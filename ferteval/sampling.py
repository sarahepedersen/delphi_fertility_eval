"""Autoregressive forecasting (Phase 2b).

Completes the trajectories of *incomplete* women (censored while still in the reproductive
window) by rolling the model forward as a **competing-exponentials point process** — exactly
the generative law Delphi is trained under. At each step the model gives per-token logits
``logit_k`` (log rates); we draw per-token waiting times ``t_k = -log(U_k)·exp(-logit_k)``
(i.e. ``t_k ~ Exp(λ_k)``), take the **minimum** (next token = argmin, Δt = min), advance the
age, and stop on the ``death`` token or the completion age. Tokens that cannot legitimately
be emitted (padding, censoring, the BIRTH_year cohort marker) are masked out.

The rollout is **batched** (all trajectories step together) and driven only by ``model.forward``
— no dependence on the fork's ``generate()`` signature. On real data the trained logits are
already per-day rates, so the same code runs unchanged. Monte-Carlo: ``n_samples`` independent
completions per woman, each becoming a pseudo-woman in the forecast dataset (balanced
replication leaves occurrence-exposure rates, ratios and means unbiased).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import EvalConfig
from .extraction import FertilityData, patient_sequences
from .loaders import DelphiBundle
from .vocab import TokenVocab

DAYS_PER_YEAR = 365.25


@dataclass
class ForecastConfig:
    n_samples: int = 25
    max_age_years: float = 50.0   # roll forward until this age (then mark complete)
    age_cap_years: float = 55.0   # hard stop
    max_steps: int = 80           # safety cap on rollout length
    temperature: float = 1.0
    batch_size: int = 128         # trajectories per forward (bounds GPU memory)
    seed: int = 1337

    @classmethod
    def from_cfg(cls, cfg: EvalConfig) -> "ForecastConfig":
        f = cfg.forecast
        return cls(n_samples=f.n_samples, max_age_years=f.max_age, age_cap_years=f.age_cap,
                   temperature=f.temperature, batch_size=cfg.inference.batch_size, seed=cfg.metrics.seed)


# --------------------------------------------------------------------------- #
# seeds                                                                         #
# --------------------------------------------------------------------------- #
def build_seeds(data: np.ndarray, vocab: TokenVocab, woman_ids=None, truncate_age: float | None = None):
    """Seed prefixes ``(woman_id, tokens, ages_days)`` to roll forward.

    The trailing terminal tokens (``censoring`` / ``death``) are stripped so the forecast
    continues *past* the cutoff. ``truncate_age`` (years) additionally cuts the history at
    that age — used by the backtest to truncate already-completed cohorts.
    """
    seqs = patient_sequences(data)
    ids = range(len(seqs)) if woman_ids is None else list(woman_ids)
    terminal = {i for i in (vocab.censoring_id, vocab.end_sequence_id) if i is not None}
    out = []
    for wid in ids:
        tokens, ages = seqs[wid]
        if truncate_age is not None:
            keep = ages <= truncate_age * DAYS_PER_YEAR
            tokens, ages = tokens[keep], ages[keep]
        tokens, ages = list(map(int, tokens)), list(map(float, ages))
        while tokens and tokens[-1] in terminal:
            tokens.pop(); ages.pop()
        out.append((int(wid), tokens, ages))
    return out


# --------------------------------------------------------------------------- #
# rollout                                                                       #
# --------------------------------------------------------------------------- #
def forecast_sequences(bundle: DelphiBundle, vocab: TokenVocab, seeds, fc: ForecastConfig,
                       rng: np.random.Generator | None = None):
    """Roll each seed forward ``fc.n_samples`` times; return ``(new_id, tokens, ages_days)``."""
    import torch

    rng = rng or np.random.default_rng(fc.seed)
    model, device = bundle.model, bundle.device
    block = int(getattr(model.config, "block_size", 80))
    pad = vocab.padding_id or 0
    death_id = vocab.end_sequence_id
    blocked = _blocked_ids(vocab)
    max_age_days = fc.max_age_years * DAYS_PER_YEAR
    cap_days = fc.age_cap_years * DAYS_PER_YEAR

    trajs, nid = [], 0
    for (_wid, tokens, ages) in seeds:
        for _ in range(fc.n_samples):
            trajs.append({"id": nid, "tok": list(tokens), "age": list(ages), "active": True})
            nid += 1

    model.to(device)
    with torch.no_grad():
        for _ in range(fc.max_steps):
            active = [t for t in trajs if t["active"]]
            if not active:
                break
            last_logits, last_age = _forward_last(active, model, torch, device, block, pad, fc.batch_size)
            last_logits[:, list(blocked)] = -np.inf
            if fc.temperature != 1.0:
                last_logits = last_logits / fc.temperature

            dt, nxt = _competing_exponential(last_logits, rng)
            for i, t in enumerate(active):
                age_next = last_age[i] + dt[i]
                if age_next >= max_age_days or age_next > cap_days:
                    # the next event falls at/after completion → stop without emitting it
                    _finalize(t, death_id, max_age_days)
                    continue
                t["tok"].append(int(nxt[i])); t["age"].append(float(age_next))
                if death_id is not None and int(nxt[i]) == death_id:
                    t["active"] = False

    # any trajectory still active at the step cap is closed off as complete
    for t in trajs:
        if t["active"]:
            _finalize(t, death_id, max_age_days)
    return [(t["id"], np.array(t["tok"], dtype=np.int64), np.array(t["age"], dtype=np.float64)) for t in trajs]


def _forward_last(active, model, torch, device, block, pad, batch_size):
    """Forward the active trajectories in mini-batches; return (last-position logits [B,V]
    numpy, last age [B] days).

    Chunking bounds GPU memory: a single forward over *all* trajectories would allocate a
    logits tensor — and the model's full attention tensor ``(n_layer, B, n_head, T, T)`` —
    sized by the total trajectory count (incomplete women × n_samples), which OOMs. We
    process ``batch_size`` trajectories at a time and keep only the last-position logits.
    """
    logits_out, age_out = [], []
    bs = max(1, int(batch_size))
    for start in range(0, len(active), bs):
        chunk = active[start:start + bs]
        Lw = min(max(len(t["tok"]) for t in chunk), block)
        B = len(chunk)
        idx = torch.full((B, Lw), pad, dtype=torch.long)
        age = torch.zeros((B, Lw), dtype=torch.float32)
        last_pos = []
        for j, t in enumerate(chunk):
            tk, ag = t["tok"][-Lw:], t["age"][-Lw:]
            idx[j, :len(tk)] = torch.tensor(tk, dtype=torch.long)
            age[j, :len(ag)] = torch.tensor(ag, dtype=torch.float32)
            last_pos.append(len(tk) - 1)
            age_out.append(t["age"][-1])
        out = model(idx.to(device), age.to(device))[0]                     # (B, Lw, V)
        rows = torch.arange(B, device=out.device)
        cols = torch.tensor(last_pos, device=out.device)
        logits_out.append(out[rows, cols].float().cpu().numpy())           # (B, V)
        del out
    return np.concatenate(logits_out, axis=0), np.array(age_out, dtype=np.float64)


def _competing_exponential(logits: np.ndarray, rng: np.random.Generator):
    """Draw (Δt, next_token) per row via competing exponentials with rates exp(logit)."""
    u = rng.random(logits.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        times = -np.exp(-logits) * np.log(u)        # t_k ~ Exp(exp(logit_k)); masked → inf
    times[~np.isfinite(times)] = np.inf
    return times.min(axis=1), times.argmin(axis=1)


def _finalize(t, death_id, age_days):
    """Close a trajectory by appending a death/end marker (at a strictly later age) so
    extraction treats it as complete."""
    t["active"] = False
    if death_id is None or (t["tok"] and t["tok"][-1] == death_id):
        return
    floor = (t["age"][-1] + 1.0) if t["age"] else age_days  # keep ages strictly increasing
    t["tok"].append(int(death_id)); t["age"].append(float(max(age_days, floor)))


def _blocked_ids(vocab: TokenVocab) -> set[int]:
    """Tokens the model must not emit during a rollout."""
    ids = {vocab.padding_id, vocab.censoring_id}
    ids |= set(vocab.birth_year_to_cohort.keys())
    return {i for i in ids if i is not None}


# --------------------------------------------------------------------------- #
# convenience                                                                   #
# --------------------------------------------------------------------------- #
def forecast_incomplete(cfg: EvalConfig, bundle: DelphiBundle, vocab: TokenVocab, fd: FertilityData,
                        data: np.ndarray, fc: ForecastConfig | None = None) -> FertilityData:
    """Forecast the incomplete women of ``fd`` and return them as a forecast FertilityData."""
    fc = fc or ForecastConfig.from_cfg(cfg)
    rng = np.random.default_rng(fc.seed)
    seeds = build_seeds(data, vocab, woman_ids=fd.incomplete_women["woman_id"].tolist())
    trajs = forecast_sequences(bundle, vocab, seeds, fc, rng)
    return FertilityData.from_sequences(trajs, vocab, completion_age=cfg.demography.completion_age,
                                        repro_ages=(cfg.demography.repro_age_min, cfg.demography.repro_age_max))


def seeds_from_data(data: np.ndarray, vocab: TokenVocab, up_to_age_years: float | None = None):
    """Backwards-compatible alias for :func:`build_seeds` (truncate at ``up_to_age_years``)."""
    return build_seeds(data, vocab, truncate_age=up_to_age_years)

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
    """Roll each seed forward ``fc.n_samples`` times; return ``[(new_id, tokens, ages_days)]``.

    Fully vectorised on-device: state (token/age buffers, ``lengths``, ``active`` mask) lives on
    the model's device and every step advances the whole batch with torch ops — the competing-
    exponential sampling runs on-device, so there is **no per-step CPU sync and no per-trajectory
    Python loop**. Only the forward is chunked (by ``fc.batch_size``) to bound the attention
    tensor's memory; its last-position logits are kept on the GPU. Runs on CPU too (tests).
    """
    import torch

    n_seeds = len(seeds)
    if n_seeds == 0:
        return []
    model, device = bundle.model, bundle.device
    block = int(getattr(model.config, "block_size", 80))
    pad = vocab.padding_id or 0
    death_id = vocab.end_sequence_id
    blocked = sorted(_blocked_ids(vocab))
    max_age_days = fc.max_age_years * DAYS_PER_YEAR
    ns = int(fc.n_samples)
    seed = int(rng.integers(0, 2 ** 31 - 1)) if rng is not None else int(fc.seed)

    # seed buffers: one row per seed (right-padded), then replicated n_samples times
    st = np.full((n_seeds, block), pad, dtype=np.int64)
    sa = np.zeros((n_seeds, block), dtype=np.float32)
    sl = np.zeros(n_seeds, dtype=np.int64)
    for i, (_wid, tokens, ages) in enumerate(seeds):
        tk = np.asarray(tokens, dtype=np.int64)[-block:]
        ag = np.asarray(ages, dtype=np.float32)[-block:]
        s = len(tk)
        st[i, :s] = tk; sa[i, :s] = ag; sl[i] = max(s, 1)

    buf_tok = torch.as_tensor(st, device=device).repeat_interleave(ns, 0)
    buf_age = torch.as_tensor(sa, device=device).repeat_interleave(ns, 0)
    lengths = torch.as_tensor(sl, device=device).repeat_interleave(ns, 0)
    B = buf_tok.shape[0]
    rowsB = torch.arange(B, device=device)
    last_age = buf_age[rowsB, (lengths - 1).clamp_min(0)]
    active = torch.ones(B, dtype=torch.bool, device=device)
    try:
        g = torch.Generator(device=device); g.manual_seed(seed)
    except (RuntimeError, TypeError):
        g = None; torch.manual_seed(seed)

    def _terminate(mask):  # append death marker at completion age for the masked rows
        if death_id is None:
            return
        lf = lengths[mask]
        buf_tok[rowsB[mask], lf] = death_id
        buf_age[rowsB[mask], lf] = float(max_age_days)
        lengths[mask] = lf + 1

    model.to(device)
    with torch.no_grad():
        for _ in range(int(fc.max_steps)):
            if not bool(active.any()):
                break
            logits = _forward_last(buf_tok, buf_age, lengths, model, torch, device, int(fc.batch_size))
            logits[:, blocked] = float("-inf")
            if fc.temperature != 1.0:
                logits = logits / fc.temperature

            u = torch.rand(logits.shape, generator=g, device=device).clamp_min(1e-12)
            times = -torch.exp(-logits) * torch.log(u)          # t_k ~ Exp(exp(logit_k))
            times = torch.nan_to_num(times, nan=float("inf"), posinf=float("inf"))
            dt, nxt = times.min(dim=1)
            age_next = last_age + dt

            has_room = lengths < block
            reached = age_next >= max_age_days
            emit = active & (~reached) & has_room
            le = lengths[emit]
            buf_tok[rowsB[emit], le] = nxt[emit]
            buf_age[rowsB[emit], le] = age_next[emit]
            lengths[emit] = le + 1
            last_age[emit] = age_next[emit]

            _terminate(active & reached & has_room)             # reached completion → death@50
            died = (emit & (nxt == death_id)) if death_id is not None else torch.zeros_like(active)
            active = active & ~((active & reached) | (active & ~has_room) | died)

        _terminate(active & (lengths < block))                  # close off survivors at the step cap

    tok = buf_tok.cpu().numpy(); age = buf_age.cpu().numpy(); L = lengths.cpu().numpy()
    return [(b, tok[b, :L[b]].astype(np.int64), age[b, :L[b]].astype(np.float64)) for b in range(B)]


def _forward_last(buf_tok, buf_age, lengths, model, torch, device, batch_size):
    """Chunked forward over the whole batch; return last-position logits ``(B, V)`` **on the
    model's device** (no CPU sync). Chunking by ``batch_size`` bounds the model's attention
    tensor ``(n_layer, chunk, n_head, T, T)``, which is the real memory driver."""
    B = buf_tok.shape[0]
    bs = max(1, int(batch_size))
    outs = []
    for s in range(0, B, bs):
        e = min(s + bs, B)
        out = model(buf_tok[s:e], buf_age[s:e])[0]                     # (chunk, block, V)
        rows = torch.arange(e - s, device=out.device)
        outs.append(out[rows, lengths[s:e] - 1].float())              # (chunk, V), stays on device
        del out
    return torch.cat(outs, dim=0)


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


def seed_states(seeds, vocab: TokenVocab):
    """Per-seed state at the forecast cutoff, indexed by seed order.

    Columns: ``seed_idx, woman_id, cohort, cutoff_age, parity_at_cutoff``. Each forecast
    trajectory maps back to its seed via ``trajectory_id // n_samples``, so this lets the
    recuperation analysis condition the model's forecast on the woman's state at the cutoff.
    """
    import pandas as pd

    child_ids = list(vocab.child_id_set)
    rows = []
    for idx, (wid, tokens, ages) in enumerate(seeds):
        toks = np.asarray(tokens, dtype=np.int64)
        cohort = vocab.cohort_of_sequence(toks)
        rows.append({
            "seed_idx": idx,
            "woman_id": int(wid),
            "cohort": int(cohort) if cohort is not None else -1,
            "cutoff_age": (float(max(ages)) / DAYS_PER_YEAR) if len(ages) else float("nan"),
            "parity_at_cutoff": int(np.isin(toks, child_ids).sum()),
        })
    return pd.DataFrame(rows)

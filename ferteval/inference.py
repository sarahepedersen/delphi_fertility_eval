"""Model inference: per-position child-token scores + bookkeeping for the metrics.

This is the fertility-adapted core of the upstream ``evaluate_auc_pipeline``. Given the
eval batch ``(x, a, y, b)`` (input tokens/ages, target tokens/ages) we run a chunked
forward pass and keep, for every position, the model's logits for the child token(s).
We also precompute everything the metric layer needs to define case/control sets and
subgroups without touching the model again:

* ``pred_idx[p, t]`` — the last *input* position strictly before ``b[p, t] - offset``,
  i.e. where the one-step-ahead prediction for target ``t`` is read (mirrors upstream
  ``(a[:, :, None] < b[:, None, :] - offset).sum(1) - 1``).
* ``child_tot_x[p, t]`` — running parity: number of child tokens in ``x[p, :t+1]``.
* ``cohort[p]`` — birth-year cohort from the sequence's BIRTH_year token.

``torch`` is imported lazily inside :func:`run_inference` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EvalConfig
from .loaders import DelphiBundle, load_data, make_eval_batch
from .vocab import TokenVocab


@dataclass
class InferenceResult:
    x: np.ndarray            # (B, T) input tokens
    a: np.ndarray            # (B, T) input ages (days)
    y: np.ndarray            # (B, T) target tokens
    b: np.ndarray            # (B, T) target ages (days)
    child_logits: np.ndarray  # (B, T, n_child) logits for child_ids (input-position aligned)
    child_ids: list[int]     # column order of child_logits
    log_z: np.ndarray        # (B, T) log-partition = logsumexp over full vocab (for probabilities)
    pred_idx: np.ndarray     # (B, T) int; input pos to read score for target t (-1 if none)
    child_tot_x: np.ndarray   # (B, T) running child count along x
    cohort: np.ndarray       # (B,) birth-year cohort (-1 if missing)
    offset_days: float

    @property
    def age_years(self) -> np.ndarray:
        return self.a / 365.25

    def any_child_score(self) -> np.ndarray:
        """(B, T) 'any birth' score = logsumexp over child logits (monotone in total intensity)."""
        return _logsumexp(self.child_logits, axis=-1)

    def child_score(self, token_id: int) -> np.ndarray:
        """(B, T) score for a specific child token id (e.g. son or daughter)."""
        col = self.child_ids.index(token_id)
        return self.child_logits[:, :, col]

    def child_prob(self) -> np.ndarray:
        """(B, T) P(next token is a birth) = softmax mass on child tokens."""
        return np.exp(self.any_child_score() - self.log_z)


# --------------------------------------------------------------------------- #
# main entry                                                                   #
# --------------------------------------------------------------------------- #
def run_inference(cfg: EvalConfig, bundle: DelphiBundle, vocab: TokenVocab) -> InferenceResult:
    """Load data, assemble the eval batch, and run the chunked forward pass."""
    import torch

    data = load_data(cfg.paths.data)
    p2i = bundle.get_p2i(data)
    batch = make_eval_batch(bundle, data, p2i, cfg)  # (x, a, y, b) tensors on device
    return _infer_from_batch(cfg, bundle, vocab, batch, torch)


def _infer_from_batch(cfg, bundle, vocab, batch, torch) -> InferenceResult:
    x_t, a_t, y_t, b_t = batch
    child_ids = list(vocab.child_ids)
    child_cols = torch.tensor(child_ids, device=x_t.device, dtype=torch.long)

    logits_chunks = []
    logz_chunks = []
    bs = cfg.inference.batch_size
    bundle.model.to(bundle.device)
    with torch.no_grad():
        for dd in zip(*[torch.split(t, bs) for t in (x_t, a_t, y_t, b_t)]):
            dd = [t.to(bundle.device) for t in dd]
            logits = bundle.model(*dd)[0]                 # (b, T, vocab)
            sel = logits.index_select(-1, child_cols)     # (b, T, n_child)
            logz = torch.logsumexp(logits.float(), dim=-1)  # (b, T) full-vocab normalizer
            logits_chunks.append(sel.float().cpu().numpy())
            logz_chunks.append(logz.cpu().numpy())
    child_logits = np.concatenate(logits_chunks, axis=0)  # (B, T, n_child)
    log_z = np.concatenate(logz_chunks, axis=0)           # (B, T)

    x = _to_np(x_t)
    a = _to_np(a_t).astype(np.float64)
    y = _to_np(y_t)
    b = _to_np(b_t).astype(np.float64)

    offset = float(cfg.inference.offset_days)
    pred_idx = compute_pred_idx(a, b, offset)
    child_tot_x = running_child_count(x, child_ids)
    cohort = sequence_cohorts(x, vocab)

    return InferenceResult(
        x=x, a=a, y=y, b=b,
        child_logits=child_logits, child_ids=child_ids, log_z=log_z,
        pred_idx=pred_idx, child_tot_x=child_tot_x, cohort=cohort,
        offset_days=offset,
    )


# --------------------------------------------------------------------------- #
# bookkeeping (pure numpy — unit-testable without torch)                       #
# --------------------------------------------------------------------------- #
def compute_pred_idx(a: np.ndarray, b: np.ndarray, offset: float) -> np.ndarray:
    """Last input position strictly before ``b[:, t] - offset`` for each target ``t``.

    Mirrors upstream: ``(a[:, :, None] < b[:, None, :] - offset).sum(1) - 1``.
    Returns ``(B, T)`` int array; -1 means no valid input position exists.
    """
    less = a[:, :, np.newaxis] < (b[:, np.newaxis, :] - offset)  # (B, T_in, T_tgt)
    return less.sum(axis=1) - 1


def running_child_count(x: np.ndarray, child_ids: list[int]) -> np.ndarray:
    """Cumulative number of child tokens along each input sequence (inclusive)."""
    is_child = np.isin(x, list(child_ids))
    return np.cumsum(is_child, axis=1)


def sequence_cohorts(x: np.ndarray, vocab: TokenVocab) -> np.ndarray:
    """Birth-year cohort per sequence from its BIRTH_year token (-1 if absent)."""
    mapping = vocab.birth_year_to_cohort
    out = np.full(x.shape[0], -1, dtype=np.int64)
    if not mapping:
        return out
    birth_ids = np.array(sorted(mapping.keys()))
    for i in range(x.shape[0]):
        row = x[i]
        hits = row[np.isin(row, birth_ids)]
        if hits.size:
            out[i] = mapping[int(hits[0])]
    return out


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def _logsumexp(arr: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(arr, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

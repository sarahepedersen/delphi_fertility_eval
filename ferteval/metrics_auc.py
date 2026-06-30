"""Discrimination (AUC) metrics for the fertility model.

Two layers:

1. **AUC + nonparametric bootstrap variance** — a point AUC (tie-correct Mann-Whitney)
   plus a CPU resampling bootstrap for the standard error and percentile CI. Pooling
   across subgroups uses a *stratified* bootstrap (resample within each subgroup, then
   recompute the mean-of-subgroup-AUCs) so the pooled CI accounts for subgroup sizes.
   Runs anywhere — no CUDA, no closed-form variance assumptions.

2. **Fertility subgroup AUC** — a parity-conditional, discrete-time-hazard AUC
   stratified by (age × birth-cohort), replacing Delphi's (age × sex) disease-onset split.

Design choice (documented deliberately): because births are *recurrent*, we frame
discrimination as a discrete-time hazard. For the transition n -> n+1 the **risk set**
is women who reached parity n. A **positive** woman-year is the (n+1)-th birth; the
**negative** woman-years are the ``no_event`` years of women who reached parity n but
did *not* progress (final parity == n). The model score is the one-step-ahead child
hazard read at ``pred_idx``. n=0 recovers the classic first-birth (parous vs
nulliparous) onset AUC. Within each subgroup we dedupe to one row per patient (random
pick) to avoid pseudo-replication before computing the AUC / bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .inference import InferenceResult


# =========================================================================== #
# AUC point estimate + bootstrap                                              #
# =========================================================================== #
def roc_auc(case: np.ndarray, control: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score_case > score_control) + 0.5 P(tie), tie-correct."""
    m, n = len(case), len(control)
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(np.concatenate([np.asarray(case), np.asarray(control)]))  # 1-based midranks
    return float((r[:m].sum() - m * (m + 1) / 2.0) / (m * n))


def bootstrap_auc(case: np.ndarray, control: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Nonparametric bootstrap: resample cases & controls with replacement, AUC each time.

    Returns an array of length ``n_boot`` of bootstrap AUCs (CPU, vectorised). Ties are
    broken arbitrarily within a resample, which is negligible for continuous model scores.
    """
    case = np.asarray(case, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    m, n = len(case), len(control)
    if m == 0 or n == 0 or n_boot <= 0:
        return np.full(max(n_boot, 0), np.nan)

    bc = case[rng.integers(0, m, size=(n_boot, m))]
    bk = control[rng.integers(0, n, size=(n_boot, n))]
    combined = np.concatenate([bc, bk], axis=1)               # (n_boot, m+n)
    ranks = combined.argsort(axis=1).argsort(axis=1)          # 0-based ranks per replicate
    case_rank_sum = ranks[:, :m].sum(axis=1)
    u = case_rank_sum - m * (m - 1) / 2.0
    return u / (m * n)


def _ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if samples.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _se(samples: np.ndarray) -> float:
    """Bootstrap SE; NaN (no warning) when there are too few replicates to estimate it."""
    return float(np.std(samples, ddof=1)) if samples.size >= 2 else float("nan")


# =========================================================================== #
# Fertility event-row construction                                            #
# =========================================================================== #
@dataclass
class EventRows:
    """Flat per-woman-year rows for one AUC target."""

    score: np.ndarray
    age: np.ndarray      # years, at the prediction position
    cohort: np.ndarray   # birth-year cohort
    pid: np.ndarray      # patient index (for dedup)
    label: np.ndarray    # 1 = positive woman-year, 0 = negative

    def __len__(self) -> int:
        return len(self.score)


def _gather_at_pred(values_2d: np.ndarray, p: np.ndarray, pidx: np.ndarray) -> np.ndarray:
    return values_2d[p, pidx]


def build_parity_transition_rows(result: InferenceResult, score2d: np.ndarray, n: int, no_event_id: int) -> EventRows:
    """Rows for the parity transition n -> n+1 (n=0 => first birth).

    Positive  = the (n+1)-th birth woman-year of women who progressed.
    Negative  = no_event woman-years of women who reached parity n and did NOT progress.
    """
    y, a, pred_idx, cohort = result.y, result.a, result.pred_idx, result.cohort
    child_set = list(result.child_ids)

    is_birth = np.isin(y, child_set)
    is_noevent = (y == no_event_id)
    birth_ordinal = np.cumsum(is_birth, axis=1)          # (B,T) 1-indexed order at birth positions
    parity_before = birth_ordinal - is_birth             # births strictly before this position
    final_births = birth_ordinal[:, -1]                  # (B,) total births per woman

    valid = pred_idx >= 0
    pos_mask = is_birth & (birth_ordinal == (n + 1)) & valid
    neg_mask = is_noevent & (parity_before == n) & (final_births[:, None] == n) & valid

    return _rows_from_masks(pos_mask, neg_mask, score2d, a, pred_idx, cohort)


def build_child_sex_rows(result: InferenceResult, son_id: int, daughter_id: int) -> EventRows:
    """Rows for son-vs-daughter discrimination among birth woman-years."""
    y, a, pred_idx, cohort = result.y, result.a, result.pred_idx, result.cohort
    score2d = result.child_score(son_id)  # higher => more son-like
    valid = pred_idx >= 0
    pos_mask = (y == son_id) & valid
    neg_mask = (y == daughter_id) & valid
    return _rows_from_masks(pos_mask, neg_mask, score2d, a, pred_idx, cohort)


def _rows_from_masks(pos_mask, neg_mask, score2d, a, pred_idx, cohort) -> EventRows:
    def collect(mask, label):
        p, t = np.where(mask)
        if p.size == 0:
            return None
        pidx = pred_idx[p, t]
        return EventRows(
            score=_gather_at_pred(score2d, p, pidx),
            age=_gather_at_pred(a, p, pidx) / 365.25,
            cohort=cohort[p],
            pid=p,
            label=np.full(p.size, label, dtype=np.int64),
        )

    pos = collect(pos_mask, 1)
    neg = collect(neg_mask, 0)
    parts = [r for r in (pos, neg) if r is not None]
    if not parts:
        return EventRows(*(np.array([]) for _ in range(5)))
    return EventRows(
        score=np.concatenate([r.score for r in parts]),
        age=np.concatenate([r.age for r in parts]),
        cohort=np.concatenate([r.cohort for r in parts]),
        pid=np.concatenate([r.pid for r in parts]),
        label=np.concatenate([r.label for r in parts]),
    )


# =========================================================================== #
# Subgroup AUC over (age x cohort)                                            #
# =========================================================================== #
def subgroup_auc(
    rows: EventRows,
    age_edges: list[int],
    cohort_edges: list[int],
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
) -> tuple[pd.DataFrame, np.ndarray]:
    """AUC within each (age bracket x cohort bracket), deduped to one row per patient.

    Returns ``(df_subgroups, boots)`` where ``df_subgroups`` has one row per non-empty
    subgroup (point AUC + bootstrap SE + percentile CI) and ``boots`` is a
    ``(n_subgroups, n_bootstrap)`` matrix of bootstrap AUCs aligned to ``df_subgroups``
    rows, used by :func:`aggregate_bootstrap` for the pooled (stratified) CI.
    """
    records, boot_rows = [], []
    pos = rows.label == 1
    neg = ~pos
    for ai in range(len(age_edges) - 1):
        a_lo, a_hi = age_edges[ai], age_edges[ai + 1]
        in_age = (rows.age >= a_lo) & (rows.age < a_hi)
        for ci in range(len(cohort_edges) - 1):
            c_lo, c_hi = cohort_edges[ci], cohort_edges[ci + 1]
            in_cohort = (rows.cohort >= c_lo) & (rows.cohort < c_hi)
            sel = in_age & in_cohort

            case_idx = _dedup_one_per_patient(np.where(sel & pos)[0], rows.pid, rng)
            ctrl_idx = _dedup_one_per_patient(np.where(sel & neg)[0], rows.pid, rng)
            if len(case_idx) < 2 or len(ctrl_idx) < 2:
                continue

            case_scores = rows.score[case_idx]
            ctrl_scores = rows.score[ctrl_idx]
            auc_val = roc_auc(case_scores, ctrl_scores)
            boots = bootstrap_auc(case_scores, ctrl_scores, n_bootstrap, rng)
            lo, hi = _ci(boots)
            records.append({
                "age": a_lo,
                "cohort": c_lo,
                "auc": auc_val,
                "auc_se": _se(boots),
                "auc_lo": lo,
                "auc_hi": hi,
                "n_diseased": int(len(case_idx)),
                "n_healthy": int(len(ctrl_idx)),
            })
            boot_rows.append(boots)

    df = pd.DataFrame.from_records(records)
    boots = np.vstack(boot_rows) if boot_rows else np.empty((0, n_bootstrap))
    return df, boots


def aggregate_bootstrap(df_subgroups: pd.DataFrame, boots: np.ndarray) -> pd.DataFrame:
    """Pool subgroup AUCs as an (unweighted) mean, with a stratified-bootstrap CI.

    The pooled AUC is the mean of subgroup point AUCs; its sampling distribution is the
    per-replicate mean across subgroups (``boots.mean(axis=0)``), giving an SE and a
    percentile CI without any normality assumption.
    """
    if df_subgroups.empty or boots.size == 0:
        return pd.DataFrame()

    pooled_boots = boots.mean(axis=0)  # (n_bootstrap,) stratified-bootstrap pooled AUC
    lo, hi = _ci(pooled_boots)
    return pd.DataFrame([{
        "auc": float(df_subgroups["auc"].mean()),
        "auc_se": _se(pooled_boots),
        "auc_lo": lo,
        "auc_hi": hi,
        "n_subgroups": int(len(df_subgroups)),
        "n_diseased": int(df_subgroups["n_diseased"].sum()),
        "n_healthy": int(df_subgroups["n_healthy"].sum()),
    }])


# --------------------------------------------------------------------------- #
def _dedup_one_per_patient(idx: np.ndarray, pid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep one randomly-chosen row per patient (anti-pseudoreplication)."""
    if idx.size == 0:
        return idx
    perm = rng.permutation(idx.size)
    _, first = np.unique(pid[idx][perm], return_index=True)
    return idx[perm[first]]

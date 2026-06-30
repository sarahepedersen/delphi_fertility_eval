"""Predicted-vs-observed calibration for the fertility model.

Discrimination (AUC) tells you whether higher-risk woman-years are ranked above
lower-risk ones; calibration tells you whether the predicted *probabilities* are
right. We frame each valid prediction position as a woman-year with:

* predicted ``p`` = ``P(next token is a birth)`` (softmax mass on child tokens), and
* observed ``label`` = 1 if a birth actually occurred that year.

We report a reliability curve (quantile-binned predicted vs observed), the Expected
Calibration Error (ECE), and per-cohort calibration-in-the-large (mean predicted vs
mean observed) plus a logistic calibration slope/intercept — the cohort-subgroup
calibration you asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inference import InferenceResult


@dataclass
class CalibrationRows:
    pred: np.ndarray      # predicted P(birth) per woman-year
    label: np.ndarray     # observed birth (1/0)
    age: np.ndarray       # years at prediction
    cohort: np.ndarray    # birth-year cohort
    parity_before: np.ndarray

    def __len__(self) -> int:
        return len(self.pred)


def build_calibration_rows(result: InferenceResult, no_event_id: int, parity: int | None = None) -> CalibrationRows:
    """One row per valid woman-year (target in child ∪ no_event, valid pred_idx).

    If ``parity`` is given, restrict to woman-years entered at that parity.
    """
    y, a, pred_idx, cohort = result.y, result.a, result.pred_idx, result.cohort
    child_set = list(result.child_ids)

    is_birth = np.isin(y, child_set)
    is_noevent = (y == no_event_id)
    parity_before = np.cumsum(is_birth, axis=1) - is_birth
    valid = (pred_idx >= 0) & (is_birth | is_noevent)
    if parity is not None:
        valid = valid & (parity_before == parity)

    p, t = np.where(valid)
    pidx = pred_idx[p, t]
    prob2d = result.child_prob()
    return CalibrationRows(
        pred=prob2d[p, pidx],
        label=is_birth[p, t].astype(np.int64),
        age=a[p, pidx] / 365.25,
        cohort=cohort[p],
        parity_before=parity_before[p, t],
    )


# --------------------------------------------------------------------------- #
# core metrics                                                                 #
# --------------------------------------------------------------------------- #
def reliability_curve(pred: np.ndarray, label: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table: predicted vs observed per bin."""
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    if pred.size == 0:
        return pd.DataFrame(columns=["bin", "pred_mean", "obs_mean", "count", "pred_lo", "pred_hi"])

    # quantile edges, deduped (handles spiky/degenerate predicted distributions)
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([pred.min(), pred.max() + 1e-12])
    bins = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, len(edges) - 2)

    records = []
    for b in range(len(edges) - 1):
        m = bins == b
        if not m.any():
            continue
        records.append({
            "bin": b,
            "pred_mean": float(pred[m].mean()),
            "obs_mean": float(label[m].mean()),
            "count": int(m.sum()),
            "pred_lo": float(edges[b]),
            "pred_hi": float(edges[b + 1]),
        })
    return pd.DataFrame.from_records(records)


def expected_calibration_error(pred: np.ndarray, label: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) * |obs_b - pred_b| over quantile bins."""
    rc = reliability_curve(pred, label, n_bins)
    if rc.empty:
        return float("nan")
    w = rc["count"] / rc["count"].sum()
    return float((w * (rc["obs_mean"] - rc["pred_mean"]).abs()).sum())


def calibration_slope_intercept(pred: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """Logistic calibration slope & intercept: fit label ~ sigmoid(intercept + slope*logit(pred)).

    Perfect calibration => slope 1, intercept 0. Slope<1 indicates over-extreme predictions.
    """
    pred = np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1 - 1e-6)
    label = np.asarray(label, dtype=np.int64)
    if np.unique(label).size < 2:
        return float("nan"), float("nan")
    logit = np.log(pred / (1 - pred)).reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression

        # C=inf == unregularised fit, and avoids the deprecated penalty=None arg
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        lr.fit(logit, label)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return _logit_fit_newton(logit.ravel(), label)


# --------------------------------------------------------------------------- #
# per-cohort calibration                                                       #
# --------------------------------------------------------------------------- #
def per_cohort_calibration(rows: CalibrationRows, cohort_edges: list[int], n_bins: int = 10) -> pd.DataFrame:
    """Calibration-in-the-large + slope + ECE for each birth-cohort bracket."""
    records = []
    for ci in range(len(cohort_edges) - 1):
        c_lo, c_hi = cohort_edges[ci], cohort_edges[ci + 1]
        m = (rows.cohort >= c_lo) & (rows.cohort < c_hi)
        if m.sum() < 2:
            continue
        pred, label = rows.pred[m], rows.label[m]
        slope, intercept = calibration_slope_intercept(pred, label)
        records.append({
            "cohort": c_lo,
            "cohort_hi": c_hi,
            "n_woman_years": int(m.sum()),
            "n_births": int(label.sum()),
            "mean_pred": float(pred.mean()),
            "mean_obs": float(label.mean()),
            "calib_in_large": float(label.mean() - pred.mean()),
            "slope": slope,
            "intercept": intercept,
            "ece": expected_calibration_error(pred, label, n_bins),
        })
    return pd.DataFrame.from_records(records)


def _logit_fit_newton(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Tiny IRLS logistic fit (fallback if sklearn is unavailable)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        grad = X.T @ (y - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[1]), float(beta[0])

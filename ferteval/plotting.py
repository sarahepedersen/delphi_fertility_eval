"""Matplotlib figures for AUC-by-subgroup and calibration reliability curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_auc_by_age(df_subgroups: pd.DataFrame, out_path: str | Path, title: str = "AUC by age") -> Path | None:
    """One line per cohort: AUC vs age bracket."""
    if df_subgroups.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cohort, g in df_subgroups.groupby("cohort"):
        g = g.sort_values("age")
        ax.plot(g["age"], g["auc"], marker="o", label=f"cohort {int(cohort)}")
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_path)


def plot_auc_by_cohort(df_subgroups: pd.DataFrame, out_path: str | Path, title: str = "AUC by cohort") -> Path | None:
    """One line per age bracket: AUC vs cohort."""
    if df_subgroups.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for age, g in df_subgroups.groupby("age"):
        g = g.sort_values("cohort")
        ax.plot(g["cohort"], g["auc"], marker="o", label=f"age {int(age)}")
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xlabel("Birth cohort")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_path)


def plot_reliability(reliability: pd.DataFrame, out_path: str | Path, title: str = "Calibration") -> Path | None:
    """Reliability curve (predicted vs observed) with the diagonal."""
    if reliability.empty:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    hi = max(reliability["pred_mean"].max(), reliability["obs_mean"].max()) * 1.1
    ax.plot([0, hi], [0, hi], ls="--", c="grey", lw=1, label="perfect")
    ax.plot(reliability["pred_mean"], reliability["obs_mean"], marker="o", label="model")
    ax.set_xlabel("Predicted P(birth)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_reliability_by_cohort(rows_by_cohort: dict, out_path: str | Path, title: str = "Calibration by cohort") -> Path | None:
    """Overlay reliability curves for several cohorts. ``rows_by_cohort``: {label: reliability_df}."""
    curves = {k: v for k, v in rows_by_cohort.items() if not v.empty}
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    hi = max(v["obs_mean"].max() for v in curves.values()) * 1.15 or 0.1
    ax.plot([0, hi], [0, hi], ls="--", c="grey", lw=1, label="perfect")
    for label, rc in curves.items():
        ax.plot(rc["pred_mean"], rc["obs_mean"], marker="o", ms=3, label=str(label))
    ax.set_xlabel("Predicted P(birth)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

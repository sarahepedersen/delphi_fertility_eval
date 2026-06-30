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


# =========================================================================== #
# Demographic figures (phase 2)                                               #
# =========================================================================== #
def plot_ccf_by_cohort(ccf: pd.DataFrame, out_path, title="Cumulated cohort fertility") -> "Path | None":
    """CCF(age) per cohort; solid over observed ages, dashed once exposure runs out."""
    if ccf.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cohort, g in ccf.groupby("cohort"):
        g = g.sort_values("age").reset_index(drop=True)
        obs = g[g["observed"]]
        (line,) = ax.plot(obs["age"], obs["ccf"], label=f"{int(cohort)}")
        if (~g["observed"]).any():  # forecast / unobserved tail — start it at the last observed point
            tail = g.loc[obs.index.max():] if not obs.empty else g
            ax.plot(tail["age"], tail["ccf"], ls="--", color=line.get_color())
    ax.set_xlabel("Age"); ax.set_ylabel("Cumulated births / woman"); ax.set_title(title)
    ax.legend(title="cohort", fontsize=7, ncol=2)
    return _save(fig, out_path)


def plot_ppr(ppr: pd.DataFrame, out_path, title="Parity progression ratios") -> "Path | None":
    if ppr.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cohort, g in ppr.groupby("cohort"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["ppr"], marker="o", label=f"{int(cohort)}")
    ax.set_xlabel("Parity n (n→n+1)"); ax.set_ylabel("PPR"); ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.legend(title="cohort", fontsize=7, ncol=2)
    return _save(fig, out_path)


def plot_km_survival(km: pd.DataFrame, out_path, title="Time to first birth (KM)") -> "Path | None":
    if km.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cohort, g in km.groupby("cohort"):
        g = g.sort_values("age")
        ax.step(g["age"], g["survival"], where="post", label=f"{int(cohort)}")
    ax.set_xlabel("Age"); ax.set_ylabel("P(no birth yet)"); ax.set_ylim(0, 1.02); ax.set_title(title)
    ax.legend(title="cohort", fontsize=7, ncol=2)
    return _save(fig, out_path)


def plot_mab1_timeseries(mab: pd.DataFrame, out_path, x="period", title="Mean age at first birth") -> "Path | None":
    if mab.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    g = mab.sort_values(x)
    ax.plot(g[x], g["mean_age"], marker=".")
    ax.set_xlabel("Calendar year" if x == "period" else "Cohort"); ax.set_ylabel("Mean age (yrs)")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_asfr(asfr: pd.DataFrame, out_path, periods=None, title="ASFR by age") -> "Path | None":
    if asfr.empty:
        return None
    avail = sorted(asfr["period"].unique())
    periods = periods or avail[:: max(1, len(avail) // 6)]  # show ~6 schedules
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for yr in periods:
        g = asfr[asfr["period"] == yr].sort_values("age")
        ax.plot(g["age"], g["asfr"], label=f"{int(yr)}")
    ax.set_xlabel("Age"); ax.set_ylabel("ASFR"); ax.set_title(title)
    ax.legend(title="year", fontsize=7, ncol=2)
    return _save(fig, out_path)


def plot_age_parity_surface(surface: pd.DataFrame, out_dir, prefix="age_parity") -> list:
    """One heatmap (age × parity) per selected year."""
    paths = []
    for year, g in surface.groupby("year"):
        grid = g.pivot_table(index="parity", columns="age", values="rate")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        im = ax.pcolormesh(grid.columns, grid.index, grid.to_numpy(), shading="nearest")
        fig.colorbar(im, ax=ax, label="rate")
        ax.set_xlabel("Age"); ax.set_ylabel("Parity (entered)"); ax.set_title(f"Age × parity rate, {int(year)}")
        paths.append(_save(fig, f"{out_dir}/{prefix}__{int(year)}.png"))
    return paths


def plot_birth_order_age_profile(prof: pd.DataFrame, out_path, title="Age profile of birth order") -> "Path | None":
    if prof.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for order, g in prof.groupby("order"):
        g = g.sort_values("age")
        ax.plot(g["age"], g["density"], label=f"order {int(order)}")
    ax.set_xlabel("Age"); ax.set_ylabel("Share of births"); ax.set_title(title)
    ax.legend(fontsize=7)
    return _save(fig, out_path)


def plot_lexis_surface(lexis: pd.DataFrame, out_path, title="First-birth intensity (Lexis)") -> "Path | None":
    """Heatmap cohort (x) × age (y) of first-birth intensity; unobserved cells left blank."""
    if lexis.empty:
        return None
    grid = lexis.pivot_table(index="age", columns="cohort", values="intensity")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.pcolormesh(grid.columns, grid.index, grid.to_numpy(), shading="nearest")
    fig.colorbar(im, ax=ax, label="first-birth intensity")
    ax.set_xlabel("Birth cohort"); ax.set_ylabel("Age"); ax.set_title(title)
    return _save(fig, out_path)


def plot_ccf_completed(ccf_obs: pd.DataFrame, ccf_full: pd.DataFrame, out_path,
                       title="CCF: observed + model forecast") -> "Path | None":
    """Per cohort: solid observed CCF, dashed model-forecast continuation past the cutoff."""
    if ccf_full.empty:
        return None
    obs = ccf_obs.set_index(["cohort", "age"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cohort, g in ccf_full.groupby("cohort"):
        g = g.sort_values("age")
        observed_flag = [bool(obs["observed"].get((cohort, a), False)) for a in g["age"]]
        g = g.assign(obs=observed_flag)
        solid = g[g["obs"]]
        (line,) = ax.plot(solid["age"], solid["ccf"], label=f"{int(cohort)}")
        if (~g["obs"]).any():  # forecast continuation, anchored at the last observed point
            tail = g.loc[solid.index.max():] if not solid.empty else g
            ax.plot(tail["age"], tail["ccf"], ls="--", color=line.get_color())
    ax.set_xlabel("Age"); ax.set_ylabel("Cumulated births / woman"); ax.set_title(title)
    ax.legend(title="cohort", fontsize=7, ncol=2)
    return _save(fig, out_path)


def plot_backtest_ccf(bt: pd.DataFrame, out_path, title="Backtest: forecast vs observed CCF") -> "Path | None":
    """Scatter of forecast vs observed completed CCF, one colour per truncation age."""
    if bt.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = min(bt["ccf_observed"].min(), bt["ccf_forecast"].min())
    hi = max(bt["ccf_observed"].max(), bt["ccf_forecast"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1, label="perfect")
    for age, g in bt.groupby("truncation_age"):
        ax.scatter(g["ccf_observed"], g["ccf_forecast"], label=f"trunc @ {int(age)}")
    ax.set_xlabel("Observed CCF"); ax.set_ylabel("Forecast CCF"); ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_observed_vs_forecast(df, x, y, out_path, group="source", title="Observed vs forecast") -> "Path | None":
    """Generic overlay: same metric, one line per source (observed / forecast)."""
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for src, g in df.groupby(group):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker=".", label=str(src))
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title); ax.legend(fontsize=8)
    return _save(fig, out_path)

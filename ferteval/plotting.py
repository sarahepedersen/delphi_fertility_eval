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
    ax.set_xlabel("Age at first birth"); ax.set_ylabel("P(no birth yet)")
    ax.set_xlim(15, 50); ax.set_ylim(0, 1.02); ax.set_title(title)
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


def plot_lexis_surface(lexis: pd.DataFrame, out_path, title="First-birth intensity (Lexis)",
                       forecast_boundary_period: int | None = None) -> "Path | None":
    """Heatmap cohort (x) × age (y) of first-birth intensity; unobserved cells left blank.

    If ``forecast_boundary_period`` (the calendar year the data ends) is given, draw the
    line ``age = period − cohort`` marking the observed/forecast frontier: everything below
    it is observed, everything above is model forecast.
    """
    if lexis.empty:
        return None
    grid = lexis.pivot_table(index="age", columns="cohort", values="intensity")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.pcolormesh(grid.columns, grid.index, grid.to_numpy(), shading="nearest")
    fig.colorbar(im, ax=ax, label="first-birth intensity")
    if forecast_boundary_period is not None:
        cohorts = grid.columns.to_numpy()
        ages = float(forecast_boundary_period) - cohorts   # age = period − cohort
        ax.plot(cohorts, ages, color="red", lw=2, label=f"data cutoff ({forecast_boundary_period}) — forecast above")
        ax.set_ylim(grid.index.min(), grid.index.max())
        ax.legend(fontsize=8, loc="lower left")
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


def plot_backtest_ccf(bt: pd.DataFrame, out_path, title="Backtest: forecast vs observed CCF by cohort") -> "Path | None":
    """Completed CCF by cohort bin: observed truth (line) vs forecast from each truncation age."""
    if bt.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    obs = bt.drop_duplicates("cohort").sort_values("cohort")  # truth is the same across truncation ages
    ax.plot(obs["cohort"], obs["ccf_observed"], "k-o", lw=2, label="observed (truth)")
    for age, g in bt.groupby("truncation_age"):
        g = g.sort_values("cohort")
        ax.plot(g["cohort"], g["ccf_forecast"], marker="s", ls="--", label=f"forecast @ trunc {int(age)}")
    ax.set_xlabel("Birth-cohort bin"); ax.set_ylabel("Completed CCF"); ax.set_title(title)
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


# =========================================================================== #
# CCF decomposition (phase 3)                                                 #
# =========================================================================== #
def plot_ccf_decomposition(trend: pd.DataFrame, out_path, title="ΔCCF vs reference cohort") -> "Path | None":
    """Grouped bars: childlessness vs family-size contribution to ΔCCF per cohort, with total."""
    if trend.empty:
        return None
    g = trend.sort_values("cohort")
    x = g["cohort"].to_numpy(); w = (x[1] - x[0]) * 0.35 if len(x) > 1 else 3
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, g["childlessness_effect"], width=w, label="childlessness", color="tab:red")
    ax.bar(x + w / 2, g["familysize_effect"], width=w, label="family size", color="tab:blue")
    ax.plot(x, g["dccf"], "k-o", lw=1.5, label="ΔCCF (total)")
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_xlabel("Birth cohort"); ax.set_ylabel(f"Contribution to ΔCCF (ref {int(g['reference'].iloc[0])})")
    ax.set_title(title); ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_childlessness_familysize(comp_obs: pd.DataFrame, comp_fc: pd.DataFrame | None, out_path,
                                  title="Childlessness & family size by cohort") -> "Path | None":
    """Two panels: childlessness P(0) and mean-children-among-mothers by cohort (observed + forecast)."""
    if comp_obs.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, col, lab in ((ax[0], "childless", "childlessness P(0)"), (ax[1], "mcm", "mean children | mother")):
        o = comp_obs.sort_values("cohort")
        a.plot(o["cohort"], o[col], "k-o", label="observed")
        if comp_fc is not None and not comp_fc.empty:
            f = comp_fc.sort_values("cohort")
            a.plot(f["cohort"], f[col], "--s", color="tab:blue", label="forecast-completed")
        a.set_xlabel("Birth cohort"); a.set_ylabel(lab); a.legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, out_path)


def plot_parity_distribution(dist_obs: pd.DataFrame, dist_fc: pd.DataFrame | None, out_path,
                             title="Parity distribution by cohort") -> "Path | None":
    """One line per final parity: share vs cohort (observed solid, forecast dashed)."""
    if dist_obs.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, g in dist_obs.groupby("parity"):
        g = g.sort_values("cohort")
        (line,) = ax.plot(g["cohort"], g["share"], marker="o", ms=3, label=f"parity {int(p)}")
        if dist_fc is not None and not dist_fc.empty:
            gf = dist_fc[dist_fc["parity"] == p].sort_values("cohort")
            ax.plot(gf["cohort"], gf["share"], ls="--", color=line.get_color())
    ax.set_xlabel("Birth cohort"); ax.set_ylabel("Share of women"); ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, title="solid=obs, dashed=forecast")
    return _save(fig, out_path)


def plot_recuperation(childless_curve: pd.DataFrame, dist: pd.DataFrame | None, out_path,
                      title="Recuperation: forecast for the nulliparous") -> "Path | None":
    """P(end childless | nulliparous at cutoff age) vs age; + forecast final-parity mix if given."""
    if childless_curve.empty:
        return None
    if dist is not None and not dist.empty:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax0, ax1 = ax
    else:
        fig, ax0 = plt.subplots(figsize=(6, 4.5)); ax1 = None
    c = childless_curve.sort_values("cutoff_age")
    ax0.plot(c["cutoff_age"], c["p_childless"], "-o", color="tab:red")
    ax0.set_xlabel("Age still nulliparous at cutoff"); ax0.set_ylabel("P(end childless)")
    ax0.set_ylim(0, 1.02); ax0.set_title("P(childless | nulliparous at age)")
    if ax1 is not None:
        nulli = dist[dist["parity_at_cutoff"] == 0]
        for cbin, g in nulli.groupby("cutoff_age"):
            g = g.sort_values("final_parity")
            ax1.plot(g["final_parity"], g["share"], marker="o", label=f"cutoff {int(cbin)}")
        ax1.set_xlabel("Forecast completed parity"); ax1.set_ylabel("Share"); ax1.legend(fontsize=7)
        ax1.set_title("Forecast parity mix | nulliparous at cutoff")
    fig.suptitle(title)
    return _save(fig, out_path)


def plot_backtest_components(bt: pd.DataFrame, out_path, title="Component backtest (forecast vs truth)") -> "Path | None":
    """Forecast vs observed childlessness, family size, and parity-distribution TVD by cohort."""
    if bt.empty:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    _bt_panel(ax[0], bt, "childless_obs", "childless_fc", "childlessness P(0)")
    _bt_panel(ax[1], bt, "mcm_obs", "mcm_fc", "mean children | mother")
    for age, g in bt.groupby("truncation_age"):
        g = g.sort_values("cohort")
        ax[2].plot(g["cohort"], g["tvd"], marker="s", ls="--", label=f"trunc {int(age)}")
    ax[2].set_xlabel("Cohort"); ax[2].set_ylabel("parity-dist TVD"); ax[2].set_title("Parity-dist distance")
    ax[2].legend(fontsize=7)
    fig.suptitle(title)
    return _save(fig, out_path)


def plot_recuperation_backtest(rec: pd.DataFrame, out_path,
                               title="Recuperation backtest (forecast vs truth)") -> "Path | None":
    """Validated recuperation: forecast vs observed P(childless | nulliparous at age) and mean
    recuperated parity, across truncation ages."""
    if rec.empty:
        return None
    r = rec.sort_values("truncation_age")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(r["truncation_age"], r["p_childless_observed"], "k-o", label="observed (truth)")
    ax[0].plot(r["truncation_age"], r["p_childless_forecast"], "--s", color="tab:red", label="forecast")
    ax[0].set_xlabel("Age still nulliparous (truncation)"); ax[0].set_ylabel("P(end childless)")
    ax[0].set_ylim(0, 1.02); ax[0].set_title("P(childless | nulliparous at age)"); ax[0].legend(fontsize=8)
    ax[1].plot(r["truncation_age"], r["mean_final_observed"], "k-o", label="observed (truth)")
    ax[1].plot(r["truncation_age"], r["mean_final_forecast"], "--s", color="tab:blue", label="forecast")
    ax[1].set_xlabel("Age still nulliparous (truncation)"); ax[1].set_ylabel("Mean completed parity")
    ax[1].set_title("Recuperated parity | nulliparous at age"); ax[1].legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, out_path)


def plot_childlessness_auc(auc: pd.DataFrame, out_path,
                           title="Childlessness predictability (AUC) by cohort") -> "Path | None":
    """AUC for predicting who ends childless (among the nulliparous), by cohort — one line per
    prediction age. A rising trend = childlessness is more predictable in later cohorts."""
    if auc.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for age, g in auc.groupby("prediction_age"):
        g = g.sort_values("cohort")
        line = ax.plot(g["cohort"], g["auc"], marker="o", label=f"predict at age {int(age)}")[0]
        if {"auc_lo", "auc_hi"}.issubset(g.columns):
            ax.fill_between(g["cohort"], g["auc_lo"], g["auc_hi"], color=line.get_color(), alpha=0.15)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xlabel("Birth cohort"); ax.set_ylabel("AUC (childless vs eventual mother)")
    ax.set_ylim(0.4, 1.0); ax.set_title(title); ax.legend(fontsize=8)
    return _save(fig, out_path)


def _bt_panel(ax, bt, obs_col, fc_col, label):
    obs = bt.drop_duplicates("cohort").sort_values("cohort")
    ax.plot(obs["cohort"], obs[obs_col], "k-o", lw=2, label="observed (truth)")
    for age, g in bt.groupby("truncation_age"):
        g = g.sort_values("cohort")
        ax.plot(g["cohort"], g[fc_col], marker="s", ls="--", label=f"forecast @ {int(age)}")
    ax.set_xlabel("Cohort"); ax.set_ylabel(label); ax.set_title(label); ax.legend(fontsize=7)

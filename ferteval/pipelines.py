"""End-to-end orchestration: load model + data, compute metrics, write tables + figures.

``run_auc`` and ``run_calibration`` are the two phase-1 entry points. Both accept a
fully-resolved :class:`EvalConfig`, and both also accept a pre-built
``(InferenceResult, TokenVocab)`` so the CLI can run inference once and reuse it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as calib
from . import decomposition as decomp
from . import demography as demog
from . import metrics_auc as M
from . import plotting
from . import sampling
from .config import EvalConfig
from .extraction import FertilityData, merge
from .inference import InferenceResult, run_inference
from .loaders import load_data, load_model
from .vocab import TokenVocab


# --------------------------------------------------------------------------- #
# shared setup                                                                 #
# --------------------------------------------------------------------------- #
def prepare(cfg: EvalConfig) -> tuple[InferenceResult, TokenVocab]:
    """Load vocab + model + data and run the (single) forward pass."""
    cfg.require_paths("delphi_repo", "ckpt", "data")
    vocab = TokenVocab.from_config(cfg)
    bundle = load_model(cfg)
    result = run_inference(cfg, bundle, vocab)
    return result, vocab


def resolve_cohort_edges(cfg: EvalConfig, cohort: np.ndarray) -> list[int]:
    """Cohort bracket edges from config, or a single catch-all bin if no cohort tokens resolved."""
    valid = cohort[cohort >= 0]
    if valid.size == 0:
        return [-2, -1]  # single bin capturing the -1 'missing cohort' sentinel
    return cfg.bins.cohort.edge_list()


# --------------------------------------------------------------------------- #
# AUC                                                                          #
# --------------------------------------------------------------------------- #
def run_auc(cfg: EvalConfig, result: InferenceResult | None = None, vocab: TokenVocab | None = None) -> dict:
    if result is None or vocab is None:
        result, vocab = prepare(cfg)

    if vocab.no_event_id is None:
        raise ValueError("no_event token is required for AUC risk sets; set tokens.no_event / token_ids.no_event.")

    rng = np.random.default_rng(cfg.metrics.seed)
    age_edges = cfg.bins.age.edges()
    cohort_edges = resolve_cohort_edges(cfg, result.cohort)
    any_child = result.any_child_score()

    events = _auc_event_specs(cfg, vocab)
    unpooled_frames, pooled_records = [], []
    for name, builder in events:
        rows = builder(result, any_child)
        if len(rows) == 0:
            continue
        df_sub, boots = M.subgroup_auc(rows, age_edges, cohort_edges, rng, n_bootstrap=cfg.metrics.n_bootstrap)
        if df_sub.empty:
            continue
        pooled = M.aggregate_bootstrap(df_sub, boots)
        df_sub.insert(0, "event", name)
        unpooled_frames.append(df_sub)
        if not pooled.empty:
            pooled.insert(0, "event", name)
            pooled_records.append(pooled)

    df_unpooled = pd.concat(unpooled_frames, ignore_index=True) if unpooled_frames else pd.DataFrame()
    df_pooled = pd.concat(pooled_records, ignore_index=True) if pooled_records else pd.DataFrame()

    out = Path(cfg.paths.out)
    _write_table(df_unpooled, out / "auc_subgroups")
    _write_table(df_pooled, out / "auc_pooled")
    if not df_unpooled.empty:
        for name, g in df_unpooled.groupby("event"):
            plotting.plot_auc_by_age(g, out / f"auc_by_age__{_slug(name)}.png", title=f"AUC by age — {name}")
            plotting.plot_auc_by_cohort(g, out / f"auc_by_cohort__{_slug(name)}.png", title=f"AUC by cohort — {name}")

    return {"subgroups": df_unpooled, "pooled": df_pooled}


def _auc_event_specs(cfg: EvalConfig, vocab: TokenVocab):
    """List of (event_name, builder(result, any_child_score) -> EventRows)."""
    specs = []
    no_event_id = vocab.no_event_id

    if cfg.metrics.auc_first_birth:
        specs.append(("first_birth (0->1)",
                      lambda r, s, n=0: M.build_parity_transition_rows(r, s, n, no_event_id)))

    if cfg.metrics.auc_parity_progression:
        for n in range(1, cfg.metrics.max_parity + 1):
            specs.append((f"progression ({n}->{n + 1})",
                          lambda r, s, n=n: M.build_parity_transition_rows(r, s, n, no_event_id)))

    if cfg.metrics.auc_child_sex and vocab.child_son_id is not None and vocab.child_daughter_id is not None:
        son, dau = vocab.child_son_id, vocab.child_daughter_id
        specs.append(("child_sex (son vs daughter)",
                      lambda r, _s, son=son, dau=dau: M.build_child_sex_rows(r, son, dau)))

    return specs


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
def run_calibration(cfg: EvalConfig, result: InferenceResult | None = None, vocab: TokenVocab | None = None) -> dict:
    if result is None or vocab is None:
        result, vocab = prepare(cfg)
    if vocab.no_event_id is None:
        raise ValueError("no_event token is required for calibration; set tokens.no_event / token_ids.no_event.")

    n_bins = cfg.metrics.calibration_n_bins
    cohort_edges = resolve_cohort_edges(cfg, result.cohort)

    rows = calib.build_calibration_rows(result, vocab.no_event_id)
    reliability = calib.reliability_curve(rows.pred, rows.label, n_bins)
    ece = calib.expected_calibration_error(rows.pred, rows.label, n_bins)
    slope, intercept = calib.calibration_slope_intercept(rows.pred, rows.label)
    per_cohort = calib.per_cohort_calibration(rows, cohort_edges, n_bins)

    overall = pd.DataFrame([{
        "n_woman_years": len(rows),
        "n_births": int(rows.label.sum()),
        "mean_pred": float(rows.pred.mean()) if len(rows) else float("nan"),
        "mean_obs": float(rows.label.mean()) if len(rows) else float("nan"),
        "ece": ece,
        "slope": slope,
        "intercept": intercept,
    }])

    out = Path(cfg.paths.out)
    _write_table(reliability, out / "reliability")
    _write_table(per_cohort, out / "calibration_by_cohort")
    _write_table(overall, out / "calibration_overall")

    plotting.plot_reliability(reliability, out / "reliability.png",
                              title=f"Calibration (ECE={ece:.3f}, slope={slope:.2f})")
    cohort_curves = {
        f"cohort {int(c)}": calib.reliability_curve(rows.pred[rows.cohort == c], rows.label[rows.cohort == c], n_bins)
        for c in np.unique(rows.cohort[rows.cohort >= 0])
    }
    plotting.plot_reliability_by_cohort(cohort_curves, out / "reliability_by_cohort.png")

    return {"reliability": reliability, "by_cohort": per_cohort, "overall": overall}


# --------------------------------------------------------------------------- #
# Demography (observed)                                                         #
# --------------------------------------------------------------------------- #
def run_demography(cfg: EvalConfig, fd: FertilityData | None = None) -> dict:
    """Extract observed FertilityData, compute every estimator, write tables + figures."""
    if fd is None:
        cfg.require_paths("data")
        vocab = TokenVocab.from_config(cfg)
        data = load_data(cfg.paths.data)
        fd = FertilityData.from_bin(
            data, vocab,
            completion_age=cfg.demography.completion_age,
            repro_ages=(cfg.demography.repro_age_min, cfg.demography.repro_age_max),
        )

    dg = cfg.demography
    cohort_edges = cfg.bins.cohort.edge_list()
    tables = demog.run_all(fd, dg.selected_years, dg.max_parity, cohort_edges,
                           cohort_min=dg.cohort_min, period_min=dg.resolved_period_min(),
                           period_max=dg.period_max, mab_by=dg.mab_by)

    out = Path(cfg.paths.out) / "demography"
    for name, df in tables.items():
        _write_table(df, out / name)

    # figures — per-plot display cutoffs (tables written above stay complete)
    asfr_plot = tables["asfr"]
    if dg.asfr_exclude_periods:
        asfr_plot = asfr_plot[~asfr_plot["period"].isin(dg.asfr_exclude_periods)]
    plotting.plot_ccf_by_cohort(_cap(tables["ccf_curve"], "cohort", dg.ccf_max_cohort), out / "ccf_by_cohort.png")
    plotting.plot_ppr(_cap(tables["parity_progression_ratios"], "cohort", dg.ppr_max_cohort), out / "ppr.png")
    plotting.plot_km_survival(_cap(tables["time_to_first_birth"], "cohort", dg.ttfb_max_cohort),
                              out / "time_to_first_birth.png")
    plotting.plot_mab1_timeseries(tables["mean_age_first_birth"], out / "mean_age_first_birth.png", x=dg.mab_by)
    plotting.plot_asfr(asfr_plot, out / "asfr.png")
    plotting.plot_age_parity_surface(tables["age_parity_surface"], out)
    plotting.plot_birth_order_age_profile(tables["birth_order_age_profile"], out / "birth_order_age_profile.png")
    plotting.plot_lexis_surface(tables["lexis_first_birth"], out / "lexis_first_birth.png")

    return {"fertility_data": fd, "tables": tables}


# --------------------------------------------------------------------------- #
# Forecasting: complete incomplete cohorts + backtest                          #
# --------------------------------------------------------------------------- #
def run_forecast(cfg: EvalConfig, bundle=None, vocab: TokenVocab | None = None) -> dict:
    """Complete incomplete cohorts with the model, fill the CCF tail / Lexis corner, and
    backtest forecast accuracy on already-completed cohorts."""
    cfg.require_paths("delphi_repo", "ckpt", "data")
    vocab = vocab or TokenVocab.from_config(cfg)
    bundle = bundle or load_model(cfg)
    data = load_data(cfg.paths.data)
    dg = cfg.demography
    fc = sampling.ForecastConfig.from_cfg(cfg)

    observed = FertilityData.from_bin(data, vocab, dg.completion_age, (dg.repro_age_min, dg.repro_age_max))
    forecast = sampling.forecast_incomplete(cfg, bundle, vocab, observed, data, fc)
    completed = merge(observed, forecast)

    out = Path(cfg.paths.out) / "forecast"
    cohort_edges = cfg.bins.cohort.edge_list()
    tables = demog.run_all(completed, dg.selected_years, dg.max_parity, cohort_edges,
                           cohort_min=dg.cohort_min, period_min=dg.resolved_period_min(),
                           period_max=dg.period_max, mab_by=dg.mab_by)
    for name, df in tables.items():
        _write_table(df, out / f"completed__{name}")

    # observed → completed CCF overlay (forecast fills the dashed tail), by cohort bin.
    # Respect cohort_min, and drop cohorts born after max_display_cohort (too early to forecast).
    cap = cfg.forecast.max_display_cohort
    obs_c = observed.filter_cohorts(min_cohort=dg.cohort_min).binned_cohorts(cohort_edges)
    full_c = completed.filter_cohorts(min_cohort=dg.cohort_min).binned_cohorts(cohort_edges)
    ccf_obs = demog.ccf_curve(obs_c)
    ccf_full = demog.ccf_curve(full_c)
    if cap is not None:
        ccf_obs = ccf_obs[ccf_obs["cohort"] <= cap]
        ccf_full = ccf_full[ccf_full["cohort"] <= cap]
    plotting.plot_ccf_completed(ccf_obs, ccf_full, out / "ccf_observed_vs_completed.png")

    # Lexis surface with a line marking the data cutoff (forecast beyond it). The frontier is
    # the calendar year the data ends = the latest observed period (cohort + age).
    boundary_period = int(observed.exposure["period"].max()) if len(observed.exposure) else None
    plotting.plot_lexis_surface(tables["lexis_first_birth"], out / "lexis_completed.png",
                                title="First-birth intensity (observed + forecast)",
                                forecast_boundary_period=boundary_period)

    backtest = _backtest(cfg, bundle, vocab, data, observed, fc)
    _write_table(backtest, out / "backtest_ccf")
    plotting.plot_backtest_ccf(backtest, out / "backtest_ccf.png")

    return {"observed": observed, "forecast": forecast, "completed": completed,
            "tables": tables, "backtest": backtest}


def _backtest(cfg: EvalConfig, bundle, vocab: TokenVocab, data, observed: FertilityData,
              fc) -> pd.DataFrame:
    """Truncate completed cohorts at each cutoff age, forecast forward, compare CCF to truth.

    Results are grouped into cohort bins of width ``forecast.backtest_cohort_step`` (default
    5 years) rather than per individual birth year.
    """
    dg = cfg.demography
    step = max(1, cfg.forecast.backtest_cohort_step)
    cols = ["truncation_age", "cohort", "ccf_observed", "ccf_forecast", "error"]

    per_cohort = demog.completed_cohort_fertility(observed)
    complete_cohorts = set(per_cohort.loc[per_cohort["fully_observed"], "cohort"])
    if not complete_cohorts:
        return pd.DataFrame(columns=cols)
    lo = (min(complete_cohorts) // step) * step
    hi = (max(complete_cohorts) // step + 1) * step
    edges = list(range(int(lo), int(hi) + 1, step))

    w = observed.women
    complete_women = w[((~w["censored"]) | (w["exit_age"] >= dg.completion_age)) & w["cohort"].isin(complete_cohorts)]
    ids = complete_women["woman_id"].tolist()
    if not ids:
        return pd.DataFrame(columns=cols)

    # observed truth per cohort bin (restricted to the completed women)
    obs_truth = demog.completed_cohort_fertility(observed.subset(ids).binned_cohorts(edges)).set_index("cohort")["ccf"]

    rows = []
    for k, age in enumerate(cfg.forecast.backtest_truncation_ages):
        seeds = sampling.build_seeds(data, vocab, woman_ids=ids, truncate_age=age)
        rng = np.random.default_rng(fc.seed + k + 1)
        trajs = sampling.forecast_sequences(bundle, vocab, seeds, fc, rng)
        fcfd = FertilityData.from_sequences(trajs, vocab, dg.completion_age, (dg.repro_age_min, dg.repro_age_max))
        fc_ccf = demog.completed_cohort_fertility(fcfd.binned_cohorts(edges)).set_index("cohort")["ccf"]
        for cbin in obs_truth.index:
            if cbin in fc_ccf.index:
                o, f = float(obs_truth[cbin]), float(fc_ccf[cbin])
                rows.append({"truncation_age": age, "cohort": int(cbin),
                             "ccf_observed": o, "ccf_forecast": f, "error": f - o})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# CCF decomposition: childlessness vs family size, recuperation, component backtest #
# --------------------------------------------------------------------------- #
def run_decomposition(cfg: EvalConfig, bundle=None, vocab: TokenVocab | None = None) -> dict:
    """Decompose the (forecast-completed) CCF trend into childlessness vs family size, probe
    recuperation from the model's state-conditional forecasts, and backtest the breakdown."""
    cfg.require_paths("delphi_repo", "ckpt", "data")
    vocab = vocab or TokenVocab.from_config(cfg)
    bundle = bundle or load_model(cfg)
    data = load_data(cfg.paths.data)
    dg, dc = cfg.demography, cfg.decomposition
    fc = sampling.ForecastConfig.from_cfg(cfg)
    cohort_edges = cfg.bins.cohort.edge_list()

    observed = FertilityData.from_bin(data, vocab, dg.completion_age, (dg.repro_age_min, dg.repro_age_max))
    observed = observed.filter_cohorts(min_cohort=dg.cohort_min)

    # forecast the incomplete women (tracking each seed's state at the cutoff)
    seeds = sampling.build_seeds(data, vocab, woman_ids=observed.incomplete_women["woman_id"].tolist())
    states = sampling.seed_states(seeds, vocab)
    trajs = sampling.forecast_sequences(bundle, vocab, seeds, fc, np.random.default_rng(fc.seed))
    forecast_fd = FertilityData.from_sequences(trajs, vocab, dg.completion_age, (dg.repro_age_min, dg.repro_age_max))
    completed = merge(observed, forecast_fd)

    # components on observed (completed cohorts only) and on the completed (obs + forecast) view
    obs_bin = observed.binned_cohorts(cohort_edges)
    full_bin = completed.binned_cohorts(cohort_edges)
    obs_comp = decomp.parity_components(obs_bin, dc.max_parity)
    full_comp = decomp.parity_components(full_bin, dc.max_parity)
    trend = decomp.cohort_trend_attribution(full_comp, dc.reference_cohort, dc.max_parity)

    # recuperation from the model's state-conditional forecasts (genuinely incomplete women)
    rec_dist, rec_curve = decomp.recuperation_table(forecast_fd, states, fc.n_samples,
                                                    dc.recuperation_age_bins, dc.max_parity)
    # component + validated recuperation backtest + childlessness predictability (one forecast pass)
    bt = _backtest_components(cfg, bundle, vocab, data, observed, fc)
    backtest, rec_backtest, childless_auc = bt["components"], bt["recuperation"], bt["childlessness_auc"]

    out = Path(cfg.paths.out) / "decomposition"
    for name, df in [("components_observed", obs_comp), ("components_completed", full_comp),
                     ("ccf_trend_attribution", trend), ("recuperation_parity_mix", rec_dist),
                     ("recuperation_childless_curve", rec_curve), ("backtest_components", backtest),
                     ("recuperation_backtest", rec_backtest), ("childlessness_auc", childless_auc)]:
        _write_table(df, out / name)

    dist_obs = demog.parity_distribution(obs_bin, dc.max_parity)
    dist_full = demog.parity_distribution(full_bin, dc.max_parity)
    plotting.plot_ccf_decomposition(trend, out / "ccf_decomposition.png")
    plotting.plot_childlessness_familysize(obs_comp, full_comp, out / "childlessness_familysize.png")
    plotting.plot_parity_distribution(dist_obs, dist_full, out / "parity_distribution.png")
    plotting.plot_recuperation(rec_curve, rec_dist, out / "recuperation.png")
    plotting.plot_backtest_components(backtest, out / "backtest_components.png")
    plotting.plot_recuperation_backtest(rec_backtest, out / "recuperation_backtest.png")
    plotting.plot_childlessness_auc(childless_auc, out / "childlessness_auc.png")

    return {"observed": observed, "completed": completed, "components_observed": obs_comp,
            "components_completed": full_comp, "trend": trend, "recuperation": (rec_dist, rec_curve),
            "backtest": backtest, "recuperation_backtest": rec_backtest, "childlessness_auc": childless_auc}


def _backtest_components(cfg: EvalConfig, bundle, vocab: TokenVocab, data, observed: FertilityData,
                         fc) -> dict:
    """Backtest the decomposition, the recuperation curve, and childlessness predictability.

    Returns a dict of DataFrames:
      * ``components`` — forecast vs held-out truth for childlessness, family size, mean age at
        first birth, PPRs, and parity-distribution TVD, per (cohort bin × truncation age).
      * ``recuperation`` — validated recuperation: forecast vs observed
        ``P(childless | nulliparous at age)`` and mean recuperated parity.
      * ``childlessness_auc`` — AUC for predicting who ends childless, among women nulliparous
        at each prediction age, per cohort (score = forecast P(childless), label = actual
        childlessness). Comparing AUC across cohorts answers "is childlessness more predictable
        in later cohorts?".
    """
    dg, dc = cfg.demography, cfg.decomposition
    step = max(1, cfg.forecast.backtest_cohort_step)
    cols = ["truncation_age", "cohort", "ccf_obs", "ccf_fc", "childless_obs", "childless_fc",
            "mcm_obs", "mcm_fc", "mab_obs", "mab_fc", "a1_obs", "a1_fc", "a2_obs", "a2_fc", "tvd"]
    rec_cols = ["truncation_age", "n_women", "p_childless_forecast", "p_childless_observed",
                "mean_final_forecast", "mean_final_observed"]

    empty = {"components": pd.DataFrame(columns=cols), "recuperation": pd.DataFrame(columns=rec_cols),
             "childlessness_auc": pd.DataFrame()}
    per_cohort = demog.completed_cohort_fertility(observed)
    complete_cohorts = set(per_cohort.loc[per_cohort["fully_observed"], "cohort"])
    if not complete_cohorts:
        return empty
    edges = list(range((min(complete_cohorts) // step) * step, (max(complete_cohorts) // step + 1) * step + 1, step))
    w = observed.women
    ids = w[((~w["censored"]) | (w["exit_age"] >= dg.completion_age)) & w["cohort"].isin(complete_cohorts)]["woman_id"].tolist()
    if not ids:
        return empty

    obs_fd = observed.subset(ids).binned_cohorts(edges)
    obs_comp = decomp.parity_components(obs_fd, dc.max_parity).set_index("cohort")
    obs_mab = demog.mean_age_first_birth(obs_fd, by="cohort").set_index("cohort")["mean_age"]
    obs_final = observed.women.set_index("woman_id")["final_parity"]  # held-out truth per woman

    rows, rec_rows, predict_rows = [], [], []
    ns = fc.n_samples

    # Build every truncation-age's seeds up front and roll them ALL forward in ONE batched pass
    all_seeds, age_ranges, states_by_age = [], {}, {}
    for age in cfg.forecast.backtest_truncation_ages:
        seeds_a = sampling.build_seeds(data, vocab, woman_ids=ids, truncate_age=age)
        age_ranges[age] = (len(all_seeds), len(all_seeds) + len(seeds_a))
        all_seeds.extend(seeds_a)
        states_by_age[age] = sampling.seed_states(seeds_a, vocab)
    if not all_seeds:
        return empty
    all_trajs = sampling.forecast_sequences(bundle, vocab, all_seeds, fc, np.random.default_rng(fc.seed + 1))

    for age in cfg.forecast.backtest_truncation_ages:
        s0, s1 = age_ranges[age]
        r0 = s0 * ns
        trajs = [(tid - r0, tok, ag) for (tid, tok, ag) in all_trajs[s0 * ns:s1 * ns]]  # re-id to 0..
        states = states_by_age[age]
        fcfd_raw = FertilityData.from_sequences(trajs, vocab, dg.completion_age, (dg.repro_age_min, dg.repro_age_max))
        fcfd = fcfd_raw.binned_cohorts(edges)
        fc_comp = decomp.parity_components(fcfd, dc.max_parity).set_index("cohort")
        fc_mab = demog.mean_age_first_birth(fcfd, by="cohort").set_index("cohort")["mean_age"]
        tvd = decomp.parity_dist_tvd(obs_fd, fcfd, dc.max_parity).set_index("cohort")["tvd"]
        for cbin in obs_comp.index:
            if cbin not in fc_comp.index:
                continue
            o, f = obs_comp.loc[cbin], fc_comp.loc[cbin]
            rows.append({"truncation_age": age, "cohort": int(cbin),
                         "ccf_obs": o["ccf"], "ccf_fc": f["ccf"],
                         "childless_obs": o["childless"], "childless_fc": f["childless"],
                         "mcm_obs": o["mcm"], "mcm_fc": f["mcm"],
                         "mab_obs": float(obs_mab.get(cbin, np.nan)), "mab_fc": float(fc_mab.get(cbin, np.nan)),
                         "a1_obs": o["a1"], "a1_fc": f["a1"], "a2_obs": o["a2"], "a2_fc": f["a2"],
                         "tvd": float(tvd.get(cbin, np.nan))})

        # women nulliparous at this prediction age (the informative risk set)
        nulli_seed_idx = states.index[states["parity_at_cutoff"] == 0]
        if len(nulli_seed_idx):
            nulli = states.loc[nulli_seed_idx]
            nulli_ids = nulli["woman_id"]
            fw = fcfd_raw.women[["woman_id", "final_parity"]].copy()
            fw["seed_idx"] = fw["woman_id"] // fc.n_samples
            # per-woman forecast P(childless) = fraction of her samples ending childless
            p_childless = fw.groupby("seed_idx")["final_parity"].apply(lambda s: float((s == 0).mean()))

            # (a) aggregate validated recuperation curve
            fc_nulli = fw[fw["seed_idx"].isin(nulli_seed_idx)]["final_parity"]
            obs_nulli = obs_final.reindex(nulli_ids).dropna()
            rec_rows.append({"truncation_age": age, "n_women": int(len(nulli_ids)),
                             "p_childless_forecast": float((fc_nulli == 0).mean()),
                             "p_childless_observed": float((obs_nulli == 0).mean()),
                             "mean_final_forecast": float(fc_nulli.mean()),
                             "mean_final_observed": float(obs_nulli.mean())})

            # (b) per-woman predictability rows: forecast P(childless) vs actual childlessness, by cohort
            predict_rows.append(pd.DataFrame({
                "prediction_age": age,
                "cohort": _bin_lower(nulli["cohort"].to_numpy(), edges),
                "score": nulli.index.to_series().map(p_childless).to_numpy(),
                "label": (obs_final.reindex(nulli_ids).to_numpy() == 0).astype(int),
            }))

    predict = pd.concat(predict_rows, ignore_index=True) if predict_rows else pd.DataFrame(
        columns=["prediction_age", "cohort", "score", "label"])
    auc = decomp.childlessness_auc(predict, n_bootstrap=min(cfg.metrics.n_bootstrap, 200), seed=cfg.metrics.seed)
    return {"components": pd.DataFrame(rows, columns=cols),
            "recuperation": pd.DataFrame(rec_rows, columns=rec_cols),
            "childlessness_auc": auc}


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
# --------------------------------------------------------------------------- #
def _cap(df: pd.DataFrame, col: str, cap: int | None) -> pd.DataFrame:
    """Display filter: keep rows with ``df[col] <= cap`` (no-op when cap is None)."""
    return df[df[col] <= cap] if (cap is not None and not df.empty and col in df.columns) else df


def _bin_lower(values: np.ndarray, edges: list[int]) -> np.ndarray:
    """Map each value to the lower edge of its bin (same convention as binned_cohorts)."""
    e = np.asarray(edges)
    return e[np.clip(np.digitize(values, e) - 1, 0, len(e) - 2)]


def _write_table(df: pd.DataFrame, path_stem: Path) -> None:
    """Write CSV always, plus parquet when an engine is available."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path_stem.with_suffix(".parquet"), index=False)
    except Exception:
        pass  # pyarrow/fastparquet not installed — CSV is enough


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")

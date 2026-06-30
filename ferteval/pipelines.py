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
                      lambda r, s, son=son, dau=dau: M.build_child_sex_rows(r, son, dau)))

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
    tables = demog.run_all(fd, dg.selected_years, dg.max_parity)

    out = Path(cfg.paths.out) / "demography"
    for name, df in tables.items():
        _write_table(df, out / name)

    # figures
    plotting.plot_ccf_by_cohort(tables["ccf_curve"], out / "ccf_by_cohort.png")
    plotting.plot_ppr(tables["parity_progression_ratios"], out / "ppr.png")
    plotting.plot_km_survival(tables["time_to_first_birth"], out / "time_to_first_birth.png")
    plotting.plot_mab1_timeseries(tables["mean_age_first_birth"], out / "mean_age_first_birth.png")
    plotting.plot_asfr(tables["asfr"], out / "asfr.png")
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
    tables = demog.run_all(completed, dg.selected_years, dg.max_parity)
    for name, df in tables.items():
        _write_table(df, out / f"completed__{name}")

    # observed → completed CCF overlay (forecast fills the dashed tail)
    ccf_obs = demog.ccf_curve(observed)
    ccf_full = demog.ccf_curve(completed)
    plotting.plot_ccf_completed(ccf_obs, ccf_full, out / "ccf_observed_vs_completed.png")
    plotting.plot_lexis_surface(tables["lexis_first_birth"], out / "lexis_completed.png",
                                title="First-birth intensity (observed + forecast)")

    backtest = _backtest(cfg, bundle, vocab, data, observed, fc)
    _write_table(backtest, out / "backtest_ccf")
    plotting.plot_backtest_ccf(backtest, out / "backtest_ccf.png")

    return {"observed": observed, "forecast": forecast, "completed": completed,
            "tables": tables, "backtest": backtest}


def _backtest(cfg: EvalConfig, bundle, vocab: TokenVocab, data, observed: FertilityData,
              fc) -> pd.DataFrame:
    """Truncate completed cohorts at each cutoff age, forecast forward, compare CCF to truth."""
    dg = cfg.demography
    obs_ccf = demog.completed_cohort_fertility(observed)
    complete_cohorts = set(obs_ccf.loc[obs_ccf["fully_observed"], "cohort"])
    obs_lookup = obs_ccf.set_index("cohort")["ccf"]
    w = observed.women
    complete_women = w[((~w["censored"]) | (w["exit_age"] >= dg.completion_age)) & w["cohort"].isin(complete_cohorts)]

    rows = []
    for k, age in enumerate(cfg.forecast.backtest_truncation_ages):
        ids = complete_women["woman_id"].tolist()
        if not ids:
            continue
        seeds = sampling.build_seeds(data, vocab, woman_ids=ids, truncate_age=age)
        rng = np.random.default_rng(fc.seed + k + 1)
        trajs = sampling.forecast_sequences(bundle, vocab, seeds, fc, rng)
        fcfd = FertilityData.from_sequences(trajs, vocab, dg.completion_age, (dg.repro_age_min, dg.repro_age_max))
        fc_ccf = demog.completed_cohort_fertility(fcfd).set_index("cohort")["ccf"]
        for cohort in sorted(complete_cohorts):
            if cohort in fc_ccf.index:
                obs_v, fc_v = float(obs_lookup[cohort]), float(fc_ccf[cohort])
                rows.append({"truncation_age": age, "cohort": int(cohort),
                             "ccf_observed": obs_v, "ccf_forecast": fc_v, "error": fc_v - obs_v})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
# --------------------------------------------------------------------------- #
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

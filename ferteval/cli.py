"""Command-line interface.

Subcommands:
  auc           — discrimination (AUC) by age x cohort, for first-birth / parity transitions
  calibration   — predicted-vs-observed reliability, ECE, per-cohort calibration
  all           — run both (single forward pass, reused)
  build-meta    — generate meta.pkl (stoi/itos) from a Delphi labels.csv

Examples:
  python -m ferteval.cli auc --delphi-repo ~/Delphi --ckpt ckpt.pt \
      --data val.bin --meta meta.pkl --out reports/run1
  python -m ferteval.cli build-meta --labels labels.csv --out meta.pkl
"""

from __future__ import annotations

import argparse

from . import build_meta
from .config import EvalConfig


# --------------------------------------------------------------------------- #
# argument plumbing                                                            #
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Optional user config YAML (merged over defaults)")
    p.add_argument("--delphi-repo", default=None, help="Path to your Delphi fork (model.py + utils.py)")
    p.add_argument("--ckpt", default=None, help="Path to the trained checkpoint (.pt)")
    p.add_argument("--data", default=None, help="Path to eval sequences (.bin)")
    p.add_argument("--meta", default=None, help="Path to meta.pkl (token name<->id)")
    p.add_argument("--labels-csv", default=None, help="Optional labels CSV with token metadata")
    p.add_argument("--out", default=None, help="Output directory for tables + figures")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    p.add_argument("--dataset-subset-size", type=int, default=None, help="Number of patients (-1 = all)")
    p.add_argument("--n-bootstrap", type=int, default=None, help="bootstrap replicates for AUC SE + CIs (CPU)")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-event-token-rate", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None, help="forecast: Monte-Carlo trajectories per woman")
    p.add_argument("--seed", type=int, default=None)


def _overrides_from_args(args: argparse.Namespace) -> dict:
    paths, inference, metrics, forecast = {}, {}, {}, {}
    if args.delphi_repo is not None: paths["delphi_repo"] = args.delphi_repo
    if args.ckpt is not None: paths["ckpt"] = args.ckpt
    if args.data is not None: paths["data"] = args.data
    if args.meta is not None: paths["meta"] = args.meta
    if args.labels_csv is not None: paths["labels_csv"] = args.labels_csv
    if args.out is not None: paths["out"] = args.out
    if args.device is not None: inference["device"] = args.device
    if args.dataset_subset_size is not None: inference["dataset_subset_size"] = args.dataset_subset_size
    if args.block_size is not None: inference["block_size"] = args.block_size
    if args.batch_size is not None: inference["batch_size"] = args.batch_size
    if args.no_event_token_rate is not None: inference["no_event_token_rate"] = args.no_event_token_rate
    if args.n_bootstrap is not None: metrics["n_bootstrap"] = args.n_bootstrap
    if args.seed is not None: metrics["seed"] = args.seed
    if getattr(args, "n_samples", None) is not None: forecast["n_samples"] = args.n_samples
    out = {}
    if paths: out["paths"] = paths
    if inference: out["inference"] = inference
    if metrics: out["metrics"] = metrics
    if forecast: out["forecast"] = forecast
    return out


def _load_cfg(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig.load(args.config, _overrides_from_args(args))


# --------------------------------------------------------------------------- #
# command handlers                                                             #
# --------------------------------------------------------------------------- #
def _cmd_auc(args):
    from . import pipelines
    cfg = _load_cfg(args)
    res = pipelines.run_auc(cfg)
    _print_pooled(res.get("pooled"))


def _cmd_calibration(args):
    from . import pipelines
    cfg = _load_cfg(args)
    res = pipelines.run_calibration(cfg)
    print(res["overall"].to_string(index=False))


def _cmd_demography(args):
    from . import pipelines
    cfg = _load_cfg(args)
    res = pipelines.run_demography(cfg)
    tables = res["tables"]
    ccf = tables["completed_cohort_fertility"]
    if not ccf.empty:
        print("Completed cohort fertility (final CCF by cohort):")
        print(ccf.to_string(index=False))
    print(f"\nWrote demography tables + figures to {cfg.paths.out}/demography")


def _cmd_forecast(args):
    from . import pipelines
    cfg = _load_cfg(args)
    res = pipelines.run_forecast(cfg)
    bt = res["backtest"]
    if not bt.empty:
        mae = bt["error"].abs().mean()
        print(f"Backtest CCF mean abs error: {mae:.3f} over {len(bt)} (cohort × truncation) cells")
    print(f"Wrote forecast (completed + backtest) tables + figures to {cfg.paths.out}/forecast")


def _cmd_decompose(args):
    from . import pipelines
    cfg = _load_cfg(args)
    res = pipelines.run_decomposition(cfg)
    trend = res["trend"]
    if not trend.empty:
        latest = trend.sort_values("cohort").iloc[-1]
        print(f"ΔCCF vs cohort {int(latest['reference'])} for cohort {int(latest['cohort'])}: "
              f"{latest['dccf']:+.3f}  = childlessness {latest['childlessness_effect']:+.3f} "
              f"+ family-size {latest['familysize_effect']:+.3f}")
    bt = res["backtest"]
    if not bt.empty:
        print(f"Component backtest: childlessness MAE {(bt['childless_fc'] - bt['childless_obs']).abs().mean():.3f}, "
              f"parity-dist TVD mean {bt['tvd'].mean():.3f}")
    ca = res["childlessness_auc"]
    if not ca.empty and ca["auc"].notna().any():
        print("Childlessness predictability AUC (by cohort, oldest→newest):")
        for age, g in ca.groupby("prediction_age"):
            g = g.sort_values("cohort")
            aucs = ", ".join(f"{int(c)}:{a:.2f}" for c, a in zip(g["cohort"], g["auc"]) if a == a)
            print(f"  predict@{int(age)}: {aucs}")
    print(f"Wrote decomposition tables + figures to {cfg.paths.out}/decomposition")


def _cmd_all(args):
    from . import pipelines
    cfg = _load_cfg(args)
    result, vocab = pipelines.prepare(cfg)
    auc_res = pipelines.run_auc(cfg, result, vocab)
    cal_res = pipelines.run_calibration(cfg, result, vocab)
    _print_pooled(auc_res.get("pooled"))
    print(cal_res["overall"].to_string(index=False))


def _cmd_build_meta(args):
    build_meta.run(args)


def _print_pooled(pooled):
    if pooled is None or pooled.empty:
        print("No AUC results (empty risk sets — check token resolution and subgroup bins).")
        return
    cols = [c for c in ("event", "auc", "auc_se", "n_diseased", "n_healthy", "n_subgroups") if c in pooled.columns]
    print(pooled[cols].to_string(index=False))


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ferteval", description="Delphi fertility evaluation suite")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("auc", _cmd_auc), ("calibration", _cmd_calibration), ("all", _cmd_all),
                          ("demography", _cmd_demography), ("forecast", _cmd_forecast),
                          ("decompose", _cmd_decompose)):
        p = sub.add_parser(name, help=f"run {name}")
        _add_common(p)
        p.set_defaults(func=handler)

    p_meta = sub.add_parser("build-meta", help="generate meta.pkl from a labels.csv")
    build_meta.add_arguments(p_meta)
    p_meta.set_defaults(func=_cmd_build_meta)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

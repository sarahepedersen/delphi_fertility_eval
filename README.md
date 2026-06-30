# ferteval — Delphi fertility model evaluation suite

Evaluation suite for Delphi-2M-style autoregressive models of **female fertility
sequences** (`BIRTH_year` cohort token → stream of `CHILD` events with yearly `no_event`
tokens, plus `padding` / `censoring` / `death`). It adapts the upstream
[gerstung-lab/Delphi](https://github.com/gerstung-lab/Delphi) `evaluate_auc.py` machinery
to fertility and adds proper cohort-subgroup calibration.

**Phase 1 (implemented):** discrimination (AUC) and predicted-vs-observed calibration.
**Phase 2 (interfaces reserved in `demography.py` / `sampling.py`):** completed cohort
fertility, parity progression ratios, ASFR, etc., computed identically on real and
model-forecasted sequences.

## Install

Use the conda env that already has your Delphi fork's deps (torch, numpy, pandas, ...):

```bash
conda activate delphi
pip install -e .            # or: pip install -e '.[parquet,test]'
```

`torch` is an *optional* dependency on purpose — the pure-numpy metrics are unit-testable
without it — but you need it (in the same env as your fork) to load checkpoints.

## Quickstart

```bash
# 1. (once) build meta.pkl token name<->id map from your Delphi labels.csv
python -m ferteval.cli build-meta --labels labels.csv --out meta.pkl

# 2. discrimination + calibration in one pass
python -m ferteval.cli all \
    --delphi-repo /path/to/your/Delphi \   # contains model.py + utils.py
    --ckpt /path/to/ckpt.pt \
    --data /path/to/val.bin \
    --meta meta.pkl \
    --out reports/run1
```

Subcommands: `auc`, `calibration`, `all`, `build-meta`. Run `--help` on any of them.
Anything in `configs/fertility_default.yaml` can be overridden on the CLI or via
`--config my.yaml`.

## What it computes

**AUC (`metrics_auc.py`)** — a parity-conditional, discrete-time-hazard AUC stratified by
**age × birth cohort** (replacing Delphi's age × sex split, since the population is
female-only). For the transition `n → n+1` the risk set is women who reached parity `n`;
positives are the `n+1`-th birth woman-years, negatives are the `no_event` years of women
who reached parity `n` but did not progress. `n=0` is the classic first-birth onset AUC.
Uncertainty via a **nonparametric CPU bootstrap**: per-subgroup SE + percentile CI, and a
stratified bootstrap for the pooled estimate. Control replicates with `--n-bootstrap`.

**Calibration (`calibration.py`)** — each valid woman-year gets a predicted
`P(next token is a birth)` (softmax mass on child tokens) and an observed birth label.
Outputs a reliability curve, ECE, and **per-cohort** calibration-in-the-large + logistic
slope/intercept.

Outputs (CSV + parquet + PNG) land in `--out`: `auc_subgroups`, `auc_pooled`,
`reliability`, `calibration_by_cohort`, `calibration_overall`, and figures.

## Token-scheme flexibility

The same suite runs against the single-`CHILD` model and the `CHILD_SON`/`CHILD_DAUGHTER`
model — set `tokens.child` (one or two names) in the config, or pin ids via `token_ids`.
Parity is reconstructed from the running count of child tokens, so it is agnostic to
whether a birth is one token or two. Set `metrics.auc_child_sex: true` (with
`tokens.child_son`/`child_daughter`) to add son-vs-daughter discrimination.

## Layout

```
ferteval/
  config.py        loaders.py     inference.py    vocab.py     build_meta.py
  metrics_auc.py   calibration.py plotting.py     pipelines.py cli.py
  demography.py    sampling.py    # phase-2 stubs (finalized interfaces)
configs/fertility_default.yaml
tests/             # fake Delphi repo + synthetic data; full pipeline runs with no real data
```

## Testing

```bash
python -m pytest -q          # 18 tests; no real data or trained model required
```

The tests stand up a fake Delphi repo (`tests/fake_delphi/`) and synthetic `.bin` so the
loaders → inference → metrics path is exercised end-to-end.

"""Decomposing completed cohort fertility (CCF) into interpretable components.

CCF changes across cohorts (or between observed and model-forecast) can arise from very
different mechanisms with the same mean:

* **childlessness**: more women ending at parity 0, vs
* **family size among mothers**: mothers
  having fewer children, and
* **postponement** (tempo), which only lowers CCF if postponers fail to recuperate — probed
  separately via the model's state-conditional forecasts (see :func:`recuperation_table`).

Two additive decompositions of ``ΔCCF`` are provided, both summing to ΔCCF exactly:

* **two-factor** ``CCF = parous × MCM`` (childlessness effect + family-size effect), and
* **PPR** ``CCF = Σ_{n≥1} Π_{k<n} a_k`` attributed to each progression ``a_k`` (``a0`` is the
  childlessness component).

Everything is computed from :func:`ferteval.demography.parity_distribution`, so components are
internally consistent (``CCF = Σ_p p·P(p)`` exactly) and work per **cohort bin** on any
:class:`~ferteval.extraction.FertilityData` (observed-completed or forecast-completed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import demography as demog
from .extraction import FertilityData


# --------------------------------------------------------------------------- #
# components                                                                    #
# --------------------------------------------------------------------------- #
def parity_components(fd: FertilityData, max_parity: int = 6) -> pd.DataFrame:
    """Per cohort: CCF, childlessness, parous share, mean-children-among-mothers, parity
    dispersion, tail shares, and the progression ratios ``a_k``."""
    dist = demog.parity_distribution(fd, max_parity)
    rows = []
    for cohort, g in dist.groupby("cohort"):
        s = g.set_index("parity")["share"].reindex(range(max_parity + 1)).fillna(0.0)
        parities = s.index.to_numpy()
        ccf = float((parities * s.to_numpy()).sum())
        childless = float(s.loc[0])
        parous = 1.0 - childless
        mcm = ccf / parous if parous > 0 else np.nan
        var = float(((parities - ccf) ** 2 * s.to_numpy()).sum())
        # survival of the parity distribution and progression ratios a_k = P(≥k+1)/P(≥k)
        cum = {k: float(s.loc[k:].sum()) for k in range(max_parity + 2)}
        rec = {
            "cohort": int(cohort), "n_women": int(g["n_women"].iloc[0]),
            "ccf": ccf, "childless": childless, "parous": parous, "mcm": mcm,
            "variance": var, "sd": float(np.sqrt(var)),
            "p2": float(s.loc[2]) if 2 in s.index else 0.0,
            "p3plus": float(s.loc[3:].sum()) if max_parity >= 3 else 0.0,
        }
        for k in range(max_parity + 1):
            rec[f"a{k}"] = (cum[k + 1] / cum[k]) if cum[k] > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# two-factor decomposition: CCF = parous × MCM                                 #
# --------------------------------------------------------------------------- #
def two_factor_decomposition(a: pd.Series, b: pd.Series) -> dict:
    """Das Gupta split of ``ΔCCF = CCF_b − CCF_a`` into childlessness and family-size effects."""
    dccf = float(b["ccf"] - a["ccf"])
    childlessness_effect = float((b["parous"] - a["parous"]) * (a["mcm"] + b["mcm"]) / 2.0)
    familysize_effect = float((a["parous"] + b["parous"]) / 2.0 * (b["mcm"] - a["mcm"]))
    return {
        "dccf": dccf,
        "childlessness_effect": childlessness_effect,
        "familysize_effect": familysize_effect,
        "residual": dccf - childlessness_effect - familysize_effect,
    }


# --------------------------------------------------------------------------- #
# PPR decomposition: CCF = Σ_{n≥1} Π_{k<n} a_k                                  #
# --------------------------------------------------------------------------- #
def _ccf_from_ppr(a_vec: np.ndarray) -> float:
    """CCF implied by progression ratios a_0..a_K: a0 + a0 a1 + a0 a1 a2 + ..."""
    total, prod = 0.0, 1.0
    for ak in a_vec:
        prod *= ak
        total += prod
    return total


def ppr_decomposition(a: pd.Series, b: pd.Series, max_parity: int = 6) -> dict:
    """Attribute ``ΔCCF`` to each progression ratio ``a_k`` (sums to ΔCCF exactly).

    Uses a sequential swap A→B averaged over forward and backward orderings — each ordering
    telescopes to ΔCCF, so their average does too, at a fraction of full Das Gupta cost.
    """
    keys = [f"a{k}" for k in range(max_parity + 1)]
    va = np.array([float(a[k]) for k in keys])
    vb = np.array([float(b[k]) for k in keys])
    va = np.nan_to_num(va)
    vb = np.nan_to_num(vb)

    def sequential(order):
        eff = np.zeros(len(keys))
        cur = va.copy()
        base = _ccf_from_ppr(cur)
        for k in order:
            cur[k] = vb[k]
            new = _ccf_from_ppr(cur)
            eff[k] = new - base
            base = new
        return eff

    fwd = sequential(list(range(len(keys))))
    bwd = sequential(list(reversed(range(len(keys)))))
    eff = (fwd + bwd) / 2.0
    out = {f"{keys[k]}_effect": float(eff[k]) for k in range(len(keys))}
    out["dccf"] = float(_ccf_from_ppr(vb) - _ccf_from_ppr(va))
    return out


# --------------------------------------------------------------------------- #
# cohort-trend attribution                                                      #
# --------------------------------------------------------------------------- #
def cohort_trend_attribution(components: pd.DataFrame, reference_cohort: int | None = None,
                             max_parity: int = 6) -> pd.DataFrame:
    """Decompose each cohort's CCF change vs a reference cohort (default: oldest)."""
    if components.empty:
        return components
    comp = components.sort_values("cohort").reset_index(drop=True)
    ref_cohort = reference_cohort if reference_cohort is not None else int(comp["cohort"].iloc[0])
    ref = comp[comp["cohort"] == ref_cohort]
    if ref.empty:
        return pd.DataFrame()
    ref = ref.iloc[0]

    rows = []
    for _, b in comp.iterrows():
        tf = two_factor_decomposition(ref, b)
        pp = ppr_decomposition(ref, b, max_parity)
        rows.append({"cohort": int(b["cohort"]), "reference": ref_cohort, "ccf": float(b["ccf"]),
                     **{k: v for k, v in tf.items() if k != "dccf"}, "dccf": tf["dccf"],
                     **{k: v for k, v in pp.items() if k != "dccf"}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# recuperation / conditional forecast probing                                  #
# --------------------------------------------------------------------------- #
def recuperation_table(forecast_fd: FertilityData, seed_states: pd.DataFrame, n_samples: int,
                       age_bins=(25, 30, 35, 40, 45), max_parity: int = 6):
    """From a forecast, stratify women by their state at the cutoff and report the model's
    forecast completed-parity mix + P(eventual childless).

    Returns ``(dist, childless_curve)``:
      * ``dist`` — forecast P(final parity=p) by (parity_at_cutoff, cutoff-age bin).
      * ``childless_curve`` — P(end childless | nulliparous at cutoff age bin) vs age.

    Each forecast pseudo-woman maps to its seed via ``woman_id // n_samples``.
    """
    w = forecast_fd.women[["woman_id", "final_parity"]].copy()
    w["seed_idx"] = w["woman_id"] // n_samples
    st = seed_states.set_index("seed_idx")[["parity_at_cutoff", "cutoff_age"]]
    w = w.join(st, on="seed_idx").dropna(subset=["parity_at_cutoff"])
    edges = list(age_bins)
    # bin cutoff age to the lower edge of its bracket
    idx = np.clip(np.digitize(w["cutoff_age"].to_numpy(), edges) - 1, 0, len(edges) - 2)
    w["cutoff_bin"] = np.array(edges[:-1])[idx]

    # completed-parity distribution by (parity_at_cutoff, cutoff bin)
    dist_rows = []
    for (par0, cbin), g in w.groupby(["parity_at_cutoff", "cutoff_bin"]):
        fp = np.clip(g["final_parity"].to_numpy(), 0, max_parity)
        for p in range(max_parity + 1):
            dist_rows.append({"parity_at_cutoff": int(par0), "cutoff_age": int(cbin),
                              "final_parity": p, "share": float((fp == p).mean()), "n": len(fp)})
    dist = pd.DataFrame(dist_rows)

    # P(eventual childless | nulliparous at cutoff) vs cutoff age
    nulli = w[w["parity_at_cutoff"] == 0]
    cc_rows = []
    for cbin, g in nulli.groupby("cutoff_bin"):
        cc_rows.append({"cutoff_age": int(cbin), "p_childless": float((g["final_parity"] == 0).mean()),
                        "mean_final_parity": float(g["final_parity"].mean()), "n": len(g)})
    childless_curve = pd.DataFrame(cc_rows).sort_values("cutoff_age").reset_index(drop=True)
    return dist, childless_curve


# --------------------------------------------------------------------------- #
# helper for the component backtest                                            #
# --------------------------------------------------------------------------- #
def childlessness_auc(rows: pd.DataFrame, n_bootstrap: int = 200, seed: int = 0) -> pd.DataFrame:
    """AUC for predicting *lifetime childlessness*, per (prediction_age, cohort).

    ``rows`` has columns ``prediction_age, cohort, score, label`` for women **nulliparous at
    the prediction age** — ``score`` = the model's forecast ``P(childless)``, ``label`` = 1 if she
    actually ended childless. AUC = P(score_childless > score_eventual-mother); >0.5 means the
    model ranks the eventual-childless above the recuperators. Bootstrap CI via the vendored
    resampler. Comparing AUC across cohorts answers "is childlessness more predictable later?".
    """
    from .metrics_auc import bootstrap_auc, roc_auc, _ci, _se

    out = []
    for (age, cohort), g in rows.groupby(["prediction_age", "cohort"]):
        s = g["score"].to_numpy(); y = g["label"].to_numpy()
        case, control = s[y == 1], s[y == 0]
        n1, n0 = len(case), len(control)
        rec = {"prediction_age": int(age), "cohort": int(cohort), "n": int(n1 + n0),
               "n_childless": int(n1), "base_rate": (n1 / (n1 + n0)) if (n1 + n0) else np.nan}
        if n1 >= 2 and n0 >= 2:
            boots = bootstrap_auc(case, control, n_bootstrap, np.random.default_rng(seed))
            lo, hi = _ci(boots)
            rec.update({"auc": roc_auc(case, control), "auc_se": _se(boots), "auc_lo": lo, "auc_hi": hi})
        else:
            rec.update({"auc": np.nan, "auc_se": np.nan, "auc_lo": np.nan, "auc_hi": np.nan})
        out.append(rec)
    return pd.DataFrame(out)


def parity_dist_tvd(fd_a: FertilityData, fd_b: FertilityData, max_parity: int = 6) -> pd.DataFrame:
    """Total-variation distance TVD = ½ Σ_p |P_a(p) − P_b(p)| between two parity distributions,
    per cohort present in both."""
    da = demog.parity_distribution(fd_a, max_parity).pivot_table(index="cohort", columns="parity", values="share")
    db = demog.parity_distribution(fd_b, max_parity).pivot_table(index="cohort", columns="parity", values="share")
    common = da.index.intersection(db.index)
    rows = []
    for c in common:
        tvd = 0.5 * float(np.nansum(np.abs(da.loc[c].to_numpy() - db.loc[c].to_numpy())))
        rows.append({"cohort": int(c), "tvd": tvd})
    return pd.DataFrame(rows)

"""Demographic estimators.

Every function takes a :class:`~ferteval.extraction.FertilityData` and returns a tidy
DataFrame, so the identical code path serves observed and model-forecast data. Rates are
**occurrence-exposure** (events / woman-years); ``period = cohort + age`` throughout.

Implements the priority set: completed cohort fertility (cumulated from cohort ASFR, which
handles censoring), parity progression ratios, parity distribution, time-to-event
(Kaplan-Meier), period mean age at first birth, ASFR by year×cohort, age×parity fertility
surface, age profile of birth order, and the Lexis first-birth intensity surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extraction import FertilityData

REPRO_AGES = list(range(15, 50))


# --------------------------------------------------------------------------- #
# rates                                                                        #
# --------------------------------------------------------------------------- #
def asfr(fd: FertilityData) -> pd.DataFrame:
    """Age-specific fertility rates ASFR(age, period) = births / woman-years.

    Returns one row per (age, period[, cohort]) cell with births, exposure, asfr.
    Each cell's cohort = period − age (Lexis identity).
    """
    births = fd.births.copy()
    if births.empty:
        return pd.DataFrame(columns=["age", "period", "cohort", "births", "exposure", "asfr"])
    births["age_int"] = np.floor(births["age"]).astype(int)
    keys = ["age_int", "period"]

    num = births.groupby(keys).size().rename("births").reset_index()
    den = fd.exposure.groupby(["age", "period"]).size().rename("exposure").reset_index()
    den = den.rename(columns={"age": "age_int"})

    out = den.merge(num, on=keys, how="left").fillna({"births": 0})
    out["cohort"] = out["period"] - out["age_int"]
    out["asfr"] = out["births"] / out["exposure"].where(out["exposure"] > 0)
    return out.rename(columns={"age_int": "age"}).sort_values(["period", "age"]).reset_index(drop=True)


def cohort_asfr(fd: FertilityData) -> pd.DataFrame:
    """ASFR by (cohort, age) — exposure and births grouped on the woman's birth cohort."""
    births = fd.births.copy()
    cols = ["cohort", "age", "births", "exposure", "asfr"]
    if births.empty:
        return pd.DataFrame(columns=cols)
    births = births[births["cohort"] >= 0]
    births["age_int"] = np.floor(births["age"]).astype(int)
    num = births.groupby(["cohort", "age_int"]).size().rename("births").reset_index()
    den = fd.exposure[fd.exposure["cohort"] >= 0].groupby(["cohort", "age"]).size().rename("exposure").reset_index()
    den = den.rename(columns={"age": "age_int"})
    out = den.merge(num, on=["cohort", "age_int"], how="left").fillna({"births": 0})
    out["asfr"] = out["births"] / out["exposure"].where(out["exposure"] > 0)
    return out.rename(columns={"age_int": "age"})[cols].sort_values(["cohort", "age"]).reset_index(drop=True)


def tfr(fd: FertilityData) -> pd.DataFrame:
    """Total fertility rate per period year = sum of ASFR over age."""
    a = asfr(fd)
    if a.empty:
        return pd.DataFrame(columns=["period", "tfr"])
    return a.groupby("period")["asfr"].sum().rename("tfr").reset_index()


# --------------------------------------------------------------------------- #
# completed cohort fertility                                                    #
# --------------------------------------------------------------------------- #
def ccf_curve(fd: FertilityData) -> pd.DataFrame:
    """Cumulated cohort fertility CCF(age) = running sum of cohort ASFR over age.

    Cumulating occurrence-exposure rates (rather than averaging raw parity) keeps the curve
    well-defined under censoring: ages with no exposure simply do not contribute. Columns:
    cohort, age, asfr, ccf, exposure, observed.
    """
    ca = cohort_asfr(fd)
    if ca.empty:
        return pd.DataFrame(columns=["cohort", "age", "asfr", "ccf", "exposure", "observed"])
    rows = []
    for cohort, g in ca.groupby("cohort"):
        g = g.set_index("age").reindex(REPRO_AGES)
        observed = g["exposure"].fillna(0) > 0
        contrib = g["asfr"].fillna(0.0)
        ccf = contrib.cumsum()
        for age in REPRO_AGES:
            rows.append({"cohort": int(cohort), "age": age, "asfr": float(contrib.loc[age]),
                         "ccf": float(ccf.loc[age]), "exposure": float(g["exposure"].fillna(0).loc[age]),
                         "observed": bool(observed.loc[age])})
    return pd.DataFrame(rows)


def completed_cohort_fertility(fd: FertilityData) -> pd.DataFrame:
    """Final CCF per cohort (CCF at the oldest reproductive age) + whether fully observed."""
    curve = ccf_curve(fd)
    if curve.empty:
        return pd.DataFrame(columns=["cohort", "ccf", "fully_observed"])
    out = []
    for cohort, g in curve.groupby("cohort"):
        out.append({"cohort": int(cohort), "ccf": float(g["ccf"].iloc[-1]),
                    "fully_observed": bool(g["observed"].iloc[-1])})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# parity                                                                        #
# --------------------------------------------------------------------------- #
def _completed_women(fd: FertilityData) -> pd.DataFrame:
    """Women whose childbearing is finished (died, or observed to the completion age)."""
    w = fd.women
    return w[(~w["censored"]) | (w["exit_age"] >= fd.completion_age)]


def parity_progression_ratios(fd: FertilityData, max_parity: int = 6) -> pd.DataFrame:
    """PPR(n) = B(n+1)/B(n), B(n) = women reaching parity ≥ n, by cohort (completed women)."""
    w = _completed_women(fd)
    w = w[w["cohort"] >= 0]
    rows = []
    for cohort, g in w.groupby("cohort"):
        fp = g["final_parity"].to_numpy()
        counts = {n: int((fp >= n).sum()) for n in range(0, max_parity + 2)}
        for n in range(0, max_parity + 1):
            bn, bn1 = counts[n], counts[n + 1]
            rows.append({"cohort": int(cohort), "n": n, "B_n": bn, "B_n1": bn1,
                         "ppr": (bn1 / bn) if bn > 0 else np.nan})
    return pd.DataFrame(rows)


def parity_distribution(fd: FertilityData, max_parity: int = 6) -> pd.DataFrame:
    """Share of (completed) women ending at each final parity, by cohort."""
    w = _completed_women(fd)
    w = w[w["cohort"] >= 0]
    rows = []
    for cohort, g in w.groupby("cohort"):
        fp = np.clip(g["final_parity"].to_numpy(), 0, max_parity)
        n = len(fp)
        for p in range(0, max_parity + 1):
            rows.append({"cohort": int(cohort), "parity": p, "share": float((fp == p).mean()) if n else np.nan,
                         "n_women": n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# time to event (Kaplan-Meier)                                                  #
# --------------------------------------------------------------------------- #
def time_to_event(fd: FertilityData, transition: int = 0, by_cohort: bool = True) -> pd.DataFrame:
    """KM survival for the age at the (transition→transition+1) birth.

    transition=0 → age at first birth. The risk set is women who reached parity
    ``transition``; event = the next birth (with its age); else right-censored at exit_age.
    """
    durations, events, cohorts = _duration_event(fd, transition)
    frames = []
    if by_cohort:
        for c in np.unique(cohorts[cohorts >= 0]):
            m = cohorts == c
            km = _km(durations[m], events[m])
            km["cohort"] = int(c)
            frames.append(km)
    else:
        km = _km(durations, events)
        km["cohort"] = -1
        frames.append(km)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["age", "survival", "at_risk", "cohort"])


def _duration_event(fd: FertilityData, transition: int):
    births = fd.births
    nth = births[births["parity"] == transition + 1].groupby("woman_id")["age"].min().to_dict()
    prev = births[births["parity"] == transition].groupby("woman_id")["age"].min().to_dict() if transition > 0 else None
    dur, ev, coh = [], [], []
    for w in fd.women.itertuples(index=False):
        wid = w.woman_id
        if transition > 0 and wid not in prev:
            continue  # never reached the starting parity → not at risk
        start = float(prev[wid]) if transition > 0 else float(w.entry_age)
        if wid in nth:
            dur.append(float(nth[wid]) - start); ev.append(1)
        else:
            dur.append(float(w.exit_age) - start); ev.append(0)
        coh.append(int(w.cohort))
    return np.array(dur), np.array(ev), np.array(coh)


def _km(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier survival on (duration, event) with no left truncation."""
    if len(durations) == 0:
        return pd.DataFrame(columns=["age", "survival", "at_risk"])
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    times = np.unique(d)
    surv, rows = 1.0, []
    for t in times:
        at_risk = int((d >= t).sum())
        n_events = int(((d == t) & (e == 1)).sum())
        if at_risk > 0:
            surv *= (1 - n_events / at_risk)
        rows.append({"age": float(t), "survival": surv, "at_risk": at_risk})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# period mean age at first birth                                               #
# --------------------------------------------------------------------------- #
def mean_age_first_birth(fd: FertilityData, by: str = "period") -> pd.DataFrame:
    """Mean age at first birth by calendar period (default) or by cohort."""
    fb = fd.births[fd.births["parity"] == 1].copy()
    if fb.empty:
        return pd.DataFrame(columns=[by, "mean_age", "n"])
    key = "period" if by == "period" else "cohort"
    fb = fb[fb[key] >= 0]
    g = fb.groupby(key)["age"]
    return pd.DataFrame({by: g.mean().index, "mean_age": g.mean().to_numpy(), "n": g.size().to_numpy()})


# --------------------------------------------------------------------------- #
# age x parity fertility surface                                               #
# --------------------------------------------------------------------------- #
def age_parity_surface(fd: FertilityData, years: list[int], max_parity: int = 5) -> pd.DataFrame:
    """Order-specific occurrence-exposure rate by age and parity for selected period years.

    rate(age, p, year) = births of order p+1 at that age/year / woman-years at that
    age/year entered at parity p. Returns tidy (year, age, parity, births, exposure, rate).
    """
    births = fd.births.copy()
    rows = []
    if births.empty:
        return pd.DataFrame(columns=["year", "age", "parity", "births", "exposure", "rate"])
    births["age_int"] = np.floor(births["age"]).astype(int)
    for year in years:
        b_y = births[births["period"] == year]
        e_y = fd.exposure[fd.exposure["period"] == year]
        num = b_y.groupby(["age_int", "parity"]).size().rename("births")
        den = e_y.groupby(["age", "parity_at_age"]).size().rename("exposure")
        for (age, par_start), exposure in den.items():
            if par_start > max_parity:
                continue
            order = par_start + 1
            n_births = int(num.get((age, order), 0))
            rows.append({"year": year, "age": int(age), "parity": int(par_start),
                         "births": n_births, "exposure": int(exposure),
                         "rate": n_births / exposure if exposure > 0 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# age profile of birth order                                                    #
# --------------------------------------------------------------------------- #
def birth_order_age_profile(fd: FertilityData, max_order: int = 5, period_range: tuple[int, int] | None = None,
                            age_grid: list[int] | None = None) -> pd.DataFrame:
    """Normalised age distribution of the n-th birth, per birth order (peaks shift right)."""
    births = fd.births.copy()
    if period_range is not None:
        lo, hi = period_range
        births = births[(births["period"] >= lo) & (births["period"] <= hi)]
    age_grid = age_grid or REPRO_AGES
    rows = []
    for order in range(1, max_order + 1):
        b = births[births["parity"] == order]
        ages = np.floor(b["age"]).astype(int)
        counts = ages.value_counts()
        total = counts.sum()
        for age in age_grid:
            c = int(counts.get(age, 0))
            rows.append({"order": order, "age": age, "count": c,
                         "density": (c / total) if total > 0 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Lexis first-birth intensity surface                                          #
# --------------------------------------------------------------------------- #
def lexis_first_birth(fd: FertilityData) -> pd.DataFrame:
    """First-birth intensity on a cohort × age grid = first births / nulliparous exposure.

    Cells with no observed nulliparous exposure (observed=False) are the unobserved
    top-right corner that the forecast fills in. Columns: cohort, age, first_births,
    exposure, intensity, observed.
    """
    fb = fd.births[(fd.births["parity"] == 1) & (fd.births["cohort"] >= 0)].copy()
    nulli = fd.exposure[(fd.exposure["parity_at_age"] == 0) & (fd.exposure["cohort"] >= 0)]
    cols = ["cohort", "age", "first_births", "exposure", "intensity", "observed"]
    if nulli.empty:
        return pd.DataFrame(columns=cols)
    fb["age_int"] = np.floor(fb["age"]).astype(int)
    num = fb.groupby(["cohort", "age_int"]).size().rename("first_births")
    den = nulli.groupby(["cohort", "age"]).size().rename("exposure")
    rows = []
    for (cohort, age), exposure in den.items():
        fbn = int(num.get((cohort, age), 0))
        rows.append({"cohort": int(cohort), "age": int(age), "first_births": fbn,
                     "exposure": int(exposure), "intensity": fbn / exposure if exposure > 0 else np.nan,
                     "observed": exposure > 0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# convenience: run them all                                                     #
# --------------------------------------------------------------------------- #
def run_all(fd: FertilityData, selected_years: list[int], max_parity: int = 6) -> dict[str, pd.DataFrame]:
    """Compute every observed estimator; returned dict keys double as output filenames."""
    return {
        "ccf_curve": ccf_curve(fd),
        "completed_cohort_fertility": completed_cohort_fertility(fd),
        "parity_progression_ratios": parity_progression_ratios(fd, max_parity),
        "parity_distribution": parity_distribution(fd, max_parity),
        "time_to_first_birth": time_to_event(fd, transition=0),
        "mean_age_first_birth": mean_age_first_birth(fd, by="period"),
        "asfr": asfr(fd),
        "tfr": tfr(fd),
        "age_parity_surface": age_parity_surface(fd, selected_years, max_parity - 1),
        "birth_order_age_profile": birth_order_age_profile(fd, max_parity - 1),
        "lexis_first_birth": lexis_first_birth(fd),
    }

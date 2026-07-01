"""Shared fertility data substrate.

Every demographic estimator consumes one object — :class:`FertilityData` — so the same
code path serves observed sequences and model-forecasted sequences. It holds three tidy
tables:

* ``women``    — one row per woman (cohort, entry/exit, censoring, final parity).
* ``births``   — one row per birth (parity / birth order, age, calendar period).
* ``exposure`` — one row per observed woman-year (the denominators for rates), tagged with
  the parity entering that age. ``period = cohort + age`` throughout (Lexis identity).

The crucial distinction for forecasting: a woman who reaches the ``death`` token is
**complete** (out of risk); a woman who ends on ``censoring`` while still under the
completion age is **incomplete** — her childbearing is unfinished and she is the unit the
forecasting engine rolls forward.

Extraction needs only ``data`` + a :class:`~ferteval.vocab.TokenVocab` — not the model — so
the observed-data demography pipeline runs without the Delphi fork or a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .vocab import TokenVocab

DAYS_PER_YEAR = 365.25

# Raw Delphi .bin files store token ids one BELOW the model/meta index: Delphi's
# get_batch applies `tokens = tokens + 1` at load time (so it can manufacture the reserved
# Padding=0 and No-event=1 tokens, which never appear in the raw file). When we read the
# raw bin directly (extraction / forecast seeds) we apply the same +1 so ids line up with
# meta.pkl. The AUC/inference path goes through get_batch and must NOT be shifted here.
TOKEN_OFFSET = 1


@dataclass
class FertilityData:
    source: str               # 'observed' | 'forecast' | 'observed+forecast'
    women: pd.DataFrame
    births: pd.DataFrame
    exposure: pd.DataFrame
    completion_age: float = 50.0

    # ------------------------------------------------------------------ #
    # constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bin(cls, data: np.ndarray, vocab: TokenVocab, completion_age: float = 50.0,
                 repro_ages: tuple[int, int] = (15, 50)) -> "FertilityData":
        """Build from a real Delphi ``.bin`` array ``(N,3)`` = [pid, age_days, token_id]."""
        p2i = _patient_index(data)
        recs = []
        for wid, (start, length) in enumerate(p2i):
            rows = data[start:start + length]
            recs.append(_extract_woman(wid, rows[:, 2] + TOKEN_OFFSET, rows[:, 1], vocab, completion_age, repro_ages))
        return _assemble(recs, "observed", completion_age)

    @classmethod
    def from_sequences(cls, sequences, vocab: TokenVocab, completion_age: float = 50.0,
                       repro_ages: tuple[int, int] = (15, 50)) -> "FertilityData":
        """Build from forecasted sequences: an iterable of ``(woman_id, tokens, ages_days)``."""
        recs = [_extract_woman(wid, np.asarray(tokens), np.asarray(ages), vocab, completion_age, repro_ages)
                for wid, tokens, ages in sequences]
        return _assemble(recs, "forecast", completion_age)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @property
    def incomplete_women(self) -> pd.DataFrame:
        """Censored women still in the reproductive window — the forecast targets."""
        w = self.women
        return w[w["censored"] & (w["exit_age"] < self.completion_age)]

    def cohorts(self) -> np.ndarray:
        c = self.women["cohort"].to_numpy()
        return np.unique(c[c >= 0])

    def with_source(self, label: str) -> "FertilityData":
        return FertilityData(label, self.women, self.births, self.exposure, self.completion_age)

    def filter_cohorts(self, min_cohort: int | None = None, max_cohort: int | None = None) -> "FertilityData":
        """Drop women/births/exposure whose cohort falls outside [min_cohort, max_cohort]."""
        if min_cohort is None and max_cohort is None:
            return self

        def flt(df):
            m = pd.Series(True, index=df.index)
            if min_cohort is not None:
                m &= df["cohort"] >= min_cohort
            if max_cohort is not None:
                m &= df["cohort"] <= max_cohort
            return df[m]

        return FertilityData(self.source, flt(self.women), flt(self.births), flt(self.exposure), self.completion_age)

    def subset(self, woman_ids) -> "FertilityData":
        """Restrict all three tables to a set of woman_ids."""
        ids = set(woman_ids)
        return FertilityData(
            self.source,
            self.women[self.women["woman_id"].isin(ids)],
            self.births[self.births["woman_id"].isin(ids)],
            self.exposure[self.exposure["woman_id"].isin(ids)],
            self.completion_age,
        )

    def binned_cohorts(self, edges) -> "FertilityData":
        """Return a copy whose ``cohort`` column is replaced by its cohort-bin lower edge.

        ``edges`` are bin boundaries (e.g. [1930, 1940, ...]); a birth year in [e_i, e_{i+1})
        is relabelled to ``e_i``. Years outside [edges[0], edges[-1]) become -1 (which every
        estimator drops). Used so CCF / PPR / time-to-event group by cohort *bin* rather than
        by every individual birth year. period/period_exact columns are left untouched
        (the cohort-family estimators don't use them).
        """
        edges = list(edges)
        if len(edges) < 2:
            return self
        e = np.asarray(edges)

        def remap(df):
            c = df["cohort"].to_numpy()
            idx = np.clip(np.digitize(c, e) - 1, 0, len(e) - 2)
            binned = np.where((c >= e[0]) & (c < e[-1]), e[idx], -1).astype(np.int64)
            return df.assign(cohort=binned)

        return FertilityData(self.source, remap(self.women), remap(self.births),
                             remap(self.exposure), self.completion_age)


def merge(observed: FertilityData, forecast: FertilityData) -> FertilityData:
    """Combine observed-complete women with model forecasts into one completed dataset.

    The observed *incomplete* women are dropped (the forecast replaces them) and the
    forecast woman_ids are offset so they never collide with observed ids.
    """
    drop = set(observed.incomplete_women["woman_id"])
    keep = ~observed.women["woman_id"].isin(drop)
    offset = (int(observed.women["woman_id"].max()) + 1) if len(observed.women) else 0

    def shift(df):
        return df.assign(woman_id=df["woman_id"] + offset)

    women = pd.concat([observed.women[keep], shift(forecast.women)], ignore_index=True)
    kept = set(women["woman_id"])
    births = pd.concat([observed.births[observed.births["woman_id"].isin(kept)], shift(forecast.births)],
                       ignore_index=True)
    exposure = pd.concat([observed.exposure[observed.exposure["woman_id"].isin(kept)], shift(forecast.exposure)],
                         ignore_index=True)
    return FertilityData("observed+forecast", women, births, exposure, observed.completion_age)


# --------------------------------------------------------------------------- #
# per-woman extraction                                                         #
# --------------------------------------------------------------------------- #
def _extract_woman(wid, tokens, ages_days, vocab: TokenVocab, completion_age, repro_ages):
    tokens = np.asarray(tokens).astype(np.int64)
    ages = np.asarray(ages_days, dtype=np.float64) / DAYS_PER_YEAR

    child_set = vocab.child_id_set
    son, daughter = vocab.child_son_id, vocab.child_daughter_id
    death_id, censor_id = vocab.end_sequence_id, vocab.censoring_id
    by_map = vocab.birth_year_to_cohort

    cohort = -1
    for t in tokens:
        if int(t) in by_map:
            cohort = by_map[int(t)]
            break

    births, timeline_ages = [], []
    exit_age, exit_reason, censored = None, "end_of_record", True
    parity = 0
    for t, a in zip(tokens, ages):
        t = int(t)
        if t in by_map:
            continue  # the cohort marker is not a timeline event
        timeline_ages.append(a)
        if t in child_set:
            parity += 1
            sex = "son" if (son is not None and t == son) else ("daughter" if (daughter is not None and t == daughter) else None)
            births.append((parity, a, sex))
        if death_id is not None and t == death_id:
            exit_age, exit_reason, censored = a, "death", False
            break
        if censor_id is not None and t == censor_id:
            exit_age, exit_reason, censored = a, "censored", True
            break

    if exit_age is None:
        exit_age = timeline_ages[-1] if timeline_ages else 0.0
        exit_reason, censored = "end_of_record", True
    entry_age = timeline_ages[0] if timeline_ages else 0.0

    woman = {
        "woman_id": wid, "cohort": cohort, "entry_age": entry_age, "exit_age": exit_age,
        "exit_reason": exit_reason, "censored": censored, "final_parity": parity,
    }
    birth_rows = [
        {"woman_id": wid, "cohort": cohort, "parity": p, "age": a,
         "period": int(np.floor(cohort + a)) if cohort >= 0 else -1,
         "period_exact": (cohort + a) if cohort >= 0 else np.nan, "child_sex": sex}
        for (p, a, sex) in births
    ]

    lo, hi = repro_ages
    birth_ages = sorted(a for (_, a, _) in births)
    exp_rows = []
    # Ages come from integer-day timestamps, so entry_age can sit a hair above its integer
    # (e.g. 15.0007). Floor it so the woman is counted as exposed from her entry age-year.
    floor_entry = int(np.floor(entry_age))
    upper = min(exit_age, completion_age, hi)
    for ag in range(int(lo), int(hi)):
        if floor_entry <= ag < upper:
            par = int(np.searchsorted(birth_ages, ag, side="left"))  # births strictly before age ag
            exp_rows.append({"woman_id": wid, "cohort": cohort, "age": ag,
                             "period": int(cohort + ag) if cohort >= 0 else -1, "parity_at_age": par})
    return woman, birth_rows, exp_rows


def _assemble(recs, source, completion_age) -> FertilityData:
    women = pd.DataFrame([r[0] for r in recs])
    births = pd.DataFrame([b for r in recs for b in r[1]],
                          columns=["woman_id", "cohort", "parity", "age", "period", "period_exact", "child_sex"])
    exposure = pd.DataFrame([e for r in recs for e in r[2]],
                            columns=["woman_id", "cohort", "age", "period", "parity_at_age"])
    for df in (women, births, exposure):
        df["source"] = source
    return FertilityData(source, women, births, exposure, completion_age)


def patient_sequences(data: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-woman (tokens, ages_days), indexed by woman_id. Tokens are shifted to the model
    index (raw bin + TOKEN_OFFSET) so seeds fed to the model match its vocabulary."""
    return [(data[s:s + n, 2].astype(np.int64) + TOKEN_OFFSET, data[s:s + n, 1].astype(np.float64))
            for (s, n) in _patient_index(data)]


def _patient_index(data: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, length) runs per patient id (data assumed grouped by pid)."""
    pids = data[:, 0]
    n = len(pids)
    out, start = [], 0
    for i in range(1, n + 1):
        if i == n or pids[i] != pids[start]:
            out.append((start, i - start))
            start = i
    return out

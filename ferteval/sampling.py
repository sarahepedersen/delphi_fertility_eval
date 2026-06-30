"""Autoregressive trajectory forecasting (PHASE 2 — interfaces reserved).

Mirrors upstream ``sampling_trajectories.ipynb``: roll each woman forward from a seed
prefix (at minimum her ``BIRTH_year`` cohort token) by repeatedly sampling the next
(token, age) pair from the model until an ``end_sequence``/death token or an age cap is
reached. The resulting sequences feed :func:`ferteval.demography.births_table_from_sequences`
so demographic measures are computed identically on real and synthetic data.

Reuses phase-1 infrastructure directly: :class:`ferteval.loaders.DelphiBundle` for the
model and :class:`ferteval.vocab.TokenVocab` for token semantics. Bodies raise
``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EvalConfig
from .loaders import DelphiBundle
from .vocab import TokenVocab


@dataclass
class ForecastConfig:
    n_samples_per_seed: int = 1
    max_age_years: float = 55.0
    temperature: float = 1.0
    seed: int = 1337


def forecast_sequences(
    cfg: EvalConfig,
    bundle: DelphiBundle,
    vocab: TokenVocab,
    seeds: np.ndarray,
    fc: ForecastConfig | None = None,
):
    """Generate forecasted sequences from seed prefixes.

    Args:
        seeds: array of seed prefixes (e.g. one BIRTH_year token per woman, optionally
            with partial observed history) to condition each rollout on.
    Returns:
        A list/array of generated (token, age) sequences. [phase 2]
    """
    raise NotImplementedError("phase 2: autoregressive rollout to end_sequence/death")


def seeds_from_data(data: np.ndarray, vocab: TokenVocab, up_to_age_years: float | None = None) -> np.ndarray:
    """Build seed prefixes from real data — e.g. each woman's cohort token plus any
    history up to ``up_to_age_years`` (for conditional forecasting / backtesting). [phase 2]"""
    raise NotImplementedError("phase 2")

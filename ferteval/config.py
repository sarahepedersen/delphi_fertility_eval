"""Configuration objects for the fertility evaluation suite.

A run is fully described by an :class:`EvalConfig`, built by deep-merging (in order):
the packaged default YAML, an optional user YAML, and CLI overrides. Keeping paths
out of the checked-in YAML makes configs machine-independent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "fertility_default.yaml"


# --------------------------------------------------------------------------- #
# Nested config sections                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    delphi_repo: str | None = None
    ckpt: str | None = None
    data: str | None = None
    meta: str | None = None
    labels_csv: str | None = None
    out: str = "reports/run"


@dataclass
class Tokens:
    child: list[str] = field(default_factory=lambda: ["CHILD"])
    child_son: str | None = None
    child_daughter: str | None = None
    no_event: str | None = "no_event"
    end_sequence: str | None = "death"
    padding: str | None = "padding"
    censoring: str | None = "censoring"
    birth_year_prefix: str | None = "BIRTH_"
    birth_year_regex: str = r"(\d{4})"


@dataclass
class AgeBins:
    min: int = 15
    max: int = 50
    step: int = 5

    def edges(self) -> list[int]:
        return list(range(self.min, self.max + 1, self.step))


@dataclass
class CohortBins:
    edges: list[int] | None = None
    min: int = 1930
    max: int = 1990
    step: int = 10

    def edge_list(self) -> list[int]:
        if self.edges is not None:
            return list(self.edges)
        return list(range(self.min, self.max + 1, self.step))


@dataclass
class Bins:
    age: AgeBins = field(default_factory=AgeBins)
    cohort: CohortBins = field(default_factory=CohortBins)


@dataclass
class Inference:
    device: str = "auto"
    block_size: int = 80
    batch_size: int = 128
    select: str = "left"
    padding_mode: str = "random"
    no_event_token_rate: int = 1
    offset_days: float = 365.25
    dataset_subset_size: int = -1
    get_batch_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    seed: int = 1337
    n_bootstrap: int = 1000
    auc_first_birth: bool = True
    auc_parity_progression: bool = True
    max_parity: int = 5
    auc_child_sex: bool = False
    calibration_n_bins: int = 10


@dataclass
class EvalConfig:
    paths: Paths = field(default_factory=Paths)
    tokens: Tokens = field(default_factory=Tokens)
    token_ids: dict[str, Any] = field(default_factory=dict)
    bins: Bins = field(default_factory=Bins)
    inference: Inference = field(default_factory=Inference)
    metrics: Metrics = field(default_factory=Metrics)

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config_path: str | Path | None = None, overrides: dict | None = None) -> "EvalConfig":
        """Build a config from the packaged default + optional user YAML + overrides."""
        data = _read_yaml(_DEFAULT_CONFIG)
        if config_path is not None:
            data = _deep_merge(data, _read_yaml(Path(config_path)))
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        d = d or {}
        return cls(
            paths=Paths(**(d.get("paths") or {})),
            tokens=Tokens(**(d.get("tokens") or {})),
            token_ids=dict(d.get("token_ids") or {}),
            bins=Bins(
                age=AgeBins(**((d.get("bins") or {}).get("age") or {})),
                cohort=CohortBins(**((d.get("bins") or {}).get("cohort") or {})),
            ),
            inference=Inference(**(d.get("inference") or {})),
            metrics=Metrics(**(d.get("metrics") or {})),
        )

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    def require_paths(self, *names: str) -> None:
        """Raise a clear error if a required path is unset."""
        missing = [n for n in names if getattr(self.paths, n) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required path(s): {', '.join(missing)}. "
                f"Pass them via --{missing[0].replace('_', '-')} (or in your config YAML)."
            )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out

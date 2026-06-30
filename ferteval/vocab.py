"""Token-name resolution for fertility sequences.

The same evaluation suite must run against several token schemes:

* a single ``CHILD`` token,
* ``CHILD_SON`` / ``CHILD_DAUGHTER``,
* (future) parity-specific child tokens.

:class:`TokenVocab` resolves *logical* tokens (the child set, ``no_event``,
``end_sequence``/death, ``padding``, ``censoring``, and the ``BIRTH_year`` cohort
tokens) to concrete integer ids, using — in priority order — explicit id overrides
from the config, then a name→id map loaded from ``meta.pkl`` and/or a labels CSV.

Parity is **not** a token property here: it is reconstructed downstream from the
running count of child tokens within a sequence, so this resolver is agnostic to
whether births are one token or two (son/daughter).
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import EvalConfig, Tokens


@dataclass
class TokenVocab:
    name_to_id: dict[str, int]
    id_to_name: dict[int, str]

    # resolved logical groups
    child_ids: list[int]
    child_son_id: int | None
    child_daughter_id: int | None
    no_event_id: int | None
    end_sequence_id: int | None
    padding_id: int | None
    censoring_id: int | None
    birth_year_to_cohort: dict[int, int]  # token_id -> birth year

    # -------------------------------------------------------------- #
    # Construction                                                   #
    # -------------------------------------------------------------- #
    @classmethod
    def from_config(cls, cfg: EvalConfig) -> "TokenVocab":
        name_to_id = load_name_map(cfg.paths.meta, cfg.paths.labels_csv)
        return cls.resolve(name_to_id, cfg.tokens, cfg.token_ids)

    @classmethod
    def resolve(
        cls,
        name_to_id: dict[str, int],
        tokens: Tokens,
        token_ids: dict[str, Any] | None = None,
    ) -> "TokenVocab":
        token_ids = token_ids or {}
        id_to_name = {int(v): k for k, v in name_to_id.items()}

        def one(logical: str, name: str | None) -> int | None:
            """Resolve a single-id logical token (override > name lookup)."""
            if logical in token_ids and token_ids[logical] is not None:
                return int(token_ids[logical])
            if name is None:
                return None
            return _lookup(name_to_id, name, logical)

        # --- child set (one or more names) -----------------------------------
        if "child" in token_ids and token_ids["child"] is not None:
            child_ids = [int(x) for x in _as_list(token_ids["child"])]
        else:
            child_ids = [_lookup(name_to_id, n, "child") for n in _as_list(tokens.child)]
        child_ids = sorted(dict.fromkeys(child_ids))  # dedupe, keep order-stable
        if not child_ids:
            raise ValueError("No child token resolved; set tokens.child or token_ids.child.")

        child_son_id = one("child_son", tokens.child_son)
        child_daughter_id = one("child_daughter", tokens.child_daughter)
        no_event_id = one("no_event", tokens.no_event)
        end_sequence_id = one("end_sequence", tokens.end_sequence)
        padding_id = one("padding", tokens.padding)
        censoring_id = one("censoring", tokens.censoring)

        # --- birth-year / cohort tokens --------------------------------------
        birth_year_to_cohort = _resolve_birth_years(name_to_id, tokens, token_ids)

        return cls(
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            child_ids=child_ids,
            child_son_id=child_son_id,
            child_daughter_id=child_daughter_id,
            no_event_id=no_event_id,
            end_sequence_id=end_sequence_id,
            padding_id=padding_id,
            censoring_id=censoring_id,
            birth_year_to_cohort=birth_year_to_cohort,
        )

    # -------------------------------------------------------------- #
    # Convenience                                                    #
    # -------------------------------------------------------------- #
    @property
    def child_id_set(self) -> set[int]:
        return set(self.child_ids)

    @property
    def ignore_ids(self) -> set[int]:
        """Tokens that are neither events of interest nor cohort markers."""
        ids = {self.no_event_id, self.padding_id, self.censoring_id}
        return {i for i in ids if i is not None}

    def cohort_of_sequence(self, token_ids: Iterable[int]) -> int | None:
        """Return the birth-year cohort for a sequence, or None if no marker is present."""
        for t in token_ids:
            t = int(t)
            if t in self.birth_year_to_cohort:
                return self.birth_year_to_cohort[t]
        return None

    def describe(self) -> str:
        son = self.id_to_name.get(self.child_son_id) if self.child_son_id is not None else None
        return (
            f"TokenVocab(child={[self.id_to_name.get(i, i) for i in self.child_ids]}, "
            f"son={son}, no_event={self.no_event_id}, end_sequence={self.end_sequence_id}, "
            f"padding={self.padding_id}, censoring={self.censoring_id}, "
            f"n_cohort_tokens={len(self.birth_year_to_cohort)})"
        )


# --------------------------------------------------------------------------- #
# name map loading                                                            #
# --------------------------------------------------------------------------- #
def load_name_map(meta_path: str | Path | None, labels_csv: str | Path | None = None) -> dict[str, int]:
    """Build a token-name -> id map from meta.pkl and/or a labels CSV.

    Accepts the common Delphi / nanoGPT shapes:
      * meta.pkl with a ``stoi`` (name->id) and/or ``itos`` (id->name) entry,
      * meta.pkl that is itself a flat {name: id} or {id: name} dict,
      * a labels CSV with name + index/id columns.
    """
    name_to_id: dict[str, int] = {}

    if meta_path is not None:
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        name_to_id.update(_name_map_from_meta(meta))

    if labels_csv is not None:
        name_to_id.update(_name_map_from_csv(labels_csv))

    if not name_to_id:
        raise ValueError(
            "Could not build a token name->id map. Provide paths.meta (meta.pkl with "
            "stoi/itos) or paths.labels_csv, or pin ids directly via token_ids."
        )
    return name_to_id


def _name_map_from_meta(meta: Any) -> dict[str, int]:
    if isinstance(meta, dict):
        if "stoi" in meta and isinstance(meta["stoi"], dict):
            return {str(k): int(v) for k, v in meta["stoi"].items()}
        if "itos" in meta and isinstance(meta["itos"], dict):
            return {str(v): int(k) for k, v in meta["itos"].items()}
        # flat dict — infer direction from key/value types
        keys, vals = list(meta.keys()), list(meta.values())
        if keys and all(isinstance(k, str) for k in keys) and all(_is_int(v) for v in vals):
            return {str(k): int(v) for k, v in meta.items()}
        if keys and all(_is_int(k) for k in keys) and all(isinstance(v, str) for v in vals):
            return {str(v): int(k) for k, v in meta.items()}
    raise ValueError(
        "Unrecognised meta.pkl structure; expected a dict with 'stoi'/'itos' or a "
        "flat {name: id} / {id: name} mapping."
    )


def _name_map_from_csv(labels_csv: str | Path) -> dict[str, int]:
    import pandas as pd

    df = pd.read_csv(labels_csv)
    cols = {c.lower(): c for c in df.columns}
    name_col = next((cols[c] for c in ("name", "token", "label") if c in cols), None)
    id_col = next((cols[c] for c in ("index", "id", "token_id", "idx") if c in cols), None)
    if name_col is None or id_col is None:
        raise ValueError(
            f"labels_csv {labels_csv} must have a name column (name/token/label) and an "
            f"id column (index/id/token_id); found {list(df.columns)}."
        )
    return {str(n): int(i) for n, i in zip(df[name_col], df[id_col]) if pd.notna(i)}


# --------------------------------------------------------------------------- #
# birth-year / cohort resolution                                             #
# --------------------------------------------------------------------------- #
def _resolve_birth_years(
    name_to_id: dict[str, int], tokens: Tokens, token_ids: dict[str, Any]
) -> dict[int, int]:
    # explicit override: {token_id: year} or [(id, year), ...]
    if token_ids.get("birth_year") is not None:
        raw = token_ids["birth_year"]
        if isinstance(raw, dict):
            return {int(k): int(v) for k, v in raw.items()}
        return {int(i): int(y) for i, y in raw}

    if not tokens.birth_year_prefix:
        return {}

    pattern = re.compile(tokens.birth_year_regex)
    out: dict[int, int] = {}
    for name, tid in name_to_id.items():
        if name.startswith(tokens.birth_year_prefix):
            m = pattern.search(name)
            if m:
                out[int(tid)] = int(m.group(1))
    return out


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def _lookup(name_to_id: dict[str, int], name: str, logical: str) -> int:
    if name in name_to_id:
        return int(name_to_id[name])
    raise ValueError(
        f"Token name {name!r} (for logical '{logical}') not found in the vocabulary. "
        f"Set tokens.{logical} to the correct name or pin token_ids.{logical}. "
        f"Available names ({len(name_to_id)}): {sorted(name_to_id)}"
    )


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _is_int(x: Any) -> bool:
    return isinstance(x, int) or (hasattr(x, "__int__") and not isinstance(x, (str, bytes)))

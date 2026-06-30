"""Generate a ``meta.pkl`` token name<->id map from a Delphi ``labels.csv``.

Upstream Delphi's ``train.py`` doesn't read ``meta.pkl`` (vocab_size is a config
constant), and the data-prep step assigns each token an id equal to its **row position**
in ``labels.csv`` (whose first field per row is the token / ICD code name). This utility
makes that mapping explicit and writes it in the nanoGPT ``meta.pkl`` shape that the rest
of ``ferteval`` (see :mod:`ferteval.vocab`) reads:

    {"stoi": {name: id}, "itos": {id: name}, "vocab_size": int}

It accepts both comma-delimited CSVs (with a header) and the upstream space-delimited
``labels.csv`` (first whitespace field = name, id = row index).
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

_NAME_COLS = ("name", "token", "label", "code", "event")
_ID_COLS = ("index", "id", "token_id", "idx")


def build_meta_from_labels(
    labels_path: str | Path,
    name_column: str | None = None,
    id_column: str | None = None,
    id_start: int = 0,
    has_header: bool | None = None,
    whole_line: bool = False,
) -> dict:
    """Build a ``{stoi, itos, vocab_size}`` dict from a labels file.

    Args:
        labels_path: path to ``labels.csv`` (comma- or whitespace-delimited).
        name_column: column holding the token name. Auto-detected (name/token/label/
            code/event) for CSVs; for whitespace files the first field is used.
        id_column: column holding an explicit integer id. If omitted, ids are assigned
            by row order starting at ``id_start``.
        id_start: first id when assigning by row order (default 0).
        has_header: force header on/off. ``None`` auto-detects.
        whole_line: treat each *entire* (stripped) line as one token name, with no
            delimiter splitting. Use this when token names themselves contain spaces
            (e.g. ``"no event"``) — otherwise whitespace splitting keeps only the first
            word. Ignored when ``id_column``/CSV columns are needed.

    Returns:
        dict with ``stoi``, ``itos``, ``vocab_size``.
    """
    rows, header = _read_rows(Path(labels_path), has_header, whole_line=whole_line)
    if not rows:
        raise ValueError(f"{labels_path}: no data rows found.")

    name_idx, id_idx = _resolve_columns(header, rows[0], name_column, id_column)

    stoi: dict[str, int] = {}
    itos: dict[int, str] = {}
    for row_pos, row in enumerate(rows):
        name = str(row[name_idx]).strip()
        if name == "":
            continue
        tid = int(float(row[id_idx])) if id_idx is not None else id_start + row_pos
        if name in stoi and stoi[name] != tid:
            raise ValueError(f"Duplicate token name {name!r} with conflicting ids {stoi[name]} vs {tid}.")
        if tid in itos and itos[tid] != name:
            raise ValueError(f"Duplicate id {tid} for names {itos[tid]!r} and {name!r}.")
        stoi[name] = tid
        itos[tid] = name

    vocab_size = (max(itos) + 1) if itos else 0
    return {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}


def write_meta(meta: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(meta, fh)
    return out


# --------------------------------------------------------------------------- #
# parsing helpers                                                              #
# --------------------------------------------------------------------------- #
def _read_rows(path: Path, has_header: bool | None, whole_line: bool = False) -> tuple[list[list[str]], list[str] | None]:
    """Read the labels file into rows + optional header, sniffing the delimiter."""
    text = path.read_text().splitlines()
    lines = [ln for ln in text if ln.strip() != ""]
    if not lines:
        return [], None

    if whole_line:
        parsed = [[ln.strip()] for ln in lines]  # each full line = one token name
    elif "," in lines[0]:
        parsed = list(csv.reader(lines))
    else:
        parsed = [ln.split() for ln in lines]  # whitespace-delimited (upstream style)

    if has_header is None:
        has_header = _looks_like_header(parsed[0])
    if has_header:
        return parsed[1:], parsed[0]
    return parsed, None


def _looks_like_header(first_row: list[str]) -> bool:
    cells = [c.strip().lower() for c in first_row]
    return any(c in _NAME_COLS or c in _ID_COLS for c in cells)


def _resolve_columns(
    header: list[str] | None,
    first_row: list[str],
    name_column: str | None,
    id_column: str | None,
) -> tuple[int, int | None]:
    """Return (name_col_index, id_col_index|None)."""
    if header is not None:
        lower = [h.strip().lower() for h in header]

        def find(requested, candidates):
            if requested is not None:
                if requested in header:
                    return header.index(requested)
                if requested.lower() in lower:
                    return lower.index(requested.lower())
                raise ValueError(f"Column {requested!r} not in header {header}.")
            for c in candidates:
                if c in lower:
                    return lower.index(c)
            return None

        name_idx = find(name_column, _NAME_COLS)
        if name_idx is None:
            name_idx = 0  # fall back to first column
        id_idx = find(id_column, _ID_COLS)
        return name_idx, id_idx

    # no header: first field is the name; ids by row order unless a numeric id column index given
    name_idx = int(name_column) if (name_column and name_column.isdigit()) else 0
    id_idx = int(id_column) if (id_column and id_column.isdigit()) else None
    return name_idx, id_idx


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--out", required=True, help="Output meta.pkl path")
    parser.add_argument("--name-column", default=None, help="Name column (auto-detected if omitted)")
    parser.add_argument("--id-column", default=None, help="Explicit id column (else ids = row order)")
    parser.add_argument("--id-start", type=int, default=0, help="First id when assigning by row order")
    parser.add_argument(
        "--has-header", choices=["auto", "yes", "no"], default="auto", help="Whether the file has a header row"
    )
    parser.add_argument(
        "--whole-line", action="store_true",
        help="Treat each full line as one token name (use when names contain spaces, e.g. 'no event')",
    )


def run(args: argparse.Namespace) -> dict:
    has_header = {"auto": None, "yes": True, "no": False}[args.has_header]
    meta = build_meta_from_labels(
        args.labels,
        name_column=args.name_column,
        id_column=args.id_column,
        id_start=args.id_start,
        has_header=has_header,
        whole_line=getattr(args, "whole_line", False),
    )
    out = write_meta(meta, args.out)
    print(f"Wrote {out} — vocab_size={meta['vocab_size']}, {len(meta['stoi'])} named tokens.")
    sample = list(meta["stoi"].items())[:8]
    print("First tokens:", sample)
    return meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build meta.pkl (stoi/itos) from a Delphi labels.csv")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

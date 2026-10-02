"""The seal as a picture (M2_CONTRACT §3, §11): one cell per row, on its side of the seal.

The split question's preview draws the rows the seal is drawn from as cells, so its basis is seen
rather than described: a grouped seal moves whole units; a seal whose grouping was abandoned leaves
units on both sides; an undetermined one is drawn by row and says what an identifier suggests
without claiming it; a chronological one lays the units out by their last observation.

Only row identity and the draw travel: which side each row is on, which unit it belongs to, and
(chronological) each unit's last time. No outcome value is read or sent. The counts are over every
row the seal is drawn from; the cells are whole units in file order, at most
:data:`~turbotab.core.consequences.SEAL_CELLS` rows, so the picture reads the same at 300 rows or
300,000.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from turbotab.core.consequences import SEAL_CELLS, SealCells


def _codes(values: Sequence[Any], row_ids: np.ndarray) -> np.ndarray:
    """Each value's unit code; a blank identifier is a unit of its own (it matches nobody)."""
    import pandas as pd

    keyed = [f"__row_{rid}" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
             for v, rid in zip(values, row_ids)]
    return pd.factorize(np.asarray(keyed, dtype=object), sort=False)[0].astype(np.int64)


def _shown_rows(unit: np.ndarray | None, n: int, limit: int) -> np.ndarray:
    """Positions (in file order) of whole units, first-seen first, up to ``limit`` rows."""
    if unit is None:
        return np.arange(min(n, limit))
    first = {}
    for pos, u in enumerate(unit.tolist()):
        first.setdefault(u, pos)
    sizes = np.bincount(unit, minlength=len(first))
    picked: list[int] = []
    total = 0
    for u in sorted(first, key=first.get):
        if picked and total + int(sizes[u]) > limit:
            break
        picked.append(u)
        total += int(sizes[u])
    return np.flatnonzero(np.isin(unit, picked))


def _evidence(column: str, values: Sequence[Any], held: np.ndarray) -> str | None:
    import pandas as pd

    s = pd.Series(list(values), dtype=object)
    present = s.notna().to_numpy()
    held_values = set(s[held & present].astype(str))
    train_values = set(s[~held & present].astype(str))
    if not held_values or len(set(s[present].astype(str))) == int(present.sum()):
        return None  # nothing held out, or the column never repeats: it suggests nothing
    both = len(held_values & train_values)
    return (f"`{column}` repeats: `{both:,}` of its `{len(held_values):,}` held-out values "
            f"also train.")


def seal_cells(*, draw: Any, universe: Any, info: dict[str, Any], store: Any,
               suggested: str | None = None, limit: int = SEAL_CELLS,
               read: list[str] | None = None) -> SealCells:
    """The seal ``draw`` (``seal.seal_inputs``) as cells, from ``draw_split``'s ``info``.

    ``universe`` is every row the held-out rows were drawn over, aligned with ``draw.groups``;
    ``suggested`` names a column the grain reading suggests repeats (undetermined bases only).
    Each column read over the universe is appended to ``read``, so the preview's basis can say so.
    """
    read = read if read is not None else []
    from turbotab.core import seal

    ids = np.asarray(universe, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    held = np.isin(ids, np.asarray(info.get("sealed", []), dtype=np.int64))
    basis = draw.basis
    n = int(len(ids))

    unit: np.ndarray | None = None
    if basis.state == "grouped" and draw.groups is not None:
        unit = _codes(np.asarray(draw.groups, dtype=object)[order], ids)
    elif basis.state == "abandoned" and basis.column and basis.column in store.columns:
        values = store.materialize([basis.column], ids)[basis.column].to_numpy(dtype=object)
        read.append(basis.column)
        unit = _codes(values, ids)

    chron = draw.chronology if draw.chronology is not None and draw.chronology.drawn else None
    times: np.ndarray | None = None
    dated = False
    if chron is not None and chron.time_column in store.columns:
        import pandas as pd

        raw = store.materialize([chron.time_column], ids)[chron.time_column]
        read.append(chron.time_column)
        times = seal.read_times(raw)
        dated = not pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw)
        if unit is None:  # by row, chronologically: each row is its own unit on the axis
            unit = np.arange(n, dtype=np.int64)

    shown = _shown_rows(unit, n, limit)
    hold = held[shown].astype(int).tolist()
    shown_unit: list[int] | None = None
    unit_time: list[float] | None = None
    time_start = time_end = None
    if unit is not None:
        _, local = np.unique(unit[shown], return_inverse=True)
        # Number the drawn units first-seen first, so unit 0 is the first in the file.
        first_seen: dict[int, int] = {}
        for code in local.tolist():
            first_seen.setdefault(code, len(first_seen))
        shown_unit = [first_seen[c] for c in local.tolist()]
        if times is not None:
            last = np.full(len(first_seen), np.nan)
            for pos, u in zip(shown.tolist(), shown_unit):
                t = times[pos]
                if not np.isnan(t) and (np.isnan(last[u]) or t > last[u]):
                    last[u] = t
            lo, hi = np.nanmin(last), np.nanmax(last)
            span = (hi - lo) or 1.0
            unit_time = [float((t - lo) / span) if not np.isnan(t) else 0.0 for t in last]
            time_start, time_end = seal._show_time(lo, dated), seal._show_time(hi, dated)

    n_units = n_holdout_units = straddle = None
    if unit is not None and basis.state in ("grouped", "abandoned"):
        sides = np.zeros((int(unit.max()) + 1, 2), dtype=bool)
        sides[unit[held], 1] = True
        sides[unit[~held], 0] = True
        n_units = int(sides.any(axis=1).sum())
        n_holdout_units = int(sides[:, 1].sum())
        straddle = int((sides[:, 0] & sides[:, 1]).sum())
    elif basis.state == "one_row_per_unit":
        n_units, n_holdout_units, straddle = n, int(held.sum()), 0

    evidence = None
    if basis.state == "undetermined" and suggested and suggested in store.columns:
        values = store.materialize([suggested], ids)[suggested].to_numpy(dtype=object)
        read.append(suggested)
        evidence = _evidence(suggested, values, held)

    return SealCells(
        state=basis.state, label=basis.label, exploratory=bool(draw.exploratory),
        column=basis.column, chronological=chron is not None,
        time_column=chron.time_column if chron is not None else None,
        boundary=chron.boundary if chron is not None else None,
        time_start=time_start, time_end=time_end,
        n_rows=n, n_holdout=int(held.sum()), n_units=n_units, n_holdout_units=n_holdout_units,
        straddle=straddle, evidence=evidence, hold=hold, unit=shown_unit, unit_time=unit_time,
    )


__all__ = ["seal_cells"]

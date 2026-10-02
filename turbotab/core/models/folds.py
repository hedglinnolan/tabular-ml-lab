"""Cross-validation folds that respect time: forward chaining by whole unit (audit MA-11, A17).

When the user says the task predicts later outcomes from earlier ones, random folds let every model
learn from the future it is scored on, so cross-validation flatters it (random-fold CV R² 0.27–0.30
against forward chaining's 0.23–0.24 on the audit's drift fixture). Roberts et al. 2017 (Ecography
40:913): "We recommend that block cross-validation be used wherever dependence structures exist in
a dataset".

**The scheme.** Units (a person, or a row when nothing repeats) are ordered by their order key —
the seal's rank of each unit's last observation, the same order the chronological holdout uses —
and cut into ``B`` contiguous blocks, balanced by rows. Fold ``j`` (``j = 1 … B − 1``) is scored
by a model fit on blocks ``0 … j − 1``: scikit-learn's ``TimeSeriesSplit`` convention (``n_splits``
scored folds over ``n_splits + 1`` blocks), applied to whole units. Block 0 only ever trains.
Units with no readable time sort first, so they train every fold and are never scored.

**Every block holds every class.** Contiguous time blocks cannot be stratified, and a rare event can
leave a block without one: its AUC is undefined and the fit fails (A17: 34% of rare-event splits).
For a classification outcome the cuts are moved, as little as possible from the row-balanced ones,
so that every block (the first training block included) holds every class; when the events are too
few for ``B`` such blocks, fewer blocks are made and a note says so.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

UNDATED = -np.inf  # the order key of a unit with no readable time: earliest, so it only trains


def _segments_from_right(classes: Sequence[frozenset], need: frozenset) -> np.ndarray:
    """``R[i]``: how many disjoint runs, each holding every class in ``need``, fit in units ``i…``.

    Scanning from the right and closing a run as soon as it holds every class gives the most runs
    for every suffix at once (an exchange argument: any run can be shrunk to end where the greedy
    one does without losing a run to its right).
    """
    n = len(classes)
    out = np.zeros(n + 1, dtype=np.int64)
    seen: set = set()
    count = 0
    for i in range(n - 1, -1, -1):
        seen |= classes[i]
        if need <= seen:
            count += 1
            seen = set()
        out[i] = count
    return out


def _first_full_end(classes: Sequence[frozenset], start: int, need: frozenset) -> int | None:
    """The smallest ``end`` such that units ``start … end − 1`` hold every class (None: never)."""
    seen: set = set()
    for i in range(start, len(classes)):
        seen |= classes[i]
        if need <= seen:
            return i + 1
    return None


def forward_blocks(order: Any, units: Any, n_blocks: int, labels: Any | None = None
                   ) -> tuple[np.ndarray, int, list[str]]:
    """Block numbers (``0 … B − 1``) per row, time-ordered by whole unit; ``B``; and notes.

    ``order`` (per row; NaN or ``-inf`` when a row's time is unknown) ranks the units: a unit sorts
    by its largest key. ``units`` (per row) keeps a unit's rows in one block. ``labels`` (per row,
    classification only) asks that every block hold every class. Ties sort by the unit's label, so
    the blocks never depend on the order the rows arrive in.
    """
    import pandas as pd

    order = np.asarray(order, dtype=float)
    order = np.where(np.isnan(order), UNDATED, order)
    keys = np.asarray([str(u) for u in np.asarray(units, dtype=object)], dtype=object)
    n = len(keys)
    notes: list[str] = []
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0, notes
    table = pd.DataFrame({"unit": keys, "order": order})
    per_unit = table.groupby("unit", sort=True).agg(order=("order", "max"), rows=("order", "size"))
    per_unit = per_unit.reset_index().sort_values(["order", "unit"], kind="mergesort")
    names = per_unit["unit"].to_numpy(dtype=object)
    rows = per_unit["rows"].to_numpy(dtype=np.int64)
    n_units = len(names)
    want = max(1, min(int(n_blocks), n_units))

    classes: list[frozenset] | None = None
    need: frozenset = frozenset()
    if labels is not None:
        lab = pd.Series(np.asarray(labels, dtype=object)).astype(str).to_numpy(dtype=object)
        by_unit = pd.DataFrame({"unit": keys, "label": lab}).groupby("unit", sort=True)["label"]
        held = {u: frozenset(v) for u, v in by_unit.unique().items()}
        classes = [held[u] for u in names]
        need = frozenset().union(*classes)
        if len(need) < 2:
            classes = None  # one class: there is nothing to cover

    reach = _segments_from_right(classes, need) if classes is not None else None
    blocks = want
    if reach is not None and reach[0] < blocks:
        if reach[0] >= 2:
            notes.append(f"The rarest class fills only `{int(reach[0])}` time-ordered blocks that "
                         f"each hold every class, so there are `{int(reach[0]) - 1}` folds instead "
                         f"of `{want - 1}`.")
            blocks = int(reach[0])
        else:
            notes.append("Too few rows of the rarest class to give every time-ordered fold one of "
                         "each class: some folds cannot be scored on every class.")
            reach = None
            classes = None

    cum = np.concatenate([[0], np.cumsum(rows)])
    total = float(cum[-1])
    cuts: list[int] = []
    start = 0
    for j in range(1, blocks):
        ideal_rows = total * j / blocks
        lo = start + 1  # a block holds at least one unit
        hi = n_units - (blocks - j)  # and leaves one for each block after it
        if classes is not None and reach is not None:
            first = _first_full_end(classes, start, need)
            lo = max(lo, first if first is not None else n_units)
            feasible = np.flatnonzero(reach[: n_units + 1] >= blocks - j)
            hi = min(hi, int(feasible.max()) if len(feasible) else lo)
        hi = max(hi, lo)
        candidates = np.arange(lo, hi + 1)
        cut = int(candidates[np.argmin(np.abs(cum[candidates] - ideal_rows))])
        cuts.append(cut)
        start = cut
    unit_block = np.zeros(n_units, dtype=np.int64)
    for j, cut in enumerate(cuts, start=1):
        unit_block[cut:] = j
    block_of = dict(zip(names.tolist(), unit_block.tolist()))
    return np.asarray([block_of[k] for k in keys], dtype=np.int64), blocks, notes


def forward_pairs(blocks: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) position pairs: block ``j`` scored by a model fit on every earlier block."""
    blocks = np.asarray(blocks, dtype=np.int64)
    out = []
    for j in sorted(set(blocks.tolist())):
        if j == 0:
            continue
        out.append((np.flatnonzero(blocks < j), np.flatnonzero(blocks == j)))
    return out


__all__ = ["UNDATED", "forward_blocks", "forward_pairs"]

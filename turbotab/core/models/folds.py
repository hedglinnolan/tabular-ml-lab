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

**The comparison substrate** (MODELING_SEQUENCE ruling 4, MS6). Under prediction, whatever
validation supplies the headline score, families are compared with each other and with the
no-predictor baseline, and the choice among them is corrected, on **repeated k-fold cross-validation,
at least :data:`COMPARISON_REPEATS` × K** (:func:`comparison_folds`). Bouckaert & Frank (PAKDD 2004):
"replicability improved even further by using 10-times 10-fold cross-validation instead of random
subsampling … for best replicability we recommend the latter one." A one-partition comparison rests
on df = K − 1 and replicates poorly across partitions. The first repeats are the split's own folds,
so the headline's one k-fold run is a subset of the substrate; the rest are drawn as the split draws
its folds (grouped by unit, stratified for classes or events, seeded ``seed + r``). Time-ordered
folds are the same in every repeat, so they run once. Internal–external validation's folds are the
clusters, so the substrate is drawn afresh beside them.

**Whole units in every resample.** When rows repeat within a unit, each fold, each bootstrap
resample (:func:`unit_rows`, :func:`draw_units`) and each nested cross-validation fold
(:func:`nested_folds`) takes or leaves a unit's rows together, so no unit is on both sides of any
split (MODELING_SEQUENCE §2: repeated units *imply* grouped folds, a grouped holdout and a bootstrap
by unit). When the seal could not keep units whole (one unit, or too few to hold any out), the
folds are drawn by row and every score is stated as within-unit performance, never silently
(``validation.unit_spans``).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

UNDATED = -np.inf  # the order key of a unit with no readable time: earliest, so it only trains
COMPARISON_REPEATS = 10  # the substrate's repeats at least (Bouckaert & Frank 2004: 10 × 10)
TIME_ORDERED_ONCE = ("Time-ordered folds are the same in every repeat, so the comparisons rest on one "
                     "run of them.")


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


# ── the comparison substrate and whole-unit resampling (module docstring) ─────


def unit_labels(groups: Any, n: int) -> np.ndarray | None:
    """Each row's unit as a string (None: every row its own unit); a missing label is a unit of its
    own row, as the split maps it."""
    if groups is None:
        return None
    import pandas as pd

    values = np.asarray(groups, dtype=object)
    if len(values) != n:
        raise ValueError(f"{len(values)} unit labels for {n} rows")
    return np.asarray([f"__missing_{i}" if (v is None or (isinstance(v, float) and np.isnan(v))
                                            or v is pd.NA) else str(v)
                       for i, v in enumerate(values)], dtype=object)


def kfold_assignment(n: int, *, strata: Any = None, groups: Any = None, folds: int = 5,
                     seed: int = 0) -> np.ndarray:
    """A fold number per row (``0 … K − 1``), drawn as the split draws its folds: whole units when
    ``groups`` are given, stratified by ``strata`` (classes, or the event) when given and possible,
    shuffled by ``seed``. With one unit no fold can leave a unit out, so the rows are folded by row;
    the fit never asks for that silently: when the seal cannot keep units whole it passes no
    groups, and every score says it is within-unit performance (``validation.unit_spans``)."""
    import warnings

    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    fold = np.zeros(n, dtype=np.int64)
    if n < 2:
        return fold
    labels = unit_labels(groups, n)
    units = len(pd.unique(labels)) if labels is not None else n
    k = max(2, min(int(folds), units))
    idx = np.arange(n)
    plain = KFold(k, shuffle=True, random_state=seed)
    strata = None if strata is None else np.asarray(strata, dtype=object).astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a rare class warns; it falls back below
        if labels is not None and units >= 2:
            parts = (StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, strata, groups=labels)
                     if strata is not None else
                     GroupKFold(k, shuffle=True, random_state=seed).split(idx, groups=labels))
        elif strata is not None and pd.Series(strata).value_counts().max() >= k:
            parts = StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, strata)
        else:
            parts = plain.split(idx)
        try:
            for f, (_, test) in enumerate(parts):
                fold[test] = f
        except ValueError:
            splitter = (GroupKFold(k, shuffle=True, random_state=seed).split(idx, groups=labels)
                        if labels is not None else plain.split(idx))
            for f, (_, test) in enumerate(splitter):
                fold[test] = f
    return fold


def comparison_folds(headline: Sequence[Any], *, validation: str, scheme: str, n: int,
                     strata: Any = None, groups: Any = None, folds: int = 5, seed: int = 0,
                     repeats: int = COMPARISON_REPEATS) -> tuple[list[np.ndarray], int, str | None]:
    """The comparison substrate's fold columns (module docstring); how many of them are the
    headline's own (they come first, so the headline is their subset); and a note, or None.

    ``headline``: the split's fold columns over the training rows (``fold``, ``fold_r1``, …).
    """
    columns = [np.asarray(c).astype(np.int64) for c in headline]
    if scheme == "time_ordered":
        return columns[:1], 1, TIME_ORDERED_ONCE
    want = max(int(repeats), COMPARISON_REPEATS)
    if validation == "internal_external":
        return [kfold_assignment(n, strata=strata, groups=groups, folds=folds, seed=seed + r)
                for r in range(want)], 0, None
    shared = len(columns)
    for r in range(shared, want):
        columns.append(kfold_assignment(n, strata=strata, groups=groups, folds=folds, seed=seed + r))
    return columns, shared, None


def unit_rows(groups: Any, n: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """(each row's unit code, each unit's rows) with units in order of first appearance; every row
    is its own unit without ``groups``."""
    import pandas as pd

    if groups is None:
        return np.arange(n), [np.asarray([i]) for i in range(n)]
    codes = pd.factorize(unit_labels(groups, n))[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    return codes, [order[bounds[u]:bounds[u + 1]] for u in range(codes.max() + 1)]


def draw_units(rng: np.random.Generator, n_units: int) -> np.ndarray:
    """One bootstrap draw of whole units: ``n_units`` unit indices with replacement, the one draw
    every resample here makes (so an independent implementation can replay it)."""
    return rng.integers(0, n_units, n_units)


def nested_folds(n: int, *, groups: Any = None, folds: int = 5,
                 rng: np.random.Generator) -> np.ndarray:
    """One repetition's folds for Bates, Hastie & Tibshirani's nested cross-validation: ``1 … K``
    per row, ``0`` for the units left over (Bates et al.'s ``nested_cv_helper``: the folds are
    equal in units, and the ``U mod K`` units a shuffle leaves last sit out this repetition)."""
    codes, rows_of = unit_rows(groups, n)
    U = len(rows_of)
    k = int(folds)
    unit_fold = np.zeros(U, dtype=np.int64)
    used = (U // k) * k
    unit_fold[:used] = np.arange(used) % k + 1
    unit_fold[:used] = unit_fold[:used][rng.permutation(used)]
    out = np.zeros(n, dtype=np.int64)
    for u, rows in enumerate(rows_of):
        out[rows] = unit_fold[u]
    return out


__all__ = ["COMPARISON_REPEATS", "TIME_ORDERED_ONCE", "UNDATED", "comparison_folds", "draw_units",
           "forward_blocks", "forward_pairs", "kfold_assignment", "nested_folds", "unit_labels",
           "unit_rows"]

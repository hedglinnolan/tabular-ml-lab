"""Tier A: the seal as a picture (M2_CONTRACT §3, §11) draws exactly the seal that was drawn.

The cells are the rows the seal is drawn from, on the side the draw put them; a grouped seal never
draws a unit on both sides; an abandoned one counts every unit it split; an undetermined one says
nothing it does not know (no units, no straddle count) and never reads the outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbotab.core import seal
from turbotab.core.decisions import GrainSpec, ProjectState, TemporalSpec
from turbotab.core.seal_picture import seal_cells
from turbotab.core.stages.rows import draw_split


class Store:
    """The two things the picture reads from a DataStore."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)
        self.reads: list[list[str]] = []

    def materialize(self, columns, row_ids=None):
        self.reads.append(list(columns))
        out = self.frame[list(columns)]
        return out if row_ids is None else out.loc[np.asarray(row_ids)]


def _table(n_people: int = 120, visits: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pid = np.repeat([f"P{i:03d}" for i in range(n_people)], visits)
    start = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 600, n_people), unit="D")
    dates = np.concatenate([[s + pd.Timedelta(days=90 * k) for k in range(visits)] for s in start])
    frame = pd.DataFrame({"pid": pid, "visit_date": pd.Series(dates).dt.strftime("%Y-%m-%d"),
                          "y": rng.normal(size=len(pid))})
    # Shuffle, so file order is not unit order (a stacked export).
    frame = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    frame.index.name = "row_id"
    return frame


def _draw(state: ProjectState, frame: pd.DataFrame, holdout: float = 0.2):
    store = Store(frame)
    universe = np.arange(len(frame), dtype=np.int64)
    d = seal.seal_inputs(state, universe, store, "regression", holdout=holdout, seed=0)
    assignment, info = draw_split(universe, holdout=holdout, seed=0, folds=5, universe=universe,
                                  **d.split_args())
    return store, universe, d, assignment, info


def _units_of(frame: pd.DataFrame, cells, column: str) -> list[str]:
    """The unit label of each drawn cell, read back from the file by row position."""
    shown = _shown_ids(frame, cells)
    return frame.loc[shown, column].tolist()


def _shown_ids(frame: pd.DataFrame, cells) -> np.ndarray:
    # Cells are whole units, in file order: the rows of the units drawn.
    return np.arange(len(frame))[: len(cells.hold)] if cells.unit is None else _rows_of_units(frame, cells)


def _rows_of_units(frame, cells):
    ids = np.arange(len(frame))
    first = {}
    for pos, u in enumerate(frame["pid"].tolist()):
        first.setdefault(u, pos)
    picked, total = [], 0
    sizes = frame["pid"].value_counts()
    for u in sorted(first, key=first.get):
        if picked and total + int(sizes[u]) > 600:
            break
        picked.append(u)
        total += int(sizes[u])
    return ids[frame["pid"].isin(picked).to_numpy()]


def test_a_grouped_seal_draws_whole_units_on_the_side_the_draw_put_them():
    frame = _table()
    state = ProjectState(grain=GrainSpec(grain="repeated", id_column="pid"))
    store, universe, d, assignment, info = _draw(state, frame)
    cells = seal_cells(draw=d, universe=universe, info=info, store=store)
    assert cells.state == "grouped" and not cells.exploratory and cells.column == "pid"
    assert cells.n_rows == len(frame) and cells.n_holdout == info["n_holdout"]
    assert cells.straddle == 0 and cells.n_units == 120
    side = dict(zip(assignment["row_id"], assignment["partition"] == "holdout"))
    shown = _shown_ids(frame, cells)
    assert [int(side[r]) for r in shown] == cells.hold  # each cell on the side the draw put it
    labels = _units_of(frame, cells, "pid")
    for u in set(cells.unit):  # a unit's cells are one unit's rows, all on one side
        members = [i for i, x in enumerate(cells.unit) if x == u]
        assert len({labels[i] for i in members}) == 1
        assert len({cells.hold[i] for i in members}) == 1
    assert len(set(labels)) == len(set(cells.unit))
    assert "y" not in {c for read in store.reads for c in read}  # the outcome is never read


def test_an_abandoned_seal_counts_every_unit_it_split():
    frame = _table()
    state = ProjectState(grain=GrainSpec(grain="one_row_per_unit", acknowledged=True),
                         roles={"pid": "identifier"})
    store, universe, d, assignment, info = _draw(state, frame)
    cells = seal_cells(draw=d, universe=universe, info=info, store=store)
    assert cells.state == "abandoned" and cells.exploratory and cells.column == "pid"
    held = set(assignment.loc[assignment["partition"] == "holdout", "row_id"])
    by_unit = frame.assign(held=[r in held for r in range(len(frame))]).groupby("pid")["held"]
    assert cells.straddle == int(((by_unit.sum() > 0) & (by_unit.sum() < by_unit.size())).sum()) > 0
    assert cells.n_holdout_units == int((by_unit.sum() > 0).sum())


def test_an_undetermined_seal_claims_no_units_and_reports_only_what_an_identifier_suggests():
    frame = _table()
    # Undetermined comes only from an explicit "I don't know" (M2 §12.2); no grain draws no seal.
    state = ProjectState(grain=GrainSpec(grain="unknown"))
    store, universe, d, assignment, info = _draw(state, frame)
    cells = seal_cells(draw=d, universe=universe, info=info, store=store, suggested="pid")
    assert cells.state == "undetermined" and cells.exploratory
    assert cells.unit is None and cells.straddle is None and cells.n_units is None
    held = set(assignment.loc[assignment["partition"] == "holdout", "row_id"])
    held_ids = set(frame.loc[sorted(held), "pid"])
    both = held_ids & set(frame.loc[sorted(set(range(len(frame))) - held), "pid"])
    assert cells.evidence == (f"`pid` repeats: `{len(both):,}` of its `{len(held_ids):,}` held-out "
                              f"values also train.")
    assert cells.hold == [int(r in held) for r in range(min(600, len(frame)))]


def test_a_chronological_seal_lays_units_out_by_their_last_visit_the_latest_held_out():
    frame = _table()
    state = ProjectState(grain=GrainSpec(grain="repeated", id_column="pid"),
                         temporal=TemporalSpec(temporal=True, time_column="visit_date"))
    store, universe, d, assignment, info = _draw(state, frame)
    cells = seal_cells(draw=d, universe=universe, info=info, store=store)
    assert cells.chronological and cells.time_column == "visit_date" and cells.boundary
    assert cells.unit_time is not None and len(cells.unit_time) == len(set(cells.unit))
    held_t = [cells.unit_time[u] for u, h in zip(cells.unit, cells.hold) if h]
    train_t = [cells.unit_time[u] for u, h in zip(cells.unit, cells.hold) if not h]
    assert held_t and train_t and min(held_t) >= max(train_t)  # the latest units are held out
    assert cells.straddle == 0

"""The generic diff: it shows what changed most, and stays fast on wide tables."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from turbotab.core.consequences import (
    CAPTION_WORDS, MAX_VIEWS, TITLE_WORDS, PreviewResult, diff_views, words,
)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(10, 2, size=n),
        "kind": rng.choice(["x", "y"], size=n),
    }, index=pd.Index(np.arange(100, 100 + n), name="row_id"))


def _budgets(views):
    for v in views:
        assert words(v.title) <= TITLE_WORDS, v.title
        assert words(v.caption) <= CAPTION_WORDS, v.caption
    PreviewResult(kind="x", views=views, basis="b")  # every view validates in the result


def test_the_most_changed_column_is_the_one_shown():
    before = _frame()
    after = before.copy()
    after["a"] = after["a"] + 0.05  # a nudge
    after["c"] = after["c"] * 3  # a big move
    views = diff_views(before, after, None)
    dist = next(v for v in views if v.kind == "distribution")
    assert dist.column == "c"
    focus = next(v for v in views if v.kind == "table_focus")
    assert focus.columns_after[0] == "c" and focus.n_affected_columns == 2
    assert all(col in ("a", "c") for _, col in focus.changed)
    _budgets(views)


def test_nothing_changed_shows_nothing():
    before = _frame()
    assert diff_views(before, before.copy(), None) == []


def test_dropped_rows_lead_with_the_row_flow():
    before = _frame()
    after = before.iloc[: len(before) // 2]
    views = diff_views(before, after, None)
    assert views[0].kind == "row_flow"
    assert views[0].after[-1].n == len(after) and views[0].after[-1].dropped == len(before) - len(after)
    _budgets(views)


def test_added_and_removed_columns_show_in_the_lineage():
    before = _frame()
    after = before.drop(columns=["b"]).assign(a_log=np.log1p(before["a"].abs()))
    views = diff_views(before, after, None)
    lineage = next(v for v in views if v.kind == "lineage")
    assert set(lineage.emphasis) == {"a_log", "b"}
    matrix = {n.column for n in lineage.after.nodes if n.lane == "matrix"}
    assert "a_log" in matrix and "b" not in matrix
    _budgets(views)


def test_a_relationship_that_moves_is_shown_only_when_it_moves_a_lot():
    before = _frame()
    after = before.copy()
    after["b"] = before["a"] * 2 + np.random.default_rng(1).normal(scale=0.1, size=len(before))
    views = diff_views(before, after, None)
    rel = next(v for v in views if v.kind == "relationship")
    assert set(rel.emphasis) == {"a", "b"} and abs(rel.r_after) > 0.9 and abs(rel.r_before) < 0.2
    assert len(rel.points_before) <= 800
    small = before.copy()
    small["b"] = small["b"] + 0.01
    assert not any(v.kind == "relationship" for v in diff_views(before, small, None))
    assert len(views) <= MAX_VIEWS


def test_a_wide_frame_with_one_changed_column_is_fast():
    rng = np.random.default_rng(0)
    before = pd.DataFrame(rng.normal(size=(1_000, 20_000)), columns=[f"g{i:05d}" for i in range(20_000)])
    after = before.copy()
    after["g01234"] = after["g01234"] * 4 + 2
    started = time.perf_counter()
    views = diff_views(before, after, None)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"{elapsed:.2f}s"
    assert views[0].kind == "distribution" and views[0].column == "g01234"
    _budgets(views)

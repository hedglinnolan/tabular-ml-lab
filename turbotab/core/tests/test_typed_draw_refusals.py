"""The split and cohort stages refuse in words with a code and ways forward (SWARM_PLAN §8's typed
fit-time refusal; the build's package 3, item (e)).

The held-out draw's refusals (a cut-point that holds no row out, leaves none to train, reads no
time, reads the outcome or names a column the table lacks; a draw by time whose column is missing,
unreadable or mostly undated; the grain not yet answered), the missing column internal–external
validation folds by, and the cohort's too few rows reached the fit as a ``ValueError``: the fit's
status served the error with no refusal, so the swarm's record and its crash sweep classed each as
a crash. Each is now raised as ``graph.StageRefusal``: its words exactly as before (what a person
reads does not move: a ``ValueError`` with the same text), a code, and its ways forward.

**Known truth.** Each code is the answer-time refusal's own for the same condition, read here from
that refusal (``decisions.validate`` on the same table), never from the stage; each way forward
is built here from the state the test writes (the share exit is the answer-time exit's label and
the recorded split with the usual share in place of the cut-point). Each guard both ways: the same
table and answers with the condition removed draw, or count, with no refusal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import seal
from turbotab.core.decisions import (CutPoint, ExclusionRule, GrainSpec, ProjectState, Refusal,
                                     SplitSpec, TemporalSpec, parse_decision, validate)
from turbotab.core.graph import StageRefusal
from turbotab.core.stages.rows import cohort_stage, split_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.stage_harness import Ingested

CYCLES = (2011.0, 2013.0, 2015.0, 2016.99, 2017.0, 2019.0)


def _write(tmp_path, frame: pd.DataFrame, name: str = "draw.csv") -> Ingested:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return Ingested(path, tmp_path)


def cycles(tmp_path, *, undated: int = 0, text_wave: bool = False) -> tuple[Ingested, pd.DataFrame]:
    """Sixty rows, ten in each of six cycles; ``undated`` rows with no cycle; with ``text_wave`` a
    column ``wave`` none of whose values reads as a date or a number."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"cycle": np.repeat(CYCLES, 10), "x": rng.normal(size=60)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=60)
    if undated:
        frame.loc[: undated - 1, "cycle"] = np.nan
    if text_wave:
        frame["wave"] = np.where(np.arange(60) % 2, "early", "late")
    return _write(tmp_path, frame, f"cycles{undated}{int(text_wave)}.csv"), frame


def state(split: SplitSpec, **slots) -> ProjectState:
    return ProjectState(target="y", task="regression", roles={"x": "covariate"},
                        missing="complete_case", grain=GrainSpec(grain="one_row_per_unit"),
                        split=split, **slots)


def cut_split(column: str = "cycle", at=2017) -> SplitSpec:
    return SplitSpec(holdout=0.0, seed=3, folds=3, fold_order="random",
                     cut_point=CutPoint(column=column, at=at))


def drawn(t: Ingested, s: ProjectState):
    info = t.run(target_info_stage, s)
    cohort = t.run(cohort_stage, s, {"target_info": info})
    return t.run(split_stage, s, {"cohort": cohort, "target_info": info})


def refused_by_the_draw(t: Ingested, s: ProjectState) -> StageRefusal:
    with pytest.raises(StageRefusal) as found:
        drawn(t, s)
    refusal = found.value
    # What a person reads does not move: a ValueError with the same words (``shown_as``).
    assert isinstance(refusal, ValueError) and str(refusal) == refusal.message
    assert StageRefusal.shown_as == "ValueError"
    return refusal


def answer_time_code(t: Ingested, frame: pd.DataFrame, split: SplitSpec) -> str:
    """The code the answer-time check gives the same cut-point on the same table (the
    construction's truth): ``decisions.validate`` of the split, with the table as its context."""
    body = {"kind": "set_split", **split.model_dump(mode="json")}
    with t.store() as store:
        ctx = {"columns": list(frame.columns), "target": "y", "store": store,
               "state": ProjectState(target="y", task="regression")}
        with pytest.raises(Refusal) as found:
            validate(body, ctx)
    return found.value.code


def share_exit(split: SplitSpec) -> dict:
    """The answer-time cut-point refusal's way forward, from the recorded split: the usual share
    held out instead, the scheme kept (``seal._the_cut_point_reads_its_column``)."""
    body = {**split.model_dump(mode="json"), "kind": "set_split", "holdout": seal.USUAL,
            "cut_point": None, "fold_order": None}
    return {"label": f"Hold out {seal.USUAL:.0%} instead",
            "decision": parse_decision(body).model_dump(mode="json")}


# ── the cut-point's draw ─────────────────────────────────────────────────────


@pytest.mark.parametrize("column, at", [("wave", 2017), ("cycle", "soon"), ("cycle", 2030),
                                        ("cycle", 2011), ("y", 0.5)],
                         ids=["no_such_column", "unreadable", "holds_nothing", "trains_nothing",
                              "on_the_outcome"])
def test_a_cut_point_the_draw_refuses_is_refused_with_the_answers_code_and_its_way_forward(
        tmp_path, column, at):
    t, frame = cycles(tmp_path)
    split = cut_split(column, at)
    refusal = refused_by_the_draw(t, state(split))
    assert refusal.code == answer_time_code(t, frame, split)
    assert refusal.exits == [share_exit(split)]


def test_more_than_a_tenth_undated_is_refused_with_the_answers_code(tmp_path):
    t, frame = cycles(tmp_path, undated=7)
    assert int(frame["cycle"].isna().sum()) == 7 and 7 > 0.1 * len(frame)
    split = cut_split()
    refusal = refused_by_the_draw(t, state(split))
    assert refusal.code == answer_time_code(t, frame, split) == "cut_point_undated"
    assert refusal.exits == [share_exit(split)]


@pytest.mark.parametrize("rule, code", [
    (ExclusionRule(column="cycle", high=2016, reason="early"), "cut_point_holds_no_analyzed_row"),
    (ExclusionRule(column="cycle", low=2017, reason="late"), "cut_point_trains_no_analyzed_row"),
])
def test_an_exclusion_that_empties_a_side_of_the_cut_point_is_refused_with_its_code(
        tmp_path, rule, code):
    """The draw's own refusals ('No analyzed row would be held out', '... would train before
    ...'): the cut-point reads on the rows with the outcome recorded, so the answer-time check
    passes it, and the exclusion recorded after it empties one side (counts by pandas)."""
    t, frame = cycles(tmp_path)
    kept = frame["cycle"].between(rule.low if rule.low is not None else -np.inf,
                                  rule.high if rule.high is not None else np.inf)
    later = frame["cycle"] >= 2017
    assert int((kept & later).sum()) == 0 or int((kept & ~later).sum()) == 0
    split = cut_split()
    refusal = refused_by_the_draw(t, state(split, exclusions=[rule]))
    assert refusal.code == code
    assert refusal.exits == [share_exit(split)]


def test_a_cut_point_with_rows_on_both_sides_is_drawn_with_no_refusal(tmp_path):
    """Silent: the same table and cut-point at 2017 hold twenty rows out and train forty (pandas),
    and an exclusion of the earliest cycle leaves both sides analyzed rows."""
    t, frame = cycles(tmp_path)
    split = drawn(t, state(cut_split()))
    a = split.frames["assignment"]
    assert int((a["partition"] == "holdout").sum()) == int((frame["cycle"] >= 2017).sum()) == 20
    thinned = drawn(t, state(cut_split(), exclusions=[
        ExclusionRule(column="cycle", low=2012, reason="not the first")]))
    assert int((thinned.frames["assignment"]["partition"] == "holdout").sum()) == 20


# ── the draw by time, by share ───────────────────────────────────────────────

RANDOM_INSTEAD = {"label": "Draw the held-out rows at random instead",
                  "decision": {"kind": "set_temporal", "temporal": False, "time_column": None}}


@pytest.mark.parametrize("column, undated, text, code", [
    ("wave", 0, False, "unknown_column"),
    ("wave", 0, True, "time_column_unreadable"),
    ("cycle", 7, False, "time_column_undated"),
], ids=["no_such_column", "unreadable", "undated"])
def test_a_draw_by_time_whose_column_cannot_order_is_refused_with_its_code(tmp_path, column,
                                                                            undated, text, code):
    t, frame = cycles(tmp_path, undated=undated, text_wave=text)
    split = SplitSpec(holdout=0.2, seed=3, folds=3)
    s = state(split, temporal=TemporalSpec(temporal=True, time_column=column))
    refusal = refused_by_the_draw(t, s)
    assert refusal.code == code
    assert refusal.exits == [RANDOM_INSTEAD]
    parse_decision(refusal.exits[0]["decision"])  # a decision the model reads


def test_a_draw_by_time_on_a_readable_column_is_drawn_with_no_refusal(tmp_path):
    """Silent: the latest fifth by cycle held out (the chronological draw), with seven undated rows
    of sixty refused and six (a tenth) drawn."""
    t, frame = cycles(tmp_path)
    split = SplitSpec(holdout=0.2, seed=3, folds=3)
    out = drawn(t, state(split, temporal=TemporalSpec(temporal=True, time_column="cycle")))
    held = out.frames["assignment"].loc[lambda a: a["partition"] == "holdout", "row_id"]
    assert set(frame.loc[held, "cycle"]) <= {2017.0, 2019.0}
    t6, _ = cycles(tmp_path, undated=6)
    drawn(t6, state(split, temporal=TemporalSpec(temporal=True, time_column="cycle")))


# ── internal–external validation's cluster ───────────────────────────────────


def test_a_cluster_column_the_table_lacks_is_refused_with_the_answers_code(tmp_path):
    """The answer-time check refuses a cluster that is no column (``unknown_column``, with its way
    forward); a table that no longer holds it at the draw is refused the same way, not crashed."""
    t, frame = cycles(tmp_path)
    split = SplitSpec(holdout=0.2, seed=3, folds=3, validation="internal_external", cluster="site")
    with pytest.raises(Refusal) as answered:
        validate({"kind": "set_split", **split.model_dump(mode="json")},
                 {"columns": list(frame.columns), "target": "y"})
    refusal = refused_by_the_draw(t, state(split))
    assert refusal.code == answered.value.code == "unknown_column"
    assert refusal.exits == [{"label": e["label"], "decision": e["decision"]}
                             for e in answered.value.exits]
    present = SplitSpec(holdout=0.2, seed=3, folds=3, validation="internal_external",
                        cluster="cycle")
    drawn(t, state(present))  # silent: a column of the table with six levels


# ── the grain first ──────────────────────────────────────────────────────────


def test_a_draw_with_no_grain_is_refused_with_a_code(tmp_path):
    t, _ = cycles(tmp_path)
    s = state(SplitSpec(holdout=0.2, seed=3, folds=3)).model_copy(update={"grain": None})
    refusal = refused_by_the_draw(t, s)
    assert refusal.code == "grain_first" and refusal.message == seal.GRAIN_FIRST
    assert refusal.exits == []


# ── the cohort's too few rows ────────────────────────────────────────────────


def test_a_cohort_that_leaves_too_few_rows_is_refused_with_the_answers_code(tmp_path):
    """Three rows kept by an exclusion (pandas), below the floor: the cohort refuses with the
    answer-time code (``too_few_rows``, ``core.row_floor``) and its way forward, the rule dropped,
    which the decision model reads; with the floor's own count of rows kept it is silent."""
    from turbotab.core.row_floor import FEWEST_ROWS

    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"age": np.arange(20, 80), "x": rng.normal(size=60)})
    frame["y"] = frame["x"] + rng.normal(scale=0.5, size=60)
    t = _write(tmp_path, frame, "ages.csv")
    narrow = ExclusionRule(column="age", low=30, high=32, reason="a narrow band")
    assert int(frame["age"].between(30, 32).sum()) == 3 < FEWEST_ROWS
    s = ProjectState(target="y", task="regression", roles={"x": "covariate", "age": "covariate"},
                     missing="complete_case", grain=GrainSpec(grain="one_row_per_unit"),
                     exclusions=[narrow])
    info = t.run(target_info_stage, s)
    with pytest.raises(StageRefusal) as found:
        t.run(cohort_stage, s, {"target_info": info})
    refusal = found.value
    assert refusal.code == "too_few_rows" and str(refusal) == refusal.message
    assert refusal.message.startswith("The answers recorded leave 3 of the 60 rows in the table")
    drop = d.SetExclusions(rules=[]).model_dump(mode="json")
    assert [e["decision"] for e in refusal.exits] == [drop]
    assert refusal.exits[0]["label"].startswith("Drop the rule: ")
    wide = narrow.model_copy(update={"high": 30 + FEWEST_ROWS - 1})
    assert int(frame["age"].between(30, 30 + FEWEST_ROWS - 1).sum()) == FEWEST_ROWS
    out = t.run(cohort_stage, s.model_copy(update={"exclusions": [wide]}), {"target_info": info})
    assert out.data["n_final"] == FEWEST_ROWS

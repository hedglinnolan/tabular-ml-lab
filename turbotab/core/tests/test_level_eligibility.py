"""The categorical eligibility rule (SWARM_PLAN §5 and §8): exclude rows by a level of a coded
column, the codes that record a refused or a don't-know answer named on their own.

Every expected count here comes from the table each test builds, never from the engine: a refused
or don't-know answer cannot confirm that a row is eligible (STROBE item 13a's "confirmed eligible"),
so, like a blank, it leaves on a line of its own, named with its code, unless the rule keeps it,
and a code the rule does not name is read as an answer like any other level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import LevelRule, ProjectState, Refusal, validate
from turbotab.core.stages.rows import cohort_flow, cohort_stage, rule_keep
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests.stage_harness import Ingested

# NHANES DIQ010, "Doctor told you have diabetes": 1 yes, 2 no, 3 borderline, 7 refused, 9 don't
# know. Ten rows: two yes, three no, one borderline, one refused, two don't know, one blank.
CODES = [1, 2, 3, 7, 9, np.nan, 2, 1, 9, 2]
REASON = "diagnosed diabetes"


def coded() -> pd.DataFrame:
    return pd.DataFrame({"DIQ010": CODES, "y": np.arange(10, dtype=float)})


def bang(**over) -> LevelRule:
    return LevelRule(**{"column": "DIQ010", "exclude": [1], "refused": [7], "dont_know": [9],
                        "reason": REASON, **over})


def flow(rule, frame=None):
    steps, kept = cohort_flow(coded() if frame is None else frame, target="y", rules=[rule],
                              missing=None, predictor_columns=[])
    return {s["key"]: s for s in steps}, kept


# ── the participant flow ─────────────────────────────────────────────────────


def test_a_refused_or_dont_know_answer_leaves_on_a_named_line_of_its_own():
    steps, kept = flow(bang())
    blank = steps["exclusion:0:not_recorded"]
    assert (blank["label"], blank["dropped"], blank["n"]) == ("`DIQ010` not recorded", 1, 9)
    refused = steps["exclusion:0:refused"]
    assert (refused["label"], refused["dropped"], refused["n"]) == ("`DIQ010` refused (`7`)", 1, 8)
    unknown = steps["exclusion:0:dont_know"]
    assert (unknown["label"], unknown["dropped"], unknown["n"]) == ("`DIQ010` don't know (`9`)", 2, 6)
    for line in (refused, unknown):
        assert line["reason"] == "a refused or don't-know answer cannot confirm the row is eligible"
    level = steps["exclusion:0"]
    assert (level["label"], level["dropped"], level["n"], level["reason"]) == (
        "`DIQ010` not `1`", 2, 4, REASON)
    # Who stays: the three no's and the borderline answer.
    assert sorted(kept.tolist()) == [1, 2, 6, 9]


def test_a_rule_that_keeps_unconfirmed_answers_keeps_them_and_each_line_says_so():
    steps, kept = flow(bang(missing="keep"))
    assert steps["exclusion:0:not_recorded"]["label"] == "`DIQ010` not recorded: `1` kept"
    assert steps["exclusion:0:refused"]["label"] == "`DIQ010` refused (`7`): `1` kept"
    assert steps["exclusion:0:dont_know"]["label"] == "`DIQ010` don't know (`9`): `2` kept"
    assert all(steps[f"exclusion:0:{k}"]["dropped"] == 0
               for k in ("not_recorded", "refused", "dont_know"))
    level = steps["exclusion:0"]
    assert level["label"] == "`DIQ010` not `1`, or not recorded, refused or don't know"
    assert level["dropped"] == 2 and sorted(kept.tolist()) == [1, 2, 3, 4, 5, 6, 8, 9]


def test_a_code_the_rule_does_not_name_is_an_answer_like_any_other():
    """Only a named code is read as refused or don't know: with none named, a 7 or a 9 is a level
    the rule does not exclude, so it stays, and no line names it."""
    steps, kept = flow(bang(refused=[], dont_know=[]))
    assert "exclusion:0:refused" not in steps and "exclusion:0:dont_know" not in steps
    assert steps["exclusion:0:not_recorded"]["dropped"] == 1  # the blank still cannot confirm
    assert sorted(kept.tolist()) == [1, 2, 3, 4, 6, 8, 9]


def test_a_level_matches_however_it_is_spelled():
    """``1`` names the level whether the column holds 1.0 (a coded column read as numbers) or the
    text "1"; a text column's own words are levels too."""
    as_text = coded().assign(DIQ010=[None if pd.isna(v) else str(int(v)) for v in CODES])
    assert rule_keep(as_text, bang()).tolist() == rule_keep(coded(), bang()).tolist()
    assert rule_keep(coded(), bang(exclude=["1"], refused=["7.0"])).tolist() == \
        rule_keep(coded(), bang()).tolist()
    words = pd.DataFrame({"told": ["Yes", "No", "Refused", "Don't know", "No"]})
    rule = LevelRule(column="told", exclude=["Yes"], refused=["Refused"], dont_know=["Don't know"],
                     reason=REASON)
    assert rule_keep(words, rule).tolist() == [False, True, False, False, True]


def test_the_rows_stage_applies_the_rule(tmp_path):
    """Through the real target and cohort stages, over a table on disk."""
    path = tmp_path / "coded.csv"  # the ten rows six times over: enough for the design's floor
    pd.concat([coded()] * 6, ignore_index=True).assign(age=np.arange(60) % 30 + 40).to_csv(
        path, index=False)
    table = Ingested(path, tmp_path)
    state = ProjectState(target="y", task="regression", roles={"age": "covariate"},
                         exclusions=[bang()], missing="complete_case")
    info = table.run(target_info_stage, state)
    cohort = table.run(cohort_stage, state, {"target_info": info})
    steps = {s["key"]: s for s in cohort.data["steps"]}
    assert [steps[k]["dropped"] for k in ("exclusion:0:not_recorded", "exclusion:0:refused",
                                          "exclusion:0:dont_know", "exclusion:0")] == [6, 6, 12, 12]
    assert cohort.data["n_final"] == 24


# ── the rule's own shape ─────────────────────────────────────────────────────


def test_a_level_named_in_two_roles_is_refused():
    with pytest.raises(ValidationError, match="`7`"):
        bang(exclude=[1, 7])
    with pytest.raises(ValidationError, match="`9`"):
        bang(refused=[7, "9.0"])  # 9.0 and 9 name one level
    with pytest.raises(ValidationError):
        bang(exclude=[])  # a rule that excludes no level excludes nobody


def test_distinct_levels_are_accepted():
    rule = bang(exclude=[1, 3], refused=[7, 77], dont_know=[9, 99])
    assert rule.reads() == ["DIQ010"] and d.as_rule(rule.model_dump(mode="json")) == rule


# ── the validators ───────────────────────────────────────────────────────────


class FrameStore:
    """The few DataStore calls the validators make, over a frame."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns=None, row_ids=None):
        out = self.frame[list(columns) if columns is not None else self.columns]
        return out if row_ids is None else out.loc[np.asarray(row_ids)]


COLUMN_INFO = {"DIQ010": {"dtype": "numeric", "n_unique": 5}, "y": {"dtype": "numeric", "n_unique": 10},
               "told": {"dtype": "categorical", "n_unique": 4}}


def context(frame: pd.DataFrame) -> dict:
    return {"columns": list(frame.columns), "column_info": COLUMN_INFO, "target": "y",
            "task": "regression", "store": FrameStore(frame),
            "state": ProjectState(target="y", task="regression")}


def answer(rule) -> dict:
    return {"kind": "set_exclusions", "rules": [rule.model_dump(mode="json")]}


def test_a_rule_on_a_column_the_table_lacks_is_refused():
    with pytest.raises(Refusal) as found:
        validate(answer(bang(column="DIQ999")), context(coded()))
    assert found.value.code == "unknown_column" and "`DIQ999`" in found.value.message
    assert found.value.exits[0]["decision"] == {"kind": "set_exclusions", "rules": []}


def test_a_rule_on_a_categorical_column_is_accepted():
    """The range rule's "not numbers" refusal is a range's: a level rule reads text levels too."""
    frame = pd.DataFrame({"told": ["Yes", "No", "Refused", "No"], "y": [1.0, 2.0, 3.0, 4.0]})
    rule = LevelRule(column="told", exclude=["Yes"], refused=["Refused"], reason=REASON)
    validate(answer(rule), context(frame))
    validate(answer(bang()), context(coded()))


def test_a_rule_whose_excluded_levels_no_row_holds_is_refused():
    """Excluding `Yes` from a column coded 1 and 2 would exclude nobody: refused, with the levels
    the column does hold, rather than recorded as an exclusion that removed no one."""
    with pytest.raises(Refusal) as found:
        validate(answer(bang(exclude=["Yes"])), context(coded()))
    assert found.value.code == "no_such_level"
    assert "`Yes`" in found.value.message and "`1`, `2`, `3`, `7`, `9`" in found.value.message
    # One excluded level the column holds is enough, and a refused code nobody gave is no reason.
    validate(answer(bang(exclude=[1, "Yes"], refused=[7, 77])), context(coded()))


# ── the preview ──────────────────────────────────────────────────────────────


def preview_caption(tmp_path, rules) -> str:
    from turbotab.core import fact_previews, row_previews  # noqa: F401 - registers the builders
    from turbotab.core.consequences import PreviewContext, plan
    from turbotab.core.datastore import DataStore, ingest
    from turbotab.core.decisions import parse_decision

    path = tmp_path / "coded.csv"
    coded().to_csv(path, index=False)
    ingest(path, tmp_path / "coded.parquet")
    with DataStore(tmp_path / "coded.parquet", 2 << 30) as store:
        ctx = PreviewContext(project_id="p", state=ProjectState(target="y"), datastore=store,
                             artifact={}.get, training_row_ids=None, cohort_row_ids=None,
                             sealed_row_ids=None)
        decision = parse_decision({"kind": "set_exclusions",
                                   "rules": [r.model_dump(mode="json") for r in rules]})
        return plan(decision, ctx, basis="b").views[0].caption


def test_the_preview_counts_an_excluded_level_and_the_unconfirmed_answers_apart(tmp_path):
    """Of ten rows, the two yes's are at the excluded level, and the blank, the refused answer and
    the two don't-knows are not confirmed eligible; never "outside these ranges"."""
    assert preview_caption(tmp_path, [bang()]) == (
        "`6` of `10` rows would leave: `2` at an excluded level, `4` not confirmed eligible; `4` "
        "stay.")


def test_the_preview_of_ranges_alone_says_what_it_said(tmp_path):
    """Silent: a decision of range rules alone keeps its caption's words."""
    rule = d.ExclusionRule(column="DIQ010", low=2, high=3, reason="a range")
    assert preview_caption(tmp_path, [rule]) == (
        "`6` of `10` rows would leave: `5` outside these ranges, `1` with no value to check; `4` "
        "stay.")


# ── the methods sentence ─────────────────────────────────────────────────────


def test_the_sentence_counts_what_the_flow_counts_and_names_each_code():
    text = voice.sentence_for(d.SetExclusions(rules=[bang()]), ProjectState(target="y"),
                              {"frame": coded()})
    # the blank, the refused answer, the two don't-knows and the two yes's
    assert text == ("`6` rows with `DIQ010` `1`, or refused (`7`), don't know (`9`) or not "
                    "recorded were excluded as diagnosed diabetes.")
    kept = voice.sentence_for(d.SetExclusions(rules=[bang(exclude=[1, 3], missing="keep")]),
                              ProjectState(target="y"), {"frame": coded()})
    assert kept == "`3` rows with `DIQ010` `1` or `3` were excluded as diagnosed diabetes."

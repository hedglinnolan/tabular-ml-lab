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
    assert rule_keep(coded(), bang(refused=["07"])).tolist() == rule_keep(coded(), bang()).tolist()
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
    """The rows at the excluded level leave as the rule's reason, and the blank, refused and
    don't-know answers on a clause of their own as not confirmed eligible (the package 4
    verifier's V4), each count the flow's: of ``coded()``'s ten rows, the two coded 1, then the
    refused answer, the two don't-knows and the blank (by pandas)."""
    frame = coded()
    at_level = int((frame["DIQ010"] == 1).sum())
    unconfirmed = int((frame["DIQ010"].isna() | frame["DIQ010"].isin([7, 9])).sum())
    assert (at_level, unconfirmed) == (2, 4)
    text = voice.sentence_for(d.SetExclusions(rules=[bang()]), ProjectState(target="y"),
                              {"frame": frame})
    assert text == (
        f"`{at_level}` rows with `DIQ010` `1` were excluded as diagnosed diabetes, and "
        f"`{unconfirmed}` with `DIQ010` refused (`7`), don't know (`9`) or not recorded were "
        f"excluded as not confirmed eligible.")
    kept = voice.sentence_for(d.SetExclusions(rules=[bang(exclude=[1, 3], missing="keep")]),
                              ProjectState(target="y"), {"frame": coded()})
    assert kept == "`3` rows with `DIQ010` `1` or `3` were excluded as diagnosed diabetes."


# ── repair round 1: the verifier's problems ──────────────────────────────────


def test_the_refusal_of_a_level_no_row_holds_says_only_what_it_checks():
    """P8: no row holds the excluded level, so no row would leave at it; the blank, the refused
    answer and the don't-know answer would still leave as not confirmed eligible (three rows, by
    construction), and the refusal counts them rather than saying the rule excludes no one. With
    those answers kept, none would leave, and the refusal says nothing of them."""
    frame = pd.DataFrame({"DIQ010": [2, 2, 7, 9, np.nan, 3], "y": np.arange(6, dtype=float)})
    with pytest.raises(Refusal) as found:
        validate(answer(bang()), context(frame))
    assert found.value.code == "no_such_level"
    assert found.value.message == (
        "No row of `DIQ010` holds `1`, so no row would leave at an excluded level; `3` would "
        "leave only as not confirmed eligible. `DIQ010` holds `2`, `3`, `7`, `9`.")
    with pytest.raises(Refusal) as found:
        validate(answer(bang(missing="keep")), context(frame))
    assert found.value.message == (
        "No row of `DIQ010` holds `1`, so no row would leave at an excluded level. `DIQ010` holds "
        "`2`, `3`, `7`, `9`.")


def test_a_text_level_that_is_not_a_number_is_matched_as_written():
    """Silent, beside the spellings of one number: a text level that reads as no number names
    only itself, so `yes` does not name `Yes`."""
    words = pd.DataFrame({"told": ["Yes", "No", "yes"]})
    rule = LevelRule(column="told", exclude=["yes"], reason=REASON)
    assert rule_keep(words, rule).tolist() == [True, True, False]


def test_with_unconfirmed_answers_kept_the_preview_counts_only_the_rows_that_leave(tmp_path):
    """P13: with the blank, refused and don't-know answers kept, only the two rows at the excluded
    level leave, and the caption says no count of unconfirmed rows that leave."""
    assert preview_caption(tmp_path, [bang(missing="keep")]) == (
        "`2` of `10` rows would leave at an excluded level; `8` stay.")


# ── repair round 2: the verifier's problems ──────────────────────────────────

# Eight rows: a refused answer with no outcome recorded (row 2), a child's don't-know (row 6) that
# an adults-only rule before the level rule removes, and three unconfirmed answers that reach the
# level rule (the don't-know in row 3, the blank in row 4, the refused answer in row 7). No row is
# coded 1.
REACHING = pd.DataFrame({"age": [30, 30, 30, 30, 30, 30, 12, 30],
                         "DIQ010": [2, 2, 7, 9, np.nan, 3, 9, 7],
                         "y": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0]})
ADULTS = d.ExclusionRule(column="age", low=18, reason="adults only")


def test_the_refusal_counts_only_the_unconfirmed_rows_the_flow_brings_to_the_rule():
    """V1: the rows that would still leave as not confirmed eligible are counted as the flow counts
    them: among the rows with the outcome recorded that the rules before it keep, not over every
    row of the table (five here, by pandas, against the three that reach the rule)."""
    frame = REACHING
    unconfirmed = frame["DIQ010"].isna() | frame["DIQ010"].isin([7, 9])
    reaching = frame["y"].notna() & (frame["age"] >= 18)
    n = int((unconfirmed & reaching).sum())
    assert (n, int(unconfirmed.sum())) == (3, 5)
    ctx = {**context(frame), "column_info": {**COLUMN_INFO,
                                             "age": {"dtype": "numeric", "n_unique": 2}}}
    rules = [ADULTS.model_dump(mode="json"), bang().model_dump(mode="json")]
    with pytest.raises(Refusal) as found:
        validate({"kind": "set_exclusions", "rules": rules}, ctx)
    assert found.value.code == "no_such_level"
    assert found.value.message == (
        f"No row of `DIQ010` holds `1`, so no row would leave at an excluded level; `{n}` would "
        f"leave only as not confirmed eligible. `DIQ010` holds `2`, `3`, `7`, `9`.")
    # Alone, the rule reaches every adult and child with the outcome recorded: four, by pandas.
    alone = int((unconfirmed & frame["y"].notna()).sum())
    assert alone == 4
    with pytest.raises(Refusal) as found:
        validate(answer(bang()), ctx)
    assert f"`{alone}` would leave only as not confirmed eligible" in found.value.message


def sentence(rules, frame=None):
    return voice.sentence_for(d.SetExclusions(rules=rules), ProjectState(target="y"),
                              {"frame": coded() if frame is None else frame})


def test_the_sentence_counts_the_unconfirmed_rows_the_flow_brings_to_the_rule():
    """V4 beside V1: after the adults-only rule, only the rows it keeps with the outcome recorded
    reach the level rule, and each clause counts those (by pandas), with the total in all."""
    frame = REACHING.assign(DIQ010=[2, 1, 7, 9, np.nan, 3, 9, 1])
    reaching = frame["y"].notna() & (frame["age"] >= 18)
    children = int((frame["y"].notna() & (frame["age"] < 18)).sum())
    at_level = int((reaching & (frame["DIQ010"] == 1)).sum())
    unconfirmed = int((reaching & (frame["DIQ010"].isna() | frame["DIQ010"].isin([7, 9]))).sum())
    assert (children, at_level, unconfirmed) == (1, 2, 2)
    assert sentence([ADULTS, bang()], frame) == (
        f"`{children}` row with `age` below `18` or not recorded was excluded as adults only; "
        f"`{at_level}` rows with "
        f"`DIQ010` `1` were excluded as diagnosed diabetes, and `{unconfirmed}` with `DIQ010` "
        f"refused (`7`), don't know (`9`) or not recorded were excluded as not confirmed eligible, "
        f"`{children + at_level + unconfirmed}` in all.")


def test_without_the_rows_the_sentence_counts_no_clause_it_cannot_and_keeps_the_reasons_apart():
    """Silent on the split: given only the flow's total for the rule (no rows to split it), the
    sentence says the total and each reason, never a split it did not count; with no count at
    all, neither; given the unconfirmed count beside the total (as the export would pass it), the
    split; and a rule that keeps its unconfirmed answers says one clause, as before."""
    assert voice.sentence_for(d.SetExclusions(rules=[bang()]), ProjectState(target="y"),
                              {"exclusion_counts": [6]}) == (
        "`6` rows with `DIQ010` `1`, refused (`7`), don't know (`9`) or not recorded were "
        "excluded, at `1` as diagnosed diabetes and otherwise as not confirmed eligible.")
    assert voice.sentence_for(d.SetExclusions(rules=[bang()]), ProjectState(target="y"), {}) == (
        "Rows with `DIQ010` `1` were excluded as diagnosed diabetes, and rows with `DIQ010` "
        "refused (`7`), don't know (`9`) or not recorded were excluded as not confirmed eligible.")
    # Given the flow's unconfirmed count beside its total (the export's counts), the split is said.
    assert voice.sentence_for(d.SetExclusions(rules=[bang()]), ProjectState(target="y"),
                              {"exclusion_counts": [6], "exclusion_unconfirmed": [4]}) == (
        "`2` rows with `DIQ010` `1` were excluded as diagnosed diabetes, and `4` with `DIQ010` "
        "refused (`7`), don't know (`9`) or not recorded were excluded as not confirmed eligible.")
    assert sentence([bang(missing="keep")]) == (
        "`2` rows with `DIQ010` `1` were excluded as diagnosed diabetes.")

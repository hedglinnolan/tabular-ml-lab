"""The record's assembler (DRIVER_PLAN D3), first on constructed inputs, then end to end.

The constructed tests take their truths from construction: a transcript whose counts are written
here, decisions and previews built by hand, a curve built in ``core/mattered.py``'s own models. The
end-to-end test drives ``clinical_risk.csv`` (its declared truth: ``truths.FIXTURE_TRUTHS``) in
an in-process server in analyst mode (``server_drive.local_server``), every decision posted with a
note, through two fits, a preview of each kind, the seal and a change after it, to the export. Its
record is compared with what each route serves, with the run's own notes, with hashlib over the
fixture's bytes and the bundle's, with pandas over the fixture, and with git for the engine's
commit; then every check runs on it and gives what the run's construction implies. Five more runs
on the same server harness: the NHANES journey under Estimate with a change after the lock;
``clinical_risk.csv`` under Predict with reverts between fits (a revert alone, then a revert with
a new record); the NHANES journey with a rule tried and undone before the lock, then a revert
after it; the NHANES journey with each adjustment group previewed with its own answer, one fit and
a lock no fit record's plan holds; and ``clinical_risk.csv`` with a revert and its revert posted
before a fit completed, then a third. Each one's truths come from its construction, read against
what ``GET …/fits`` serves and, for which records are in force, ``provenance.in_force``.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from research.swarm import assemble as A
from research.swarm import checks as C
from research.swarm.record import Claim, validate

# ── spend ────────────────────────────────────────────────────────────────────

# Two assistant messages; the first streamed over two lines that share its id, the later line with
# its final output count; a user line with no usage.
TRANSCRIPT = [
    {"type": "user", "message": {"role": "user", "content": "the card"}},
    {"type": "assistant", "message": {"id": "msg_a", "usage": {
        "input_tokens": 3, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 40_000,
        "output_tokens": 16}}},
    {"type": "assistant", "message": {"id": "msg_a", "usage": {
        "input_tokens": 3, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 40_000,
        "output_tokens": 200}}},
    {"type": "assistant", "message": {"id": "msg_b", "usage": {
        "input_tokens": 2, "cache_creation_input_tokens": 1_000, "cache_read_input_tokens": 45_000,
        "output_tokens": 75}}},
]
FRESH = (3 + 500 + 200) + (2 + 1_000 + 75)
CACHED = 40_000 + 45_000


def transcript(path: Path, lines: list[dict[str, Any]] = TRANSCRIPT) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")
    return path


def test_spend_counts_each_message_once_at_its_final_counts(tmp_path):
    got = A.spend(transcript(tmp_path / "t.jsonl"))
    assert (got.fresh, got.cached, got.messages) == (FRESH, CACHED, 2)
    assert got.fresh != (3 + 500 + 16) + (2 + 1_000 + 75)  # not the streamed line's partial count


def test_spend_is_refused_when_a_line_cannot_be_counted_once(tmp_path):
    lines = TRANSCRIPT + [{"type": "assistant", "message": {"usage": {"output_tokens": 9}}}]
    with pytest.raises(A.SpendRefused, match="no message id"):
        A.spend(transcript(tmp_path / "t.jsonl", lines))


# ── what a decision answers, and what was examined before it ─────────────────


def test_each_decision_answers_its_questions_keyed_as_a_fixtures_truth_keys_them():
    assert A.answers_of({"kind": "confirm_readings", "items": [
        {"reading": "code_or_count", "column": "age", "value": "amount"},
        {"reading": "role", "column": "id", "value": "identifier"}]}) == [
        ("code_or_count:age", "amount"), ("role:id", "identifier")]
    assert A.answers_of({"kind": "set_column_unit", "column": "kcal", "unit": "kcal",
                         "days": 1}) == [("unit:kcal", "kcal"), ("day_count:kcal", "1")]
    assert A.answers_of({"kind": "set_roles", "roles": {"a": "exposure"}, "unconfirmed": []}) == [
        ("role:a", "exposure")]
    assert A.answers_of({"kind": "apply_repair", "finding_id": "below_detection__crp",
                         "option": "half_limit", "params": {}}) == [
        ("detection_limit:crp", "half_limit")]
    assert A.answers_of({"kind": "select_models", "models": ["linear"]}) == [
        ("models", {"models": ["linear"]})]


YES = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
NO = {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "no"}
# One decision of each kind the server keeps under a key, or in several entries (DB-V4).
KEYED = [
    {"kind": "set_adjustment", "exposure": "sugar", "answers": {"age": YES, "gender": YES}},
    {"kind": "set_forms", "forms": {"sugar": {"form": "linear"}, "age": {"form": "linear"}}},
    {"kind": "set_exposure_form", "column": "sugar", "form": "linear"},
    {"kind": "set_modification", "modifier": "gender"},
    {"kind": "view_outcome", "view": "distribution", "target": "glucose"},
    {"kind": "apply_repair", "finding_id": "outliers__kcal", "option": "winsorize"},
    {"kind": "dismiss_finding", "finding_id": "outliers__kcal"},
    {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings", "lines": []},
    {"kind": "join_files", "file": "f0123456789", "on": "seqn"},
    {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
    {"kind": "set_validation", "validation": "kfold", "folds": 5},
    {"kind": "set_outcome_scale", "column": "glucose", "scale": "log"},
]


def test_each_answer_the_server_keeps_apart_is_its_own_question():
    """DB-V4: a decision answers one question per entry the server writes (its own
    ``provenance._writes``, the slots and keys that decide which record is in force), so each
    covariate's adjustment answer is its own question, two kinds that write one entry answer one
    question, and a decision's value is the entry the server keeps."""
    from turbotab.core import decisions as D
    from turbotab.core import provenance as P

    for body in KEYED:
        model = D.parse_decision(body)
        served = model.model_dump(mode="json")
        expected = [f"adjust:{k}" if s == "adjustment" else s if k is None else f"{s}:{k}"
                    for s, k in P._writes(model)]
        assert [q for q, _ in A.answers_of(served)] == expected, body["kind"]
    assert [q for q, _ in A.answers_of(KEYED[0])] == ["adjust:age", "adjust:gender"]
    assert A.answers_of(KEYED[0])[0][1] == {**D.CovariateAnswers(**YES).model_dump(),
                                            "exposure": "sugar"}
    # One entry, two kinds: a column's form and a finding's disposition.
    forms = dict(A.answers_of(KEYED[1]))
    assert A.answers_of(KEYED[2]) == [("exposure_forms:sugar", forms["exposure_forms:sugar"])]
    assert A.answers_of(KEYED[5])[0][0] == A.answers_of(KEYED[6])[0][0] == "findings:outliers__kcal"
    # A split's scheme is written onto the draw: one slot, one question; the outcome's scale
    # writes the target.
    assert A.answers_of(KEYED[9])[0][0] == A.answers_of(KEYED[10])[0][0] == "split"
    assert A.answers_of(KEYED[11])[0][0] == A.answers_of({"kind": "set_target",
                                                          "column": "glucose"})[0][0] == "target"
    assert A.answers_of({"kind": "revert", "decision_id": "r3"}) == [
        ("revert:r3", {"decision_id": "r3"})]
    with pytest.raises(A.UnreadDecision, match="set_adjustment"):
        A.answers_of({"kind": "set_adjustment", "exposure": "sugar", "answers": {"age": 1}})


def test_a_field_the_server_fills_is_no_difference_and_any_other_field_is():
    asked = {"family": "linear", "target": None, "scores": None}
    assert A.same_option(asked, {"family": "linear", "target": "y", "scores": {"linear": {}}})
    assert not A.same_option(asked, {"family": "elastic_net", "target": "y", "scores": None})
    assert not A.same_option({"models": ["linear"]}, {"models": None})
    assert A.same_option("exposure", "exposure") and not A.same_option("exposure", "excluded")


def _rec(seq, minute, decision, note="why"):
    return {"id": f"r{seq}", "seq": seq, "at": f"2026-10-18T10:{minute:02d}:00Z", "note": note,
            "post_seal": False, "after_estimates": False, "recorded_by": "you",
            "decision": decision}


def _preview(minute, decision, refused=None):
    return {"at": f"2026-10-18T10:{minute:02d}:00Z", "decision": decision, "refused": refused}


ROLE_A = {"kind": "confirm_role", "column": "a", "role": "excluded"}
RECORDS = [_rec(1, 1, {"kind": "set_roles", "roles": {"a": "exposure", "b": "covariate"}}),
           _rec(2, 2, {"kind": "confirm_role", "column": "b", "role": "identifier"}),
           _rec(3, 3, {"kind": "confirm_role", "column": "b", "role": "covariate"}),
           _rec(4, 9, ROLE_A)]
PREVIEWS = [_preview(4, ROLE_A), _preview(5, {**ROLE_A, "role": "covariate"}, "reading_unsettled"),
            _preview(6, {**ROLE_A, "role": "flag"}), _preview(10, {**ROLE_A, "role": "time"})]
FITS = [{"id": "fit1", "seq": 1, "at": "2026-10-18T10:05:00Z", "decisions": [1, 3]}]


def test_an_option_is_examined_when_previewed_or_fitted_before_the_decision_or_while_it_stood():
    got = A.examined_of(RECORDS[3], RECORDS, PREVIEWS, FITS)
    assert [(e["how"], e["value"], e["alternative"]) for e in got] == [
        ("preview", "excluded", False),  # the option chosen, previewed: not an alternative
        ("preview", "covariate", False),  # refused, so nothing was previewed
        ("preview", "flag", True),
        ("preview", "time", True),  # at 10:10, while the answer stood: previewed, then kept
        ("fit", "exposure", True),  # record #1's answer, in force in fit 1
    ]  # record #2 was never in force in a fit
    assert got[-1]["seq"] == 1 and got[-1]["attempt"] == "fit1"
    # Record #1 answered role:a first, and its window closes at record #4 (10:09): the previews at
    # 10:04–10:06 examined it too, while it stood.
    first = A.examined_of(RECORDS[0], RECORDS, PREVIEWS, FITS)
    assert [(e["value"], e["alternative"]) for e in first] == [
        ("excluded", True), ("covariate", False), ("flag", True)]


def test_a_preview_after_the_questions_next_answer_examines_that_answer_not_this_one():
    later = _rec(5, 12, {**ROLE_A, "role": "flag"})
    got = A.examined_of(RECORDS[3], RECORDS + [later], PREVIEWS, FITS)
    assert ("preview", "time") in [(e["how"], e["value"]) for e in got]  # 10:10, before #5
    after = _preview(13, {**ROLE_A, "role": "identifier"})
    got = A.examined_of(RECORDS[3], RECORDS + [later], PREVIEWS + [after], FITS)
    assert "identifier" not in [e["value"] for e in got]


def test_an_option_examined_the_same_way_twice_is_listed_once():
    block = _rec(2, 2, {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": "a", "value": "exposure"}]})
    records = [RECORDS[0], block, _rec(3, 9, ROLE_A)]
    fits = [{"id": "fit1", "seq": 1, "at": "2026-10-18T10:05:00Z", "decisions": [1, 2]}]
    previews = [_preview(4, {**ROLE_A, "role": "flag"}), _preview(6, {**ROLE_A, "role": "flag"})]
    got = A.examined_of(records[2], records, previews, fits)
    assert [(e["how"], e["value"], e.get("seq")) for e in got] == [
        ("preview", "flag", None), ("fit", "exposure", 1)]  # #2 held the same option in fit 1


SPLIT_DRAW = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
SCHEME = {"kind": "set_validation", "folds": 5, "validation": "kfold"}


def test_a_validation_scheme_is_compared_with_the_split_by_what_each_leaves_in_the_slot():
    """DRV-INT (DRV-B's verifier, P3): ``set_split`` draws the held-out rows and sets the scheme;
    ``set_validation`` writes only the scheme onto the draw (``SetValidation.scheme()``). So the
    split held in the only fit, its scheme the one #3 records again, is no alternative examined
    for #3, and neither is a preview of that scheme; a split or a scheme with ten folds is."""
    records = [_rec(1, 1, {"kind": "set_target", "column": "y"}), _rec(2, 2, SPLIT_DRAW),
               _rec(3, 6, SCHEME)]
    fits = [{"id": "fit1", "seq": 1, "at": "2026-10-18T10:04:00Z", "decisions": [1, 2]}]
    same = [_preview(5, {**SCHEME})]
    got = A.examined_of(records[2], records, same, fits)
    assert [(e["how"], e["alternative"]) for e in got] == [("preview", False), ("fit", False)]
    other = [_rec(1, 1, {"kind": "set_target", "column": "y"}),
             _rec(2, 2, {**SPLIT_DRAW, "folds": 10}), _rec(3, 6, SCHEME)]
    got = A.examined_of(other[2], other, [_preview(5, {**SCHEME, "folds": 10})], fits)
    assert [(e["how"], e["alternative"]) for e in got] == [("preview", True), ("fit", True)]
    # The other way round: a draw chosen after a scheme previewed alone, the same or another.
    drawn = [_rec(1, 1, {"kind": "set_target", "column": "y"}), _rec(2, 6, SPLIT_DRAW)]
    for folds, alternative in ((5, False), (10, True)):
        got = A.examined_of(drawn[1], drawn, [_preview(4, {**SCHEME, "folds": folds})], [])
        assert [(e["how"], e["alternative"]) for e in got] == [("preview", alternative)], folds


def test_a_scheme_answer_leaves_the_draw_standing_so_another_draw_is_an_alternative_to_it():
    """DRV-INT round 1 (P2): a scheme answer (#2) leaves in the split slot the draw in force at
    it (#1's 20% held out) with its scheme (``decisions._scheme_onto_the_draw``). A preview of a
    draw holding out 30% while #2 stands examined another held-out share, so it is an
    alternative to #2; a preview of #1's own draw with #2's scheme, or of #2's scheme again, is
    not. The same by fit: a 30% draw (#1) in force in fit 1, then a 20% draw (#2) and a scheme
    answer (#3) with the same folds: #1's draw is an alternative examined for #3."""
    records = [_rec(1, 1, SPLIT_DRAW), _rec(2, 2, SCHEME)]
    other = _preview(3, {**SPLIT_DRAW, "holdout": 0.3})
    got = A.examined_of(records[1], records, [other], [])
    assert [(e["how"], e["value"]["holdout"], e["alternative"]) for e in got] == [
        ("preview", 0.3, True)]
    same = [_preview(3, {**SPLIT_DRAW}), _preview(4, {**SCHEME})]
    got = A.examined_of(records[1], records, same, [])
    assert [(e["how"], e["alternative"]) for e in got] == [("preview", False), ("preview", False)]
    fitted = [_rec(1, 1, {**SPLIT_DRAW, "holdout": 0.3}), _rec(2, 3, SPLIT_DRAW),
              _rec(3, 4, SCHEME)]
    fits = [{"id": "fit1", "seq": 1, "at": "2026-10-18T10:02:00Z", "decisions": [1]}]
    got = A.examined_of(fitted[2], fitted, [], fits)
    assert [(e["how"], e["seq"], e["alternative"]) for e in got] == [("fit", 1, True)]
    kept = [_rec(1, 1, SPLIT_DRAW), _rec(2, 3, {**SPLIT_DRAW, "holdout": 0.3}),
            _rec(3, 4, SCHEME)]  # #1 fitted, then the 30% draw, then the scheme: 20% is examined
    got = A.examined_of(kept[2], kept, [], fits)
    assert [(e["how"], e["seq"], e["alternative"]) for e in got] == [("fit", 1, True)]
    back = [_rec(1, 1, SPLIT_DRAW), _rec(2, 4, SCHEME)]  # the fitted draw, its scheme kept
    got = A.examined_of(back[1], back, [], fits)
    assert [(e["how"], e["seq"], e["alternative"]) for e in got] == [("fit", 1, False)]


def _adjust(seq, minute, **answers):
    return _rec(seq, minute, {"kind": "set_adjustment", "exposure": "sugar", "answers": answers})


def _adjust_preview(minute, **answers):
    return _preview(minute, {"kind": "set_adjustment", "exposure": "sugar", "answers": answers})


def test_a_preview_of_another_covariates_answer_is_not_examined_for_this_one():
    """DB-V4: group one (`age`, `gender`, #1) stood while group two (`bmi`) was previewed with its
    own answer and then recorded (#2): the preview examined `bmi`'s question, not group one's, so
    #1 had no option examined, and #2 had its own answer previewed, which is no alternative."""
    first, second = _adjust(1, 1, age=YES, gender=YES), _adjust(2, 3, bmi=NO)
    previews = [_adjust_preview(2, bmi=NO)]
    assert A.examined_of(first, [first, second], previews, []) == []
    got = A.examined_of(second, [first, second], previews, [])
    assert [(e["question"], e["how"], e["alternative"]) for e in got] == [
        ("adjust:bmi", "preview", False)]


def test_a_preview_that_answers_this_covariate_otherwise_is_an_alternative():
    """DB-V4, the other way: `age` previewed with other answers while #1 stood is an alternative
    examined for #1's `age`; the same answers previewed are not."""
    first = _adjust(1, 1, age=YES, gender=YES)
    previews = [_adjust_preview(2, age=NO), _adjust_preview(3, age=YES)]
    got = A.examined_of(first, [first], previews, [])
    assert [(e["question"], e["value"]["causes_exposure"], e["alternative"]) for e in got] == [
        ("adjust:age", "no", True), ("adjust:age", "yes", False)]


# ── materiality ──────────────────────────────────────────────────────────────

MOVED = [{"family": "linear", "against": "fit1", "movement": {"value": 0.77}, "why": None}]


def _fits():
    return [{"id": "fit1", "seq": 1, "decisions": [1, 2], "changed": [1, 2], "movement": [],
             "movement_why": "No earlier fit is recorded to compare this one with.",
             "withheld": None},
            {"id": "fit2", "seq": 2, "decisions": [1, 2, 3], "changed": [3], "movement": MOVED,
             "movement_why": None, "withheld": None},
            {"id": "fit3", "seq": 3, "decisions": [1, 2, 4, 5], "changed": [4, 5],
             "movement": MOVED, "movement_why": None, "withheld": None}]


def test_a_decision_has_the_fits_movement_only_where_it_alone_changed():
    fits = _fits()
    assert A.fits_materiality({"seq": 3}, fits) == {"source": "fits", "attempt": "fit2",
                                                    "movement": MOVED}
    together = A.fits_materiality({"seq": 4}, fits)
    assert together["source"] is None and together["attempt"] == "fit3"
    assert together["why"] == ("It took effect at fit #3 together with record #5: the fit "
                               "records measure the change between two fits, not each record's "
                               "share of it.")
    assert A.fits_materiality({"seq": 1}, fits)["why"] == fits[0]["movement_why"]
    assert A.fits_materiality({"seq": 6}, fits) == {"source": None, "why": A.NOT_IN_A_FIT}
    quiet = copy.deepcopy(fits)
    quiet[1].update(movement=[], withheld="Fit was not pressed.")
    assert A.fits_materiality({"seq": 3}, quiet)["why"] == "Fit was not pressed."
    swapped = copy.deepcopy(fits)
    swapped[1]["movement"] = [{"family": "elastic_net", "against": "fit1", "movement": None,
                               "why": "`elastic_net` was not fitted in the previous fit."}]
    assert A.fits_materiality({"seq": 3}, swapped) == {
        "source": None, "attempt": "fit2",
        "why": "`elastic_net`: `elastic_net` was not fitted in the previous fit."}


# ── a fit's change from the fit before, both ways (DB-V2) ────────────────────


def _fit(seq: int, minute: int, decisions: list[int], **kw: Any) -> dict[str, Any]:
    return {"id": f"fit{seq}", "seq": seq, "at": f"2026-10-18T10:{minute:02d}:00Z",
            "decisions": decisions, **kw}


def _excluded(column: str) -> dict[str, Any]:
    return {"kind": "confirm_role", "column": column, "role": "excluded"}


def _revert(seq: int, minute: int, target: int, note: str = "why") -> dict[str, Any]:
    return _rec(seq, minute, {"kind": "revert", "decision_id": f"r{target}"}, note)


def _rule(low: int) -> dict[str, Any]:
    return {"kind": "set_exclusions", "rules": [{"column": "kcal", "low": low, "high": 5000,
                                                 "reason": "implausible intake"}]}


def _moved(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The attempts with each fit's served movement (constructed): its own value, ``0.1 × seq``."""
    for a in attempts:
        a.update(movement=[{"family": "linear", "against": None, "why": None,
                            "movement": {"value": round(0.1 * a["seq"], 1)}}],
                 movement_why=None, withheld=None)
    return attempts


TARGET = _rec(1, 1, {"kind": "set_target", "column": "y"})
MODELS = _rec(2, 2, {"kind": "select_models", "models": ["linear"]})


def test_a_fit_after_a_revert_alone_is_the_reverts_change_with_its_reason_and_measure():
    """Fits {1,2}, {1,2,3}, {1,2,3,4} and {1,2,4}: the last follows only revert #5 of #3, so its
    change is the revert's (record #3 withdrawn by it), its reason the revert's note, and the
    revert's materiality that fit's served movement."""
    note = "Back to `a`: excluding it lowered the AUC."
    records = [TARGET, MODELS, _rec(3, 4, _excluded("a")), _rec(4, 6, _excluded("b")),
               _revert(5, 8, 3, note)]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [1, 2, 3]), _fit(3, 7, [1, 2, 3, 4]),
            _fit(4, 9, [1, 2, 4])]
    got = _moved(A.attempts_of(fits, records))
    assert [(a["changed"], a["withdrawn"], a["reinstated"]) for a in got] == [
        ([1, 2], [], []), ([3], [], []), ([4], [], []), ([5], [{"seq": 3, "by": 5}], [])]
    assert got[3]["reasons"] == [{"seq": 5, "note": note}]
    assert A.fits_materiality(records[4], got, records) == {
        "source": "fits", "attempt": "fit4", "movement": got[3]["movement"]}
    assert A.fits_materiality(records[2], got, records) == {  # #3 alone changed before fit 2
        "source": "fits", "attempt": "fit2", "movement": got[1]["movement"]}


def test_a_penalized_familys_served_rows_are_assembled_apart_from_the_fits_effects():
    """P4: GET …/fits serves ridge's shrunk row of `x` under the fit's ``effects`` beside the
    linear model's (a family with no coefficient table, ``stages.effects.has_coefficient_table``);
    the attempt keeps ridge's under ``penalized`` and the linear model's, and a family with no
    rows, under ``effects``. A fit with no penalized family keeps an empty ``penalized``."""
    linear = [{"feature": "x", "estimate": 0.30, "ci_low": 0.10, "ci_high": 0.50}]
    ridge = [{"feature": "x", "estimate": 0.12, "ci_low": None, "ci_high": None}]
    fits = [_fit(1, 3, [1, 2], effects={"linear": linear, "ridge": ridge, "random_forest": None}),
            _fit(2, 5, [1, 2], effects={"linear": linear})]
    got = A.attempts_of(fits, [TARGET, MODELS])
    assert got[0]["effects"] == {"linear": linear, "random_forest": None}
    assert got[0]["penalized"] == {"ridge": ridge}
    assert (got[1]["effects"], got[1]["penalized"]) == ({"linear": linear}, {})


def test_a_revert_and_a_new_record_before_one_fit_share_its_change():
    """Revert #4 of #3 and record #5 both came before fit 3: neither has its movement alone."""
    records = [TARGET, MODELS, _rec(3, 4, _excluded("a")), _revert(4, 6, 3),
               _rec(5, 7, _excluded("b"))]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [1, 2, 3]), _fit(3, 8, [1, 2, 5])]
    got = _moved(A.attempts_of(fits, records))
    assert (got[2]["changed"], got[2]["withdrawn"], got[2]["reinstated"]) == (
        [4, 5], [{"seq": 3, "by": 4}], [])
    assert [r["seq"] for r in got[2]["reasons"]] == [4, 5]
    tail = ": the fit records measure the change between two fits, not each record's share of it."
    assert A.fits_materiality(records[4], got, records) == {
        "source": None, "attempt": "fit3",
        "why": "It took effect at fit #3 together with record #4" + tail}
    assert A.fits_materiality(records[3], got, records) == {
        "source": None, "attempt": "fit3",
        "why": "It took effect at fit #3 together with record #5" + tail}


def test_a_record_restored_by_reverting_what_replaced_it_is_the_reverts_change():
    """Rule A (#2), replaced by rule B (#3), then #3 reverted (#5): fit 4 has rule A again, the
    revert's change alone, so the revert has its movement; rule A first took effect at fit 1."""
    records = [TARGET, _rec(2, 2, _rule(500)), _rec(3, 4, _rule(600)),
               _rec(4, 6, _excluded("a")), _revert(5, 8, 3)]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [1, 3]), _fit(3, 7, [1, 3, 4]), _fit(4, 9, [1, 2, 4])]
    got = _moved(A.attempts_of(fits, records))
    assert (got[1]["changed"], got[1]["withdrawn"]) == ([3], [{"seq": 2, "by": 3}])  # replaced
    assert (got[3]["changed"], got[3]["withdrawn"], got[3]["reinstated"]) == (
        [5], [{"seq": 3, "by": 5}], [{"seq": 2, "by": 5}])
    assert A.fits_materiality(records[4], got, records)["attempt"] == "fit4"
    assert A.fits_materiality(records[4], got, records)["source"] == "fits"
    assert A.fits_materiality(records[1], got, records)["why"] == A.FIRST_FIT


def test_a_record_restored_by_reverting_its_revert_is_that_reverts_change():
    """#2 reverted by #3 (fit 2), then #3 reverted by #5: #2 is back at fit 4 by #5 alone."""
    records = [TARGET, _rec(2, 2, _excluded("a")), _revert(3, 4, 2),
               _rec(4, 6, _excluded("b")), _revert(5, 8, 3)]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [1]), _fit(3, 7, [1, 4]), _fit(4, 9, [1, 2, 4])]
    got = _moved(A.attempts_of(fits, records))
    assert (got[1]["changed"], got[1]["withdrawn"]) == ([3], [{"seq": 2, "by": 3}])
    assert (got[3]["changed"], got[3]["reinstated"]) == ([5], [{"seq": 2, "by": 5}])
    assert A.fits_materiality(records[4], got, records)["source"] == "fits"
    assert A.fits_materiality(records[2], got, records)["attempt"] == "fit2"


def test_a_record_first_in_force_by_a_revert_says_so_and_has_no_measure_of_its_own():
    """Rule A (#2) was replaced by rule B (#3) before any fit; reverting #3 (#5) brings rule A in
    for the first time at fit 3, a change the revert is credited with."""
    records = [TARGET, _rec(2, 2, _rule(500)), _rec(3, 3, _rule(600)),
               _rec(4, 5, _excluded("a")), _revert(5, 7, 3)]
    fits = [_fit(1, 4, [1, 3]), _fit(2, 6, [1, 3, 4]), _fit(3, 8, [1, 2, 4])]
    got = _moved(A.attempts_of(fits, records))
    assert A.fits_materiality(records[1], got, records) == {
        "source": None, "attempt": "fit3",
        "why": "It took effect at fit #3, restored by record #5, which is credited with that "
               "change."}


ANSWERS_UNREAD = "This fit cannot be matched to your recorded decisions."


def test_a_record_is_in_no_fit_only_where_every_fits_records_are_served():
    """DRV-INT (DRV-B's verifier, P4): the only fit's records are not served (``decisions``
    null), so whether record #1 was in force at it is not known: its materiality says so with the
    engine's why, never "no fit completed". A fit after one whose records are not served has no
    change known, and a record first held there has the earlier fit's why. With every fit's
    records served, a record in none of them was in no fit (the test below)."""
    records = [_rec(1, 1, {"kind": "set_target", "column": "y"}), _rec(2, 5, _excluded("a"))]
    unread = _fit(1, 4, None, movement=[], movement_why=ANSWERS_UNREAD, withheld="Not served.")
    got = A.attempts_of([unread], records)
    assert got[0]["changed"] is None
    for rec in records:
        assert A.fits_materiality(rec, got, records) == {
            "source": None, "why": f"Fit #1's records are not served, so whether this record was "
                                   f"in force at it is not known: Not served."}
    got = A.attempts_of([unread, _fit(2, 6, [1, 2], movement=[], movement_why="x")], records)
    assert got[1]["changed"] is None
    assert A.fits_materiality(records[1], got, records) == {
        "source": None, "attempt": "fit2",
        "why": "The fit before this record's first fit, fit #1, has records the server does not "
               "serve, so the change between them is not known: Not served."}


def test_a_revert_no_fit_measured_says_why_and_a_record_never_fitted_says_so():
    records = [TARGET, _rec(2, 2, _excluded("a")), _revert(3, 3, 2)]
    fits = [_fit(1, 4, [1])]  # #2 was withdrawn before any fit
    got = _moved(A.attempts_of(fits, records))
    assert A.fits_materiality(records[2], got, records) == {
        "source": None, "why": "No fit's change from the fit before it is credited to this revert "
                               "of record #2."}
    assert A.fits_materiality(records[1], got, records) == {"source": None, "why": A.NOT_IN_A_FIT}


# DB-V6: a revert the fit before read, with its own revert, is not the change after it.

CHAIN = [TARGET, MODELS, _rec(3, 4, _excluded("a")), _rec(4, 5, _excluded("b")),
         _revert(5, 6, 3, "Back to `a`."), _revert(6, 7, 5, "No, without `a` after all."),
         _revert(7, 9, 6, "Back to `a` again: the AUC fell without it.")]


def test_a_revert_the_fit_before_read_with_its_revert_is_not_credited_with_a_later_shift():
    """The verifier's run: `a` (#3) and `b` (#4) excluded after fit 1; #3 reverted (#5) and that
    revert reverted (#6), all appended before fit 2 completed, which read them (its records hold
    #3: the longest prefix keying its fit). Then #6 reverted (#7), the only record appended before
    fit 3, which withdraws #3 again: fit 3's change is #7's, with #7's note and fit 3's movement;
    #5 is credited with no fit's change."""
    fits = [_fit(1, 3, [1, 2]), _fit(2, 8, [1, 2, 3, 4]), _fit(3, 10, [1, 2, 4])]
    got = _moved(A.attempts_of(fits, CHAIN))
    assert (got[1]["changed"], got[1]["withdrawn"], got[1]["reinstated"]) == ([3, 4], [], [])
    assert (got[2]["changed"], got[2]["withdrawn"], got[2]["reinstated"]) == (
        [7], [{"seq": 3, "by": 7}], [])
    assert got[2]["reasons"] == [{"seq": 7, "note": CHAIN[6]["note"]}]
    assert A.fits_materiality(CHAIN[6], got, CHAIN) == {
        "source": "fits", "attempt": "fit3", "movement": got[2]["movement"]}
    assert A.fits_materiality(CHAIN[4], got, CHAIN) == {
        "source": None, "why": "No fit's change from the fit before it is credited to this revert "
                               "of record #3."}


def test_reverts_the_fit_before_did_not_read_share_the_change_they_made_together():
    """The other way: fit 2 completed before #5, #6 and #7 were appended, so fit 3 read all three,
    which together withdrew #3: no one of them is credited, and none has fit 3's movement."""
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [1, 2, 3, 4]), _fit(3, 10, [1, 2, 4])]
    got = _moved(A.attempts_of(fits, CHAIN))
    assert (got[2]["changed"], got[2]["withdrawn"]) == ([], [{"seq": 3, "by": None}])
    for revert in CHAIN[4:]:
        assert A.fits_materiality(revert, got, CHAIN)["source"] is None


def test_a_revert_appended_while_the_fit_before_ran_and_unread_by_it_is_credited():
    """#4 reverts #3 before fit 2 completed, but fit 2's records hold #3, so it had not read #4
    (reading it withdraws #3): fit 3, which holds #3 no longer, read it, and the change is #4's."""
    records = [TARGET, MODELS, _rec(3, 4, _excluded("a")), _revert(4, 5, 3)]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 6, [1, 2, 3]), _fit(3, 8, [1, 2])]
    got = _moved(A.attempts_of(fits, records))
    assert (got[2]["changed"], got[2]["withdrawn"]) == ([4], [{"seq": 3, "by": 4}])
    assert A.fits_materiality(records[3], got, records)["source"] == "fits"


def test_beside_a_new_answer_a_revert_is_credited_only_where_its_whole_chain_was_read():
    """Rule A (#3) withdrawn by #4 and replaced by rule B (#5), both before fit 3: the revert
    withdrew it (the log's own rule), so #4 is credited and #5 is new. The other way: #4 appended
    before fit 2 completed and unread by it, its revert #5 and rule B (#6) after: fit 3 read both
    reverts (rule B, after them, is in force), which net to nothing, so rule B replaced rule A."""
    records = [TARGET, MODELS, _rec(3, 4, _rule(500)), _revert(4, 6, 3), _rec(5, 7, _rule(600))]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [1, 2, 3]), _fit(3, 8, [1, 2, 5])]
    got = A.attempts_of(fits, records)
    assert (got[2]["changed"], got[2]["withdrawn"]) == ([4, 5], [{"seq": 3, "by": 4}])
    records = [TARGET, MODELS, _rec(3, 4, _rule(500)), _revert(4, 5, 3), _revert(5, 7, 4),
               _rec(6, 8, _rule(600))]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 6, [1, 2, 3]), _fit(3, 9, [1, 2, 6])]
    got = A.attempts_of(fits, records)
    assert (got[2]["changed"], got[2]["withdrawn"]) == ([6], [{"seq": 3, "by": 6}])


def test_a_record_that_left_force_by_no_record_found_withholds_every_measure_of_that_fit():
    """Record #2, the task answered for the outcome `y`, left force before fit 2 with no revert of
    it and no later answer to its question: it stands only while `y` is the outcome, and #3 named
    another (``provenance.in_force``: fit 2's records are #3 alone). The change is not wholly #3's,
    so #3 has no measure (a number is right or it is refused). DRV-INT round 1: the log was
    `set_lens` (#2) left out of fit 2's records, which no prefix of the log holds by the server's
    rule, so that change is now not known (the test above); a task answer is one the server's own
    rule leaves out."""
    records = [_rec(1, 1, {"kind": "set_target", "column": "y"}),
               _rec(2, 2, {"kind": "set_task", "column": "y", "task": "regression"}),
               _rec(3, 4, {"kind": "set_target", "column": "z"})]
    fits = [_fit(1, 3, [1, 2]), _fit(2, 5, [3])]
    got = _moved(A.attempts_of(fits, records))
    assert (got[1]["changed"], got[1]["withdrawn"]) == (
        [3], [{"seq": 1, "by": 3}, {"seq": 2, "by": None}])
    assert A.fits_materiality(records[2], got, records) == {
        "source": None, "attempt": "fit2",
        "why": "It took effect at fit #2 while record #2 left force by no record found between "
               "the two fits: the fit records measure the change between two fits, not each "
               "record's share of it."}


# DRV-INT (DRV-B's verifier, P1): a credit to one record is confirmed by the server's own rule.

def _age(low: int) -> dict[str, Any]:
    return {"kind": "set_exclusions", "rules": [{"column": "age", "low": low, "high": 90,
                                                 "reason": "r"}]}


def test_two_reverts_between_fits_that_restore_and_withdraw_one_answer_share_the_change():
    """The verifier's probe: rule age >= 18 (#1), replaced by age >= 30 (#2), fit 1 (#2 in force);
    #2 reverted (#3, which brings #1 back) and #1 reverted (#4), both before fit 2 (no rule in
    force). By the log's own rule, #3 alone after fit 1's records leaves rule #1 in force, not fit
    2's records, so #3 is not credited with fit 2's whole change: #3 and #4 moved together, and
    neither has fit 2's movement. The other way: #3 alone before fit 2 is credited, with it."""
    records = [_rec(1, 1, _age(18)), _rec(2, 2, _age(30)), _revert(3, 5, 2), _revert(4, 6, 1)]
    fits = [_fit(1, 3, [2]), _fit(2, 7, [])]
    got = _moved(A.attempts_of(fits, records))
    assert got[1]["changed"] == [3, 4]
    assert [r["seq"] for r in got[1]["reasons"]] == [3, 4]
    tail = ": the fit records measure the change between two fits, not each record's share of it."
    for seq, other in ((3, 4), (4, 3)):
        m = A.fits_materiality(records[seq - 1], got, records)
        assert m == {"source": None, "attempt": "fit2",
                     "why": f"It took effect at fit #2 together with record #{other}" + tail}, seq
    alone = records[:3]
    got = _moved(A.attempts_of([_fit(1, 3, [2]), _fit(2, 7, [1])], alone))
    assert (got[1]["changed"], got[1]["withdrawn"]) == ([3], [{"seq": 2, "by": 3}])
    assert A.fits_materiality(alone[2], got, alone) == {
        "source": "fits", "attempt": "fit2", "movement": got[1]["movement"]}


def test_a_credit_the_log_cannot_account_for_is_withheld_and_says_so():
    """Fit 2's records ({1, 3}) are those of no prefix of the log by the server's rule (#2 holds a
    question of its own, and nothing withdrew it), though #3 is the one record new since fit 1:
    the change is not known, so no record is credited with it and none has its movement; the why
    says the log does not account for it. A record that changed nothing the log cannot account
    for keeps its credit (the first test above)."""
    records = [_rec(1, 1, _excluded("a")), _rec(2, 2, _excluded("b")), _rec(3, 4, _excluded("c"))]
    got = _moved(A.attempts_of([_fit(1, 3, [1]), _fit(2, 5, [1, 3])], records))
    assert got[1]["changed"] is None
    m = A.fits_materiality(records[2], got, records)
    assert (m["source"], m["attempt"], m["why"]) == (None, "fit2", A.UNACCOUNTED)


def _server_fits(recs: list[Any], fits: list[tuple[int, int]], slots: tuple[str, ...]
                 ) -> list[dict[str, Any]]:
    """Fit records written as ``ProjectService._write_fit_record`` writes them: fit ``k`` keyed
    by the first ``n`` records' answers in ``slots`` and completing at minute ``done``; its
    records in force those of the longest prefix appended by then that keys it
    (``runs.keyed_prefix`` over ``runs.appended_by``). ``_read`` is how many records it read."""
    from datetime import timedelta

    from turbotab.core import decisions as D, provenance, runs

    def key_of(state: Any) -> str:
        dumped = state.model_dump(mode="json")
        return json.dumps({k: dumped.get(k) for k in slots}, sort_keys=True)

    out = []
    for k, (n, done) in enumerate(fits, start=1):
        at = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=done)
        prefix = runs.keyed_prefix(runs.appended_by(recs, at), key_of(D.fold(recs[:n])), key_of)
        live = provenance.in_force(prefix)
        out.append({"id": f"fit{k}", "seq": k, "at": at.isoformat(),
                    "decisions": sorted(r.seq for r in prefix if r.id in live),
                    "_read": len(prefix)})
    return out


def _log(records: list[dict[str, Any]]) -> list[Any]:
    from turbotab.core.decisions import DecisionRecord

    return [DecisionRecord.model_validate(r) for r in records]


def test_a_fit_whose_records_two_prefixes_with_different_answers_hold_has_no_change_known():
    """DRV-INT round 1 (P1): ``provenance.in_force`` counts a draw (#1) as superseded by the scheme
    written onto it (#2), while ``decisions.fold`` keeps the draw. Fit 1 is keyed on #1–#2; the
    exclusion rule (#3) keys fit 2; the draw's revert (#4) is appended while fit 2 runs, so the
    server's prefix for fit 2 is #1–#3 (with #4 the draw is gone, which keys another fit). The
    prefixes #1–#3 and #1–#4 both hold #2 and #3 in force but fold to different splits, so what
    ``GET …/fits`` serves cannot tell which fit 2 read: its change is not known, #4 (which fit 2
    never read) is named in none, and #3 has no measure and says why. The other way: #4 appended
    after fit 2 completed, one prefix holds fit 2's records, and #3 alone is credited with fit 2's
    change and its movement."""
    records = [_rec(1, 1, {"kind": "set_split", "holdout": 0.2}),
               _rec(2, 2, {"kind": "set_validation", "folds": 10}), _rec(3, 4, _age(30)),
               _revert(4, 5, 1)]
    fits = _server_fits(_log(records), [(2, 3), (3, 6)], ("split", "exclusions"))
    assert [(f["decisions"], f["_read"]) for f in fits] == [([2], 2), ([2, 3], 3)]
    served = [{k: v for k, v in f.items() if k != "_read"} for f in fits]
    got = _moved(A.attempts_of(served, records))
    assert (got[1]["changed"], got[1]["reasons"]) == (None, [])
    assert A.fits_materiality(records[2], got, records) == {
        "source": None, "attempt": "fit2", "why": A.UNACCOUNTED}
    assert A.Prefixes(records).read_through(served[1]) is None
    late = [*records[:3], _revert(4, 7, 1)]
    fits = _server_fits(_log(late), [(2, 3), (3, 6)], ("split", "exclusions"))
    served = [{k: v for k, v in f.items() if k != "_read"} for f in fits]
    got = _moved(A.attempts_of(served, late))
    assert (got[1]["changed"], got[1]["withdrawn"], got[1]["reinstated"]) == ([3], [], [])
    assert A.fits_materiality(late[2], got, late) == {
        "source": "fits", "attempt": "fit2", "movement": got[1]["movement"]}


def _role(rnd: Any) -> dict[str, Any]:
    return {"kind": "confirm_role", "column": rnd.choice("ab"),
            "role": rnd.choice(["covariate", "excluded"])}


def _kinds(rnd: Any) -> dict[str, Any]:
    """An exclusion rule, a missing-values answer, a held-out draw, a validation scheme written
    onto the draw, or the sensitivity answer, which no fit reads (the verifier's DRV-INT round 1
    simulation, drvint/probes/sim_kinds.py)."""
    k = rnd.random()
    if k < 0.3:
        return _age(rnd.choice([18, 30, 40]))
    if k < 0.45:
        return {"kind": "set_missing", "strategy": rnd.choice(["complete_case", "impute"])}
    if k < 0.6:
        return {"kind": "set_split", "holdout": rnd.choice([0.2, 0.3]),
                "folds": rnd.choice([5, 10])}
    if k < 0.75:
        return {"kind": "set_validation", "folds": rnd.choice([5, 10])}
    return {"kind": "set_sensitivity", "analyses": []}


# Each simulation: the answers it draws, the slots its fits are keyed by, its seeds, and the chances
# that a record is a revert before a fit starts and while it runs.
SIMULATIONS = {
    "roles": (_role, ("role_confirmations",), 150, (0.45, 0.6)),
    "kinds": (_kinds, ("exclusions", "missing", "split"), 300, (0.4, 0.5)),
}


def _simulated(seed: int, simulation: str = "roles", events: int = 14
               ) -> tuple[list[Any], list[dict[str, Any]]]:
    """A log of answers and reverts drawn at random (``SIMULATIONS``), and its fits as the server
    records them (``ProjectService._write_fit_record``): each fit keyed by the answers in force
    when it started, completing after up to two more records, recorded once per key, its records
    in force those of the longest prefix appended by its completion that keys it
    (``runs.keyed_prefix``). The verifier's simulations (DRV-B, dbv3/probes/sim.py; DRV-INT round
    1, drvint/probes/sim_kinds.py)."""
    import random
    from datetime import timedelta

    from turbotab.core import decisions as D, provenance, runs

    body, slots, _, (p_before, p_during) = SIMULATIONS[simulation]
    rnd, recs, t = random.Random(seed), [], 0.0
    t0 = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)

    def key_of(state: Any) -> str:
        dumped = state.model_dump(mode="json")
        return json.dumps({k: dumped.get(k) for k in slots}, sort_keys=True)

    def append(p_revert: float) -> None:
        nonlocal t
        t += 1.0
        decision = ({"kind": "revert", "decision_id": rnd.choice(recs).id}
                    if recs and rnd.random() < p_revert else body(rnd))
        r = D.DecisionRecord(id=f"r{len(recs) + 1}", seq=len(recs) + 1,
                             at=t0 + timedelta(seconds=t), note=f"n{len(recs) + 1}",
                             decision=D.parse_decision(decision))
        try:
            D.fold(recs + [r])
        except D.Refusal:
            return
        recs.append(r)

    fits, keys = [], set()
    for _ in range(events):
        for _ in range(rnd.randint(0, 2)):
            append(p_before)
        if not recs:
            continue
        key = key_of(D.fold(recs))
        for _ in range(rnd.randint(0, 2)):
            append(p_during)
        t += 0.5
        if key in keys:
            continue
        keys.add(key)
        done = t0 + timedelta(seconds=t)
        prefix = runs.keyed_prefix(runs.appended_by(recs, done), key, key_of)
        live = provenance.in_force(prefix)
        fits.append({"id": f"fit{len(fits) + 1}", "seq": len(fits) + 1, "at": done.isoformat(),
                     "decisions": sorted(r.seq for r in prefix if r.id in live),
                     "_read": len(prefix), "_key": key})
    return recs, fits


@pytest.mark.parametrize("simulation", sorted(SIMULATIONS))
def test_every_credit_to_one_record_is_that_records_change_by_the_servers_rule(simulation):
    """Over simulated runs (roles confirmed and reverted; exclusions, missing values, the draw,
    the scheme written onto it and an answer no fit reads, with reverts): every record credited
    with a fit's change, or with a shift of it, is among the records this fit read that the fit
    before did not; and wherever a fit's whole change is credited to one record, that record
    alone, after the records the fit before read, keys this fit by the server's own rule
    (``decisions.fold``). The truth is the server's prefix for each fit (``runs.keyed_prefix``);
    the assembler reads only what ``GET …/fits`` serves. Some runs do credit one record, so the
    rule is exercised."""
    from turbotab.core import decisions as D

    _, slots, seeds, _ = SIMULATIONS[simulation]
    credited = 0
    for seed in range(seeds):
        recs, fits = _simulated(seed, simulation)
        records = [r.model_dump(mode="json") for r in recs]
        got = A.attempts_of([{k: v for k, v in f.items() if not k.startswith("_")} for f in fits],
                            records)
        for i in range(1, len(fits)):
            before, fit, change = fits[i - 1], fits[i], got[i]
            if fit["_read"] > before["_read"]:
                read = set(range(before["_read"] + 1, fit["_read"] + 1))
                named = set(change["changed"] or ()) | {
                    s["by"] for s in change["withdrawn"] + change["reinstated"] if s["by"]}
                assert named <= read, (seed, i, change, sorted(read))
            loose = [s for s in change["withdrawn"] + change["reinstated"] if s["by"] is None]
            if not change["changed"] or len(change["changed"]) != 1 or loose:
                continue
            (x,) = change["changed"]
            credited += 1
            assert before["_read"] < x <= fit["_read"], (seed, i, change)
            alone = D.fold(recs[:before["_read"]] + [recs[x - 1]]).model_dump(mode="json")
            assert json.dumps({k: alone.get(k) for k in slots}, sort_keys=True) == fit["_key"], (
                seed, i, change)
    assert credited > 20


def _curve_doc() -> dict[str, Any]:
    from turbotab.core import mattered as M

    def choice(key, label, decision):
        return M.MatteredChoice(key=key, label=label, primary="recorded", decision=decision,
                                options=[M.MatteredOption(key="recorded", label="As recorded"),
                                         M.MatteredOption(key="other", label="Changed")])

    recorded = {"adjusted:age": "recorded", "exclusion:0": "recorded"}
    curve = M.MatteredCurve(
        estimate_label="β", choices=[choice("adjusted:age", "`age` adjusted", "set_adjustment"),
                                     choice("exclusion:0", "The rule", "set_exclusions")],
        specs=[M.MatteredSpec(key="s0", picks=recorded, primary=True, estimate=0.3, low=0.1,
                              high=0.5, n=100),
               M.MatteredSpec(key="s1", picks={**recorded, "adjusted:age": "other"},
                              primary=False, estimate=0.4, low=0.2, high=0.6, n=100)],
        not_estimated=[M.MatteredNotEstimated(key="s2", picks={**recorded, "exclusion:0": "other"},
                                              primary=False, reason="A reading waits.")],
        total=3, drawn=3, sampled=False, rule=None, crossing="one_at_a_time", summary=None,
        level=0.95, sealed=None, plan_lock="L", weighted=False, clustered=False, exits=[])
    return M.Mattered(status="fresh", fresh=True, key="k", plan_lock="L", curve=curve, says=None,
                      methods=None, reported=None, examined=[], error=None).model_dump(mode="json")


def test_under_estimate_a_choice_the_curve_varies_has_its_departures_beside_the_primary():
    doc = _curve_doc()
    served = A.Read(200, doc)
    earlier = _adjust(4, 4, age=NO)
    adjust = _adjust(5, 5, age=YES)  # `age` answered again
    rules = _rec(6, 6, {"kind": "set_exclusions", "rules": []})
    in_force = A.in_force_of([earlier, adjust, rules])
    got = A.mattered_materiality(adjust, served, in_force)
    assert got == {"source": "mattered", "primary": doc["curve"]["specs"][0], "departures": [
        {"choice": "adjusted:age", "label": "`age` adjusted", "option": "other",
         "spec": doc["curve"]["specs"][1], "not_estimated": None}]}
    # The rule's one departure was not estimated: no number, so its why is the curve's reason.
    assert A.mattered_materiality(rules, served, in_force) == {
        "source": None, "why": "The curve estimated no specification that changes this "
                               "decision's choices: `The rule` Changed: A reading waits."}
    assert A.mattered_materiality(earlier, served, in_force)["why"] == (
        "Superseded by record #5, which answered again each question this record answered: the "
        "curve served is drawn for the plan in force now, so it varies that record's choices, not "
        "this one's.")
    assert A.mattered_materiality(_rec(7, 7, {"kind": "set_purpose"}), served, in_force) is None
    refused = A.Read(409, {"error": {"code": "curve_sealed", "message": "Opens after the lock.",
                                     "exits": []}})
    assert A.mattered_materiality(adjust, refused, in_force) == {"source": None,
                                                                 "why": "Opens after the lock."}


def _groups_curve() -> dict[str, Any]:
    """A curve over `age` and `gender` adjusted for and `bmi` left out, each departure estimated
    (constructed numbers), in ``core/mattered.py``'s own models."""
    from turbotab.core import mattered as M

    keys = {"adjusted:age": "`age` adjusted", "adjusted:gender": "`gender` adjusted",
            "left_out:bmi": "`bmi` left out"}
    recorded = dict.fromkeys(keys, "recorded")
    choices = [M.MatteredChoice(key=k, label=label, primary="recorded", decision="set_adjustment",
                                options=[M.MatteredOption(key="recorded", label="As recorded"),
                                         M.MatteredOption(key="other", label="Changed")])
               for k, label in keys.items()]
    specs = [M.MatteredSpec(key="s0", picks=recorded, primary=True, estimate=0.3, low=0.1,
                            high=0.5, n=100)]
    specs += [M.MatteredSpec(key=f"s{i}", picks={**recorded, k: "other"}, primary=False,
                             estimate=0.3 + i / 10, low=0.1, high=0.7, n=100)
              for i, k in enumerate(keys, start=1)]
    curve = M.MatteredCurve(
        estimate_label="β", choices=choices, specs=specs, not_estimated=[], total=4, drawn=4,
        sampled=False, rule=None, crossing="one_at_a_time", summary=None, level=0.95,
        sealed=None, plan_lock="L", weighted=False, clustered=False, exits=[])
    return M.Mattered(status="fresh", fresh=True, key="k", plan_lock="L", curve=curve, says=None,
                      methods=None, reported=None, examined=[], error=None).model_dump(mode="json")


def test_under_estimate_each_covariates_choices_are_credited_to_its_own_answer_in_force():
    """DB-V4: two groups answered in two records, both in force (the server keeps each covariate's
    answer under its column): each carries the departures of its own covariates' choices, and
    neither is superseded. Answering `age` again (#3) takes `age`'s choice to #3 and leaves #1 in
    force for `gender`."""
    doc = _groups_curve()
    served = A.Read(200, doc)
    first, second = _adjust(1, 1, age=YES, gender=YES), _adjust(2, 2, bmi=NO)

    def choices(rec, records):
        got = A.mattered_materiality(rec, served, A.in_force_of(records))
        assert got["source"] == "mattered", got
        assert got["primary"] == doc["curve"]["specs"][0]
        return [d["choice"] for d in got["departures"]]

    assert choices(first, [first, second]) == ["adjusted:age", "adjusted:gender"]
    assert choices(second, [first, second]) == ["left_out:bmi"]
    again = _adjust(3, 3, age=NO)
    assert choices(first, [first, second, again]) == ["adjusted:gender"]
    assert choices(again, [first, second, again]) == ["adjusted:age"]
    assert A.choice_question("left_out:bmi", "set_adjustment") == "adjust:bmi"
    assert A.choice_question("exclusion:1", "set_exclusions") == "exclusions"


# ── the result ───────────────────────────────────────────────────────────────

SCORES = {"linear": {"auc": 0.65, "log_loss": 0.54}}


def test_under_predict_the_result_is_the_openings_scores_once():
    records = [_rec(1, 1, {"kind": "select_models", "models": ["linear"]}),
               _rec(2, 3, {"kind": "open_seal", "family": "linear", "target": "y",
                           "metric": "log_loss", "n_holdout": 96, "scores": SCORES}),
               _rec(3, 4, {"kind": "select_models", "models": ["linear", "elastic_net"]})]
    opened = {"seq": 2, "current": True, "note": "the reported result"}
    fits = [{"id": "fit1", "target": "y", "changed_after_seal": False, "at_opening": opened,
             "plan_digest": "d1", "decisions": [1]},
            {"id": "fit2", "target": "y", "changed_after_seal": True, "at_opening": None,
             "plan_digest": "d2", "decisions": [1, 2, 3]}]
    intervals = {"auc": {"ci_low": 0.52, "ci_high": 0.78}}
    doc = {"at_opening": opened, "families": [{"family": "linear", "intervals": intervals}]}
    got = A.result_of("prediction", records, fits, A.Read(200, doc))
    assert (got["attempt"], got["decisions"], got["scores"], got["intervals"]) == (
        "fit1", [1], SCORES, intervals)
    later = {**doc, "at_opening": {**opened, "current": False, "note": "changed after it"}}
    got = A.result_of("prediction", records, fits, A.Read(200, later))
    assert got["intervals"] is None and got["intervals_why"] == "changed after it"
    assert A.result_of("prediction", records[:1], fits, A.Read(200, doc))["refused"]


# The plan a constructed lock records: a numeric outcome's difference in its mean, whose number is
# the locked fit's exposure row itself (``estimand.measure_facts``: a difference, conditional and
# marginal).
PLAN = {"estimand": {"exposure": "x", "measure": "mean_difference"}}


def _reported(rows: dict[str, Any], ratio: bool = False) -> list[dict[str, Any]]:
    """Each family's exposure rows as the result reports them (``record.Estimate``): on a ratio
    scale the rows' ``ratio`` columns, on a difference scale the coefficient itself."""
    cols = ("ratio", "ratio_low", "ratio_high") if ratio else ("estimate", "ci_low", "ci_high")
    return [{"family": f, "feature": row["feature"], "setting": None, "estimate": row.get(cols[0]),
             "ci_low": row.get(cols[1]), "ci_high": row.get(cols[2]), "why": None}
            for f, rs in rows.items() for row in rs or ()]


def test_under_estimate_the_result_is_the_fit_the_lock_in_force_was_recorded_on():
    """A fit the scheduler held for Fit completes after the lock (its records hold it), and GET
    …/fits marks it the one the lock was recorded on (``locked``); a fit after a change made with
    the estimates in view holds another plan."""
    effect = [{"feature": "x", "estimate": 0.3, "ci_low": 0.1, "ci_high": 0.5}]
    records = [_rec(1, 1, {"kind": "set_estimand"}), _rec(2, 2, {"kind": "set_adjustment"}),
               _rec(3, 3, {"kind": "lock_plan", "digest": "lockdigest", "plan": PLAN}),
               _rec(4, 6, {"kind": "set_exclusions", "rules": []})]
    fits = [{"id": "fit1", "plan_digest": "d1", "decisions": [1], "effects": {}, "locked": False},
            {"id": "fit2", "plan_digest": "lockdigest", "decisions": [1, 2, 3],
             "effects": {"linear": effect}, "locked": True},
            {"id": "fit3", "plan_digest": "d3", "decisions": [1, 2, 3, 4],
             "effects": {"linear": effect}, "locked": False}]
    got = A.result_of("inference", records, fits, A.Read(200, {}))
    assert (got["attempt"], got["lock"], got["plan_digest"], got["estimates"]) == (
        "fit2", 3, "lockdigest", _reported({"linear": effect}))
    assert (got["measure"], got["scale"], got["source"]) == ("mean_difference", "difference",
                                                             "fits")
    assert A.result_of("inference", records[:2], fits, A.Read(200, {}))["refused"]


A_ROWS = [{"feature": "sugar", "estimate": -0.0276, "ci_low": -0.0395, "ci_high": -0.0156}]
B_ROWS = [{"feature": "sugar", "estimate": -0.0262, "ci_low": -0.0384, "ci_high": -0.0140}]
TRIED = [_rec(1, 1, {"kind": "set_estimand"}), _rec(2, 2, _rule(500)), _rec(3, 4, _rule(600)),
         _rec(4, 6, _rule(500)), _rec(5, 7, {"kind": "lock_plan", "digest": "plan-A",
                                             "plan": PLAN})]
TRIED_FITS = [{"id": "fit1", "plan_digest": "plan-A", "decisions": [1, 2],
               "effects": {"linear": A_ROWS}, "locked": True},
              {"id": "fit2", "plan_digest": "plan-B", "decisions": [1, 3],
               "effects": {"linear": B_ROWS}, "locked": False}]


def test_under_estimate_the_result_is_the_fit_whose_plan_is_the_locks_not_the_last_before_it():
    """DB-V1's E1: rule A fitted (fit 1), rule B fitted (fit 2), rule A again (#4, no new fit:
    its key was recorded), then the lock on rule A's plan. Its E3: a revert of #4 after the lock
    makes rule B's fit current again. The result is fit 1 both times, the fit GET …/fits marks
    as the one the lock was recorded on."""
    got = A.result_of("inference", TRIED, TRIED_FITS, A.Read(200, {}))
    assert (got["attempt"], got["decisions"], got["estimates"], got["plan_digest"]) == (
        "fit1", [1, 2], _reported({"linear": A_ROWS}), "plan-A")
    reverted = {**_rec(6, 9, {"kind": "revert", "decision_id": "r4"}), "after_estimates": True}
    got = A.result_of("inference", TRIED + [reverted], TRIED_FITS, A.Read(200, {}))
    assert (got["attempt"], got["estimates"]) == ("fit1", _reported({"linear": A_ROWS}))


def test_under_estimate_the_result_is_the_locked_fit_whatever_its_plans_digest():
    """DRV-INT (DB's [next, engine] line): an answer the fit does not read (the model sequence,
    the sensitivity analyses) came between the fit and the press, so the lock records a digest no
    fit record holds; GET …/fits marks the fit the lock was recorded on all the same (its key), and
    the result is that fit's, under the lock's digest. Before, it was refused as not known."""
    other = [*TRIED[:4], _rec(5, 7, {"kind": "lock_plan", "digest": "plan-C", "plan": PLAN})]
    got = A.result_of("inference", other, TRIED_FITS, A.Read(200, {}))
    assert (got["attempt"], got["lock"], got["plan_digest"], got["estimates"]) == (
        "fit1", 5, "plan-C", _reported({"linear": A_ROWS}))
    assert got.get("refused") is None


def test_under_estimate_a_lock_no_fit_record_is_marked_with_is_refused():
    """No fit record is marked as the one the lock was recorded on (that fit never completed, say):
    the result is refused, never the fit whose plan's digest is the lock's, nor the last fit before
    the lock; nor where two are marked."""
    unmarked = [{**f, "locked": False} for f in TRIED_FITS]
    got = A.result_of("inference", TRIED, unmarked, A.Read(200, {}))
    assert (got["attempt"], got["lock"], got["plan_digest"], got.get("estimates")) == (
        None, 5, "plan-A", None)
    assert (got["refused"], got["refused_kind"]) == (A.no_locked_fit(5), "not_known")
    twice = [{**f, "locked": True} for f in TRIED_FITS]
    got = A.result_of("inference", TRIED, twice, A.Read(200, {}))
    assert got["attempt"] is None and got["refused"] == A.locked_twice(5, ["fit1", "fit2"])
    assert got["refused_kind"] == "not_known"


def test_a_result_with_no_numbers_says_whether_none_was_reported_served_or_known():
    """DB-V5: the plan never locked is none reported; the locked fit estimating no row, none
    served; its rows withheld by the route now, or an opening with no scores while its fit serves
    held-out scores, a result the record cannot know."""
    unlocked = A.result_of("inference", TRIED[:4], TRIED_FITS, A.Read(200, {}))
    assert unlocked["refused_kind"] == "not_reported"
    empty = [{**TRIED_FITS[0], "effects": {"linear": None}, "withheld": None}, TRIED_FITS[1]]
    got = A.result_of("inference", TRIED, empty, A.Read(200, {}))
    assert (got["refused"], got["refused_kind"]) == ("The locked fit serves no exposure rows.",
                                                     "not_served")
    gated = [{**empty[0], "withheld": "A question the estimates rest on is open."}, empty[1]]
    got = A.result_of("inference", TRIED, gated, A.Read(200, {}))
    assert (got["refused"], got["refused_kind"]) == ("A question the estimates rest on is open.",
                                                     "not_known")
    records = [_rec(1, 1, {"kind": "select_models", "models": ["linear"]}),
               _rec(2, 3, {"kind": "open_seal", "family": "linear", "target": "y",
                           "metric": "auc", "n_holdout": 96, "scores": None})]
    opened = {"id": "fit1", "target": "y", "changed_after_seal": False, "decisions": [1],
              "at_opening": {"seq": 2}, "plan_digest": "d1", "holdout": None}
    got = A.result_of("prediction", records, [opened], A.Read(200, {}))
    assert (got["refused"], got["refused_kind"]) == ("The opening recorded no held-out scores.",
                                                     "not_served")
    scored = {**opened, "holdout": {"linear": {"auc": 0.7}}}
    assert A.result_of("prediction", records, [scored], A.Read(200, {}))["refused_kind"] == (
        "not_known")
    assert A.result_of("prediction", records[:1], [], A.Read(200, {}))["refused_kind"] == (
        "not_reported")


def test_a_lock_the_server_withdrew_is_not_the_lock_in_force():
    effect = [{"feature": "x", "estimate": 0.3, "ci_low": 0.1, "ci_high": 0.5}]
    records = [_rec(1, 1, {"kind": "set_estimand"}),
               _rec(2, 2, {"kind": "lock_plan", "digest": "first"}),
               _rec(3, 3, {"kind": "revert", "decision_id": "r2"}),
               _rec(4, 4, {"kind": "set_adjustment"}),
               _rec(5, 5, {"kind": "lock_plan", "digest": "second"})]
    fits = [{"id": "fit1", "decisions": [1, 4, 5], "effects": {"linear": effect}}]
    got = A.result_of("inference", records, fits, A.Read(200, {}))
    assert (got["lock"], got["plan_digest"]) == (5, "second")
    withdrawn = A.result_of("inference", records[:4], fits, A.Read(200, {}))
    assert withdrawn["refused"] == "The plan was never locked, so no estimate was served."
    rules = _rec(6, 6, {"kind": "set_exclusions", "rules": []})
    undone = _rec(7, 7, {"kind": "revert", "decision_id": "r6"})
    assert A.mattered_materiality(rules, A.Read(200, _curve_doc()),
                                  A.in_force_of([rules, undone]),
                                  A.cancelled([rules, undone])) == {"source": None,
                                                                    "why": A.WITHDRAWN}


# ── DRV-FIX (a): under Estimate, the number the app reports for the estimand ─────────────────

LOG_ODDS = {"feature": "x", "estimate": 0.5, "ci_low": 0.3, "ci_high": 0.7, "p": 0.0001,
            "ratio": math.exp(0.5), "ratio_low": math.exp(0.3), "ratio_high": math.exp(0.7)}
SETTING = "one unit higher than observed"
CONTRAST = {"setting": SETTING, "risk_low": 0.30, "risk_high": 0.40, "rd": 0.10, "rr": 4 / 3,
            "rd_low": 0.05, "rd_high": 0.15, "rr_low": 1.15, "rr_high": 1.52,
            "rr_unbounded": False, "n_boot": 1000, "n_limit": 0, "n_failed": 0}


def _locked(measure: str, *after: dict[str, Any]) -> list[dict[str, Any]]:
    """A decision log as GET /projects/{pid} serves it: the goal, the outcome and the estimand
    (`x`'s total effect as ``measure``), then the plan's lock recording the plan those hold as
    the server records it (``plan_lock.plan_of`` and ``plan_lock.digest`` over the log's fold),
    then ``after``, each made with the estimates in view."""
    from turbotab.core import decisions as D, plan_lock

    log = [_rec(1, 1, {"kind": "set_purpose", "purpose": "inference"}),
           _rec(2, 2, {"kind": "set_target", "column": "y"}),
           _rec(3, 3, {"kind": "set_estimand", "exposure": "x", "measure": measure})]
    plan = plan_lock.plan_of(D.fold([D.DecisionRecord.model_validate(r) for r in log]))
    log.append({**_rec(4, 4, {"kind": "lock_plan", "plan": plan,
                              "digest": plan_lock.digest(plan)}, note=None),
                "recorded_by": "turbotab"})
    return log + [{**_rec(5 + i, 6 + i, d), "after_estimates": True} for i, d in enumerate(after)]


def _locked_fit() -> list[dict[str, Any]]:
    return [{"id": "fit1", "fit_key": "key1", "decisions": [1, 2, 3, 4],
             "effects": {"linear": [dict(LOG_ODDS)]}, "withheld": None, "locked": True}]


def _results(measure: str, **marginal: Any) -> dict[str, Any]:
    """GET …/results under Estimate, served for the locked fit's key with the effects stage
    current: the fit's exposure row (the log odds) and the effects stage's standardization of
    ``measure`` (one contrast, :data:`CONTRAST`, unless ``marginal`` says otherwise)."""
    served = {"declared": measure, "method": "g-computation", "contrasts": [dict(CONTRAST)],
              "refused": None, "interval_refused": None, "exits": [], "concerns": [], **marginal}
    return {"purpose": "inference", "fit": {"key": "key1", "fresh": True, "status": "fresh"},
            "effects": {"key": "e1", "fresh": True, "status": "fresh"}, "withheld": None,
            "estimand": {"exposure": "x", "measure": measure},
            "families": [{"family": "linear", "label": "Logistic regression",
                          "effect": [dict(LOG_ODDS)], "sequence": [], "marginal": served,
                          "e_values": []}]}


@pytest.mark.parametrize("measure,cols,scale", [("risk_difference", ("rd", "rd_low", "rd_high"),
                                                 "difference"),
                                                ("risk_ratio", ("rr", "rr_low", "rr_high"),
                                                 "ratio")])
def test_under_estimate_a_marginal_measure_is_the_results_standardization_not_the_log_odds(
        measure, cols, scale):
    """The estimand declares a marginal measure; the locked fit's row is the conditional log odds
    per unit of `x` (0.5, 0.3 to 0.7), and the effects stage's standardization, as GET …/results
    serves it for that fit, the risk difference 0.10 (0.05 to 0.15) or ratio 4/3 (1.15 to 1.52),
    each constructed apart: the result is the latter, on its scale, read from the results with
    the fit key they served it for."""
    got = A.result_of("inference", _locked(measure), _locked_fit(),
                      A.Read(200, _results(measure)))
    assert (got.get("refused"), got["attempt"], got["lock"]) == (None, "fit1", 4)
    assert (got["measure"], got["scale"], got["conditioning"], got["source"],
            got["served_fit"]) == (measure, scale, "marginal", "results", "key1")
    assert got["estimates"] == [{"family": "linear", "feature": "x", "setting": SETTING,
                                 "estimate": CONTRAST[cols[0]], "ci_low": CONTRAST[cols[1]],
                                 "ci_high": CONTRAST[cols[2]], "why": None}]
    assert got["estimates"][0]["estimate"] not in (LOG_ODDS["estimate"], LOG_ODDS["ratio"])
    assert validate_result(got)


def validate_result(raw: dict[str, Any]) -> bool:
    from research.swarm.record import Result

    Result.model_validate(raw)
    return True


def test_a_marginal_measure_the_results_no_longer_serve_for_the_locked_plan_is_not_known():
    """The results serve what the plan in force gives now. Both ways: a change to the plan after
    the lock (an exclusion rule), and a fit or effects stage that is not the locked plan's or not
    current, leave the reported risk difference not known; the same rule withdrawn again by its
    revert leaves the plan the lock's (the server's own rule, ``plan_lock.plan_changed``), and the
    number stands."""
    rules = {"kind": "set_exclusions", "rules": [{"column": "x", "low": 0, "high": 9,
                                                  "reason": "range"}]}
    doc = _results("risk_difference")
    changed = A.result_of("inference", _locked("risk_difference", rules), _locked_fit(),
                          A.Read(200, doc))
    assert (changed["refused"], changed["refused_kind"]) == (A.plan_changed_why(4),
                                                             "not_known")
    assert changed.get("estimates") is None and changed["attempt"] == "fit1"
    undone = A.result_of("inference", _locked("risk_difference", rules,
                                              {"kind": "revert", "decision_id": "r5"}),
                         _locked_fit(), A.Read(200, doc))
    assert undone.get("refused") is None and undone["estimates"][0]["estimate"] == 0.10
    for part, served in (("fit", {"key": "key2", "fresh": True, "status": "fresh"}),
                         ("fit", {"key": "key1", "fresh": False, "status": "stale"}),
                         ("effects", {"key": "e1", "fresh": False, "status": "running"})):
        got = A.result_of("inference", _locked("risk_difference"), _locked_fit(),
                          A.Read(200, {**doc, part: served}))
        assert (got["refused"], got["refused_kind"]) == (A.not_current_why(4), "not_known"), part
    gated = A.result_of("inference", _locked("risk_difference"), _locked_fit(),
                        A.Read(200, {**doc, "withheld": "A question the estimates rest on is open."}))
    assert (gated["refused"], gated["refused_kind"]) == (
        "A question the estimates rest on is open.", "not_known")


def test_a_marginal_measure_the_effects_stage_refused_is_refused_with_its_words():
    """The effects stage refuses the marginal risks (under multiple imputation, say), serves none
    for a family it does not standardize, or serves the risks with their interval refused: the
    result is refused with the stage's words where no family has a number, each family's why
    beside it otherwise, and a number with no interval says why."""
    words = ("The marginal risks are not pooled over multiple imputations here: the bootstrap "
             "would have to repeat the imputation in every resample.")
    refused = _results("risk_difference", contrasts=[], refused=words)
    got = A.result_of("inference", _locked("risk_difference"), _locked_fit(), A.Read(200, refused))
    assert (got["refused"], got["refused_kind"], got.get("estimates")) == (words, "not_served",
                                                                          None)
    assert validate_result(got)
    two = _results("risk_difference")
    two["families"].append({"family": "elastic_net", "label": "Elastic net", "effect": None,
                            "sequence": [], "marginal": None, "e_values": []})
    got = A.result_of("inference", _locked("risk_difference"), _locked_fit(), A.Read(200, two))
    assert [(e["family"], e["estimate"], e["why"]) for e in got["estimates"]] == [
        ("linear", 0.10, None),
        ("elastic_net", None, "The results serve no marginal risk difference for `elastic_net`.")]
    floor = "Too few units for an interval."
    bare = _results("risk_difference", interval_refused=floor,
                    contrasts=[{**CONTRAST, "rd_low": None, "rd_high": None}])
    got = A.result_of("inference", _locked("risk_difference"), _locked_fit(), A.Read(200, bare))
    assert got["estimates"] == [{"family": "linear", "feature": "x", "setting": SETTING,
                                 "estimate": 0.10, "ci_low": None, "ci_high": None, "why": floor}]


@pytest.mark.parametrize("measure,ratio,scale,conditioning", [
    ("odds_ratio", True, "ratio", "conditional"),
    ("mean_difference", False, "difference", "conditional and marginal")])
def test_a_conditional_measure_is_the_locked_fits_row_on_its_scale(measure, ratio, scale,
                                                                   conditioning):
    """A conditional measure's number is the locked fit's exposure row as GET …/fits serves it,
    on the measure's scale: the odds ratio its ``ratio`` columns (exp of the log odds and of its
    interval's ends), a difference the coefficient itself. A change to the plan after the lock
    leaves it standing (the fit record is the locked fit's); where the results serve the locked
    plan, a row they serve otherwise leaves the number not known."""
    log = _locked(measure)
    got = A.result_of("inference", log, _locked_fit(), A.Read(200, _results(measure)))
    assert (got.get("refused"), got["measure"], got["scale"], got["conditioning"],
            got["source"]) == (None, measure, scale, conditioning, "fits")
    assert got["estimates"] == _reported({"linear": [LOG_ODDS]}, ratio=ratio)
    assert got["estimates"][0]["estimate"] == (math.exp(0.5) if ratio else 0.5)
    rules = {"kind": "set_exclusions", "rules": []}
    later = A.result_of("inference", _locked(measure, rules), _locked_fit(), A.Read(200, {}))
    assert later["estimates"] == got["estimates"]
    other = _results(measure)
    other["families"][0]["effect"] = [{**LOG_ODDS, "estimate": 0.6, "ratio": math.exp(0.6)}]
    differ = A.result_of("inference", log, _locked_fit(), A.Read(200, other))
    assert (differ["refused"], differ["refused_kind"]) == (A.rows_differ_why(4), "not_known")


# The ruling of 2026-10-23 on the shrunk effect: ridge declares a shrunk table with no intervals
# (``ridge.py``), so its fit's row of the exposure is no estimate of the effect; the effects stage
# refuses where no chosen family has a coefficient table, in the words its construction gives
# (``stages.effects.no_table_refusal``, the linear model offered first).
SHRUNK = {"feature": "x", "estimate": 0.42, "ci_low": None, "ci_high": None, "p": None}
NO_TABLE_WORDS = ("None of the chosen model families has a coefficient table, so there is no "
                  "effect to report; choose the linear model.")
NO_TABLE_REFUSAL = {"stage": "effects", "code": "no_coefficient_table", "message": NO_TABLE_WORDS,
                    "exits": [{"label": "Add Linear model, which has a coefficient table",
                               "decision": {"kind": "select_models",
                                            "models": ["ridge", "linear"]}}]}


def _fit_of(effects: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**_locked_fit()[0], "effects": effects}]


def _served_for(families: list[dict[str, Any]], effects_stage: dict[str, Any]) -> dict[str, Any]:
    """GET …/results for the locked fit's key: ``families`` as served, the effects stage as
    ``effects_stage`` serves its status."""
    return {**_results("mean_difference"), "effects": effects_stage,
            "families": [{"label": None, "sequence": [], "marginal": None, "e_values": [], **f}
                         for f in families]}


def test_a_locked_fit_whose_families_have_no_coefficient_table_is_refused_in_the_engines_words():
    """Ridge alone, locked: its fit's shrunk row of `x` is no estimate. The results serve none for
    it and, in its place, the effects stage's refusal; the record takes no estimate from the fit's
    row and is refused in the stage's words, none served. Where the results serve another fit (a
    change of models after the lock), the stage's words for it are not the locked fit's, and the
    refusal says only that no family of the locked fit has a table."""
    refused_stage = {"key": None, "fresh": False, "status": "error"}
    doc = _served_for([{"family": "ridge", "effect": None, "refusal": NO_TABLE_REFUSAL}],
                      refused_stage)
    got = A.result_of("inference", _locked("mean_difference"), _fit_of({"ridge": [SHRUNK]}),
                      A.Read(200, doc))
    assert (got["refused"], got["refused_kind"], got.get("estimates")) == (
        NO_TABLE_WORDS, "not_served", None)
    assert got["attempt"] == "fit1" and validate_result(got)
    other = {**doc, "fit": {"key": "key2", "fresh": True, "status": "fresh"}}
    got = A.result_of("inference", _locked("mean_difference"), _fit_of({"ridge": [SHRUNK]}),
                      A.Read(200, other))
    assert (got["refused"], got["refused_kind"]) == (A.NO_TABLE, "not_served")
    assert json.dumps(SHRUNK["estimate"]) not in json.dumps(got)


def test_a_locked_fit_with_a_coefficient_table_keeps_its_row_and_a_shrunk_one_beside_it_is_left():
    """Both ways: the linear model's row of `x` is the estimate, alone or beside ridge, whose
    shrunk row the results serve no effect for (and the stage, fitting the linear model, refuses
    nothing); the record takes the linear model's row only, and the rows the results serve agree."""
    fresh = {"key": "e1", "fresh": True, "status": "fresh"}
    alone = A.result_of("inference", _locked("mean_difference"), _fit_of({"linear": [LOG_ODDS]}),
                        A.Read(200, _served_for([{"family": "linear", "effect": [LOG_ODDS],
                                                  "refusal": None}], fresh)))
    assert (alone.get("refused"), alone["source"]) == (None, "fits")
    assert alone["estimates"] == _reported({"linear": [LOG_ODDS]})
    doc = _served_for([{"family": "linear", "effect": [LOG_ODDS], "refusal": None},
                       {"family": "ridge", "effect": None, "refusal": None}], fresh)
    both = A.result_of("inference", _locked("mean_difference"),
                       _fit_of({"linear": [LOG_ODDS], "ridge": [SHRUNK]}), A.Read(200, doc))
    assert (both.get("refused"), both["estimates"]) == (None, _reported({"linear": [LOG_ODDS]}))
    assert validate_result(both)


def test_a_locked_fit_whose_families_report_no_coefficients_is_refused_in_the_engines_words():
    """The random forest alone, locked: it reports no coefficients, so the fit record serves no
    family's rows of `x` (``effects`` empty, as ``GET …/fits`` serves a family with none), and the
    effects stage refuses as it does for ridge, no chosen family having a coefficient table. The
    record is refused in the stage's words as the results serve them in its slot, none served.
    Both ways: where the route's gate withheld the rows, its reason still wins and the numbers are
    not known; where a family of the fit has a coefficient table and the fit serves no rows, the
    record says only that, since the stage gives no refusal for it."""
    forest = {**NO_TABLE_REFUSAL, "exits": [{"label": "Add Linear model, which has a coefficient "
                                                      "table",
                                             "decision": {"kind": "select_models",
                                                          "models": ["random_forest", "linear"]}}]}
    refused_stage = {"key": None, "fresh": False, "status": "error"}
    doc = _served_for([{"family": "random_forest", "effect": None, "refusal": forest}],
                      refused_stage)
    lock = [{**_locked_fit()[0], "families": ["random_forest"], "effects": {}}]
    got = A.result_of("inference", _locked("mean_difference"), lock, A.Read(200, doc))
    assert (got["refused"], got["refused_kind"], got.get("estimates")) == (
        NO_TABLE_WORDS, "not_served", None)
    assert validate_result(got)
    withheld = [{**lock[0], "withheld": "The plan is not locked."}]
    got = A.result_of("inference", _locked("mean_difference"), withheld, A.Read(200, doc))
    assert (got["refused"], got["refused_kind"]) == ("The plan is not locked.", "not_known")
    tabled = [{**lock[0], "families": ["linear"]}]
    got = A.result_of("inference", _locked("mean_difference"), tabled, A.Read(200, doc))
    assert (got["refused"], got["refused_kind"]) == ("The locked fit serves no exposure rows.",
                                                     "not_served")


@pytest.fixture(scope="module")
def walking_locks() -> dict[str, dict[str, Any]]:
    """The walking table's short Estimate journeys, ridge alone, the random forest alone and the
    linear model alone, each locked by Fit, as the server serves them
    (``test_results_no_coefficient_table_api``)."""
    from turbotab.server.tests.test_results_no_coefficient_table_api import _locked_journey

    return {"ridge": _locked_journey(["ridge"]), "random_forest": _locked_journey(["random_forest"]),
            "linear": _locked_journey(["linear"])}


def test_on_the_served_journeys_a_ridge_lock_is_refused_and_a_linear_lock_is_least_squares(
        walking_locks):
    """End to end on the server's own documents: ridge alone, locked, is refused, none served, in
    the effects stage's words as its status serves them, which are those its construction gives;
    the linear model alone is its row of `walks_daily`, statsmodels' least squares with HC3
    intervals on the 120 complete cases (``_ols_hc3``)."""
    from turbotab.server.tests.test_results_no_coefficient_table_api import _ols_hc3

    def result(name: str) -> dict[str, Any]:
        j = walking_locks[name]
        return A.result_of("inference", j["view"]["decisions"], j["fits"],
                           A.Read(200, j["results"]))

    stage = walking_locks["ridge"]["view"]["stages"]["effects"]["refusal"]
    got = result("ridge")
    assert (got["refused"], got["refused_kind"], got.get("estimates")) == (
        stage["message"], "not_served", None)
    assert stage["message"] == NO_TABLE_WORDS and validate_result(got)
    got = result("linear")
    truth = _ols_hc3()
    assert got.get("refused") is None and got["source"] == "fits" and validate_result(got)
    [e] = got["estimates"]
    assert (e["family"], e["feature"], e["why"]) == ("linear", "walks_daily", None)
    assert (e["estimate"], e["ci_low"], e["ci_high"]) == pytest.approx(
        (truth["estimate"], truth["ci_low"], truth["ci_high"]), rel=1e-9)


def test_on_the_served_journey_a_random_forest_lock_is_refused_in_the_effects_stages_words(
        walking_locks):
    """The random forest alone, locked, end to end: its fit record serves no family's rows, the
    effects stage refuses (no chosen family has a coefficient table), the results serve that
    refusal in the family's slot, and the record is refused in those words, none served."""
    j = walking_locks["random_forest"]
    status = j["view"]["stages"]["effects"]
    assert status["status"] == "error" and status["refusal"]["code"] == "no_coefficient_table"
    [record] = [f for f in j["fits"] if f.get("locked")]
    assert record["families"] == ["random_forest"] and not any(record["effects"].values())
    [slot] = j["results"]["families"]
    assert (slot["family"], slot["effect"], slot["refusal"]) == (
        "random_forest", None, status["refusal"])
    got = A.result_of("inference", j["view"]["decisions"], j["fits"], A.Read(200, j["results"]))
    assert (got["refused"], got["refused_kind"], got.get("estimates")) == (
        status["refusal"]["message"], "not_served", None)
    assert status["refusal"]["message"] == NO_TABLE_WORDS and validate_result(got)


def test_a_lock_whose_plan_declares_no_estimand_reports_no_estimate():
    no_estimand = [*_locked("odds_ratio")[:3], _rec(4, 4, {"kind": "lock_plan", "plan": {},
                                                            "digest": "d"})]
    got = A.result_of("inference", no_estimand, _locked_fit(), A.Read(200, {}))
    assert (got["refused"], got["refused_kind"]) == (A.no_estimand_why(4), "not_reported")


# ── DRV-FIX (f): the run's end ──────────────────────────────────────────────


def test_the_runs_end_is_when_its_last_command_returned_unless_stop_recorded_one(tmp_path):
    """SWARM_PLAN §4's run holds its start and its end. The record is assembled with the server
    still up, before ``stop`` writes driver.json's ``ended``, so the run's end is when the last
    command the driver ran for it returned (``commands.jsonl``' ``done``; a line logged before
    ``done`` was kept, its ``at``), and driver.json's own ``ended`` where it holds one."""
    from research.swarm import driver as drv

    def assembled(name: str, ended: str | None, lines: list[dict[str, Any]]) -> Any:
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        run = drv.Run(folder, {"run": name, "pid": 1, "port": 1, "project": "p1",
                               "started": "2026-10-18T10:00:00+00:00", "ended": ended})
        run.save()
        for line in lines:
            run.log({"command": "ask", "args": {}, "summary": {"exit": 0}, **line})
        path, ok = A.assemble(folder, meta=META, client=FakeClient(_served_routes()),
                              corpus=tmp_path / "corpus", mattered_wait=0)
        assert ok, path.read_text("utf-8")[:2000]
        return validate(json.loads(path.read_text("utf-8"))).run

    lines = [{"at": "2026-10-18T10:10:00+00:00", "done": "2026-10-18T10:12:00+00:00"},
             {"at": "2026-10-18T10:20:00+00:00", "done": "2026-10-18T10:24:30+00:00"}]
    run = assembled("pilot-" + "1" * 32, None, lines)
    assert (run.started.isoformat(), run.ended.isoformat()) == ("2026-10-18T10:00:00+00:00",
                                                                "2026-10-18T10:24:30+00:00")
    older = assembled("pilot-" + "2" * 32, None, [{"at": "2026-10-18T10:20:00+00:00"}])
    assert older.ended.isoformat() == "2026-10-18T10:20:00+00:00"
    stopped = assembled("pilot-" + "3" * 32, "2026-10-18T10:30:00+00:00", lines)
    assert stopped.ended.isoformat() == "2026-10-18T10:30:00+00:00"
    assert assembled("pilot-" + "4" * 32, None, []).ended is None  # nothing ran: no end known


# ── reading the server ───────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status_code, self._body = status, body
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.text = self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        if isinstance(self._body, bytes):
            raise ValueError("not JSON")
        return self._body


class FakeClient:
    """Serves ``routes`` by path suffix and keeps the order of every GET."""

    def __init__(self, routes: dict[str, Any]):
        self.routes, self.asked = routes, []

    def get(self, url: str) -> FakeResponse:
        self.asked.append(url)
        for suffix, answer in self.routes.items():
            if url.endswith(suffix):
                answer = answer() if callable(answer) else answer
                return answer if isinstance(answer, FakeResponse) else FakeResponse(200, answer)
        return FakeResponse(404, {"error": {"code": "unknown", "message": url, "exits": []}})


def test_the_analyst_log_is_read_before_any_route_that_can_render_a_claim():
    states = iter(["running", "running", "fresh"])
    client = FakeClient({"/analyst-log": {"claims": [], "previews": []},
                         "/mattered": lambda: {"status": next(states)}})
    served = A.fetch(client, "p1", poll=0.0)
    assert client.asked[0] == "/api/projects/p1/analyst-log"
    assert client.asked.count("/api/projects/p1/mattered") == 3  # waited while it computed
    assert served.mattered.body == {"status": "fresh"}


def _run_dir(folder: Path, **driver: Any) -> Path:
    folder.mkdir(parents=True)
    (folder / A.DRIVER_FILE).write_text(json.dumps({"run": folder.name, "project": "p1",
                                                    "port": 1, **driver}), "utf-8")
    return folder


def test_the_record_reads_the_run_directory_the_driver_writes(tmp_path):
    """DRV-INT: the run directory as ``research.swarm.driver`` writes it (``Run.save``,
    ``Run.log`` of each command's ``summary``): its run, project, budget, start and end, its
    server, each refusal it met with the command after it, and the analyst's own export. Each
    value is the one written here; a refused export is not the export reached."""
    from research.swarm import driver as drv

    folder = tmp_path / "runs" / "pilot-0123456789abcdef0123456789abcdef"
    folder.mkdir(parents=True)
    meta = {"run": folder.name, "pid": 4242, "port": 1, "project": "p1",
            "budget": {"fits": 3, "seconds": 600.0}, "started": "2026-10-18T10:00:00+00:00",
            "ended": "2026-10-18T10:30:00+00:00", "workers": 2, "analyst": "scripted"}
    run = drv.Run(folder, meta)
    run.save()
    way = {"label": "Keep the score at the opening", "decision": {"kind": "open_seal"}}
    refused = {"refused": {"by": "server", "status": 409, "code": "no_reseal",
                           "message": "The held-out rows open once.",
                           "exits": drv.opts.exits([way])}}
    exported = {"exported": {"path": str(folder / "export.zip"), "bytes": 9, "sha256": "ab"}}
    for at, name, code, payload in (("10:10", "answer", 1, refused),
                                    ("10:20", "export", 1, {"refused": {
                                        "by": "server", "code": "result_not_ready",
                                        "message": "Not yet.", "exits": []}}),
                                    ("10:25", "export", 0, exported)):
        run.log({"at": f"2026-10-18T{at}:00+00:00", "command": name, "args": {},
                 "summary": drv.summary(name, code, payload)})
    path, ok = A.assemble(folder, meta=META, client=FakeClient(_served_routes()),
                          corpus=tmp_path / "corpus", mattered_wait=0)
    assert ok, path.read_text("utf-8")[:2000]
    rec = validate(json.loads(path.read_text("utf-8")))
    assert rec.run.id == folder.name
    assert (rec.run.started.isoformat(), rec.run.ended.isoformat()) == (meta["started"],
                                                                        meta["ended"])
    assert rec.run.fit_budget == meta["budget"]
    assert (rec.run.server["pid"], rec.run.server["port"]) == (4242, 1)
    assert [(s.command, s.code, s.message, s.exits, s.then) for s in rec.surfacings.refusals] == [
        ("answer", "no_reseal", "The held-out rows open once.", [way["label"]], "export"),
        ("export", "result_not_ready", "Not yet.", [], "export")]
    assert rec.export.reached
    only_refused = tmp_path / "runs" / "pilot-fedcba9876543210fedcba9876543210"
    only_refused.mkdir()
    drv.Run(only_refused, {**meta, "run": only_refused.name}).save()
    drv.Run(only_refused, meta).log({"at": meta["ended"], "command": "export", "args": {},
                                     "summary": drv.summary("export", 1, refused)})
    path, ok = A.assemble(only_refused, meta=META, client=FakeClient(_served_routes()),
                          corpus=tmp_path / "corpus", mattered_wait=0)
    assert ok and not validate(json.loads(path.read_text("utf-8"))).export.reached


def test_a_server_that_serves_no_analyst_log_leaves_a_failed_run(tmp_path):
    refused = FakeResponse(403, {"error": {"code": "analyst_mode_only", "message": "not analyst",
                                           "exits": []}})
    client = FakeClient({"/analyst-log": refused, "/p1": {"decisions": [], "state": {}}})
    path, ok = A.assemble(_run_dir(tmp_path / "runs" / "r1"), client=client,
                          corpus=tmp_path / "corpus", mattered_wait=0)
    assert not ok and path == tmp_path / "corpus" / "failed" / "r1.json"
    assert "analyst_mode_only" in json.loads(path.read_text("utf-8"))["assembled"]["unreadable"]


def _served_routes() -> dict[str, Any]:
    """A server's answers for a run that declared nothing, enough for a record that validates."""
    def refused(status: int, code: str) -> FakeResponse:
        return FakeResponse(status, {"error": {"code": code, "message": code, "exits": []}})

    return {"/analyst-log": {"claims": [], "previews": []}, "/fits": {"records": []},
            "/results": {}, "/mattered": refused(409, "not_offered"), "/materiality": {"rows": []},
            "/export": refused(409, "not_ready"),
            "/api/health": {"version": "2.0.0.dev0", "mode": "local", "workers": 2},
            "/p1": {"decisions": [], "state": {}, "summary": {"source_name": "table.csv"}}}


META = {"design": 2, "analyst_model": "scripted", "prompt_version": "v1", "dataset": {"id": "d"}}


@pytest.mark.parametrize("route,name", [("/fits", "The fit records"),
                                        ("/materiality", "The materiality ledger"),
                                        ("/results", "The results")])
def test_a_route_that_fails_to_serve_leaves_a_failed_run_not_an_empty_record(tmp_path, route,
                                                                            name):
    """DB-V8: these routes refuse only a project the server does not hold, so a failed read is no
    record of an empty run (no fit, no concern): a failed run. Silent where each serves."""
    run = _run_dir(tmp_path / "runs" / "r1", started="2026-10-18T10:00:00+00:00")
    path, ok = A.assemble(run, meta=META, client=FakeClient(_served_routes()),
                          corpus=tmp_path / "served", mattered_wait=0)
    assert ok and validate(json.loads(path.read_text("utf-8"))).run.fits == 0
    routes = {**_served_routes(), route: FakeResponse(500, {"detail": "Internal Server Error"})}
    path, ok = A.assemble(run, meta=META, client=FakeClient(routes), corpus=tmp_path / "failed",
                          mattered_wait=0)
    assert not ok and path == tmp_path / "failed" / "failed" / "r1.json"
    assert json.loads(path.read_text("utf-8"))["assembled"]["unreadable"].startswith(
        f"{name} could not be read (HTTP 500)")


# ── end to end ───────────────────────────────────────────────────────────────

FIXTURE = "clinical_risk.csv"
RUN_ID = "e2e-clinical-1"
SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
ROLE = {"kind": "confirm_role", "column": "sodium_mmol_l", "role": "excluded"}
AFTER_SEAL = {"kind": "select_models", "models": ["linear", "elastic_net"]}


@contextmanager
def analyst_server(home: Path):
    """``server_drive.local_server`` in analyst mode, as the driver runs its server: the harness
    takes no analyst flag, so its ``Settings`` are given one. Claims render strictly, as in every
    core test (``core/tests/conftest.py``)."""
    from turbotab.core import config
    from turbotab.core.tests.acceptance.server_drive import local_server

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "Settings", functools.partial(config.Settings, analyst=True))
        mp.setenv("TURBOTAB_CLAIMS_STRICT", "1")
        with local_server(home) as client:
            yield client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload(r: Any, command: str) -> tuple[int, dict[str, Any]]:
    """What the driver would print for the server's response ``r`` to ``command``, and its exit
    status: a refusal as the driver's ``refused`` (``driver.refused``), the export as its
    ``exported``, anything else as served."""
    from research.swarm import driver as drv

    if r.status_code != 200:
        error = r.json().get("error") or {}
        return 1, {"refused": {"by": "server", "status": r.status_code, "code": error.get("code"),
                               "message": error.get("message"),
                               "exits": drv.opts.exits(error.get("exits"))}}
    if command == "export":
        return 0, {"exported": {"path": "export.zip", "bytes": len(r.content),
                                "sha256": hashlib.sha256(r.content).hexdigest()}}
    return 0, {"results": r.json()} if command == "results" else {}


class ScriptedDriver:
    """A scripted analyst's client: every decision posted with a note, kept by the seq it was
    recorded at, and each command logged as a line of ``commands.jsonl`` as the driver logs it
    (``driver.main``: its time, name and ``driver.summary`` of what it printed)."""

    def __init__(self, client: Any):
        self.c, self.commands, self.notes, self.n = client, [], {}, 0

    @property
    def app(self) -> Any:
        return self.c.app

    def log(self, command: str, r: Any) -> None:
        from research.swarm import driver as drv

        code, payload = _payload(r, command)
        self.commands.append({"at": _now(), "command": command, "args": {},
                              "summary": drv.summary(command, code, payload)})

    def get(self, url: str, **kw: Any) -> Any:
        r = self.c.get(url, **kw)
        for command in ("results", "export"):
            if url.endswith(f"/{command}"):
                self.log(command, r)
        return r

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:  # noqa: A002
        if url.endswith("/decisions"):
            self.n += 1
            note = f"Reason {self.n}: {json.get('kind')} as the evidence on its card reads."
            kw.setdefault("params", {})["note"] = note
            r = self.c.post(url, json=json, **kw)
            self.log("answer", r)
            if r.status_code == 200:  # TurboTab may record its own answers in the same request
                mine = next(d for d in r.json()["decisions"] if d["note"] == note)
                self.notes[mine["seq"]] = note
            return r
        r = self.c.post(url, json=json, **kw)
        command = "preview" if url.endswith("/preview") else "fit" if url.endswith("/fit") else None
        if command:
            self.log(command, r)
        return r


def _first_fit(driver: ScriptedDriver, pid: str) -> None:
    """``clinical_risk.csv`` answered through its first fit, the linear model's, with Fit pressed
    (a preview of excluding sodium and of the blanks' strategy on the way)."""
    from turbotab.core.tests.acceptance.server_drive import press_fit
    from turbotab.core.tests.truths import FIXTURE_TRUTHS
    from turbotab.server.tests.conftest import answer_settled, declare, prepare, wait_for

    declare(pid, FIXTURE_TRUTHS[FIXTURE], fixture=FIXTURE)
    wait_for(driver, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    for body in ({"kind": "set_lens", "lenses": ["clinical"]},
                 {"kind": "set_target", "column": "readmit_30d"},
                 {"kind": "set_purpose", "purpose": "prediction"}, SPLIT):
        prepare(driver, pid, body)
        assert answer_settled(driver, pid, None, body).status_code == 200, body
    for body in (ROLE, {"kind": "set_missing", "strategy": "impute"}):
        assert driver.post(f"/api/projects/{pid}/preview", json=body).status_code == 200
    models = {"kind": "select_models", "models": ["linear"]}
    prepare(driver, pid, models)
    assert answer_settled(driver, pid, None, models).status_code == 200
    wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
    assert press_fit(driver, pid)


def _drive(driver: ScriptedDriver, pid: str) -> dict[str, Any]:
    from turbotab.core.tests.truths import answer_refusal
    from turbotab.server.tests.conftest import answer_settled, truth_of, wait_for

    _first_fit(driver, pid)
    assert answer_settled(driver, pid, None, ROLE).status_code == 200
    wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
    assert driver.get(f"/api/projects/{pid}/results").status_code == 200
    url = f"/api/projects/{pid}/decisions"
    opening = {"kind": "open_seal", "family": "linear"}
    opened = answer_refusal(lambda d: driver.post(url, json=d), driver.post(url, json=opening),
                            truth_of(pid), lambda: driver.post(url, json=opening))
    assert opened.status_code == 200, opened.text
    assert answer_settled(driver, pid, None, AFTER_SEAL).status_code == 200
    wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
    resealed = driver.post(url, json={"kind": "reseal", "reason": "a better draw"})
    assert resealed.status_code == 409
    exported = driver.get(f"/api/projects/{pid}/export")
    assert exported.status_code == 200
    return {"reseal": resealed.json()["error"]}


def finish(run_dir: Path, driver: ScriptedDriver, pid: str, start: str,
           dataset: dict[str, Any]) -> None:
    """The run directory as the driver leaves it (``driver.cmd_start``'s ``driver.json``, with the
    run's own fields beside it): its commands, the analyst's transcript (here the constructed one)
    and driver.json."""
    (run_dir / A.COMMANDS_FILE).write_text(
        "".join(json.dumps(c) + "\n" for c in driver.commands), "utf-8")
    transcript(run_dir / "transcript.jsonl")
    (run_dir / A.DRIVER_FILE).write_text(json.dumps({
        "run": run_dir.name, "pid": os.getpid(), "port": None, "project": pid,
        "budget": {"fits": 10, "seconds": 600.0}, "started": start, "ended": _now(),
        "design": 2, "analyst_model": "scripted", "effort": None, "prompt_version": "e2e-1",
        "transcript": "transcript.jsonl", "dataset": dataset}), "utf-8")


@pytest.fixture(scope="module")
def e2e(tmp_path_factory) -> dict[str, Any]:
    """One scripted run to its record, and what the server serves once it is assembled."""
    folder = tmp_path_factory.mktemp("swarm_e2e")
    run_dir = folder / "runs" / RUN_ID
    start = _now()
    with analyst_server(run_dir / "home") as client:
        made = client.post("/api/projects", json={"path": str(SAMPLE)})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        driver = ScriptedDriver(client)
        seen = _drive(driver, pid)
        finish(run_dir, driver, pid, start,
               {"id": "clinical-risk", "control": "known_answer", "fixture": FIXTURE})
        log_before = client.get(f"/api/projects/{pid}/analyst-log").json()
        began = time.monotonic()
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        seconds = time.monotonic() - began
        base = f"/api/projects/{pid}"
        served = {name: client.get(f"{base}{route}").json() for name, route in (
            ("view", ""), ("fits", "/fits"), ("results", "/results"))}
        bundle = client.get(f"{base}/export").content
    return {"path": path, "ok": ok, "notes": driver.notes, "served": served, "bundle": bundle,
            "log_before": log_before, "seconds": seconds, "corpus": folder / "corpus", **seen}


SAMPLE = Path(__file__).resolve().parents[3] / "turbotab" / "sample_data" / FIXTURE


def test_the_run_leaves_one_record_that_validates(e2e):
    assert e2e["ok"], e2e["path"].read_text("utf-8")[:2000]
    assert e2e["path"] == e2e["corpus"] / f"{RUN_ID}.json"
    assert not (e2e["corpus"] / "failed").exists()
    rec = validate(json.loads(e2e["path"].read_text("utf-8")))
    assert rec.run.id == RUN_ID and rec.run.design == 2


@pytest.fixture(scope="module")
def rec(e2e):
    return validate(json.loads(e2e["path"].read_text("utf-8")))


def test_each_decision_is_the_logs_record_with_the_reason_the_driver_posted(rec, e2e):
    log = e2e["served"]["view"]["decisions"]
    assert [(d.id, d.seq, d.kind) for d in rec.decisions] == [
        (r["id"], r["seq"], r["decision"]["kind"]) for r in log]
    assert {d.seq: d.reason for d in rec.decisions} == e2e["notes"]
    assert all(d.source == "analyst" for d in rec.decisions)  # TurboTab recorded nothing itself
    after = [d for d in rec.decisions if d.post_seal]
    assert [(d.kind, d.phase, d.option["models"]) for d in after] == [
        ("select_models", "after_seal", AFTER_SEAL["models"])]


def test_each_attempt_is_the_fit_record_as_served(rec, e2e):
    served = e2e["served"]["fits"]["records"]
    assert len(rec.attempts) == len(served) == 3
    for attempt, fit in zip(rec.attempts, served):
        mine = attempt.model_dump(mode="json")
        for key in ("id", "fit_key", "plan_digest", "decisions", "families", "scores",
                    "movement", "movement_why", "holdout", "changed_after_seal", "at_opening"):
            assert mine[key] == fit[key], key
    role = next(d for d in rec.decisions if d.kind == "confirm_role")
    assert rec.attempts[1].changed == [role.seq]
    assert rec.run.fits == 3
    assert rec.run.fit_seconds == pytest.approx(sum(f["seconds"] for f in served))


def test_the_role_change_has_the_engines_movement_and_its_examined_options(rec, e2e):
    """Dropping sodium between fits 1 and 2 was the only change, the rows and folds unchanged
    (the fixture has no blanks), so its materiality is fit 2's served paired movement."""
    role = next(d for d in rec.decisions if d.kind == "confirm_role")
    fit2 = e2e["served"]["fits"]["records"][1]
    assert role.materiality.source == "fits" and role.materiality.attempt == fit2["id"]
    assert role.materiality.movement == fit2["movement"]
    assert fit2["movement"][0]["movement"]["value"] is not None
    examined = [(e.how, e.value, e.alternative) for e in role.examined
                if e.question == "role:sodium_mmol_l"]
    assert ("preview", "excluded", False) in examined  # the option chosen, previewed
    assert any(how == "fit" and value != "excluded" and alt for how, value, alt in examined)


def test_the_result_is_the_first_openings_scores_and_no_later_record_reached_it(rec, e2e):
    log = e2e["served"]["view"]["decisions"]
    opening = next(r for r in log if r["decision"]["kind"] == "open_seal")
    after = [r["seq"] for r in log if r["post_seal"]]
    r = rec.result
    assert (r.opening, r.family, r.scores) == (opening["seq"], "linear",
                                               opening["decision"]["scores"])
    opened = next(f for f in e2e["served"]["fits"]["records"]
                  if (f["at_opening"] or {}).get("seq") == opening["seq"]
                  and not f["changed_after_seal"])
    assert r.attempt == opened["id"] and r.decisions == opened["decisions"]
    assert after and not set(after) & set(r.decisions)
    # A change after the opening: the intervals served are the changed fit's, not the opening's.
    assert r.intervals is None
    assert r.intervals_why == e2e["served"]["results"]["at_opening"]["note"]


def test_the_run_dataset_regime_and_export_are_the_servers_and_the_files(rec, e2e):
    frame = pd.read_csv(SAMPLE)
    assert rec.dataset.files[0].sha256 == hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert rec.regime.n_rows == len(frame) and rec.regime.missing == {}
    assert rec.regime.events == int((frame["readmit_30d"] == 1).sum())
    assert rec.regime.n_analyzed == len(frame.dropna())
    assert rec.export.reached and rec.export.sha256 == hashlib.sha256(e2e["bundle"]).hexdigest()
    assert rec.export.methods.startswith("# Methods")
    assert rec.export.checklist["checklist"]
    head = subprocess.run(["git", "-C", str(SAMPLE.parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert rec.run.engine_commit == head
    assert (rec.run.spend.fresh, rec.run.spend.cached) == (FRESH, CACHED)
    # The claims are the analyst log as it stood when the run ended, before any read of assembly.
    assert rec.claims == [Claim.model_validate(c) for c in e2e["log_before"]["claims"]]


def test_the_reseal_the_server_refused_is_a_surfacing(rec, e2e):
    refusal = next(s for s in rec.surfacings.refusals if s.code == e2e["reseal"]["code"])
    assert refusal.command == "answer" and refusal.message == e2e["reseal"]["message"]
    assert refusal.then == "export"


def test_every_check_gives_what_the_runs_construction_implies(rec):
    got = {o.check: o for o in C.run_checks([rec])}
    assert got["seal_integrity"].outcome == "pass"
    assert got["logging"].outcome == "pass", got["logging"].problems
    assert got["single_pass"].outcome == "pass"  # an alternative examined, and three fits
    assert got["robustness"].outcome == "pass"
    assert got["null_decoy"].outcome == "not_checked"  # not a decoy
    # The fixture's fact readings were answered from its truth, and it declares no answer.
    known = got["known_answer"]
    assert known.outcome == "not_checked" and known.problems == []
    assert "no known answer is declared" in known.unchecked[0]


# ── end to end under Estimate ────────────────────────────────────────────────

NHANES_TRUTH = "_tt_tmp_nhanes.csv"
WIDER = {"kind": "set_exclusions", "rules": [
    {"column": "kcal", "low": 600, "high": 4500, "reason": "implausible intake"}]}


@pytest.fixture(scope="module")
def estimate_e2e(tmp_path_factory) -> dict[str, Any]:
    """The NHANES journey under Estimate (sugar and glucose, ``test_fit_lock_repair``'s drive, its
    truth ``truths.FIXTURE_TRUTHS``' own), every decision with a note: a preview of dropping the
    exclusion rule, the linear fit, Fit pressed (the lock), then the rule changed with the
    estimates in view, a second fit and its curve, and the export."""
    from turbotab.core.tests.acceptance.server_drive import settle_forms
    from turbotab.server.tests.conftest import wait_for
    from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press

    folder = tmp_path_factory.mktemp("swarm_e2e_estimate")
    run_dir = folder / "runs" / "e2e-nhanes-1"
    start = _now()
    with analyst_server(run_dir / "home") as client:
        driver = ScriptedDriver(client)
        drive = _inference_drive(driver)
        pid = drive.pid
        preview = driver.post(f"/api/projects/{pid}/preview",
                              json={"kind": "set_exclusions", "rules": []})
        assert preview.status_code == 200, preview.text
        drive.decide({"kind": "select_models", "models": ["linear"]})
        wait_for(driver, pid, {"fit": "fresh"}, timeout=400)
        assert press(driver, pid)["locked"] is True
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        assert driver.get(f"/api/projects/{pid}/results").status_code == 200
        drive.decide(WIDER)
        settle_forms(drive)
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        assert driver.get(f"/api/projects/{pid}/export").status_code == 200
        finish(run_dir, driver, pid, start,
               {"id": "nhanes", "control": "known_answer", "fixture": NHANES_TRUTH})
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        base = f"/api/projects/{pid}"
        served = {name: client.get(f"{base}{route}").json() for name, route in (
            ("view", ""), ("fits", "/fits"), ("mattered", "/mattered"))}
        in_force = _server_in_force(client, pid)
    assert ok, path.read_text("utf-8")[:2000]
    return {"rec": validate(json.loads(path.read_text("utf-8"))), "served": served,
            "notes": driver.notes, "in_force": in_force}


def _server_in_force(client: Any, pid: str) -> set[int]:
    """The seqs of the records the server holds in force now (``provenance.in_force``, its own
    rule over its own log)."""
    from turbotab.core import provenance

    records = client.app.state.service.log(pid).records()
    ids = provenance.in_force(records)
    return {r.seq for r in records if r.id in ids}


def _adjustment_choices_hold(rec: Any, curve: dict[str, Any], in_force: set[int]) -> None:
    """DB-V4 on the server: each covariate's choices on the curve are credited to the one
    adjustment record the server holds in force for it, never to another group's, and none of
    those records is called superseded."""
    groups = [d for d in rec.decisions if d.kind == "set_adjustment"]
    varied = [c["key"] for c in curve["choices"] if c["decision"] == "set_adjustment"]
    assert len(groups) > 1 and varied
    credited: list[str] = []
    for d in groups:
        assert (d.seq in in_force) == ("Superseded" not in (d.materiality.why or "")), d.seq
        if d.seq not in in_force:
            continue
        columns = set(d.option["answers"])
        mine = [k for k in varied if k.split(":", 1)[1] in columns]
        got = [x["choice"] for x in d.materiality.departures or []]
        assert sorted(set(got)) == sorted(mine), (d.seq, got, mine)
        credited += mine
    assert sorted(credited) == sorted(varied)


def test_under_estimate_the_result_is_the_fit_the_lock_was_recorded_on(estimate_e2e):
    rec, served = estimate_e2e["rec"], estimate_e2e["served"]
    lock = next(r for r in served["view"]["decisions"] if r["decision"]["kind"] == "lock_plan")
    changed = next(r for r in served["view"]["decisions"] if r["after_estimates"])
    assert changed["decision"]["rules"][0]["low"] == WIDER["rules"][0]["low"]
    r = rec.result
    assert (r.lock, r.plan_digest) == (lock["seq"], lock["decision"]["digest"])
    fits = served["fits"]["records"]
    locked = next(f for f in fits if f["id"] == r.attempt)
    assert changed["seq"] not in locked["decisions"]
    assert (r.measure, r.scale, r.source) == ("mean_difference", "difference", "fits")
    assert [e.model_dump() for e in r.estimates] == _reported(
        {f: rows for f, rows in locked["effects"].items() if rows})
    later = [f for f in fits if changed["seq"] in (f["decisions"] or ())]
    assert later and all(f["id"] != r.attempt for f in later)  # the secondary analysis is not it
    mine = {d.seq: d.reason for d in rec.decisions if d.source == "analyst"}
    assert mine == estimate_e2e["notes"]
    own = [d for d in rec.decisions if d.source == "app"]  # TurboTab's split and its lock
    assert [d.kind for d in own] == ["set_split", "lock_plan"]
    assert all(d.reason is None for d in own)


def test_under_estimate_a_choice_the_curve_varies_carries_its_departures(estimate_e2e):
    rec, curve = estimate_e2e["rec"], estimate_e2e["served"]["mattered"]["curve"]
    rules = [d for d in rec.decisions if d.kind == "set_exclusions"]
    first, wider = rules[0], rules[-1]
    assert wider.phase == "after_estimates"
    assert first.materiality.source is None and f"#{wider.seq}" in first.materiality.why
    assert wider.materiality.source == "mattered"
    keys = [c["key"] for c in curve["choices"] if c["decision"] == "set_exclusions"]
    departures = [s for s in curve["specs"] + curve["not_estimated"]
                  if not s["primary"] and any(s["picks"][k] != "recorded" for k in keys)]
    assert [d["choice"] for d in wider.materiality.departures] == keys
    assert [d["spec"] or {"reason": d["not_estimated"]} for d in wider.materiality.departures] == [
        s if "estimate" in s else {"reason": s["reason"]} for s in departures]
    assert wider.materiality.primary == next(s for s in curve["specs"] if s["primary"])
    # Both options it set aside were examined: dropping the rule (previewed) and the old rule (fit).
    assert sorted((e.how, e.alternative) for e in wider.examined) == [("fit", True),
                                                                     ("preview", True)]
    # Each adjustment group carries its own covariates' choices (DB-V4), not the last group all.
    _adjustment_choices_hold(rec, curve, estimate_e2e["in_force"])


def test_under_estimate_every_check_gives_what_the_runs_construction_implies(estimate_e2e):
    got = {o.check: o for o in C.run_checks([estimate_e2e["rec"]])}
    assert got["seal_integrity"].outcome == "pass", got["seal_integrity"].problems
    assert got["logging"].outcome == "pass", got["logging"].problems
    assert got["single_pass"].outcome == "pass"  # the rule's removal previewed before the lock
    assert got["robustness"].outcome == "pass"
    known = got["known_answer"]
    assert known.outcome == "not_checked" and known.problems == []  # kcal's unit and days declared


# ── end to end: reverts between fits (DB-V2) ─────────────────────────────────

ALBUMIN = {"kind": "confirm_role", "column": "albumin_g_dl", "role": "excluded"}
HEMOGLOBIN = {"kind": "confirm_role", "column": "hemoglobin_g_dl", "role": "excluded"}


def _seq_of(driver: ScriptedDriver) -> int:
    """The seq of the record the driver posted last (each post carries its own note)."""
    return max(driver.notes)


@pytest.fixture(scope="module")
def revert_e2e(tmp_path_factory) -> dict[str, Any]:
    """``clinical_risk.csv`` under Predict in analyst mode, every record posted with a note:
    after the first fit, sodium excluded (fit 2), albumin excluded (fit 3), the sodium record
    reverted, the only change before fit 4 (a plan no fit held, so a new fit); then the albumin
    record reverted (fit 1's plan again, whose fit is recorded, so no new record) and hemoglobin
    excluded, both before fit 5."""
    from turbotab.server.tests.conftest import answer_settled, wait_for

    folder = tmp_path_factory.mktemp("swarm_e2e_revert")
    run_dir = folder / "runs" / "e2e-revert-1"
    start = _now()
    seqs: dict[str, int] = {}
    with analyst_server(run_dir / "home") as client:
        made = client.post("/api/projects", json={"path": str(SAMPLE)})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        driver = ScriptedDriver(client)
        _first_fit(driver, pid)

        def record(name: str, body: dict[str, Any]) -> None:
            assert answer_settled(driver, pid, None, body).status_code == 200, body
            seqs[name] = _seq_of(driver)

        def revert(name: str, of: str) -> None:
            log = client.get(f"/api/projects/{pid}").json()["decisions"]
            target = next(r["id"] for r in log if r["seq"] == seqs[of])
            r = driver.post(f"/api/projects/{pid}/decisions",
                            json={"kind": "revert", "decision_id": target})
            assert r.status_code == 200, r.text
            seqs[name] = _seq_of(driver)

        def fitted() -> None:
            wait_for(driver, pid, {"fit": "fresh"}, timeout=600)

        for step, name, body in ((record, "sodium", ROLE), (record, "albumin", ALBUMIN),
                                 (revert, "sodium back", "sodium"),
                                 (revert, "albumin back", "albumin"),
                                 (record, "hemoglobin", HEMOGLOBIN)):
            step(name, body)
            fitted()
        finish(run_dir, driver, pid, start, {"id": "clinical-risk", "fixture": FIXTURE})
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        fits = client.get(f"/api/projects/{pid}/fits").json()["records"]
    assert ok, path.read_text("utf-8")[:2000]
    return {"rec": validate(json.loads(path.read_text("utf-8"))), "fits": fits, "seqs": seqs,
            "notes": driver.notes}


def test_on_the_server_a_fit_after_a_revert_alone_is_the_reverts_change(revert_e2e):
    rec, fits, seqs, notes = (revert_e2e[k] for k in ("rec", "fits", "seqs", "notes"))
    assert len(rec.attempts) == len(fits) == 5  # fit 1's plan again made no new fit record
    by = {a.seq: a for a in rec.attempts}
    assert (by[2].changed, by[3].changed) == ([seqs["sodium"]], [seqs["albumin"]])
    assert by[4].changed == [seqs["sodium back"]]
    assert [w.model_dump() for w in by[4].withdrawn] == [{"seq": seqs["sodium"],
                                                          "by": seqs["sodium back"]}]
    assert [(r.seq, r.note) for r in by[4].reasons] == [(seqs["sodium back"],
                                                         notes[seqs["sodium back"]])]
    undone = next(d for d in rec.decisions if d.seq == seqs["sodium back"])
    assert undone.kind == "revert" and undone.materiality.source == "fits"
    assert undone.materiality.attempt == fits[3]["id"]
    assert undone.materiality.movement == fits[3]["movement"]
    assert fits[3]["movement"][0]["movement"]["value"] is not None
    sodium = next(d for d in rec.decisions if d.seq == seqs["sodium"])
    assert (sodium.materiality.source, sodium.materiality.attempt) == ("fits", fits[1]["id"])


def test_on_the_server_a_revert_and_a_new_record_before_one_fit_share_its_change(revert_e2e):
    rec, fits, seqs = (revert_e2e[k] for k in ("rec", "fits", "seqs"))
    last = rec.attempts[4]
    assert last.changed == sorted([seqs["albumin back"], seqs["hemoglobin"]])
    assert [w.model_dump() for w in last.withdrawn] == [{"seq": seqs["albumin"],
                                                         "by": seqs["albumin back"]}]
    assert [r.seq for r in last.reasons] == last.changed
    by_seq = {d.seq: d for d in rec.decisions}
    for mine, other in (("albumin back", "hemoglobin"), ("hemoglobin", "albumin back")):
        m = by_seq[seqs[mine]].materiality
        assert (m.source, m.attempt) == (None, fits[4]["id"])
        assert m.why.startswith(f"It took effect at fit #{last.seq} together with record "
                                f"#{seqs[other]}:")


def test_on_the_server_after_reverts_logging_passes_and_the_referee_sees_each_revert(revert_e2e):
    from research.swarm import referee

    rec, seqs, notes = (revert_e2e[k] for k in ("rec", "seqs", "notes"))
    got = {o.check: o for o in C.run_checks([rec])}
    assert got["logging"].outcome == "pass", got["logging"].problems
    text = referee.prompt(rec, 4)
    assert f"Record #{seqs['sodium back']}, `revert` of record #{seqs['sodium']}" in text
    assert notes[seqs["sodium back"]] in text


# ── end to end: a rule tried before the lock, and a revert after it (DB-V1) ──


@pytest.fixture(scope="module")
def relock_e2e(tmp_path_factory) -> dict[str, Any]:
    """The NHANES journey under Estimate, every decision with a note: the drive's rule 500–5000
    fitted (fit 1), rule 600–4500 fitted (fit 2), rule 500–5000 recorded again (fit 1's plan, its
    fit recorded), then Fit pressed (the lock), assembled (E1). Then the restoring record reverted
    with the estimates in view, which makes fit 2 current again, assembled again (E3)."""
    from turbotab.core import plan_lock
    from turbotab.core.tests.acceptance.server_drive import settle_forms
    from turbotab.server.tests.conftest import wait_for
    from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press
    from turbotab.server.tests.test_quest_api import rule

    folder = tmp_path_factory.mktemp("swarm_e2e_relock")
    run_dir = folder / "runs" / "e2e-relock-1"
    start = _now()
    dataset = {"id": "nhanes", "control": "known_answer", "fixture": NHANES_TRUTH}
    with analyst_server(run_dir / "home") as client:
        driver = ScriptedDriver(client)
        drive = _inference_drive(driver)
        pid = drive.pid
        service = client.app.state.service
        drive.decide({"kind": "select_models", "models": ["linear"]})
        wait_for(driver, pid, {"fit": "fresh"}, timeout=400)
        for body in (rule(600, 4500), rule(500, 5000)):
            drive.decide(body)
            settle_forms(drive)
            wait_for(driver, pid, {"fit": "fresh"}, timeout=400)
        assert press(driver, pid)["locked"] is True
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        assert driver.get(f"/api/projects/{pid}/results").status_code == 200

        def snapshot(name: str) -> dict[str, Any]:
            finish(run_dir, driver, pid, start, dataset)
            path, ok = A.assemble(run_dir, client=client, corpus=folder / name)
            assert ok, path.read_text("utf-8")[:2000]
            records = service.log(pid).records()
            return {"rec": validate(json.loads(path.read_text("utf-8"))),
                    "locked_key": service._locked_fit(pid, records,
                                                      plan_lock.current_lock(records)),
                    "fits": client.get(f"/api/projects/{pid}/fits").json()["records"],
                    "log": client.get(f"/api/projects/{pid}").json()["decisions"],
                    "results": client.get(f"/api/projects/{pid}/results").json()}

        e1 = snapshot("corpus_e1")
        restore = next(r for r in reversed(e1["log"])
                       if r["decision"]["kind"] == "set_exclusions")
        reverted = driver.post(f"/api/projects/{pid}/decisions",
                               json={"kind": "revert", "decision_id": restore["id"]})
        assert reverted.status_code == 200, reverted.text
        settle_forms(drive)
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        e3 = snapshot("corpus_e3")
    return {"e1": e1, "e3": e3}


def _rows(fit: dict[str, Any]) -> dict[str, Any]:
    return {f: rows for f, rows in fit["effects"].items() if rows}


def test_under_estimate_a_rule_tried_and_undone_before_the_lock_leaves_the_locked_fit(relock_e2e):
    """E1: by construction the lock is on the drive's rule, fit 1's plan; the server's own rule
    for the fit the lock was recorded on (``_locked_fit``) names the same key."""
    e1 = relock_e2e["e1"]
    rules = [r for r in e1["log"] if r["decision"]["kind"] == "set_exclusions"]
    drives, tried, again = rules
    assert (tried["decision"]["rules"][0]["low"], again["decision"]["rules"][0]["low"]) == (600,
                                                                                         500)
    fits = e1["fits"]
    assert len(fits) == 2  # the drive's rule again made no new fit record
    locked, other = fits
    assert drives["seq"] in locked["decisions"] and tried["seq"] not in locked["decisions"]
    assert locked["fit_key"] == e1["locked_key"]
    r = e1["rec"].result
    assert r.refused is None and (r.attempt, r.decisions) == (locked["id"], locked["decisions"])
    assert [e.model_dump() for e in r.estimates] == _reported(_rows(locked)) and r.estimates
    assert tried["seq"] not in r.decisions
    assert C.seal_integrity(e1["rec"]).outcome == "pass"
    # DRV-FIX (a): the results serve the locked plan (the server's own plan rule over the log as
    # served, the locked fit current), so the record's rows were held to the rows they serve
    lock = next(x for x in e1["log"] if x["decision"]["kind"] == "lock_plan")
    assert A.serves_lock(e1["results"], locked, lock, e1["log"]) is None
    assert r.source == "fits" and _reported({f["family"]: f["effect"] for f in e1["results"]
                                             ["families"] if f["effect"]}) == _reported(
        _rows(locked))


def test_under_estimate_a_revert_after_the_lock_leaves_the_locked_fits_estimate(relock_e2e):
    """E3: the revert makes fit 2 current again and its estimate is served, but the primary result
    stays the locked fit's, read from GET …/fits; the record HEAD assembled before DB-V1's repair
    (fit 2's estimate under the lock's digest) is not passed by seal integrity."""
    e3 = relock_e2e["e3"]
    assert e3["log"][-1]["decision"]["kind"] == "revert" and e3["log"][-1]["after_estimates"]
    locked = next(f for f in e3["fits"] if f["fit_key"] == e3["locked_key"])
    other = next(f for f in e3["fits"] if f["id"] != locked["id"])
    assert _rows(other) and _rows(other) != _rows(locked)
    rec = e3["rec"]
    assert rec.result.attempt == locked["id"]
    # DRV-FIX (a): the revert after the lock changed the plan by the server's own rule, so the
    # results serve another plan's estimate, and the locked fit's own rows stand
    lock = next(x for x in e3["log"] if x["decision"]["kind"] == "lock_plan")
    assert A.serves_lock(e3["results"], locked, lock, e3["log"]) == (A.plan_changed_why(
        lock["seq"]), "not_known")
    assert [e.model_dump() for e in rec.result.estimates] == _reported(_rows(locked))
    assert C.seal_integrity(rec).outcome == "pass"
    wrong = rec.model_dump(mode="json", by_alias=True)
    wrong["result"].update(attempt=other["id"], decisions=other["decisions"],
                           estimates=_reported(_rows(other)))
    assert C.seal_integrity(validate(wrong)).outcome != "pass"


# ── end to end: own answers previewed, one fit, and a lock the fit records do not name ──
# (DB-V4, DB-V5)

# Constructed: no interval near it, a value of the NHANES estimand's measure.
KNOWN = {"family": "linear", "feature": "sugar", "measure": "mean_difference", "value": 9.0}


class PreviewingDriver(ScriptedDriver):
    """A scripted analyst that previews each adjustment group with its own answer before it posts
    it, as the verifier's run did: nothing else is previewed."""

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:  # noqa: A002
        if url.endswith("/decisions") and (json or {}).get("kind") == "set_adjustment":
            super().post(url[: -len("/decisions")] + "/preview", json=json)
        return super().post(url, json=json, **kw)


@pytest.fixture(scope="module")
def lone_e2e(tmp_path_factory) -> dict[str, Any]:
    """The NHANES journey under Estimate in analyst mode, every decision with a note and each
    adjustment group previewed with its own answer: the linear fit, then the sensitivity answer
    (a slot of the plan the fit does not read, so no new fit), then Fit pressed, which locks a plan
    whose digest no fit record holds. Assembled twice, as a known-answer control (``KNOWN``) and
    as a null decoy."""
    from turbotab.core import plan_lock
    from turbotab.server.tests.conftest import wait_for
    from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press

    folder = tmp_path_factory.mktemp("swarm_e2e_lone")
    run_dir = folder / "runs" / "e2e-lone-1"
    start = _now()
    with analyst_server(run_dir / "home") as client:
        driver = PreviewingDriver(client)
        drive = _inference_drive(driver)
        pid = drive.pid
        drive.decide({"kind": "select_models", "models": ["linear"]})
        wait_for(driver, pid, {"fit": "fresh"}, timeout=400)
        posted = driver.post(f"/api/projects/{pid}/decisions",
                             json={"kind": "set_sensitivity", "analyses": []})
        assert posted.status_code == 200, posted.text
        assert press(driver, pid)["locked"] is True
        wait_for(driver, pid, {"fit": "fresh", "mattered": "fresh"}, timeout=400)
        assert driver.get(f"/api/projects/{pid}/results").status_code == 200
        assert driver.get(f"/api/projects/{pid}/export").status_code == 200
        finish(run_dir, driver, pid, start, {"id": "nhanes", "control": "known_answer",
                                             "fixture": NHANES_TRUTH, "known": KNOWN})
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        decoy, decoy_ok = A.assemble(run_dir, meta={"dataset": {"id": "nhanes-decoy",
                                                                "control": "null_decoy"}},
                                     client=client, corpus=folder / "decoy")
        records = client.app.state.service.log(pid).records()
        locked_key = client.app.state.service._locked_fit(pid, records,
                                                          plan_lock.current_lock(records))
        served = {name: client.get(f"/api/projects/{pid}{route}").json() for name, route in (
            ("view", ""), ("fits", "/fits"), ("mattered", "/mattered"))}
        in_force = _server_in_force(client, pid)
    assert ok, path.read_text("utf-8")[:2000]
    assert decoy_ok, decoy.read_text("utf-8")[:2000]
    return {"rec": validate(json.loads(path.read_text("utf-8"))),
            "decoy": validate(json.loads(decoy.read_text("utf-8"))), "served": served,
            "locked_key": locked_key, "in_force": in_force}


def test_on_the_server_a_run_that_previewed_only_its_own_answers_is_single_pass(lone_e2e):
    """DB-V4 (3): each adjustment group was previewed with its own answer, which is no
    alternative, while the group before it stood; nothing else was previewed, and the run fit
    once: a single pass. The Estimate run above, which previewed dropping its rule, is not."""
    rec = lone_e2e["rec"]
    groups = [d for d in rec.decisions if d.kind == "set_adjustment"]
    assert len(groups) > 1
    assert all(e.how == "preview" and not e.alternative for d in groups for e in d.examined)
    assert all(d.examined for d in groups)  # each previewed its own answer
    got = C.single_pass(rec)
    assert got.outcome == "fail", got.notes
    assert got.problems == ["Single pass: before the seal's opening or the plan's lock, no "
                            "decision had an alternative examined, and the run made 1 fit."]


def test_on_the_server_each_adjustment_group_is_credited_with_its_own_choices(lone_e2e):
    """DB-V4 (1, 2): the server holds every group in force; each carries its own covariates'
    departures, and the assembler's record in force for each question is the server's own."""
    rec, served, in_force = lone_e2e["rec"], lone_e2e["served"], lone_e2e["in_force"]
    _adjustment_choices_hold(rec, served["mattered"]["curve"], in_force)
    held = set(A.in_force_of(served["view"]["decisions"]).values())
    for d in rec.decisions:
        if d.kind in ("set_adjustment", "set_exclusions"):
            assert (d.seq in held) == (d.seq in in_force), d.seq


def test_on_the_server_the_locked_fit_is_the_result_whatever_its_plans_digest(lone_e2e):
    """DRV-INT (DB's [next, engine] line): the lock's digest is no fit record's (the sensitivity
    answer came after the fit), but GET …/fits marks the fit the lock was recorded on, the one the
    server's own rule names (``_locked_fit``, its key), and the result is that fit's estimate.
    Before, the record could not identify it and the known answer and the decoy went unjudged.
    Now they are judged: the known answer 9.0 (constructed: no interval near it) fails, and the
    decoy check reads the served interval; seal integrity passes."""
    rec, decoy, served = lone_e2e["rec"], lone_e2e["decoy"], lone_e2e["served"]
    lock = next(r for r in served["view"]["decisions"] if r["decision"]["kind"] == "lock_plan")
    fits = served["fits"]["records"]
    assert all(f["plan_digest"] != lock["decision"]["digest"] for f in fits)
    [locked] = [f for f in fits if f["locked"]]
    assert locked["fit_key"] == lone_e2e["locked_key"]  # the server's own rule names it
    rows = {f: rows for f, rows in locked["effects"].items() if rows}
    assert any(row["estimate"] is not None for rs in rows.values() for row in rs)
    r = rec.result
    assert (r.refused, r.lock, r.plan_digest) == (None, lock["seq"], lock["decision"]["digest"])
    assert (r.attempt, r.decisions) == (locked["id"], locked["decisions"])
    assert [e.model_dump() for e in r.estimates] == _reported(rows)
    known = C.known_answer(rec)
    assert known.outcome == "fail" and all("excludes the known answer 9.0" in p
                                           for p in known.problems), known
    excludes = any(row["ci_low"] > 0 or row["ci_high"] < 0 for rs in rows.values() for row in rs
                   if row["estimate"] is not None)
    assert C.null_decoy(decoy).outcome == ("fail" if excludes else "pass")
    assert C.seal_integrity(rec).outcome == "pass"


# ── end to end: a revert the fit before read, with its revert (DB-V6) ─────────


@pytest.fixture(scope="module")
def chain_e2e(tmp_path_factory) -> dict[str, Any]:
    """``clinical_risk.csv`` under Predict in analyst mode, every record posted with a note: after
    the first fit, sodium excluded, albumin excluded, the sodium record reverted and that revert
    reverted, posted one after another; once the fit is fresh, the second revert reverted (which
    withdraws sodium again), and the fit fresh again."""
    from turbotab.server.tests.conftest import answer_settled, wait_for

    folder = tmp_path_factory.mktemp("swarm_e2e_chain")
    run_dir = folder / "runs" / "e2e-chain-1"
    start = _now()
    seqs: dict[str, int] = {}
    with analyst_server(run_dir / "home") as client:
        made = client.post("/api/projects", json={"path": str(SAMPLE)})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        driver = ScriptedDriver(client)
        _first_fit(driver, pid)
        wait_for(driver, pid, {"fit": "fresh"}, timeout=600)

        def revert(name: str, of: str) -> None:
            r = driver.post(f"/api/projects/{pid}/decisions",
                            json={"kind": "revert", "decision_id": f"{ids[seqs[of]]}"})
            assert r.status_code == 200, r.text
            seqs[name] = _seq_of(driver)
            ids.update({d["seq"]: d["id"] for d in r.json()["decisions"]})

        for name, body in (("sodium", ROLE), ("albumin", ALBUMIN)):
            assert answer_settled(driver, pid, None, body).status_code == 200, body
            seqs[name] = _seq_of(driver)
        ids = {d["seq"]: d["id"] for d in client.get(f"/api/projects/{pid}").json()["decisions"]}
        revert("back", "sodium")
        revert("out", "back")
        wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
        revert("back again", "out")
        wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
        finish(run_dir, driver, pid, start, {"id": "clinical-risk", "fixture": FIXTURE})
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        fits = client.get(f"/api/projects/{pid}/fits").json()["records"]
        log = client.get(f"/api/projects/{pid}").json()["decisions"]
    assert ok, path.read_text("utf-8")[:2000]
    return {"rec": validate(json.loads(path.read_text("utf-8"))), "fits": fits, "seqs": seqs,
            "log": log, "notes": driver.notes}


def test_on_the_server_a_shift_is_credited_to_the_revert_the_fit_before_had_not_read(chain_e2e):
    """The fit after the last revert withdrew sodium; the record credits it to that revert where
    the two reverts before it were appended by the time the fit before completed (the fit before
    read them), and to no record where they were not (three reverts then made it together):
    never to the first revert, and no measure but the credited revert's own."""
    rec, fits, seqs, log, notes = (chain_e2e[k] for k in ("rec", "fits", "seqs", "log", "notes"))
    sodium = seqs["sodium"]
    at = {r["seq"]: datetime.fromisoformat(r["at"]) for r in log}
    last = next(i for i in range(len(fits) - 1, 0, -1)
                if sodium in fits[i - 1]["decisions"] and sodium not in fits[i]["decisions"])
    attempt = rec.attempts[last]
    shift = next(w for w in attempt.withdrawn if w.seq == sodium)
    assert shift.by != seqs["back"]
    read_before = all(at[seqs[n]] <= datetime.fromisoformat(fits[last - 1]["at"])
                      for n in ("back", "out"))
    by_seq = {d.seq: d for d in rec.decisions}
    if read_before:  # the verifier's run
        assert (shift.by, attempt.changed) == (seqs["back again"], [seqs["back again"]])
        assert [(r.seq, r.note) for r in attempt.reasons] == [
            (seqs["back again"], notes[seqs["back again"]])]
        m = by_seq[seqs["back again"]].materiality
        assert (m.source, m.attempt, m.movement) == ("fits", fits[last]["id"],
                                                     fits[last]["movement"])
    else:
        assert shift.by is None and by_seq[seqs["back again"]].materiality.source is None
    assert by_seq[seqs["back"]].materiality.attempt != fits[last]["id"]


# ── end to end: two reverts between fits restore an answer and withdraw it (DRV-INT) ──


@pytest.fixture(scope="module")
def restore_e2e(tmp_path_factory) -> dict[str, Any]:
    """``clinical_risk.csv`` under Predict in analyst mode, every record posted with a note: after
    the first fit, a roles answer with albumin left out ("A", a fit), the elastic net added (a
    fit), a roles answer with hemoglobin left out in A's place ("B", the fit before); then B
    reverted, which brings A back (a plan fitted already, so no new fit record), and A reverted,
    posted one after the other: the last fit has the first fit's roles and both families, a plan
    no fit held before, on the same rows and folds. DRV-B's verifier's probe on the server."""
    from turbotab.server.tests.conftest import answer_settled, wait_for

    folder = tmp_path_factory.mktemp("swarm_e2e_restore")
    run_dir = folder / "runs" / "e2e-restore-1"
    start = _now()
    seqs: dict[str, int] = {}
    with analyst_server(run_dir / "home") as client:
        made = client.post("/api/projects", json={"path": str(SAMPLE)})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        driver = ScriptedDriver(client)
        _first_fit(driver, pid)
        roles = client.get(f"/api/projects/{pid}").json()["state"]["roles"]
        assert "excluded" not in (roles["albumin_g_dl"], roles["hemoglobin_g_dl"])  # predictors
        a = {"kind": "set_roles", "roles": {**roles, "albumin_g_dl": "excluded"}}
        b = {"kind": "set_roles", "roles": {**roles, "hemoglobin_g_dl": "excluded"}}
        both = {"kind": "select_models", "models": ["linear", "elastic_net"]}
        for name, body in (("A", a), ("both", both), ("B", b)):
            assert answer_settled(driver, pid, None, body).status_code == 200, body
            seqs[name] = max(s for s, n in driver.notes.items() if body["kind"] in n)
            wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
        ids = {d["seq"]: d["id"] for d in client.get(f"/api/projects/{pid}").json()["decisions"]}
        for name, of in (("B back", "B"), ("A back", "A")):
            r = driver.post(f"/api/projects/{pid}/decisions",
                            json={"kind": "revert", "decision_id": ids[seqs[of]]})
            assert r.status_code == 200, r.text
            seqs[name] = _seq_of(driver)
        wait_for(driver, pid, {"fit": "fresh"}, timeout=600)
        finish(run_dir, driver, pid, start, {"id": "clinical-risk", "fixture": FIXTURE})
        path, ok = A.assemble(run_dir, client=client, corpus=folder / "corpus")
        fits = client.get(f"/api/projects/{pid}/fits").json()["records"]
        in_force = _server_in_force(client, pid)
    assert ok, path.read_text("utf-8")[:2000]
    return {"rec": validate(json.loads(path.read_text("utf-8"))), "fits": fits, "seqs": seqs,
            "in_force": in_force}


def test_on_the_server_two_reverts_that_restore_and_withdraw_an_answer_share_the_change(
        restore_e2e):
    """By construction the last fit holds neither roles answer, and the fit before it B: the
    revert of B alone would have left A in force, so it is not credited with the last fit's
    change. Both reverts moved together, and neither has the movement GET …/fits serves for that
    fit (a number: same rows, same folds, the linear model's predictors changed)."""
    rec, fits, seqs = (restore_e2e[k] for k in ("rec", "fits", "seqs"))
    last, before = fits[-1], fits[-2]
    assert {seqs["B"], seqs["both"]} <= set(before["decisions"])
    assert not {seqs["A"], seqs["B"]} & set(last["decisions"])
    assert seqs["both"] in last["decisions"]
    assert set(last["decisions"]) == restore_e2e["in_force"]
    attempt = rec.attempts[-1]
    assert attempt.changed == sorted([seqs["B back"], seqs["A back"]])
    linear = next(m for m in last["movement"] if m["family"] == "linear")
    assert linear["movement"] is not None  # served, and credited to neither revert alone
    by_seq = {d.seq: d for d in rec.decisions}
    for mine, other in (("B back", "A back"), ("A back", "B back")):
        m = by_seq[seqs[mine]].materiality
        assert (m.source, m.attempt) == (None, last["id"])
        assert m.why.startswith(f"It took effect at fit #{attempt.seq} together with record "
                                f"#{seqs[other]}:")

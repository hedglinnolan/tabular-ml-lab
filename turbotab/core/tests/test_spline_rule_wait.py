"""Harrell's rule waits for the rows it reads under prediction as under inference (the ruling of
2026-10-23 on the verifier's probe at d07a7ef5: under prediction a spline posted right after an
answer that sent the rows to be read again recorded the rule's default k, "no effective sample
size was read", where the rows read a moment later give another, so its k hung on how quickly the
post landed).

* The decision context alone (``decisions.validate`` with a context that reads rows, as the
  server's does): under prediction, with the outcome and its task answered and the analyzed rows,
  or the outcome's task, still being read, the declaration is refused ``not_yet`` with the spline
  at a k given as the ways forward, as under inference; the rows read, it records k by the rule on
  their size (counted here with pandas, k from Harrell's text). Where the outcome, or its task, is
  not answered (a matter of the answers' order), or the rows read give no size (a yes/no outcome
  in words, its event level not answered), the rule's default stands under prediction, said so,
  and under inference the refusal says what it waits for, never that the rows are being read;
  rows whose counting failed, was stopped or waits on an answer are said to be not counted, and
  the rule waits for them under either purpose.
* The engine (short reference journeys, ``turbotab.core.reference.journeys``, on the table
  ``test_replay_spline_rule`` writes, whose complete cases are known by construction): under
  prediction a spline of ``age`` with no k posted right after the reading of ``age`` is confirmed
  (which rebuilds the working table) records k by the rule on the complete cases, the record the
  same as that of the post made once the rows are read, on every run; with the rows under 45
  excluded after it, the record keeps that k and size. A spline posted before the outcome is
  answered records the default, said so. Under inference the same post records the rule's k on
  the complete cases, as before.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.tests import modeling_fixtures as mf

RULE = "Harrell's rule (3 knots below an effective sample size of 30, 5 from 100, else 4"


def harrell(n: float) -> int:
    """Harrell's rule as RMS §2.4.6 states it: 3 knots below an effective sample size of 30, 5
    from 100, else 4."""
    return 3 if n < 30 else 5 if n >= 100 else 4


class _Store:
    """A store a decision context reads (the server's ``DataStore`` shape)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns: list[str], rows: Any = None) -> pd.DataFrame:
        out = self.frame[list(columns)]
        return out if rows is None else out.iloc[np.asarray(rows, dtype=int)]

    def whole_numbers(self, columns: list[str]) -> dict[str, Any]:
        return {}

    def text_numbers(self, columns: list[str]) -> dict[str, Any]:
        return {}


def _frame(n: int = 140, seed: int = 4) -> pd.DataFrame:
    """``y`` numeric with about 15% blank, ``x`` a skewed amount (no whole numbers to read)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    y[rng.random(n) < 0.15] = np.nan
    return pd.DataFrame({"pid": np.arange(n), "x": rng.gamma(3.0, 2.0, n),
                         "z": rng.normal(size=n), "y": y})


ROLES = {"pid": "identifier", "x": "covariate", "z": "covariate"}


def _state(purpose: str, **slots: Any) -> d.ProjectState:
    base: dict[str, Any] = dict(roles=dict(ROLES), role_confirmations=dict(ROLES), target="y",
                                task="regression", purpose=purpose, lens=["clinical"])
    if purpose == "inference":
        base["roles"] = {**ROLES, "x": "exposure"}
        base["role_confirmations"] = dict(base["roles"])
        base["estimand"] = d.EstimandSpec(exposure="x", measure="mean_difference")
    return mf.state(**{**base, **slots})


def _ctx(state: d.ProjectState, frame: pd.DataFrame, rows: Any, **stages: str) -> dict[str, Any]:
    """A context that reads rows: ``rows`` the analyzed rows (None while they are being read),
    ``stages`` a stage's status (``"cancelled"`` for a stage stopped by request)."""
    def stage(name: str) -> Any:
        status = stages.get(name)
        if status is None:
            return None
        if status == "cancelled":
            return SimpleNamespace(status="stale", cancelled=True)
        return SimpleNamespace(status=status, cancelled=False)

    return {"state": state, "store": lambda: _Store(frame), "analyzed": lambda: rows,
            "stage": stage}


SPLINE = {"kind": "set_exposure_form", "column": "x", "form": "spline"}


def _refused(ctx: dict[str, Any]) -> d.Refusal:
    with pytest.raises(d.Refusal) as refused:
        d.validate(SPLINE, ctx)
    return refused.value


def _k_given(refusal: d.Refusal) -> None:
    """The ways forward: the spline at each k given, none of them by the rule."""
    assert [e["decision"]["knots"] for e in refusal.exits] == [3, 4, 5]
    assert [e["label"] for e in refusal.exits] == [
        f"A spline of `x` with {k} knots, k given" for k in (3, 4, 5)]


BEING_READ = ("k by Harrell's rule reads the effective sample size of the analyzed rows, and they "
              "are being read for the answers just given; declare `x`'s spline again in a moment, "
              "or give its k.")


@pytest.mark.parametrize("purpose", ["prediction", None])
@pytest.mark.parametrize("stages", [{"cohort": "running"}, {"cohort": "stale"}, {}])
def test_under_prediction_the_rule_waits_while_the_analyzed_rows_are_being_read(stages, purpose):
    """The outcome and its task answered, the cohort being counted again (or a context that
    states no stages): refused ``not_yet``, the words today's under inference, the spline at a k
    given the ways forward; never the rule's default in its place. So too before the purpose is
    answered, where the rule stated its default as under prediction."""
    frame = _frame()
    refusal = _refused(_ctx(_state(purpose), frame, None, **stages))
    assert refusal.code == "not_yet" and refusal.message == BEING_READ
    _k_given(refusal)
    taken = d.validate(refusal.exits[2]["decision"], _ctx(_state("prediction"), frame, None))
    assert (taken.knots, taken.knots_rule, taken.n_effective) == (5, None, None)


@pytest.mark.parametrize("purpose", ["prediction", "inference", None])
@pytest.mark.parametrize("rows", [slice(0, 140), slice(0, 90), slice(0, 31)])
def test_once_the_rows_are_read_the_rule_reads_their_size(purpose, rows):
    """The same declaration once the rows are read: k by the rule on the analyzed rows' outcome
    (its recorded values counted with pandas), under either purpose."""
    frame = _frame()
    ids = np.arange(len(frame))[rows]
    n = int(frame["y"].iloc[ids].notna().sum())
    done = d.validate(SPLINE, _ctx(_state(purpose), frame, ids, cohort="fresh"))
    assert (done.knots, done.knots_rule, done.n_effective) == (harrell(n), "harrell", n)


def test_under_prediction_the_rule_waits_while_the_outcomes_task_is_being_detected():
    """The outcome answered, its task not answered and the target stage reading it: refused
    ``not_yet`` as while the rows are read (the rows wait on that reading too)."""
    refusal = _refused(_ctx(_state("prediction", task=None), _frame(), None,
                            target_info="running", cohort="idle"))
    assert refusal.code == "not_yet" and refusal.message == BEING_READ
    _k_given(refusal)


def test_under_prediction_before_the_outcome_is_answered_the_default_stands_said_so():
    """No outcome answered: the rule reads nothing, and its default is recorded, said so."""
    from turbotab.core.voice import sentence_for

    st = _state("prediction", target=None, task=None)
    done = d.validate(SPLINE, _ctx(st, _frame(), None, target_info="blocked", cohort="blocked"))
    assert (done.knots, done.knots_rule, done.n_effective) == (4, "harrell", None)
    said = sentence_for(done, st)
    assert f"k = 4, the default of {RULE}; no effective sample size was read)" in said
    assert "4 by Harrell's rule" not in said


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_under_prediction_a_task_neither_answered_nor_being_detected_keeps_the_default(status):
    """The outcome answered, its task not answered and the target stage not reading it (it
    failed, or was stopped): the default stands, said so, as before."""
    done = d.validate(SPLINE, _ctx(_state("prediction", task=None), _frame(), None,
                                   target_info=status, cohort="idle"))
    assert (done.knots, done.knots_rule, done.n_effective) == (4, "harrell", None)


def test_under_inference_before_the_outcome_is_answered_the_refusal_says_so():
    """Under inference the rule waits as before, and says what it waits for: no outcome chosen,
    never "being read for the answers just given"."""
    st = _state("inference", target=None, task=None)
    refusal = _refused(_ctx(st, _frame(), None, target_info="blocked", cohort="blocked"))
    assert refusal.code == "not_yet"
    assert refusal.message == (
        "k by Harrell's rule reads the effective sample size of the analyzed rows' outcome, and "
        "no outcome is chosen yet; choose it, then declare `x`'s spline again, or give its k.")
    _k_given(refusal)


def test_under_inference_a_task_neither_answered_nor_being_detected_is_said_so():
    refusal = _refused(_ctx(_state("inference", task=None), _frame(), None,
                            target_info="error", cohort="idle"))
    assert refusal.code == "not_yet"
    assert refusal.message == (
        "k by Harrell's rule reads the effective sample size of the analyzed rows, which the "
        "outcome's task decides, and the task of `y` is not answered yet; answer it, then declare "
        "`x`'s spline again, or give its k.")
    _k_given(refusal)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
@pytest.mark.parametrize("status, why", [
    ("error", "the step that counts them failed"),
    ("cancelled", "the step that counts them was stopped"),
    ("blocked", "the step that counts them waits on an earlier answer")])
def test_rows_whose_counting_failed_are_said_to_be_not_counted(purpose, status, why):
    """The outcome and its task answered, the cohort not being counted for the answers as they
    stand: the rule waits under either purpose, and its words say why, never "being read"."""
    refusal = _refused(_ctx(_state(purpose), _frame(), None, target_info="fresh", cohort=status))
    assert refusal.code == "not_yet"
    assert refusal.message == (
        f"k by Harrell's rule reads the effective sample size of the analyzed rows, and they are "
        f"not being counted for the answers as they stand ({why}); give `x`'s spline its k, or "
        f"declare it again once they are counted.")
    _k_given(refusal)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_rows_read_whose_outcome_gives_no_size_keep_the_default_or_say_why(purpose):
    """The rows read, a yes/no outcome recorded as words and its event level not answered: the
    effective sample size (the smaller of the events and the non-events) cannot be counted. Under
    prediction the default stands, said so, as before; under inference the refusal says why,
    never that the rows are being read."""
    frame = _frame().assign(y=lambda f: np.where(f["z"] > 0, "yes", "no"))
    ctx = _ctx(_state(purpose, task="binary", event=None), frame, np.arange(len(frame)),
               target_info="fresh", cohort="fresh")
    if purpose == "prediction":
        done = d.validate(SPLINE, ctx)
        assert (done.knots, done.knots_rule, done.n_effective) == (4, "harrell", None)
        return
    refusal = _refused(ctx)
    assert refusal.code == "not_yet"
    assert refusal.message == (
        "k by Harrell's rule reads the effective sample size of the analyzed rows, and their "
        "outcome gives none (no value of it is recorded on them, or its event level is not "
        "answered yet); give `x`'s spline its k, or declare it again once it gives one.")
    _k_given(refusal)


def test_a_context_that_reads_no_rows_keeps_the_default_under_either_purpose():
    """A client's own context (no store): nothing is read, nothing waits, the default stated."""
    for purpose in ("prediction", "inference"):
        done = d.validate(SPLINE, {"state": _state(purpose)})
        assert (done.knots, done.knots_rule, done.n_effective) == (4, "harrell", None)


def test_a_k_given_never_waits():
    for purpose in ("prediction", "inference"):
        done = d.validate({**SPLINE, "knots": 3}, _ctx(_state(purpose), _frame(), None,
                                                        cohort="running"))
        assert (done.knots, done.knots_rule, done.n_effective) == (3, None, None)


# ── the engine: journeys ─────────────────────────────────────────────────────


def _right_after_the_reading(*after: dict[str, Any]):
    """``age`` read as an amount (a form waits for that reading; the answer rebuilds the working
    table, so every stage the rule reads is computed again), then at once, waiting for nothing, a
    spline of ``age`` with no k given; then ``after``."""
    def go(run, view) -> None:
        from turbotab.core.reference import journeys as J

        J.post_answer(run, {"kind": "confirm_reading", "reading": "code_or_count",
                            "column": "age", "value": "amount"})
        J.post_answer(run, {"kind": "set_exposure_form", "column": "age", "form": "spline"})
        for body in after:
            J.post_answer(run, body)
    return go


def _age_forms(path: Path) -> list[dict]:
    """Every ``set_exposure_form`` record of ``age`` (the journeys' own posts), in order."""
    from turbotab.core.tests.test_replay_spline_rule import _files, _records

    return [r for r in _records(_files(path)["decisions.jsonl"])
            if r["decision"]["kind"] == "set_exposure_form" and r["decision"]["column"] == "age"]


def _form(record: dict) -> dict:
    return {k: v for k, v in record["decision"].items() if k not in ("kind", "column")}


def _rule_words(k: int, n: int) -> str:
    return f"k = {k} by {RULE}; here {n:,}, the number of analyzed rows)"


@pytest.fixture(scope="module")
def waited(tmp_path_factory):
    """The Predict journey whose spline is posted once the rows are read (as
    ``test_replay_spline_rule``'s), at the recorded seed: what the record is when the post lands
    after the reading, whatever the timing."""
    from turbotab.core.tests import test_replay_spline_rule as R

    folder = tmp_path_factory.mktemp("waited")
    return R._journey(folder, "waited", "prediction", answers=dict(R.PREDICT),
                      before={"models": R._spline_once_the_rows_are_read(
                          "target_info", "working", "cohort", "split")})


@pytest.mark.slow
def test_a_predict_spline_posted_once_the_rows_are_read_records_the_rules_k_as_before(waited):
    """Under prediction, posted once the rows are read: k = 5 by the rule on the 120 complete cases
    (pandas; RMS's rule), its words in the record's sentence, as recorded before the ruling."""
    from turbotab.core.tests.test_replay_spline_rule import complete, SEED

    [record] = _age_forms(waited)
    n = len(complete(SEED))
    assert n == 120 and harrell(n) == 5
    assert (_form(record)["knots"], _form(record)["knots_rule"], _form(record)["n_effective"]) == (
        harrell(n), "harrell", float(n))
    assert _rule_words(harrell(n), n) in record["sentence"]


@pytest.mark.slow
@pytest.mark.parametrize("seed, run", [(7, 1), (7, 2), (8, 1)])
def test_a_predict_spline_posted_right_after_a_reading_waits_for_the_rows(
        waited, tmp_path, seed, run):
    """The verifier's variant C: under prediction, a spline of ``age`` with no k posted at once
    after the reading of ``age`` is confirmed. Before the ruling it recorded k = 4, the default,
    "no effective sample size was read"; now it waits for the rows and records k by the rule on
    the draw's complete cases (120 at seed 7, 130 at seed 8; pandas, RMS's rule), on every run; at
    the recorded seed its record and sentence are those of the post made once the rows are read."""
    from turbotab.core.tests import test_replay_spline_rule as R

    path = R._journey(tmp_path / "c", "c", "prediction", seed=seed, answers=dict(R.PREDICT),
                      before={"models": _right_after_the_reading()})
    [record] = _age_forms(path)
    n = len(R.complete(seed))
    assert n == R.N - R.BLANKS[seed] and harrell(n) == 5
    form = _form(record)
    assert (form["knots"], form["knots_rule"], form["n_effective"]) == (
        harrell(n), "harrell", float(n)), record["sentence"]
    assert _rule_words(harrell(n), n) in record["sentence"]
    assert "no effective sample size was read" not in record["sentence"]
    if seed == R.SEED:
        [before] = _age_forms(waited)
        assert (record["decision"], record["sentence"]) == (before["decision"], before["sentence"])


@pytest.mark.slow
def test_a_predict_spline_posted_right_after_a_reading_then_an_exclusion_keeps_the_k_read(
        tmp_path):
    """The verifier's variant D: the same, then the rows under 45 excluded. The spline's record
    states k = 5 on the 120 complete cases it waited for (pandas), not the default, and the
    exclusion (85 complete cases aged 45 and over, k = 4) leaves it as recorded."""
    from turbotab.core.tests import test_replay_spline_rule as R

    aged = {"kind": "set_exclusions", "rules": [
        {"kind": "range", "column": "age", "low": R.AGED, "reason": "45 and over"}]}
    path = R._journey(tmp_path / "d", "d", "prediction", answers=dict(R.PREDICT),
                      before={"models": _right_after_the_reading(aged)})
    [record] = _age_forms(path)
    later = [r["seq"] for r in R._records(R._files(path)["decisions.jsonl"])
             if r["decision"]["kind"] == "set_exclusions" and r["decision"]["rules"]]
    assert later and min(later) > record["seq"]
    n, kept = len(R.complete(R.SEED)), int((R.complete(R.SEED)["age"] >= R.AGED).sum())
    assert (n, kept) == (120, 85) and harrell(n) != harrell(kept)
    form = _form(record)
    assert (form["knots"], form["knots_rule"], form["n_effective"]) == (
        harrell(n), "harrell", float(n)), record["sentence"]
    assert _rule_words(harrell(n), n) in record["sentence"]


@pytest.mark.slow
def test_a_predict_spline_posted_before_the_outcome_is_answered_keeps_the_default(tmp_path):
    """The verifier's variant A: posted before the outcome is answered (a matter of the answers'
    order), the record states the rule's default, said so, as before the ruling."""
    from turbotab.core.tests import test_replay_spline_rule as R

    path = R._journey(tmp_path / "a", "a", "prediction", answers=dict(R.PREDICT),
                      before={"target": _right_after_the_reading()})
    [first] = _age_forms(path)
    form = _form(first)
    assert (form["knots"], form["knots_rule"], form["n_effective"]) == (4, "harrell", None)
    assert f"k = 4, the default of {RULE}; no effective sample size was read)" in first["sentence"]


@pytest.mark.slow
def test_an_inference_spline_posted_right_after_a_reading_records_the_rules_k_as_before(tmp_path):
    """Under inference the rule waited already: the same post records k = 5 by the rule on the 120
    complete cases (pandas; RMS's rule)."""
    from turbotab.core.tests import test_replay_spline_rule as R

    path = R._journey(tmp_path / "i", "i", "inference", exposure="walks_daily",
                      roles=dict(R.ROLES), answers={"missing": R.COMPLETE_CASES},
                      before={"models": _right_after_the_reading()})
    [record] = _age_forms(path)
    n = len(R.complete(R.SEED))
    form = _form(record)
    assert (form["knots"], form["knots_rule"], form["n_effective"]) == (
        harrell(n), "harrell", float(n)), record["sentence"]
    assert _rule_words(harrell(n), n) in record["sentence"]

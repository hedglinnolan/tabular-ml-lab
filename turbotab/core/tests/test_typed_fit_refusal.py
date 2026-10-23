"""A typed fit-time refusal (SWARM_PLAN §8; the sixth review's §3 item 2): a stage that refuses in
words carries a code and its ways forward, as an answer-time refusal does, and the stage's status
serves them apart from an error the engine did not type as a refusal, so a script classes a
refused fit without reading its prose.

Each guard both ways, on stages built here: a refusal in words is served with its code and exits
(a light stage, a heavy one in a job worker, and a stage waiting on a refused one), and so is a
reading the fit waits on (``readings.Unsettled``, with the ask's code and its own exits); a crash,
the same ``ValueError`` text with no refusal, is served with none, as is scikit-learn's own error.
The fit's own refusals are held to their construction: an empty selection (columns of a Hadamard
matrix, exactly uncorrelated with the outcome, so nothing is kept) refuses with ``nothing_kept``
and the two selections the words name; the design stage's refusals in words keep their words and
gain a code, the pipeline's among them (amounts not numbers yet, a tuned family with no per-row
loss, blanks with no missing-values answer), each with the ways forward its words name. A
selection's fit-time ways forward are the recorded answer with only its method changed (its
pre-selection answer kept), as the answer-time refusal builds them, wherever the stage ran; with
no answer recorded, as raised."""
from __future__ import annotations

import pickle
import uuid

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard

from turbotab.core import decisions as d
from turbotab.core.graph import Engine, Graph, Stage, StageRefusal, StageStatus
from turbotab.core.tests.test_graph import Projects, RecordingBus, wait_for
from turbotab.core.tests.toy_stages import ToyState

GRAPH = "turbotab.core.tests.test_typed_fit_refusal:refusing_graph"
WORDS = "The toy refuses in words: choose no selection."
EXITS = [{"label": "No selection", "decision": {"kind": "set_selection", "method": "none"}},
         {"label": "Wait for the table", "decision": None}]


def stage_refuses(ctx):
    raise StageRefusal("toy_refused", WORDS, exits=EXITS)


def stage_crashes(ctx):
    raise ValueError(WORDS)  # the same words, untyped: a crash


def stage_after(ctx):
    return {"after": ctx.inputs}


# A reading the fit waits on (``readings.Unsettled``: it asks, never guesses), with its own ways
# forward: one confirmation per answer the reading can take.
ASKED = "`age` holds whole numbers, which may be codes or amounts; the fit waits for the answer."
ASKED_EXITS = [
    {"label": "`age` is an amount", "decision": {"kind": "confirm_reading", "reading":
                                                 "code_or_count", "column": "age",
                                                 "value": "amount"}},
    {"label": "`age` holds codes", "decision": {"kind": "confirm_reading", "reading":
                                                "code_or_count", "column": "age", "value": "code"}}]


def stage_waits(ctx):
    from turbotab.core.readings import Unsettled

    raise Unsettled(ASKED, exits=ASKED_EXITS)


def refusing_graph() -> Graph:
    """R refuses (light), RH refuses in a job worker, C and CH fail with the same words untyped,
    AR waits on RH, AC on CH; U and UH wait on an unsettled reading (light, in a job worker)."""
    return Graph([
        Stage("R", 1, (), (), stage_refuses),
        Stage("RH", 1, (), (), stage_refuses, heavy=True),
        Stage("C", 1, (), (), stage_crashes),
        Stage("CH", 1, (), (), stage_crashes, heavy=True),
        Stage("AR", 1, ("RH",), (), stage_after),
        Stage("AC", 1, ("CH",), (), stage_after),
        Stage("U", 1, (), (), stage_waits),
        Stage("UH", 1, (), (), stage_waits, heavy=True),
    ])


def job_refuses() -> None:
    raise StageRefusal("toy_refused", WORDS, exits=EXITS)


def job_crashes() -> None:
    raise ValueError(WORDS)


def normalized(exits):
    """Each way forward as an answer-time refusal stores it: its decision parsed by the decision
    model and dumped as JSON (``decisions.Refusal``)."""
    return [{"label": e["label"], "decision": None if e["decision"] is None
             else d.parse_decision(e["decision"]).model_dump(mode="json")} for e in exits]


@pytest.fixture
def statuses(runner, tmp_path):
    projects, bus = Projects(tmp_path), RecordingBus()
    engine = Engine(graph_factory=GRAPH, runner=runner, bus=bus, project_ctx=projects)
    pid = uuid.uuid4().hex[:8]
    projects.states[pid] = ToyState()
    engine.on_decision(pid)
    try:
        yield wait_for(engine, pid, lambda s: all(x.status == "error" for x in s.values()),
                       timeout=30)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stage", ["R", "RH"])
def test_a_refusal_in_words_is_served_with_its_code_and_exits(statuses, stage):
    """Light or in a job worker: the error text is the one a ValueError with these words gives
    (what a person reads does not move), and the refusal is served beside it."""
    status = statuses[stage]
    assert status.error == f"ValueError: {WORDS}"
    assert status.refusal is not None
    assert status.refusal.model_dump(mode="json") == {
        "stage": stage, "code": "toy_refused", "message": WORDS, "exits": normalized(EXITS)}


@pytest.mark.parametrize("stage", ["C", "CH"])
def test_a_crash_with_the_same_words_is_served_with_no_refusal(statuses, stage):
    status = statuses[stage]
    assert status.error == f"ValueError: {WORDS}"
    assert status.refusal is None


@pytest.mark.parametrize("stage", ["U", "UH"])
def test_a_stage_waiting_on_an_unsettled_reading_is_served_refused_with_its_own_ways_forward(
        statuses, stage):
    """The verifier's V4: a reading the fit waits on (``readings.Unsettled``) is a refusal in words
    with its own ways forward, served with the answer-time ask's code (``reading_unsettled``);
    its error text is the one it had (what a person reads does not move)."""
    status = statuses[stage]
    assert status.error == f"Unsettled: {ASKED}"
    assert status.refusal is not None
    assert status.refusal.model_dump(mode="json") == {
        "stage": stage, "code": "reading_unsettled", "message": ASKED,
        "exits": normalized(ASKED_EXITS)}


def test_a_scikit_learn_value_error_is_still_served_with_no_refusal():
    """The guard's other side: an error the engine did not type, scikit-learn's own ValueError on
    an empty design, carries no refusal; nor does a ValueError that merely holds ``exits``."""
    from sklearn.linear_model import LinearRegression

    from turbotab.core.jobs import failure_of

    with pytest.raises(ValueError) as raised:
        LinearRegression().fit(np.empty((0, 1)), np.empty(0))
    text, refusal = failure_of(raised.value)
    assert text == f"ValueError: {raised.value}" and refusal is None
    lookalike = ValueError(ASKED)
    lookalike.exits = ASKED_EXITS
    assert failure_of(lookalike) == (f"ValueError: {ASKED}", None)


def test_a_stage_waiting_on_a_refused_one_serves_that_refusal_and_on_a_crash_none(statuses):
    assert statuses["AR"].error == "Needs 'RH', which failed."
    assert statuses["AR"].refusal == statuses["RH"].refusal
    assert statuses["AR"].refusal.stage == "RH"
    assert statuses["AC"].error == "Needs 'CH', which failed."
    assert statuses["AC"].refusal is None


def test_the_status_without_a_refusal_serves_none():
    assert StageStatus(stage="s", status="fresh").model_dump(mode="json")["refusal"] is None


def test_a_job_that_refuses_reports_the_refusal_and_one_that_crashes_none(runner):
    """The job pipe carries the refusal whole (its code, words and exits); a crash carries none."""
    import time

    ids = {"refuses": runner.submit(job_refuses, label="refuses"),
           "crashes": runner.submit(job_crashes, label="crashes")}
    end = time.monotonic() + 30
    views = {}
    while time.monotonic() < end and len(views) < 2:
        for name, job in ids.items():
            view = runner.get(job)
            if view.state in ("done", "error", "cancelled"):
                views[name] = view
        time.sleep(0.05)
    assert views["refuses"].state == views["crashes"].state == "error"
    assert views["refuses"].error == views["crashes"].error == f"ValueError: {WORDS}"
    assert views["refuses"].refusal == {"code": "toy_refused", "message": WORDS,
                                        "exits": normalized(EXITS)}
    assert views["crashes"].refusal is None


def test_a_refusal_is_a_value_error_with_its_words_and_pickles_whole():
    refusal = StageRefusal("toy_refused", WORDS, exits=EXITS)
    assert isinstance(refusal, ValueError) and str(refusal) == WORDS
    again = pickle.loads(pickle.dumps(refusal))
    assert (again.code, again.message, again.exits) == ("toy_refused", WORDS, normalized(EXITS))


def test_a_way_forward_that_is_no_decision_is_refused_when_raised():
    with pytest.raises(Exception):
        StageRefusal("toy_refused", WORDS, exits=[{"label": "x", "decision": {"kind": "no_such"}}])


# ── the fit's own refusals ───────────────────────────────────────────────────

N = 64


def orthogonal(carried: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Six candidates exactly uncorrelated with the outcome (columns of a Hadamard matrix); with
    ``carried``, the first carries the outcome."""
    H = hadamard(N).astype(float)
    y = pd.Series(H[:, 1], name="y")
    X = pd.DataFrame(H[:, 2:8], columns=[f"x{i}" for i in range(6)])
    if carried:
        X["x0"] = H[:, 1] + 0.5 * H[:, 9]
    return X, y


def selector(method: str, columns: list[str]):
    from turbotab.core.models.variable_selection import Selector

    return Selector(method, "regression", None, None, None, 50, 5, 0, None, columns)


@pytest.mark.parametrize("method", ["univariable", "elastic_net", "stepwise"])
def test_a_selection_that_keeps_nothing_refuses_with_nothing_kept_and_its_two_ways_forward(method):
    from turbotab.core.models.variable_selection import LABELS, nothing_kept

    X, y = orthogonal(carried=False)
    assert np.allclose(X.T.to_numpy() @ (y - y.mean()).to_numpy(), 0)  # the construction
    fitted = selector(method, list(X.columns)).fit(X, y)
    with pytest.raises(StageRefusal) as refused:
        fitted.transform(X)
    r = refused.value
    assert r.code == "nothing_kept" and str(r) == nothing_kept(method, N)  # the words as before
    # The words name No selection and in-fold screening; the exits are those two selections.
    assert r.exits == normalized([
        {"label": LABELS["none"], "decision": {"kind": "set_selection", "method": "none"}},
        {"label": LABELS["screening"], "decision": {"kind": "set_selection",
                                                    "method": "screening"}}])


@pytest.mark.parametrize("method", ["univariable", "elastic_net", "stepwise"])
def test_a_selection_that_keeps_a_predictor_refuses_nothing(method):
    X, y = orthogonal(carried=True)
    kept = selector(method, list(X.columns)).fit(X, y).transform(X)
    assert "x0" in kept.columns


def _design(tmp_path, st, rows=None, frame=None):
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = frame if frame is not None else mf.nhanes_like(60, seed=1)
    paths = mf.ingest_frame(frame, tmp_path)
    ids = np.arange(len(frame)) if rows is None else rows
    return design_stage(mf.context(st, {"split": mf.split_bundle(ids),
                                        "target_info": mf.target_info("regression")}, paths))


def test_a_design_handed_no_rows_refuses_with_a_code_and_its_words(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    with pytest.raises(StageRefusal) as refused:
        _design(tmp_path, mf.state(models=["linear"]), rows=np.array([], dtype=np.int64))
    assert refused.value.code == "no_rows" and refused.value.exits == []
    assert str(refused.value).startswith("There are no training rows to build the models on")


def test_a_design_with_no_predictor_refuses_with_a_code(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    roles = {c: ("identifier" if c == "SEQN" else "excluded") for c in mf.NHANES_ROLES}
    with pytest.raises(StageRefusal) as refused:
        _design(tmp_path, mf.state(models=["linear"], roles=roles))
    assert refused.value.code == "no_predictors"
    assert "nothing to model the outcome with" in str(refused.value)


def test_a_design_whose_outcome_takes_one_value_refuses_with_a_code(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(60, seed=1)
    frame["glucose"] = 100.0
    with pytest.raises(StageRefusal) as refused:
        _design(tmp_path, mf.state(models=["linear"]), frame=frame)
    assert refused.value.code == "one_outcome_value"
    assert "in every one of the" in str(refused.value)


def test_a_design_that_can_be_built_refuses_nothing(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    out = _design(tmp_path, mf.state(models=["linear"]))
    assert out is not None


def test_a_design_waiting_on_an_unsettled_reading_is_typed_with_its_two_ways_forward(tmp_path):
    """The verifier's V4, reproduced: with no reading confirmed, the fixture's `age` (whole years,
    by its construction) may be codes or amounts, so the design asks (``readings.Unsettled``) and
    fits nothing. The failure is typed with the ask's code and its own two ways forward, one
    confirmation per answer, its error text as before."""
    from turbotab.core.jobs import failure_of
    from turbotab.core.readings import Unsettled
    from turbotab.core.tests import modeling_fixtures as mf

    with pytest.raises(Unsettled) as asked:
        _design(tmp_path, mf.state(models=["linear"], shape_confirmations={}))
    text, refusal = failure_of(asked.value)
    assert text == f"Unsettled: {asked.value}"
    assert refusal["code"] == "reading_unsettled" and refusal["message"] == str(asked.value)
    assert [e["decision"] for e in refusal["exits"]] == [
        {"kind": "confirm_reading", "reading": "code_or_count", "column": "age", "value": value}
        for value in ("amount", "code")]
    assert [e["label"] for e in refusal["exits"]] == [e["label"] for e in asked.value.exits]


# ── the design's refusals in words from the pipeline (the verifier's V4) ─────


def _amounts_state(**slots):
    return d.ProjectState(target="y", purpose="prediction",
                          roles={"crp": "covariate", "age": "covariate"},
                          shape_confirmations={"code_or_count:crp": "amount",
                                               "code_or_count:age": "amount"}, **slots)


AMOUNTS_WORDS = ("`crp` is recorded as amounts, but some of its values are not numbers yet (values "
                 "below a detection limit, or commas that read two ways); answer how they read "
                 "before the fit.")


def test_amounts_below_a_detection_limit_refuse_with_the_limits_answers():
    """`crp` recorded as amounts holds `<0.20` ten times (the construction): the design's backstop
    keeps its words and offers the below-detection repair's two readings (half the limit, the
    limit over √2) and codes instead, as the fit's ask offers them."""
    from turbotab.core.models.pipeline import design_spec

    frame = pd.DataFrame({"crp": ["1.2", "<0.20", "3.4", "0.5"] * 10,
                          "age": np.arange(40, dtype=float)})
    with pytest.raises(StageRefusal) as refused:
        design_spec(_amounts_state(), frame, ["crp", "age"])
    r = refused.value
    assert r.code == "reading_unsettled" and str(r) == AMOUNTS_WORDS
    repairs, codes = r.exits[:2], r.exits[2:]
    assert [(e["decision"]["kind"], e["decision"]["finding_id"], e["decision"]["option"])
            for e in repairs] == [("apply_repair", "below_detection__crp", "half_limit"),
                                  ("apply_repair", "below_detection__crp", "limit_root2")]
    read = [e["decision"]["params"]["columns"]["crp"] for e in repairs]
    assert [(c["decimal"], c["thousands"]) for c in read] == [(".", False), (".", False)]
    assert [c["factor"] for c in read] == pytest.approx([0.5, 1 / np.sqrt(2)])
    assert [e["decision"] for e in codes] == [{"kind": "confirm_reading", "reading":
                                               "code_or_count", "column": "crp", "value": "code"}]


def test_amounts_whose_commas_read_two_ways_refuse_with_the_two_readings_of_the_commas():
    """Every `crp` value is `d,ddd`: a comma that splits thousands or marks decimals. The backstop
    offers the text-number repair's two readings and codes instead."""
    from turbotab.core.models.pipeline import design_spec

    frame = pd.DataFrame({"crp": ["1,234", "2,500", "3,125", "4,750"] * 10,
                          "age": np.arange(40, dtype=float)})
    with pytest.raises(StageRefusal) as refused:
        design_spec(_amounts_state(), frame, ["crp", "age"])
    r = refused.value
    assert r.code == "reading_unsettled" and str(r) == AMOUNTS_WORDS
    assert [(e["decision"]["kind"], e["decision"].get("option") or e["decision"].get("value"))
            for e in r.exits] == [("apply_repair", "thousands"), ("apply_repair", "decimal_comma"),
                                  ("confirm_reading", "code")]
    assert [e["decision"]["params"]["columns"]["crp"] for e in r.exits[:2]] == [
        {"decimal": ".", "thousands": True}, {"decimal": ",", "thousands": False}]


def test_amounts_that_are_numbers_refuse_nothing():
    from turbotab.core.models.pipeline import design_spec

    frame = pd.DataFrame({"crp": [1.2, 0.1, 3.4, 0.5] * 10, "age": np.arange(40, dtype=float)})
    assert design_spec(_amounts_state(), frame, ["crp", "age"]).numeric == ["crp", "age"]


class _Tuned:
    """A family tuned for every task (its ``tuning`` a mapping, read by ``tuning_for``)."""

    key, label, tasks = "toy_tuned", "Toy tuned model", ("regression", "time_to_event")
    tuning = {"time_to_event": object(), "regression": object()}


def test_a_tuned_family_on_an_outcome_with_no_per_row_loss_refuses_with_a_code():
    """The losses a plan tunes on are mse, log loss and the RPS (``tuning.Loss``); a time-to-event
    outcome's primary is the time-dependent Brier score (``metrics.PRIMARY``), none of them, so a
    tuned family cannot be fit. The words name no way forward, and none is offered."""
    from typing import get_args

    from turbotab.core.models.metrics import PRIMARY
    from turbotab.core.models.pipeline import make_plans
    from turbotab.core.models.tuning import Loss

    assert PRIMARY["time_to_event"] not in get_args(Loss)  # the construction
    with pytest.raises(StageRefusal) as refused:
        make_plans([_Tuned()], "time_to_event", np.zeros(40))
    r = refused.value
    assert r.code == "no_per_row_loss" and r.exits == []
    assert str(r) == ("Toy tuned model is tuned on a strictly proper per-row loss, which this "
                      "outcome has none of, so it cannot be fit here.")
    assert make_plans([_Tuned()], "time_to_event", np.zeros(40), strict=False) == {}


def test_a_family_with_nothing_to_tune_refuses_no_loss():
    """The guard's other side: a family with no tuning on the same outcome makes no plan and
    refuses nothing."""
    from turbotab.core.models.pipeline import make_plans

    untuned = _Tuned()
    untuned.tuning = None
    assert make_plans([untuned], "time_to_event", np.zeros(40)) == {}


def _blank_design(purpose):
    from turbotab.core.models.pipeline import design_spec

    X = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0] * 5})
    st = d.ProjectState(target="y", purpose=purpose, roles={"x": "covariate"},
                        shape_confirmations={"code_or_count:x": "amount"})
    return design_spec(st, X, ["x"]), X


# Each purpose's answers to the missing-values question that the leash accepts as they stand
# (MODELING_SEQUENCE ruling 6, WP7): under prediction a fill in each training fold, with
# indicators or with blanks as their own level, or complete cases; under inference multiple
# imputation or complete cases (a single fill there is recorded only with its attestation).
ACCEPTED_FILLS = {
    "prediction": [{"strategy": "impute"}, {"strategy": "impute", "indicators": True},
                   {"strategy": "complete_case"},
                   {"strategy": "impute", "categorical": "missing_category"}],
    "inference": [{"strategy": "multiple_imputation"}, {"strategy": "complete_case"}],
}


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_blanks_with_no_missing_values_answer_refuse_with_the_answers_that_handle_them(purpose):
    """Five of twenty rows have a blank `x` and no missing-values strategy was chosen: the linear
    model cannot use them. The words are kept, and the ways forward are the missing-values answers
    the purpose accepts, each recorded as it stands."""
    from turbotab.core.models.base import get_family
    from turbotab.core.models.pipeline import warnings_for

    spec, X = _blank_design(purpose)
    rows = "analyzed rows" if purpose == "inference" else "training rows"
    with pytest.raises(StageRefusal) as refused:
        warnings_for(spec, X, ["linear"], {"linear": get_family("linear")}, rows_word=rows)
    r = refused.value
    assert r.code == "no_missing_strategy"
    assert str(r) == (f"5 {rows} have a missing predictor value and no missing-values strategy "
                      f"was chosen; Linear model cannot use them. Choose complete cases or "
                      f"imputation first.")
    assert [e["decision"] for e in r.exits] == [
        d.parse_decision({"kind": "set_missing", **body}).model_dump(mode="json")
        for body in ACCEPTED_FILLS[purpose]]
    state = d.ProjectState(target="y", purpose=purpose)
    for e in r.exits:
        d.validate(e["decision"], {"state": state})  # each way forward is accepted as it stands


def test_blanks_a_family_handles_or_no_blanks_refuse_nothing():
    """The guard's other side: a family that handles blanks itself, or no blank at all."""
    from turbotab.core.models.base import families, get_family
    from turbotab.core.models.pipeline import warnings_for

    spec, X = _blank_design("prediction")
    handles = next(f for f in families("regression") if f.handles_missing)
    assert warnings_for(spec, X, [handles.key], {handles.key: handles}) == []
    full = X.fillna(2.0)
    assert warnings_for(spec, full, ["linear"], {"linear": get_family("linear")}) == []


def test_a_batch_confounded_with_the_outcome_is_typed_as_the_answer_refuses_it():
    """The design's backstop says the answer-time refusal's words (``confounding_refusal``); typed,
    it carries that refusal's code and exits, found by its words among the table's columns."""
    from turbotab.core.methods.batch import confounding_refusal
    from turbotab.core.stages.modeling import _confounding_typed

    answer = confounding_refusal("plate", "y")
    typed = _confounding_typed(answer.message, ["age", "plate", "y"], "y")
    assert (typed.code, typed.message, typed.exits) == (answer.code, answer.message, answer.exits)
    assert _confounding_typed(None, ["age", "plate", "y"], "y") is None


def test_a_grouping_too_small_for_intercepts_is_typed_with_the_cluster_only_way_forward():
    from turbotab.core.stages.modeling import _small_groups_typed

    st = d.ProjectState(purpose="inference", target="y",
                        clusters=d.ClusterSpec(column="site", adjust="fixed_effects"))
    typed = _small_groups_typed("too few rows per group", st)
    assert typed.code == "incidental_parameters"
    assert typed.exits == normalized([{
        "label": "Cluster the intervals by `site` only",
        "decision": {"kind": "set_clusters", "column": "site", "adjust": "cluster_only"}}])
    # Under prediction no grouping gets intercepts: the refusal names no way forward of that kind.
    plain = _small_groups_typed("too few rows per group",
                                st.model_copy(update={"purpose": "prediction"}))
    assert plain.code == "incidental_parameters" and plain.exits == []


# ── stepwise's in-fit backstop (the verifier's P8) ───────────────────────────


def wide(rows: int, columns: int) -> tuple[pd.DataFrame, pd.Series]:
    """``columns`` candidates on ``rows`` rows, drawn here: backward elimination starts from all of
    them, which an intercept and every column can be fit to only when columns < rows − 1."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(rows, columns)), columns=[f"x{i}" for i in range(columns)])
    return X, pd.Series(X["x0"] + rng.normal(size=rows), name="y")


def answer_time_stepwise_refusal(monkeypatch, columns: int, rows: int):
    """The refusal the selection answer gives where the smallest training fold has ``rows`` rows
    for ``columns`` candidates (``_stepwise_has_rows``, its room read as given)."""
    from turbotab.core.models import variable_selection as vs

    monkeypatch.setattr(vs, "stepwise_room", lambda *a, **k: (columns, rows))
    with pytest.raises(d.Refusal) as refused:
        vs._stepwise_has_rows(d.SetSelection(method="stepwise"), None, None, "regression")
    return refused.value


def test_stepwise_with_more_columns_than_its_rows_allow_refuses_typed_as_the_answer_does(
        monkeypatch):
    """10 candidates on 8 training rows: backward elimination cannot start (10 ≥ 8 − 1). The fit
    refuses with the answer-time code and its three ways forward (elastic net, screening, no
    selection), and its words are the ones a person read before."""
    X, y = wide(8, 10)
    with pytest.raises(StageRefusal) as refused:
        selector("stepwise", list(X.columns)).fit(X, y)
    r = refused.value
    answer = answer_time_stepwise_refusal(monkeypatch, 10, 8)
    assert r.code == answer.code == "stepwise_needs_rows"
    assert [(e["label"], e["decision"]["kind"], e["decision"]["method"]) for e in r.exits] == [
        (e["label"], e["decision"]["kind"], e["decision"]["method"]) for e in answer.exits]
    assert [e["decision"]["method"] for e in r.exits] == ["elastic_net", "screening", "none"]
    assert str(r) == ("Backward elimination starts from all 10 candidate columns, and these "
                      "training rows number 8: a model with every column and an intercept needs "
                      "more rows than that. Choose the elastic net or in-fold screening in the "
                      "selection question.")
    from turbotab.core.jobs import failure_of

    text, refusal = failure_of(r)
    assert text == f"ValueError: {r}" and refusal["code"] == "stepwise_needs_rows"


def test_stepwise_with_room_for_every_column_refuses_nothing():
    """6 candidates on 64 rows (6 < 64 − 1): the backstop stays silent and the fit runs."""
    X, y = wide(64, 6)
    fitted = selector("stepwise", list(X.columns)).fit(X, y)
    assert "x0" in fitted.transform(X).columns


# ── a fit-time way forward keeps the recorded selection (the verifier's V3) ──

RECORDED = d.SelectionSpec(method="stepwise", pre_selected="no", keep=3)


def answer_time_exits(monkeypatch, recorded: d.SelectionSpec):
    """The ways forward the selection answer's own refusal gives for ``recorded`` (each the
    answer with only its method changed, ``decision.model_copy``), as a refusal stores them."""
    from turbotab.core.models import variable_selection as vs

    monkeypatch.setattr(vs, "stepwise_room", lambda *a, **k: (10, 8))
    answer = d.SetSelection(**recorded.model_dump())
    with pytest.raises(d.Refusal) as refused:
        vs._stepwise_has_rows(answer, None, None, "regression")
    return [{"label": e["label"], "decision": d.parse_decision(
        e["decision"].model_dump(mode="json") if hasattr(e["decision"], "model_dump")
        else e["decision"]).model_dump(mode="json")} for e in refused.value.exits]


def stepwise_backstop() -> StageRefusal:
    X, y = wide(8, 10)
    with pytest.raises(StageRefusal) as refused:
        selector("stepwise", list(X.columns)).fit(X, y)
    return refused.value


def test_the_stepwise_backstops_ways_forward_are_the_recorded_answers_as_the_answer_gives_them(
        monkeypatch):
    """The recorded selection answered pre_selected "no" (TRIPOD+AI 9a) and keep 3: each way
    forward on that state is that answer with only its method changed, labeled and recorded as
    the answer-time refusal's ways forward are."""
    r = stepwise_backstop().on_answers(d.ProjectState(purpose="prediction", selection=RECORDED))
    assert r.exits == answer_time_exits(monkeypatch, RECORDED)
    assert [(e["decision"]["method"], e["decision"]["pre_selected"], e["decision"]["keep"])
            for e in r.exits] == [("elastic_net", "no", 3), ("screening", "no", 3),
                                  ("none", "no", 3)]


def test_nothing_kepts_ways_forward_keep_the_recorded_answer_but_its_method():
    from turbotab.core.models.variable_selection import LABELS

    recorded = d.SelectionSpec(method="univariable", threshold=0.01, pre_selected="other_data")
    X, y = orthogonal(carried=False)
    fitted = selector("univariable", list(X.columns)).fit(X, y)
    with pytest.raises(StageRefusal) as refused:
        fitted.transform(X)
    r = refused.value.on_answers(d.ProjectState(purpose="prediction", selection=recorded))
    assert r.exits == normalized([
        {"label": LABELS[m], "decision": {**recorded.model_dump(mode="json"),
                                          "kind": "set_selection", "method": m}}
        for m in ("none", "screening")])


def test_with_no_selection_recorded_or_a_refusal_that_changes_none_the_ways_forward_stay():
    """Where the state records no selection the ways forward are as raised (the method alone, on
    the model's defaults); a refusal whose ways forward are not changes of an answer (the toy's)
    is the same on a state that records one."""
    raised = stepwise_backstop()
    before = [dict(e) for e in raised.exits]
    assert raised.on_answers(d.ProjectState(purpose="prediction")).exits == before
    assert raised.on_answers(None).exits == before
    toy = StageRefusal("toy_refused", WORDS, exits=EXITS)
    assert toy.on_answers(d.ProjectState(selection=RECORDED)).exits == normalized(EXITS)


def test_a_refusal_whose_ways_forward_were_made_the_recorded_answers_pickles_whole():
    r = stepwise_backstop().on_answers(d.ProjectState(purpose="prediction", selection=RECORDED))
    again = pickle.loads(pickle.dumps(r))
    assert (again.code, again.message, again.exits) == (r.code, r.message, r.exits)


SELECTING = "turbotab.core.tests.test_typed_fit_refusal:selecting_graph"


def stage_selects(ctx):
    X, y = wide(8, 10)
    selector("stepwise", list(X.columns)).fit(X, y)  # the backstop refuses
    return {}


def selecting_graph() -> Graph:
    """S runs the stepwise backstop on this engine's threads, SH in a job worker."""
    return Graph([Stage("S", 1, (), (), stage_selects),
                  Stage("SH", 1, (), (), stage_selects, heavy=True)])


@pytest.mark.parametrize("stage", ["S", "SH"])
def test_the_served_ways_forward_keep_the_selection_the_fit_ran_with(runner, tmp_path, stage):
    """Light or in a job worker, the status serves each way forward as the project's recorded
    selection with only its method changed; a project with none recorded is served them as
    raised."""
    projects, bus = Projects(tmp_path), RecordingBus()
    engine = Engine(graph_factory=SELECTING, runner=runner, bus=bus, project_ctx=projects)
    recorded, bare = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
    projects.states[recorded] = ToyState(purpose="prediction", selection=RECORDED)
    projects.states[bare] = ToyState(purpose="prediction")
    try:
        served = {}
        for pid in (recorded, bare):
            engine.on_decision(pid)
            served[pid] = wait_for(engine, pid, lambda s: all(x.status == "error"
                                                              for x in s.values()), timeout=30)
    finally:
        engine.shutdown()
    kept = served[recorded][stage].refusal
    assert kept is not None and kept.code == "stepwise_needs_rows"
    assert [(e["decision"]["method"], e["decision"]["pre_selected"]) for e in kept.exits] == [
        ("elastic_net", "no"), ("screening", "no"), ("none", "no")]
    raised = served[bare][stage].refusal
    assert [e["decision"]["pre_selected"] for e in raised.exits] == [None, None, None]
    assert [e["label"] for e in kept.exits] == [e["label"] for e in raised.exits]

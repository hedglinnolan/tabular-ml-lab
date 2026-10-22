"""A typed fit-time refusal (SWARM_PLAN §8; the sixth review's §3 item 2): a stage that refuses in
words carries a code and its ways forward, as an answer-time refusal does, and the stage's status
serves them apart from a crash, so a script classes a refused fit without reading its prose.

Each guard both ways, on stages built here: a refusal in words is served with its code and exits
(a light stage, a heavy one in a job worker, and a stage waiting on a refused one); a crash, the
same ``ValueError`` text with no refusal, is served with none. The fit's own refusals are held to
their construction: an empty selection (columns of a Hadamard matrix, exactly uncorrelated with
the outcome, so nothing is kept) refuses with ``nothing_kept`` and the two selections the words
name; the design stage's refusals in words keep their words and gain a code."""
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


def refusing_graph() -> Graph:
    """R refuses (light), RH refuses in a job worker, C and CH fail with the same words untyped,
    AR waits on RH, AC on CH."""
    return Graph([
        Stage("R", 1, (), (), stage_refuses),
        Stage("RH", 1, (), (), stage_refuses, heavy=True),
        Stage("C", 1, (), (), stage_crashes),
        Stage("CH", 1, (), (), stage_crashes, heavy=True),
        Stage("AR", 1, ("RH",), (), stage_after),
        Stage("AC", 1, ("CH",), (), stage_after),
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

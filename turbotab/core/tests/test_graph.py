"""The stage graph's semantics (BLUEPRINT §4). Tier A: this is what "live" means."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.events import EventBus
from turbotab.core.graph import (
    Engine,
    Graph,
    GraphError,
    ProjectContext,
    Stage,
    StageStatus,
    latest_key,
    read_artifact,
    read_meta,
    write_artifact,
)
from turbotab.core.tests import toy_stages as toy
from turbotab.core.tests.toy_stages import ToyState

TOY = "turbotab.core.tests.toy_stages:toy_graph"
HEAVY = "turbotab.core.tests.toy_stages:heavy_graph"
FAILING = "turbotab.core.tests.toy_stages:failing_graph"


class RecordingBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str, dict]] = []
        self._events_lock = threading.Lock()

    def publish(self, pid, event_type, data):
        with self._events_lock:
            self.events.append((pid, event_type, data))
        super().publish(pid, event_type, data)

    def take(self, pid: str, event_type: str) -> list[dict]:
        with self._events_lock:
            return [data for p, t, data in self.events if p == pid and t == event_type]

    def clear(self) -> None:
        with self._events_lock:
            self.events.clear()


class Projects:
    """A stand-in workspace: pid -> state and settings."""

    def __init__(self, root: Path):
        self.root = root
        self.states: dict[str, ToyState] = {}
        self.settings: dict[str, dict] = {}

    def cache(self, pid: str) -> Path:
        return self.root / pid / "cache"

    def __call__(self, pid: str) -> ProjectContext:
        return ProjectContext(
            project_id=pid,
            state=self.states[pid],
            cache_root=self.cache(pid),
            paths={},
            settings=self.settings.get(pid, {}),
            fingerprint="fp-1",
        )


@pytest.fixture
def make_engine(runner, tmp_path):
    made: list[Engine] = []

    def make(factory: str):
        projects, bus = Projects(tmp_path), RecordingBus()
        engine = Engine(graph_factory=factory, runner=runner, bus=bus, project_ctx=projects)
        made.append(engine)
        return engine, projects, bus

    yield make
    for engine in made:
        engine.shutdown()


def _pid() -> str:
    return uuid.uuid4().hex[:8]


def wait_for(engine: Engine, pid: str, predicate, timeout: float = 15.0) -> dict[str, StageStatus]:
    end = time.monotonic() + timeout
    while True:
        status = engine.status(pid)
        if predicate(status):
            return status
        if time.monotonic() > end:
            raise AssertionError({name: (s.status, s.error) for name, s in status.items()})
        time.sleep(0.02)


def all_fresh(status: dict[str, StageStatus]) -> bool:
    return all(s.status == "fresh" for s in status.values())


# ── the graph itself ─────────────────────────────────────────────────────────


def _noop(ctx):
    return {}


def test_order_is_topological_and_bad_graphs_are_refused():
    graph = toy.toy_graph()
    names = [s.name for s in graph.order()]
    assert names == ["A", "B", "C", "D"]
    assert graph.upstream("D") == {"A", "B", "C"}
    with pytest.raises(GraphError, match="unknown stage"):
        Graph([Stage("X", 1, ("nope",), (), _noop)]).order()
    with pytest.raises(GraphError, match="cycle"):
        Graph([Stage("P", 1, ("Q",), (), _noop), Stage("Q", 1, ("P",), (), _noop)]).order()
    with pytest.raises(GraphError, match="twice"):
        Graph([Stage("P", 1, (), (), _noop), Stage("P", 1, (), (), _noop)])


# ── keys ─────────────────────────────────────────────────────────────────────


def test_a_changed_slot_changes_keys_exactly_downstream(make_engine):
    engine, projects, _ = make_engine(TOY)
    pid = _pid()
    projects.states[pid] = ToyState(x="1", y="1")
    k1 = engine.keys(pid)
    assert all(k1.values())

    projects.states[pid] = ToyState(x="1", y="2")
    engine.on_decision(pid)
    k2 = engine.keys(pid)
    assert {s for s in k1 if k1[s] != k2[s]} == {"B", "D"}

    projects.states[pid] = ToyState(x="2", y="2")
    engine.on_decision(pid)
    k3 = engine.keys(pid)
    assert {s for s in k2 if k2[s] != k3[s]} == {"A", "B", "C", "D"}

    projects.states[pid] = ToyState(x="2", y="2", purpose="inference")  # read by no stage
    engine.on_decision(pid)
    assert engine.keys(pid) == k3

    projects.states[pid] = ToyState(x="1", y="1")
    engine.on_decision(pid)
    assert engine.keys(pid) == k1  # keys are a function of inputs, nothing else


def test_a_stage_reading_a_slot_the_state_lacks_is_an_error(make_engine, tmp_path):
    engine, projects, _ = make_engine(TOY)
    pid = _pid()
    from turbotab.core.decisions import ProjectState

    projects.states[pid] = ProjectState()  # has no x or y
    with pytest.raises(GraphError, match="slot 'x'"):
        engine.keys(pid)


# ── staleness, freshness, blocking ───────────────────────────────────────────


def test_after_a_change_the_old_artifact_is_served_stale_then_fresh(make_engine):
    engine, projects, bus = make_engine(TOY)
    pid = _pid()
    projects.settings[pid] = {"delay": 0.15}
    projects.states[pid] = ToyState(x="1", y="1")
    engine.on_decision(pid)
    wait_for(engine, pid, all_fresh)
    before = engine.get(pid, "B")
    assert before.fresh and before.artifact == {"value": "B(A(1),1)"}
    calls = dict(toy.CALLS)
    bus.clear()

    projects.states[pid] = ToyState(x="1", y="2")
    engine.on_decision(pid)
    during = engine.get(pid, "B")
    assert (during.fresh, during.key, during.artifact) == (False, before.key, before.artifact)
    assert during.status in {"queued", "running"}
    d_during = engine.get(pid, "D")
    assert not d_during.fresh and d_during.artifact["value"].startswith("D(B(A(1),1)")
    now = engine.status(pid)
    assert now["A"].status == now["C"].status == "fresh"

    wait_for(engine, pid, all_fresh)
    after = engine.get(pid, "B")
    assert after.fresh and after.artifact == {"value": "B(A(1),2)"} and after.key != before.key
    assert engine.get(pid, "D").artifact == {"value": "D(B(A(1),2),C(A(1)))"}

    # Only B and D ran again, and only their statuses were announced.
    ran = {name for (p, name), n in toy.CALLS.items() if p == pid and n != calls.get((p, name))}
    assert ran == {"B", "D"}
    events = bus.take(pid, "stage")
    assert {e["stage"] for e in events} == {"B", "D"}
    for name in ("B", "D"):
        statuses = [e["status"] for e in events if e["stage"] == name]
        assert statuses[0] == "stale" and statuses[-1] == "fresh", statuses
        assert "running" in statuses

    # Toggling back finds the earlier artifacts: fresh at once, nothing reruns.
    calls = dict(toy.CALLS)
    projects.states[pid] = ToyState(x="1", y="1")
    engine.on_decision(pid)
    assert all_fresh(engine.status(pid))
    assert engine.get(pid, "B").artifact == {"value": "B(A(1),1)"}
    assert {k: v for k, v in toy.CALLS.items() if k[0] == pid} == {
        k: v for k, v in calls.items() if k[0] == pid
    }


def test_blocked_stages_say_which_slots_are_missing_transitively(make_engine):
    engine, projects, _ = make_engine(TOY)
    pid = _pid()
    projects.states[pid] = ToyState()
    status = engine.status(pid)
    assert {n: (s.status, set(s.missing), s.key) for n, s in status.items()} == {
        "A": ("blocked", {"x"}, None),
        "B": ("blocked", {"x", "y"}, None),
        "C": ("blocked", {"x"}, None),
        "D": ("blocked", {"x", "y"}, None),
    }
    empty = engine.get(pid, "A")
    assert (empty.status, empty.key, empty.fresh, empty.artifact) == ("blocked", None, False, None)

    projects.states[pid] = ToyState(x="1")
    engine.on_decision(pid)
    status = wait_for(engine, pid, lambda s: s["A"].status == s["C"].status == "fresh")
    assert (status["B"].status, status["B"].missing) == ("blocked", ["y"])
    assert (status["D"].status, status["D"].missing) == ("blocked", ["y"])

    projects.states[pid] = ToyState(x="1", y="1")
    engine.on_decision(pid)
    wait_for(engine, pid, all_fresh)

    # Unsetting a slot blocks again, and the last artifact is still served, stale.
    projects.states[pid] = ToyState(y="1")
    engine.on_decision(pid)
    assert engine.status(pid)["A"].status == "blocked"
    stale = engine.get(pid, "A")
    assert (stale.fresh, stale.status, stale.artifact) == (False, "blocked", {"value": "A(1)"})


# ── heavy stages: supersession and cancellation ──────────────────────────────


def test_rapid_decisions_cancel_superseded_heavy_jobs(make_engine):
    engine, projects, bus = make_engine(HEAVY)
    pid = _pid()
    projects.settings[pid] = {"steps": 20}  # one second of cooperative sleeping
    for value in range(6):
        projects.states[pid] = ToyState(y=str(value))
        engine.on_decision(pid)
        time.sleep(0.05)
    status = wait_for(engine, pid, all_fresh, timeout=20)

    h = engine.get(pid, "H")
    assert h.fresh and h.artifact["y"] == "5" and h.artifact["partial"] is False
    assert engine.get(pid, "K").artifact == {"from_h": "5"}
    written = [p.name for p in projects.cache(pid).joinpath("H").iterdir() if p.is_dir()]
    assert written == [status["H"].key], "only the last key's artifact was computed"

    def finals() -> dict[str, str]:
        out: dict[str, str] = {}
        for event in bus.take(pid, "job"):
            out[event["job_id"]] = event["state"]
        return out

    end = time.monotonic() + 5
    while sorted(finals().values()).count("cancelled") < 5 and time.monotonic() < end:
        time.sleep(0.02)
    assert sorted(finals().values()) == ["cancelled"] * 5 + ["done"]


def test_a_job_cancelled_by_request_waits_for_ensure(make_engine, runner):
    engine, projects, _ = make_engine(HEAVY)
    pid = _pid()
    projects.settings[pid] = {"steps": 200}
    projects.states[pid] = ToyState(y="1")
    engine.on_decision(pid)
    status = wait_for(engine, pid, lambda s: s["H"].status == "running")
    runner.cancel(status["H"].job_id)
    status = wait_for(engine, pid, lambda s: s["H"].status not in {"queued", "running"})
    assert status["H"].status == "idle"
    # The status says who stopped it, and so does the stage waiting on it.
    assert status["H"].cancelled and status["K"].cancelled and status["K"].status == "idle"
    engine.on_decision(pid)  # nothing changed: the cancel is respected
    assert engine.status(pid)["H"].status == "idle"
    job_id = engine.ensure(pid, "K")  # ensuring the dependent reruns what it waits on
    status = engine.status(pid)
    assert job_id is None and status["H"].job_id is not None
    assert not status["H"].cancelled and not status["K"].cancelled
    runner.cancel(status["H"].job_id)


# ── failures stay where they happen ──────────────────────────────────────────


def test_a_failing_stage_is_an_error_and_other_branches_keep_going(make_engine):
    engine, projects, bus = make_engine(FAILING)
    pid = _pid()
    projects.states[pid] = ToyState()
    engine.on_decision(pid)
    status = wait_for(
        engine, pid, lambda s: s["E"].status == "error" and s["G"].status == "fresh", timeout=20
    )
    assert "stopped unexpectedly" in status["E"].error
    assert (status["F"].status, status["F"].error) == ("error", "ValueError: boom")
    assert (status["E2"].status, status["E2"].error) == ("error", "Needs 'E', which failed.")

    # An error is not retried behind the user's back ...
    jobs_before = len(bus.take(pid, "job"))
    engine.on_decision(pid)
    time.sleep(0.1)
    assert len(bus.take(pid, "job")) == jobs_before
    # ... but ensure() retries it.
    assert engine.ensure(pid, "E") is not None
    wait_for(engine, pid, lambda s: s["E"].status == "error", timeout=20)


# ── the cache ────────────────────────────────────────────────────────────────


def test_artifacts_round_trip_in_the_three_formats(tmp_path):
    root = tmp_path / "cache"
    data = {"a": 1, "b": [1.5, float("nan")], "n": np.int64(3), "t": ("x", "y")}
    assert write_artifact(root, "S", "k1", 2, data) == "json"
    assert read_artifact(root, "S", "k1") == {"a": 1, "b": [1.5, None], "n": 3, "t": ["x", "y"]}
    meta = read_meta(root, "S", "k1")
    assert {k: meta[k] for k in ("stage", "key", "version", "format")} == {
        "stage": "S", "key": "k1", "version": 2, "format": "json"
    }
    assert "created_at" in meta

    frame = pd.DataFrame({"__row_id": [0, 1], "v": [0.5, None]})
    assert write_artifact(root, "S", "k2", 2, frame) == "parquet"
    pd.testing.assert_frame_equal(read_artifact(root, "S", "k2"), frame)

    array = np.arange(6).reshape(2, 3)
    assert write_artifact(root, "S", "k3", 2, array) == "joblib"
    np.testing.assert_array_equal(read_artifact(root, "S", "k3"), array)

    assert latest_key(root, "S") == "k3"
    assert latest_key(root, "S", exclude="k3") in {"k1", "k2"}
    assert write_artifact(root, "S", "k1", 2, {"other": True}) == "json"  # first write wins
    assert read_artifact(root, "S", "k1")["a"] == 1


def test_status_serializes_to_the_contract_shape():
    fields = set(StageStatus(stage="s", status="idle").model_dump(mode="json"))
    assert fields == {
        "stage", "status", "key", "fresh", "missing", "error", "job_id", "progress", "updated_at",
        "cancelled",
    }

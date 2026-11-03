"""A project's stages are settled once no stage reads queued or running (PP2 scope, candidate 3,
the graph part): a stage reads fresh only once its run has finished and the stages after it are
scheduled, and an artifact a job writes after its project was removed recreates no folder.

The windows are built, not waited for: a runner that computes a heavy stage's job as a worker would
(the job writes its own artifact) and holds the engine's ``on_done`` until the test releases it,
and a ``write_artifact`` that holds a light stage's run between its write and the engine's
``_finish_run``.
"""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest

from turbotab.core import graph
from turbotab.core.events import EventBus
from turbotab.core.graph import Engine, ProjectContext, StageStatus, has_artifact, read_artifact, write_artifact
from turbotab.core.jobs import JobView
from turbotab.core.tests.toy_stages import ToyState

TOY = "turbotab.core.tests.toy_stages:toy_graph"  # A -> B, A -> C, B and C -> D; all light
HEAVY = "turbotab.core.tests.toy_stages:heavy_graph"  # H (heavy, a job) -> K (light)
BUSY = ("queued", "running")


class Projects:
    """A stand-in workspace: ``projects/<pid>/cache`` made when the project is added, as
    ``Workspace.cache_dir`` makes it, and never again."""

    def __init__(self, root: Path):
        self.root = root
        self.states: dict[str, ToyState] = {}
        self.settings: dict[str, dict] = {}

    def folder(self, pid: str) -> Path:
        return self.root / "projects" / pid

    def cache(self, pid: str) -> Path:
        return self.folder(pid) / "cache"

    def add(self, state: ToyState, settings: dict | None = None) -> str:
        pid = uuid.uuid4().hex[:8]
        self.states[pid] = state
        self.settings[pid] = dict(settings or {})
        self.cache(pid).mkdir(parents=True)
        return pid

    def __call__(self, pid: str) -> ProjectContext:
        return ProjectContext(project_id=pid, state=self.states[pid], cache_root=self.cache(pid),
                              paths={}, settings=self.settings[pid], fingerprint="fp-1")


class GatedRunner:
    """A job runner that computes each job on a thread of this process, as a worker computes it
    (``run_stage_job`` writes the artifact itself), then holds the job's ``on_done`` until
    ``release`` is set: the window between a heavy stage's write and the engine hearing it is done.
    ``hold_start`` also holds each job before it computes, until ``start`` is set."""

    def __init__(self, hold_start: bool = False):
        self.start, self.release = threading.Event(), threading.Event()
        if not hold_start:
            self.start.set()
        self.started, self.finished = threading.Event(), threading.Event()
        self.views: dict[str, JobView] = {}
        self._threads: list[threading.Thread] = []

    def submit(self, fn, *args, label, stage=None, on_progress=None, on_done=None) -> str:
        job_id = uuid.uuid4().hex
        self.views[job_id] = JobView(job_id=job_id, label=label, stage=stage, state="running")
        thread = threading.Thread(target=self._work, args=(job_id, fn, args, on_progress, on_done),
                                  daemon=True)
        self._threads.append(thread)
        thread.start()
        return job_id

    def _work(self, job_id, fn, args, on_progress, on_done) -> None:
        if on_progress is not None:
            on_progress(self.views[job_id])  # the job is running (the engine's run.started)
        self.started.set()
        self.start.wait(30)
        try:
            result, state, error = fn(*args), "done", None
        except Exception as exc:  # noqa: BLE001 - a job that raises reports an error, as a worker's does
            result, state, error = None, "error", f"{type(exc).__name__}: {exc}"
        self.views[job_id] = self.views[job_id].model_copy(update={"state": state, "error": error})
        self.finished.set()
        self.release.wait(30)
        if on_done is not None:
            on_done(self.views[job_id], result)

    def get(self, job_id: str) -> JobView:
        return self.views[job_id]

    def cancel(self, job_id: str) -> bool:
        return False

    def close(self) -> None:
        self.start.set()
        self.release.set()
        for thread in self._threads:
            thread.join(10)


@pytest.fixture
def engines():
    made: list[tuple[Engine, GatedRunner]] = []

    def make(factory: str, projects: Projects, runner: GatedRunner) -> Engine:
        engine = Engine(graph_factory=factory, runner=runner, bus=EventBus(), project_ctx=projects)
        made.append((engine, runner))
        return engine

    yield make
    for engine, runner in made:
        runner.start.set()
        runner.release.set()
        engine.shutdown()
        runner.close()


def wait_for(engine: Engine, pid: str, predicate, timeout: float = 15.0) -> dict[str, StageStatus]:
    end = time.monotonic() + timeout
    while True:
        status = engine.status(pid)
        if predicate(status):
            return status
        if time.monotonic() > end:
            raise AssertionError({name: (s.status, s.error) for name, s in status.items()})
        time.sleep(0.02)


def settled(status: dict[str, StageStatus]) -> bool:
    """What a caller that waits for a project's stages reads: no stage queued or running."""
    return not any(s.status in BUSY for s in status.values())


# ── a stage reads fresh only once the stages after it are scheduled ─────────


def test_a_heavy_stage_reads_running_until_the_engine_has_scheduled_the_stage_after_it(
        engines, tmp_path):
    projects, runner = Projects(tmp_path), GatedRunner()
    engine = engines(HEAVY, projects, runner)
    pid = projects.add(ToyState(y="1"), {"steps": 1})
    engine.on_decision(pid)
    assert runner.finished.wait(15)
    key = engine.keys(pid)["H"]
    # The window, by construction: H's job wrote its artifact; the engine has not heard it is done.
    assert has_artifact(projects.cache(pid), "H", key)
    window = engine.status(pid)
    assert (window["H"].status, window["H"].fresh, window["K"].status) == ("running", False, "idle")
    assert not settled(window)
    assert engine.get(pid, "H").fresh is False  # served as it reads: not fresh yet

    runner.release.set()
    first = wait_for(engine, pid, lambda s: s["H"].status == "fresh")
    assert first["K"].status in (*BUSY, "fresh")  # scheduled under the same lock that freed H
    done = wait_for(engine, pid, lambda s: all(v.status == "fresh" for v in s.values()))
    assert settled(done)
    assert read_artifact(projects.cache(pid), "K", done["K"].key) == {"from_h": "1"}


def test_a_light_stage_reads_running_until_the_stages_after_it_are_scheduled(
        engines, tmp_path, monkeypatch):
    written, release = threading.Event(), threading.Event()
    real = graph.write_artifact

    def held_after_writing(cache_root, stage, key, version, obj):
        fmt = real(cache_root, stage, key, version, obj)
        if stage == "A":
            written.set()
            release.wait(30)
        return fmt

    monkeypatch.setattr(graph, "write_artifact", held_after_writing)
    projects, runner = Projects(tmp_path), GatedRunner()
    engine = engines(TOY, projects, runner)
    pid = projects.add(ToyState(x="1", y="1"))
    try:
        engine.on_decision(pid)
        assert written.wait(15)
        key = engine.keys(pid)["A"]
        assert has_artifact(projects.cache(pid), "A", key)
        window = engine.status(pid)
        assert [window[s].status for s in "ABCD"] == ["running", "idle", "idle", "idle"]
        assert not settled(window)
    finally:
        release.set()
    first = wait_for(engine, pid, lambda s: s["A"].status == "fresh")
    assert first["B"].status in (*BUSY, "fresh") and first["C"].status in (*BUSY, "fresh")
    done = wait_for(engine, pid, lambda s: all(v.status == "fresh" for v in s.values()))
    assert read_artifact(projects.cache(pid), "D", done["D"].key) == {"value": "D(B(A(1),1),C(A(1)))"}


def test_an_artifact_with_no_run_in_flight_reads_fresh_at_once(engines, tmp_path):
    """The other way: a stage whose artifact exists and whose engine runs nothing for it (here a
    second engine over the same cache, as after a restart) reads fresh, and nothing is submitted."""
    projects, runner = Projects(tmp_path), GatedRunner()
    runner.release.set()
    engine = engines(HEAVY, projects, runner)
    pid = projects.add(ToyState(y="1"), {"steps": 1})
    engine.on_decision(pid)
    computed = wait_for(engine, pid, lambda s: all(v.status == "fresh" for v in s.values()))

    again = GatedRunner()
    restarted = engines(HEAVY, projects, again)
    status = restarted.status(pid)
    assert {name: (s.status, s.key) for name, s in status.items()} == {
        name: ("fresh", s.key) for name, s in computed.items()}
    assert again.views == {}


# ── a late write recreates no removed project's folder ───────────────────────


def test_write_artifact_makes_the_stage_folder_only_under_an_existing_cache_root(tmp_path):
    project = tmp_path / "projects" / "p"
    cache = project / "cache"
    cache.mkdir(parents=True)
    assert write_artifact(cache, "S", "k1", 1, {"a": 1}) == "json"  # a stage's first artifact
    assert read_artifact(cache, "S", "k1") == {"a": 1}

    shutil.rmtree(project)
    with pytest.raises(OSError):
        write_artifact(cache, "S", "k2", 1, {"a": 2})
    assert not project.exists()


def test_a_job_that_ends_after_its_project_was_removed_writes_nothing(engines, tmp_path):
    projects, runner = Projects(tmp_path), GatedRunner(hold_start=True)
    engine = engines(HEAVY, projects, runner)
    pid = projects.add(ToyState(y="1"), {"steps": 1})
    engine.on_decision(pid)
    assert runner.started.wait(15)
    shutil.rmtree(projects.folder(pid))  # the copy is removed while its job is in flight
    runner.start.set()
    runner.release.set()
    status = wait_for(engine, pid, lambda s: s["H"].status == "error")
    assert status["K"].status == "error" and status["K"].error == "Needs 'H', which failed."
    assert not projects.folder(pid).exists()

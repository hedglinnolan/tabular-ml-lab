"""The job runner: results, progress, cancellation that stops CPU, crash recovery."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from turbotab.core.jobs import JobView
from turbotab.core.tests import toy_stages as toy


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_job_returns_its_result_through_on_done(runner):
    done = threading.Event()
    seen: dict = {}

    def on_done(view: JobView, result):
        seen["view"], seen["result"] = view, result
        done.set()

    job_id = runner.submit(toy.add, 2, 3, label="add", stage="S", on_done=on_done)
    assert done.wait(15)
    assert seen["result"] == 5
    view = seen["view"]
    assert (view.job_id, view.label, view.stage, view.state, view.error) == (job_id, "add", "S", "done", None)
    assert runner.get(job_id).state == "done"


def test_progress_is_relayed_and_a_cooperative_cancel_is_quick(runner):
    updates: list[JobView] = []
    job_id = runner.submit(toy.cooperative, 200, label="coop", on_progress=updates.append)
    assert _wait_until(lambda: any(u.progress for u in updates if u.progress is not None))
    assert updates[0].state == "running"
    assert any(u.message and u.message.startswith("step") for u in updates)
    started = time.monotonic()
    assert runner.cancel(job_id) is True
    view = runner.wait(job_id, timeout=5)
    assert view.state == "cancelled"
    assert time.monotonic() - started < 1.0  # it heard the flag; nothing was killed
    assert runner.cancel(job_id) is False  # already finished


def test_cancelling_a_cpu_bound_job_that_never_listens_stops_it(runner, tmp_path: Path):
    pidfile = tmp_path / "pid"
    job_id = runner.submit(toy.spin, str(pidfile), 60.0, label="spin")
    assert _wait_until(lambda: pidfile.exists() and pidfile.read_text() != "")
    pid = int(pidfile.read_text())
    started = time.monotonic()
    runner.cancel(job_id)
    view = runner.wait(job_id, timeout=10)
    elapsed = time.monotonic() - started
    assert view.state == "cancelled"
    assert elapsed < 3.0, f"cancel took {elapsed:.2f}s"
    assert _wait_until(lambda: _gone(pid), timeout=2.0), "the worker is still alive"
    # the pool respawned the worker and keeps working
    next_id = runner.submit(toy.add, 1, 1, label="after")
    assert runner.wait(next_id, timeout=15).state == "done"


def test_a_crashing_worker_marks_the_job_error_and_the_pool_keeps_working(runner):
    crashed = runner.submit(toy.crash, label="crash")
    view = runner.wait(crashed, timeout=15)
    assert view.state == "error"
    assert "exit code 7" in (view.error or "")
    raised = runner.submit(toy.raise_value_error, label="raise")
    view = runner.wait(raised, timeout=15)
    assert view.state == "error" and view.error == "ValueError: nope"
    ok = runner.submit(toy.add, 20, 22, label="ok")
    assert runner.wait(ok, timeout=15).state == "done"


def test_a_queued_job_can_be_cancelled_before_it_starts(runner, tmp_path: Path):
    blockers = [runner.submit(toy.cooperative, 100, label=f"b{i}") for i in range(runner.workers)]
    queued = runner.submit(toy.add, 1, 2, label="queued")
    assert runner.get(queued).state == "queued"
    assert runner.cancel(queued) is True
    assert runner.get(queued).state == "cancelled"
    for b in blockers:
        runner.cancel(b)
    for b in blockers:
        assert runner.wait(b, timeout=5).state == "cancelled"


def test_a_local_function_is_refused_up_front(runner):
    def local() -> None:
        pass

    with pytest.raises(ValueError, match="top-level"):
        runner.submit(local, label="local")
    with pytest.raises(KeyError):
        runner.get("no-such-job")


def test_workers_start_on_demand_and_retire_when_idle():
    """No worker at startup; one per job needed, up to the maximum; gone after the idle timeout."""
    from turbotab.core.jobs import JobRunner

    with JobRunner(workers=2, idle_seconds=0.6) as pool:
        assert pool.worker_pids() == [None, None] and pool.live_workers == 0
        first = pool.submit(toy.add, 1, 2, label="first")
        assert pool.wait(first, timeout=20).state == "done"
        assert pool.live_workers == 1  # one job needed one worker
        pid = next(p for p in pool.worker_pids() if p is not None)
        assert _wait_until(lambda: pool.live_workers == 0, timeout=5.0), "the idle worker was not retired"
        assert _wait_until(lambda: _gone(pid), timeout=5.0), "the retired worker process is still alive"
        # Work after retirement starts a worker again.
        again = pool.submit(toy.add, 2, 2, label="again")
        assert pool.wait(again, timeout=20).state == "done"
        assert pool.live_workers == 1


def test_a_worker_with_work_is_not_retired(tmp_path: Path):
    from turbotab.core.jobs import JobRunner

    with JobRunner(workers=1, idle_seconds=0.2) as pool:
        job = pool.submit(toy.cooperative, 20, label="busy")  # longer than the idle timeout
        assert pool.wait(job, timeout=30).state == "done"
        # A retirement mid-job would have killed it and reported an error.

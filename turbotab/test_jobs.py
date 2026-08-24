"""L7's job-queue gates.

Two, and the first is the one that matters:

1. **Two concurrent jobs produce results identical to two sequential ones.**
   That is the whole reason randomness has to be explicit. `models/nn_whuber.py`,
   `utils/seed.py` and `utils/datasets.py` all seed *process* state, which is
   safe with one run at a time and a data race under a pool.
2. **Cancel does something**, and when it does not, the queue says so rather
   than reporting a stop it did not achieve (`T0-LIVE-002`).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from turbotab.jobs import Cancelled, JobQueue, JobStatus, isolated_rng

pytestmark = pytest.mark.timeout(120)


@pytest.fixture
def queue():
    q = JobQueue(max_workers=4, base_seed=42)
    yield q
    q.shutdown(wait=False)


# ── gate 1 · concurrency does not change results ─────────────────────────

def _draw(ctx, n=2000):
    """A worker that uses the generator it was given."""
    return float(ctx.rng.normal(size=n).sum())


def _draw_from_global(ctx, n=2000):
    """A worker that reaches for the global RNG, as the engine's model code does.

    Included on purpose: the queue has to survive the code it actually has, not
    the code it wishes it had.
    """
    return float(np.random.normal(size=n).sum())


@pytest.mark.parametrize("worker", [_draw, _draw_from_global],
                         ids=["explicit-generator", "global-rng"])
def test_two_concurrent_jobs_match_two_sequential_ones(queue, worker):
    """The gate.

    Sequential first, then the same two jobs launched together. Same seeds, same
    answers — otherwise "reproducible" is a claim that depends on thread timing,
    and reproducibility is a manuscript claim.
    """
    globalish = worker is _draw_from_global
    seq = []
    for seed in (7, 99):
        h = queue.submit(f"seq-{seed}", worker, seed=seed, uses_global_rng=globalish)
        seq.append(queue.wait(h.id).result)

    a = queue.submit("con-7", worker, seed=7, uses_global_rng=globalish)
    b = queue.submit("con-99", worker, seed=99, uses_global_rng=globalish)
    con = [queue.wait(a.id).result, queue.wait(b.id).result]

    assert con == seq, (
        f"concurrent results {con} differ from sequential {seq} — the jobs are "
        "sharing randomness")


def test_many_concurrent_jobs_are_each_reproducible(queue):
    """Eight jobs at once, four workers, two distinct seeds interleaved."""
    handles = [queue.submit(f"j{i}", _draw_from_global, seed=(7 if i % 2 == 0 else 99),
                            uses_global_rng=True)
               for i in range(8)]
    results = [queue.wait(h.id).result for h in handles]

    evens = {r for r, h in zip(results, handles) if h.name[1:].isdigit()
             and int(h.name[1:]) % 2 == 0}
    odds = {r for r, h in zip(results, handles) if h.name[1:].isdigit()
            and int(h.name[1:]) % 2 == 1}
    assert len(evens) == 1, f"same seed gave different answers under load: {evens}"
    assert len(odds) == 1, f"same seed gave different answers under load: {odds}"
    assert evens != odds, "different seeds gave the same answer"


def test_a_job_cannot_leave_the_global_rng_changed(queue):
    """Containment, checked directly.

    The engine's global seeding is not removed by this queue — it is contained.
    A worker that reseeds process state must not change what the next caller
    draws.
    """
    np.random.seed(12345)
    before = np.random.normal(size=5).tolist()

    np.random.seed(12345)
    h = queue.submit("reseeds-globally", lambda ctx: np.random.seed(999) or 1,
                     seed=5, uses_global_rng=True)
    queue.wait(h.id)
    after = np.random.normal(size=5).tolist()

    assert after == before, (
        "a job left the process RNG reseeded — the next job's 'reproducible' "
        "result would depend on when this one ran")


def test_isolated_rng_restores_state_even_on_failure():
    np.random.seed(4242)
    expected = np.random.normal(size=3).tolist()

    np.random.seed(4242)
    with pytest.raises(RuntimeError):
        with isolated_rng(seed=1):
            np.random.seed(7)
            raise RuntimeError("boom")
    assert np.random.normal(size=3).tolist() == expected


def test_the_seed_is_recorded_on_the_handle(queue):
    """A result has to be traceable to the randomness that produced it."""
    h = queue.wait(queue.submit("x", _draw, seed=1234).id)
    assert h.seed == 1234
    assert h.to_dict()["seed"] == 1234


# ── gate 2 · cancel actually cancels · T0-LIVE-002 ───────────────────────

def _cooperative(ctx, started: threading.Event, steps=400):
    started.set()
    for i in range(steps):
        ctx.raise_if_cancelled()
        ctx.progress(i / steps, f"step {i}")
        time.sleep(0.005)
    return "ran to completion"


def _uncooperative(ctx, started: threading.Event):
    """Never checks its token. The queue must not claim to have stopped it."""
    started.set()
    time.sleep(0.05)
    return "ignored the cancel"


def test_cancel_stops_a_cooperative_job(queue):
    started = threading.Event()
    h = queue.submit("long thing", _cooperative, started)
    assert started.wait(5), "job never started"

    queue.cancel(h.id)
    done = queue.wait(h.id, timeout=10)

    assert done.status is JobStatus.CANCELLED
    assert done.result is None
    assert done.progress < 1.0, "a cancelled job reported itself complete"


def test_a_queued_job_cancelled_before_it_starts_never_runs():
    q = JobQueue(max_workers=1, base_seed=1)
    try:
        blocker = threading.Event()
        q.submit("blocker", lambda ctx: blocker.wait(5))
        later = q.submit("later", lambda ctx: "should not run")
        q.cancel(later.id)
        blocker.set()

        done = q.wait(later.id, timeout=10)
        assert done.status is JobStatus.CANCELLED
        assert done.result is None
    finally:
        q.shutdown(wait=False)


def test_the_queue_does_not_claim_a_cancel_it_did_not_achieve(queue):
    """The honest half, and the difference from `T0-LIVE-002`.

    The existing Cancel button sets a flag nothing reads, so training runs to
    completion while the UI says it stopped. Here a job that ignores its token
    is reported `DONE`, with a message saying the work was not stopped.
    """
    started = threading.Event()
    h = queue.submit("stubborn", _uncooperative, started)
    assert started.wait(5)
    queue.cancel(h.id)

    done = queue.wait(h.id, timeout=10)
    assert done.status is JobStatus.DONE, (
        "the queue reported a cancel for a job that ran to completion")
    assert done.result == "ignored the cancel"
    assert "not stopped" in done.message


def test_progress_is_observable_while_the_job_runs(queue):
    started = threading.Event()
    h = queue.submit("long thing", _cooperative, started)
    assert started.wait(5)

    deadline = time.time() + 5
    seen = 0.0
    while time.time() < deadline:
        seen = queue.get(h.id).progress
        if 0 < seen < 1:
            break
        time.sleep(0.01)
    assert 0 < seen < 1, "progress was never observable mid-flight"
    queue.cancel(h.id)
    queue.wait(h.id, timeout=10)


def test_a_failing_job_is_reported_not_swallowed(queue):
    h = queue.wait(queue.submit("bad", lambda ctx: 1 / 0).id)
    assert h.status is JobStatus.FAILED
    assert "ZeroDivisionError" in h.error
    assert h.result is None


def test_jobs_import_without_streamlit():
    import turbotab.jobs as jobs
    assert "streamlit" not in open(jobs.__file__, encoding="utf-8").read()


# ── T0-LIVE-002: the decorative cancel is gone from Classic ──────────────

def test_classic_no_longer_offers_a_cancel_it_cannot_honor():
    """`T0-LIVE-002`, resolved by removal rather than by wiring.

    The button set `cancel_training`, which nothing read. Wiring it would still
    have overstated: Streamlit runs one script per session on one thread, so
    while the training loop runs no widget is interactive and the button cannot
    be clicked at all.

    Real cancellation lives in `turbotab.jobs`. This asserts Classic no longer
    claims it.
    """
    import re

    src = open("pages/06_Train_and_Compare.py", encoding="utf-8").read()
    # The button CALL, not the phrase: the file explains at length why the
    # button was removed, and naming a thing is not doing it.
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))

    assert not re.search(r"st\.button\([^)]*Cancel", code), (
        "the decorative cancel button is back in Classic")
    assert "cancel_training" not in code, (
        "the flag nothing reads is being set again")
    assert "cannot be interrupted once started" in src.lower(), (
        "Classic should say plainly that a run cannot be stopped")

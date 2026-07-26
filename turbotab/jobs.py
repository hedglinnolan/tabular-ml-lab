"""
turbotab.jobs — work you can watch, and stop.

`PRODUCT_VISION.md` §05 traces three of the four things Streamlit could not do
to the same absence: nothing owns long work. *"Did it break?"* is the most common
complaint about the current app and it is a direct consequence. This is the
component whose absence caused the migration.

Two properties matter more than the API:

**Explicit randomness.** Every worker is handed a `numpy.random.Generator`. It
is never expected to reach for global state, because `models/nn_whuber.py`,
`utils/seed.py` and `utils/datasets.py` all call `np.random.seed` /
`torch.manual_seed` on process state — safe with one run at a time and silently
corrupting under a worker pool (`TRANSITION_PLAN.md` §04). Reproducibility is a
manuscript claim, so this is the failure that costs most.

Global mutation cannot be removed from those modules from here, so the queue
does the next honest thing: a job that admits it touches process RNG
(``uses_global_rng=True``) runs under a queue-wide lock, inside
:func:`isolated_rng`, which snapshots and restores the global numpy and torch
state around it.

**Snapshot and restore alone is not enough, and it is worth saying why.** In a
thread pool the jobs share one process RNG, so two workers that both reseed it
interleave: A seeds, B seeds, A draws from B's stream. Restoring afterwards
tidies up but does not make either answer reproducible — measured, not assumed,
by `test_two_concurrent_jobs_match_two_sequential_ones[global-rng]`, which fails
without the lock. So such jobs are **serialized**, and that cost is the reason
to pass a `Generator` instead. Workers that use ``ctx.rng`` run fully parallel
and need none of this.

**Cancellation that means something.** `T0-LIVE-002`: the existing Cancel button
sets a flag nothing reads, so training runs to completion regardless. Here cancel
sets a `threading.Event` the worker is given, and the worker is expected to check
it. A job that ignores its token is reported as `finished`, not `cancelled` —
the queue never claims to have stopped something it did not stop.

Headless: no Streamlit, no HTTP, no global queue instance.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Cancelled(Exception):
    """Raised by a worker that noticed its cancel token."""


@dataclass
class JobHandle:
    """What the caller holds. Serializable apart from the result."""

    id: str
    name: str                       # plain language: what is happening
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0           # 0..1
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    seed: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "status": self.status.value,
            "progress": round(float(self.progress), 4), "message": self.message,
            "error": self.error, "seed": self.seed,
            "elapsed": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 3),
            "terminal": self.is_terminal,
        }


class JobContext:
    """What a worker is given: a generator, a cancel token, a progress channel."""

    def __init__(self, handle: JobHandle, rng: np.random.Generator,
                 cancel: threading.Event, lock: threading.Lock):
        self.rng = rng
        self._handle = handle
        self._cancel = cancel
        self._lock = lock

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        """The cooperative half of cancellation.

        A worker that never calls this cannot be stopped, and the queue will say
        so rather than reporting a cancel it did not achieve.
        """
        if self._cancel.is_set():
            raise Cancelled(f"{self._handle.name} was cancelled")

    def progress(self, fraction: float, message: str = "") -> None:
        with self._lock:
            self._handle.progress = max(0.0, min(1.0, float(fraction)))
            if message:
                self._handle.message = message


@contextmanager
def isolated_rng(seed: int):
    """Run a block with global RNG state contained.

    The engine still contains `np.random.seed` and `torch.manual_seed` calls on
    process state. Under a worker pool those are a data race between jobs — job
    A reseeds, job B draws, and B's "reproducible" result depends on A's timing.

    This does not fix those call sites. It makes them harmless to neighbours:
    the global state is snapshotted, seeded deterministically for this job, and
    restored on the way out.
    """
    np_state = np.random.get_state()
    torch_state = None
    try:
        import torch
        torch_state = torch.get_rng_state()
    except Exception:
        torch = None

    np.random.seed(seed % (2 ** 32 - 1))
    if torch_state is not None:
        torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)


class JobQueue:
    """A small, observable queue. One per project host; never a module global."""

    def __init__(self, max_workers: int = 4, base_seed: int = 42):
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="turbotab-job")
        self._jobs: Dict[str, JobHandle] = {}
        self._cancels: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        # Serializes jobs that touch process-global RNG. Separate from the
        # bookkeeping lock, so observing a job never waits on one running.
        self._global_rng_lock = threading.Lock()
        self._base_seed = int(base_seed)

    # ── submitting ──────────────────────────────────────────────────────────

    def submit(self, name: str, fn: Callable[..., Any], *args,
               seed: Optional[int] = None, uses_global_rng: bool = False,
               **kwargs) -> JobHandle:
        """Run `fn(ctx, *args, **kwargs)` off the caller's thread.

        `fn` receives a :class:`JobContext` first: its generator, its cancel
        token, its progress channel. The seed is recorded on the handle, so a
        result can always be traced to the randomness that produced it.

        `uses_global_rng=True` declares that the work reaches for process RNG —
        which the neural-net wrapper and the dataset helpers currently do. Those
        jobs are serialized against each other, because two of them running at
        once draw from one interleaved stream and neither result is
        reproducible. It is a correctness requirement, not caution, and the way
        out of the queue it creates is to pass `ctx.rng` down instead.
        """
        job_seed = self._base_seed if seed is None else int(seed)
        handle = JobHandle(id=uuid.uuid4().hex[:12], name=name, seed=job_seed)
        cancel = threading.Event()
        with self._lock:
            self._jobs[handle.id] = handle
            self._cancels[handle.id] = cancel

        def run():
            with self._lock:
                if cancel.is_set():
                    handle.status = JobStatus.CANCELLED
                    handle.finished_at = time.time()
                    return
                handle.status = JobStatus.RUNNING
                handle.started_at = time.time()
            ctx = JobContext(handle, np.random.default_rng(job_seed), cancel, self._lock)
            try:
                if uses_global_rng:
                    # Serialized: see the module docstring. Held for the whole
                    # job, because the interleaving happens between the seed and
                    # the draw.
                    with self._global_rng_lock, isolated_rng(job_seed):
                        value = fn(ctx, *args, **kwargs)
                else:
                    value = fn(ctx, *args, **kwargs)
            except Cancelled:
                with self._lock:
                    handle.status = JobStatus.CANCELLED
                    handle.message = "cancelled"
            except BaseException:
                with self._lock:
                    handle.status = JobStatus.FAILED
                    handle.error = traceback.format_exc(limit=6)
            else:
                with self._lock:
                    # A worker that ignored its token ran to completion. Say
                    # that, rather than reporting a cancel that did not happen.
                    handle.status = JobStatus.DONE
                    handle.result = value
                    handle.progress = 1.0
                    if cancel.is_set():
                        handle.message = ("finished before it noticed the cancel "
                                          "— the work was not stopped")
            finally:
                with self._lock:
                    handle.finished_at = time.time()

        self._pool.submit(run)
        return handle

    # ── observing ───────────────────────────────────────────────────────────

    def get(self, job_id: str) -> JobHandle:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job {job_id!r}.")
            return self._jobs[job_id]

    def list(self) -> List[JobHandle]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def wait(self, job_id: str, timeout: float = 30.0) -> JobHandle:
        deadline = time.time() + timeout
        while time.time() < deadline:
            handle = self.get(job_id)
            if handle.is_terminal:
                return handle
            time.sleep(0.005)
        raise TimeoutError(f"Job {job_id} did not finish within {timeout}s.")

    # ── stopping ────────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> JobHandle:
        """Ask a job to stop. `T0-LIVE-002`'s replacement.

        Sets the token the worker was given. A job that has not started yet is
        cancelled outright; a running job stops at its next checkpoint. Whether
        it actually stopped is visible on the handle afterwards, because a
        button that claims control it does not have is the app asserting
        something false.
        """
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job {job_id!r}.")
            handle = self._jobs[job_id]
            if handle.is_terminal:
                return handle
            self._cancels[job_id].set()
            handle.message = "cancelling…"
        return handle

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

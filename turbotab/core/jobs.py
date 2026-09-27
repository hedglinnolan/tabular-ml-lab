"""Jobs: work in persistent worker processes that can be watched and stopped
(docs/turbotab-next/BLUEPRINT.md §5).

* Workers are spawned once (``spawn`` context) and pre-import numpy, pandas and
  scikit-learn, so a job starts in milliseconds rather than seconds.
* Each worker talks to the runner over its own pipe, never a shared queue: a
  worker that is killed can only break its own channel.
* Cancel is two-step. The job's shared flag is set (``current().cancelled()``
  turns true and ``current().progress()`` raises :class:`Cancelled`); if the job
  is still running after a short grace period its worker is terminated and a
  fresh one spawned. CPU stops within about 1.5 s either way.
* A worker that dies mid-job (``os._exit``, a segfault, the OOM killer) marks
  that job ``error``; the slot is respawned and the pool keeps working.
* Callbacks (``on_progress(view)``, ``on_done(view, result)``) all run on the
  runner's single dispatcher thread, never on the caller's thread and never
  while the runner's lock is held, so a callback may call ``submit``/``cancel``.

Honesty rule carried from ``turbotab/jobs.py``: a job that ignores its cancel
flag and returns normally is reported ``done``, not ``cancelled``; the runner
never claims to have stopped something it did not stop. Jobs that were stopped
(cooperatively, by raising :class:`Cancelled`, or by termination) are
``cancelled``.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import signal
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import wait as mp_wait
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "error", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"done", "error", "cancelled"})

GRACE_SECONDS = 1.0  # cooperative window after cancel, before terminate
KILL_SECONDS = 0.5  # SIGTERM -> SIGKILL window
PROGRESS_INTERVAL = 0.1  # worker-side throttle for progress messages
KEEP_FINISHED = 1000  # finished jobs remembered for get()
MAX_FAILED_STARTS = 3  # a slot whose worker dies before ready this often is given up
PRELOAD = ("numpy", "pandas", "sklearn")


class JobView(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    job_id: str
    label: str
    stage: str | None = None
    state: JobState
    progress: float | None = None
    message: str | None = None
    error: str | None = None


class Cancelled(BaseException):
    """Raised by work that noticed its cancel flag.

    A ``BaseException`` (like ``asyncio.CancelledError``) so that a stage's own
    ``except Exception`` cannot swallow it.
    """


# ── worker side ──────────────────────────────────────────────────────────────


class JobContext:
    """What a running job reaches through :func:`current`."""

    def __init__(self, job_id: str, conn: Any, flag: Any, send_lock: threading.Lock):
        self.job_id = job_id
        self._conn = conn
        self._flag = flag
        self._send_lock = send_lock
        self._last = float("-inf")

    def cancelled(self) -> bool:
        return bool(self._flag.value)

    def progress(self, fraction: float, message: str = "") -> None:
        """Report progress; raises :class:`Cancelled` once cancel was requested."""
        if self.cancelled():
            raise Cancelled()
        now = time.monotonic()
        if now - self._last < PROGRESS_INTERVAL and fraction < 1.0:
            return
        self._last = now
        with self._send_lock:
            self._conn.send(("progress", self.job_id, float(fraction), str(message)))


_CURRENT: JobContext | None = None


def current() -> JobContext | None:
    """The job running in this worker process, or ``None`` outside a job."""
    return _CURRENT


def _worker_main(conn: Any, flag: Any, preload: tuple[str, ...]) -> None:
    global _CURRENT
    # Ctrl-C in the server's terminal reaches the whole process group; the
    # runner decides when workers stop, not the keyboard.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in preload:
        try:
            importlib.import_module(name)
        except Exception:  # an optional package missing must not kill the pool
            pass
    send_lock = threading.Lock()
    try:
        conn.send(("ready", os.getpid()))
    except OSError:
        return
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return  # the runner went away
        if message[0] == "stop":
            return
        _, job_id, payload = message
        _CURRENT = JobContext(job_id, conn, flag, send_lock)
        try:
            fn, args = pickle.loads(payload)
            reply: tuple[Any, ...] = ("done", job_id, fn(*args))
        except Cancelled:
            reply = ("cancelled", job_id)
        except BaseException as exc:  # noqa: BLE001 - everything is reported
            reply = ("error", job_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())
        finally:
            _CURRENT = None
        try:
            with send_lock:
                conn.send(reply)
        except (EOFError, OSError):
            return
        except Exception as exc:  # the result itself would not pickle
            with send_lock:
                conn.send(("error", job_id, f"The job's result could not be sent back: {exc}", ""))


# ── runner side ──────────────────────────────────────────────────────────────


class _Worker:
    def __init__(self, ctx: Any, index: int, preload: tuple[str, ...], failed_starts: int = 0):
        self.index = index
        self.conn, child = ctx.Pipe(duplex=True)
        self.flag = ctx.RawValue("b", 0)
        self.process = ctx.Process(
            target=_worker_main,
            args=(child, self.flag, preload),
            name=f"turbotab-worker-{index}",
            daemon=True,
        )
        self.process.start()
        child.close()
        self.alive = True
        self.ready = False
        self.pid: int | None = None
        self.job: _Job | None = None
        self.deadline: float | None = None  # terminate at this time (after cancel)
        self.kill_at: float | None = None  # SIGKILL at this time (after terminate)
        self.failed_starts = failed_starts


@dataclass(eq=False)
class _Job:
    id: str
    label: str
    stage: str | None
    payload: bytes
    on_progress: Callable[[JobView], Any] | None
    on_done: Callable[[JobView, Any], Any] | None
    state: str = "queued"
    progress: float | None = None
    message: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    worker: _Worker | None = None

    def view(self) -> JobView:
        return JobView(
            job_id=self.id,
            label=self.label,
            stage=self.stage,
            state=self.state,  # type: ignore[arg-type]
            progress=self.progress,
            message=self.message,
            error=self.error,
        )


def default_workers() -> int:
    """``TURBOTAB_WORKERS`` if set, else one fewer than the CPU count (at least 1)."""
    configured = os.environ.get("TURBOTAB_WORKERS", "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, (os.cpu_count() or 2) - 1)


def _check_importable(fn: Callable[..., Any]) -> None:
    if inspect.isfunction(fn) and ("<locals>" in fn.__qualname__ or fn.__name__ == "<lambda>"):
        raise ValueError(
            f"{fn.__qualname__} cannot run in a worker process: a job function "
            "must be a top-level function of an importable module"
        )


class JobRunner:
    """A fixed pool of worker processes running one job each at a time, FIFO."""

    def __init__(
        self,
        workers: int,
        *,
        preload: tuple[str, ...] = PRELOAD,
        grace: float = GRACE_SECONDS,
    ):
        if workers < 1:
            raise ValueError("a JobRunner needs at least one worker")
        self._ctx = mp.get_context("spawn")
        self._preload = tuple(preload)
        self._grace = grace
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[_Job] = deque()
        self._outbox: deque[Callable[[], Any]] = deque()
        self._closed = False
        self._wake_r, self._wake_w = self._ctx.Pipe(duplex=False)
        self._workers: list[_Worker | None] = [
            _Worker(self._ctx, i, self._preload) for i in range(workers)
        ]
        self._thread = threading.Thread(target=self._loop, name="turbotab-jobs", daemon=True)
        self._thread.start()

    # ── public API ──

    @property
    def workers(self) -> int:
        return len(self._workers)

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        label: str,
        stage: str | None = None,
        on_progress: Callable[[JobView], Any] | None = None,
        on_done: Callable[[JobView, Any], Any] | None = None,
    ) -> str:
        """Queue ``fn(*args)`` for a worker; returns the job id.

        ``fn`` must be importable by the worker (a top-level function). The
        arguments are pickled here, so an unpicklable argument fails now rather
        than on the worker.
        """
        _check_importable(fn)
        payload = pickle.dumps((fn, args), protocol=pickle.HIGHEST_PROTOCOL)
        job = _Job(uuid.uuid4().hex, label, stage, payload, on_progress, on_done)
        with self._lock:
            if self._closed:
                raise RuntimeError("the job runner has been shut down")
            self._jobs[job.id] = job
            self._queue.append(job)
            self._prune()
        self._wake()
        return job.id

    def cancel(self, job_id: str) -> bool:
        """Stop a queued or running job. False if unknown or already finished."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in TERMINAL:
                return False
            if job.state == "queued":
                try:
                    self._queue.remove(job)
                except ValueError:
                    pass
                self._finish(job, "cancelled")
            elif not job.cancel_requested:
                job.cancel_requested = True
                worker = job.worker
                if worker is not None:
                    worker.flag.value = 1
                    worker.deadline = time.monotonic() + self._grace
        self._wake()
        return True

    def get(self, job_id: str) -> JobView:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.view()

    def wait(self, job_id: str, timeout: float | None = None) -> JobView:
        """Block until the job is finished (or ``timeout``); returns its view."""
        end = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    raise KeyError(job_id)
                if job.state in TERMINAL:
                    return job.view()
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return job.view()
                self._changed.wait(remaining)

    def worker_pids(self) -> list[int | None]:
        with self._lock:
            return [w.pid if w is not None and w.alive else None for w in self._workers]

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in list(self._jobs.values()):
                if job.state not in TERMINAL:
                    job.state = "cancelled"
            self._queue.clear()
            self._outbox.clear()
            self._changed.notify_all()
        self._wake()
        self._thread.join(timeout=5)
        workers = [w for w in self._workers if w is not None and w.alive]
        for w in workers:
            if w.job is None:
                try:
                    w.conn.send(("stop",))
                except OSError:
                    pass
            else:
                w.process.terminate()
        end = time.monotonic() + 2.0
        for w in workers:
            w.process.join(timeout=max(0.0, end - time.monotonic()))
            if w.process.is_alive():
                w.process.kill()
                w.process.join(timeout=1.0)
            w.alive = False
            w.conn.close()
        self._wake_r.close()
        self._wake_w.close()

    def __enter__(self) -> "JobRunner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ── dispatcher thread ──

    def _wake(self) -> None:
        try:
            self._wake_w.send_bytes(b"!")
        except OSError:
            pass

    def _loop(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
                watch: dict[Any, tuple[str, _Worker | None]] = {self._wake_r: ("wake", None)}
                pending_deadline = False
                for w in self._workers:
                    if w is not None and w.alive:
                        watch[w.conn] = ("conn", w)
                        watch[w.process.sentinel] = ("exit", w)
                        pending_deadline = pending_deadline or w.deadline is not None
            # Everything else wakes us (pipes, exits, the wake channel); only a
            # pending cancel deadline needs a clock.
            ready = mp_wait(list(watch), timeout=0.05 if pending_deadline else 2.0)
            with self._lock:
                if self._closed:
                    return
                for obj in ready:
                    what, w = watch[obj]
                    if what == "wake":
                        self._drain_wake()
                    elif w is not None and w.alive:
                        self._drain(w)
                        if what == "exit":
                            self._on_exit(w)
                self._check_deadlines()
                self._dispatch()
                outbox = list(self._outbox)
                self._outbox.clear()
            for callback in outbox:
                try:
                    callback()
                except Exception:
                    log.exception("a job callback failed")

    def _drain_wake(self) -> None:
        try:
            while self._wake_r.poll():
                self._wake_r.recv_bytes()
        except (EOFError, OSError):
            pass

    def _drain(self, w: _Worker) -> None:
        try:
            while w.alive and w.conn.poll():
                self._handle(w, w.conn.recv())
        except (EOFError, OSError):
            self._on_exit(w)

    def _handle(self, w: _Worker, message: tuple[Any, ...]) -> None:
        tag = message[0]
        if tag == "ready":
            w.ready = True
            w.pid = message[1]
            w.failed_starts = 0
            return
        job = w.job
        if job is None or job.id != message[1]:
            return
        if tag == "progress":
            job.progress = message[2]
            job.message = message[3] or None
            if job.on_progress is not None:
                self._outbox.append(_bind(job.on_progress, job.view()))
        elif tag == "done":
            self._finish(job, "done", result=message[2])
        elif tag == "cancelled":
            self._finish(job, "cancelled")
        elif tag == "error":
            if message[3]:
                log.warning("job %s (%s) failed:\n%s", job.id, job.label, message[3])
            self._finish(job, "error", error=message[2])

    def _finish(self, job: _Job, state: str, *, result: Any = None, error: str | None = None) -> None:
        job.state = state
        job.error = error
        worker = job.worker
        job.worker = None
        if worker is not None and worker.job is job:
            worker.job = None
            worker.deadline = None
            worker.kill_at = None
        job.payload = b""
        self._changed.notify_all()
        if job.on_done is not None:
            self._outbox.append(_bind(job.on_done, job.view(), result))

    def _on_exit(self, w: _Worker) -> None:
        if not w.alive:
            return
        w.alive = False
        w.process.join(timeout=1.0)
        code = w.process.exitcode
        w.conn.close()
        job = w.job
        if job is not None:
            if job.cancel_requested:
                self._finish(job, "cancelled")
            else:
                self._finish(
                    job,
                    "error",
                    error=f"The worker process running this job stopped unexpectedly (exit code {code}).",
                )
        failed = w.failed_starts + (0 if w.ready else 1)
        if self._closed:
            return
        if failed >= MAX_FAILED_STARTS:
            log.error("worker slot %d failed to start %d times; giving it up", w.index, failed)
            self._workers[w.index] = None
            return
        self._workers[w.index] = _Worker(self._ctx, w.index, self._preload, failed)

    def _check_deadlines(self) -> None:
        now = time.monotonic()
        for w in self._workers:
            if w is None or not w.alive or w.job is None or w.deadline is None:
                continue
            if w.kill_at is None and now >= w.deadline:
                w.process.terminate()
                w.kill_at = now + KILL_SECONDS
            elif w.kill_at is not None and now >= w.kill_at:
                w.process.kill()
                w.kill_at = now + 60.0

    def _dispatch(self) -> None:
        live = [w for w in self._workers if w is not None and w.alive]
        if not live:
            while self._queue:
                self._finish(self._queue.popleft(), "error", error="No worker process could be started.")
            return
        for w in live:
            if not self._queue:
                return
            if not w.ready or w.job is not None:
                continue
            job = self._queue.popleft()
            w.flag.value = 0
            try:
                w.conn.send(("run", job.id, job.payload))
            except OSError:
                self._queue.appendleft(job)  # the exit sentinel will respawn the worker
                continue
            job.state = "running"
            job.worker = w
            w.job = job
            w.deadline = None
            w.kill_at = None
            if job.on_progress is not None:
                self._outbox.append(_bind(job.on_progress, job.view()))

    def _prune(self) -> None:
        finished = sum(1 for j in self._jobs.values() if j.state in TERMINAL)
        if finished <= KEEP_FINISHED:
            return
        for job_id in [j.id for j in self._jobs.values() if j.state in TERMINAL]:
            del self._jobs[job_id]
            finished -= 1
            if finished <= KEEP_FINISHED:
                return


def _bind(fn: Callable[..., Any], *args: Any) -> Callable[[], Any]:
    return lambda: fn(*args)

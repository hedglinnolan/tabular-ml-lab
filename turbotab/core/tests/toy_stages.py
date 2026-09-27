"""Toy stages and job functions for the runtime tests.

They live in an importable module because heavy stages and jobs run in spawned
worker processes, which rebuild the graph from ``"module:function"`` and find
job functions by reference.
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path

from turbotab.core.decisions import ProjectState
from turbotab.core.graph import Graph, Stage, StageContext
from turbotab.core.jobs import current


class ToyState(ProjectState):
    x: str | None = None
    y: str | None = None


# Light stages run on the engine's threads, in this process, so tests can count
# how often each one actually ran.
CALLS: Counter[str] = Counter()
_CALLS_LOCK = threading.Lock()


def _ran(name: str, ctx: StageContext) -> None:
    with _CALLS_LOCK:
        CALLS[(ctx.project_id, name)] += 1  # type: ignore[index]
    time.sleep(float(ctx.settings.get("delay", 0.0)))


def stage_a(ctx: StageContext) -> dict:
    _ran("A", ctx)
    return {"value": f"A({ctx.state.x})"}


def stage_b(ctx: StageContext) -> dict:
    _ran("B", ctx)
    return {"value": f"B({ctx.inputs['A']['value']},{ctx.state.y})"}


def stage_c(ctx: StageContext) -> dict:
    _ran("C", ctx)
    return {"value": f"C({ctx.inputs['A']['value']})"}


def stage_d(ctx: StageContext) -> dict:
    _ran("D", ctx)
    return {"value": f"D({ctx.inputs['B']['value']},{ctx.inputs['C']['value']})"}


def toy_graph() -> Graph:
    """A reads x; B deps A, reads y; C deps A; D deps B and C."""
    return Graph(
        [
            Stage("A", 1, (), ("x",), stage_a, requires=("x",)),
            Stage("B", 1, ("A",), ("y",), stage_b, requires=("y",)),
            Stage("C", 1, ("A",), (), stage_c),
            Stage("D", 1, ("B", "C"), (), stage_d),
        ]
    )


# ── heavy ────────────────────────────────────────────────────────────────────


def stage_h(ctx: StageContext) -> dict:
    """Sleeps in small steps, checking cancelled(). On cancel it *returns* a
    partial result: the runtime, not the stage, must refuse to cache it."""
    steps = int(ctx.settings.get("steps", 20))
    for i in range(steps):
        if ctx.cancelled():
            return {"y": ctx.state.y, "partial": True}
        time.sleep(0.05)
    return {"y": ctx.state.y, "partial": False, "pid": os.getpid()}


def stage_after_h(ctx: StageContext) -> dict:
    return {"from_h": ctx.inputs["H"]["y"]}


def heavy_graph() -> Graph:
    return Graph(
        [
            Stage("H", 1, (), ("y",), stage_h, heavy=True, requires=("y",)),
            Stage("K", 1, ("H",), (), stage_after_h),
        ]
    )


def stage_exit(ctx: StageContext) -> dict:
    os._exit(3)


def stage_boom(ctx: StageContext) -> dict:
    raise ValueError("boom")


def stage_ok(ctx: StageContext) -> dict:
    return {"ok": True}


def stage_needs_e(ctx: StageContext) -> dict:
    return {"e": ctx.inputs["E"]}


def failing_graph() -> Graph:
    """E kills its worker; F raises; G is fine; E2 needs E."""
    return Graph(
        [
            Stage("E", 1, (), (), stage_exit, heavy=True),
            Stage("F", 1, (), (), stage_boom),
            Stage("G", 1, (), (), stage_ok),
            Stage("E2", 1, ("E",), (), stage_needs_e),
        ]
    )


# ── plain job functions for the runner ───────────────────────────────────────


def add(a: int, b: int) -> int:
    return a + b


def spin(pidfile: str, seconds: float) -> str:
    """CPU-bound and deaf to cancellation."""
    Path(pidfile).write_text(str(os.getpid()))
    end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < end:
        n += 1
    return "spun"


def cooperative(steps: int) -> str:
    job = current()
    assert job is not None
    for i in range(steps):
        job.progress(i / steps, f"step {i}")  # raises Cancelled once cancelled
        time.sleep(0.05)
    return "finished"


def crash() -> None:
    os._exit(7)


def raise_value_error() -> None:
    raise ValueError("nope")

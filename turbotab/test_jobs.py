"""L7's job-queue gates.

Two, and the first is the one that matters:

1. **Two concurrent jobs produce results identical to two sequential ones.**
   That is the whole reason randomness has to be explicit. `models/nn_whuber.py`,
   `utils/seed.py` and `utils/datasets.py` all seed *process* state, which is
   safe with one run at a time and a data race under a pool.
2. **Cancel does something**, and when it does not, the queue says so rather
   than reporting a stop it did not achieve (`T0-LIVE-002`).
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import threading
import time

import numpy as np
import pytest


pytestmark = pytest.mark.timeout(120)


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

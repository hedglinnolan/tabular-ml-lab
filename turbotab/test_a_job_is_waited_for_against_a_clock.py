"""`TEST-040` — no test in this tree polls a job by counting iterations.

**The class, not the instance.** Four sites spun `for _ in range(N)` over
`/job/{id}` with no wait; two more slept but still failed with the job's own
status. All six report *the app had not answered yet* as *the app produced the
wrong answer*, which is the assertion `README.md`'s governing rule forbids —
made by the instrument instead of by the app.

It matters more than an app defect would. The number this project reports every
loop, and that every adjudication re-measures, is a suite pass count. The same
commit returned `1423 passed` on one machine and `1 failed, 1422 passed` on a
loaded one. A count that differs between two honest runs of one tree is a
record that cannot settle anything, and `TEST-030` is the same axis in the
other direction — order-dependence producing false *passes*.

So this file guards two things: that the helper actually distinguishes the two
outcomes, and that nobody reintroduces the shape it replaced.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from turbotab import jobwait as JW

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The tokens that make a `range()` loop a *job poll* rather than any other
#: bounded loop. Derived from the six sites rather than guessed: every one of
#: them fetched `/job/` and broke on `terminal`.
_POLL_MARKERS = ("/job/", '"terminal"', "['terminal']", '["terminal"]')


def _bounded_job_polls():
    """Every `for _ in range(N)` loop in the tree that polls a job.

    An AST walk rather than a grep, because the question is *does this loop
    poll a job* and a grep answers *does this text appear* — trap 5.
    """
    out = []
    for path in sorted(ROOT.rglob("test_*.py")):
        if {"venv", ".venv", "node_modules", "__pycache__"} & set(path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            it = node.iter
            if not (isinstance(it, ast.Call)
                    and getattr(it.func, "id", "") == "range"):
                continue
            body = ast.unparse(node)
            if any(m in body for m in _POLL_MARKERS):
                out.append((path.relative_to(ROOT), node.lineno))
    return out


def test_no_test_polls_a_job_by_counting_iterations():
    """The standing check. Iterations are not time.

    A bigger `N` is not the fix — `range(4000)` still elapses in milliseconds
    on a fast machine and still fails with the job's status on a slow one. It
    only moves the load at which the lie appears.
    """
    offenders = _bounded_job_polls()
    assert not offenders, (
        "these poll a job with a bounded iteration count rather than against a "
        f"deadline: {offenders}. Use `turbotab.jobwait.settle` / `settle_done` "
        "— a timeout must be a different kind of event from a wrong answer.")


def test_the_sweep_can_see_the_shape_it_is_looking_for():
    """The positive control, because everything above is an absence claim.

    A sweep that read nothing would report a clean tree it never looked at.
    This synthesizes the exact shape the check exists to catch and requires
    the detector to find it — so a broken walk fails here rather than passing
    quietly there.
    """
    src = (
        "def test_x(client, job):\n"
        "    for _ in range(200):\n"
        "        job = client.get(f'/job/{job[\"id\"]}').json()\n"
        "        if job['terminal']:\n"
        "            break\n"
    )
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
    assert len(loops) == 1
    body = ast.unparse(loops[0])
    assert any(m in body for m in _POLL_MARKERS), (
        "the marker set does not match the shape this check exists to catch, "
        "so the sweep above is vacuous")

    # And the walk reaches the tree: it must find this very file, which is the
    # only guarantee that `rglob` and the parse both worked. An assertion that
    # cannot fail is not a control.
    walked = {p for p in ROOT.rglob("test_*.py")
              if not ({"venv", ".venv", "node_modules"} & set(p.parts))}
    assert pathlib.Path(__file__).resolve() in walked, (
        "the sweep does not reach its own file, so its empty result above "
        "says nothing about the tree")


def test_a_timeout_is_not_returned_as_a_job_state():
    """The load-bearing property: `settle` never hands back a running job.

    If it did, the caller's `assert status == "done"` would fire on a timeout
    and we would be exactly back where we started, one layer down.
    """
    class NeverFinishes:
        def get(self, _url):
            class R:
                @staticmethod
                def json():
                    return {"id": "j", "status": "running", "terminal": False,
                            "progress": 0.5}
            return R()

    with pytest.raises(JW.JobDidNotSettle) as caught:
        JW.settle(NeverFinishes(), {"id": "j", "name": "Training"},
                  timeout=0.15, poll=0.01)
    message = str(caught.value)
    assert "did not finish in 0.15 seconds" in message, message
    assert "TIMEOUT, not a wrong answer" in message, (
        "the message has to say which of the two things happened; that "
        "distinction is the whole of TEST-040")
    assert "'running'" in message, "the last status belongs in the message"


def test_a_job_that_fails_is_reported_as_a_failure_not_a_timeout():
    """The other direction, and it is why `settle` and `settle_done` are two
    functions. A job that reaches `error` is terminal — the app answered, and
    it answered badly. That must surface as the app's error, immediately,
    with no waiting."""
    class FailsAtOnce:
        def get(self, _url):
            class R:
                @staticmethod
                def json():
                    return {"id": "j", "status": "error", "terminal": True,
                            "error": "the fit refused: target is constant"}
            return R()

    state = JW.settle(FailsAtOnce(), {"id": "j"}, timeout=30)
    assert state["status"] == "error", "a terminal job is returned, not raised"

    with pytest.raises(AssertionError) as caught:
        JW.settle_done(FailsAtOnce(), {"id": "j"}, timeout=30)
    assert "target is constant" in str(caught.value), (
        "the app's own error is what the caller should see")
    assert not isinstance(caught.value, JW.JobDidNotSettle), (
        "a failed fit is not a timeout, and conflating them is the defect "
        "this file exists to close")


def test_it_returns_as_soon_as_the_job_is_terminal():
    """A generous timeout costs nothing when the job finishes. If `settle`
    slept out its deadline the default of 120 s would make the suite
    unrunnable, so this pins the early return."""
    calls = {"n": 0}

    class FinishesOnThirdPoll:
        def get(self, _url):
            calls["n"] += 1
            done = calls["n"] >= 3
            class R:
                @staticmethod
                def json():
                    return {"id": "j", "status": "done" if done else "running",
                            "terminal": done}
            return R()

    state = JW.settle_done(FinishesOnThirdPoll(), {"id": "j"},
                           timeout=30, poll=0.01)
    assert state["status"] == "done"
    assert calls["n"] == 3, calls

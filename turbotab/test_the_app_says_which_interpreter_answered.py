"""`TEST-087` — `/dev/status` names the interpreter, and the page says so.

**The extension `TEST-084` needed, found one drive after `TEST-084` landed.**
`L60-B3` made the app say which BUILD is answering, because three drives had a
version question attached to them. It worked exactly as designed: run 4 opened
with *"Build is fresh and consistent… everything on screen is trustworthy"*,
`rev` matched `HEAD`, `page_newer_than_engine` was `false`, and the driver
re-checked it mid-drive.

**All of that was true and the app still could not fit a model**, because the
banner reports the CODE's vintage and the failure was the ENVIRONMENT. Honest
and insufficient in the same breath, which is worse than absent: it licensed a
conclusion it could not support.

The same shape three times, one layer further out each time. `PM_TRANSITION.md`
§07 item 8: *ahead 31* was a fact about the remote, not the working copy.
`TEST-084`: the process was serving from the right directory and the wrong
code. Now: the right code in the wrong environment.

**And `ps` cannot answer it, which is why the app has to.** `venv/bin/python` is
a symlink to the Homebrew interpreter, so `ps` prints the resolved path and a
complete virtualenv looks identical to the bare system Python. `L60-E` read
exactly that and wrote *"every uvicorn on this host runs SYSTEM Python"* — for
two processes that were running two different virtualenvs, one complete and one
not. Only `sys.prefix` inside the process knows.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, pageharness as PH                          # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_dev_status_names_the_interpreter_and_the_environment(client):
    """The payload. Served unconditionally, like the build half — a driver who
    cannot tell the environment produces findings nobody can reproduce."""
    body = client.get("/dev/status").json()
    env = body.get("environment")
    assert env, "/dev/status carries no environment block at all"
    assert env["python"] == sys.executable, (env["python"], sys.executable)
    assert env["prefix"] == sys.prefix, (env["prefix"], sys.prefix)
    assert isinstance(env["engine_stack_ok"], bool)
    assert isinstance(env["missing"], list)


def test_the_two_halves_are_independent(client):
    """**The point of the row, asserted.** A build stamp that was clean while
    the environment was broken is exactly what happened, so the payload must be
    able to say `build fine, environment broken` — the two must not be one
    field wearing two names."""
    body = client.get("/dev/status").json()
    # Namespaced rather than disjoint: both blocks carry a `why`, and that is
    # right — each explains ITS own fact. What must not happen is a fact from
    # one appearing inside the other, because a reader who checked `build` and
    # found it clean would take that as covering both, which is exactly what
    # run 4 did.
    for owned_by_environment in ("python", "prefix", "engine_stack_ok",
                                 "missing"):
        assert owned_by_environment not in body["build"], (
            f"{owned_by_environment} is inside the BUILD block, where a reader "
            f"looking for the code's vintage would take it for one")
    for owned_by_build in ("rev", "page_mtime", "page_newer_than_engine"):
        assert owned_by_build not in body["environment"], (
            f"{owned_by_build} is inside the ENVIRONMENT block")
    assert body["build"]["why"] != body["environment"]["why"] or (
        body["build"]["why"] is None), (
        "the two blocks explain themselves with the same sentence, so a "
        "reader cannot tell which fact it is about")


def test_a_healthy_environment_reports_the_absence_of_nothing(client):
    """Under `venv/` the stack is complete, so `missing` is empty and `why` is
    `None` rather than an empty string. Returning nothing where there is
    nothing to say is the habit this project keeps."""
    env = client.get("/dev/status").json()["environment"]
    assert env["engine_stack_ok"] is True, env
    assert env["missing"] == [], env
    assert env["why"] is None and env["fix"] is None, env


def test_the_probe_does_not_import_the_stack_it_asks_about():
    """**The cost, asserted rather than assumed.** `_SERVED_BUILD` is stamped
    at API import — in the server AND in every one of two thousand test
    processes — so a probe that imported scikit-learn, xgboost and lightgbm to
    find out whether they are there would add seconds to each of them.

    `find_spec` answers the question that actually failed, which is ABSENCE.
    The launcher does the real import, where it can afford to.
    """
    import subprocess

    probe = ("import sys, time;"
             "t = time.perf_counter();"
             "from ml import engine_stack;"
             "engine_stack.report();"
             "print(int('sklearn' in sys.modules), round(time.perf_counter() - t, 3))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    imported, seconds = done.stdout.split()
    assert imported == "0", (
        "the probe imported scikit-learn, so every API import and every test "
        "process now pays for it")
    assert float(seconds) < 1.0, f"the probe took {seconds}s"


def _reader() -> str:
    return ('__emit({devBuild: __harness.html("devBuild"),\n'
            '        cls: (__harness.el("devBuild") || {}).className});')


def _status(*, stack_ok: bool, stale: bool = False) -> dict:
    return {
        "enabled": False, "session": None, "flag": "TURBOTAB_DEV_CHECKS",
        "build": {"rev": "abc1234",
                  "engine_loaded_at": 1.0 if stale else 2.0,
                  "page_mtime": 2.0 if stale else 1.0,
                  "page_newer_than_engine": stale,
                  "why": ("The page is re-read from disk on every request. "
                          "Restart the server." if stale else None)},
        "environment": (
            {"python": sys.executable, "prefix": sys.prefix,
             "engine_stack_ok": True, "missing": [], "why": None, "fix": None}
            if stack_ok else
            {"python": "/tmp/bare/.venv/bin/python",
             "prefix": "/tmp/bare/.venv",
             "engine_stack_ok": False,
             "missing": ["sklearn", "xgboost", "lightgbm"],
             "why": "This interpreter cannot import sklearn, so the model "
                    "registry cannot be built and every model-shelf request "
                    "will fail.",
             "fix": "Start the server from the environment that has them: "
                    "`make turbotab`."}),
    }


def test_the_page_renders_a_broken_environment_rather_than_only_serving_it(capsys):
    """**A payload nobody reads is the defect this fixes**, and it is the exact
    mistake `TEST-084` was written to avoid making twice.

    Driven through the harness against a `/dev/status` that reports a broken
    stack, so what is asserted is what the page DID with it. It reuses the
    build banner's element — one place for *what you cannot conclude from this
    session* — rather than a second one nobody looks at.
    """
    if not PH.available():
        pytest.skip("no JS engine on this machine")

    # THE NEGATIVE FIRST. A banner that rendered on a healthy server would be
    # a mark that asserts nothing (`DESIGN_LANGUAGE.md` §02) and the positive
    # below would prove very little.
    clean = PH.run(_reader(), routes={"/dev/status": _status(stack_ok=True)})
    assert not (clean["devBuild"] or "").strip(), (
        f"a healthy server rendered a banner: {clean['devBuild']!r}")

    broken = PH.run(_reader(), routes={"/dev/status": _status(stack_ok=False)})
    said = broken["devBuild"] or ""
    assert "cannot build the model shelf" in said, (
        f"the server said the environment cannot fit a model and the page said "
        f"nothing — which is the state four drives were in: {said!r}")
    assert "/tmp/bare/.venv/bin/python" in said, (
        "the banner does not name the interpreter, so a driver still cannot "
        "tell which Python answered")
    assert "make turbotab" in said, "the banner offers no way out"
    assert "devbuild" in (broken["cls"] or ""), (
        "the banner has content and no class, so the stylesheet keeps it hidden")
    with capsys.disabled():
        print(f"\n  healthy: silent · broken stack: {len(said)} chars on screen")


def test_both_banners_can_be_true_at_once():
    """The two facts are independent and the page must be able to say both.

    Run 4's session was clean on the build and broken on the environment. The
    reverse, and the both-at-once case, are equally possible — a page that
    could only ever show one would hide whichever it did not choose.
    """
    if not PH.available():
        pytest.skip("no JS engine on this machine")

    out = PH.run(_reader(),
                 routes={"/dev/status": _status(stack_ok=False, stale=True)})
    said = out["devBuild"] or ""
    assert "newer page than its engine" in said, said[:400]
    assert "cannot build the model shelf" in said, said[:400]

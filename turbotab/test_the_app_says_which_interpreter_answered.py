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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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



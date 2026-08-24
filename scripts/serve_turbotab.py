#!/usr/bin/env python
"""Start TurboTab — or refuse, and say which package and which interpreter.

**`DRIVE-035`, and the whole of what four human drives cost.** There was no
launch command. `make serve` starts the *Streamlit* app, `README.md` says
nothing about TurboTab, and `turbotab/README.md`'s "Run it" section — written
when TurboTab was a walking skeleton with no Train step — tells you to build
`turbotab/.venv` from `turbotab/requirements.txt` and run uvicorn from that.
**That environment is deliberately empty of scikit-learn**, and
`tests/test_the_guided_door_installs_without_the_app.py` exists to keep it that
way. So the documented way to start the app was guaranteed to lose `GET
/models`, and a test guarded the guarantee.

The refusal below is the point of this file. A server that starts happily and
then answers twenty-one characters of *Internal Server Error* at the Train step
is the failure this exists to prevent: it is the governing rule's *assert
something false* branch wearing a green terminal.

**The probe runs in the interpreter that will serve.** That is not a detail —
it is the entire lesson. Every test, every probe and every number in four loops
was measured under `venv/`, and every browser request was answered by
`turbotab/.venv`. A check that runs anywhere else is asking a different
interpreter the same question.

Usage
-----
    make turbotab                 # macOS/Linux alias; uses ./venv explicitly
    venv/bin/python scripts/serve_turbotab.py --port 8777 --reload
    venv\\Scripts\\python scripts\\serve_turbotab.py --port 8777   # Windows
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rev() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              timeout=5)
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _broken_why(module: str) -> str:
    """The sentence for a module that is FOUND and will not import.

    `report()['why']` covers absence, which is the case four drives hit and the
    one `find_spec` can see. This is the other one, and it gets its own words
    rather than borrowing them: *missing* and *broken* are different problems
    and a refusal that called the second one *missing* would send a person to
    reinstall a package that is already there.
    """
    return (f"This interpreter finds {module} and cannot import it, so the "
            f"model registry cannot be built. The package is present and "
            f"something about the install is wrong — a partial install, a "
            f"shadowing file on PYTHONPATH, or a wheel built for another "
            f"platform.")


def _broken_fix(module: str) -> str:
    from ml import engine_stack

    return (f"Run `{sys.executable} -c \"import {module}\"` to see the whole "
            f"traceback, then reinstall: "
            f"`{sys.executable} -m pip install --force-reinstall "
            f"{engine_stack._distribution(module)}`.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start the TurboTab server.")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reload", action="store_true",
                        help="restart the engine when a source file changes. "
                             "Without it a long-running process serves a NEW "
                             "page against the Python it started with "
                             "(TEST-084).")
    parser.add_argument("--check-only", action="store_true",
                        help="run the environment check, print it, and exit "
                             "without binding the port.")
    args = parser.parse_args(argv)

    from ml import engine_stack

    # FLUSHED, and it is not tidiness. Python block-buffers stdout when it is
    # redirected, and uvicorn logs to stderr — so `make turbotab > log` put the
    # banner AFTER the server's startup lines, or after the process exited.
    # The whole requirement is that the first line of the terminal answers
    # *which interpreter*, and a buffered first line is not a first line.
    def say(text: str = "") -> None:
        print(text, flush=True)

    # PRINTED BEFORE ANYTHING ELSE, because the first line of the terminal is
    # what three drives could not answer: which interpreter, and which build.
    say("TurboTab")
    say(f"  interpreter  {sys.executable}")
    say(f"  environment  {sys.prefix}")
    say(f"  build        {_rev()}")

    # THE REAL IMPORT, not `find_spec`. The launcher runs once and the server
    # it is about to start imports this stack on its first `/models` request
    # anyway, so doing it here costs a few seconds and catches a package that
    # is present and broken as well as one that is absent. `/dev/status` uses
    # the cheap check because it runs at API import in every test process;
    # this one can afford to be right about more.
    failure = engine_stack.import_failure()
    if failure is not None:
        module, error = failure
        report = engine_stack.report()
        absent = report["missing"] or [module]
        say()
        say("REFUSED — this interpreter cannot build the model shelf.")
        say(f"  missing      {', '.join(absent)}")
        say(f"  first error  {module}: {error}")
        say(f"  why          {report['why'] or _broken_why(module)}")
        say(f"  fix          {report['fix'] or _broken_fix(module)}")
        say()
        say("The port was not bound. A server that starts here answers every")
        say("model request with an unhandled ModuleNotFoundError, which the")
        say("browser reads as 21 characters of Internal Server Error.")
        return 2

    say(f"  model shelf  ready — {', '.join(n for n, _d, _w in engine_stack.ESTIMATOR_STACK)}")
    extras = engine_stack.report()["extras_missing"]
    if extras:
        # NOT a refusal. `torch` is absent on purpose (`TEST-038`) and `shap`
        # shortens one explanation rather than the shelf.
        say(f"  optional     absent: {', '.join(extras)} "
              f"(the shelf is unaffected)")
    if args.check_only:
        return 0

    say(f"  serving      http://{args.host}:{args.port}/")
    say()

    import uvicorn

    uvicorn.run("turbotab.api:app", host=args.host, port=args.port,
                reload=bool(args.reload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

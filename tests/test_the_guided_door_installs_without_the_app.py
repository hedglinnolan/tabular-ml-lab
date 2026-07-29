"""The portability claim, with its second signal restored.

`turbotab/requirements.txt` states it: *the whole diagnose → profile → detect
path needs pandas and numpy and nothing else — no scikit-learn, no scipy, no
statsmodels, and none of the 1.1 GB of torch the Streamlit app installs.*

**Two independent signals used to hold that, and one of them was lost.**

* `tests/test_engine_is_headless.py` blocks `streamlit` at import and asserts
  the core modules load anyway. That one still works.
* `turbotab/.venv` was *empty of the app's dependencies*, so the claim was true
  of a real environment and not only of an import blocker. At L19 the app's
  requirements were installed into it to run the full suite, and that signal
  went quiet — not wrong, just gone, which is the shape of failure this project
  keeps naming.

The environments are separated again: `./venv` is the full one the Makefile
already named (`PYTHON := ./venv/bin/python`), and `turbotab/.venv` is the
Guided door's. This file is the second signal made executable, so it cannot go
quiet a second time without something saying so.

**It skips rather than fails when the venv is absent**, and that is deliberate:
the claim is about what a Guided-door environment CONTAINS, and a machine that
has not built one has no opinion. `test_engine_is_headless` carries the claim
that does not depend on an environment.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUIDED_VENV = ROOT / "turbotab" / ".venv"

# What `turbotab/requirements.txt` says the Guided door does NOT need. Each is
# named rather than inferred from a diff, because a list computed from what
# happens to be installed would agree with any environment it was run in.
FORBIDDEN = ("sklearn", "scipy", "statsmodels", "matplotlib", "streamlit",
             "torch", "shap", "xgboost", "lightgbm")


def _interpreter() -> pathlib.Path:
    for name in ("bin/python", "Scripts/python.exe"):
        candidate = GUIDED_VENV / name
        if candidate.exists():
            return candidate
    pytest.skip("turbotab/.venv is not built on this machine")


def test_the_guided_environment_carries_none_of_the_apps_dependencies():
    """The claim, against a real environment rather than an import blocker."""
    python = _interpreter()
    probe = (
        "import importlib.util, json, sys;"
        f"print(json.dumps([m for m in {list(FORBIDDEN)!r} "
        "if importlib.util.find_spec(m) is not None]))"
    )
    out = subprocess.run([str(python), "-c", probe], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    present = json.loads(out.stdout.strip().splitlines()[-1])
    assert present == [], (
        f"turbotab/.venv now carries {present}, so the portability claim in "
        f"turbotab/requirements.txt is true only of the import blocker and not "
        f"of any environment. Install the app's dependencies into ./venv — the "
        f"Makefile already names it — and keep this one minimal.")


def test_the_guided_environment_carries_what_the_door_actually_needs():
    """The other half. An environment that is minimal because it is empty
    proves nothing, and would pass the test above perfectly."""
    python = _interpreter()
    probe = (
        "import importlib.util, json;"
        "print(json.dumps([m for m in ['pandas','numpy','fastapi','pytest'] "
        "if importlib.util.find_spec(m) is None]))"
    )
    out = subprocess.run([str(python), "-c", probe], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_the_engine_runs_in_that_environment_and_not_only_imports():
    """`ml.import_doctor.diagnose` on a real frame, in the minimal environment.

    Importing is the weaker claim: a module can import and then reach for
    scikit-learn on the first call. This runs the path the requirements file
    makes its claim about.
    """
    python = _interpreter()
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import pandas as pd;"
        "from ml import import_doctor;"
        "df = pd.read_csv(%r);"
        "print(len(import_doctor.diagnose(df)))"
        % (str(ROOT), str(ROOT / "turbotab" / "sample_data" / "clinic_visits.csv"))
    )
    out = subprocess.run([str(python), "-c", probe], capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-1500:]
    assert int(out.stdout.strip().splitlines()[-1]) > 0

"""Classic's CI collects its tests on this tree (BLUEPRINT §9.1, §10).

Classic must keep working on ``turbotab-next``, and its CI is how that is shown. Some of Classic's
tests read the legacy app's record when pytest imports them: ``docs/turbotab/data/``,
``docs/turbotab/tools/`` and ``docs/turbotab/DOMAIN_SCIENCE.md``. When the legacy app was retired,
the first version of that change archived those paths too. Then pytest stopped at collection
("Interrupted: 2 errors during collection"), and neither CI tier ran a single test. A collection
error does not fail one test. It stops the whole run, so each tier is collected here exactly as
``.github/workflows/ci.yml`` runs it.

The commands are read from ``ci.yml`` rather than restated, so a tier CI adds is collected too.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CI = REPO / ".github" / "workflows" / "ci.yml"


def ci_pytest_runs(text: str) -> list[list[str]]:
    """The paths and ``--ignore`` flags of every ``python -m pytest`` command in a workflow."""
    joined = re.sub(r"\\\n\s*", " ", text)
    runs = []
    for line in joined.splitlines():
        found = re.search(r"python -m pytest\s+(.*)$", line)
        if found:
            runs.append([a for a in shlex.split(found.group(1))
                         if not a.startswith("-") or a.startswith("--ignore=")])
    return runs


RUNS = ci_pytest_runs(CI.read_text("utf-8"))

# Classic's tests import these at module level, and only Classic's requirements.txt installs them.
# The v2 workflow (.github/workflows/v2.yml) installs turbotab/server/requirements.txt alone, so
# there every Classic tier would fail to collect for a reason that says nothing about this tree.
CLASSIC_ONLY = ("streamlit", "torch")
MISSING = [name for name in CLASSIC_ONLY if importlib.util.find_spec(name) is None]


def collect(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                           *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def test_the_workflow_is_read():
    """The positive control: an empty parametrization would pass every collection below."""
    assert len(RUNS) >= 3, RUNS
    assert ["tests/", "--ignore=tests/integration"] == RUNS[0][:2], RUNS[0]
    assert any(run == ["tests/integration"] for run in RUNS), RUNS


@pytest.mark.parametrize("args", RUNS, ids=[" ".join(r)[:60] for r in RUNS])
def test_each_ci_tier_collects_without_an_error(args):
    if MISSING:
        pytest.skip(f"Classic's requirements.txt is not installed here ({', '.join(MISSING)} missing); "
                    "its tiers collect only where Classic's CI installs it, as the venv does")
    out = collect(args)
    tail = "\n".join((out.stdout + out.stderr).strip().splitlines()[-25:])
    assert out.returncode == 0, f"pytest {' '.join(args)} does not collect:\n{tail}"
    counted = re.search(r"(\d+) tests? collected", out.stdout)
    assert counted and int(counted.group(1)) > 0, tail
    assert not re.search(r"\d+ errors?\b", out.stdout.splitlines()[-1]), tail


def test_a_module_that_fails_on_import_is_caught(tmp_path):
    """The negative control: a test module that reads a missing file on import fails collection."""
    probe = tmp_path / "test_probe.py"
    probe.write_text('from pathlib import Path\nDATA = (Path(__file__).parent / "absent.json").read_text()\n'
                     "def test_x():\n    assert DATA\n", "utf-8")
    out = collect([str(probe)])
    assert out.returncode != 0
    assert "error" in out.stdout.splitlines()[-1]

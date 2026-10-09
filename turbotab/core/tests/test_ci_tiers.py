"""TurboTab v2's two CI tiers (audit of 2026-10-09, recommendation 7).

The fast tier (``.github/workflows/v2.yml``) runs on every push in about ten minutes; the full tier
(``.github/workflows/v2-full.yml``) runs the whole suite and the browser tests. The first version
of the split ran the full tier only on a schedule and by "Run workflow". GitHub reads both only
from the default branch, ``main``, which had no such file, so the acceptance suite and the
browser tests stopped running anywhere: not on a push to ``turbotab-next``, not on a pull request
to ``main``. It also moved the guarantee tests (no held-out row reaches a fit, the cross-validated
scores against scikit-learn's own, the held-out seal) out of the fast tier. These tests read the
workflows as GitHub does and collect the fast tier's tests as its commands do.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml  # installed with uvicorn[standard] (turbotab/server/requirements.txt)

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"


def workflow(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text("utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # YAML 1.1 reads the key `on` as true
    return data


FAST = workflow("v2.yml")
FULL = workflow("v2-full.yml")


def pytest_args(run: str) -> list[str]:
    """The arguments of the one ``python -m pytest`` command in a step's script."""
    found = re.search(r"python -m pytest\s+(.*?)(?:\|\||$)", re.sub(r"\\\n\s*", " ", run), re.M)
    assert found, run
    return shlex.split(found.group(1))


def runs(job: dict) -> list[str]:
    return [step.get("run", "") for step in job["steps"]]


def collect(args: list[str]) -> set[str]:
    kept = [a for a in args if a not in ("-q", "-rfE") and not a.startswith("--junitxml")]
    if "-n" in kept:
        i = kept.index("-n")
        del kept[i:i + 2]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                           *kept], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return {line for line in done.stdout.splitlines() if "::" in line}


def test_the_full_tier_runs_where_the_work_lands_and_before_a_release():
    on = FULL["on"]
    assert "turbotab-next" in on["push"]["branches"], on
    assert "main" in on["pull_request"]["branches"], on
    assert "schedule" in on and "workflow_dispatch" in on, on
    for name, job in FULL["jobs"].items():
        assert "if" not in job, f"{name} would skip some of the triggers above: {job['if']}"


def test_the_nightly_tests_the_integration_branch_not_the_default_one():
    """A schedule runs on the default branch's latest commit; main changes only through pull
    requests, which this tier runs already, so the nightly checks out turbotab-next."""
    for name, job in FULL["jobs"].items():
        checkout = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout"))
        ref = checkout.get("with", {}).get("ref", "")
        assert re.fullmatch(r"\$\{\{ github\.event_name == 'schedule' && 'turbotab-next' \|\| '' \}\}",
                            ref), (name, ref)


def test_the_full_tier_runs_every_test_and_the_browser_tests():
    commands = [run for job in FULL["jobs"].values() for run in runs(job)]
    suites = [pytest_args(run) for run in commands if "python -m pytest" in run]
    assert suites == [["turbotab/core", "turbotab/server", "-q", "-n", "2", "-rfE",
                       "--junitxml=pytest-report.xml"]], suites
    assert any("npx playwright test" in run for run in commands), commands


def test_the_fast_tier_runs_on_every_push_and_pull_request():
    on = FAST["on"]
    assert {"turbotab-next", "ci/**"} <= set(on["push"]["branches"]), on
    assert "main" in on["pull_request"]["branches"], on


# The guarantees the fast tier must hold on every push (audit recommendations 4 and 5): checked
# by name, so that marking one slow or moving it out of every shard fails here.
GUARANTEES = (
    "turbotab/core/tests/test_modeling.py::test_every_fitted_step_sees_only_training_fold_rows[prediction]",
    "turbotab/core/tests/test_modeling.py::test_every_fitted_step_sees_only_training_fold_rows[inference]",
    "turbotab/core/tests/test_modeling.py::test_cv_metrics_equal_an_independent_cross_validate[regression-glucose]",
    "turbotab/core/tests/test_modeling.py::test_cv_metrics_equal_an_independent_cross_validate[binary-glucose_high]",
    "turbotab/core/tests/test_modeling.py::test_cv_metrics_equal_an_independent_cross_validate[multiclass-glucose_band]",
    "turbotab/server/tests/test_seal_api.py::test_no_held_out_score_reaches_any_response_until_the_seal_is_opened",
    "turbotab/server/tests/test_seal_api.py::test_a_change_after_the_opening_is_marked_post_seal_and_the_results_say_so",
)


@pytest.fixture(scope="module")
def shards() -> dict[str, set[str]]:
    job = FAST["jobs"]["fast-tests"]
    (run,) = [r for r in runs(job) if "python -m pytest" in r]
    found = {}
    for shard in job["strategy"]["matrix"]["include"]:
        command = run.replace("${{ matrix.paths }}", shard["paths"])
        assert "${{" not in pytest_args(command), command
        found[shard["shard"]] = collect(pytest_args(command))
    return found


def test_the_fast_shards_together_run_each_fast_test_once(shards):
    """Every test but the acceptance suite and those marked slow, in exactly one shard: a new test
    file lands in a shard by itself, and none runs twice."""
    expected = collect(["turbotab/core", "turbotab/server", "--ignore=turbotab/core/tests/acceptance",
                        "-m", "not slow"])
    assert len(expected) > 1000, len(expected)  # the positive control
    union = set().union(*shards.values())
    assert union == expected, (sorted(expected - union)[:10], sorted(union - expected)[:10])
    assert sum(map(len, shards.values())) == len(union), {k: len(v) for k, v in shards.items()}


def test_the_fast_tier_keeps_the_guarantee_tests(shards):
    union = set().union(*shards.values())
    assert [g for g in GUARANTEES if g not in union] == []

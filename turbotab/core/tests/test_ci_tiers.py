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

import importlib.util
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
    # Reporting and timeout flags change nothing collected; pytest-timeout is pinned for CI, and
    # venv-next may lack it.
    kept = [a for a in args if a not in ("-q", "-rfE") and not a.startswith(("--junitxml", "--timeout"))]
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


def shard_commands(job: dict) -> dict[str, str]:
    """Each matrix shard's ``python -m pytest`` script, with its paths filled in."""
    (run,) = [r for r in runs(job) if "python -m pytest" in r]
    found = {}
    for shard in job["strategy"]["matrix"]["include"]:
        command = run.replace("${{ matrix.paths }}", shard["paths"])
        assert "${{" not in " ".join(pytest_args(command)), command
        found[shard["shard"]] = command
    return found


FULL_SUITE = FULL["jobs"]["full-tests"]


def test_the_full_shards_together_run_every_test_once_and_the_browser_tests_run():
    """The full suite ran past its job's 120 minutes on 2026-10-11, 10-13 and twice on 10-16 (it
    took 60 to 111 minutes when it finished), so it runs in shards; together they still run
    every test, slow and acceptance ones included, each once."""
    shards = {name: collect(pytest_args(run)) for name, run in shard_commands(FULL_SUITE).items()}
    assert len(shards) >= 2, shards.keys()
    expected = collect(["turbotab/core", "turbotab/server"])
    assert len(expected) > 3000, len(expected)  # the positive control
    union = set().union(*shards.values())
    assert union == expected, (sorted(expected - union)[:10], sorted(union - expected)[:10])
    assert sum(map(len, shards.values())) == len(union), {k: len(v) for k, v in shards.items()}
    commands = [run for job in FULL["jobs"].values() for run in runs(job)]
    assert any("npx playwright test" in run for run in commands), commands


def test_each_full_shard_fails_a_hung_test_by_name_and_reports_before_the_jobs_clock_ends():
    """A test past 20 minutes is killed by name, even inside C code (the thread method), and the
    whole run is interrupted at its own deadline, which leaves pytest time to write the report
    the annotations read; GitHub's job timeout leaves no report at all."""
    limit = FULL_SUITE["timeout-minutes"]
    for name, run in shard_commands(FULL_SUITE).items():
        args = pytest_args(run)
        for flag in ("-n", "-rfE", "--timeout=1200", "--timeout-method=thread", "--junitxml=pytest-report.xml"):
            assert flag in args, (name, flag, args)
        deadline = re.search(r"timeout --signal=INT --kill-after=(\d+)m (\d+)m python -m pytest", run)
        assert deadline, (name, run)
        kill_after, minutes = map(int, deadline.groups())
        assert minutes + kill_after + 10 <= limit, (name, minutes, kill_after, limit)
    always = [s for s in FULL_SUITE["steps"] if "annotate_durations.py" in s.get("run", "")]
    assert [s.get("if") for s in always] == ["always()"], always


def test_the_durations_annotation_ranks_files_and_tests_by_their_time(tmp_path):
    spec = importlib.util.spec_from_file_location("annotate_durations",
                                                  REPO / ".github" / "scripts" / "annotate_durations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="4">'
        '<testcase classname="turbotab.core.tests.acceptance.test_a" name="test_x" time="300"/>'
        '<testcase classname="turbotab.core.tests.acceptance.test_a" name="test_y" time="240"/>'
        '<testcase classname="turbotab.server.tests.test_b.TestGroup" name="test_z" time="600"/>'
        '<testcase classname="turbotab.core.tests.test_c" name="test_w[1]" time="1.5"/>'
        '<testcase time="0.000"/>'  # what an interrupted xdist run leaves
        '</testsuite></testsuites>', "utf-8")
    files, tests, total = module.slowest(report)
    assert files == [("turbotab/server/tests/test_b.py", 600.0),
                     ("turbotab/core/tests/acceptance/test_a.py", 540.0),
                     ("turbotab/core/tests/test_c.py", 1.5)]
    assert tests[0] == ("turbotab/server/tests/test_b.py::TestGroup::test_z", 600.0)
    assert tests[-1] == ("turbotab/core/tests/test_c.py::test_w[1]", 1.5)
    assert total == (4, 1141.5)
    assert module.slowest(tmp_path / "missing.xml") is None


def test_the_full_tiers_timeout_has_its_plugin_installed_at_a_pinned_version():
    # pytest refuses --timeout before collecting a test unless pytest-timeout is installed.
    server = REPO / "turbotab" / "server"
    dev = [line.split("#")[0].strip() for line in (server / "requirements-dev.txt").read_text("utf-8").splitlines()]
    pins = (server / "constraints.txt").read_text("utf-8").splitlines()
    assert "pytest-timeout" in dev, dev
    assert any(re.fullmatch(r"pytest-timeout==\d+(\.\d+)+", line.strip()) for line in pins), pins


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

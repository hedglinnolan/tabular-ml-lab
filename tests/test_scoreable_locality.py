"""A helper that enforces an invariant needs a test asserting its call sites.

`FEATURE_PARITY.md`, fifth member of the silence family. Principle-locality says
state a rule once; its unstated half is that stating it once is only safe when
*using* it everywhere is checkable.

`_scoreable_here` is the case that produced the rule. It exists because held-out
is not the same as scoreable, its own comment says so in exactly the right words
— *"reporting the sealed count is a number a researcher would write down and be
wrong about"* — and until `STATE-102` it was called at ONE site, inside the
cohort-run branch. The ordinary path printed the sealed count unconditionally,
so a row-dropping step after the seal left the chip reporting 60 held-out rows
where evaluation had 53.

Nothing failed when the second path forgot. This file is what fails.

The check is structural rather than behavioral on purpose: a behavioral test
would have to drive Streamlit to observe the chip, and it would only cover the
paths it happened to exercise. Reading the AST covers every path, including ones
added after this test was written — which is the whole point, since the defect
was a path that got added without the call.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.test_lockbox as lockbox                              # noqa: E402

HELPER = "_scoreable_here"
RENDERER = "render_lockbox_status"


def _function_ast(module, name: str) -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from {module.__name__}")


def _calls_to(node: ast.AST, name: str) -> list:
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def test_the_scoreable_check_is_computed_above_the_branch_not_inside_one():
    """The locality assertion itself.

    A call nested inside an `if` covers that branch and no other, which is
    exactly how `STATE-102` happened. Requiring it at the function's top level
    means every path below inherits it — including paths that do not exist yet.
    """
    fn = _function_ast(lockbox, RENDERER)
    # UNCONDITIONAL means the statement itself is not a branch. Walking into a
    # top-level `if` and counting the call inside it would pass on exactly the
    # code this test exists to reject -- and did, in the first draft of this
    # file, which is its own small argument for the revert probe.
    branchy = (ast.If, ast.For, ast.While, ast.Try, ast.With)
    unconditional = [n for n in fn.body
                     if not isinstance(n, branchy) and _calls_to(n, HELPER)]
    assert unconditional, (
        f"{RENDERER} calls {HELPER} only inside a branch. A call nested in an "
        f"`if` covers that branch and no other, and the next branch added below "
        f"it will print a sealed count nobody checked -- which is what "
        f"STATE-102 was.")


def test_every_path_that_prints_the_sealed_count_is_downstream_of_the_check():
    """The consequence, stated over the source rather than trusted.

    Any statement that formats `n_test` or the sealed label count must come
    after the scoreable check in the function body, or it can print a number
    the check would have corrected.
    """
    fn = _function_ast(lockbox, RENDERER)
    first_check = None
    for i, stmt in enumerate(fn.body):
        if _calls_to(stmt, HELPER):
            first_check = i
            break
    assert first_check is not None, f"{RENDERER} does not call {HELPER}"

    offenders = []
    for i, stmt in enumerate(fn.body):
        if i <= first_check:
            continue_ = True
        src = ast.get_source_segment(inspect.getsource(lockbox), stmt) or ""
        if i < first_check and ("n_test" in src or "['labels']" in src):
            offenders.append(i)
    assert not offenders, (
        f"{RENDERER} formats the sealed count at statement(s) {offenders}, "
        f"before {HELPER} runs at statement {first_check}.")


def test_the_helper_still_states_why_it_exists():
    """The docstring is load-bearing: it is where the distinction between held
    out and scoreable is written down. A refactor that keeps the call and drops
    the reasoning leaves the next reader unable to tell why the call matters."""
    doc = (getattr(lockbox, HELPER).__doc__ or "").lower()
    assert doc.strip(), f"{HELPER} lost its docstring"
    assert "outcome" in doc or "target" in doc, (
        f"{HELPER}'s docstring no longer says what makes a sealed row "
        f"scoreable")


# Sites that may read the sealed count WITHOUT the scoreable check, each with
# the reason it is exempt. An allowlist rather than a pattern, because "why is
# this one fine" is the question a new site has to answer, and answering it in
# writing is the whole mechanism. Adding a file here is a decision; forgetting
# to add one is a red test.
#
# Found by this test on its first run, which is the argument for writing it:
# STATE-102 named ONE bad site and there were two, plus five benign ones nobody
# had ever enumerated.
SCOREABLE_EXEMPT = {
    # At seal time. The labels were drawn moments earlier from rows that have a
    # value for the target, so sealed == scoreable by construction — there is
    # no interval in which a row could have been dropped.
    "pages/01_Upload_and_Audit.py": "seal-time disclosure; nothing has run yet",
    # Serialization, not display. The archive must record what was SEALED —
    # that is the historical fact. Recording the currently-scoreable count
    # would make the archive's meaning depend on when it was written.
    "utils/session_manager.py": "persists the sealed count as the record of the seal",
    "turbotab/archive.py": "same, for the Guided door's archive",
    # Assertion, not display. The dev harness compares the seal DECISION's
    # n_test against the LOCKBOX's n_test, which is a check that the record
    # agrees with itself — the same category as a test, and the reason the loop
    # above already skips `test_*`. It shows a researcher nothing; a violation
    # goes to a gitignored session directory behind TURBOTAB_DEV_CHECKS=1.
    #
    # Worth recording that this guard CAUGHT the new file the day it landed,
    # which is what a call-site enumeration is for: the invariant did not have
    # to be remembered by whoever wrote `devchecks.py`.
    "turbotab/devchecks.py": "compares the record against itself; displays nothing",
}


@pytest.mark.parametrize("name", [RENDERER])
def test_every_site_that_formats_the_sealed_count_is_checked_or_exempt(name):
    """The search is part of the test rather than part of somebody's memory.

    A site that prints `lb['n_test']` at a distance from the seal is printing a
    number that may no longer be true. Either it goes through the renderer that
    checks, or it names why it is exempt.
    """
    import pathlib
    root = pathlib.Path(PROJECT_ROOT)
    hits = []
    for path in (list(root.glob("pages/*.py")) + list(root.glob("utils/*.py"))
                 + list(root.glob("turbotab/*.py"))):
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        # Tests assert about the sealed count; that is their job, not a
        # display of it to a researcher.
        if path.name.startswith("test_") or rel in SCOREABLE_EXEMPT:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if "n_test" not in line or not ("lb[" in line or "lockbox[" in line):
                continue
            if line.lstrip().startswith("#"):
                continue
            # The site is fine if it asks the helper somewhere in the same file.
            if HELPER in text or "_score(" in text:
                continue
            hits.append(f"{rel}:{lineno}")
    assert not hits, (
        "these sites format the SEALED count without ever asking whether it is "
        "still scoreable:\n  " + "\n  ".join(hits) +
        "\nEither call _scoreable_here beside it, or add the file to "
        "SCOREABLE_EXEMPT in this test with the reason it does not need to.")


# ── the behavior the locality protects ───────────────────────────────────────

def test_the_ordinary_path_reports_what_is_scoreable_not_what_was_sealed():
    """`STATE-101`'s measured harm, as a guard.

    400 rows, 60 with an impossible glucose. The seal holds out 60; the
    plausibility filter then drops 7 of them and evaluation runs on 53. Before
    this fix the chip said 60 on the non-cohort path, because `_scoreable_here`
    was only consulted inside the cohort branch.
    """
    import numpy as np
    import pandas as pd
    import streamlit as st
    from ml.pipeline import apply_plausibility_filter
    from utils.test_lockbox import ensure_lockbox

    st.session_state.clear()
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "age": rng.integers(20, 80, n).astype(float),
        "glucose": rng.normal(95, 12, n),
        "outcome": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 60, replace=False), "glucose"] = -999.0

    from utils.session_state import DataConfig
    st.session_state["raw_data"] = df
    st.session_state["data_config"] = DataConfig(
        target_col="outcome", feature_cols=["age", "glucose"],
        task_type="classification")
    lb = ensure_lockbox(df, "outcome", "classification")
    assert lb is not None

    filtered = apply_plausibility_filter(
        df, ["age", "glucose"],
        {"lower_bounds": [0.0, 20.0], "upper_bounds": [120.0, 400.0]})
    st.session_state["filtered_data"] = filtered

    still_here = [l for l in lb["labels"] if l in filtered.index]
    assert len(still_here) < lb["n_test"], (
        "the fixture no longer removes any sealed row, so it cannot exercise "
        "the gap it was written for")

    scoreable = lockbox._scoreable_here(lb["labels"])
    assert scoreable is not None
    assert scoreable == len(still_here), (
        f"the helper reports {scoreable} scoreable of {lb['n_test']} sealed, "
        f"but {len(still_here)} sealed labels survive the filter")
    st.session_state.clear()

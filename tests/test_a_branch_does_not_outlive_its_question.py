"""A cohort branch is only comparable under the question it answered.

`utils/cohorts.py` is about to start banking a whole cohort's results — every
fit, frame, matrix, score and figure — so that switching from the women to the
men and back does not destroy either. The persistence is the easy half. The
half that decides whether the feature is safe is *invalidation*: not "a branch
was lost" but "a branch survived a change that made it stale, and its numbers
reached the manuscript."

The rule is default-destructive. `reset_downstream_results` drops the archive
on every call unless the caller explicitly says otherwise, and the only caller
that may say otherwise is the cohort switch — the one change that alters *who
the rows are* and nothing else. Every other caller changed the question:

    the data · the target · the feature list · the engineering recipe ·
    the selection · the preprocessing rule · the row filter · the seal ·
    the quarantine regime · a restored session

There are nineteen call sites today, in nine files. Seventeen changed the
question; two are the cohort switch. The design this implements enumerated
fourteen — it missed page 01's two new-file resets, page 02's staleness guard,
`set_data`'s schema branch and the session restore. That is the argument for
putting the drop INSIDE the reset instead of beside its call sites: a caller
written by someone who has never heard of a cohort branch — or simply not
noticed by whoever counted — cannot leak one by forgetting.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import streamlit as st

from turbotab import cascade
from turbotab.cascade import BRANCH_ARCHIVE_KEY
from utils.session_state import reset_data_dependent_state, reset_downstream_results

ROOT = pathlib.Path(__file__).resolve().parent.parent

_RESET_FUNCS = {"reset_downstream_results", "reset_data_dependent_state"}

# The one file whose calls are allowed to preserve the archive. A cohort switch
# is "same question, different people": the branches it did not touch are still
# answers to the question being asked.
_MAY_PRESERVE = {"utils/cohorts.py"}

# Every file that invalidates downstream results, and what its calls mean.
# `archive/restore` is the cohort switch; everything else changed the question
# and must drop every branch. A file appearing here that is not in this map is
# a new caller nobody classified — which is exactly when someone should be
# made to think about whether it changes the question.
_CALLER_KIND = {
    "pages/01_Upload_and_Audit.py": "drop-all",     # new file; config save; exploratory flip
    "pages/02_EDA.py": "drop-all",                  # staleness guard: target column is gone
    "pages/03_Feature_Engineering.py": "drop-all",  # the recipe changed
    "pages/04_Feature_Selection.py": "drop-all",    # the selection changed
    "pages/05_Preprocess.py": "drop-all",           # the row-filter rule changed rows
    "utils/session_state.py": "drop-all",           # set_data; the full data reset
    "utils/session_manager.py": "drop-all",         # a different session is being restored
    "utils/test_lockbox.py": "drop-all",            # the seal moved
    "utils/cohort_ui.py": "archive/restore",        # who the rows are
}

_SOURCE_DIRS = ("pages", "utils", "ml", "turbotab", "launcher", "scripts")


def _source_files():
    out = [ROOT / "app.py"]
    for d in _SOURCE_DIRS:
        out.extend(sorted((ROOT / d).rglob("*.py")))
    return [p for p in out if p.exists() and "__pycache__" not in p.parts]


def _reset_calls(path: pathlib.Path):
    """Every call to a reset function in one file, with its keywords.

    Import aliases are resolved rather than matched by name: page 01 calls the
    reset as `_rdr` inside a widget callback, and a scan that grepped for the
    function's own name would report that call site as absent — which is the
    one shape of blind spot this test exists to have no version of.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in _RESET_FUNCS:
                    aliases[a.asname or a.name] = a.name
        # The defining module calls them by their own names, with no import to
        # bind. `set_data` and `reset_data_dependent_state` are two of the
        # eighteen call sites and would be invisible to an import-only scan.
        elif isinstance(node, ast.FunctionDef) and node.name in _RESET_FUNCS:
            aliases[node.name] = node.name
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name):
            name = aliases.get(fn.id)
        elif isinstance(fn, ast.Attribute) and fn.attr in _RESET_FUNCS:
            name = fn.attr
        else:
            name = None
        if name is None:
            continue
        kwargs = {k.arg: k.value for k in node.keywords if k.arg is not None}
        calls.append((name, node.lineno, kwargs))
    return calls


def _all_reset_calls():
    found = {}
    for path in _source_files():
        rel = path.relative_to(ROOT).as_posix()
        calls = _reset_calls(path)
        if calls:
            found[rel] = calls
    return found


# ── the static gate: nobody else may preserve ────────────────────────────

def test_only_the_cohort_switch_may_preserve_the_archive():
    """The whole safety property, in one assertion.

    `preserve_branches=True` says "the question did not change". Only a cohort
    switch can say that truthfully. A page that discovers the flag and passes
    it to keep a branch alive across a recipe change would be publishing one
    cohort's models under another cohort's Methods.
    """
    offenders = []
    for rel, calls in _all_reset_calls().items():
        if rel in _MAY_PRESERVE:
            continue
        for name, line, kwargs in calls:
            node = kwargs.get("preserve_branches")
            if node is None:
                continue
            if not (isinstance(node, ast.Constant) and node.value is False):
                offenders.append(f"{rel}:{line} — {name}(preserve_branches=...)")
    assert not offenders, (
        "these call sites keep archived cohort branches across a change that "
        "invalidates them: " + "; ".join(offenders) + ". Only a cohort switch "
        "leaves the question intact.")


def test_every_caller_of_the_cascade_is_classified():
    """A new caller is a decision, not an accident.

    The drop is default-on precisely so that forgetting is safe, and this test
    is the other half of that: forgetting is safe, but it does not stay
    invisible. A file that starts invalidating downstream results lands here
    until someone writes down whether it changed the question.
    """
    found = _all_reset_calls()
    unclassified = {
        rel: [f"{rel}:{line} {name}(…)" for name, line, _ in calls]
        for rel, calls in found.items() if rel not in _CALLER_KIND
    }
    assert not unclassified, (
        "these files invalidate downstream results and are not in _CALLER_KIND: "
        f"{unclassified}. Add each with 'drop-all' or 'archive/restore' — and if "
        "it is 'archive/restore', it belongs behind utils.cohorts.switch_branch, "
        "not at its own call site.")

    # And the classification must not outlive its callers.
    stale = set(_CALLER_KIND) - set(found)
    assert not stale, f"_CALLER_KIND names files that no longer call a reset: {sorted(stale)}"


def test_the_reset_never_empties_a_container_in_place():
    """`BRANCH-002`, statically.

    A cohort switch snapshots the live objects and then calls the reset to make
    room for the next branch. The reset assigns fresh containers — `= {}`,
    `= None`, `pop` — so the snapshot keeps the old ones. One `.clear()` here
    would empty the snapshot through its own reference and lose the branch that
    had just been banked, silently, with the archive still present and holding
    empty shells.
    """
    tree = ast.parse((ROOT / "utils" / "session_state.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "reset_downstream_results")
    in_place = [
        f"line {node.lineno}"
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"clear", "popitem"}
    ]
    assert not in_place, (
        f"reset_downstream_results empties a container in place at {in_place}. "
        "A cohort snapshot holds that same object; assign a fresh one instead.")


def test_the_alias_resolution_actually_finds_the_aliased_call():
    """Guard the guard.

    Page 01's exploratory toggle calls the reset as `_rdr` inside a nested
    function. Every assertion above is only worth what this scan sees, so the
    one call site that would defeat a naive scan is named here: if the scan
    stops finding it, the gates above have gone quiet rather than green.
    """
    calls = _reset_calls(ROOT / "pages" / "01_Upload_and_Audit.py")
    assert any(name == "reset_downstream_results" for name, _, _ in calls), (
        "the reset call inside page 01's exploratory-mode callback is no longer "
        "found — either it moved, or the alias resolution in _reset_calls broke")


# ── the behavioral gate: the archive actually goes ───────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        for key in (BRANCH_ARCHIVE_KEY, "raw_data", "filtered_data", "data_config",
                    "_raw_data_fingerprint", "cohort_run", "cohort_runs_done",
                    "trained_models", "model_results", "workflow_provenance",
                    "insight_ledger", "methodology_log", "test_lockbox",
                    "selected_features", "pre_fe_feature_cols"):
            st.session_state.pop(key, None)
    _wipe()
    yield
    _wipe()


@pytest.mark.parametrize("flags", [
    pytest.param({}, id="full-reset"),
    pytest.param({"clear_feature_engineering": False}, id="keep-FE"),
    pytest.param({"clear_feature_selection": False}, id="keep-selection"),
    pytest.param({"clear_feature_engineering": False,
                  "clear_feature_selection": False}, id="keep-both"),
])
def test_every_partial_reset_still_drops_every_branch(flags):
    """Partial invalidation is still invalidation of the question.

    `clear_feature_engineering=False` means "the recipe I just applied
    survives" — it does not mean "the women's fitted models survive", because
    they were fitted under the recipe that was just replaced.
    """
    st.session_state[BRANCH_ARCHIVE_KEY] = {("sex", "Female"): {"trained_models": {}}}
    reset_downstream_results(**flags)
    assert BRANCH_ARCHIVE_KEY not in st.session_state


def test_the_cohort_switch_may_keep_them():
    st.session_state[BRANCH_ARCHIVE_KEY] = {("sex", "Female"): {"trained_models": {}}}
    reset_downstream_results(clear_feature_engineering=True, preserve_branches=True)
    assert st.session_state[BRANCH_ARCHIVE_KEY].keys() == {("sex", "Female")}


def test_a_new_dataset_drops_them():
    st.session_state[BRANCH_ARCHIVE_KEY] = {("sex", "Female"): {"trained_models": {}}}
    reset_data_dependent_state()
    assert BRANCH_ARCHIVE_KEY not in st.session_state


def test_corrected_values_under_the_same_schema_drop_them():
    """`set_data`'s same-schema branch — the re-upload of cleaned data.

    The rows are the same people; the values are not. Every banked branch was
    fitted on numbers that no longer exist.
    """
    import numpy as np
    import pandas as pd
    from utils.session_state import set_data

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sex": rng.choice(["M", "F"], 200),
                       "age": rng.integers(20, 80, 200),
                       "y": rng.choice([0, 1], 200)})
    set_data(df)
    st.session_state[BRANCH_ARCHIVE_KEY] = {("sex", "F"): {"trained_models": {}}}

    corrected = df.copy()
    corrected.loc[corrected.index[:20], "age"] = 41
    set_data(corrected)
    assert BRANCH_ARCHIVE_KEY not in st.session_state


def test_the_reset_leaves_a_snapshot_of_its_own_objects_intact():
    """`BRANCH-002`, behaviorally: the snapshot is taken by reference.

    This is what makes `switch_branch` able to snapshot and then reset in that
    order. If the reset ever emptied a live container in place, this test sees
    it as the snapshot going empty — which is what the archive would silently
    hold.
    """
    live = {}
    for key in sorted(cascade.BRANCH_KEYS):
        st.session_state[key] = {"belongs_to": key}
        live[key] = st.session_state[key]

    reset_downstream_results()

    emptied = [k for k, obj in live.items() if obj != {"belongs_to": k}]
    assert not emptied, (
        f"the reset mutated these objects rather than replacing them: {emptied}. "
        "A branch snapshot holding them would come back empty.")


# ── the derived key set ──────────────────────────────────────────────────

def test_branch_keys_are_derived_from_the_graph_not_hand_listed():
    """The set a switch archives cannot drift from the set a reset clears.

    Hand-listing it is how a result key added on page 07 ends up in neither —
    cleared on a switch and never restored, so the women's SHAP is simply gone
    and nothing says so.
    """
    assert cascade.BRANCH_KEYS <= (cascade.all_result_keys() | cascade._FE_FRAME_STATE)
    assert not (cascade.BRANCH_KEYS & cascade.SHARED_DECISION_KEYS)


def test_what_was_fitted_is_per_branch_and_what_was_chosen_is_not():
    """The rule, spelled out on the keys that make it concrete.

    `preprocessing` owns both halves — a config the researcher chose and a
    pipeline fitted on one cohort's rows — which is why the boundary is a
    named list of decisions rather than a stage boundary.
    """
    for fitted in ("trained_models", "model_results", "fitted_estimators",
                   "preprocessing_pipeline", "X_train", "X_test", "shap_results",
                   "pdp_results", "permutation_importance", "eda_results",
                   "table1_df", "df_engineered", "filtered_data",
                   "external_validation_results", "manuscript_export_context"):
        assert fitted in cascade.BRANCH_KEYS, f"{fitted} was computed from rows"

    for chosen in ("preprocessing_config", "preprocessing_config_by_model",
                   "feature_selection_results", "consensus_features"):
        assert chosen not in cascade.BRANCH_KEYS, f"{chosen} is a decision"

    # And the decisions the cascade never touches are not in the graph at all,
    # so they cannot arrive here by accident.
    for never in ("data_config", "selected_features", "fe_recipe", "raw_data",
                  "test_lockbox", "cohort_runs_done", "cohort_run"):
        assert never not in cascade.BRANCH_KEYS


def test_the_archive_key_is_not_itself_a_branch_key():
    """It holds the branches; archiving it into a branch would nest forever."""
    assert BRANCH_ARCHIVE_KEY not in cascade.BRANCH_KEYS

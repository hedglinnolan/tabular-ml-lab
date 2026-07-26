"""L5 gate: the declared DAG reproduces both existing cascade implementations.

Two implementations exist today and they disagree:

1. `utils.session_state.reset_downstream_results` — the production function,
   three flags, called from three places.
2. `pages/03_Feature_Engineering.py:1262-1282` — a hand-rolled copy that clears
   nineteen keys inline. `TRANSITION_PLAN.md` §03 says it misses at least eleven,
   plus the ledger rollback and the provenance clearing.

`turbotab.cascade` declares the graph once. Before it can replace either, it has
to agree with the first exactly, and it has to *cover* the second while naming
what the second forgot. Nothing here re-implements a cascade: the production
function is run for real through a Streamlit script, and page 03's key list is
read out of its source by AST rather than transcribed.
"""
import ast
import os

import pytest

from turbotab import cascade

pytestmark = pytest.mark.timeout(300)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keys the production function clears that are not *results* of a stage. They
# are configuration the cascade happens to reset, or bookkeeping. Excluded from
# the comparison by name, so the exclusion is arguable rather than silent.
_NOT_STAGE_RESULTS = {
    # Restored from `pre_fe_feature_cols` rather than cleared; the FE branch
    # rewrites the selection instead of dropping it.
    "selected_features",
    "pre_fe_feature_cols",
    # `feature_engineering_applied` and `engineered_feature_names` are flags on
    # the FE stage, tested separately below.
    "feature_engineering_applied", "engineered_feature_names",
}


_PROBE = """
import streamlit as st, sys
sys.path.insert(0, __ROOT__)
from utils.session_state import reset_downstream_results

reset_downstream_results(**st.session_state["_flags"])
# "Cleared" means what the production function means by it: the key is
# absent, None, or reset to an EMPTY container. reset_downstream_results
# uses all three — `pop(k)`, `= None`, and `= {}` / `= []` — and reading
# only the first two would report a dozen keys as survivors.
def _cleared(k):
    if k not in st.session_state:
        return True
    v = st.session_state[k]
    if v is None:
        return True
    return isinstance(v, (dict, list, set, tuple)) and len(v) == 0

st.session_state["_survivors"] = sorted(
    k for k in st.session_state["_seeded"] if not _cleared(k))
"""


def _production_cleared(flags):
    """Which of the DAG's keys the real function actually clears."""
    from streamlit.testing.v1 import AppTest

    seeded = sorted(cascade.all_result_keys())
    # str.format is unusable here: the probe contains dict and set
    # literals, whose braces it would read as fields.
    at = AppTest.from_string(_PROBE.replace("__ROOT__", repr(ROOT)))
    for k in seeded:
        at.session_state[k] = f"seeded::{k}"
    at.session_state["_seeded"] = seeded
    at.session_state["_flags"] = flags
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return set(seeded) - set(at.session_state["_survivors"])


def _page03_cleared_keys():
    """Read page 03's inline cascade out of its source, by AST.

    Transcribing the list would mean this test compares the DAG to my copy of
    page 03 rather than to page 03.
    """
    path = os.path.join(ROOT, "pages", "03_Feature_Engineering.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    keys = set()
    for node in ast.walk(tree):
        # st.session_state["x"] = ...  /  st.session_state.pop("x", None)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)
                        and "session_state" in ast.dump(t.value)):
                    keys.add(t.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and "session_state" in ast.dump(node.func.value)
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


# ── graph sanity ─────────────────────────────────────────────────────────

def test_every_result_key_belongs_to_exactly_one_stage():
    """A key on two stages has two owners and no clear invalidation rule."""
    seen = {}
    for s in cascade.STAGES:
        for k in s.produces:
            assert k not in seen, f"{k!r} is produced by both {seen[k]} and {s.name}"
            seen[k] = s.name


def test_stages_are_declared_in_dependency_order():
    """`descendants()` walks STAGES once and relies on this."""
    position = {s.name: i for i, s in enumerate(cascade.STAGES)}
    for s in cascade.STAGES:
        for dep in s.depends_on:
            assert position[dep] < position[s.name], (
                f"{s.name} is declared before its dependency {dep}")


def test_the_graph_is_connected_to_data():
    for s in cascade.STAGES:
        if s.name == "data":
            continue
        assert s.name in cascade.descendants("data"), f"{s.name} is unreachable"


# ── gate 1 · agreement with the production function ──────────────────────

@pytest.mark.parametrize("flags", [
    pytest.param({}, id="full-reset"),
    pytest.param({"clear_feature_engineering": False}, id="keep-FE"),
    pytest.param({"clear_feature_selection": False}, id="keep-selection"),
    pytest.param({"clear_feature_engineering": False,
                  "clear_feature_selection": False}, id="keep-both"),
])
def test_the_dag_matches_the_production_cascade(flags):
    """Every flag combination, key for key.

    This is the gate. If it fails, the DAG is not yet a description of what the
    app does and must not replace it.
    """
    production = _production_cleared(flags)
    declared = cascade.keys_for_reset_downstream_results(
        clear_feature_engineering=flags.get("clear_feature_engineering", True),
        clear_feature_selection=flags.get("clear_feature_selection", True),
    )

    only_production = production - declared
    only_declared = declared - production
    assert not only_production, (
        f"the real cascade clears keys the DAG does not: {sorted(only_production)}")
    assert not only_declared, (
        f"the DAG would clear keys the real cascade leaves: {sorted(only_declared)}")


def test_keeping_a_stage_does_not_keep_its_descendants():
    """Feature Selection applying a new selection keeps the selection and drops
    everything built on the old feature set. That asymmetry is the point."""
    kept = cascade.keys_for_reset_downstream_results(clear_feature_selection=False)
    assert "feature_selection_results" not in kept
    assert "consensus_features" not in kept
    for downstream in ("preprocessing_pipeline", "X_train", "trained_models"):
        assert downstream in kept, f"{downstream} survived a selection change"


# ── gate 2 · covering page 03, and naming what it forgot ─────────────────

def test_the_dag_covers_page_03s_hand_rolled_cascade():
    page03 = _page03_cleared_keys()
    declared = cascade.keys_to_clear("data")
    # Page 03 touches session-state keys that are not results (widget lists,
    # its own scratch). Only the result keys are in scope.
    page03_results = page03 & cascade.all_result_keys()

    assert page03_results, "the AST scan found no cascade in page 03 — it moved"
    uncovered = page03_results - declared
    assert not uncovered, (
        f"page 03 clears result keys the DAG does not: {sorted(uncovered)}")


def test_page_03_misses_keys_the_dag_would_clear():
    """Records the gap rather than asserting a count.

    `TRANSITION_PLAN.md` §03 says page 03 misses at least eleven keys. This is
    the live list — the to-do for reconciling the two, and the reason the DAG is
    worth having.
    """
    missed = cascade.missing_from(_page03_cleared_keys())
    assert len(missed) >= 10, (
        "page 03's cascade now covers nearly everything the DAG does — if it was "
        f"reconciled, update this test. Currently missing: {sorted(missed)}")
    # The specific ones the transition plan named.
    for named in ("cv_results", "dataset_profile", "eda_results",
                  "train_indices", "target_transformer", "cv_strategy",
                  "baseline_results", "calibration_results", "bootstrap_results"):
        assert named in missed, (
            f"{named} is no longer missing from page 03 — the plan's list is stale")


def test_provenance_sections_follow_the_same_graph():
    """A Methods section describing work that no longer exists asserts something
    false, so the record's sections are invalidated with the stages that wrote
    them."""
    full = cascade.provenance_sections_to_clear("data")
    for sec in ("eda", "split", "preprocessing", "training", "explainability", "coach"):
        assert sec in full, f"provenance section {sec!r} is never cleared"

    kept = cascade.provenance_sections_to_clear("data", keep={"feature_selection"})
    assert "feature_selection" not in kept
    assert "split" in kept


# ── partial invalidation from a mid-graph change ──────────────────────────

def test_a_change_at_preprocessing_leaves_upstream_alone():
    keys = cascade.keys_to_clear("preprocessing")
    assert "X_train" in keys and "trained_models" in keys
    assert "df_engineered" not in keys, "an upstream stage was invalidated"
    assert "feature_selection_results" not in keys
    # Descriptive work over the data is not downstream of preprocessing.
    assert "dataset_profile" not in keys


def test_a_change_at_training_does_not_redo_the_split():
    keys = cascade.keys_to_clear("training")
    assert "shap_results" in keys and "report_data" in keys
    assert "X_train" not in keys, "the split was invalidated by a training change"

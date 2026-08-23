"""Characterization tests for the invalidation cascade — the production function.

`TRANSITION_PLAN.md` §03 puts it plainly: *no test calls the production
`reset_downstream_results()`. Three separate re-implementations test
themselves.* The single most important behavior to preserve has, today, zero
real coverage — while three copies of it are green.

So this file calls the real function. It runs inside a Streamlit script through
`AppTest.from_string`, because `reset_downstream_results` writes to
`st.session_state` and there is no honest way to exercise it without one. A mock
session-state dict would be a fourth re-implementation.

Two things are pinned:

1. **Which keys the cascade clears.** Seed every downstream key, run it, record
   what survives. `pages/03` hand-rolls its own version that misses at least
   eleven of these (`TRANSITION_PLAN.md` §03) — this is the list that hand-rolled
   version has to be reconciled against.
2. **That partial invalidation is real.** `clear_feature_engineering=False` and
   `clear_feature_selection=False` are live call sites, so a naive
   full-cascade DAG cannot replace this function. Each flag is pinned separately.

**Gate for L4:** deliberately break the cascade and this file goes red. See
`test_the_cascade_guard_would_catch_a_removed_key`, which does exactly that by
running a mutated copy of the function.

Run:  venv/bin/python -m pytest tests/integration/test_characterization_cascade.py -v
"""
import pytest

pytestmark = pytest.mark.timeout(300)


# Every key the production cascade is expected to clear on a full reset. Read
# off `utils/session_state.py::reset_downstream_results` and then *verified by
# running it* — the test below fails if the function stops clearing any of them.
FULL_RESET_KEYS = [
    # feature engineering
    "df_engineered", "engineering_log",
    # pipelines
    "preprocessing_pipeline", "preprocessing_config",
    "preprocess_built_model_keys",
    # splits and targets
    "X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
    "feature_names", "train_indices", "val_indices", "test_indices",
    # Row LABELS, added with the split extraction (L6) and pinned here
    # afterwards: they name the same partition the *_indices keys name, so a
    # cascade that cleared the positions and kept the labels would leave page 07
    # resolving the held-out rows of a split that no longer exists.
    "train_row_labels", "val_row_labels", "test_row_labels",
    "split_config", "target_transformer", "target_label_encoder",
    "y_train_original", "y_val_original", "y_test_original",
    "cv_strategy", "cv_groups_train",
    # The row FILTER, reclassified from PRESERVED_KEYS — see below.
    "filtered_data",
    # models and metrics
    "cv_results",
    # analysis results
    "shap_results", "shap_matplotlib_figs", "bootstrap_results",
    "baseline_results", "calibration_results", "sensitivity_seed_results",
    "hypothesis_test_results", "table1_df", "table1_metadata",
    "custom_table1_tests", "table1_custom_test_footnotes", "dataset_profile",
    # Manuscript-facing results that used to be in NEITHER of the function's
    # inline key lists (`STATE-038`): the export read them while the reset that
    # destroyed the model they came from left them standing. They are now in the
    # registry the function clears from, and pinned here so a key can only leave
    # that registry deliberately.
    "pdp_results", "sensitivity_dropout_results", "sensitivity_dropout_baseline",
    "bland_altman_results", "preprocessing_summary",
    # report artifacts
    "report_data", "methods_section", "flow_diagram", "tripod_tracker",
    "latex_report", "report_best_model", "report_model_selection",
    "report_explain_selection", "report_include_results", "report_include_llm",
    "manuscript_context", "manuscript_export_context", "compiled_pdf",
    "manuscript_table1_df", "manuscript_table1_metadata",
    # coach evidence
    "coach_probe_result", "_coach_applied",
]

# Keys that survive only because a flag says so.
FEATURE_SELECTION_KEYS = ["feature_selection_results", "consensus_features"]

# Keys the cascade must NOT touch: configuration, not results.
#
# `filtered_data` was on this list and has been RECLASSIFIED into
# FULL_RESET_KEYS. It was pinned here when it was inert: `get_data()` let
# `df_engineered` shadow it, so a stale filter changed nothing and leaving it
# alone was harmless. `STATE-037` changed that — `get_data()` now masks the
# active frame's ROWS by `filtered_data` whenever it exists, in either page
# order. A filter left over from a superseded preprocessing config therefore
# silently changes who every downstream number describes, which makes it
# data-dependent state and not configuration. Popping it is what keeps the
# cascade honest; the pin, not the fix, was the stale half.
PRESERVED_KEYS = ["raw_data", "data_config", "task_mode",
                  "random_seed", "test_lockbox", "exploratory_mode"]


_SCRIPT = """
import streamlit as st
import sys
sys.path.insert(0, {root!r})
from utils.session_state import reset_downstream_results

flags = st.session_state["_flags"]
reset_downstream_results(**flags)

# A key counts as cleared if it is absent OR present-but-None: the function
# uses both `pop` and `= None`, and both mean "this result is gone".
st.session_state["_survivors"] = sorted(
    k for k in st.session_state["_seeded"]
    if k in st.session_state and st.session_state[k] is not None
)
"""


def _run_cascade(seed_keys, flags=None, extra_seed=None):
    """Seed session state, call the real cascade, return what survived."""
    import os
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    at = AppTest.from_string(_SCRIPT.format(root=root))

    for k in seed_keys:
        at.session_state[k] = f"seeded::{k}"
    for k, v in (extra_seed or {}).items():
        at.session_state[k] = v
    at.session_state["_seeded"] = list(seed_keys) + list((extra_seed or {}).keys())
    at.session_state["_flags"] = flags or {}
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return set(at.session_state["_survivors"])


# ── what a full reset clears ─────────────────────────────────────────────

def test_full_reset_clears_every_downstream_key():
    """The list the hand-rolled cascade in `pages/03` has to match."""
    survivors = _run_cascade(FULL_RESET_KEYS + FEATURE_SELECTION_KEYS)
    assert survivors == set(), (
        f"the cascade left {sorted(survivors)} behind — either the function "
        "stopped clearing them, or this list is out of date. Do not delete the "
        "assertion; find out which.")


def test_the_cascade_leaves_configuration_alone():
    """Invalidation clears *results*, not the answers that produced them."""
    survivors = _run_cascade(PRESERVED_KEYS)
    assert survivors == set(PRESERVED_KEYS), (
        f"the cascade destroyed configuration: {sorted(set(PRESERVED_KEYS) - survivors)}")


def test_the_row_filter_is_cleared_under_every_flag():
    """`filtered_data` is who the results are about, so it goes with them.

    The reclassification out of PRESERVED_KEYS, asserted rather than merely
    removed. It has to hold under the PARTIAL calls too: page 03's save and page
    01's exploratory toggle both pass `clear_feature_engineering=False`, and a
    filter that survived those would keep masking the frame every later stage is
    computed on while the page that set it is gone.
    """
    for flags in ({},
                  {"clear_feature_engineering": False},
                  {"clear_feature_selection": False},
                  {"clear_feature_engineering": False,
                   "clear_feature_selection": False}):
        survivors = _run_cascade(["filtered_data"], flags=flags)
        assert survivors == set(), (
            f"the row filter survived reset_downstream_results(**{flags}) — "
            "get_data() masks by it, so every downstream number would describe "
            "a set of people chosen by a superseded config")


# ── partial invalidation is a real call ──────────────────────────────────

def test_feature_selection_survives_when_its_flag_says_so():
    """`clear_feature_selection=False` is what Feature Selection calls when it
    *applies* a selection: everything built on the old feature set is stale, but
    the selection just made — and its record — must survive.

    This is the case that stops a naive full-cascade DAG from replacing the
    function (`ARCHITECTURE.md` §02).
    """
    kept = _run_cascade(FULL_RESET_KEYS + FEATURE_SELECTION_KEYS,
                        flags={"clear_feature_selection": False})
    assert set(FEATURE_SELECTION_KEYS) <= kept, (
        "clear_feature_selection=False did not preserve the selection")
    assert not (kept & set(FULL_RESET_KEYS)), (
        "the flag preserved more than the selection")


def test_feature_engineering_survives_when_its_flag_says_so():
    kept = _run_cascade(FULL_RESET_KEYS, flags={"clear_feature_engineering": False})
    assert "df_engineered" in kept, (
        "clear_feature_engineering=False still dropped the engineered frame")
    assert "X_train" not in kept, "the rest of the cascade did not run"


def test_restore_pre_fe_features_puts_the_old_selection_back():
    """The FE branch restores `selected_features` from `pre_fe_feature_cols`, so
    the config refers to columns that still exist after the engineered ones go.
    """
    import os
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    at = AppTest.from_string(_SCRIPT.format(root=root))
    at.session_state["pre_fe_feature_cols"] = ["age", "bmi"]
    at.session_state["selected_features"] = ["age", "bmi", "age_x_bmi"]
    at.session_state["_seeded"] = ["selected_features"]
    at.session_state["_flags"] = {}
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["selected_features"] == ["age", "bmi"], (
        "the engineered column was left in the selection after FE was cleared")


# ── the gate ─────────────────────────────────────────────────────────────

def test_the_cascade_guard_would_catch_a_removed_key():
    """The L4 gate, executed rather than asserted.

    "Deliberately break the cascade and the suite goes red" is only a
    meaningful gate if someone checks. This runs a *mutated* copy of
    `reset_downstream_results` — one line removed, so `cv_results` is no longer
    cleared — and asserts the check above would have caught it.

    Nothing in the repository is modified: the mutation is applied to a copy of
    the source, executed in its own module.
    """
    import os
    import re
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = open(os.path.join(root, "utils", "session_state.py"), encoding="utf-8").read()

    mutated, n = re.subn(r"^(\s*)st\.session_state\.cv_results = None\s*$",
                         r"\1pass  # MUTATED: cascade no longer clears cv_results",
                         src, count=1, flags=re.M)
    assert n == 1, ("the mutation target moved — this gate is now checking "
                    "nothing, which is the failure mode it exists to prevent")

    script = f"""
import streamlit as st, sys, types
sys.path.insert(0, {root!r})
import utils                      # real package, so relative imports resolve
mod = types.ModuleType("mutated_session_state")
mod.__dict__["__file__"] = {os.path.join(root, "utils", "session_state.py")!r}
exec(compile({mutated!r}, "mutated_session_state", "exec"), mod.__dict__)

mod.reset_downstream_results()
st.session_state["_survivors"] = sorted(
    k for k in st.session_state["_seeded"]
    if k in st.session_state and st.session_state[k] is not None
)
"""
    at = AppTest.from_string(script)
    for k in FULL_RESET_KEYS:
        at.session_state[k] = f"seeded::{k}"
    at.session_state["_seeded"] = list(FULL_RESET_KEYS)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    survivors = set(at.session_state["_survivors"])
    assert "cv_results" in survivors, (
        "the mutation did not survive the cascade, so this gate proves nothing")
    # And that is exactly what the real test asserts against.
    assert survivors != set(), "the guard would not have gone red"

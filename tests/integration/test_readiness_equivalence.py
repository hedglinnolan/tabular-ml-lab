"""L6, second half: the step-completion model, extracted and pinned.

`utils/theme.py:685 render_sidebar_workflow` held the only implementation of the
app's step model — ten predicates over session state — plus the quick/advanced
split. `TRANSITION_PLAN.md` §02.4: *delete `theme.py` as "just styling" during
the cut and the step model goes with it.* `ARCHITECTURE.md` §05 adds what it
actually is: **the Router's readiness function, filed under CSS.**

The ten predicates below are transcribed from `theme.py` as it stood before the
extraction, and they are the reference `turbotab.readiness` must reproduce. A
transcription is weaker evidence than driving the real code, which is why the
split block was done the other way — but these are single-expression predicates
over session-state keys, quoted here verbatim, and the second test in this file
checks the page no longer carries its own copy.
"""
import itertools
import os
import re

import pytest

from turbotab import readiness

pytestmark = pytest.mark.timeout(300)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Verbatim from utils/theme.py before the extraction. `get_data() is not None`
# is represented by the `working_data` key, which is what the adapter passes.
def _reference(s):
    return {
        "upload":      s.get("working_data") is not None,
        "eda":         (s.get("data_config") is not None
                        and getattr(s.get("data_config"), "target_col", None) is not None),
        "features":    s.get("feature_selection_results") is not None,
        "preprocess":  s.get("preprocessing_pipeline") is not None,
        "train":       bool(s.get("trained_models")),
        "explain":     (len(s.get("permutation_importance", {}) or {}) > 0
                        or len(s.get("shap_results", {}) or {}) > 0),
        "sensitivity": s.get("sensitivity_seed_results") is not None,
        "report":      s.get("report_data") is not None,
        "stats":       (s.get("hypothesis_test_results") is not None
                        or len(s.get("custom_table1_tests", []) or []) > 0),
        "engineering": bool(s.get("feature_engineering_applied", False)),
    }


class _Cfg:
    def __init__(self, target_col=None):
        self.target_col = target_col


# A spread of states, including the empties that distinguish `is not None` from
# truthiness — `trained_models = {}` is "no models", `permutation_importance = {}`
# is "not run", and a predicate that confuses the two reports a step complete
# when nothing happened.
_STATES = [
    {},
    {"working_data": object()},
    {"working_data": object(), "data_config": _Cfg("glucose")},
    {"data_config": _Cfg(None)},
    {"trained_models": {}},
    {"trained_models": {"rf": object()}},
    {"permutation_importance": {}},
    {"permutation_importance": {"rf": 1}},
    {"shap_results": {"rf": 1}},
    {"custom_table1_tests": []},
    {"custom_table1_tests": ["t"]},
    {"hypothesis_test_results": {}},
    {"feature_engineering_applied": False},
    {"feature_engineering_applied": True},
    {"feature_selection_results": {}, "preprocessing_pipeline": object(),
     "sensitivity_seed_results": {}, "report_data": {}},
]


@pytest.mark.parametrize("state", _STATES, ids=range(len(_STATES)))
def test_the_extracted_predicates_match_the_page(state):
    """Ten predicates, every state, key for key."""
    got = readiness.assess(state).completed
    assert got == _reference(state), (
        f"readiness disagrees with theme.py for {state!r}")


def test_the_page_no_longer_carries_its_own_step_model():
    """The extraction, checked structurally.

    A second implementation is the failure mode Decision C names: not two UIs,
    but two implementations that drift until they disagree and nobody can say
    which is right. `theme.py` must now ask, not compute.
    """
    src = open(os.path.join(ROOT, "utils", "theme.py"), encoding="utf-8").read()
    assert "turbotab.readiness" in src or "from turbotab import readiness" in src, (
        "theme.py does not use the shared readiness model")

    # The inline predicate expressions must be gone, not merely unused.
    for expression in (
        "st.session_state.get('feature_selection_results') is not None",
        "st.session_state.get('preprocessing_pipeline') is not None",
        "bool(st.session_state.get('trained_models'))",
        "st.session_state.get('report_data') is not None",
    ):
        assert expression not in src, (
            f"theme.py still computes a step predicate itself: {expression}")


def test_quick_and_advanced_keep_the_same_membership_as_the_page():
    """The disclosure split, which decides which questions are optional."""
    core = {s.key for s in readiness.CORE_STEPS}
    advanced = {s.key for s in readiness.ADVANCED_STEPS}
    assert core == {"upload", "eda", "features", "preprocess",
                    "train", "explain", "report"}
    assert advanced == {"engineering", "sensitivity", "stats"}
    assert not (core & advanced) and len(core | advanced) == 10


def test_progress_counts_every_step_not_just_the_visible_ones():
    """`theme.py` reports `completed/len(all_items)` over core + advanced."""
    r = readiness.assess({"working_data": object()})
    assert r.n_total == 10
    assert r.n_complete == 1
    assert abs(r.progress - 0.1) < 1e-9


def test_next_step_walks_the_core_path():
    r = readiness.assess({"working_data": object()})
    assert r.next_step().key == "eda"
    r = readiness.assess({"working_data": object(), "data_config": _Cfg("y")})
    assert r.next_step().key == "features"


def test_a_step_is_blocked_until_the_ones_before_it_are_done():
    """The half of a readiness model a Router actually needs: not only what is
    done, but what may be asked."""
    blocked = {s.key for s in readiness.assess({}).blocked_steps()}
    assert "train" in blocked and "report" in blocked
    assert "upload" not in blocked

    done_all_core = {
        "working_data": object(), "data_config": _Cfg("y"),
        "feature_selection_results": {}, "preprocessing_pipeline": object(),
        "trained_models": {"rf": 1}, "permutation_importance": {"rf": 1},
        "report_data": {},
    }
    assert readiness.assess(done_all_core).blocked_steps() == []


def test_readiness_imports_without_streamlit():
    src = open(os.path.join(ROOT, "turbotab", "readiness.py"), encoding="utf-8").read()
    assert "streamlit" not in src

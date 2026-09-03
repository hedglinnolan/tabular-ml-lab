"""Pages 07 and 08 let you choose which group they are about.

Before branches there was nothing to pick between: the previous cohort's work
did not exist. Now it does, and the control that reaches it has to satisfy two
things that pull against each other.

It must render ABOVE the "train a model first" gate, because a researcher
standing in an untrained branch is precisely the person who needs to switch to
a trained one, and a control rendered after `st.stop()` is invisible to them.
On page 07 that means above the TASK-MODE gate too — it stops the script four
lines earlier, and "above the trained_models gate" would still be dead there.

And picking must *be* the switch, not a display filter. If the page could show
one cohort's SHAP while `active_cohort()` named another, the sidebar chip, the
export, `get_data()` and the provenance record would all disagree with the
screen — and the manuscript is generated from the ones the researcher cannot
see.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from turbotab.cascade import BRANCH_ARCHIVE_KEY
from utils.cohorts import (
    EVERYONE, active_cohort, branch_has_models, branch_key, branch_label,
    known_branches, plan_cohorts, switch_branch,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

_WIPE = (BRANCH_ARCHIVE_KEY, "cohort_run", "cohort_runs_done", "raw_data",
         "filtered_data", "data_config", "_raw_data_fingerprint", "test_lockbox",
         "trained_models", "model_results", "_cohort_filter_broken",
         "branch_pick_07_Explainability", "_branch_pick_07_Explainability_seeded_for",
         "branch_pick_08_Sensitivity_Analysis",
         "_branch_pick_08_Sensitivity_Analysis_seeded_for")


@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        for key in _WIPE:
            st.session_state.pop(key, None)
    _wipe()
    yield
    _wipe()


def study(n=400, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["Male", "Female"], n),
        "age": rng.integers(20, 80, n),
        "y": rng.choice([0, 1], n, p=[0.5, 0.5]),
    })


def target_for(df, label):
    plan = plan_cohorts(df, "sex", "y", "classification")
    cell = next(c for c in plan.viable if c.label == label)
    return (df, plan, cell, "y", [])


def two_branches(train_female=True):
    """A session with Female (trained) and Male (not) archived."""
    from utils.session_state import DataConfig
    df = study()
    st.session_state["raw_data"] = df
    st.session_state["data_config"] = DataConfig(
        target_col="y", feature_cols=["age"], task_type="classification")
    switch_branch(target_for(df, "Female"))
    if train_female:
        st.session_state["trained_models"] = {"ridge": object()}
    switch_branch(target_for(df, "Male"))
    return df


def render(page_key="07_Explainability"):
    """Render the picker, capturing the selectbox it draws."""
    from utils.cohort_ui import render_branch_picker
    drawn = {}

    def fake_selectbox(label, options, **kw):
        drawn["label"] = label
        drawn["options"] = list(options)
        key = kw.get("key")
        return st.session_state.get(key, options[0] if options else None)

    real = st.selectbox
    st.selectbox = fake_selectbox
    try:
        render_branch_picker(page_key)
    finally:
        st.selectbox = real
    return drawn


# ── where it renders ─────────────────────────────────────────────────────

def _call_line(path, func):
    """The line a top-level call to `func` sits on, by AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == func):
            return node.lineno
    return None


def _stop_lines(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "stop")


@pytest.mark.parametrize("page,key", [
    ("pages/07_Explainability.py", "07_Explainability"),
    ("pages/08_Sensitivity_Analysis.py", "08_Sensitivity_Analysis"),
])
def test_the_picker_renders_before_the_page_can_stop(page, key):
    """Above EVERY `st.stop()`, not just the trained-models one.

    Page 07's task-mode gate stops the script five lines before its
    trained-models gate. A picker placed "above the trained_models gate" would
    be dead in non-prediction mode — a state a confused researcher lands in
    often, and one the picker cannot help them out of if it never renders.
    """
    path = ROOT / page
    at = _call_line(path, "render_branch_picker")
    assert at is not None, f"{page} does not render the branch picker"
    stops = _stop_lines(path)
    assert stops, f"{page} has no gates at all — check this test still means something"
    assert at < stops[0], (
        f"the picker is at line {at}, below the first st.stop() at {stops[0]}")


def test_page_08_states_which_group_its_numbers_describe():
    """Its only statement of scope was the lockbox chip, which says how many
    rows were held out — not who they are."""
    src = (ROOT / "pages/08_Sensitivity_Analysis.py").read_text(encoding="utf-8")
    assert "render_cohort_note(" in src


# ── what it offers ───────────────────────────────────────────────────────

class TestWhatThePickerOffers:

    def test_nothing_at_all_until_there_is_a_second_branch(self):
        """One branch is not a choice, and a disabled control that explains a
        feature nobody asked for is clutter on every page."""
        from utils.session_state import DataConfig
        df = study()
        st.session_state["raw_data"] = df
        st.session_state["data_config"] = DataConfig(
            target_col="y", feature_cols=["age"], task_type="classification")
        assert render() == {}, "the picker drew a control with one branch"

        switch_branch(target_for(df, "Female"))
        assert render()["options"], "two branches and still no picker"

    def test_it_names_every_branch_and_whether_it_has_models(self):
        two_branches()
        drawn = render()
        joined = " | ".join(drawn["options"])
        assert "sex = Female" in joined and "sex = Male" in joined
        assert "Everyone" in joined, "the whole-study branch is a destination too"
        assert "trained" in joined and "not yet trained" in joined, joined

    def test_the_trained_flag_is_read_per_branch_not_from_the_live_keys(self):
        two_branches(train_female=True)
        assert branch_has_models(("sex", "Female")) is True
        assert branch_has_models(("sex", "Male")) is False
        drawn = render()
        female = next(o for o in drawn["options"] if "Female" in o)
        male = next(o for o in drawn["options"] if "Male" in o)
        assert "(trained)" in female, female
        assert "not yet trained" in male, male

    def test_it_opens_on_the_branch_that_is_actually_active(self):
        two_branches()
        render()
        shown = st.session_state["branch_pick_07_Explainability"]
        assert shown.startswith(branch_label(("sex", "Male"))), shown

    def test_it_reseeds_when_the_branch_changes_underneath_it(self):
        """The sidebar chooser and page 06's button both switch too. A widget
        holding the previous branch would show the wrong group as selected and
        switch back to it on the next interaction."""
        df = two_branches()
        render()
        assert "Male" in st.session_state["branch_pick_07_Explainability"]

        switch_branch(target_for(df, "Female"))     # switched from elsewhere
        render()
        assert "Female" in st.session_state["branch_pick_07_Explainability"]

    def test_each_page_keeps_its_own_widget_key(self):
        """One key shared across pages would make Streamlit's widget cull on a
        page change look like a deliberate switch."""
        two_branches()
        render("07_Explainability")
        render("08_Sensitivity_Analysis")
        assert "branch_pick_07_Explainability" in st.session_state
        assert "branch_pick_08_Sensitivity_Analysis" in st.session_state


# ── picking is switching ─────────────────────────────────────────────────

class TestPickingIsSwitching:

    def _pick(self, page_key, wanted):
        """Drive the picker as if the researcher chose `wanted`."""
        from utils.cohort_ui import render_branch_picker
        switched = {}

        def fake_selectbox(label, options, **kw):
            key = kw.get("key")
            hit = next(o for o in options if wanted in o)
            st.session_state[key] = hit
            return hit

        def fake_rerun(*a, **k):
            switched["reran"] = True

        real_sb, real_rerun = st.selectbox, st.rerun
        st.selectbox, st.rerun = fake_selectbox, fake_rerun
        try:
            render_branch_picker(page_key)
        finally:
            st.selectbox, st.rerun = real_sb, real_rerun
        return switched

    def test_picking_a_group_makes_it_the_active_cohort(self):
        """Not a display filter. `active_cohort()` is what the sidebar chip,
        `get_data()`, the provenance record and the export all read."""
        two_branches()
        assert active_cohort()["label"] == "Male"

        self._pick("07_Explainability", "Female")
        assert active_cohort()["label"] == "Female"
        assert branch_key(active_cohort()) == ("sex", "Female")

    def test_picking_a_trained_group_restores_its_models(self):
        two_branches(train_female=True)
        assert not st.session_state.get("trained_models"), "Male has none"

        self._pick("07_Explainability", "Female")
        assert st.session_state.get("trained_models"), (
            "the picker switched but the branch was not restored")

    def test_picking_an_untrained_group_still_switches(self):
        """It then stops at the page's own gate, which says where to go. A
        picker that refused would leave no way to reach the group at all."""
        df = two_branches(train_female=True)
        switch_branch(target_for(df, "Female"))     # stand in the trained one
        assert st.session_state.get("trained_models")

        self._pick("07_Explainability", "Male")
        assert active_cohort()["label"] == "Male"
        assert not st.session_state.get("trained_models")

    def test_picking_everyone_ends_the_restriction(self):
        two_branches()
        self._pick("07_Explainability", "Everyone")
        assert active_cohort() is None

    def test_picking_the_branch_already_active_does_nothing(self):
        """A no-op switch would archive and restore on every rerun of the page,
        and every rerun would call st.rerun()."""
        two_branches()
        switched = self._pick("07_Explainability", "Male")
        assert not switched, "the picker reran on a selection that did not change"

    def test_the_rows_follow_the_pick(self):
        from utils.session_state import get_data
        two_branches()
        self._pick("07_Explainability", "Female")
        seen = get_data()
        assert set(seen["sex"].unique()) == {"Female"}


# ── the order it offers them in ──────────────────────────────────────────

def test_everyone_is_offered_last():
    """It is not one of the groups being compared, and putting it first would
    make the default read as 'no cohort' on a page about one."""
    two_branches()
    assert known_branches()[-1] == EVERYONE

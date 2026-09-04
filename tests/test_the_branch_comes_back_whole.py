"""Switching cohorts archives a branch; switching back restores it whole.

The old switch was destructive by construction: it reset everything below the
shared decisions and rebuilt from them, so what the women's run produced —
models, SHAP, the sensitivity sweep, the Methods draft — was gone the moment
the men's run began. A `CohortRun` holding a label, two row counts and one
metric was the only trace.

These tests pin the replacement. Four slices have to travel with a branch, and
they are keyed four different ways:

    session keys        named by `cascade.BRANCH_KEYS`, derived from the graph
    ledger insights     keyed by the PAGE that produced them
    methodology entries keyed by a step name that maps to a page
    provenance sections attributes on one dataclass

Miss any one and the failure is silent and specific: the men's `report.md`
prints the women's explainability findings, or the Methods describes a split
that no longer exists.

The other half of the contract is what must NOT travel. A decision — a recipe,
a config, a selection — is the study's, not the branch's, and restoring one
would silently change the question under every other branch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from turbotab import cascade
from turbotab.cascade import BRANCH_ARCHIVE_KEY
from utils.cohorts import (
    EVERYONE, active_cohort, archive_current, branch_key, branch_has_models,
    known_branches, plan_cohorts, snapshot_current, switch_branch,
)

_WIPE = (
    BRANCH_ARCHIVE_KEY, "cohort_run", "cohort_runs_done", "_cohort_filter_broken",
    "raw_data", "filtered_data", "data_config", "_raw_data_fingerprint",
    "test_lockbox", "insight_ledger", "methodology_log", "workflow_provenance",
    "cohort_replay_pending", "cohort_decisions_pending", "selected_features",
    "pre_fe_feature_cols", "exploratory_mode", "fe_recipe",
)


@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        for key in _WIPE:
            st.session_state.pop(key, None)
        for key in cascade.BRANCH_KEYS:
            st.session_state.pop(key, None)
    _wipe()
    yield
    _wipe()


def study(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["Male", "Female"], n),
        "smoker": rng.choice(["never", "former", "current"], n),
        "age": rng.integers(20, 80, n),
        "bmi": rng.normal(28, 5, n),
        "diabetes": rng.choice([0, 1], n, p=[0.5, 0.5]),
    })


def target_for(df, label, column="sex", target="diabetes"):
    """The `switch_branch` payload for one group."""
    plan = plan_cohorts(df, column, target, "classification")
    cell = next(c for c in plan.viable if c.label == label)
    return (df, plan, cell, target, [])


def fit_a_branch(tag):
    """Stand in for a completed run: a fit, a frame, a score, a figure."""
    st.session_state["trained_models"] = {f"ridge_{tag}": object()}
    st.session_state["model_results"] = {f"ridge_{tag}": {"metrics": {"auc": 0.5}}}
    st.session_state["X_test"] = pd.DataFrame({"age": [1, 2, 3]})
    st.session_state["shap_results"] = {f"ridge_{tag}": [tag]}
    st.session_state["df_engineered"] = pd.DataFrame({"age": [1, 2, 3]})
    st.session_state["report_data"] = {"who": tag}


# ── the round trip ───────────────────────────────────────────────────────

class TestTheRoundTrip:

    def test_female_male_female_returns_the_same_objects(self):
        """Not equal values — the SAME objects.

        The archive holds live references on purpose (`snapshot_current`), so a
        page that adds a model while a branch is active updates that branch's
        archive too. Identity is the assertion that proves the snapshot was not
        quietly round-tripped through a serializer.
        """
        df = study()
        st.session_state["raw_data"] = df

        switch_branch(target_for(df, "Female"))
        fit_a_branch("female")
        female = {k: st.session_state[k] for k in
                  ("trained_models", "model_results", "shap_results", "report_data")}

        switch_branch(target_for(df, "Male"))
        assert st.session_state.get("trained_models") in ({}, None), (
            "the men inherited the women's models")

        fit_a_branch("male")
        male_models = st.session_state["trained_models"]

        switch_branch(target_for(df, "Female"))
        for key, obj in female.items():
            assert st.session_state[key] is obj, f"{key} came back as a different object"
        assert st.session_state["trained_models"] is not male_models

    def test_every_branch_key_makes_the_trip(self):
        """The whole derived set, not the handful a test happens to name.

        A key that is archived but never restored is a result that silently
        disappears on the way back; one restored but never archived is the
        previous branch's number under this branch's heading.
        """
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))

        planted = {}
        for key in sorted(cascade.BRANCH_KEYS):
            planted[key] = {"belongs_to": key}
            st.session_state[key] = planted[key]

        switch_branch(target_for(df, "Male"))
        switch_branch(target_for(df, "Female"))

        missing = [k for k, obj in planted.items() if st.session_state.get(k) is not obj]
        assert not missing, f"these keys did not survive the round trip: {missing}"

    def test_a_key_absent_when_archived_is_absent_when_restored(self):
        """A restore reproduces a state, it does not approximate one.

        Pages read `model_results` and `eda_results` by bare attribute access.
        Restoring a branch that never had `shap_results` must leave it absent
        rather than inventing an empty one, because absent is what it was when
        that branch was a working state.
        """
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        st.session_state.pop("shap_results", None)

        switch_branch(target_for(df, "Male"))
        st.session_state["shap_results"] = {"ridge": ["male"]}

        switch_branch(target_for(df, "Female"))
        assert "shap_results" not in st.session_state

    def test_everyone_is_a_branch_too(self):
        """Going back to the whole study restores it rather than emptying the app."""
        df = study()
        st.session_state["raw_data"] = df
        fit_a_branch("everyone")
        whole_study = st.session_state["trained_models"]

        switch_branch(target_for(df, "Female"))
        fit_a_branch("female")
        assert st.session_state["trained_models"] is not whole_study

        switch_branch(None)
        assert active_cohort() is None
        assert st.session_state["trained_models"] is whole_study

    def test_the_first_visit_to_a_group_builds_it_and_the_second_restores_it(self):
        df = study()
        st.session_state["raw_data"] = df
        assert switch_branch(target_for(df, "Female")) is False, "first visit is a build"
        switch_branch(target_for(df, "Male"))
        assert switch_branch(target_for(df, "Female")) is True, "second visit is a restore"


# ── what must NOT travel ─────────────────────────────────────────────────

class TestTheDecisionsStayWithTheStudy:

    def test_a_shared_decision_is_never_archived(self):
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        for key in cascade.SHARED_DECISION_KEYS:
            st.session_state[key] = {"chosen": "once"}
        chosen = {k: st.session_state[k] for k in cascade.SHARED_DECISION_KEYS}

        snap = snapshot_current()
        assert not (set(snap.keys) & cascade.SHARED_DECISION_KEYS)

        switch_branch(target_for(df, "Male"))
        # The new-branch path runs the ordinary reset, which clears the
        # selection results as it always did — what matters is that switching
        # BACK does not resurrect a per-branch copy of them.
        switch_branch(target_for(df, "Female"))
        for key, obj in chosen.items():
            assert st.session_state.get(key) is not obj, (
                f"{key} was restored per-branch; it is a decision, not a measurement")

    def test_the_selection_itself_survives_every_switch(self):
        """`selected_features` is the question. It must be identical in both runs."""
        df = study()
        st.session_state["raw_data"] = df
        st.session_state["selected_features"] = ["age", "bmi"]
        switch_branch(target_for(df, "Female"))
        assert st.session_state["selected_features"] == ["age", "bmi"]
        switch_branch(target_for(df, "Male"))
        assert st.session_state["selected_features"] == ["age", "bmi"]


# ── the three non-key slices ─────────────────────────────────────────────

def _insight(page, ident, auto=True):
    from utils.insight_ledger import Insight
    return Insight(id=ident, source_page=page, category="methodology",
                   severity="info", finding=f"{ident} on {page}",
                   implication="—", recommended_action="—",
                   auto_generated=auto)


class TestTheLedgerTravelsWithItsBranch:

    def setup_ledger(self):
        from utils.insight_ledger import InsightLedger
        ledger = InsightLedger()
        st.session_state["insight_ledger"] = ledger
        return ledger

    def test_an_explainability_finding_does_not_leak_into_the_next_group(self):
        """The bug this closes reached `report.md`.

        The reset pruned only 02/05/06 and merely un-resolved 07/08/09, so a
        SHAP finding measured on the women was still in the ledger when the
        men's report was written, and page 10 printed it there.
        """
        df = study()
        st.session_state["raw_data"] = df
        ledger = self.setup_ledger()

        switch_branch(target_for(df, "Female"))
        ledger.add(_insight("07_Explainability", "shap-female"))
        ledger.add(_insight("01_Upload_and_Audit", "audit-shared"))

        switch_branch(target_for(df, "Male"))
        ids = {i.id for i in ledger.insights}
        assert "shap-female" not in ids, "the women's SHAP finding followed the men"
        assert "audit-shared" in ids, "a study-level finding was lost"

        switch_branch(target_for(df, "Female"))
        assert "shap-female" in {i.id for i in ledger.insights}

    def test_a_hand_written_note_travels_too(self):
        """`prune_auto_generated` spares what a person wrote — correctly, for an
        invalidation. A cohort switch is the other case: the note is about one
        group of people and belongs to that group's branch."""
        df = study()
        st.session_state["raw_data"] = df
        ledger = self.setup_ledger()

        switch_branch(target_for(df, "Female"))
        ledger.add(_insight("08_Sensitivity_Analysis", "by-hand", auto=False))

        switch_branch(target_for(df, "Male"))
        assert "by-hand" not in {i.id for i in ledger.insights}

        switch_branch(target_for(df, "Female"))
        assert "by-hand" in {i.id for i in ledger.insights}

    def test_a_restored_branch_does_not_duplicate_its_own_insights(self):
        df = study()
        st.session_state["raw_data"] = df
        ledger = self.setup_ledger()
        switch_branch(target_for(df, "Female"))
        ledger.add(_insight("06_Train_and_Compare", "one"))

        switch_branch(target_for(df, "Male"))
        switch_branch(target_for(df, "Female"))
        switch_branch(target_for(df, "Male"))
        switch_branch(target_for(df, "Female"))

        assert [i.id for i in ledger.insights].count("one") == 1


class TestTheMethodologyLogTravels:

    def test_a_training_entry_belongs_to_its_branch(self):
        df = study()
        st.session_state["raw_data"] = df
        st.session_state["methodology_log"] = [
            {"step": "Upload & Audit", "action": "loaded"},
        ]
        switch_branch(target_for(df, "Female"))
        st.session_state["methodology_log"].append(
            {"step": "Model Training", "action": "fitted on the women"})

        switch_branch(target_for(df, "Male"))
        steps = [e["action"] for e in st.session_state["methodology_log"]]
        assert "fitted on the women" not in steps
        assert "loaded" in steps, "a study-level step was dropped"

        switch_branch(target_for(df, "Female"))
        steps = [e["action"] for e in st.session_state["methodology_log"]]
        assert "fitted on the women" in steps


class TestTheProvenanceTravels:

    def test_the_split_section_describes_the_branch_it_belongs_to(self):
        from utils.workflow_provenance import WorkflowProvenance
        df = study()
        st.session_state["raw_data"] = df
        prov = WorkflowProvenance()
        st.session_state["workflow_provenance"] = prov

        switch_branch(target_for(df, "Female"))
        from utils.workflow_provenance import SplitProvenance
        female_split = SplitProvenance(strategy="random", train_n=100,
                                       val_n=20, test_n=30)
        prov.split = female_split

        switch_branch(target_for(df, "Male"))
        assert prov.split is not female_split, "the men inherited the women's split record"

        switch_branch(target_for(df, "Female"))
        assert prov.split is female_split

    def test_the_decision_sections_are_not_per_branch(self):
        """The FE recipe and the selection are the study's. Archiving them
        per-branch would let two branches disagree about the question."""
        from utils.cohorts import _branch_provenance_sections
        sections = _branch_provenance_sections()
        assert "feature_engineering" not in sections
        assert "feature_selection" not in sections
        assert "split" in sections and "training" in sections
        # And the set is derived from the record, not a literal: sections the
        # cascade's own stage graph never declared still have to travel.
        for late_addition in ("sensitivity", "statistical_validation",
                              "external_validation"):
            assert late_addition in sections, (
                f"{late_addition} is a per-branch record and is not being archived")


# ── the archive itself ───────────────────────────────────────────────────

class TestTheArchive:

    def test_the_switch_is_the_only_thing_that_keeps_it(self):
        """`switch_branch` is the one caller allowed to pass preserve_branches."""
        from utils.session_state import reset_downstream_results
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        switch_branch(target_for(df, "Male"))
        assert st.session_state.get(BRANCH_ARCHIVE_KEY)

        reset_downstream_results()          # any other caller
        assert BRANCH_ARCHIVE_KEY not in st.session_state

    def test_archiving_twice_is_not_two_branches(self):
        """`archive_current` is idempotent, which is what lets `switch_branch`
        call it unconditionally as its first line."""
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        before = dict(st.session_state[BRANCH_ARCHIVE_KEY])
        archive_current()
        archive_current()
        after = st.session_state[BRANCH_ARCHIVE_KEY]
        assert set(after) == set(before) | {("sex", "Female")}
        # And the whole-study branch the switch banked on its way out is still
        # exactly one entry, not one per call.
        assert len(after) == 2, list(after)

    def test_the_branches_are_offered_in_the_run_order_with_everyone_last(self):
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        switch_branch(target_for(df, "Male"))
        keys = known_branches()
        assert keys[-1] == EVERYONE, "everyone is not one of the groups"
        assert ("sex", "Female") in keys and ("sex", "Male") in keys

    def test_a_branch_knows_whether_it_has_been_trained(self):
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        fit_a_branch("female")
        switch_branch(target_for(df, "Male"))

        assert branch_has_models(("sex", "Female")) is True
        assert branch_has_models(("sex", "Male")) is False
        assert branch_key(active_cohort()) == ("sex", "Male")


# ── the cohort filter still holds ────────────────────────────────────────

class TestTheRowsAreStillRight:

    def test_a_restored_branch_still_sees_only_its_own_people(self):
        """The point of the whole feature. A restore that brought back the rows
        without bringing back the filter would be the silent full-study run."""
        from utils.session_state import get_data
        df = study()
        st.session_state["raw_data"] = df

        switch_branch(target_for(df, "Female"))
        switch_branch(target_for(df, "Male"))
        switch_branch(target_for(df, "Female"))

        seen = get_data()
        assert set(seen["sex"].unique()) == {"Female"}
        assert len(seen) < len(df)

    def test_the_previous_frame_never_reaches_the_next_branch(self):
        """`filtered_data` is a row subset of the branch that made it.

        The old code popped it by hand before switching, because `apply_cohort`
        reading the previous cohort's frame returns an EMPTY one with no flag
        set. It is a branch key now, so the archive does that by construction —
        this test is what says so.
        """
        from utils.cohorts import cohort_filter_broken
        df = study()
        st.session_state["raw_data"] = df

        switch_branch(target_for(df, "Female"))
        st.session_state["filtered_data"] = df[df["sex"] == "Female"]

        switch_branch(target_for(df, "Male"))
        live = st.session_state.get("filtered_data")
        assert live is None or set(live["sex"].unique()) == {"Male"}
        assert not cohort_filter_broken()


# ── branches live for the session ────────────────────────────────────────

class TestTheArchiveIsNotSaved:

    def test_no_save_path_writes_a_branch_archive(self):
        """A stated limit, not an oversight.

        A snapshot holds fitted estimators, SHAP explainers and matplotlib
        figures. `session_manager` writes JSON and parquet from four explicit
        allowlists, and the archive is in none of them — which is what makes
        "branches live for the session" true rather than aspirational. A future
        save path that adds it would be pickling arbitrary fitted objects into
        a file the app reloads, and this test is where that decision surfaces.
        """
        from utils import session_manager as sm
        allowlisted = (set(sm._DATAFRAME_KEYS) | set(sm._DATACLASS_KEYS)
                       | set(sm._PLAIN_KEYS) | set(sm._SAFE_WIDGET_KEYS))
        assert BRANCH_ARCHIVE_KEY not in allowlisted, (
            "the branch archive would be written to a save file; it holds "
            "fitted estimators and figures")

    def test_restoring_a_session_leaves_no_branch_behind(self):
        """The restore runs the full data reset, which drops the archive.

        Without this, a restored session would carry a `cohort_run` naming a
        group whose branch is gone — and the picker would offer a branch that
        cannot be restored.
        """
        from utils.session_state import reset_data_dependent_state
        df = study()
        st.session_state["raw_data"] = df
        switch_branch(target_for(df, "Female"))
        switch_branch(target_for(df, "Male"))
        assert st.session_state.get(BRANCH_ARCHIVE_KEY)

        reset_data_dependent_state()        # what _clear_downstream_state calls
        assert BRANCH_ARCHIVE_KEY not in st.session_state
        assert active_cohort() is None


# ── the archive must survive the reset that runs beside it ───────────────

class TestTheResetCannotReachIntoTheArchive:
    """The snapshot holds live objects on purpose, and the reset mutates some
    of them IN PLACE. Those two facts collided.

    `reset_downstream_results` rolls back resolutions — `resolved = False`,
    `resolved_by = ""` — on the insights of its cleared pages. The snapshot
    taken moments earlier holds those same `Insight` objects, so the rollback
    reached a branch that was already archived and un-resolved a finding in it.
    Switching back showed a resolved finding reopened; worse, page 10 fills the
    manuscript's Limitations from unresolved insights, so the previous group's
    exported report gained a limitation it had addressed.
    """

    def _ledger(self):
        from utils.insight_ledger import InsightLedger
        ledger = InsightLedger()
        st.session_state["insight_ledger"] = ledger
        return ledger

    def test_a_resolved_finding_stays_resolved_in_the_branch_it_belongs_to(self):
        df = study()
        st.session_state["raw_data"] = df
        ledger = self._ledger()

        switch_branch(target_for(df, "Female"))
        found = _insight("05_Preprocess", "imbalance")
        ledger.add(found)
        found.resolved = True
        found.resolved_by = "class_weight=balanced"
        found.resolved_on_page = "06_Train_and_Compare"

        switch_branch(target_for(df, "Male"))      # the reset runs here

        archived = st.session_state[BRANCH_ARCHIVE_KEY][("sex", "Female")]
        banked = [i for i in archived.ledger if i.id == "imbalance"]
        assert banked, "the finding was not archived at all"
        assert banked[0].resolved is True, (
            "the reset un-resolved an insight inside an already-archived branch")
        assert banked[0].resolved_by == "class_weight=balanced"

        switch_branch(target_for(df, "Female"))
        live = ledger.get("imbalance")
        assert live is not None and live.resolved is True, (
            "the restored branch shows a finding the researcher had resolved as open")

    def test_the_live_ledger_is_still_rolled_back_for_the_new_branch(self):
        """The fix must not stop the reset doing its job. A finding on a page
        OUTSIDE the branch set — resolved on a cleared page — is still rolled
        back, because the new branch has not earned that resolution."""
        df = study()
        st.session_state["raw_data"] = df
        ledger = self._ledger()

        switch_branch(target_for(df, "Female"))
        shared = _insight("04_Feature_Selection", "collinear")
        ledger.add(shared)
        shared.resolved = True
        shared.resolved_on_page = "04_Feature_Selection"

        switch_branch(target_for(df, "Male"))
        live = ledger.get("collinear")
        assert live is not None, "a shared-page finding must not be pruned"
        assert live.resolved is False, (
            "a resolution earned on a cleared page survived the reset")


class TestPageTwoTravelsWithItsResults:
    """Page 02's keys are per-branch — `eda_results`, `dataset_profile`,
    `dataset_profile_scope`, `table1_df` are all in `BRANCH_KEYS` — so its
    ledger and methodology entries have to be too. They were not, so a restored
    branch showed its own profile numbers beside the OTHER group's
    distributional findings: the exact confusion `dataset_profile_scope` exists
    to prevent.
    """

    def test_an_eda_finding_does_not_outlive_its_cohort(self):
        from utils.insight_ledger import InsightLedger
        df = study()
        st.session_state["raw_data"] = df
        ledger = InsightLedger()
        st.session_state["insight_ledger"] = ledger

        switch_branch(target_for(df, "Female"))
        st.session_state["eda_results"] = {"profile": "the women"}
        ledger.add(_insight("02_EDA", "skew-in-the-women"))

        switch_branch(target_for(df, "Male"))
        assert "skew-in-the-women" not in {i.id for i in ledger.insights}, (
            "the women's EDA finding is on screen under the men's heading")

        switch_branch(target_for(df, "Female"))
        assert st.session_state["eda_results"] == {"profile": "the women"}
        assert "skew-in-the-women" in {i.id for i in ledger.insights}, (
            "the finding did not come back with the results it describes")

    def test_the_page_set_follows_the_key_set(self):
        """The rule, asserted rather than remembered: a page whose results are
        per-branch keys must be a branch page, or its findings and its numbers
        get separated."""
        from turbotab.cascade import BRANCH_PAGES
        assert "02_EDA" in BRANCH_PAGES, (
            "eda_results and dataset_profile are branch keys; page 02's "
            "insights must travel with them")
        for shared in ("03_Feature_Engineering", "04_Feature_Selection"):
            assert shared not in BRANCH_PAGES, (
                f"{shared} records a decision shared by every branch")

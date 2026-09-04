"""One seal, drawn before the split — so two cohort runs open two disjoint halves.

`opened_count` is study-wide and cohort-blind. Run the women, run the men, and
it reads 2. The chip printed that number, and the Methods wrote a sentence from
it, and both of them mean the same rows were scored twice — which is the one
thing it does not mean. The seal is drawn on the whole study before any cohort
exists (`utils/cohorts.py`, invariant 1), so each run is scored against its own
slice and no row is scored twice.

The other half is a seal that stops existing. Two paths end one: the stale-label
retirement DELETES the dict, and a redraw rebuilds it with `opened_count: 0`.
Either way the counter goes to zero, after which the Methods is free to say the
set was "accessed only for the final evaluation" about a study whose first
held-out set had been opened four times. The sequence that reaches it is
ordinary: run Female, run Male, go back to analyzing everyone, change the test
fraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.cohorts import plan_cohorts, start_cohort, clear_cohort
from utils.test_lockbox import (
    EVERYONE_TAG, RETIRED_KEY, cohort_open_breakdown, ensure_lockbox,
    get_lockbox, lockbox_open_count, open_tag, opens_by_cohort, opens_here,
    record_lockbox_open, retire_seal, retired_seals,
)

_SWEEP = "Sensitivity Analysis (seed sweep, re-split over the sealed rows)"


@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        for key in ("test_lockbox", RETIRED_KEY, "cohort_run", "raw_data",
                    "filtered_data", "data_config", "_raw_data_fingerprint",
                    "_cohort_filter_broken", "_lockbox_redrawn",
                    "_lockbox_not_sealed", "cohort_runs_done"):
            st.session_state.pop(key, None)
    _wipe()
    yield
    _wipe()


def study(n=400, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["Male", "Female"], n),
        "age": rng.integers(20, 80, n),
        "y": rng.choice([0, 1], n, p=[0.5, 0.5]),
    })


def seal(df, fraction=0.2):
    return ensure_lockbox(df, "y", "classification", fraction=fraction, seed=7)


def enter(df, label):
    plan = plan_cohorts(df, "sex", "y", "classification")
    cell = next(c for c in plan.viable if c.label == label)
    return start_cohort(df, plan, cell, "y")


# ── the recording ────────────────────────────────────────────────────────

class TestWhoseSliceWasOpened:

    def test_an_opening_with_no_cohort_belongs_to_everyone(self):
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")
        assert opens_by_cohort() == {EVERYONE_TAG: {"Train & Compare": 1}}
        assert open_tag() == EVERYONE_TAG

    def test_two_groups_scored_once_each_are_not_one_set_scored_twice(self):
        df = study()
        st.session_state["raw_data"] = df
        seal(df)

        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")

        assert lockbox_open_count() == 2, "the study total is still the total"
        assert opens_here("sex=Female") == 1
        assert opens_here("sex=Male") == 1
        assert opens_here() == 1, "the ACTIVE run has been scored once"

    def test_the_breakdown_names_the_groups(self):
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")

        breakdown = cohort_open_breakdown()
        assert "Female 1" in breakdown and "Male 1" in breakdown, breakdown

    def test_one_group_alone_gets_no_breakdown(self):
        """A breakdown of one is noise — the total already says it."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        assert cohort_open_breakdown() == ""

    def test_a_seed_sweep_is_not_a_scoring_run_for_this_group_either(self):
        """The chip already draws this distinction for the study total: page 08
        re-partitions the sealed rows and reports no held-out number."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        record_lockbox_open(_SWEEP)

        assert opens_here() == 2
        assert opens_here(scoring_only=True) == 1

    def test_the_opened_at_strings_keep_the_format_the_chip_parses(self):
        """The tag goes in a structured field on purpose.

        `render_lockbox_status` reads the source out of `opened_at` by requiring
        the entry to END with ')'. A ` [sex=Female]` suffix would make every
        entry unparseable, the source list would come back empty, and the seed
        sweep would be re-classified as a scoring run — flipping page 08 from
        a blue notice to a red warning.
        """
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")

        entries = get_lockbox()["opened_at"]
        assert entries and all(e.endswith(")") for e in entries), entries
        assert all("(" in e for e in entries)
        src = entries[0][entries[0].find("(") + 1:-1].strip()
        assert src == "Train & Compare", src


# ── a seal that stops existing ───────────────────────────────────────────

class TestARetiredSealLeavesItsRecord:

    def test_a_redraw_carries_the_previous_counts_out(self):
        """The exact sequence from the design note: two groups, back to
        everyone, change the test fraction."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df, fraction=0.2)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")
        clear_cohort()

        seal(df, fraction=0.3)          # the redraw the cohort guard no longer blocks

        assert lockbox_open_count() == 0, "a fresh seal starts at zero, as before"
        retired = retired_seals()
        assert len(retired) == 1, retired
        assert retired[0]["opened_count"] == 2
        assert set(retired[0]["opened_by_cohort"]) == {"sex=Female", "sex=Male"}

    def test_a_deleted_seal_leaves_its_record_too(self):
        """`_cannot_seal` pops the dict outright — there is no successor to
        carry anything into, so the record is parked beside it."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")

        # The refusal path: the frame can no longer be sealed AND the rows the
        # old seal names are not in it. `_cannot_seal` deletes the dict there.
        shrunk = df.iloc[:50].drop(columns=["y"])
        assert ensure_lockbox(shrunk, "y", "classification") is None
        assert get_lockbox() is None
        assert retired_seals()[0]["opened_count"] == 1

    def test_an_unopened_seal_leaves_nothing_behind(self):
        """A seal drawn and replaced without ever being scored against has no
        disclosure to make, and a record of it would put one in the Methods."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df, fraction=0.2)
        seal(df, fraction=0.3)
        assert retired_seals() == []

    def test_a_new_dataset_drops_the_retired_record(self):
        """It exists so a redraw cannot erase a disclosure. Carried into a new
        study it would make one about openings that study never had."""
        from utils.session_state import reset_data_dependent_state
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")
        retire_seal(get_lockbox())
        assert retired_seals()

        reset_data_dependent_state()
        assert retired_seals() == []


# ── what the manuscript says ─────────────────────────────────────────────

class TestTheMethodsSaysWhatTheCounterMeasured:

    def test_two_groups_produce_a_per_group_sentence_not_a_total(self):
        from ml.narrative_engine import _lockbox_cohort_access
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")

        said = _lockbox_cohort_access()
        assert said is not None
        assert "disjoint slice" in said
        assert "split by sex" in said
        assert "Female 1" in said and "Male 1" in said
        # The forking path the sequential design cannot close, disclosed.
        assert "sequentially" in said

    def test_one_cohort_alone_does_not_get_the_sentence(self):
        from ml.narrative_engine import _lockbox_cohort_access
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        assert _lockbox_cohort_access() is None

    def test_a_retired_seal_is_disclosed(self):
        from ml.narrative_engine import _retired_seal_clause
        df = study()
        st.session_state["raw_data"] = df
        seal(df, fraction=0.2)
        record_lockbox_open("Train & Compare")
        record_lockbox_open("Train & Compare")
        seal(df, fraction=0.35)

        said = _retired_seal_clause()
        assert said is not None
        assert "accessed 2 times" in said, said
        assert "not the only one" in said

    def test_no_retired_seal_means_no_sentence_about_one(self):
        from ml.narrative_engine import _retired_seal_clause
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")
        assert _retired_seal_clause() is None


# ── what the chip says ───────────────────────────────────────────────────

class TestTheChipStopsClaimingTheSamePeople:

    def _render(self, df):
        """Collect every caption and warning the chip emits."""
        import utils.test_lockbox as tlb
        said = []
        real_caption, real_warning, real_info = st.caption, st.warning, st.info
        st.caption = lambda body, **kw: said.append(str(body))
        st.warning = lambda body, **kw: said.append(str(body))
        st.info = lambda body, **kw: said.append(str(body))
        try:
            tlb.render_lockbox_status()
        finally:
            st.caption, st.warning, st.info = real_caption, real_warning, real_info
        return " ".join(said)

    def test_the_cohort_caption_no_longer_claims_the_same_held_out_people(self):
        """It said "every run is evaluated against the same held-out people".

        Every run is evaluated against its own SLICE of one set. Saying they
        share the people is the sentence a reviewer would read as a paired
        comparison, which these disjoint slices cannot support.
        """
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        text = self._render(df)
        assert "same held-out people" not in text, text
        assert "its own slice" in text, text

    def test_the_cohort_caption_reports_this_run_and_the_study(self):
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")

        text = self._render(df)
        assert "This slice has been scored once" in text, text
        assert "Female 1" in text and "Male 1" in text, text
        assert "no row was scored twice" in text, text

    def test_the_whole_study_caption_says_it_too(self):
        """The state after *Go back to analyzing everyone* is where the
        archived-vs-live distinction matters most, and it is the one caption
        that has no cohort branch to put it in."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")
        clear_cohort()

        text = self._render(df)
        assert "Female 1" in text and "Male 1" in text, text
        assert "disjoint slices" in text, text


# ── disjoint across groups is a claim, and it can be false ───────────────

class TestTheDisjointnessClaimIsEarned:
    """"Two groups, once each" is only *not* "one set scored twice" because the
    slices do not overlap. That is a claim about the history, and there are two
    histories where it is false — both reachable by ordinary use."""

    def test_a_whole_study_opening_covers_every_slice(self):
        """Train on everyone, then split by sex and train the women. The seal
        has been opened twice, and the second opening's rows are a SUBSET of
        the first's. Nothing here is disjoint."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")        # everyone
        enter(df, "Female")
        record_lockbox_open("Train & Compare")

        from utils.test_lockbox import opens_are_disjoint
        assert opens_are_disjoint() is False
        assert cohort_open_breakdown(), "the counts are still worth showing"

    def test_two_groups_alone_are_disjoint(self):
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        enter(df, "Male")
        record_lockbox_open("Train & Compare")

        from utils.test_lockbox import opens_are_disjoint
        assert opens_are_disjoint() is True

    def test_the_methods_will_not_assert_disjointness_it_does_not_have(self):
        """It falls through to the plain "accessed N times" warning, which is
        the honest description of a history that includes a whole-study run."""
        from ml.narrative_engine import _lockbox_cohort_access
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        assert _lockbox_cohort_access() is None

    def test_the_chip_says_which_of_the_two_it_is(self):
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")
        enter(df, "Female")
        record_lockbox_open("Train & Compare")

        said = TestTheChipStopsClaimingTheSamePeople()._render(df)
        assert "no row was scored twice" not in said, said
        assert "scored inside it as well" in said, said

    def test_the_breakdown_always_adds_up_to_the_total(self):
        """A seal restored from a save file has a count and no per-cohort map.
        Printing a breakdown beside a total it does not sum to is worse than
        printing no breakdown."""
        from utils.test_lockbox import opens_per_cohort
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        record_lockbox_open("Train & Compare")
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        get_lockbox().pop("opened_by_cohort")        # the pre-field shape

        assert sum(opens_per_cohort().values()) == lockbox_open_count() == 2


class TestScoredMeansScored:

    def test_a_seed_sweep_does_not_make_the_caption_say_scored_twice(self):
        """The chip's own notice explains that the sweep re-partitions the
        sealed rows and reports no held-out number. The per-run caption beside
        it counted the sweep anyway, so one sentence said "scored twice" while
        the next said only one scoring run had happened."""
        df = study()
        st.session_state["raw_data"] = df
        seal(df)
        enter(df, "Female")
        record_lockbox_open("Train & Compare")
        record_lockbox_open(_SWEEP)

        said = TestTheChipStopsClaimingTheSamePeople()._render(df)
        assert "This slice has been scored once" in said, said


class TestARedrawOnTheSameRowsStillZeroesTheCounter:

    def test_the_record_survives_a_redraw_that_lands_on_the_same_labels(self):
        """The retire was guarded by the LABELS changing; the counter is zeroed
        whenever the dict is rebuilt.

        The signature is stale-ed by hand rather than by changing a parameter,
        because every parameter that moves the signature also moves the draw.
        This is the shape of the bug, isolated: a rebuild whose split lands on
        the same rows still resets `opened_count`, and under the old guard the
        openings were simply gone.
        """
        df = study()
        st.session_state["raw_data"] = df
        seal(df, fraction=0.2)
        record_lockbox_open("Train & Compare")
        before = list(get_lockbox()["labels"])

        get_lockbox()["signature"] = "stale-so-the-early-return-does-not-fire"
        ensure_lockbox(df, "y", "classification", fraction=0.2, seed=7)

        assert list(get_lockbox()["labels"]) == before, (
            "the setup did not reproduce the same draw; the test is not "
            "exercising the same-labels path")
        assert lockbox_open_count() == 0, "a rebuilt seal starts at zero"
        assert retired_seals(), (
            "a redraw that reproduced the same rows zeroed the counter and "
            "left no record that the set had ever been opened")
        assert retired_seals()[0]["opened_count"] == 1


class TestTheRecordSurvivesASavedSession:

    def test_a_retired_seal_is_written_to_the_save_file(self):
        """It exists so a redraw cannot erase a disclosure. A save that drops it
        erases the same disclosure by a different route."""
        from utils import session_manager as sm
        assert "test_lockbox_retired" in sm._PLAIN_KEYS

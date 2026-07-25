"""Two defects the pre-PR audit reproduced in regions the suite never covered.

SUBJECT IDs. detect_repeated_subjects() matched "id" as a bare SUBSTRING, so
uric_acid, folic_acid, linoleic_acid, lipid and NHANES's whole RID* family —
including RIDAGEYR, which is age in years — all read as subject identifiers.
Among the survivors it then ranked by MOST distinct values, exactly backwards,
so a near-continuous lab value beat the real subject_id sitting beside it. The
lockbox was grouped on that covariate: on NHANES all 10 test ages were unseen
in training, stratification was skipped entirely, and with genuine repeated
measures 49 of 120 real subjects landed on BOTH sides — while page 01 printed
"Splitting by row would put the same person in both training and testing",
claiming to have prevented the leak it had just created.

JOIN KEYS. A contiguous integer run demoted its candidate to "low" whatever it
was called, and combine_ui drops "low" from the dropdown. A study numbering
participants 1..N — or this app's own execute_stack, which turns two stacked
cycles into SEQN 1..200 — therefore lost its correct SEQN<->SEQN key and had a
measurement column pre-selected instead. Every existing test offset IDs to
range(83732, ...) or range(1000, ...), so the region where a real ID is also
contiguous was never exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.join_doctor import find_key_candidates, suggest_best
from utils.test_lockbox import detect_repeated_subjects, _name_looks_like_a_subject_id

RNG = np.random.RandomState(0)


@pytest.fixture(autouse=True)
def _deterministic():
    RNG.seed(0)


# ── a biomarker is not a person ──────────────────────────────────────────

class TestSubjectIdNamesAreWholeTokens:

    @pytest.mark.parametrize("name", [
        "uric_acid", "folic_acid", "linoleic_acid", "amino_acid_score",
        "lipid", "oxidized_ldl", "residual", "RIDAGEYR", "RIDEXPRG",
        "valid", "rapid_test", "humidity",
    ])
    def test_these_are_not_subject_ids(self, name):
        assert not _name_looks_like_a_subject_id(name)

    @pytest.mark.parametrize("name", [
        "id", "ID", "SEQN", "subject_id", "SubjectID", "patient_id",
        "participant id", "MRN", "USUBJID", "record_id", "person",
    ])
    def test_these_are(self, name):
        assert _name_looks_like_a_subject_id(name)

    def test_nhanes_age_is_not_chosen_as_the_grouping_column(self):
        n = 1500
        df = pd.DataFrame({
            "SEQN": RNG.randint(83732, 83732 + n, n),
            "RIDAGEYR": RNG.randint(18, 80, n),
            "uric_acid": np.round(RNG.uniform(2, 9, n), 1),
            "y": RNG.randint(0, 2, n),
        })
        got = detect_repeated_subjects(df)
        assert got is None or got[0] not in ("RIDAGEYR", "uric_acid")

    def test_the_real_subject_column_beats_a_finer_biomarker(self):
        """Repeated measures: 120 people x 4 visits, plus a near-continuous lab."""
        subj = np.repeat(np.arange(1, 121), 4)
        df = pd.DataFrame({
            "subject_id": subj,
            "linoleic_acid": np.round(RNG.uniform(1, 40, len(subj)), 2),
            "visit": np.tile([1, 2, 3, 4], 120),
            "y": RNG.randint(0, 2, len(subj)),
        })
        got = detect_repeated_subjects(df)
        assert got is not None and got[0] == "subject_id", got


# ── a numbered participant is still a participant ────────────────────────

def _two_cycles(start=1, n=200):
    ids = np.arange(start, start + n)
    demo = pd.DataFrame({"SEQN": ids, "age": RNG.randint(20, 80, n),
                         "sex": RNG.choice(["F", "M"], n)})
    labs = pd.DataFrame({"SEQN": ids, "glucose": np.round(RNG.uniform(70, 200, n), 1)})
    return demo, labs


class TestContiguousIdsAreStillKeys:

    def test_the_correct_key_is_offered_not_discarded(self):
        demo, labs = _two_cycles()
        cands = find_key_candidates(demo, labs)
        seqn = [c for c in cands if c.left_col == "SEQN" and c.right_col == "SEQN"]
        assert seqn, "the true key vanished entirely"
        assert seqn[0].confidence != "low", (
            "combine_ui removes 'low' from the dropdown, so the user cannot pick it")

    def test_it_outranks_a_measurement_that_happens_to_overlap(self):
        demo, labs = _two_cycles()
        best = suggest_best(demo, labs)
        assert best is not None
        assert (best.left_col, best.right_col) == ("SEQN", "SEQN"), (
            f"pre-selected {best.left_col}<->{best.right_col} instead")

    def test_it_is_offered_but_never_asserted(self):
        demo, labs = _two_cycles()
        seqn = next(c for c in find_key_candidates(demo, labs)
                    if c.left_col == "SEQN" == c.right_col)
        assert seqn.confidence == "medium", (
            "the app cannot PROVE two contiguous runs are the same people")

    def test_two_anonymous_row_counters_are_still_refused(self):
        a = pd.DataFrame({"Unnamed: 0": range(1, 101), "gdp": RNG.uniform(0, 9, 100)})
        b = pd.DataFrame({"Unnamed: 0": range(1, 101), "rainfall": RNG.uniform(0, 9, 100)})
        cands = [c for c in find_key_candidates(a, b)
                 if c.left_col == "Unnamed: 0" == c.right_col]
        assert all(c.confidence == "low" for c in cands), (
            "two unrelated exports overlap 100% by construction")

    def test_a_non_contiguous_id_is_unaffected(self):
        demo, labs = _two_cycles(start=83732)
        best = suggest_best(demo, labs)
        assert best is not None and best.left_col == "SEQN"
        assert best.confidence == "high"

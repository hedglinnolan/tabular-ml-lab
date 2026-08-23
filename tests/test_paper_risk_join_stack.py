"""Regressions for the join/stack canonicalization defects (paper-risk sprint).

One class per finding id. Each defends a merge or stack that produced a
CONFIDENTLY WRONG analysis frame — rows silently dropped, rows silently
mis-paired, one variable split in two, or a participant number handed to the
model as a predictor.

The rule being defended throughout: the app may be silent, and it may refuse,
but it must never assert something false.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.join_doctor import (
    diagnose_join, execute_join, find_key_candidates, key_reading,
    normalize_key, suggest_best,
)
from utils.combine import plan_stack

RNG = np.random.RandomState(0)


@pytest.fixture(autouse=True)
def _deterministic():
    RNG.seed(0)


class TestImport233CaseFoldingIsDecidedForThePair:
    """normalize_key decided folding from whichever column it was handed.

    One stray-case duplicate on ONE side disabled folding for that side alone,
    so the two files compared in two different canonical spaces: rows vanished
    AND a surviving row was paired with the wrong partner, with nothing on
    screen saying so.
    """

    @staticmethod
    def _frames():
        sites = pd.DataFrame({"specimen": ["A01", "B02"], "site": ["x", "y"]})
        labs = pd.DataFrame({"specimen": ["A01", "a01", "B02"],
                             "g": [1.0, 2.0, 3.0]})
        return sites, labs

    def test_one_sided_case_collision_does_not_drop_or_mispair_rows(self):
        sites, labs = self._frames()
        out, _ = execute_join(sites, labs, "specimen", "specimen", "inner")
        by_id = dict(zip(out["specimen"].astype(str).str.upper(), out["g"]))
        # B02 survives, and A01 carries ITS OWN lab value, not a01's.
        assert set(by_id) == {"A01", "B02"}
        assert by_id["A01"] == 1.0
        assert by_id["B02"] == 3.0

    def test_the_pair_shares_one_canonical_space(self):
        sites, labs = self._frames()
        reading = key_reading(sites["specimen"], labs["specimen"])
        left = normalize_key(sites["specimen"], reading.fold_case, reading.keep_tokens)
        right = normalize_key(labs["specimen"], reading.fold_case, reading.keep_tokens)
        # Whatever the decision, both sides must have made the SAME one.
        assert set(left) <= set(right)

    def test_refusing_to_fold_is_disclosed_with_the_actual_collision(self):
        sites, labs = self._frames()
        d = diagnose_join(sites, labs, "specimen", "specimen", "inner")
        hit = [w for w in d.warnings
               if "Capitalization is being treated as meaningful in BOTH files" in w]
        assert hit, d.warnings
        assert "'A01'" in hit[0] and "'a01'" in hit[0]

    def test_case_folding_still_happens_when_it_is_safe_on_both_sides(self):
        a = pd.DataFrame({"k": ["A01", "B02"], "x": [1, 2]})
        b = pd.DataFrame({"k": ["a01", "b02"], "y": [3, 4]})
        out, _ = execute_join(a, b, "k", "k", "inner")
        assert len(out) == 2


class TestImport234Float64IdCollapseIsRefused:
    """A key column that arrived as float64 has already lost digits above 2^53.

    _canon_scalar defended only against float conversion it performed itself,
    then canonicalized the collapsed digits as exact — two participants became
    one row identity, and the resulting duplicate was explained to the
    researcher as repeated visits producing a fan-out.
    """

    @staticmethod
    def _frames():
        left = pd.DataFrame({"SEQN": [9007199254740993.0, 9007199254740992.0,
                                      9007199254740995.0],
                             "age": [50, 60, 70]})
        right = pd.DataFrame({"SEQN": [9007199254740993.0, 9007199254740995.0],
                              "chol": [4.0, 5.0]})
        return left, right

    def test_diagnose_blocks_instead_of_explaining_the_collision_as_a_fan_out(self):
        left, right = self._frames()
        d = diagnose_join(left, right, "SEQN", "SEQN", "inner")
        assert not d.can_proceed
        assert any("lost their last digits" in b for b in d.blocking)
        # And it must not narrate the collision as ordinary study structure.
        assert not any("several rows per ID" in w for w in d.warnings)

    def test_execute_join_refuses_rather_than_merging_collapsed_ids(self):
        left, right = self._frames()
        with pytest.raises(ValueError, match="lost their last digits"):
            execute_join(left, right, "SEQN", "SEQN", "inner")

    def test_the_limit_itself_is_already_unsafe(self):
        """2^53 may BE a collapsed 2^53+1 — the pair the module's docstring names."""
        left = pd.DataFrame({"SEQN": [9007199254740993.0, 9007199254740992.0],
                             "age": [50, 60]})
        right = pd.DataFrame({"SEQN": [9007199254740993.0], "chol": [4.0]})
        assert not diagnose_join(left, right, "SEQN", "SEQN", "inner").can_proceed

    def test_ordinary_numeric_ids_are_untouched(self):
        left = pd.DataFrame({"SEQN": [1.0, 2.0, np.nan], "age": [50, 60, 70]})
        right = pd.DataFrame({"SEQN": [1.0, 2.0], "chol": [4.0, 5.0]})
        out, _ = execute_join(left, right, "SEQN", "SEQN", "inner")
        assert len(out) == 2


class TestImport236IndexPenaltyAppliesWhenEitherSideIsACounter:
    """index_like was `_looks_like_row_index(left) AND ...(right)`.

    A measurement paired against a bare 1..N counter therefore escaped both the
    penalty and the confidence downgrade, and outranked the honest
    counter-to-counter pairing that WAS penalised — the penalty made the
    ranking strictly worse than no penalty, and the junk pairing was offered at
    medium confidence, above combine_ui's low-confidence filter.
    """

    @staticmethod
    def _frames():
        a = pd.DataFrame({"row": range(50), "age": RNG.randint(18, 80, 50)})
        b = pd.DataFrame({"row": range(50), "chol": RNG.normal(200, 30, 50)})
        return a, b

    def test_measurement_against_a_counter_is_penalised(self):
        a, b = self._frames()
        junk = [c for c in find_key_candidates(a, b)
                if (c.left_col, c.right_col) == ("age", "row")]
        assert junk, "the pairing under test must still be produced"
        assert junk[0].index_like
        assert junk[0].confidence == "low"

    def test_it_never_outranks_the_honest_counter_pairing(self):
        a, b = self._frames()
        cands = find_key_candidates(a, b)
        rank = {(c.left_col, c.right_col): i for i, c in enumerate(cands)}
        assert rank[("row", "row")] < rank[("age", "row")]

    def test_nothing_is_suggested_for_two_unrelated_files(self):
        a, b = self._frames()
        assert suggest_best(a, b) is None

    def test_a_named_identifier_pair_is_still_offered(self):
        """The counter penalty must not make a real study's SEQN unjoinable."""
        demo = pd.DataFrame({"SEQN": range(1, 51),
                             "age": RNG.randint(18, 80, 50)})
        labs = pd.DataFrame({"SEQN": list(range(1, 51)) * 2,
                             "chol": RNG.normal(200, 30, 100)})
        best = suggest_best(demo, labs)
        assert best is not None and best.left_col == "SEQN"


class TestImport248RealKeyValuesAreNotDeletedAsMissingTokens:
    """_canon_scalar deleted any value whose text was in _KEY_MISSING_TOKENS.

    A study centre legitimately named 'NA' was dropped from the merged table,
    and the app then asserted the row had 'no ID at all' — a false reason for a
    value the app refused to read.
    """

    @staticmethod
    def _frames():
        a = pd.DataFrame({"centre": ["NA", "NB", "NC", "ND"], "x": [1, 2, 3, 4]})
        b = pd.DataFrame({"centre": ["NA", "NB", "NC", "ND"], "y": [5, 6, 7, 8]})
        return a, b

    def test_a_centre_named_na_survives_the_merge(self):
        a, b = self._frames()
        out, _ = execute_join(a, b, "centre", "centre", "inner")
        assert sorted(out["centre"].astype(str)) == ["NA", "NB", "NC", "ND"]

    def test_keeping_it_is_disclosed(self):
        a, b = self._frames()
        d = diagnose_join(a, b, "centre", "centre", "inner")
        assert any("'NA'" in n and "real ID" in n for n in d.notes)
        assert not any("no ID at all" in w for w in d.warnings)

    def test_na_in_a_column_of_numbers_is_still_a_blank_and_the_reason_is_true(self):
        a = pd.DataFrame({"pid": ["1001", "1002", "NA"], "x": [1, 2, 3]})
        b = pd.DataFrame({"pid": ["1001", "1002", "NA"], "y": [4, 5, 6]})
        out, _ = execute_join(a, b, "pid", "pid", "inner")
        assert sorted(out["pid"].astype(str)) == ["1001", "1002"]
        d = diagnose_join(a, b, "pid", "pid", "inner")
        # The disclosure must name the spelling it discarded, not claim the row
        # had no ID in it.
        assert any("'NA'" in w and "no value" in w for w in d.warnings)

    def test_two_unknown_subjects_are_still_never_the_same_subject(self):
        a = pd.DataFrame({"id": ["A1", "unknown", "unknown"], "x": [1, 2, 3]})
        b = pd.DataFrame({"id": ["A1", "unknown"], "y": [1, 2]})
        out, _ = execute_join(a, b, "id", "id", "inner")
        assert out["id"].astype(str).tolist() == ["A1"]


class TestImport254StackNamesThatDifferOnlyByCaseOrSpacing:
    """plan_stack intersected RAW labels, so 'age' and 'Age' stacked into two
    half-empty columns that are one variable, and nothing said so.

    The 'Only N of M columns appear in every file' warning was gated on
    overlap < 0.5, so a survey file where two of twenty headers differ only in
    capitalization produced no warning at all.
    """

    def test_the_colliding_pairs_are_named(self):
        c1 = pd.DataFrame({"SEQN": [1, 2], "age": [10, 20], "glucose": [1.0, 2.0]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "Age": [30, 40], "Glucose ": [3.0, 4.0]})
        plan = plan_stack({"c1": c1, "c2": c2})
        assert set(plan.name_variants) == {"age", "glucose"}
        text = " ".join(plan.warnings)
        assert "'age' / 'Age'" in text
        assert "'Glucose '" in text and "'glucose'" in text

    def test_it_fires_regardless_of_overall_overlap(self):
        """The high-overlap case is the dangerous one: no other warning fires."""
        base = {f"v{i}": [1, 2] for i in range(18)}
        w1 = pd.DataFrame({**base, "SEQN": [1, 2], "age": [1, 2]})
        w2 = pd.DataFrame({**base, "SEQN": [3, 4], "Age": [3, 4]})
        plan = plan_stack({"w1": w1, "w2": w2})
        assert "age" in plan.name_variants
        assert any("only by capitalization or spacing" in w for w in plan.warnings)

    def test_columns_are_not_merged_silently(self):
        """Reported as a question, never renamed behind the researcher's back."""
        c1 = pd.DataFrame({"SEQN": [1, 2], "age": [10, 20]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "Age": [30, 40]})
        plan = plan_stack({"c1": c1, "c2": c2})
        assert "age" in plan.all_columns and "Age" in plan.all_columns

    def test_matching_headers_produce_no_such_warning(self):
        c1 = pd.DataFrame({"SEQN": [1, 2], "age": [10, 20]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "age": [30, 40]})
        plan = plan_stack({"c1": c1, "c2": c2})
        assert not plan.name_variants

    def test_one_file_holding_both_spellings_meant_two_columns(self):
        c1 = pd.DataFrame({"SEQN": [1, 2], "age": [10, 20], "Age": [1, 2]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "age": [30, 40], "Age": [3, 4]})
        plan = plan_stack({"c1": c1, "c2": c2})
        assert not plan.name_variants


class TestImport256IdentifiersAreNotOfferedAsPredictors:
    """reserved_columns() returned ['__source_file'] and nothing else.

    Neither the join key recorded by execute_join nor the lockbox's group_col
    was registered anywhere pages/01 reads, so the subject ID arrived in the
    feature multiselect pre-ticked — memorized row identity on a random split,
    and the top of the SHAP table in the manuscript.
    """

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from utils.combine import clear_registered_reserved_columns
        clear_registered_reserved_columns()
        yield
        clear_registered_reserved_columns()

    def test_a_registered_join_key_is_reserved(self):
        from utils.combine import (
            SOURCE_COLUMN, is_reserved_column, reserved_columns,
            set_reserved_columns,
        )
        set_reserved_columns(["SEQN"], "the ID these files were merged on",
                             role="join_key")
        assert is_reserved_column("SEQN")
        assert "SEQN" in reserved_columns()
        assert SOURCE_COLUMN in reserved_columns()

    def test_a_registered_group_column_is_reserved(self):
        from utils.combine import is_reserved_column, register_reserved_column
        register_reserved_column("subject_id", "the column the split was drawn by",
                                 role="group_col")
        assert is_reserved_column("subject_id")

    def test_the_reason_is_available_for_disclosure(self):
        from utils.combine import register_reserved_column, reserved_column_reason
        register_reserved_column("SEQN", "the ID these files were merged on",
                                 role="join_key")
        assert "merged on" in reserved_column_reason("SEQN")

    def test_registering_a_role_replaces_the_previous_key_for_it(self):
        """A key from an abandoned preview must not bar a real predictor."""
        from utils.combine import is_reserved_column, set_reserved_columns
        set_reserved_columns(["age"], "join key", role="join_key")
        set_reserved_columns(["SEQN"], "join key", role="join_key")
        assert is_reserved_column("SEQN")
        assert not is_reserved_column("age")

    def test_roles_do_not_clobber_each_other(self):
        from utils.combine import (
            is_reserved_column, register_reserved_column, set_reserved_columns,
        )
        register_reserved_column("subject_id", "split column", role="group_col")
        set_reserved_columns(["SEQN"], "join key", role="join_key")
        assert is_reserved_column("subject_id") and is_reserved_column("SEQN")

    def test_ordinary_columns_are_still_offered(self):
        from utils.combine import is_reserved_column, set_reserved_columns
        set_reserved_columns(["SEQN"], "join key", role="join_key")
        assert not is_reserved_column("age")
        assert not is_reserved_column("glucose")

    def test_the_page_filters_the_feature_pool_with_this_predicate(self):
        """The pool pages/01 builds is exactly all_cols minus target and reserved."""
        from utils.combine import (
            SOURCE_COLUMN, is_reserved_column, set_reserved_columns,
        )
        set_reserved_columns(["SEQN"], "join key", role="join_key")
        all_cols = ["SEQN", "age", "glucose", SOURCE_COLUMN, "chol"]
        target = "chol"
        pool = [c for c in all_cols if c != target and not is_reserved_column(c)]
        assert pool == ["age", "glucose"]

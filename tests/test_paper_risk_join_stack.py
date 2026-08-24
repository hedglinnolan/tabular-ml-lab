"""Regressions for the join/stack canonicalization defects (paper-risk sprint).

One class per finding id. Each defends a merge or stack that produced a
CONFIDENTLY WRONG analysis frame — rows silently dropped, rows silently
mis-paired, one variable split in two, or a participant number handed to the
model as a predictor.

The rule being defended throughout: the app may be silent, and it may refuse,
but it must never assert something false.
"""
from __future__ import annotations

import ast
import io
import pathlib

import numpy as np
import pandas as pd
import pytest

from ml.join_doctor import (
    diagnose_join, execute_join, find_key_candidates, key_reading,
    normalize_key, numeric_key_precision_loss, suggest_best,
)
from utils.combine import plan_stack

RNG = np.random.RandomState(0)

REPO = pathlib.Path(__file__).resolve().parent.parent
PAGE_01 = REPO / "pages" / "01_Upload_and_Audit.py"


@pytest.fixture(autouse=True)
def _deterministic():
    RNG.seed(0)


def _first_sentence(text: str) -> str:
    """What the researcher reads before deciding whether to read on."""
    head, _, _ = text.partition(". ")
    return head


class _FakeStreamlit:
    """Enough of the Streamlit surface to RUN a render function in a test.

    The point of these tests is that the real registration site runs, not a
    copy of it, so the widgets answer deterministically and everything else is
    recorded and discarded.
    """

    def __init__(self, press: bool = True, radio_index: int | None = None):
        self.session_state: dict = {}
        self.press = press
        self.radio_index = radio_index
        self.said: list = []

    def selectbox(self, label, options, index=0, **kw):
        return list(options)[index or 0]

    def radio(self, label, options, index=0, **kw):
        return list(options)[self.radio_index if self.radio_index is not None else index]

    def multiselect(self, label, options, default=None, **kw):
        return list(default if default is not None else options)

    def toggle(self, *a, **kw):
        return False

    def button(self, *a, **kw):
        return self.press

    def columns(self, spec, **kw):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def expander(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def _record(*a, **kw):
            self.said.append((name, a[0] if a else None))
        return _record


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

    # ── the check is about VALUES, not about float64 ─────────────────────
    # Gating on float64 and hardcoding 2^53 let two live shapes through, and
    # both were narrated to the researcher as ordinary repeated visits.

    @staticmethod
    def _float32_frames():
        """float32 stops counting whole numbers at 2^24 — MRN territory."""
        left = pd.DataFrame({"SEQN": pd.Series([16777217.0, 16777216.0, 16777219.0],
                                               dtype="float32"),
                             "age": [50, 60, 70]})
        right = pd.DataFrame({"SEQN": pd.Series([16777217.0, 16777219.0], dtype="float32"),
                              "chol": [4.0, 5.0]})
        return left, right

    def test_float32_keys_are_refused_at_their_own_limit(self):
        left, right = self._float32_frames()
        loss = numeric_key_precision_loss(left["SEQN"])
        assert loss is not None and loss[2] == 2 ** 24
        d = diagnose_join(left, right, "SEQN", "SEQN", "inner")
        assert not d.can_proceed
        assert any("16,777,216" in b for b in d.blocking)
        # Never as fan-out: two participants collapsed into one is not a visit.
        assert not any("several rows per ID" in w for w in d.warnings)
        with pytest.raises(ValueError, match="lost their last digits"):
            execute_join(left, right, "SEQN", "SEQN", "inner")

    def test_a_parquet_upload_preserves_float32_and_is_still_refused(self):
        """The live reach path: parquet round-trips float32 unchanged."""
        pytest.importorskip("pyarrow")
        left, right = self._float32_frames()
        buf = io.BytesIO()
        left.to_parquet(buf, index=False)
        back = pd.read_parquet(io.BytesIO(buf.getvalue()))
        assert back["SEQN"].dtype == np.float32
        assert not diagnose_join(back, right, "SEQN", "SEQN", "inner").can_proceed

    def test_float16_keys_are_refused_at_2_to_the_11(self):
        s = pd.Series([2048.0, 2049.0], dtype="float16")
        loss = numeric_key_precision_loss(s)
        assert loss is not None and loss[2] == 2 ** 11

    def test_extension_float_dtypes_are_covered(self):
        for dtype, limit in (("Float32", 2 ** 24), ("Float64", 2 ** 53)):
            s = pd.Series([float(limit), float(limit) + 4, None], dtype=dtype)
            loss = numeric_key_precision_loss(s)
            assert loss is not None and loss[2] == limit, dtype

    def test_an_object_column_of_python_floats_is_checked_not_skipped(self):
        """object dtype is what the app's own stack path produces."""
        vals = [9007199254740993.0, 9007199254740992.0, 9007199254740995.0]
        left = pd.DataFrame({"SEQN": pd.Series(vals, dtype=object), "age": [50, 60, 70]})
        right = pd.DataFrame({"SEQN": pd.Series([vals[0], vals[2]], dtype=object),
                              "chol": [4.0, 5.0]})
        assert numeric_key_precision_loss(left["SEQN"]) is not None
        d = diagnose_join(left, right, "SEQN", "SEQN", "inner")
        assert not d.can_proceed
        assert not any("several rows per ID" in w for w in d.warnings)
        with pytest.raises(ValueError, match="lost their last digits"):
            execute_join(left, right, "SEQN", "SEQN", "inner")

    def test_a_text_id_stacked_onto_a_numeric_one_is_caught(self):
        """pd.concat of a text ID and a float ID gives object holding floats."""
        a = pd.DataFrame({"SEQN": ["9007199254740993", "9007199254740995"], "x": [1, 2]})
        b = pd.DataFrame({"SEQN": [9007199254740993.0, 9007199254740995.0], "y": [3, 4]})
        stacked = pd.concat([a, b], ignore_index=True)
        assert stacked["SEQN"].dtype == object
        assert numeric_key_precision_loss(stacked["SEQN"]) is not None

    def test_ids_kept_as_text_are_still_joinable(self):
        """The refusal's own advice — re-import the ID as text — must work."""
        left = pd.DataFrame({"SEQN": ["9007199254740993", "9007199254740992"],
                             "age": [50, 60]})
        right = pd.DataFrame({"SEQN": ["9007199254740993"], "chol": [4.0]})
        assert numeric_key_precision_loss(left["SEQN"]) is None
        out, _ = execute_join(left, right, "SEQN", "SEQN", "inner")
        assert len(out) == 1

    def test_ordinary_float32_and_object_keys_are_left_alone(self):
        small = pd.Series([1.0, 2.0, 3.0], dtype="float32")
        assert numeric_key_precision_loss(small) is None
        assert numeric_key_precision_loss(pd.Series(["A1", "B2", None], dtype=object)) is None
        assert numeric_key_precision_loss(pd.Series([1.0, 2.0, np.nan], dtype=object)) is None
        # Integers are exact at any magnitude — nothing has been lost.
        assert numeric_key_precision_loss(pd.Series([2 ** 60, 2 ** 60 + 1])) is None


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

    # ── evidence from the pair, not a shape quorum ───────────────────────
    # Keeping a token only when HALF the column shared its shape, and refusing
    # to decide at all below two other shapes, deleted the centre 'NA' from
    # exactly the two shapes real coding schemes take.

    @staticmethod
    def _both_sides(values):
        a = pd.DataFrame({"region": list(values), "x": range(len(values))})
        b = pd.DataFrame({"region": list(values), "y": range(len(values))})
        return a, b

    @pytest.mark.parametrize("values", [
        ["NA", "EU", "APAC", "LATAM"],      # codes of mixed length
        ["NA", "EMEA", "APAC"],             # no other code shares its shape
        ["NA", "NB"],                       # one other value: the old rule bailed
        ["NA", "Boston", "Seattle"],        # centre names, not codes
    ])
    def test_a_region_both_files_name_survives_whatever_its_neighbours_look_like(self, values):
        a, b = self._both_sides(values)
        out, _ = execute_join(a, b, "region", "region", "inner")
        assert sorted(out["region"].astype(str)) == sorted(values)
        d = diagnose_join(a, b, "region", "region", "inner")
        assert not any("no ID" in w for w in d.warnings), d.warnings

    def test_a_dash_used_as_a_code_by_both_files_survives(self):
        a = pd.DataFrame({"sp": ["-", "A", "B", "C"], "x": [1, 2, 3, 4]})
        b = pd.DataFrame({"sp": ["-", "A", "B", "C"], "y": [5, 6, 7, 8]})
        out, _ = execute_join(a, b, "sp", "sp", "inner")
        assert sorted(out["sp"].astype(str)) == ["-", "A", "B", "C"]

    def test_the_note_states_the_evidence_that_was_actually_used(self):
        """A false reason for a right decision still teaches something untrue."""
        a, b = self._both_sides(["NA", "Boston", "Seattle"])
        note = " ".join(diagnose_join(a, b, "region", "region", "inner").notes)
        assert "'NA'" in note and "real ID" in note
        # 'NA' shares its shape with nothing here, so the note must not say so.
        assert "shape of the other codes" not in note
        assert "both files use it" in note

    def test_a_token_only_one_file_uses_is_not_kept_by_pair_evidence(self):
        a = pd.DataFrame({"centre": ["NA", "Boston", "Seattle"], "x": [1, 2, 3]})
        b = pd.DataFrame({"centre": ["Boston", "Seattle"], "y": [4, 5]})
        assert not key_reading(a["centre"], b["centre"]).keep_tokens

    def test_a_column_of_subject_numbers_still_reads_na_as_a_blank(self):
        """The guard that stops the pair rule fusing unreadable IDs."""
        a = pd.DataFrame({"pid": ["1001", "1002", "NA"], "x": [1, 2, 3]})
        b = pd.DataFrame({"pid": ["1001", "1002", "NA"], "y": [4, 5, 6]})
        out, _ = execute_join(a, b, "pid", "pid", "inner")
        assert sorted(out["pid"].astype(str)) == ["1001", "1002"]

    # ── the disclosure has to be true in its FIRST sentence ──────────────

    def test_the_refusal_disclosure_leads_with_the_spelling_not_with_no_id(self):
        """It led with 'have no ID at all' and corrected itself further down.

        A researcher acts on the opening claim: they go looking for a gap in
        their file that is not there. Correcting it in sentence two is not a
        disclosure, it is a retraction nobody reads.
        """
        a = pd.DataFrame({"pid": ["1001", "1002", "NA"], "x": [1, 2, 3]})
        b = pd.DataFrame({"pid": ["1001", "1002", "NA"], "y": [4, 5, 6]})
        hits = [w for w in diagnose_join(a, b, "pid", "pid", "inner").warnings
                if "'NA'" in w]
        assert hits, "the discarded spelling must still be disclosed"
        lead = _first_sentence(hits[0])
        assert "no ID at all" not in lead
        assert "'NA'" in lead and "no value" in lead
        # And the reason precedes the consequence, not the other way round.
        assert hits[0].index("'NA'") < hits[0].index("dropped")

    def test_a_genuinely_blank_id_is_still_called_exactly_that(self):
        a = pd.DataFrame({"pid": ["1001", "1002", None], "x": [1, 2, 3]})
        b = pd.DataFrame({"pid": ["1001", "1002"], "y": [4, 5]})
        hits = [w for w in diagnose_join(a, b, "pid", "pid", "inner").warnings
                if "no ID" in w]
        assert hits and "no ID at all" in _first_sentence(hits[0])


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

    def test_underscore_and_space_are_the_same_spelling(self):
        """One export sanitizes its headers, the next does not."""
        c1 = pd.DataFrame({"SEQN": [1, 2], "blood pressure": [120, 130]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "blood_pressure": [140, 150]})
        plan = plan_stack({"c1": c1, "c2": c2})
        assert set(plan.name_variants) == {"blood pressure"}
        text = " ".join(plan.warnings)
        assert "'blood pressure'" in text and "'blood_pressure'" in text

    def test_the_underscore_variant_fires_at_high_overlap_too(self):
        base = {f"v{i}": [1, 2] for i in range(18)}
        w1 = pd.DataFrame({**base, "SEQN": [1, 2], "blood pressure": [1, 2]})
        w2 = pd.DataFrame({**base, "SEQN": [3, 4], "Blood_Pressure": [3, 4]})
        plan = plan_stack({"w1": w1, "w2": w2})
        assert "blood pressure" in plan.name_variants
        assert any("only by capitalization or spacing" in w for w in plan.warnings)

    def test_the_underscore_variants_are_not_merged_silently(self):
        c1 = pd.DataFrame({"SEQN": [1, 2], "blood pressure": [120, 130]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "blood_pressure": [140, 150]})
        plan = plan_stack({"c1": c1, "c2": c2})
        assert "blood pressure" in plan.all_columns
        assert "blood_pressure" in plan.all_columns

    def test_a_different_variable_is_not_reported_as_a_variant(self):
        """'age' vs 'age_years' is a question the app has no business asking."""
        c1 = pd.DataFrame({"SEQN": [1, 2], "age": [10, 20]})
        c2 = pd.DataFrame({"SEQN": [3, 4], "age_years": [30, 40]})
        assert not plan_stack({"c1": c1, "c2": c2}).name_variants

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

    # ── the REAL page and the REAL registration sites ────────────────────
    # A test that re-implements the page's filter inline passes whatever the
    # page does. These run the page's own expression and the app's own
    # registration call, so deleting either fails the test.

    @staticmethod
    def _page_tree():
        return ast.parse(PAGE_01.read_text(encoding="utf-8"))

    def test_the_page_binds_the_filter_to_utils_combines_predicate(self):
        aliases = {(a.name, a.asname)
                   for n in ast.walk(self._page_tree())
                   if isinstance(n, ast.ImportFrom) and n.module == "utils.combine"
                   for a in n.names}
        assert ("is_reserved_column", "_is_reserved") in aliases

    def test_the_pages_own_feature_filter_drops_a_registered_join_key(self):
        """The page's expression is compiled from its source and run here."""
        from utils.combine import (
            SOURCE_COLUMN, is_reserved_column, set_reserved_columns,
        )
        exprs = [n.value for n in ast.walk(self._page_tree())
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "feature_options"
                         for t in n.targets)
                 and isinstance(n.value, ast.ListComp)]
        assert exprs, "pages/01 no longer builds feature_options as a comprehension"
        code = compile(ast.Expression(body=exprs[0]), str(PAGE_01), "eval")
        set_reserved_columns(["SEQN"], "the ID these files were merged on",
                             role="join_key")
        pool = eval(code, {"all_cols": ["SEQN", "age", "glucose", SOURCE_COLUMN, "chol"],
                           "target_col": "chol",
                           "_is_reserved": is_reserved_column})
        assert pool == ["age", "glucose"]
        assert is_reserved_column("SEQN")

    def test_the_page_registers_the_group_column_where_the_seal_is_drawn(self):
        """The seal's own call, executed with the real registration function.

        Role-scoped REPLACEMENT since `DRIVE-068`: the same call has to release
        the column when the seal has no group column, or a withdrawn subject
        declaration leaves it barred from the predictors forever. Both
        directions are driven here, from the page's own source.
        """
        from utils.combine import is_reserved_column, set_reserved_columns
        calls = [n for n in ast.walk(self._page_tree())
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_set_reserved"]
        assert calls, "pages/01 no longer registers the lockbox group column"
        node = ast.Expression(body=calls[0])
        ast.fix_missing_locations(node)
        code = compile(node, str(PAGE_01), "eval")
        env = {"_set_reserved": set_reserved_columns,
               "_GROUP_COL_RESERVED_REASON": "split by"}
        eval(code, dict(env, _lb={"group_col": "subject_id"}))
        assert is_reserved_column("subject_id")
        eval(code, dict(env, _lb=None))
        assert not is_reserved_column("subject_id"), (
            "the seal drew no group column and the page kept it reserved")

    def test_the_real_combine_step_registers_the_join_key_at_commit(self, monkeypatch):
        import utils.combine_ui as combine_ui
        from utils.combine import is_reserved_column, reserved_column_reason

        fake = _FakeStreamlit(press=True)
        monkeypatch.setattr(combine_ui, "st", fake)
        demo = pd.DataFrame({"SEQN": [1, 2, 3], "age": [50, 60, 70]})
        labs = pd.DataFrame({"SEQN": [1, 2, 3], "chol": [4.0, 5.0, 6.0]})

        out = combine_ui.render_combine_step({"demo": demo, "labs": labs})
        assert out is not None and "chol" in out.columns
        assert is_reserved_column("SEQN")
        assert "merged on" in reserved_column_reason("SEQN")

    def test_nothing_is_reserved_until_the_user_commits(self, monkeypatch):
        """The key of an abandoned preview must not bar a real predictor."""
        import utils.combine_ui as combine_ui
        from utils.combine import is_reserved_column

        fake = _FakeStreamlit(press=False)
        monkeypatch.setattr(combine_ui, "st", fake)
        demo = pd.DataFrame({"SEQN": [1, 2, 3], "age": [50, 60, 70]})
        labs = pd.DataFrame({"SEQN": [1, 2, 3], "chol": [4.0, 5.0, 6.0]})

        assert combine_ui.render_combine_step({"demo": demo, "labs": labs}) is None
        assert not is_reserved_column("SEQN")

"""Regressions for defects found by the multi-file / JSON stress audits.

Each test names the finding it locks down. These exist because the same class
of bug came back twice: a fix landed for the reported case, the underlying rule
stayed wrong, and a slightly different file walked straight back into it.

The unifying rule being defended: the app may be silent, and it may refuse, but
it must never assert something false. 'high' confidence is the tier the UI
pre-selects, so 'high' is the app asserting — nothing uncertain may reach it.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from ml.import_doctor import (
    ALL_CHECKS, apply_fix, check_constant_columns, check_empty_rows_and_columns,
    check_numeric_sentinels, check_numeric_stored_as_text, diagnose,
    has_duplicate_labels, numeric_conversion_would_lose, reinfer_types,
)
from ml.join_doctor import find_key_candidates, suggest_best

RNG = np.random.RandomState(0)


# ── a measurement is not an identifier ───────────────────────────────────

class TestMeasurementIsNotAKey:
    """'age' matching 'age' across two survey cycles was rated `high`.

    It presents as an identically-named key with 77% coverage, and joining on
    it produced 144 rows of Cartesian nonsense that the app promised and then
    delivered — consistently, confidently wrong.
    """

    def _cycles(self):
        return (pd.DataFrame({"SEQN": range(1000, 1100), "age": RNG.randint(18, 80, 100)}),
                pd.DataFrame({"SEQN": range(1100, 1200), "age": RNG.randint(18, 80, 100)}))

    def test_age_is_not_auto_suggested(self):
        assert suggest_best(*self._cycles()) is None

    def test_age_is_rated_low_if_offered_at_all(self):
        for c in find_key_candidates(*self._cycles()):
            assert c.confidence == "low"

    def test_duplicates_on_both_sides_is_the_rule(self):
        left, right = self._cycles()
        cands = [c for c in find_key_candidates(left, right) if c.left_col == "age"]
        assert cands and cands[0].repeats_on_both_sides

    def test_a_shared_category_is_not_a_key(self):
        a = pd.DataFrame({"SEQN": range(83732, 83832), "sex": RNG.randint(1, 3, 100)})
        b = pd.DataFrame({"pid": range(90000, 90100), "sex": RNG.randint(1, 3, 100)})
        assert suggest_best(a, b) is None


# ── a row counter is not an identifier, even when the names agree ────────

class TestRowCounterIsNotAKey:
    """Matching names used to defeat the row-counter guard (1.0 >= 0.85).

    Two unrelated exports both carrying 'Unnamed: 0' overlap 100% by
    construction, and the app cannot tell that from two files that genuinely
    list the same people in the same order — so it must not assert either.
    """

    @pytest.mark.parametrize("name", ["row", "index", "id", "n", "Unnamed: 0",
                                      "rownum", "obs", "seq"])
    def test_counter_never_auto_suggested(self, name):
        a = pd.DataFrame({name: range(1, 51), "survey_score": RNG.rand(50)})
        b = pd.DataFrame({name: range(1, 51), "gdp": RNG.rand(50)})
        assert suggest_best(a, b) is None, f"{name!r} was auto-suggested as a key"

    def test_counter_is_still_offered_for_manual_choice(self):
        a = pd.DataFrame({"row": range(1, 51), "x": RNG.rand(50)})
        b = pd.DataFrame({"row": range(1, 51), "y": RNG.rand(50)})
        assert suggest_best(a, b, include_low=True) is not None


# ── repeated measures must be joinable at all ────────────────────────────

class TestRepeatedMeasuresAreJoinable:
    """_MIN_UNIQUENESS rejected any column repeating on >50% of rows.

    Three visits per subject puts the subject ID at 0.33 uniqueness, so it was
    discarded before pairing and the join became undiscoverable — for most of
    longitudinal nutrition research.
    """

    @pytest.mark.parametrize("visits", [2, 3, 4, 10, 25])
    def test_one_to_many_is_found(self, visits):
        subjects = pd.DataFrame({"SEQN": range(83732, 83782),
                                 "age": RNG.randint(18, 80, 50)})
        rows = pd.DataFrame({"SEQN": np.repeat(range(83732, 83782), visits),
                             "bp": RNG.normal(120, 10, 50 * visits)})
        best = suggest_best(subjects, rows)
        assert best is not None, f"{visits} visits/subject: no key found"
        assert best.left_col == "SEQN" and best.confidence == "high"

    def test_many_to_one_is_found(self):
        rows = pd.DataFrame({"SEQN": np.repeat(range(83732, 83782), 3),
                             "bp": RNG.normal(120, 10, 150)})
        subjects = pd.DataFrame({"SEQN": range(83732, 83782),
                                 "age": RNG.randint(18, 80, 50)})
        best = suggest_best(rows, subjects)
        assert best is not None and best.left_col == "SEQN"

    def test_a_real_one_to_one_key_still_rates_high(self):
        a = pd.DataFrame({"SEQN": range(83732, 83832), "age": RNG.randint(18, 80, 100)})
        b = pd.DataFrame({"SEQN": range(83732, 83832), "glucose": RNG.normal(100, 20, 100)})
        best = suggest_best(a, b)
        assert best.left_col == "SEQN" and best.confidence == "high"

    def test_differently_named_key_is_still_found_by_value(self):
        a = pd.DataFrame({"SEQN": range(83732, 83832), "age": RNG.randint(18, 80, 100)})
        b = pd.DataFrame({"patient_id": range(83732, 83832), "g": RNG.normal(100, 20, 100)})
        best = suggest_best(a, b)
        assert best is not None and best.right_col == "patient_id"


# ── duplicate column labels ──────────────────────────────────────────────

class TestDuplicateLabels:
    """df['bp'] returns a DataFrame when 'bp' names two columns.

    Two checks died on the ambiguous Series comparison, diagnose() swallowed
    the exception, and the user was shown a clean bill of health on a file with
    empty and constant columns in it.
    """

    def _frame(self):
        df = pd.DataFrame({
            "bp": [120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
            "bp2": [80, 85, 90, 95, 100, 105, 110, 115, 120, 125],
            "age": [45, 52, 999, 38, 61, 999, 47, 55, 60, 42],
            "empty": [None] * 10,
            "const": [1] * 10,
        })
        return df.rename(columns={"bp2": "bp"})

    def test_helper_detects_duplicates(self):
        assert has_duplicate_labels(self._frame())
        assert not has_duplicate_labels(pd.DataFrame({"a": [1], "b": [2]}))

    @pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__)
    def test_no_check_raises(self, check):
        check(self._frame())   # must not raise

    def test_duplicate_finding_is_reported_first(self):
        ids = [f.id for f in diagnose(self._frame())]
        assert "duplicate_columns" in ids

    def test_per_column_fixes_are_withheld_while_labels_are_ambiguous(self):
        # A fix that says "recode 999 in bp" cannot say WHICH bp.
        ids = [f.id for f in diagnose(self._frame())]
        assert not any(i.startswith("sentinel_missing__") for i in ids)

    def test_full_diagnosis_returns_after_the_rename(self):
        df = self._frame()
        dup = [f for f in diagnose(df) if f.id == "duplicate_columns"][0]
        fixed, _ = apply_fix(df, dup)
        ids = [f.id for f in diagnose(fixed)]
        assert "sentinel_missing__age" in ids
        assert "constant_columns" in ids

    def test_checks_see_each_duplicate_independently(self):
        assert [f.id for f in check_constant_columns(self._frame())] == ["constant_columns"]
        assert [f.id for f in check_empty_rows_and_columns(self._frame())] == ["empty_columns"]
        assert [f.id for f in check_numeric_sentinels(self._frame())] == ["sentinel_missing__age"]

    def test_a_failed_check_is_disclosed_not_swallowed(self, monkeypatch):
        import ml.import_doctor as mod

        def boom(df):
            raise RuntimeError("synthetic")
        monkeypatch.setattr(mod, "check_constant_columns", boom)
        monkeypatch.setattr(mod, "ALL_CHECKS",
                            tuple(boom if c is check_constant_columns else c
                                  for c in ALL_CHECKS))
        ids = [f.id for f in mod.diagnose(pd.DataFrame({"a": range(20)}))]
        assert "checks_failed" in ids


# ── numeric columns arriving as text ─────────────────────────────────────

class TestNumericStoredAsText:
    """The `raw_numeric >= 0.99` skip assumed a fresh read_csv.

    A frame built by promoting a header is entirely object dtype, so the
    flagship Excel case (title row above the header) produced an all-text frame
    that the doctor declared clean and on which age.mean() raised TypeError.
    """

    EXCEL = ("Nutrition Cohort Study 2024,,,\nExported 2024-03-14,,,\n,,,\n"
             "subject_id,age,bmi,site\n"
             + "".join(f"S{i:03d},{30 + i},{22 + i * 0.4:.1f},Boston\n" for i in range(1, 13))
             + "Total,,,\n")

    def test_pure_numeric_text_is_no_longer_invisible(self):
        df = pd.DataFrame({"age": ["31", "32", "33", "34", "35", "36", "37"]})
        assert [f.id for f in diagnose(df)] == ["numeric_as_text__age"]

    def test_promoting_a_header_restores_numeric_dtypes(self):
        raw = pd.read_csv(io.StringIO(self.EXCEL))
        fixed, _ = apply_fix(raw, diagnose(raw)[0])
        assert pd.api.types.is_numeric_dtype(fixed["age"])
        assert pd.api.types.is_numeric_dtype(fixed["bmi"])
        assert fixed["age"].mean() == pytest.approx(36.5)

    def test_promoting_keeps_a_text_identifier_as_text(self):
        raw = pd.read_csv(io.StringIO(self.EXCEL))
        fixed, _ = apply_fix(raw, diagnose(raw)[0])
        assert not pd.api.types.is_numeric_dtype(fixed["subject_id"])

    @pytest.mark.parametrize("vals,why", [
        (["007", "008", "009", "010", "011", "012"], "leading zero"),
        (["02139", "02140", "02141", "10001", "10002", "94105"], "leading zero"),
        ([str(9007199254740993 + i) for i in range(6)], "digits"),
    ])
    def test_identifiers_are_never_converted(self, vals, why):
        df = pd.DataFrame({"pid": vals})
        assert numeric_conversion_would_lose(df["pid"]) is not None
        assert not pd.api.types.is_numeric_dtype(reinfer_types(df)["pid"])
        assert [f.id for f in check_numeric_stored_as_text(df)] == []

    def test_big_integer_ids_keep_every_digit(self):
        vals = [str(9007199254740993 + i) for i in range(6)]
        out = reinfer_types(pd.DataFrame({"pid": vals}))
        assert list(out["pid"]) == vals

    def test_a_genuine_measurement_is_offered_at_high_confidence(self):
        df = pd.DataFrame({"glucose": ["95", "102", "110", "88", "121", "99"]})
        found = check_numeric_stored_as_text(df)
        assert len(found) == 1 and found[0].confidence == "high"
        out, _ = apply_fix(df, found[0])
        assert pd.api.types.is_numeric_dtype(out["glucose"])


# ── JSON: where the rows are must be answerable, never guessed ───────────

class TestJsonRowSetChoice:
    """The loader told users to do something the app gave them no way to do.

    For an ambiguous payload it raised "Pick which key holds your rows", but
    records_key was never wired to a widget and cached_parse_upload had no
    parameter to carry it. And when several recognised wrapper keys were
    present it silently took whichever came first in _JSON_WRAPPER_KEYS.
    """

    PAYLOAD = {
        "patients": [{"id": 1, "age": 40}, {"id": 2, "age": 55}],
        "visits": [{"id": 1, "bp": 120}, {"id": 1, "bp": 118}, {"id": 2, "bp": 130}],
        "meta": [{"exported": "2026-01-01"}],
    }

    def _bytes(self, obj):
        import json as _json
        return io.BytesIO(_json.dumps(obj).encode())

    def test_candidates_are_reported(self):
        from data_processor import inspect_json
        layout = inspect_json(self._bytes(self.PAYLOAD))
        assert layout.needs_a_choice
        assert layout.candidates == ["meta", "patients", "visits"]

    @pytest.mark.parametrize("key,rows,cols", [
        ("patients", 2, ["id", "age"]),
        ("visits", 3, ["id", "bp"]),
        ("meta", 1, ["exported"]),
    ])
    def test_each_candidate_is_loadable(self, key, rows, cols):
        from data_processor import load_json
        df = load_json(self._bytes(self.PAYLOAD), records_key=key)
        assert len(df) == rows and list(df.columns) == cols

    def test_records_key_reaches_through_load_tabular_data(self):
        from data_processor import load_tabular_data
        df = load_tabular_data(self._bytes(self.PAYLOAD), filename="x.json",
                               records_key="visits")
        assert len(df) == 3

    def test_records_key_reaches_through_the_upload_cache(self):
        import json as _json
        from utils.perf_cache import cached_parse_upload
        raw = _json.dumps(self.PAYLOAD).encode()
        df = cached_parse_upload(raw, "x.json", False, 0, "patients")
        assert len(df) == 2 and "age" in df.columns

    def test_a_wrapper_guess_is_disclosed(self):
        from data_processor import inspect_json
        layout = inspect_json(self._bytes({"data": [{"a": 1}], "results": [{"b": 2}]}))
        assert layout.chosen_key == "data"
        assert layout.candidates == ["data", "results"]
        assert "data" in layout.note and "change it" in layout.note

    def test_an_unambiguous_wrapper_still_says_what_it_did(self):
        from data_processor import inspect_json
        layout = inspect_json(self._bytes({"data": [{"a": 1}, {"a": 2}]}))
        assert layout.chosen_key == "data" and not layout.needs_a_choice
        assert "data" in layout.note

    @pytest.mark.parametrize("raw,expect", [
        (b"", "empty"),
        (b'{"a": [1,2', "not valid JSON"),
        (b"42", "single value"),
    ])
    def test_bad_input_returns_a_message_not_an_exception(self, raw, expect):
        from data_processor import inspect_json
        layout = inspect_json(io.BytesIO(raw))
        assert expect in layout.error

    def test_geojson_is_named_as_such(self):
        import json as _json
        from data_processor import inspect_json
        raw = _json.dumps({"type": "FeatureCollection", "features": []}).encode()
        assert "GeoJSON" in inspect_json(io.BytesIO(raw)).error

    def test_truncated_json_is_not_mislabelled_as_json_lines(self):
        from data_processor import inspect_json
        layout = inspect_json(io.BytesIO(b'{"a": [1,2'))
        assert layout.kind == "not_tabular"

    def test_real_json_lines_is_still_recognised(self):
        from data_processor import inspect_json
        raw = b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'
        layout = inspect_json(io.BytesIO(raw))
        assert layout.kind == "lines" and "3" in layout.note

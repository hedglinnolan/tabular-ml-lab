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


# ── files that need BOTH operations ──────────────────────────────────────

class TestMixedRelationships:
    """Step 2 asked ONE question for ALL files, and the NHANES shape — two
    cycles of two domains — has no right answer among the two it offered.

        relationship_hint said: link          (the wrong operation)
        stack everything -> 400 rows, correct answer is 200, every column ~50% null
        link everything  -> proposes joining two cycles on `age`

    This is the shape of the project's own demo dataset.
    """

    def _two_by_two(self):
        return {
            "demo_2017": pd.DataFrame({"SEQN": range(1000, 1100),
                                       "age": RNG.randint(18, 80, 100),
                                       "sex": RNG.randint(1, 3, 100)}),
            "demo_2019": pd.DataFrame({"SEQN": range(1100, 1200),
                                       "age": RNG.randint(18, 80, 100),
                                       "sex": RNG.randint(1, 3, 100)}),
            "labs_2017": pd.DataFrame({"SEQN": range(1000, 1100),
                                       "glucose": RNG.normal(100, 20, 100)}),
            "labs_2019": pd.DataFrame({"SEQN": range(1100, 1200),
                                       "glucose": RNG.normal(100, 20, 100)}),
        }

    def test_the_shape_is_recognised(self):
        from utils.combine import plan_combination
        assert plan_combination(self._two_by_two()).shape == "stack_then_link"

    def test_files_are_grouped_by_what_they_measure(self):
        from utils.combine import plan_combination
        groups = {g.label: set(g.members) for g in plan_combination(self._two_by_two()).groups}
        assert groups == {"demo": {"demo_2017", "demo_2019"},
                          "labs": {"labs_2017", "labs_2019"}}

    def test_the_plan_is_described_in_plain_language(self):
        from utils.combine import plan_combination
        text = plan_combination(self._two_by_two()).describe().lower()
        assert "stacked" in text and "linked" in text
        for jargon in ("inner join", "outer join", "union all", "cardinality"):
            assert jargon not in text

    def test_grouping_uses_columns_not_filenames(self):
        from utils.combine import plan_combination
        frames = {
            "file_A": pd.DataFrame({"SEQN": range(50), "age": range(50)}),
            "export final(2)": pd.DataFrame({"SEQN": range(50, 100), "age": range(50)}),
            "bloods": pd.DataFrame({"SEQN": range(100), "glucose": RNG.rand(100)}),
        }
        groups = [set(g.members) for g in plan_combination(frames).groups]
        assert {"file_A", "export final(2)"} in groups
        assert {"bloods"} in groups

    def test_three_domains_by_three_cycles(self):
        from utils.combine import plan_combination
        frames = {}
        for dom, cols in [("demo", ["age", "sex"]), ("labs", ["glucose", "chol"]),
                          ("diet", ["kcal", "fiber"])]:
            for i, yr in enumerate(["2015", "2017", "2019"]):
                frames[f"{dom}_{yr}"] = pd.DataFrame(
                    {"SEQN": range(i * 100, i * 100 + 100),
                     **{c: RNG.rand(100) for c in cols}})
        plan = plan_combination(frames)
        assert plan.shape == "stack_then_link"
        assert sorted(len(g.members) for g in plan.groups) == [3, 3, 3]

    @pytest.mark.parametrize("frames_fn,expected", [
        (lambda: {"1999-2000": pd.DataFrame({"SEQN": range(100), "age": range(100)}),
                  "2001-2002": pd.DataFrame({"SEQN": range(100, 230), "age": range(130)})},
         "stack"),
        (lambda: {"demographics": pd.DataFrame({"SEQN": range(200), "age": range(200)}),
                  "labs": pd.DataFrame({"SEQN": range(200), "glucose": RNG.rand(200)}),
                  "diet": pd.DataFrame({"SEQN": range(200), "kcal": range(200)})},
         "link"),
        (lambda: {"only": pd.DataFrame({"x": [1, 2, 3]})}, "single"),
    ])
    def test_the_shapes_that_already_worked_are_unchanged(self, frames_fn, expected):
        from utils.combine import plan_combination
        assert plan_combination(frames_fn()).shape == expected


class TestSourceColumnIsNeverAPredictor:
    """Stacking two groups then linking them produces '__source_file_demo' and
    '__source_file_labs' — the join suffixes the collision — and the feature
    pool's exact-match check let both through. A model that can see which file
    a row came from predicts the batch.
    """

    @pytest.mark.parametrize("name", [
        "__source_file", "__source_file_demo", "__source_file_labs",
        "__source_file_x", "__source_file_1",
    ])
    def test_every_suffixed_variant_is_reserved(self, name):
        from utils.combine import is_reserved_column
        assert is_reserved_column(name)

    @pytest.mark.parametrize("name", ["age", "source_file", "sex", "SEQN", "file_source"])
    def test_ordinary_columns_are_not_reserved(self, name):
        from utils.combine import is_reserved_column
        assert not is_reserved_column(name)


# ── two-level column headers ─────────────────────────────────────────────

class TestMultiIndexColumns:
    """Parquet round-trips MultiIndex columns and Excel two-row headers make
    them. diagnose_join raised KeyError on "('key', 'SEQN')" — the printed
    form of a tuple label — and pandas refuses to merge a 2-level frame
    against a 1-level one.
    """

    def _frames(self):
        return (pd.DataFrame({("key", "SEQN"): range(100), ("demo", "age"): range(100)}),
                pd.DataFrame({"SEQN": range(100), "glucose": RNG.rand(100)}))

    def test_diagnose_does_not_raise(self):
        from ml.join_doctor import diagnose_join, find_key_candidates
        a, b = self._frames()
        cand = [c for c in find_key_candidates(a, b) if "SEQN" in c.left_col][0]
        assert diagnose_join(a, b, cand.left_col, cand.right_col).matched_keys == 100

    def test_join_delivers_what_it_promised(self):
        from ml.join_doctor import diagnose_join, execute_join, find_key_candidates
        a, b = self._frames()
        cand = [c for c in find_key_candidates(a, b) if "SEQN" in c.left_col][0]
        d = diagnose_join(a, b, cand.left_col, cand.right_col)
        out, _ = execute_join(a, b, cand.left_col, cand.right_col, "inner")
        assert len(out) == d.predicted_rows == 100

    def test_loading_flattens_a_multiindex_parquet(self):
        from data_processor import load_tabular_data
        a, _ = self._frames()
        buf = io.BytesIO()
        a.to_parquet(buf)
        buf.seek(0)
        loaded = load_tabular_data(buf, filename="x.parquet")
        assert not isinstance(loaded.columns, pd.MultiIndex)
        assert list(loaded.columns) == ["key_SEQN", "demo_age"]

    def test_blank_and_unnamed_sublevels_collapse(self):
        from data_processor import flatten_columns
        m = pd.DataFrame({("SEQN", ""): range(5), ("demo", "age"): range(5),
                          ("Unnamed: 2_level_0", "bmi"): range(5)})
        assert list(flatten_columns(m).columns) == ["SEQN", "demo_age", "bmi"]

    def test_a_flat_frame_is_untouched(self):
        from data_processor import flatten_columns
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert list(flatten_columns(df).columns) == ["a", "b"]


# ── blockers must name the real obstacle ─────────────────────────────────

class TestBlockingMessagesNameTheCause:
    """"Check you picked the right columns" is wrong advice when the columns
    ARE right and something invisible stops the match."""

    def test_timezone_mismatch_is_named(self):
        from ml.join_doctor import diagnose_join
        left = pd.DataFrame({"visit_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                             "a": [1, 2]})
        right = pd.DataFrame({"visit_date": pd.to_datetime(
            ["2020-01-01", "2020-01-02"]).tz_localize("UTC"), "b": [7, 8]})
        msg = diagnose_join(left, right, "visit_date", "visit_date",
                            "inner", "clinic.csv", "labs.csv").blocking[0]
        assert "timezone" in msg
        assert "labs.csv" in msg and "clinic.csv" in msg
        assert "Check you picked the right columns" not in msg

    def test_differing_timezones_are_named(self):
        from ml.join_doctor import diagnose_join
        base = pd.to_datetime(["2020-01-01", "2020-01-02"])
        left = pd.DataFrame({"d": base.tz_localize("Europe/London"), "a": [1, 2]})
        right = pd.DataFrame({"d": base.tz_localize("UTC").tz_convert("America/New_York"),
                              "b": [7, 8]})
        msg = diagnose_join(left, right, "d", "d").blocking[0]
        assert "timezone" in msg.lower()

    def test_non_date_mismatch_keeps_the_original_advice(self):
        from ml.join_doctor import diagnose_join
        x = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        y = pd.DataFrame({"id": ["x", "y"], "w": [1, 2]})
        assert "Check you picked the right columns" in diagnose_join(x, y, "id", "id").blocking[0]

    def test_a_genuinely_absent_column_says_so(self):
        from ml.join_doctor import diagnose_join
        x = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        y = pd.DataFrame({"id": ["x", "y"], "w": [1, 2]})
        with pytest.raises(ValueError, match="is not in this file"):
            diagnose_join(x, y, "nope", "id")

    def test_date_vs_text_blocks_and_predicts_zero(self):
        """Finding 7: diagnose used to green-light a pair execute could not merge."""
        from ml.join_doctor import diagnose_join, execute_join
        left = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-02"]), "a": [1, 2]})
        right = pd.DataFrame({"d": ["2020-01-01", "2020-01-02"], "b": [7, 8]})
        d = diagnose_join(left, right, "d", "d")
        assert not d.can_proceed
        out, _ = execute_join(left, right, "d", "d", "inner")
        assert len(out) == d.predicted_rows == 0


# ── sentinel codes vs genuine coded scales ───────────────────────────────

class TestSentinelFalsePositives:
    """Finding 26. Excluding 7/8/9 as candidates BEFORE measuring the spread
    manufactures the gap that condemns them: in a real 1-9 Likert scale the 7s
    and 8s are answers, removing them leaves 1-6, and the 9 then looks like an
    outlier in a distribution it belongs to.

    A false positive here is worse than a miss. The user accepts the fix, real
    answers become NaN, and nothing about the result looks wrong.
    """

    @pytest.mark.parametrize("label,vals", [
        ("Likert 1-9 where 9 is a real answer", list(range(1, 10)) * 22),
        ("counts 0-9", list(range(10)) * 20),
        ("age including a 9-year-old", [9, 25, 31, 44, 52, 63, 71, 38, 29, 55] * 20),
        ("percentages reaching 99", list(range(90, 101)) * 10),
        ("a 1-7 scale", list(range(1, 8)) * 20),
        ("0/1 binary", [0, 1] * 50),
        ("months 1-12", list(range(1, 13)) * 20),
    ])
    def test_genuine_data_is_not_flagged(self, label, vals):
        assert check_numeric_sentinels(pd.DataFrame({"q": vals})) == [], label

    @pytest.mark.parametrize("label,vals,expected", [
        ("1-5 with 9 = refused", [1, 2, 3, 4, 5] * 39 + [9] * 5, [9]),
        ("1-5 with a 7/8/9 code block", [1, 2, 3, 4, 5] * 39 + [7, 8, 9] * 3, [7, 8, 9]),
        ("1-5 with 99", [1, 2, 3, 4, 5] * 39 + [99] * 5, [99]),
        ("triglycerides with 999",
         [120, 150, 180, 210, 145, 167, 190, 175, 160, 155] * 20 + [999] * 4, [999]),
        ("BMI with 99",
         [18.0, 22.5, 27.1, 31.4, 24.8, 29.9, 33.2, 20.1, 26.7, 35.0] * 20 + [99] * 3, [99]),
        ("survey -9 = don't know", [1, 2, 3, 4, 5] * 39 + [-9] * 5, [-9]),
        ("age with 999", [25, 31, 44, 52, 63, 71, 38, 29, 55, 47] * 20 + [999] * 3, [999]),
        ("1-4 with both -9 and 999", [1, 2, 3, 4] * 49 + [-9] * 3 + [999] * 3, [-9, 999]),
    ])
    def test_real_sentinels_still_fire(self, label, vals, expected):
        found = check_numeric_sentinels(pd.DataFrame({"q": vals}))
        assert found, f"{label}: nothing flagged"
        assert sorted(int(v) for v in found[0].params["values"]) == expected, label

    def test_a_code_block_is_never_split_down_the_middle(self):
        """Distance alone left the 7 behind and recoded only 8 and 9 — which
        would silently keep 'refused' in the data as the number seven."""
        found = check_numeric_sentinels(
            pd.DataFrame({"q": [1, 2, 3, 4, 5] * 39 + [7, 8, 9] * 3}))
        assert sorted(int(v) for v in found[0].params["values"]) == [7, 8, 9]

    def test_sentinel_findings_are_never_auto_applied(self):
        found = check_numeric_sentinels(
            pd.DataFrame({"q": [1, 2, 3, 4, 5] * 39 + [9] * 5}))
        assert not found[0].auto_suggestable


class TestLossyFixesAreNeverPreSelected:
    """Findings 34 and 36: a fix that discards values must say so and must not
    be pre-selected. Re-verified rather than assumed."""

    def test_coercion_that_blanks_values_is_low_confidence(self):
        df = pd.DataFrame({"sodium": ["140 mmol/L"] * 96 + ["not measured"] * 4})
        found = [f for f in diagnose(df) if f.fix_kind == "coerce_numeric"]
        assert found and found[0].confidence == "low"
        assert not found[0].auto_suggestable
        assert "4" in found[0].fix_label          # the loss is stated up front

    def test_the_description_records_what_was_lost(self):
        df = pd.DataFrame({"sodium": ["140 mmol/L"] * 96 + ["not measured"] * 4})
        found = [f for f in diagnose(df) if f.fix_kind == "coerce_numeric"][0]
        out, desc = apply_fix(df, found)
        assert "could not be read" in desc
        assert int(out["sodium"].notna().sum()) == 96

    def test_none_as_a_real_answer_is_a_question_not_a_recommendation(self):
        from ml.import_doctor import check_text_missing_tokens
        df = pd.DataFrame({"medication": ["None", "Statin", None, "None", "Beta-blocker",
                                          "None", "Statin", None, "None", "Metformin",
                                          "None", "Statin"]})
        found = check_text_missing_tokens(df)
        assert found and found[0].confidence == "low"
        assert not found[0].auto_suggestable


# ── decimal comma vs thousands separator ─────────────────────────────────

class TestCommaReading:
    """Finding 44, and a regression I introduced fixing it.

    The original bug stripped the comma from European decimals, turning 22,5
    into 225. The fix over-corrected: _DECIMAL_COMMA's dot-thousands branch
    used `*`, so it matched a plain "45,000" with zero dot groups and claimed
    every US thousands-separated number as a European decimal — $45,000 became
    45.0, a 1000x under-scale, at 'high' confidence and pre-selected, with a
    description that never mentioned an interpretation had been chosen.

    Strictly worse than the original: bigger error, commoner format.
    """

    def _convert(self, vals):
        df = pd.DataFrame({"v": vals})
        found = [f for f in diagnose(df) if f.fix_kind == "coerce_numeric"]
        assert found, f"not flagged: {vals[:3]}"
        out, _ = apply_fix(df, found[0])
        return found[0], list(out["v"])

    @pytest.mark.parametrize("vals,expected", [
        (["45,000", "52,000", "61,500", "38,000", "72,000", "55,000"],
         [45000, 52000, 61500, 38000, 72000, 55000]),
        (["450,000", "512,000", "615,000", "380,000", "720,000", "550,000"],
         [450000, 512000, 615000, 380000, 720000, 550000]),
    ])
    def test_us_thousands_are_not_rescaled(self, vals, expected):
        _, got = self._convert(vals)
        assert got == expected

    @pytest.mark.parametrize("vals,expected", [
        (["22,5", "28,4", "31,0", "24,5", "19,8", "27,1"],
         [22.5, 28.4, 31.0, 24.5, 19.8, 27.1]),
        (["5,55", "6,10", "4,98", "7,25", "5,04", "6,33"],
         [5.55, 6.10, 4.98, 7.25, 5.04, 6.33]),
        (["1.234,5", "2.100,7", "980,3", "1.050,25", "3.400,0", "2.750,8"],
         [1234.5, 2100.7, 980.3, 1050.25, 3400.0, 2750.8]),
    ])
    def test_european_decimals_still_convert(self, vals, expected):
        _, got = self._convert(vals)
        assert got == pytest.approx(expected)

    def test_an_undecidable_comma_is_not_pre_selected(self):
        finding, _ = self._convert(["45,000", "52,000", "61,500",
                                    "38,000", "72,000", "55,000"])
        assert finding.confidence == "low"
        assert not finding.auto_suggestable

    def test_the_assumption_is_stated_in_the_finding(self):
        finding, _ = self._convert(["45,000", "52,000", "61,500",
                                    "38,000", "72,000", "55,000"])
        assert "thousands" in finding.detail and "decimals" in finding.detail

    def test_unambiguous_european_is_still_high_confidence(self):
        finding, _ = self._convert(["22,5", "28,4", "31,0", "24,5", "19,8", "27,1"])
        assert finding.confidence == "high" and finding.auto_suggestable

    def test_a_column_with_a_settling_value_is_not_called_ambiguous(self):
        from ml.import_doctor import comma_reading_is_ambiguous
        # "1.234,5" proves the whole column is European, so "5,555" is decimal.
        s = pd.Series(["1.234,5", "5,555", "2.100,7", "980,3", "3,141", "2,718"])
        assert not comma_reading_is_ambiguous(s)

    def test_units_are_unaffected(self):
        finding, got = self._convert(["140 mmol/L", "138 mmol/L", "142 mmol/L",
                                      "139 mmol/L", "141 mmol/L", "137 mmol/L"])
        assert got == [140, 138, 142, 139, 141, 137]
        assert finding.confidence == "high"


# ── the join diagnosis must name the real problem ────────────────────────

class TestDtypeMismatchIsJudgedOnTheUnderlyingType:
    """Findings 4 and 40. `is_numeric_dtype` was asked about the CONTAINER.

    A Categorical of integers is not "numeric" by that test, so a working
    category-vs-number join was blocked with the false claim that one file
    stores the key as text (4). And when two columns shared no values at all
    the type blocker fired ahead of the real problem, telling the user that
    fixing the types would "match 0 IDs" — advice to do something pointless
    (40).
    """

    def _diag(self, left, right):
        from ml.join_doctor import diagnose_join
        return diagnose_join(left, right, "id", "id", "inner", "fileA", "fileB")

    def test_category_of_ints_against_ints_is_not_a_mismatch(self):
        d = self._diag(pd.DataFrame({"id": pd.Categorical([1, 2, 3] * 10), "a": range(30)}),
                       pd.DataFrame({"id": [1, 2, 3] * 10, "b": range(30)}))
        assert not d.dtype_mismatch and not d.blocking and d.matched_keys == 3

    def test_float_against_int_is_not_a_type_problem(self):
        d = self._diag(pd.DataFrame({"id": [1.0, 2.0, 3.0], "a": [1, 2, 3]}),
                       pd.DataFrame({"id": [1, 2, 3], "b": [7, 8, 9]}))
        assert not d.dtype_mismatch

    @pytest.mark.parametrize("left_ids", [["001", "002", "003"], ["1000", "1001", "1002"]])
    def test_text_against_numbers_is_still_caught(self, left_ids):
        right_ids = [int(v) for v in left_ids]
        d = self._diag(pd.DataFrame({"id": left_ids, "a": [1, 2, 3]}),
                       pd.DataFrame({"id": right_ids, "b": [7, 8, 9]}))
        assert d.dtype_mismatch
        assert "text" in d.blocking[0] and "numbers" in d.blocking[0]

    def test_zero_overlap_is_not_blamed_on_the_types(self):
        d = self._diag(pd.DataFrame({"id": ["a01", "a02", "a03"], "a": [1, 2, 3]}),
                       pd.DataFrame({"id": [999, 998, 997], "b": [7, 8, 9]}))
        assert not d.dtype_mismatch
        assert "nothing to join on" in d.blocking[0]
        assert "matches 0 IDs" not in d.blocking[0]

    def test_case_and_spacing_is_reported_as_spacing(self):
        d = self._diag(
            pd.DataFrame({"id": [f"A{i:02d}" for i in range(1, 21)], "a": range(20)}),
            pd.DataFrame({"id": [f"a{i:02d} " for i in range(1, 21)], "b": range(20)}))
        assert d.needs_normalization and not d.dtype_mismatch
        assert d.matched_keys == 20


class TestBlowUpIsRefused:
    """Finding 23. A predicted 25,000,000-row many-to-many was allowed through
    with only a warning. On the laptop this app runs on that exhausts memory
    and the tab stops responding, which reads as 'the app is broken'."""

    def test_a_cartesian_blow_up_blocks(self):
        from ml.join_doctor import diagnose_join
        x = pd.DataFrame({"k": [1] * 5000, "a": range(5000)})
        y = pd.DataFrame({"k": [1] * 5000, "b": range(5000)})
        d = diagnose_join(x, y, "k", "k")
        assert d.predicted_rows == 25_000_000
        assert not d.can_proceed
        assert "will not finish" in d.blocking[0]

    def test_the_message_says_what_to_do_instead(self):
        from ml.join_doctor import diagnose_join
        x = pd.DataFrame({"k": [1] * 5000, "a": range(5000)})
        y = pd.DataFrame({"k": [1] * 5000, "b": range(5000)})
        msg = diagnose_join(x, y, "k", "k").blocking[0]
        assert "one row per subject" in msg

    @pytest.mark.parametrize("visits", [3, 10, 25])
    def test_a_legitimate_one_to_many_is_not_blocked(self, visits):
        from ml.join_doctor import diagnose_join
        s = pd.DataFrame({"SEQN": range(83732, 83782), "age": RNG.randint(18, 80, 50)})
        v = pd.DataFrame({"SEQN": np.repeat(range(83732, 83782), visits),
                          "bp": RNG.normal(120, 10, 50 * visits)})
        d = diagnose_join(s, v, "SEQN", "SEQN")
        assert d.can_proceed and d.predicted_rows == 50 * visits


class TestBlankFillIsDisclosed:
    """Finding 41. A left join where half the rows match nothing produced no
    warning at all — the researcher gets a variable that is 50% missing by
    construction and blames the data."""

    def _frames(self):
        return (pd.DataFrame({"id": range(100), "x": RNG.rand(100)}),
                pd.DataFrame({"id": range(50, 100), "y": RNG.rand(50)}))

    @pytest.mark.parametrize("how", ["left", "outer"])
    def test_unmatched_left_rows_are_disclosed(self, how):
        from ml.join_doctor import diagnose_join
        left, right = self._frames()
        d = diagnose_join(left, right, "id", "id", how, "demographics", "labs")
        blob = " ".join(d.warnings)
        assert "50" in blob and "blank" in blob and "labs" in blob

    def test_an_inner_join_still_says_rows_are_dropped(self):
        from ml.join_doctor import diagnose_join
        left, right = self._frames()
        d = diagnose_join(left, right, "id", "id", "inner", "demographics", "labs")
        assert "dropped" in " ".join(d.warnings)

    def test_a_complete_match_warns_about_nothing(self):
        from ml.join_doctor import diagnose_join
        left = pd.DataFrame({"id": range(100), "x": RNG.rand(100)})
        right = pd.DataFrame({"id": range(100), "y": RNG.rand(100)})
        d = diagnose_join(left, right, "id", "id", "left", "a", "b")
        assert not d.warnings


# ── before/after: does the map match what actually happens? ──────────────

class TestChangeMapMatchesReality:
    """The map is only worth showing if it is true. Every field is checked
    against the frame the engine actually produces — a map that disagrees with
    the result would be worse than no map at all."""

    def _frames(self):
        return (pd.DataFrame({"SEQN": range(1000, 1200), "age": RNG.randint(18, 80, 200),
                              "site": ["A"] * 200}),
                pd.DataFrame({"SEQN": range(1050, 1230), "glucose": RNG.normal(100, 20, 180),
                              "site": ["B"] * 180}))

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_predicted_rows_and_columns_match_the_result(self, how):
        from ml.join_doctor import execute_join
        from utils.combine_preview import describe_join
        left, right = self._frames()
        cm = describe_join(left, right, "SEQN", "SEQN", how, "demographics", "labs")
        out, _ = execute_join(left, right, "SEQN", "SEQN", how, "demographics", "labs")
        assert cm.after_rows == len(out)
        assert cm.after_cols == out.shape[1]

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_predicted_column_names_match_the_result(self, how):
        from ml.join_doctor import execute_join
        from utils.combine_preview import describe_join
        left, right = self._frames()
        cm = describe_join(left, right, "SEQN", "SEQN", how, "demographics", "labs")
        out, _ = execute_join(left, right, "SEQN", "SEQN", how, "demographics", "labs")
        assert [c.name for c in cm.columns] == [str(c) for c in out.columns]

    def test_every_row_is_accounted_for(self):
        from utils.combine_preview import describe_join
        left, right = self._frames()
        cm = describe_join(left, right, "SEQN", "SEQN", "inner", "demographics", "labs")
        # 150 matched + 50 left-only + 30 right-only = everything brought in.
        assert sum(g.n for g in cm.row_groups) == 200 + 180 - 150

    def test_a_collision_is_reported_with_its_new_name(self):
        from utils.combine_preview import describe_join
        left, right = self._frames()
        cm = describe_join(left, right, "SEQN", "SEQN", "inner", "demographics", "labs")
        renamed = {c.renamed_from: c.name for c in cm.renamed_columns}
        assert renamed == {"site": "site_demographics", "site": "site_labs"} or \
            set(renamed.values()) == {"site_demographics", "site_labs"}

    def test_stack_map_matches_the_stacked_frame(self):
        from utils.combine import execute_stack
        from utils.combine_preview import describe_stack
        frames = {"c17": pd.DataFrame({"SEQN": range(100), "age": RNG.randint(18, 80, 100),
                                       "glucose": RNG.rand(100)}),
                  "c19": pd.DataFrame({"SEQN": range(100, 230), "age": RNG.randint(18, 80, 130)})}
        cm = describe_stack(frames)
        out, _ = execute_stack(frames)
        assert cm.after_rows == len(out)
        assert cm.after_cols == out.shape[1]
        assert [c.name for c in cm.columns] == [str(c) for c in out.columns]


class TestConsequencesAreAboutTheStudy:
    """The point of the map. A row count is mechanically true and analytically
    useless; these say what the merge does to the ANALYSIS."""

    def test_fanout_says_n_is_no_longer_people(self):
        from utils.combine_preview import describe_join
        subj = pd.DataFrame({"SEQN": range(1000, 1050), "age": RNG.randint(18, 80, 50)})
        visits = pd.DataFrame({"SEQN": np.repeat(range(1000, 1050), 3),
                               "bp": RNG.normal(120, 10, 150)})
        cm = describe_join(subj, visits, "SEQN", "SEQN", "inner", "subjects", "visits")
        assert any("no longer the number of people" in c for c in cm.consequences)

    def test_dropping_rows_says_the_cohort_is_selected(self):
        from utils.combine_preview import describe_join
        left = pd.DataFrame({"SEQN": range(200), "age": RNG.randint(18, 80, 200)})
        right = pd.DataFrame({"SEQN": range(100, 200), "g": RNG.rand(100)})
        cm = describe_join(left, right, "SEQN", "SEQN", "inner", "demographics", "labs")
        assert any("subsample" in c and "biased" in c for c in cm.consequences)

    def test_blank_filling_is_named_as_missing_by_construction(self):
        from utils.combine_preview import describe_join
        left = pd.DataFrame({"SEQN": range(200), "age": RNG.randint(18, 80, 200)})
        right = pd.DataFrame({"SEQN": range(100, 200), "g": RNG.rand(100)})
        cm = describe_join(left, right, "SEQN", "SEQN", "left", "demographics", "labs")
        assert any("missing by construction" in c for c in cm.consequences)

    def test_stacking_names_the_batch_variable(self):
        from utils.combine_preview import describe_stack
        frames = {"a": pd.DataFrame({"SEQN": range(50), "x": RNG.rand(50)}),
                  "b": pd.DataFrame({"SEQN": range(50, 100), "x": RNG.rand(50)})}
        cm = describe_stack(frames)
        assert any("batch variable" in c for c in cm.consequences)

    def test_a_clean_one_to_one_join_says_nothing_alarming(self):
        from utils.combine_preview import describe_join
        left = pd.DataFrame({"SEQN": range(200), "age": RNG.randint(18, 80, 200)})
        right = pd.DataFrame({"SEQN": range(200), "g": RNG.rand(200)})
        cm = describe_join(left, right, "SEQN", "SEQN", "inner", "a", "b")
        assert cm.consequences == []


class TestSamePeopleDecidesGrouping:
    """Column-name overlap was the only signal, so two cycles that both gained
    a column (0.67 overlap, under the 0.8 threshold) were classed as different
    measurements on the same people — and the app proposed linking two cycles
    that share no participants at all."""

    AGE = staticmethod(lambda n: RNG.randint(18, 80, n))

    def test_cycles_with_schema_drift_still_stack(self):
        from utils.combine import plan_combination
        frames = {"cycle_2017": pd.DataFrame({"SEQN": range(1000, 1100),
                                              "age": self.AGE(100), "glucose": RNG.rand(100)}),
                  "cycle_2019": pd.DataFrame({"SEQN": range(1100, 1230), "age": self.AGE(130)})}
        assert plan_combination(frames).shape == "stack"

    def test_same_ids_means_link_not_stack(self):
        from utils.combine import plan_combination
        frames = {"demographics": pd.DataFrame({"SEQN": range(200), "age": self.AGE(200)}),
                  "labs": pd.DataFrame({"SEQN": range(200), "glucose": RNG.rand(200)})}
        assert plan_combination(frames).shape == "link"

    def test_an_id_named_column_outvotes_a_coincidental_match(self):
        from utils.combine import _same_people
        # SEQN is disjoint; `age` happens to be unique and identical. The ID
        # column decides — otherwise every pair of cycles reads as one cohort.
        a = pd.DataFrame({"SEQN": range(0, 50), "age": range(50)})
        b = pd.DataFrame({"SEQN": range(50, 100), "age": range(50)})
        assert _same_people(a, b) is False

    @pytest.mark.parametrize("name,expected", [
        ("SEQN", True), ("subject_id", True), ("participant", True), ("MRN", True),
        ("pid", True), ("record_id", True), ("usubjid", True),
        ("age", False), ("bmi", False), ("glucose", False), ("site", False),
    ])
    def test_id_name_detection(self, name, expected):
        from utils.combine import _looks_like_an_id_name
        assert _looks_like_an_id_name(name) is expected

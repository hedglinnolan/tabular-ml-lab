"""
Continuous-vs-discrete color choice for scatter overlays.

The EDA Feature Explorer used to hand the raw column to plotly express and let
it infer from dtype. A classification target coded 0/1 is numeric, so it came
out as two shades of one continuous ramp instead of two distinct colors — the
plot silently stopped answering the question it was there to answer.
"""
import numpy as np
import pandas as pd

from utils.column_utils import color_by_category


class TestClassificationTarget:

    def test_integer_coded_binary_target_is_discrete(self):
        assert color_by_category(pd.Series([0, 1, 1, 0, 1]), is_classification_target=True)

    def test_flag_wins_over_high_cardinality(self):
        """A many-class target is still categorical, however many codes it has."""
        codes = pd.Series(np.arange(40) % 25)
        assert color_by_category(codes, is_classification_target=True)

    def test_float_coded_classes_are_discrete_when_flagged(self):
        assert color_by_category(pd.Series([0.0, 1.0, 2.0]), is_classification_target=True)


class TestWithoutTheFlag:

    def test_strings_are_discrete(self):
        assert color_by_category(pd.Series(["male", "female", "male"]))

    def test_booleans_are_discrete(self):
        assert color_by_category(pd.Series([True, False, True]))

    def test_categorical_dtype_is_discrete(self):
        assert color_by_category(pd.Series(["a", "b"]).astype("category"))

    def test_small_integer_codes_are_discrete(self):
        assert color_by_category(pd.Series([1, 2, 3, 1, 2, 3]))

    def test_continuous_measurements_are_not_discrete(self):
        rng = np.random.default_rng(0)
        assert not color_by_category(pd.Series(rng.normal(100, 15, 200)))

    def test_many_distinct_integers_are_continuous(self):
        """Ages are whole numbers but they are a measurement, not a code."""
        rng = np.random.default_rng(1)
        ages = pd.Series(rng.integers(18, 90, 300))
        assert not color_by_category(ages)

    def test_boundary_at_max_discrete_levels(self):
        assert color_by_category(pd.Series(list(range(10)) * 3))
        assert not color_by_category(pd.Series(list(range(11)) * 3))

    def test_respects_a_custom_level_cap(self):
        codes = pd.Series(list(range(8)) * 3)
        assert color_by_category(codes, max_discrete_levels=10)
        assert not color_by_category(codes, max_discrete_levels=5)


class TestEdges:

    def test_all_missing_numeric_falls_back_to_continuous(self):
        assert not color_by_category(pd.Series([np.nan, np.nan], dtype=float))

    def test_missing_values_do_not_break_the_whole_number_check(self):
        assert color_by_category(pd.Series([0.0, 1.0, np.nan, 1.0]))

    def test_empty_series(self):
        assert not color_by_category(pd.Series([], dtype=float))

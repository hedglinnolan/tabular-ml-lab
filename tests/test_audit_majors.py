"""Four majors from the pre-PR audit.

* Key discovery truncated each side to the FIRST 200,000 distinct values in row
  order — a positional head-truncation applied independently per file, which is
  exactly what _key_tokens' own docstring promises it never does ("Sampling
  each file independently compares two different random subsets, so on files
  above the sample size the measured overlap collapses toward zero and the true
  key stops being proposed exactly when the data is large enough to matter").
* A Categorical join key crashed right/outer joins with a raw pandas message,
  seconds after the app predicted an exact row count.
* check_numeric_stored_as_text gated on >= 0.99 and then asserted "Every value
  is a plain number" at HIGH confidence — the tier the UI pre-selects — without
  counting what the conversion would blank. The same situation at a 90% parse
  rate was correctly reported as low confidence with the count shown.
* The holdout slider re-drew the lockbox without group_col, silently
  downgrading a subject-level split to a row-wise one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml.join_doctor as jd
from ml.import_doctor import diagnose
from ml.join_doctor import execute_join, find_key_candidates

RNG = np.random.RandomState(0)


@pytest.fixture(autouse=True)
def _deterministic_test_data():
    RNG.seed(0)


@pytest.fixture
def small_cap(monkeypatch):
    """Shrink the cap so the >200k region is testable in milliseconds."""
    monkeypatch.setattr(jd, "_MAX_DISTINCT", 500)


class TestSamplingKeepsTheSameKeySpace:

    def _pair(self, n):
        ids = np.arange(100000, 100000 + n)
        left = pd.DataFrame({"SEQN": ids, "age": RNG.randint(20, 80, n)})
        right = pd.DataFrame({"SEQN": RNG.permutation(ids),
                              "glucose": RNG.rand(n)})
        return left, right

    def test_the_key_survives_well_past_the_cap(self, small_cap):
        left, right = self._pair(2000)          # 4x the cap
        cands = [c for c in find_key_candidates(left, right) if c.left_col == "SEQN"]
        assert cands, "the true key vanished once the file got large"
        assert cands[0].coverage_left > 0.9 and cands[0].coverage_right > 0.9

    def test_different_row_orders_do_not_break_the_overlap(self, small_cap):
        """The old head-truncation compared two disjoint slices."""
        left, right = self._pair(3000)
        c = next(c for c in find_key_candidates(left, right) if c.left_col == "SEQN")
        assert c.coverage_left > 0.9

    def test_the_reported_count_describes_the_files_not_the_sample(self, small_cap):
        left, right = self._pair(2000)
        c = next(c for c in find_key_candidates(left, right) if c.left_col == "SEQN")
        assert 1800 <= c.n_matched <= 2200, f"reported {c.n_matched} of 2000"

    def test_an_estimate_is_never_asserted(self, small_cap):
        left, right = self._pair(2000)
        c = next(c for c in find_key_candidates(left, right) if c.left_col == "SEQN")
        assert c.sampled and c.confidence != "high"
        assert "estimate" in c.headline("demographics", "labs")

    def test_below_the_cap_nothing_is_sampled_or_hedged(self, small_cap):
        left, right = self._pair(200)
        c = next(c for c in find_key_candidates(left, right) if c.left_col == "SEQN")
        assert not c.sampled and c.n_matched == 200
        assert "estimate" not in c.headline()


class TestCategoricalKeysDoNotCrash:

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_every_join_type_completes(self, how):
        left = pd.DataFrame({"SEQN": pd.Categorical(["a", "b", "c"]),
                             "age": [1, 2, 3]})
        right = pd.DataFrame({"SEQN": pd.Categorical(["b", "c", "d"]),
                              "glucose": [4.0, 5.0, 6.0]})
        merged, _ = execute_join(left, right, "SEQN", "SEQN", how, "demo", "labs")
        assert isinstance(merged, pd.DataFrame) and len(merged) > 0

    def test_the_right_only_key_value_survives_an_outer_join(self):
        left = pd.DataFrame({"SEQN": pd.Categorical(["a", "b"]), "age": [1, 2]})
        right = pd.DataFrame({"SEQN": pd.Categorical(["b", "d"]), "g": [4.0, 6.0]})
        merged, _ = execute_join(left, right, "SEQN", "SEQN", "outer", "demo", "labs")
        assert "d" in set(merged["SEQN"].astype(str))


class TestHighConfidenceMeansNothingIsLost:

    def test_a_column_that_fully_parses_stays_high(self):
        df = pd.DataFrame({"glucose": [str(v) for v in range(100, 200)],
                           "y": RNG.randint(0, 2, 100)})
        f = next(f for f in diagnose(df) if f.id == "numeric_as_text__glucose")
        assert f.confidence == "high" and "Every value is a plain number" in f.detail
        assert "blanks" not in f.fix_label

    def test_a_column_that_nearly_parses_is_not_asserted(self):
        vals = [str(v) for v in range(100, 199)] + ["not measured"]
        df = pd.DataFrame({"glucose": vals, "y": RNG.randint(0, 2, 100)})
        f = next(f for f in diagnose(df) if f.id == "numeric_as_text__glucose")
        assert f.confidence == "low", "a data-destroying fix was pre-selectable"
        assert "Every value is a plain number" not in f.detail
        assert "1" in f.fix_label and "blank" in f.fix_label.lower()

"""Above the distinct-value cap, key discovery must still tell the truth.

Three defects the re-audit drew, all in the sampling the cap forces:

  - The keep-hash ran on RAW values. pd.util.hash_array buckets 1 (int64),
    1.0 (float64) and '1' (object) differently, so two files holding the
    identical ID set retained DISJOINT regions of the key space whenever their
    dtypes differed — the exact "two different random subsets" failure the
    sampling exists to avoid, and cross-dtype matching is a supported case.

  - The rescale used the LEFT column's ratio alone and coverage was never
    rescaled at all. A shared ID survives into the measured intersection only
    if it clears the STRICTER of the two thresholds, so when the files differ
    in size the true key was understated by that ratio, and the min_coverage
    gate — testing an un-rescaled number — dropped it before it was scored.

  - `if self.sampled: return "medium"` was written as a ceiling and read first,
    so it acted as a FLOOR: every junk pair on a large file jumped from 'low'
    to 'medium', where combine_ui offers it and pre-selects the top score. The
    honest "no shared ID was found" message stopped appearing.
"""
import numpy as np
import pandas as pd
import pytest

import ml.join_doctor as jd
from ml.join_doctor import find_key_candidates


CAP = 500


@pytest.fixture(autouse=True)
def small_cap(monkeypatch):
    """Exercise the real sampling branch without building 200k-row frames."""
    monkeypatch.setattr(jd, "_MAX_DISTINCT", CAP)
    yield


def ids(n, start=0):
    """Study IDs with gaps, so they are not mistaken for a row counter."""
    return 100_000 + 7 * np.arange(start, start + n)


def test_the_same_ids_stored_as_int_and_float_still_match():
    n = 4 * CAP
    left = pd.DataFrame({"subject_id": ids(n).astype("int64"),
                         "bmi": np.linspace(18, 40, n)})
    right = pd.DataFrame({"subject_id": ids(n).astype("float64"),
                          "crp": np.linspace(0, 9, n)})
    best = find_key_candidates(left, right)
    assert best, "the true key vanished entirely"
    top = best[0]
    assert (top.left_col, top.right_col) == ("subject_id", "subject_id")
    assert top.coverage_left > 0.8, f"coverage collapsed to {top.coverage_left:.2f}"
    assert top.n_matched > 0.8 * n, f"reported {top.n_matched:,} of {n:,} shared IDs"


def test_a_small_file_joined_to_a_huge_one_keeps_its_coverage():
    """5,000 participants against 100,000 specimen rows, at the shrunk cap."""
    n_small, n_big = CAP, 20 * CAP
    left = pd.DataFrame({"subject_id": ids(n_small), "bmi": np.linspace(18, 40, n_small)})
    right = pd.DataFrame({"subject_id": np.r_[ids(n_small), ids(n_big - n_small, n_small)],
                          "crp": np.linspace(0, 9, n_big)})
    best = find_key_candidates(left, right)
    assert best, "the true key was filtered out before it could be scored"
    top = best[0]
    assert (top.left_col, top.right_col) == ("subject_id", "subject_id")
    # every one of the small file's people is in the big file
    assert top.coverage_left > 0.8, f"coverage_left {top.coverage_left:.2f}"
    assert top.n_matched > 0.8 * n_small, f"n_matched {top.n_matched:,} of {n_small:,}"


def test_two_unrelated_big_files_are_not_offered_a_key():
    """The honest 'no shared ID' answer must survive the cap."""
    n = 3 * CAP
    rng = np.random.default_rng(4)
    # a plain row counter — the only thing two unrelated exports share
    left = pd.DataFrame({"row": np.arange(n), "bmi": rng.normal(27, 4, n)})
    right = pd.DataFrame({"row": np.arange(n), "crp": rng.normal(3, 1, n)})
    usable = [c for c in find_key_candidates(left, right) if c.confidence != "low"]
    assert not usable, (
        "offered " + ", ".join(f"{c.left_col}~{c.right_col} ({c.confidence})"
                               for c in usable))


def test_an_estimate_is_never_asserted_as_high():
    n = 4 * CAP
    left = pd.DataFrame({"subject_id": ids(n), "bmi": np.linspace(18, 40, n)})
    right = pd.DataFrame({"subject_id": ids(n), "crp": np.linspace(0, 9, n)})
    top = find_key_candidates(left, right)[0]
    assert top.sampled, "the fixture should have tripped the cap"
    assert top.confidence == "medium", f"an estimate reached {top.confidence}"


def test_the_ceiling_does_not_lift_a_row_counter():
    """A sampled row-counter pair stays 'low' — the ceiling is not a floor."""
    n = 3 * CAP
    c = jd.KeyCandidate(
        left_col="Unnamed: 0", right_col="row", coverage_left=1.0, coverage_right=1.0,
        n_matched=n, left_unique=n, right_unique=n, left_rows=n, right_rows=n,
        name_similarity=0.1, index_like=True, sampled=True,
    )
    assert c.confidence == "low"


def test_a_measurement_that_repeats_on_both_sides_stays_low_when_sampled():
    n = 3 * CAP
    c = jd.KeyCandidate(
        left_col="glucose", right_col="glucose", coverage_left=0.9, coverage_right=0.9,
        n_matched=n // 2, left_unique=n // 4, right_unique=n // 4,
        left_rows=n, right_rows=n, name_similarity=1.0, sampled=True,
        left_has_duplicates=True, right_has_duplicates=True,
    )
    assert c.repeats_on_both_sides and c.confidence == "low"


def test_below_the_cap_nothing_is_marked_as_an_estimate():
    n = CAP // 2
    left = pd.DataFrame({"subject_id": ids(n), "bmi": np.linspace(18, 40, n)})
    right = pd.DataFrame({"subject_id": ids(n), "crp": np.linspace(0, 9, n)})
    top = find_key_candidates(left, right)[0]
    assert not top.sampled
    assert top.n_matched == n and top.confidence == "high"

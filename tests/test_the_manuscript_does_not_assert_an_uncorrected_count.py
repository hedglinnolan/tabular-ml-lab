"""`AUDIT-001` and `AUDIT-004` — the manuscript stops reporting numbers whose
basis it does not name.

Found by the L29 anti-pattern audit, in code Streamlit users run today. The
generated Methods draft ended with:

    N of M tests yielded statistically significant results (p < 0.05).

no correction named and none applied. `research/METABOLOMICS_PACK.md` §06.3:

> Plotting raw p-values with a line at p = 0.05 on a 3,000-feature untargeted
> dataset **is an anti-pattern and would be flagged in review.**

and §10 lists *asterisks without the test or correction*. The pack's coaching
supplies the arithmetic that makes it concrete: at an uncorrected p < 0.05 you
would expect about 150 of 3,000 by chance, so 187 observed is consistent with
nothing happening.

**This is the governing rule failing in the artifact that is the product.** The
app even knew: `pages/09_Hypothesis_Testing.py` has warned on screen about the
family-wise error rate the whole time. The warning and the draft disagreed, and
the draft is the thing that leaves the building.

`AUDIT-004` is the same defect in a different sentence and rides here for that
reason: `quick_probe_baselines` deleted every row with a missing value in the
target or in any feature and reported an MAE with no statement of what it was
about. A reported n that is not the n the reader assumes is an uncorrected
count wearing different clothes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import multiplicity                                          # noqa: E402
from ml.narrative_engine import NarrativeEngine                      # noqa: E402
from utils.workflow_provenance import WorkflowProvenance             # noqa: E402


def _wide_family(n_tests: int = 400, seed: int = 0,
                 n_real: int = 0) -> WorkflowProvenance:
    """A recorded family the size of an untargeted panel, with `n_real` hits.

    The null p-values are drawn on [0.05, 1) rather than [0, 1). The first
    draft used the full interval and the corrected count came back 13 rather
    than the planted 12 — because one uniform draw fell under the BH line at
    rank 13, which is Benjamini-Hochberg working correctly and a fixture whose
    answer nobody could state in advance. Bounding the nulls away makes the
    corrected count exactly what was planted, so the test asserts the fix
    rather than a draw.
    """
    rng = np.random.default_rng(seed)
    prov = WorkflowProvenance()
    for i in range(n_tests):
        p = 1e-8 if i < n_real else float(rng.uniform(0.05, 1.0))
        prov.record_statistical_test("Mann-Whitney U", f"mz_{i:03d}", 1.0, p)
    return prov


def _paragraph(prov: WorkflowProvenance) -> str:
    return NarrativeEngine(prov).generate().statistical_validation


# ── the gate ────────────────────────────────────────────────────────────────

def test_a_wide_uncorrected_family_produces_no_count_of_significant_tests():
    """The gate. 400 features, a binary outcome, nothing going on: the
    paragraph a reviewer would accept says the tests are uncorrected and
    declines to count them."""
    text = _paragraph(_wide_family())

    assert "Mann-Whitney U" in text
    assert "No correction for multiple comparisons was applied" in text
    assert "not interpretable as a count of findings" in text
    assert "roughly 20 of 400 would be expected to do so by chance alone" in text

    # THE ASSERTION THAT MATTERS: no count of significant tests, in any of the
    # shapes the old sentence could take.
    assert "yielded statistically significant" not in text
    assert "significant results (p < 0.05)" not in text
    assert not any(f"{k} of 400 tests" in text for k in range(0, 401))


def test_a_corrected_family_reports_its_count_and_names_the_method():
    """The other branch. A corrected count IS a result, so it is stated —
    with the method and the threshold, which is what makes it checkable."""
    prov = _wide_family(n_real=12)
    summary = prov.apply_multiplicity_correction()
    assert summary["n_significant"] == 12, summary["n_significant"]

    text = _paragraph(prov)
    assert "Benjamini-Hochberg FDR" in text
    assert "across the 400 tests reported here" in text
    assert "12 remained significant at q < 0.05" in text
    assert "not interpretable" not in text


def test_the_correction_is_an_act_rather_than_something_the_draft_does():
    """Benjamini-Hochberg over *everything run in this session* is a decision
    about what the family is, and the app does not get to make it silently —
    the same rule the packs run on. Drafting twice must not correct anything."""
    prov = _wide_family(n_real=12)
    first, second = _paragraph(prov), _paragraph(prov)
    assert first == second
    assert "No correction" in first
    assert multiplicity.correction_of(
        prov.statistical_validation.tests_run) is None


def test_the_engine_is_not_reimplemented():
    """`ml/feature_selection.py:186` already calls `multipletests`. A
    hand-rolled Benjamini-Hochberg beside it is the two-engines failure."""
    source = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "ml", "multiplicity.py"), encoding="utf-8").read()
    assert "from statsmodels.stats.multitest import multipletests" in source
    for invented in ("sort(", "argsort", "cumsum", "* n / rank"):
        assert invented not in source, (
            f"{invented!r} suggests a second Benjamini-Hochberg was written")


# ── the arithmetic, checked where the answer is known ──────────────────────

def test_benjamini_hochberg_matches_the_hand_computable_case():
    """`LOOP.md` §06.3. Four p-values, BH by hand: q_i = min over j>=i of
    p_j * n / j."""
    tests = [{"test_name": "t", "p_value": p}
             for p in (0.001, 0.008, 0.040, 0.900)]
    summary = multiplicity.adjust(tests)
    q = [t["q_value"] for t in summary["tests"]]
    #  q4 = 0.900        q3 = min(0.900, 0.040*4/3) = 0.05333
    #  q2 = min(q3, 0.008*4/2) = 0.016     q1 = min(q2, 0.001*4/1) = 0.004
    assert q == pytest.approx([0.004, 0.016, 0.0533333333, 0.900], abs=1e-9)
    assert summary["n_significant"] == 2, (
        "the third p is 0.04 and survives raw thresholding; BH is what "
        "separates it")


def test_a_test_with_no_p_value_is_carried_rather_than_counted_as_one():
    """Padding the family with `p = 1` would shrink everybody else's q, and a
    test that reported no p is not a test that failed to reach significance."""
    tests = [{"test_name": "a", "p_value": 0.01},
             {"test_name": "b", "p_value": None},
             {"test_name": "c", "p_value": 0.02}]
    summary = multiplicity.adjust(tests)
    assert summary["n_adjusted"] == 2 and summary["n_without_p"] == 1
    assert summary["tests"][1]["q_value"] is None
    assert summary["tests"][0]["q_value"] == pytest.approx(0.02)


def test_a_partly_corrected_family_reads_as_uncorrected():
    """If some tests carry a correction and some do not, the family has not
    been corrected, and naming the method of the corrected subset would
    describe a family that does not exist."""
    assert multiplicity.correction_of([
        {"correction": "fdr_bh"}, {"correction": None}]) is None
    assert multiplicity.correction_of([{"correction": "fdr_bh"}]) == "fdr_bh"
    assert multiplicity.correction_of([]) is None


def test_the_expected_count_is_arithmetic_a_reader_can_check():
    assert multiplicity.expected_by_chance(3000) == pytest.approx(150.0)
    assert multiplicity.expected_by_chance(0) == 0.0
    assert multiplicity.family_wise_error_rate(1) == pytest.approx(0.05)
    assert multiplicity.family_wise_error_rate(0) == 0.0


def test_the_test_list_is_stable_between_drafts():
    """`list(set(...))` is not ordered, so the same analysis drafted twice
    produced different prose. A record that changes when nothing changed is
    not a record."""
    prov = WorkflowProvenance()
    for name in ("Shapiro-Wilk", "Breusch-Pagan", "Levene", "Mann-Whitney U"):
        prov.record_statistical_test(name, "residuals", 1.0, 0.5)
    text = _paragraph(prov)
    assert "Breusch-Pagan, Levene, Mann-Whitney U, Shapiro-Wilk" in text


# ── AUDIT-004 · the same defect in a different sentence ────────────────────

class _Signals:
    task_type_final = "regression"


def test_the_quick_baseline_reports_what_it_dropped():
    """`NUTRITION_PACK.md` §06 lists *silent listwise deletion with no N
    cascade* first. On a wide table with scattered missingness this removes
    most of the rows, and the MAE is then about whoever happened to be
    complete."""
    from ml.eda_actions import quick_probe_baselines

    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(0, 1, size=(200, 6)),
                         columns=[f"f{i}" for i in range(6)])
    frame["y"] = rng.normal(0, 1, 200)
    # One missing value in a different row of each feature: listwise deletion
    # removes six rows to save none.
    for i in range(6):
        frame.loc[i * 5, f"f{i}"] = np.nan

    out = quick_probe_baselines(frame, "y", [f"f{i}" for i in range(6)],
                                _Signals(), None)
    cascade = [f for f in out["findings"] if "of 200 rows" in f]
    assert cascade, out["findings"]
    assert "194 of 200 rows" in cascade[0]
    assert "6 were removed" in cascade[0]
    assert "about those 194 rows" in cascade[0]


def test_a_complete_table_gets_no_cascade_sentence():
    """A cascade that fires when nothing was dropped is noise, and noise is
    how a real cascade stops being read."""
    from ml.eda_actions import quick_probe_baselines

    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(0, 1, size=(120, 4)),
                         columns=[f"f{i}" for i in range(4)])
    frame["y"] = rng.normal(0, 1, 120)
    out = quick_probe_baselines(frame, "y", [f"f{i}" for i in range(4)],
                                _Signals(), None)
    assert not [f for f in out["findings"] if "were removed" in f]


def test_non_numeric_features_are_named_rather_than_silently_dropped():
    from ml.eda_actions import quick_probe_baselines

    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(0, 1, size=(120, 3)),
                         columns=["a", "b", "c"])
    frame["site"] = ["A", "B"] * 60
    frame["y"] = rng.normal(0, 1, 120)
    out = quick_probe_baselines(frame, "y", ["a", "b", "c", "site"],
                                _Signals(), None)
    assert [f for f in out["findings"]
            if "3 of 4 selected features are numeric" in f], out["findings"]

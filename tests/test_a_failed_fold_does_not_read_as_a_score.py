"""A cross-validation fold that dies must not reach the reader as a number.

`cross_val_score` defaults to `error_score=np.nan`, so a fold whose fit or
whose scoring raises comes back as a NaN in the score array. Plain `np.mean`
then poisons the mean and the SD, and the Cross-Validation Results table
printed `Mean Score: nan` — a cell that reads as a measurement when it is an
absence. That is the same defect class as a 999.0 VIF sentinel, and it lands
hardest at p/n > 1, where folds fail most.

**The audit's framing needed one correction and this file pins it down.** Total
fold failure does NOT arrive as NaN: scikit-learn raises `ValueError` when every
fit fails, and pages/06 already catches that and skips CV with a message. The
defect is exactly the PARTIAL failure, 1..k-1 folds.

**Two routes reach a NaN fold and only one of them announces itself.** A failed
FIT raises `FitFailedWarning` in the parent process, where it can be read; a
failed SCORE (the estimator fits, then `predict` raises) warns from inside the
loky worker, and the parent hears nothing. The score arrays are identical. So
`np.isnan(scores)` is the detector and the warning is reason-enrichment only —
asserted below in both directions, because a fix written as if the warning were
the detector would silently miss half the cases.

**What is deliberately NOT done, and is asserted as such:** `np.mean` is not
swapped for `np.nanmean`. That would delete the only visible symptom the defect
has and publish a mean over the survivors under a header promising a mean over
the folds. A fold usually dies on the harder training slice, so the survivors
are biased optimistic; the survivor mean exists only inside the disclosure,
welded to its own denominator.
"""
from __future__ import annotations

import ast
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.base import BaseEstimator, RegressorMixin              # noqa: E402
from sklearn.linear_model import LinearRegression                   # noqa: E402
from sklearn.model_selection import KFold, cross_val_score          # noqa: E402

from ml.eval import (                                               # noqa: E402
    cv_fold_disclosure, describe_fold_failures, perform_cross_validation,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── estimators that fail on exactly one fold ─────────────────────────────
#
# A row marker cannot single out one fold's TRAIN slice (each row is in k-1 of
# them), but it can single out one fold's TEST slice: the sentinel row is absent
# from exactly one train slice, so a fit that requires it fails exactly once.

class _FitFailsWhenSentinelHeldOut(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        if not np.any(np.asarray(X)[:, 0] > 900):
            raise ValueError("sentinel row absent from train slice")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class _ScoreFailsWhenSentinelHeldOut(BaseEstimator, RegressorMixin):
    """Fits fine every time; `predict` raises on the fold holding the sentinel."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        if np.any(np.asarray(X)[:, 0] > 900):
            raise ValueError("sentinel row in test slice")
        return np.full(len(X), self.mean_)


class _AlwaysFails(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        raise ValueError("this estimator never fits")

    def predict(self, X):                                # pragma: no cover
        return np.zeros(len(X))


def _frame(n=50, seed=0, sentinel_row=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=n)
    if sentinel_row is not None:
        X[sentinel_row, 0] = 999.0
    return X, y


# ── the clean path is untouched ──────────────────────────────────────────

def test_a_complete_fold_loop_discloses_nothing():
    """The ordinary path stores no disclosure, exactly like `MINE-027`'s."""
    X, y = _frame()
    res = perform_cross_validation(LinearRegression(), X, y, cv_folds=5,
                                   task_type='regression')
    assert res['fold_failures'] is None
    assert np.isfinite(res['mean']) and np.isfinite(res['std'])
    assert cv_fold_disclosure(res['scores']) is None
    assert describe_fold_failures("LR", res['fold_failures']) == ''


def test_the_clean_path_numbers_are_bit_for_bit_what_they_were():
    """No result moves. The disclosure is a tap on the scores, not a filter."""
    X, y = _frame()
    res = perform_cross_validation(LinearRegression(), X, y, cv_folds=5,
                                   task_type='regression')
    expected = -cross_val_score(LinearRegression(), X, y,
                                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                scoring='neg_mean_squared_error')
    assert res['scores'].tolist() == expected.tolist()
    assert res['mean'] == float(np.mean(expected))
    assert res['std'] == float(np.std(expected))


# ── a partial failure is recorded, and stays an absence ──────────────────

def test_a_dead_fold_is_named_by_number_and_by_reason():
    X, y = _frame()
    res = perform_cross_validation(_FitFailsWhenSentinelHeldOut(), X, y,
                                   cv_folds=5, task_type='regression')
    disclosure = res['fold_failures']
    assert disclosure is not None, "a fold died and the run said nothing"
    assert disclosure['n_folds'] == 5
    assert disclosure['n_failed'] == 1
    assert disclosure['n_scored'] == 4
    # 1-based, in split order, and it is the fold whose score is NaN.
    nan_positions = [i + 1 for i in np.flatnonzero(~np.isfinite(res['scores']))]
    assert disclosure['failed_folds'] == nan_positions
    assert any("sentinel row absent from train slice" in r
               for r in disclosure['reasons']), disclosure['reasons']


def test_the_mean_stays_nan_rather_than_becoming_a_survivor_mean():
    """`np.nanmean` here would delete the symptom instead of explaining it."""
    X, y = _frame()
    res = perform_cross_validation(_FitFailsWhenSentinelHeldOut(), X, y,
                                   cv_folds=5, task_type='regression')
    assert not np.isfinite(res['mean'])
    assert not np.isfinite(res['std'])
    survivors = res['scores'][np.isfinite(res['scores'])]
    assert res['mean'] != pytest.approx(float(np.mean(survivors)))
    # The survivor mean exists, but only where its denominator is attached.
    disclosure = res['fold_failures']
    assert disclosure['mean_over_scored'] == pytest.approx(float(np.mean(survivors)))
    assert disclosure['n_scored'] == len(survivors)


def test_a_fold_that_dies_while_scoring_is_caught_by_the_array_not_the_warning():
    """The warning route is silent here; the detector must not depend on it."""
    X, y = _frame()
    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter("always")
        res = perform_cross_validation(_ScoreFailsWhenSentinelHeldOut(), X, y,
                                       cv_folds=5, task_type='regression')
    assert not any(w.category.__name__ == "FitFailedWarning" for w in heard), (
        "this route is supposed to be the one the parent cannot hear; if it now "
        "warns, the test no longer covers the silent case")
    disclosure = res['fold_failures']
    assert disclosure is not None, "the silent route went undetected"
    assert disclosure['n_failed'] == 1
    assert disclosure['reasons'] == [], (
        "no reason is available on this route and inventing one would be worse "
        "than admitting it")


def test_the_prose_admits_when_there_is_no_reason():
    X, y = _frame()
    res = perform_cross_validation(_ScoreFailsWhenSentinelHeldOut(), X, y,
                                   cv_folds=5, task_type='regression')
    text = describe_fold_failures("RF", res['fold_failures'])
    assert "no reason" in text
    assert "1 of 5" in text


def test_the_warning_stream_is_tapped_not_swallowed():
    """Recording the FitFailedWarning must not hide it from the caller."""
    X, y = _frame()
    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter("always")
        perform_cross_validation(_FitFailsWhenSentinelHeldOut(), X, y,
                                 cv_folds=5, task_type='regression')
    assert any(w.category.__name__ == "FitFailedWarning" for w in heard), (
        "the disclosure consumed the warning instead of passing it on")


def test_the_tap_still_re_emits_when_the_call_raises():
    """A `with` block re-emitting after the call swallows on the raising path.

    `error_score` defaults to NaN, so a failed FIT returns rather than raises —
    but the call can still raise outright (a scorer that raises, a splitter that
    refuses the data, a worker killed for memory), and the warnings captured on
    the way to that exception are often the best account of what went wrong. The
    re-emission is in a `finally` so they survive, and the exception itself is
    unchanged.
    """
    import ml.eval as eval_mod

    X, y = _frame()

    def _warns_then_raises(*args, **kwargs):
        warnings.warn("a witness to the failure", UserWarning)
        raise RuntimeError("the fold loop refused")

    original = eval_mod.cross_val_score
    eval_mod.cross_val_score = _warns_then_raises
    try:
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError) as excinfo:
                perform_cross_validation(LinearRegression(), X, y, cv_folds=5,
                                         task_type='regression')
    finally:
        eval_mod.cross_val_score = original

    assert "the fold loop refused" in str(excinfo.value), (
        "the real failure must reach the caller unchanged, not be replaced")
    assert any("a witness to the failure" in str(w.message) for w in heard), (
        "warnings captured before the raise were swallowed by the tap")


# ── total failure is a different, already-handled case ───────────────────

def test_every_fold_failing_still_raises_rather_than_returning_nan():
    """Pinned because the fix must not duplicate or fight pages/06's handler.

    scikit-learn raises when `num_failed_fits == num_fits`, and
    pages/06_Train_and_Compare.py already catches it, warns, and stores
    `cv_results=None`. Only the partial case reaches the table.
    """
    X, y = _frame(sentinel_row=None)
    with pytest.raises(ValueError):
        perform_cross_validation(_AlwaysFails(), X, y, cv_folds=5,
                                 task_type='regression')


# ── the prose every surface shares ───────────────────────────────────────

def test_the_disclosure_prose_never_prints_a_bare_survivor_mean():
    disclosure = cv_fold_disclosure(np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
                                    ["ValueError: singular matrix"])
    text = describe_fold_failures("SVR", disclosure)
    assert "nan" not in text.lower()
    assert "1 of 5" in text and "fold 3" in text
    assert "3.0000" in text, "the survivor mean is stated"
    assert "4 folds that did complete" in text, "…and never without its denominator"
    assert "optimistic" in text, "…and never without the caveat"
    assert "ValueError: singular matrix" in text


def test_the_disclosure_survives_every_fold_failing_if_it_is_ever_asked():
    disclosure = cv_fold_disclosure(np.array([np.nan, np.nan]))
    assert disclosure['mean_over_scored'] is None
    assert disclosure['n_scored'] == 0
    text = describe_fold_failures("SVR", disclosure)
    assert "nothing to average" in text
    assert "nan" not in text.lower()


# ── the surfaces that publish the numbers ────────────────────────────────

def _page_source(name: str) -> str:
    with open(os.path.join(ROOT, "pages", name), encoding="utf-8") as fh:
        return fh.read()


def _cv_block(source: str, start: str, stop: str) -> str:
    i = source.index(start)
    return source[i:source.index(stop, i)]


def test_train_and_compare_does_not_hand_a_nan_to_the_results_table():
    block = _cv_block(_page_source("06_Train_and_Compare.py"),
                      'st.subheader("Cross-Validation Results")',
                      "# Boxplot of CV scores")
    assert "fold_failures" in block, "the table cannot tell a dead fold from a score"
    assert "'Folds Scored'" in block, "no denominator column"
    assert "'Mean Score': '—' if _failed" in block, (
        "the mean is still printed unguarded, so a NaN still reads as a score")


def test_the_boxplot_labels_a_model_that_lost_a_fold():
    """Plotly drops the NaN, so an unlabeled box looks tighter than the rest."""
    block = _cv_block(_page_source("06_Train_and_Compare.py"),
                      "# Boxplot of CV scores",
                      "# Pairwise statistical comparison")
    assert "fold_failures" in block and "folds)" in block


def test_the_exported_report_does_not_write_nan_into_the_manuscript():
    block = _cv_block(_page_source("10_Report_Export.py"),
                      '"### Cross-Validation Results"',
                      "compare_models_paired_cv")
    assert "fold_failures" in block
    assert "Folds Scored" in block
    assert "describe_fold_failures" in block, "the failure is never explained"
    assert '"—" if failed else f"{cv[\'mean\']:.4f}"' in block


# ── downstream consumers of a NaN std ────────────────────────────────────

def test_the_coach_does_not_write_maximum_fold_sd_equals_nan():
    """`nan < x` is False, so an unfiltered NaN walked past the early return."""
    from ml.model_coach import _detect_high_cv_variance

    results = {
        "svr": {"metrics": {"RMSE": 1.0},
                "cv_results": {"mean": float('nan'), "std": float('nan'),
                               "fold_failures": {"n_failed": 1, "n_folds": 5,
                                                 "n_scored": 4}}},
        "rf": {"metrics": {"RMSE": 2.0},
               "cv_results": {"mean": 1.5, "std": 0.01}},
    }
    insights = _detect_high_cv_variance(results, "regression")
    for insight in insights:
        for field in ("finding", "manuscript_text"):
            assert "nan" not in insight[field].lower(), insight[field]


def test_the_coach_still_fires_on_a_genuinely_noisy_run():
    """The NaN filter must not turn the detector off for everyone else."""
    from ml.model_coach import _detect_high_cv_variance

    results = {
        "a": {"metrics": {"RMSE": 1.0}, "cv_results": {"mean": 1.0, "std": 0.9}},
        "b": {"metrics": {"RMSE": 1.2}, "cv_results": {"mean": 1.2, "std": 0.8}},
    }
    assert _detect_high_cv_variance(results, "regression"), (
        "a real high-variance run stopped being reported")


# ── the Methods section ──────────────────────────────────────────────────

_METHODS_KWARGS = dict(
    data_config={"feature_cols": ["a", "b"], "target_col": "y"},
    preprocessing_config={}, model_configs={}, split_config={},
    n_total=200, n_train=140, n_val=30, n_test=30,
    feature_names=["a", "b"], target_name="y", task_type="regression",
    metrics_used=["RMSE"], cv_folds=5,
)


def test_the_methods_section_does_not_claim_a_fold_loop_that_did_not_finish():
    """`cv_results` truthiness cannot see the difference; the disclosure can."""
    from ml.publication import generate_methods_section

    partial = {
        "rf": {"metrics": {"RMSE": 1.0},
               "cv_results": {"mean": float('nan'), "std": float('nan'),
                              "fold_failures": {"n_folds": 5, "n_scored": 4,
                                                "n_failed": 1,
                                                "failed_folds": [3],
                                                "mean_over_scored": 1.1,
                                                "reasons": []}}},
    }
    text = generate_methods_section(selected_model_results=partial,
                                    **_METHODS_KWARGS)
    assert "did not complete" in text
    assert "4 of 5 folds" in text


def test_the_narrative_engine_says_it_too_because_it_is_the_path_page_10_takes():
    """publication.py is the FALLBACK. This generator is what a reader sees.

    `_build_methods_section_for_export` tries NarrativeEngine first and only
    reaches `generate_methods_section` when provenance is empty — so a
    disclosure that lives solely in publication.py is a disclosure the exported
    manuscript does not carry on any ordinary run. The incompleteness rides on
    the training record for the same reason `hyperopt_trials` does.
    """
    from ml.narrative_engine import NarrativeEngine
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_training(
        models_trained=["rf", "xgb_reg"],
        use_cv=True, cv_folds=5,
        cv_models_run=["rf", "xgb_reg"],
        cv_incomplete={"xgb_reg": [3, 5]},
    )
    ctx = prov.get_methods_context()
    assert ctx["cv_incomplete"] == {"xgb_reg": [3, 5]}

    text = NarrativeEngine(prov)._gen_model_development()
    assert "did not complete" in text
    assert "3 of 5 folds" in text
    # And the over-claim it replaces must not survive beside it.
    assert "is not substituted for one" in text


def test_the_narrative_engine_stays_silent_when_every_fold_loop_finished():
    from ml.narrative_engine import NarrativeEngine
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_training(models_trained=["rf"], use_cv=True, cv_folds=5,
                         cv_models_run=["rf"])
    text = NarrativeEngine(prov)._gen_model_development()
    assert "did not complete" not in text
    assert "5-fold cross-validation was used for evaluation" in text


def test_an_older_training_record_without_the_field_still_loads():
    """`cv_incomplete` defaults to empty, and empty is 'nothing incomplete'.

    Unlike `cv_models_run`, absence here is not ambiguous: a record that lists
    cross-validated models and no incompleteness is a record of loops that
    finished, which is what every run before this field wrote.
    """
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_training(models_trained=["rf"], use_cv=True, cv_folds=5,
                         cv_models_run=["rf"])
    payload = prov.to_dict()
    payload["training"].pop("cv_incomplete")
    restored = WorkflowProvenance.from_dict(payload)
    assert restored.training.cv_incomplete == {}
    assert restored.get_methods_context()["cv_incomplete"] == {}


def test_the_page_records_the_incompleteness_it_already_displays():
    src = _page_source("06_Train_and_Compare.py")
    assert "cv_incomplete={" in src, (
        "the training record must carry what the CV table already shows")
    assert "fold_failures']['n_scored']" in src


def test_a_complete_fold_loop_gets_no_extra_sentence():
    from ml.publication import generate_methods_section

    clean = {"rf": {"metrics": {"RMSE": 1.0},
                    "cv_results": {"mean": 1.0, "std": 0.1}}}
    text = generate_methods_section(selected_model_results=clean,
                                    **_METHODS_KWARGS)
    assert "did not complete" not in text
    assert "5-fold cross-validation was used for internal validation" in text


# ── the paired comparison ────────────────────────────────────────────────

def test_the_paired_test_reports_the_folds_it_was_actually_computed_over():
    """`n_paired` travels with `p`, because the test drops unpaired folds.

    `ml/stats_tests.py` opens `paired_location_test` with
    `d = d[~np.isnan(d)]`. That is not changed — but it means a pair where one
    model lost a fold still yields a finite `p`, labelled "paired t-test",
    computed over a self-selected subset of the loop. The denominator has to
    come back with it or a display cannot tell the two cases apart.
    """
    from ml.eval import compare_models_paired_cv

    complete = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    holed = np.array([1.5, 2.5, np.nan, 4.5, np.nan])
    results = {
        "a": {"cv_results": {"scores": complete}},
        "b": {"cv_results": {"scores": complete + 0.5}},
        "c": {"cv_results": {"scores": holed}},
    }
    paired = compare_models_paired_cv(["a", "b", "c"], results)

    ab = paired[("a", "b")]
    assert ab["n_folds"] == 5 and ab["n_paired"] == 5
    assert np.isfinite(ab["mean_delta"]) and np.isfinite(ab["p"])

    ac = paired[("a", "c")]
    assert ac["n_folds"] == 5 and ac["n_paired"] == 3, (
        "two folds are unpaired and the count must say so")
    assert not np.isfinite(ac["mean_delta"])
    # The defect this guards: the statistic itself is still finite.
    assert np.isfinite(ac["p"]), (
        "if this ever becomes NaN the display gate below is redundant, not "
        "wrong — but today a survivor p is exactly what comes back")


def test_neither_surface_publishes_a_paired_test_over_unpaired_folds():
    """Blanking Mean Δ alone left a real-looking p beside the hole.

    A `Mean Δ` of "—" next to `p = 0.547` and `Significant: No` reads as an
    effect size that could not be computed beside a test that could — when in
    fact the test ran on the folds both models happened to survive. Both
    surfaces gate the whole comparison on `n_paired == n_folds`.
    """
    for name in ("06_Train_and_Compare.py", "10_Report_Export.py"):
        src = _page_source(name)
        assert '_complete = v.get("n_paired") == v.get("n_folds")' in src, (
            f"{name} must gate on the paired-fold denominator")
        assert '_p_ok = _complete and p is not None and np.isfinite(p)' in src, (
            f"{name} must not print a p for an incomplete pair")
        assert "unpaired folds" in src, (
            f"{name} must name the absence rather than leave an empty cell")


def test_the_pages_still_parse():
    for name in ("06_Train_and_Compare.py", "10_Report_Export.py"):
        ast.parse(_page_source(name), filename=name)

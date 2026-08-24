"""Paper-risk sprint, ML-core slice: the number in the manuscript is the number
that was computed, over the population it was computed on, for the class it is
about.

One test (or class) per finding. Each fails if its fix is reverted:

- `MINE-027`   metrics computed on the surviving rows carry the count of the
               rows that did not survive.
- `MODELS-009` the baselines get their own preprocessing, not whichever model's
               pipeline was first in checkbox order.
- `T0-BUILD-006` which outcome level is the event is stated, not left to the
               alphabet in silence.
- `TEST-018`   MAD capping may trim extremes; it may not delete a predictor.
- `STATE-011`  cluster features are ADDED, as every description of them says.
- `STATE-003`  positional plausibility bounds refuse a column set they were not
               built for, instead of gating one biomarker against another's band.
- `MINE-030`   a seed sweep reports its denominator and its failures.
- `MINE-025`   'representative' is not a property of an insertion-ordered dict.
- `RECORD-017` a naive timestamp is not stamped UTC.
- `SWEEP-023`  seed 0 is a seed.
- `CONTRACT-010` the recorded selection criterion is the metric the ranking used.
- `MISC-095`   torch is optional; the module imports without it and refuses at
               use with a message a person can read.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
PAGE_06 = (REPO / "pages" / "06_Train_and_Compare.py").read_text(encoding="utf-8")
PAGE_05 = (REPO / "pages" / "05_Preprocess.py").read_text(encoding="utf-8")


# ── MINE-027 · the denominator travels with the metric ───────────────────

def test_mine_027_regression_metrics_disclose_dropped_nonfinite_pairs():
    """A degenerate back-transform makes some predictions NaN. The metrics are
    then computed on the rows the model handled — biased optimistic, because the
    dropped rows are the ones it blew up on — and used to be returned as the
    model's held-out R²/RMSE with the only trace in a logger nobody reads."""
    from ml.eval import calculate_regression_metrics, regression_scoring_disclosure

    y_true = np.arange(10, dtype=float)
    y_pred = y_true + 0.1
    y_pred[:3] = np.nan

    disclosure = regression_scoring_disclosure(y_true, y_pred)

    assert disclosure['n_dropped_nonfinite'] == 3
    assert disclosure['n_scored'] == 7
    assert disclosure['n_pairs'] == 10
    assert np.isfinite(calculate_regression_metrics(y_true, y_pred)['R2'])


def test_mine_027_the_disclosure_is_not_a_metric():
    """It rode the metrics dict, and everything downstream ITERATES that dict:
    `n_dropped_nonfinite=30` became a Test Set Metric tile, two columns of the
    comparison table, and a term in the narrative's Methods sentence."""
    from ml.eval import calculate_regression_metrics

    y_true = np.arange(10, dtype=float)
    y_pred = y_true + 0.1
    y_pred[:3] = np.nan

    assert set(calculate_regression_metrics(y_true, y_pred)) == {
        'MAE', 'RMSE', 'R2', 'MedianAE'}


def test_mine_027_the_ordinary_path_discloses_nothing():
    """No truncation, no disclosure: a field that only ever reads zero would be
    stored beside every result and rendered by every consumer."""
    from ml.eval import calculate_regression_metrics, regression_scoring_disclosure

    y_true = np.arange(10, dtype=float)
    metrics = calculate_regression_metrics(y_true, y_true + 0.1)
    assert set(metrics) == {'MAE', 'RMSE', 'R2', 'MedianAE'}
    assert regression_scoring_disclosure(y_true, y_true + 0.1) is None


def test_mine_027_the_page_surfaces_the_truncation():
    assert "regression_scoring_disclosure" in PAGE_06, (
        "Train & Compare renders the regression metrics; if it does not read "
        "the dropped-pair count, the truncation is disclosed to nobody")
    assert "'test_scoring': _test_scoring" in PAGE_06, (
        "the disclosure must be stored BESIDE the metrics so the export can "
        "publish it — and outside them so nothing renders it as a score")


def test_mine_027_the_export_publishes_the_denominator():
    """pages/10's headline metrics and the .tex table are where the truncated
    numbers get quoted; the N has to be there, not in a Streamlit warning."""
    page10 = (REPO / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
    assert "test_scoring" in page10 and "held-out rows" in page10

    from ml.latex_report import _metrics_to_latex_table

    tex = _metrics_to_latex_table(
        {"rf": {"metrics": {"RMSE": 3.2, "R2": 0.41},
                "test_scoring": {"n_dropped_nonfinite": 3, "n_scored": 7,
                                 "n_pairs": 10}}},
        task_type="regression")
    assert "computed on 7 of 10 pairs" in tex
    assert "3 non-finite pair(s) excluded" in tex
    # And it stays a footnote, never a column.
    assert "n_dropped_nonfinite" not in tex

    clean = _metrics_to_latex_table(
        {"rf": {"metrics": {"RMSE": 3.2, "R2": 0.41}}}, task_type="regression")
    assert "Note:" not in clean


def test_mine_027_the_narrative_states_it_in_prose():
    from ml.narrative_engine import NarrativeEngine
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_training(models_trained=["rf"], primary_model="rf",
                         metrics_by_model={"rf": {"RMSE": 3.2, "R2": 0.41}})
    engine = NarrativeEngine(
        prov,
        manuscript_context={"selected_model_results": {
            "rf": {"metrics": {"RMSE": 3.2, "R2": 0.41},
                   "test_scoring": {"n_dropped_nonfinite": 3, "n_scored": 7,
                                    "n_pairs": 10}}}})
    text = engine._gen_model_evaluation()

    assert "n_dropped_nonfinite" not in text, (
        "a disclosure printed as `key=value` beside RMSE reads as a metric")
    assert "could not be scored" in text
    assert "remaining 7" in text


# ── MODELS-009 · the comparator owns its preprocessing ───────────────────

class TestModels009BaselinesHaveTheirOwnPipeline:

    def _frame(self, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "age": rng.normal(50, 10, n),
            "bmi": rng.normal(27, 4, n),
            "sex": rng.choice(["male", "female"], n),
        })
        df.loc[df.index[:5], "age"] = np.nan          # the imputer has work to do
        return df

    def test_the_baseline_matrix_does_not_come_from_a_model_pipeline(self):
        from ml.baseline_models import prepare_baseline_matrices

        df = self._frame()
        train, test = df.iloc[:40], df.iloc[40:]
        X_train_t, X_test_t, described = prepare_baseline_matrices(train, test)

        assert np.isfinite(X_train_t).all(), "baseline features still hold NaNs"
        assert X_train_t.shape[1] == X_test_t.shape[1]
        assert "median imputation" in described and "one-hot" in described

    def test_the_recipe_is_stated_on_every_baseline_result(self):
        from ml.baseline_models import prepare_baseline_matrices, train_baseline_models

        df = self._frame()
        rng = np.random.default_rng(4)
        y = rng.normal(0, 1, len(df))
        X_train_t, X_test_t, described = prepare_baseline_matrices(df.iloc[:40], df.iloc[40:])

        results = train_baseline_models(
            X_train_t, y[:40], X_test_t, y[40:],
            task_type="regression", n_bootstrap=25,
            preprocessing_description=described)

        assert results, "no baselines were produced"
        for name, res in results.items():
            assert res["preprocessing"] == described, (
                f"{name} does not record the preprocessing its numbers came from")

    def test_the_page_no_longer_borrows_the_first_models_pipeline(self):
        assert "Use the first model's preprocessing pipeline for baselines" not in PAGE_06
        block = PAGE_06.split("BASELINE MODEL COMPARISON", 1)[1].split(
            "CALIBRATION ANALYSIS", 1)[0]
        assert "fitted_preprocessing_pipelines" not in block, (
            "the baselines are transformed through a user model's pipeline "
            "again, so the comparator moves with checkbox order")
        assert "prepare_baseline_matrices" in block


# ── T0-BUILD-006 · the event class is stated ─────────────────────────────

class TestT0Build006TheEventClassIsDisclosed:

    def _clinical(self, n: int = 80) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        return pd.DataFrame({
            "age": rng.normal(60, 8, n),
            "bmi": rng.normal(28, 4, n),
            "outcome": np.where(rng.random(n) < 0.5, "improved", "stable"),
        })

    def _split(self):
        from ml.splits import SplitSpec, make_split
        return make_split(self._clinical(), ["age", "bmi"], "outcome",
                          "classification", SplitSpec(random_state=0))

    def test_the_mapping_is_recorded(self):
        split = self._split()
        assert split.class_encoding is not None
        assert split.class_encoding["positive_class"] == "stable", (
            "LabelEncoder sorts, so 'stable' is class 1 — if that changed, every "
            "published metric moved with it")
        assert split.class_encoding["encoding"] == {"improved": 0, "stable": 1}

    def test_the_mapping_is_disclosed_in_the_notes_the_page_renders(self):
        split = self._split()
        joined = " ".join(split.notes)
        assert "'stable' is class 1" in joined or '"stable" is class 1' in joined, (
            f"the event class is not stated anywhere the page can show it: {joined}")
        assert "alphabetical" in joined

    def test_the_page_records_the_encoding(self):
        assert "class_encoding" in PAGE_06, (
            "Train & Compare does not carry the outcome encoding into the record")


# ── TEST-018 · capping may not delete a predictor ────────────────────────

class TestTest018MadCappingDoesNotFlattenAColumn:

    def _matrix(self, n: int = 500):
        rng = np.random.default_rng(0)
        smoker = (rng.random(n) < 0.30).astype(float)      # mad == 0
        age = rng.normal(55, 12, n)
        age[0] = 400.0                                     # a genuine outlier
        return np.column_stack([smoker, age])

    def test_a_binary_flag_survives_mad_capping(self):
        from ml.preprocess_operators import OutlierCapping

        X = self._matrix()
        out = OutlierCapping(method="mad", params={"threshold": 3.5}).fit_transform(X)

        assert len(np.unique(out[:, 0])) == 2, (
            "the 30%-prevalence flag was collapsed onto its median: the model is "
            "fitted on a predictor that no longer exists, and the recipe still "
            "says MAD capping")
        assert out[:, 0].var() > 0

    def test_a_real_outlier_is_still_capped(self):
        from ml.preprocess_operators import OutlierCapping

        X = self._matrix()
        capper = OutlierCapping(method="mad", params={"threshold": 3.5}).fit(X)
        out = capper.transform(X)

        assert out[0, 1] < 400.0, "MAD capping stopped capping"
        assert capper.uncapped_columns_ == [0], (
            "the zero-MAD column is not named on the fitted step, so the recipe "
            "cannot say which columns were left uncapped")

    def test_the_recipe_names_the_uncapped_columns(self):
        from ml.pipeline import _describe_outlier
        from ml.preprocess_operators import OutlierCapping

        capper = OutlierCapping(method="mad", params={"threshold": 3.5}).fit(self._matrix())
        assert "left uncapped" in _describe_outlier(capper)


# ── STATE-011 · cluster features are added ───────────────────────────────

class TestState011KMeansAddsRatherThanReplaces:

    def _frame(self, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        return pd.DataFrame({
            "age": rng.normal(50, 10, n),
            "bmi": rng.normal(27, 4, n),
            "glucose": rng.normal(95, 12, n),
        })

    def _pipeline(self, n_clusters: int = 4):
        from ml.pipeline import build_preprocessing_pipeline
        return build_preprocessing_pipeline(
            numeric_features=["age", "bmi", "glucose"],
            categorical_features=[],
            use_kmeans_features=True,
            kmeans_n_clusters=n_clusters,
            kmeans_add_distances=True,
        )

    def test_the_original_predictors_are_still_in_the_matrix(self):
        df = self._frame()
        out = self._pipeline().fit_transform(df)
        out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)

        assert out.shape[1] == 3 + 4, (
            "enabling cluster features discarded the study's predictors and "
            "trained the model on distances to centroids alone")

    def test_the_feature_names_keep_the_predictors(self):
        from ml.pipeline import get_feature_names_after_transform

        pipe = self._pipeline()
        pipe.fit(self._frame())
        names = get_feature_names_after_transform(pipe, ["age", "bmi", "glucose"])

        assert "age" in names and "glucose" in names
        assert any(n.startswith("kmeans_dist_cluster_") for n in names)

    def test_the_recipe_says_added_because_they_are_added(self):
        from ml.pipeline import get_pipeline_recipe

        pipe = self._pipeline()
        pipe.fit(self._frame())
        recipe = get_pipeline_recipe(pipe)

        assert "added alongside the existing predictors" in recipe
        assert "REPLACED" not in recipe

    def test_the_page_copy_no_longer_promises_columns_it_removes(self):
        assert "**alongside** your existing predictors" in PAGE_05


# ── STATE-003 · a band belongs to the column it was computed for ─────────

class TestState003PlausibilityBoundsAreNotAppliedToStrangers:

    def test_the_gate_refuses_a_column_set_it_was_not_built_for(self):
        from ml.preprocess_operators import PlausibilityGate

        gate = PlausibilityGate(lower_bounds=[70.0, 0.5, 10.0],
                                upper_bounds=[140.0, 1.5, 60.0]).fit(None)
        with pytest.raises(ValueError, match="positional"):
            gate.transform(np.array([[95.0, 0.9], [200.0, 1.1]]))

    def test_the_gate_still_gates_the_columns_it_was_built_for(self):
        from ml.preprocess_operators import PlausibilityGate

        X = np.array([[95.0, 0.9], [200.0, 1.1]])
        out = PlausibilityGate(lower_bounds=[70.0, None],
                               upper_bounds=[140.0, None]).fit(X).transform(X)
        assert np.isnan(out[1, 0]) and not np.isnan(out[0, 0])
        assert not np.isnan(out[:, 1]).any()

    def test_the_page_realigns_the_bounds_to_the_columns_it_passes(self):
        assert "_align_to(numeric_features_safe)" in PAGE_05, (
            "the filtered numeric feature list is passed with unfiltered "
            "positional bounds again")
        build_calls = PAGE_05.count("unit_harmonization_factors=uf")
        assert build_calls == 0, (
            "a build call still hands over the unfiltered positional factors")

    def test_bounds_keyed_by_name_survive_the_realignment(self):
        from ml.pipeline import build_plausibility_bounds

        features = ["glucose", "creatinine", "age"]
        bounds = build_plausibility_bounds(features, [1.0, 1.0, 1.0])
        by_name = bounds["bounds_by_feature"]

        assert set(by_name) == set(features), (
            "build_plausibility_bounds no longer returns a name-keyed view, "
            "which is what makes realignment possible at all")


# ── Wave-2 handoff · the lockbox remainder is grouped by subject ─────────

class TestLockboxRemainderIsDividedBySubject:
    """The lockbox branch split the train/val remainder with a plain
    `train_test_split`, so a DECLARED subject could still have rows in both
    train and validation. The seal keeps owning the test set — routing to the
    grouped branch instead would draw a fresh test set past the seal and trade a
    train/validation leak for a train/test one."""

    def _longitudinal(self, n_subjects: int = 30, per: int = 4) -> pd.DataFrame:
        rng = np.random.default_rng(2)
        n = n_subjects * per
        return pd.DataFrame({
            "subject": np.repeat([f"S{i}" for i in range(n_subjects)], per),
            "age": rng.normal(50, 10, n),
            "bmi": rng.normal(27, 4, n),
            "y": rng.normal(0, 1, n),
        })

    def _split(self, df, sealed):
        from ml.splits import SplitSpec, make_split
        return make_split(df, ["age", "bmi"], "y", "regression",
                          SplitSpec(random_state=0, entity_id_col="subject"),
                          sealed)

    def test_no_subject_spans_train_and_validation(self):
        df = self._longitudinal()
        sealed = list(df.index[df.subject.isin([f"S{i}" for i in range(24, 30)])])
        split = self._split(df, sealed)

        train = set(df.loc[split.train_labels, "subject"])
        val = set(df.loc[split.val_labels, "subject"])
        assert split.strategy == "lockbox", "the seal must still own the test set"
        assert set(split.test_labels) == set(sealed)
        assert not (train & val), f"{len(train & val)} subject(s) span train and validation"

    def test_the_folds_inherit_the_grouping(self):
        df = self._longitudinal()
        sealed = list(df.index[df.subject.isin([f"S{i}" for i in range(24, 30)])])
        split = self._split(df, sealed)

        assert split.cv_strategy == "group"
        assert split.cv_groups_train is not None
        assert len(split.cv_groups_train) == len(split.train_labels)

    def test_a_seal_that_already_straddles_subjects_is_disclosed(self):
        df = self._longitudinal()
        sealed = list(df.sample(24, random_state=1).index)
        split = self._split(df, sealed)

        assert any("rows in BOTH the sealed test set" in n for n in split.notes), (
            "the split cannot repair a row-sealed lockbox, but it may not stay "
            "silent about a subject that sits on both sides of it")


# ── Wave-2 handoff · the probe's help text asks the seal ─────────────────

def test_the_coach_probe_help_text_does_not_assert_an_exclusion_it_cannot_see():
    """`train_row_mask` returns an all-True mask when nothing is sealed, so
    "TRAINING rows only" was a claim about an exclusion that may not exist —
    the shape of MINE-005."""
    assert "quarantine_is_active" in PAGE_05
    probe_block = PAGE_05.split("run_coach_probe", 1)[0][-1500:]
    assert "no test set is sealed" in probe_block, (
        "the probe still promises TRAINING rows only with no lockbox in force")


# ── MINE-030 · the seed sweep reports its denominator ────────────────────

class TestMine030SeedFailuresAreCounted:

    def _analysis(self):
        from ml.sensitivity import sensitivity_random_seeds

        def train(seed):
            if seed in (1, 7, 13):
                raise RuntimeError("convergence failure")
            return seed

        return sensitivity_random_seeds(
            train_fn=train, eval_fn=lambda m: {"RMSE": 1.0 + 0.01 * m},
            seeds=[0, 1, 7, 13, 99, 123, 456], baseline_seed=42)

    def test_attempts_and_failures_are_separate_numbers(self):
        analysis = self._analysis()
        assert analysis.n_attempted == 7
        assert analysis.n_failed == 3
        assert len(analysis.variations) == 4

    def test_the_description_that_reaches_the_manuscript_says_both(self):
        description = self._analysis().description
        assert "4 of 7" in description, description
        assert "failed" in description

    def test_the_summary_table_carries_the_denominator(self):
        from ml.sensitivity import sensitivity_summary_table

        table = sensitivity_summary_table([self._analysis()], "RMSE")
        assert int(table.iloc[0]["N attempted"]) == 7
        assert int(table.iloc[0]["N failed"]) == 3


# ── MINE-025 · no member of that dict is representative ──────────────────

class TestMine025PcaIsNotDescribedFromAnArbitraryModel:

    def test_agreeing_configs_collapse(self):
        from ml.publication import pca_technique_sentence

        sentence = pca_technique_sentence({
            "ridge": {"mode": "Fixed Components", "n_components": 10},
            "rf": {"mode": "Fixed Components", "n_components": 10},
        })
        assert "10 components" in sentence
        assert "preprocessing pipeline" in sentence, (
            "the clause sits in the feature-engineering subsection while its "
            "number comes from the preprocessing configs, and must say so")

    def test_disagreeing_configs_are_named_per_model(self):
        from ml.publication import pca_technique_sentence

        sentence = pca_technique_sentence({
            "ridge": {"mode": "Fixed Components", "n_components": 10},
            "rf": {"mode": "Fixed Components", "n_components": 5},
        })
        assert "ridge: 10 components" in sentence and "rf: 5 components" in sentence
        assert not re.fullmatch(r"PCA dimensionality reduction \(\d+ components\)", sentence)

    def test_the_answer_does_not_depend_on_insertion_order(self):
        from ml.publication import pca_technique_sentence

        a = {"ridge": {"mode": "Fixed Components", "n_components": 10},
             "rf": {"mode": "Fixed Components", "n_components": 5}}
        b = {"rf": a["rf"], "ridge": a["ridge"]}
        first_a = re.search(r"(\d+) components", pca_technique_sentence(a)).group(1)
        first_b = re.search(r"(\d+) components", pca_technique_sentence(b)).group(1)
        # Both orders name both models; neither promotes one to 'the study's'.
        assert "ridge" in pca_technique_sentence(b) and "rf" in pca_technique_sentence(a)
        assert (first_a, first_b) == ("10", "5")


# ── RECORD-017 · a naive timestamp is not UTC ────────────────────────────

class TestRecord017TheAuditTrailDoesNotStampLocalTimeAsUtc:

    def test_a_naive_timestamp_is_not_labeled_utc(self):
        from ml.publication import _format_audit_timestamp

        rendered = _format_audit_timestamp("2026-08-23T14:05:00")
        assert rendered == "2026-08-23 14:05"
        assert "UTC" not in rendered, (
            "every Record timestamp is datetime.now() — local wall clock — so "
            "labeling it UTC is a false provenance claim in an exported artifact")

    def test_an_aware_timestamp_is_converted_and_labeled(self):
        from ml.publication import _format_audit_timestamp

        assert _format_audit_timestamp("2026-08-23T14:05:00+02:00") == "2026-08-23 12:05 UTC"

    def test_an_unparseable_timestamp_still_renders(self):
        from ml.publication import _format_audit_timestamp

        assert _format_audit_timestamp("") == "Timestamp unavailable"
        assert "not a date" in _format_audit_timestamp("not a date")


# ── SWEEP-023 · seed 0 is a seed ─────────────────────────────────────────

class TestSweep023SeedZeroIsNotFortyTwo:
    """The live falsehood is `... or 42` in pages/10's reproducibility manifest,
    which this slice does not own (handoff). What is checked here is that seed 0
    survives every hop this slice DOES own, so the manifest's fix has something
    true to read."""

    def test_the_split_honors_seed_zero(self):
        from ml.splits import SplitSpec, make_split

        rng = np.random.default_rng(1)
        df = pd.DataFrame({"age": rng.normal(50, 10, 60),
                           "bmi": rng.normal(27, 4, 60),
                           "y": rng.normal(0, 1, 60)})

        zero = make_split(df, ["age", "bmi"], "y", "regression", SplitSpec(random_state=0))
        forty_two = make_split(df, ["age", "bmi"], "y", "regression", SplitSpec(random_state=42))

        assert list(zero.test_labels) != list(forty_two.test_labels), (
            "seed 0 produced the seed-42 partition, so a manifest reporting 42 "
            "would be indistinguishable from the truth")

    def test_the_page_does_not_coerce_a_falsy_seed(self):
        seed_lines = [l for l in PAGE_06.splitlines()
                      if "random_seed" in l and " or 42" in l]
        assert not seed_lines, (
            "a falsy-seed fallback on Train & Compare would publish 42 for the "
            f"most common seed in ML tutorials: {seed_lines}")


# ── CONTRACT-010 · the recorded criterion is the metric that ranked ──────

class TestContract010TheSelectionCriterionMatchesTheTask:

    def test_the_dead_session_key_is_gone(self):
        assert "st.session_state.get('task_type', '')" not in PAGE_06, (
            "nothing in the repository writes session_state['task_type'], so "
            "this read always took the Accuracy branch — including for every "
            "regression study")

    def test_the_criterion_reads_the_resolved_task_type(self):
        block = PAGE_06.split("_selection_metric", 1)[1][:200]
        assert "task_type_final_local" in block, (
            "the recorded selection criterion must come from the same resolved "
            "task type the ranking loop above used")

    def test_the_phrase_is_the_holdout_phrase_for_each_metric(self):
        from ml.holdout_selection import criterion_phrase

        assert "RMSE" in criterion_phrase("RMSE")
        assert "Accuracy" in criterion_phrase("Accuracy")


# ── MISC-101 · cross-validation is routine, so it defaults on ────────────

class TestMisc101CrossValidationDefaultsOnAtRuntime:
    """The predecessor of this class grepped PAGE_06 for the literal
    `get('use_cv', True)`. That string was present and the checkbox still
    rendered unchecked, because `init_session_state` seeds the key first and the
    page's fallback never runs. Every assertion here reads the live value."""

    @pytest.fixture
    def session(self):
        import streamlit as st
        st.session_state.clear()
        yield st.session_state
        st.session_state.clear()

    def test_the_seeded_default_is_on(self, session):
        from utils.session_state import init_session_state

        init_session_state()

        assert session["use_cv"] is True, (
            "the seed runs before the page's get('use_cv', True), so the seed "
            "IS the default the checkbox renders")

    def test_a_new_dataset_returns_to_the_default(self, session):
        from utils.session_state import (init_session_state,
                                         reset_data_dependent_state)

        init_session_state()
        session["use_cv"] = False
        reset_data_dependent_state()

        assert session["use_cv"] is True, (
            "the re-seed on a new dataset must land on the shipped default")

    def test_the_checkbox_renders_checked(self):
        """The rendered widget, not the value behind it."""
        pytest.importorskip("streamlit.testing.v1")
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from streamlit.testing.v1 import AppTest
        from tests.integration.conftest import (build_test_dataframe,
                                                inject_data_state)

        at = AppTest.from_file(str(REPO / "pages" / "06_Train_and_Compare.py"),
                               default_timeout=60)
        inject_data_state(at, build_test_dataframe())
        # The page stops above the checkbox without a built pipeline.
        at.session_state["preprocessing_pipelines_by_model"] = {
            "ridge": Pipeline([("imp", SimpleImputer())])}
        at.run()

        boxes = [c for c in at.checkbox if c.key == "train_use_cv"]
        assert boxes, "the Enable Cross-Validation checkbox did not render"
        assert boxes[0].value is True, (
            "the checkbox a user sees on arrival is the default, whatever the "
            "fallback in the source says")

    def test_the_neural_network_exclusion_is_stated(self):
        assert "Cross-validation skipped for Neural Network" in PAGE_06, (
            "the NN exclusion is a stated skip, not a silent one")


# ── MISC-095 · torch is optional ─────────────────────────────────────────

class TestMisc095TheNeuralNetworkModuleImportsWithoutTorch:

    def test_the_module_imports(self):
        import models.nn_whuber as nn_whuber

        assert hasattr(nn_whuber, "NNWeightedHuberWrapper")
        assert isinstance(nn_whuber.TORCH_AVAILABLE, bool)

    def test_the_import_is_guarded_at_source(self):
        source = (REPO / "models" / "nn_whuber.py").read_text(encoding="utf-8")
        assert not re.search(r"^import torch$", source, re.MULTILINE), (
            "an unguarded module-scope `import torch` takes the whole module "
            "down in the deliberately-torchless environment")
        assert "from __future__ import annotations" in source, (
            "the torch.Tensor annotations are evaluated at import time without it")

    @pytest.mark.parametrize("attr", ["NNWeightedHuberWrapper",
                                      "SklearnCompatibleNNRegressor",
                                      "SklearnCompatibleNNClassifier"])
    def test_construction_refuses_with_a_readable_message(self, attr):
        import models.nn_whuber as nn_whuber

        if nn_whuber.TORCH_AVAILABLE:
            pytest.skip("torch is installed here; the refusal path is not live")
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            getattr(nn_whuber, attr)()

    def test_the_registry_factory_refuses_the_same_way(self):
        import models.nn_whuber as nn_whuber
        from ml.model_registry import _create_nn

        if nn_whuber.TORCH_AVAILABLE:
            pytest.skip("torch is installed here; the refusal path is not live")
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            _create_nn("regression", 42)

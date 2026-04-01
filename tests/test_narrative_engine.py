"""Tests for NarrativeEngine — end-to-end manuscript generation from provenance.

Validates that the NarrativeEngine produces complete, correct, per-model-aware
manuscript sections from WorkflowProvenance + InsightLedger input.
"""

import pytest
from utils.workflow_provenance import WorkflowProvenance
from utils.insight_ledger import Insight, InsightLedger
from ml.narrative_engine import NarrativeEngine, ManuscriptDraft


@pytest.fixture
def full_provenance():
    """A complete workflow provenance record simulating a realistic analysis."""
    prov = WorkflowProvenance()

    prov.record_upload("glucose", "regression", ["age", "bmi", "insulin", "bp", "skin_thickness"], 768)
    prov.record_eda_analysis("Correlation Matrix")
    prov.record_eda_analysis("Distribution Plots")
    prov.record_table1()
    prov.record_feature_selection(
        method="consensus (mutual_info + f_regression)",
        n_before=5, n_after=3,
        features_kept=["age", "bmi", "insulin"],
        consensus_methods=["mutual_info", "f_regression"],
    )
    prov.record_split(
        strategy="stratified",
        train_n=538, val_n=115, test_n=115,
        random_seed=42,
    )
    prov.record_preprocessing(
        configs_by_model={
            "ridge": {"numeric_scaling": "standard", "categorical_encoding": "onehot",
                      "numeric_power_transform": "yeo-johnson", "numeric_outlier_treatment": "none"},
            "rf": {"numeric_scaling": "none", "categorical_encoding": "onehot",
                   "numeric_power_transform": "none", "numeric_outlier_treatment": "none"},
            "histgb_reg": {"numeric_scaling": "none", "categorical_encoding": "ordinal",
                          "numeric_power_transform": "none",
                          "numeric_outlier_treatment": "percentile_clip",
                          "numeric_outlier_params": {"lower": 5, "upper": 95}},
        },
        imputation_method="median",
    )
    prov.record_training(
        models_trained=["ridge", "rf", "histgb_reg"],
        primary_model="rf",
        selection_criteria="validation RMSE",
        use_cv=True, cv_folds=5,
        hyperparameters={
            "ridge": {"alpha": 1.0},
            "rf": {"n_estimators": 100, "max_depth": None},
            "histgb_reg": {"max_iter": 100, "learning_rate": 0.1},
        },
        metrics_by_model={
            "ridge": {"RMSE": 12.3, "R2": 0.78},
            "rf": {"RMSE": 10.1, "R2": 0.85},
            "histgb_reg": {"RMSE": 11.5, "R2": 0.81},
        },
    )
    prov.record_explainability(["SHAP", "Permutation Importance"], ["ridge", "rf", "histgb_reg"])
    prov.record_sensitivity(seed_stability=True, seed_stability_cv=2.3, feature_dropout=True)
    prov.record_statistical_test("Shapiro-Wilk", "residuals", 0.97, 0.03)
    prov.record_statistical_test("Breusch-Pagan", "residuals", 4.2, 0.12)

    return prov


@pytest.fixture
def full_ledger():
    """InsightLedger with resolved and acknowledged insights."""
    ledger = InsightLedger()

    # Resolved: skewness
    ledger.upsert(Insight(
        id="eda_skew_group", source_page="02_EDA", category="distribution",
        severity="warning", finding="3 features exhibit right skewness",
        implication="May affect linear models",
        model_scope=["linear", "neural"],
    ))
    ledger.resolve("eda_skew_group", "Yeo-Johnson for Ridge; raw for RF/HistGB", "05_Preprocess",
                   {"action_type": "power_transform", "per_model": {"ridge": "yeo-johnson", "rf": "none", "histgb_reg": "none"}})

    # Resolved: missing data
    ledger.upsert(Insight(
        id="eda_missing_moderate", source_page="02_EDA", category="data_quality",
        severity="warning", finding="2 features have 5-10% missing values",
        implication="May bias results",
    ))
    ledger.resolve("eda_missing_moderate", "Median imputation", "05_Preprocess",
                   {"action_type": "imputation", "method": "median"})

    # Acknowledged limitation
    ledger.upsert(Insight(
        id="eda_sufficiency_borderline", source_page="02_EDA", category="methodology",
        severity="warning", finding="Sample size (n=768) is adequate but not large for 5 predictors",
        implication="Limited power for complex interactions",
    ))
    ledger.acknowledge("eda_sufficiency_borderline", "Proceeded with regularization")

    # Strength
    ledger.upsert(Insight(
        id="eda_opportunity_clean_data", source_page="02_EDA", category="data_quality",
        severity="info", finding="Low overall missingness (<5%)",
        implication="Favorable for analysis",
    ))
    ledger.auto_acknowledge_gate("Preprocessing", source_pages=["02_EDA"])

    return ledger


class TestNarrativeEngineGeneration:

    def test_generates_all_sections(self, full_provenance, full_ledger):
        engine = NarrativeEngine(full_provenance, full_ledger)
        draft = engine.generate()

        assert draft.study_design != ""
        assert draft.predictor_variables != ""
        assert draft.missing_data != ""
        assert draft.data_preprocessing != ""
        assert draft.model_development != ""
        assert draft.model_evaluation != ""
        assert draft.sensitivity_analysis != ""
        assert draft.statistical_validation != ""
        assert draft.data_observations != ""
        assert draft.software_environment != ""

    def test_study_design_content(self, full_provenance):
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "regression" in draft.study_design
        assert "768" in draft.study_design
        assert "glucose" in draft.study_design
        assert "stratified" in draft.study_design
        assert "538" in draft.study_design  # training n
        assert "seed=42" in draft.study_design

    def test_predictor_variables_with_selection(self, full_provenance):
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "consensus" in draft.predictor_variables.lower()
        assert "5" in draft.predictor_variables  # before
        assert "3" in draft.predictor_variables  # after
        assert "age" in draft.predictor_variables
        assert "bmi" in draft.predictor_variables

    def test_per_model_preprocessing(self, full_provenance):
        """The core differentiator: per-model preprocessing described separately."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        # Should mention per-model differences using human-readable names
        assert "Ridge Regression" in draft.data_preprocessing
        assert "Random Forest" in draft.data_preprocessing
        assert "Histogram-based Gradient Boosting" in draft.data_preprocessing

        # Ridge gets scaling + transform
        assert "z-score" in draft.data_preprocessing.lower() or "standardization" in draft.data_preprocessing.lower()
        assert "yeo-johnson" in draft.data_preprocessing.lower()

        # HistGB gets outlier clipping
        assert "percentile" in draft.data_preprocessing.lower() or "clip" in draft.data_preprocessing.lower()

    def test_identical_preprocessing_not_per_model(self):
        """When all models share preprocessing, don't describe per-model."""
        prov = WorkflowProvenance()
        prov.record_upload("target", "regression", ["a", "b"], 100)
        same = {"numeric_scaling": "standard", "categorical_encoding": "onehot"}
        prov.record_preprocessing({"ridge": same, "rf": same}, "mean")

        engine = NarrativeEngine(prov)
        draft = engine.generate()

        assert "All models shared" in draft.data_preprocessing
        assert "Ridge Regression" not in draft.data_preprocessing

    def test_model_development(self, full_provenance):
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "Ridge Regression" in draft.model_development
        assert "Random Forest" in draft.model_development
        assert "Histogram-based Gradient Boosting" in draft.model_development
        assert "5-fold" in draft.model_development
        assert "Random Forest" in draft.model_development  # primary model

    def test_hyperparameters_in_model_development(self, full_provenance):
        """#81: Hyperparameters should be described in human-readable prose."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        # Ridge should mention alpha and L2 regularization
        assert "alpha=1" in draft.model_development or "alpha=1.0" in draft.model_development
        assert "L2" in draft.model_development

        # Random Forest should mention number of trees and depth
        assert "100 trees" in draft.model_development
        assert "unrestricted depth" in draft.model_development or "max depth" in draft.model_development

        # HistGB should mention learning rate and iterations
        assert "learning rate" in draft.model_development.lower()
        assert "100 boosting iterations" in draft.model_development

    def test_model_evaluation_metrics(self, full_provenance):
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "RMSE" in draft.model_evaluation  # spelled out as "root mean squared error (RMSE)"
        assert "R²" in draft.model_evaluation    # spelled out as "coefficient of determination (R²)"
        assert "10.1" in draft.model_evaluation  # RF RMSE value
        assert "Random Forest" in draft.model_evaluation  # human-readable model name

    def test_confidence_intervals_in_evaluation(self):
        """#83: Confidence intervals should be included when available."""
        prov = WorkflowProvenance()
        prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
        prov.record_training(
            models_trained=["ridge", "rf"],
            primary_model="rf",
            metrics_by_model={
                "ridge": {
                    "RMSE": 12.34,
                    "RMSE_ci_lower": 11.95,
                    "RMSE_ci_upper": 12.73,
                    "R2": 0.72,
                    "R2_ci_lower": 0.68,
                    "R2_ci_upper": 0.76,
                },
                "rf": {
                    "RMSE": 10.12,
                    "R2": 0.85,
                    # No CIs for RF
                },
            },
        )

        engine = NarrativeEngine(prov)
        draft = engine.generate()

        # Ridge should show CIs
        assert "12.34" in draft.model_evaluation
        assert "95% CI" in draft.model_evaluation
        assert "11.95" in draft.model_evaluation or "12.0" in draft.model_evaluation  # formatted
        assert "12.73" in draft.model_evaluation or "12.7" in draft.model_evaluation

        # RF should show metrics without CIs
        assert "10.12" in draft.model_evaluation or "10.1" in draft.model_evaluation
        assert "0.85" in draft.model_evaluation

    def test_sensitivity_analysis(self, full_provenance):
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "seed stability" in draft.sensitivity_analysis.lower()
        assert "feature dropout" in draft.sensitivity_analysis.lower()

    def test_statistical_validation(self, full_provenance):
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "Shapiro-Wilk" in draft.statistical_validation
        assert "Breusch-Pagan" in draft.statistical_validation
        assert "1 of 2" in draft.statistical_validation  # 1 significant at p<0.05

    def test_data_observations_from_ledger(self, full_provenance, full_ledger):
        engine = NarrativeEngine(full_provenance, full_ledger)
        draft = engine.generate()

        # Resolved insights
        assert "skew" in draft.data_observations.lower() or "Yeo-Johnson" in draft.data_observations
        # Acknowledged limitations
        assert "noted and accepted" in draft.data_observations
        # Strengths
        assert "favorable" in draft.data_observations.lower()


class TestManuscriptDraft:

    def test_to_markdown(self, full_provenance, full_ledger):
        engine = NarrativeEngine(full_provenance, full_ledger)
        draft = engine.generate()
        md = draft.to_markdown()

        assert md.startswith("## Methods")
        assert "### Study Design" in md
        assert "### Data Preprocessing" in md
        assert "### Model Development" in md

    def test_to_latex(self, full_provenance, full_ledger):
        engine = NarrativeEngine(full_provenance, full_ledger)
        draft = engine.generate()
        latex = draft.to_latex()

        assert "\\section{Methods}" in latex
        assert "\\subsection{Study Design}" in latex
        assert "\\subsection{Data Preprocessing}" in latex

    def test_sections_dict_excludes_empty(self):
        draft = ManuscriptDraft()
        draft.study_design = "Some content"
        # All others empty

        assert "Study Design" in draft.sections
        assert len(draft.sections) == 1

    def test_completeness_warnings(self):
        """Empty provenance should generate completeness warnings."""
        prov = WorkflowProvenance()
        engine = NarrativeEngine(prov)
        draft = engine.generate()

        assert len(draft.warnings) > 0
        assert any("[PLACEHOLDER]" in w for w in draft.warnings)


class TestResultsAndDiscussion:
    """Tests for #82: Results and Discussion sections."""

    def test_results_section_generated(self, full_provenance):
        """Results section should report best model and metrics."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert draft.results != ""
        assert "Random Forest" in draft.results  # best model
        assert "R²" in draft.results or "R2" in draft.results
        assert "10.1" in draft.results  # RF RMSE
        assert "0.85" in draft.results  # RF R2

    def test_results_comparative_performance(self, full_provenance):
        """Results should compare all models."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        # Should mention comparison across models
        assert "Table 1" in draft.results or "candidate models" in draft.results
        # Should mention ranking
        assert "lowest" in draft.results.lower() or "highest" in draft.results.lower()

    def test_results_notes_simple_vs_complex(self):
        """Results should note when simple models are competitive."""
        prov = WorkflowProvenance()
        prov.record_upload("glucose", "regression", ["age", "bmi"], 500)
        prov.record_training(
            models_trained=["ridge", "rf"],
            primary_model="rf",
            metrics_by_model={
                "ridge": {"RMSE": 10.2, "R2": 0.84},
                "rf": {"RMSE": 10.1, "R2": 0.85},
            },
        )

        engine = NarrativeEngine(prov)
        draft = engine.generate()

        # Should note that linear model is competitive
        assert "linear" in draft.results.lower() or "comparable" in draft.results.lower()

    def test_discussion_section_generated(self, full_provenance, full_ledger):
        """Discussion section should have structured skeleton."""
        engine = NarrativeEngine(full_provenance, full_ledger)
        draft = engine.generate()

        assert draft.discussion != ""
        # Check for subsection headers
        assert "Principal Findings" in draft.discussion
        assert "Comparison with Prior Work" in draft.discussion
        assert "Strengths and Limitations" in draft.discussion
        assert "Practical Implications" in draft.discussion or "Clinical" in draft.discussion
        assert "Conclusions" in draft.discussion

    def test_discussion_principal_findings_auto_generated(self, full_provenance):
        """Principal Findings should auto-summarize results."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "Random Forest" in draft.discussion  # best model
        assert "85%" in draft.discussion or "0.85" in draft.discussion  # R2 percentage
        assert "compared across" in draft.discussion.lower()  # comparison note

    def test_discussion_placeholders(self, full_provenance):
        """Discussion should include placeholders for human completion."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()

        assert "[To be completed by the investigator" in draft.discussion
        # At least 2 placeholders (Prior Work, Implications)
        assert draft.discussion.count("[To be completed by the investigator") >= 2

    def test_discussion_strengths_and_limitations_from_ledger(self, full_provenance, full_ledger):
        """Strengths and limitations should auto-populate from InsightLedger."""
        engine = NarrativeEngine(full_provenance, full_ledger)
        draft = engine.generate()

        # Should include acknowledged limitations
        assert "Sample size" in draft.discussion or "adequate but not large" in draft.discussion
        # Should include strengths
        assert "Low overall missingness" in draft.discussion or "Favorable" in draft.discussion

    def test_results_and_discussion_in_markdown(self, full_provenance):
        """Results and Discussion should render as top-level sections in markdown."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()
        md = draft.to_markdown()

        assert "## Methods" in md
        assert "## Results" in md
        assert "## Discussion" in md
        # Results should come before Discussion
        assert md.index("## Results") < md.index("## Discussion")

    def test_results_and_discussion_in_latex(self, full_provenance):
        """Results and Discussion should render as top-level sections in LaTeX."""
        engine = NarrativeEngine(full_provenance)
        draft = engine.generate()
        latex = draft.to_latex()

        assert "\\section{Methods}" in latex
        assert "\\section{Results}" in latex
        assert "\\section{Discussion}" in latex


class TestEdgeCases:

    def test_minimal_provenance(self):
        """Only upload — should generate study design but warn about missing sections."""
        prov = WorkflowProvenance()
        prov.record_upload("target", "classification", ["a", "b"], 200)

        engine = NarrativeEngine(prov)
        draft = engine.generate()

        assert "classification" in draft.study_design
        assert "200" in draft.study_design
        assert draft.data_preprocessing == ""
        assert draft.model_development == ""
        assert len(draft.warnings) > 0

    def test_no_ledger(self):
        """NarrativeEngine works without InsightLedger."""
        prov = WorkflowProvenance()
        prov.record_upload("target", "regression", ["a"], 100)
        prov.record_training(["ridge"], "ridge", "RMSE")

        engine = NarrativeEngine(prov, ledger=None)
        draft = engine.generate()

        assert draft.data_observations == ""
        assert "Ridge Regression" in draft.model_development

    def test_single_model_no_cv(self):
        """Single model without CV should still produce valid narrative."""
        prov = WorkflowProvenance()
        prov.record_upload("target", "regression", ["a", "b"], 100)
        prov.record_preprocessing(
            {"ridge": {"numeric_scaling": "standard"}},
            "mean",
        )
        prov.record_training(
            ["ridge"], "ridge", "RMSE",
            use_cv=False,
            hyperparameters={"ridge": {"alpha": 0.5}},
        )

        engine = NarrativeEngine(prov)
        draft = engine.generate()

        assert "Ridge Regression" in draft.model_development
        assert "cross-validation" not in draft.model_development.lower()
        assert "All models shared" in draft.data_preprocessing
        assert "alpha=0.5" in draft.model_development

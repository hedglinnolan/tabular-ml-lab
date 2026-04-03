"""Tests for the pre-export manuscript consistency validator."""

from ml.manuscript_validator import validate_manuscript_bundle


def test_validate_manuscript_bundle_passes_for_consistent_bundle():
    manuscript_context = {
        'population_counts': {
            'analysis_total': 950,
            'train_n': 700,
            'val_n': 150,
            'test_n': 100,
        },
        'feature_counts': {'original': 26, 'selected': 18},
        'feature_names_for_manuscript': [f'f{i}' for i in range(18)],
        'included_models': ['ridge', 'rf'],
        'best_metric_name': 'RMSE',
    }
    report_text = """
## Abstract (Draft)
**Methods:** Of 1,000 observations, 950 remained for analysis after exclusions. The workflow began with 26 predictor variables and retained 18 predictors for final modeling.
"""
    methods_text = """
## Methods
### Study Design
A regression analysis was performed on a dataset of 950 observations.

### Predictor Variables
The workflow began with 26 predictor variables and retained 18 predictors for final modeling.

### Model Development
Ridge Regression and Random Forest were trained with validation RMSE as the selection criterion.

### Model Evaluation
Ridge Regression and Random Forest were evaluated on the held-out test set using RMSE and R².
"""
    latex_text = r"""
\section{Abstract}
Methods text only.
\section{Methods}
Ridge Regression and Random Forest were evaluated with clean LaTeX output.
"""

    report = validate_manuscript_bundle(
        manuscript_context=manuscript_context,
        methods_text=methods_text,
        report_text=report_text,
        latex_text=latex_text,
        task_type='regression',
    )

    assert report.passed
    assert all(check.status == "PASS" for check in report.checks)


def test_validate_manuscript_bundle_flags_cross_section_and_artifact_failures():
    manuscript_context = {
        'population_counts': {
            'analysis_total': 950,
            'train_n': 700,
            'val_n': 150,
            'test_n': 100,
        },
        'feature_counts': {'original': 18, 'selected': 18},
        'feature_names_for_manuscript': [f'f{i}' for i in range(18)],
        'included_models': ['ridge', 'histgb_reg'],
        'best_metric_name': 'Accuracy',
    }
    report_text = """
## Abstract (Draft)
**Methods:** A total of 1,000 observations were available for analysis. Feature selection reduced the feature set and retained 12 predictors for final modeling. favorable to analysis.
"""
    methods_text = """
## Methods
### Study Design
A regression analysis was performed on a dataset of 950 observations.

### Predictor Variables
The final modeling set contained 18 predictors.

### Model Development
Ridge Regression was trained with validation Accuracy as the selection criterion.

### Model Evaluation
Ridge Regression was evaluated on the held-out test set.
"""
    latex_text = r"""
\section{Methods}
[PLACEHOLDER] ## ** HISTGB_REG no action needed.. Table X
"""

    report = validate_manuscript_bundle(
        manuscript_context=manuscript_context,
        methods_text=methods_text,
        report_text=report_text,
        latex_text=latex_text,
        task_type='regression',
    )

    failed_names = {check.name for check in report.failed_checks}

    assert "Analysis population is consistent across abstract and study design" in failed_names
    assert "Final predictor count is consistent across abstract and methods" in failed_names
    assert "Model names match between development and evaluation sections" in failed_names
    assert "Selection metric language matches task type" in failed_names
    assert "Abstract feature-selection language matches actual reduction" in failed_names
    assert "LaTeX output is free of markdown and note artifacts" in failed_names
    assert "No internal model keys leak into export text" in failed_names
    assert "No coaching language patterns remain in export text" in failed_names
    assert "No obvious dangling punctuation or placeholder references remain" in failed_names


def test_validate_manuscript_bundle_flags_rendered_selection_metric_and_primary_conflict():
    manuscript_context = {
        'population_counts': {
            'analysis_total': 950,
            'train_n': 700,
            'val_n': 150,
            'test_n': 100,
        },
        'feature_counts': {'original': 26, 'selected': 18},
        'feature_names_for_manuscript': [f'f{i}' for i in range(18)],
        'included_models': ['ridge', 'nn'],
        'best_metric_name': 'RMSE',
        'best_model_by_metric': 'nn',
        'manuscript_primary_model': None,
    }
    report_text = """
## Abstract (Draft)
**Methods:** Of 1,000 observations, 950 remained for analysis after exclusions. The workflow began with 26 predictor variables and retained 18 predictors for final modeling.
"""
    methods_text = """
## Methods
### Study Design
A regression analysis was performed on a dataset of 950 observations.

### Predictor Variables
The workflow began with 26 predictor variables and retained 18 predictors for final modeling.

### Model Development
Neural Network (MLP) was selected as the primary model, based on validation Accuracy.

### Model Evaluation
Ridge Regression and Neural Network (MLP) were evaluated on the held-out test set using RMSE and R².
"""
    latex_text = r"""
\section{Results}
\subsection{Model Performance}
The best model by RMSE was \textbf{Neural Network (MLP)}. No manuscript-primary model was explicitly selected in the workflow.
"""

    report = validate_manuscript_bundle(
        manuscript_context=manuscript_context,
        methods_text=methods_text,
        report_text=report_text,
        latex_text=latex_text,
        task_type='regression',
    )

    failed_names = {check.name for check in report.failed_checks}

    assert "Selection metric language matches task type" in failed_names
    assert "Primary model statements are internally consistent" in failed_names

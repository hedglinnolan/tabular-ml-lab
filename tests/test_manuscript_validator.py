"""Tests for the pre-export manuscript consistency validator."""

import pandas as pd

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
    table1_df = pd.DataFrame(
        {"Overall (N=950)": ["1.0", "2.0"]},
        index=["f0, median [IQR]", "f1, median [IQR]"],
    )
    for idx in range(2, 18):
        table1_df.loc[f"f{idx}, median [IQR]"] = "1.0"

    report = validate_manuscript_bundle(
        manuscript_context=manuscript_context,
        methods_text=methods_text,
        report_text=report_text,
        latex_text=latex_text,
        task_type='regression',
        table1_df=table1_df,
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


def test_validate_manuscript_bundle_allows_best_by_metric_without_primary_model_conflict():
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
    methods_text = """
## Methods
### Study Design
A regression analysis was performed on a dataset of 950 observations.

### Predictor Variables
The workflow began with 26 predictor variables and retained 18 predictors for final modeling.

### Model Development
Neural Network (MLP) achieved the best held-out performance on validation RMSE, but no manuscript-primary model was explicitly selected.

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
        report_text="## Abstract (Draft)\n**Methods:** Of 1,000 observations, 950 remained for analysis after exclusions.",
        latex_text=latex_text,
        task_type='regression',
    )

    failed_names = {check.name for check in report.failed_checks}
    assert "Primary model statements are internally consistent" not in failed_names


def test_validate_manuscript_bundle_allows_investigator_placeholders_but_rejects_raw_markdown_artifacts():
    manuscript_context = {
        'population_counts': {
            'analysis_total': 950,
            'train_n': 700,
            'val_n': 150,
            'test_n': 100,
        },
        'feature_counts': {'original': 26, 'selected': 18},
        'feature_names_for_manuscript': [f'f{i}' for i in range(18)],
        'included_models': ['ridge'],
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
Ridge Regression achieved the best held-out performance on validation RMSE, but no manuscript-primary model was explicitly selected.

### Model Evaluation
Ridge Regression was evaluated on the held-out test set using RMSE and R².
"""
    latex_text = r"""
\section{Introduction}
[PLACEHOLDER: Add clinical background.]
\section{Discussion}
[PLACEHOLDER: Compare the performance to prior work. Add an appropriate benchmark for typical regression performance in this domain.]
\section{Methods}
Clean LaTeX output only.
"""

    report = validate_manuscript_bundle(
        manuscript_context=manuscript_context,
        methods_text=methods_text,
        report_text=report_text,
        latex_text=latex_text,
        task_type='regression',
    )

    failed_names = {check.name for check in report.failed_checks}

    assert "LaTeX output is free of markdown and note artifacts" not in failed_names
    assert "No obvious dangling punctuation or placeholder references remain" not in failed_names


def test_validate_manuscript_bundle_flags_table1_population_and_feature_coverage():
    manuscript_context = {
        'population_counts': {
            'analysis_total': 950,
            'train_n': 700,
            'val_n': 150,
            'test_n': 100,
        },
        'feature_counts': {'original': 26, 'selected': 3},
        'feature_names_for_manuscript': ['age', 'bmi', 'hdl'],
        'included_models': ['ridge'],
        'best_metric_name': 'RMSE',
    }
    report_text = """
## Abstract (Draft)
**Methods:** Of 1,000 observations, 950 remained for analysis after exclusions. The workflow began with 26 predictor variables and retained 3 predictors for final modeling.
"""
    methods_text = """
## Methods
### Study Design
A regression analysis was performed on a dataset of 950 observations.

### Predictor Variables
The workflow began with 26 predictor variables and retained 3 predictors for final modeling.

### Model Development
Ridge Regression achieved the best held-out performance on validation RMSE, but no manuscript-primary model was explicitly selected.

### Model Evaluation
Ridge Regression was evaluated on the held-out test set using RMSE and R².
"""
    latex_text = r"\section{Methods}Clean LaTeX output only."
    table1_df = pd.DataFrame(
        {"Overall (N=1000)": ["47 [31, 63]", "27.7 [24.1, 32.1]"]},
        index=["age, median [IQR]", "bmi, median [IQR]"],
    )

    report = validate_manuscript_bundle(
        manuscript_context=manuscript_context,
        methods_text=methods_text,
        report_text=report_text,
        latex_text=latex_text,
        task_type='regression',
        table1_df=table1_df,
    )

    failed_names = {check.name for check in report.failed_checks}

    assert "Table 1 population matches the analysis cohort" in failed_names
    assert "Table 1 includes all finalized predictors" in failed_names

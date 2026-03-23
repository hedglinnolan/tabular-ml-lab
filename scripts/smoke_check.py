#!/usr/bin/env python
"""
Smoke Check Script for Glucose MLP Interactive

Validates key functions and imports without running Streamlit.
Run with: python scripts/smoke_check.py
"""
import sys
import os
import traceback
from typing import List, Tuple
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Track test results
results: List[Tuple[str, bool, str]] = []


def test(name: str, requires: List[str] = None):
    """Decorator to wrap tests and track results."""
    def decorator(func):
        def wrapper():
            # Check for required packages
            if requires:
                for pkg in requires:
                    try:
                        __import__(pkg)
                    except ImportError:
                        results.append((name, None, f"SKIPPED (requires {pkg})"))
                        print(f"SKIP {name}: SKIPPED (requires {pkg})")
                        return
            
            try:
                func()
                results.append((name, True, "PASS"))
                print(f"PASS {name}")
            except ImportError as e:
                # Skip tests that fail due to missing packages
                pkg_name = str(e).replace("No module named ", "").strip("'")
                if pkg_name in ['streamlit', 'torch', 'shap']:
                    results.append((name, None, f"SKIPPED (requires {pkg_name})"))
                    print(f"SKIP {name}: SKIPPED (requires {pkg_name})")
                else:
                    results.append((name, False, str(e)))
                    print(f"FAIL {name}: {e}")
                    if "--verbose" in sys.argv:
                        traceback.print_exc()
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"FAIL {name}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# Import Tests - Ensure modules can be loaded without errors
# ============================================================

@test("Import: utils.session_state")
def test_import_session_state():
    from utils.session_state import (
        init_session_state, get_data, set_data, DataConfig,
        TaskTypeDetection, CohortStructureDetection
    )


@test("Import: ml.model_registry")
def test_import_model_registry():
    from ml.model_registry import get_registry, ModelSpec, ModelCapabilities


@test("Import: ml.model_coach")
def test_import_model_coach():
    from ml.model_coach import (
        coach_recommendations, CoachRecommendation, GROUP_DISPLAY_NAMES,
        compute_model_recommendations, ModelRecommendation, RecommendationBucket,
        TrainingTimeTier, CoachOutput
    )


@test("Import: ml.dataset_profile")
def test_import_dataset_profile():
    from ml.dataset_profile import (
        compute_dataset_profile, DatasetProfile, FeatureProfile, TargetProfile,
        DataSufficiencyLevel, WarningLevel, DataWarning, get_profile_summary_text
    )


@test("Import: ml.eda_recommender")
def test_import_eda_recommender():
    from ml.eda_recommender import compute_dataset_signals, recommend_eda, DatasetSignals


@test("Import: ml.triage")
def test_import_triage():
    from ml.triage import detect_task_type, detect_cohort_structure


@test("Import: ml.outliers")
def test_import_outliers():
    from ml.outliers import detect_outliers, outlier_rate


@test("Import: ml.physiology_reference")
def test_import_physiology_reference():
    from ml.physiology_reference import (
        load_reference_bundle, match_variable_key, get_reference_interval,
        load_nhanes_reference
    )


@test("Import: ml.preprocess_operators")
def test_import_preprocess_operators():
    from ml.preprocess_operators import UnitHarmonizer, PlausibilityGate, OutlierCapping


@test("Import: ml.eval")
def test_import_eval():
    from ml.eval import (
        calculate_regression_metrics, calculate_classification_metrics,
        perform_cross_validation, analyze_residuals, compare_importance_ranks
    )


@test("Import: models.nn_whuber")
def test_import_nn_wrapper():
    from models.nn_whuber import NNWeightedHuberWrapper, SimpleMLP


@test("Import: models.glm")
def test_import_glm():
    from models.glm import GLMWrapper


@test("Import: models.rf")
def test_import_rf():
    from models.rf import RFWrapper


@test("Import: data_processor")
def test_import_data_processor():
    from data_processor import load_and_preview_csv, get_numeric_columns


@test("Import: visualizations.plot_bland_altman")
def test_import_plot_bland_altman():
    from visualizations import plot_bland_altman


# ============================================================
# Functional Tests - Test key functionality
# ============================================================

@test("Registry: get_registry returns dict with models")
def test_registry_structure():
    from ml.model_registry import get_registry
    registry = get_registry()
    assert isinstance(registry, dict), "Registry should be a dict"
    assert len(registry) > 0, "Registry should not be empty"
    # Check required models exist
    required_models = ['nn', 'rf', 'glm', 'huber', 'ridge', 'logreg']
    for model in required_models:
        assert model in registry, f"Model '{model}' should be in registry"


@test("Registry: NN model has architecture params")
def test_nn_architecture_params():
    from ml.model_registry import get_registry
    registry = get_registry()
    nn_spec = registry.get('nn')
    assert nn_spec is not None, "NN spec should exist"
    schema = nn_spec.hyperparam_schema
    assert 'num_layers' in schema, "NN should have num_layers param"
    assert 'layer_width' in schema, "NN should have layer_width param"
    assert 'architecture_pattern' in schema, "NN should have architecture_pattern param"
    assert 'activation' in schema, "NN should have activation param"


@test("Coach: recommendations are merged by group")
def test_coach_merging():
    from ml.model_coach import coach_recommendations, _merge_recommendations_by_group, CoachRecommendation
    
    # Create test recommendations with same group
    recs = [
        CoachRecommendation(
            group='Linear',
            recommended_models=['glm'],
            why=['Reason 1'],
            when_not_to_use=['Caveat 1'],
            suggested_preprocessing=['Preprocess 1'],
            priority=1
        ),
        CoachRecommendation(
            group='Linear',
            recommended_models=['ridge'],
            why=['Reason 2'],
            when_not_to_use=['Caveat 2'],
            suggested_preprocessing=['Preprocess 2'],
            priority=2
        ),
    ]
    
    merged = _merge_recommendations_by_group(recs)
    assert len(merged) == 1, "Should merge into single Linear recommendation"
    assert 'glm' in merged[0].recommended_models, "Should include glm"
    assert 'ridge' in merged[0].recommended_models, "Should include ridge"
    assert merged[0].priority == 1, "Should use lowest priority"


@test("Coach: display_name property works")
def test_coach_display_name():
    from ml.model_coach import CoachRecommendation, GROUP_DISPLAY_NAMES
    
    rec = CoachRecommendation(
        group='Linear',
        recommended_models=['glm'],
        why=['Test'],
        when_not_to_use=[],
        suggested_preprocessing=[],
        priority=1
    )
    
    assert rec.display_name == 'Linear Models', f"Expected 'Linear Models', got '{rec.display_name}'"


@test("NN: SimpleMLP accepts activation parameter")
def test_nn_activation():
    import torch
    from models.nn_whuber import SimpleMLP
    
    # Test with different activations
    for activation in ['relu', 'tanh', 'leaky_relu', 'elu']:
        model = SimpleMLP(input_dim=10, hidden=[32, 32], dropout=0.1, output_dim=1, activation=activation)
        assert model.activation_name == activation, f"Activation should be {activation}"
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 1), f"Output shape should be (5, 1)"


@test("NN: NNWeightedHuberWrapper accepts activation parameter")
def test_nn_wrapper_activation():
    from models.nn_whuber import NNWeightedHuberWrapper
    
    wrapper = NNWeightedHuberWrapper(
        hidden_layers=[32, 16],
        dropout=0.1,
        task_type='regression',
        activation='tanh'
    )
    
    assert wrapper.activation == 'tanh', "Activation should be tanh"
    assert wrapper.hidden_layers == [32, 16], "Hidden layers should be [32, 16]"


@test("Data: DataConfig can be created")
def test_data_config():
    from utils.session_state import DataConfig
    
    config = DataConfig(
        target_col='glucose',
        feature_cols=['feature1', 'feature2'],
        datetime_col=None,
        task_type='regression'
    )
    
    assert config.target_col == 'glucose'
    assert len(config.feature_cols) == 2


@test("Detection: TaskTypeDetection final property works")
def test_task_type_detection():
    from utils.session_state import TaskTypeDetection
    
    # Without override
    det = TaskTypeDetection(detected='regression', confidence='high', reasons=['Test'])
    assert det.final == 'regression', "Final should be detected value"
    
    # With override
    det.override_enabled = True
    det.override_value = 'classification'
    assert det.final == 'classification', "Final should be override value when enabled"


@test("DatasetProfile: compute_dataset_profile works")
def test_dataset_profile():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile, DataSufficiencyLevel
    
    # Create simple test data
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100)
    })
    
    profile = compute_dataset_profile(
        df,
        target_col='target',
        feature_cols=['feature1', 'feature2', 'category'],
        task_type='regression',
        outlier_method='iqr'
    )
    
    assert profile.n_rows == 100, "Should have 100 rows"
    assert profile.n_features == 3, "Should have 3 features"
    assert profile.n_numeric == 2, "Should have 2 numeric features"
    assert profile.n_categorical == 1, "Should have 1 categorical feature"
    assert profile.target_profile is not None, "Should have target profile"
    assert profile.target_profile.task_type == 'regression', "Should be regression"
    assert profile.data_sufficiency in DataSufficiencyLevel, "Should have valid sufficiency level"


@test("DatasetProfile: classification target detection")
def test_profile_classification():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(200),
        'target': np.random.choice([0, 1], 200, p=[0.8, 0.2])  # Imbalanced
    })
    
    profile = compute_dataset_profile(
        df,
        target_col='target',
        feature_cols=['feature1'],
        task_type='classification',
        outlier_method='iqr'
    )
    
    assert profile.target_profile.task_type == 'classification'
    assert profile.target_profile.n_classes == 2
    assert profile.target_profile.is_imbalanced, "Should detect imbalance"


@test("DatasetProfile: warnings generation")
def test_profile_warnings():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile, WarningLevel
    
    # Create data with issues
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': [1.0] * 20 + [np.nan] * 5,  # Small sample with missing
        'target': list(range(25))
    })
    
    profile = compute_dataset_profile(
        df,
        target_col='target',
        feature_cols=['feature1'],
        task_type='regression',
        outlier_method='iqr'
    )
    
    # Should have sample size warning
    assert len(profile.warnings) > 0, "Should have warnings for small sample"
    warning_categories = [w.category for w in profile.warnings]
    assert 'sample_size' in warning_categories, "Should warn about small sample"


@test("Coach: compute_model_recommendations works with profile")
def test_coach_with_profile():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile
    from ml.model_coach import compute_model_recommendations, RecommendationBucket
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500),
        'target': np.random.randn(500)
    })
    
    profile = compute_dataset_profile(
        df,
        target_col='target',
        feature_cols=['feature1', 'feature2'],
        task_type='regression',
        outlier_method='iqr'
    )
    
    coach_output = compute_model_recommendations(profile)
    
    assert coach_output is not None, "Should return coach output"
    assert len(coach_output.recommended_models) > 0 or len(coach_output.worth_trying_models) > 0, \
        "Should have some recommendations"
    assert coach_output.dataset_summary, "Should have dataset summary"
    assert coach_output.preprocessing_recommendations is not None, "Should have preprocessing recs"
    assert coach_output.baseline_eda, "Should have baseline EDA recommendations"
    assert coach_output.advanced_eda_by_family, "Should have advanced EDA by family"


@test("Coach: ModelRecommendation has required fields")
def test_model_recommendation_fields():
    from ml.model_coach import ModelRecommendation, RecommendationBucket, TrainingTimeTier
    
    rec = ModelRecommendation(
        model_key='ridge',
        model_name='Ridge Regression',
        group='Linear',
        bucket=RecommendationBucket.RECOMMENDED,
        rationale='Good for this dataset',
        dataset_fit_summary='Good fit',
        strengths=['Interpretable'],
        weaknesses=[],
        risks=[],
        training_time=TrainingTimeTier.FAST,
        interpretability='high',
        requires_scaling=True,
        requires_encoding=True,
        handles_missing=False,
        plain_language_summary='Ridge is a regularized linear model.',
        when_to_use='When features are correlated',
        when_to_avoid='When relationships are nonlinear',
        priority=10
    )
    
    assert rec.model_key == 'ridge'
    assert rec.bucket == RecommendationBucket.RECOMMENDED
    assert rec.training_time == TrainingTimeTier.FAST


@test("Outliers: detect_outliers IQR")
def test_detect_outliers_iqr():
    import pandas as pd
    from ml.outliers import detect_outliers

    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is outlier
    mask, info = detect_outliers(s, method="iqr")
    assert isinstance(mask, pd.Series), "Should return Series mask"
    assert mask.dtype == bool, "Mask should be boolean"
    assert info["method"] == "iqr", "Info should record method"
    assert info.get("lower") is not None or info.get("upper") is not None, "Should have bounds"


@test("Physiology: load_reference_bundle structure")
def test_load_reference_bundle():
    from ml.physiology_reference import load_reference_bundle

    bundle = load_reference_bundle()
    assert "nhanes" in bundle, "Bundle should have nhanes"
    assert "clinical" in bundle, "Bundle should have clinical"
    nhanes = bundle["nhanes"]
    assert "variables" in nhanes, "NHANES should have variables"
    assert len(nhanes["variables"]) > 0, "NHANES variables should not be empty"


@test("Preprocess: UnitHarmonizer transform")
def test_unit_harmonizer():
    import numpy as np
    from ml.preprocess_operators import UnitHarmonizer

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    h = UnitHarmonizer(conversion_factors=[2.0, 0.5])
    out = h.fit_transform(X)
    assert out.shape == X.shape
    assert out[0, 0] == 2.0 and out[0, 1] == 1.0, "Should scale by factors"


@test("Preprocess: OutlierCapping fit_transform")
def test_outlier_capping():
    import numpy as np
    from ml.preprocess_operators import OutlierCapping

    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    c = OutlierCapping(method="percentile", params={"lower_q": 0.1, "upper_q": 0.9})
    out = c.fit_transform(X)
    assert out.shape == X.shape
    assert out.min() >= np.nanmin(c.lower_bounds_) and out.max() <= np.nanmax(c.upper_bounds_), \
        "Output should be clipped to bounds"


@test("Eval: compare_importance_ranks")
def test_compare_importance_ranks():
    import numpy as np
    from ml.eval import compare_importance_ranks

    names = ["ridge", "lasso"]
    fnames = ["f1", "f2", "f3"]
    perm = {
        "ridge": {"importances_mean": np.array([0.1, 0.5, 0.3])},
        "lasso": {"importances_mean": np.array([0.2, 0.4, 0.4])},
    }
    fn_by_model = {"ridge": fnames, "lasso": fnames}
    out = compare_importance_ranks(names, perm, fn_by_model, top_k=2)
    assert isinstance(out, dict), "Should return dict"
    assert ("ridge", "lasso") in out, "Should have ridge vs lasso"
    r = out[("ridge", "lasso")]
    assert "spearman" in r and "top_k_overlap" in r and "n_features" in r, "Result should have expected keys"


@test("Viz: plot_bland_altman returns figure")
def test_plot_bland_altman():
    import numpy as np
    from visualizations import plot_bland_altman

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    fig = plot_bland_altman(a, b, title="Test", label_a="A", label_b="B")
    assert fig is not None, "Should return figure"
    assert hasattr(fig, "add_trace") and hasattr(fig, "update_layout"), "Should be Plotly Figure"


@test("Eval: analyze_residuals_extended, analyze_pred_vs_actual, analyze_bland_altman")
def test_eval_extended_stats():
    import numpy as np
    from ml.eval import (
        analyze_residuals_extended,
        analyze_pred_vs_actual,
        analyze_bland_altman,
        analyze_confusion_matrix,
    )

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([1.1, 2.1, 2.9, 4.2, 4.7])
    r = analyze_residuals_extended(y, p)
    assert "skew" in r and "iqr" in r, "Extended residuals should have skew, iqr"
    pva = analyze_pred_vs_actual(y, p)
    assert "correlation" in pva and "bias_by_quintile" in pva
    ba = analyze_bland_altman(y, p)
    assert "mean_diff" in ba and "pct_outside_loa" in ba
    yt = np.array([0, 0, 1, 1, 1])
    yp = np.array([0, 1, 1, 1, 0])
    cm = analyze_confusion_matrix(yt, yp)
    assert "per_class" in cm and "top_confusions" in cm


@test("Plot narrative: narrative_residuals, narrative_pred_vs_actual, narrative_bland_altman")
def test_plot_narrative():
    from ml.plot_narrative import (
        narrative_residuals,
        narrative_pred_vs_actual,
        narrative_bland_altman,
        narrative_permutation_importance,
    )
    from ml.eval import analyze_residuals_extended, analyze_pred_vs_actual, analyze_bland_altman

    import numpy as np
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([1.1, 2.1, 2.9, 4.2, 4.7])
    r = analyze_residuals_extended(y, p)
    n1 = narrative_residuals(r, model_name="ridge")
    assert isinstance(n1, str) and len(n1) > 0
    pva = analyze_pred_vs_actual(y, p)
    n2 = narrative_pred_vs_actual(pva, model_name="ridge")
    assert isinstance(n2, str)
    ba = analyze_bland_altman(y, p)
    n3 = narrative_bland_altman(ba, label_a="A", label_b="B")
    assert isinstance(n3, str) and len(n3) > 0
    perm = {"importances_mean": np.array([0.1, 0.5, 0.3]), "feature_names": ["f1", "f2", "f3"]}
    n4 = narrative_permutation_importance(perm, model_name="ridge")
    assert isinstance(n4, str) and "f2" in n4

    from ml.plot_narrative import narrative_robustness, narrative_partial_dependence
    rob = {("m1", "m2"): {"spearman": 0.9, "top_k_overlap": 4}}
    n5 = narrative_robustness(rob)
    assert isinstance(n5, str) and len(n5) > 0
    pd_data = {"feat1": {"values": [0, 1, 2], "average": [0.1, 0.2, 0.3]}}
    n6 = narrative_partial_dependence(pd_data, model_name="ridge")
    assert isinstance(n6, str) and len(n6) > 0


@test("Import: utils.llm_ui")
def test_import_llm_ui():
    from utils.llm_ui import render_interpretation_with_llm_button



@test("Upload flow: load_and_preview_csv + reconcile_state_with_df")
def test_upload_flow():
    import io

    class MockSession(dict):
        def __getattr__(self, k):
            return self.get(k)
        def __setattr__(self, k, v):
            self[k] = v

    from data_processor import load_and_preview_csv, get_numeric_columns, get_selectable_columns
    from utils.state_reconcile import reconcile_state_with_df
    from utils.reconcile import reconcile_target_features
    from utils.session_state import DataConfig

    csv = "a,b,c\n1,2,3\n4,5,6\n7,8,9"
    buf = io.BytesIO(csv.encode("utf-8"))
    df = load_and_preview_csv(buf)
    assert df is not None and len(df) == 3 and list(df.columns) == ["a", "b", "c"]

    numeric = get_numeric_columns(df)
    assert "a" in numeric and "b" in numeric and "c" in numeric

    session = MockSession()
    session["data_config"] = DataConfig(target_col="c", feature_cols=["a", "b"])
    reconcile_state_with_df(df, session)
    cfg = session.get("data_config")
    assert cfg is not None and cfg.target_col == "c" and set(cfg.feature_cols) == {"a", "b"}

    num_sel, cat_sel = get_selectable_columns(df)
    t, f = reconcile_target_features(df, "c", ["a", "b"], (num_sel, cat_sel))
    assert t == "c" and set(f) == {"a", "b"}


@test("Stats tests: correlation, normality, paired")
def test_stats_tests():
    import numpy as np
    from ml.stats_tests import (
        correlation_test,
        normality_check,
        paired_location_test,
        two_sample_location_test,
        categorical_association_test,
    )
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    r, p, name = correlation_test(x, y, method="pearson")
    assert abs(r) > 0.9 and 0 <= p <= 1 and "Pearson" in name
    stat, pn, nn = normality_check(x)
    assert nn == "Shapiro–Wilk"
    diff = np.array([0.1, -0.1, 0.2, 0.0, -0.1])
    st, pp, nm = paired_location_test(diff, parametric=True)
    assert nm == "paired t-test"
    a, b = np.array([1, 2, 3]), np.array([2, 3, 4])
    st2, p2, n2 = two_sample_location_test(a, b, parametric=True)
    assert "t-test" in n2
    cont = np.array([[10, 20], [15, 25]])
    st3, p3, n3 = categorical_association_test(cont, use_fisher=False)
    assert "chi" in n3.lower()


@test("EDA: narrative helpers + data_sufficiency_check")
def test_eda_narratives_and_actions():
    import pandas as pd
    from ml.plot_narrative import (
        narrative_eda_linearity,
        narrative_eda_influence,
        narrative_eda_normality,
        narrative_eda_sufficiency,
        narrative_eda_multicollinearity,
    )
    from ml.eda_actions import data_sufficiency_check, linearity_scatter
    from ml.eda_recommender import DatasetSignals

    assert isinstance(narrative_eda_linearity({}), str)
    assert isinstance(narrative_eda_influence({}), str)
    assert isinstance(narrative_eda_normality({}), str)
    assert isinstance(narrative_eda_sufficiency({"ratio": 25, "n_rows": 100, "n_features": 4}), str)
    assert isinstance(narrative_eda_multicollinearity({"vif": [("x", 5.0)]}), str)

    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    signals = DatasetSignals(
        n_rows=3, n_cols=2, numeric_cols=["x", "y"],
        task_type_final="regression", target_name="y",
    )
    session = {}
    out = data_sufficiency_check(df, "y", ["x"], signals, session)
    assert "findings" in out and "figures" in out and "stats" in out
    assert out["stats"].get("n_rows") == 3 and out["stats"].get("n_features") == 1

    out2 = linearity_scatter(df, "y", ["x"], signals, session)
    assert "findings" in out2 and "figures" in out2 and "stats" in out2


@test("Data: prepare_data with categorical target")
def test_prepare_data_categorical():
    import pandas as pd
    from data_processor import prepare_data

    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": ["A", "B", pd.NA, "A", "B"],
    })
    out = prepare_data(df, target_col="target", feature_cols=["x1", "x2"], test_size=0.2, val_size=0.2, seed=42)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = out
    assert len(y_train) > 0, "Should have training samples after dropping NaN target"
    assert len(feat_names) == 2, "Should have 2 feature names"
    assert y_train.dtype == object or str(y_train.dtype) == "object", "Categorical target should remain object/string"


@test("Data: load_csv encoding fallback")
def test_load_csv_encoding_fallback():
    import io
    from data_processor import load_csv

    csv_bytes = b"a,b\n1,2\n\xe9,3"
    buf = io.BytesIO(csv_bytes)
    df = load_csv(buf)
    assert df is not None, "Should load with fallback encoding"
    assert len(df) >= 2, "Should have at least 2 rows"
    assert "a" in df.columns and "b" in df.columns, "Should have columns a, b"


@test("Utils: make_unique_columns")
def test_make_unique_columns():
    from utils.column_utils import make_unique_columns

    result = make_unique_columns(["a", "a", "b", "a"])
    assert result == ["a", "a_1", "b", "a_2"], f"Expected ['a','a_1','b','a_2'], got {result}"


@test("DatasetProfile: empty DataFrame raises ValueError")
def test_dataset_profile_empty_guard():
    import pandas as pd
    from ml.dataset_profile import compute_dataset_profile

    try:
        compute_dataset_profile(
            pd.DataFrame(),
            target_col="x",
            feature_cols=[],
            task_type="regression",
            outlier_method="iqr",
        )
        assert False, "Should have raised ValueError for empty DataFrame"
    except ValueError as e:
        assert "empty" in str(e).lower(), f"Expected empty-related message, got: {e}"


# ============================================================
# Main execution
# ============================================================

def run_all_tests():
    """Run all registered tests."""
    print("\n" + "=" * 60)
    print("Glucose MLP Interactive - Smoke Check")
    print("=" * 60 + "\n")
    
    # Run import tests
    print("Import Tests:")
    print("-" * 40)
    test_import_session_state()
    test_import_model_registry()
    test_import_model_coach()
    test_import_dataset_profile()
    test_import_eda_recommender()
    test_import_triage()
    test_import_outliers()
    test_import_physiology_reference()
    test_import_preprocess_operators()
    test_import_eval()
    test_import_nn_wrapper()
    test_import_glm()
    test_import_rf()
    test_import_data_processor()
    test_import_plot_bland_altman()
    test_import_llm_ui()
    
    print("\nFunctional Tests:")
    print("-" * 40)
    test_registry_structure()
    test_nn_architecture_params()
    test_coach_merging()
    test_coach_display_name()
    test_nn_activation()
    test_nn_wrapper_activation()
    test_data_config()
    test_task_type_detection()
    test_dataset_profile()
    test_profile_classification()
    test_profile_warnings()
    test_coach_with_profile()
    test_model_recommendation_fields()
    test_detect_outliers_iqr()
    test_load_reference_bundle()
    test_unit_harmonizer()
    test_outlier_capping()
    test_compare_importance_ranks()
    test_stats_tests()
    test_plot_bland_altman()
    test_eval_extended_stats()
    test_plot_narrative()
    test_upload_flow()
    test_eda_narratives_and_actions()
    test_prepare_data_categorical()
    test_load_csv_encoding_fallback()
    test_make_unique_columns()
    test_dataset_profile_empty_guard()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, success, _ in results if success is True)
    failed = sum(1 for _, success, _ in results if success is False)
    skipped = sum(1 for _, success, _ in results if success is None)
    total = len(results)
    
    if failed == 0:
        print(f"All tests passed! ({passed} passed, {skipped} skipped)")
    else:
        print(f"FAIL {failed}/{total} tests failed ({passed} passed, {skipped} skipped):")
        for name, success, msg in results:
            if success is False:
                print(f"   - {name}: {msg}")
    
    if skipped > 0:
        print(f"\nSkipped tests (missing optional dependencies):")
        for name, success, msg in results:
            if success is None:
                print(f"   - {name}: {msg}")
    
    print("=" * 60 + "\n")
    
    # Return success if no tests failed (skipped is OK)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

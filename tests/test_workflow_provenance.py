"""Tests for WorkflowProvenance — creation, record_* methods, readers, and round-trip."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.workflow_provenance import (
    WorkflowProvenance,
    UploadProvenance,
    EDAProvenance,
    FeatureEngineeringProvenance,
    FeatureSelectionProvenance,
    PreprocessingProvenance,
    ModelPreprocessingConfig,
    TrainingProvenance,
    ExplainabilityProvenance,
    SensitivityProvenance,
    StatisticalValidationProvenance,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_prov() -> WorkflowProvenance:
    return WorkflowProvenance()


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------

def test_creation_all_sections_none():
    prov = make_prov()
    assert prov.upload is None
    assert prov.eda is None
    assert prov.feature_engineering is None
    assert prov.feature_selection is None
    assert prov.split is None
    assert prov.preprocessing is None
    assert prov.training is None
    assert prov.explainability is None
    assert prov.sensitivity is None
    assert prov.statistical_validation is None
    assert prov.schema_version == 1


# ---------------------------------------------------------------------------
# 2. record_* methods
# ---------------------------------------------------------------------------

def test_record_upload():
    prov = make_prov()
    prov.record_upload(
        target_col='glucose',
        task_type='regression',
        feature_cols=['age', 'bmi', 'bp'],
        n_samples=500,
        data_source='csv',
    )
    assert prov.upload is not None
    assert prov.upload.target_col == 'glucose'
    assert prov.upload.task_type == 'regression'
    assert prov.upload.feature_cols == ['age', 'bmi', 'bp']
    assert prov.upload.n_samples == 500
    assert prov.upload.n_features == 3
    assert prov.upload.data_source == 'csv'
    assert prov.upload.timestamp != ''


def test_record_cleaning():
    prov = make_prov()
    prov.record_upload(
        target_col='glucose', task_type='regression',
        feature_cols=['a', 'b'], n_samples=100,
    )
    prov.record_cleaning(
        action='Drop rows with missing values',
        rows_before=100,
        rows_after=90,
        details={'cols': ['a']},
    )
    assert len(prov.upload.cleaning_actions) == 1
    entry = prov.upload.cleaning_actions[0]
    assert entry['action'] == 'Drop rows with missing values'
    assert entry['rows_before'] == 100
    assert entry['rows_after'] == 90
    # record_cleaning updates n_samples
    assert prov.upload.n_samples == 90


def test_record_cleaning_without_upload_is_noop():
    prov = make_prov()
    # Should not raise; no-op when upload is None
    prov.record_cleaning(action='drop', rows_before=100, rows_after=90)
    assert prov.upload is None


def test_record_eda_analysis():
    prov = make_prov()
    prov.record_eda_analysis('Distribution Plot')
    assert prov.eda is not None
    assert 'Distribution Plot' in prov.eda.analyses_run
    # Duplicate should not be added
    prov.record_eda_analysis('Distribution Plot')
    assert prov.eda.analyses_run.count('Distribution Plot') == 1
    prov.record_eda_analysis('Correlation Matrix')
    assert len(prov.eda.analyses_run) == 2


def test_record_table1():
    prov = make_prov()
    assert prov.eda is None
    prov.record_table1()
    assert prov.eda is not None
    assert prov.eda.table1_generated is True


def test_record_feature_engineering():
    prov = make_prov()
    prov.record_feature_engineering(
        transforms=['log_transform', 'polynomial'],
        n_created=5,
        n_before=10,
        n_after=15,
    )
    assert prov.feature_engineering is not None
    assert prov.feature_engineering.transforms_applied == ['log_transform', 'polynomial']
    assert prov.feature_engineering.n_features_created == 5
    assert prov.feature_engineering.n_features_before == 10
    assert prov.feature_engineering.n_features_after == 15


def test_record_feature_selection():
    prov = make_prov()
    prov.record_feature_selection(
        method='consensus',
        n_before=20,
        n_after=10,
        features_kept=['a', 'b', 'c'],
        consensus_methods=['lasso', 'random_forest'],
    )
    assert prov.feature_selection is not None
    assert prov.feature_selection.method == 'consensus'
    assert prov.feature_selection.n_features_before == 20
    assert prov.feature_selection.n_features_after == 10
    assert prov.feature_selection.features_kept == ['a', 'b', 'c']
    assert prov.feature_selection.consensus_methods == ['lasso', 'random_forest']


def test_record_preprocessing():
    prov = make_prov()
    configs = {
        'LogisticRegression': {
            'numeric_scaling': 'standard',
            'categorical_encoding': 'onehot',
            'numeric_outlier_treatment': 'none',
        },
        'RandomForest': {
            'numeric_scaling': 'none',
            'categorical_encoding': 'ordinal',
            'numeric_outlier_treatment': 'clip',
        },
    }
    prov.record_preprocessing(configs_by_model=configs, imputation_method='median')
    assert prov.preprocessing is not None
    assert set(prov.preprocessing.models_configured) == {'LogisticRegression', 'RandomForest'}
    assert prov.preprocessing.per_model['LogisticRegression'].scaling == 'standard'
    assert prov.preprocessing.per_model['RandomForest'].scaling == 'none'
    assert prov.preprocessing.per_model['LogisticRegression'].imputation == 'median'


def test_record_training():
    prov = make_prov()
    prov.record_training(
        models_trained=['LR', 'RF', 'XGB'],
        primary_model='XGB',
        selection_criteria='validation RMSE',
        use_cv=True,
        cv_folds=5,
        use_hyperopt=False,
        class_weight_balanced=False,
        metrics_by_model={'LR': {'RMSE': 1.2}, 'RF': {'RMSE': 0.9}, 'XGB': {'RMSE': 0.8}},
    )
    assert prov.training is not None
    assert prov.training.models_trained == ['LR', 'RF', 'XGB']
    assert prov.training.primary_model == 'XGB'
    assert prov.training.selection_criteria == 'validation RMSE'
    assert prov.training.use_cv is True
    assert prov.training.cv_folds == 5
    assert prov.training.metrics_by_model['XGB']['RMSE'] == 0.8


def test_record_explainability():
    prov = make_prov()
    prov.record_explainability(
        methods=['shap', 'permutation_importance'],
        models=['XGB', 'RF'],
    )
    assert prov.explainability is not None
    assert 'shap' in prov.explainability.methods_used
    assert 'XGB' in prov.explainability.models_explained


def test_record_sensitivity():
    prov = make_prov()
    prov.record_sensitivity(seed_stability=True, seed_stability_cv=3.5)
    assert prov.sensitivity is not None
    assert prov.sensitivity.seed_stability is True
    assert prov.sensitivity.seed_stability_cv == 3.5
    assert prov.sensitivity.feature_dropout is False

    # Update with dropout
    prov.record_sensitivity(
        seed_stability=prov.sensitivity.seed_stability,
        seed_stability_cv=prov.sensitivity.seed_stability_cv,
        feature_dropout=True,
    )
    assert prov.sensitivity.feature_dropout is True
    assert prov.sensitivity.seed_stability is True


def test_record_statistical_test():
    prov = make_prov()
    prov.record_statistical_test(
        test_name="Pearson correlation",
        variable="age ~ bmi",
        statistic=0.42,
        p_value=0.003,
    )
    assert prov.statistical_validation is not None
    assert len(prov.statistical_validation.tests_run) == 1
    entry = prov.statistical_validation.tests_run[0]
    assert entry['test_name'] == 'Pearson correlation'
    assert entry['variable'] == 'age ~ bmi'
    assert entry['p_value'] == 0.003

    # Second test appends
    prov.record_statistical_test(test_name="Mann-Whitney U", variable="glucose by group")
    assert len(prov.statistical_validation.tests_run) == 2


# ---------------------------------------------------------------------------
# 3. get_completeness()
# ---------------------------------------------------------------------------

def test_get_completeness_empty():
    prov = make_prov()
    comp = prov.get_completeness()
    assert all(v is False for v in comp.values())
    assert set(comp.keys()) == {
        'upload', 'eda', 'feature_engineering', 'feature_selection',
        'split', 'preprocessing', 'training', 'explainability',
        'sensitivity', 'statistical_validation',
    }


def test_get_completeness_partial():
    prov = make_prov()
    prov.record_upload(target_col='y', task_type='classification',
                       feature_cols=['a', 'b'], n_samples=200)
    prov.record_table1()
    comp = prov.get_completeness()
    assert comp['upload'] is True
    assert comp['eda'] is True
    assert comp['feature_engineering'] is False
    assert comp['training'] is False


def test_get_completeness_full():
    prov = make_prov()
    prov.record_upload(target_col='y', task_type='regression',
                       feature_cols=['a'], n_samples=100)
    prov.record_eda_analysis('Distribution')
    prov.record_feature_engineering(transforms=['log'], n_created=1, n_before=1, n_after=2)
    prov.record_feature_selection(method='manual', n_before=2, n_after=1, features_kept=['a'])
    prov.record_split(strategy='random', train_n=70, val_n=15, test_n=15)
    prov.record_preprocessing(configs_by_model={'LR': {}}, imputation_method='mean')
    prov.record_training(models_trained=['LR'], primary_model='LR')
    prov.record_explainability(methods=['shap'], models=['LR'])
    prov.record_sensitivity(seed_stability=True)
    prov.record_statistical_test(test_name='t-test', variable='a')
    comp = prov.get_completeness()
    assert all(v is True for v in comp.values())


# ---------------------------------------------------------------------------
# 4. get_methods_context()
# ---------------------------------------------------------------------------

def test_get_methods_context_returns_expected_keys():
    prov = make_prov()
    prov.record_upload(target_col='glucose', task_type='regression',
                       feature_cols=['age', 'bmi'], n_samples=300)
    prov.record_training(
        models_trained=['XGB'],
        primary_model='XGB',
        selection_criteria='validation RMSE',
        use_cv=True,
        cv_folds=5,
    )
    ctx = prov.get_methods_context()

    # Upload keys
    assert ctx['target_name'] == 'glucose'
    assert ctx['task_type'] == 'regression'
    assert ctx['n_total'] == 300

    # Training keys
    assert ctx['models_trained'] == ['XGB']
    assert ctx['primary_model'] == 'XGB'
    assert ctx['use_cv'] is True
    assert ctx['cv_folds'] == 5


def test_get_methods_context_empty():
    prov = make_prov()
    ctx = prov.get_methods_context()
    assert ctx == {}


# ---------------------------------------------------------------------------
# 5. to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    prov = make_prov()
    prov.record_upload(target_col='y', task_type='classification',
                       feature_cols=['a', 'b', 'c'], n_samples=250, data_source='csv')
    prov.record_cleaning(action='drop nulls', rows_before=250, rows_after=230)
    prov.record_eda_analysis('Histogram')
    prov.record_table1()
    prov.record_feature_selection(
        method='consensus', n_before=10, n_after=5,
        features_kept=['a', 'b'], consensus_methods=['lasso', 'rf'],
    )
    prov.record_split(strategy='stratified', train_n=160, val_n=35, test_n=35, random_seed=123)
    prov.record_preprocessing(
        configs_by_model={'LR': {'numeric_scaling': 'standard', 'categorical_encoding': 'onehot'}},
        imputation_method='median',
    )
    prov.record_training(
        models_trained=['LR', 'RF'],
        primary_model='RF',
        selection_criteria='validation F1',
        use_cv=False,
        use_hyperopt=True,
    )
    prov.record_explainability(methods=['shap'], models=['RF'])
    prov.record_sensitivity(seed_stability=True, seed_stability_cv=2.1, feature_dropout=True)
    prov.record_statistical_test(test_name='chi2', variable='a ~ b', statistic=5.3, p_value=0.02)

    d = prov.to_dict()
    prov2 = WorkflowProvenance.from_dict(d)

    # Upload
    assert prov2.upload.target_col == 'y'
    assert prov2.upload.n_features == 3
    assert len(prov2.upload.cleaning_actions) == 1

    # EDA
    assert 'Histogram' in prov2.eda.analyses_run
    assert prov2.eda.table1_generated is True

    # Feature selection
    assert prov2.feature_selection.method == 'consensus'
    assert prov2.feature_selection.features_kept == ['a', 'b']

    # Split
    assert prov2.split.strategy == 'stratified'
    assert prov2.split.random_seed == 123

    # Preprocessing
    assert prov2.preprocessing.per_model['LR'].scaling == 'standard'
    assert prov2.preprocessing.per_model['LR'].imputation == 'median'

    # Training
    assert prov2.training.primary_model == 'RF'
    assert prov2.training.use_hyperopt is True

    # Explainability
    assert prov2.explainability.methods_used == ['shap']

    # Sensitivity
    assert prov2.sensitivity.seed_stability is True
    assert prov2.sensitivity.seed_stability_cv == 2.1
    assert prov2.sensitivity.feature_dropout is True

    # Statistical validation
    assert len(prov2.statistical_validation.tests_run) == 1
    assert prov2.statistical_validation.tests_run[0]['test_name'] == 'chi2'


def test_from_dict_empty():
    prov = WorkflowProvenance.from_dict({})
    assert prov.upload is None
    assert prov.schema_version == 1


def test_to_dict_is_json_serializable():
    import json
    prov = make_prov()
    prov.record_upload(target_col='y', task_type='regression',
                       feature_cols=['a'], n_samples=50)
    d = prov.to_dict()
    # Should not raise
    json.dumps(d)


# ---------------------------------------------------------------------------
# 6. configs_differ()
# ---------------------------------------------------------------------------

def test_configs_differ_single_model():
    prov = make_prov()
    prov.record_preprocessing(
        configs_by_model={'LR': {'numeric_scaling': 'standard'}},
        imputation_method='mean',
    )
    assert prov.preprocessing.configs_differ() is False


def test_configs_differ_identical_models():
    prov = make_prov()
    prov.record_preprocessing(
        configs_by_model={
            'LR': {'numeric_scaling': 'standard', 'categorical_encoding': 'onehot'},
            'SVM': {'numeric_scaling': 'standard', 'categorical_encoding': 'onehot'},
        },
        imputation_method='mean',
    )
    assert prov.preprocessing.configs_differ() is False


def test_configs_differ_different_scaling():
    prov = make_prov()
    prov.record_preprocessing(
        configs_by_model={
            'LR': {'numeric_scaling': 'standard'},
            'RF': {'numeric_scaling': 'none'},
        },
        imputation_method='mean',
    )
    assert prov.preprocessing.configs_differ() is True


def test_configs_differ_different_encoding():
    prov = make_prov()
    prov.record_preprocessing(
        configs_by_model={
            'LR': {'categorical_encoding': 'onehot'},
            'RF': {'categorical_encoding': 'ordinal'},
        },
        imputation_method='mean',
    )
    assert prov.preprocessing.configs_differ() is True


# ---------------------------------------------------------------------------
# 7. record_upload resets downstream sections
# ---------------------------------------------------------------------------

def test_record_upload_resets_downstream():
    prov = make_prov()
    # Populate downstream sections first
    prov.record_feature_engineering(transforms=['log'], n_created=2, n_before=5, n_after=7)
    prov.record_feature_selection(method='manual', n_before=7, n_after=4, features_kept=['a'])
    prov.record_preprocessing(configs_by_model={'LR': {}}, imputation_method='mean')
    prov.record_training(models_trained=['LR'], primary_model='LR')

    assert prov.feature_engineering is not None
    assert prov.feature_selection is not None
    assert prov.preprocessing is not None
    assert prov.training is not None

    # Re-uploading should reset them
    prov.record_upload(target_col='new_target', task_type='classification',
                       feature_cols=['x', 'y'], n_samples=100)

    assert prov.feature_engineering is None
    assert prov.feature_selection is None
    assert prov.preprocessing is None
    assert prov.training is None
    # upload itself should be updated
    assert prov.upload.target_col == 'new_target'


def test_record_upload_does_not_reset_eda():
    prov = make_prov()
    prov.record_eda_analysis('Correlation')
    prov.record_upload(target_col='y', task_type='regression',
                       feature_cols=['a'], n_samples=50)
    # EDA is not downstream of upload config
    assert prov.eda is not None

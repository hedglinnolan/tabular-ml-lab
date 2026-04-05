"""Tests for NN recommendation engine (issue #95)."""
import pytest
from ml.nn_recommender import recommend_nn_config, NNRecommendation
from ml.dataset_profile import DataSufficiencyLevel, TargetProfile


def _make_target_profile(task_type='regression', skewness=0.5, outlier_rate=0.02):
    return TargetProfile(
        name='target', task_type=task_type, n_unique=100,
        skewness=skewness, outlier_rate=outlier_rate
    )


class TestRecommendationLogic:
    def test_large_dataset_wider_deeper(self):
        rec = recommend_nn_config(
            n_samples=20000, n_features=30,
            target_profile=_make_target_profile(),
            data_sufficiency=DataSufficiencyLevel.ABUNDANT,
        )
        assert rec.params['num_layers'] >= 3
        assert rec.params['layer_width'] >= 64

    def test_small_dataset_conservative(self):
        rec = recommend_nn_config(
            n_samples=150, n_features=5,
            target_profile=_make_target_profile(),
            data_sufficiency=DataSufficiencyLevel.SCARCE,
        )
        assert rec.params['num_layers'] <= 2
        assert rec.params['layer_width'] <= 64
        assert rec.params['dropout'] >= 0.1

    def test_skewed_target_recommends_huber(self):
        rec = recommend_nn_config(
            n_samples=5000, n_features=20,
            target_profile=_make_target_profile(skewness=2.5),
            data_sufficiency=DataSufficiencyLevel.ADEQUATE,
            task_type='regression',
        )
        assert rec.params['loss_function'] == 'huber'

    def test_high_outlier_rate_recommends_weighted_huber(self):
        rec = recommend_nn_config(
            n_samples=5000, n_features=20,
            target_profile=_make_target_profile(outlier_rate=0.15),
            data_sufficiency=DataSufficiencyLevel.ADEQUATE,
            task_type='regression',
        )
        assert rec.params['loss_function'] == 'weighted_huber'
        assert rec.params['grad_clip_norm'] is not None

    def test_well_behaved_target_recommends_mse(self):
        rec = recommend_nn_config(
            n_samples=5000, n_features=20,
            target_profile=_make_target_profile(skewness=0.3, outlier_rate=0.01),
            data_sufficiency=DataSufficiencyLevel.ADEQUATE,
            task_type='regression',
        )
        assert rec.params['loss_function'] == 'mse'

    def test_classification_no_regression_loss(self):
        rec = recommend_nn_config(
            n_samples=5000, n_features=20,
            target_profile=_make_target_profile(task_type='classification'),
            data_sufficiency=DataSufficiencyLevel.ADEQUATE,
            task_type='classification',
        )
        # Classification uses BCE/CE automatically, loss_function defaults to mse (ignored)
        assert rec.params['loss_function'] == 'mse'

    def test_batchnorm_for_deep_networks(self):
        rec = recommend_nn_config(
            n_samples=10000, n_features=25,
            data_sufficiency=DataSufficiencyLevel.ADEQUATE,
        )
        if rec.params['num_layers'] >= 3:
            assert rec.params['use_batchnorm'] is True

    def test_batchnorm_disabled_for_shallow(self):
        rec = recommend_nn_config(
            n_samples=200, n_features=5,
            data_sufficiency=DataSufficiencyLevel.SCARCE,
        )
        assert rec.params['use_batchnorm'] is False

    def test_high_ratio_low_regularization(self):
        rec = recommend_nn_config(
            n_samples=50000, n_features=10,
            data_sufficiency=DataSufficiencyLevel.ABUNDANT,
        )
        assert rec.params['dropout'] <= 0.1
        assert rec.params['weight_decay'] <= 1e-5

    def test_low_ratio_high_regularization(self):
        rec = recommend_nn_config(
            n_samples=100, n_features=50,
            data_sufficiency=DataSufficiencyLevel.SCARCE,
        )
        assert rec.params['dropout'] >= 0.2
        assert rec.params['weight_decay'] >= 1e-4

    def test_cosine_for_large_datasets(self):
        rec = recommend_nn_config(
            n_samples=20000, n_features=20,
            data_sufficiency=DataSufficiencyLevel.ABUNDANT,
        )
        assert rec.params['lr_scheduler'] == 'cosine_warm_restarts'

    def test_reduce_on_plateau_for_small_datasets(self):
        rec = recommend_nn_config(
            n_samples=500, n_features=10,
            data_sufficiency=DataSufficiencyLevel.LIMITED,
        )
        assert rec.params['lr_scheduler'] == 'reduce_on_plateau'

    def test_interactions_reduce_depth(self):
        rec = recommend_nn_config(
            n_samples=10000, n_features=30,
            data_sufficiency=DataSufficiencyLevel.ADEQUATE,
            has_engineered_interactions=True,
        )
        assert rec.params['num_layers'] <= 3

    def test_funnel_for_high_dim(self):
        rec = recommend_nn_config(
            n_samples=500, n_features=50,
            data_sufficiency=DataSufficiencyLevel.LIMITED,
        )
        assert rec.params['architecture_pattern'] == 'funnel'


class TestRecommendationCompleteness:
    def test_all_params_present(self):
        rec = recommend_nn_config(n_samples=1000, n_features=20)
        expected_keys = {
            'num_layers', 'layer_width', 'architecture_pattern', 'dropout',
            'weight_decay', 'lr', 'batch_size', 'loss_function', 'use_batchnorm',
            'lr_scheduler', 'grad_clip_norm', 'activation', 'epochs', 'patience'
        }
        assert expected_keys.issubset(set(rec.params.keys()))

    def test_all_params_have_reasoning(self):
        rec = recommend_nn_config(n_samples=1000, n_features=20)
        for key in rec.params:
            assert key in rec.reasoning, f"Missing reasoning for {key}"
            assert len(rec.reasoning[key]) > 0, f"Empty reasoning for {key}"

    def test_config_source_is_recommended(self):
        rec = recommend_nn_config(n_samples=1000, n_features=20)
        assert rec.config_source == "recommended"

    def test_return_type(self):
        rec = recommend_nn_config(n_samples=1000, n_features=20)
        assert isinstance(rec, NNRecommendation)
        assert isinstance(rec.params, dict)
        assert isinstance(rec.reasoning, dict)

    def test_no_target_profile_still_works(self):
        rec = recommend_nn_config(
            n_samples=1000, n_features=20, target_profile=None
        )
        assert rec.params['loss_function'] == 'mse'

    def test_nhanes_like_benchmark(self):
        """Verify NHANES-like config is close to AutoGluon's optimal."""
        tp = TargetProfile(
            name='glucose', task_type='regression', n_unique=1000,
            skewness=1.53, outlier_rate=0.05
        )
        rec = recommend_nn_config(
            n_samples=19784, n_features=26,
            target_profile=tp,
            data_sufficiency=DataSufficiencyLevel.ABUNDANT,
            task_type='regression',
        )
        # AutoGluon winner: 4 layers x 128, lr=0.0003, wd=1e-6
        assert rec.params['num_layers'] >= 3
        assert rec.params['layer_width'] >= 64
        assert rec.params['lr'] <= 0.001
        assert rec.params['weight_decay'] <= 1e-5


class TestProvenanceFields:
    def test_training_provenance_new_fields(self):
        from utils.workflow_provenance import TrainingProvenance
        tp = TrainingProvenance(
            nn_config_source='recommended',
            nn_config_reasoning={'lr': '0.001 — standard'},
            nn_config_modifications={'lr': 0.0005},
        )
        assert tp.nn_config_source == 'recommended'
        assert 'lr' in tp.nn_config_reasoning
        assert tp.nn_config_modifications['lr'] == 0.0005

    def test_record_training_accepts_nn_fields(self):
        from utils.workflow_provenance import WorkflowProvenance
        prov = WorkflowProvenance()
        prov.record_training(
            models_trained=['nn'],
            nn_config_source='recommended',
            nn_config_reasoning={'lr': 'test'},
            nn_config_modifications={},
        )
        assert prov.training.nn_config_source == 'recommended'

    def test_methods_context_includes_nn_fields(self):
        from utils.workflow_provenance import WorkflowProvenance
        prov = WorkflowProvenance()
        prov.record_training(
            models_trained=['nn'],
            nn_config_source='recommended+modified',
            nn_config_reasoning={'lr': '0.001'},
            nn_config_modifications={'lr': 0.0005},
        )
        ctx = prov.get_methods_context()
        assert ctx['nn_config_source'] == 'recommended+modified'
        assert 'lr' in ctx['nn_config_modifications']

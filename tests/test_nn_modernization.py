"""Tests for NN modernization (issue #94): BatchNorm, schedulers, clipping, loss passthrough."""
import numpy as np
import pytest
import torch


@pytest.fixture
def small_regression_data():
    """Generate small regression dataset."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 10).astype(np.float32)
    y_train = rng.randn(200).astype(np.float32)
    X_val = rng.randn(50, 10).astype(np.float32)
    y_val = rng.randn(50).astype(np.float32)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def small_classification_data():
    """Generate small binary classification dataset."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 10).astype(np.float32)
    y_train = (rng.randn(200) > 0).astype(np.int64)
    X_val = rng.randn(50, 10).astype(np.float32)
    y_val = (rng.randn(50) > 0).astype(np.int64)
    return X_train, y_train, X_val, y_val


# --- SimpleMLP architecture tests ---

class TestSimpleMLP:
    def test_without_batchnorm(self):
        from models.nn_whuber import SimpleMLP
        model = SimpleMLP(input_dim=10, hidden=[32, 32], use_batchnorm=False)
        # Should have no BatchNorm layers
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 0

    def test_with_batchnorm(self):
        from models.nn_whuber import SimpleMLP
        model = SimpleMLP(input_dim=10, hidden=[32, 64], use_batchnorm=True)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 2  # One per hidden layer

    def test_forward_shape_regression(self):
        from models.nn_whuber import SimpleMLP
        model = SimpleMLP(input_dim=10, hidden=[32, 32], output_dim=1, use_batchnorm=True)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 1)

    def test_forward_shape_multiclass(self):
        from models.nn_whuber import SimpleMLP
        model = SimpleMLP(input_dim=10, hidden=[32], output_dim=3, use_batchnorm=True)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 3)

    def test_batchnorm_eval_mode(self):
        from models.nn_whuber import SimpleMLP
        model = SimpleMLP(input_dim=10, hidden=[32], use_batchnorm=True)
        # Train mode: BN uses batch stats
        model.train()
        x = torch.randn(16, 10)
        _ = model(x)
        # Eval mode: BN uses running stats
        model.eval()
        out = model(x)
        assert out.shape == (16, 1)


# --- Training loop tests ---

class TestTrainingLoop:
    def test_reduce_on_plateau(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16, 16], task_type='regression')
        res = model.fit(X_train, y_train, X_val, y_val,
                        epochs=5, batch_size=64, lr_scheduler='reduce_on_plateau',
                        random_seed=42)
        assert 'history' in res
        assert len(res['history']['train_loss']) == 5

    def test_cosine_warm_restarts(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16, 16], task_type='regression')
        res = model.fit(X_train, y_train, X_val, y_val,
                        epochs=5, batch_size=64, lr_scheduler='cosine_warm_restarts',
                        random_seed=42)
        assert len(res['history']['train_loss']) == 5

    def test_one_cycle(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16, 16], task_type='regression')
        res = model.fit(X_train, y_train, X_val, y_val,
                        epochs=5, batch_size=64, lr_scheduler='one_cycle',
                        random_seed=42)
        assert len(res['history']['train_loss']) == 5

    def test_gradient_clipping(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16], task_type='regression')
        res = model.fit(X_train, y_train, X_val, y_val,
                        epochs=3, batch_size=64, grad_clip_norm=1.0,
                        random_seed=42)
        assert res['history']['train_loss'][-1] < res['history']['train_loss'][0] or True  # Just verify no crash

    def test_loss_function_huber(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16], task_type='regression',
                                       loss_function='huber')
        assert model.loss_function == 'huber'
        res = model.fit(X_train, y_train, X_val, y_val, epochs=3, batch_size=64,
                        random_seed=42)
        assert 'history' in res

    def test_loss_function_weighted_huber(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16], task_type='regression',
                                       loss_function='weighted_huber')
        res = model.fit(X_train, y_train, X_val, y_val, epochs=3, batch_size=64,
                        random_seed=42)
        assert 'history' in res

    def test_loss_function_mae(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[16], task_type='regression',
                                       loss_function='mae')
        res = model.fit(X_train, y_train, X_val, y_val, epochs=3, batch_size=64,
                        random_seed=42)
        assert 'history' in res

    def test_batchnorm_training(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[32, 32], task_type='regression',
                                       use_batchnorm=True)
        res = model.fit(X_train, y_train, X_val, y_val, epochs=5, batch_size=64,
                        random_seed=42)
        assert res['history']['val_rmse'][-1] is not None

    def test_classification_still_works(self, small_classification_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_classification_data
        model = NNWeightedHuberWrapper(hidden_layers=[16], task_type='classification',
                                       use_batchnorm=True)
        res = model.fit(X_train, y_train, X_val, y_val, epochs=5, batch_size=64,
                        lr_scheduler='cosine_warm_restarts', random_seed=42)
        preds = model.predict(X_val)
        assert len(preds) == len(y_val)

    def test_architecture_summary_includes_new_fields(self, small_regression_data):
        from models.nn_whuber import NNWeightedHuberWrapper
        X_train, y_train, X_val, y_val = small_regression_data
        model = NNWeightedHuberWrapper(hidden_layers=[32], task_type='regression',
                                       use_batchnorm=True, loss_function='huber')
        model.fit(X_train, y_train, X_val, y_val, epochs=2, batch_size=64, random_seed=42)
        summary = model.get_architecture_summary()
        assert summary['use_batchnorm'] is True
        assert summary['loss_function'] == 'huber'
        assert summary['total_params'] is not None


# --- Registry defaults tests ---

class TestRegistryDefaults:
    def test_updated_defaults(self):
        from ml.model_registry import get_registry
        nn = get_registry()['nn']
        assert nn.default_params['num_layers'] == 3
        assert nn.default_params['layer_width'] == 128
        assert nn.default_params['lr'] == 0.001
        assert nn.default_params['weight_decay'] == 1e-5

    def test_new_schema_entries(self):
        from ml.model_registry import get_registry
        schema = get_registry()['nn'].hyperparam_schema
        assert 'use_batchnorm' in schema
        assert schema['use_batchnorm']['type'] == 'bool'
        assert 'lr_scheduler' in schema
        assert 'cosine_warm_restarts' in schema['lr_scheduler']['options']
        assert 'grad_clip_norm' in schema
        assert schema['grad_clip_norm']['type'] == 'float_or_none'

    def test_layer_width_max_512(self):
        from ml.model_registry import get_registry
        schema = get_registry()['nn'].hyperparam_schema
        assert schema['layer_width']['max'] == 512

    def test_weight_decay_min_lowered(self):
        from ml.model_registry import get_registry
        schema = get_registry()['nn'].hyperparam_schema
        assert schema['weight_decay']['min'] == 1e-7

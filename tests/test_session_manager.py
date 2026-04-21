"""Tests for the pickle-free session save/load format.

Security-critical: these tests verify that the session manager does not
accept pickle input and cannot be coerced into deserializing arbitrary
Python. Every path that touches user bytes is JSON or Parquet only.
"""
from __future__ import annotations

import io
import json
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

# Streamlit session-state mock -------------------------------------------
# We mock st.session_state with a plain dict so these tests run without a
# real Streamlit runtime.

class _FakeSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def fake_session(monkeypatch):
    from utils import session_manager
    state = _FakeSessionState()

    class _FakeStreamlit:
        session_state = state

    monkeypatch.setattr(session_manager, "st", _FakeStreamlit())
    # Some helpers on the load path import session_state.reset_data_dependent_state.
    # Replace that with a no-op against our fake dict so tests don't touch
    # real Streamlit state.
    from utils import session_state as ss_module
    monkeypatch.setattr(ss_module, "st", _FakeStreamlit())
    return state


def _populated_state(state):
    """Put a realistic mix of values into the fake session state."""
    from utils.session_state import (
        DataConfig, SplitConfig, ModelConfig,
        TaskTypeDetection, CohortStructureDetection,
    )
    from utils.insight_ledger import Insight, InsightLedger

    state["raw_data"] = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "bmi": [22.5, 24.1, 26.3, 23.8],
        "glucose": [95, 102, 110, 98],
    })
    state["df_engineered"] = state["raw_data"].assign(age_sq=state["raw_data"].age ** 2)
    state["datasets_registry"] = {
        "primary": state["raw_data"],
        "external": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
    }
    state["data_config"] = DataConfig(
        target_col="glucose", feature_cols=["age", "bmi"], task_type="regression",
    )
    state["split_config"] = SplitConfig(train_size=0.7, val_size=0.15, test_size=0.15, random_state=42)
    state["model_config"] = ModelConfig(nn_epochs=150)
    state["task_type_detection"] = TaskTypeDetection(detected="regression", confidence="high")
    state["cohort_structure_detection"] = CohortStructureDetection(detected="cross_sectional")
    state["selected_features"] = ["age", "bmi"]
    state["engineered_feature_names"] = ["age", "bmi", "age_sq"]
    state["methodology_log"] = [{"step": "EDA", "action": "correlation matrix"}]
    state["random_seed"] = 42
    state["cv_folds"] = 5
    state["use_cv"] = True
    state["workflow_mode"] = "quick"
    state["current_page"] = "05_Preprocess"

    ledger = InsightLedger()
    ledger.add(Insight(
        id="test_insight",
        source_page="02_EDA",
        category="data_quality",
        severity="warning",
        finding="Missingness in age column",
        implication="May bias results",
        recommended_action="Impute or drop",
    ))
    state["insight_ledger"] = ledger

    # Things that must NEVER be written to the save file.
    state["trained_models"] = {"rf": object()}
    state["fitted_estimators"] = {"rf": object()}
    state["X_train"] = np.array([[1, 2], [3, 4]])
    state["y_train"] = np.array([0, 1])
    state["openai_api_key"] = "sk-SECRETKEY-DO-NOT-LEAK"
    state["anthropic_api_key"] = "sk-ant-SECRETKEY-DO-NOT-LEAK"

    # Safe widget keys
    state["llm_backend"] = "ollama"
    state["ollama_model"] = "qwen3.5:9b"


# --- Save -----------------------------------------------------------------

def test_save_produces_valid_zip(fake_session):
    from utils.session_manager import _collect_session_data
    _populated_state(fake_session)
    archive_bytes, manifest = _collect_session_data()
    assert zipfile.is_zipfile(io.BytesIO(archive_bytes))
    assert manifest["schema_version"] == "2.0"
    assert "data_config" in manifest["saved_keys"]
    assert "insight_ledger" in manifest["saved_keys"]


def test_save_never_writes_api_keys(fake_session):
    """Hard guarantee: API keys must not appear anywhere in the archive."""
    from utils.session_manager import _collect_session_data
    _populated_state(fake_session)
    archive_bytes, _ = _collect_session_data()
    assert b"sk-SECRETKEY-DO-NOT-LEAK" not in archive_bytes
    assert b"sk-ant-SECRETKEY-DO-NOT-LEAK" not in archive_bytes
    assert b"openai_api_key" not in archive_bytes
    assert b"anthropic_api_key" not in archive_bytes


def test_save_never_writes_trained_models(fake_session):
    """Trained models, fitted estimators, and splits must not be persisted."""
    from utils.session_manager import _collect_session_data
    _populated_state(fake_session)
    archive_bytes, manifest = _collect_session_data()
    for forbidden in (
        "trained_models", "fitted_estimators",
        "X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
    ):
        assert forbidden not in manifest["saved_keys"]

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        # Ensure no member claims to hold these artifacts
        for name in zf.namelist():
            assert "X_train" not in name and "trained_models" not in name


def test_save_on_empty_state_contains_only_manifest(fake_session):
    from utils.session_manager import _collect_session_data
    archive_bytes, manifest = _collect_session_data()
    assert manifest["saved_keys"] == []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        names = zf.namelist()
        # config.json, widget_state.json, manifest.json are always written
        # (even empty) because they are structural.
        assert "manifest.json" in names
        assert "config.json" in names


# --- Load -----------------------------------------------------------------

def test_round_trip_preserves_values(fake_session):
    from utils.session_manager import _collect_session_data, _restore_session_data
    from utils.session_state import DataConfig

    _populated_state(fake_session)
    archive_bytes, _ = _collect_session_data()

    # Simulate a fresh session by clearing + restoring.
    fake_session.clear()
    restored_count, manifest = _restore_session_data(archive_bytes)
    assert restored_count > 0
    assert manifest["workflow_step"] == "05_Preprocess"

    # Scalar / list round-trip
    assert fake_session["random_seed"] == 42
    assert fake_session["selected_features"] == ["age", "bmi"]

    # Dataclass round-trip
    dc = fake_session["data_config"]
    assert isinstance(dc, DataConfig)
    assert dc.target_col == "glucose"
    assert dc.feature_cols == ["age", "bmi"]

    # DataFrame round-trip
    assert fake_session["raw_data"].shape == (4, 3)
    assert list(fake_session["raw_data"].columns) == ["age", "bmi", "glucose"]

    # InsightLedger round-trip
    ledger = fake_session["insight_ledger"]
    assert len(ledger) == 1
    assert ledger._insights[0].id == "test_insight"

    # datasets_registry round-trip
    registry = fake_session["datasets_registry"]
    assert set(registry.keys()) == {"primary", "external"}

    # Widget state is deferred to _pending_widget_state_restore
    pending = fake_session.get("_pending_widget_state_restore", {})
    assert pending.get("llm_backend") == "ollama"
    assert pending.get("ollama_model") == "qwen3.5:9b"


def test_round_trip_does_not_restore_api_keys(fake_session):
    """Even if someone hand-crafts a save file containing API keys, they
    are filtered out on load because the widget whitelist excludes them."""
    from utils import session_manager
    from utils.session_manager import _restore_session_data

    members = {
        "manifest.json": json.dumps({
            "schema_version": session_manager.SAVE_SCHEMA_VERSION,
            "saved_at": "2026-04-21T00:00:00",
            "workflow_step": "01",
            "saved_keys": [], "skipped_keys": [], "members": [],
        }).encode(),
        "config.json": b"{}",
        # Attacker-crafted widget state trying to inject an API key:
        "widget_state.json": json.dumps({
            "llm_backend": "openai",
            "openai_api_key": "sk-INJECTED",
            "anthropic_api_key": "sk-INJECTED",
        }).encode(),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    _restore_session_data(buf.getvalue())
    pending = fake_session.get("_pending_widget_state_restore", {})
    assert "openai_api_key" not in pending
    assert "anthropic_api_key" not in pending
    assert pending.get("llm_backend") == "openai"


# --- Rejection paths ------------------------------------------------------

def test_rejects_pickle_file(fake_session):
    from utils.session_manager import _restore_session_data, SessionLoadError

    # A legit pickle payload. If our code ever called pickle.loads on this,
    # Python would reconstruct the dict. We assert it refuses to try.
    payload = pickle.dumps({"raw_data": "hello"})
    assert payload[0] == 0x80  # PROTO opcode

    with pytest.raises(SessionLoadError, match="pickle"):
        _restore_session_data(payload)


def test_rejects_non_zip(fake_session):
    from utils.session_manager import _restore_session_data, SessionLoadError
    with pytest.raises(SessionLoadError, match="[Nn]ot a valid"):
        _restore_session_data(b"not a zip file, just some bytes")


def test_rejects_oversized_upload(fake_session):
    from utils import session_manager
    from utils.session_manager import _restore_session_data, SessionLoadError

    giant = b"\x00" * (session_manager._MAX_UPLOAD_BYTES + 1)
    with pytest.raises(SessionLoadError, match="too large"):
        _restore_session_data(giant)


def test_rejects_path_traversal(fake_session):
    from utils.session_manager import _restore_session_data, SessionLoadError

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.json", b"{}")
    with pytest.raises(SessionLoadError, match="Unsafe member path"):
        _restore_session_data(buf.getvalue())


def test_rejects_absolute_path(fake_session):
    from utils.session_manager import _restore_session_data, SessionLoadError

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("/etc/passwd", b"{}")
    with pytest.raises(SessionLoadError, match="Unsafe member path"):
        _restore_session_data(buf.getvalue())


def test_rejects_too_many_members(fake_session):
    from utils import session_manager
    from utils.session_manager import _restore_session_data, SessionLoadError

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(session_manager._MAX_MEMBERS + 5):
            zf.writestr(f"entry_{i}.json", b"{}")
    with pytest.raises(SessionLoadError, match="max"):
        _restore_session_data(buf.getvalue())


def test_rejects_missing_manifest(fake_session):
    from utils.session_manager import _restore_session_data, SessionLoadError

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.json", b"{}")
    with pytest.raises(SessionLoadError, match="manifest.json"):
        _restore_session_data(buf.getvalue())


def test_rejects_unsupported_version(fake_session):
    from utils.session_manager import _restore_session_data, SessionLoadError

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({
            "schema_version": "99.99",
            "saved_at": "2026-04-21T00:00:00",
            "workflow_step": "01",
            "saved_keys": [], "skipped_keys": [], "members": [],
        }).encode())
    with pytest.raises(SessionLoadError, match="schema version"):
        _restore_session_data(buf.getvalue())


def test_unknown_config_keys_are_ignored_not_restored(fake_session):
    """A hostile save file cannot inject arbitrary session state keys."""
    from utils import session_manager
    from utils.session_manager import _restore_session_data

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({
            "schema_version": session_manager.SAVE_SCHEMA_VERSION,
            "saved_at": "2026-04-21T00:00:00",
            "workflow_step": "01",
            "saved_keys": [], "skipped_keys": [], "members": [],
        }).encode())
        zf.writestr("config.json", json.dumps({
            # Whitelisted:
            "random_seed": 123,
            # Not whitelisted -- must be ignored:
            "arbitrary_evil_key": "attacker_payload",
            "__class__": "attacker_payload",
            "trained_models": {"rf": "attacker_payload"},
        }).encode())

    _restore_session_data(buf.getvalue())
    assert fake_session["random_seed"] == 123
    assert "arbitrary_evil_key" not in fake_session
    # trained_models is in NEVER_PERSIST so it was cleared by
    # _clear_downstream_state; we assert it was not overwritten from the file.
    assert fake_session.get("trained_models") != {"rf": "attacker_payload"}

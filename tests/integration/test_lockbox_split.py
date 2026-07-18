"""End-to-end: the Prepare Splits button on Train & Compare must use the
upload lockbox as THE test set (Tier 2, AppTest).

This pins the structural leakage fix at the widget level: frozen labels in,
identical test indices out — not just the unit-level partition math.
"""
import numpy as np
import pandas as pd
import pytest

from tests.integration.conftest import build_test_dataframe, inject_data_state


@pytest.fixture
def apptest_train_page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file("pages/06_Train_and_Compare.py")


def _click_button(at, label_fragment):
    for b in at.button:
        if label_fragment in (b.label or ""):
            return b.click()
    raise AssertionError(f"No button matching '{label_fragment}' found")


def _inject_pipeline(at, df, target_col="glucose"):
    """Satisfy page 06's preprocessing gate with a minimal per-model pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    numeric = [c for c in df.columns
               if c != target_col and df[c].dtype.kind in "fi"]
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), numeric)],
        remainder="drop",
    )
    at.session_state["preprocessing_pipelines_by_model"] = {"ridge": pre}
    at.session_state["preprocessing_config_by_model"] = {"ridge": {}}


def test_prepare_splits_uses_lockbox_labels(apptest_train_page):
    at = apptest_train_page
    df = build_test_dataframe(n=200)
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)

    # Freeze a lockbox exactly as page 01 would (labels are df.index values)
    rng = np.random.RandomState(42)
    test_labels = sorted(rng.choice(df.index.values, size=30, replace=False).tolist())
    at.session_state["test_lockbox"] = {
        "labels": test_labels,
        "fraction": 0.15,
        "seed": 42,
        "n_total": len(df),
        "n_test": len(test_labels),
        "signature": "test-fixture",
        "stratified": False,
    }

    at.run(timeout=120)
    assert not at.exception, f"page errored before interaction: {at.exception}"

    _click_button(at, "Prepare Splits")
    at.run(timeout=180)
    assert not at.exception, f"Prepare Splits errored: {at.exception}"

    test_indices = at.session_state["test_indices"]
    assert test_indices is not None and len(test_indices) > 0

    # df has a default RangeIndex, so positional indices == index labels.
    # Every produced test index must come from the lockbox, and (allowing for
    # rows dropped by target-NaN masking — none in this fixture) cover it.
    assert set(test_indices) == set(test_labels), (
        "Prepare Splits did not use the lockbox labels as the test set"
    )

    # Train/val must be disjoint from the lockbox
    train_indices = set(at.session_state["train_indices"])
    val_indices = set(at.session_state["val_indices"])
    assert not (train_indices & set(test_labels))
    assert not (val_indices & set(test_labels))


def test_prepare_splits_exploratory_mode_ignores_lockbox(apptest_train_page):
    at = apptest_train_page
    df = build_test_dataframe(n=200)
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    at.session_state["test_lockbox"] = {
        "labels": list(df.index[:30]),
        "fraction": 0.15,
        "seed": 42,
        "n_total": len(df),
        "n_test": 30,
        "signature": "test-fixture",
        "stratified": False,
    }
    at.session_state["exploratory_mode"] = True

    at.run(timeout=120)
    _click_button(at, "Prepare Splits")
    at.run(timeout=180)
    assert not at.exception

    # With quarantine explicitly off, the page draws its own split; the test
    # set need not (and with this seed, will not) equal the lockbox labels.
    test_indices = set(at.session_state["test_indices"])
    assert test_indices != set(df.index[:30])

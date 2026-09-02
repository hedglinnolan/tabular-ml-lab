"""Drive the real Train & Compare page and read the price above the button.

The unit tests pin the arithmetic. This one pins the promise, in the runtime
that shows it: with a pipeline built on Preprocess and splits prepared, the
caption under the fit counts states a duration for this machine BEFORE the
click, names the rows it was measured on, and changes with the frame — a
600-row study and a 6,000-row study are not quoted the same sentence.

Each page is a fresh AppTest carrying the previous one's session state, the
same journey shape as test_the_next_cohort_rebuilds_the_same_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

pytestmark = pytest.mark.timeout(900)

FEATURES = ["age", "bmi", "chol", "sbp"]


def _study(n, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age": rng.integers(20, 80, n).astype(float),
                       "bmi": np.round(rng.normal(28, 5, n), 1),
                       "chol": np.round(rng.normal(200, 30, n), 1),
                       "sbp": np.round(rng.normal(125, 15, n), 1)})
    logit = 0.04 * (df["age"] - 50) + 0.1 * (df["bmi"] - 28) + rng.normal(0, 1, n)
    df["y"] = (logit > 0).astype(int)
    return df


def _state(n):
    """The session as it stands when the researcher opens Preprocess."""
    from utils.session_state import DataConfig, set_data
    from utils.test_lockbox import ensure_lockbox
    from ml.dataset_profile import compute_dataset_profile

    st.session_state.clear()
    df = _study(n)
    set_data(df)
    st.session_state["task_mode"] = "prediction"
    st.session_state["data_config"] = DataConfig(
        target_col="y", feature_cols=list(FEATURES), task_type="classification")
    st.session_state["selected_features"] = list(FEATURES)
    st.session_state["data_audit"] = {"n_rows": len(df), "n_cols": len(df.columns)}
    try:
        st.session_state["dataset_profile"] = compute_dataset_profile(
            df, "y", FEATURES, "classification")
    except Exception:
        pass
    ensure_lockbox(df, "y", "classification")
    # Two models with different curves: a forest and a linear model.
    st.session_state["train_model_rf"] = True
    st.session_state["train_model_logreg"] = True
    st.session_state["_coach_applied"] = True
    st.session_state["use_cv"] = True
    st.session_state["cv_folds"] = 3
    state = {k: v for k, v in st.session_state.items()}
    st.session_state.clear()
    return state


def _open(page, state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def _ok(at, label):
    from tests.integration.conftest import HARNESS_ONLY_EXCEPTIONS
    real = [str(e.value)[:600] for e in at.exception
            if not any(p in str(e.value) for p in HARNESS_ONLY_EXCEPTIONS)]
    assert not real, (label, real)


def _click(at, key=None, label=None):
    if key is not None:
        at.button(key=key).click()
    else:
        hit = [b for b in at.button if label in str(b.label)]
        assert hit, f"no button labeled {label!r}; have {[str(b.label) for b in at.button]}"
        hit[0].click()
    at.run()


def _priced_page_06(n):
    at = _open("pages/05_Preprocess.py", _state(n))
    at.run()
    _ok(at, f"05 render ({n} rows)")
    _click(at, key="preprocess_build_button")
    _ok(at, f"05 build ({n} rows)")
    state = dict(at.session_state.filtered_state)

    at = _open("pages/06_Train_and_Compare.py", state)
    at.run()
    _ok(at, f"06 render ({n} rows)")
    _click(at, label="Prepare Splits")
    _ok(at, f"06 splits ({n} rows)")
    captions = [str(c.value) for c in at.caption]
    cost = dict(at.session_state["_train_cost"])
    buttons = [str(b.label) for b in at.button]
    return captions, cost, buttons


@pytest.fixture(scope="module")
def priced():
    small_captions, small_cost, small_buttons = _priced_page_06(600)
    large_captions, large_cost, _ = _priced_page_06(6_000)
    return {"small": (small_captions, small_cost), "large": (large_captions, large_cost),
            "buttons": small_buttons}


def _time_line(captions):
    hits = [c for c in captions if "Time on this machine" in c]
    assert len(hits) == 1, captions
    return hits[0]


def test_the_estimate_is_stated_before_the_click(priced):
    captions, cost = priced["small"]
    line = _time_line(captions)
    assert "Train Models about" in line or "Train Models under a second" in line, line
    assert "with hyperparameter optimization" in line
    assert "a floor" in line, "Optuna's number is called what it is"
    assert "Train Models" in " ".join(priced["buttons"]), "the buttons are there to be clicked"
    assert cost["standard"] > 0 and cost["optimized"] > cost["standard"]


def test_the_estimate_names_what_it_was_measured_on(priced):
    captions, cost = priced["small"]
    line = _time_line(captions)
    assert ("measured just now on your full training set" in line
            or "projected from sample fits of" in line), line
    assert cost["provenance"] and cost["provenance"] in line


def test_every_selected_model_is_quoted_or_named_as_not_estimated(priced):
    captions, cost = priced["small"]
    line = _time_line(captions)
    assert "RF" in line and "LOGREG" in line
    assert set(cost["per_model"]) == {"rf", "logreg"}
    assert cost["unestimated"] == []


def test_the_memory_arithmetic_is_kept_and_the_line_is_not_shown_for_a_small_frame(priced):
    """The bytes are exact (rows x post-preprocessing width x 8) whatever the
    size; the LINE appears only once the folds hold enough to be worth one,
    which a 600-row study with four predictors does not."""
    captions, cost = priced["small"]
    assert cost["cv_folds"] == 3
    assert cost["train_matrix_bytes"] % (cost["n_train"] * 8) == 0, "rows x width x 8"
    assert cost["train_matrix_bytes"] >= cost["n_train"] * 4 * 8, "at least the four predictors"
    assert not [c for c in captions if "Memory while cross-validating" in c]


def test_a_larger_frame_is_quoted_a_larger_number(priced):
    """The whole point: the sentence is a function of the data."""
    _, small = priced["small"]
    _, large = priced["large"]
    assert large["n_train"] > small["n_train"]
    assert large["standard"] >= small["standard"], (small, large)
    assert large["per_model"]["rf"]["fit"] >= small["per_model"]["rf"]["fit"]


def test_the_quote_is_kept_for_the_run_to_restate(priced):
    """`_train_models` reads `_train_cost` back to quote each model as it
    starts; the keys it reads must be the ones the caption wrote."""
    _, cost = priced["small"]
    assert {"standard", "optimized", "per_model", "provenance", "n_train", "unestimated"} <= set(cost)
    for entry in cost["per_model"].values():
        assert {"fit", "cv", "optuna"} == set(entry)

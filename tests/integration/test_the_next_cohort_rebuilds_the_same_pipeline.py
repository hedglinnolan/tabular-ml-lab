"""Drive the real pages through a cohort switch and rebuild the second group.

The unit tests pin the mechanism. This one pins the promise, in the runtime
that broke it: Preprocess is configured by hand for the women, Train &
Compare fits them, the button switches to the men, and the men's pipeline is
rebuilt from the carried settings by the same page that lost them before.

Each page is a fresh AppTest carrying the previous one's session state, which
is how a multipage app looks to Streamlit's script runner: one script per run,
state in between. What a fresh AppTest cannot imitate is the widget cull that
happens on a live page switch — which is exactly why the fix does not depend
on any widget key surviving one.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils import replay as R

pytestmark = pytest.mark.timeout(900)

FEATURES = ["age", "bmi", "chol"]
DECIDED = {"numeric_scaling": "robust", "numeric_imputation": "mean",
           "numeric_outlier_treatment": "mad", "numeric_power_transform": "yeo-johnson"}
MAD_K = 3.0
LOGREG_C = 0.5


def _study(n=600):
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                       "age": rng.integers(20, 80, n).astype(float),
                       "bmi": np.round(rng.normal(28, 5, n), 1),
                       "chol": np.round(rng.normal(200, 30, n), 1)})
    logit = 0.04 * (df["age"] - 50) + 0.1 * (df["bmi"] - 28) + rng.normal(0, 1, n)
    df["y"] = (logit > 0).astype(int)
    return df


def _female_state():
    """The session as it stands when the researcher opens Preprocess."""
    from utils.cohorts import plan_cohorts, start_cohort
    from utils.session_state import DataConfig, set_data
    from utils.test_lockbox import ensure_lockbox, train_row_mask
    from ml.dataset_profile import compute_dataset_profile

    st.session_state.clear()
    df = _study()
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
    plan = plan_cohorts(df, "sex", "y", "classification",
                        train_mask=train_row_mask(df.index))
    start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")
    st.session_state["train_model_logreg"] = True
    st.session_state["_coach_applied"] = True          # the researcher's pick, not the coach's
    st.session_state["use_cv"] = False
    state = {k: v for k, v in st.session_state.items()}
    st.session_state.clear()
    return state


def _open(page, state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def _state_of(at):
    return dict(at.session_state.filtered_state)


def _ok(at, label):
    # st.page_link() cannot resolve a page under a single-file AppTest, and
    # page 05 renders one after a successful build. That is the harness's
    # failure, not the page's; conftest names every wording it has had.
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


def _decision_keys(cfg):
    return {k: v for k, v in cfg.items()
            if k not in ("numeric_features", "categorical_features", "n_output_features",
                         "overrides", "unit_harmonization_config", "plausibility_bounds")}


@pytest.fixture(scope="module")
def journey():
    """One pass through both groups; every test below reads from it."""
    out = {}

    # ── Female: Preprocess, by hand ────────────────────────────────────
    at = _open("pages/05_Preprocess.py", _female_state())
    at.run()
    _ok(at, "05 female first render")
    at.radio(key="preprocess_config_mode").set_value(R.ADVANCED_MODE_LABEL)
    at.run()
    _ok(at, "05 female advanced")
    at.selectbox(key="preprocess_logreg_numeric_scaling").set_value(DECIDED["numeric_scaling"])
    at.selectbox(key="preprocess_logreg_numeric_imputation_display").set_value(DECIDED["numeric_imputation"])
    at.selectbox(key="preprocess_logreg_numeric_outlier_treatment").set_value(DECIDED["numeric_outlier_treatment"])
    at.selectbox(key="preprocess_logreg_numeric_power_transform").set_value(DECIDED["numeric_power_transform"])
    at.run()
    _ok(at, "05 female choices")
    at.number_input(key="preprocess_logreg_outlier_mad_threshold").set_value(MAD_K)
    at.run()
    _click(at, key="preprocess_build_button")
    _ok(at, "05 female build")
    out["female_config"] = dict(at.session_state["preprocessing_config_by_model"]["logreg"])
    out["female_built"] = list(at.session_state["preprocess_built_model_keys"])
    state = _state_of(at)

    # ── Female: Train & Compare, then the button ───────────────────────
    at = _open("pages/06_Train_and_Compare.py", state)
    at.run()
    _ok(at, "06 female first render")
    _click(at, label="Prepare Splits")           # the model picker renders past the splits gate
    _ok(at, "06 female splits")
    out["female_pick_shown"] = at.checkbox(key="train_model_logreg").value
    at.number_input(key="logreg_C").set_value(LOGREG_C)
    at.run()
    _ok(at, "06 female hyperparameter")
    _click(at, key="train_models_button")
    _ok(at, "06 female train")
    out["female_trained"] = sorted(at.session_state["trained_models"].keys())
    out["next_button_present"] = bool(at.button(key="cohort_next_btn"))
    out["warnings_before_switch"] = [str(w.value) for w in at.warning]
    _click(at, key="cohort_next_btn")
    _ok(at, "06 switch")
    out["gate_warnings"] = [str(w.value) for w in at.warning]
    out["after_switch"] = {
        "cohort": dict(at.session_state["cohort_run"]),
        "config_by_model": dict(at.session_state["preprocessing_config_by_model"] or {}),
        "built": list(at.session_state["preprocess_built_model_keys"] or [])
        if "preprocess_built_model_keys" in _state_of(at) else [],
        # Deep-copied: the pages that claim these decisions mutate the parked
        # dict in place, and the state is carried between AppTests by reference.
        "pending": copy.deepcopy(at.session_state["cohort_decisions_pending"]),
        "trained": dict(at.session_state["trained_models"] or {}),
    }
    out["sidebar_after_switch"] = [str(c.value) for c in at.sidebar.caption]
    state = _state_of(at)

    # ── Male: Preprocess shows the carried settings, rebuilds from them ─
    at = _open("pages/05_Preprocess.py", state)
    at.run()
    _ok(at, "05 male first render")
    out["male_mode"] = at.radio(key="preprocess_config_mode").value
    out["male_widgets"] = {
        "numeric_scaling": at.selectbox(key="preprocess_logreg_numeric_scaling").value,
        "numeric_imputation": at.selectbox(key="preprocess_logreg_numeric_imputation_display").value,
        "numeric_outlier_treatment": at.selectbox(key="preprocess_logreg_numeric_outlier_treatment").value,
        "numeric_power_transform": at.selectbox(key="preprocess_logreg_numeric_power_transform").value,
        "mad_k": at.number_input(key="preprocess_logreg_outlier_mad_threshold").value,
    }
    out["male_infos"] = [str(i.value) for i in at.info]
    _click(at, key="preprocess_build_button")
    _ok(at, "05 male build")
    out["male_config"] = dict(at.session_state["preprocessing_config_by_model"]["logreg"])
    out["male_built"] = list(at.session_state["preprocess_built_model_keys"])
    state = _state_of(at)

    # ── Male: Train & Compare has the pick and the hyperparameter ──────
    at = _open("pages/06_Train_and_Compare.py", state)
    at.run()
    _ok(at, "06 male render")
    out["male_pending_before_splits"] = "cohort_decisions_pending" in _state_of(at)
    _click(at, label="Prepare Splits")
    _ok(at, "06 male splits")
    out["male_pick_shown"] = at.checkbox(key="train_model_logreg").value
    out["male_C"] = at.number_input(key="logreg_C").value
    out["male_captions"] = [str(c.value) for c in at.caption]
    out["male_markdown"] = " ".join(str(m.value) for m in at.markdown)
    out["male_pending_left"] = "cohort_decisions_pending" in _state_of(at)
    # The sidebar renders at the top of a run and the claim happens further
    # down it, so the sidebar reports the claim one rerun later — the next
    # widget interaction, in practice.
    at.run()
    _ok(at, "06 male rerun")
    out["male_sidebar"] = [str(c.value) for c in at.sidebar.caption]
    out["male_C_after_rerun"] = at.number_input(key="logreg_C").value
    return out


class TestTheFemaleRunWasWhatWeThink:

    def test_the_hand_configuration_was_built(self, journey):
        cfg = journey["female_config"]
        for k, v in DECIDED.items():
            assert cfg[k] == v, (k, cfg)
        assert cfg["numeric_outlier_params"]["threshold"] == MAD_K
        assert journey["female_built"] == ["logreg"]

    def test_the_pick_reached_train_and_compare_and_a_model_was_fitted(self, journey):
        assert journey["female_pick_shown"] is True
        assert journey["female_trained"] == ["logreg"]
        assert journey["next_button_present"]


class TestTheSwitchParksEveryDecisionAndFitsNothing:

    def test_the_people_changed(self, journey):
        assert journey["after_switch"]["cohort"]["label"] == "Male"

    def test_nothing_fitted_on_the_women_survived(self, journey):
        a = journey["after_switch"]
        assert a["config_by_model"] == {}, "a config with no pipeline behind it"
        assert a["built"] == []
        assert a["trained"] == {}

    def test_the_decisions_are_waiting_with_their_origin(self, journey):
        p = journey["after_switch"]["pending"]
        assert p["from_label"] == "Female"
        cfg = p["preprocess"]["config_by_model"]["logreg"]
        for k, v in DECIDED.items():
            assert cfg[k] == v
        assert p["models"]["picks"] == ["logreg"]
        assert p["models"]["hyperparams"]["logreg"]["logreg_C"] == LOGREG_C

    def test_the_gate_says_where_the_settings_went(self, journey):
        gate = " | ".join(journey["gate_warnings"])
        assert "from the Female run are waiting on the **Preprocess** page" in gate, gate
        assert "were reset" not in gate

    def test_the_sidebar_says_so_too(self, journey):
        side = " | ".join(journey["sidebar_after_switch"])
        assert "Carried from the Female run" in side, side
        assert "preprocessing settings for LOGREG" in side


class TestTheMenGetTheSamePipelineSpecification:

    def test_preprocess_opens_in_advanced_mode_showing_the_womens_choices(self, journey):
        assert journey["male_mode"] == R.ADVANCED_MODE_LABEL
        w = journey["male_widgets"]
        assert w["numeric_scaling"] == DECIDED["numeric_scaling"]
        assert w["numeric_imputation"] == DECIDED["numeric_imputation"]
        assert w["numeric_outlier_treatment"] == DECIDED["numeric_outlier_treatment"]
        assert w["numeric_power_transform"] == DECIDED["numeric_power_transform"]
        assert w["mad_k"] == MAD_K

    def test_the_page_says_what_it_carried_and_that_the_fit_is_new(self, journey):
        infos = " | ".join(journey["male_infos"])
        assert "Carried from the Female run" in infos, infos
        assert "preprocessing settings for LOGREG" in infos
        assert "refit" in infos

    def test_the_rebuilt_specification_equals_the_first(self, journey):
        a = _decision_keys(journey["female_config"])
        b = _decision_keys(journey["male_config"])
        assert b == a, {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert journey["male_built"] == ["logreg"], (
            "the coach must not have added its own picks after the reset")

    def test_train_and_compare_has_the_pick_and_the_hyperparameter(self, journey):
        assert journey["male_pick_shown"] is True
        assert journey["male_C"] == LOGREG_C
        assert "Tuned for this model" in journey["male_markdown"]
        caps = " | ".join(journey["male_captions"])
        assert "Carried from the Female run" in caps and "hyperparameter settings for LOGREG" in caps
        assert "tuned by Optuna on the previous group were not carried" in caps

    def test_nothing_is_left_waiting_and_the_sidebar_is_quiet(self, journey):
        assert journey["male_pending_before_splits"] is True, (
            "claimed before the splits gate, the values would die with the run that stops there")
        assert journey["male_pending_left"] is False
        assert not any("Carried from" in c for c in journey["male_sidebar"])
        assert journey["male_C_after_rerun"] == LOGREG_C, "the value is the widget's now, not a seed"

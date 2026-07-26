"""Decisions replay onto the next group; fits do not.

"Now run the same analysis on Male" carries an assumption: the same analysis.
The switch used to clear feature engineering and preprocessing outright, which
is the safe half of the rule and only half of it. Clearing the FITS is right —
a scaler's mean, an imputer's median and a PCA's components are numbers learned
from the previous group, and reusing them leaks one group into the other's
results. Clearing the DECISIONS is not: "bmi_squared = bmi²" means the same
thing for anyone, and a run that quietly lost it is not answering the same
question as the run it is compared against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils import replay as R
from utils.cohorts import plan_cohorts, start_cohort
from utils.session_state import DataConfig, get_data, set_data
from utils.test_lockbox import ensure_lockbox, train_row_mask


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _study(n=600):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"sex": rng.choice(["Female", "Male"], n),
                         "age": rng.integers(20, 80, n),
                         "bmi": np.round(rng.normal(28, 5, n), 1),
                         "chol": np.round(rng.normal(200, 30, n), 1),
                         "y": rng.integers(0, 2, n)})


def _in_female(df):
    set_data(df)
    st.session_state["data_config"] = DataConfig(
        target_col="y", feature_cols=["age", "bmi", "chol"],
        task_type="classification")
    ensure_lockbox(df, "y", "classification")
    plan = plan_cohorts(df, "sex", "y", "classification",
                        train_mask=train_row_mask(df.index))
    start_cohort(df, plan, next(c for c in plan.viable if c.label == "Female"), "y")
    return plan


class TestPureFormulasReplayExactly:

    def test_a_ratio_is_recomputed_from_the_new_rows(self):
        df = _study()
        _in_female(df)
        R.record("ratio", {"pairs": [["chol", "bmi"]]}, ["chol_div_bmi"])
        male = df[df["sex"] == "Male"]
        out, made, missed = R.replay_onto(male, R.recipe())
        assert made == ["chol_div_bmi"] and not missed
        expected = male["chol"] / male["bmi"]
        pd.testing.assert_series_equal(out["chol_div_bmi"], expected,
                                       check_names=False)

    def test_an_interaction_replays(self):
        df = _study()
        R.record("interaction", {"left": "bmi", "right": "chol", "op": "*"},
                 ["bmi_x_chol"])
        out, made, _ = R.replay_onto(df, R.recipe())
        assert made == ["bmi_x_chol"]
        pd.testing.assert_series_equal(out["bmi_x_chol"], df["bmi"] * df["chol"],
                                       check_names=False)

    def test_a_square_replays(self):
        df = _study()
        R.record("interaction", {"left": "bmi", "right": None, "op": "square"},
                 ["bmi_squared"])
        out, made, _ = R.replay_onto(df, R.recipe())
        assert made == ["bmi_squared"]
        assert np.allclose(out["bmi_squared"], df["bmi"] ** 2)

    def test_polynomial_terms_replay(self):
        df = _study()
        R.record("polynomial",
                 {"columns": ["age", "bmi"], "degree": 2, "interaction_only": False},
                 ["age^2", "age bmi", "bmi^2"])
        out, made, missed = R.replay_onto(df, R.recipe())
        assert not missed and len(made) == 3
        assert np.allclose(out["bmi^2"], df["bmi"].astype(float) ** 2)


class TestFittedStepsAreRefitOnTheNewGroupsTrainingRows:

    def test_pca_is_refit_not_reused(self):
        df = _study()
        _in_female(df)
        R.record("pca", {"columns": ["age", "bmi", "chol"], "n_components": 2,
                         "seed": 42}, ["PCA_1", "PCA_2"], R.REFIT)
        female = get_data()
        male = df[df["sex"] == "Male"]
        f_out, _, _ = R.replay_onto(female, R.recipe(), train_row_mask(female.index))
        m_out, made, _ = R.replay_onto(male, R.recipe(), train_row_mask(male.index))
        assert made == ["PCA_1", "PCA_2"]
        # A reused fit would give the two groups the same component loadings and
        # so identical statistics on overlapping input; a refit does not.
        assert abs(float(f_out["PCA_1"].std()) - float(m_out["PCA_1"].std())) > 1e-9

    def test_the_sealed_rows_do_not_influence_the_refit(self):
        """A replay must not open the lockbox any more than the original did."""
        df = _study()
        _in_female(df)
        male = df[df["sex"] == "Male"]
        mask = train_row_mask(male.index)
        assert 0 < int(mask.sum()) < len(male)
        R.record("pca", {"columns": ["age", "bmi", "chol"], "n_components": 2,
                         "seed": 42}, ["PCA_1", "PCA_2"], R.REFIT)
        with_sealed, _, _ = R.replay_onto(male, R.recipe(),
                                          pd.Series(True, index=male.index))
        train_only, _, _ = R.replay_onto(male, R.recipe(), mask)
        assert not np.allclose(with_sealed["PCA_1"], train_only["PCA_1"]), (
            "the refit gave the same answer with and without the sealed rows, "
            "which means the mask is not being honored")


class TestNothingIsLostInSilence:

    def test_an_unreplayable_step_is_named(self):
        df = _study()
        R.record("umap", {"n_components": 2}, ["UMAP_1", "UMAP_2"], R.MANUAL)
        _, made, missed = R.replay_onto(df, R.recipe())
        assert not made and len(missed) == 1 and "UMAP" in missed[0]

    def test_the_summary_says_the_runs_diverged(self):
        text = R.replay_summary(["a", "b"], ["UMAP (2 components) → 2 features"])
        assert "not answering quite the same question" in text

    def test_a_ratio_whose_denominator_has_zeros_here_is_reported(self):
        df = _study()
        df.loc[df.index[:5], "bmi"] = 0
        R.record("ratio", {"pairs": [["chol", "bmi"]]}, ["chol_div_bmi"])
        _, made, missed = R.replay_onto(df, R.recipe())
        assert not made and missed and "zeros" in missed[0]


class TestTheSwitchCarriesTheDecisions:

    def test_staging_survives_the_reset_and_restores(self):
        from utils.session_state import reset_downstream_results
        df = _study()
        _in_female(df)
        R.record("ratio", {"pairs": [["chol", "bmi"]]}, ["chol_div_bmi"])
        st.session_state["preprocess_logreg_numeric_scaling"] = "robust"
        R.stage_for_replay("cohort switch")
        reset_downstream_results(clear_feature_engineering=True)
        restored = R.restore_decisions()
        assert R.recipe() and restored
        assert st.session_state["preprocess_logreg_numeric_scaling"] == "robust"

    def test_the_button_text_matches_what_happens(self):
        import inspect
        import utils.cohort_ui as cu
        src = inspect.getsource(cu.render_next_cohort)
        assert "rebuilt from their formulas" in src
        assert "refit on" in src

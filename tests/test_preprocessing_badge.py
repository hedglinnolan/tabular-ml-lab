"""What Train & Compare says about preprocessing must match what it does.

Preprocess is where you declare the models you will train: pick them there,
build, and each gets a pipeline tuned to it, arriving pre-selected on Train &
Compare. Picking none is also legal and builds one shared pipeline.

Training then resolves `get_preprocessing_pipeline(model_key) or pipeline`, and
that fallback is where the danger is. It is NOT generic: set_preprocessing_pipelines
takes the 'default' entry if one exists, and otherwise the FIRST prepared
model's pipeline. So adding a model on Train & Compare that was never prepared
trains it with another model's preprocessing — if that model had PCA enabled,
this one silently gets PCA too, and Explainability then describes PC1 and PC2
for a model the researcher believes saw raw predictors.

A single "✅ Preprocessed / ⚠️ No pipeline" badge could not tell those apart. It
called a perfectly good shared pipeline a problem, and called borrowing another
model's transform the same thing as having nothing. These tests pin all four
states to what training actually resolves.
"""
from __future__ import annotations

import re

import pytest
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.session_state import (
    get_preprocessing_pipeline, set_preprocessing_pipelines,
)

PAGE = "pages/06_Train_and_Compare.py"


@pytest.fixture(autouse=True)
def clean_state():
    for key in ("preprocessing_pipeline", "preprocessing_pipelines_by_model",
                "preprocessing_config", "preprocessing_config_by_model",
                "preprocess_built_model_keys"):
        st.session_state.pop(key, None)
    yield
    for key in ("preprocessing_pipeline", "preprocessing_pipelines_by_model",
                "preprocessing_config", "preprocessing_config_by_model",
                "preprocess_built_model_keys"):
        st.session_state.pop(key, None)


def a_pipeline() -> Pipeline:
    return Pipeline([("scale", StandardScaler())])


class TestSharedPipelineIsRealPreprocessing:
    """The state Preprocess actually leaves behind on a first pass."""

    def test_first_pass_builds_only_a_shared_pipeline(self):
        """No models ticked yet -> one 'default' pipeline, empty per-model list."""
        built = {"default": a_pipeline()}
        set_preprocessing_pipelines(built, {"default": {}}, {})
        per_model = [k for k in built if k != "default"]
        st.session_state["preprocess_built_model_keys"] = per_model

        assert per_model == []
        assert st.session_state["preprocessing_pipeline"] is not None

    def test_every_model_still_resolves_to_that_pipeline(self):
        """Which is why "No pipeline" was false: training finds one."""
        set_preprocessing_pipelines({"default": a_pipeline()}, {"default": {}}, {})
        st.session_state["preprocess_built_model_keys"] = []
        shared = st.session_state["preprocessing_pipeline"]
        for model_key in ("logreg", "rf", "xgb", "nn"):
            resolved = get_preprocessing_pipeline(model_key) or shared
            assert resolved is shared

    def test_per_model_pipeline_wins_when_one_exists(self):
        tuned = a_pipeline()
        set_preprocessing_pipelines(
            {"default": a_pipeline(), "rf": tuned},
            {"default": {}, "rf": {}}, {})
        assert get_preprocessing_pipeline("rf") is tuned
        assert get_preprocessing_pipeline("logreg") is not tuned


class TestBorrowedPipelineIsDisclosed:
    """Selecting a model here that Preprocess never prepared.

    Supported on purpose — but what it silently inherits has to be said.
    """

    @staticmethod
    def prepared_ridge_with_pca_and_plain_rf():
        ridge = Pipeline([("scale", StandardScaler()),
                          ("pca", PCA(n_components=2))])
        rf = Pipeline([("scale", StandardScaler())])
        set_preprocessing_pipelines({"ridge": ridge, "rf": rf},
                                    {"ridge": {}, "rf": {}}, {})
        st.session_state["preprocess_built_model_keys"] = ["ridge", "rf"]
        return ridge, rf

    def test_unprepared_model_inherits_the_first_models_pipeline(self):
        ridge, _ = self.prepared_ridge_with_pca_and_plain_rf()
        fallback = st.session_state["preprocessing_pipeline"]
        assert fallback is ridge
        resolved = get_preprocessing_pipeline("xgb_clf") or fallback
        assert resolved is ridge

    def test_and_therefore_inherits_its_dimensionality_reduction(self):
        """The harm: SHAP would explain PC1/PC2 for a model never given PCA."""
        self.prepared_ridge_with_pca_and_plain_rf()
        fallback = st.session_state["preprocessing_pipeline"]
        resolved = get_preprocessing_pipeline("xgb_clf") or fallback
        assert "pca" in dict(resolved.steps)

    def test_no_generic_default_exists_to_fall_back_on(self):
        """Which is why the fallback is another model's and not a neutral one."""
        self.prepared_ridge_with_pca_and_plain_rf()
        by_model = st.session_state["preprocessing_pipelines_by_model"]
        assert "default" not in by_model
        assert next(iter(by_model)) == "ridge"


class TestBadgeMatchesResolution:
    """The four states the card can honestly be in."""

    @staticmethod
    def badge(model_key: str) -> str:
        """The page's own rule, kept in step with it by the source check below."""
        per_model = st.session_state.get("preprocess_built_model_keys", [])
        by_model = st.session_state.get("preprocessing_pipelines_by_model", {}) or {}
        if model_key in per_model:
            return "tuned"
        if "default" in by_model:
            return "shared"
        if by_model:
            return f"borrows:{next(iter(by_model))}"
        return "none"

    def test_shared_pipeline_is_not_reported_as_missing(self):
        set_preprocessing_pipelines({"default": a_pipeline()}, {"default": {}}, {})
        st.session_state["preprocess_built_model_keys"] = []
        assert self.badge("logreg") == "shared"

    def test_tuned_pipeline_is_named_as_such(self):
        set_preprocessing_pipelines(
            {"default": a_pipeline(), "rf": a_pipeline()},
            {"default": {}, "rf": {}}, {})
        st.session_state["preprocess_built_model_keys"] = ["rf"]
        assert self.badge("rf") == "tuned"
        assert self.badge("logreg") == "shared"

    def test_borrowing_is_not_reported_as_tuned_or_as_nothing(self):
        set_preprocessing_pipelines(
            {"ridge": a_pipeline(), "rf": a_pipeline()},
            {"ridge": {}, "rf": {}}, {})
        st.session_state["preprocess_built_model_keys"] = ["ridge", "rf"]
        assert self.badge("ridge") == "tuned"
        assert self.badge("xgb_clf") == "borrows:ridge"

    def test_nothing_built_is_still_a_warning(self):
        """The one case where the original message was right."""
        assert self.badge("logreg") == "none"

    def test_page_source_still_implements_these_four_states(self):
        """A guard against the page quietly reverting to the per-model-only check.

        Cheap, but it is what would have caught the original defect: the page
        must consult the shared pipeline, not only `_prep_built`.
        """
        source = open(PAGE, encoding="utf-8").read()
        block = source[source.index('if model_key in _prep_built'):]
        block = block[:block.index("border =")]
        assert "Tuned for this model" in block
        assert "Shared pipeline" in block
        assert "No pipeline" in block
        assert "_prep_fallback_owner" in block, (
            "borrowing another model's pipeline must be named, not hidden")
        assert re.search(r"elif\s+_prep_is_generic", block), (
            "a genuinely shared pipeline must not be reported as a problem")

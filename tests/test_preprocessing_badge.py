"""What Train & Compare says about preprocessing must match what it does.

The defect this file pins down was a loop with no exit. Preprocess can only
tune a pipeline PER MODEL for models already ticked on Train & Compare, but the
recommended order puts Preprocess first — so on a first pass nothing is ticked,
it builds one shared pipeline, and `preprocess_built_model_keys` comes back
empty. Train & Compare read that list alone and stamped "⚠️ No pipeline" on
every model card, seconds after Preprocess reported success. The researcher
goes back to Preprocess, presses Build again, gets the same shared pipeline,
and is told again that there is no pipeline.

Underneath, training was fine the whole time: it resolves
`get_preprocessing_pipeline(model_key) or pipeline`, so the shared pipeline was
being applied. The badge was the only thing that was wrong, which is the worst
version of this bug — the app was lying about work it had correctly done.
"""
from __future__ import annotations

import re

import pytest
import streamlit as st
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


class TestBadgeMatchesResolution:
    """The three states the card can honestly be in."""

    @staticmethod
    def badge(model_key: str) -> str:
        """The page's own rule, kept in step with it by the source check below."""
        per_model = st.session_state.get("preprocess_built_model_keys", [])
        shared = st.session_state.get("preprocessing_pipeline")
        if model_key in per_model:
            return "tuned"
        if shared is not None:
            return "shared"
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

    def test_nothing_built_is_still_a_warning(self):
        """The one case where the original message was right."""
        assert self.badge("logreg") == "none"

    def test_page_source_still_implements_these_three_states(self):
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
        assert re.search(r"elif\s+pipeline\s+is\s+not\s+None", block), (
            "the badge must fall back to the shared pipeline")

"""An exported number must say which participants it is about.

`CONTRACT-017` made `dataset_profile` training-scope on page 02: the profile
drives the model coach, and a coach that has seen the held-out people is
choosing models with them. That fixed a leak and created a disclosure question,
because page 10 copies the profile's `p_n_ratio`, `total_missing_rate` and data
sufficiency straight into the exported metadata and into the Methods section.

The number changed and the manuscript did not. An exported figure whose
population is unstated is read as being about the whole study, which makes it a
live assertion of something false in a document the author is about to submit —
not a convergence item.

Worse, the missing-data sentence mixes two populations: the feature *count*
comes from the training-scope profile and the per-feature *rates* come from
`data_audit` over the whole frame. Both are now stated.

Findings: CONTRACT-017 (the scope), and the export half raised out of L11.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from ml.publication import generate_methods_section


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def scope_fields():
    """`pages/10`'s helper, imported without executing the page."""
    import ast
    import os
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "pages", "10_Report_Export.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_profile_scope_fields")
    module = types.ModuleType("_scope_probe")
    module.st = st
    module.__dict__["Dict"] = dict
    module.__dict__["Any"] = object
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"),
         module.__dict__)
    return module._profile_scope_fields


# ── the helper says what page 02 recorded ────────────────────────────────

def test_a_training_scoped_profile_is_labeled_with_its_n():
    st.session_state["dataset_profile_scope"] = {
        "rows": "training", "n_rows": 168, "n_rows_total": 200,
        "reason": "held-out test rows are excluded to prevent selection leakage",
    }
    fields = scope_fields()()
    assert fields["row_scope"] == "training"
    assert fields["row_scope_n"] == 168
    assert "training rows only, n=168" in fields["row_scope_note"]
    assert "200" in fields["row_scope_note"], (
        "the note does not say what the training rows are a subset of")


def test_an_unsealed_analysis_says_all_rows():
    st.session_state["dataset_profile_scope"] = {
        "rows": "all", "n_rows": 200, "n_rows_total": 200,
        "reason": "no rows are sealed in this analysis",
    }
    fields = scope_fields()()
    assert fields["row_scope"] == "all"
    assert "all rows, n=200" in fields["row_scope_note"]


def test_an_unrecorded_scope_is_unknown_rather_than_assumed():
    """Page 02 may not have run. Guessing 'all' would be the original defect."""
    fields = scope_fields()()
    assert fields["row_scope"] == "unknown"
    assert fields["row_scope_n"] is None
    assert "not recorded" in fields["row_scope_note"]


# ── page 02 records it ───────────────────────────────────────────────────

def test_page_02_records_the_scope_beside_the_profile():
    import ast
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pages", "02_EDA.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    written = {
        t.slice.value
        for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
        and isinstance(t.slice.value, str)
        and "session_state" in ast.dump(t.value)
    }
    assert "dataset_profile" in written
    assert "dataset_profile_scope" in written, (
        "page 02 masks the profile to training rows and does not record that it "
        "did, so nothing downstream can state the population")


# ── the Methods sentence states both populations ─────────────────────────

def methods(summary):
    return generate_methods_section(
        data_config={"feature_cols": ["a", "b"], "target_col": "y",
                     "task_type": "classification"},
        preprocessing_config={},
        model_configs={"ridge": {}},
        split_config={"train_size": 0.7, "val_size": 0.15, "test_size": 0.15},
        n_total=200, n_train=140, n_val=30, n_test=30,
        feature_names=["a", "b"], target_name="y",
        task_type="classification", metrics_used=["auc"],
        missing_data_summary=summary,
    )


def test_the_missing_data_sentence_names_the_training_population():
    text = methods({
        "n_features_with_missing": 2, "total_features": 7,
        "min_missing_rate": 0.05, "max_missing_rate": 0.31,
        "row_scope": "training", "row_scope_n": 168,
        "rates_row_scope": "all", "rates_n_rows": 200,
    })
    assert "2 of 7 features had missing values" in text
    assert "counted on the 168 training rows" in text, (
        "the feature count is a training-set number presented as a study number")
    assert "over all 200 records" in text, (
        "the rates are whole-frame numbers presented beside a training-set count "
        "with nothing distinguishing them")


def test_an_unsealed_analysis_adds_no_qualifier():
    """The fix must not clutter a study where the whole frame was profiled."""
    text = methods({
        "n_features_with_missing": 2, "total_features": 7,
        "min_missing_rate": 0.05, "max_missing_rate": 0.31,
        "row_scope": "all", "row_scope_n": 200,
    })
    assert "2 of 7 features had missing values (missing rates" in text
    assert "training rows" not in text


def test_a_summary_with_no_scope_still_renders():
    """Older saved sessions carry no scope. They must not break the export."""
    text = methods({"n_features_with_missing": 2, "total_features": 7})
    assert "2 of 7 features had missing values." in text

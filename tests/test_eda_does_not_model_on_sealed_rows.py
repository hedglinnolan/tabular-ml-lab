"""Explore must not fit models on, or profile, the rows the lockbox sealed.

`utils/test_lockbox.py`'s module docstring states the contract: "Every
target-aware step upstream of Train & Compare — feature-engineering fits,
feature selection, target-association views — operates on training rows only,
via train_row_mask()." `pages/04_Feature_Selection.py` obeys it and says so on
screen. `pages/02_EDA.py` never called it: `df = get_data()`, unfiltered, into
everything.

This closes the two paths where that costs something real:

  - `dataset_profile`, which drives the model coach's picks. A profile computed
    on held-out people lets the sealed test set help choose the models.
  - `quick_probe_baselines`, which runs its *own* 80/20 split and fits
    constant / GLM / shallow-RF models — a modeling step wearing an EDA
    costume, reporting a score on rows it was shown.

Finding: CONTRACT-017 (PARTIAL — the display analyses named in its note still
see every row, and close at L11 convergence of pages/02).

The page's module body cannot be imported outside Streamlit, so the structural
half is read with `ast`: the page must hand these two paths the masked frame.
The behavioral half is asserted against the engine functions the page calls,
and end-to-end through AppTest in
`tests/integration/test_eda_profile_is_scoped_to_training_rows.py`.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from ml.dataset_profile import compute_dataset_profile
from ml.eda_actions import quick_probe_baselines
from utils.test_lockbox import ensure_lockbox, train_row_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "pages", "02_EDA.py")

# The frame the page must NOT hand to a modeling path.
FULL_FRAME = "df"
# The masked frame, and the helper that chooses between them.
MASKED_FRAME = "_train_df"
CHOOSER = "_frame_for_action"


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def page_tree() -> ast.Module:
    with open(PAGE, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=PAGE)


def study(n=240):
    rng = np.random.default_rng(11)
    age = rng.integers(20, 80, n).astype(float)
    bmi = rng.normal(27, 4, n)
    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "glucose": 60 + 0.6 * age + 1.4 * bmi + rng.normal(0, 6, n),
    })


# ── the page calls train_row_mask at all ─────────────────────────────────

def test_the_page_masks_the_sealed_rows():
    tree = page_tree()
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "train_row_mask" in called, (
        "pages/02 does not call train_row_mask; the lockbox contract names "
        "this page's target-aware steps explicitly")


# ── the profile that drives the coach ────────────────────────────────────

def test_the_dataset_profile_is_computed_on_the_masked_frame():
    tree = page_tree()
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_compute_profile"
    ]
    assert calls, "pages/02 no longer computes a dataset profile"
    for call in calls:
        frame = call.args[0]
        assert isinstance(frame, ast.Name) and frame.id == MASKED_FRAME, (
            f"the dataset profile is computed on {ast.dump(frame)} rather than "
            f"{MASKED_FRAME}; it drives the model coach, so the sealed test set "
            "would help choose the models")


def test_the_profile_cache_key_follows_the_masked_frame():
    """A profile keyed on the full frame serves the leaky answer to the fix."""
    tree = page_tree()
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_compute_profile"
    ]
    for call in calls:
        data_id = next((kw.value for kw in call.keywords if kw.arg == "data_id"),
                       None)
        assert isinstance(data_id, ast.Name) and data_id.id == "_train_fingerprint", (
            "the profile cache key does not follow the frame it profiles; "
            "reusing the full-frame fingerprint puts the leak straight back")


# ── the action that fits models ──────────────────────────────────────────

def test_quick_probe_baselines_is_registered_as_a_modeling_action():
    tree = page_tree()
    assigns = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_TRAIN_ONLY_ACTIONS"
                for t in node.targets)
    ]
    assert assigns, "pages/02 has no _TRAIN_ONLY_ACTIONS roster"
    named = {elt.value for elt in ast.walk(assigns[0].value)
             if isinstance(elt, ast.Constant)}
    assert "quick_probe_baselines" in named, (
        "quick_probe_baselines runs its own 80/20 split and fits models; it "
        "must be scoped to training rows")


def test_no_eda_action_is_run_on_the_unmasked_frame():
    """Every `action_func(...)` call routes its frame through the chooser."""
    tree = page_tree()
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "action_func" and node.args
    ]
    assert calls, "pages/02 no longer dispatches EDA actions"

    offenders = []
    for call in calls:
        frame = call.args[0]
        routed = (
            isinstance(frame, ast.Call) and isinstance(frame.func, ast.Name)
            and frame.func.id == CHOOSER
        ) or (
            isinstance(frame, ast.Name) and frame.id == "_action_df"
        )
        if not routed:
            offenders.append(ast.dump(frame))
    assert not offenders, (
        f"EDA actions are dispatched on an unrouted frame: {offenders}. Every "
        f"call site must go through {CHOOSER}, or a modeling action added to "
        "the roster will still be handed every row")


# ── what the mask actually buys, against the real engine functions ───────

def test_a_probe_on_the_masked_frame_never_touches_a_sealed_row():
    df = study()
    st.session_state["raw_data"] = df
    lockbox = ensure_lockbox(df, "glucose", "regression")
    sealed = set(lockbox["labels"])
    assert sealed, "the fixture sealed nothing; the test would be vacuous"

    mask = train_row_mask(df.index)
    train_df = df.loc[mask]

    assert not (set(train_df.index) & sealed), (
        "the masked frame still contains sealed rows")
    assert len(train_df) == len(df) - len(sealed)

    class Signals:
        task_type_final = "regression"

    result = quick_probe_baselines(train_df, "glucose", ["age", "bmi"],
                                   Signals(), st.session_state)
    assert result.get("figures") or result.get("findings"), (
        "the probe produced nothing on the masked frame")


def test_the_profile_of_the_masked_frame_describes_the_training_rows():
    df = study()
    st.session_state["raw_data"] = df
    ensure_lockbox(df, "glucose", "regression")

    mask = train_row_mask(df.index)
    train_df = df.loc[mask]

    features = ["age", "bmi"]
    scoped = compute_dataset_profile(train_df, "glucose", features, "regression")
    full = compute_dataset_profile(df, "glucose", features, "regression")

    assert scoped.n_rows == int(mask.sum())
    assert scoped.n_rows < full.n_rows, (
        "the scoped profile counts every row; nothing was quarantined")

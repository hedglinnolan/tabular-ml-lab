"""A saved one-group analysis must come back as a one-group analysis.

session_manager is a strict whitelist — "anything not listed is dropped on save
and ignored on load" — and neither the active cohort run nor the banked
comparison runs were listed. So saving mid-run and restoring produced a session
covering the WHOLE study, with no indication that a filter had been dropped and
the previous group's result gone. The test-set lockbox got explicit schema-2.1
handling for exactly this reason; the run that decides WHO the analysis is
about never did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_session_manager import fake_session  # noqa: F401
from utils import session_manager
from utils.cohorts import CohortRun


@pytest.fixture
def state(fake_session):
    return fake_session


def _study(n=80):
    return pd.DataFrame({"sex": ["F", "M"] * (n // 2),
                         "y": [0, 1] * (n // 2),
                         "age": list(range(n))})


def _seed_state(state):
    df = _study()
    state["raw_data"] = df
    state["cohort_run"] = {
        "column": "sex", "value": "F", "label": "F",
        "labels": [int(i) for i in df.index[df["sex"] == "F"]],
        "n_rows": 40, "n_total": 80, "position": 1, "of": 2,
        "order": ["F", "M"], "target_col": "y", "dropped_features": ["sex"],
    }
    state["cohort_runs_done"] = [CohortRun(
        column="sex", label="F", n_train=34, n_test=6, completed=True,
        metrics={"ROC-AUC": 0.71}, target_col="y", task_type="classification",
        data_fingerprint="fp-1")]
    return df


def _round_trip(state, df):
    blob, _ = session_manager._collect_session_data()
    for k in ("cohort_run", "cohort_runs_done", "raw_data",
              "_raw_data_fingerprint"):
        state.pop(k, None)
    session_manager._restore_session_data(blob)


def test_the_active_run_comes_back(state):
    df = _seed_state(state)
    _round_trip(state, df)
    run = state.get("cohort_run")
    assert run, "the restored session silently covered the whole study"
    assert run["column"] == "sex" and run["label"] == "F"
    assert len(run["labels"]) == 40 and run["n_total"] == 80


def test_the_banked_comparison_comes_back(state):
    df = _seed_state(state)
    _round_trip(state, df)
    runs = state.get("cohort_runs_done") or []
    assert len(runs) == 1
    assert runs[0].label == "F" and runs[0].metrics["ROC-AUC"] == 0.71
    assert runs[0].target_col == "y", "the question it answered came back too"


def test_the_data_fingerprint_is_rebuilt(state):
    """Without it every restored run reads as belonging to other data."""
    df = _seed_state(state)
    _round_trip(state, df)
    assert state.get("_raw_data_fingerprint"), "runs would be filtered out"


def test_a_session_with_no_run_restores_unchanged(state):
    state["raw_data"] = _study()
    blob, _ = session_manager._collect_session_data()
    state.pop("raw_data", None)
    session_manager._restore_session_data(blob)
    assert state.get("cohort_run") is None
    assert state.get("raw_data") is not None

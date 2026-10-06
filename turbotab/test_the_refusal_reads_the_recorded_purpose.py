"""`AUDIT-005` — the refusal that claimed too much now reads the purpose.

`missingness.py` said the outcome cannot be in the imputation model and that
**"there is no configuration in which this is acceptable, so it is not offered
as a choice."** `research/CLINICAL_SURVEY_PACK.md` §A2 marks the opposite
**[SETTLED]**:

> *Imputing with the outcome EXCLUDED from the imputation model. Biases
> associations toward the null. The outcome MUST be in the imputation model.*

**Both are right, about different purposes.** Under prediction the imputer
writes the outcome into features the app will not have at deployment. Under
inference the imputation model is part of the estimation, and omitting the
outcome makes the imputed covariates conditionally independent of it — the
association shrinks toward the null.

`DOMAIN_SCIENCE.md` §01.3 is the frame: the prediction/inference fork is one of
the seven convergences, *the advice inverts*, and it names five decisions where
it does. This is one of the five.

**The defect was the absolute, not the refusal.** The app records the purpose
and this module never read it, so it asserted a universal it had the information
to qualify — which makes it an `AUDIT-008` instance as well as a governing-rule
one. The prediction branch is unchanged and is still a blocker.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import missingness as M, purpose as P                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "glucose": rng.normal(120, 25, n),
        "hba1c": rng.normal(7, 1.2, n),
    })
    frame.loc[frame.index[:20], "glucose"] = np.nan
    return frame


# ── the two branches ────────────────────────────────────────────────────────

def test_under_prediction_the_outcome_in_scope_is_still_a_blocker():
    """Unchanged, and it has to be: at deployment the leak has nowhere to come
    from, so the features would carry information the app will not have."""
    reading = M.outcome_in_scope("hba1c", P.PREDICTION)
    assert reading["refuse"] is True
    assert "not offered as a choice here" in reading["message"]
    assert "PREDICTING an outcome for a new person" in reading["message"]


def test_under_inference_the_same_configuration_is_correct():
    """Not a softer refusal — not a refusal. Under an association objective the
    thing being blocked is the thing the literature requires."""
    reading = M.outcome_in_scope("hba1c", P.INFERENCE)
    assert reading["refuse"] is False
    assert "correct rather than a leak" in reading["message"]
    assert "biases the association toward the null" in reading["message"]
    assert reading["source"].startswith("research/CLINICAL_SURVEY_PACK.md#A2")


def test_each_branch_names_the_other_so_the_reader_can_check_the_fork():
    """§01.3's point is that the advice inverts. A message that stated only its
    own branch would read as a universal again, which is the defect."""
    predicting = M.outcome_in_scope("y", P.PREDICTION)["message"]
    inferring = M.outcome_in_scope("y", P.INFERENCE)["message"]
    assert "estimating how strongly" in predicting
    assert "were this model for prediction" in inferring.lower()
    assert "CLINICAL_SURVEY_PACK.md" in predicting


def test_the_absolute_is_gone_from_the_module():
    """The sentence itself, asserted absent. *There is no configuration in
    which this is acceptable* was false in an app that records which
    configuration it is in."""
    source = (ROOT / "turbotab" / "missingness.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.split("\n")
                     if not line.lstrip().startswith("#"))
    assert "There is no configuration in which this is acceptable" not in code


# ── the unanswered case, which is a decision rather than a default ─────────

def test_an_unanswered_purpose_refuses_and_names_the_question():
    """The purpose is a CHOICE the constitution says is always asked and never
    inferred, so an unanswered purpose is not evidence for the inference
    branch. The safe branch stands AND the question that would change it is
    named — the recorded-absence rule rather than a shrug."""
    reading = M.outcome_in_scope("hba1c", None)
    assert reading["refuse"] is True
    assert "purpose question has not been answered" in reading["message"]
    assert "nothing in your data reveals it" in reading["message"]


# ── through the record, which is where it matters ─────────────────────────

def test_declare_refuses_under_prediction_and_records_under_inference():
    kwargs = dict(column="glucose", branch="numeric", mechanism=M.NOT_SURE,
                  strategy_key=M.IMPUTE_MICE, target="hba1c",
                  uses_columns=["age", "hba1c"], n_missing=20)
    with pytest.raises(M.MissingnessRefusal, match="not offered as a choice"):
        M.declare(purpose=P.PREDICTION, **kwargs)
    with pytest.raises(M.MissingnessRefusal):
        M.declare(**kwargs)                       # purpose unanswered

    record = M.declare(purpose=P.INFERENCE, **kwargs)
    note = record["outcome_in_scope"]
    assert note and note["refuse"] is False
    assert note["purpose"] == P.INFERENCE
    assert note["evidence_status"] == "SETTLED"


def test_the_note_is_absent_when_the_outcome_is_not_in_scope():
    """A note that fires when nothing happened is noise, and noise is how a
    real note stops being read."""
    record = M.declare(column="glucose", branch="numeric",
                       mechanism=M.NOT_SURE, strategy_key=M.IMPUTE_MICE,
                       target="hba1c", uses_columns=["age"], n_missing=20,
                       purpose=P.INFERENCE)
    assert record["outcome_in_scope"] is None


# ── and from outside, because a refusal a client cannot reach is a claim ──



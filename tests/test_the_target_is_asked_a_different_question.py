"""The outcome's question is different in kind from a feature's.

For a feature, binary-versus-numeric is a **reading**: the values mean the same
thing either way and the question is how to store them. For the outcome the
reading is nearly forced — two-level text is binary classification — and the
decision that matters is **which level is the event being predicted**.

That choice sets the sign of every effect estimate, decides what sensitivity and
specificity are the sensitivity and specificity *of*, and determines what the
model is trained to detect. Asking "is this binary?" about an outcome answers a
question nobody had and skips the one that matters.

**Never pre-selected, at any confidence.** Convention may be offered as a
suggestion with its reasoning shown — `responder`, `improved`, `case` are
conventionally the event — but `alive`/`dead` has no correct default. Whether
the event is death or survival is the research question, not a property of the
data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.binary_text import (EVENT_CONVENTIONS, apply_positive_class,
                            binary_text_finding, detect_binary_text,
                            positive_class_finding)


def outcome(levels=("responder", "non-responder"), n=80, seed=4):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(list(levels), n, p=[0.4, 0.6]))


# ── the question that gets asked ─────────────────────────────────────────

def test_the_target_is_asked_which_level_is_the_event():
    f = positive_class_finding("outcome", outcome())
    assert f is not None
    assert f.title == "Which of these is the event you are predicting?"
    assert "binary" not in f.title.lower(), (
        "the outcome is being asked how to read it rather than what it means")
    assert f.fix_kind == "set_positive_class"


def test_the_same_column_as_a_feature_is_asked_how_to_read_it():
    f = binary_text_finding("meds_chol", outcome(("True", "False")))
    assert f is not None
    assert f.fix_kind == "read_as_binary"
    assert "binary variable written as text" in f.title


def test_the_target_routes_to_its_own_question_and_features_do_not():
    df = pd.DataFrame({"outcome": outcome(),
                       "meds": outcome(("True", "False"), seed=9)})
    kinds = {f.affected_columns[0]: f.fix_kind
             for f in detect_binary_text(df, target="outcome")}
    assert kinds == {"outcome": "set_positive_class", "meds": "read_as_binary"}


def test_with_no_target_yet_every_column_is_read_as_a_feature():
    """Before a target is chosen there is no outcome column to ask about."""
    df = pd.DataFrame({"outcome": outcome()})
    kinds = {f.fix_kind for f in detect_binary_text(df)}
    assert kinds == {"read_as_binary"}


def test_the_question_states_why_it_matters():
    why = positive_class_finding("outcome", outcome()).why_it_matters.lower()
    assert "sign" in why and "sensitivity" in why, (
        "the question does not say what the choice changes")


# ── never pre-selected ───────────────────────────────────────────────────

@pytest.mark.parametrize("levels", [
    ("responder", "non-responder"),      # conventional vocabulary
    ("yes", "no"),                       # a known boolean pair
    ("alive", "dead"),                   # no correct default exists
    ("arm_a", "arm_b"),                  # no convention at all
])
def test_the_event_is_never_pre_selected(levels):
    f = positive_class_finding("outcome", outcome(levels))
    assert f.auto_suggestable is False, (
        f"{levels} pre-selected an event; which level is the event is the "
        "research question, not a property of the data")
    assert f.confidence != "high"


def test_a_convention_is_offered_with_its_reasoning_shown():
    f = positive_class_finding("outcome", outcome())
    assert f.params["suggested"] == "responder"
    assert "conventionally the event" in f.params["suggested_reason"]
    assert "suggestion, not an answer" in f.why_it_matters


def test_alive_dead_gets_no_suggestion_at_all():
    """The case that settled the rule. Neither level is conventionally the event."""
    f = positive_class_finding("survival", outcome(("alive", "dead")))
    assert f.params["suggested"] is None
    assert f.params["suggested_reason"] is None
    assert not (EVENT_CONVENTIONS & {"alive", "dead"}), (
        "a mortality vocabulary was added to the conventions table; whether the "
        "event is death or survival is the research question")


# ── applying the choice ──────────────────────────────────────────────────

def test_applying_without_a_choice_refuses_rather_than_defaulting():
    s = outcome()
    f = positive_class_finding("outcome", s)
    df = pd.DataFrame({"outcome": s})
    with pytest.raises(ValueError) as exc:
        apply_positive_class(df, f)
    assert "research question" in str(exc.value)


def test_the_chosen_level_becomes_one():
    s = outcome()
    f = positive_class_finding("outcome", s)
    df = pd.DataFrame({"outcome": s})

    out, desc = apply_positive_class(df, f, event="responder")
    assert set(out["outcome"].dropna().unique()) == {0, 1}
    assert (out.loc[s == "responder", "outcome"] == 1).all()
    assert "responder as the event" in desc

    other, desc2 = apply_positive_class(df, f, event="non-responder")
    assert (other.loc[s == "non-responder", "outcome"] == 1).all(), (
        "the user's contrary choice was overridden by the convention")
    assert "non-responder as the event" in desc2


def test_the_input_frame_is_never_mutated():
    s = outcome()
    df = pd.DataFrame({"outcome": s})
    before = df["outcome"].tolist()
    apply_positive_class(df, positive_class_finding("outcome", s), event="responder")
    assert df["outcome"].tolist() == before


def test_a_level_that_is_not_in_the_column_is_refused():
    s = outcome()
    f = positive_class_finding("outcome", s)
    with pytest.raises(ValueError) as exc:
        apply_positive_class(pd.DataFrame({"outcome": s}), f, event="maybe")
    assert "not one of the two levels" in str(exc.value)


def test_rows_with_no_outcome_stay_missing_and_are_counted():
    s = pd.Series(["responder", "non-responder", None] * 10)
    f = positive_class_finding("outcome", s)
    out, desc = apply_positive_class(pd.DataFrame({"outcome": s}), f,
                                     event="responder")
    assert out["outcome"].isna().sum() == 10
    assert "no outcome recorded" in desc

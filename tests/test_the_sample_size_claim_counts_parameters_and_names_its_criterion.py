"""`AUDIT-020` and `AUDIT-021` — one design, and this file is the guard on it.

`CLINICAL_SURVEY_PACK.md` §A5.4 makes two statements about sample size and the
app got both wrong in the same sentence:

* **the denominator** — *count parameters, not variables; a 4-knot spline is 3
  parameters, a 5-level factor is 4.* `ml/dataset_profile.py` divided by
  `len(feature_cols)`, a count of DataFrame columns;
* **the criterion** — *the events-per-variable rule of 10 is a legacy heuristic
  that both under- and over-estimates requirements depending on prevalence and
  expected model strength; use the criteria-based calculation.*
  **[SETTLED that EPV≥10 is superseded.]** `ml/model_coach.py` printed
  `guideline ≥ 10`.

Together those produced, on a 200-row frame with one 5-level factor and 26
minority events: **`EPV = 26.0`, "Good events per variable (26.0).
Classification models have adequate signal."** The registry-correct number is
`6.5`, which by the same function's own bands is the *low* band. The app was one
band too reassuring and cited a retired rule for it.

## The fixture rule

`GUIDED-097`: every claim about a journey step runs against **two fixtures of
different target shape.** Here they are an integer `0/1` outcome and a
**string-labeled** one (`alive`/`dead`), because a string target is where
`float()` succeeds for one and raises for the other. **Not covered: multiclass
(k > 2) and a continuous target** — EPV is not defined for a regression outcome
and the branch does not fire, which is stated rather than left silent.

## What is deliberately NOT asserted here

That the app computes Riley's minimum. It does not, and `ml/sample_size.py`
records why: the calculation needs an anticipated model R² the app never asks
for. These tests assert the app **says which criterion it is not computing**,
which is the honest form of a capability that does not exist.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml import sample_size as SS                                      # noqa: E402
from ml.dataset_profile import compute_dataset_profile                # noqa: E402
from ml.model_coach import model_viability, select_top_picks          # noqa: E402
from turbotab import resolution as RES                                # noqa: E402

#: 200 rows, one 5-level factor, 26 minority events. §A5.4 charges 4 parameters
#: for the factor, so the correct EPV is 26/4 = 6.5 and the defect reported 26.0.
N_ROWS = 200
N_LEVELS = 5
N_EVENTS = 26
CORRECT_PARAMETERS = N_LEVELS - 1
CORRECT_EPV = N_EVENTS / CORRECT_PARAMETERS


def _one_factor_frame(target_shape: str) -> pd.DataFrame:
    """The same frame under two target shapes. `GUIDED-097`."""
    rng = np.random.default_rng(20)
    grade = rng.choice(list("ABCDE")[:N_LEVELS], size=N_ROWS)
    flag = np.array([1] * N_EVENTS + [0] * (N_ROWS - N_EVENTS))
    rng.shuffle(flag)
    if target_shape == "int01":
        target = flag
    elif target_shape == "string":
        target = np.where(flag == 1, "dead", "alive")
    else:                                                     # pragma: no cover
        raise AssertionError(f"unknown target shape {target_shape!r}")
    return pd.DataFrame({"grade": grade, "outcome": target})


TARGET_SHAPES = ["int01", "string"]


# ── AUDIT-020 · the denominator ─────────────────────────────────────────────

@pytest.mark.parametrize("target_shape", TARGET_SHAPES)
def test_events_per_variable_divides_by_parameters_not_by_columns(target_shape):
    """`AUDIT-020`. A 5-level factor is 4 parameters, so EPV is 6.5 and not 26."""
    profile = compute_dataset_profile(_one_factor_frame(target_shape),
                                      target_col="outcome",
                                      task_type="classification")
    assert profile.n_candidate_parameters == CORRECT_PARAMETERS, (
        f"§A5.4 charges {CORRECT_PARAMETERS} parameters for a "
        f"{N_LEVELS}-level factor; the profile charged "
        f"{profile.n_candidate_parameters}")
    assert profile.events_per_variable == pytest.approx(CORRECT_EPV), (
        f"EPV is {profile.events_per_variable}, not {CORRECT_EPV}. A column "
        f"count as the denominator gives {float(N_EVENTS)}, which is the "
        f"defect AUDIT-020 named")
    # The column count is a different, true fact and it is still reported.
    assert profile.n_features == 1


@pytest.mark.parametrize("target_shape", TARGET_SHAPES)
def test_both_doors_charge_the_same_parameters_for_the_same_frame(target_shape):
    """The cross-door form. Guided charged `nunique − 1` and Classic charged 1
    for the same column, so one file produced two different sample-size claims
    depending on which door the researcher opened."""
    frame = _one_factor_frame(target_shape)
    profile = compute_dataset_profile(frame, target_col="outcome",
                                      task_type="classification")
    guided = RES.candidate_parameters(frame, "outcome")["total"]
    assert profile.n_candidate_parameters == guided == CORRECT_PARAMETERS


@pytest.mark.parametrize("target_shape", TARGET_SHAPES)
def test_the_exported_sufficiency_sentence_states_the_denominator_it_used(target_shape):
    """The narrative is what `pages/10_Report_Export.py` writes into the report
    and what `turbotab/api.py` serializes to the Guided door, so the count it
    divided by has to be in the sentence a reader sees."""
    profile = compute_dataset_profile(_one_factor_frame(target_shape),
                                      target_col="outcome",
                                      task_type="classification")
    narrative = profile.sufficiency_narrative
    assert f"{CORRECT_PARAMETERS:,} candidate parameters" in narrative, narrative
    assert "(EPV = 6.5)" in narrative, narrative
    assert "26.0" not in narrative, (
        "the column-count EPV is still in the exported sentence: " + narrative)


@pytest.mark.parametrize("target_shape", TARGET_SHAPES)
def test_the_exported_sentence_no_longer_certifies_adequate_signal(target_shape):
    """`AUDIT-020`'s prose half, and `AGENT_ONBOARD.md`'s governing rule.

    *"Classification models have adequate signal"* is an adequacy verdict, and
    §A5.4 says only the criteria-based calculation can reach one. It was
    asserted from the heuristic that section marks superseded, and exported into
    the manuscript. No band may say it — including the band a well-powered study
    lands in, which is why this runs on a frame at EPV 30 as well.
    """
    frame = _one_factor_frame(target_shape)
    wide = frame.copy()
    rng = np.random.default_rng(21)
    # 120 events over 4 parameters is EPV 30 — the old top band.
    flag = np.array([1] * 120 + [0] * (N_ROWS - 120))
    rng.shuffle(flag)
    wide["outcome"] = (flag if target_shape == "int01"
                       else np.where(flag == 1, "dead", "alive"))
    for f in (frame, wide):
        narrative = compute_dataset_profile(
            f, target_col="outcome", task_type="classification"
        ).sufficiency_narrative
        assert "adequate signal" not in narrative, narrative
        assert "Good events per variable" not in narrative, narrative


def test_no_events_per_parameter_is_reported_when_nothing_can_be_spent():
    """Trap 9. A frame whose only predictor is constant spends 0 parameters, and
    `minority / 0` used to be `float('inf')` — the value of a perfectly powered
    study, asserted from a frame with no usable predictor at all."""
    rng = np.random.default_rng(22)
    flag = np.array([1] * N_EVENTS + [0] * (N_ROWS - N_EVENTS))
    rng.shuffle(flag)
    frame = pd.DataFrame({"constant": ["z"] * N_ROWS, "outcome": flag})
    profile = compute_dataset_profile(frame, target_col="outcome",
                                      task_type="classification")
    assert profile.n_candidate_parameters == 0
    assert profile.events_per_variable is None
    assert "EPV" not in profile.sufficiency_narrative


# ── AUDIT-021 · the criterion ───────────────────────────────────────────────

def _low_epv_profile(target_shape: str):
    """40 minority events, 8 numeric predictors — EPV 5.0, the row's own case."""
    rng = np.random.default_rng(23)
    n = 400
    data = {f"x{i}": rng.normal(size=n) for i in range(8)}
    flag = np.array([1] * 40 + [0] * (n - 40))
    rng.shuffle(flag)
    data["outcome"] = (flag if target_shape == "int01"
                       else np.where(flag == 1, "dead", "alive"))
    return compute_dataset_profile(pd.DataFrame(data), target_col="outcome",
                                   task_type="classification")


@pytest.mark.parametrize("target_shape", TARGET_SHAPES)
def test_the_coach_headline_does_not_present_ten_as_the_field_s_guideline(target_shape):
    """`AUDIT-021`, Classic door — `pages/05_Preprocess.py:206` renders this
    headline and `:293-300` records it into workflow provenance as material the
    manuscript can cite."""
    profile = _low_epv_profile(target_shape)
    assert profile.events_per_variable == pytest.approx(5.0)
    _, _, headline = select_top_picks(profile)
    assert "guideline" not in headline, headline
    assert "EPV = 5.0" in headline, headline
    assert "caution trigger" in headline, (
        "the headline states 10 without saying whose number it is: " + headline)
    assert "superseded" in headline, headline
    assert "criteria-based minimum" in headline, (
        "the headline retires the rule of 10 without naming what replaced it: "
        + headline)


@pytest.mark.parametrize("target_shape", TARGET_SHAPES)
def test_no_model_verdict_states_an_epv_threshold_as_a_requirement(target_shape):
    """`AUDIT-021`, Guided door — `turbotab/models.py:98` calls exactly this
    function and renders a clause under every model card. The `nn` clause read
    *"needs roughly 500+ rows and EPV≥10"*, which states the superseded rule as
    a requirement of the software."""
    verdicts = model_viability(_low_epv_profile(target_shape))
    for key, (_verdict, clause) in verdicts.items():
        assert "EPV≥10" not in clause, f"{key}: {clause}"
        assert "EPV >= 10" not in clause, f"{key}: {clause}"
        assert "guideline" not in clause, f"{key}: {clause}"
    # The number itself is still reported — the shelf is not shortened.
    assert "EPV=5.0" in verdicts["logreg"][1]


def test_the_caution_threshold_is_named_as_the_app_s_own_and_not_the_field_s():
    """The constant carries the disclosure, so no future call site can emit the
    number without the sentence being one import away. §08 check 2: the value is
    unchanged — what the denominator correction moved is which quantity it
    gates."""
    assert SS.CAUTION_EPV == 10.0
    assert "superseded" in SS.SUPERSEDED_SHORT
    assert "legacy heuristic" in SS.SUPERSEDED
    assert SS.EVIDENCE["evidence_status"] == "SETTLED"
    assert SS.EVIDENCE["source"].endswith("#A5.4 Sample size")
    assert (ROOT / "docs" / "turbotab" / "research"
            / "CLINICAL_SURVEY_PACK.md").exists()


def test_the_theory_reference_does_not_teach_the_retired_rule_as_current():
    """§08 check 5 — what the same lens finds one surface over. The Theory
    Reference is where a researcher goes to learn the concept, and it taught
    *"The classic rule: at least 10-20 events per variable"* with nothing saying
    it had been superseded."""
    from utils.theory_anchors import THEORY_ANCHORS

    entry = THEORY_ANCHORS["sample_size"]
    text = " ".join(str(v) for v in entry.values())
    assert "legacy heuristic" in text, text
    assert "criteria-based minimum" in text, text
    assert "Count PARAMETERS, not columns" in text, text
    assert "The classic rule: at least 10" not in text, text


# ── the surfaces the strings actually reach ─────────────────────────────────
#
# These are claims about a FILE — does this module hand that field to that
# writer — so they are read statically rather than driven. Trap 5's reservation
# is the other direction: a claim about BEHAVIOR must be driven, and the two
# above that are about behavior are.

def test_the_narrative_is_the_string_both_doors_publish():
    export = (ROOT / "pages" / "10_Report_Export.py").read_text(encoding="utf-8")
    assert "profile.sufficiency_narrative" in export, (
        "the report no longer exports the sufficiency narrative; this file's "
        "claim that the corrected sentence reaches the manuscript is stale")
    api = (ROOT / "turbotab" / "api.py").read_text(encoding="utf-8")
    assert '"sufficiency_narrative": prof.get("sufficiency_narrative")' in api, (
        "the Guided door no longer serializes the sufficiency narrative")
    preprocess = (ROOT / "pages" / "05_Preprocess.py").read_text(encoding="utf-8")
    assert "_coach_headline" in preprocess

"""`GUIDED-177` · the feature-selection preview is evidence for the method chosen.

## What was measured before the fix

Driven through the real API on `clinic_visits.csv` with both of its targets:
**five methods are offered** — `mutual_info`, `lasso`, `rfe`, `univariate`,
`stability` — and all five produced **one** ranking, with **one** measure
string. `selection.evidence` took no method argument at all; it computed
`abs(pearson)` unconditionally and labeled every row *absolute correlation
with the outcome*, under a recorded sentence reading *the top 5 features by
mutual information with `glucose`*.

Second half of the same measurement, and it is the worse one: on the
**classification** shape — `outcome`, a string label — every one of the seven
numeric candidates came back `score: None` with the measure *not numeric — not
ranked here*. The columns are floats. It was the OUTCOME that a correlation
could not read, and the sentence blamed the feature.

After: **six distinct measure strings across the six requests** on each shape,
and the two methods that have a per-column statistic compute it with the same
sklearn scorer `pipeline_plan._selector` fits inside the fold.

## The line this file asserts

A method's preview is possible when its score is a property of ONE column
against the outcome. `mutual_info` and `univariate` are; `lasso`, `rfe` and
`stability` rank by what survives a fit over all candidates at once, and
getting that means running the selector, which is a selection. **Those three
stay offered** — the shelf is never shortened — and say what was not computed
instead of borrowing another method's number.

## Fixture shapes — `GUIDED-097`

`TARGET_SHAPES` runs the load-bearing claims against a continuous target, a
binary-string target and a three-level target. `SHAPES_NOT_COVERED` names the
rest, with the reason, in the file rather than in a report.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import selection as _sel         # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Three target shapes, and the multiclass one is here because a
#: three-level outcome is the case where `mutual_info_classif` and `f_classif`
#: diverge most from anything Pearson could report.
TARGET_SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c"),
    "binary_string": ("clinic_visits.csv", "outcome"),
    "multiclass": ("multiclass_stage.csv", "disease_stage"),
}

#: And the ones this file does NOT cover.
SHAPES_NOT_COVERED = {
    "binary_numeric": (
        "`leaky_sepsis.csv` has a 0/1 target and no missing values. It is the "
        "shape where the OLD behavior was least visibly wrong — a correlation "
        "against 0/1 is a real if crude statistic — so it is the weakest of "
        "the four for this claim, and it is the one dropped."),
    "survival": (
        "No fixture carries a time-and-event pair, and neither sklearn scorer "
        "used here accepts one. A survival outcome would need a third branch "
        "in `_preview_measure`, not a third fixture."),
    "wide": (
        "`metabolomics_untargeted.csv` has 396 numeric columns. The preview "
        "caps at `top=12` rows but scores every candidate first, so the wide "
        "case is a timing question rather than a correctness one, and it is "
        "not driven here."),
}

#: The one string the old code put on every row of every method.
PEARSON = "absolute correlation with the outcome"

#: What `/features` offers, read from the module rather than restated.
OFFERED = sorted(_sel.METHODS)

#: The three that rank by a fit rather than by a per-column statistic.
NO_SCORE = sorted(_sel._NO_PER_FEATURE_SCORE)

#: The two that have a per-column statistic and therefore a real preview.
PREVIEWABLE = sorted(set(OFFERED) - set(NO_SCORE))


# ── 4 · the sharpest one: MI sees what a correlation cannot ──────────────────

def test_mutual_information_ranks_the_column_correlation_ranks_last():
    """**Why the substitution mattered most for the method it was paired with.**

    Mutual information is chosen precisely because it reads non-monotone
    association. On a column with `y = (x - 0.5)**2` the Pearson correlation is
    near zero by construction and the MI is the largest in the table — so the
    correlation preview put the single most informative column LAST under a
    sentence promising a mutual-information selection.

    Direct call rather than a fixture upload, because no shipped CSV contains a
    deliberately non-monotone pair and inventing one in `sample_data` to make a
    statistical point would be a fixture manufacturing its own result.
    """
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0, 1, n)
    noise = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "curved": x,
        "noise": noise,
        "linear": rng.uniform(0, 1, n),
    })
    df["y"] = (df["curved"] - 0.5) ** 2 + 0.02 * rng.normal(0, 1, n)
    df["linear"] = df["y"] * 0.6 + 0.4 * rng.uniform(0, 1, n)
    cands = ["curved", "noise", "linear"]

    corr = _sel.evidence(df, "y", cands)
    assert corr["measure"] == PEARSON                                # control
    corr_order = [r["feature"] for r in corr["ranked"]]
    assert corr_order[-1] == "curved", (
        "this fixture cannot tell the two measures apart: the correlation "
        f"ranking is {corr_order}, and `curved` is meant to be last in it")

    mi = _sel.evidence(df, "y", cands, method="mutual_info",
                       task_type="regression")
    mi_order = [r["feature"] for r in mi["ranked"]]
    assert mi["measure"] == "mutual information with the outcome"
    assert mi_order[0] == "curved", (
        f"mutual information ranked {mi_order}; the non-monotone column is "
        f"not first, so this preview is not reading what MI reads")
    assert mi_order != corr_order, (
        "the two measures produced the same ordering, so this file cannot "
        "tell a method-aware preview from the old one")
    # AND THE NUMBERS ARE NOT THE CORRELATION'S. Same ordering by accident
    # would still be caught above; same values would mean the branch is dead.
    assert ([r["score"] for r in mi["ranked"]]
            != [r["score"] for r in corr["ranked"]])


# ── 5 · the string-outcome half: the reason named the wrong column ───────────

def test_a_label_outcome_is_named_as_the_reason_a_correlation_was_not_computed():
    """Every numeric candidate came back *not numeric — not ranked here* on a
    project whose target is a string, which is false of a column of floats.

    The refusal is the outcome's, so the sentence is about the outcome.
    """
    df = pd.DataFrame({"age": [40.0, 50.0, 60.0, 70.0, 55.0],
                       "chol": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "status": ["died", "lived", "died", "lived", "died"]})
    body = _sel.evidence(df, "status", ["age", "chol"])

    assert body["is_ranked"] is False
    assert "status" in body["measure"], body["measure"]
    for row in body["ranked"]:
        assert "not numeric — not ranked here" != row["measure"], (
            f"{row['feature']!r} is a float column and the preview says it is "
            f"not numeric")

    # And a method that CAN read a label outcome does read it.
    mi = _sel.evidence(df, "status", ["age", "chol"],
                       method="mutual_info", task_type="classification")
    assert mi["is_ranked"] is True
    assert any(r["score"] is not None for r in mi["ranked"]), mi["ranked"]

"""What the manuscript may say about choosing a model on the held-out set.

`AUDIT-030`, ruled at the L45 adjudication.

## The word that was false

Classic's Train page ranks the trained models by
`model_results[name]['metrics']` (`pages/06_Train_and_Compare.py:1546`), and
that dict is the **test** dict written at `:1496` from `test_metrics`. It then
recorded the decision as `selection_criteria='validation <metric>'`, and
`ml/narrative_engine.py` rendered *"<Model> was selected as the primary model,
based on validation <metric>."* into the Methods section.

**There is no validation split behind that sentence.** A real validation split
exists and is used — hyperparameter optimization at `pages/06:1260` and `:1318`
draws one — but nothing stores a per-model validation score, and the ranking
that names the primary model never sees one. So the word was false, and the
governing rule is that the app may be silent and may refuse and must never
assert something false.

`GUIDED-104`'s precedent does not cover it. That row let a run note correct a
recorded scope, and it was accepted **because the weaker claim was still true**
— the app really had fitted over the training rows, it had merely implied a
stronger thing. Here there is no weaker true reading of `validation`.

## The two sentences that replace it

**What was compared.** The models were ranked on the held-out set. That is a
fact, it is short, and it is what a reader needs to interpret every number
beside it.

**What that costs.** Choosing among N fitted models by their held-out scores
makes the chosen model's reported performance optimistic, because the reported
score is a maximum over N draws on the same rows.
`research/CLINICAL_SURVEY_PACK.md` §A5.5 lists *"reporting apparent performance
without optimism correction"* flatly as an anti-pattern and names bootstrap
optimism correction as the recommended default.

**And the number is declined, deliberately.** This door computes no optimism
correction, so the sentence states the **direction** and refuses the magnitude.
`unquantified` is the honest word rather than a hedge — the same posture as the
not-estimable annotation box, and the same discipline as returning `(None, None)`
from a calibration that could not be fitted rather than the values of a perfect
one.

## Why this is a module rather than an f-string

The false phrase was composed in **five** places, and `AUDIT-030` names two of
them: `pages/06_Train_and_Compare.py:1580`, `ml/narrative_engine.py:587` and
`:1091`, and — the two the row missed — `utils/workflow_provenance.py:657` and
`:659`, where the same words are the *default* a sparse record falls back to.
A claim composed in five places is a claim with five chances to drift back.
"""
from __future__ import annotations

#: The surface the comparison actually happened on. One word, in one place, so
#: `validation` cannot come back through a producer nobody updated.
HELD_OUT = "held-out"


def criterion_phrase(metric: str = "") -> str:
    """What was compared, as it reads inside the Methods sentence.

    Returns a phrase, never an empty string that a caller would then have to
    decide about: a selection whose metric is unknown is still a selection made
    on the held-out set, and saying so is the true short sentence.
    """
    metric = str(metric or "").strip()
    return f"the {HELD_OUT} {metric}" if metric else f"{HELD_OUT} performance"


def optimism_sentence(n_models: int) -> str:
    """The cost of having chosen among `n_models` on the held-out set.

    Returns `""` when there was no choice to make. **One model is not a
    selection**, and attaching an optimism caveat to a single fitted model would
    be the second, uncalibrated layer of caution this project forbids — it makes
    a real concern and a routine one read identically, which is the failure the
    evidence badge exists to prevent.
    """
    n = int(n_models or 0)
    if n < 2:
        return ""
    return (
        f"Because the reported model was chosen by comparing {n} models' scores "
        f"on the {HELD_OUT} set, its reported performance is optimistic relative "
        f"to a model chosen without those rows; the size of that optimism was "
        f"not estimated here."
    )

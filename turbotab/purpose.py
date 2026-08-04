"""What is this model for? — question 2.5 of the pre-seal sequence.

`DOMAIN_SCIENCE.md` §01.3, and it is the deepest of the seven convergences.

> The same dataset, the same target and the same lens can require **opposite**
> handling depending on whether the user wants prediction or inference.

The research names five places across four domains where the advice does not
shade — it **inverts**:

| Domain | The question | Prediction | Inference |
|---|---|---|---|
| Clinical EHR | missing labs | missing-indicator method | **never** — biases the estimate |
| Clinical labs | values below LOD | censoring indicator + substitution | censored regression |
| Nutrition | two recalls per person | the mean is fine for ranking | not adequate for a prevalence claim |
| Survey | a 30-item instrument | item-level with penalization | scale score with attenuation correction |
| Metabolomics | features vs compounds | feature-level is fine | compound-level is what you may claim |

TurboTab assumed prediction throughout — reasonably, for a predictive-modeling
app — and never asked. **A tool that gives the inference answer to someone
building a bedside model is wrong, and so is the reverse.**

## Why it is a CHOICE and not a FACT

Nothing in the data reveals it. Two studies with the same table, the same
outcome and the same lens can want opposite things, and the difference lives
entirely in the research question. So by the routing constitution it is a
question of **choice**: always asked, never skippable at any confidence, and no
default at any confidence either — a pre-selected purpose would be the app
deciding what the user's paper is about.

## Why it is worth its place in a sequence that fights for length

`OPENING_SEQUENCE.md`'s own test is *if the answer were wrong, would a
downstream number be wrong or misleading?* Emphatically. It fires **once** and
it changes the default on roughly a dozen decisions per pack — the best ratio in
the opening sequence.

## Where it goes

Immediately after the target, because it is about what the target is *for*, and
before the grain, because the answer changes what several later questions
default to. Position **2.5**, by the same reasoning as 1.5: every other position
is cited across three documents and the fixture table, and a fractional number
is itself true — this was inserted between two that were already fixed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from turbotab import exits as _exits

PREDICTION = "prediction"
INFERENCE = "inference"
PURPOSES = (PREDICTION, INFERENCE)


class PurposeError(Exception):
    """A purpose the app cannot honestly record."""


TITLE = "What is this model for?"
WHY = ("Both are legitimate and they are optimized for different things, so "
       "several later answers change depending on which you say. Nothing in "
       "the file reveals it — only you know what the paper claims.")
CONSUMER = (
    "The missing-data route reads it first: a was-it-missing indicator carries "
    "the clinician's judgment and is observable at deployment, which makes it "
    "useful for prediction and a known source of bias for an association "
    "estimate. Three more places read it: whether a value below the limit of "
    "detection may be substituted, whether the outcome may sit inside the "
    "imputation model, and whether class weighting is contraindicated. Those "
    "four are the whole list today. The research names further forks this "
    "answer would decide — which repeated measurements may be averaged, and "
    "whether a survey instrument enters as a scale score or item by item — "
    "and this app implements neither, so those calls stay yours. Answering "
    "wrongly does not raise an error, it produces a complete set of numbers "
    "optimized for a claim you are not making."
)

OPTIONS: List[Dict[str, str]] = [
    {"key": PREDICTION,
     "label": "Predicting an outcome for a new person",
     "note": "Records that the number at the bedside is what has to be right. "
             "Handling that carries information about a patient — a "
             "was-it-missing indicator, a censoring flag — becomes legitimate, "
             "because it is available at deployment too."},
    {"key": INFERENCE,
     "label": "Estimating how strongly something is associated with the outcome",
     "note": "Records that the coefficient has to be unbiased. Several "
             "conveniences that help a predictor become sources of bias here, "
             "and the app will say so at the point each one is offered rather "
             "than in a lecture now."},
]


def question() -> Dict[str, Any]:
    """The question, as the Router and the page both read it."""
    return {
        "key": "state_purpose",
        "clause": "lockbox-01",
        "seq": "2.5",
        "title": TITLE,
        "why": WHY,
        "consumer": CONSUMER,
        "options": list(OPTIONS),
    }


def normalize(answer: str) -> str:
    answer = str(answer or "").strip()
    if answer not in PURPOSES:
        raise PurposeError(
            f"{answer!r} is not one of {list(PURPOSES)}. There is no third "
            f"answer and no default: a pre-selected purpose would be the app "
            f"deciding what your paper is about.")
    return answer


def methods_sentence(answer: str) -> str:
    """The sentence the record keeps and the manuscript carries.

    Both answers get one, and the wording is deliberately about the **objective**
    rather than about the software: it is a sentence a methods section can carry
    as it stands.
    """
    if answer == INFERENCE:
        return ("The model was built to estimate the strength of association "
                "between the predictors and the outcome; handling was chosen "
                "to keep the coefficients unbiased rather than to maximize "
                "predictive accuracy.")
    return ("The model was built to predict the outcome for a new individual; "
            "handling was chosen to maximize predictive accuracy at deployment "
            "rather than to keep any single coefficient unbiased.")


# ── the first consumer ──────────────────────────────────────────────────────

# A missing-indicator under an inference objective. Not a style question: the
# indicator is a known source of bias in the estimate, and the research is
# unambiguous. Blocked with both exits rather than refused, because the user may
# have a reason — and because §09's CONSEQUENCE is resolve-or-attest, never a
# hard stop.
INDICATOR_UNDER_INFERENCE = (
    "You said this model is for estimating how strongly `{column}` is "
    "associated with the outcome, and a was-it-missing indicator is a known "
    "source of bias in that estimate. The indicator carries the clinician's "
    "decision to order a test, so under a prediction objective it is "
    "legitimate and often helpful — the same column, the same data, and the "
    "opposite answer.\n\n"
    "Multiple imputation, or a model that states its missingness mechanism, is "
    "what an association estimate wants. If you have a reason to keep the "
    "indicator, say so and it is recorded as a stated limitation."
)

INDICATOR_EXITS = (
    {"id": "impute_median", "kind": "resolve",
     "label": "Impute instead, inside the training folds",
     "detail": "The value is filled from the training folds only, and the "
               "estimate is not conditioned on whether it was recorded.",
     # `GUIDED-183`. This carried no `retry`, and `showRefusal` emits
     # ` disabled` for an exit with no `retry.payload` — so **the SAFE way out
     # rendered greyed out beside a live "keep it anyway"**, which is the exact
     # inversion §09 forbids. `GUIDED-087` is the same shape and its build is
     # `missingness.blocker_exits`; this is that build, on the path nothing
     # tested.
     #
     # Both spellings, for the reason `missingness.card_option_for_strategy`
     # gives: `api.py` reads `card_option` in preference to `strategy`, and the
     # request this is merged into came from a door that posts `card_option`.
     "retry": {"payload": {"strategy": "impute_median",
                           "card_option": "impute_median"},
               "how": "Sent again with a training-fold median in place of the "
                      "was-it-missing indicator.",
               "typed": None}},
    _exits.attest(
        "Keep the indicator — I know what this absence is",
        "Recorded as a stated limitation: the association estimate is "
        "conditioned on the missingness pattern, and the methods "
        "section says so.",
        _exits.ACKNOWLEDGE_SIGNAL_LOSS),
)


def blocks_indicator(purpose: Optional[str], strategy: str) -> bool:
    """Whether this strategy is contraindicated by the recorded purpose.

    `None` — the question unanswered — blocks nothing. The app does not get to
    infer a purpose and then hold the user to it.
    """
    return purpose == INFERENCE and strategy == "indicator"


def indicator_blocker(column: str) -> Dict[str, Any]:
    return {
        "kind": "indicator_under_inference",
        "column": column,
        "message": INDICATOR_UNDER_INFERENCE.format(column=column),
        "exits": [dict(e) for e in INDICATOR_EXITS],
        "acknowledgment_kind": "typed",
        # The badge, because this is a pack-grade claim and every pack-grade
        # claim says where the field stands (`GUIDED-047`).
        "evidence_status": "SETTLED",
        "source": ("research/CLINICAL_SURVEY_PACK.md#A2 · ★ Missing data — "
                   "where TurboTab differentiates itself"),
    }

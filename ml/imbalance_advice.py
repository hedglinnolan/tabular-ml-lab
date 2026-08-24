"""What to say about class imbalance, and to whom. `GUIDED-049`.

`DOMAIN_SCIENCE.md` §03b. The research commissioned four anti-pattern registries
as pack CONTENT, and the first pass over them found a defect in shipped code:

* `ml/dataset_profile.py` advised *"Use class weights in training"* and
  *"Consider SMOTE or other resampling"*;
* `ml/eda_recommender.py` repeated it;
* `ml/narrative_engine.py` wrote it **into the generated manuscript** —
  *"To address class imbalance, class_weight='balanced' was applied…"* —
  unconditionally, whenever the flag was set.

Van den Goorbergh, van Smeden, Timmerman & Van Calster (*JAMIA*
2022;29:1525) showed random undersampling, random oversampling and SMOTE all
produce **strong overestimation of minority-class probability without improving
discrimination**, and that any apparent sensitivity gain is reproducible by
simply shifting the decision threshold. Replicated for machine-learning methods
by Carriero et al. (*Stat Med* 2025). Rare outcomes are a real problem — but the
problem is **small-sample overfitting, not imbalance**, and the remedy is
penalization and adequate sample size.

## Why this is a defect and not a feature request

The app recommended a step that damages the property clinical prediction cares
about most, and then **asserted it in the artifact that is the product**. Under
the governing rule that is the serious kind: not silence, not refusal — the app
asserting something false, in a manuscript.

## Why the capability is not deleted

Because the advice is not wrong everywhere. For a classifier operated at a
**fixed operating point**, where the decision is the output and no calibrated
probability is claimed, rebalancing is a defensible way to move the point — and
`DOMAIN_SCIENCE.md` §04's scope filter is explicit that the app builds the
science that changes a sentence a reviewer would challenge, not the science that
changes what a practitioner prefers.

So it is **routed by the purpose** (`GUIDED-048`), which is the question that
exists precisely because the same data wants opposite handling:

| Purpose | Verdict |
|---|---|
| **inference** | contraindicated — the estimate is what is being claimed |
| **prediction** | contraindicated — probability is what is being claimed, and this degrades its calibration |
| unanswered | the citation is stated and nothing is recommended |

Prediction and inference land in the same place here for *different* reasons,
and both are stated: that is the honest reading, not a shortcut.

The one place it survives is a fixed-operating-point classifier, and the app
cannot currently tell one from a risk model — so it is **offered with the
citation shown**, never recommended, and never written into the manuscript
without the qualification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

CITATION = ("van den Goorbergh et al., JAMIA 2022;29:1525; replicated for "
            "machine-learning methods by Carriero et al., Stat Med 2025")

EVIDENCE = {
    "evidence_status": "SETTLED",
    "source": "research/CLINICAL_SURVEY_PACK.md#A5 · Modeling",
}

CONTRAINDICATED = (
    "Rebalancing — class weights, SMOTE, over- or under-sampling — is "
    "contraindicated here. It produces strong overestimation of "
    "minority-class probability without improving discrimination, and the "
    "apparent sensitivity gain is reproducible by shifting the decision "
    "threshold instead. The real problem behind a rare outcome is "
    "small-sample overfitting, and the remedy for that is penalization and "
    "adequate sample size. ({citation}.)")

INFERENCE_EXTRA = (
    "Under an association objective it is worse than unhelpful: reweighting "
    "the outcome distribution changes the intercept and the estimate you are "
    "reporting.")

UNANSWERED = (
    "Whether rebalancing is appropriate depends on what this model is for, "
    "and that has not been recorded yet. For a risk model or an association "
    "estimate it is contraindicated; for a classifier read at a fixed "
    "operating point it is defensible. ({citation}.)")

FIXED_POINT_NOTE = (
    "Defensible only for a classifier read at a fixed operating point, where "
    "the decision is the output and no calibrated probability is claimed. If "
    "a probability is reported, this degrades it.")

# What the app may say instead, and it is the part that makes the removal
# honest rather than merely subtractive.
INSTEAD: List[str] = [
    "Report PR-AUC and calibration alongside discrimination",
    "Choose the decision threshold explicitly, from the costs of the two errors",
    "Penalize the fit, and check whether the sample supports the model at all",
]


def advice(purpose: Optional[str]) -> Dict[str, Any]:
    """The advisory this purpose earns, with its badge and its citation.

    `recommended` is the field every caller reads. It is **never True**: there
    is no purpose under which this app recommends rebalancing, because the two
    purposes it can distinguish are the two where it is contraindicated.
    """
    if purpose == "inference":
        text = CONTRAINDICATED.format(citation=CITATION) + " " + INFERENCE_EXTRA
    elif purpose == "prediction":
        text = CONTRAINDICATED.format(citation=CITATION)
    else:
        text = UNANSWERED.format(citation=CITATION)
    return {
        "advisory": text,
        "instead": list(INSTEAD),
        "recommended": False,
        "offered_note": FIXED_POINT_NOTE,
        "citation": CITATION,
        **EVIDENCE,
    }


def manuscript_sentence(purpose: Optional[str]) -> str:
    """What the manuscript says when rebalancing WAS applied.

    It is still said — the reader has to know what was done — but it is no
    longer said approvingly, and it carries the limitation. An unconditional
    *"to address class imbalance…"* is the app endorsing the step in the one
    artifact that is the product.
    """
    limitation = (
        "This is reported as a limitation: rebalancing is known to "
        "overestimate minority-class probability without improving "
        "discrimination, so the predicted probabilities should not be read as "
        "calibrated risks")
    if purpose == "inference":
        limitation += (", and the reweighting changes the intercept of the "
                       "reported association")
    return (
        "Class weighting (class_weight='balanced') was applied to supported "
        "classifiers, weighting each class inversely proportional to its "
        f"frequency in the training data. {limitation} ({CITATION}).")

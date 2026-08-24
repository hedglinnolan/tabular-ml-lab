"""What the app may say about sample size — the denominator and the criterion.

`AUDIT-020` and `AUDIT-021` are **one design**, and this module is it. Both are
`CLINICAL_SURVEY_PACK.md` §A5.4, read at build; both are about the same
sentence; and fixing either one alone leaves the sentence false in the other
half. The number was computed over the wrong denominator *and* compared against
a threshold the section marks superseded, so the app reported `EPV = 26.0` where
the registry-correct value was `6.5`, and then called `10` "the guideline".

## The two halves, quoted

**The denominator** — §A5.4, first paragraph:

> Inputs: number of candidate predictor **parameters** — *count parameters, not
> variables*; a 4-knot spline is 3 parameters, a 5-level factor is 4.

So a categorical column with `k` observed levels spends `k − 1` parameters,
which is exactly what the one-hot encoder in `turbotab/pipeline_plan` and
`ml/pipeline` produces. `ml/dataset_profile.py` counted DataFrame columns, so a
5-level factor cost 1 instead of 4 and every EPV over a frame with a categorical
predictor was **too high** — the direction that makes the app more reassuring
than the data supports.

**The criterion** — §A5.4, the block quote:

> The events-per-variable rule of 10 is a legacy heuristic that both under- and
> over-estimates requirements depending on prevalence and expected model
> strength; use the criteria-based calculation.
> **[SETTLED that EPV≥10 is superseded; the newer thresholds 0.9 and 0.05 are
> themselves CONVENTION — chosen, not derived.]**

The criteria-based calculation is Riley et al.'s minimum (*Stat Med* 2019,
Part II for binary outcomes): global shrinkage ≥ 0.9, an absolute difference
≤ 0.05 between apparent and adjusted Nagelkerke R², and a margin of error ≤ 0.05
on the overall outcome risk.

## Why this module does not compute Riley's minimum

Because one of its three inputs is an **anticipated model R²**, and the app does
not hold one. `turbotab/resolution.py`'s "What is deliberately NOT here" already
records this as specified-unbuilt for the same reason, alongside metabolomics'
detectable-fold-change curve and nutrition's λ. Inventing an anticipated R² to
make a criterion computable would be the app asserting a study parameter the
researcher never stated — worse than the defect being fixed.

So the honest form is the one the governing rule already names: **be silent
about the criterion it cannot compute, and never assert the superseded rule is
the field's guideline.** `EPV` is still reported, because `40 events for 8
parameters` is a fact about the data rather than a verdict on the study, and
`PRODUCT_VISION.md`'s shelf rule is that a wrong sentence is corrected rather
than deleted.

## The number 10, and why it did not move

`10` survives as `CAUTION_EPV` — **the app's own trigger for saying "keep the
lineup small"**, and it is labeled as that everywhere it is emitted. It is not
presented as an adequacy threshold, and it is deliberately **not moved in the
same loop as the denominator correction** (`AGENT_ONBOARD.md` §08, check 2: a
threshold that moves under pressure is indistinguishable from a relaxed gate).
Correcting *which quantity is gated* is the permitted half of that rule, and
that is exactly what the denominator change is.

Note the direction: because a categorical column now costs `k − 1` instead of
`1`, EPV goes **down** on any frame with a categorical predictor, so more
datasets trip the caution trigger. The correction cannot make the app quieter.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as _pdt

#: Where every claim in this module comes from. Read at build from the file,
#: never from recollection.
SOURCE = "research/CLINICAL_SURVEY_PACK.md#A5.4 Sample size"

#: `[SETTLED that EPV≥10 is superseded]` — the badge §A5.4 attaches to the
#: position, verbatim from the pack.
EVIDENCE: Dict[str, str] = {
    "evidence_status": "SETTLED",
    "source": SOURCE,
}

#: The literature the sentence points at, so the reader can check it without
#: reading this repository. `ml/imbalance_advice.CITATION` established the shape.
CITATION = "Riley et al., Stat Med 2019; pmsampsize"

#: The app's own caution trigger, and it is not the field's guideline. Below
#: this many events per candidate parameter the coach keeps the model lineup
#: small and says why. Every string that uses it says whose number it is.
CAUTION_EPV = 10.0

#: What §A5.4 says about the rule of 10, in one sentence, wherever the app
#: reports EPV. This is the sentence whose absence was `AUDIT-021`.
SUPERSEDED = (
    "Events per candidate parameter is a legacy heuristic — it both under- and "
    "over-estimates what a model needs, depending on prevalence and expected "
    "model strength. The field's criterion is the criteria-based minimum "
    f"({CITATION}), which needs an anticipated model R² this app does "
    "not hold and therefore does not compute."
)

#: The short form, for a one-line headline that already carries the number.
SUPERSEDED_SHORT = (
    "Ten per parameter is this app's caution trigger, not the field's "
    "criterion: the rule of 10 is superseded by the criteria-based minimum "
    f"({CITATION}), which needs an anticipated model R² this app does "
    "not hold."
)

#: §A5.4's flagged warning, kept beside the count because the count is the thing
#: it qualifies. Nothing in this module enforces it — the profile counts the
#: columns it is handed — so it is stated rather than implied.
CANDIDATES_COUNT_EVEN_IF_DROPPED = (
    "Candidate predictors count toward sample size even if they are later "
    "dropped: screening 40 variables and keeping 8 means sizing for 40, "
    "because data-driven selection consumes degrees of freedom whether or not "
    "it appears in the final model."
)


def parameters_for_column(series: Any) -> int:
    """How many parameters one column spends. §A5.4's rule, one column at a time.

    Numeric is 1. A categorical with `k` observed levels is `k − 1` — the width
    the one-hot encoder actually produces, which is why the count describes the
    fit rather than the spreadsheet. A constant column is 0, and that is not a
    special case: `k = 1` gives `k − 1 = 0`, and a column with one value buys
    the model nothing.

    A boolean is numeric to pandas and a 2-level factor to a statistician, and
    both readings give 1, so the branch it does not take costs nothing.
    """
    if isinstance(series, pd.DataFrame):                      # pragma: no cover
        return 0
    if _pdt.is_numeric_dtype(series):
        return 1
    return max(0, int(series.dropna().nunique()) - 1)


def candidate_parameters(frame: pd.DataFrame,
                         feature_cols: Optional[Sequence[str]] = None,
                         *, exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """The §A5.4 parameter count, with the per-column breakdown that explains it.

    This is the one implementation. `turbotab/resolution.candidate_parameters`
    delegates to it and `ml/dataset_profile` reads it, so the two doors cannot
    give a researcher different numbers for the same frame — which is the state
    `AUDIT-020` found them in (Guided charged `nunique − 1`, Classic charged 1).

    Args:
        frame: the data.
        feature_cols: the columns the model may spend parameters on. `None`
            means every column in `frame` except `exclude`.
        exclude: columns to drop — the target, and a grouping column where one
            exists.

    Returns:
        `{"total", "numeric", "from_categorical", "per_column", "largest"}`
        plus this module's badge and source.
    """
    drop = {str(c) for c in exclude}
    if feature_cols is None:
        cols = [c for c in frame.columns if str(c) not in drop]
    else:
        cols = [c for c in feature_cols if str(c) not in drop]

    numeric = 0
    from_categorical = 0
    per_column: List[Dict[str, Any]] = []
    for column in cols:
        series = frame[column]
        if isinstance(series, pd.DataFrame):                  # pragma: no cover
            continue
        spent = parameters_for_column(series)
        if _pdt.is_numeric_dtype(series):
            numeric += spent
            kind = "numeric"
        else:
            from_categorical += spent
            kind = f"{int(series.dropna().nunique())} observed levels"
        per_column.append({"column": str(column), "parameters": spent,
                           "kind": kind})

    ordered = sorted(per_column, key=lambda d: -d["parameters"])
    return {
        "total": int(numeric + from_categorical),
        "numeric": int(numeric),
        "from_categorical": int(from_categorical),
        "per_column": per_column,
        "largest": ordered[:5],
        **EVIDENCE,
    }


def events_per_parameter(minority_class_size: Optional[int],
                         n_parameters: Optional[int]) -> Optional[float]:
    """EPV over §A5.4's denominator, or `None`.

    `None` rather than a number wherever the quotient would be a guess: no event
    count, no parameter count, or a parameter count of zero. The last one is the
    one that used to return `float('inf')` for `p = 0`, and infinite events per
    parameter is the value of a *perfectly* powered study — returning it from a
    frame with no usable predictors asserts exactly the thing the app cannot
    know (`AGENT_ONBOARD.md` §07, trap 9).
    """
    if minority_class_size is None or n_parameters is None:
        return None
    if n_parameters <= 0:
        return None
    return float(minority_class_size) / float(n_parameters)


def epv_sentence(epv: Optional[float],
                 minority_class_size: Optional[int],
                 n_parameters: Optional[int]) -> str:
    """What the app says about an EPV, with the counts and without a verdict.

    The bands are the app's own and are labeled as the app's own. **No band says
    "adequate"** — the previous top band said *"Good events per variable (26.0).
    Classification models have adequate signal,"* which is an adequacy claim
    §A5.4 says only the criteria-based calculation can make, asserted from the
    heuristic that section marks superseded, and exported into the manuscript.

    Empty string where there is no honest number, because the app may be silent.
    """
    if epv is None or minority_class_size is None or not n_parameters:
        return ""
    counts = (f"{minority_class_size:,} minority-class events for "
              f"{n_parameters:,} candidate parameters (EPV = {epv:.1f})")
    if epv < 5:
        band = ("Very little information per parameter. Overfitting risk is "
                "high for any model.")
    elif epv < CAUTION_EPV:
        band = "Little information per parameter. Prefer penalized fits."
    elif epv < 20:
        band = "Moderate information per parameter."
    else:
        band = ""
    return " ".join(x for x in (counts + ".", band, SUPERSEDED) if x)

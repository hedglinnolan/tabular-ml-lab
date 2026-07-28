"""turbotab.obligations — what a decision arms, to be discharged elsewhere.

Clause §05 is the only clause in the constitution whose **obligation fires at a
different step from the action that arms it**, and `STATE-103` records why that
makes it the one most likely to go unbuilt: every other clause is checkable
where it happens. The seal states its basis at the seal. The grain is asked
before the seal. A structural repair posts its receipt when it executes. This
one deliberately spends its friction later.

> A train-only trim is a **legitimate choice**, so it does not earn a blocker —
> friction is spent where an operation is almost certainly an error, and this
> one is not. What is illegitimate is reporting a single aggregate metric
> afterward as though nothing happened.
>
> So the trim is a CHOICE that silently **arms a requirement**, and the blocker
> fires at export if the stratified in-range / out-of-range breakdown is absent.

**This module is the arming half only, and the split is deliberate.** The firing
half needs an export, and there is no Report step; building a blocker with
nothing to block would be untestable, and a half-built clause that looks whole
is precisely the failure `STATE-103` was filed about. So:

    STATE-103   the arming half — a trim records the obligation      (here)
    STATE-105   the firing half — export refuses without the breakdown (open)

Two rows rather than one, because under a single row the arming half landing
would read as progress on the whole clause. Under two, this one closes and the
other stays visibly open.

**An obligation is a promise about what a later number must say**, so it carries
what the later step needs to say it: which column was narrowed, to what range,
how many training rows fell outside it, and the sentence a report has to be able
to produce. A record that only said *"something was trimmed"* would leave the
Report step re-deriving facts that were known at trim time and are not
recoverable afterwards — the trim has already happened by then.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# The one kind this module knows. Named rather than implicit so a second kind —
# and clause §07's stability assumption is a candidate — is added as a value
# rather than by widening what "obligation" silently means.
EXTRAPOLATION = "extrapolation_breakdown"

# Where it is discharged. Not built: `STEP_LABELS` in the Router already names
# steps that do not exist yet, on the principle that naming where an item
# resurfaces is more honest than "later".
DISCHARGED_AT = "report"


class ObligationError(Exception):
    """An obligation was recorded or discharged dishonestly."""


def arm_extrapolation(df: pd.DataFrame, column: str,
                      train_labels: Sequence[Any],
                      minimum: Optional[float] = None,
                      maximum: Optional[float] = None,
                      reason: str = "") -> Dict[str, Any]:
    """Record what a train-only trim obliges the report to say.

    Nothing is trimmed here and nothing is blocked. The trim is a legitimate
    choice; this is the memory of it.

    The counts are computed **now**, against the frame as it stands, because
    they are not recoverable later: after the trim the out-of-range training
    rows are gone, and the report cannot count what is no longer there.
    """
    if column not in df.columns:
        raise ObligationError(f"No column named '{column}' in this table.")
    if minimum is None and maximum is None:
        raise ObligationError(
            "A trim with no bounds narrows nothing, so there is no "
            "extrapolation to disclose.")
    if not (reason or "").strip():
        raise ObligationError(
            "A trim's reason is what the report has to print beside the "
            "breakdown. Without it the disclosure would say that some rows were "
            "outside a range nobody can explain.")

    s = df[column]
    train = s.loc[[l for l in train_labels if l in df.index]]
    held = s.drop(index=[l for l in train_labels if l in df.index])

    def _outside(series: pd.Series) -> int:
        out = pd.Series(False, index=series.index)
        if minimum is not None:
            out |= series < minimum
        if maximum is not None:
            out |= series > maximum
        return int(out.fillna(False).sum())

    bounds = []
    if minimum is not None:
        bounds.append(f"≥ {minimum:g}")
    if maximum is not None:
        bounds.append(f"≤ {maximum:g}")
    range_text = f"`{column}` {' and '.join(bounds)}"

    return {
        "kind": EXTRAPOLATION,
        # §04, said out loud at the point of the CHOICE rather than left to be
        # inferred. A trim and an eligibility criterion look identical in a
        # spreadsheet; the difference is who the model is FOR versus how the fit
        # is stabilized, and only one of them changes N.
        "not_a_population_restriction": (
            "This narrows the TRAINING rows only. It does not change who your "
            "study is about: the held-out rows are untouched, N is unchanged, "
            "and nothing here belongs in participant flow. If you meant to "
            "restrict the population the model is for, that is the eligibility "
            "question, it is asked before the seal, and it does change N."),
        "discharged_at": DISCHARGED_AT,
        "column": str(column),
        "minimum": None if minimum is None else float(minimum),
        "maximum": None if maximum is None else float(maximum),
        "reason": reason.strip(),
        "range": range_text,
        "n_train_trimmed": _outside(train),
        # THE NUMBER THE REPORT ACTUALLY NEEDS. The held-out rows were never
        # trimmed — §04 forbids it — so some of them are outside the range the
        # model was fitted on, and a single aggregate metric averages those in
        # silently. That is the extrapolation clause §05 is about.
        "n_test_outside_range": _outside(held),
        "n_test_total": int(len(held)),
        "requires": (
            "A metric stratified by whether the held-out row falls inside the "
            "trimmed range. A single aggregate number reports performance on a "
            "population the model was not fitted for, without saying so."),
        "sentence": (
            f"The model was fitted on training rows with {range_text} "
            f"({reason.strip()}). {_outside(held)} of {len(held)} held-out rows "
            f"fall outside that range, so performance must be reported "
            f"separately for in-range and out-of-range rows rather than as one "
            f"number."),
        "discharged": False,
    }


def outstanding(obligations: Sequence[Dict[str, Any]],
                at_step: str = DISCHARGED_AT) -> List[Dict[str, Any]]:
    """Obligations this step is responsible for and that are not yet met.

    The firing half reads this. It exists here rather than at the Report step so
    that when Report is built it consumes a record it did not invent — the
    arming step and the firing step must not each have their own idea of what
    was armed, which is how a two-step obligation becomes two half-obligations.
    """
    return [o for o in obligations
            if o.get("discharged_at") == at_step and not o.get("discharged")]

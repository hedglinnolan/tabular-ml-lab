"""The predictors that count toward sample size — `AUDIT-023`.

`research/CLINICAL_SURVEY_PACK.md` §A5.4, flagged ⚠:

> *"**Candidate predictors count toward sample size even if they are later
> dropped.** If you screen 40 variables and keep 8, you must size for 40 —
> data-driven selection consumes degrees of freedom whether or not it appears in
> the final model. This is the sample-size mistake PROBAST most often catches."*

The Guided door gets this right and says so in its own record
(`turbotab/resolution.py:377`, *"counted including any later dropped by feature
selection"*). The Classic door did not, and the way it failed is worth stating
precisely because the sentence read plausibly.

## What went wrong

`pages/04_Feature_Selection.py` applies a selection by assigning
`data_config.feature_cols = consensus` — **in place**. Nothing else recorded the
list that was screened, so after the press the app had no way to know 40
variables had ever been considered. `pages/02_EDA.py` then reads
`data_config.feature_cols` as its feature list, `ml/regime.py:185` sets
`n_features = len(feature_cols)`, and the EDA sufficiency insight writes

> *"the modest ratio of observations to candidate predictors (60 observations,
> 8 predictors)"*

into `manuscript_text`, which the report carries as a limitation. **The number
is the kept count and the word beside it is `candidate`.** Two consequences, and
the second is the one that matters:

1. the sentence is false about what was screened, and
2. the sufficiency verdict — the thing that decides whether the app warns at all
   — was computed with p = 8 when §A5.4 says it must be computed with p = 40.

## What this module does

It does not stop the selection from being applied; applying it is the whole
point of the page, and the shelf is never shortened. It records what was
screened so the sizing arithmetic has the denominator §A5.4 names, and it
composes the phrase that names both numbers.

`sufficiency_over_candidates` takes the **more severe** of the profile's verdict
and the verdict recomputed over the screened count. Monotone on purpose: a
larger denominator can only make the data look scarcer, so this can never
silence a warning the old path raised — it can only raise one the old path
missed. That property is what makes it safe to change an existing verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

#: `ml.dataset_profile.DataSufficiencyLevel` values, most sufficient first.
#: Duplicated as strings rather than imported so that reading a verdict does not
#: drag `dataset_profile`'s sklearn and reference-table imports into a caller
#: that only wants the word.
SEVERITY_ORDER = ("abundant", "adequate", "limited", "scarce", "critical")


@dataclass(frozen=True)
class CandidateCount:
    """How many predictors were screened, and how many survived.

    `screened` is the number §A5.4 says to size for. `retained` is what the
    model is actually offered. `from_record` says whether a selection was
    recorded at all — where it is False the two are equal because nothing was
    dropped, and a caller must not report that as *"40 screened, 40 retained"*
    as though a screening step had run.
    """
    screened: int
    retained: int
    from_record: bool

    @property
    def dropped(self) -> int:
        return max(0, self.screened - self.retained)


def screened_candidates(
    current_features: Sequence[str],
    provenance: Optional[Any] = None,
) -> List[str]:
    """Every column that was offered to the selection, plus the ones kept.

    An ordered union: the recorded screened list first, then anything in the
    current list that is not in it. The union rather than the record alone
    because feature engineering can add columns after a selection was recorded,
    and those are candidates too.
    """
    kept = [str(c) for c in (current_features or [])]
    record = getattr(provenance, "feature_selection", None) if provenance else None
    screened = [str(c) for c in (getattr(record, "candidates_screened", None) or [])]
    out = list(screened)
    seen = set(out)
    for column in kept:
        if column not in seen:
            out.append(column)
            seen.add(column)
    return out


def candidate_count(
    current_features: Sequence[str],
    provenance: Optional[Any] = None,
) -> CandidateCount:
    """The two numbers, from the record where there is one."""
    kept = [str(c) for c in (current_features or [])]
    record = getattr(provenance, "feature_selection", None) if provenance else None
    screened_list = screened_candidates(kept, provenance)
    has_record = bool(record and (getattr(record, "candidates_screened", None)
                                  or getattr(record, "n_features_before", 0)))
    screened = len(screened_list)
    if record is not None:
        # A record whose list was never populated still carries the count, and a
        # count is enough for the arithmetic even when the names are gone.
        screened = max(screened, int(getattr(record, "n_features_before", 0) or 0))
    return CandidateCount(screened=max(screened, len(kept)),
                          retained=len(kept),
                          from_record=bool(has_record))


def candidate_phrase(count: CandidateCount) -> str:
    """`'40 candidate predictors, 8 of which were retained after selection'`.

    Where nothing was dropped this is the plain count, because *"8 candidate
    predictors, 8 of which were retained"* invites a reader to look for a
    screening step that did not happen.
    """
    noun = "candidate predictor" if count.screened == 1 else "candidate predictors"
    if count.dropped > 0:
        return (f"{count.screened} {noun}, {count.retained} of which "
                f"{'was' if count.retained == 1 else 'were'} retained after "
                f"feature selection")
    return f"{count.screened} {noun}"


def more_severe(*levels: Optional[str]) -> Optional[str]:
    """The scarcest of the verdicts given, ignoring anything unrecognized."""
    known = [str(l).strip().lower() for l in levels
             if l and str(l).strip().lower() in SEVERITY_ORDER]
    if not known:
        return None
    return max(known, key=SEVERITY_ORDER.index)


def sufficiency_over_candidates(
    profile_level: Optional[str],
    n_rows: int,
    count: CandidateCount,
    task_type: str,
    minority_class_size: Optional[int] = None,
) -> str:
    """The sufficiency verdict, sized for the predictors that were screened.

    The profile's own verdict is computed with p = the kept count, which §A5.4
    forbids for sizing. This recomputes it with p = `count.screened` using the
    same shipped `assess_data_sufficiency` — no new thresholds — and returns
    whichever of the two is more severe.
    """
    from ml.dataset_profile import assess_data_sufficiency

    level, _ = assess_data_sufficiency(
        int(n_rows), int(max(count.screened, 1)), str(task_type),
        minority_class_size)
    return more_severe(profile_level, level.value) or level.value

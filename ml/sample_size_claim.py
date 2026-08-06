"""What the sample size does and does not support — `AUDIT-022`.

The generated manuscript's **Strengths** list carried
`f"Sample size of {analysis_total:,} observations"` behind one gate,
`if analysis_total > 0`. A study of 40 rows and a study of 40,000 both earned
the same bullet, under a heading that asserts the item is a methodological
strength. A list that says the same thing regardless of the number is not a
claim about this study, and on 40 rows it is a false one: the same document
could print

    **Strengths:**
    - Sample size of 40 observations

    **Limitations:**
    - Sample size may be insufficient (40 rows, 8 features, 5.00:1 samples per feature)

because the limitations half is drawn from the EDA insight ledger and the
strengths half consulted nothing.

## What the registry requires

`research/CLINICAL_SURVEY_PACK.md` §A5.4 requires sample-size adequacy to come
from Riley et al.'s criteria-based calculation over the candidate predictor
**parameters**, the anticipated prevalence and the anticipated model R², and is
[SETTLED] that the events-per-variable rule of 10 is superseded. **This module
does not implement Riley.** `ml/dataset_profile.assess_data_sufficiency` is a
heuristic on n, the predictor-to-sample ratio and events per variable, and it
says so; the Guided door's `turbotab/resolution.py` is where the criteria-based
calculation lives.

So the honest form of the strengths bullet is not a better verdict — it is
**stating which check was run**. Where the heuristic rated the data sufficient,
the bullet says that and names the check. Where it rated it otherwise, or where
no check ran at all, the count is still reported and is simply not claimed as a
strength.

## No threshold moved

`SUPPORTIVE_LEVELS` maps `DataSufficiencyLevel`'s existing five-value vocabulary
onto *may this be called a strength*. It introduces no cutoff:
`assess_data_sufficiency`'s numbers (n<50, n<100, n<500, p/n>0.5, p/n>1.0,
EPV<5, EPV<10) are untouched and are not read here. `AGENT_ONBOARD.md` §08's
second check is about a gate whose quantity moved under pressure; the quantity
this module gates on is the verdict that already existed.

## The shelf is never shortened

Every branch returns a `text` that states the N. The caller decides where it is
printed — under **Strengths** when `is_strength`, as a plain statement of the
count otherwise — but the count is never dropped. `PRODUCT_VISION.md`'s rule:
a sentence that is wrong is corrected, not removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: The `ml.dataset_profile.DataSufficiencyLevel` values whose own narrative says
#: the data are sufficient. `limited`, `scarce` and `critical` are the other
#: three and none of them supports a strength claim.
SUPPORTIVE_LEVELS = ("abundant", "adequate")

#: Every level the check can return, so an unrecognized string is treated as
#: *no verdict was evaluated* rather than silently as an unfavorable one — the
#: two are different states and the sentences differ.
KNOWN_LEVELS = ("abundant", "adequate", "limited", "scarce", "critical")

#: Named once, and it trails the claim rather than interrupting it. The
#: manuscript must say WHICH check produced the verdict it is quoting, because
#: §A5.4 is [SETTLED] that the events-per-variable heuristic is superseded by
#: Riley et al.'s criteria-based calculation — so a reader who sees "adequate"
#: and assumes Riley has been told something false by omission.
_WHICH_CHECK = ("That verdict is a heuristic on the sample size, the "
                "predictor-to-sample ratio and events per variable; no "
                "criteria-based minimum sample size (Riley et al.) was "
                "computed for this analysis.")


@dataclass(frozen=True)
class SampleSizeClaim:
    """One sentence about the sample size, and where it may be printed.

    `text` always states the count. `is_strength` is True only when a
    sufficiency verdict was evaluated and was favorable; a caller that prints
    `text` under a **Strengths** heading without checking it is making the
    claim this module exists to stop.
    """
    text: str
    is_strength: bool
    basis: str


def _population(analysis_total: int, cohort_column: str, cohort_value) -> str:
    """The count, with the group it is a count OF where the run is restricted.

    A restricted N read as the study's N is the defect
    `tests/test_manuscript_discloses_cohort.py` was filed for; the disclosure
    travels with the number here rather than being re-derived by each caller.
    """
    base = f"{analysis_total:,} observations"
    if cohort_column:
        return (f"{base} within {cohort_column} = {cohort_value} "
                f"(the analysis was restricted to this group)")
    return base


def _density(analysis_total: int, n_candidate_predictors: Optional[int]) -> str:
    """Observations per candidate predictor, or nothing.

    §A5.4's ⚠ clause: candidate predictors count toward sample size even if
    they are later dropped, so the denominator is the number **screened**.
    `ml.candidate_predictors` is what supplies it; where the caller cannot, this
    returns the empty string rather than a ratio computed over the kept set,
    because a ratio against the wrong denominator is `AUDIT-023`.
    """
    if not n_candidate_predictors or n_candidate_predictors <= 0:
        return ""
    ratio = analysis_total / n_candidate_predictors
    shown = f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"
    return (f" — {shown} observations per candidate predictor over the "
            f"{n_candidate_predictors:,} screened")


def sample_size_claim(
    analysis_total: int,
    *,
    sufficiency: Optional[str] = None,
    n_candidate_predictors: Optional[int] = None,
    cohort_column: str = "",
    cohort_value: object = "",
    verdict_scope_note: str = "",
) -> Optional[SampleSizeClaim]:
    """The sample-size sentence this run has earned.

    `sufficiency` is a `DataSufficiencyLevel.value`. `verdict_scope_note`
    describes the rows the verdict was computed on — page 02 quarantines the
    sealed test rows out of the profile, so the verdict usually describes the
    training rows while `analysis_total` counts all of them, and a verdict whose
    population is unstated will be read as being about everyone.

    Returns `None` for an empty cohort, matching the caller's existing
    `if analysis_total > 0` gate: there is no count to state.
    """
    if not analysis_total or analysis_total <= 0:
        return None

    population = _population(analysis_total, cohort_column, cohort_value)
    density = _density(analysis_total, n_candidate_predictors)
    level = (sufficiency or "").strip().lower()
    scope = f", computed on {verdict_scope_note}" if verdict_scope_note else ""

    if level in SUPPORTIVE_LEVELS:
        return SampleSizeClaim(
            text=(f"Sample size of {population}{density}, which the "
                  f"data-sufficiency check rated {level}{scope}. "
                  f"{_WHICH_CHECK}"),
            is_strength=True,
            basis=f"data-sufficiency verdict: {level}")

    if level in KNOWN_LEVELS:
        return SampleSizeClaim(
            text=(f"Sample size of {population}{density}. The "
                  f"data-sufficiency check rated this {level}{scope}, so the "
                  f"count is stated here and is not claimed as a "
                  f"methodological strength."),
            is_strength=False,
            basis=f"data-sufficiency verdict: {level}")

    return SampleSizeClaim(
        text=(f"Sample size of {population}{density}. No sample-size criterion "
              f"was evaluated for this analysis, so the count is stated "
              f"without a verdict on whether it is adequate."),
        is_strength=False,
        basis="no sample-size criterion was evaluated")

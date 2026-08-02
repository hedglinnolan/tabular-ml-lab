"""The clinical pack's import detectors — `research/CLINICAL_SURVEY_PACK.md` §A1.

Until L41 this pack held **one prior and zero detectors** against a 1,209-line
research file, and `packs.py` carried a comment arguing that the thinness was
the point because physiologic bounds and unit harmonization already live in the
core. That argument was true of **§A1.2** and it was never true of **§A1.3**,
which describes machinery nothing in this repository has: censoring tokens,
detection limits inferred from the data, columns typed as text because a
qualifier is embedded in them, and number formats that make a numeric column
parse as a string.

## What is here, hardest-first

`LOOP.md` §02: *order a batch hardest-first, judging hardest by what is most
likely to break the abstraction rather than by effort.* The ordering below is
that judgment, and the first entry earned its place — it is the one that bent
the contract.

1. **`censored_values`** (§A1.3). A **per-analyte table**, not one reading. Every
   other detector in this repository answers one question about a table; this one
   answers the same question separately for each analyte and has to carry the
   answers as rows. See `CensoringReading` and the note on the contract below.
2. **`text_numeric`** (§A1.3) — a column typed as text that is >80%
   numeric-parseable. Near-certain evidence of an embedded qualifier.
3. **`mixed_result_type`** (§A1.3) — quantitative and qualitative results in one
   column. A troponin holding both `0.04` and `negative`, which is the one a
   generic profiler reads as a clean categorical.
4. **`mixed_units`** (§A1.1) — a bimodal analyte at a known conversion ratio.
   **A hard stop.**
5. **`default_value_mass`** (§03b) — excess mass at 120/80, 98.6 and 0.
6. **`temporal_implausibility`** (§A1.2) — the trajectory rather than the value.
7. **`number_format`** (§A1.3) — thousands separators, European decimal commas,
   scientific notation.

And the coaching sentence the pack exists for, `impossible_vs_extreme` (§03b),
which is a **correction to behavior the app ships** rather than a new capability.

## What the batch did to the detector contract

`Pack.detectors` is `Callable[[DataFrame], Optional[Dict]]` and L28 split it
rather than widening it, on the argument that widening rests on one example.
**Seven examples later it still holds**, and the one that looked like it would
break it did not:

`censored_values` produces a table with one row per analyte, and the temptation
was to emit one finding per analyte. That would have been wrong for a reason
that has nothing to do with the contract — `nutrition.atwater_finding` already
learned it at `GUIDED-058`: **a varying finding id cannot be bound to
anything.** `LooksFor` names an id, `prior_columns` looks a detector up by
`f["id"] == detector`, and both would need to know N spellings of one finding.
So the per-analyte table lives in `params["analytes"]`, the finding is one
finding, and `affected_columns` carries the analytes. The contract did not bend;
the *shape of the payload* did, and it was already allowed to.

**The one thing the contract genuinely cannot express** is the recorded purpose,
and that is deliberate rather than a gap. See `substitution_position` below.

## Where the field stands, per claim

§A1.3 states three positions and this module carries all three without
collapsing them, because collapsing them is precisely what the evidence badge
exists to prevent:

- The **10% warning threshold is CONVENTION.** The research says so in its own
  words — *"TurboTab uses 10% as its warning threshold, which is a convention,
  not a proven cutoff."*
- The **substitution question is DISPUTED**, and the dispute has a shape rather
  than being a shrug: below roughly 5–10% censoring, LOD/√2 is widely used and
  rarely changes conclusions materially; above ~20% it is not defensible.
- The **prediction/inference asymmetry is CONVENTION** — well-argued, not
  formally settled. *Below detection* is real, reproducible clinical information
  available at deployment, so a censoring indicator plus a substituted value is
  often defensible for prediction and never for an unbiased exposure–outcome
  estimate.

And one thing that is not a position at all: **`TNTC` and `QNS` are not
censoring.** Too numerous to count and quantity not sufficient are measurement
*failures* — the assay produced no number — and treating them as extreme values
would put a number where none exists.

## Every number here is cited, and where one is not, it says so

The 10% threshold, the 5–10% and 20% band, the 80% parse rate and the >5 cm /
±30% / 30-day temporal limits are all the research's own and ship as stated.
**`_DEFAULT_MASS_LIMIT` is not.** §03b names the artifact — *excess mass at
120/80, 98.6, 0 and at round numbers* — and states no cut point for how much
mass is excess, exactly as `nutrition.DRIFT_LIMIT` found for the drift row. So
0.05, and the choice to measure it against the column's own second-most-common
value rather than against a uniform expectation, are this module's own. That is
the same gap `GUIDED-061` records for nutrition and it is recorded here rather
than left for a bare-number scan that structurally cannot see it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from turbotab.packs import (CONVENTION_STATUS, DISPUTED, SETTLED, Claim,
                            Evidence, _finding)

CLINICAL = "clinical"


# ═════════════════════════════════════════════════════════════════════════════
# 1 · §A1.3 · censored lab values — the per-analyte table
# ═════════════════════════════════════════════════════════════════════════════

CENSORING_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value formats and censored values")

#: The warning threshold, and it is **CONVENTION**. The research names it as one
#: in its own sentence, so the badge is not an interpretation.
CENSORING_WARN = 0.10

#: The band the substitution dispute lives in. Below the first, LOD/√2 is widely
#: used and rarely changes conclusions materially; above the second it is not
#: defensible. Both are the research's.
SUBSTITUTION_DEFENSIBLE_BELOW = 0.05
SUBSTITUTION_INDEFENSIBLE_ABOVE = 0.20

#: §A1.3's own token list, verbatim, plus the case and spacing variants a real
#: extract carries. Split by what the token MEANS, because that is the split the
#: research makes and it decides the routing: left censoring is a value that
#: exists and is below the limit; a measurement failure is no value at all.
_LEFT_TOKENS = ("<lod", "<lloq", "undetectable", "not detected", "nondetectable",
                "non-detectable", "trace", "below detection", "bld")
_RIGHT_TOKENS = (">uloq", ">ll", "above range", "above reportable")

#: **NOT CENSORING.** Too numerous to count, quantity not sufficient, and the
#: rest are measurement failures — the specimen was unusable or the count was
#: uncountable. §A1.3: *"Treat them as missing, not as extreme values."*
_FAILURE_TOKENS = ("tntc", "qns", "hemolyzed", "hemolysed", "see comment",
                   "pending", "cancelled", "canceled", "lipemic", "clotted",
                   "insufficient", "unsuitable")

#: `<0.3`, `≤ 0.30`, `> 1500`, `≥1500`. The relational operator is the signal and
#: the number beside it is the limit.
_RELATIONAL = re.compile(r"^\s*(<=|>=|<|>|≤|≥)\s*([0-9]*\.?[0-9]+)\s*$")


@dataclass(frozen=True)
class AnalyteCensoring:
    """One row of §A1.3's censoring summary table.

    The research specifies the table's columns exactly — *analyte | n | % below
    LOD | LOD value(s) | % above ULOQ | handling chosen* — and this is those
    columns with the last one absent, because **handling is the user's and this
    module never chooses it.**
    """
    column: str
    n: int
    n_left: int
    n_right: int
    n_failure: int
    #: The detection limit, *"usually inferable as the modal `<X` value"* —
    #: §A1.3's own phrasing, and the modal one rather than the minimum because a
    #: single mistyped `<0.03` beside two hundred `<0.3` would otherwise move it
    #: by an order of magnitude.
    detection_limit: Optional[float]
    #: Every distinct limit seen, so a column with two of them is visible rather
    #: than averaged. Two detection limits in one analyte is an assay change
    #: mid-study, which is a real finding and not a rounding problem.
    limits_seen: Tuple[float, ...]
    upper_limit: Optional[float]
    failure_tokens: Tuple[str, ...]

    @property
    def left_fraction(self) -> float:
        return (self.n_left / self.n) if self.n else 0.0

    @property
    def right_fraction(self) -> float:
        return (self.n_right / self.n) if self.n else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyte": self.column,
            "n": self.n,
            "n_below_lod": self.n_left,
            "pct_below_lod": round(100.0 * self.left_fraction, 1),
            "detection_limit": self.detection_limit,
            "detection_limits_seen": list(self.limits_seen),
            "n_above_uloq": self.n_right,
            "pct_above_uloq": round(100.0 * self.right_fraction, 1),
            # CARRIED SEPARATELY AND NAMED, because the whole point is that
            # these are not censoring. A summary that folded them into
            # `n_below_lod` would be the app asserting a detection limit that
            # was never reached.
            "n_measurement_failure": self.n_failure,
            "measurement_failure_tokens": list(self.failure_tokens),
        }


def _cells(series: pd.Series) -> List[str]:
    return [str(v).strip() for v in series.dropna()]


def read_censoring(series: pd.Series) -> Optional[AnalyteCensoring]:
    """One analyte's censoring, or `None` where the column carries none.

    `None` rather than a zero row, and the distinction matters at the boundary:
    a summary table listing every numeric column with 0% censored would bury the
    two analytes that have it, which is the presentation §A1.3 asks for turned
    into the presentation it warns against.
    """
    cells = _cells(series)
    if not cells:
        return None

    n_left = n_right = n_failure = 0
    limits: List[float] = []
    upper: List[float] = []
    failures: List[str] = []
    for raw in cells:
        low = raw.lower()
        if any(token in low for token in _FAILURE_TOKENS):
            n_failure += 1
            for token in _FAILURE_TOKENS:
                if token in low and token not in failures:
                    failures.append(token)
            continue
        match = _RELATIONAL.match(raw)
        if match:
            operator, number = match.group(1), float(match.group(2))
            if operator in ("<", "<=", "≤"):
                n_left += 1
                limits.append(number)
            else:
                n_right += 1
                upper.append(number)
            continue
        if any(token in low for token in _LEFT_TOKENS):
            n_left += 1
            continue
        if any(token in low for token in _RIGHT_TOKENS):
            n_right += 1

    if not (n_left or n_right or n_failure):
        return None

    # THE MODAL `<X`, which is §A1.3's own inference rule. `None` where nothing
    # numeric was attached — `undetectable` is censoring and carries no limit,
    # and inventing one from the column's minimum would be the app supplying a
    # number the assay did not.
    detection_limit = None
    if limits:
        counts: Dict[float, int] = {}
        for value in limits:
            counts[value] = counts.get(value, 0) + 1
        detection_limit = max(sorted(counts), key=lambda v: (counts[v], -v))
    return AnalyteCensoring(
        column=str(series.name), n=len(cells), n_left=n_left, n_right=n_right,
        n_failure=n_failure, detection_limit=detection_limit,
        limits_seen=tuple(sorted(set(limits))),
        upper_limit=(min(upper) if upper else None),
        failure_tokens=tuple(failures))


def censoring_table(df: pd.DataFrame) -> List[AnalyteCensoring]:
    """§A1.3's censoring summary table, one row per analyte that has any.

    Ordered by censored fraction, worst first — the analyte whose conclusions
    are most at risk is the one a reader should meet first, and alphabetical
    order would bury it behind whatever starts with an `a`.
    """
    rows: List[AnalyteCensoring] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue          # a parsed number cannot still carry its qualifier
        reading = read_censoring(series)
        if reading is not None:
            rows.append(reading)
    return sorted(rows, key=lambda r: (-r.left_fraction, r.column))


def substitution_position(fraction: float) -> Dict[str, str]:
    """Where the field stands on substituting a number for a censored value.

    **DISPUTED, and the dispute has a shape.** Returning a single verdict here
    would be the app picking a side while wearing a badge that says it has not,
    so this returns both positions and the band the fraction falls in.
    """
    if fraction < SUBSTITUTION_DEFENSIBLE_BELOW:
        band = (
            f"At {fraction:.1%} censored this is below the roughly 5% where "
            f"LOD/√2 substitution is widely used and rarely changes conclusions "
            f"materially. That is the position most likely to be uncontested "
            f"here; it is not a proof that substitution is correct.")
    elif fraction <= SUBSTITUTION_INDEFENSIBLE_ABOVE:
        band = (
            f"At {fraction:.1%} censored this sits inside the contested band. "
            f"Below roughly 5–10% substitution is widely used; above about 20% "
            f"it is not defensible. Both positions are live at this fraction "
            f"and the app is not choosing between them.")
    else:
        band = (
            f"At {fraction:.1%} censored, substitution is above the ~20% the "
            f"literature calls indefensible. Maximum likelihood or censored "
            f"regression assuming a lognormal distribution, or "
            f"distribution-based multiple imputation of the sub-limit values, "
            f"is what this fraction needs.")
    return {
        "band": band,
        "for_substitution": (
            "Simple substitution (LOD, LOD/2, LOD/√2) is the common practice, "
            "it is trivially reproducible, and at low censored fractions it "
            "rarely moves a conclusion."),
        "against_substitution": (
            "Substitution biases estimates and the bias grows with the "
            "censored fraction. Every substituted value is a number the assay "
            "did not produce, plotted as if it had been measured."),
    }


#: **The prediction/inference asymmetry, and it is CONVENTION.** §A1.3 marks it
#: *well-argued, not formally settled*, and this is the sentence that changes
#: with the recorded purpose rather than with the data.
#:
#: **Why this is a function of `purpose` and not a detector branch.** The
#: detector contract takes a DataFrame, and that is correct here rather than
#: limiting: the censored fraction is a fact about the table and is the same
#: under either objective. What inverts is the *handling*, and handling is a
#: decision — so this is read at the decision, beside
#: `purpose.blocks_indicator`, which is the same shape for the same reason.
PREDICTION_ASYMMETRY = (
    "`{column}` is left-censored at {limit}, and *below detection* is real, "
    "reproducible clinical information that is available at deployment too. "
    "Under a prediction objective a censoring indicator plus a substituted "
    "value is often defensible for exactly that reason. Under an inference "
    "objective it is not: the same column, the same data, and the opposite "
    "answer, because a substituted value biases the exposure–outcome estimate "
    "and the bias grows with the censored fraction."
)

INFERENCE_SUBSTITUTION_BLOCK = (
    "You said this model is for estimating how strongly `{column}` is "
    "associated with the outcome, and {pct:.0%} of its values are censored at "
    "the assay's limit. Substituting a number for them biases that estimate, "
    "and the bias grows with the censored fraction.\n\n"
    "Under a prediction objective the same substitution is often defensible — "
    "*below detection* is observable at the bedside too — which is why this is "
    "a question about your objective rather than about the column. Maximum "
    "likelihood or censored regression assuming a lognormal distribution, or "
    "distribution-based multiple imputation of the sub-limit values, is what an "
    "association estimate wants. If you have a reason to substitute anyway, say "
    "so and it is recorded as a stated limitation."
)


def blocks_substitution(purpose: Optional[str], fraction: float) -> bool:
    """Whether the recorded purpose contraindicates substituting these values.

    **`None` — the question unanswered — blocks nothing**, exactly as
    `purpose.blocks_indicator` does. The app does not get to infer an objective
    and then hold somebody to it.

    Gated on the fraction as well as the purpose, because the research does not
    say substitution is always wrong for inference — it says the bias grows with
    the fraction and that below roughly 5% it rarely matters. Blocking at 2%
    would make the app more confident than the literature it cites.
    """
    from turbotab import purpose as _purpose

    return (purpose == _purpose.INFERENCE
            and fraction >= SUBSTITUTION_DEFENSIBLE_BELOW)


def substitution_blocker(column: str, fraction: float) -> Dict[str, Any]:
    """The block, in the shape `purpose.indicator_blocker` already established."""
    from turbotab import exits as _exits

    return {
        "kind": "substitution_under_inference",
        "column": column,
        "message": INFERENCE_SUBSTITUTION_BLOCK.format(column=column,
                                                       pct=fraction),
        "exits": [
            {"id": "keep_censored", "kind": "resolve",
             "label": "Leave the censored values as missing",
             "detail": "No number is invented. The column keeps its blanks and "
                       "the censoring indicator records which rows were below "
                       "the limit."},
            dict(_exits.attest(
                "Substitute anyway — I know what this censoring is",
                "Recorded as a stated limitation: the association estimate is "
                "conditioned on substituted sub-limit values, and the methods "
                "section says so.",
                _exits.ACKNOWLEDGE_SIGNAL_LOSS)),
        ],
        "acknowledgment_kind": "typed",
        # CONVENTION rather than SETTLED, because §A1.3 marks the asymmetry
        # *well-argued, not formally settled* and a SETTLED badge here would be
        # the app being more certain than its own source.
        "evidence_status": CONVENTION_STATUS,
        "source": ("research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value formats "
                   "and censored values"),
    }


def censored_values_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """§A1.3, as one finding carrying a per-analyte table."""
    table = censoring_table(df)
    if not table:
        return None

    censored = [row for row in table if row.n_left or row.n_right]
    failures_only = [row for row in table if not (row.n_left or row.n_right)]
    worst = max((row.left_fraction for row in censored), default=0.0)

    if censored:
        lead = censored[0]
        limit = ("its detection limit" if lead.detection_limit is None
                 else f"{lead.detection_limit:g}")
        headline = (
            f"{len(censored)} analyte{'' if len(censored) == 1 else 's'} carry "
            f"censored values")
        detail = (
            f"`{lead.column}` records {lead.left_fraction:.1%} of its values as "
            f"below {limit} — left-censored at the assay's limit of detection. "
            f"TurboTab has not substituted a number for them.")
        if lead.n_right:
            detail += (
                f" A further {lead.right_fraction:.1%} are above the upper "
                f"limit of quantitation.")
        if len(censored) > 1:
            detail += (
                f" The same reading applies to "
                f"`{'`, `'.join(r.column for r in censored[1:])}`; the table "
                f"below carries each analyte's fraction and limit separately, "
                f"because a detection limit is a property of an assay and not "
                f"of a study.")
    else:
        lead = failures_only[0]
        headline = "Measurement failures are recorded as text in a lab column"
        detail = (
            f"`{lead.column}` carries {'`, `'.join(lead.failure_tokens)} in "
            f"place of a number.")

    if failures_only or any(row.n_failure for row in censored):
        with_failures = [row for row in table if row.n_failure]
        detail += (
            f" `{'`, `'.join(sorted({t for r in with_failures for t in r.failure_tokens}))}` "
            f"are **not** censoring at a detection limit — they are "
            f"measurement failures, and they route to missing rather than to "
            f"an extreme value. Treating them as small numbers would put a "
            f"value where the assay produced none.")

    over_threshold = worst >= CENSORING_WARN
    severity = "warning" if over_threshold else "info"
    position = substitution_position(worst)

    return _finding(
        "pack::clinical::censored_values", severity,
        headline, detail,
        ("A censored value is not missing at random and it is not a small "
         "number: it is a value the assay could not resolve, and every "
         "handling of it changes the estimate. Substitution biases results and "
         "the bias grows with the censored fraction, so which handling is "
         "defensible depends on how much of the column is censored and on "
         "what the model is for."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=CENSORING_EVIDENCE,
        claims=(
            Claim(key="threshold",
                  statement=(f"{CENSORING_WARN:.0%} censored is where this app "
                             f"starts warning"),
                  evidence=Evidence(
                      status=CONVENTION_STATUS,
                      source=("research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value "
                              "formats and censored values"))),
            Claim(key="substitution",
                  statement=position["band"],
                  evidence=Evidence(
                      status=DISPUTED,
                      source=("research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value "
                              "formats and censored values"),
                      both_sides=(position["for_substitution"] + " "
                                  + position["against_substitution"]))),
            Claim(key="purpose_asymmetry",
                  statement=(
                      "Below detection is real information available at "
                      "deployment, so a censoring indicator plus a substituted "
                      "value is often defensible for prediction and never for "
                      "an unbiased exposure-outcome estimate."),
                  evidence=Evidence(
                      status=CONVENTION_STATUS,
                      source=("research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value "
                              "formats and censored values"))),
        ),
        columns=[row.column for row in table],
        params={
            # §A1.3'S CENSORING SUMMARY TABLE, as data. One row per analyte,
            # because a detection limit belongs to an assay: folding two
            # analytes into one fraction would report a limit neither of them
            # has.
            "analytes": [row.to_dict() for row in table],
            "worst_censored_fraction": round(worst, 4),
            "warn_threshold": CENSORING_WARN,
            "over_warn_threshold": over_threshold,
            "substitution_band": position["band"],
            "substitution_defensible_below": SUBSTITUTION_DEFENSIBLE_BELOW,
            "substitution_indefensible_above": SUBSTITUTION_INDEFENSIBLE_ABOVE,
            "prediction_asymmetry": PREDICTION_ASYMMETRY.format(
                column=(censored[0].column if censored else lead.column),
                limit=("its detection limit"
                       if not censored or censored[0].detection_limit is None
                       else f"{censored[0].detection_limit:g}")),
        },
        # NO REPAIR. Every handling here is a decision the user makes with
        # information the table does not contain, which is `DOMAIN_SCIENCE.md`
        # §01.2's litmus exactly: the data cannot distinguish the causes.
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 2 · §A1.3 · a text column that is mostly numbers
# ═════════════════════════════════════════════════════════════════════════════

TEXT_NUMERIC_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value formats and censored values")

#: §A1.3's own rate: *"columns typed as text that are >80% numeric-parseable — a
#: near-certain sign of embedded qualifiers."*
NUMERIC_PARSE_RATE = 0.80

#: Below this many values the rate is not a rate. Not the research's — it says
#: nothing about a minimum n — so it is this module's own, chosen because 20
#: values is where one stray cell stops being 33% of the column.
_MIN_CELLS = 20


def _parses(text: str) -> bool:
    try:
        float(text.replace(",", "").replace(" ", ""))
        return True
    except ValueError:
        return False


def numeric_parse_rate(series: pd.Series) -> Optional[Tuple[float, List[str]]]:
    """`(rate, the values that did not parse)`, or `None` on a numeric column."""
    if pd.api.types.is_numeric_dtype(series):
        return None
    cells = _cells(series)
    if len(cells) < _MIN_CELLS:
        return None
    unparsed = [c for c in cells if not _parses(c)]
    return (len(cells) - len(unparsed)) / len(cells), unparsed


def text_numeric_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Columns typed as text that are mostly numbers, with what blocked them."""
    hits: List[Dict[str, Any]] = []
    for column in df.columns:
        read = numeric_parse_rate(df[column])
        if read is None:
            continue
        rate, unparsed = read
        if rate < NUMERIC_PARSE_RATE or not unparsed:
            continue
        distinct = sorted({u for u in unparsed})
        hits.append({
            "column": str(column),
            "numeric_parse_rate": round(rate, 4),
            "n_unparsed": len(unparsed),
            "blocking_values": distinct[:8],
            "n_distinct_blocking_values": len(distinct),
        })
    if not hits:
        return None

    hits.sort(key=lambda h: (-h["numeric_parse_rate"], h["column"]))
    lead = hits[0]
    return _finding(
        "pack::clinical::text_numeric", "warning",
        (f"{len(hits)} column{'' if len(hits) == 1 else 's'} arrived as text "
         f"and {'is' if len(hits) == 1 else 'are'} mostly numbers"),
        (f"`{lead['column']}` parses as a number in "
         f"{lead['numeric_parse_rate']:.1%} of its rows. The "
         f"{lead['n_distinct_blocking_values']} value"
         f"{'' if lead['n_distinct_blocking_values'] == 1 else 's'} that stop "
         f"it are `{'`, `'.join(str(v) for v in lead['blocking_values'])}`"
         + ("" if lead["n_distinct_blocking_values"] <= 8 else " and others")
         + ". A column above 80% numeric-parseable is near-certain evidence of "
           "a qualifier embedded in the result rather than a genuinely "
           "categorical measurement."
         + ("" if len(hits) == 1 else
            f" The same reading applies to "
            f"`{'`, `'.join(h['column'] for h in hits[1:])}`.")),
        ("The whole column is typed as text because of a handful of cells, so "
         "every numeric summary, every plot and every model silently treats a "
         "continuous laboratory measurement as a category. The values that "
         "block the parse are the finding — they are what the column is "
         "actually recording besides a number."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=TEXT_NUMERIC_EVIDENCE,
        columns=[h["column"] for h in hits],
        params={"columns": hits, "parse_rate_threshold": NUMERIC_PARSE_RATE},
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 3 · §A1.3 · quantitative and qualitative results in one column
# ═════════════════════════════════════════════════════════════════════════════

MIXED_RESULT_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value formats and censored values")

#: §A1.3's own example is a troponin holding both `0.04` and `negative`. The
#: vocabulary is the qualitative result set a lab actually reports, and it is
#: deliberately NOT the censoring set — `<0.3` is a quantitative result with a
#: bound on it, `negative` is a different kind of answer entirely.
_QUALITATIVE = ("negative", "positive", "reactive", "non-reactive",
                "nonreactive", "normal", "abnormal", "detected",
                "not detected", "equivocal", "indeterminate", "borderline",
                "present", "absent", "reads negative", "neg", "pos")


def _qualitative_cells(series: pd.Series) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw in _cells(series):
        low = raw.lower()
        for token in _QUALITATIVE:
            if low == token:
                counts[token] = counts.get(token, 0) + 1
                break
    return counts


def mixed_result_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A result column carrying both a number and a verdict.

    **The one a generic profiler gets exactly backwards.** A column of
    `0.04`/`0.31`/`negative` reads as a low-cardinality categorical to anything
    counting distinct values, so it is one-hot encoded and the numbers are
    thrown away — the opposite of the correct handling, applied confidently.
    """
    hits: List[Dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        cells = _cells(series)
        if len(cells) < _MIN_CELLS:
            continue
        qualitative = _qualitative_cells(series)
        if not qualitative:
            continue
        n_numeric = sum(1 for c in cells if _parses(c))
        if not n_numeric:
            continue      # wholly qualitative is a categorical, correctly
        hits.append({
            "column": str(column),
            "n": len(cells),
            "n_quantitative": n_numeric,
            "n_qualitative": sum(qualitative.values()),
            "qualitative_values": sorted(qualitative),
        })
    if not hits:
        return None

    hits.sort(key=lambda h: (-h["n_qualitative"], h["column"]))
    lead = hits[0]
    return _finding(
        "pack::clinical::mixed_result_type", "warning",
        (f"`{lead['column']}` records both a measured value and a verdict"),
        (f"{lead['n_quantitative']:,} of its rows carry a number and "
         f"{lead['n_qualitative']:,} carry "
         f"`{'`, `'.join(lead['qualitative_values'])}`. Those are two different "
         f"kinds of result in one field — a quantitative assay and a "
         f"qualitative one — and they cannot be compared on any scale."
         + ("" if len(hits) == 1 else
            f" The same is true of "
            f"`{'`, `'.join(h['column'] for h in hits[1:])}`.")),
        ("A generic profiler reads this as a categorical, because that is what "
         "counting distinct values says. It is then one-hot encoded and every "
         "measured number in the column is discarded — the opposite of the "
         "right handling, applied with no signal that anything was lost. "
         "Splitting the field into a value and a qualitative result is a "
         "decision about what the two assays mean, which the table cannot "
         "answer."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=MIXED_RESULT_EVIDENCE,
        columns=[h["column"] for h in hits],
        params={"columns": hits},
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 4 · §A1.1 · a bimodal analyte at a known conversion ratio — A HARD STOP
# ═════════════════════════════════════════════════════════════════════════════

MIXED_UNITS_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#A1.1 Unit harmonization")

#: §A1.1's conversion table, verbatim, plus §03b's three physical ones. **These
#: are the analytes whose factor is a constant.** The reason the list is short
#: and closed is the hard stop itself: conversion depends on molecular weight
#: for many analytes — most drug levels, most hormones — and official LOINC/UCUM
#: services return errors for a missing molecular weight. An open-ended list
#: would be this module inventing factors it cannot verify.
CONVERSIONS: Tuple[Tuple[str, str, float, str, str], ...] = (
    ("glucose", r"gluc|glc\b", 18.0, "mmol/L", "mg/dL"),
    ("creatinine", r"creat|\bcr\b", 88.4, "mg/dL", "µmol/L"),
    ("total cholesterol", r"cholesterol|\bchol\b|\btc\b", 38.67, "mmol/L", "mg/dL"),
    ("triglycerides", r"triglyc|\btg\b", 88.57, "mmol/L", "mg/dL"),
    ("bilirubin", r"bilirubin|\btbili\b", 17.1, "mg/dL", "µmol/L"),
    ("calcium", r"calcium|\bca\b", 4.0, "mmol/L", "mg/dL"),
    ("hemoglobin", r"hemoglobin|haemoglobin|\bhgb\b|\bhb\b", 10.0, "g/dL", "g/L"),
    ("height", r"height|stature|\bht\b", 2.54, "inches", "cm"),
    ("weight", r"weight|\bwt\b|mass", 2.205, "kg", "lb"),
    ("temperature", r"temp\b|temperature", None, "°C", "°F"),
)

#: How close the observed ratio has to sit to the tabled factor. Not the
#: research's — §A1.1 says *"flag when component means differ by a ratio near a
#: known conversion factor"* and never says how near — so 8% is this module's
#: own. Chosen because the tightest neighboring pair in the table above,
#: 88.4 and 88.57, are 0.2% apart, so any tolerance that separates a real
#: conversion from noise separates those two as well; and because a real
#: two-site split reproduces its factor to within rounding.
_RATIO_TOLERANCE = 0.08

#: A mode holding less than this share of the column is a tail, not a second
#: population. This module's own, for the same reason.
_MIN_MODE_SHARE = 0.08


@dataclass(frozen=True)
class UnitSplit:
    column: str
    analyte: str
    ratio: float
    factor: float
    low_median: float
    high_median: float
    n_low: int
    n_high: int
    unit_low: str
    unit_high: str

    def to_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "analyte": self.analyte,
                "observed_ratio": round(self.ratio, 3),
                "tabled_factor": self.factor,
                "median_low": self.low_median, "median_high": self.high_median,
                "n_low": self.n_low, "n_high": self.n_high,
                "implied_unit_low": self.unit_low,
                "implied_unit_high": self.unit_high}


def _split_at_gap(values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The two clusters, split at the widest gap on the log scale.

    A log-scale gap rather than a fitted mixture, and the reason is §A1.1's own:
    the signature being looked for is *component means differing by a ratio*, so
    the quantity is a ratio and the natural scale is the log. Two clusters an
    exact factor apart are two clusters a constant apart in logs, which a gap
    finds without an optimizer, without a seed, and deterministically — which
    the Router's determinism requirement makes a hard requirement rather than a
    convenience.
    """
    positive = values[values > 0]
    if len(positive) < 40:
        return None
    logs = np.sort(np.log(positive))
    gaps = np.diff(logs)
    if not len(gaps):
        return None
    where = int(np.argmax(gaps))
    low, high = np.exp(logs[:where + 1]), np.exp(logs[where + 1:])
    n = len(positive)
    if min(len(low), len(high)) < max(4, int(_MIN_MODE_SHARE * n)):
        return None
    return low, high


def read_unit_split(series: pd.Series) -> Optional[UnitSplit]:
    """Whether this column looks like one analyte reported in two units."""
    name = str(series.name).lower()
    for analyte, pattern, factor, unit_low, unit_high in CONVERSIONS:
        if not re.search(pattern, name):
            continue
        if factor is None:
            # TEMPERATURE IS NOT A RATIO. °F = °C × 1.8 + 32 is affine, so a
            # log-scale ratio test does not apply to it and pretending it does
            # would produce a factor nobody can check. Handled by its own
            # reading below, and skipped here rather than approximated.
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
        split = _split_at_gap(values)
        if split is None:
            continue
        low, high = split
        low_median, high_median = float(np.median(low)), float(np.median(high))
        if low_median <= 0:
            continue
        ratio = high_median / low_median
        if abs(ratio - factor) / factor > _RATIO_TOLERANCE:
            continue
        return UnitSplit(
            column=str(series.name), analyte=analyte, ratio=ratio,
            factor=factor, low_median=round(low_median, 3),
            high_median=round(high_median, 3), n_low=len(low), n_high=len(high),
            unit_low=unit_low, unit_high=unit_high)
    return None


def mixed_units_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A bimodal analyte at a tabled conversion ratio. **Never converted.**

    `DOMAIN_SCIENCE.md` §01.2's first hard stop, and the reason is stated rather
    than implied: the detection is unambiguous and the action is forbidden,
    because conversion depends on molecular weight for many analytes and the
    official LOINC/UCUM services themselves error on it. High-confidence
    detection, irreversible-if-wrong action, and no signal in the data that
    resolves it — declaration separated from execution.
    """
    splits = [s for s in (read_unit_split(df[c]) for c in df.columns)
              if s is not None]
    if not splits:
        return None

    lead = splits[0]
    return _finding(
        "pack::clinical::mixed_units", "critical",
        (f"`{lead.column}` holds two populations a factor of "
         f"{lead.factor:g} apart"),
        (f"Its values split into {lead.n_low:,} around {lead.low_median:g} and "
         f"{lead.n_high:,} around {lead.high_median:g} — an observed ratio of "
         f"{lead.ratio:.2f} against the {lead.unit_low}-to-{lead.unit_high} "
         f"conversion factor of {lead.factor:g}. That almost always means two "
         f"sites or two eras reported different units into the same field. "
         f"**TurboTab has not converted anything and will not.** Conversion "
         f"depends on molecular weight for many analytes, and the official "
         f"LOINC/UCUM services return errors for a missing one — so the app "
         f"detects this, states it, and requires you to confirm the units per "
         f"analyte against the source data dictionary."
         + ("" if len(splits) == 1 else
            f" `{'`, `'.join(s.column for s in splits[1:])}` split the same way.")),
        ("A mixed-unit predictor is not a noisy predictor. It is a variable "
         "whose meaning changes between rows, and no amount of regularization "
         "repairs that — the model learns a relationship that is true of "
         "neither population. It also passes every distributional check the app "
         "has, because two clean unimodal populations at a constant ratio look "
         "like one skewed one."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=MIXED_UNITS_EVIDENCE,
        columns=[s.column for s in splits],
        params={
            "columns": [s.to_dict() for s in splits],
            "ratio_tolerance": _RATIO_TOLERANCE,
            # NAMED AS A HARD STOP IN THE PAYLOAD, not only in the prose.
            # `GUIDED-064`'s class: the machine-readable form must not be
            # lossier than the sentence, and *never auto-convert* is the whole
            # content of this finding.
            "hard_stop": "never_auto_convert",
            "hard_stop_because": (
                "Conversion depends on molecular weight for many analytes and "
                "official LOINC/UCUM services error on a missing one. The app "
                "detects and declares; the user executes."),
        },
        # NO REPAIR, and this is the one where the absence is the content.
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 5 · §03b · repeated-digit / default-value mass
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_MASS_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source="research/CLINICAL_SURVEY_PACK.md#A1.2 ★ Reference ranges vs physiological plausibility — the distinction that matters")

#: §03b's own list: *excess mass at 120/80, 98.6, 0, and at round numbers.*
#: A value here is only a finding when it holds far more of the column than the
#: values around it — 120 is also a perfectly ordinary systolic reading.
DEFAULT_VALUES: Dict[str, Tuple[float, ...]] = {
    "bp_sys": (120.0, 0.0),
    "bp_di": (80.0, 0.0),
    "temperature_f": (98.6,),
    "temperature_c": (37.0,),
}

_DEFAULT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("bp_sys", r"sbp|systolic|bp_sys"),
    ("bp_di", r"dbp|diastolic|bp_di"),
    ("temperature_f", r"temp.*_f$|temp_f|fahrenheit"),
    ("temperature_c", r"temp.*_c$|temp_c|celsius"),
)

#: **NOT THE RESEARCH'S.** §03b names the artifact and states no cut point for
#: how much mass is excess, so this and the choice to measure against the
#: column's own runner-up value are this module's own. Recorded here rather than
#: left for a bare-number scan, which structurally cannot tell an invented
#: threshold from a cited one — the same gap `GUIDED-061` holds open for
#: `nutrition.DRIFT_LIMIT`.
_DEFAULT_MASS_LIMIT = 0.05
#: How many times the runner-up a default value has to hold before the spike is
#: a spike rather than a popular reading. Same provenance.
_DEFAULT_MASS_RATIO = 2.0


def read_default_mass(series: pd.Series) -> List[Dict[str, Any]]:
    if not pd.api.types.is_numeric_dtype(series):
        return []
    name = str(series.name).lower()
    kind = next((k for k, pattern in _DEFAULT_PATTERNS
                 if re.search(pattern, name)), None)
    if kind is None:
        return []
    values = series.dropna()
    if len(values) < _MIN_CELLS:
        return []
    counts = values.value_counts()
    out = []
    for candidate in DEFAULT_VALUES[kind]:
        if candidate not in counts.index:
            continue
        share = float(counts[candidate]) / len(values)
        others = counts.drop(index=candidate)
        runner_up = float(others.iloc[0]) / len(values) if len(others) else 0.0
        if share < _DEFAULT_MASS_LIMIT:
            continue
        if runner_up and share < _DEFAULT_MASS_RATIO * runner_up:
            continue
        out.append({"column": str(series.name), "value": float(candidate),
                    "n": int(counts[candidate]), "share": round(share, 4),
                    "next_most_common_share": round(runner_up, 4)})
    return out


def default_value_mass_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Excess mass at 120/80, 98.6 and 0 — an artifact, not a measurement."""
    hits: List[Dict[str, Any]] = []
    for column in df.columns:
        hits.extend(read_default_mass(df[column]))
    if not hits:
        return None

    hits.sort(key=lambda h: (-h["share"], h["column"]))
    lead = hits[0]
    return _finding(
        "pack::clinical::default_value_mass", "warning",
        (f"`{lead['column']}` piles up on {lead['value']:g}"),
        (f"{lead['share']:.1%} of its values are exactly {lead['value']:g}, "
         f"against {lead['next_most_common_share']:.1%} for the next most "
         f"common reading. That is a documented EHR artifact — value preference "
         f"and manual entry rather than measurement."
         + ("" if len(hits) == 1 else
            " The same spike appears at "
            + ", ".join(f"{h['value']:g} in `{h['column']}`" for h in hits[1:])
            + ".")
         + " TurboTab has not removed or replaced any of them: a genuine "
           "120/80 and a transcribed one are the same number, and nothing in "
           "the column separates them."),
        ("A spike at a default value is a measurement that was probably never "
         "taken, and it biases every summary toward the default. It is also "
         "invisible to every distributional check the app has, because a "
         "spike at a plausible value is inside every plausible range. The "
         "count is what makes it actionable — a reviewer asking how vitals "
         "were captured wants this number."),
        confidence="medium", pack=CLINICAL, marker="offered",
        evidence=DEFAULT_MASS_EVIDENCE,
        columns=sorted({h["column"] for h in hits}),
        params={"values": hits, "mass_threshold": _DEFAULT_MASS_LIMIT,
                "ratio_over_runner_up": _DEFAULT_MASS_RATIO,
                # SAID IN THE PAYLOAD. Both numbers above are this module's own
                # and the research states none, so a consumer reading `params`
                # gets the same disclosure a reader of the docstring does.
                "thresholds_are_this_apps_own": True},
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 6 · §A1.2 · temporal plausibility — the trajectory, not the value
# ═════════════════════════════════════════════════════════════════════════════

TEMPORAL_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source="research/CLINICAL_SURVEY_PACK.md#A1.2 ★ Reference ranges vs physiological plausibility — the distinction that matters")

#: §A1.2's own limits, from Kahn et al.'s harmonized framework: *adult height
#: changing >5 cm between visits; weight changing >30% in <30 days.* The
#: research marks the plausibility set as CONVENTION and notes that Kahn's
#: limits are institution- and observation-specific, so the badge is CONVENTION
#: rather than SETTLED.
HEIGHT_JUMP_CM = 5.0
WEIGHT_CHANGE_FRACTION = 0.30
WEIGHT_CHANGE_WINDOW_DAYS = 30

#: **Adult only.** §A1.2 is explicit: *never apply adult bounds to pediatric or
#: growth data; use age-and-sex-specific modified z-score flags.* A 9 cm height
#: change is a finding in a 60-year-old and a normal year in a 12-year-old, so
#: rows below this age are excluded from the height reading and the exclusion is
#: reported rather than silent.
ADULT_AGE = 20

_PERSON_PATTERNS = r"subject_id|patient_id|participant_id|person_id|record_id|\bid$"
_DATE_PATTERNS = r"date|visit_dt|timestamp|collected|drawn"
_HEIGHT_PATTERNS = r"height|stature|\bht_"
_WEIGHT_PATTERNS = r"weight|\bwt_|body_mass"
_AGE_PATTERNS = r"^age$|age_years|age_at"
_DEATH_PATTERNS = r"death|deceased|dod\b|date_of_death"


def _first_matching(df: pd.DataFrame, pattern: str) -> Optional[str]:
    for column in df.columns:
        if re.search(pattern, str(column).lower()):
            return str(column)
    return None


def _as_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


@dataclass
class TemporalReading:
    person_column: str
    date_column: str
    n_people: int
    height_jumps: List[Dict[str, Any]] = field(default_factory=list)
    weight_swings: List[Dict[str, Any]] = field(default_factory=list)
    after_death: List[Dict[str, Any]] = field(default_factory=list)
    pediatric_rows_excluded: int = 0
    #: Cells the CORE already calls physiologically impossible, dropped before
    #: any trajectory is traced. Reported rather than silently subtracted — see
    #: `_atemporally_impossible`.
    impossible_cells_excluded: int = 0

    @property
    def any(self) -> bool:
        return bool(self.height_jumps or self.weight_swings or self.after_death)


def _atemporally_impossible(df: pd.DataFrame) -> Dict[str, set]:
    """Row indices the CORE flags as physiologically impossible, per column.

    **A temporal claim is only about values that are individually believable**,
    and this exists because the first version of this detector was not.

    Kahn et al.'s framework separates atemporal plausibility from temporal
    plausibility, and the split is not decorative: they have different remedies.
    Run against `clinical_longitudinal.csv`, which seeds `height_cm = 0.0` as an
    entry error, the first version reported *"an adult changed height by
    164.7 cm, from 164.7 to 0"*. Every word of that is arithmetic and none of it
    is a trajectory — one cell is an entry error the atemporal check already
    flags, and dressing it as a trajectory would have produced a second finding
    about a defect that already has one, in more alarming language.

    So the impossible cells come out first, and the count of them goes in the
    payload: a trajectory computed over 3 of a person's 4 visits is a different
    claim from one computed over all 4.
    """
    from turbotab import engine

    out: Dict[str, set] = {}
    try:
        report = engine.plausibility(df)
    except Exception:
        return out
    for row in report.get("impossible") or []:
        rows = row.get("all_rows")
        if rows:
            out.setdefault(str(row["column"]), set()).update(int(r) for r in rows)
    return out


def read_temporal(df: pd.DataFrame) -> Optional[TemporalReading]:
    """The trajectory checks §A1.2 asks for, or `None` where the chain is absent.

    **`None` when there is no person column or no date**, because temporal
    plausibility is a claim about a sequence and a cross-section has none. That
    is a refusal to compute rather than a clean result, and the difference
    matters: a table with one row per patient must not come back reporting zero
    implausible trajectories.
    """
    person = _first_matching(df, _PERSON_PATTERNS)
    date_column = _first_matching(df, _DATE_PATTERNS)
    if not person or not date_column:
        return None
    dates = _as_dates(df[date_column])
    if dates.isna().all():
        return None

    impossible = _atemporally_impossible(df)
    work = df.copy()
    work["__when"] = dates
    work = work.dropna(subset=["__when"]).sort_values([person, "__when"])
    if work.groupby(person).size().max() < 2:
        return None            # no repeated measurements: nothing to trace

    reading = TemporalReading(person_column=person, date_column=date_column,
                              n_people=int(work[person].nunique()))

    def _believable(column: str) -> pd.Series:
        """The column with the atemporally-impossible cells blanked."""
        series = work[column].astype(float).copy()
        bad = impossible.get(column) or set()
        if bad:
            hit = series.index.isin(bad)
            reading.impossible_cells_excluded += int(hit.sum())
            series[hit] = np.nan
        return series

    age_column = _first_matching(df, _AGE_PATTERNS)
    height = _first_matching(df, _HEIGHT_PATTERNS)
    if height and pd.api.types.is_numeric_dtype(work[height]):
        traced = work.assign(__value=_believable(height))
        if age_column and pd.api.types.is_numeric_dtype(work[age_column]):
            adult = traced[traced[age_column] >= ADULT_AGE]
            reading.pediatric_rows_excluded = int(len(traced) - len(adult))
            traced = adult
        for who, rows in traced.groupby(person):
            values = rows["__value"].dropna()
            if len(values) < 2:
                continue
            # **CONSECUTIVE VISITS, not max minus min.** §A1.2's rule is
            # *"adult height changing >5 cm between visits"*, and a spread over
            # four visits is a different quantity: 2 cm of measurement noise
            # each visit sums to 6 and reports a jump nobody made.
            steps = values.diff().abs().dropna()
            if steps.empty or float(steps.max()) <= HEIGHT_JUMP_CM:
                continue
            at = int(steps.to_numpy().argmax())
            reading.height_jumps.append({
                "person": str(who), "column": height,
                "from": float(values.iloc[at]), "to": float(values.iloc[at + 1]),
                "change_cm": round(float(steps.iloc[at]), 1)})

    weight = _first_matching(df, _WEIGHT_PATTERNS)
    if weight and pd.api.types.is_numeric_dtype(work[weight]):
        traced = work.assign(__value=_believable(weight))
        for who, rows in traced.groupby(person):
            values = rows[["__value", "__when"]].dropna()
            for i in range(1, len(values)):
                before, after = values.iloc[i - 1], values.iloc[i]
                days = (after["__when"] - before["__when"]).days
                if not (0 < days < WEIGHT_CHANGE_WINDOW_DAYS):
                    continue
                if before["__value"] <= 0:
                    continue
                change = (after["__value"] - before["__value"]) / before["__value"]
                if abs(change) > WEIGHT_CHANGE_FRACTION:
                    reading.weight_swings.append({
                        "person": str(who), "column": weight,
                        "from": float(before["__value"]),
                        "to": float(after["__value"]),
                        "days": int(days), "change": round(float(change), 3)})

    death = _first_matching(df, _DEATH_PATTERNS)
    if death:
        died = _as_dates(df[death])
        if not died.isna().all():
            work["__died"] = died.reindex(work.index)
            late = work.dropna(subset=["__died"])
            late = late[late["__when"] > late["__died"]]
            for _, row in late.iterrows():
                reading.after_death.append({
                    "person": str(row[person]),
                    "measured": str(row["__when"].date()),
                    "died": str(row["__died"].date())})
    return reading


def temporal_implausibility_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Kahn et al.'s temporal half. The app only ever did the atemporal one."""
    reading = read_temporal(df)
    if reading is None or not reading.any:
        return None

    parts: List[str] = []
    if reading.height_jumps:
        worst = max(reading.height_jumps, key=lambda h: h["change_cm"])
        parts.append(
            f"{len(reading.height_jumps)} adult"
            f"{'' if len(reading.height_jumps) == 1 else 's'} change height by "
            f"more than {HEIGHT_JUMP_CM:g} cm between visits — the largest is "
            f"{worst['change_cm']:g} cm, from {worst['from']:g} to "
            f"{worst['to']:g}")
    if reading.weight_swings:
        worst = max(reading.weight_swings, key=lambda w: abs(w["change"]))
        parts.append(
            f"{len(reading.weight_swings)} weight change"
            f"{'' if len(reading.weight_swings) == 1 else 's'} exceed "
            f"{WEIGHT_CHANGE_FRACTION:.0%} inside "
            f"{WEIGHT_CHANGE_WINDOW_DAYS} days — the largest is "
            f"{worst['change']:+.0%} over {worst['days']} days, from "
            f"{worst['from']:g} to {worst['to']:g}")
    if reading.after_death:
        parts.append(
            f"{len(reading.after_death)} measurement"
            f"{'' if len(reading.after_death) == 1 else 's'} are timestamped "
            f"after the recorded date of death")

    detail = "; ".join(parts) + "."
    if reading.impossible_cells_excluded:
        detail += (
            f" {reading.impossible_cells_excluded:,} cell"
            f"{'' if reading.impossible_cells_excluded == 1 else 's'} the "
            f"atemporal check already calls impossible were excluded before "
            f"tracing: a height recorded as 0 is an entry error with its own "
            f"finding, and reading it as a trajectory would report one defect "
            f"twice in more alarming language.")
    if reading.pediatric_rows_excluded:
        detail += (
            f" {reading.pediatric_rows_excluded:,} rows below age {ADULT_AGE} "
            f"were excluded from the height reading: a 9 cm year is normal in a "
            f"growing child, and applying an adult bound to pediatric data is "
            f"the error §A1.2 names explicitly.")

    return _finding(
        "pack::clinical::temporal_implausibility", "warning",
        (f"{len(reading.height_jumps) + len(reading.weight_swings) + len(reading.after_death)} "
         f"trajectories are not believable"),
        detail,
        ("Kahn et al.'s framework separates ATEMPORAL plausibility — is this "
         "value believable — from TEMPORAL plausibility — is this trajectory "
         "believable. The app already does the first, at every value, and has "
         "never done the second. Each value flagged here is individually "
         "plausible and the sequence is not, so no per-value check can find "
         "them: a height of 160 cm and a height of 169 cm are both ordinary, "
         "and they cannot both belong to the same adult."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=TEMPORAL_EVIDENCE,
        columns=sorted({j["column"] for j in
                        reading.height_jumps + reading.weight_swings}),
        params={
            "person_column": reading.person_column,
            "date_column": reading.date_column,
            "n_people": reading.n_people,
            "height_jumps": reading.height_jumps,
            "weight_swings": reading.weight_swings,
            "measurements_after_death": reading.after_death,
            "height_jump_cm": HEIGHT_JUMP_CM,
            "weight_change_fraction": WEIGHT_CHANGE_FRACTION,
            "weight_change_window_days": WEIGHT_CHANGE_WINDOW_DAYS,
            "adult_age": ADULT_AGE,
            "pediatric_rows_excluded": reading.pediatric_rows_excluded,
            # WHAT WAS NOT TRACED. A trajectory computed over 3 of a person's 4
            # visits is a different claim from one over all 4, and a payload
            # that reported only the hits would let the exclusion read as
            # coverage.
            "impossible_cells_excluded": reading.impossible_cells_excluded,
        },
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 7 · §A1.3 · number formats
# ═════════════════════════════════════════════════════════════════════════════

NUMBER_FORMAT_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#A1.3 Lab value formats and censored values")

_THOUSANDS = re.compile(r"^\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$")
_EURO_DECIMAL = re.compile(r"^\s*-?\d+,\d{1,2}\s*$")
_SCIENTIFIC = re.compile(r"^\s*-?\d+(\.\d+)?[eE][+-]?\d+\s*$")

_FORMATS = (
    ("thousands_separator", _THOUSANDS,
     "a thousands separator, so every value parses as text"),
    ("european_decimal_comma", _EURO_DECIMAL,
     "a decimal comma, which parses as text in an English locale and as a "
     "thousands separator in the ambiguous cases"),
    # SCIENTIFIC NOTATION IS ONLY REACHABLE ON A COLUMN THAT IS ALREADY TEXT,
    # and saying so is more useful than the pattern is.
    #
    # §A1.3 asks for it, and it is listed there beside separators and decimal
    # commas as though the three fail the same way. They do not, here: pandas
    # parses `1.2E+03` natively, so a column of nothing but scientific notation
    # arrives as `float64` and this detector never sees it. The pattern earns
    # its place only when something ELSE in the column blocks the parse — a
    # censoring token, a qualitative result — and then it tells the reader that
    # the numbers underneath are in exponent form.
    #
    # The alternative was to re-read the source file for this one check.
    # Declined: the detector contract is a DataFrame, every other reading in
    # this module is about the loaded frame, and a detector that quietly
    # reached past it would be reporting on an object no other finding
    # describes. The limit is stated instead, in the fixture's `.md` and in
    # SHAPES_NOT_COVERED.
    ("scientific_notation", _SCIENTIFIC,
     "scientific notation, which is reported here only because something else "
     "in the column already blocks the parse — on its own it reads as a "
     "number"),
)


def read_number_formats(series: pd.Series) -> List[Dict[str, Any]]:
    if pd.api.types.is_numeric_dtype(series):
        return []
    cells = _cells(series)
    if len(cells) < _MIN_CELLS:
        return []
    out = []
    for key, pattern, _because in _FORMATS:
        n = sum(1 for c in cells if pattern.match(c))
        if n / len(cells) >= 0.5:
            out.append({"column": str(series.name), "format": key, "n": n,
                        "share": round(n / len(cells), 4),
                        "example": next(c for c in cells if pattern.match(c))})
    return out


def number_format_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Numbers a reader will not parse as numbers.

    Separate from `text_numeric` deliberately, and the split is not cosmetic: a
    thousands separator makes a column text and `float()` fails, while a
    European decimal comma makes `1,05` parse as text here and as `105` in a
    locale-aware reader. **The second can be silently wrong by a factor of a
    hundred**, and reporting it as *some values did not parse* would lose that.
    """
    hits: List[Dict[str, Any]] = []
    for column in df.columns:
        hits.extend(read_number_formats(df[column]))
    if not hits:
        return None

    because = dict((key, text) for key, _pattern, text in _FORMATS)
    hits.sort(key=lambda h: (h["format"], h["column"]))
    lines = "; ".join(
        f"`{h['column']}` carries {because[h['format']]} (`{h['example']}`)"
        for h in hits)
    ambiguous = [h for h in hits if h["format"] == "european_decimal_comma"]
    detail = lines + "."
    if ambiguous:
        detail += (
            f" `{ambiguous[0]['column']}` is the one to check first: "
            f"`{ambiguous[0]['example']}` is one value in a decimal-comma "
            f"locale and a hundred times that in a locale-aware reader that "
            f"takes the comma for a thousands separator. Nothing in the column "
            f"settles which, so TurboTab has not parsed it either way.")

    return _finding(
        "pack::clinical::number_format", "warning",
        (f"{len(hits)} column{'' if len(hits) == 1 else 's'} write numbers in a "
         f"format that does not parse"),
        detail,
        ("A column that does not parse as numeric is treated as a category by "
         "everything downstream, so a continuous measurement silently becomes "
         "hundreds of one-hot levels. The decimal-comma case is worse than "
         "that: it can parse successfully and be wrong by a factor of a "
         "hundred, which no error anywhere would report."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=NUMBER_FORMAT_EVIDENCE,
        columns=sorted({h["column"] for h in hits}),
        params={"columns": hits},
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# 8 · §03b · the coaching sentence — impossible is not the same as extreme
# ═════════════════════════════════════════════════════════════════════════════

IMPOSSIBLE_VS_EXTREME_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#A1.2 ★ Reference ranges vs physiological plausibility — the distinction that matters")


def _plausibility(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The CORE's reading, read. Never a second copy of the bounds.

    `ml/card_evidence.plausibility_report` already holds every band and every
    tier; recomputing them here would be the two-engines failure inside a pack,
    and a pack whose bounds disagreed with the core's would be worse than a
    pack with none.
    """
    try:
        from turbotab import engine
        return engine.plausibility(df)
    except Exception:
        return None


def impossible_vs_extreme_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """**A correction to behavior the app ships**, not a new capability.

    §03b names this the coaching line that matters most, and it is a correction
    rather than an addition because the app already flags both categories and
    already offers a generic outlier treatment that cannot tell them apart:

    > *"4 systolic values are below 30 mmHg — physiologically impossible in a
    > living outpatient, almost certainly entry errors. **This is different from
    > the 812 values above 140 mmHg, which are abnormal but real and must be
    > kept:** excluding abnormal-but-possible values would remove the sickest
    > patients and bias the model toward the healthy."*

    The two counts in one sentence are the whole content. Reported separately
    they read as two severities of the same thing, which is exactly the reading
    that deletes the sickest patients.
    """
    report = _plausibility(df)
    if not report:
        return None

    # THE CORE ALREADY KNOWS WHICH COLUMNS IT CANNOT READ, and this reads it.
    #
    # `whole_column_suspect` is set when so much of a column falls outside the
    # physiologic band that the reading is about the COLUMN rather than about
    # its entries — a unit that was not what the matcher assumed, or a variable
    # that is not what the name says. On `clinical_labs.csv` that is `glucose`:
    # 42% of it is below the mg/dL floor because 120 of its rows are in mmol/L,
    # and the honest reading is `mixed_units`, at `critical`, one detector up.
    #
    # Without this filter the coaching sentence would have said *120 glucose
    # values are physiologically impossible and are almost certainly entry
    # errors, set them to missing* about 120 correctly-measured values. That is
    # the exact false authoritative assertion this pack exists to prevent, and
    # the capability that stops it was already in the core with nothing reading
    # it — `AUDIT-008`, arriving inside a detector written to avoid it.
    suspect = {row["column"] for row in
               (report.get("impossible") or []) + (report.get("improbable") or [])
               if row.get("whole_column_suspect")
               or row.get("reading") not in (None, "entries")}
    impossible = {row["column"]: row for row in report.get("impossible") or []
                  if row["column"] not in suspect}
    improbable = {row["column"]: row for row in report.get("improbable") or []
                  if row["column"] not in suspect}
    if not impossible:
        return None

    pairs: List[Dict[str, Any]] = []
    for column, row in impossible.items():
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        band = improbable.get(column)
        # ABNORMAL BUT POSSIBLE: outside the clinically-normal band and inside
        # the physiologic one. Computed from the bounds the CORE reported, so
        # there is one set of numbers rather than two.
        n_abnormal = 0
        if band is not None:
            inside_possible = series[(series >= row["low"]) & (series <= row["high"])]
            n_abnormal = int(((inside_possible < band["low"])
                              | (inside_possible > band["high"])).sum())
        pairs.append({
            "column": column,
            "variable": row.get("variable"),
            "unit": row.get("unit"),
            "n_impossible": int(row["n_flagged"]),
            "impossible_band": [row["low"], row["high"]],
            "n_abnormal_but_possible": n_abnormal,
            "normal_band": ([band["low"], band["high"]] if band else None),
        })

    pairs = [p for p in pairs if p["n_impossible"]]
    if not pairs:
        return None
    pairs.sort(key=lambda p: (-p["n_abnormal_but_possible"], -p["n_impossible"]))
    lead = next((p for p in pairs if p["n_abnormal_but_possible"]), pairs[0])

    detail = (
        f"{lead['n_impossible']} value"
        f"{'' if lead['n_impossible'] == 1 else 's'} in `{lead['column']}` are "
        f"outside {lead['impossible_band'][0]:g}–{lead['impossible_band'][1]:g}"
        f"{' ' + lead['unit'] if lead['unit'] else ''} — physiologically "
        f"impossible in a living outpatient, and almost certainly entry errors. "
        f"TurboTab recommends setting them to missing and reporting the count.")
    if lead["n_abnormal_but_possible"]:
        detail += (
            f" **This is different from the "
            f"{lead['n_abnormal_but_possible']:,} values outside "
            f"{lead['normal_band'][0]:g}–{lead['normal_band'][1]:g}"
            f"{' ' + lead['unit'] if lead['unit'] else ''}, which are abnormal "
            f"but real and must be kept**: excluding abnormal-but-possible "
            f"values would remove the sickest patients and bias the model "
            f"toward the healthy.")
    if len(pairs) > 1:
        detail += (
            " The same two categories are present in "
            + ", ".join(f"`{p['column']}`" for p in pairs if p is not lead)
            + ".")

    return _finding(
        "pack::clinical::impossible_vs_extreme", "warning",
        (f"`{lead['column']}` holds impossible values and abnormal ones, and "
         f"they are different categories"),
        detail,
        ("Physiologically impossible and statistically extreme are different "
         "categories, and no generic outlier rule can tell them apart. A ±3 SD "
         "screen or an IQR fence on this column removes both — the four entry "
         "errors and the sickest patients in the cohort — and reports one "
         "number for having done it. The impossible values are a data-quality "
         "repair with a count that belongs in the paper; the abnormal ones are "
         "the case mix the model exists to learn."),
        confidence="high", pack=CLINICAL, marker="offered",
        evidence=IMPOSSIBLE_VS_EXTREME_EVIDENCE,
        columns=[p["column"] for p in pairs],
        params={"columns": pairs,
                "reference_version": report.get("reference_version"),
                # WHAT WAS SET ASIDE AND WHY. A column the core cannot read is
                # not a column with no findings, and reporting only what was
                # checked would make the silence read as coverage.
                "columns_the_core_could_not_read": sorted(suspect)},
        # NO REPAIR HERE EITHER, and for a different reason from the others:
        # the core already offers the impossible-value repair through its own
        # finding. A second offer would be two paths to one edit.
        fix_label="", fix_kind="none")


#: Every detector, in the order they run. Hardest-first, which is the order they
#: were built in and the order `LOOP.md` §02 asks a widened part to hold.
DETECTORS = (censored_values_finding, text_numeric_finding,
             mixed_result_finding, mixed_units_finding,
             default_value_mass_finding, temporal_implausibility_finding,
             number_format_finding, impossible_vs_extreme_finding)


def findings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Every clinical import reading this table supports. Empty is common."""
    out = []
    for detector in DETECTORS:
        found = detector(df)
        if found:
            out.append(found)
    return out

"""The nutrition pack's first real content — two detectors and a refusal.

`research/NUTRITION_PACK.md` §01, §03 and §07. This is the loop that tests
whether the pack architecture carries domain content, and nutrition went first
for one reason: **it is the pack with a refusal in it.** A pack that can only
add findings has not been tested.

## What is here

**The Atwater reconstruction** (§01). `E_hat = 4P + 4C + 9F + 7A` against the
declared energy column, with the full ratio table. Nothing else in the app can
infer an energy unit — the impossibility bands know physiology and not
arithmetic — and the row that matters most is the last one: *the ratio drifts
with total energy* means **mixed units across rows**, which is a multi-source
merge and a hard fail rather than a conversion. A single global factor can be
applied; a drifting one cannot, and applying the median factor to a mixed table
would silently corrupt every row it did not describe.

**NHANES survey design** (§01). The design variables, and the sentence that
costs the most to get wrong: **dietary analyses take the dietary weights**
(`WTDRD1` for day 1, `WTDR2D` for both days) **and not `WTMEC2YR`**. Two further
readings, each a finding of its own: a weight column with no strata or PSU is a
**partially specified design**, and a stratum with a single PSU **breaks
Taylor-series variance estimation** and has standard remedies rather than a
shrug.

**The prevalence-of-inadequacy refusal** (§07, figure E). See
`prevalence_of_inadequacy` below — it is the part of this module that matters
most and the reason this pack was built first.

## Every number here is cited, and one is not

The Atwater coefficients (4/4/9/7 kcal per gram) and the kJ factor (4.184) are
`SETTLED` and hard-coded. The **acceptance band** for the ratio — the 0.90–1.10
row — is the research's own table and ships as stated.

**What the gate covers, exactly.** The sentence that stood here said
`docs/turbotab/tools/evidence.py check` kept it true that no `[verify-at-build]`
number ships from this module as a constant. That was false when it was written:
the gate set `PACKS = turbotab/packs.py` and scanned that one file, so it had
never opened this one (`GUIDED-059`). It does now — the scan runs over every
module that emits a finding, a refusal or a figure spec, and this is one.

**And what the gate still cannot see, which is the more useful half.**
`[verify-at-build]` marks a number the research *tried to read from primary text
and could not*. `DRIFT_LIMIT` below is not marked anywhere, because the research
never states a cut point for the drift row at all — it gives the reading and the
verdict and no number. So 0.25, and the choice to express drift as the
10th-to-90th spread over the median, are the implementation's own, and no bare-
number scan can tell that from a number the research supplied. That is
`GUIDED-061`, it is open, and the badge above `atwater_finding` reads SETTLED
over a detector whose gate is invented. The badge now reaches that finding,
which makes the gap easier to see rather than smaller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml import name_registry as registry
from turbotab.packs import (CONVENTION_STATUS, DISPUTED, SETTLED, Evidence,
                            PackRefusal,
                            _finding)

DIETARY = "dietary"

# ── §01 · the Atwater reconstruction ─────────────────────────────────────────

# kcal per gram. SETTLED, and the reconstruction's whole basis.
ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0}

# 1 kcal = 4.184 kJ. The pack's own "#1 embarrassment risk" list opens with the
# unit-conversion table, so this is a named constant rather than a literal.
KCAL_PER_KJ = 4.184

ATWATER_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/NUTRITION_PACK.md#01 · Import and structural recognition")

_NAME_PATTERNS = {
    "energy": r"energy|kcal|kj|calor",
    "protein": r"prot",
    "carbohydrate": r"carb|cho\b",
    "fat": r"\bfat|lipid|tfat",
    "alcohol": r"alco|etoh",
}


def _match(df: pd.DataFrame, role: str) -> Optional[str]:
    """The first numeric column whose name matches this role's pattern.

    Name matching alone is explicitly *not* how the pack recognizes a nutrient —
    §01 says match on three signals jointly. This is the name signal only, and
    it is used to find the columns the reconstruction then **tests**: the
    Atwater check is itself the second signal, so using names to assemble it and
    arithmetic to judge it is the two-signal structure rather than a shortcut
    past it.
    """
    pattern = re.compile(_NAME_PATTERNS[role], re.I)
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and pattern.search(str(column)):
            return str(column)
    return None


@dataclass(frozen=True)
class AtwaterReading:
    """The ratio table's verdict, with the numbers behind it."""
    verdict: str
    ratio: Optional[float]
    drift: Optional[float]
    energy_column: Optional[str]
    macro_columns: Dict[str, str]
    n_used: int
    sentence: str


# The ratio table, verbatim from §01. Ordered, because the drift row is tested
# BEFORE any factor is applied — a table with mixed units also has a median
# ratio, and reading the median first is how the mixed case gets converted.
PASS_LOW, PASS_HIGH = 0.90, 1.10
KJ_RATIO, INVERSE_RATIO = 4.184, 1.0 / 4.184
# Above this, the ratio is not stable enough across the energy range to be one
# factor. Expressed as the spread of the per-row ratio relative to its median,
# which is the quantity the research's "drifts with total energy" describes.
DRIFT_LIMIT = 0.25


def atwater(df: pd.DataFrame) -> Optional[AtwaterReading]:
    """`E_hat = 4P + 4C + 9F + 7A` against the declared energy column.

    Returns `None` where the table does not carry what the check needs — no
    energy column, or fewer than two macronutrients, in which case the
    reconstruction would rest on so little that a passing ratio would mean
    nothing.
    """
    energy_col = _match(df, "energy")
    macros = {role: _match(df, role) for role in
              ("protein", "carbohydrate", "fat", "alcohol")}
    macros = {k: v for k, v in macros.items() if v}
    if not energy_col or len(macros) < 2:
        return None

    declared = pd.to_numeric(df[energy_col], errors="coerce")
    reconstructed = sum(
        pd.to_numeric(df[col], errors="coerce").fillna(0.0) * ATWATER[role]
        for role, col in macros.items())
    usable = declared.notna() & (reconstructed > 0) & (declared > 0)
    if int(usable.sum()) < 10:
        return None

    ratios = (declared[usable] / reconstructed[usable]).astype(float)
    median = float(ratios.median())

    # THE PERCENT-OF-ENERGY ROW IS TESTED BEFORE THE DRIFT ROW, and the
    # reason is a genuine overlap between two rows of the research's table that
    # only showed up when a fixture was built for each.
    #
    # For true percent-of-energy data the ratio `E / E_hat` is PROPORTIONAL to
    # total energy — the macro columns sum to 100 for everybody while energy
    # varies — so it drifts, perfectly, and the drift gate would call it a
    # multi-source merge. Both rows describe the same observation.
    #
    # The research resolves it in the percent row itself: *"check whether the
    # four sum to ~100"*. That is a property of the macros alone and is
    # independent of the ratio, so it separates the two cleanly, and using the
    # table's own tiebreaker is better than inventing a correlation test that
    # would have to be tuned.
    macro_sum = sum(pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                    for col in macros.values())[usable]
    if 90.0 <= float(macro_sum.median()) <= 110.0:
        return AtwaterReading(
            verdict="macros_not_grams", ratio=median, drift=None,
            energy_column=energy_col, macro_columns=macros,
            n_used=int(usable.sum()),
            sentence=(
                f"The macronutrient columns sum to about "
                f"{float(macro_sum.median()):.0f} on a typical row, so they are "
                f"percentages of energy rather than grams. The Atwater "
                f"reconstruction does not apply to percentages, and every "
                f"downstream step that treats them as grams — densities, "
                f"energy adjustment, implausible-intake screens — would be "
                f"computing on the wrong quantity."))

    # THE DRIFT ROW IS TESTED NEXT, and the ordering is the finding.
    #
    # A table with mixed units across rows also has a median ratio, and it will
    # often sit near one of the clean factors. Reading the median first and the
    # spread second is exactly how a multi-source merge gets "converted" by a
    # factor that describes some of its rows — so the spread is the gate, and no
    # factor is proposed until it passes.
    spread = float((ratios.quantile(0.9) - ratios.quantile(0.1)) / max(median, 1e-9))
    if spread > DRIFT_LIMIT:
        return AtwaterReading(
            verdict="mixed_units", ratio=median, drift=spread,
            energy_column=energy_col, macro_columns=macros,
            n_used=int(usable.sum()),
            sentence=(
                f"The ratio of declared to reconstructed energy is not one "
                f"number across your rows — it spreads by {spread:.0%} around a "
                f"median of {median:.2f}. That is the signature of a "
                f"multi-source merge in which different rows carry different "
                f"units. There is no single factor to apply, so this is not a "
                f"conversion the app can offer; the rows have to be separated "
                f"by source first."))

    if PASS_LOW <= median <= PASS_HIGH:
        verdict, sentence = "pass", (
            f"Declared energy reconstructs to 4·protein + 4·carbohydrate + "
            f"9·fat + 7·alcohol within {abs(1 - median):.0%} across "
            f"{int(usable.sum()):,} rows, so the units are internally "
            f"consistent — kilocalories and grams. Every downstream step is a "
            f"function of total energy, and a unit error there is invisible in "
            f"the final tables.")
    elif abs(median - KJ_RATIO) / KJ_RATIO < 0.05:
        verdict, sentence = "energy_in_kj", (
            f"Declared energy is about {median:.2f}× the reconstruction, which "
            f"is the kilojoule factor ({KCAL_PER_KJ} kJ per kcal). The energy "
            f"column is in kJ while the macronutrients are in grams.")
    elif abs(median - INVERSE_RATIO) / INVERSE_RATIO < 0.05:
        verdict, sentence = "energy_inverse", (
            f"Declared energy is about {median:.3f}× the reconstruction, which "
            f"is the inverse of the kilojoule factor — the two columns are "
            f"mislabeled relative to each other.")
    elif median < 0.5 or median > 2.0:
        # The percent case that the sum test above did NOT catch — a subset of
        # the macros expressed as percentages, so they do not sum to 100. Kept
        # separate rather than folded in, because the app is much less sure
        # here and the sentence says so.
        verdict, sentence = "unexplained", (
            f"Declared energy is {median:.2f}× the reconstruction and no clean "
            f"unit factor explains it. The macronutrient columns may be "
            f"percentages of energy rather than grams, but they do not sum to "
            f"about 100, so that reading does not fit either. Reported rather "
            f"than repaired.")
    else:
        verdict, sentence = "unexplained", (
            f"Declared energy is {median:.2f}× the reconstruction, which is "
            f"outside the acceptance band and is not a unit factor the app "
            f"recognizes. Reported rather than repaired.")

    return AtwaterReading(verdict=verdict, ratio=median, drift=spread,
                          energy_column=energy_col, macro_columns=macros,
                          n_used=int(usable.sum()), sentence=sentence)


def atwater_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The reconstruction as a finding. Never a repair on the mixed-unit case."""
    reading = atwater(df)
    if reading is None or reading.verdict == "pass":
        return None
    severity = "critical" if reading.verdict == "mixed_units" else "warning"
    titles = {
        "mixed_units": "Different rows carry different energy units",
        "energy_in_kj": "The energy column is in kilojoules",
        "energy_inverse": "The energy and macronutrient columns are mislabeled",
        "macros_not_grams": "The macronutrient columns look like percentages",
        "unexplained": "Declared energy does not reconstruct from the macronutrients",
    }
    return _finding(
        # ONE ID PER DETECTOR, and the verdict is a parameter — which it already
        # was. The id used to be `atwater_{verdict}`, and a varying id cannot be
        # bound to anything: `LooksFor` names an id, `prior_columns` looks a
        # detector up by `f["id"] == detector`, and both would have had to know
        # five spellings of one finding. The title is where the verdict belongs,
        # because that is what a reader sees. Found by wiring the detector into
        # the pack (`GUIDED-058`); nothing had needed to refer to it before.
        "pack::dietary::atwater", severity,
        titles[reading.verdict], reading.sentence,
        ("Total energy is the denominator of every energy adjustment, every "
         "nutrient density and every implausible-intake screen, so a unit "
         "error here propagates into every result and is invisible in the "
         "final tables."),
        confidence="high", pack=DIETARY, marker="derived",
        # ATTACHED, at last. This constant was built at `GUIDED-047` and reached
        # nothing until `GUIDED-059`: the only reference to it in the repository
        # was a test asserting the constant was well-formed, which is a guard
        # testing its own description rather than the app.
        evidence=ATWATER_EVIDENCE,
        columns=[reading.energy_column] + list(reading.macro_columns.values()),
        params={"ratio": reading.ratio, "drift": reading.drift,
                "n_used": reading.n_used, "verdict": reading.verdict,
                "energy_column": reading.energy_column},
        # NO FIX ON THE MIXED CASE. `fix_kind="none"` is the engine refusing to
        # guess, and it is the right refusal: there is no single factor, so any
        # repair would corrupt the rows it did not describe.
        fix_label="", fix_kind="none")


# ── §01 · NHANES survey design ───────────────────────────────────────────────

DESIGN_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/NUTRITION_PACK.md#01 · Import and structural recognition")

# The dietary weights, and the one that is NOT for dietary analyses.
DIETARY_WEIGHTS = ("WTDRD1", "WTDR2D")
EXAM_WEIGHT = "WTMEC2YR"
INTERVIEW_WEIGHT = "WTINT2YR"
STRATA, PSU = "SDMVSTRA", "SDMVPSU"
# The survey cycle. §01 lists it beside the weights and the design variables and
# nothing else reads it — it is here as CORROBORATION only, because a column
# literally called `SDDSRVYR` is an NHANES file saying so in its own vocabulary.
SURVEY_CYCLE = "SDDSRVYR"

# Every design variable the pack recognizes by its EXACT name. Deliberately not
# the generic `strata|stratum` and `psu|cluster` fallbacks below: those are
# `GUIDED-069`'s unfixed half, and letting a weak signal corroborate another
# weak signal would launder both. A table with columns `weight` and `cluster`
# must not become a survey design.
EXACT_DESIGN_NAMES = DIETARY_WEIGHTS + (EXAM_WEIGHT, INTERVIEW_WEIGHT,
                                        STRATA, PSU, SURVEY_CYCLE)


def _body_measurement(df: pd.DataFrame, column: str) -> Optional[Tuple[str, bool]]:
    """`(variable, plausible)` where the engine recognizes this as physiology.

    Through `physiology_reference.match_variable_key`, which is exact against
    the key or a declared alias after case and separators are stripped — never
    by substring. Borrowing the vetted matcher is the opposite of adding
    another name list, and it is the same call `clinical_reference_columns` and
    `_reference_column` already make.

    `plausible` is the second half and it is what makes the reading a reading
    rather than a name lookup: a column called `weight` holding 34,000 is not a
    body weight, whatever it is called — 34,000 kg is outside the impossibility
    band `physiology_reference` publishes (0.4–650), and NHANES sampling weights
    live in the tens of thousands. The band is the engine's, not this module's.
    """
    try:
        from ml.physiology_reference import (get_impossibility_band,
                                             load_reference_bundle,
                                             match_variable_key)
        reference = load_reference_bundle()["nhanes"]
    except Exception:                                      # pragma: no cover
        return None
    variable = match_variable_key(str(column), reference)
    if not variable:
        return None
    band = get_impossibility_band(reference, variable)
    if not band:                                           # pragma: no cover
        return variable, True
    floor, ceiling = float(band[0]), float(band[1])
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:                                       # pragma: no cover
        return variable, True
    return variable, bool(floor <= float(values.median()) <= ceiling)


def survey_design(df: pd.DataFrame) -> Dict[str, Any]:
    """Which design variables this table carries, and what is missing.

    Exact names first, then the generic patterns §01 lists, because a column
    literally called `SDMVPSU` is a different quality of evidence from one
    called `psu` and the finding says which.

    ## A generic weight needs a second signal, and guard #2 is why

    §01 says to flag generic `weight|wt|pweight`, and taking that as sufficient
    fired `partial_design` on `clinic_visits.csv` — *"There is a survey weight
    in this table and no strata or PSU column"* — about a column holding
    `107 kg`. **A body weight called a sampling weight is the dietary pack
    asserting something false on a clinical table, authoritatively**, which is
    the exact failure `DOMAIN_PACKS.md` §05 says would embarrass us and §03's
    guard #2 forbids. It was invisible until the detectors were wired to an
    upload (`GUIDED-058`); the module's own tests used the NHANES names.

    ## The cost, and how much of it was recoverable (`GUIDED-070`)

    The first repair rejected a generic weight name unless the column was
    numeric and unrecognized by the engine's physiology reference. That killed
    the false assertion and took a real capability with it — **a genuine
    sampling weight called `weight` was missed even in a table carrying
    `SDMVSTRA` and `SDMVPSU`** — and that loss lived in a report and a comment
    and in no ledger row, which is `LOOP.md` §06.1's most common gap.

    Most of it was recoverable, because *the corroboration a generic weight
    name lacks is sitting in the columns beside it*: **nobody names a column
    `SDMVSTRA` by accident.** Three signals, jointly, which is §01's own
    instruction applied where it had not been:

    1. **The table says it is a complex survey**, in NHANES's own vocabulary —
       at least one of `EXACT_DESIGN_NAMES`. Absent that, a generic weight name
       is the only evidence there is, and it is not enough.
    2. **A survey weight is numeric.** `107 kg` is a measurement a person typed.
    3. **Its values are not a plausible body measurement.** The engine
       recognizes `weight` and `wt` as physiology and publishes an
       impossibility band for each; a column called `weight` whose median is
       34,000 is not a body weight in any unit, and NHANES sampling weights
       live in the tens of thousands. This is the signal signal 2 of the first
       repair stood in for, and it is a reading rather than a name lookup.

    **The corroboration is exact names only**, never the generic
    `strata|stratum` and `psu|cluster` fallbacks below. Those are `GUIDED-069`'s
    unfixed half, and letting one weak signal corroborate another would launder
    both — a table with columns `weight` and `cluster` would become a survey
    design, which is the original defect arriving through the back door.

    What is still lost, and it is smaller and stated: a survey weight named
    generically in a table with **no** exactly-named design variable. The app
    says nothing about that table's design, and silence over a false assertion
    is the governing rule's own ordering. Every rejection records its reason, so
    the absence is inspectable rather than absent.
    """
    present = {str(c).upper(): str(c) for c in df.columns}
    found = {
        "dietary_weights": [present[w] for w in DIETARY_WEIGHTS if w in present],
        "exam_weight": present.get(EXAM_WEIGHT),
        "interview_weight": present.get(INTERVIEW_WEIGHT),
        "strata": present.get(STRATA),
        "psu": present.get(PSU),
    }
    if not found["strata"]:
        found["strata"] = next((str(c) for c in df.columns
                                if re.fullmatch(r"strata|stratum", str(c), re.I)), None)
    if not found["psu"]:
        found["psu"] = next((str(c) for c in df.columns
                             if re.fullmatch(r"psu|cluster", str(c), re.I)), None)

    # The exactly-named design variables this table carries. Computed before the
    # generic pass because it is what licenses it.
    found["recognized_design"] = [present[name] for name in EXACT_DESIGN_NAMES
                                  if name in present]

    named = [str(c) for c in df.columns
             if re.fullmatch(r"weight|wt|pweight", str(c), re.I)]
    generic, rejected = [], []
    for column in named:
        reading = _body_measurement(df, column)
        if not pd.api.types.is_numeric_dtype(df[column]):
            rejected.append((column, "it is not numeric"))
        elif not found["recognized_design"]:
            rejected.append(
                (column, "the table carries no design variable this pack "
                         "recognizes by its exact name, so a generic weight "
                         "name is the only evidence there is"))
        elif reading and reading[1]:
            rejected.append(
                (column, f"the engine's physiology reference recognizes it as "
                         f"{reading[0]} and its values are plausible for that "
                         f"measurement"))
        else:
            generic.append(column)
    found["generic_weights"] = generic
    found["rejected_weights"] = rejected
    found["any_weight"] = bool(found["dietary_weights"] or found["exam_weight"]
                               or found["interview_weight"] or generic)
    return found


def lonely_psu(df: pd.DataFrame, design: Dict[str, Any]) -> List[Any]:
    """Strata containing exactly one PSU.

    Named for the remedy's own vocabulary (`options(survey.lonely.psu=)`), and
    it is a real break rather than a warning: Taylor-series linearization
    estimates a stratum's variance from the spread BETWEEN its PSUs, and one
    PSU has no spread. The variance contribution is undefined, not small.
    """
    if not design.get("strata") or not design.get("psu"):
        return []
    counts = df.groupby(design["strata"])[design["psu"]].nunique()
    return [k for k, v in counts.items() if int(v) == 1]


# ── the three design readings, as three detectors ────────────────────────────
#
# SPLIT RATHER THAN WIDENED, and the choice is a deliverable (`GUIDED-058`).
# `Pack.detectors` is a tuple of `Callable[[pd.DataFrame], Optional[Dict]]`.
# `atwater_finding` matched it as written; `design_findings` returned a `List`
# and did not, so the pack path was open to one of the two and closed to the
# other. Two ways out:
#
# **Widen the contract** so a detector may return a list. One line in `packs.
# findings`, and it costs the type its meaning: every caller then has to handle
# both shapes, `prior_columns`'s `f["id"] == detector` lookup gets ambiguous,
# and the widening would rest on ONE example — which is the same mistake
# `GUIDED-056` refused to make with the figure `tier` enum, one layer down.
#
# **Split.** Three findings that were already independent become three
# detectors. What it cost: the early `return out` after the partial-design
# reading, which suppressed the lonely-PSU check, had to become a precondition
# — and it turned out to be one already, since `lonely_psu` needs BOTH strata
# and PSU and the partial case is exactly their absence. So the control flow
# was expressible as guards on each detector rather than as an order between
# them, which is the evidence that these were three things sharing a function
# rather than one thing with three outputs. `design_findings` stays as the
# composed reading, because it is a better unit to test and to read.
#
# The honest cost of splitting: `survey_design(df)` now runs three times per
# table instead of once. It is a column-name scan over a dict, so the cost is
# real and negligible, and paying it buys a contract that stayed narrow.


def survey_weights_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The weight to use, and the one not to. Findings, never repairs."""
    design = survey_design(df)
    if not design["any_weight"] or not design["dietary_weights"]:
        return None
    names = ", ".join(f"`{w}`" for w in design["dietary_weights"])
    detail = (
        f"This table carries {names}"
        + (f" and `{design['exam_weight']}`." if design["exam_weight"] else ".")
        + " Dietary analyses take the dietary weights: `WTDRD1` for day-1 "
          "analyses and `WTDR2D` for anything using both days. They add "
          "adjustments for recall non-response and for the deliberate "
          "weekday/weekend allocation of recall days that the examination "
          "weight does not carry.")
    if design["exam_weight"]:
        detail += (f" `{design['exam_weight']}` is the examination weight "
                   f"and is not the right one here.")
    return _finding(
        "pack::dietary::survey_weights", "warning",
        "Use the dietary weights, not the examination weight", detail,
        ("Unweighted or wrongly-weighted estimates are biased toward the "
         "oversampled groups, because NHANES deliberately oversamples "
         "specific race, age and income groups — so an unweighted mean is "
         "not a US-population mean."),
        confidence="high", pack=DIETARY, marker="derived",
        evidence=DESIGN_EVIDENCE,
        columns=design["dietary_weights"],
        params={"use": design["dietary_weights"],
                "not": [w for w in (design["exam_weight"],) if w]},
        fix_label="", fix_kind="none")


def partial_design_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A weight with no strata or PSU. Weights fix the estimate, not the interval."""
    design = survey_design(df)
    if not design["any_weight"] or (design["strata"] and design["psu"]):
        return None
    missing = [name for name, key in (("strata", "strata"), ("PSU", "psu"))
               if not design[key]]
    return _finding(
        "pack::dietary::partial_design", "warning",
        "The survey design is only partially specified",
        (f"There is a survey weight in this table and no "
         f"{' or '.join(missing)} column. Weights alone correct the point "
         f"estimates toward the population; the strata and PSU are what "
         f"make the standard errors right. Without them the intervals are "
         f"too narrow, and nothing on screen shows it."),
        ("NCHS states that variance estimates computed under a "
         "simple-random-sample assumption are generally too low and biased "
         "for NHANES."),
        confidence="high", pack=DIETARY, marker="derived",
        evidence=DESIGN_EVIDENCE,
        columns=(design["dietary_weights"]
                 or design["generic_weights"]
                 or [w for w in (design["exam_weight"],) if w]),
        params={"missing": missing},
        fix_label="", fix_kind="none")


def lonely_psu_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A stratum with one PSU. The estimator does not degrade, it breaks.

    Its precondition — strata AND PSU both present — is the exact negation of
    `partial_design_finding`'s, which is why the composed reading's early
    `return` was expressible as a guard rather than as an order.
    """
    design = survey_design(df)
    if not design["any_weight"] or not (design["strata"] and design["psu"]):
        return None
    lonely = lonely_psu(df, design)
    if not lonely:
        return None
    shown = ", ".join(str(s) for s in lonely[:5])
    return _finding(
        "pack::dietary::lonely_psu", "critical",
        f"{len(lonely):,} stratum/strata contain a single PSU",
        (f"Strata {shown}{'…' if len(lonely) > 5 else ''} each contain one "
         f"primary sampling unit. Taylor-series linearization estimates a "
         f"stratum's variance from the spread between its PSUs, so a "
         f"stratum with one PSU contributes an undefined variance rather "
         f"than a small one — the estimator does not degrade, it breaks. "
         f"The standard remedies are to collapse the stratum with a "
         f"neighbor, to centre the lonely PSU at the population mean, or "
         f"to certainty-adjust it; which one is a decision about your "
         f"design, not about your data."),
        ("A variance estimate that is undefined and silently computed is "
         "the failure this whole product exists to remove."),
        confidence="high", pack=DIETARY, marker="derived",
        evidence=DESIGN_EVIDENCE,
        columns=[design["strata"], design["psu"]],
        params={"strata": [str(s) for s in lonely], "n": len(lonely)},
        fix_label="", fix_kind="none")


DESIGN_DETECTORS = (survey_weights_finding, partial_design_finding,
                    lonely_psu_finding)


def design_findings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Every reading the design variables support, composed from the three.

    Kept because it is the better unit to read and to test, and because the
    ORDER is part of the reading: which weight, then whether the design is
    complete, then whether a complete one is estimable.
    """
    return [f for f in (detector(df) for detector in DESIGN_DETECTORS) if f]


# ── §03 · the variance decomposition the shrinkage plot is drawn from ────────

# Named for the decomposition rather than for the figure, and separate from
# `figure_specs.SHRINKAGE_EVIDENCE` on purpose: that one badges the figure's
# publication-grade CHECKLIST, this one badges the arithmetic underneath it.
# Two claims that happen to share a section.
USUAL_INTAKE_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/NUTRITION_PACK.md#03 · ★ Repeated recalls and "
            "measurement error"))

# WHAT THIS IS AND WHAT IT IS NOT, in the payload and in the caption, because
# the difference is a documented failure rather than a nuance.
#
# §03 specifies the NCI method — Box–Cox toward normality, a linear mixed model
# with a person-level random effect and covariates, back-transformed by Monte
# Carlo integration, implemented in MIXTRAN/DISTRIB — or ISU/ISUF, MSM, SPADE.
# None of those is here and none of them is claimed. What is here is the
# one-way random-effects decomposition §03 states in full, with the classical
# shrinkage that follows from it:
#
#     lambda_i = sigma_b^2 / (sigma_b^2 + sigma_w^2 / n_i)
#     usual_i  = grand_mean + lambda_i * (mean_i - grand_mean)
#
# That is enough to show the narrowing, which is what the figure argues, and it
# is not enough to hand anyone a person's usual intake — §03 is explicit that
# even the NCI method's individual output must not go into a Table 1 or a
# per-person deliverable. So the method travels with the numbers.
SHRINKAGE_METHOD = (
    "variance-component shrinkage of each person's mean toward the grand mean, "
    "not the NCI method")


@dataclass(frozen=True)
class VarianceComponents:
    """σ²_w, σ²_b and what follows from them, for one nutrient.

    `lambda_at(n)` is the attenuation factor for a mean of `n` days — the
    quantity §03 says a reviewer in this field asks for, and the one that says
    how much of a diet–outcome slope survives the measurement error.
    """
    within: float
    between: float
    n_people: int
    n_rows: int
    days_median: float

    @property
    def ratio(self) -> Optional[float]:
        """σ²_w : σ²_b. `None` where σ²_b is zero — a ratio to nothing."""
        return None if self.between <= 0 else self.within / self.between

    @property
    def icc(self) -> Optional[float]:
        total = self.within + self.between
        return None if total <= 0 else self.between / total

    def lambda_at(self, n_days: float) -> Optional[float]:
        if self.between <= 0 or n_days <= 0:
            return None
        return float(self.between / (self.between + self.within / n_days))


class UsualIntakeRefusal(PackRefusal):
    """A usual-intake distribution the data cannot support.

    §03, verbatim on the case that matters: *"With one recall you cannot do
    that separation from your own data at all."* Returning a third density
    anyway would be the figure making its argument with fabricated evidence —
    and the argument is the whole figure.
    """


def variance_components(df: pd.DataFrame, *, person_col: str,
                        value_col: str) -> Optional[VarianceComponents]:
    """One-way random-effects decomposition of one nutrient across a person's days.

    `None` — never zeros — where the design cannot support it. σ²_w is
    estimated from the spread WITHIN a person, so a table with one row per
    person has nothing to estimate it from, and §03 says to report that rather
    than to substitute an external ratio silently.

    The unbalanced case uses the standard `n0` rather than the mean group size,
    because NHANES-shaped data has people with one recall beside people with
    two and treating n0 as the average would bias σ²_b upward.
    """
    if person_col not in df.columns or value_col not in df.columns:
        return None
    values = pd.to_numeric(df[value_col], errors="coerce")
    frame = pd.DataFrame({"person": df[person_col], "value": values}).dropna()
    if frame.empty:
        return None
    sizes = frame.groupby("person")["value"].size()
    k, total = int(len(sizes)), int(len(frame))
    if k < 2 or total <= k:
        # Every person has one row: within-person variance is not identifiable.
        return None

    grand = float(frame["value"].mean())
    means = frame.groupby("person")["value"].mean()
    ss_within = float(((frame["value"]
                        - frame["person"].map(means)) ** 2).sum())
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ms_within = ss_within / (total - k)
    ms_between = ss_between / (k - 1)
    n0 = (total - float((sizes ** 2).sum()) / total) / (k - 1)
    if n0 <= 0:                                            # pragma: no cover
        return None
    # CLAMPED AT ZERO AND NOT BELOW. A negative method-of-moments estimate of
    # σ²_b means the between-person signal is smaller than the noise; it does
    # not mean the variance is negative, and reporting it as such would be a
    # number nobody can read.
    between = max((ms_between - ms_within) / n0, 0.0)
    return VarianceComponents(
        within=float(ms_within), between=float(between), n_people=k,
        n_rows=total, days_median=float(sizes.median()))


def usual_intake_series(df: pd.DataFrame, *, person_col: str,
                        value_col: str) -> Dict[str, Any]:
    """The three densities the shrinkage plot overlays, or a refusal.

    Raises `UsualIntakeRefusal` rather than returning two series or a flat
    third one: `shrinkage_payload` already refuses a missing series, and a
    third density that is a spike at the grand mean would satisfy that check
    while making the figure's argument out of nothing.
    """
    components = variance_components(df, person_col=person_col,
                                     value_col=value_col)
    if components is None:
        raise UsualIntakeRefusal(
            f"Separating day-to-day variation from between-person difference "
            f"needs at least two days for some of your participants, and "
            f"`{value_col}` has one row per person. The within-person variance "
            f"is not identifiable from these data at all — not by this app and "
            f"not by any method without an external variance ratio, which is "
            f"an assumption rather than a measurement and would have to be "
            f"labeled as one.",
            evidence=USUAL_INTAKE_EVIDENCE,
            offer={"draw": "per_nutrient_distribution",
                   "label": f"The observed distribution of {value_col}",
                   "caption_note": (
                       "One day per person, labeled as one day. No usual-"
                       "intake claim is made from it and no percentile of it "
                       "is a percentile of usual intake."),
                   "forbidden": "usual_intake_from_one_day"})

    if components.between <= 0:
        raise UsualIntakeRefusal(
            f"Across {components.n_people:,} people, the variation between "
            f"one person's days is as large as the variation between people "
            f"for `{value_col}`, so the between-person variance estimates to "
            f"zero. Every modeled usual intake would be the same number, and a "
            f"density drawn from it would be a spike that says the population "
            f"is identical rather than that the data cannot see the "
            f"difference.",
            evidence=USUAL_INTAKE_EVIDENCE,
            offer={"draw": "per_nutrient_distribution",
                   "label": f"The observed distribution of {value_col}",
                   "caption_note": (
                       "Drawn from the observed days, with the variance "
                       "decomposition reported beside it so the reason the "
                       "third density is absent is visible."),
                   "forbidden": "usual_intake_with_no_between_variance"})

    values = pd.to_numeric(df[value_col], errors="coerce")
    frame = pd.DataFrame({"person": df[person_col], "value": values}).dropna()
    grand = float(frame["value"].mean())
    means = frame.groupby("person")["value"].mean()
    sizes = frame.groupby("person")["value"].size()

    # THE FIRST DAY, not a random one. `single_day` has to be reproducible from
    # the table, or the figure is not the same figure twice.
    single = frame.groupby("person")["value"].first()
    shrunk = {}
    for person, mean in means.items():
        lam = components.lambda_at(float(sizes[person]))
        shrunk[person] = grand + (lam or 0.0) * (float(mean) - grand)

    return {
        "series": {
            "single_day": [float(v) for v in single.to_numpy()],
            "mean_of_days": [float(v) for v in means.to_numpy()],
            "usual_intake": [float(shrunk[p]) for p in means.index],
        },
        "components": components,
        "n_days": float(sizes.median()),
        "method": SHRINKAGE_METHOD,
    }


# ── §07 figure E · the refusal ───────────────────────────────────────────────

PREVALENCE_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/NUTRITION_PACK.md#07 · EDA and presentation")

# Nutrients with an AI and no EAR. A prevalence of inadequacy cannot be computed
# from an AI **by anyone** — the AI is set where an EAR could not be
# established, so there is no requirement distribution to sit below.
#
# A NAME REGISTRY, not a substring list, and `test_every_substring_match_
# against_a_name_is_declared` is what made that explicit. The first version
# asked `any(a in name for a in AI_ONLY)`, and `"iron" in "environment"` is
# True — a refusal firing on a column that has nothing to do with iron, in the
# one code path whose whole job is to refuse correctly. Exact key or declared
# alias; an unrecognized name yields `None`, which is an answer.
AI_ONLY = registry.build({
    "fiber": ("fibre", "dietary fiber", "dietary fibre", "total fiber",
              "DR1TFIBE", "DR2TFIBE"),
    "potassium": ("DR1TPOTA", "DR2TPOTA", "K"),
    "vitamin k": ("vitamin_k", "phylloquinone", "DR1TVK"),
    "chromium": (),
    "manganese": (),
    "choline": ("DR1TCHL",),
    "biotin": (),
    "pantothenic acid": ("vitamin b5",),
})

# The one nutrient whose requirement distribution is skewed, so the cut-point
# method does not apply and the probability approach does.
SKEWED_REQUIREMENT = registry.build({
    "iron": ("DR1TIRON", "DR2TIRON", "fe"),
})

USUAL_INTAKE = "usual_intake"
SINGLE_DAY = "single_day"
NAIVE_MEAN = "naive_mean"


class PrevalenceRefusal(PackRefusal):
    """A prevalence of inadequacy the app must not compute.

    Carries `offer` — what it CAN draw instead — because a refusal that offers
    nothing is indistinguishable from a missing feature, and the user still has
    a real question. And it carries a badge, because *"nobody can compute this,
    not the app and not you with a spreadsheet"* is the strongest sentence this
    pack says and was the only one going out with no citation (`GUIDED-059`).

    Everything it does is `PackRefusal`'s. The subclass exists so a caller can
    catch this refusal specifically, and so the gate that walks refusal call
    sites finds it by subclassing rather than by a name somebody remembered to
    add to a list.
    """


def prevalence_of_inadequacy(nutrient: str, *, basis: str,
                             reference_kind: str,
                             stratum: Optional[str] = None) -> Dict[str, Any]:
    """The EAR cut-point method, and the four cases where it must refuse.

    **This is why nutrition went first.** A pack that can only add findings has
    not been tested; this is the path that says whether the architecture holds
    when the correct answer is *no*.

    Four refusals, each for a different reason and each stating what it can draw
    instead:

    1. **Against an AI.** Fiber and potassium have an Adequate Intake and no
       EAR. An AI is set precisely *because* a requirement distribution could
       not be established, so there is nothing for an intake to be below.
       Nobody can compute this — not the app, and not the user with a
       spreadsheet.
    2. **Against the RDA.** The RDA is an individual-level target covering 97–98%
       of requirements. Using it as a group cut-point counts most of the
       adequately-nourished population as inadequate.
    3. **From single-day intakes.** The one-day distribution is wider than the
       usual-intake distribution, so a tail proportion computed from it is
       overstated — in both tails.
    4. **From a naive mean of days.** Narrower than one day and still wider than
       usual intake. *"We averaged the two 24-hour recalls to obtain usual
       intake"* followed by a prevalence claim is a documented failure, not a
       simplification.

    And one routing rule rather than a refusal: **iron in menstruating women**
    has a skewed requirement distribution, so the cut-point method does not
    apply and the probability approach does.
    """
    canonical_ai = registry.match(nutrient, AI_ONLY)
    canonical_skewed = registry.match(nutrient, SKEWED_REQUIREMENT)

    if canonical_ai:
        raise PrevalenceRefusal(
            f"{nutrient} has an Adequate Intake, not an Estimated Average "
            f"Requirement. An AI is set precisely because a requirement "
            f"distribution could not be established, so there is nothing for "
            f"an intake to be below — a prevalence of inadequacy cannot be "
            f"computed from an AI, and neither can anyone else compute one.",
            evidence=PREVALENCE_EVIDENCE,
            offer={
                "draw": "distribution_against_ai",
                "label": f"Usual intake of {nutrient} against the AI",
                "caption_note": (
                    "The vertical line is the Adequate Intake. The proportion "
                    "below it is NOT a prevalence of inadequacy and is not "
                    "labeled as one."),
                "forbidden": "prevalence_of_inadequacy",
            })

    if reference_kind.lower() == "rda":
        raise PrevalenceRefusal(
            f"The RDA is an individual-level target set to cover 97–98% of "
            f"requirements. Counting everyone below it as inadequate counts "
            f"most of an adequately-nourished population as deficient. "
            f"Prevalence of inadequacy is computed against the EAR.",
            evidence=PREVALENCE_EVIDENCE,
            offer={
                "draw": "distribution_against_ear_and_rda",
                "label": f"Usual intake of {nutrient} against the EAR and RDA",
                "caption_note": (
                    "The shaded area below the EAR is the prevalence of "
                    "inadequacy. The RDA is drawn as a reference for an "
                    "individual and no area is shaded against it."),
                "forbidden": "prevalence_against_rda",
            })

    if basis in (SINGLE_DAY, NAIVE_MEAN):
        which = ("a single day's intake" if basis == SINGLE_DAY
                 else "a naive mean of the available days")
        raise PrevalenceRefusal(
            f"Prevalence of inadequacy needs a usual-intake distribution, and "
            f"this is {which}. The observed distribution is wider than the "
            f"usual-intake distribution — day-to-day variation is still in it —"
            f" so a tail proportion computed from it is overstated, in both "
            f"tails. Averaging two recalls narrows it and does not remove it.",
            evidence=PREVALENCE_EVIDENCE,
            offer={
                "draw": "shrinkage",
                "label": "What usual-intake modeling would change",
                "caption_note": (
                    "Three densities of the same nutrient — one day, the mean "
                    "of the available days, and modeled usual intake. The "
                    "narrowing between them is the size of the error a "
                    "prevalence computed from the first two would carry."),
                "forbidden": "prevalence_from_observed_intake",
            })

    method = "cut_point"
    note = ""
    if canonical_skewed and (stratum or "").strip().lower() in (
            "menstruating", "menstruating women", "women_menstruating"):
        method = "probability_approach"
        note = ("Iron in menstruating women has a skewed requirement "
                "distribution, so the EAR cut-point method does not apply and "
                "the probability approach is used instead.")
    return {"nutrient": nutrient, "basis": basis, "method": method,
            "reference_kind": "EAR", "note": note,
            **PREVALENCE_EVIDENCE.to_dict()}

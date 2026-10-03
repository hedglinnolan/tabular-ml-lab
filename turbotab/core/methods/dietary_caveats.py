"""The caveats a dietary result carries, each stated where it is sourced (audit WP15: IN-20, IN-22).

**What adjusting for energy asks.** Adjusting a nutrient's coefficient for total energy makes it a
substitution: more of that nutrient in place of other energy at the same total. Leaving energy out,
or partitioning it, asks about adding energy. Tomova et al. 2022 (*AJCN* 115:189, PMC8755101):
"It remains underappreciated that adjusting for total energy and adjusting for remaining energy
intake evaluate very different causal estimands. These 2 estimands relate to very different
questions that require very different interpretations." The energy finding states this choice
(:data:`ENERGY_WHY`); it no longer says adjustment "is not in dispute" (audit IN-20, ledger #86).

**An energy-related outcome** (:func:`energy_related`). NUTRITION_PACK §04, Diagnostic: "Detect
whether the outcome is itself energy-related (weight, BMI, adiposity, diabetes) — if so, escalate
the mediation/collider warning." Tomova et al. 2022: total energy "can be conceptualized as a
'collider'", and "Adjusting for TE (purple) opens conditional dependencies between the exposure and
all competing energy sources". The outcome is read from its name by whole tokens, never
substrings (audit IN-01): ``body_weight_kg`` and ``BMXBMI`` read as energy-related; ``fat_g`` and
``WTMEC2YR`` (a survey weight) do not.

**Measurement error** (:func:`measurement_error_line`). Self-reported intakes are measured with
error. Keogh et al. 2020 (STRATOS Part 1, *Stat Med* 39:2197, PMC7450672, §3.1.1): under classical
non-differential error "|βX*| ≤ |βX|" for a single error-prone covariate; §3.1.3, with several:
"the estimated coefficients in model (11) may be larger or smaller than the true target values in
a rather unpredictable manner." Freedman et al. 2011 (*JNCI* 103:1086, PMC3143422): "In
multivariable disease models with two or more mismeasured exposures, estimated relative risks may
become attenuated, inflated, or can even change direction". Under inference the coefficient table
carries that limitation, with the reports each row averages and the fact that the table is not
corrected (the ``calibration`` stage corrects energy-adjusted exposures when asked).
"""
from __future__ import annotations

import re

from typing import Any, Iterable, Sequence

from turbotab.core.voice import count, listing, plural, tick

NUT04 = "research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature"
DISPUTED = {"status": "DISPUTED", "source": NUT04}

# ── what adjusting for energy asks ───────────────────────────────────────────

ENERGY_TITLE = "Adjusting for total energy decides what a nutrient's coefficient means"
ENERGY_WHY = (
    "Adjusting for total energy makes a nutrient's coefficient a substitution: more of it in place "
    "of other energy at the same total. Leaving energy out, or partitioning it, asks about adding "
    "energy instead. The two answer different questions (Tomova et al. 2022), so which to ask is "
    "the study's choice and the method follows from it. For prediction, models that keep total "
    "energy predict alike.")

# ── an energy-related outcome ────────────────────────────────────────────────

# Whole tokens of an outcome's name, and what each says the outcome is (NUTRITION_PACK §04's list:
# weight, BMI, adiposity, diabetes; the audit adds waist). NHANES names are whole tokens too.
_KIND_BY_TOKEN = {
    "weight": "body weight", "bodyweight": "body weight", "wt": "body weight",
    "bmxwt": "body weight",
    "bmi": "BMI", "bmxbmi": "BMI", "quetelet": "BMI",
    "waist": "waist size", "bmxwaist": "waist size", "whr": "waist size", "whtr": "waist size",
    "adiposity": "adiposity", "obesity": "adiposity", "obese": "adiposity",
    "overweight": "adiposity", "fatmass": "adiposity", "bodyfat": "adiposity",
    "adipose": "adiposity",
    "diabetes": "diabetes", "diabetic": "diabetes", "t2d": "diabetes", "t2dm": "diabetes",
    "diq010": "diabetes",
}
# Two tokens that name adiposity together: ``fat_mass_kg``, ``body_fat_pct``, ``percent_fat``.
_PAIRS = {("fat", "mass"): "adiposity", ("body", "fat"): "adiposity", ("percent", "fat"): "adiposity",
          ("pct", "fat"): "adiposity", ("fat", "pct"): "adiposity", ("fat", "percent"): "adiposity"}
# Adiposity sites named with "fat" or "adipose" (audit WP15 repair: ``visceral_fat`` and
# ``trunk_fat`` pushed no dispute), and hip circumference, an adiposity measure as waist is.
_PAIRS.update({(site, w): "adiposity" for site in ("visceral", "trunk", "android", "gynoid",
                                                    "abdominal", "liver", "hepatic")
               for w in ("fat", "adipose", "adiposity")})
_PAIRS.update({("hip", w): "hip size" for w in ("circumference", "circ", "cm", "girth")})
# Spellings that run words together, read on the name's joined words (``BodyMassIndex``,
# ``body_mass_index``; the paediatric BMI z-scores ``bmiz``, ``zbmi``, ``BMIz``; ``weightloss_kg``;
# gestational weight gain ``GWG``).
_JOINED = (
    (re.compile(r"bodymassindex|bmiz|zbmi|bmisds?|bmipct|bmipercentile|bmicentile"), "BMI"),
    (re.compile(r"weightloss|weightgain|weightchange|^gwg|gestationalweightgain"), "body weight"),
    (re.compile(r"fatmass|bodyfat|visceralfat|trunkfat|androidfat|gynoidfat"), "adiposity"),
    (re.compile(r"waistcirc|waisthip|waisttohip|waistheight"), "waist size"),
    (re.compile(r"hipcirc"), "hip size"),
)
# A "weight" that is not the participant's body: a survey's sampling weight, or a birth weight.
_NOT_BODY = {"sample", "sampling", "survey", "svy", "design", "birth", "pweight", "probability"}

# Child growth, as the growth references name their indices (audit WP13 gate repair: the WHO
# ``anthro`` package's own outputs went unread). WHO anthro (R package documentation): "zlen
# Length/height-for-age z-score", "zwei Weight-for-age z-score", "zwfl Weight-for-length/height
# z-score", "zbmi BMI-for-age z-score", "zac Arm circumference-for-age z-score", "zts Triceps
# skinfold-for-age z-score", "zss Subscapular skinfold-for-age z-score"; the survey literature's
# WAZ, HAZ/LAZ, WHZ/WLZ and BAZ, and the same indices spelled ``wfa_z``, ``wfh_z``, ``hfa_z``.
# Each is body size or adiposity in a growing child, which energy intake feeds and which drives
# intake in turn: the pack's energy-related list (weight, BMI, adiposity) for children. A flag
# beside the index (WHO anthro's ``fwei``, ``zwei_flag``) is not the index.
_GROWTH_INDEX: dict[str, str] = {
    **dict.fromkeys(("zwei", "waz", "wfaz", "wfa", "zwfa"), "body weight"),
    **dict.fromkeys(("zwfl", "zwfh", "whz", "wlz", "wfhz", "wflz", "wfh", "wfl"), "body weight"),
    **dict.fromkeys(("zbmi", "baz", "bfaz", "bmiz", "bfa", "bmifa"), "BMI"),
    **dict.fromkeys(("zlen", "zhgt", "haz", "laz", "hfaz", "lfaz", "hfa", "lfa"), "child growth"),
    **dict.fromkeys(("zac", "zmuac", "acfaz", "muacz", "muac"), "body size"),
    **dict.fromkeys(("zts", "zss", "tsfaz", "ssfaz"), "adiposity"),
}
_INDEX_QUALIFIERS = {"z", "zscore", "score", "sd", "sds", "who", "cdc", "who2006", "cdc2000",
                     "iotf", "ref"}
# Standard abbreviations of the pack's own list (weight, BMI, adiposity, diabetes; waist): body
# weight in animal diet studies (``BW``, ``bw_g``, ``final_bw``, ``BWG``), visceral adipose tissue
# (``VAT``), the fat-mass index (``FMI``), waist circumference and waist-to-height (``WC``,
# ``WHtR``), diabetes as ``DM`` (``incident_dm``, MESA's ``dm031c``) or ``diab``; and the NHANES
# examination and DXA names (BMX_J "BMXHIP - Hip Circumference (cm)"; DXX_J "DXDTOPF - Total
# Percent Fat", "DXDTOFAT - Total Fat (g)", "DXXTRFAT - Trunk Fat (g)").
_ABBREVIATIONS: dict[str, str] = {
    **dict.fromkeys(("bw", "bwg", "bwgain", "bodywt", "bwt"), "body weight"),
    **dict.fromkeys(("vat", "fmi", "fmindex"), "adiposity"),
    **dict.fromkeys(("wc", "whtr", "wthr"), "waist size"),
    **dict.fromkeys(("dm", "diab", "t1dm", "iddm", "niddm"), "diabetes"),
    "bmxhip": "hip size",
    **dict.fromkeys(("dxdtopf", "dxdtofat", "dxxtrfat", "dxxtrpf", "dxdtrpf", "dxxandfat",
                     "dxxgynfat"), "adiposity"),
}
_ABBREVIATION_JOINED = re.compile(r"^(?:dm\d+[a-z]?|bw\d+)$")
_NOT_AN_INDEX = {"flag", "flags", "flagged", "imputed", "missing", "f"}


def _tokens(name: str) -> list[str]:
    """The name's words by the one tokenizer every name reading shares (audit WP13): it also splits
    a run of capitals from the word after it, so ``BMIChange`` and ``T2DIncident`` read."""
    from turbotab.core.recognizers import tokens

    return tokens(name)


def energy_related(outcome: str | None) -> str | None:
    """What the outcome's name says it is, when that is energy-related; else None.

    ``"body weight"``, ``"BMI"``, ``"waist size"``, ``"adiposity"`` or ``"diabetes"``, read from
    whole tokens of the name (``weight_kg`` → body weight; ``bmi_change`` → BMI). A weight named
    with a sampling or birth token is not the participant's body weight.
    """
    if not outcome:
        return None
    tokens = _tokens(outcome)
    words = set(tokens)
    for a, b in zip(tokens, tokens[1:]):
        if (a, b) in _PAIRS:
            return _PAIRS[(a, b)]
    if not words & _NOT_AN_INDEX:
        core = [t for t in tokens if t not in _INDEX_QUALIFIERS]
        joined_core = "".join(core)
        index = (_GROWTH_INDEX.get(joined_core)
                 or next((_GROWTH_INDEX[t] for t in core if t in _GROWTH_INDEX), None))
        if index is not None and (len(core) <= 2 or joined_core in _GROWTH_INDEX):
            return index
        for t in [*tokens, "".join(tokens)]:
            if t in _ABBREVIATIONS:
                kind = _ABBREVIATIONS[t]
                if kind == "body weight" and _NOT_BODY & words:
                    continue
                return kind
            if _ABBREVIATION_JOINED.match(t):
                return "diabetes" if t.startswith("dm") else "body weight"
    joined = "".join(tokens)
    for pattern, kind in _JOINED:
        if pattern.search(joined) and not (kind == "body weight" and _NOT_BODY & set(tokens)):
            return kind
    for t in tokens:
        kind = _KIND_BY_TOKEN.get(t)
        if kind == "body weight" and _NOT_BODY & set(tokens):
            continue
        if kind is not None:
            return kind
    return None


# An outcome the names cannot place is read against the table's own body-size columns: one that
# tracks a weight, BMI, waist or fat measure closely is a body-size outcome whatever it is called.
# TurboTab's own cut-off (|r| >= 0.7: half the variance shared).
RELATED_R = 0.7
_BODY_KINDS = ("body weight", "BMI", "waist size", "adiposity", "hip size", "body size",
               "child growth")


def recognized_otherwise(outcome: str) -> bool:
    """The outcome's name reads as something the pack does not list as energy-related: a clinical
    analyte (lipids, glucose, HbA1c, blood pressure, CRP …), a nutrient or total energy."""
    from turbotab.core.recognizers import is_nutrient, reads_as_total_energy
    from turbotab.core.units import analyte_of

    return bool(analyte_of(outcome) or is_nutrient(outcome) or reads_as_total_energy(outcome)
                or _crp_like(outcome) or _NHANES_NOT_BODY.match(str(outcome).upper()))


# NHANES variable names carry their file's prefix: LBX/LBD a laboratory result (TRIGLY_J
# "LBDLDNSI - LDL-Cholesterol, NIH equation 2 (mmol/L)"), BPX a blood pressure, URX/URD a urine
# result. Each is a measure the pack does not list as energy-related; the body-measures (BMX) and
# DXA (DXX, DXD) files are read by name above.
_NHANES_NOT_BODY = re.compile(r"^(?:LB[XD]|BPX|BPXO|URX|URD)[A-Z0-9]+$")


def _crp_like(outcome: str) -> bool:
    from turbotab.core.detectors.plausibility import variable_of

    try:
        return variable_of(outcome) is not None
    except Exception:  # noqa: BLE001 - the band table is a second reader, never a failure
        return False


def outcome_relation(outcome: str | None, frame: Any = None) -> dict[str, Any] | None:
    """How the outcome stands to energy balance, and how that was read (audit IN-20, WP13 gate
    repair): ``{"kind", "basis", ...}`` with ``basis``

    * ``"name"``: the name reads as energy-related (:func:`energy_related`);
    * ``"values"``: the name does not, but the outcome tracks one of the table's body-size columns
      at |r| >= 0.7 (``via``, ``r``);
    * ``"other"``: the name reads as something the pack does not list (an analyte, a nutrient): no
      dispute;
    * ``"unconfirmed"``: nothing places it. The leash (BLUEPRINT §11.3): the dispute is then
      stated as a condition the researcher answers, never silently dropped."""
    if not outcome:
        return None
    kind = energy_related(outcome)
    if kind is not None:
        return {"kind": kind, "basis": "name"}
    if frame is not None and outcome in getattr(frame, "columns", ()):
        import numpy as np
        import pandas as pd

        y = pd.to_numeric(frame[outcome], errors="coerce")
        best = None
        for c in frame.columns:
            other = energy_related(str(c)) if c != outcome else None
            if other not in _BODY_KINDS:
                continue
            x = pd.to_numeric(frame[c], errors="coerce")
            both = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(both) < 10 or both["x"].nunique() < 3 or both["y"].nunique() < 3:
                continue
            r = float(both["x"].corr(both["y"]))
            if np.isfinite(r) and abs(r) >= RELATED_R and (best is None or abs(r) > abs(best[2])):
                best = (str(c), other, r)
        if best is not None:
            return {"kind": best[1], "basis": "values", "via": best[0], "r": best[2]}
    if recognized_otherwise(outcome):
        return {"kind": None, "basis": "other"}
    return {"kind": None, "basis": "unconfirmed"}


def unconfirmed_line(outcome: str) -> str:
    """The energy card's line when nothing places the outcome (≤ 22 words)."""
    return (f"If {tick(outcome)} measures body size, adiposity or diabetes, energy may be on its "
            f"causal path, and adjusting for it is disputed.")


def unconfirmed_why(outcome: str) -> str:
    """The energy finding's added sentences when nothing places the outcome."""
    return (f"Nothing here says whether {tick(outcome)} is energy-related. If it measures body "
            f"size, adiposity, growth or diabetes, whether to adjust for reported energy is itself "
            f"disputed: energy may lie on the causal path to the outcome, and adiposity drives "
            f"under-reporting, so adjusting can be over-adjustment and collider bias at once. Say "
            f"which it is; if it is, present the adjusted and the unadjusted models.")


def values_line(outcome: str, kind: str, via: str, r: float) -> str:
    """The energy card's DISPUTED line when the outcome is read by its values (≤ 22 words)."""
    return (f"{tick(outcome)} tracks {tick(via)} (r {r:.2f}), so it reads as {kind}: adjusting "
            f"for energy is disputed.")


def dispute_line(outcome: str, kind: str) -> str:
    """The energy card's DISPUTED line (≤ 22 words, ``COMPOSED_BUDGETS["card_line"]``)."""
    return (f"{tick(outcome)} reads as {kind}: energy may be on its causal path and a collider, "
            f"so adjusting for it is disputed.")


def dispute_why(outcome: str, kind: str) -> str:
    """The energy finding's added sentences when the outcome is energy-related (NUTRITION_PACK §04,
    DISPUTED: "Conditioning on reported total energy here can be simultaneously over-adjustment
    (mediator) and collider-stratification bias. Present both adjusted and unadjusted models, and
    flag this in limitations.")."""
    return (f"{tick(outcome)} reads as {kind}, and whether to adjust for reported energy is then "
            f"itself disputed: energy may lie on the causal path to the outcome, and adiposity "
            f"drives under-reporting, so adjusting can be over-adjustment and collider bias at "
            f"once. Present the adjusted and the unadjusted models and name it as a limitation.")


# ── measurement error, under inference ───────────────────────────────────────

def measurement_error_line(exposures: Sequence[str], energy: Iterable[str] = (), *,
                           reports: float = 1.0) -> str | None:
    """The coefficient table's measurement-error limitation, or None with no dietary exposure.

    ``exposures`` are the self-reported intakes among the model's exposures; ``energy`` the
    total-energy columns also in the model (each error-prone too); ``reports`` how many reports
    each row's values average (``stages.proposals.recall_days``).
    """
    exposures = list(dict.fromkeys(exposures))
    if not exposures:
        return None
    energy = [e for e in dict.fromkeys(energy) if e not in exposures]
    n = len(exposures)
    each = "each " if n > 1 else ""
    how = (f"{each}the mean of {count(int(reports))} reports per person" if reports > 1
           else f"{each}from a single report per row")
    head = (f"{listing(exposures, limit=4)} {plural(n, 'is a', 'are')} self-reported "
            f"{plural(n, 'intake')} measured with error, {how}, and this table does not correct for "
            f"it")
    if n + len(energy) == 1:
        return (f"{head}: as the model's one error-prone exposure, its coefficient is attenuated "
                f"toward zero under classical error (Keogh et al. 2020).")
    with_energy = f" (here also {listing(energy, limit=2)})" if energy else ""
    return (f"{head}: with several error-prone intakes in one model{with_energy}, a coefficient can "
            f"be attenuated, inflated or change sign (Freedman et al. 2011).")


__all__ = ["DISPUTED", "ENERGY_TITLE", "ENERGY_WHY", "dispute_line", "dispute_why",
           "energy_related", "measurement_error_line", "outcome_relation", "recognized_otherwise",
           "unconfirmed_line", "unconfirmed_why", "values_line"]

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
from typing import Iterable, Sequence

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
# A "weight" that is not the participant's body: a survey's sampling weight, or a birth weight.
_NOT_BODY = {"sample", "sampling", "survey", "svy", "design", "birth", "pweight", "probability"}


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def energy_related(outcome: str | None) -> str | None:
    """What the outcome's name says it is, when that is energy-related; else None.

    ``"body weight"``, ``"BMI"``, ``"waist size"``, ``"adiposity"`` or ``"diabetes"``, read from
    whole tokens of the name (``weight_kg`` → body weight; ``bmi_change`` → BMI). A weight named
    with a sampling or birth token is not the participant's body weight.
    """
    if not outcome:
        return None
    tokens = _tokens(outcome)
    for a, b in zip(tokens, tokens[1:]):
        if (a, b) in _PAIRS:
            return _PAIRS[(a, b)]
    for t in tokens:
        kind = _KIND_BY_TOKEN.get(t)
        if kind == "body weight" and _NOT_BODY & set(tokens):
            continue
        if kind is not None:
            return kind
    return None


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
           "energy_related", "measurement_error_line"]

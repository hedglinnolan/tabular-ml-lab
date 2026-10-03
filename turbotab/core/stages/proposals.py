"""The ``proposals`` stage: what the field usually does, offered for the exclusions and
energy-adjustment questions — never pre-selected (M1_CONTRACT §3).

Two readings, both from ``docs/turbotab/research/NUTRITION_PACK.md``:

* **Exclusions** (§02, Diagnostic 1). The fixed kcal screens in circulation — Willett 2013's
  sex-specific 500–3,500 kcal/d for women and 800–4,000 for men, the Nurses' Health Study and
  Health Professionals Follow-up Study's 500–3,500 and 800–4,200, and the sex-neutral 500–5,000
  and 500–3,500 — each as an :class:`~turbotab.core.decisions.ExclusionRule` with the number of
  rows it would remove. The pack says the conventions *genuinely differ across literatures* and
  that the app must show how N moves with the choice, so every screen is offered with its count
  and its CONVENTION badge, and none is chosen. Each is attributed to its source (audit MI-02):
  Banna et al. 2017 (*Front Nutr* 4:45), quoting Willett's *Nutritional Epidemiology* (3rd ed.,
  2013): "an allowable range of 800–4,000 kcal/day for men may be used"; Pan et al. 2011 (*AJCN*
  94:1088), across NHS, NHS II and HPFS: "daily energy intake <800 or >4200 kcal/d for men and
  <500 or >3500 kcal/d for women".
* **Energy** (§04). The energy column, the energy-bearing nutrients, the strata a residual can be
  computed within, which of the five models can run on these columns, and the field's usual
  method (the Willett residual, CONVENTION) — first in order, never selected.
* **Missing values** (M1_CONTRACT §12.4, under every lens). Each predictor with blanks, how many,
  and whether the blanks likely mean "not asked": a mostly blank (≥ 50%) yes/no or
  medication-like column, where a blank is a skipped question rather than an unknown value
  (DRIVE_RUBRIC §4: median fill on ``meds_hbp`` would put every unknown on medication). The
  missing-values question offers to leave those columns out before complete cases.
* **Coach lines** (M2_CONTRACT §6). ``coach`` holds at most one data-grounded line per decision
  card — the exclusions card (rows outside the pack's plausible intake) and the missing-values card
  (the blankest "not asked" column) — from :func:`turbotab.core.coach.card_lines`. Never a choice.

Counts are made as the participant flow makes them: among rows with the outcome measured, when an
outcome has been chosen. :func:`rule_excludes` is the one definition of what a rule removes.

Owner: the M1 "voice" agent.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.decisions import ExclusionRule, RangeByLevel
from turbotab.core.graph import StageContext

EXCLUSION_EVIDENCE = {
    "status": "CONVENTION",
    "source": "research/NUTRITION_PACK.md#02 · Implausible intake exclusions",
}
ENERGY_EVIDENCE = {
    "status": "CONVENTION",
    "source": "research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature",
}
KCAL_PER_KJ = 4.184
# Audit IN-07's remedy: "refuse a screen that would remove more than half the rows, with a units
# exit". An intake screen removes the implausible few; one that would remove most rows says the
# energy column's unit is not the one the screen reads it in (a child's kJ, a weekly total).
MOST_ROWS = 0.5
MAX_STRATA_LEVELS = 10
USUAL_METHOD = "residual"
NOT_ASKED_SHARE = 0.5  # a yes/no or medication column blank on at least this share
MAX_MISSING_COLUMNS = 200  # predictors with blanks the reading lists, most blank first
PREDICTOR_ROLES = ("exposure", "covariate", "energy")
_MEDICATION = {
    "med", "meds", "medication", "medications", "medicine", "medicines", "rx", "drug", "drugs",
    "prescription", "prescribed", "treated", "treatment", "taking", "insulin", "statin", "statins",
    "antihypertensive", "antihypertensives",
}


# ── what a rule removes ──────────────────────────────────────────────────────

def level_key(value: Any) -> str:
    """A level as a rule names it: ``1.0`` and ``1`` are the same level, text is stripped."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return str(int(value)) if float(value).is_integer() else repr(float(value))
    return str(value).strip()


def rule_excludes(frame: pd.DataFrame, rule: Any) -> pd.Series:
    """True where ``rule`` removes the row.

    A row is removed when its value is below ``low`` or above ``high`` (both inclusive bounds are
    kept). With ``by``, a row whose level of ``by.column`` has its own range is judged by that
    range, and any other row by ``low``/``high``. A missing value is never removed by a range:
    missing values are the missing-data question's, not an exclusion's.
    """
    from turbotab.core.decisions import as_rule

    rule = as_rule(rule)
    if getattr(rule, "kind", "range") == "goldberg":
        # Judged outside its cut-offs; a row it cannot screen is never removed here either.
        from turbotab.core.methods.misreporting import screen

        s = screen(frame, rule)
        return (s["screened"] & ~s["inside"]).astype(bool)
    values = pd.to_numeric(frame[rule.column], errors="coerce")
    low = pd.Series(np.nan if rule.low is None else float(rule.low), index=frame.index)
    high = pd.Series(np.nan if rule.high is None else float(rule.high), index=frame.index)
    if rule.by is not None and rule.by.ranges:
        levels = frame[rule.by.column].map(level_key)
        for level, (lo, hi) in rule.by.ranges.items():
            at = levels == level_key(level)
            low[at] = np.nan if lo is None else float(lo)
            high[at] = np.nan if hi is None else float(hi)
    below = (values < low).fillna(False)
    above = (values > high).fillna(False)
    return (below | above).astype(bool)


# ── recognizing the columns ──────────────────────────────────────────────────

_NUMERIC = ("numeric", "integer")
_TEXTUAL = ("categorical", "text", "boolean")
_SEX_NAMES = {"sex", "gender", "riagendr", "sex_at_birth", "gender_identity"}
_FEMALE = {"f", "female", "woman", "women", "w", "girl", "fem"}
_MALE = {"m", "male", "man", "men", "boy", "masc"}
_NHANES_SEX = {"1": "male", "2": "female"}  # RIAGENDR, NHANES codebook
_FLAG = re.compile(r"^(imputed|flag|is_imputed)_|_(imputed|flag|imp)$", re.I)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def _dtype(info: Mapping[str, Any] | None) -> str:
    return str((info or {}).get("dtype") or "")


def is_energy_name(column: str) -> bool:
    """The name reads as total energy intake: the one recognizer every stage shares
    (:func:`turbotab.core.recognizers.reads_as_total_energy`; whole words, so ``alc_kcal`` is
    alcohol's energy and ``energy_expenditure_kcal`` is not intake)."""
    from turbotab.core.recognizers import reads_as_total_energy

    return reads_as_total_energy(column)


_INTAKE_WORDS = {"intake", "intakes", "ei", "tei", "dietary", "diet", "food", "foods", "consumed",
                 "consumption", "reported"}


def energy_column(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str]) -> str | None:
    """The total-energy column: the one the roles name, else the one its name declares.

    Among names that read as total energy intake, one whose median is no day's energy in kcal or
    kJ is passed over (:func:`turbotab.core.recognizers.energy_median_contradicts`, when the
    column's summary carries a median); then a name that says intake (``energy_intake``,
    ``DR1TKCAL``, ``ENERC_KCAL``, ``TEI``) comes first, then one in kcal, then any other, in table
    order. The unit alone never outranks the word "intake" (audit WP13 repair: ``exercise_kcal``
    outranked ``energy_intake``)."""
    from turbotab.core.recognizers import energy_median_contradicts, tokens

    named = [c for c, r in roles.items() if r == "energy" and c in columns]
    if named:
        return named[0]
    def codebook(c: str) -> bool:
        # The codebook states what the variable is and its unit (DR1TOT_L "DR1TKCAL - Energy
        # (kcal)"; NUTR_DEF ENERC_KCAL): its values are not second-guessed here, and a median no
        # day's intake has is the implausible-intake finding's unit question instead.
        return bool(re.fullmatch(r"(?i)(DR[12X][TI]KCAL|ENERC_?KCAL|ENERC_?KJ|ENER_?KCAL|"
                                 r"ENER_?KJ)", str(c)))

    candidates = [c for c, info in columns.items()
                  if _dtype(info) in _NUMERIC and is_energy_name(c)
                  and roles.get(c) not in ("identifier", "cluster", "flag", "design", "time",
                                           "excluded")
                  and (codebook(c) or energy_median_contradicts((info or {}).get("median")) is None)]
    if not candidates:
        return None
    order = list(columns)

    def rank(c: str) -> tuple[int, int, int]:
        words = set(tokens(c))
        says_intake = 0 if (words & _INTAKE_WORDS or codebook(c)) else 1
        in_kcal = 0 if words & {"kcal", "kcals", "calories", "calorie", "kilocalories"} \
            or str(c).upper().endswith("KCAL") else 1
        return says_intake, in_kcal, order.index(c)

    return sorted(candidates, key=rank)[0]


def energy_bearing(column: str) -> bool:
    """A nutrient whose amount carries energy by a known Atwater factor (not a share of energy)."""
    from turbotab.core.methods.energy import energy_factor, nutrient_role, unit_of

    if unit_of(column) == "density" or is_energy_name(column):
        return False
    try:
        role = nutrient_role(column)
    except ValueError:
        return False
    return role is not None and energy_factor(column).factor is not None


def _with_medians(info: Mapping[str, Mapping[str, Any]],
                  frame: pd.DataFrame | None) -> dict[str, Mapping[str, Any]]:
    """The column records, each energy-named one carrying its median from ``frame`` when its
    record has none (what :func:`energy_column` checks a day's energy by)."""
    out = dict(info)
    if frame is None:
        return out
    for c, i in info.items():
        if (i or {}).get("median") is None and c in frame.columns and is_energy_name(c):
            values = pd.to_numeric(frame[c], errors="coerce")
            if values.notna().any():
                out[c] = {**(i or {}), "median": float(values.median())}
    return out


def nutrient_candidates(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str], *,
                        energy: str | None, target: str | None,
                        frame: pd.DataFrame | None = None) -> list[str]:
    """The energy-bearing nutrients energy adjustment works with: the exposures the roles name
    (the roles stage has already read their values), or, with no roles, every column the name
    reads as one whose values corroborate it (:func:`turbotab.core.recognizers.intake_check`,
    read on ``frame`` when given)."""
    from turbotab.core.recognizers import intake_check

    out = []
    for c, info in columns.items():
        if c in (energy, target) or _dtype(info) not in _NUMERIC or _FLAG.search(c):
            continue
        if roles and roles.get(c) not in (None, "exposure"):
            continue
        if not energy_bearing(c):
            continue
        if not roles and frame is not None and c in frame.columns:
            e = frame[energy] if energy and energy in frame.columns else None
            check = intake_check(c, frame[c], energy=e)
            if check is not None and not check.corroborated:
                continue
        out.append(c)
    return out


def not_adjusted(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str], *,
                 energy: str | None, target: str | None, nutrients: Sequence[str]) -> list[dict[str, str]]:
    """Exposures energy adjustment leaves as they are, each with a short reason (never silently)."""
    from turbotab.core.methods.energy import energy_factor
    from turbotab.core.stages.rows import _is_nutrient

    out = []
    for c, info in columns.items():
        if c in (energy, target) or c in nutrients or _dtype(info) not in _NUMERIC or _FLAG.search(c):
            continue
        # The exposures, as the roles have them; before any roles, the columns named as nutrients.
        if roles.get(c) != "exposure" and (roles or not _is_nutrient(c) or is_energy_name(c)):
            continue
        reading = energy_factor(c)
        if reading.unit == "density":
            reason = "already a share of energy"
        elif reading.factor is None and "more than one" in reading.reason:
            reason = "names two nutrients, so its energy is ambiguous"
        elif reading.role is not None and reading.unit in ("milligrams", "micrograms", "IU"):
            reason = "not in grams, which the energy factors need"
        elif not _is_nutrient(c) and reading.unit in ("grams", "kcal", "kj"):
            # A food or food group (``fatty_fish_g``) carries energy, but by no Atwater factor
            # the app knows: fatty fish is about 2 kcal/g, not fat's 9 (audit IN-01).
            reason = "a food, not a nutrient: no energy factor unless one is declared"
        else:
            reason = "carries no energy"
        out.append({"column": c, "reason": reason})
    return out


def sex_column(columns: Mapping[str, Mapping[str, Any]], frame: pd.DataFrame | None,
               roles: Mapping[str, str], *, screen: bool = False) -> tuple[str | None, dict[str, str]]:
    """The sex column and which of its levels are ``female`` and ``male``. For a screen
    (``screen``), a column left out of the model still counts."""
    skip = (("identifier", "cluster", "flag", "design") if screen
            else ("identifier", "cluster", "flag", "design", "excluded"))
    for c in columns:
        if c.lower() not in _SEX_NAMES and not ({"sex", "gender"} & set(_tokens(c))):
            continue
        if roles.get(c) in skip:
            continue
        if frame is None or c not in frame.columns:
            continue
        mapping: dict[str, str] = {}
        for value in frame[c].dropna().unique():
            key = level_key(value)
            low = key.lower()
            if c.lower() == "riagendr" and key in _NHANES_SEX:
                mapping[key] = _NHANES_SEX[key]
            elif low in _FEMALE:
                mapping[key] = "female"
            elif low in _MALE:
                mapping[key] = "male"
        if set(mapping.values()) == {"female", "male"}:
            return c, mapping
    return None, {}


def strata_candidates(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str], *,
                      exclude: Iterable[str], sex: str | None) -> list[str]:
    """Categorical columns with 2–10 levels a residual could be computed within; sex first."""
    skip = set(exclude)
    out = []
    for c, info in columns.items():
        if c in skip or _FLAG.search(c):
            continue
        if roles.get(c) in ("identifier", "cluster", "flag", "design", "excluded", "energy",
                            "exposure"):
            continue
        n_unique = int((info or {}).get("n_unique") or 0)
        if c != sex and (_dtype(info) not in _TEXTUAL or not 2 <= n_unique <= MAX_STRATA_LEVELS):
            continue
        out.append(c)
    if sex in out:
        out.remove(sex)
        out.insert(0, sex)
    return out


OPTION_REASON_WORDS = 20  # teaching.COMPOSED_BUDGETS["option_reason"]
_UNIT_UNSAID = re.compile(r"does not say what unit|could not confirm it is grams")
_MIXED_UNITS = re.compile(r"no single factor to apply")
_TICKED = re.compile(r"`([^`]+)`")


def option_reason(reason: str) -> str:
    """Why a method cannot run, as the option shows it: one sentence within the budget.

    The methods' own reasons are written for the methods record and can run to 50 words; the card
    shows one sentence beside the option (DRIVE_RUBRIC §2.5: no walls). The known long ones are
    restated; anything else keeps its last sentence (the specific cause, as the card reads it),
    else its first.
    """
    from turbotab.core.consequences import clip_words
    from turbotab.core.voice import finish, listing, words

    if words(reason) <= OPTION_REASON_WORDS:
        return reason
    if _MIXED_UNITS.search(reason):
        # The Atwater reading's mixed-units verdict (turbotab/nutrition.py): restated whole, never
        # cut mid-clause (repair round: it read "…; the rows have.").
        return finish("Declared and reconstructed energy differ by no single factor across rows, a "
                      "multi-source merge: separate the rows by source first")
    if _UNIT_UNSAID.search(reason):
        head = reason.split(" does not say", 1)[0].split(" could not", 1)[0]
        columns = _TICKED.findall(head)
        if columns:
            say = "does" if len(columns) == 1 else "do"
            return finish(f"{listing(columns, limit=3)} {say} not say grams, and this table cannot "
                          f"confirm it")
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", reason)
    for sentence in (sentences[-1], sentences[0]):
        if words(sentence) <= OPTION_REASON_WORDS and _TICKED.search(sentence):
            return finish(sentence)
    # Clipped at a clause, never inside one: the longest run of whole clauses within the budget.
    last = sentences[-1]
    clauses = re.split(r"(?<=[;:])\s+", last.rstrip("."))
    kept = ""
    for clause in clauses:
        joined = f"{kept} {clause}".strip()
        if words(joined) > OPTION_REASON_WORDS:
            break
        kept = joined
    if kept:
        return finish(kept.rstrip(";:"))
    return finish(clip_words(last, OPTION_REASON_WORDS))


def nested_reason(nested: Mapping[str, str], nutrients: Sequence[str]) -> str | None:
    """The partition option's reason when a total sits beside its parts, within the budget.

    This is where the energy card's old nested-parts note now lives (M2_CONTRACT §6): one short
    reason on the option it is about, and the term card for "nested" says what follows from it
    (a substitution moves the parts with their total).
    """
    from turbotab.core.voice import listing, tick

    parts = [n for n in nutrients if nested.get(n) in nutrients]
    if not parts:
        return None
    parents = list(dict.fromkeys(nested[n] for n in parts))
    are = "is" if len(parts) == 1 else "are"
    return (f"{listing(parts, limit=3)} {are} nested in {listing(parents, limit=2)}: a partition "
            f"would count that energy twice.")


def energy_unit_reading(frame: pd.DataFrame, energy: str) -> dict[str, str]:
    """The energy column's unit and how it was read (audit IN-07), first signal that speaks:

    1. ``name``: a ``_kj`` suffix;
    2. ``atwater``: the reconstruction from the macronutrients (NUTRITION_PACK §01: a ratio near
       4.18 is kJ, near 1 is kcal), which outranks a ``kcal`` in the name;
    3. ``name``: a ``kcal`` suffix or word, or a codebook that states kcal (``DR1TKCAL``);
    4. ``magnitude``: with no macronutrients to reconstruct from, the pack's median-magnitude prior
       ("energy 1,600–2,600 kcal (7,000–11,000 → kJ)");
    5. ``assumed``: kcal, said as an assumption.
    """
    from turbotab.core.methods.energy import atwater_check, unit_of
    from turbotab.core.recognizers import ENERGY_PRIOR_SOURCE, energy_unit, energy_unit_by_magnitude
    from turbotab.core.voice import tick

    col = tick(energy)
    if unit_of(energy) == "kj":
        return {"unit": "kj", "basis": "name", "sentence": f"{col} says kJ in its name."}
    try:
        reading = atwater_check(frame, energy) if energy in frame.columns else None
    except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
        reading = None
    if reading is not None and reading.verdict == "energy_in_kj":
        return {"unit": "kj", "basis": "atwater",
                "sentence": f"{col} is about {reading.ratio:.2f}× the energy its macronutrients "
                            f"carry: kilojoules."}
    if reading is not None and reading.verdict == "pass":
        return {"unit": "kcal", "basis": "atwater",
                "sentence": f"{col} matches the energy its macronutrients carry: kcal."}
    if unit_of(energy) == "kcal" or energy_unit(energy) == "kcal":
        return {"unit": "kcal", "basis": "name", "sentence": f"{col} says kcal in its name."}
    if energy in frame.columns:
        by = energy_unit_by_magnitude(frame[energy])
        if by is not None:
            median = float(pd.to_numeric(frame[energy], errors="coerce").median())
            word = "kJ" if by == "kj" else "kcal"
            return {"unit": by, "basis": "magnitude",
                    "sentence": f"{col}'s median, {tick(f'{median:,.0f}')}, is a day's energy in "
                                f"{word} ({ENERGY_PRIOR_SOURCE})."}
    median = (float(pd.to_numeric(frame[energy], errors="coerce").median())
              if energy in frame.columns else float("nan"))
    if math.isfinite(median) and median > 5_000:
        # No population eats a median of more than 5,000 kcal a day: the column is in kJ outside
        # the adult band (children), or a total over more than a day. Kcal is only the reading the
        # counts are made in, and the screens refuse what it would remove (audit IN-07).
        return {"unit": "kcal", "basis": "assumed",
                "sentence": f"Nothing says what unit {col} is in, and its median, "
                            f"{tick(f'{median:,.0f}')}, is no day's intake in kcal; kcal is assumed "
                            f"only to count, so check the unit."}
    return {"unit": "kcal", "basis": "assumed",
            "sentence": f"Nothing says what unit {col} is in; kcal is assumed."}


def _energy_unit(frame: pd.DataFrame, energy: str) -> str:
    """``kj`` when the column is in kilojoules, else ``kcal`` (:func:`energy_unit_reading`)."""
    return energy_unit_reading(frame, energy)["unit"]


# ── the proposals ────────────────────────────────────────────────────────────

def _kcal(value: float) -> str:
    return f"{value:,.0f}"


# The sex-specific screens, each with its men's upper bound and the source its rule's reason names
# (audit MI-02): (key, label, men's upper kcal/d, reason source). Women's range is 500–3,500 in both.
SEX_SPECIFIC_SCREENS: tuple[tuple[str, str, float, str], ...] = (
    ("willett_2013_by_sex", "Willett 2013", 4000.0, "Willett 2013's sex-specific cut-offs"),
    ("nhs_hpfs_by_sex", "NHS/HPFS", 4200.0,
     "the Nurses' Health Study and Health Professionals Follow-up Study cut-offs"),
)


def exclusion_proposals(frame: pd.DataFrame, *, energy: str, unit: str, sex: str | None,
                        sex_levels: Mapping[str, str], base: pd.Series) -> list[dict[str, Any]]:
    """The pack's fixed kcal screens, each with the rows it would remove from ``base``."""
    factor = KCAL_PER_KJ if unit == "kj" else 1.0
    per_day = "kcal a day"

    def bounds(low: float, high: float) -> tuple[float, float]:
        return round(low * factor, 1), round(high * factor, 1)

    def in_kj(low: float, high: float) -> str:
        if unit != "kj":
            return ""
        lo, hi = bounds(low, high)
        return f" ({_kcal(lo)}–{_kcal(hi)} kJ)"

    screens: list[tuple[str, str, ExclusionRule]] = []
    if sex is not None:
        women = [k for k, v in sex_levels.items() if v == "female"]
        men = [k for k, v in sex_levels.items() if v == "male"]
        for key, name, men_high, source in SEX_SPECIFIC_SCREENS:
            ranges: dict[str, tuple[float | None, float | None]] = {}
            for k in women:
                ranges[k] = bounds(500, 3500)
            for k in men:
                ranges[k] = bounds(800, men_high)
            screens.append((
                key,
                f"{name}, by sex: women 500–3,500 and men 800–{_kcal(men_high)} {per_day}"
                + (" (compared in kJ)" if unit == "kj" else ""),
                ExclusionRule(column=energy, by=RangeByLevel(column=sex, ranges=ranges),
                              reason=f"implausible intakes ({source})"),
            ))
    for low, high in ((500, 5000), (500, 3500)):
        lo, hi = bounds(low, high)
        screens.append((
            f"sex_neutral_{low}_{high}",
            f"Sex-neutral: {_kcal(low)}–{_kcal(high)} {per_day}{in_kj(low, high)}",
            ExclusionRule(column=energy, low=lo, high=hi,
                          reason=f"implausible intakes (sex-neutral {_kcal(low)}–{_kcal(high)} kcal a day)"),
        ))
    out = []
    present = int((pd.to_numeric(frame[energy], errors="coerce").notna() & base).sum())
    for key, label, rule in screens:
        affected = int((rule_excludes(frame, rule) & base).sum())
        out.append({"key": key, "rule": rule.model_dump(mode="json"), "label": label,
                    "affected": affected, "evidence": dict(EXCLUSION_EVIDENCE),
                    "refused": screen_refusal(energy, affected, present)})
    return out


def screen_refusal(energy: str, affected: int, present: int) -> str | None:
    """Why an intake screen is refused (it would remove more than half the rows that hold energy),
    or None. The sentence sends the user to the unit, the likelier error (audit IN-07)."""
    from turbotab.core.voice import tick

    if not present or affected <= MOST_ROWS * present:
        return None
    return (f"Refused: it would remove {tick(f'{affected:,}')} of {tick(f'{present:,}')} rows; "
            f"check the unit of {tick(energy)} first.")


# ── the Goldberg screen (audit ME-16) ────────────────────────────────────────
# Read only from exact names, each corroborated by its values' scale: a recognizer that reads a
# substring sends birth weight or a survey weight to body weight (audit IN-01, IN-10).
_AGE_NAMES = {"age", "age_years", "age_yrs", "age_yr", "ageyears", "ridageyr"}
_WEIGHT_NAMES = {"weight", "weight_kg", "wt_kg", "body_weight", "body_weight_kg", "bodyweight",
                 "bmxwt"}
_HEIGHT_NAMES = {"height", "height_cm", "ht_cm", "stature", "stature_cm", "bmxht", "height_m"}
GOLDBERG_PAL = 1.55  # Black 2000: "not necessarily the value of choice"; shown, never chosen
GOLDBERG_EVIDENCE = {
    "status": "CONVENTION",
    "source": "Black 2000, Int J Obes 24:1119 (Goldberg cut-offs for EI:BMR)",
}


# A screen reads a column whatever its model role, so a weight or height left out of the model
# ("excluded") still serves the Goldberg screen (repair round: the menu was too tight); a column
# that names rows, flags others or describes the design is never read as a body measure.
_NOT_A_MEASURE = ("identifier", "cluster", "flag", "design")


def _named(info: Mapping[str, Mapping[str, Any]], frame: pd.DataFrame, names: set[str],
           roles: Mapping[str, str], lo: float, hi: float) -> str | None:
    """The column whose whole name is one of ``names`` and whose median is within [lo, hi]."""
    for c in info:
        if str(c).lower() not in names or c not in frame.columns:
            continue
        if roles.get(c) in _NOT_A_MEASURE:
            continue
        values = pd.to_numeric(frame[c], errors="coerce")
        median = values.median()
        if values.notna().sum() and lo <= float(median) <= hi:
            return str(c)
    return None


def body_columns(info: Mapping[str, Mapping[str, Any]], frame: pd.DataFrame,
                 roles: Mapping[str, str]) -> dict[str, Any]:
    """Age (years), weight (kg) and height (cm or m) columns, when their names and values agree."""
    age = _named(info, frame, _AGE_NAMES, roles, 10, 100)
    weight = _named(info, frame, _WEIGHT_NAMES, roles, 25, 200)
    height = _named(info, frame, _HEIGHT_NAMES, roles, 100, 230)
    unit = "cm"
    if height is None:
        height, unit = _named(info, frame, _HEIGHT_NAMES, roles, 1.0, 2.3), "m"
    return {"age": age, "weight": weight, "height": height, "height_unit": unit}


def goldberg_proposal(frame: pd.DataFrame, info: Mapping[str, Mapping[str, Any]], *, energy: str,
                      unit: str, sex: str | None, sex_levels: Mapping[str, str],
                      roles: Mapping[str, str], target: str | None,
                      base: pd.Series, days: float = 1.0,
                      days_note: str | None = None) -> dict[str, Any] | None:
    """The Goldberg screen with Schofield's BMR, offered with its count when the columns are read.

    PAL 1.55 and the days of intake each row's energy averages (``recall_days``: one, unless the
    working table averaged each person's recalls) are stated in the label; a rule that would read
    the outcome (weight as the outcome) is not offered (audit RO-01).
    """
    from turbotab.core.decisions import GoldbergRule

    body = body_columns(info, frame, roles)
    if sex is None or body["age"] is None or body["weight"] is None:
        return None
    height = body["height"]
    rule = GoldbergRule(
        column=energy, energy_unit="kj" if unit == "kj" else "kcal", days=days, sex=sex,
        female=[k for k, v in sex_levels.items() if v == "female"],
        male=[k for k, v in sex_levels.items() if v == "male"],
        age=body["age"], weight=body["weight"], height=height, height_unit=body["height_unit"],
        equation="schofield_height" if height else "schofield", pal=GOLDBERG_PAL,
        reason="implausible energy reports (Goldberg cut-offs, Black 2000)")
    if target is not None and target in rule.reads():
        return None
    affected = int((rule_excludes(frame, rule) & base).sum())
    eq = "weight and height" if height else "weight"
    n_days = f"{days:g} {'day' if days == 1 else 'days'}"
    label = (f"Goldberg: energy over Schofield BMR ({eq}) outside the 95% cut-offs for PAL "
             f"{GOLDBERG_PAL} and {n_days} of intake" + (f" ({days_note})" if days_note else ""))
    present = int((pd.to_numeric(frame[energy], errors="coerce").notna() & base).sum())
    return {"key": "goldberg_schofield", "rule": rule.model_dump(mode="json"), "label": label,
            "affected": affected, "evidence": dict(GOLDBERG_EVIDENCE),
            "refused": screen_refusal(energy, affected, present)}


def _marked(text: str, columns: Iterable[str]) -> str:
    """Wrap this table's column names in backticks, longest first, whole words only."""
    for c in sorted(set(columns), key=len, reverse=True):
        text = re.sub(rf"(?<![`\w]){re.escape(c)}(?![`\w])", f"`{c}`", text)
    return text


def partition_check(frame: pd.DataFrame, energy: str, nutrients: Sequence[str]) -> dict[str, Any] | None:
    """Why a partition of ``nutrients`` cannot run on ``frame``'s rows, or None (energy.py)."""
    from turbotab.core.methods.energy import partition_refusal
    from turbotab.core.methods.nesting import nested_components

    present = [n for n in nutrients if n in frame.columns]
    nested = nested_components(frame, present)
    return partition_refusal(frame, energy, list(nutrients), nested=nested)


def energy_reading(frame: pd.DataFrame, columns: Mapping[str, Mapping[str, Any]],
                   roles: Mapping[str, str], *, energy: str | None, nutrients: list[str],
                   sex: str | None, target: str | None,
                   purpose: str | None = None) -> dict[str, Any]:
    from turbotab.core.methods.energy import applicable_methods, rank_methods
    from turbotab.core.voice import finish

    from turbotab.core.methods.nesting import nested_components

    verdicts = applicable_methods(list(columns), energy, nutrients)
    if verdicts["partition"]["ok"] and energy in frame.columns:
        # The checks the fit makes on the data (the Atwater reconstruction), and the two it
        # cannot: a total beside its parts, and nutrients that out-weigh total energy.
        refused = partition_check(frame, energy, nutrients)
        if refused is not None:
            present = [n for n in nutrients if n in frame.columns]
            nested = nested_reason(nested_components(frame, present), nutrients)
            verdicts["partition"] = {"ok": False, "reason": nested or refused["reason"]}
            # The all-components model is a partition over every source: it refuses alike.
            if verdicts.get("all_components", {}).get("ok"):
                reason = str(nested or refused["reason"]).replace(
                    "a partition would", "the all-components model would", 1)
                verdicts["all_components"] = {"ok": False, "reason": reason}
    applicability = {}
    for m, v in verdicts.items():
        reason = finish(_marked(str(v["reason"]), columns))
        applicability[m] = {"ok": bool(v["ok"]),
                            "reason": reason if v["ok"] else option_reason(reason)}
    usual = next((m for m in (USUAL_METHOD, "standard") if applicability.get(m, {}).get("ok")), None)
    # Soundness for the declared purpose orders the methods, beside the customary one (north star
    # 5; BLUEPRINT §12 ruling 2): under inference the all-components model first, with the dispute
    # and its precision cost in one line; under prediction a line that the choice matters little.
    ranked = rank_methods(purpose, applicability)
    ranking = {"purpose": purpose if purpose in ("inference", "prediction") else None,
               "order": ranked["order"], "line": ranked["line"]}
    r_with_energy: dict[str, float] = {}
    if energy and energy in frame.columns:
        e = pd.to_numeric(frame[energy], errors="coerce")
        for n in nutrients:
            if n in frame.columns:
                r = e.corr(pd.to_numeric(frame[n], errors="coerce"))
                if r is not None and not math.isnan(r):
                    r_with_energy[n] = round(float(r), 3)
    exclude = [c for c in (energy, target, *nutrients) if c]
    # Audit IN-20: an energy-related outcome (weight, BMI, waist, adiposity, diabetes) pushes the
    # DISPUTED note onto the energy card (NUTRITION_PACK §04: "escalate the mediation/collider
    # warning"), as a card line now and as a badged field for the card's presentation.
    from turbotab.core.methods.dietary_caveats import DISPUTED, dispute_line, energy_related

    notes: list[str] = []
    dispute = None
    kind = energy_related(target)
    if kind is not None and target is not None:
        line = finish(dispute_line(target, kind))
        notes.append(line)
        dispute = {"outcome": target, "kind": kind, "note": line, "evidence": dict(DISPUTED)}
    return {
        "energy_column": energy,
        "nutrients": nutrients,
        "strata_candidates": strata_candidates(columns, roles, exclude=exclude, sex=sex),
        "applicability": applicability,
        "usual": usual,
        "usual_evidence": dict(ENERGY_EVIDENCE) if usual else None,
        "ranking": ranking,
        "r_with_energy": r_with_energy,
        # The nested-parts note that ran ~70 words above the options is folded into the
        # partition option's reason and the "nested" term card; notes stay for data lines that
        # fit the card's budget (COMPOSED_BUDGETS["card_line"]).
        "notes": notes,
        "outcome_dispute": dispute,
        "not_adjusted": not_adjusted(columns, roles, energy=energy, target=target, nutrients=nutrients),
    }


def gappy_predictors(columns: Sequence[Mapping[str, Any]], roles: Mapping[str, str],
                     target: str | None) -> list[str]:
    """Predictors with any blank (by the ingest's counts), most blank first, at most 200."""
    gappy = [c for c in columns if roles.get(str(c["name"])) in PREDICTOR_ROLES
             and str(c["name"]) != target and int(c.get("n_missing") or 0) > 0]
    gappy.sort(key=lambda c: -int(c.get("n_missing") or 0))
    return [str(c["name"]) for c in gappy[:MAX_MISSING_COLUMNS]]


def _yes_no(info: Mapping[str, Any] | None) -> bool:
    return _dtype(info) == "boolean" or 1 <= int((info or {}).get("n_unique") or 0) <= 2


def _blank_reason(share: float, yes_no: bool, medication: bool) -> str:
    from turbotab.core.voice import tick

    pct = tick(f"{share:.0%}")
    if share >= NOT_ASKED_SHARE:
        if yes_no and medication:
            return f"A yes/no medication question blank on {pct} of rows: blank usually means not asked."
        if yes_no:
            return f"A yes/no answer blank on {pct} of rows: blank usually means not asked."
        if medication:
            return f"Medication use blank on {pct} of rows: blank usually means not asked."
        return f"Blank on {pct} of rows; nothing says the blanks mean not asked."
    return f"Blank on {pct} of rows."


def missing_reading(frame: pd.DataFrame, columns: Sequence[Mapping[str, Any]],
                    roles: Mapping[str, str], target: str | None) -> dict[str, Any]:
    """Each predictor with blanks among rows with the outcome measured, and which mean "not asked".

    ``leave_out`` is the offer the missing-values question makes: the likely-not-asked columns,
    and the rows on which at least one of them is blank (the rows complete cases would lose to
    them alone, at most).
    """
    info = {str(c["name"]): c for c in columns}
    base = frame[target].notna() if target and target in frame.columns else pd.Series(True, index=frame.index)
    n_base = int(base.sum())
    out = []
    for c in gappy_predictors(columns, roles, target):
        if c not in frame.columns or not n_base:
            continue
        n = int((frame[c].isna() & base).sum())
        if not n:
            continue
        share = n / n_base
        yes_no = _yes_no(info.get(c))
        medication = bool(set(_tokens(c)) & _MEDICATION)
        out.append({"column": c, "n_missing": n, "share": round(share, 4),
                    "likely_not_asked": bool(share >= NOT_ASKED_SHARE and (yes_no or medication)),
                    "reason": _blank_reason(share, yes_no, medication)})
    out.sort(key=lambda e: (-e["share"], e["column"]))
    likely = [e["column"] for e in out if e["likely_not_asked"]]
    leave_out = None
    if likely:
        n_rows = int((frame[likely].isna().any(axis=1) & base).sum())
        leave_out = {"columns": likely, "n_rows": n_rows,
                     "share": round(n_rows / n_base, 4) if n_base else 0.0}
    return {"columns": out, "leave_out": leave_out}


def build_proposals(frame: pd.DataFrame, columns: Sequence[Mapping[str, Any]], *,
                    lens: Sequence[str] | None, target: str | None,
                    roles: Mapping[str, str] | None = None,
                    purpose: str | None = None,
                    days: tuple[float, str | None] = (1.0, None)) -> dict[str, Any]:
    """The proposals artifact from a frame holding (at least) the columns it reads.

    ``columns`` are the ingest's column records (``name``, ``dtype``, ``n_unique``,
    ``n_missing``); ``roles`` the confirmed roles, else the proposed ones, else empty;
    ``purpose`` the declared purpose, which orders the energy methods by soundness; ``days`` the
    recall days each row's energy averages, with a note when they differ (``recall_days``).
    """
    from turbotab.core.voice import tick

    info = {str(c["name"]): c for c in columns}
    roles = dict(roles or {})
    missing = missing_reading(frame, columns, roles, target)
    # WP7: the missing-data methods, soundest first for the declared purpose, each with its
    # "customary" and "sound" labels and its rung (turbotab.core.methods.missing).
    from turbotab.core.methods.missing import below_detection_options, methods_for

    missing["methods"] = methods_for(purpose)
    missing["below_detection"] = below_detection_options(purpose)
    n_base = int(frame[target].notna().sum()) if target and target in frame.columns else len(frame)
    if "dietary" not in (lens or []):
        return {"exclusions": [], "energy": None, "missing": missing, "n_base": n_base,
                "coach": _card_lines(frame, target=target, energy=None, unit="kcal", missing=missing),
                "basis": "Only the missing-values reading is proposed: the dietary lens is not chosen.",
                "energy_unit": None}
    energy = energy_column(_with_medians(info, frame), roles)
    nutrients = nutrient_candidates(info, roles, energy=energy, target=target, frame=frame)
    sex, sex_levels = sex_column(info, frame, roles)
    if target and target in frame.columns:
        base = frame[target].notna()
        # Every row with the outcome, held-out rows included, and the basis says so: these
        # readings answer questions asked before the seal (M2_CONTRACT §3, the basis audit).
        basis = (f"Counted across all {tick(f'{int(base.sum()):,}')} rows with {tick(target)} "
                 f"measured.")
    else:
        base = pd.Series(True, index=frame.index)
        basis = f"Counted across all {tick(f'{len(frame):,}')} rows; no outcome is chosen yet."
    exclusions: list[dict[str, Any]] = []
    unit = "kcal"
    unit_reading = None
    if energy is not None and energy in frame.columns:
        unit_reading = energy_unit_reading(frame, energy)
        unit = unit_reading["unit"]
        if energy != target:  # an eligibility rule never reads the outcome (audit RO-01)
            # A screen reads a column whatever its model role: sex left out of the model still
            # serves Willett's sex-specific cut-offs, as it serves the Goldberg screen (the methods
            # gate, item E: the menu was too tight, BLUEPRINT §11.3).
            screen_sex, screen_levels = sex_column(info, frame, roles, screen=True)
            by_sex = screen_sex if screen_sex != target else None  # never a rule on the outcome
            exclusions = exclusion_proposals(frame, energy=energy, unit=unit, sex=by_sex,
                                             sex_levels=screen_levels, base=base)
            goldberg = goldberg_proposal(frame, info, energy=energy, unit=unit, sex=screen_sex,
                                         sex_levels=screen_levels, roles=roles, target=target,
                                         base=base, days=days[0], days_note=days[1])
            if goldberg is not None:
                exclusions.append(goldberg)
    reading = None
    if energy is not None or nutrients:
        reading = energy_reading(frame, info, roles, energy=energy, nutrients=nutrients, sex=sex,
                                 target=target, purpose=purpose)
    return {"exclusions": exclusions, "energy": reading, "missing": missing, "n_base": n_base,
            "coach": _card_lines(frame, target=target, energy=energy, unit=unit, missing=missing),
            "basis": basis, "energy_unit": unit_reading}


def _card_lines(frame: pd.DataFrame, *, target: str | None, energy: str | None, unit: str,
                missing: Mapping[str, Any]) -> dict[str, Any]:
    """The decision cards' coach lines, at most one per question (turbotab.core.coach)."""
    from turbotab.core.coach import card_lines

    lines = card_lines(frame, target=target, energy=energy, unit=unit, missing=missing)
    return {key: line.model_dump(mode="json") for key, line in lines.items()}


def needed_columns(columns: Sequence[Mapping[str, Any]], *, target: str | None,
                   roles: Mapping[str, str]) -> list[str]:
    """The few columns the proposals read: energy (every column named as total energy, so the
    values can pass over one that is no day's intake), sex, the outcome and the nutrients."""
    info = {str(c["name"]): c for c in columns}
    energy = energy_column(info, roles)
    energies = [c for c, i in info.items() if _dtype(i) in _NUMERIC and is_energy_name(c)]
    nutrients = nutrient_candidates(info, roles, energy=energy, target=target)
    sexes = [c for c in info if c.lower() in _SEX_NAMES or {"sex", "gender"} & set(_tokens(c))]
    body = [c for c in info if str(c).lower() in _AGE_NAMES | _WEIGHT_NAMES | _HEIGHT_NAMES]
    wanted = [energy, *energies, target, *sexes, *nutrients, *body]
    return list(dict.fromkeys(c for c in wanted if c and c in info))


def roles_from(state_roles: Mapping[str, str] | None, artifact: Any) -> dict[str, str]:
    """The confirmed roles, else the roles stage's proposals (a Bundle, a dict, or None)."""
    if state_roles:
        return dict(state_roles)
    data = getattr(artifact, "data", artifact)
    if isinstance(data, Mapping):
        out = {}
        for entry in data.get("columns") or []:
            column, proposed = entry.get("column"), entry.get("proposed")
            if column and proposed:
                out[str(column)] = str(proposed)
        return out
    return {}


def proposals_stage(ctx: StageContext) -> dict[str, Any]:
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.working import table_info

    state = ctx.state
    columns = table_info(ctx)["columns"]
    roles = roles_from(state.roles, ctx.inputs.get("roles"))
    target = state.target
    wanted = needed_columns(columns, target=target, roles=roles) if "dietary" in (state.lens or []) else []
    wanted = list(dict.fromkeys([*wanted, *([target] if target else []),
                                 *gappy_predictors(columns, roles, target)]))
    with open_store(ctx) as store:  # no column wanted still reads the row ids, so N is right
        ctx.progress(0.2, "Reading the energy, nutrient and blank columns")
        frame = store.materialize(wanted)
        survey = _survey_proposal(state, store)
    ctx.progress(0.6, "Counting what each exclusion rule would remove")
    out = build_proposals(frame, columns, lens=state.lens, target=target, roles=roles,
                          purpose=state.purpose, days=recall_days(ctx.inputs.get("working")))
    out["survey"] = survey
    # WP12a (repair round): the exposure-form question's options, ordered by soundness for the
    # declared purpose, each with its "customary in" and "sound for" labels (north star 5).
    from turbotab.core.methods.exposure_form import options as form_options

    out["exposure_forms"] = form_options(state.purpose)
    return out


def _survey_proposal(state: Any, store: Any) -> dict[str, Any] | None:
    """The survey question's options on this table (audit §5 WP10), read with the roles."""
    from turbotab.core.methods.survey import proposal

    return proposal(state, store) if state.roles is not None else None


def recall_days(working: Any) -> tuple[float, str | None]:
    """How many recalls each analysis row's values average: one, unless the working table combined
    each person's rows by the mean, then the number of rows each person had. When that number
    differs between people the smallest is used (the Goldberg limits widen as days fall, so no one
    is held to a narrower limit than their own) and the note says so (NUTRITION_PACK §02: a
    multi-day cut-off on fewer days over-excludes)."""
    from turbotab.core.stages.working import row_map

    data = getattr(working, "data", None) or {}
    aggregation = data.get("aggregation") or {}
    if aggregation.get("method") != "mean":
        return 1.0, None
    counts = row_map(working).groupby("row_id").size()
    if counts.empty:
        return 1.0, None
    fewest, most = int(counts.min()), int(counts.max())
    if fewest == most:
        return float(fewest), None
    return float(fewest), f"the fewest recalls anyone has; others have up to {most}"


__all__ = [
    "ENERGY_EVIDENCE", "EXCLUSION_EVIDENCE", "build_proposals", "energy_bearing", "energy_column",
    "MOST_ROWS", "exclusion_proposals", "gappy_predictors", "level_key", "missing_reading",
    "needed_columns", "screen_refusal",
    "nested_reason",
    "nutrient_candidates", "partition_check", "proposals_stage", "recall_days",
    "roles_from", "rule_excludes", "sex_column", "strata_candidates",
]

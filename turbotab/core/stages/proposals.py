"""The ``proposals`` stage: what the field usually does, offered for the exclusions and
energy-adjustment questions — never pre-selected (M1_CONTRACT §3).

Two readings, both from ``docs/turbotab-next/reference/research/NUTRITION_PACK.md``:

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


def _codebook_energy(c: str) -> bool:
    """A codebook's total-energy variable (DR1TOT_L "DR1TKCAL - Energy (kcal)"; NUTR_DEF
    ENERC_KCAL; the Framingham FFQ's "NUT_CALOR … CALORIES, (kcal)"): what it is is documented."""
    return bool(re.fullmatch(r"(?i)(DR[12X][TI]KCAL|ENERC_?KCAL|ENERC_?KJ|ENER_?KCAL|ENER_?KJ|"
                             r"NUT_CALOR)", str(c)))


def energy_column_reading(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str],
                          frame: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """The total-energy column and how it was read, or None when no column reads as one.

    The roles name it (``basis: "roles"``), else a name that reads as total energy intake, checked
    by its values where it can be:

    * a column whose median is no day's energy in kcal or kJ is passed over
      (:func:`turbotab.core.recognizers.energy_median_contradicts`), unless a codebook documents it;
    * with the macronutrients in the table, each candidate is read against the energy they carry
      (:func:`turbotab.core.recognizers.energy_against_macros`): one that does not follow them is
      no energy intake whatever its name (a device's energy expenditure, ``Calories`` or
      ``Kcals``), and is listed in ``rejected`` with why; one that follows them is read "by its
      values" (``basis: "values"``) and ranks first;
    * then a name that says intake (``energy_intake``, ``DR1TKCAL``, ``ENERC_KCAL``, ``TEI``), then
      one in kcal, then any other, in table order. The unit alone never outranks the word
      "intake" (audit WP13 repair: ``exercise_kcal`` outranked ``energy_intake``).

    ``basis`` is ``"values"``, ``"codebook"``, ``"name"`` (only the name says so: nothing to check
    it against) or ``"roles"``; ``why`` is the clause a proposal states."""
    from turbotab.core.readings import by_values
    from turbotab.core.recognizers import energy_median_contradicts, macro_candidates, tokens

    named = [c for c, r in roles.items() if r == "energy" and c in columns]
    if named:
        return {"column": named[0], "basis": "roles", "why": "the roles name it", "r": None,
                "rejected": [], "others": []}
    candidates = [c for c, info in columns.items()
                  if _dtype(info) in _NUMERIC and is_energy_name(c)
                  and roles.get(c) not in ("identifier", "cluster", "flag", "design", "time",
                                           "excluded")
                  and (_codebook_energy(c)
                       or energy_median_contradicts((info or {}).get("median")) is None)]
    if not candidates:
        return None
    verdicts: dict[str, Any] = {}
    if frame is not None:
        # Every macronutrient total the names read, so the values choose among duplicates for
        # each candidate (BLUEPRINT §14: an InBody body ``Protein`` beside a recall's ``protein_g``).
        # The registry's one test for total energy (``readings.KIND_RULES["role:energy"]``).
        totals = macro_candidates(frame, exclude=candidates)
        for c in candidates:
            if c in frame.columns:
                verdicts[c] = by_values("role:energy", frame, c, candidates=totals).detail
    rejected = [{"column": c, "why": verdicts[c].why, "r": verdicts[c].r} for c in candidates
                if verdicts.get(c) is not None and not verdicts[c].corroborated
                and not _codebook_energy(c)]
    kept = [c for c in candidates if c not in {x["column"] for x in rejected}]
    if not kept:
        return {"column": None, "basis": None, "why": None, "r": None, "rejected": rejected,
                "others": []}
    order = list(columns)

    def rank(c: str) -> tuple[int, int, int, int]:
        words = set(tokens(c))
        by_values = 0 if verdicts.get(c) is not None and verdicts[c].by_values else 1
        says_intake = 0 if (words & _INTAKE_WORDS or _codebook_energy(c)) else 1
        in_kcal = 0 if words & {"kcal", "kcals", "calories", "calorie", "kilocalories"} \
            or str(c).upper().endswith("KCAL") else 1
        return by_values, says_intake, in_kcal, order.index(c)

    chosen = sorted(kept, key=rank)[0]
    verdict = verdicts.get(chosen)
    if verdict is not None and verdict.by_values:
        basis, why = "values", verdict.why
    elif _codebook_energy(chosen):
        basis, why = "codebook", "its codebook names it total energy"
    elif verdict is not None:
        basis, why = "name", verdict.why
    else:
        basis, why = "name", ("only its name says it is total energy intake; no macronutrients "
                              "are here to check it against")
    # The other names that read as total energy, and why each was not the one: they are not a
    # second energy column, and the roles proposal says so rather than calling them "not a
    # nutrient".
    others = [{"column": c, "r": verdicts[c].r if verdicts.get(c) is not None else None}
              for c in kept if c != chosen]
    return {"column": chosen, "basis": basis, "why": why,
            "r": verdict.r if verdict is not None else None, "rejected": rejected,
            "others": others}


def energy_column(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str],
                  frame: pd.DataFrame | None = None) -> str | None:
    """The total-energy column (:func:`energy_column_reading`), or None."""
    reading = energy_column_reading(columns, roles, frame)
    return reading["column"] if reading is not None else None


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
                        frame: pd.DataFrame | None = None,
                        settled: Iterable[str] | None = None,
                        energy_unit: str | None = None) -> list[str]:
    """The energy-bearing nutrients energy adjustment works with: the exposures the roles name
    (the roles stage has already read their values), or, with no roles, every column the name
    reads as one whose values corroborate it (BLUEPRINT §14 rule 1: plausible amounts rising with
    total energy at r ≥ 0.3, duplicates resolved; read on ``frame`` when given).

    ``settled`` (BLUEPRINT §14 rule 2): the columns whose roles a number-changing default may read
    (:func:`turbotab.core.readings.settled_columns`); an exposure outside it, carried by a bulk
    confirm below high confidence, is no default nutrient until confirmed on its own."""
    from turbotab.core.readings import by_values_table

    allowed = set(settled) if settled is not None else None
    checks = {}
    # Without the settled roles (a caller that has no record of confirmations), only the values'
    # own reading pre-fills the card, whatever roles are given (BLUEPRINT §14 rule 2), by the
    # registry's one test (``readings.KIND_RULES["role:exposure"]``).
    if (not roles or allowed is None) and frame is not None:
        checks = {c: v.detail for c, v in by_values_table(
            "role:exposure", frame, energy=energy if energy in frame.columns else None,
            energy_unit=energy_unit, skip=[c for c in (target,) if c]).items()}
    out = []
    for c, info in columns.items():
        if c in (energy, target) or _dtype(info) not in _NUMERIC or _FLAG.search(c):
            continue
        if roles and roles.get(c) not in (None, "exposure"):
            continue
        if not energy_bearing(c):
            continue
        if allowed is not None and c not in allowed:
            continue
        if (not roles or allowed is None) and frame is not None and c in frame.columns:
            check = checks.get(c)
            if check is not None and not check.by_values:
                continue  # only the values' own reading pre-fills a number-changing default
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
        elif energy_bearing(c):
            # BLUEPRINT §14 rule 2: an energy-bearing exposure the card does not pre-fill was
            # proposed below high confidence (its values do not corroborate it, or it rode along
            # unconfirmed in a bulk confirm); it is adjusted once its role is confirmed on its own.
            reason = "proposed below high confidence; confirm its role to adjust it"
        else:
            reason = "carries no energy"
        out.append({"column": c, "reason": reason})
    return out


def sex_column(columns: Mapping[str, Mapping[str, Any]], frame: pd.DataFrame | None,
               roles: Mapping[str, str], *, screen: bool = False, state: Any = None,
               guess: bool = False) -> tuple[str | None, dict[str, str]]:
    """The sex column and which of its levels are ``female`` and ``male``, as the readings ledger
    holds them (BLUEPRINT §14.3): labels that spell the sexes, or numeric codes as the user
    confirmed them (``confirm_reading`` sex_coding). Numeric codes nobody confirmed are no sex
    column here (NHANES RIAGENDR codes 1 male, 2 female; other studies 1 female), unless ``guess``
    asks for the best guess a question leads with. For a screen (``screen``), a column left out of
    the model still counts."""
    from turbotab.core.readings import parse_sex_coding, sex_coding_reading

    skip = (("identifier", "cluster", "flag", "design") if screen
            else ("identifier", "cluster", "flag", "design", "excluded"))
    for c in columns:
        if c.lower() not in _SEX_NAMES and not ({"sex", "gender"} & set(_tokens(c))):
            continue
        if roles.get(c) in skip:
            continue
        if frame is None or c not in frame.columns:
            continue
        found = sex_coding_reading(state, c, frame[c])
        if found.value is None or not (found.settled or guess):
            continue
        mapping = {level_key(k): v for k, v in (parse_sex_coding(found.value) or {}).items()}
        present = {level_key(v) for v in frame[c].dropna().unique()}
        mapping = {k: v for k, v in mapping.items() if k in present}
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


def recorded_energy_unit(state: Any, energy: str | None) -> Any:
    """What the user recorded for the energy column, unit and days (``set_column_unit``, or a
    unit's or a day count's own confirmation: one store, read through the one accessor,
    ``readings.unit_record``), or None. Either part may be None while unrecorded."""
    from turbotab.core.readings import unit_record

    return unit_record(state, energy)


# The bases that settle the energy unit; a magnitude prior or nothing at all only proposes one
# (audit WP13 gate repair: the toddlers' 4,040 kJ read as kcal set the screens' bounds, the
# implausible-intake count and the coach's "likely over-reporting", and no decision confirmed it).
# The bases that settle an energy column's unit (BLUEPRINT §14.3): the user's record, and the
# Atwater identity where its ratio admits one reading only (near 1: kcal). A name's ``kcal`` or
# ``_kj`` is the best guess a question leads with, never a settlement.
SETTLED_UNIT_BASES = ("decision", "atwater")
ENERGY_WORDS = {"kj": "kJ", "kcal": "kcal"}


def energy_unit_reading(frame: pd.DataFrame, energy: str, recorded: Any = None) -> dict[str, Any]:
    """The energy column's unit and day count as the readings ledger holds them (BLUEPRINT §14.1):
    the unit (:func:`_energy_unit_named`), and how many days each value spans
    (:func:`turbotab.core.readings.day_count_reading`). ``confirmed`` only when both are settled.
    ``recorded`` is what the user recorded (:func:`recorded_energy_unit`): the unit and the days
    are each read from it when recorded, whichever decision recorded them (BLUEPRINT §14.3: the
    fifth gate's ``confirm_reading`` kcal and 4 days were ignored by the screens).

    The Atwater identity settles kcal and says nothing about days: the gate's 2-day totals
    (``energy_kcal_day1_day2``, ``kcal_sum_d1_d2``, ``Energy (kcal) - 2 recalls``) followed their
    2-day macronutrients, were read as one day's intake, and a 5,000 kcal screen removed 135 of
    500 rows where one true daily value exceeded it. A ratio that fits several readings (4.00: kJ,
    or a 4-day kcal total beside daily means) settles neither, and the question offers each as one
    answer, unit and days together (``candidates``)."""
    from turbotab.core.readings import day_count_candidates, day_count_reading
    from turbotab.core.voice import tick

    unit_rec = getattr(recorded, "unit", None) if recorded is not None else None
    days_rec = getattr(recorded, "days", None) if recorded is not None else None
    if isinstance(recorded, Mapping):
        unit_rec, days_rec = recorded.get("unit"), recorded.get("days")
    if unit_rec is not None and str(unit_rec) not in ENERGY_WORDS:
        # A unit recorded for total energy that is no energy unit (``g``): the screens read
        # nothing in it, and say why (an answer is never read as another).
        return {"unit": "kcal", "basis": "not_energy", "days": int(days_rec or 1),
                "confirmed": False, "unit_settled": False, "recorded_unit": str(unit_rec),
                "sentence": f"{tick(energy)} is recorded in {unit_rec}, which is no unit of energy; "
                            f"record kcal or kJ before screening by it."}
    reading = _energy_unit_named(frame, energy, str(unit_rec) if unit_rec is not None else None)
    if reading["basis"] == "days" and days_rec is None:
        return reading
    if days_rec is not None:
        days = int(days_rec)
        reading["days"] = days
        reading["days_recorded"] = True
        reading["confirmed"] = bool(reading.get("unit_settled"))
        if reading["basis"] == "days":
            reading["basis"] = "name"
        if reading["basis"] == "decision":
            word = ENERGY_WORDS[str(reading["unit"])]
            over = f", a total over {days} days" if days > 1 else ""
            reading["sentence"] = f"{tick(energy)} is in {word}{over}, as recorded."
        elif not reading["unit_settled"]:
            reading["sentence"] = (f"{reading['sentence']} Each value spans {days} "
                                   f"day{'s' if days != 1 else ''}, as recorded.")
        return reading
    values = frame[energy] if energy in frame.columns else None
    days = day_count_reading(energy, values, unit=str(reading["unit"]))
    reading["days_reading"] = days.to_dict()
    if not days.settled:
        reading["confirmed"] = False
        reading["days_unsettled"] = True
        found = day_count_candidates(energy, values, unit=str(reading["unit"]))
        found += [d for u, d in reading.get("candidates") or [] if d not in found]
        reading["days_candidates"] = found
        if reading["basis"] != "atwater_ambiguous":
            reading["sentence"] = (f"{reading['sentence']} How many days each value spans is not "
                                   f"settled: {days.evidence}; record it before screening by "
                                   f"{tick(energy)}.")
    return reading


def _energy_unit_named(frame: pd.DataFrame, energy: str, recorded: Any = None) -> dict[str, Any]:
    """The energy column's unit, how it was read, and whether that settles it (audit IN-07), first
    signal that speaks:

    0. ``decision``: the unit the user recorded (``set_column_unit``), with the days each value
       totals;
    1. ``name``: a ``_kj`` suffix;
    2. ``atwater``: the reconstruction from the macronutrients (NUTRITION_PACK §01: a ratio near
       4.18 is kJ, near 1 is kcal), which outranks a ``kcal`` in the name;
    3. ``name``: a ``kcal`` suffix or word, or a codebook that states kcal (``DR1TKCAL``);
    4. ``magnitude``: with no macronutrients to reconstruct from, the pack's median-magnitude prior
       ("energy 1,600–2,600 kcal (7,000–11,000 → kJ)");
    5. ``assumed``: nothing says; kcal is the reading counts are made in, said as an assumption.

    Before any of 1–5, a day count in the name (``kcal_2d``, ``kcal_4day_total``; ``days``) is a
    question: a total over the days or a mean of them (BLUEPRINT §14 rule 3).

    ``confirmed`` is True for the first three. A magnitude or an assumption is a proposal: the
    screens are offered with their counts but refused until the unit is recorded, and no line
    calls a row an under- or over-report in it (the leash, BLUEPRINT §11.3)."""
    from turbotab.core.methods.energy import unit_of
    from turbotab.core.recognizers import (
        ENERGY_PRIOR_SOURCE, day_count, energy_unit, energy_unit_by_magnitude,
    )
    from turbotab.core.voice import tick

    col = tick(energy)
    if recorded is not None:
        unit = str(recorded)
        word = "kJ" if unit == "kj" else "kcal"
        return {"unit": unit, "basis": "decision", "days": 1, "confirmed": True,
                "unit_settled": True, "sentence": f"{col} is in {word}, as recorded."}

    def out(unit: str, basis: str, sentence: str) -> dict[str, Any]:
        return {"unit": unit, "basis": basis, "days": 1, "confirmed": basis in SETTLED_UNIT_BASES,
                "unit_settled": basis in SETTLED_UNIT_BASES, "sentence": sentence}

    if unit_of(energy) == "per_period":
        # ``kcal_week``: the name says a total over a period, not a day's intake; how many days it
        # totals is recorded, never guessed.
        return out(energy_unit(energy) or "kcal", "assumed",
                   f"{col} names a total over a week, month or year; record how many days it "
                   f"totals before screening by it.")
    spanned = day_count(energy)
    if spanned is not None:
        # BLUEPRINT §14 rule 3: ``kcal_2d``, ``energy_kcal_7d``, ``kcal_4day_total`` carry a day
        # count, a total over the days as readily as a mean of them; every screen, band and count
        # is read in it, so it is asked (the gate: a 2-day total read as a day's intake lost 111
        # of 500 rows to a 5,000 kcal screen no true daily value exceeded).
        unit = "kj" if unit_of(energy) == "kj" else (energy_unit(energy) or "kcal")
        word = "kJ" if unit == "kj" else "kcal"
        reading = out(unit, "days", f"{col}'s name carries `{spanned}` days: a total over them or "
                                    f"a day's mean, in {word}? Record it before screening by it.")
        reading["days"] = spanned
        reading["days_in_name"] = spanned
        return reading
    if unit_of(energy) == "kj":
        return out("kj", "name", f"{col} says kJ in its name.")
    # The registry's value test for the unit (BLUEPRINT §14.3: ``readings.KIND_RULES
    # ["unit:energy"]``): kcal and kJ sit 4.184× apart against the macronutrients' energy.
    from turbotab.core.readings import by_values

    verdict = by_values("unit:energy", frame, energy) if energy in frame.columns else None
    if verdict is not None and verdict.settles:
        return out(str(verdict.value), "atwater", f"{col} {verdict.evidence}.")
    if verdict is not None and verdict.candidates:
        # BLUEPRINT §14.3 (amendment after the fifth gate): the ratio fits more than one reading
        # (4.00: kJ, or a 4-day kcal total beside daily-mean macronutrients); the unit and the
        # days are asked together, the name's unit first among them when it spells one.
        named = "kj" if unit_of(energy) == "kj" else ("kcal" if unit_of(energy) == "kcal"
                                                      or energy_unit(energy) == "kcal" else None)
        pairs = sorted(verdict.candidates, key=lambda p: (p[0] != named, p[1]))
        reading = out(str(pairs[0][0]), "atwater_ambiguous",
                      f"{col} {verdict.evidence}; record its unit and the days each value spans "
                      f"before screening by it.")
        reading["days"] = int(pairs[0][1])
        reading["candidates"] = [list(p) for p in pairs]
        return reading
    if unit_of(energy) == "kcal" or energy_unit(energy) == "kcal":
        return out("kcal", "name", f"{col} says kcal in its name.")
    if energy in frame.columns:
        by = energy_unit_by_magnitude(frame[energy])
        if by is not None:
            median = float(pd.to_numeric(frame[energy], errors="coerce").median())
            word = "kJ" if by == "kj" else "kcal"
            return out(by, "magnitude",
                       f"Only {col}'s median, {tick(f'{median:,.0f}')}, says {word} "
                       f"({ENERGY_PRIOR_SOURCE}); record its unit before screening by it.")
    median = (float(pd.to_numeric(frame[energy], errors="coerce").median())
              if energy in frame.columns else float("nan"))
    if math.isfinite(median) and median > 5_000:
        # No population eats a median of more than 5,000 kcal a day: the column is in kJ outside
        # the adult band (children), or a total over more than a day. Kcal is only the reading the
        # counts are made in, and the screens refuse what it would remove (audit IN-07).
        return out("kcal", "assumed",
                   f"Nothing says what unit {col} is in, and its median, {tick(f'{median:,.0f}')}, "
                   f"is no day's intake in kcal; record its unit before screening by it.")
    return out("kcal", "assumed", f"Nothing says what unit {col} is in; record its unit before "
                                  f"screening by it.")


def _energy_unit(frame: pd.DataFrame, energy: str) -> str:
    """``kj`` when the column is in kilojoules, else ``kcal`` (:func:`energy_unit_reading`)."""
    return energy_unit_reading(frame, energy)["unit"]


def unit_refusal(energy: str, reading: Mapping[str, Any] | None) -> str | None:
    """Why no screen may be chosen on ``energy`` yet: its unit is only proposed (a median-magnitude
    prior, or nothing at all), so the screen's bounds would be read in a unit nobody stated. None
    once the unit is settled (recorded, spelled out, or reconstructed from the macronutrients)."""
    from turbotab.core.voice import tick

    if reading is None or reading.get("confirmed", True):
        return None
    word = "kJ" if reading.get("unit") == "kj" else "kcal"
    if reading.get("basis") == "not_energy":
        return (f"Refused until {tick(energy)}'s unit is recorded as kcal or kJ: it is recorded in "
                f"{reading.get('recorded_unit')}, which is no unit of energy.")
    if reading.get("basis") == "atwater_ambiguous":
        fits = " or ".join(f"{ENERGY_WORDS[u]}" + (f" over {d} days" if int(d) > 1 else "")
                           for u, d in reading.get("candidates") or [])
        return (f"Refused until {tick(energy)}'s unit and days are recorded: against the energy "
                f"its macronutrients carry it reads as {fits}, which its values cannot tell "
                f"apart, and the screen's bounds are read in it.")
    if reading.get("days_unsettled") and reading.get("basis") in SETTLED_UNIT_BASES:
        evidence = (reading.get("days_reading") or {}).get("evidence") or "nothing settles it"
        return (f"Refused until {tick(energy)}'s days are recorded: {evidence}, so whether each "
                f"value is one day's intake or a total over several is not settled, and the "
                f"screen's bounds are read in a day's.")
    if reading.get("basis") == "name":
        return (f"Refused until {tick(energy)}'s unit and days are recorded: only its name says "
                f"{word}, and the screen's bounds are read in a day's {word}.")
    if reading.get("basis") == "days":
        return (f"Refused until {tick(energy)}'s days are recorded: its name carries "
                f"{tick(str(reading.get('days_in_name')))} days, a total or a mean, and the "
                f"screen's bounds are read in it.")
    if reading.get("basis") == "magnitude":
        return (f"Refused until {tick(energy)}'s unit is recorded: only its median says {word}, "
                f"and the screen's bounds are read in it.")
    return (f"Refused until {tick(energy)}'s unit is recorded: nothing says whether it is kcal or "
            f"kJ, a day's intake or a total over several days.")


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
                        sex_levels: Mapping[str, str], base: pd.Series, days: int = 1,
                        unit_reading: Mapping[str, Any] | None = None,
                        settled: Iterable[str] | None = None,
                        settled_sex: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """The pack's fixed kcal screens, each with the rows it would remove from ``base``, read in
    ``unit`` (a value totalling ``days`` days is screened at ``days`` times a day's bounds). While
    the unit is only proposed (``unit_reading`` not confirmed) every screen is refused with the
    reason (:func:`unit_refusal`): its count is shown, never applied."""
    factor = (KCAL_PER_KJ if unit == "kj" else 1.0) * max(int(days or 1), 1)
    per_day = "kcal a day"

    def bounds(low: float, high: float) -> tuple[float, float]:
        return round(low * factor, 1), round(high * factor, 1)

    def in_kj(low: float, high: float) -> str:
        if unit != "kj" and factor == 1.0:
            return ""
        lo, hi = bounds(low, high)
        word = "kJ" if unit == "kj" else "kcal"
        over = f" over {int(days)} days" if days and int(days) > 1 else ""
        return f" ({_kcal(lo)}–{_kcal(hi)} {word}{over})"

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
                + (" (compared in kJ)" if unit == "kj" else "")
                + (f" (over {int(days)} days)" if days and int(days) > 1 else ""),
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
    unsettled = role_refusal(energy, settled) or unit_refusal(energy, unit_reading)
    # BLUEPRINT §14.1: a sex-specific screen reads the sex column's levels by its role too.
    sex_waiting = (f"Refused until `{sex}`'s role is confirmed on its own: it was proposed below "
                   f"high confidence, and the screen reads its levels."
                   if sex is not None and settled_sex is not None and sex not in set(settled_sex)
                   else None)
    for key, label, rule in screens:
        affected = int((rule_excludes(frame, rule) & base).sum())
        by_sex = getattr(rule, "by", None) is not None
        out.append({"key": key, "rule": rule.model_dump(mode="json"), "label": label,
                    "affected": affected, "evidence": dict(EXCLUSION_EVIDENCE),
                    "refused": unsettled or (sex_waiting if by_sex else None)
                    or screen_refusal(energy, affected, present)})
    return out


def role_refusal(energy: str, settled: Iterable[str] | None) -> str | None:
    """Why no screen may be chosen on ``energy`` yet: its role rode along unconfirmed in a bulk
    confirm below high confidence (BLUEPRINT §14 rule 2). None once it is settled."""
    from turbotab.core.voice import tick

    if settled is None or energy in set(settled):
        return None
    return (f"Refused until {tick(energy)} is confirmed as total energy intake on its own: it was "
            f"proposed below high confidence.")


def body_refusal(body: Mapping[str, Any], sex: str | None, settled: Iterable[str] | None,
                 state: Any = None, frame: pd.DataFrame | None = None) -> str | None:
    """Why the Goldberg screen may not be chosen yet (BLUEPRINT §14.1, the readings ledger): a body
    measure's unit is known only from its median (kg and lb overlap there; the gate's US women's
    ``weight`` in pounds excluded 136 of 500 rows against 5 in kg), or its role, or the sex
    column's, rode along unconfirmed. None once each is settled."""
    from turbotab.core.readings import body_unit_reading, listing

    def values(c: str) -> Any:
        return frame[c] if frame is not None and c in frame.columns else None

    units = [c for c, m in ((body.get("age"), "age"), (body.get("weight"), "weight"),
                            (body.get("height"), "height")) if c
             and not body_unit_reading(c, m, state, values(c)).settled]
    roles = ([c for c in (body.get("age"), body.get("weight"), body.get("height"), sex)
              if c and c not in set(settled)] if settled is not None else [])
    if units:
        one = len(units) == 1
        return (f"Refused until {listing(units)}'s {'unit is' if one else 'units are'} recorded: "
                f"only {'its median says' if one else 'their medians say'} kg, cm or years, which "
                f"tells kg from lb no better than a heavy cohort from a light one.")
    if roles:
        return (f"Refused until {listing(roles)} {'is' if len(roles) == 1 else 'are'} confirmed "
                f"on {'its' if len(roles) == 1 else 'their'} own: proposed below high confidence.")
    return None


def sex_refusal(column: str) -> str:
    """Why a screen that reads a numeric sex column nobody's coding settled is refused."""
    from turbotab.core.voice import tick

    return (f"Refused until {tick(column)}'s coding is confirmed: which code is female is a "
            f"guess (NHANES codes 1 male, 2 female; other studies 1 female), and the counts "
            f"shown read the guess.")


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
                 roles: Mapping[str, str], units: Mapping[str, Any] | None = None,
                 state: Any = None) -> dict[str, Any]:
    """Age (years), weight (kg or lb) and height (cm or m) columns, when their names and values
    agree, each read in the unit the user recorded for it (the one accessor,
    ``readings.unit_record``; BLUEPRINT §14.3, every confirmation is honored):

    * an age recorded in another unit than years (months) is no age in years: not read;
    * a weight recorded in lb is read in lb (converted exactly by the screen), in kg as kg, and
      in any other unit (g, a length) not at all;
    * a height recorded in cm or m is read in it, and in any other unit (inches, which the BMR
      equation does not take) not at all: the weight-only equation is used;
    * an unrecorded measure is found by its median's band, as proposed (its own refusal asks)."""
    from turbotab.core.readings import _get, unit_record

    def recorded(column: str | None) -> Any:
        if column is None:
            return None
        spec = (unit_record(state, column) if state is not None
                else unit_record(None, column, units=units or {}))
        return _get(spec, "unit") if spec is not None else None

    def named(names: set[str]) -> list[str]:
        return [str(c) for c in info if str(c).lower() in names and c in frame.columns
                and roles.get(c) not in _NOT_A_MEASURE]

    def median(column: str) -> float:
        values = pd.to_numeric(frame[column], errors="coerce")
        return float(values.median()) if values.notna().sum() else float("nan")

    age = _named(info, frame, _AGE_NAMES, roles, 10, 100)
    if age is not None and recorded(age) not in (None, "years"):
        age = None
    weight, weight_unit = None, "kg"
    for c in named(_WEIGHT_NAMES):
        said = recorded(c)
        factor = {None: 1.0, "kg": 1.0, "lb": 0.45359237}.get(said)
        if factor is not None and 25 <= median(c) * factor <= 200:
            weight, weight_unit = c, ("lb" if said == "lb" else "kg")
            break
    height, height_unit = None, "cm"
    for c in named(_HEIGHT_NAMES):
        said = recorded(c)
        if said in ("cm", "m"):
            height, height_unit = c, str(said)
            break
        if said is None and 100 <= median(c) <= 230:
            height, height_unit = c, "cm"
            break
        if said is None and 1.0 <= median(c) <= 2.3:
            height, height_unit = c, "m"
            break
    return {"age": age, "weight": weight, "weight_unit": weight_unit, "height": height,
            "height_unit": height_unit}


def goldberg_proposal(frame: pd.DataFrame, info: Mapping[str, Mapping[str, Any]], *, energy: str,
                      unit: str, sex: str | None, sex_levels: Mapping[str, str],
                      roles: Mapping[str, str], target: str | None,
                      base: pd.Series, days: float = 1.0,
                      days_note: str | None = None,
                      unit_reading: Mapping[str, Any] | None = None,
                      units: Mapping[str, Any] | None = None,
                      settled: Iterable[str] | None = None,
                      state: Any = None) -> dict[str, Any] | None:
    """The Goldberg screen with Schofield's BMR, offered with its count when the columns are read.

    PAL 1.55 and the days of intake each row's energy averages (``recall_days``: one, unless the
    working table averaged each person's recalls) are stated in the label; a rule that would read
    the outcome (weight as the outcome) is not offered (audit RO-01).
    """
    from turbotab.core.decisions import GoldbergRule

    body = body_columns(info, frame, roles, units, state)
    if sex is None or body["age"] is None or body["weight"] is None:
        return None
    if unit_reading is not None and int(unit_reading.get("days") or 1) > 1:
        return None  # Goldberg reads a day's mean intake; a total over days is not one
    height = body["height"]
    # The weight's unit as the ledger holds it (BLUEPRINT §14.3): a recorded pound is converted to
    # kilograms exactly, so the count shown is the screen's own, never one read in the wrong unit.
    weight_unit = str(body.get("weight_unit") or "kg")
    rule = GoldbergRule(
        column=energy, energy_unit="kj" if unit == "kj" else "kcal", days=days, sex=sex,
        female=[k for k, v in sex_levels.items() if v == "female"],
        male=[k for k, v in sex_levels.items() if v == "male"],
        age=body["age"], weight=body["weight"], weight_unit=weight_unit, height=height,
        height_unit=body["height_unit"],
        equation="schofield_height" if height else "schofield", pal=GOLDBERG_PAL,
        reason="implausible energy reports (Goldberg cut-offs, Black 2000)")
    if target is not None and target in rule.reads():
        return None
    affected = int((rule_excludes(frame, rule) & base).sum())
    eq = "weight and height" if height else "weight"
    if weight_unit == "lb":
        eq = eq.replace("weight", "weight converted from lb", 1)
    n_days = f"{days:g} {'day' if days == 1 else 'days'}"
    label = (f"Goldberg: energy over Schofield BMR ({eq}) outside the 95% cut-offs for PAL "
             f"{GOLDBERG_PAL} and {n_days} of intake" + (f" ({days_note})" if days_note else ""))
    present = int((pd.to_numeric(frame[energy], errors="coerce").notna() & base).sum())
    return {"key": "goldberg_schofield", "rule": rule.model_dump(mode="json"), "label": label,
            "affected": affected, "evidence": dict(GOLDBERG_EVIDENCE),
            "refused": role_refusal(energy, settled) or unit_refusal(energy, unit_reading)
            or body_refusal(body, sex, settled if state is not None and getattr(state, "roles", None)
                            else None, state, frame)
            or screen_refusal(energy, affected, present)}


def _marked(text: str, columns: Iterable[str]) -> str:
    """Wrap this table's column names in backticks, longest first, whole words only."""
    for c in sorted(set(columns), key=len, reverse=True):
        text = re.sub(rf"(?<![`\w]){re.escape(c)}(?![`\w])", f"`{c}`", text)
    return text


def partition_check(frame: pd.DataFrame, energy: str, nutrients: Sequence[str],
                    state: Any = None) -> dict[str, Any] | None:
    """Why a partition of ``nutrients`` cannot run on ``frame``'s rows, or None (energy.py). The
    parts of totals as the ledger holds them (``readings.nesting``: each confirmation stands over
    the names' and values' guess)."""
    from turbotab.core.methods.energy import partition_refusal
    from turbotab.core.readings import nesting

    present = [n for n in nutrients if n in frame.columns]
    nested = nesting(state, frame=frame, columns=present)
    return partition_refusal(frame, energy, list(nutrients), nested=nested)


def energy_reading(frame: pd.DataFrame, columns: Mapping[str, Mapping[str, Any]],
                   roles: Mapping[str, str], *, energy: str | None, nutrients: list[str],
                   sex: str | None, target: str | None,
                   purpose: str | None = None,
                   settled: Iterable[str] | None = None,
                   energy_unit: str | None = None,
                   waiting: Iterable[str] = (), state: Any = None) -> dict[str, Any]:
    from turbotab.core.methods.energy import applicable_methods, rank_methods
    from turbotab.core.readings import nesting
    from turbotab.core.voice import finish

    verdicts = applicable_methods(list(columns), energy, nutrients)
    if verdicts["partition"]["ok"] and energy in frame.columns:
        # The checks the fit makes on the data (the Atwater reconstruction), and the two it
        # cannot: a total beside its parts, and nutrients that out-weigh total energy.
        refused = partition_check(frame, energy, nutrients, state)
        if refused is not None:
            present = [n for n in nutrients if n in frame.columns]
            nested = nested_reason(nesting(state, frame=frame, columns=present), nutrients)
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
    unsettled_energy = role_refusal(energy, settled) if energy else None
    if unsettled_energy:
        # BLUEPRINT §14 rule 2: the energy column rode along unconfirmed; every method waits.
        applicability = {m: {"ok": False, "reason": option_reason(unsettled_energy)}
                         for m in applicability}
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
    # Audit WP13 gate repair: the outcome is read by its name, else by its values against the
    # table's body-size columns; when nothing places it, the dispute is stated as a condition the
    # researcher answers (the leash), never dropped because a name list missed the outcome.
    from turbotab.core.methods.dietary_caveats import (
        DISPUTED, dispute_line, outcome_relation, unconfirmed_line, values_line,
    )

    notes: list[str] = []
    dispute = None
    relation = (outcome_relation(target, frame, energy=energy, energy_unit=energy_unit)
                if target is not None else None)
    if relation is not None and relation["kind"] is not None:
        kind = relation["kind"]
        line = finish(values_line(target, kind, relation["via"], relation["r"])
                      if relation["basis"] == "values" else dispute_line(target, kind))
        notes.append(line)
        dispute = {"outcome": target, "kind": kind, "note": line, "evidence": dict(DISPUTED),
                   "basis": relation["basis"]}
    elif relation is not None and relation["basis"] == "unconfirmed":
        line = finish(unconfirmed_line(target))
        notes.append(line)
        dispute = {"outcome": target, "kind": None, "note": line, "evidence": dict(DISPUTED),
                   "basis": "unconfirmed"}
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
        "not_adjusted": not_adjusted(columns, roles, energy=energy, target=target,
                                     nutrients=nutrients),
        # BLUEPRINT §14 rule 2: the energy column and nutrients this card would read that wait for
        # their own confirmation (recorded below high confidence in a bulk confirm).
        "unconfirmed": sorted(set(waiting)),
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
                    days: tuple[float, str | None] = (1.0, None),
                    units: Mapping[str, Any] | None = None,
                    settled: Iterable[str] | None = None,
                    state: Any = None) -> dict[str, Any]:
    """The proposals artifact from a frame holding (at least) the columns it reads.

    ``columns`` are the ingest's column records (``name``, ``dtype``, ``n_unique``,
    ``n_missing``); ``roles`` the confirmed roles, else the proposed ones, else empty;
    ``purpose`` the declared purpose, which orders the energy methods by soundness; ``days`` the
    recall days each row's energy averages, with a note when they differ (``recall_days``);
    ``units`` the columns' recorded units (``set_column_unit``); ``settled`` the columns whose
    roles a number-changing default may read (:func:`turbotab.core.readings.settled_columns`;
    BLUEPRINT §14), every role as given when None.
    """
    from turbotab.core.readings import unit_record
    from turbotab.core.voice import tick

    info = {str(c["name"]): c for c in columns}
    roles = dict(roles or {})
    missing = missing_reading(frame, columns, roles, target)
    # WP7: the missing-data methods, soundest first for the declared purpose, each with its
    # "customary" and "sound" labels and its rung (turbotab.core.methods.missing).
    from turbotab.core.methods.missing import (below_detection_options,
                                               imputation_model_options, methods_for)

    missing["methods"] = methods_for(purpose)
    missing["below_detection"] = below_detection_options(purpose)
    # MS1–MS2: multiple imputation's sub-answers, each labeled (compatible; passive; single-level).
    missing["imputation_models"] = imputation_model_options(purpose)
    n_base = int(frame[target].notna().sum()) if target and target in frame.columns else len(frame)
    if "dietary" not in (lens or []):
        from turbotab.core import custom_sound

        return {"exclusions": [], "energy": None, "missing": missing, "n_base": n_base,
                "labels": {"missing": custom_sound.missing(purpose).model_dump(mode="json"),
                           "exclusions": None, "energy_adjustment": None},
                "coach": _card_lines(frame, target=target, energy=None, unit="kcal", missing=missing),
                "basis": "Only the missing-values reading is proposed: the dietary lens is not chosen.",
                "energy_unit": None}
    settled = set(settled) if settled is not None else None
    energy_read = energy_column_reading(_with_medians(info, frame), roles, frame)
    energy = energy_read["column"] if energy_read is not None else None
    settled_unit = None
    if energy is not None and energy in frame.columns:
        recorded = unit_record(state, energy, units=units)
        first = energy_unit_reading(frame, energy, recorded)
        # A share of a day's energy reads the unit alone: settled by the record or the Atwater
        # identity, whatever the days (BLUEPRINT §14.3).
        settled_unit = first["unit"] if first.get("unit_settled") else None
    nutrients = nutrient_candidates(info, roles, energy=energy, target=target, frame=frame,
                                    settled=settled, energy_unit=settled_unit)
    # With no record of confirmations (a caller holding roles alone), the energy column sets the
    # screens and the card only as its values' own reading: following the energy its
    # macronutrients carry (BLUEPRINT §14 rule 2).
    energy_settled = settled
    if settled is None and roles and energy is not None:
        from turbotab.core.readings import by_values

        verdict = by_values("role:energy", frame, energy) if energy in frame.columns else None
        energy_settled = {energy} if verdict is not None and verdict.settles else set()
    sex, sex_levels = sex_column(info, frame, roles, state=state)
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
        recorded = unit_record(state, energy, units=units)
        unit_reading = energy_unit_reading(frame, energy, recorded)
        unit = unit_reading["unit"]
        if energy != target:  # an eligibility rule never reads the outcome (audit RO-01)
            # A screen reads a column whatever its model role: sex left out of the model still
            # serves Willett's sex-specific cut-offs, as it serves the Goldberg screen (the methods
            # gate, item E: the menu was too tight, BLUEPRINT §11.3).
            screen_sex, screen_levels = sex_column(info, frame, roles, screen=True, state=state)
            guessed_sex = None
            if screen_sex is None:
                # A numeric sex column nobody's coding settled: the screens that read it are
                # offered under the best guess and refused until it is confirmed (BLUEPRINT §14.2).
                guessed_sex, guessed_levels = sex_column(info, frame, roles, screen=True,
                                                         state=state, guess=True)
                if guessed_sex is not None:
                    screen_sex, screen_levels = guessed_sex, guessed_levels
            by_sex = screen_sex if screen_sex != target else None  # never a rule on the outcome
            exclusions = exclusion_proposals(frame, energy=energy, unit=unit, sex=by_sex,
                                             sex_levels=screen_levels, base=base,
                                             days=int(unit_reading.get("days") or 1),
                                             unit_reading=unit_reading, settled=energy_settled,
                                             settled_sex=(settled if state is not None
                                                          and getattr(state, "roles", None)
                                                          else None))
            goldberg = goldberg_proposal(frame, info, energy=energy, unit=unit, sex=screen_sex,
                                         sex_levels=screen_levels, roles=roles, target=target,
                                         base=base, days=days[0], days_note=days[1],
                                         unit_reading=unit_reading, units=units,
                                         settled=energy_settled, state=state)
            if goldberg is not None:
                exclusions.append(goldberg)
            if guessed_sex is not None:
                why = sex_refusal(guessed_sex)
                for offer in exclusions:
                    reads = offer["rule"].get("sex") or (offer["rule"].get("by") or {}).get("column")
                    if reads == guessed_sex and not offer.get("refused"):
                        offer["refused"] = why
    reading = None
    if energy is not None or nutrients:
        # What the card would read that waits for its own confirmation (BLUEPRINT §14 rule 2):
        # the energy column, and every energy-bearing exposure the card does not pre-fill.
        waiting = [c for c, r in roles.items() if r == "exposure" and energy_bearing(c)
                   and c not in nutrients and c != target]
        if energy is not None and energy_settled is not None and energy not in energy_settled:
            waiting.append(energy)
        reading = energy_reading(frame, info, roles, energy=energy, nutrients=nutrients, sex=sex,
                                 target=target, purpose=purpose, settled=energy_settled,
                                 energy_unit=settled_unit, waiting=waiting, state=state)
    # WP17 (north star 5; audit RO-06, RO-07): every option labeled customary and sound for the
    # declared purpose, soundest first, with the tension in one line.
    from turbotab.core import custom_sound

    applicable = ([m for m, v in (reading or {}).get("applicability", {}).items() if v.get("ok")]
                  if reading else None)
    labels = {
        "missing": custom_sound.missing(purpose).model_dump(mode="json"),
        "exclusions": custom_sound.exclusions(purpose, [e["key"] for e in exclusions])
        .model_dump(mode="json"),
        "energy_adjustment": (custom_sound.energy(purpose, applicable).model_dump(mode="json")
                              if reading else None),
    }
    return {"exclusions": exclusions, "energy": reading, "missing": missing, "n_base": n_base,
            "labels": labels,
            "coach": _card_lines(frame, target=target, energy=energy, unit=unit, missing=missing,
                                 unit_reading=unit_reading,
                                 settled=(energy is None or energy_settled is None
                                          or energy in energy_settled)),
            "basis": basis, "energy_unit": unit_reading}


def _card_lines(frame: pd.DataFrame, *, target: str | None, energy: str | None, unit: str,
                missing: Mapping[str, Any],
                unit_reading: Mapping[str, Any] | None = None,
                settled: bool = True) -> dict[str, Any]:
    """The decision cards' coach lines, at most one per question (turbotab.core.coach)."""
    from turbotab.core.coach import card_lines

    lines = card_lines(frame, target=target, energy=energy, unit=unit, missing=missing,
                       unit_reading=unit_reading, settled=settled)
    return {key: line.model_dump(mode="json") for key, line in lines.items()}


def needed_columns(columns: Sequence[Mapping[str, Any]], *, target: str | None,
                   roles: Mapping[str, str]) -> list[str]:
    """The few columns the proposals read: energy (every column named as total energy, so the
    values can pass over one that is no day's intake), sex, the outcome and the nutrients."""
    from turbotab.core.recognizers import AmbiguousNutrient, read_nutrient

    info = {str(c["name"]): c for c in columns}
    energy = energy_column(info, roles)
    energies = [c for c, i in info.items() if _dtype(i) in _NUMERIC and is_energy_name(c)]
    nutrients = nutrient_candidates(info, roles, energy=energy, target=target)

    def total(c: str) -> bool:
        try:
            reading = read_nutrient(c)
        except AmbiguousNutrient:
            return False
        return reading is not None and reading.macro in ("protein", "carbohydrate", "fat",
                                                         "alcohol") and reading.part is None

    # The macronutrient totals the energy column is read against (energy_column_reading), and the
    # body-size columns an unrecognized outcome is read against (``outcome_relation``).
    macros = [c for c, i in info.items() if _dtype(i) in _NUMERIC and c != target and total(c)]
    from turbotab.core.methods.dietary_caveats import energy_related

    bodies = [c for c, i in info.items() if _dtype(i) in _NUMERIC and c != target
              and energy_related(c) is not None][:20]
    macros = [*macros, *bodies]
    sexes = [c for c in info if c.lower() in _SEX_NAMES or {"sex", "gender"} & set(_tokens(c))]
    body = [c for c in info if str(c).lower() in _AGE_NAMES | _WEIGHT_NAMES | _HEIGHT_NAMES]
    wanted = [energy, *energies, target, *sexes, *nutrients, *macros, *body]
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
        # P1-FU: the substitution option says the swap Table 2 will, on the values' reading of
        # the parts inside their totals (``estimand.values_nesting``, the design's own test).
        from turbotab.core.estimand import values_nesting

        nested = values_nesting(state, store=store) if state.purpose == "inference" else None
    ctx.progress(0.6, "Counting what each exclusion rule would remove")
    from turbotab.core.readings import settled_columns

    out = build_proposals(frame, columns, lens=state.lens, target=target, roles=roles,
                          purpose=state.purpose,
                          days=recall_days(ctx.inputs.get("working"), state),
                          units=getattr(state, "column_units", None) or {},
                          settled=settled_columns(state, ctx.inputs.get("roles")), state=state)
    out["survey"] = survey
    # WP12a (repair round): the exposure-form question's options, ordered by soundness for the
    # declared purpose, each with its "customary in" and "sound for" labels (north star 5).
    from turbotab.core.methods.exposure_form import options as form_options

    out["exposure_forms"] = form_options(state.purpose)
    # WP17 (MODELING_SEQUENCE §1 steps 2–3): under inference, what the exposure and effect question
    # offers (only the measures the engine fits), and the adjustment card: covariates grouped by the
    # pack's guess, one tap per group.
    from turbotab.core.estimand import adjustment_card, estimand_card

    task = state.task
    if task is None and target and target in frame.columns and state.purpose == "inference":
        # The task as target_info reads it (``ml.triage``'s detection, settled by the readings'
        # value test where it settles), for the measures the card offers.
        from turbotab import engine
        from turbotab.core.readings import task_reading

        values = frame[target].reset_index(drop=True)
        detection = engine.detect_task_type(values.to_frame(), target)
        if detection.get("detected") == "classification":
            task = "binary" if int(values.nunique(dropna=True)) <= 2 else "multiclass"
        else:
            task = "regression"
        found = task_reading(target, task, detection.get("confidence"), values=values)
        if found.settled:
            task = str(found.value)
    # ESTIMAND (ruling 9): a yes/no outcome's event share ranks the marginal measures.
    from turbotab.core.estimand import event_share

    share = (event_share(frame[target], state.event)
             if task == "binary" and target and target in frame.columns else None)
    out["estimand"] = estimand_card(state, task, prevalence=share, nested=nested)
    out["adjustment"] = adjustment_card(state)
    # LEASH (the routing gate's leash note): the grouping question's card, every column it asks
    # about with the guess it shows and that guess's evidence (``estimand.grouping_card``).
    from turbotab.core.estimand import grouping_card

    out["grouping"] = grouping_card(state, ctx.inputs.get("roles"))
    # ESTIMAND (MODELING_SEQUENCE §1 row 11): Model 1 is declared beside the adjustment answers,
    # before any estimate is displayed (the effects stage reports the sequence).
    from turbotab.core.estimand import adjustment_answer, model_sequence_card

    out["model_sequence"] = (model_sequence_card(state)
                             if adjustment_answer(state) is not None else None)
    return out


def _survey_proposal(state: Any, store: Any) -> dict[str, Any] | None:
    """The survey question's options on this table (audit §5 WP10), read with the roles."""
    from turbotab.core.methods.survey import proposal

    return proposal(state, store) if state.roles is not None else None


def recall_days(working: Any, state: Any = None) -> tuple[float, str | None]:
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
    if state is not None:
        # BLUEPRINT §14.1: a combined row averages recall days only when the user answered that a
        # unit's rows are repeats (replicate recalls); averaged time points are no recall days, and
        # one day is the conservative reading (the Goldberg limits widen as days fall).
        from turbotab.core.readings import repeat_kind_reading

        found = repeat_kind_reading(state, None)
        if found is None or not found.settled or found.value != "repeats":
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
    "needed_columns", "screen_refusal", "unit_refusal", "energy_column_reading",
    "energy_unit_reading", "recorded_energy_unit", "SETTLED_UNIT_BASES",
    "nested_reason",
    "nutrient_candidates", "partition_check", "proposals_stage", "recall_days",
    "roles_from", "rule_excludes", "sex_column", "strata_candidates",
]

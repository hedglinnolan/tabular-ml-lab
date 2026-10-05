"""M1 row-side stages: column roles, the cohort (participant flow), the split.

Owner: the M1 "rows" agent. Contract: docs/turbotab-next/M1_CONTRACT.md §3.

* ``roles`` proposes a role for every column but the outcome, with a plain reason. It reads
  names and the profile's column summaries through the one set of recognizers every stage shares
  (:mod:`turbotab.core.recognizers`, audit WP13: whole words, codebooks, corroboration): energy
  and nutrients, identifiers and clusters, acquisition columns, time and survey weights.
  A proposal is a suggestion the user confirms with ``set_roles``; it is never an answer.
* ``cohort`` is the participant flow: ``loaded`` → ``outcome_measured`` → one step per exclusion
  rule → ``complete_cases`` (only under ``missing == "complete_case"``). It counts every row.
* ``split`` draws the held-out rows and the cross-validation folds on the cohort: grouped by the
  identifier when it repeats, stratified for classification, seeded and deterministic.

The pure functions (:func:`cohort_flow`, :func:`draw_split`, :func:`propose_roles`) are what the
stages and the consequence previews share, so a preview and the stage it foretells cannot disagree.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.stages.data import LENSES, open_store

PREDICTOR_ROLES = ("exposure", "covariate", "energy")
REASON_WORDS = 16
OMICS = ("metabolomics", "genomics")
CHUNK_COLUMNS = 1_000  # columns materialized at once when counting missing values


# ── shared helpers ────────────────────────────────────────────────────────────


def predictors(roles: Mapping[str, str] | None, order: Sequence[str] | None = None,
               drop: Iterable[str] = ()) -> list[str]:
    """Columns whose role puts them in the model (exposure, covariate or energy), in table order.

    ``drop``: columns the missing-values answer left out (``SetMissing.drop_columns``).
    """
    if not roles:
        return []
    gone = set(drop)
    names = [c for c, r in roles.items() if r in PREDICTOR_ROLES and c not in gone]
    if order is None:
        return names
    rank = {c: i for i, c in enumerate(order)}
    return sorted(names, key=lambda c: rank.get(c, len(rank)))


def energy_bearing(column: str) -> bool:
    """True when ``column`` is an amount of a macronutrient that carries energy.

    Protein, carbohydrate, fat, alcohol or fiber, in grams or unmarked units (Atwater
    factors apply), or already in kcal. A share of energy (``protein_pct_kcal``) is not an
    amount that carries energy, and neither is a total-energy column.
    """
    from turbotab.core.methods.energy import energy_factor, nutrient_role

    try:
        role = nutrient_role(column)
    except ValueError:  # names two macronutrients: ambiguous, not a clean amount
        return False
    if role is None:
        return False
    return energy_factor(column).factor is not None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _tokens(name: str) -> list[str]:
    return [t for t in _norm(name).split("_") if t]


def _words(text: str) -> int:
    return len(text.split())


def _fmt_num(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.4g}"


# ── roles ─────────────────────────────────────────────────────────────────────

_COVARIATE_TOKENS = {
    "age", "sex", "gender", "bmi", "race", "ethnicity", "ethnic", "education", "educ", "income",
    "pir", "poverty", "smoking", "smoker", "smoke", "marital", "height", "weight", "waist", "hip",
    "activity", "exercise", "met", "mets", "meds", "medication", "medications", "bp", "sbp", "dbp",
    "systolic", "diastolic", "diabetes", "hypertension", "region", "country", "occupation",
    "insurance", "parity", "menopause", "pregnant", "pregnancy",
}
# Fasting status is a known confounder (METABOLOMICS_PACK §01: "age, sex, BMI, fasting status,
# medication, site…"): adjusted for, never an acquisition column (audit IN-02).
_FASTING_TOKENS = {"fasting", "fasted", "fast"}
_FLAG_PREFIX = ("imputed_", "is_imputed_", "flag_", "imp_")
_FLAG_SUFFIX = ("_imputed", "_flag", "_flagged", "_imp")
_UNIT_LABELS = {
    "grams": "g", "milligrams": "mg", "micrograms": "µg", "IU": "IU", "kcal": "kcal",
    "kj": "kJ", "density": "share of energy",
}
_INTAKE_UNITS = ("grams", "milligrams", "micrograms", "IU", "density")
_SURVEY_STRATA = {"strata", "stratum", "psu", "cluster", "sampling_weight"}


def _id_like(name: str) -> bool:
    """The name reads as an identifier of any kind: the one recognizer every stage shares
    (:func:`turbotab.core.recognizers.is_identifier`; audit IN-06)."""
    from turbotab.core.recognizers import is_identifier

    return is_identifier(name)


def _flag_base(name: str, by_lower: Mapping[str, str]) -> tuple[bool, str | None]:
    low = str(name).lower()
    for prefix in _FLAG_PREFIX:
        if low.startswith(prefix) and len(low) > len(prefix):
            return True, by_lower.get(low[len(prefix):])
    for suffix in _FLAG_SUFFIX:
        if low.endswith(suffix) and len(low) > len(suffix):
            return True, by_lower.get(low[: -len(suffix)])
    return False, None


def _unit(name: str) -> str | None:
    from turbotab.core.methods.energy import unit_of

    unit = unit_of(name)
    if unit == "unmarked":
        return "kcal" if _norm(name) in ("kcal", "calories", "kilocalories") else None
    return _UNIT_LABELS.get(unit, unit)


def _is_nutrient(name: str) -> bool:
    """The name declares a nutrient intake (:func:`turbotab.core.recognizers.is_nutrient`)."""
    from turbotab.core.recognizers import is_nutrient

    return is_nutrient(name)


# A day's food and drink together weigh a few kilograms; a column whose median is beyond 10 kg is
# no day's intake in grams of anything (a DXA fat mass in grams, 25,000; TurboTab's own bound,
# stated as such).
MAX_DAY_GRAMS = 10_000.0


_RATIO_WORDS = {"per_body": "per kg of body weight", "density": "per unit of energy",
                "per_period": "per week, month or year", "concentration": "per volume"}


def _ratio_unit(name: str) -> str | None:
    """How the name's unit expression is a ratio, said in words (``protein_g_kg`` → "per kg of body
    weight"), or None for an amount (:func:`turbotab.core.recognizers.amount_unit`)."""
    from turbotab.core.methods.energy import unit_of

    return _RATIO_WORDS.get(unit_of(name))


def _doubt(check: Any) -> str:
    """The clause a name-only nutrient proposal carries: what did, and did not, corroborate it."""
    why = str(getattr(check, "why", "") or "")
    r = getattr(check, "r", None)
    if getattr(check, "duplicate_of", None):
        return why  # another column reads as the same nutrient (BLUEPRINT §14)
    if "no intake's name does" in why:
        return why
    if "too few values" in why:
        return "too few values to check"
    if r is not None and "does not rise" in why:
        return f"it does not track total energy (r = {r:.2f})"
    if "name and unit" in why:
        return "only its name and unit say so"
    if "codebook name" in why:
        return "only its codebook name says so"
    return "only its name says so"


def _intake_by_unit(name: str, median: Any = None) -> bool:
    """An amount in an intake unit (``_g``, ``_mg``, a share of energy) that names no specimen,
    score, body scan or concentration, and whose median (when known) is a day's amount: a food or
    nutrient intake under the dietary lens (``fatty_fish_g``)."""
    from turbotab.core.methods.energy import unit_of
    from turbotab.core.recognizers import concentration_unit, tokens

    unit = unit_of(name)
    if unit not in _INTAKE_UNITS or concentration_unit(name):
        return False
    words = set(tokens(name))
    if words & {"serum", "plasma", "blood", "urine", "urinary", "score", "index", "mass", "body",
                "dose", "meds", "medication", "dxa", "dexa", "scan", "tissue", "sample",
                "specimen", "biopsy"}:
        return False
    try:
        m = float(median) if median is not None else None
    except (TypeError, ValueError):
        m = None
    scale = {"grams": 1.0, "milligrams": 1e-3, "micrograms": 1e-6}.get(unit)
    return not (m is not None and scale is not None and m * scale > MAX_DAY_GRAMS)


def _design_role(name: str, summary: Mapping[str, Any], exact: set[str],
                 design_in_table: bool = False) -> str | None:
    """A reason when this column is part of a survey design, else None.

    A sampling weight is read by :func:`turbotab.core.recognizers.reads_as_survey_weight`: an
    NHANES weight by name, unambiguous survey vocabulary, or a name that is a weight only beside a
    survey design (``sample_wt``, a bare ``weight`` far above any body weight, ``_LLCPWT``, a word
    ending in ``wt``): the design columns in the table are the corroboration, so a tissue mass on a
    metabolomics sample sheet is no sampling weight (audit WP13 gate repair); never a birth or body
    weight (audit IN-10)."""
    from turbotab.core.recognizers import reads_as_survey_weight

    upper = str(name).upper()
    if upper in exact or upper.startswith(("SDMV",)):
        return "An NHANES survey design variable: a weight, stratum or sampling unit."
    median = summary.get("median")
    low = summary.get("min")
    # A sampling weight is never negative; zero is "not in this subsample" (the CDC GLU_J codebook,
    # WTSAF2YR: "0 | No Lab Result or Not Fasting for 8 to <24 hours | 325"; BLUEPRINT §14), so a
    # subsample's real weight is no covariate for its zeros.
    positive = low is None or _non_negative(low)
    if reads_as_survey_weight(name, median=None if median is None else float(median),
                              design_in_table=design_in_table) and positive:
        if upper.startswith(("WTDR", "WTMEC", "WTINT", "WTSA", "WTSB")):
            return "An NHANES survey design variable: a weight, stratum or sampling unit."
        if design_in_table:
            return ("A sampling weight beside the survey's design columns: how each row was "
                    "sampled, not a body measurement.")
        return "A sampling weight: how each row was sampled, not a body measurement."
    tokens = set(_tokens(name))
    if tokens & _SURVEY_STRATA and not tokens & {"id", "ids"}:
        return "Names a sampling stratum or cluster of the survey design."
    return None


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return True


def _non_negative(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return True


def _design_named(name: str, exact: set[str]) -> bool:
    """An NHANES design variable or weight by its published name or the tutorial's multi-cycle
    grammar (``SDMVSTRA``, ``WTMEC2YR``, ``WTSAF6YR``, ``MEC6YR``)."""
    from turbotab.core.recognizers import MULTI_CYCLE_WEIGHT, NHANES_WEIGHT_PREFIX, weight_tier

    upper = str(name).upper()
    return (upper in exact or upper.startswith("SDMV") or weight_tier(upper) is not None
            or bool(MULTI_CYCLE_WEIGHT.fullmatch(upper))
            or (bool(re.fullmatch(r"[A-Z0-9]+", upper)) and upper.startswith(NHANES_WEIGHT_PREFIX)))


def characteristic_fits(name: str, values: Any) -> bool:
    """A sex, age or BMI column's values fit what its name says (BLUEPRINT §14: high only where the
    values corroborate): a sex has two or three levels; an age is a number from 0 to 120 at its
    median, never negative; a BMI's median lies within 10–80 kg/m²."""
    import pandas as pd

    s = pd.Series(values).dropna()
    if s.empty:
        return False
    words = set(_tokens(name))
    if words & {"sex", "gender"}:
        return 2 <= int(s.nunique()) <= 3
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 0.95 * len(s) or x.empty:
        return False
    if "age" in words:
        return float(x.min()) >= 0 and float(x.median()) <= 120
    if "bmi" in words:
        return 10 <= float(x.median()) <= 80
    return False


def design_values_fit(name: str, values: Any) -> bool:
    """A design column's values fit its kind (BLUEPRINT §14: high only where the values
    corroborate): a weight is numeric, never negative, positive somewhere and takes more than two
    values; a stratum or PSU holds whole-number codes."""
    import pandas as pd

    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if x.empty:
        return False
    tokens = set(_tokens(name))
    upper = str(name).upper()
    if tokens & {"psu", "strata", "stratum"} or upper in ("SDMVSTRA", "SDMVPSU", "SDDSRVYR"):
        return bool((x == x.round()).all())
    return bool(float(x.min()) >= 0 and float(x.max()) > 0 and x.nunique() > 2)


def _design_in_table(columns: Sequence[Mapping[str, Any]], exact: set[str]) -> bool:
    """The table names its survey design: an NHANES design variable, or a stratum or primary
    sampling unit by its name (``SDMVSTRA``, ``_PSU``, ``strata``)."""
    for c in columns:
        name = str(c["name"])
        tokens = set(_tokens(name))
        if name.upper() in exact or name.upper().startswith("SDMV"):
            return True
        if tokens & {"psu", "strata", "stratum"} and not tokens & {"id", "ids"}:
            return True
    return False


ACQUISITION_REASON = {
    "inference": "Acquisition/batch: adjusted for under inference, so batch differences are not "
                 "read as biology.",
    "prediction": "Acquisition/batch: left out for prediction, since new samples come from new "
                  "batches.",
    None: "Acquisition/batch: a covariate, so batch differences are not read as biology.",
}


def acquisition_proposal(purpose: str | None) -> tuple[str, str]:
    """``(role, reason)`` for a batch, plate or run-order column (audit IN-02): a covariate under
    inference (the pack's default is to model the batch: METABOLOMICS_PACK §442; Nygaard et al.
    2016, batch blocked in the model rather than removed beforehand), left out under prediction,
    where new samples come from batches the model never saw."""
    key = purpose if purpose in ("inference", "prediction") else None
    return ("excluded" if key == "prediction" else "covariate"), ACQUISITION_REASON[key]


def _acquisition_values_fit(name: str, n_unique: int, n_present: int) -> bool:
    """A run order or a well position takes many values; one that holds two is a yes/no
    (``sleep_well`` 0/1), whatever its name (audit WP13 repair)."""
    from turbotab.core.recognizers import acquisition_kind

    if acquisition_kind(name) in ("run_order", "well") and n_present:
        return n_unique > 2
    return True


def acquisition_corroborated(columns: Sequence[Mapping[str, Any]], lens: Sequence[str] | None,
                             acquisition: Iterable[str]) -> str | None:
    """Why the table's acquisition-named columns read as how samples were acquired, or None.

    An acquisition word is one signal; the other is that the table is an assay: an assay lens
    (metabolomics, genomics), or a second acquisition column of another kind (a batch beside a run
    order or a plate). In a plate-size feeding study ``plate`` (small/large) is the intervention,
    and leaving it out "since new samples come from new batches" would drop the exposure (audit
    WP13 gate repair). Without that second signal the column is kept, as an ordinary predictor,
    and the proposal says so."""
    from turbotab.core.recognizers import acquisition_kind

    if any(k in OMICS for k in (lens or [])):
        return "the table is read under an assay lens"
    kinds = {acquisition_kind(c) for c in acquisition} - {None}
    if len(kinds) >= 2:
        return "the table records several acquisition columns"
    return None


def propose_roles(
    columns: Sequence[Mapping[str, Any]],
    *,
    lens: Sequence[str] | None,
    target: str | None,
    n_rows: int,
    energy_column: str | None = None,
    acquisition: Iterable[str] = (),
    purpose: str | None = None,
    fractional: Iterable[str] = (),
    intake: Mapping[str, Any] | None = None,
    energy_info: Mapping[str, Any] | None = None,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
    repeating: str | None = None,
    named_unit: str | None = None,
) -> list[dict[str, Any]]:
    """One proposal per column (the outcome and the row identity excepted), in table order.

    ``columns`` are the profile's column summaries (``name``, ``dtype``, ``n``, ``n_missing``,
    ``n_unique``, ``median``…). ``energy_column`` is the total-energy column the recognizer reads;
    ``acquisition`` the batch, plate and run-order columns (:func:`_acquisition_columns`);
    ``purpose`` the declared purpose, which sets an acquisition column's default; ``fractional``
    the identifier-named columns whose values have fractional parts (measurements, never IDs);
    ``intake`` the values' verdict on each column the name reads as a nutrient
    (:func:`turbotab.core.recognizers.corroborated_nutrients`); ``energy_info`` how the
    total-energy column was read (:func:`turbotab.core.stages.proposals.energy_column_reading`);
    ``facts`` what the values say about each column named like an identifier (``id``), a flag
    (``flag``) or a time (``time``: the share of ``repeating``'s units it varies within), and
    whether a design column's values fit (``design_values``); ``repeating`` the identifier whose
    values name repeating units.

    Every name is read by :mod:`turbotab.core.recognizers` (audit WP13): whole words, never
    substrings; a study's arms and groups are exposures; a site or household is a cluster.

    **Confidence says what the values corroborated** (BLUEPRINT §14, Recognition's leash): ``high``
    only where the values agree with the name, codebook names included (a nutrient plausible as an
    intake that rises with total energy at r ≥ 0.3; total energy following its macronutrients; an
    identifier with a unit structure; a flag that marks its base's blanks; a time that changes
    within units; a survey design's values beside its design); a reading the name alone makes is
    ``medium`` at most and its reason says so; ``low`` is the lens's default for a column nothing
    recognized. Every proposal below high carries ``attention`` and needs its own confirmation
    (``confirm_role``) before any number-changing default reads it. Nothing here is applied."""
    from turbotab import nutrition
    from turbotab.core.recognizers import (
        TIME_CONSTANT, TIME_VARIES, acquisition_kind, has_id_tail, id_kind, is_rate, reads_as_time,
        study_group,
    )

    lenses = [k for k in (lens or []) if k in LENSES]
    dietary = "dietary" in lenses
    omics = any(k in OMICS for k in lenses)
    exact_design = {c.upper() for c in getattr(nutrition, "EXACT_DESIGN_NAMES", ())}
    design_in_table = _design_in_table(columns, exact_design)
    acquisition = set(acquisition) | {str(c["name"]) for c in columns
                                      if acquisition_kind(c["name"]) is not None}
    assay = acquisition_corroborated(columns, lens, acquisition)
    fractional = set(fractional)
    intake = dict(intake or {})
    facts = dict(facts or {})
    energy_info = dict(energy_info or {})
    rejected_energy = {str(x["column"]): x for x in energy_info.get("rejected") or []}
    other_energy = {str(x["column"]): x for x in energy_info.get("others") or []}
    lens_default = "covariate" if (dietary or omics) else "exposure"
    by_lower = {str(c["name"]).lower(): str(c["name"]) for c in columns}
    unit_word = f"`{repeating}`" if repeating else "unit"
    # The design the table names corroborates its weights only where its strata and PSUs hold the
    # whole-number codes a design's do (BLUEPRINT §14).
    design_fits = all(f.get("design_values", True) for c, f in facts.items()
                      if set(_tokens(c)) & {"psu", "strata", "stratum"}
                      or str(c).upper() in ("SDMVSTRA", "SDMVPSU"))
    out: list[dict[str, Any]] = []

    for summary in columns:
        name = str(summary["name"])
        if name == target or name.startswith("__"):
            continue
        dtype = str(summary.get("dtype") or "")
        numeric = dtype in ("numeric", "integer")
        n_present = max(0, int(summary.get("n", n_rows - int(summary.get("n_missing", 0)))))
        n_unique = int(summary.get("n_unique") or 0)
        unique = bool(n_present) and n_unique >= n_present
        tokens = set(_tokens(name))
        fact = facts.get(name) or {}
        proposal = {"column": name, "linked_to": None, "unit": _unit(name), "nested_in": None,
                    "kind": None}

        def put(role: str, confidence: str, reason: str, **extra: Any) -> None:
            out.append({**proposal, "proposed": role, "confidence": confidence, "reason": reason,
                        **extra})

        is_flag, base = _flag_base(name, by_lower)
        flag = fact.get("flag")
        design = _design_role(name, summary, exact_design, design_in_table)
        kind = None if name in fractional else id_kind(name)
        unit_check = fact.get("id")
        time_share = fact.get("time")
        timed = "time" in fact  # the values were read against the repeating units
        if is_flag and flag is not None and flag.verdict == "never":
            # BLUEPRINT §14: continuous values are never a flag (the gate's ``sbp_imp``).
            put(lens_default, "low", f"Named like a flag, but {flag.why}: kept as a predictor "
                                     f"until you say otherwise.")
        elif is_flag and flag is not None and flag.verdict == "indicator":
            put("covariate", "low", f"Named like a flag, but {flag.why}, kept as a predictor.")
        elif is_flag:
            if flag is not None and flag.verdict == "flag" and base:
                # BLUEPRINT §14.3 (amendment after the fifth gate): a survey's skip-pattern gate
                # marks its follow-up's blanks exactly as a flag does (NHANES ALQ111 "No" skips
                # ALQ130), and it is a characteristic the model needs; the user says which.
                put("flag", "medium", f"Marks `{base}`'s missing values: {flag.why}. A survey's "
                                      f"skip-pattern gate marks them the same way and stays in the "
                                      f"model: say which it is.", linked_to=base)
            elif flag is not None and base:
                put("flag", "medium", f"Named like a flag on `{base}`, but {flag.why}.",
                    linked_to=base)
            elif base:
                put("flag", "medium", f"Named like a flag on `{base}`; its values are not read.",
                    linked_to=base)
            else:
                put("flag", "medium", "Named like a flag that marks other values; not a measurement.")
        elif design is not None:
            fits = fact.get("design_values")
            confirmed = bool(_design_named(name, exact_design) and design_in_table and fits
                             and design_fits)
            # A reading below high says what was not shown (BLUEPRINT §14).
            doubt = ("" if confirmed else
                     " Its values were not read." if fits is None else
                     " Its values do not fit one." if not fits else
                     " The design's strata and PSUs do not hold codes." if not design_fits else
                     " Its name is no published design variable.")
            # BLUEPRINT §14.3: a published design name is a name, and positive weights and
            # whole-number codes fit a measurement and a predictor's codes as well; the user
            # confirms each design column (the survey question then names its part).
            put("design", "medium", design + (doubt or " Confirm it is part of the design."))
        elif name in acquisition and _acquisition_values_fit(name, n_unique, n_present) and assay:
            role, reason = acquisition_proposal(purpose)
            put(role, "medium", reason, kind="acquisition")
        elif name in acquisition and _acquisition_values_fit(name, n_unique, n_present):
            put("covariate", "low", "Named like an acquisition column, but nothing says this is an "
                                    "assay: kept in the model.")
        elif (kind in ("subject", "record", "cluster") or _norm(name) == "seqn") \
                and unit_check is not None and unit_check.verdict == "never":
            # BLUEPRINT §14: a 0/1, yes/no, Self/Proxy or kin-label column is never an identifier.
            put(lens_default, "low", f"Named like an identifier, but {unit_check.why}: kept as a "
                                     f"predictor until you say otherwise.")
        elif _norm(name) == "seqn" or kind == "subject":
            what = ("`SEQN` is the NHANES respondent number" if _norm(name) == "seqn"
                    else "Named like a participant's identifier")
            rows = fact.get("rows")
            if unit_check is None:
                put("identifier", "medium", f"{what}; its values are not read yet.")
            elif rows is not None and rows.settles:
                put("identifier", "high", f"{what}; {rows.evidence}.")
            elif name == named_unit and unit_check.verdict == "units":
                put("identifier", "high", f"The grain answer names it as the unit; "
                                          f"{unit_check.why}.")
            elif unit_check.verdict == "units":
                # BLUEPRINT §14.3 (the gate's MEPS ``PID``, a roster's ``person_no``): a value that
                # repeats may name a unit, or number people within a household; the values cannot
                # tell, so it is asked.
                put("identifier", "medium",
                    f"{what}, and {(rows.evidence if rows is not None else unit_check.why)}: "
                    f"whether it names each unit, or numbers people within a group (a household's "
                    f"line number), is asked.")
            else:
                put("identifier", "medium", f"{what}, but {unit_check.why}: say whether it names "
                                            f"units.")
        elif kind == "record" and n_unique > 2 and (n_unique > CATEGORY_LEVELS or unique):
            confirmed = unit_check is not None and unit_check.verdict == "units"
            rows = fact.get("rows")
            if confirmed and rows is not None and rows.settles:
                # One value per row that no measurement would take names each row (BLUEPRINT §14.3:
                # :func:`turbotab.core.readings.names_rows`).
                put("identifier", "high", f"Named like an identifier; it names rows or samples: "
                                          f"{rows.evidence}.")
            elif confirmed and name == named_unit:
                # A repeating code the grain answer names is the user's own unit.
                put("identifier", "high", f"The grain answer names it as the unit; "
                                          f"{unit_check.why}.")
            elif confirmed and unique:
                put("identifier", "medium",
                    f"Named like an identifier, one value per row, but "
                    f"{rows.evidence if rows is not None else 'its values were not read'}: say "
                    f"whether it names rows or measures something.")
            elif confirmed:
                # BLUEPRINT §14.1 (the gate's ``stratum_id``, HCHS/SOL ``PSU_ID``, ``recruiter_id``):
                # more than ten repeating codes are no unit structure by themselves; a stratum, a
                # sampling unit or an interviewer repeats the same way. Whether its rows belong
                # together, and so whether the intervals cluster by it, is the user's to say.
                put("identifier", "medium",
                    f"Named like an identifier, and {unit_check.why}: whether it names each unit, "
                    f"or groups units (a stratum, a sampling unit, an interviewer), is asked.")
            else:
                put("identifier", "medium", "Named like an identifier; its values are not read yet.")
        elif kind == "record" and n_unique > 2 and n_present and n_unique < n_present:
            # An identifier names units; a ``trt_id`` or ``tx_id`` holding three values on 312
            # rows is a code for groups, such as a trial's arms or a study's sites (audit WP13
            # gate repair: one more arm than ``arm_id``'s two brought "Names each unit; 3 units"
            # back, and the seal grouped by the arm).
            put(lens_default, "low",
                f"Named like an identifier, but `{n_unique:,}` values on `{n_present:,}` rows: a "
                f"group code, not a unit.")
        elif kind == "record" and unit_check is not None:
            put(lens_default, "low", f"Named like an identifier, but {unit_check.why}: kept as a "
                                     f"predictor until you say otherwise.")
        elif kind == "cluster" and unique:
            rows = fact.get("rows")
            put("identifier", "high" if rows is not None and rows.settles else "medium",
                "Names a group, such as a household, and is unique on every row"
                + (f": {rows.evidence}." if rows is not None else "."))
        elif kind == "cluster":
            put("cluster", "medium", "Named like a group of participants, such as a site or "
                                     "household; not a trait.")
        elif kind == "visit" and unique:
            put("identifier", "medium", "Names each record, such as a visit or encounter.")
        elif kind == "visit" and timed and time_share is not None and time_share >= TIME_VARIES:
            # BLUEPRINT §14.3: a measurement changes within units too; whether it orders the rows
            # is the user's (the repeats answer naming it settles it).
            put("time", "medium", f"A visit or encounter index that changes within each "
                                  f"{unit_word}{_rising(fact)}: say whether it orders the rows.")
        elif kind == "visit":
            put("time", "medium", "Named like a visit or encounter index: when a row was "
                                  "measured.")
        elif n_unique <= 1:
            put("excluded", "high", "Every row holds the same value, so it cannot explain anything.")
        elif not numeric and dtype in ("categorical", "text") and n_present and n_unique >= 0.95 * n_present:
            put("identifier", "medium", f"`{n_unique:,}` different values in `{n_present:,}` rows: it names rows.")
        elif energy_column is not None and name == energy_column:
            basis = energy_info.get("basis")
            r = energy_info.get("r")
            if basis == "values":
                put("energy", "high" if dietary else "medium",
                    f"Total energy intake: its values follow the macronutrients' energy (r = {r:.2f}).")
            elif basis == "codebook":
                # BLUEPRINT §14: a codebook name is a name; no macronutrients here corroborate it.
                put("energy", "medium",
                    "Total energy intake as its codebook names it; no macronutrients here to check "
                    "it against.")
            elif r is not None:
                put("energy", "medium",
                    f"Named as total energy intake; it follows the macronutrients' energy only "
                    f"loosely (r = {r:.2f}).")
            elif basis == "name":
                put("energy", "medium",
                    "Named as total energy intake; no macronutrients here to check it against.")
            else:  # read from the name alone, its values not read (a preview of the lens)
                put("energy", "medium",
                    "Named as total energy intake, the column energy adjustment works against.")
        elif numeric and dietary and name in other_energy and energy_column is not None:
            r = other_energy[name].get("r")
            tracks = f" (this one: r = {r:.2f})" if r is not None else ""
            put("covariate", "low",
                f"Also named as energy, but `{energy_column}` is the intake{tracks}; say what this is.")
        elif numeric and dietary and name in rejected_energy:
            # Named like total energy, but the values say it is not what people ate: a device's
            # energy expenditure (Fitbit ``Calories``, ActiLife ``Kcals``) beside a diet record.
            r = rejected_energy[name].get("r")
            put("covariate", "low",
                f"Named like total energy, but it does not track the macronutrients' energy "
                f"(r = {r:.2f}).")
        elif dtype == "datetime":
            if timed and time_share is not None and time_share >= TIME_VARIES:
                # BLUEPRINT §14.3 (amendment after the fifth gate): an assay's run or batch date
                # changes within units exactly as a visit date does; the user says which.
                put("time", "medium", f"Holds dates that change within each {unit_word}: when each "
                                      f"row was recorded, or a run or batch date (an assay's), "
                                      f"which changes the same way; say which.")
            elif timed and time_share is not None and time_share <= TIME_CONSTANT:
                # BLUEPRINT §14: a date constant within units (a birth or randomization date) is
                # never the evidence of when a unit's rows were measured.
                put("excluded", "low", f"A date that is the same on every row of a {unit_word}: "
                                       f"when it began, not when each row was measured.")
            elif timed and time_share is not None:
                put("time", "medium", f"Holds dates that change within only "
                                      f"`{time_share:.0%}` of {unit_word}s: say what they order.")
            elif timed:
                put("time", "medium", "Holds dates; no unit repeats here, so nothing shows they "
                                      "order a unit's rows.")
            else:
                put("time", "medium", "Holds dates or times: when a row was recorded.")
        elif (numeric and dietary and _is_nutrient(name) and name in intake
              and not intake[name].corroborated):
            # The name says a nutrient; the values say otherwise, or nothing but the name says so
            # beside an energy column (audit WP13 repair: ``lipid_disorder`` 0/1, ``dxa_fat_g`` at
            # a median of 25,000 g, ``BreathAlcohol``; BLUEPRINT §14: children's DXA arm fat).
            from turbotab.core.recognizers import read_nutrient

            try:
                what = read_nutrient(name).nutrient
            except Exception:  # noqa: BLE001 - two nutrients named: said plainly
                what = "a nutrient"
            put("covariate", "low", f"Named like {what}, but {intake[name].why}: not read as a "
                                    f"nutrient until you say it is one.")
        elif numeric and dietary and _is_nutrient(name):
            carries = energy_bearing(name)
            check = intake.get(name)
            unit = _ratio_unit(name)
            head = "A nutrient that carries energy" if carries else "A nutrient intake"
            if unit is not None:
                put("exposure", "medium",
                    f"A nutrient intake {unit}: an exposure, not a day's amount.")
            elif check is not None and check.by_values and "add up to" in check.why:
                put("exposure", "high", f"{head}: an exposure; {check.why}.")
            elif check is not None and check.by_values:
                put("exposure", "high",
                    f"{head}: an exposure; it rises with total energy (r = {check.r:.2f}).")
            else:
                put("exposure", "medium", f"Named as {head[0].lower()}{head[1:]}; {_doubt(check)}.")
        elif numeric and dietary and _intake_by_unit(name, summary.get("median")):
            put("exposure", "medium", "An intake by its unit, not a nutrient: an exposure under the "
                                      "dietary lens.")
        elif study_group(name) and has_id_tail(name) and n_unique > CATEGORY_LEVELS:
            # ``group_id`` with dozens of groups numbers clusters (therapy groups, litters), not
            # the arms a trial compares.
            put("cluster", "medium", f"`{n_unique:,}` groups, too many to be a study's arms: groups "
                                     f"rows, such as therapy groups or litters; not a trait.")
        elif study_group(name):
            put("exposure", "medium", "A study arm or group: what the study compares, an exposure.")
        elif tokens & _FASTING_TOKENS:
            put("covariate", "medium", "Fasting status: a known confounder, adjusted for rather than studied.")
        elif reads_as_time(name) and timed and time_share is not None and time_share >= TIME_VARIES:
            # BLUEPRINT §14.3 (the gate: a sleep diary's ``hours``, an activity log's ``days``):
            # any measurement changes within units, so varying is no evidence of time. The best
            # guess follows the values: one that rises with each unit's rows may order them; one
            # that rises and falls is a measurement in a time unit. Either is asked.
            if (fact.get("rising") or 0.0) >= TIME_VARIES:
                put("time", "medium", f"Named like a time, and it rises with each {unit_word}'s "
                                      f"rows: say whether it orders them.")
            else:
                put(lens_default, "medium",
                    f"Named like a time unit, but its values rise and fall within each "
                    f"{unit_word}, as a measurement's do (hours slept, days active): read as a "
                    f"measurement until you say otherwise.")
        elif reads_as_time(name) and timed and (time_share is None or time_share <= TIME_CONSTANT):
            # BLUEPRINT §14: "a time column varies within units". A crossover's ``period`` or a
            # menstrual ``cycle_day`` on one row per unit orders nothing; it stays a predictor.
            why = ("no unit repeats here, so it orders nothing" if time_share is None else
                   f"it is the same on every row of a {unit_word}")
            put(lens_default, "low", f"Named like a time, but {why}: kept as a predictor until "
                                     f"you say otherwise.")
        elif reads_as_time(name):
            put("time", "medium", "Named like a time: a date, year, cycle, visit or recall.")
        elif tokens & _COVARIATE_TOKENS:
            # BLUEPRINT §14: high only where the values fit the characteristic the name says (a sex
            # with at most three levels, an age from 0 to 120, a BMI from 10 to 80). §14.3: the
            # reading is "a predictor", against a column left out (an identifier, a constant) and the
            # time axis of repeated rows (an age at each visit): one that changes within most units
            # may be that axis, so it is asked (``readings.KIND_RULES["role:covariate"]``).
            axis = timed and time_share is not None and time_share >= TIME_VARIES
            settles = fact.get("covariate") is not None and fact["covariate"].settles
            put("covariate", "high" if (tokens & {"age", "sex", "gender", "bmi"} and settles)
                else "medium",
                "A person's characteristic, usually adjusted for rather than studied."
                + (f" It changes within each {unit_word}: say whether it is the time axis."
                   if axis else ""))
        elif dtype == "text":
            # BLUEPRINT §14.3 (the gate: country of birth, 70 labels on 800 rows): a label count is
            # no evidence of free text, so leaving the column out is asked. The best guess follows
            # the values: labels mostly seen once and several words long read as free text; labels
            # that repeat read as a category with many levels.
            text = fact.get("text") or {}
            once, words = float(text.get("once_share") or 0.0), float(text.get("words") or 0.0)
            if once >= 0.5 and words >= 3:
                put("excluded", "medium",
                    f"Free text: `{once:.0%}` of rows hold a label seen once, about "
                    f"`{words:.0f}` words long; models cannot use it as is. Left out unless you "
                    f"say it is a category.")
            else:
                top = float(text.get("top_share") or 0.0)
                put(lens_default, "medium",
                    f"A category with `{n_unique:,}` labels (the most common on `{top:.0%}` of "
                    f"rows): many levels for a model; say whether to keep it or leave it out.")
        elif is_rate(name) and not (dietary or omics):
            put("exposure", "low", "An amount per day or week: a rate, read as an exposure for now.")
        elif omics and numeric:
            put("exposure", "medium" if n_unique > 2 else "low",
                "A measured feature: an exposure under the omics lens.")
        elif dietary or omics:
            put("covariate", "low", "Not a nutrient or feature: read as a covariate until you say otherwise.")
        elif dtype == "boolean" or (not numeric):
            put("covariate", "low", "A category that describes the row: read as a covariate for now.")
        else:
            put("exposure", "low", "A measured column: read as an exposure until you say otherwise.")
    for p in out:
        p["attention"] = p["confidence"] != "high"
    return out


# NHANES variables whose numbers are codes for categories (the codebooks' "Code or Value" tables):
# race and Hispanic origin, education, marital status, citizenship, birthplace, income bands,
# interview language, pregnancy status, military service, self-rated health, insurance.
NHANES_CODED = frozenset({
    "RIDRETH1", "RIDRETH3", "DMDEDUC2", "DMDEDUC3", "DMDMARTL", "DMDCITZN", "DMDBORN4",
    "DMDBORN2", "INDHHIN2", "INDFMIN2", "SIALANG", "SIAPROXY", "RIDEXPRG", "DMQMILIZ", "HUQ010",
    "HIQ011", "SMQ040", "SMQ020", "DIQ010", "BPQ020", "ALQ101", "ALQ111", "PAQ605", "PAQ650",
    "OCD150", "DMDHREDU", "DMDHRMAR", "DMDHRGND", "DMDHREDZ", "DMDHRMAZ",
})
CATEGORY_LEVELS = 10  # whole-number columns with at most this many values may be codes


def categorical_proposals(columns: Sequence[Mapping[str, Any]],
                          proposals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Predictors whose numbers may be codes for categories, to declare with ``set_categorical``.

    A whole-number column with 3 to CATEGORY_LEVELS values (two levels make one indicator either
    way), or a known NHANES coded variable with at least three. Entered as one number, RIDRETH3's
    codes 1, 2, 3, 4, 6, 7 would be one straight line across ethnic groups (audit MA-15). A
    suggestion only: counts and scores take few values too.
    """
    roles = {str(p["column"]): str(p["proposed"]) for p in proposals}
    out: list[dict[str, Any]] = []
    for s in columns:
        name = str(s["name"])
        if roles.get(name) not in PREDICTOR_ROLES:
            continue
        n = int(s.get("n_unique") or 0)
        if str(s.get("dtype")) not in ("integer", "numeric") or n < 3:
            continue
        if name.upper() in NHANES_CODED:
            out.append({"column": name, "levels": n, "confidence": "high",
                        "reason": f"An NHANES coded variable: its `{n}` values name groups, not amounts."})
        elif str(s.get("dtype")) == "integer" and n <= CATEGORY_LEVELS:
            out.append({"column": name, "levels": n, "confidence": "medium",
                        "reason": f"Whole numbers with `{n}` values: if they are codes for groups, one "
                                  f"slope across them means nothing."})
    return out


def _energy_reading(columns: Sequence[Mapping[str, Any]]) -> str | None:
    """The total-energy column read from names and summaries alone (:func:`_energy_info` without
    the values), or None."""
    return _energy_info(columns).get("column")


def _energy_info(columns: Sequence[Mapping[str, Any]], store: Any = None) -> dict[str, Any]:
    """The total-energy column and how it was read, by the one recognizer every stage shares
    (``stages.proposals.energy_column_reading``: ``DR2TKCAL``, ``ENERC_KCAL``, ``TotalKcal`` and
    ``energy_kj`` alike; never a macronutrient's own kcal such as ``alc_kcal``), with its values
    read against the macronutrients' energy when ``store`` is given: a device's energy
    expenditure named ``Calories`` is no intake (audit WP13 gate repair)."""
    from turbotab.core.recognizers import AmbiguousNutrient, read_nutrient
    from turbotab.core.stages.proposals import energy_column_reading, is_energy_name

    info = {str(c["name"]): c for c in columns}

    def total(c: str) -> bool:
        try:
            reading = read_nutrient(c)
        except AmbiguousNutrient:
            return False
        return reading is not None and reading.macro in ("protein", "carbohydrate", "fat",
                                                         "alcohol") and reading.part is None

    numeric = [c for c, i in info.items() if str(i.get("dtype") or "") in ("numeric", "integer")]
    try:
        frame = None
        if store is not None:
            wanted = [c for c in numeric if is_energy_name(c) or total(c)]
            frame = store.materialize(wanted) if wanted else None
        reading = energy_column_reading(info, {}, frame)
    except Exception as exc:  # a recognizer failure must not fail the stage
        from turbotab import devchecks

        devchecks.swallowed("roles::energy_reading", exc,
                            "the energy column could not be read; no column is proposed as energy")
        reading = None
    return dict(reading or {"column": None, "rejected": []})


def _acquisition_columns(columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Batch, plate, well and run-order columns, by whole names
    (:func:`turbotab.core.recognizers.acquisition_kind`). The legacy reader also claimed a study's
    arms, groups, phenotype, fasting status and site, and dropped them from the model as "an
    acquisition column" (audit IN-02); those are read on their own now."""
    from turbotab.core.recognizers import acquisition_kind

    return [str(c["name"]) for c in columns if acquisition_kind(c["name"]) is not None]


def intake_checks(store: Any, columns: Sequence[Mapping[str, Any]], *, energy: str | None,
                  target: str | None) -> dict[str, Any]:
    """The values' verdict on every numeric column the name reads as a nutrient, codebook names
    included, against the total-energy column when there is one, in its settled unit, with
    duplicates resolved (:func:`turbotab.core.recognizers.corroborated_nutrients`; BLUEPRINT §14)."""
    from turbotab.core.readings import by_values_table

    numeric = [str(c["name"]) for c in columns
               if str(c.get("dtype") or "") in ("numeric", "integer")
               and str(c["name"]) not in (energy, target)]
    names = [c for c in numeric if _is_nutrient(c)]
    if not names:
        return {}
    frame = store.materialize(list(dict.fromkeys([*names, *([energy] if energy else [])])))
    unit = proposed = None
    if energy and energy in frame.columns:
        from turbotab.core.stages.proposals import energy_unit_reading

        reading = energy_unit_reading(frame, energy)
        # The unit alone (a share is the same over one day or several): settled by the record or
        # the Atwater identity; a name's or a median's unit only sets what a contradiction reads.
        unit = reading["unit"] if reading.get("unit_settled") else None
        proposed = reading.get("unit")
    # The registry's one test for a nutrient intake (``readings.KIND_RULES["role:exposure"]``), over
    # the whole table: its verdicts carry each column's intake check for the words.
    verdicts = by_values_table("role:exposure", frame,
                               energy=energy if energy in frame.columns else None,
                               energy_unit=unit, skip=[c for c in (target,) if c],
                               proposed_unit=proposed if energy else None)
    return {c: v.detail for c, v in verdicts.items()}


def value_facts(store: Any, columns: Sequence[Mapping[str, Any]], *, target: str | None,
                fractional: Iterable[str] = (), named_unit: str | None = None,
                ) -> tuple[dict[str, dict[str, Any]], str | None]:
    """What the values say about the columns named like an identifier, a flag, a time or a survey
    design (BLUEPRINT §14 rule 1), and the identifier whose values name repeating units, or None:
    the unit the grain answer names (``named_unit``), else a person's repeating identifier (the one
    with the fewest units). A repeating ``*_id`` read only by its name (a stratum, a PSU, an
    interviewer) is no unit a time is measured within (BLUEPRINT §14.1: the cluster reading is the
    user's to settle).

    ``id``: :func:`turbotab.core.recognizers.identifier_values`; ``flag``:
    :func:`~turbotab.core.recognizers.flag_values` against the column it would mark; ``time``: the
    share of the repeating units within which a date or a time-named column changes
    (:func:`~turbotab.core.recognizers.within_unit_variation`; None when no unit repeats);
    ``design_values``: :func:`design_values_fit`."""
    from turbotab import nutrition
    from turbotab.core.recognizers import (
        flag_values, id_kind, identifier_values, reads_as_time, within_unit_variation,
    )

    fractional = set(fractional)
    by_lower = {str(c["name"]).lower(): str(c["name"]) for c in columns}
    exact = {c.upper() for c in getattr(nutrition, "EXACT_DESIGN_NAMES", ())}
    in_table = _design_in_table(columns, exact)
    ids, flags, times, designs, traits, texts = [], {}, [], [], [], []
    for c in columns:
        name = str(c["name"])
        if name == target or name.startswith("__"):
            continue
        if str(c.get("dtype") or "") == "text":
            texts.append(name)
        kind = id_kind(name)
        if name not in fractional and (kind is not None or _norm(name) == "seqn"):
            ids.append(name)
        is_flag, base = _flag_base(name, by_lower)
        if is_flag:
            flags[name] = base
        if str(c.get("dtype") or "") == "datetime" or reads_as_time(name) or kind == "visit":
            times.append(name)
        if _design_role(name, c, exact, in_table) is not None:
            designs.append(name)
        if set(_tokens(name)) & {"age", "sex", "gender", "bmi"}:
            traits.append(name)
    wanted = list(dict.fromkeys([*ids, *flags, *(b for b in flags.values() if b), *times,
                                 *designs, *traits, *texts]))
    facts: dict[str, dict[str, Any]] = {}
    if not wanted:
        return facts, None
    # The identifier and covariate readings are settled only by the registry's tests (BLUEPRINT
    # §14.3: one test per kind, ``readings.by_values``); a flag's and a time's values are only their
    # guesses' evidence (no value test settles either).
    from turbotab.core.readings import by_values

    frame = store.materialize(wanted)
    for name in ids:
        facts.setdefault(name, {})["id"] = identifier_values(frame[name])
        facts[name]["rows"] = by_values("role:identifier", frame[name])
    for name in texts:
        facts.setdefault(name, {})["text"] = text_values(frame[name])
    for name, base in flags.items():
        facts.setdefault(name, {})["flag"] = flag_values(
            frame[name], frame[base] if base else None, base)
    for name in designs:
        facts.setdefault(name, {})["design_values"] = design_values_fit(name, frame[name])
    repeating = None
    best = None
    for name in ids:
        check = facts[name]["id"]
        named = name == named_unit
        if check.verdict == "units" and check.n_units < check.n_rows \
                and (named or id_kind(name) == "subject" or _norm(name) == "seqn"):
            if named:
                repeating, best = name, -1
            elif best is None or (best >= 0 and check.n_units < best):
                repeating, best = name, check.n_units
    for name in list(dict.fromkeys([*times, *traits])):
        if name == repeating:
            continue
        facts.setdefault(name, {})["time"] = (within_unit_variation(frame[name], frame[repeating])
                                              if repeating else None)
        facts[name]["rising"] = (within_unit_rising(frame[name], frame[repeating])
                                 if repeating else None)
    for name in traits:
        facts.setdefault(name, {})["covariate"] = by_values(
            "role:covariate", name, frame[name],
            frame[repeating] if repeating and name != repeating else None)
    return facts, repeating


def within_unit_rising(values: Any, units: Any) -> float | None:
    """The share of repeating units whose values rise strictly with their rows, in the table's
    order (a visit index or a date does; hours slept rise and fall): the best guess a time question
    leads with, never its settlement. None when no unit repeats."""
    import pandas as pd

    frame = pd.DataFrame({"v": pd.Series(values).to_numpy(), "u": pd.Series(units).to_numpy()})
    frame = frame.dropna(subset=["u"])
    sizes = frame.groupby("u").size()
    repeating = sizes[sizes > 1].index
    if not len(repeating):
        return None
    inner = frame[frame["u"].isin(repeating)]
    try:
        v = pd.to_numeric(inner["v"], errors="coerce") if not \
            pd.api.types.is_datetime64_any_dtype(inner["v"]) else inner["v"]
        rising = v.groupby(inner["u"]).apply(lambda x: bool(x.notna().all() and
                                                         x.is_monotonic_increasing and x.is_unique))
    except Exception:  # noqa: BLE001 - values that cannot be ordered rise nowhere
        return 0.0
    return float(rising.mean())


def text_values(values: Any) -> dict[str, float]:
    """What a text column's labels look like: the share of rows holding a label seen once, the
    labels' median length in words, and the most common label's share of rows (the best guess
    between free text and a category with many levels; never a settlement)."""
    import pandas as pd

    s = pd.Series(values).dropna().astype(str)
    if s.empty:
        return {"once_share": 0.0, "words": 0.0, "top_share": 0.0}
    counts = s.value_counts()
    once = float(s.map(counts).eq(1).mean())
    words = float(s.str.split().map(len).median())
    return {"once_share": once, "words": words, "top_share": float(counts.iloc[0] / len(s))}


def _rising(fact: Mapping[str, Any]) -> str:
    share = fact.get("rising")
    if share is None:
        return ""
    return (" and rises with its rows" if share >= 0.5 else
            " but rises and falls within it, as a measurement does")


def _fractional_identifiers(store: Any, columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Identifier-named float columns whose values have fractional parts: measurements, never a
    unit's identifier (audit IN-06). A whole-number ID read as float because one cell is blank
    (``SEQN`` with a gap) stays an identifier."""
    from turbotab.core.recognizers import id_kind, reads_as_measurement

    named = [str(c["name"]) for c in columns
             if str(c.get("dtype") or "") == "numeric" and id_kind(c["name"]) is not None]
    out = []
    for name in named:
        values = store.materialize([name])[name]
        # Read with the name blinded: the question is the values', not the name's.
        if reads_as_measurement("__values__", values=values) == "its values have fractional parts":
            out.append(name)
    return out


def _repeats(store: Any, proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The identifier that repeats (several rows per unit), with its counts, or None: only one the
    values corroborate (a unit structure, BLUEPRINT §14), never a name's reading alone."""
    best: dict[str, Any] | None = None
    for p in proposals:
        if p["proposed"] != "identifier" or p.get("confidence") != "high":
            continue
        series = store.materialize([p["column"]])[p["column"]].dropna()
        if series.empty:
            continue
        counts = series.value_counts()
        n_units, most = int(len(counts)), int(counts.max())
        if most > 1 and (best is None or n_units < best["n_units"]):
            best = {"column": p["column"], "n_units": n_units, "max_rows_per_unit": most}
    return best


def _nesting(store: Any, columns: Sequence[Mapping[str, Any]], target: str | None) -> dict[str, str]:
    """Child -> parent over every row: the names admit it and the part never exceeds the total. The
    guess the proposals show (``readings.nesting``, no answer read: the roles stage reads none)."""
    from turbotab.core.methods.nesting import candidates
    from turbotab.core.readings import nesting

    names = [str(c["name"]) for c in columns if str(c["name"]) != target
             and str(c.get("dtype") or "") in ("numeric", "integer")]
    pairs = candidates(names)
    needed = list(dict.fromkeys([*pairs, *(c for kids in pairs.values() for c in kids)]))
    if not needed:
        return {}
    return nesting(None, frame=store.materialize(needed), columns=needed)


def _composition_reference(store: Any, proposals: Sequence[Mapping[str, Any]]) -> tuple[str, int] | None:
    """(the share to leave out, how many shares) when the exposures' energy shares sum to 100%."""
    from turbotab.core.methods.nesting import _is_share, compositions, reference_share

    shares = [str(p["column"]) for p in proposals if p["proposed"] == "exposure" and _is_share(p["column"])]
    if len(shares) < 2:
        return None
    frame = store.materialize(shares)
    found = compositions(frame, shares)
    return (reference_share(frame, found), len(found)) if found else None


def roles_stage(ctx: StageContext) -> dict[str, Any]:
    """Proposed roles for every column but the outcome, and whether rows repeat per unit."""
    from turbotab.core.stages.working import table_info

    n_rows = int(table_info(ctx)["n_rows"])
    ctx.progress(0.05, "Summarizing the working table's columns")
    if "profile" in ctx.inputs:  # the M1 wiring (and its tests) hand the profile in
        columns = ctx.inputs["profile"]["columns"]
    else:  # M2: the working table's own summaries, cached beside it
        with open_store(ctx) as store:
            columns = store.summaries()
    columns = [c for c in columns if not str(c["name"]).startswith("__")]
    ctx.progress(0.1, "Reading column names and summaries")
    dietary = "dietary" in (ctx.state.lens or [])
    with open_store(ctx) as store:
        energy_info = _energy_info(columns, store)
        energy = energy_info.get("column")
        fractional = _fractional_identifiers(store, columns)
        intake = (intake_checks(store, columns, energy=energy, target=ctx.state.target)
                  if dietary else {})
        grain = ctx.state.grain
        named_unit = (grain.id_column if grain is not None and grain.grain == "repeated"
                      else None)
        facts, repeating = value_facts(store, columns, target=ctx.state.target,
                                       fractional=fractional, named_unit=named_unit)
    proposals = propose_roles(
        columns,
        lens=ctx.state.lens,
        target=ctx.state.target,
        n_rows=n_rows,
        energy_column=energy,
        acquisition=_acquisition_columns(columns),
        purpose=ctx.state.purpose,
        fractional=fractional,
        intake=intake,
        energy_info=energy_info,
        facts=facts,
        repeating=repeating,
        named_unit=named_unit,
    )
    ctx.progress(0.6, "Checking whether identifiers repeat")
    with open_store(ctx) as store:
        repeats = _repeats(store, proposals)
        ctx.progress(0.8, "Checking which nutrients are parts of others")
        nested = _nesting(store, columns, ctx.state.target)
        reference = _composition_reference(store, proposals)
    for p in proposals:
        p["nested_in"] = nested.get(p["column"])
        if reference is not None and p["column"] == reference[0]:
            p["proposed"], p["confidence"] = "excluded", "medium"
            p["reason"] = (f"{reference[1]} shares sum to 100%, so one must leave; this one is "
                           f"the reference.")
    # WP18 (audit RO-10): with the outcome on its log scale, the column it is the log of is the
    # outcome itself, never a predictor; the user's own scale answer settles it.
    from turbotab.core.structural import log_outcome

    source = log_outcome(ctx.state)
    for p in proposals:
        if source is not None and p["column"] == source:
            p["proposed"], p["confidence"] = "excluded", "high"
            p["reason"] = (f"The outcome on its original scale; the analysis reads "
                           f"`{ctx.state.target}`, its natural log.")
    # The routing gate (RO-03 on NHANES linked mortality): a column the follow-up answer names is
    # the time half of a time-to-event outcome, never a predictor; the user's own answer settles it
    # (``PERMTH_INT`` was proposed an exposure).
    follow_up = getattr(ctx.state, "follow_up", None)
    if follow_up is not None and getattr(ctx.state, "task", None) == "time_to_event":
        named = {follow_up.time_column: "follow-up time", follow_up.entry_column: "entry time"}
        for p in proposals:
            what = named.get(p["column"])
            if what:
                p["proposed"], p["confidence"] = "time", "high"
                p["reason"] = (f"Named as the outcome's {what} (the follow-up question): part of "
                               f"the outcome `{ctx.state.target}`, not a predictor.")
    if repeats is not None:
        for p in proposals:
            if p["column"] == repeats["column"]:
                p["reason"] = (f"Names each unit; `{repeats['n_units']:,}` units, up to "
                               f"`{repeats['max_rows_per_unit']}` rows each.")
    from turbotab.core.readings import attention_columns

    for p in proposals:
        p["attention"] = p["confidence"] != "high"
    # The routing gate's leash note (LEASH): under inference, every column that can structurally
    # group rows, read by its values, for the grouping question (``turbotab.core.groupings``).
    groupings: list[dict[str, Any]] = []
    if ctx.state.purpose == "inference":
        from turbotab.core.groupings import structural_facts

        assay = any(k in OMICS for k in (ctx.state.lens or []))
        with open_store(ctx) as store:
            groupings = structural_facts(store, columns, target=ctx.state.target, assay=assay)
    # BLUEPRINT §14 rule 2: every proposal below high needs its own confirmation (``confirm_role``)
    # before a number-changing default reads it; a bulk ``set_roles`` records it unconfirmed.
    return {"columns": proposals, "repeats": repeats,
            "categorical": categorical_proposals(columns, proposals),
            "needs_confirmation": attention_columns(proposals), "groupings": groupings}


# ── cohort ────────────────────────────────────────────────────────────────────


def _level_key(value: Any) -> str:
    """One spelling per level: ``2``, ``2.0`` and ``"2"`` are the same level."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(value)


def _group_key(value: Any) -> str:
    """One unit per identifier as written: text identifiers are exact ("007", "07" and "7" are three
    people, audit C12); a number is its value (``7`` and ``7.0`` are one)."""
    if isinstance(value, str):
        return "s:" + value
    return "n:" + _level_key(value)


def rule_parts(frame: Any, rule: Any) -> tuple[Any, Any, Any]:
    """``(recorded, screened, inside)`` Boolean Series for ``rule`` over ``frame``.

    ``recorded``: the column has a value. ``screened``: a range applies to the row; without ``by``
    always, with ``by`` when the row's level of ``by.column`` has its own range or the rule has
    bounds of its own (a row whose sex is missing or unrecognized, under a by-sex rule with no
    bounds of its own, is screened by nothing: audit D16). ``inside``: ``low <= value <= high``
    (either bound may be open).
    """
    import pandas as pd

    rule = _as_rule(rule)
    if _is_goldberg(rule):  # EI:BMR against the Goldberg cut-offs (methods/misreporting.py)
        from turbotab.core.methods.misreporting import parts

        return parts(frame, rule)
    values = pd.to_numeric(frame[rule.column], errors="coerce").astype(float)
    low = pd.Series(np.nan if rule.low is None else float(rule.low), index=frame.index, dtype=float)
    high = pd.Series(np.nan if rule.high is None else float(rule.high), index=frame.index, dtype=float)
    screened = pd.Series(True, index=frame.index)
    if rule.by is not None:
        levels = frame[rule.by.column].map(lambda v: None if pd.isna(v) else _level_key(v))
        for key, (lo, hi) in rule.by.ranges.items():
            at = levels == _level_key(key)
            low[at] = np.nan if lo is None else float(lo)
            high[at] = np.nan if hi is None else float(hi)
        if rule.low is None and rule.high is None:
            screened = levels.isin([_level_key(k) for k in rule.by.ranges])
    inside = (low.isna() | (values >= low)) & (high.isna() | (values <= high))
    return values.notna(), screened, inside


def rule_keep(frame: Any, rule: Any) -> Any:
    """Boolean Series: True where ``rule`` keeps the row.

    A range keeps ``low <= value <= high``. A row it cannot confirm (no value, or under ``by`` no
    range for its level) is left out by default, the STROBE reading of "confirmed eligible"; with
    ``missing="keep"`` it is kept, and the flow's label says so (audit MA-19).
    """
    rule = _as_rule(rule)
    recorded, screened, inside = rule_parts(frame, rule)
    confirmed = recorded & screened
    if rule.missing == "keep":
        return ~confirmed | inside
    return confirmed & inside


def _as_rule(rule: Any) -> Any:
    from turbotab.core.decisions import as_rule

    return as_rule(rule)


def _is_goldberg(rule: Any) -> bool:
    return getattr(rule, "kind", "range") == "goldberg"


def rule_lines(rule: Any) -> tuple[str, str | None, str, str]:
    """The labels and reasons of a rule's own lines before its step: ``(not recorded label, not
    screened label or None, not recorded reason, not screened reason)``."""
    if _is_goldberg(rule):
        from turbotab.core.methods.misreporting import EQUATIONS

        inputs = [f"`{c}`" for c in rule.reads()]
        listed = ", ".join(inputs[:-1]) + f" or {inputs[-1]}" if len(inputs) > 1 else inputs[0]
        eq = EQUATIONS[rule.equation].label
        return (f"{listed} not recorded",
                f"Not screened: no {eq} BMR or PAL for their age, sex or activity",
                "a value the Goldberg screen needs is missing, so it cannot confirm the row",
                "the BMR equation or the PAL has no value for the row")
    col = f"`{rule.column}`"
    by = f"`{rule.by.column}`" if rule.by is not None else None
    return (f"{col} not recorded",
            f"{col} not screened: no range for its {by}" if by is not None else None,
            f"no value for {col}, so the rule cannot confirm the row",
            f"its {by} is missing or has no range of its own")


def rule_label(rule: Any) -> str:
    """What the rows a rule keeps are: true of every one of them (``or not recorded`` when the
    rule keeps the rows it cannot confirm)."""
    rule = _as_rule(rule)
    if _is_goldberg(rule):
        from turbotab.core.methods.misreporting import label

        return label(rule)
    col = f"`{rule.column}`"
    if rule.by is not None:
        text = f"{col} within its range for each `{rule.by.column}`"
    elif rule.low is not None and rule.high is not None:
        text = f"{col} within `{_fmt_num(rule.low)}`–`{_fmt_num(rule.high)}`"
    elif rule.low is not None:
        text = f"{col} at least `{_fmt_num(rule.low)}`"
    else:
        text = f"{col} at most `{_fmt_num(rule.high)}`"
    return text + (", or not recorded" if rule.missing == "keep" else "")


def rule_drops(steps: Sequence[Mapping[str, Any]]) -> list[int]:
    """Rows each exclusion rule left out, in rule order: its range step and the lines before it
    (not recorded, not screened) together. Read by the decision's sentence and the preview."""
    out: dict[int, int] = {}
    for s in steps:
        key = str(s["key"])
        if not key.startswith("exclusion:"):
            continue
        i = int(key.split(":")[1])
        out[i] = out.get(i, 0) + int(s["dropped"])
    return [out[i] for i in sorted(out)]


def _measured(series: Any) -> Any:
    import pandas as pd

    present = series.notna()
    if pd.api.types.is_float_dtype(series):
        present &= np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
    return present


def cohort_flow(
    frame: Any,
    *,
    target: str | None,
    rules: Sequence[Any] | None,
    missing: str | None,
    predictor_columns: Sequence[str],
    n_loaded: int | None = None,
    missing_frame: Any | None = None,
    repairs: Sequence[Any] | None = None,
    reference: Sequence[Mapping[str, Any]] | None = None,
    landmark: tuple[str, float] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """The participant flow over ``frame`` (indexed by row id): its steps and the kept row ids.

    ``frame`` holds the target and every column the rules read (a preview may run before an
    outcome is chosen: ``target=None`` leaves out ``outcome_measured``). Complete cases are judged on
    ``predictor_columns`` in ``missing_frame`` (default: ``frame``); columns absent from it are
    taken to have no missing values (the caller leaves them out when the profile says so).
    ``repairs`` are range rules from applied repairs (impossible values, ``repairs.exclusion_rules``):
    steps ``repair:<i>`` after the eligibility answer's own. ``landmark`` (``(time column, L)``,
    a time-to-event outcome's follow-up counted from L): the rows whose follow-up ended by L leave
    on a line of their own, not at risk then (never an eligibility rule: it reads the outcome).
    """
    steps: list[dict[str, Any]] = []
    n = len(frame) if n_loaded is None else int(n_loaded)
    # WP18 (audit RO-13): reference rows (pooled QC injections) left the working table before
    # anything was read from it; the flow counts them first, each on a line of its own.
    gone = [r for r in (reference or []) if int(r.get("n") or 0)]
    total = n + sum(int(r["n"]) for r in gone)
    steps.append({"key": "loaded", "label": "Rows in the table", "n": total, "dropped": 0,
                  "reason": None, "decision_id": None})
    for i, r in enumerate(gone):
        total -= int(r["n"])
        levels = " or ".join(f"`{v}`" for v in r.get("levels") or [])
        steps.append({"key": f"reference:{i}", "label": f"`{r['column']}` is not {levels}",
                      "n": total, "dropped": int(r["n"]),
                      "reason": "an instrument run (a reference row), not a participant",
                      "decision_id": None})
    import pandas as pd

    if target is not None:
        keep = _measured(frame[target])
        kept = int(keep.sum())
        steps.append({"key": "outcome_measured", "label": f"`{target}` recorded", "n": kept,
                      "dropped": n - kept, "reason": f"no value for the outcome `{target}`",
                      "decision_id": None})
    else:
        keep = pd.Series(True, index=frame.index)
        kept = len(frame)
    for i, rule in enumerate(rules or []):
        # STROBE item 13(a) counts who was "confirmed eligible": a row whose value is unknown is
        # not, so it leaves on a line of its own before the range (audit MA-19), and the range's
        # label is then true of every row it counts. With missing="keep" it stays, and both the
        # label and a line of its own say so.
        rule = _as_rule(rule)
        recorded, screened, inside = rule_parts(frame, rule)
        unknown = keep & ~recorded
        unscreened = keep & recorded & ~screened
        unrecorded_label, unscreened_label, unrecorded_why, unscreened_why = rule_lines(rule)
        lines = [(f"exclusion:{i}:not_recorded", unrecorded_label, unknown, unrecorded_why)]
        if unscreened_label is not None:
            lines.append((f"exclusion:{i}:not_screened", unscreened_label, unscreened,
                          unscreened_why))
        for key, label, rows, reason in lines:
            n_rows = int(rows.sum())
            if not n_rows:
                continue
            if rule.missing == "keep":
                steps.append({"key": key, "label": f"{label}: `{n_rows:,}` kept", "n": kept,
                              "dropped": 0, "reason": None, "decision_id": None})
            else:
                keep &= ~rows
                now = int(keep.sum())
                steps.append({"key": key, "label": label, "n": now, "dropped": kept - now,
                              "reason": reason, "decision_id": None})
                kept = now
        keep &= rule_keep(frame, rule)
        now = int(keep.sum())
        steps.append({"key": f"exclusion:{i}", "label": rule_label(rule), "n": now,
                      "dropped": kept - now, "reason": rule.reason, "decision_id": None})
        kept = now
    for i, rule in enumerate(repairs or []):
        rule = _as_rule(rule)
        keep &= rule_keep(frame, rule)
        now = int(keep.sum())
        steps.append({"key": f"repair:{i}", "label": rule_label(rule), "n": now,
                      "dropped": kept - now, "reason": rule.reason, "decision_id": None})
        kept = now
    if landmark is not None and landmark[0] in frame.columns:
        column, at = landmark
        time = pd.to_numeric(frame[column], errors="coerce")
        keep &= time > float(at)
        now = int(keep.sum())
        steps.append({"key": "landmark", "label": f"Followed past the landmark, `{column}` > "
                                                  f"`{float(at):g}`",
                      "n": now, "dropped": kept - now,
                      "reason": "follow-up ended by the landmark (an event or not): not at risk then",
                      "decision_id": None})
        kept = now
    if missing == "complete_case":
        source = frame if missing_frame is None else missing_frame
        present = [c for c in predictor_columns if c in source.columns]
        for start in range(0, len(present), CHUNK_COLUMNS):
            block = source.loc[keep[keep].index, present[start:start + CHUNK_COLUMNS]]
            complete = block.notna().all(axis=1)
            keep.loc[complete.index] &= complete
        now = int(keep.sum())
        steps.append({"key": "complete_cases", "label": "No predictor missing", "n": now,
                      "dropped": kept - now, "reason": "a predictor has no value",
                      "decision_id": None})
    return steps, keep[keep].index.to_numpy(dtype=np.int64)


def _columns_with_missing(ingest: Mapping[str, Any]) -> set[str]:
    return {str(c["name"]) for c in ingest.get("columns", []) if int(c.get("n_missing") or 0) > 0}


def cohort_inputs(state: Any, ingest: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """(columns the flow reads, predictors, predictors that can be missing), in table order.

    The predictors are those the missing-values answer kept: a column it left out is judged by
    nothing, so its blanks never drop a row. Nor does a column whose blanks become their own
    ``Missing`` level (``SetMissing.categorical == "missing_category"``): a blank there is a value.
    """
    from turbotab.core.decisions import left_out, missing_strategy
    from turbotab.core.models.pipeline import level_columns

    order = [str(c["name"]) for c in ingest.get("columns", [])]
    # The settled roles only (BLUEPRINT §14.1): a role that rode along unconfirmed drops no row by
    # its blanks; the fit waits for its confirmation (``readings.predictors_or_ask``).
    from turbotab.core.readings import predictor_columns

    preds = predictor_columns(state, order, drop=left_out(state))
    needed = [state.target] if state.target is not None else []
    for rule in [*(state.exclusions or []), *repair_rules(state)]:
        needed.extend(_as_rule(rule).reads())
    mark = landmark_of(state)
    if mark is not None:
        needed.append(mark[0])
    with_missing = _columns_with_missing(ingest)
    levels = set(level_columns(state, preds, {str(c["name"]): c for c in ingest.get("columns", [])}))
    gappy = ([c for c in preds if c in with_missing and c not in levels]
             if missing_strategy(state) == "complete_case" else [])
    return list(dict.fromkeys(needed)), preds, gappy


def landmark_of(state: Any) -> tuple[str, float] | None:
    """A time-to-event outcome's landmark, ``(time column, L)``, or None (``SetFollowUp``)."""
    spec = getattr(state, "follow_up", None)
    if spec is None or getattr(state, "task", None) != "time_to_event":
        return None
    at = getattr(spec, "landmark", None)
    return None if at is None else (str(spec.time_column), float(at))


def repair_rules(state: Any) -> list[Any]:
    """Range rules the applied repairs add to the flow (rows with an impossible value)."""
    from turbotab.core.repairs import exclusion_rules

    return exclusion_rules(state)


def compute_cohort(
    store: Any,
    state: Any,
    ingest: Mapping[str, Any],
    row_ids: Any | None = None,
    measured: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], Any, list[str]]:
    """Steps, kept row ids and predictors, reading only the columns the flow needs.

    ``measured``, when given a list, receives the ids of the rows whose outcome is recorded.
    """
    needed, preds, gappy = cohort_inputs(state, ingest)
    frame = store.materialize(needed, row_ids)
    if measured is not None and state.target is not None:
        measured.append(frame.index[_measured(frame[state.target]).to_numpy()].to_numpy(dtype=np.int64))
    missing_frame = None
    if gappy:
        missing_frame = _missing_mask(store, gappy, frame.index)
    from turbotab.core.decisions import missing_strategy

    steps, kept = cohort_flow(
        frame,
        target=state.target,
        rules=state.exclusions,
        missing=missing_strategy(state),
        predictor_columns=gappy,
        missing_frame=missing_frame,
        repairs=repair_rules(state),
        reference=ingest.get("reference_rows"),
        landmark=landmark_of(state),
    )
    return steps, kept, preds


def _missing_mask(store: Any, columns: Sequence[str], index: Any) -> Any:
    """A frame of the gappy predictors on these rows, read a block of columns at a time."""
    import pandas as pd

    ids = np.asarray(index, dtype=np.int64)
    parts = []
    for start in range(0, len(columns), CHUNK_COLUMNS):
        block = store.materialize(list(columns[start:start + CHUNK_COLUMNS]), ids)
        parts.append(block.notna())  # booleans: an eighth of the memory, and all we need
    mask = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
    return mask.where(mask, np.nan)  # absent -> NaN, so notna() in the flow reads it back


COMPARED_PREDICTORS = 60  # predictors the complete-case comparison reads (the widest tables)
COMPARED_ROWS = 50_000  # rows on each side of it, drawn evenly when there are more


def complete_case_loss(store: Any, state: Any, ingest: Mapping[str, Any], kept: Any,
                       preds: Sequence[str]) -> dict[str, Any] | None:
    """The rows complete cases dropped beside the rows they kept (``methods.missing.row_loss``):
    the outcome and the predictors, so a reader sees whether the kept rows are a different group
    (audit E14; CLINICAL_SURVEY_PACK: "Flag when listwise deletion drops >10% of rows")."""
    from turbotab.core.methods.missing import row_loss

    _, before, _ = compute_cohort(store, state.model_copy(update={"missing": None}), ingest)
    dropped = np.setdiff1d(np.asarray(before, dtype=np.int64), np.asarray(kept, dtype=np.int64))
    if not len(dropped):
        return None

    def even(ids: Any) -> Any:
        ids = np.asarray(ids, dtype=np.int64)
        return ids if len(ids) <= COMPARED_ROWS else ids[np.linspace(0, len(ids) - 1, COMPARED_ROWS).astype(int)]

    columns = [c for c in [state.target, *list(preds)[:COMPARED_PREDICTORS]] if c]
    loss = row_loss(store.materialize(columns, even(kept)), store.materialize(columns, even(dropped)),
                    state.target, list(preds)[:COMPARED_PREDICTORS])
    if loss is not None:  # the counts are the cohort's, whatever the comparison read
        loss.update(n_before=int(len(before)), n_kept=int(len(kept)), n_dropped=int(len(dropped)),
                    share=round(len(dropped) / len(before), 4))
    return loss


def cohort_stage(ctx: StageContext) -> Bundle:
    import pandas as pd

    ctx.progress(0.05, "Reading the outcome and the columns the rules use")
    from turbotab.core.decisions import missing_strategy
    from turbotab.core.stages.working import table_info

    ingest = table_info(ctx)
    measured: list[Any] = []
    loss = None
    with open_store(ctx) as store:
        steps, kept, preds = compute_cohort(store, ctx.state, ingest, measured=measured)
        cc = next((st for st in steps if st["key"] == "complete_cases"), None)
        if missing_strategy(ctx.state) == "complete_case" and cc is not None and cc["dropped"]:
            ctx.progress(0.7, "Comparing the rows complete cases drop with the rows they keep")
            loss = complete_case_loss(store, ctx.state, ingest, kept, preds)
    ctx.progress(0.9, "Counting who is left")
    data = {"steps": steps, "n_final": int(len(kept)), "predictors": preds,
            "complete_case_loss": loss}
    # "measured" (every row with an outcome) is what the split draws its held-out rows over.
    return Bundle(data=data, frames={"rows": pd.DataFrame({"row_id": kept}),
                                     "measured": pd.DataFrame({"row_id": measured[0]})})


# ── split ─────────────────────────────────────────────────────────────────────


def _feasible_stratify(labels: Any, test_fraction: float) -> bool:
    import pandas as pd

    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return False
    n = int(counts.sum())
    n_test = math.ceil(test_fraction * n)
    return n_test >= len(counts) and n - n_test >= len(counts)


def _draw_holdout(
    n: int, y: Any | None, g: Any | None, holdout: float, seed: int, notes: list[str]
) -> tuple[Any, bool]:
    """The held-out mask over ``n`` sorted rows (labels ``y``, groups ``g``), and whether it is stratified."""
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    stratified = y is not None
    mask = np.zeros(n, dtype=bool)
    if holdout <= 0:
        return mask, stratified
    if n < 2:
        notes.append("Too few rows to hold any out.")
        return mask, stratified
    if g is not None:
        if len(pd.unique(g)) < 2:
            notes.append("There is only one group, so nothing could be held out.")
            return mask, stratified
        chosen = None
        if stratified:
            label = pd.DataFrame({"g": g, "y": y}).groupby("g", sort=True)["y"].agg(lambda s: s.mode().iloc[0])
            if _feasible_stratify(label.to_numpy(), holdout):
                _, test_groups = train_test_split(label.index.to_numpy(), test_size=holdout,
                                                  random_state=seed, stratify=label.to_numpy())
                chosen = test_groups
            else:
                stratified = False
                notes.append("Some classes are too rare to stratify the held-out rows.")
        if chosen is None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
            _, test_idx = next(splitter.split(np.arange(n), groups=g))
            chosen = pd.unique(g[test_idx])
        return np.isin(g, np.asarray(list(chosen), dtype=object)), stratified
    if stratified and not _feasible_stratify(y, holdout):
        stratified = False
        notes.append("Some classes are too rare to stratify the held-out rows.")
    _, test_idx = train_test_split(np.arange(n), test_size=holdout, random_state=seed,
                                   stratify=y if stratified else None)
    mask[test_idx] = True
    return mask, stratified


def _assign_folds(
    n_train: int, y: Any | None, g: Any | None, folds: int, seed: int, stratified: bool, notes: list[str],
    order: Any | None = None,
) -> tuple[Any, int]:
    """Fold numbers for ``n_train`` training rows: grouped when ``g``, stratified when asked.

    With ``order`` (each row's unit rank in time) the folds are time-ordered blocks of whole units
    instead (:func:`turbotab.core.models.folds.forward_blocks`): fold 0 only trains, and fold j is
    scored by a model fit on folds 0 … j − 1; every block holds every class of ``y``. The count
    returned is the folds scored.
    """
    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    fold = np.zeros(n_train, dtype=np.int64)
    if n_train < 2:
        notes.append("Too few training rows for cross-validation.")
        return fold, 1 if n_train else 0
    if order is not None:
        from turbotab.core.models.folds import forward_blocks

        units = g if g is not None else np.arange(n_train)
        blocks, made, said = forward_blocks(order, units, int(folds) + 1, labels=y)
        notes.extend(said)
        if made < int(folds) + 1 and not said:
            notes.append(f"Only `{made}` training units, so `{max(1, made - 1)}` time-ordered folds "
                         f"instead of `{folds}`.")
        return blocks, max(1, made - 1)
    units = len(pd.unique(g)) if g is not None else n_train
    k = max(2, min(int(folds), units))
    if k < folds:
        notes.append(f"Only `{units}` training units, so `{k}` folds instead of `{folds}`.")
    idx = np.arange(n_train)
    plain = KFold(k, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a rare class warns; the note says so instead
        if g is not None and units >= 2:
            if stratified and y is not None:
                parts = StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, y, groups=g)
            else:
                parts = GroupKFold(k, shuffle=True, random_state=seed).split(idx, groups=g)
        elif stratified and y is not None and pd.Series(y).value_counts().max() >= k:
            parts = StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y)
        else:
            if stratified:
                notes.append("The folds could not be stratified, so they are plain random folds.")
            parts = plain.split(idx)
        try:
            for f, (_, test_idx) in enumerate(parts):
                fold[test_idx] = f
        except ValueError:
            for f, (_, test_idx) in enumerate(plain.split(idx)):
                fold[test_idx] = f
            notes.append("The folds could not be stratified, so they are plain random folds.")
    return fold, k


NOT_RECORDED = "(not recorded)"  # a cluster level of its own: rows whose cluster is missing


def _cluster_key(value: Any) -> str:
    return NOT_RECORDED if _isna(value) else _level_key(value)


def _cluster_folds(levels: Any, units: Any | None, column: str | None,
                   notes: list[str]) -> tuple[Any, list[str]]:
    """One fold per cluster level (sorted), and the levels in fold order. A unit whose rows fall in
    more than one cluster is counted in a note: its rows then sit in different folds."""
    import pandas as pd

    labels = np.asarray(levels, dtype=object)
    names = sorted(pd.unique(labels).tolist(), key=lambda v: (v == NOT_RECORDED, str(v)))
    index = {name: i for i, name in enumerate(names)}
    folds = np.asarray([index[v] for v in labels.tolist()], dtype=np.int64)
    if NOT_RECORDED in index:
        notes.append(f"`{int((labels == NOT_RECORDED).sum()):,}` training rows have no "
                     f"`{column}`; they are scored as a cluster of their own, {NOT_RECORDED}.")
    if units is not None and len(labels):
        spread = pd.DataFrame({"u": np.asarray(units, dtype=object), "c": labels}).groupby("u")["c"].nunique()
        crossing = int((spread > 1).sum())
        if crossing:
            notes.append(f"`{crossing:,}` units have rows in more than one level of `{column}`; "
                         f"those rows sit in different folds.")
    if len(names) < 2:
        notes.append(f"`{column}` has one level among the training rows, so no cluster can be "
                     f"held out.")
    return folds, [str(n) for n in names]


def draw_split(
    row_ids: Any,
    *,
    holdout: float,
    seed: int,
    folds: int,
    y: Any | None = None,
    groups: Any | None = None,
    grouped_by: str | None = None,
    universe: Any | None = None,
    held: Any | None = None,
    order: Any | None = None,
    repeats: int = 1,
    clusters: Any | None = None,
    cluster_column: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Assign each row of ``row_ids`` to ``train`` or ``holdout``, and each training row a fold.

    ``repeats`` > 1 draws that many fold assignments (repeated k-fold, audit ME-11): ``fold`` is the
    first, ``fold_r1`` … the others, each drawn as the first is with seeds ``seed + r``.
    ``clusters`` (aligned like ``groups``) makes each level of ``cluster_column`` a fold of its own
    (internal–external validation, audit E16); ``fold_labels`` in the facts name them in order.

    The held-out rows are drawn over ``universe`` (default: ``row_ids``) and ``row_ids`` keeps
    the ones that fall in it. The split stage passes every row whose outcome is measured, so an
    exclusion or a missing-values answer changes who is analyzed but never moves a row across
    the seal. ``y`` (classification labels) asks for stratification and ``groups`` keeps each
    group on one side and in one fold; both align with ``universe`` when it is given, else with
    ``row_ids``. Rows are sorted by id first: the same rows, seed and settings always give the
    same split. Returns the assignment frame (``row_id``, ``partition``, ``fold``; fold -1 when
    held out) and the facts the split artifact reports, plus ``sealed``: every row of the
    universe drawn to be held out, in or out of ``row_ids`` — the rows nothing may read.
    ``held`` (aligned like ``groups``) is a held-out mask drawn elsewhere — the seal's
    chronological draw (``turbotab/core/seal.py``) — used in place of a random draw. ``order``
    (aligned like ``groups``: each row's unit rank in time, ``seal.time_order``) makes the folds
    time-ordered: forward chaining by whole unit (audit MA-11), with fold stratification decided
    on its own, never inherited from how the held-out rows were drawn (A17).
    """
    import pandas as pd

    ranks = order  # each row's unit rank in time, aligned like ``groups``
    base = np.asarray(row_ids if universe is None else universe, dtype=np.int64)
    order = np.argsort(base, kind="stable")
    uids = base[order]

    def keyed(values: Any | None, key: Any = _level_key) -> Any | None:
        if values is None:
            return None
        return np.array([key(v) for v in np.asarray(values, dtype=object)[order]], dtype=object)

    y_u, g_u = keyed(y), keyed(groups, _group_key)
    order_u = None if ranks is None else np.asarray(ranks, dtype=float)[order]
    notes: list[str] = []
    if held is not None:
        held_u, stratified = np.asarray(held, dtype=bool)[order], False
    else:
        held_u, stratified = _draw_holdout(len(uids), y_u, g_u, holdout, seed, notes)

    rows = np.sort(np.asarray(row_ids, dtype=np.int64))
    pos = np.clip(np.searchsorted(uids, rows), 0, max(0, len(uids) - 1))
    inside = (pos < len(uids)) & (uids[pos] == rows) if len(uids) else np.zeros(len(rows), dtype=bool)
    held = np.where(inside, held_u[pos] if len(uids) else False, False)
    y_r = None if y_u is None else np.where(inside, y_u[pos], "__unknown__")
    g_r = None if g_u is None else np.where(inside, g_u[pos], np.array([f"__row_{r}" for r in rows], dtype=object))
    o_r = None if order_u is None else np.where(inside, order_u[pos] if len(uids) else np.nan, np.nan)

    train = np.flatnonzero(~held)
    fold = np.full(len(rows), -1, dtype=np.int64)
    fold_labels: list[str] | None = None
    extra: dict[str, Any] = {}
    if clusters is not None:
        c_u = keyed(clusters, _cluster_key)
        c_r = np.where(inside, c_u[pos] if len(uids) else NOT_RECORDED, NOT_RECORDED)
        train_folds, fold_labels = _cluster_folds(
            c_r[train], None if g_r is None else g_r[train], cluster_column, notes)
        k = len(fold_labels)
        if o_r is not None:
            notes.append(f"The folds are the levels of `{cluster_column}`, not time-ordered blocks.")
            o_r = None
    else:
        # Folds are stratified whenever there are classes to keep in proportion, however the
        # held-out rows were drawn (a chronological draw cannot be stratified; its folds can be).
        train_folds, k = _assign_folds(
            len(train), None if y_r is None else y_r[train], None if g_r is None else g_r[train],
            folds, seed, y_r is not None, notes, order=None if o_r is None else o_r[train])
        if repeats > 1 and o_r is not None:
            notes.append("Time-ordered folds are the same in every repeat, so they run once.")
        elif repeats > 1:
            for r in range(1, int(repeats)):
                column = np.full(len(rows), -1, dtype=np.int64)
                column[train], _ = _assign_folds(
                    len(train), None if y_r is None else y_r[train],
                    None if g_r is None else g_r[train], folds, seed + r, y_r is not None, [])
                extra[f"fold_r{r}"] = column
    fold[train] = train_folds

    frame = pd.DataFrame({"row_id": rows, "partition": np.where(held, "holdout", "train"), "fold": fold})
    for name, column in extra.items():
        frame[name] = column
    if o_r is not None:
        frame["order"] = o_r  # each row's unit rank in time: the inner splits follow it too
    info = {
        "n_train": int(len(train)),
        "n_holdout": int(held.sum()),
        "holdout": float(holdout),
        "seed": int(seed),
        "folds": int(k),
        "grouped_by": grouped_by if g_r is not None else None,
        "n_groups": int(len(pd.unique(g_r))) if g_r is not None else None,
        "stratified": bool(stratified),
        "fold_scheme": "time_ordered" if o_r is not None else "random",
        "folds_stratified": bool(y_r is not None and o_r is None and clusters is None
                                 and not any("could not be stratified" in t for t in notes)),
        "repeats": 1 + len(extra),
        "cluster": cluster_column if clusters is not None else None,
        "fold_labels": fold_labels,
        "notes": notes,
        "sealed": uids[held_u],
    }
    return frame, info


def split_note(info: Mapping[str, Any], unit_rows: int | None = None) -> str:
    """One plain sentence on how the rows were divided."""
    parts: list[str] = []
    if info["n_holdout"]:
        parts.append(f"`{info['n_holdout']:,}` rows are held out and `{info['n_train']:,}` train")
    else:
        parts.append(f"All `{info['n_train']:,}` rows train, checked by cross-validation")
    if info["grouped_by"]:
        parts.append(f"grouped by `{info['grouped_by']}` so a unit's rows stay together")
    if info["stratified"]:
        parts.append("with the outcome's classes in proportion")
    if info.get("cluster"):
        sentence = (", ".join(parts) + f"; `{info['folds']}` folds, one per level of "
                    f"`{info['cluster']}`, each scored by models fit on the others.")
    elif info.get("fold_scheme") == "time_ordered":
        sentence = (", ".join(parts) + f"; `{info['folds']}` time-ordered folds, each scored by models "
                    f"fit on the units before it.")
    elif int(info.get("repeats") or 1) > 1:
        sentence = ", ".join(parts) + f"; `{info['folds']}` folds, drawn `{info['repeats']}` times."
    else:
        sentence = ", ".join(parts) + f"; `{info['folds']}` folds."
    extra = " ".join(info.get("notes") or [])
    return f"{sentence} {extra}".strip()


def split_inputs(state: Any, cohort_rows: Any, store: Any, task: str | None) -> dict[str, Any]:
    """The labels and groups ``draw_split`` needs for these cohort rows."""
    ids = np.asarray(cohort_rows, dtype=np.int64)
    out: dict[str, Any] = {"y": None, "groups": None, "grouped_by": None}
    # BLUEPRINT §14.3: the groups are a settled cluster reading only, which the user alone settles
    # (the grain's unit, or an identifier the user confirmed while the grain names no other); an
    # identifier that rode along unconfirmed, or one the grain answer passes over, groups nothing.
    from turbotab.core.readings import cluster_rank, cluster_reading

    identifiers = [c for c, r in (state.roles or {}).items() if r == "identifier"
                   and cluster_reading(state, c).settled and cluster_reading(state, c).value == "yes"]
    columns = list(identifiers)
    # Classes, levels or a time-to-event outcome's events are kept in proportion by the folds.
    classify = task in ("binary", "multiclass", "ordinal", "time_to_event")
    if classify:
        columns.append(state.target)
    if not columns or not len(ids):
        return out
    frame = store.materialize(list(dict.fromkeys(columns)), ids)
    best: tuple[int, int, str] | None = None
    for col in identifiers:
        values = frame[col]
        n_units = int(values.nunique(dropna=True))
        rank = (cluster_rank(state, col), n_units, col)
        if 0 < n_units < int(values.notna().sum()) and (best is None or rank < best):
            best = rank
    if best is not None:
        col = best[2]
        values = frame[col].astype(object)
        # A missing identifier is its own unit per row: it cannot be matched to anyone.
        filled = [v if not _isna(v) else f"__missing_{rid}" for rid, v in zip(ids, values)]
        out["groups"], out["grouped_by"] = filled, col
    if classify:
        out["y"] = frame[state.target].to_numpy(dtype=object)
    return out


def _isna(value: Any) -> bool:
    try:
        return bool(value is None or (isinstance(value, float) and math.isnan(value)))
    except TypeError:
        return False


def split_stage(ctx: StageContext) -> Bundle:
    """Held-out rows drawn over every row with an outcome; folds over the cohort's training rows."""
    import pandas as pd

    cohort = ctx.inputs["cohort"]
    rows = cohort.frames["rows"]["row_id"].to_numpy(dtype=np.int64)
    measured = cohort.frames.get("measured")
    universe = rows if measured is None else measured["row_id"].to_numpy(dtype=np.int64)
    task = ctx.inputs["target_info"].get("task")
    spec = ctx.state.split
    ctx.progress(0.1, "Reading the identifier and the outcome")
    from turbotab.core.seal import seal_inputs  # the basis and the chronological draw (M2 §3)

    structure = ctx.inputs.get("structure")  # the grain stated from a unique identifier
    with open_store(ctx) as store:
        seal = seal_inputs(ctx.state, universe, store, task, holdout=spec.holdout, seed=spec.seed,
                           structure=getattr(structure, "data", structure))
    if seal.refusal:
        raise ValueError(seal.refusal)
    validation = getattr(spec, "validation", "kfold")
    clusters = None
    if validation == "internal_external":
        with open_store(ctx) as store:
            if spec.cluster not in store.columns:
                raise ValueError(f"The table has no column `{spec.cluster}` to fold by.")
            clusters = store.materialize([spec.cluster], universe)[spec.cluster].to_numpy(dtype=object)
    ctx.progress(0.4, "Drawing the held-out rows and the folds")
    frame, info = draw_split(
        rows, holdout=spec.holdout, seed=spec.seed, folds=spec.folds, universe=universe,
        repeats=int(spec.repeats) if validation == "repeated_kfold" else 1,
        clusters=clusters, cluster_column=spec.cluster if clusters is not None else None,
        **seal.split_args(),
    )
    info["validation"] = validation
    info["n_boot"] = int(spec.n_boot) if validation == "bootstrap" else None
    notes = info.pop("notes")
    sealed = info.pop("sealed")
    if seal.chronology is not None:
        notes.append(seal.chronology.sentence)
    data = {**info, **seal.facts(), "note": split_note({**info, "notes": notes})}
    # "sealed" holds every held-out row, including ones the cohort now excludes: an exclusion
    # relaxed later brings them back held out, so previews must never read them either.
    return Bundle(data=data, frames={"assignment": frame, "sealed": pd.DataFrame({"row_id": sealed})})


__all__ = [
    "CATEGORY_LEVELS", "NHANES_CODED", "PREDICTOR_ROLES", "categorical_proposals", "cohort_flow",
    "cohort_inputs", "cohort_stage", "compute_cohort", "draw_split", "energy_bearing",
    "intake_checks", "predictors",
    "propose_roles", "roles_stage", "rule_drops", "rule_keep", "rule_label", "rule_parts",
    "split_inputs", "split_note", "split_stage",
]

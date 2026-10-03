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


def _intake_by_unit(name: str) -> bool:
    """An amount in an intake unit (``_g``, ``_mg``, a share of energy) that names no specimen,
    score or concentration: a food or nutrient intake under the dietary lens (``fatty_fish_g``)."""
    from turbotab.core.methods.energy import unit_of
    from turbotab.core.recognizers import concentration_unit, tokens

    if unit_of(name) not in _INTAKE_UNITS or concentration_unit(name):
        return False
    words = set(tokens(name))
    return not words & {"serum", "plasma", "blood", "urine", "urinary", "score", "index", "mass",
                        "body", "dose", "meds", "medication"}


def _design_role(name: str, summary: Mapping[str, Any], exact: set[str],
                 design_in_table: bool = False) -> str | None:
    """A reason when this column is part of a survey design, else None.

    A sampling weight is read by :func:`turbotab.core.recognizers.reads_as_survey_weight`: an
    NHANES weight by name, survey vocabulary, or a bare ``weight`` far above any body weight in a
    table that names its design exactly; never a birth or body weight (audit IN-10)."""
    from turbotab.core.recognizers import reads_as_survey_weight

    upper = str(name).upper()
    if upper in exact or upper.startswith(("SDMV",)):
        return "An NHANES survey design variable: a weight, stratum or sampling unit."
    median = summary.get("median")
    if reads_as_survey_weight(name, median=None if median is None else float(median),
                              design_in_table=design_in_table):
        if upper.startswith(("WTDR", "WTMEC", "WTINT", "WTSA", "WTSB")):
            return "An NHANES survey design variable: a weight, stratum or sampling unit."
        return "A sampling weight: how each row was sampled, not a body measurement."
    tokens = set(_tokens(name))
    if tokens & _SURVEY_STRATA and not tokens & {"id", "ids"}:
        return "Names a sampling stratum or cluster of the survey design."
    return None


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
) -> list[dict[str, Any]]:
    """One proposal per column (the outcome and the row identity excepted), in table order.

    ``columns`` are the profile's column summaries (``name``, ``dtype``, ``n``, ``n_missing``,
    ``n_unique``, ``median``…). ``energy_column`` is the total-energy column the recognizer reads;
    ``acquisition`` the batch, plate and run-order columns (:func:`_acquisition_columns`);
    ``purpose`` the declared purpose, which sets an acquisition column's default; ``fractional``
    the identifier-named columns whose values have fractional parts (measurements, never IDs).

    Every name is read by :mod:`turbotab.core.recognizers` (audit WP13): whole words, never
    substrings; a study's arms and groups are exposures; a site or household is a cluster.
    """
    from turbotab import nutrition
    from turbotab.core.recognizers import acquisition_kind, id_kind, is_rate, reads_as_time, study_group

    lenses = [k for k in (lens or []) if k in LENSES]
    dietary = "dietary" in lenses
    omics = any(k in OMICS for k in lenses)
    exact_design = {c.upper() for c in getattr(nutrition, "EXACT_DESIGN_NAMES", ())}
    design_in_table = any(str(c["name"]).upper() in exact_design for c in columns)
    acquisition = set(acquisition) | {str(c["name"]) for c in columns
                                      if acquisition_kind(c["name"]) is not None}
    fractional = set(fractional)
    by_lower = {str(c["name"]).lower(): str(c["name"]) for c in columns}
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
        proposal = {"column": name, "linked_to": None, "unit": _unit(name), "nested_in": None,
                    "kind": None}

        def put(role: str, confidence: str, reason: str, **extra: Any) -> None:
            out.append({**proposal, "proposed": role, "confidence": confidence, "reason": reason, **extra})

        is_flag, base = _flag_base(name, by_lower)
        design = _design_role(name, summary, exact_design, design_in_table)
        kind = None if name in fractional else id_kind(name)
        if is_flag:
            if base:
                put("flag", "high", f"Marks which values of `{base}` were filled in, not measured.",
                    linked_to=base)
            else:
                put("flag", "medium", "Named like a flag that marks other values; not a measurement.")
        elif design is not None:
            put("design", "high" if str(name).upper() in exact_design else "medium", design)
        elif name in acquisition:
            role, reason = acquisition_proposal(purpose)
            put(role, "medium", reason, kind="acquisition")
        elif _norm(name) == "seqn":
            put("identifier", "high", "`SEQN` is the NHANES respondent number; it names people, not traits.")
        elif kind == "subject":
            put("identifier", "high", "Named like a participant's identifier; it names people, not traits.")
        elif kind == "record":
            put("identifier", "high", "Named like an identifier; it names rows or samples, not traits.")
        elif kind == "cluster" and unique:
            put("identifier", "high", "Names a group, such as a household, and is unique on every row.")
        elif kind == "cluster":
            put("cluster", "medium", "Groups participants, such as a site or household; not a trait.")
        elif kind == "visit" and unique:
            put("identifier", "medium", "Names each record, such as a visit or encounter.")
        elif kind == "visit":
            put("time", "medium", "A visit or encounter index: when a row was measured.")
        elif n_unique <= 1:
            put("excluded", "high", "Every row holds the same value, so it cannot explain anything.")
        elif not numeric and dtype in ("categorical", "text") and n_present and n_unique >= 0.95 * n_present:
            put("identifier", "medium", f"`{n_unique:,}` different values in `{n_present:,}` rows: it names rows.")
        elif energy_column is not None and name == energy_column:
            put("energy", "high" if dietary else "medium", "Total energy intake, the column energy adjustment works against.")
        elif dtype == "datetime":
            put("time", "high", "Holds dates or times: when a row was recorded.")
        elif numeric and dietary and _is_nutrient(name):
            carries = energy_bearing(name)
            reason = ("A nutrient that carries energy: an exposure under the dietary lens."
                      if carries else "A nutrient intake: an exposure under the dietary lens.")
            put("exposure", "high", reason)
        elif numeric and dietary and _intake_by_unit(name):
            put("exposure", "medium", "An intake by its unit, not a nutrient: an exposure under the "
                                      "dietary lens.")
        elif study_group(name):
            put("exposure", "medium", "A study arm or group: what the study compares, an exposure.")
        elif tokens & _FASTING_TOKENS:
            put("covariate", "medium", "Fasting status: a known confounder, adjusted for rather than studied.")
        elif reads_as_time(name):
            put("time", "medium", "Named like a time: a date, year, cycle, visit or recall.")
        elif tokens & _COVARIATE_TOKENS:
            put("covariate", "high" if tokens & {"age", "sex", "gender", "bmi"} else "medium",
                "A person's characteristic, usually adjusted for rather than studied.")
        elif dtype == "text":
            put("excluded", "high", f"Free text with `{n_unique:,}` different values; models cannot use it as is.")
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
    """The total-energy column, read from names and dtypes only by the one recognizer every stage
    shares (``stages.proposals.energy_column``: ``DR2TKCAL``, ``ENERC_KCAL``, ``TotalKcal`` and
    ``energy_kj`` alike; never a macronutrient's own kcal such as ``alc_kcal``)."""
    from turbotab.core.stages.proposals import energy_column

    try:
        return energy_column({str(c["name"]): c for c in columns}, {})
    except Exception:  # a recognizer failure must not fail the stage
        return None


def _acquisition_columns(columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Batch, plate, well and run-order columns, by whole names
    (:func:`turbotab.core.recognizers.acquisition_kind`). The legacy reader also claimed a study's
    arms, groups, phenotype, fasting status and site, and dropped them from the model as "an
    acquisition column" (audit IN-02); those are read on their own now."""
    from turbotab.core.recognizers import acquisition_kind

    return [str(c["name"]) for c in columns if acquisition_kind(c["name"]) is not None]


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
    """The identifier that repeats (several rows per unit), with its counts, or None."""
    best: dict[str, Any] | None = None
    for p in proposals:
        if p["proposed"] != "identifier":
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
    """Child -> parent over every row: the names admit it and the part never exceeds the total."""
    from turbotab.core.methods.nesting import candidates, nested_components

    names = [str(c["name"]) for c in columns if str(c["name"]) != target
             and str(c.get("dtype") or "") in ("numeric", "integer")]
    pairs = candidates(names)
    needed = list(dict.fromkeys([*pairs, *(c for kids in pairs.values() for c in kids)]))
    if not needed:
        return {}
    return nested_components(store.materialize(needed), needed)


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
    with open_store(ctx) as store:
        fractional = _fractional_identifiers(store, columns)
    proposals = propose_roles(
        columns,
        lens=ctx.state.lens,
        target=ctx.state.target,
        n_rows=n_rows,
        energy_column=_energy_reading(columns),
        acquisition=_acquisition_columns(columns),
        purpose=ctx.state.purpose,
        fractional=fractional,
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
    if repeats is not None:
        for p in proposals:
            if p["column"] == repeats["column"]:
                p["reason"] = (f"Names each unit; `{repeats['n_units']:,}` units, up to "
                               f"`{repeats['max_rows_per_unit']}` rows each.")
                p["confidence"] = "high"
    return {"columns": proposals, "repeats": repeats,
            "categorical": categorical_proposals(columns, proposals)}


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
) -> tuple[list[dict[str, Any]], Any]:
    """The participant flow over ``frame`` (indexed by row id): its steps and the kept row ids.

    ``frame`` holds the target and every column the rules read (a preview may run before an
    outcome is chosen: ``target=None`` leaves out ``outcome_measured``). Complete cases are judged on
    ``predictor_columns`` in ``missing_frame`` (default: ``frame``); columns absent from it are
    taken to have no missing values (the caller leaves them out when the profile says so).
    ``repairs`` are range rules from applied repairs (impossible values, ``repairs.exclusion_rules``):
    steps ``repair:<i>`` after the eligibility answer's own.
    """
    steps: list[dict[str, Any]] = []
    n = len(frame) if n_loaded is None else int(n_loaded)
    steps.append({"key": "loaded", "label": "Rows in the table", "n": n, "dropped": 0,
                  "reason": None, "decision_id": None})
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
    preds = predictors(state.roles, order, drop=left_out(state))
    needed = [state.target] if state.target is not None else []
    for rule in [*(state.exclusions or []), *repair_rules(state)]:
        needed.extend(_as_rule(rule).reads())
    with_missing = _columns_with_missing(ingest)
    levels = set(level_columns(state, preds, {str(c["name"]): c for c in ingest.get("columns", [])}))
    gappy = ([c for c in preds if c in with_missing and c not in levels]
             if missing_strategy(state) == "complete_case" else [])
    return list(dict.fromkeys(needed)), preds, gappy


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
    identifiers = [c for c, r in (state.roles or {}).items() if r == "identifier"]
    columns = list(identifiers)
    # Classes, levels or a time-to-event outcome's events are kept in proportion by the folds.
    classify = task in ("binary", "multiclass", "ordinal", "time_to_event")
    if classify:
        columns.append(state.target)
    if not columns or not len(ids):
        return out
    frame = store.materialize(list(dict.fromkeys(columns)), ids)
    best: tuple[int, str] | None = None
    for col in identifiers:
        values = frame[col]
        n_units = int(values.nunique(dropna=True))
        if 0 < n_units < int(values.notna().sum()) and (best is None or n_units < best[0]):
            best = (n_units, col)
    if best is not None:
        col = best[1]
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
    "cohort_inputs", "cohort_stage", "compute_cohort", "draw_split", "energy_bearing", "predictors",
    "propose_roles", "roles_stage", "rule_drops", "rule_keep", "rule_label", "rule_parts",
    "split_inputs", "split_note", "split_stage",
]

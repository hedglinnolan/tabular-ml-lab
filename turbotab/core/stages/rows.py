"""M1 row-side stages: column roles, the cohort (participant flow), the split.

Owner: the M1 "rows" agent. Contract: docs/turbotab-next/M1_CONTRACT.md §3.

* ``roles`` proposes a role for every column but the outcome, with a plain reason. It reads
  names and the profile's column summaries, and the legacy recognizers where they fit
  (``turbotab.nutrition`` for energy, nutrients and survey design; ``turbotab.core.methods.energy``
  for energy-bearing macronutrients; ``turbotab.packs.design_columns`` for acquisition columns).
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

_PERSON = {
    "subject", "subj", "participant", "patient", "person", "respondent", "sample", "record",
    "case", "household", "family", "member", "study", "individual", "pid", "sid",
}
_ID_TAILS = {"id", "ids", "uid", "uuid", "guid", "seqn"}
_NUMBER_TAILS = {"no", "num", "number", "code", "key"}
_TIME_TOKENS = {
    "date", "time", "datetime", "timestamp", "year", "yr", "cycle", "visit", "wave", "day",
    "month", "week", "period", "recall", "round", "occasion", "followup", "baseline",
}
_COVARIATE_TOKENS = {
    "age", "sex", "gender", "bmi", "race", "ethnicity", "ethnic", "education", "educ", "income",
    "pir", "poverty", "smoking", "smoker", "smoke", "marital", "height", "weight", "waist", "hip",
    "activity", "exercise", "met", "mets", "meds", "medication", "medications", "bp", "sbp", "dbp",
    "systolic", "diastolic", "diabetes", "hypertension", "region", "country", "occupation",
    "insurance", "parity", "menopause", "pregnant", "pregnancy",
}
_NUTRIENT_TOKENS = {
    "sugar", "sugars", "fiber", "fibre", "sodium", "potassium", "calcium", "iron", "zinc",
    "magnesium", "phosphorus", "selenium", "copper", "cholesterol", "caffeine", "folate",
    "folic", "niacin", "thiamin", "riboflavin", "retinol", "carotene", "vitamin", "vit", "sfa",
    "mufa", "pufa", "omega", "starch", "water", "moisture", "theobromine", "lycopene", "lutein",
    "choline", "iodine", "b12", "b6", "epa", "dha", "ala", "la",
}
_NUTRIENT_UNITS = re.compile(
    r".*(_g|_gram|_grams|_mg|_mcg|_ug|_iu|_pct_kcal|_pct_energy|_percent_energy|_per1000kcal"
    r"|_per_1000_kcal)$"
)
_FLAG_PREFIX = ("imputed_", "is_imputed_", "flag_", "imp_")
_FLAG_SUFFIX = ("_imputed", "_flag", "_flagged", "_imp")
_UNIT_LABELS = {
    "grams": "g", "milligrams": "mg", "micrograms": "µg", "IU": "IU", "kcal": "kcal",
    "kj": "kJ", "density": "share of energy",
}


def _id_like(name: str) -> bool:
    norm = _norm(name)
    tokens = norm.split("_")
    if norm in _ID_TAILS or tokens[-1] in _ID_TAILS or tokens[0] == "id":
        return True
    if tokens[-1] in _NUMBER_TAILS and any(t in _PERSON for t in tokens[:-1]):
        return True
    return len(tokens) == 1 and norm.endswith("id") and norm[:-2] in _PERSON


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
    from turbotab import nutrition

    tokens = set(_tokens(name))
    if tokens & _NUTRIENT_TOKENS:
        return True
    try:
        from turbotab.core.methods.energy import nutrient_role

        if nutrient_role(name) is not None:
            return True
    except ValueError:
        return True  # names two macronutrients: still a nutrient
    if nutrition.nutrient_columns([name]):
        return True
    return bool(_NUTRIENT_UNITS.fullmatch(_norm(name)))


def _design_role(name: str, summary: Mapping[str, Any], exact: set[str]) -> str | None:
    """A reason when this column is part of a survey or study design, else None."""
    upper = str(name).upper()
    if upper in exact or upper.startswith(("WTDR", "WTMEC", "WTINT", "WTSA", "WTSB", "SDMV")):
        return "An NHANES survey design variable: a weight, stratum or sampling unit."
    tokens = set(_tokens(name))
    if tokens & {"strata", "stratum", "psu", "cluster", "sampling_weight"}:
        return "Names a sampling stratum or cluster of the survey design."
    if tokens & {"weight", "weights", "wt"}:
        median = summary.get("median")
        survey = bool(tokens & {"survey", "sample", "sampling", "design"})
        if survey or (median is not None and float(median) > 1_000):
            return "A sampling weight: its values are far too large for body weight."
    return None


def propose_roles(
    columns: Sequence[Mapping[str, Any]],
    *,
    lens: Sequence[str] | None,
    target: str | None,
    n_rows: int,
    energy_column: str | None = None,
    acquisition: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """One proposal per column (the outcome and the row identity excepted), in table order.

    ``columns`` are the profile's column summaries (``name``, ``dtype``, ``n``, ``n_missing``,
    ``n_unique``, ``median``…). ``energy_column`` is the nutrition pack's reading of total
    energy; ``acquisition`` the columns ``packs.design_columns`` claims (batch, run order…).
    """
    from turbotab import nutrition

    lenses = [k for k in (lens or []) if k in LENSES]
    dietary = "dietary" in lenses
    omics = any(k in OMICS for k in lenses)
    exact_design = {c.upper() for c in getattr(nutrition, "EXACT_DESIGN_NAMES", ())}
    acquisition = set(acquisition)
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
        tokens = set(_tokens(name))
        proposal = {"column": name, "linked_to": None, "unit": _unit(name), "nested_in": None}

        def put(role: str, confidence: str, reason: str, **extra: Any) -> None:
            out.append({**proposal, "proposed": role, "confidence": confidence, "reason": reason, **extra})

        is_flag, base = _flag_base(name, by_lower)
        design = _design_role(name, summary, exact_design)
        if is_flag:
            if base:
                put("flag", "high", f"Marks which values of `{base}` were filled in, not measured.",
                    linked_to=base)
            else:
                put("flag", "medium", "Named like a flag that marks other values; not a measurement.")
        elif design is not None:
            put("design", "high" if str(name).upper() in exact_design else "medium", design)
        elif _norm(name) == "seqn":
            put("identifier", "high", "`SEQN` is the NHANES respondent number; it names people, not traits.")
        elif _id_like(name):
            put("identifier", "high", "Named like an identifier; it names rows or people, not traits.")
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
        elif tokens & _TIME_TOKENS and not tokens & _COVARIATE_TOKENS:
            put("time", "medium", "Named like a time: a date, year, cycle, visit or recall.")
        elif tokens & _COVARIATE_TOKENS:
            put("covariate", "high" if tokens & {"age", "sex", "gender", "bmi"} else "medium",
                "A person's characteristic, usually adjusted for rather than studied.")
        elif dtype == "text":
            put("excluded", "high", f"Free text with `{n_unique:,}` different values; models cannot use it as is.")
        elif name in acquisition:
            put("design", "medium", "An acquisition column (batch, plate or run order), not biology.")
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


def _energy_reading(columns: Sequence[Mapping[str, Any]]) -> str | None:
    """The nutrition pack's total-energy column, read from names and dtypes only."""
    import pandas as pd

    from turbotab import nutrition

    empty = pd.DataFrame({
        str(c["name"]): pd.Series(dtype=float if c.get("dtype") in ("numeric", "integer") else object)
        for c in columns
    })
    try:
        return nutrition._energy_column(empty)
    except Exception:  # a recognizer failure must not fail the stage
        return None


def _acquisition_columns(columns: Sequence[Mapping[str, Any]]) -> list[str]:
    import pandas as pd

    from turbotab import packs

    try:
        names = [str(c["name"]) for c in columns]
        # It reads names only, but returns nothing for an empty frame: give it one blank row.
        found = packs.design_columns(pd.DataFrame([[None] * len(names)], columns=names))
    except Exception:
        return []
    return [c for cols in found.values() for c in cols]


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
    columns = [c for c in ctx.inputs["profile"]["columns"] if not str(c["name"]).startswith("__")]
    n_rows = int(ctx.inputs["ingest"]["n_rows"])
    ctx.progress(0.1, "Reading column names and summaries")
    proposals = propose_roles(
        columns,
        lens=ctx.state.lens,
        target=ctx.state.target,
        n_rows=n_rows,
        energy_column=_energy_reading(columns),
        acquisition=_acquisition_columns(columns),
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
    return {"columns": proposals, "repeats": repeats}


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


def rule_keep(frame: Any, rule: Any) -> Any:
    """Boolean Series: True where ``rule`` keeps the row. Missing values are kept.

    A range keeps ``low <= value <= high`` (either bound may be open). With ``by``, a row whose
    level of ``by.column`` has its own range uses that range; other rows use the rule's own.
    """
    import pandas as pd

    rule = _as_rule(rule)
    values = pd.to_numeric(frame[rule.column], errors="coerce").astype(float)
    low = pd.Series(np.nan if rule.low is None else float(rule.low), index=frame.index, dtype=float)
    high = pd.Series(np.nan if rule.high is None else float(rule.high), index=frame.index, dtype=float)
    if rule.by is not None:
        levels = frame[rule.by.column].map(lambda v: None if pd.isna(v) else _level_key(v))
        for key, (lo, hi) in rule.by.ranges.items():
            at = levels == _level_key(key)
            low[at] = np.nan if lo is None else float(lo)
            high[at] = np.nan if hi is None else float(hi)
    inside = (low.isna() | (values >= low)) & (high.isna() | (values <= high))
    return values.isna() | inside


def _as_rule(rule: Any) -> Any:
    from turbotab.core.decisions import ExclusionRule

    return rule if isinstance(rule, ExclusionRule) else ExclusionRule.model_validate(rule)


def rule_label(rule: Any) -> str:
    rule = _as_rule(rule)
    col = f"`{rule.column}`"
    if rule.by is not None:
        return f"{col} within its range for each `{rule.by.column}`"
    if rule.low is not None and rule.high is not None:
        return f"{col} within `{_fmt_num(rule.low)}`–`{_fmt_num(rule.high)}`"
    if rule.low is not None:
        return f"{col} at least `{_fmt_num(rule.low)}`"
    return f"{col} at most `{_fmt_num(rule.high)}`"


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
        rule = _as_rule(rule)
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
        needed.append(rule.column)
        if rule.by is not None:
            needed.append(rule.by.column)
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


def cohort_stage(ctx: StageContext) -> Bundle:
    import pandas as pd

    ctx.progress(0.05, "Reading the outcome and the columns the rules use")
    ingest = ctx.inputs["ingest"]
    measured: list[Any] = []
    with open_store(ctx) as store:
        steps, kept, preds = compute_cohort(store, ctx.state, ingest, measured=measured)
    ctx.progress(0.9, "Counting who is left")
    data = {"steps": steps, "n_final": int(len(kept)), "predictors": preds}
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
    n_train: int, y: Any | None, g: Any | None, folds: int, seed: int, stratified: bool, notes: list[str]
) -> tuple[Any, int]:
    """Fold numbers for ``n_train`` training rows: grouped when ``g``, stratified when asked."""
    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    fold = np.zeros(n_train, dtype=np.int64)
    if n_train < 2:
        notes.append("Too few training rows for cross-validation.")
        return fold, 1 if n_train else 0
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
) -> tuple[Any, dict[str, Any]]:
    """Assign each row of ``row_ids`` to ``train`` or ``holdout``, and each training row a fold.

    The held-out rows are drawn over ``universe`` (default: ``row_ids``) and ``row_ids`` keeps
    the ones that fall in it. The split stage passes every row whose outcome is measured, so an
    exclusion or a missing-values answer changes who is analyzed but never moves a row across
    the seal. ``y`` (classification labels) asks for stratification and ``groups`` keeps each
    group on one side and in one fold; both align with ``universe`` when it is given, else with
    ``row_ids``. Rows are sorted by id first: the same rows, seed and settings always give the
    same split. Returns the assignment frame (``row_id``, ``partition``, ``fold``; fold -1 when
    held out) and the facts the split artifact reports, plus ``sealed``: every row of the
    universe drawn to be held out, in or out of ``row_ids`` — the rows nothing may read.
    """
    import pandas as pd

    base = np.asarray(row_ids if universe is None else universe, dtype=np.int64)
    order = np.argsort(base, kind="stable")
    uids = base[order]

    def keyed(values: Any | None) -> Any | None:
        if values is None:
            return None
        return np.array([_level_key(v) for v in np.asarray(values, dtype=object)[order]], dtype=object)

    y_u, g_u = keyed(y), keyed(groups)
    notes: list[str] = []
    held_u, stratified = _draw_holdout(len(uids), y_u, g_u, holdout, seed, notes)

    rows = np.sort(np.asarray(row_ids, dtype=np.int64))
    pos = np.clip(np.searchsorted(uids, rows), 0, max(0, len(uids) - 1))
    inside = (pos < len(uids)) & (uids[pos] == rows) if len(uids) else np.zeros(len(rows), dtype=bool)
    held = np.where(inside, held_u[pos] if len(uids) else False, False)
    y_r = None if y_u is None else np.where(inside, y_u[pos], "__unknown__")
    g_r = None if g_u is None else np.where(inside, g_u[pos], np.array([f"__row_{r}" for r in rows], dtype=object))

    train = np.flatnonzero(~held)
    fold = np.full(len(rows), -1, dtype=np.int64)
    train_folds, k = _assign_folds(
        len(train), None if y_r is None else y_r[train], None if g_r is None else g_r[train],
        folds, seed, stratified, notes)
    fold[train] = train_folds

    frame = pd.DataFrame({"row_id": rows, "partition": np.where(held, "holdout", "train"), "fold": fold})
    info = {
        "n_train": int(len(train)),
        "n_holdout": int(held.sum()),
        "holdout": float(holdout),
        "seed": int(seed),
        "folds": int(k),
        "grouped_by": grouped_by if g_r is not None else None,
        "n_groups": int(len(pd.unique(g_r))) if g_r is not None else None,
        "stratified": bool(stratified),
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
    sentence = ", ".join(parts) + f"; `{info['folds']}` folds."
    extra = " ".join(info.get("notes") or [])
    return f"{sentence} {extra}".strip()


def split_inputs(state: Any, cohort_rows: Any, store: Any, task: str | None) -> dict[str, Any]:
    """The labels and groups ``draw_split`` needs for these cohort rows."""
    ids = np.asarray(cohort_rows, dtype=np.int64)
    out: dict[str, Any] = {"y": None, "groups": None, "grouped_by": None}
    identifiers = [c for c, r in (state.roles or {}).items() if r == "identifier"]
    columns = list(identifiers)
    classify = task in ("binary", "multiclass")
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
    with open_store(ctx) as store:
        inputs = split_inputs(ctx.state, universe, store, task)
    ctx.progress(0.4, "Drawing the held-out rows and the folds")
    frame, info = draw_split(
        rows, holdout=spec.holdout, seed=spec.seed, folds=spec.folds, universe=universe,
        y=inputs["y"], groups=inputs["groups"], grouped_by=inputs["grouped_by"],
    )
    notes = info.pop("notes")
    sealed = info.pop("sealed")
    data = {**info, "note": split_note({**info, "notes": notes})}
    # "sealed" holds every held-out row, including ones the cohort now excludes: an exclusion
    # relaxed later brings them back held out, so previews must never read them either.
    return Bundle(data=data, frames={"assignment": frame, "sealed": pd.DataFrame({"row_id": sealed})})


__all__ = [
    "PREDICTOR_ROLES", "cohort_flow", "cohort_inputs", "cohort_stage", "compute_cohort",
    "draw_split", "energy_bearing", "predictors", "propose_roles", "roles_stage", "rule_keep",
    "rule_label", "split_inputs", "split_note", "split_stage",
]

"""The ``proposals`` stage: what the field usually does, offered for the exclusions and
energy-adjustment questions — never pre-selected (M1_CONTRACT §3).

Two readings, both from ``docs/turbotab/research/NUTRITION_PACK.md``:

* **Exclusions** (§02, Diagnostic 1). The fixed kcal screens in circulation — Willett's
  sex-specific 500–3,500 kcal/d for women and 800–4,200 for men, and the sex-neutral 500–5,000
  and 500–3,500 — each as an :class:`~turbotab.core.decisions.ExclusionRule` with the number of
  rows it would remove. The pack says the conventions *genuinely differ across literatures* and
  that the app must show how N moves with the choice, so every screen is offered with its count
  and its CONVENTION badge, and none is chosen.
* **Energy** (§04). The energy column, the energy-bearing nutrients, the strata a residual can be
  computed within, which of the five models can run on these columns, and the field's usual
  method (the Willett residual, CONVENTION) — first in order, never selected.
* **Missing values** (M1_CONTRACT §12.4, under every lens). Each predictor with blanks, how many,
  and whether the blanks likely mean "not asked": a mostly blank (≥ 50%) yes/no or
  medication-like column, where a blank is a skipped question rather than an unknown value
  (DRIVE_RUBRIC §4: median fill on ``meds_hbp`` would put every unknown on medication). The
  missing-values question offers to leave those columns out before complete cases.

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
    rule = rule if isinstance(rule, ExclusionRule) else ExclusionRule.model_validate(rule)
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
_ENERGY_NAME = re.compile(r"kcal|(?:^|[^a-z])kj(?:$|[^a-z])|energy|calor", re.I)
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
    from turbotab.core.methods.energy import nutrient_role, unit_of

    if not _ENERGY_NAME.search(str(column)) or unit_of(column) == "density":
        return False
    try:
        return nutrient_role(column) is None
    except ValueError:
        return False


def energy_column(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str]) -> str | None:
    """The total-energy column: the one the roles name, else the one its name declares."""
    named = [c for c, r in roles.items() if r == "energy" and c in columns]
    if named:
        return named[0]
    candidates = [c for c, info in columns.items()
                  if _dtype(info) in _NUMERIC and is_energy_name(c)
                  and roles.get(c) not in ("identifier", "flag", "design", "time", "excluded")]
    if not candidates:
        return None

    def rank(c: str) -> tuple[int, int]:
        name = c.lower()
        order = 0 if "kcal" in name else 1 if "energy" in name or "calor" in name else 2
        return order, list(columns).index(c)

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


def nutrient_candidates(columns: Mapping[str, Mapping[str, Any]], roles: Mapping[str, str], *,
                        energy: str | None, target: str | None) -> list[str]:
    out = []
    for c, info in columns.items():
        if c in (energy, target) or _dtype(info) not in _NUMERIC or _FLAG.search(c):
            continue
        if roles and roles.get(c) not in (None, "exposure"):
            continue
        if energy_bearing(c):
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
        else:
            reason = "carries no energy"
        out.append({"column": c, "reason": reason})
    return out


def sex_column(columns: Mapping[str, Mapping[str, Any]], frame: pd.DataFrame | None,
               roles: Mapping[str, str]) -> tuple[str | None, dict[str, str]]:
    """The sex column and which of its levels are ``female`` and ``male``."""
    for c in columns:
        if c.lower() not in _SEX_NAMES and not ({"sex", "gender"} & set(_tokens(c))):
            continue
        if roles.get(c) in ("identifier", "flag", "design", "excluded"):
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
        if roles.get(c) in ("identifier", "flag", "design", "excluded", "energy", "exposure"):
            continue
        n_unique = int((info or {}).get("n_unique") or 0)
        if c != sex and (_dtype(info) not in _TEXTUAL or not 2 <= n_unique <= MAX_STRATA_LEVELS):
            continue
        out.append(c)
    if sex in out:
        out.remove(sex)
        out.insert(0, sex)
    return out


def _parts_note(frame: pd.DataFrame, nutrients: Sequence[str]) -> list[str]:
    """Columns that are parts of another chosen column (``fat_sat`` of ``fat_total``)."""
    from turbotab.core.methods.energy import nutrient_role
    from turbotab.core.voice import listing, tick

    by_role: dict[str, list[str]] = {}
    for c in nutrients:
        try:
            role = nutrient_role(c)
        except ValueError:
            continue
        if role:
            by_role.setdefault(role, []).append(c)
    notes = []
    for role, cols in by_role.items():
        if len(cols) < 2 or not set(cols) <= set(frame.columns):
            continue
        values = frame[cols].apply(pd.to_numeric, errors="coerce")
        total = str(values.mean().idxmax())
        parts = [c for c in cols if c != total]
        both = values[[total, *parts]].dropna()
        if len(both) < 10:
            continue
        inside = (both[parts].sum(axis=1) <= both[total] * 1.02 + 1e-9).mean()
        if inside >= 0.95:
            notes.append(
                f"{listing(parts)} {'is a part' if len(parts) == 1 else 'are parts'} of {tick(total)}: "
                f"choosing them together counts {role}'s energy twice in a partition; a substitution "
                f"moves the parts with their total.")
    return notes


def _energy_unit(frame: pd.DataFrame, energy: str) -> str:
    """``kj`` when the column is in kilojoules (its suffix, or the Atwater reconstruction)."""
    from turbotab.core.methods.energy import unit_of

    if unit_of(energy) == "kj":
        return "kj"
    try:
        from turbotab import nutrition

        reading = nutrition.atwater(frame)
    except Exception:  # the legacy check is a second opinion, never a failure
        reading = None
    if reading is not None and reading.energy_column == energy and reading.verdict == "energy_in_kj":
        return "kj"
    return "kcal"


# ── the proposals ────────────────────────────────────────────────────────────

def _kcal(value: float) -> str:
    return f"{value:,.0f}"


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
        ranges: dict[str, tuple[float | None, float | None]] = {}
        for k in women:
            ranges[k] = bounds(500, 3500)
        for k in men:
            ranges[k] = bounds(800, 4200)
        screens.append((
            "willett_by_sex",
            f"Willett, by sex: women 500–3,500 and men 800–4,200 {per_day}"
            + (" (compared in kJ)" if unit == "kj" else ""),
            ExclusionRule(column=energy, by=RangeByLevel(column=sex, ranges=ranges),
                          reason="implausible intakes (Willett's sex-specific cut-offs)"),
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
    for key, label, rule in screens:
        affected = int((rule_excludes(frame, rule) & base).sum())
        out.append({"key": key, "rule": rule.model_dump(mode="json"), "label": label,
                    "affected": affected, "evidence": dict(EXCLUSION_EVIDENCE)})
    return out


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
                   sex: str | None, target: str | None) -> dict[str, Any]:
    from turbotab.core.methods.energy import applicable_methods
    from turbotab.core.voice import finish

    verdicts = applicable_methods(list(columns), energy, nutrients)
    if verdicts["partition"]["ok"] and energy in frame.columns:
        # The checks the fit makes on the data (the Atwater reconstruction), and the two it
        # cannot: a total beside its parts, and nutrients that out-weigh total energy.
        refused = partition_check(frame, energy, nutrients)
        if refused is not None:
            verdicts["partition"] = {"ok": False, "reason": refused["reason"]}
    applicability = {m: {"ok": bool(v["ok"]), "reason": finish(_marked(str(v["reason"]), columns))}
                     for m, v in verdicts.items()}
    usual = next((m for m in (USUAL_METHOD, "standard") if applicability.get(m, {}).get("ok")), None)
    r_with_energy: dict[str, float] = {}
    if energy and energy in frame.columns:
        e = pd.to_numeric(frame[energy], errors="coerce")
        for n in nutrients:
            if n in frame.columns:
                r = e.corr(pd.to_numeric(frame[n], errors="coerce"))
                if r is not None and not math.isnan(r):
                    r_with_energy[n] = round(float(r), 3)
    exclude = [c for c in (energy, target, *nutrients) if c]
    return {
        "energy_column": energy,
        "nutrients": nutrients,
        "strata_candidates": strata_candidates(columns, roles, exclude=exclude, sex=sex),
        "applicability": applicability,
        "usual": usual,
        "usual_evidence": dict(ENERGY_EVIDENCE) if usual else None,
        "r_with_energy": r_with_energy,
        "notes": _parts_note(frame, nutrients),
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
                    roles: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The proposals artifact from a frame holding (at least) the columns it reads.

    ``columns`` are the ingest's column records (``name``, ``dtype``, ``n_unique``,
    ``n_missing``); ``roles`` the confirmed roles, else the proposed ones, else empty.
    """
    from turbotab.core.voice import tick

    info = {str(c["name"]): c for c in columns}
    roles = dict(roles or {})
    missing = missing_reading(frame, columns, roles, target)
    n_base = int(frame[target].notna().sum()) if target and target in frame.columns else len(frame)
    if "dietary" not in (lens or []):
        return {"exclusions": [], "energy": None, "missing": missing, "n_base": n_base,
                "basis": "Only the missing-values reading is proposed: the dietary lens is not chosen."}
    energy = energy_column(info, roles)
    nutrients = nutrient_candidates(info, roles, energy=energy, target=target)
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
    if energy is not None and energy in frame.columns:
        unit = _energy_unit(frame, energy)
        exclusions = exclusion_proposals(frame, energy=energy, unit=unit, sex=sex,
                                         sex_levels=sex_levels, base=base)
    reading = None
    if energy is not None or nutrients:
        reading = energy_reading(frame, info, roles, energy=energy, nutrients=nutrients, sex=sex,
                                 target=target)
    return {"exclusions": exclusions, "energy": reading, "missing": missing, "n_base": n_base,
            "basis": basis}


def needed_columns(columns: Sequence[Mapping[str, Any]], *, target: str | None,
                   roles: Mapping[str, str]) -> list[str]:
    """The few columns the proposals read: energy, sex, the outcome and the nutrients."""
    info = {str(c["name"]): c for c in columns}
    energy = energy_column(info, roles)
    nutrients = nutrient_candidates(info, roles, energy=energy, target=target)
    sexes = [c for c in info if c.lower() in _SEX_NAMES or {"sex", "gender"} & set(_tokens(c))]
    wanted = [energy, target, *sexes, *nutrients]
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
    ctx.progress(0.6, "Counting what each exclusion rule would remove")
    return build_proposals(frame, columns, lens=state.lens, target=target, roles=roles)


__all__ = [
    "ENERGY_EVIDENCE", "EXCLUSION_EVIDENCE", "build_proposals", "energy_bearing", "energy_column",
    "exclusion_proposals", "gappy_predictors", "level_key", "missing_reading", "needed_columns",
    "nutrient_candidates", "partition_check", "proposals_stage",
    "roles_from", "rule_excludes", "sex_column", "strata_candidates",
]

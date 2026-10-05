"""Reference rows leave before the seal (audit RO-13, WP18; BLUEPRINT §13's worked example).

Pooled quality-control injections are one sample run repeatedly: the instrument being checked, not
people being measured. The audit found the pooled-QC finding critical with "No control for this
yet", silent on the usual export shape (a ``Class`` column of ``Case``, ``Control`` and ``QC``: the
legacy detector needs exactly two levels), the task then read as three-class with ``QC`` a class,
and the served text claiming the rows already "stay in the table for quality assessment and out of
the modeling rows".

So here:

* :func:`pooled_qc_finding` reads the pooled QC level of a label column with 2–10 levels by its
  variance (the legacy reading, name-blind: a minority level whose rows vary far less across the
  feature block than the rest), and its text says what is true: until they leave, they are rows
  like any other.
* The ``reference_rows`` repair family answers it, and the naming census's ``sample_roles`` finding
  where a run-type column (``class``, ``sample_type`` …) names the roles by level: "Exclude the QC
  rows", recorded as an ``apply_repair``. Where an injection order is read, the pooled-QC finding
  also offers QC-RLSC drift correction first (MS7, ``methods/qc_drift.py``): every injection is
  corrected from the pooled QCs, and then the QC rows leave here just the same.
* It executes in the **working table** (:func:`reference_filter`), before the outcome is read and
  before the seal, so the outcome's levels are the participants' (Case/Control: a binary task) and
  no QC row can be drawn into the held-out set. The participant flow counts them on a line of
  their own (``stages.rows.cohort_flow``'s ``reference`` steps).
* Its data scope is *row-local* (BLUEPRINT §13): whether a row leaves reads that row's own label.
  Once the held-out rows are drawn it is refused: the seal is drawn over the rows that remain.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.decisions import ApplyRepair, Refusal, register_validator
from turbotab.core.repairs import (
    Family,
    OfferContext,
    RepairOption,
    _applied,
    _count,
    _option,
    _plural,
    lit_str,
    register_family,
)

FAMILY = "reference_rows"
MIN_FEATURES = 30  # the assay floor the legacy reading keeps (``packs._is_assay_wide``)
MAX_LEVELS = 10
RSD_RATIO = 0.6  # the QC level's median relative SD at most this share of the other rows'
ROLE_COLUMNS = frozenset({"type", "sample_type", "role", "class", "group"})  # the pack's §01


# ── the reading ──────────────────────────────────────────────────────────────


def _rsd(block: pd.DataFrame) -> float:
    with np.errstate(all="ignore"):
        return float((block.std() / block.mean().abs()).median())


def pooled_qc_finding(df: pd.DataFrame) -> dict[str, Any] | None:
    """Rows that are one sample injected repeatedly, read by variance: in a label column of 2–10
    levels, a minority level (3 rows to 30% of the table) whose rows' median relative SD across
    the feature block is at most :data:`RSD_RATIO` of every other row's. The finding keeps the
    legacy id and shape (``pack::metabolomics::pooled_qc``); its text no longer says the rows are
    already out of the model."""
    from turbotab import packs

    if df is None or df.empty:
        return None
    cols = packs._numeric(df)
    if len(cols) < MIN_FEATURES:
        return None
    n = len(df)
    best: tuple[float, str, Any, int, float, float, list[Any]] | None = None
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        try:
            counts = s.value_counts(dropna=True)
        except TypeError:  # unhashable cells
            continue
        if not 2 <= len(counts) <= MAX_LEVELS:
            continue
        for level, n_minor in counts.items():
            n_minor = int(n_minor)
            if n_minor < 3 or n_minor > 0.3 * n:
                continue
            at = s == level
            rest = df.loc[~at & s.notna(), cols]
            if len(rest) < 5:
                continue
            rsd_qc, rsd_rest = _rsd(df.loc[at, cols]), _rsd(rest)
            if not (np.isfinite(rsd_qc) and np.isfinite(rsd_rest)) or rsd_rest <= 0:
                continue
            ratio = rsd_qc / rsd_rest
            if ratio > RSD_RATIO:
                continue
            others = [v for v in counts.index if v != level]
            if best is None or ratio < best[0]:
                best = (ratio, str(c), level, n_minor, rsd_qc, rsd_rest, others)
    if best is None:
        return None
    _, c, level, n_qc, rsd_qc, rsd_rest, others = best
    n_rest = n - n_qc - int(df[c].isna().sum())
    shown = ", ".join(f"{v!r}" for v in others[:4])
    return packs._finding(
        "pack::metabolomics::pooled_qc", "critical",
        f"{n_qc:,} rows look like pooled quality-control injections",
        (f"The {n_qc:,} rows where `{c}` is {level!r} vary far less across the {len(cols):,} "
         f"features than the {n_rest:,} rows where it is {shown} — a median relative standard "
         f"deviation of {rsd_qc:.0%} against {rsd_rest:.0%}. That is one sample injected "
         f"repeatedly, not {n_qc:,} different people."),
        ("They are not participants: modeling them is an error with no legitimate reading. Until "
         "they are excluded they are analyzed as rows like any other, and a held-out set could "
         "contain them. Excluding them here removes them before the held-out rows are drawn and "
         "counts them in the participant flow; they stay in the file for quality assessment."),
        confidence="high", pack=packs.METABOLOMICS, marker="derived",
        evidence=packs.POOLED_QC_EVIDENCE, columns=[c],
        params={"column": c, "qc_value": str(level), "n_qc": n_qc,
                "rsd_qc": round(rsd_qc, 3), "rsd_participants": round(rsd_rest, 3),
                "other_levels": [str(v) for v in others]})


# ── the repair ───────────────────────────────────────────────────────────────


def _role_levels(params: Mapping[str, Any], frame: pd.DataFrame) -> tuple[str, list[str]] | None:
    """The run-type column and its levels the naming census matched (``sample_roles``): a level
    every row of which is an instrument run. None when the roles are named only by sample names."""
    from turbotab import packs

    for match in params.get("matches") or []:
        column = str(match.get("column") or "")
        if column not in frame.columns or packs._norm_name(column) not in ROLE_COLUMNS:
            continue
        values = frame[column].dropna().astype(str)
        levels = []
        for family, patterns in packs.ROLE_PATTERNS:
            for pattern in patterns:
                rx = packs._ROLE_RE[(family, pattern)]
                levels += [v for v in values.unique() if rx.search(v) and v not in levels]
        if levels and len(levels) < values.nunique():
            return column, sorted(levels)
    return None


def _offer(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    """QC-RLSC first where the pooled-QC finding has an injection order to correct over (MS7, the
    soundest answer: drift corrected, then the rows leave), then the exclusion on its own."""
    from turbotab.core.stages.finding_words import family

    rlsc: list[RepairOption] = []
    if family(str(finding.get("id"))) == "pack::metabolomics::pooled_qc":
        from turbotab.core.methods.qc_drift import rlsc_offer

        rlsc = rlsc_offer(finding, p, oc)
    return [*rlsc, *_exclusion(finding, p, oc)]


def _exclusion(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    if p.get("column") and p.get("qc_value") is not None:
        column, levels = str(p["column"]), [str(p["qc_value"])]
    else:
        found = _role_levels(p, oc.frame)
        if found is None:
            return []
        column, levels = found
    # The outcome's own column may name them (``Class`` = ``QC``): its QC level leaves all the
    # same, and the task then reads the participants' levels.
    if not oc.has(column):
        return []
    n =int(oc.frame[column].astype(str).isin(levels).sum())
    if not n:
        return []
    shown = " or ".join(f"`{v}`" for v in levels[:3])
    return [_option(
        finding, "exclude_rows", "Exclude the QC rows",
        f"{_count(n)} {_plural(n, 'row')} where `{column}` is {shown} leave before the seal.",
        f"{_count(n)} {_plural(n, 'row')} where `{column}` is {shown} "
        f"({_plural(n, 'an instrument run', 'instrument runs')}, not "
        f"{_plural(n, 'a participant', 'participants')}) {_plural(n, 'was', 'were')} excluded as "
        f"reference rows before the held-out rows were drawn.",
        "rows", {"column": column, "levels": levels})]


def _marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    if option != EXCLUDE:
        from turbotab.core.methods.qc_drift import _marks as rlsc_marks

        return rlsc_marks(option, params)
    return {(str(params.get("column")), f"reference:{level}") for level in params.get("levels") or []}


def _columns(option: str, params: Mapping[str, Any]) -> list[str]:
    """The columns an answer takes out of the predictors: none for the exclusion (the label may be
    the outcome), the QC label, the injection order and the features the QC filters removed for
    QC-RLSC (``qc_drift._columns_out``)."""
    if option == EXCLUDE:
        return []
    from turbotab.core.methods.qc_drift import _columns_out

    return _columns_out(option, params)


EXCLUDE = "exclude_rows"
RLSC_OPTIONS = ("qc_rlsc_lc", "qc_rlsc_lc_dratio", "qc_rlsc_gc", "qc_rlsc_gc_dratio")
register_family(Family(FAMILY, 9, _offer,
                       {EXCLUDE: "rows", **{k: "values" for k in RLSC_OPTIONS}},
                       columns=_columns, marks=_marks),
                ["pack::metabolomics::pooled_qc", "pack::metabolomics::sample_roles"])


def exclusion_sentence(decision: Any, ctx: Any) -> str:
    """The methods sentence of an exclusion: the one the finding offered with its count, else the
    same words without it (a record read with no findings at hand)."""
    from turbotab.core.voice import _get, _offered_sentence

    said = _offered_sentence(decision, _get(ctx, "finding")) if ctx is not None else None
    if said:
        return said
    params = dict(decision.params or {})
    levels = params.get("levels") or params.get("qc_levels") or []
    shown = " or ".join(f"`{v}`" for v in list(levels)[:3])
    return (f"The rows where `{params.get('column')}` is {shown} (instrument runs, not "
            f"participants) were excluded as reference rows before the held-out rows were drawn")


def reference_rules(dispositions: Any, findings: Any = None) -> list[dict[str, Any]]:
    """Each applied reference-row exclusion: ``{column, levels, finding}``, one per column, levels
    merged (the pooled-QC and naming findings may both name the QC level)."""
    out: dict[str, dict[str, Any]] = {}
    for fam, fid, option, params in _applied(dispositions, findings):
        if fam.key != FAMILY or not params.get("column"):
            continue
        # The exclusion, and QC-RLSC, after which the QC rows leave just the same (MS7).
        levels = params.get("levels") if option == EXCLUDE else params.get("qc_levels")
        entry = out.setdefault(str(params["column"]),
                               {"column": str(params["column"]), "levels": [], "finding": fid})
        entry["levels"] += [str(v) for v in levels or [] if str(v) not in entry["levels"]]
    return list(out.values())


def reference_filter(rules: Sequence[Mapping[str, Any]]) -> str | None:
    """The SQL condition that keeps every row the rules do not name (a blank label is kept)."""
    from turbotab.core.repairs import ident

    parts = []
    for rule in rules:
        listed = ", ".join(lit_str(str(v)) for v in rule.get("levels") or [])
        if listed:
            col = ident(str(rule["column"]))
            parts.append(f"({col} IS NULL OR CAST({col} AS VARCHAR) NOT IN ({listed}))")
    return " AND ".join(parts) or None


def _before_the_seal(decision: ApplyRepair, ctx: Any) -> None:
    """Reference rows leave before the held-out rows are drawn: afterwards they would re-draw it."""
    from turbotab.core.decisions import _state
    from turbotab.core.repairs import family_for

    fam = family_for(decision.finding_id)
    state = _state(ctx)
    if fam is None or fam.key != FAMILY or state is None or getattr(state, "split", None) is None:
        return
    raise Refusal(
        "rows_after_the_seal",
        "The held-out rows are drawn over the rows the analysis holds, so reference rows leave "
        "before the seal; excluding them now would draw it again. Start a new analysis, or keep "
        "them in and say so.",
        exits=[{"label": "Keep the rows as they are", "decision": None}])


register_validator("apply_repair", _before_the_seal)

__all__ = ["EXCLUDE", "FAMILY", "exclusion_sentence", "pooled_qc_finding", "reference_filter",
           "reference_rules"]

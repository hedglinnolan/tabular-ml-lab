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
  Once rows are held out it is refused where the rows leaving would move the draw
  (:func:`_before_the_seal`): a leaving row with its outcome recorded draws the held-out rows
  again, and one before a row in the draw renumbers it, while the seal names rows by number.
  Pooled QCs with no outcome were never in the draw, so after the engine's own no-holdout split
  under Estimate, or after a split whose drawn rows all come before them, they still leave.
"""
from __future__ import annotations

from pathlib import Path
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


def _ctx_reader(ctx: Any, name: str, stage: str) -> Any:
    """``ctx.<name>(stage)`` (the fresh public artifact, or a fresh stage's files), else None."""
    from turbotab.core.decisions import _ctx

    reader = _ctx(ctx, name)
    if not callable(reader):
        return None
    try:
        return reader(stage)
    except Exception:  # noqa: BLE001 - an unreadable stage reads as not fresh
        return None


def _what_moves(decision: ApplyRepair, state: Any, ctx: Any) -> str | None:
    """What the rows this answer takes out (or brings back) would move of the held-out draw:
    ``"draw"`` where they could draw it again, ``"numbers"`` where they renumber rows in it, None
    where neither.

    The held-out rows are drawn over the rows with the outcome recorded, sorted by working row id
    and drawn by position, and the folds over the training rows the same way
    (``stages.rows.draw_split``); the rows that stay are numbered 0…n − 1 again, keeping their
    order (``stages.working``). So the draw moves where a row that leaves or returns has its
    outcome recorded (as the working table reads it, after its repairs). One with no outcome
    recorded keeps every drawn row's place in the draw, but renumbers each row after it, and the
    seal names rows by number: until the split is drawn again over the new working table, the
    previews and the evidence read every row but the old split's sealed numbers
    (``server.service.sealed_rows``), which would then name other rows, held-out participants
    among those read. So nothing moves only where every row that leaves or returns comes after
    every row in the draw; a grouped draw then holds still too, since it names a drawn row with no
    identifier by its working row id (``__missing_{rid}``). Read on the oriented table, where the
    working table reads the labels; ``"draw"`` wherever it cannot be read (no table at hand, rows
    combined per unit, an outcome the table does not hold), since the seal then cannot be shown to
    stay."""
    from turbotab.core import decisions
    from turbotab.core.repairs import _fill_params, evaluate, ident
    from turbotab.core.stages.rows import _measured
    from turbotab.core.stages.working import (
        TABLE, repair_expressions, row_local_additions, text_amounts,
    )

    target = getattr(state, "target", None)
    grain = getattr(state, "grain", None)
    if target is None or (grain is not None and grain.grain == "repeated"
                          and getattr(state, "unit", None) == "unit"
                          and getattr(state, "aggregation", None) is not None):
        return "draw"
    oriented = _ctx_reader(ctx, "bundle", "oriented")
    files = getattr(oriented, "files", None) or {}
    if TABLE not in files:
        return "draw"
    findings = _ctx_reader(ctx, "artifact", "findings")
    findings = getattr(findings, "data", findings)
    decision = _fill_params(decision, ctx)
    now = dict(getattr(state, "findings", None) or {})
    after = {**now, decision.finding_id: decisions._SLOT_VALUE["apply_repair"](decision)}
    source = Path(files[TABLE])
    names = [str(c["name"]) for c in (getattr(oriented, "data", None) or {}).get("columns", [])]
    try:
        repairs = repair_expressions(findings, now)
        repairs.update(text_amounts(state, source, names, repairs))
        _, derived, _ = row_local_additions(state, findings, names, repairs)
        if target not in names and target not in derived:
            return "draw"
        keep = {when: reference_filter([r for r in reference_rules(answers, findings)
                                         if r["column"] in names]) or "TRUE"
                for when, answers in (("now", now), ("after", after))}
        alias = {"now": "__tt_kept_now", "after": "__tt_kept_after", "outcome": "__tt_outcome"}
        expressions = {alias["now"]: keep["now"], alias["after"]: keep["after"],
                       alias["outcome"]: derived.get(target) or repairs.get(target, ident(target))}
        frame = evaluate(source, list(alias.values()), expressions)  # in row-id order
    except Exception:  # noqa: BLE001 - values that cannot be read: the seal cannot be shown to stay
        return "draw"
    kept_now = frame[alias["now"]].astype(bool).to_numpy()
    kept_after = frame[alias["after"]].astype(bool).to_numpy()
    changed = kept_now != kept_after
    if not changed.any():
        return None
    recorded = _measured(frame[alias["outcome"]]).to_numpy()
    if (changed & recorded).any():
        return "draw"
    drawn = np.flatnonzero((kept_now | kept_after) & recorded)
    if len(drawn) and int(np.flatnonzero(changed)[0]) < int(drawn[-1]):
        return "numbers"
    return None


def _before_the_seal(decision: ApplyRepair, ctx: Any) -> None:
    """Reference rows leave before the held-out rows are drawn where, afterwards, they would move
    the draw: while the split holds rows out (a share or a cut-point) and the rows that leave would
    draw it again or renumber rows in it (:func:`_what_moves`). Under cross-validation alone nothing
    is held out, and the folds are drawn again over the rows that stay, as after an exclusion; rows
    with no outcome recorded were never in the draw, so the pooled QCs of the usual export leave
    with every participant where the folds put them."""
    from turbotab.core.decisions import _state, holds_out
    from turbotab.core.repairs import family_for

    fam = family_for(decision.finding_id)
    state = _state(ctx)
    if fam is None or fam.key != FAMILY or state is None:
        return
    if not holds_out(getattr(state, "split", None)):
        return
    moves = _what_moves(decision, state, ctx)
    if moves is None:
        return
    if moves == "numbers":
        why = (f"The held-out rows are drawn and named by their row numbers, so reference rows "
               f"leave before the seal; excluding them now would renumber the rows with "
               f"`{state.target}` recorded that come after them.")
    else:
        why = ("The held-out rows are drawn over the rows the analysis holds, so reference rows "
               "leave before the seal; excluding them now would draw it again.")
    raise Refusal(
        "rows_after_the_seal", f"{why} Start a new analysis, or keep them in and say so.",
        exits=[{"label": "Keep the rows as they are", "decision": None}])


register_validator("apply_repair", _before_the_seal)

__all__ = ["EXCLUDE", "FAMILY", "exclusion_sentence", "pooled_qc_finding", "reference_filter",
           "reference_rules"]

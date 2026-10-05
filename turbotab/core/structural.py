"""Structural questions ask where evidence is thin (AUDIT_REPORT §5 WP18; RO-09, RO-10, RO-11, I18).

The opening sequence used to skip, as facts, several answers that are choices. This module holds the
rules that ask instead, and the refusals that keep each answer honest:

* **The lens that is none of the five** (RO-11). "Something else, or not sure" (``other``) is a
  first-class answer: the generic checks run, no field's defaults are applied, and the methods
  sentence says so. It stands alone: beside a lens that does describe the table it would be two
  answers, and the record could not say which (:func:`_other_lens_stands_alone`).
* **Predictors summarized after the outcome** (RO-09). With time points combined per unit, keeping
  the *first* outcome while a predictor is summarized from a later record (its last value, its mean,
  its change) predicts the past from the future. Under prediction that is refused; under inference
  it is blocked and recorded: refused with its exits, and kept only with the attestation the
  methods sentence carries (the reverse-causation concern). Copies of one imputed record have no
  order, so first, last and change are refused for them (:func:`_predictors_read_no_record_after_the_outcome`).
* **The outcome's order and scale** (RO-10). A text outcome of 3–10 levels is asked "are these
  levels ordered?" (``stages.target``'s ``order_question``), and an ordinal answer for text waits
  for the order of its levels; a positive, markedly skewed outcome is asked the scale it is
  analyzed on (``scale_question``: the original scale, a difference in means, or the log scale, a
  ratio of geometric means). The Router keeps the task question open until both are answered
  (:func:`task_followup`). One rule (:func:`turbotab.core.decisions.task_fits`) says which tasks an
  outcome may take, for the detection's skip and the explicit answer alike.
* **Imputed copies** (I18). NHANES ships its 1999–2006 DXA data as five imputed copies of every
  participant; analyzing them as records, or averaging them, treats imputed values as measured.
  Under inference, copies kept as records are each analyzed as a completed dataset with its own
  outcome and pooled by Rubin's rules (MS3; ``stages.modeling._copies_for_table``), so the answer
  is accepted; combining them per unit instead is blocked and recorded
  (:func:`_imputed_copies_are_not_combined_unrecorded`).

"Markedly skewed" is West et al.'s reference value, as Kim (2013, *Restor Dent Endod* 38:52–54)
reports it: "West et al. (1996) proposed a reference of substantial departure from normality as an
absolute skew value > 2." Skewness says nothing about which scale is right, so it only decides that
the question is asked; both answers are defensible and both are offered.

Imported by ``turbotab.core.decisions``, which registers these validators on import.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from turbotab.core.decisions import (
    _UNKNOWN,
    OUTCOME_RULE,
    PREDICTOR_ROLES,
    Refusal,
    SetAggregation,
    SetExclusions,
    SetLens,
    SetOutcomeScale,
    SetRepeatKind,
    SetRoles,
    SetUnit,
    _columns_of,
    _ctx,
    _state,
    _store_of,
    _target_of,
    as_rule,
    log_outcome_name,
    register_validator,
)

OTHER_LENS = "other"
OTHER_LABEL = "Something else, or not sure"
# West et al.'s reference for a substantial departure from normality (Kim 2013, quoted above).
SKEW_MARKED = 2.0
SCALE_MIN_ROWS = 20  # fewer values say too little about a tail to ask about it
ORDER_LEVELS = (3, 10)  # a text outcome of this many levels is asked whether they are ordered

# Ordered response scales a label set may be read as: a proposal the user confirms in one tap, never
# a settlement (BLUEPRINT §14: a name is a guess). Each runs lowest first.
ORDERED_SCALES: tuple[tuple[str, ...], ...] = (
    ("none", "mild", "moderate", "severe", "very severe"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("very low", "low", "medium", "high", "very high"),
    ("very low", "low", "moderate", "high", "very high"),
    ("poor", "fair", "good", "very good", "excellent"),
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("strongly disagree", "disagree", "neither agree nor disagree", "agree", "strongly agree"),
    ("underweight", "normal", "overweight", "obese"),
    ("normal", "prediabetes", "diabetes"),
    ("none", "low", "moderate", "high"),
    ("absent", "mild", "moderate", "severe"),
)


def _norm_label(value: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value).strip().lower())


def proposed_order(levels: Sequence[Any]) -> list[str] | None:
    """The labels lowest first when every one of them is a word of one ordered scale (``none``,
    ``mild``, ``moderate``, ``severe``); numbers by value; else None. A guess the user confirms."""
    values = [str(v) for v in levels]
    try:
        numbers = [float(v) for v in values]
    except ValueError:
        numbers = None
    if numbers is not None and all(math.isfinite(x) for x in numbers):
        return [v for _, v in sorted(zip(numbers, values))]
    by_label = {_norm_label(v): v for v in values}
    if len(by_label) != len(values):
        return None
    for scale in ORDERED_SCALES:
        rank = {word: i for i, word in enumerate(scale)}
        if all(k in rank for k in by_label):
            return [by_label[k] for k in sorted(by_label, key=rank.__getitem__)]
    return None


def skewness(values: Any) -> float | None:
    """The sample skewness (adjusted Fisher–Pearson, as pandas computes it), or None."""
    import pandas as pd

    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(x) < 3 or float(x.std(ddof=1) or 0) == 0:
        return None
    return float(x.skew())


def scale_question(values: Any, column: str) -> dict[str, Any] | None:
    """The outcome-scale question for a positive, markedly skewed outcome whose logarithm is not:
    its evidence and its two answers, or None when it is not asked (a value at or below 0 has no
    logarithm; fewer than :data:`SCALE_MIN_ROWS` values; a skew at most :data:`SKEW_MARKED`; or a
    log as skewed as that, which is a few extreme values, a 999 code or an entry error, rather than
    a scale: the detectors' business, and no log answers it)."""
    import numpy as np
    import pandas as pd

    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    x = x[np.isfinite(x.to_numpy(dtype=float))]
    if len(x) < SCALE_MIN_ROWS or float(x.min()) <= 0:
        return None
    skew = skewness(x)
    if skew is None or skew <= SKEW_MARKED:
        return None
    log_skew = skewness(np.log(x.to_numpy(dtype=float)))
    if log_skew is None or abs(log_skew) > SKEW_MARKED:
        return None
    from turbotab.core.voice import number

    derived = log_outcome_name(column)
    evidence = (f"`{column}` is positive and right-skewed: skewness {skew:.1f} over {len(x):,} "
                f"values (above West et al.'s 2), {log_skew:+.1f} on the log scale; median "
                f"{number(float(x.median()))}, largest {number(float(x.max()))}.")
    return {
        "question": "On which scale should the outcome be analyzed?",
        "skewness": skew,
        "log_skewness": log_skew,
        "n": int(len(x)),
        "min": float(x.min()),
        "median": float(x.median()),
        "max": float(x.max()),
        "evidence": evidence,
        "log_column": derived,
        "options": [
            {"label": "Original scale: a difference in means",
             "decision": SetOutcomeScale(column=column, scale="original").model_dump(mode="json")},
            {"label": f"Log scale (`{derived}`): a ratio of geometric means",
             "decision": SetOutcomeScale(column=column, scale="log").model_dump(mode="json")},
        ],
    }


def order_question(levels: Sequence[Any], column: str, numeric: bool) -> dict[str, Any] | None:
    """"Are these levels ordered?" for an outcome of 3–10 levels, with the order a scale's words
    propose (numbers by value), or None outside that range."""
    from turbotab.core.decisions import SetOutcomeOrder, SetTask

    lo, hi = ORDER_LEVELS
    values = [str(v) for v in levels]
    if not lo <= len(values) <= hi:
        return None
    proposal = proposed_order(values)
    options = [
        {"label": "Ordered: an ordinal outcome",
         "decision": SetTask(column=column, task="ordinal").model_dump(mode="json")},
        {"label": "Unordered: separate classes",
         "decision": SetTask(column=column, task="multiclass").model_dump(mode="json")},
    ]
    order_exits = []
    if proposal is not None:
        shown = " < ".join(f"`{v}`" for v in proposal)
        order_exits.append({"label": f"Lowest first: {shown}",
                            "decision": SetOutcomeOrder(column=column, levels=proposal)
                            .model_dump(mode="json")})
    return {
        "question": "Are these levels ordered?",
        "levels": values,
        "numeric": bool(numeric),
        "proposed_order": proposal,
        "evidence": ("Numbers are ordered by value." if numeric else
                     "The labels are the words of an ordered scale." if proposal is not None else
                     "No dtype says whether text labels are ordered."),
        "options": options,
        "order_options": order_exits,
    }


# ── the outcome on its log scale ─────────────────────────────────────────────


def log_outcome(state: Any) -> str | None:
    """The column whose logarithm is the outcome, when the target is that derived column."""
    spec = getattr(state, "outcome_scale", None)
    if spec is None or spec.scale != "log":
        return None
    return spec.column if getattr(state, "target", None) == log_outcome_name(spec.column) else None


def outcome_source(state: Any) -> str | None:
    """The outcome as the table spells it: the target, or the column a log-scale target is the
    logarithm of (what the structure stage reads within units)."""
    return log_outcome(state) or getattr(state, "target", None)


def derived_columns(state: Any) -> dict[str, str]:
    """``{derived column: source column}`` the working table adds: the log-scale outcome."""
    source = log_outcome(state)
    return {log_outcome_name(source): source} if source else {}


def task_followup(state: Any, target_info: Any) -> str | None:
    """What the task question still needs once its task is known (audit RO-10): ``"scale"`` while
    a positive, markedly skewed outcome's scale is unanswered, ``"order"`` while an ordinal text
    outcome's levels are unordered; None when nothing is pending."""
    target = getattr(state, "target", None)
    info = target_info if isinstance(target_info, Mapping) else getattr(target_info, "data", None)
    if target is None or not isinstance(info, Mapping) or info.get("column") != target:
        return None
    task = getattr(state, "task", None) or info.get("task")
    spec = getattr(state, "outcome_scale", None)
    if (info.get("scale_question") and task == "regression" and log_outcome(state) is None
            and (spec is None or spec.column != target)):
        return "scale"
    question = info.get("order_question") or {}
    if (getattr(state, "task", None) == "ordinal" and question and not question.get("numeric")
            and not getattr(state, "outcome_order", None)):
        return "order"
    return None


def task_followup_possible(state: Any) -> bool:
    """Whether the recorded task may still need a follow-up, before the outcome is read: a
    regression outcome whose scale is unanswered, or an ordinal one whose order is not declared."""
    task = getattr(state, "task", None)
    target = getattr(state, "target", None)
    spec = getattr(state, "outcome_scale", None)
    if task == "regression":
        return log_outcome(state) is None and (spec is None or spec.column != target)
    return task == "ordinal" and not getattr(state, "outcome_order", None)


# ── validators ───────────────────────────────────────────────────────────────


def _artifact(ctx: Any, stage: str) -> Mapping[str, Any] | None:
    from turbotab.core.sequence import artifact

    return artifact(ctx, stage)


def _other_lens_stands_alone(decision: SetLens, ctx: Any) -> None:
    """'Something else, or not sure' beside a lens that describes the table is two answers."""
    if OTHER_LENS not in decision.lenses or len(decision.lenses) == 1:
        return
    named = [k for k in decision.lenses if k != OTHER_LENS]
    raise Refusal(
        "lens_other_alone",
        f"'{OTHER_LABEL}' says none of the listed kinds describes this table; beside "
        f"{', '.join(named)} it would be two answers, and the record could not say which.",
        exits=[{"label": f"Read it as {', '.join(named)}", "decision": SetLens(lenses=named)},
               {"label": OTHER_LABEL, "decision": SetLens(lenses=[OTHER_LENS])}])


def repeat_kind_of(state: Any, structure: Mapping[str, Any] | None) -> str | None:
    """The repeat kind answered, else the one the ledger holds settled."""
    spec = getattr(state, "repeat_kind", None)
    if spec is not None:
        return str(spec.repeat_kind)
    from turbotab.core.stages.working import effective_repeat_kind

    return effective_repeat_kind(state, structure)


LATER_RULES = ("last", "mean", "change", "mode")  # each reads a unit's last record


def _predictors_read_no_record_after_the_outcome(decision: SetAggregation, ctx: Any) -> None:
    """Audit RO-09: combining time points must not summarize a predictor from records later than
    every record its outcome is read from. With the first outcome kept, a predictor's last value,
    mean or change reads the unit's later records. Refused under prediction (a predictor later
    than the outcome is not available when it is predicted); blocked and recorded under inference
    (reverse causation: the later value may follow from the outcome). Copies of one imputed record
    have no order, so first, last and change are refused for them."""
    state = _state(ctx)
    if state is None:
        return
    structure = _artifact(ctx, "structure")
    kind = repeat_kind_of(state, structure)
    if kind == "imputed_copies":
        taken = {decision.method, decision.outcome or "", *decision.columns.values()}
        ordered = sorted(taken & {"first", "last", "change"})
        if ordered:
            raise Refusal(
                "copies_have_no_order",
                f"The rows are imputed copies of one record, which have no order, so "
                f"{' or '.join(ordered)} picks a copy by its number, not by time. Combine the "
                f"copies by the mean, or keep them as rows.",
                exits=[{"label": "Combine the copies by the mean",
                        "decision": SetAggregation(method="mean",
                                                   outcome="mean" if decision.outcome else None)},
                       {"label": "Keep each copy as a row", "decision": SetUnit(unit="row")}])
        return
    if kind != "time_points" or decision.outcome != "first":
        return
    grain = getattr(state, "grain", None)
    skip = {outcome_source(state), getattr(grain, "id_column", None),
            getattr(getattr(state, "repeat_kind", None), "time_column", None)}
    own = {c: r for c, r in decision.columns.items() if c not in skip}
    later = [c for c, r in own.items() if r in LATER_RULES]
    if decision.method == "first" and not later:
        return
    what = (f"{', '.join(f'`{c}`' for c in later)} by {'its' if len(later) == 1 else 'their'} own "
            f"rule" if decision.method == "first" else f"every predictor by its {decision.method}")
    purpose = getattr(state, "purpose", None)
    if purpose == "inference" and decision.acknowledged:
        return
    keep = {"columns": {c: r for c, r in decision.columns.items() if r not in LATER_RULES}}
    exits: list[dict[str, Any]] = [
        {"label": "Baseline predictors, the last outcome",
         "decision": SetAggregation(method="first", outcome="last", **keep)},
        {"label": "Baseline predictors, the baseline outcome",
         "decision": SetAggregation(method="first", outcome="first", **keep)},
        {"label": "Keep each record as a row (a temporal split can follow)",
         "decision": SetUnit(unit="row")},
    ]
    said = (f"The first outcome is kept, and {what} reads the unit's later records: the predictors "
            f"would be summarized after the outcome they explain.")
    if purpose == "inference":
        exits.append({"label": "Keep it, recorded: predictors summarized after the outcome",
                      "decision": decision.model_copy(update={"acknowledged": True})})
        raise Refusal(
            "predictors_after_outcome",
            f"{said} Under inference a later value may follow from the outcome rather than cause "
            f"it (reverse causation), so this is kept only as a recorded limitation.",
            exits=exits)
    raise Refusal(
        "predictors_after_outcome",
        f"{said} A prediction can only use what is known when it is made, so a predictor read "
        f"after the outcome is refused.",
        exits=exits)


def _outcome_values(ctx: Any, column: str) -> Any:
    store = _store_of(ctx)
    if store is None or column not in set(getattr(store, "columns", ()) or ()):
        return None
    try:
        return store.materialize([column])[column]
    except Exception:  # noqa: BLE001 - no values to check: the fit decides
        return None


def _outcome_scale_fits(decision: SetOutcomeScale, ctx: Any) -> None:
    """The scale answers for the outcome (or the outcome a log-scale target is the log of); the
    log scale needs a regression outcome whose every value is above 0, and no column already
    named as its derived one; the scale is part of the outcome, so it is chosen before the seal."""
    import numpy as np
    import pandas as pd

    state = _state(ctx)
    target = _target_of(ctx)
    derived = log_outcome_name(decision.column)
    if target is not _UNKNOWN:
        if target is None:
            raise Refusal("no_target", "Choose the outcome first; the scale describes it.",
                          exits=[{"label": "Choose the outcome", "decision": None}])
        if target not in (decision.column, derived):
            raise Refusal(
                "not_the_target",
                f"The outcome is `{target}`, not `{decision.column}`; the scale answers for the "
                f"outcome.",
                exits=[{"label": f"Choose the scale of `{target}`", "decision": None}])
    if state is not None and getattr(state, "split", None) is not None:
        current = (getattr(state, "outcome_scale", None) or None)
        if current is None or current.column != decision.column or current.scale != decision.scale:
            raise Refusal(
                "scale_after_the_seal",
                "The held-out rows are drawn, and the outcome's scale is part of what they hold "
                "out; it is chosen before the seal. Start a new analysis to change it.",
                exits=[{"label": "Keep the outcome's scale", "decision": None}])
    if decision.scale != "log":
        return
    task = _ctx(ctx, "task")
    if task not in (None, "regression"):
        raise Refusal(
            "scale_needs_a_number",
            f"`{decision.column}` is a {task} outcome; the log scale belongs to a continuous one.",
            exits=[{"label": "Keep the original scale",
                    "decision": SetOutcomeScale(column=decision.column, scale="original")}])
    columns = _columns_of(ctx)
    current = getattr(state, "outcome_scale", None) if state is not None else None
    ours = current is not None and current.column == decision.column and current.scale == "log"
    if columns is not None and derived in columns and not ours:
        raise Refusal(
            "derived_name_taken",
            f"The table already has a column named `{derived}`. If it is the log of "
            f"`{decision.column}`, choose it as the outcome.",
            exits=[{"label": f"Choose `{derived}` as the outcome",
                    "decision": {"kind": "set_target", "column": derived}},
                   {"label": "Keep the original scale",
                    "decision": SetOutcomeScale(column=decision.column, scale="original")}])
    values = _outcome_values(ctx, decision.column)
    if values is None:
        return
    x = pd.to_numeric(values, errors="coerce")
    present = x[values.notna()]
    if present.isna().any():
        raise Refusal("scale_needs_a_number",
                      f"`{decision.column}` holds values that are not numbers, so it has no log.",
                      exits=[{"label": "Keep the original scale",
                              "decision": SetOutcomeScale(column=decision.column,
                                                          scale="original")}])
    finite = present[np.isfinite(present.to_numpy(dtype=float))]
    low = int((finite <= 0).sum())
    if low:
        raise Refusal(
            "log_of_zero",
            f"{low:,} value{'s' if low != 1 else ''} of `{decision.column}` "
            f"{'are' if low != 1 else 'is'} 0 or below, where a logarithm is undefined; adding a "
            f"constant first would choose the estimand by the constant.",
            exits=[{"label": "Keep the original scale",
                    "decision": SetOutcomeScale(column=decision.column, scale="original")}])


def _log_outcome_source_is_no_predictor(decision: SetRoles, ctx: Any) -> None:
    """With the outcome on its log scale, the column it is the log of is the outcome itself."""
    state = _state(ctx)
    source = log_outcome(state) if state is not None else None
    if source is None or decision.roles.get(source) not in PREDICTOR_ROLES:
        return
    roles = {**decision.roles, source: "excluded"}
    raise Refusal(
        "outcome_as_predictor",
        f"`{source}` is the outcome on its original scale (the analysis reads "
        f"`{log_outcome_name(source)}`), so it cannot also predict it.",
        exits=[{"label": f"Leave `{source}` out", "decision": SetRoles(roles=roles)}])


def _no_rule_on_the_outcome_source(decision: SetExclusions, ctx: Any) -> None:
    """Audit RO-01, for the outcome on its log scale: a rule on the column it is the log of keeps
    rows by their outcome all the same."""
    state = _state(ctx)
    source = log_outcome(state) if state is not None else None
    if source is None:
        return
    on = [i for i, rule in enumerate(decision.rules) if source in as_rule(rule).reads()]
    if not on:
        return
    kept = SetExclusions(rules=[r for i, r in enumerate(decision.rules) if i not in on])
    raise Refusal(
        "rule_on_outcome",
        f"`{source}` is the outcome, analyzed as `{log_outcome_name(source)}`. {OUTCOME_RULE} Say "
        f"who is studied by what was known before the outcome.",
        exits=[{"label": f"Drop the rule on `{source}`", "decision": kept},
               {"label": "Restrict by a variable measured before the outcome", "decision": None}])


# NHANES 1999–2006 DXA (CDC, "Multiple Imputation Details", wwwn.cdc.gov/nchs/nhanes/dxa/dxa.aspx):
# "Each of the data files contains FIVE sets of measured and imputed values." … "The extra
# variability due to imputation CANNOT be incorporated by simply analyzing a SINGLE dataset as if
# the imputed values were true values." … "The preferred statistical approach is to analyze EACH OF
# THE FIVE datasets separately … and then combining the estimates and standard errors using the
# combining rules".
COPIES_CONCERN = ("analyzing imputed copies as records, or by their mean, treats imputed values as "
                  "measured, so the intervals leave out the imputation's uncertainty")


def _imputed_copies_are_recorded(decision: SetRepeatKind, ctx: Any) -> None:
    """Audit I18: imputed copies name the column that numbers them, one of the dataset's. Under
    inference the copies kept as records are pooled by Rubin's rules in the fit (MS3), so the answer
    itself is not held; combining them per unit is (:func:`_imputed_copies_are_not_combined_unrecorded`)."""
    if decision.repeat_kind != "imputed_copies":
        return
    columns = _columns_of(ctx)
    if decision.implicate_column is not None and columns is not None \
            and decision.implicate_column not in columns:
        raise Refusal("unknown_column",
                      f"This dataset has no column named `{decision.implicate_column}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])


def _imputed_copies_are_not_combined_unrecorded(decision: SetUnit, ctx: Any) -> None:
    """Audit I18 (MS3): under inference, combining a unit's imputed copies into one row treats imputed
    values as measured (CDC: "The extra variability due to imputation CANNOT be incorporated by
    simply analyzing a SINGLE dataset as if the imputed values were true values"), so it is blocked
    and recorded; keeping each copy as a record lets the fit analyze each and pool them by Rubin's
    rules, as NCHS directs."""
    state = _state(ctx)
    spec = getattr(state, "repeat_kind", None)
    if (decision.unit != "unit" or getattr(state, "purpose", None) != "inference" or spec is None
            or getattr(spec, "repeat_kind", None) != "imputed_copies"
            or getattr(spec, "acknowledged", False)):
        return
    raise Refusal(
        "imputed_copies_combined",
        f"Each unit's rows are imputed copies, and combining them into one row per unit means "
        f"{COPIES_CONCERN}. Kept as records, each copy is analyzed as a completed dataset with its "
        f"own outcome and the estimates are pooled by Rubin's rules, as NHANES directs for its DXA "
        f"files.",
        exits=[{"label": "Keep each copy as a record, pooled by Rubin's rules",
                "decision": SetUnit(unit="row")},
               {"label": "Combine them, recorded: intervals too narrow",
                "decision": SetRepeatKind(**{**spec.model_dump(), "acknowledged": True})}])


register_validator("set_lens", _other_lens_stands_alone)
register_validator("set_aggregation", _predictors_read_no_record_after_the_outcome)
register_validator("set_outcome_scale", _outcome_scale_fits)
register_validator("set_roles", _log_outcome_source_is_no_predictor)
register_validator("set_exclusions", _no_rule_on_the_outcome_source)
register_validator("set_repeat_kind", _imputed_copies_are_recorded)
register_validator("set_unit", _imputed_copies_are_not_combined_unrecorded)

__all__ = [
    "COPIES_CONCERN", "LATER_RULES", "ORDERED_SCALES", "OTHER_LABEL", "OTHER_LENS", "SKEW_MARKED",
    "derived_columns", "log_outcome", "order_question", "outcome_source", "proposed_order",
    "task_followup_possible",
    "repeat_kind_of", "scale_question", "skewness", "task_followup",
]

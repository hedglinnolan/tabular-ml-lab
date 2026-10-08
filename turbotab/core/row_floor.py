"""Rows enough to analyze: no answer is recorded that would leave the analysis fewer rows than the
design needs, and no stage is handed fewer.

Found by the no-dead-end drive (INBOX): ``metabolomics_untargeted.csv`` with complete cases. The
features rode along unconfirmed when the missing-values answer was recorded, so their blanks
dropped no row then (BLUEPRINT §14.1); confirmed at the models question, they made 389 gappy
columns predictors, every row was blank in at least one of them, and the design stage failed on
scikit-learn's "Found array with 0 sample(s)" while the seal's step waited on the failed fit.

So a check is registered for every kind of answer (as ``sequence`` registers the order): it folds
the answer into the state and, when that changes what the cohort reads, counts the cohort's rows
before and after, over the rows a preview may read (every row but the held-out ones under
prediction; every row under inference, ruling 3). An answer that would leave fewer than
:data:`FEWEST_ROWS`, and fewer than now, is refused. The reason names the steps that remove the
rows (for complete cases, which predictors' blanks remove how many of them), and the exits are the
ways back: a fill of the blanks first, then dropping the rule that removes them. A refusal refuses
the preview too, so the option says why before it is pressed.

The check reads the working table as it stands. An answer that also reshapes that table (a repair
that blanks values) can still leave too few rows, so the cohort stage refuses the same way
(:func:`cohort_refusal`) and no stage downstream is handed an empty frame.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from turbotab.core.decisions import (
    SLOTS,
    Refusal,
    SetExclusions,
    SetMissing,
    _ctx,
    _records_in,
    _roles_record_what_rode_along,
    _rule_without,
    _state,
    _store_of,
    as_rule,
    fold_onto,
    register_validator,
    state_after,
    validate,
)
from turbotab.core.methods.exposure_form import SPLINE_MIN_VALUES

# The fewest rows any analysis is built on: the floor the feature-wise exits already use
# (``methods.omics.featurewise_missing_exits``), the fewest values a spline's knots can be placed
# on (``Hmisc::rcspline.eval`` stops below it).
FEWEST_ROWS = SPLINE_MIN_VALUES
SHOWN_COLUMNS = 3  # predictors a reason names by their blanks; the rest are counted
COMPLETE_CASES = "complete_cases"


# ── what removes the rows ────────────────────────────────────────────────────


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _flow_inputs(state: Any, ingest: Mapping[str, Any]) -> tuple[str, list[str]]:
    """What the cohort's flow reads of ``state``: its rules as one comparable text, and the
    predictors whose blanks complete cases judge."""
    from turbotab.core.stages.rows import cohort_inputs, domain_of, landmark_of, repair_rules

    _, _, gappy = cohort_inputs(state, ingest)
    rules = json.dumps([getattr(state, "target", None),
                        [_dump(r) for r in getattr(state, "exclusions", None) or []],
                        [_dump(r) for r in repair_rules(state)],
                        landmark_of(state), list(domain_of(state))], sort_keys=True, default=str)
    return rules, gappy


def _dropped(steps: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in steps:
        out[str(s["key"])] = out.get(str(s["key"]), 0) + int(s["dropped"] or 0)
    return out


def _causes(after: Sequence[Mapping[str, Any]], before: Sequence[Mapping[str, Any]] | None = None
            ) -> list[dict[str, Any]]:
    """The steps of the ``after`` flow that remove rows, in flow order: each with the rows it
    removes beyond what it removed in ``before`` (when given), where that is any."""
    was = _dropped(before) if before is not None else {}
    out = []
    for s in after:
        key = str(s["key"])
        extra = int(s["dropped"] or 0) - was.get(key, 0)
        if key != "loaded" and extra > 0:
            out.append({"key": key, "label": str(s["label"]), "dropped": extra})
    return out


def blanks_in(mask: Any, columns: Sequence[str], rows: Any) -> list[tuple[str, int]]:
    """Each of ``columns`` blank in any of ``rows``, with how many of them: most blanks first, then
    in table order. ``mask`` holds the columns' blanks as NaN (``stages.rows._missing_mask``)."""
    if mask is None or not len(rows):
        return []
    present = [c for c in columns if c in mask.columns]
    blank = mask.loc[np.asarray(rows, dtype=np.int64), present].isna().sum(axis=0)
    counted = [(c, int(blank[c])) for c in present if int(blank[c])]
    order = {c: i for i, c in enumerate(columns)}
    return sorted(counted, key=lambda cn: (-cn[1], order[cn[0]]))


def _rows(n: int) -> str:
    return f"{n:,} row{'' if n == 1 else 's'}"


def _blank_clause(blanks: Sequence[tuple[str, int]], removed: int) -> str:
    """"each of them is blank in at least one of the 389 predictors with blanks, most often
    `mz_0003` (in 61 of them), `mz_0022` (55) and `mz_0121` (50)": ``blanks`` counted over exactly
    the ``removed`` rows complete cases remove, each of which is blank somewhere."""
    n = len(blanks)
    who = "it is" if removed == 1 else "each of them is"
    if n == 1:
        return f"{who} blank in `{blanks[0][0]}`"
    shown = [f"`{c}` (in {k:,} of them)" if i == 0 else f"`{c}` ({k:,})"
             for i, (c, k) in enumerate(blanks[:SHOWN_COLUMNS])]
    rest = n - len(shown)
    named = (", ".join(shown[:-1]) + f" and {shown[-1]}" if not rest
             else ", ".join(shown) + f" and {rest:,} more")
    return f"{who} blank in at least one of the {n:,} predictors with blanks, most often {named}"


def causes_sentence(causes: Sequence[Mapping[str, Any]], blanks: Sequence[tuple[str, int]],
                    would: bool = False) -> str:
    """What removes the rows, one clause per step: complete cases by the predictors' blanks in the
    rows they remove (:func:`blanks_in`), any other step by its own line of the flow. ``would``:
    said of an answer not recorded yet."""
    parts = []
    for cause in causes:
        n = int(cause["dropped"])
        if cause["key"] == COMPLETE_CASES:
            said = f"complete cases {'would remove' if would else 'remove'} {_rows(n)}"
            if blanks:
                said += f": {_blank_clause(blanks, n)}"
            parts.append(said)
        else:
            parts.append(f"“{cause['label']}” {'would remove' if would else 'removes'} {_rows(n)}")
    if not parts:
        return ""
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "."


def _leaves(n_kept: int, n_from: int, word: str) -> str:
    return f"{'none' if n_kept == 0 else f'{n_kept:,}'} of the {n_from:,} {word}"


def _ways(causes: Sequence[Mapping[str, Any]]) -> str:
    keys = {str(c["key"]).split(":")[0] for c in causes}
    ways = []
    if COMPLETE_CASES in keys:
        ways.append("fill the blanks instead of complete cases (the missing-values question)")
    if "exclusion" in keys:
        ways.append("drop or widen the rule that removes them (the eligibility question)")
    if not ways:
        ways.append("change the answer that removes them")
    said = ways[0] if len(ways) == 1 else f"{ways[0]}, or {ways[1]}"
    return said[:1].upper() + said[1:] + "."


# ── the cohort stage: a table that changed under recorded answers ────────────


def cohort_refusal(store: Any, state: Any, ingest: Mapping[str, Any],
                   steps: Sequence[Mapping[str, Any]], kept: Any) -> str | None:
    """Why the cohort leaves fewer rows than the design needs, for the cohort stage to fail with
    before any stage is handed them; None when it leaves enough. Complete cases name the
    predictors whose blanks remove the rows they remove."""
    from turbotab.core.stages.rows import _missing_mask, cohort_inputs, compute_cohort

    n_kept = int(len(kept))
    if n_kept >= FEWEST_ROWS:
        return None
    causes = _causes(steps)
    blanks: list[tuple[str, int]] = []
    if any(c["key"] == COMPLETE_CASES for c in causes):
        # Complete cases are the flow's last step: the rows they remove are those the flow keeps
        # without them, and not with them.
        _, reach, _ = compute_cohort(store, state.model_copy(update={"missing": None}), ingest)
        gone = np.setdiff1d(np.asarray(reach, dtype=np.int64), np.asarray(kept, dtype=np.int64))
        _, _, gappy = cohort_inputs(state, ingest)
        if len(gone) and gappy:
            blanks = blanks_in(_missing_mask(store, gappy, gone), gappy, gone)
    n_from = int(steps[0]["n"]) if steps else n_kept
    return (f"The answers recorded leave {_leaves(n_kept, n_from, 'rows in the table')} to "
            f"analyze, and the design needs at least {FEWEST_ROWS}. "
            f"{causes_sentence(causes, blanks)} {_ways(causes)}")


# ── the check, at recording time ─────────────────────────────────────────────


@lru_cache(maxsize=1)
def _reads() -> tuple[str, ...]:
    """The slots the analyzed rows are counted from: the cohort's, and the working table's it
    counts them on (the stage graph's own ``reads``)."""
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    graph = load_graph(GRAPH_FACTORY)
    return tuple(dict.fromkeys([*graph["cohort"].reads, *graph["working"].reads]))


def _after(decision: Any, ctx: Any, now: Any) -> Any:
    """The state recording ``decision`` would leave: the log folded with it, else (no log) the
    decision written onto the state. None when the log refuses it (its own refusal stands)."""
    if _records_in(ctx) is None:
        return fold_onto(now, decision)
    return state_after(decision, ctx)


def _pool(ctx: Any, state: Any, store: Any) -> Any:
    """The rows a check may read (a preview's pool): under prediction every row but the newest
    split's held-out ones, under inference every row (BLUEPRINT §12 ruling 3). None: every row."""
    if getattr(state, "purpose", None) == "inference":
        return None
    reader = _ctx(ctx, "sealed")
    try:
        sealed = reader() if callable(reader) else None
    except Exception:  # noqa: BLE001 - no split to read: none is held out
        sealed = None
    if sealed is None or not len(sealed):
        return None
    everything = np.arange(int(store.n_rows), dtype=np.int64)
    return np.setdiff1d(everything, np.asarray(sealed, dtype=np.int64), assume_unique=True)


def _answers_leave_rows_to_analyze(decision: Any, ctx: Any) -> None:
    """Refuse an answer that would leave fewer rows to analyze than the design needs
    (:data:`FEWEST_ROWS`) and fewer than now, naming what removes them, with the ways back."""
    from turbotab.core.stages.rows import cohort_flows

    if decision.kind != "revert" and decision.kind not in SLOTS:
        return
    now = _state(ctx)
    if now is None:
        return
    if decision.kind == "set_roles":  # what rides along unconfirmed, as it will be recorded
        decision = _roles_record_what_rode_along(decision, ctx)
    after = _after(decision, ctx, now)
    if after is None or all(getattr(now, s, None) == getattr(after, s, None) for s in _reads()):
        return  # nothing the rows are counted from changes
    store = _store_of(ctx)
    if store is None:
        return
    try:
        ingest = {"columns": [c.to_dict() for c in store.info().columns]}
        rules_now, gappy_now = _flow_inputs(now, ingest)
        rules_after, gappy_after = _flow_inputs(after, ingest)
    except Exception:  # noqa: BLE001 - a state the flow cannot read: the stage says why
        return
    if rules_now == rules_after and set(gappy_after) <= set(gappy_now):
        return  # nothing this answer changes can remove a row
    pool = _pool(ctx, now, store)
    # The third flow stops before complete cases: the rows that reach them after this answer.
    reach = after.model_copy(update={"missing": None})
    try:
        _, mask, [(before, kept_now), (steps, kept), (_, reached)] = cohort_flows(
            store, [now, after, reach], ingest, pool)
    except Exception:  # noqa: BLE001 - the columns cannot be read now: the stage says why
        return
    n_now, n_after = int(len(kept_now)), int(len(kept))
    if n_after >= FEWEST_ROWS or n_after >= n_now:
        return
    causes = _causes(steps, before)
    blanks: list[tuple[str, int]] = []
    if any(c["key"] == COMPLETE_CASES for c in causes):
        # The rows complete cases remove after this answer that the analysis has now, and which
        # predictors are blank in them.
        gone = np.setdiff1d(np.asarray(reached, dtype=np.int64), np.asarray(kept, dtype=np.int64))
        gone = np.intersect1d(gone, np.asarray(kept_now, dtype=np.int64))
        blanks = blanks_in(mask, gappy_after, gone)
    word = "rows analyzed now" if pool is None else "rows outside the held-out ones"
    message = (f"Recorded, this answer would leave {_leaves(n_after, n_now, word)}, and the design "
               f"needs at least {FEWEST_ROWS}. {causes_sentence(causes, blanks, would=True)} "
               f"{_ways(causes)}")
    raise Refusal("too_few_rows", message, exits=_exits(decision, causes, now, ctx))


def _exits(decision: Any, causes: Sequence[Mapping[str, Any]], now: Any, ctx: Any
           ) -> list[dict[str, Any]]:
    exits: list[dict[str, Any]] = []
    if any(c["key"] == COMPLETE_CASES for c in causes):
        # The columns this answer leaves out stay left out (a missing-values answer's own).
        keep = list(decision.drop_columns) if isinstance(decision, SetMissing) else None
        exits += fill_exits(now, ctx, keep=keep)
    if isinstance(decision, SetExclusions):
        for c in causes:
            parts = str(c["key"]).split(":")
            if parts[0] != "exclusion":
                continue
            i = int(parts[1])
            label = f"Drop the rule on `{as_rule(decision.rules[i]).column}`"
            if all(e["label"] != label for e in exits):
                exits.append({"label": label, "decision": _rule_without(decision, i)})
    exits.append({"label": "Keep the answers as they are", "decision": None})
    return exits


def fill_exits(state: Any, ctx: Any, keep: Sequence[str] | None = None, limit: int = 2
               ) -> list[dict[str, Any]]:
    """The missing-values answers that fill the blanks instead of dropping their rows, soundest
    first for the purpose (``methods.missing.methods_for``), each one the record would accept now:
    a fill the leash refuses is offered as the way forward its refusal names (non-detections
    filled censoring-aware, a single fill under inference recorded as a limitation), never past
    it. The columns left out (``keep``; default: the recorded answer's) stay left out."""
    from turbotab.core.methods.missing import BELOW_DETECTION_LABELS, METHODS, methods_for

    if keep is None:
        keep = list(getattr(getattr(state, "missing", None), "drop_columns", None) or [])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for method in methods_for(getattr(state, "purpose", None)):
        if method["key"] == "complete_case" or method["rung"] == "refused":
            continue
        body = {"kind": "set_missing", **method["decision"], "drop_columns": list(keep)}
        accepted = _accepted(body, ctx)
        if accepted is None:
            continue
        key = json.dumps(accepted, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        # Named by the method it is (the most specific whose fields it carries), which a refusal's
        # way forward may have changed.
        named = max((m for m in METHODS.values()
                     if all(accepted.get(k) == v for k, v in m.decision.items())),
                    key=lambda m: len(m.decision))
        what = named.label[:1].lower() + named.label[1:]
        below = accepted.get("below_detection")
        if below:
            words = BELOW_DETECTION_LABELS[below]
            what += f", values below detection {words[:1].lower()}{words[1:]}"
        if accepted.get("acknowledged") and not body.get("acknowledged"):
            what += ", recorded as a limitation"
        out.append({"label": f"Fill the blanks instead: {what}", "decision": accepted})
        if len(out) == limit:
            break
    return out


def _accepted(body: Mapping[str, Any], ctx: Any) -> dict[str, Any] | None:
    """``body`` as the record would accept it now, or the first way forward its refusal offers
    that the record would accept and that fills the blanks (a missing-values answer other than
    complete cases); None when neither is."""
    try:
        return validate(dict(body), ctx).model_dump(mode="json")
    except Refusal as refused:
        for e in refused.exits:
            d = e.get("decision")
            if not d or d.get("kind") != "set_missing" or d.get("strategy") == "complete_case":
                continue
            try:
                return validate(d, ctx).model_dump(mode="json")
            except Refusal:
                continue
    return None


def _register() -> None:
    for kind in [*SLOTS, "revert"]:
        register_validator(kind, _answers_leave_rows_to_analyze)


_register()

__all__ = ["FEWEST_ROWS", "blanks_in", "causes_sentence", "cohort_refusal", "fill_exits"]

"""Rows enough to analyze: no answer is recorded that would leave an analysis fewer rows than its
fit needs, or one value of the outcome, and no stage is handed fewer.

Found by the no-dead-end drive (INBOX): ``metabolomics_untargeted.csv`` with complete cases. The
features rode along unconfirmed when the missing-values answer was recorded, so their blanks
dropped no row then (BLUEPRINT §14.1); confirmed at the models question, they made 389 gappy
columns predictors, every row was blank in at least one of them, and the design stage failed on
scikit-learn's "Found array with 0 sample(s)" while the seal's step waited on the failed fit.

So a check is registered for every kind of answer (as ``sequence`` registers the order): it folds
the answer into the state and, when that changes what the cohort reads, counts the rows before and
after, over the rows a preview may read (every row but the held-out ones under prediction; every
row under inference, ruling 3). It counts every analysis: the primary, and each sensitivity
analysis on the rows its own rules keep. An answer is refused when it would leave an analysis

* fewer rows than :data:`FEWEST_ROWS`, and fewer than now (``too_few_rows``); or
* rows whose outcome takes one value, where it took two or more (``one_outcome_value``): a model
  has nothing to tell apart, and the fit failed on it (``IndexError``).

The reason names the steps that remove the rows (for complete cases, which predictors' blanks
remove how many of them), and the exits are the ways back, each one the record accepts: a fill of
the blanks first (never the fill already recorded), then dropping the rule that removes them, or
the rules together when no one of them is enough. A refusal refuses the preview too, so the option
says why before it is pressed.

An answer that rewrites the working table's values (a repair that blanks codes, a text column read
as numbers) is counted on the values it would leave (:func:`_rewritten`): the columns it rewrites
are read as the working stage will compute them, the same SQL on the same source rows. One that
changes the working table's rows (combining a unit's records, leaving reference rows out) is
counted on the table the working stage itself builds for it, in a folder of its own
(:func:`_rebuilt`), every row on both sides; analyzing each record as its own row is the way back
from a combination. Behind the record the cohort stage still refuses the same way
(:func:`cohort_refusal`), the design stage refuses one outcome value on the rows it is handed, and
no stage downstream is handed an empty frame.

Every exit another validator offers is held to the same count: a complete-case answer is offered
as a way out only where the record accepts it (:func:`accepted_exit`, read by
``decisions._accepted_ways``), and a sensitivity analysis is never offered an exit that leaves it
the primary analysis.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from turbotab.core.decisions import (
    SLOTS,
    ProjectState,
    Refusal,
    SetExclusions,
    SetMissing,
    SetSensitivity,
    _ctx,
    _records_in,
    _roles_record_what_rode_along,
    _state,
    _store_of,
    as_rule,
    fold_onto,
    parse_decision,
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
KEEP = {"label": "Keep the answers as they are", "decision": None}
# Set while an exit is tried (:func:`_accepted_exit`): a refusal then needs no exits of its own, so
# trying each rule's removal never branches into its own exits (k rules, never k! validations).
_PROBING: ContextVar[bool] = ContextVar("row_floor_probing", default=False)


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


def _ways(causes: Sequence[Mapping[str, Any]], question: str = "the eligibility question") -> str:
    keys = {str(c["key"]).split(":")[0] for c in causes}
    ways = []
    if COMPLETE_CASES in keys:
        ways.append("fill the blanks instead of complete cases (the missing-values question)")
    if "exclusion" in keys:
        ways.append(f"drop or widen the rule that removes them ({question})")
    if "reference" in keys:
        ways.append("keep the reference rows in the table")
    if not ways:
        ways.append("change the answer that removes them")
    said = ways[0] if len(ways) == 1 else f"{ways[0]}, or {ways[1]}"
    return said[:1].upper() + said[1:] + "."


def _say(value: Any) -> str:
    """An outcome value as a reason names it: ``0``, not ``0.0``."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return f"{int(value)}"
    return str(value)


def outcome_values(frame: Any, target: str | None, rows: Any) -> list[Any]:
    """The outcome's recorded values among ``rows``, at most two (enough to say whether it varies)."""
    if target is None or frame is None or target not in frame.columns or not len(rows):
        return []
    seen: list[Any] = []
    for v in frame.loc[np.asarray(rows, dtype=np.int64), target].dropna():
        if not any(v == s for s in seen):
            seen.append(v)
            if len(seen) == 2:
                break
    return seen


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


def one_value_refusal(values: Sequence[Any], target: str, n_rows: int, rows_word: str) -> str | None:
    """Why rows whose outcome takes one value cannot be modeled, for the design stage to fail with
    (the fit failed on ``IndexError`` there); None when the outcome varies or no row is left."""
    if n_rows == 0 or len(values) != 1:
        return None
    return (f"`{target}` is `{_say(values[0])}` in every one of the {n_rows:,} {rows_word}, so "
            f"the models have no other value of the outcome to learn from. The row flow says "
            f"which answers removed the rows with other values.")


def thin_analysis(frame: Any, target: str, rows: Any, rows_word: str) -> str | None:
    """Why a sensitivity analysis is not fit, for the sensitivity stage to say beside it: its rules
    leave fewer rows than a fit needs, or one value of the outcome; None when it can be fit."""
    n = int(len(rows))
    if n < FEWEST_ROWS:
        return (f"This analysis keeps {'no' if n == 0 else f'{n:,}'} {rows_word}, and a fit needs "
                f"at least {FEWEST_ROWS}: widen its rules, or leave it out.")
    values = outcome_values(frame, target, rows)
    if len(values) == 1:
        return (f"`{target}` is `{_say(values[0])}` in every one of this analysis's {n:,} "
                f"{rows_word}, so there is no other value of the outcome to learn from: widen its "
                f"rules, or leave it out.")
    return None


# ── an answer that rewrites the working table's values ───────────────────────


@lru_cache(maxsize=1)
def _graph_reads() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(the slots the cohort and the working table read, the working table's alone): the stage
    graph's own ``reads``."""
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    graph = load_graph(GRAPH_FACTORY)
    working = tuple(graph["working"].reads)
    return tuple(dict.fromkeys([*graph["cohort"].reads, *working])), working


def _reads() -> tuple[str, ...]:
    """The slots an analysis's rows are counted from: the cohort's, the working table's it counts
    them on, and the sensitivity analyses' own rules."""
    return (*_graph_reads()[0], "sensitivity")


class _Rewritten:
    """The working table with the columns an answer rewrites read as it would leave them (the same
    row ids), every other column as it stands: what a cohort flow reads of a store."""

    def __init__(self, store: Any, values: Any):
        self.store, self.values = store, values
        self.n_rows = int(store.n_rows)

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys([*self.store.columns, *self.values.columns]))

    def materialize(self, columns: Sequence[str] | None = None, row_ids: Any = None) -> Any:
        import pandas as pd

        columns = self.columns if columns is None else list(columns)
        own = [c for c in columns if c not in self.values.columns]
        if own:
            frame = self.store.materialize(own, row_ids)
        else:
            ids = (np.arange(self.n_rows, dtype=np.int64) if row_ids is None
                   else np.asarray(row_ids, dtype=np.int64))
            frame = pd.DataFrame(index=pd.Index(ids, name="row_id"))
        for c in columns:
            if c in self.values.columns:
                frame[c] = self.values[c].reindex(frame.index).to_numpy()
        return frame[columns]


def _rewritten(now: Any, after: Any, ctx: Any, store: Any, ingest: Mapping[str, Any]
               ) -> tuple[Any, dict[str, Any]] | None:
    """``(store, ingest)`` over the working table as ``after`` would leave it, when it rewrites the
    values of a column an analysis reads (a repair that blanks codes, a text column read as
    numbers, a derived log outcome): those columns are evaluated by the working stage's own SQL
    (``stages.working``: the repairs' expressions, the confirmed amounts, the derived columns) on
    the source rows the working table's rows are. None when it rewrites none, or when its rows
    cannot be foreseen here (rows combined per unit, reference rows that change, no fresh working
    table): the cohort stage says so then."""
    from turbotab.core.repairs import evaluate
    from turbotab.core.stages.rows import cohort_inputs
    from turbotab.core.stages.working import (
        ROW_MAP, TABLE, repair_expressions, row_local_additions, text_amounts,
    )

    if all(getattr(now, s, None) == getattr(after, s, None) for s in _graph_reads()[1]):
        return None
    bundle = _ctx(ctx, "bundle")
    if not callable(bundle):
        return None
    oriented, working = bundle("oriented"), bundle("working")
    files = getattr(oriented, "files", None) or {}
    data = getattr(working, "data", None)
    if TABLE not in files or not isinstance(data, Mapping) or data.get("aggregation"):
        return None
    source = Path(files[TABLE])
    names = [str(c["name"]) for c in getattr(oriented, "data", {}).get("columns", [])]
    reader = _ctx(ctx, "artifact")
    findings = reader("findings") if callable(reader) else None

    def expressions(state: Any) -> tuple[dict[str, str], Any]:
        exprs = repair_expressions(findings, getattr(state, "findings", None))
        exprs.update(text_amounts(state, source, names, exprs))
        _, derived, rules = row_local_additions(state, findings, names, exprs)
        return {**exprs, **derived}, rules

    x_now, rules_now = expressions(now)
    x_after, rules_after = expressions(after)
    if rules_now != rules_after:
        return None  # reference rows leave or return: the working table's rows change
    needed, preds, _ = cohort_inputs(after, ingest)
    reads = set(needed) | set(preds)
    changed = [c for c in dict.fromkeys([*x_now, *x_after])
               if x_now.get(c) != x_after.get(c) and c in reads]
    if not changed:
        return None
    import pandas as pd

    ids = (pd.read_parquet(getattr(working, "files", {})[ROW_MAP])
           if data.get("row_map") not in (None, "identity")
           else pd.DataFrame({"row_id": np.arange(int(store.n_rows), dtype=np.int64),
                              "source_row_id": np.arange(int(store.n_rows), dtype=np.int64)}))
    read = evaluate(source, changed, x_after, ids["source_row_id"].to_numpy(dtype=np.int64))
    values = read.reindex(ids["source_row_id"].to_numpy(dtype=np.int64))
    values.index = pd.Index(ids["row_id"].to_numpy(dtype=np.int64), name="row_id")
    columns = [dict(c) for c in ingest.get("columns", [])]
    known = {str(c["name"]): c for c in columns}
    for c in changed:
        blank = int(values[c].isna().sum())
        if c in known:
            known[c]["n_missing"] = blank
        else:
            columns.append({"name": c, "n_missing": blank})
    return _Rewritten(store, values), {**ingest, "columns": columns}


class _Rebuilt:
    """The working table as an answer that changes its rows would leave it (rows combined per
    unit, reference rows that leave or return, pooled QCs corrected and then left out, a combined
    table's values rewritten), built by the working stage itself in a folder of its own: the store
    over it, its description (the cohort's ``ingest``), and what changes (``"units"``,
    ``"reference"`` or ``"values"``)."""

    def __init__(self, folder: Path, store: Any, info: dict[str, Any], how: str):
        self.folder, self.store, self.info, self.how = folder, store, info, how

    def close(self) -> None:
        import shutil

        try:
            self.store.close()
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)


def _rebuilt(now: Any, after: Any, ctx: Any, store: Any) -> _Rebuilt | None:
    """The working table ``after`` would build, when its rows are not the rows of the table now:
    the units its records combine into, or the reference rows that leave (WP18, RO-13); or, on a
    table whose rows are combined, the values a repair rewrites (each unit's row recombined). None
    when the rows stay as they are (a value rewrite is :func:`_rewritten`'s), or when the stage
    could not build it (it says why itself)."""
    import tempfile

    from turbotab.core.datastore import DataStore
    from turbotab.core.graph import StageContext
    from turbotab.core.methods.qc_drift import reference_plan
    from turbotab.core.stages.working import (
        TABLE, StructureError, aggregation_plan, repair_expressions, row_local_additions,
        text_amounts, working_stage,
    )

    if all(getattr(now, s, None) == getattr(after, s, None) for s in _graph_reads()[1]):
        return None
    bundle = _ctx(ctx, "bundle")
    if not callable(bundle):
        return None
    oriented = bundle("oriented")
    files = getattr(oriented, "files", None) or {}
    if TABLE not in files:
        return None
    findings, structure = bundle("findings"), bundle("structure")
    source = Path(files[TABLE])
    names = [str(c["name"]) for c in getattr(oriented, "data", {}).get("columns", [])]
    shape = getattr(structure, "data", structure)

    def rows_of(state: Any) -> tuple[Any, ...]:
        exprs = repair_expressions(findings, getattr(state, "findings", None))
        exprs.update(text_amounts(state, source, names, exprs))
        _, derived, rules = row_local_additions(state, findings, names, exprs)
        plan = aggregation_plan(state, set(names) | set(derived), shape)
        return plan, rules, reference_plan(state), (exprs, derived) if plan is not None else None

    try:
        before, later = rows_of(now), rows_of(after)
    except StructureError:
        return None  # the working stage refuses it in its own words
    if before == later:
        return None
    how = ("units" if before[0] != later[0] and later[0] is not None
           else "reference" if before[1:3] != later[1:3] else "values")
    folder = Path(tempfile.mkdtemp(prefix="tt-row-floor-"))
    try:
        built = working_stage(StageContext(
            project_id="row-floor", state=after,
            inputs={"oriented": oriented, "findings": findings, "structure": structure},
            paths={"data": str(folder / "raw.parquet")}, settings={}))
        table = DataStore(Path(built.files[TABLE]), int(getattr(store, "memory_budget_bytes", 1 << 30)))
        return _Rebuilt(folder, table, dict(built.data), how)
    except BaseException:
        import shutil

        shutil.rmtree(folder, ignore_errors=True)
        raise


# ── the check, at recording time ─────────────────────────────────────────────


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


def _named(state: Any) -> dict[str, Any]:
    """The sensitivity analyses the sensitivity stage fits, by label (one whose rule reads the
    outcome is refused there, on its own line)."""
    target = getattr(state, "target", None)
    return {a.label: a for a in getattr(state, "sensitivity", None) or []
            if target not in {c for r in a.rules for c in as_rule(r).reads()}}


def _analyses(now: Any, after: Any) -> list[tuple[str | None, Any, Any]]:
    """``(label, the analysis now, the analysis after)``: the primary first (label None), then each
    sensitivity analysis on the rows its own rules keep (None now: new, or its rules changed)."""
    out: list[tuple[str | None, Any, Any]] = [(None, now, after)]
    was = _named(now)
    for label, a in _named(after).items():
        b = was.get(label)
        out.append((label,
                    now.model_copy(update={"exclusions": list(b.rules)})
                    if b is not None and b.rules == a.rules else None,
                    after.model_copy(update={"exclusions": list(a.rules)})))
    return out


def _answers_leave_rows_to_analyze(decision: Any, ctx: Any) -> None:
    """Refuse an answer that would leave an analysis fewer rows than its fit needs
    (:data:`FEWEST_ROWS`) and fewer than now, or one value of the outcome where it had two,
    naming what removes the rows, with the ways back."""
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
    except Exception:  # noqa: BLE001 - a table that cannot be described: the stage says why
        return
    try:
        rewritten = _rewritten(now, after, ctx, store, ingest)
    except Exception:  # noqa: BLE001 - values that cannot be foreseen: counted as they stand
        rewritten = None
    rebuilt = None
    if rewritten is None:
        try:
            rebuilt = _rebuilt(now, after, ctx, store)
        except Exception:  # noqa: BLE001 - a table the working stage cannot build: it says why
            rebuilt = None
    try:
        _count(decision, ctx, now, after, store, ingest, rewritten, rebuilt)
    finally:
        if rebuilt is not None:
            rebuilt.close()


def _count(decision: Any, ctx: Any, now: Any, after: Any, store: Any, ingest: Mapping[str, Any],
           rewritten: Any, rebuilt: _Rebuilt | None) -> None:
    """The count itself: each analysis's rows now and after the answer, refused as the caller
    says. ``rebuilt``: the answer changes the working table's rows, so the rows after are counted
    on the table it would build, and every row is counted on both sides (a held-out row of the
    table now names no row of that one; the cohort stage counts them so too)."""
    from turbotab.core.stages.rows import cohort_flows, cohort_inputs

    if rebuilt is not None:
        store_after, ingest_after = rebuilt.store, rebuilt.info
    else:
        store_after, ingest_after = rewritten or (store, ingest)
    try:
        checked = []
        for label, a_now, a_after in _analyses(now, after):
            if a_now is not None and rewritten is None and rebuilt is None:
                rules_now, gappy_now = _flow_inputs(a_now, ingest)
                rules_after, gappy_after = _flow_inputs(a_after, ingest_after)
                if rules_now == rules_after and set(gappy_after) <= set(gappy_now):
                    continue  # nothing this answer changes can remove a row of it
            checked.append((label, a_now, a_after))
    except Exception:  # noqa: BLE001 - a state the flow cannot read: the stage says why
        return
    if not checked:
        return
    pool = None if rebuilt is not None else _pool(ctx, now, store)
    # One flow per state, read once per table: each analysis now, and after (with the rows that
    # reach complete cases, and a new analysis's rows before its rules).
    nows = [a for _, a, _ in checked if a is not None]
    afters = [after]
    for _, a_now, a_after in checked:
        afters += [a_after, a_after.model_copy(update={"missing": None})]
        if a_now is None:
            afters.append(a_after.model_copy(update={"exclusions": []}))
    try:
        frame_now, _, flows_now = (cohort_flows(store, nows, ingest, pool) if nows
                                   else (None, None, []))
        frame, mask, flows = cohort_flows(store_after, afters, ingest_after, pool)
    except Exception:  # noqa: BLE001 - the columns cannot be read now: the stage says why
        return
    n_primary = int(len(flows[0][1]))
    flows_now, flows = list(flows_now), list(flows[1:])
    for label, a_now, a_after in checked:
        before, kept_now = flows_now.pop(0) if a_now is not None else (None, None)
        (steps, kept), (_, reached) = flows.pop(0), flows.pop(0)
        if a_now is None:
            before, kept_now = flows.pop(0)  # the rows its rules screen
        n_after = int(len(kept))
        n_now = None if a_now is None else int(len(kept_now))
        target = getattr(after, "target", None)
        values = outcome_values(frame, target, kept)
        if n_after < FEWEST_ROWS:
            if n_now is not None and n_after >= n_now:
                continue
            code = "too_few_rows"
        elif len(values) == 1 and (n_now is None
                                   or len(outcome_values(frame_now, target, kept_now)) > 1):
            code = "one_outcome_value"
        else:
            continue
        causes = _causes(steps, before)
        blanks: list[tuple[str, int]] = []
        if any(c["key"] == COMPLETE_CASES for c in causes):
            # The rows complete cases remove after this answer that the analysis has before it,
            # and which predictors are blank in them (on a rebuilt table its rows are new ones).
            gone = np.setdiff1d(np.asarray(reached, dtype=np.int64), np.asarray(kept, dtype=np.int64))
            if rebuilt is None:
                gone = np.intersect1d(gone, np.asarray(kept_now, dtype=np.int64))
            _, _, gappy = cohort_inputs(a_after, ingest_after)
            blanks = blanks_in(mask, gappy, gone)
        said = causes_sentence(causes, blanks, would=True)
        where = "outside the held-out ones" if pool is not None else ""
        question = "the eligibility question" if label is None else "the sensitivity question"
        ways = _ways(causes, question)
        if rebuilt is not None and rebuilt.how != "values" and label is None:
            built = (f"{_rows(n_after)}, one per unit," if rebuilt.how == "units"
                     else f"{_rows(n_after)} once the reference rows have left the table,")
            if code == "too_few_rows":
                lead = (f"Recorded, this answer would leave {built} where {int(n_now or 0):,} "
                        f"rows are analyzed now, and the design needs at least {FEWEST_ROWS}.")
            else:
                lead = (f"Recorded, this answer would leave {built} and `{target}` is "
                        f"`{_say(values[0])}` in every one of them: the models need rows with "
                        f"another value of the outcome to learn from.")
            if not causes:
                ways = ("Analyze each record as its own row (the unit question), or keep the "
                        "answers as they are." if rebuilt.how == "units"
                        else "Keep the reference rows in the table, or keep the answers as they are.")
        elif label is None and code == "too_few_rows":
            word = "rows analyzed now" if pool is None else "rows outside the held-out ones"
            lead = (f"Recorded, this answer would leave {_leaves(n_after, int(n_now or 0), word)}, "
                    f"and the design needs at least {FEWEST_ROWS}.")
        elif label is None:
            lead = (f"Recorded, this answer would leave {_rows(n_after)}{' ' + where if where else ''}, "
                    f"and `{target}` is `{_say(values[0])}` in every one of them: the models need "
                    f"rows with another value of the outcome to learn from.")
        elif code == "too_few_rows":
            lead = (f"Recorded, this answer would leave the sensitivity analysis “{label}” "
                    f"{'no rows' if n_after == 0 else _rows(n_after)}"
                    f"{' ' + where if where else ''}, against {n_primary:,} in the primary "
                    f"analysis, and its fit needs at least {FEWEST_ROWS}.")
        else:
            lead = (f"Recorded, this answer would leave the sensitivity analysis “{label}” "
                    f"{_rows(n_after)}{' ' + where if where else ''}, and `{target}` is "
                    f"`{_say(values[0])}` in every one of them: its fit needs rows with another "
                    f"value of the outcome to learn from.")
        message = " ".join(p for p in (lead, said, ways) if p)
        raise Refusal(code, message, exits=[KEEP] if _PROBING.get()
                      else _exits(decision, causes, label, now, ctx,
                                  units=rebuilt is not None and rebuilt.how == "units"))


def _accepted_exit(decision: Any, ctx: Any) -> dict[str, Any] | None:
    """``decision`` as the record would accept it now, or None: an exit is offered only when
    taking it is accepted (the verifier's rule exit that was itself refused)."""
    probing = _PROBING.set(True)
    try:
        return validate(_dump(decision), ctx).model_dump(mode="json")
    except Refusal:
        return None
    finally:
        _PROBING.reset(probing)


accepted_exit = _accepted_exit  # for the other validators' exits (``decisions._accepted_ways``)


def _same_rules(a: Sequence[Any], b: Sequence[Any]) -> bool:
    """Whether two lists of eligibility rules keep the same rows by the same rules (in any order)."""
    def key(rules: Sequence[Any]) -> list[str]:
        return sorted(json.dumps(_dump(as_rule(r)), sort_keys=True, default=str) for r in rules)

    return key(a) == key(b)


def _rule_exits(rules: Sequence[Any], causes: Sequence[Mapping[str, Any]], make: Any,
                ctx: Any, where: str = "", primary: Sequence[Any] | None = None
                ) -> list[dict[str, Any]]:
    """Dropping each rule that removes rows, named by its own line of the flow, when the record
    accepts the answer without it; when no one rule is enough, the rules together. ``primary``: a
    sensitivity analysis's exits, never one whose rules are left the primary's (the analysis would
    be the primary analysis again: leaving it out says that)."""
    from turbotab.core.stages.rows import rule_label

    def offered(kept: list[Any]) -> bool:
        return primary is None or not _same_rules(kept, primary)

    causing = list(dict.fromkeys(int(str(c["key"]).split(":")[1]) for c in causes
                                 if str(c["key"]).startswith("exclusion:")))
    out: list[dict[str, Any]] = []
    for i in causing:
        label = f"Drop the rule{where}: {rule_label(rules[i])}"
        if any(e["label"] == label for e in out):
            continue  # the same rule twice: dropping either does the same
        kept = [r for j, r in enumerate(rules) if j != i]
        accepted = _accepted_exit(make(kept), ctx) if offered(kept) else None
        if accepted is not None:
            out.append({"label": label, "decision": accepted})
    if not out and len(causing) > 1:
        kept = [r for j, r in enumerate(rules) if j not in causing]
        accepted = _accepted_exit(make(kept), ctx) if offered(kept) else None
        if accepted is not None:
            named = [rule_label(rules[i]) for i in causing]
            out.append({"label": f"Drop the rules{where}: {', '.join(named[:-1])} and {named[-1]}",
                        "decision": accepted})
    return out


def _exits(decision: Any, causes: Sequence[Mapping[str, Any]], label: str | None, now: Any,
           ctx: Any, units: bool = False) -> list[dict[str, Any]]:
    """The ways back, each one the record accepts: a fill, a rule dropped, the analysis left out;
    ``units``: the answer combines each unit's records into one row, and analyzing each record as
    its own row keeps them."""
    from turbotab.core.decisions import SetUnit

    exits: list[dict[str, Any]] = []
    if units and label is None:
        apart = _accepted_exit(SetUnit(unit="row"), ctx)
        if apart is not None:
            exits.append({"label": "Analyze each record as its own row", "decision": apart})
    if any(c["key"] == COMPLETE_CASES for c in causes):
        # The columns this answer leaves out stay left out (a missing-values answer's own).
        keep = list(decision.drop_columns) if isinstance(decision, SetMissing) else None
        exits += fill_exits(now, ctx, keep=keep)
    if isinstance(decision, SetExclusions) and label is None:
        exits += _rule_exits(decision.rules, causes, lambda rules: SetExclusions(rules=rules), ctx)
    if isinstance(decision, SetSensitivity) and label is not None:
        i = next(k for k, a in enumerate(decision.analyses) if a.label == label)
        mine = decision.analyses[i]

        def without(rules: list[Any]) -> SetSensitivity:
            analyses = list(decision.analyses)
            analyses[i] = mine.model_copy(update={"rules": rules})
            return SetSensitivity(analyses=analyses)

        exits += _rule_exits(mine.rules, causes, without, ctx, where=f" from “{label}”",
                             primary=list(getattr(now, "exclusions", None) or []))
        left = _accepted_exit(SetSensitivity(analyses=[a for k, a in enumerate(decision.analyses)
                                                       if k != i]), ctx)
        if left is not None:
            exits.append({"label": f"Leave out the analysis “{label}”", "decision": left})
    exits.append(dict(KEEP))
    return exits


def fill_exits(state: Any, ctx: Any, keep: Sequence[str] | None = None, limit: int = 2
               ) -> list[dict[str, Any]]:
    """The missing-values answers that fill the blanks instead of dropping their rows, soundest
    first for the purpose (``methods.missing.methods_for``), each one the record would accept now
    and none the answer already recorded (taking it would change nothing): a fill the leash
    refuses is offered as the way forward its refusal names (non-detections filled
    censoring-aware, a single fill under inference recorded as a limitation), never past it. The
    columns left out (``keep``; default: the recorded answer's) stay left out."""
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
        if accepted is None or _changes_nothing(state, accepted):
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


def _changes_nothing(state: Any, decision: Mapping[str, Any]) -> bool:
    """Whether recording ``decision`` would leave ``state`` as it is (the answer already recorded)."""
    if not isinstance(state, ProjectState):
        return False
    return fold_onto(state, parse_decision(dict(decision))) == state


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

__all__ = ["FEWEST_ROWS", "accepted_exit", "blanks_in", "causes_sentence", "cohort_refusal",
           "fill_exits", "one_value_refusal", "outcome_values"]

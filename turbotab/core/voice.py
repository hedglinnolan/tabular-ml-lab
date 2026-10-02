"""The app's words: the sentence each decision records, and the text rules every surface keeps.

DESIGN_LANGUAGE §06: decisions are past tense and exact — a count, a verb, no adjectives. A sentence
that could not appear in a methods section is not a decision sentence. Backticks mark data values
(column names, levels, counts, cut-offs); the client renders them as mono chips.

:func:`sentence_for` is called once, when a decision is recorded (M1_CONTRACT §2), and the sentence
is stored on the record: the Record quotes it, it never recomposes it. One function per decision
kind, registered — adding a kind adds a function, never a branch.

``ctx`` is whatever the recorder knows about the project, as a mapping or an object. Every key is
optional; a sentence says only what its context can support and never prints a placeholder:

    columns          column names, or column dicts/objects with ``name`` (and ``n_missing``)
    n_rows           rows in the table
    records          the decision log so far (``DecisionRecord`` or dicts), for ``revert``
    frame            a pandas frame with the columns an exclusion names (and the target)
    datastore        a ``DataStore``: the columns an exclusion names are read from it
    exclusion_counts rows each exclusion rule removes, in order (overrides counting)
    n_cohort         rows entering the split (for its sentence)
    n_complete       rows left after complete cases, with ``n_before`` the rows before
    repeats          ``{column, n_units, max_rows_per_unit}`` when the identifier repeats
    detected_task    the task detection's answer: ``set_task`` names an override, and the split
                     and model sentences use it when no task was answered
    model_labels     ``{family key: label}``, overriding the built-in labels

M2 keys (the opening sequence, the seal and findings; each optional like the rest):

    seal_plan        the ``seal_plan`` artifact: the seal's basis and chronological draw (split)
    levels           the outcome's distinct levels (``set_event`` names the one coded 0)
    n_units          distinct values of the unit's identifier (grain, aggregation), with
                     ``rows_per_unit`` the most rows one unit has; counted from ``datastore`` when
                     neither is given
    n_holdout        the held-out rows the seal opens
    finding          the finding a disposition names: ``{id, title, summary, affected_columns}``
    repair           the repair option applied: ``{key, label, consequence, row_local}`` (the
                     repair registry's); a family may register its own sentence with
                     :func:`register_repair_sentence`
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

log = logging.getLogger(__name__)

# ── text rules ───────────────────────────────────────────────────────────────

_TICKED = re.compile(r"`[^`]*`")


def words(text: str | None) -> int:
    """Words as the budgets count them: whitespace-separated tokens, chips included."""
    return len(str(text or "").split())


def tick(value: Any) -> str:
    """A data value as the client shows it: in backticks."""
    return f"`{value}`"


def number(value: Any) -> str:
    """A data value (a cut-off, a level) as written: ``500.0`` → ``500``, ``3.25`` → ``3.25``."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "missing"
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.6g}"


def count(n: int) -> str:
    """A count, with thousands separators, in backticks."""
    return tick(f"{int(n):,}")


def plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def listing(items: Sequence[Any], limit: int = 4, *, ticked: bool = True) -> str:
    """``a``, ``a and b``, ``a, b and c``, or ``a, b, c and 3 more``."""
    shown = [tick(i) if ticked else str(i) for i in items]
    if len(shown) > limit:
        rest = len(shown) - (limit - 1)
        shown = shown[: limit - 1] + [f"{rest} more"]
    if not shown:
        return ""
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_ITALIC = re.compile(r"(?<![\w*`])\*(?![\s*])([^*\n]+?)(?<![\s*])\*(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_LEADERS = re.compile(r"(?m)^\s*(?:#{1,6}\s+|>\s+|[-*]\s+(?=\S))")


def strip_markdown(text: str) -> str:
    """Raw markdown the client would print literally: bold, italics, links, headings, quotes."""
    parts = re.split(r"(`[^`]*`)", str(text))
    out = []
    for i, part in enumerate(parts):
        if i % 2:  # a data value: leave it exactly as it is
            out.append(part)
            continue
        part = _BOLD.sub(lambda m: m.group(1) or m.group(2), part)
        part = _ITALIC.sub(r"\1", part)
        part = _LINK.sub(r"\1", part)
        part = _LEADERS.sub("", part)
        part = part.replace("⚠", "").replace("★", "")
        out.append(part)
    return "".join(out)


_TERMINAL = re.compile(r"[.?!][)\]\"'”’]*$")


def finish(text: str | None, *, terminal: bool = True) -> str:
    """One sentence-shaped string: no raw markdown, no doubled punctuation, one terminal mark.

    ``terminal=False`` is for labels (a lever, an option): they end without punctuation.
    Paragraph breaks (blank lines) are kept; each paragraph is finished on its own.
    """
    raw = strip_markdown(str(text or ""))
    paragraphs = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    done = [_finish_paragraph(p, terminal) for p in paragraphs]
    return "\n\n".join(p for p in done if p)


def _finish_paragraph(text: str, terminal: bool) -> str:
    parts = re.split(r"(`[^`]*`)", text)
    for i in range(0, len(parts), 2):
        p = re.sub(r"\s+", " ", parts[i])
        p = p.replace("...", "…")
        p = re.sub(r"\s+([,.;:!?])", r"\1", p)
        p = re.sub(r"([,;])(?=[A-Za-z(])", r"\1 ", p)
        p = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", p)
        p = re.sub(r"[,;:]\s*\.", ".", p)
        p = re.sub(r"(?<!\.)\.\.(?!\.)", ".", p)
        p = re.sub(r"([?!])\1+", r"\1", p)
        p = re.sub(r"\(\s+", "(", p)
        p = re.sub(r"\s+\)", ")", p)
        parts[i] = p
    text = "".join(parts).strip()
    if not text:
        return ""
    if not terminal:
        return re.sub(r"[\s.;:,]+$", "", text)
    text = re.sub(r"[\s,;:—–-]+$", "", text)
    if text.endswith("…"):
        text = text[:-1].rstrip() + "."
    if not _TERMINAL.search(text):
        text += "."
    return text


_PLACEHOLDER = re.compile(r"\{[A-Za-z_][\w.]*\}|\{\}")
_MACHINERY = (
    (re.compile(r"\bNone\b(?! of\b)"), "None"),  # Python's None; English "None of them" is fine
    (re.compile(r"\bnan\b|\bNaN\b"), "NaN"),
    (re.compile(r"\[object"), "[object"),
    (re.compile(r"\*\*|__"), "raw markdown"),
)


def machinery(text: str | None) -> list[str]:
    """What in ``text`` is the program talking instead of the app: None, NaN, placeholders, markdown."""
    text = str(text or "")
    outside = _TICKED.sub(" ", text)
    found = [name for pattern, name in _MACHINERY if name and pattern.search(outside)]
    if _PLACEHOLDER.search(text):
        found.append("{placeholder}")
    if text.count("`") % 2:
        found.append("unbalanced backtick")
    return found


# ── context ──────────────────────────────────────────────────────────────────

def _get(ctx: Any, name: str, default: Any = None) -> Any:
    if ctx is None:
        return default
    if isinstance(ctx, Mapping):
        value = ctx.get(name, default)
    else:
        value = getattr(ctx, name, default)
    return default if value is None else value


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _column_entry(ctx: Any, column: str) -> Any:
    for entry in _get(ctx, "columns", ()) or ():
        if isinstance(entry, str):
            if entry == column:
                return {"name": entry}
        elif _attr(entry, "name") == column:
            return entry
    return None


def _n_rows(ctx: Any) -> int | None:
    n = _get(ctx, "n_rows")
    if n is None:
        store = _get(ctx, "datastore")
        try:
            n = store.n_rows if store is not None else None
        except Exception:  # a sentence never fails a decision
            n = None
    if n is None:
        frame = _get(ctx, "frame")
        n = len(frame) if frame is not None else None
    return int(n) if n is not None else None


# ── sentences ────────────────────────────────────────────────────────────────

SentenceFn = Callable[[Any, Any, Any], str]
_SENTENCES: dict[str, SentenceFn] = {}


def register_sentence(kind: str) -> Callable[[SentenceFn], SentenceFn]:
    def wrap(fn: SentenceFn) -> SentenceFn:
        _SENTENCES[kind] = fn
        return fn
    return wrap


def sentence_for(decision: Any, state_before: Any = None, ctx: Any = None) -> str:
    """The one publishable sentence this decision records (backticks around data values).

    ``state_before`` is the ProjectState before the decision (None: nothing recorded yet).
    Never raises for a missing context: it says less instead.
    """
    from turbotab.core.decisions import ProjectState, parse_decision

    decision = parse_decision(decision)
    state = state_before if state_before is not None else ProjectState()
    fn = _SENTENCES.get(decision.kind)
    if fn is None:
        return finish(f"A {decision.kind.replace('_', ' ')} decision was recorded")
    try:
        return finish(fn(decision, state, ctx))
    except Exception:  # pragma: no cover - a sentence must never block a decision
        log.exception("sentence_for(%s) failed; recording the bare sentence", decision.kind)
        return finish(_SENTENCES[decision.kind](decision, state, None))


# M0 kinds — the same words the M0 Record used, now authored on the server.

_PURPOSE_CLAUSE = {
    "prediction": "models are judged on rows they never saw",
    "inference": "associations are estimated with their uncertainty",
}


@register_sentence("set_lens")
def _set_lens(d: Any, state: Any, ctx: Any) -> str:
    noun = plural(len(d.lenses), "lens", "lenses")
    return f"The table was read through the {listing(d.lenses, limit=5)} {noun}"


def _outcome_unit(ctx: Any, column: str) -> str | None:
    """The outcome's unit, from ``ctx["outcome_unit"]``, its name, or the clinical pack on its
    values (``frame`` or ``datastore``); None when nothing says."""
    from turbotab.core.units import from_name, outcome_unit

    given = _get(ctx, "outcome_unit")
    if given:
        return str(given)
    if from_name(column):
        return from_name(column)
    values = None
    frame, store = _get(ctx, "frame"), _get(ctx, "datastore")
    try:
        if frame is not None and column in frame.columns:
            values = frame[column]
        elif store is not None and column in store.columns:
            values = store.materialize([column])[column]
    except Exception:  # a sentence never fails a decision
        values = None
    if values is None or values.dtype.kind not in "iuf" or values.nunique() <= 2:
        return None  # a class label has no unit
    return outcome_unit(column, values)[0]


@register_sentence("set_target")
def _set_target(d: Any, state: Any, ctx: Any) -> str:
    text = f"{tick(d.column)} was chosen as the outcome"
    unit = _outcome_unit(ctx, d.column)
    if unit:
        text += f", in {'percent' if unit == '%' else unit}"
    entry = _column_entry(ctx, d.column)
    n = _n_rows(ctx)
    missing = _attr(entry, "n_missing") if entry is not None else None
    if n and missing is not None:
        measured = n - int(missing)
        if measured == n:
            text += f"; it is measured on all {count(n)} rows"
        else:
            text += f"; it is measured on {count(measured)} of {count(n)} rows"
    return text


@register_sentence("set_task")
def _set_task(d: Any, state: Any, ctx: Any) -> str:
    text = f"{tick(d.column)} was modeled as a {tick(d.task)} task"
    detected = _get(ctx, "detected_task")
    if detected and detected != d.task:
        text += f", overriding the detected {tick(detected)}"
    return text


@register_sentence("set_purpose")
def _set_purpose(d: Any, state: Any, ctx: Any) -> str:
    return f"The analysis was declared for {tick(d.purpose)}: {_PURPOSE_CLAUSE[d.purpose]}"


# revert — says what the slot holds again, when the log is at hand.

_SLOT_SUBJECT = {
    "lens": "the lens",
    "target": "the outcome",
    "task": "the task",
    "purpose": "the purpose",
    "roles": "the column roles",
    "energy_adjustment": "the energy adjustment",
    "exclusions": "the exclusions",
    "missing": "the handling of missing values",
    "split": "the split",
    "models": "the model families",
    "substitution": "the substitution",
    "orientation": "the table's orientation",
    "event": "the event level",
    "grain": "the grain",
    "repeat_kind": "the reading of the repeated rows",
    "unit": "the unit of analysis",
    "aggregation": "the combining of each unit's rows",
    "temporal": "the temporal question",
}
_PLURAL_SUBJECTS = {"roles", "exclusions", "models"}


def _slot_value(slot: str, value: Any) -> str | None:
    """What a slot holds, as a phrase; None when it is unset."""
    if value is None:
        return None
    if slot == "lens":
        return f"{listing(value, limit=5)}"
    if slot in ("target", "task", "purpose"):
        return tick(value)
    if slot == "roles":
        return f"the roles recorded for {count(len(value))} columns"
    if slot == "energy_adjustment":
        method = _attr(value, "method")
        return "no adjustment" if method == "none" else f"the {_METHOD_NAME.get(method, tick(method))}"
    if slot == "exclusions":
        n = len(value)
        return "no exclusions" if n == 0 else f"{count(n)} exclusion {plural(n, 'rule')}"
    if slot == "missing":
        strategy = value if isinstance(value, str) else _attr(value, "strategy")
        how = "complete cases" if strategy == "complete_case" else "imputation"
        dropped = [] if isinstance(value, str) else list(_attr(value, "drop_columns") or [])
        return f"{how}, with {listing(dropped)} left out" if dropped else how
    if slot == "split":
        holdout = float(_attr(value, "holdout") or 0)
        return "cross-validation only" if holdout == 0 else f"a {tick(f'{holdout:.0%}')} holdout"
    if slot == "models":
        return listing(value)
    if slot == "substitution":
        return f"{tick(_attr(value, 'donor'))} replaced by {tick(_attr(value, 'recipient'))}"
    if slot == "orientation":
        return ("features in rows, turned to one row per sample" if value == "feature_major"
                else "one row per sample, as supplied")
    if slot == "event":
        return tick(value)
    if slot == "grain":
        id_column = _attr(value, "id_column")
        if _attr(value, "grain") == "unknown":
            return "not known"
        if _attr(value, "grain") == "repeated":
            return f"repeated rows by {tick(id_column)}" if id_column else "repeated rows"
        return "one row per participant"
    if slot == "repeat_kind":
        return ("repeated measurements of one quantity" if _attr(value, "repeat_kind") == "repeats"
                else "different time points")
    if slot == "unit":
        return "one row per unit" if value == "unit" else "one row per record"
    if slot == "aggregation":
        return f"by their {_AGGREGATE_NOUN.get(_attr(value, 'method'), tick(_attr(value, 'method')))}"
    if slot == "temporal":
        return "temporal" if _attr(value, "temporal") else "not temporal"
    return None


@register_sentence("revert")
def _revert(d: Any, state: Any, ctx: Any) -> str:
    from turbotab.core.decisions import SLOTS, DecisionRecord, Revert, fold

    records = []
    for r in _get(ctx, "records", ()) or ():
        try:
            records.append(r if isinstance(r, DecisionRecord) else DecisionRecord.model_validate(r))
        except Exception:
            continue
    target = next((r for r in records if r.id == d.decision_id), None)
    if target is None:
        return "An earlier decision was reverted, restoring the answer before it"
    label = f"Decision {tick(f'#{target.seq}')} was reverted"
    if isinstance(target.decision, Revert):
        undone = next((r for r in records if r.id == target.decision.decision_id), None)
        if undone is not None:
            return f"{label}, so decision {tick(f'#{undone.seq}')} stands again"
        return f"{label}, so the decision it had undone stands again"
    slot = SLOTS.get(target.decision.kind)
    if slot is None:
        return f"{label}, restoring the answer before it"
    try:
        pending = DecisionRecord(
            id="__pending_revert__", seq=max(r.seq for r in records) + 1,
            at=datetime.now(timezone.utc), decision=d)
        after = getattr(fold([*records, pending]), slot)
    except Exception:
        return f"{label}, restoring the answer before it"
    if slot == "findings":  # one finding's disposition, keyed by its id
        fid = getattr(target.decision, "finding_id", None)
        entry = (after or {}).get(fid) if fid else None
        what = _finding_name(fid, _get(ctx, "finding"))
        if entry is None:
            return f"{label}, so {what} is open again"
        return f"{label}, so {what} is {_attr(entry, 'action')} again"
    if slot == "seal_opened":
        return (f"{label}, so the held-out rows are sealed again" if not after
                else f"{label}, and the held-out rows stay open")
    subject = _SLOT_SUBJECT.get(slot, slot.replace("_", " "))
    verb = "are" if slot in _PLURAL_SUBJECTS else "is"
    value = _slot_value(slot, after)
    if value is None:
        return f"{label}, so {subject} {verb} unanswered again"
    return f"{label}, so {subject} {verb} {value} again"


# set_roles

_ROLE_ORDER = ("energy", "exposure", "covariate", "identifier", "design", "flag", "time", "excluded")
_ROLE_NOUN = {
    "energy": ("energy", "energy"),
    "exposure": ("exposure", "exposures"),
    "covariate": ("covariate", "covariates"),
    "identifier": ("identifier", "identifiers"),
    "design": ("design column", "design columns"),
    "flag": ("flag", "flags"),
    "time": ("time column", "time columns"),
    "excluded": ("excluded", "excluded"),
}
_ROLE_AS = {
    "energy": "the energy column",
    "exposure": "an exposure",
    "covariate": "a covariate",
    "identifier": "an identifier",
    "design": "a design column",
    "flag": "a flag",
    "time": "a time column",
    "excluded": "excluded",
}


@register_sentence("set_roles")
def _set_roles(d: Any, state: Any, ctx: Any) -> str:
    roles: dict[str, str] = dict(d.roles)
    before: dict[str, str] = dict(state.roles or {})
    if before and set(before) == set(roles):
        changed = [c for c in roles if roles[c] != before[c]]
        if not changed:
            return f"The roles of all {count(len(roles))} columns were confirmed unchanged"
        if len(changed) <= 3:
            moves = [f"{tick(c)} became {_ROLE_AS.get(roles[c], tick(roles[c]))}" for c in changed]
            return f"{listing(moves, ticked=False)}; every other role is unchanged"
    groups = []
    for role in _ROLE_ORDER:
        cols = [c for c, r in roles.items() if r == role]
        if not cols:
            continue
        one, many = _ROLE_NOUN[role]
        noun = one if len(cols) == 1 else many
        groups.append(f"{noun} {listing(cols)}")
    return f"Column roles were set for {count(len(roles))} columns: " + "; ".join(groups)


# set_exclusions

def _range_phrase(low: Any, high: Any) -> str:
    if low is not None and high is not None:
        return f"outside {tick(number(low))}–{tick(number(high))}"
    if low is not None:
        return f"below {tick(number(low))}"
    if high is not None:
        return f"above {tick(number(high))}"
    return "with no bound"


def _rule_phrase(rule: Any) -> str:
    column = tick(rule.column)
    # A rule leaves out the rows it cannot confirm unless told to keep them (stages.rows.rule_keep).
    unconfirmed = " or not recorded" if getattr(rule, "missing", "exclude") == "exclude" else ""
    if rule.by is not None and rule.by.ranges:
        parts = []
        for i, (level, (low, high)) in enumerate(rule.by.ranges.items()):
            who = f"{tick(rule.by.column)} {tick(level)}" if i == 0 else tick(level)
            parts.append(f"{_range_phrase(low, high)} for {who}")
        text = f"{column} " + listing(parts, limit=6, ticked=False)
        if rule.low is not None or rule.high is not None:
            text += f", and {_range_phrase(rule.low, rule.high)} otherwise"
        return text + unconfirmed
    return f"{column} {_range_phrase(rule.low, rule.high)}{unconfirmed}"


def _as_reason(reason: str) -> str:
    reason = finish(reason, terminal=False)
    return reason[:1].lower() + reason[1:] if reason[:2] != reason[:2].upper() else reason


def exclusion_counts(rules: Sequence[Any], state: Any, ctx: Any) -> tuple[list[int] | None, int | None]:
    """Rows each rule removes, applied in order, and the rows the first rule starts from.

    Counted as the participant flow counts them: among rows with the outcome measured (when an
    outcome is chosen), each rule removing only rows the rules before it kept.
    """
    given = _get(ctx, "exclusion_counts")
    if given is not None and len(given) == len(rules):
        return [int(n) for n in given], _get(ctx, "n_before")
    frame = _get(ctx, "frame")
    columns: set[str] = set()
    for rule in rules:
        columns.add(rule.column)
        if rule.by is not None:
            columns.add(rule.by.column)
    target = getattr(state, "target", None)
    try:
        if frame is None:
            store = _get(ctx, "datastore")
            if store is None:
                return None, None
            known = set(store.columns)
            wanted = [c for c in {*columns, target} if c and c in known]
            if not columns <= known:
                return None, None
            frame = store.materialize(wanted)
        if not columns <= set(frame.columns):
            return None, None
        from turbotab.core.stages.rows import rule_keep  # what the participant flow removes

        keep = frame[target].notna() if target in frame.columns else None
        if keep is None:
            import pandas as pd

            keep = pd.Series(True, index=frame.index)
        n_before = int(keep.sum())
        out = []
        for rule in rules:
            hit = ~rule_keep(frame, rule) & keep
            out.append(int(hit.sum()))
            keep = keep & ~hit
        return out, n_before
    except Exception:  # a sentence never fails a decision; it says less
        log.debug("could not count exclusions", exc_info=True)
        return None, None


@register_sentence("set_exclusions")
def _set_exclusions(d: Any, state: Any, ctx: Any) -> str:
    target = getattr(state, "target", None)
    if not d.rules:
        who = f"every row with {tick(target)} measured" if target else "every row"
        return f"No rows were excluded: {who} stays in the analysis"
    counts, _ = exclusion_counts(d.rules, state, ctx)
    clauses = []
    for i, rule in enumerate(d.rules):
        rows = f"{count(counts[i])} {plural(counts[i], 'row')}" if counts is not None else "Rows"
        verb = "was" if counts is not None and counts[i] == 1 else "were"
        clause = f"{rows} with {_rule_phrase(rule)} {verb} excluded as {_as_reason(rule.reason)}"
        clauses.append(clause if i == 0 else clause[:1].lower() + clause[1:])
    text = "; ".join(clauses)
    if counts is not None and len(counts) > 1:
        text += f", {count(sum(counts))} in all"
    return text


# set_missing

@register_sentence("set_missing")
def _set_missing(d: Any, state: Any, ctx: Any) -> str:
    dropped = list(getattr(d, "drop_columns", None) or [])
    first = ""
    if dropped:
        n = len(dropped)
        first = (f"{listing(dropped)} {plural(n, 'was', 'were')} left out of the predictors; "
                 f"then ")
    levels = getattr(d, "categorical", "impute") == "missing_category"  # M2: missingness by mechanism
    if levels:
        first += ("blanks in categorical and yes/no predictors were kept as a level of their own, "
                  "`Missing`; then ")
    if d.strategy == "complete_case":
        other = "any other predictor" if dropped or levels else "any predictor"
        text = f"{first}rows missing {other} were dropped (a complete-case analysis)"
        kept, before = _get(ctx, "n_complete"), _get(ctx, "n_before")
        if kept is not None and before:
            if kept == before:
                text = (f"{first}a complete-case analysis was applied: no row is missing "
                        f"{other}, so all {count(before)} rows remain")
            else:
                text += f": {count(kept)} of {count(before)} rows remain"
        return text[0].upper() + text[1:]
    others = "the other predictors' missing values" if dropped or levels else "missing predictor values"
    text = (f"{first}{others} were imputed, learned from training rows only; no row was dropped "
            f"for a missing predictor")
    if getattr(d, "indicators", False):
        text += ", and each imputed number carries a missing indicator"
    return text[0].upper() + text[1:]


# set_split

@register_sentence("set_split")
def _set_split(d: Any, state: Any, ctx: Any) -> str:
    task = getattr(state, "task", None) or _get(ctx, "detected_task")  # answered, else detected
    target = getattr(state, "target", None)
    repeats = _get(ctx, "repeats")
    group = _attr(repeats, "column") if repeats else None
    # M2 (the seal, M2_CONTRACT §3): the seal's own basis and chronological draw for these answers
    # (the seal_plan artifact) outrank the roles' reading, so the sentence says how the rows were
    # really drawn: grouped or not, latest-first or at random, and whether the score is exploratory.
    plan = _get(ctx, "seal_plan")
    basis = _attr(plan, "basis") if plan else None
    chron = _attr(plan, "chronology") if plan else None
    if basis is not None:
        grouped = _attr(basis, "state") == "grouped" and _attr(basis, "source") != "aggregation"
        group = _attr(basis, "column") if grouped else None  # combined per unit: nothing repeats
    latest = bool(chron is not None and _attr(chron, "drawn") and _attr(chron, "time_column"))
    how = []
    if group:
        how.append(f"keeping each {tick(group)}'s rows together")
    if task in ("binary", "multiclass", "time_to_event") and target and not latest:
        how.append(f"stratified by {tick(target)}")
    manner = " (" + ", ".join([f"seed {tick(d.seed)}", *how]) + ")"
    folds = f"{tick(d.folds)}-fold cross-validation"
    compared = f"models were compared by {folds} on the rest"
    if plan is not None and _attr(plan, "time_ordered_folds"):  # audit MA-11: folds follow time
        folds = (f"cross-validation over {tick(d.folds)} time-ordered folds, each scored by models "
                 f"fit on earlier units")
        compared = f"models were compared on the rest by {folds}"
    # How a cross-validated or held-out R² is measured (audit MA-09; models/metrics.py).
    r2 = ("; R² was measured against the training rows' mean and pooled over every out-of-fold "
          "prediction" if task == "regression" else "")
    if d.holdout == 0:
        return f"No rows were held out; performance was estimated by {folds}{manner}{r2}"
    share = tick(f"{d.holdout:.0%}")
    # No count: the held-out rows are drawn over every row with the outcome recorded, and the
    # analysis count changes with any later exclusion or missing-values answer; the banner and
    # the Rows view say how many, as they stand.
    pool = f" of the rows with {tick(target)} recorded" if target else " of the rows"
    if latest:
        time = tick(_attr(chron, "time_column"))
        whole = f"whole {tick(group)} units by their last {time}" if group else f"by {time}"
        text = (f"The latest {share}{pool} ({whole}) were held out for one final score, so the "
                f"models are scored on later data than they learned from; {compared}")
    else:
        text = f"A random {share}{pool}{manner} was held out for one final score; {compared}"
    text += r2
    if basis is not None and _attr(basis, "exploratory"):
        text += f"; the held-out score is exploratory, as the split's basis is {_attr(basis, 'label')}"
    return text


# set_energy_adjustment

_METHOD_NAME = {
    "standard": "standard (multivariate) model",
    "residual": "residual method",
    "density_multivariate": "multivariate nutrient density model",
    "density": "nutrient density model",
    "partition": "energy partition model",
}


@register_sentence("set_energy_adjustment")
def _set_energy_adjustment(d: Any, state: Any, ctx: Any) -> str:
    # Every adjusted nutrient by name: a methods sentence never says "and 3 more".
    nutrients = listing(d.nutrients, limit=len(d.nutrients)) if d.nutrients else ""
    energy = tick(d.energy_column) if d.energy_column else "total energy"
    if d.method == "none":
        who = nutrients or "nutrients"
        return f"No energy adjustment was applied: {who} enter the models as absolute intakes"
    each = f"{nutrients} were each" if len(d.nutrients) > 1 else (f"{nutrients} was" if nutrients else "each nutrient was")
    where = f" within levels of {tick(d.strata)}" if d.strata else ""
    name = _METHOD_NAME[d.method]
    if d.method == "residual":
        many = len(d.nutrients) > 1
        who = f"{nutrients} were each" if many else (f"{nutrients} was" if nutrients else "each nutrient was")
        logged = ", both logged," if d.log_transform else ""
        # Under strata, one constant for every level (StratifiedEnergyAdjuster), not each level's.
        mean = "the nutrient's mean over all training rows" if d.strata else "the nutrient's mean"
        return (f"Energy was adjusted by the {name}: {who} regressed on {energy}{logged}{where} "
                f"on training rows and replaced by the residual plus {mean}")
    if d.method == "standard":
        who = nutrients or "the nutrients"
        return (f"Energy was adjusted by the {name}: {energy} enters the models beside {who}, so "
                f"each nutrient's effect is at fixed total energy")
    if d.method == "density_multivariate":
        return (f"Energy was adjusted by the {name}: {each} divided by {energy}{where}, which "
                f"stays in the models as its own term")
    if d.method == "density":
        return (f"Energy was adjusted by the {name}: {each} divided by {energy}{where}, which "
                f"leaves the models")
    who = nutrients or "the chosen nutrients"
    return (f"Energy was partitioned: {energy} was split into kcal from {who} and kcal from "
            f"everything else, each its own term")


# select_models

_FAMILY_LABEL = {
    "elastic_net": "elastic net",
    "boosted_trees": "gradient-boosted trees",
    "mixed": "a random-intercept mixed model",
    "gee": "generalized estimating equations",
    "cox": "Cox proportional hazards",
}
_LINEAR_LABEL = {
    "regression": "linear regression",
    "binary": "logistic regression",
    "multiclass": "multinomial logistic regression",
}


def _family_label(key: str, task: Any, ctx: Any) -> str:
    given = (_get(ctx, "model_labels") or {}).get(key)
    if given:
        return str(given)
    if key == "linear":
        return _LINEAR_LABEL.get(task, "a linear model")
    return _FAMILY_LABEL.get(key, tick(key))


_NUMBER_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


@register_sentence("select_models")
def _select_models(d: Any, state: Any, ctx: Any) -> str:
    task = getattr(state, "task", None) or _get(ctx, "detected_task")  # answered, else detected
    labels = [_family_label(k, task, ctx) for k in d.models]
    n = len(labels)
    head = _NUMBER_WORD.get(n, tick(n))
    return f"{head} model {plural(n, 'family', 'families')} {plural(n, 'was', 'were')} chosen: {listing(labels, limit=8, ticked=False)}"


# set_substitution

@register_sentence("set_substitution")
def _set_substitution(d: Any, state: Any, ctx: Any) -> str:
    energy = getattr(getattr(state, "energy_adjustment", None), "energy_column", None)
    fixed = f"with {tick(energy)} held fixed" if energy else "at the same total energy"
    text = (f"The substitution studied is {tick(d.donor)} replaced by {tick(d.recipient)}, in steps "
            f"of {tick(number(d.step_kcal))} kcal {fixed}")
    n_boot = int(getattr(d, "n_boot", 0) or 0)
    if n_boot:
        text += (f"; its band comes from {count(n_boot)} refits of each model on bootstrap "
                 f"resamples of training rows")
    return text


# ── M2: the opening sequence (OPENING_SEQUENCE.md §03) ───────────────────────


def _unit_of(state: Any) -> str | None:
    """The column naming the unit (person, sample) when rows repeat, as the grain answer has it."""
    grain = getattr(state, "grain", None)
    return _attr(grain, "id_column") if grain is not None else None


def _whose(state: Any) -> str:
    """``each `participant_id`'s`` when the unit is named, else ``each participant's``."""
    unit = _unit_of(state)
    return f"each {tick(unit)}'s" if unit else "each participant's"


def _unit_counts(ctx: Any, column: str | None) -> tuple[int | None, int | None, int | None]:
    """(rows, distinct units, most rows one unit has), from ``ctx`` or counted from the datastore."""
    n_rows, n_units = _n_rows(ctx), _get(ctx, "n_units")
    most = _get(ctx, "rows_per_unit")
    repeats = _get(ctx, "repeats")
    if n_units is None and repeats and _attr(repeats, "column") == column:
        n_units, most = _attr(repeats, "n_units"), _attr(repeats, "max_rows_per_unit")
    if n_units is None and column:
        store, frame = _get(ctx, "datastore"), _get(ctx, "frame")
        try:
            values = None
            if frame is not None and column in frame.columns:
                values = frame[column]
            elif store is not None and column in store.columns:
                values = store.materialize([column])[column]
            if values is not None:
                sizes = values.dropna().value_counts()
                n_rows = n_rows if n_rows is not None else int(len(values))
                n_units, most = int(len(sizes)), int(sizes.max()) if len(sizes) else 0
        except Exception:  # a sentence never fails a decision; it says less
            log.debug("could not count units", exc_info=True)

    def whole(value: Any) -> int | None:
        return int(value) if value is not None else None

    return whole(n_rows), whole(n_units), whole(most)


@register_sentence("set_orientation")
def _set_orientation(d: Any, state: Any, ctx: Any) -> str:
    if d.orientation == "sample_major":
        return "The table was confirmed as one row per sample, as supplied, and was not transposed"
    text = ("The table was supplied with features in rows and samples in columns, and was "
            "transposed to one row per sample before any diagnosis was run")
    n = _n_rows(ctx)
    if n:
        text += f"; its {count(n)} rows became measurement columns"
    return text


@register_sentence("set_feature_table")
def _set_feature_table(d: Any, state: Any, ctx: Any) -> str:
    named = (f"{tick(d.label)} names the features" if d.label
             else "the features are named by their row")
    if not d.annotations:
        return f"For the features-in-rows table, {named} and every other column is a sample"
    n = len(d.annotations)
    return (f"For the features-in-rows table, {named}; {listing(d.annotations)} "
            f"{'describes' if n == 1 else 'describe'} the features and {'stays' if n == 1 else 'stay'} "
            f"beside them, and every other column is a sample")


@register_sentence("set_categorical")
def _set_categorical(d: Any, state: Any, ctx: Any) -> str:
    if not d.columns:
        return "No numeric column was declared a set of categories"
    n = len(d.columns)
    return (f"{listing(d.columns)} {'was' if n == 1 else 'were'} declared "
            f"{'a code' if n == 1 else 'codes'} for categories and entered the models as one "
            f"indicator per level after the first")


def _levels(ctx: Any, column: str) -> list[str]:
    """The outcome's levels as written (``1.0`` and ``1`` are one level), from ``ctx`` or the data."""
    from turbotab.core.stages.proposals import level_key

    given = _get(ctx, "levels")
    if given is None:
        store, frame = _get(ctx, "datastore"), _get(ctx, "frame")
        try:
            if frame is not None and column in frame.columns:
                given = list(frame[column].dropna().unique())
            elif store is not None and column in store.columns:
                given = list(store.materialize([column])[column].dropna().unique())
        except Exception:  # a sentence never fails a decision
            given = None
    out: list[str] = []
    for value in given or []:
        key = level_key(value)
        if key and key not in out:
            out.append(key)
    return out


@register_sentence("set_event")
def _set_event(d: Any, state: Any, ctx: Any) -> str:
    from turbotab.core.stages.rows import _level_key as level_key  # "1", "1.0" and 1.0 are one level

    event = level_key(d.level)
    text = f"{tick(event)} of {tick(d.column)} was taken as the event and coded 1"
    others = [v for v in _levels(ctx, d.column) if level_key(v) != event]
    if 0 < len(others) <= 3:
        text += f"; {listing(others)} {plural(len(others), 'was', 'were')} coded 0"
    return text


@register_sentence("set_follow_up")
def _set_follow_up(d: Any, state: Any, ctx: Any) -> str:
    text = (f"{tick(d.column)} was analyzed as a time to event, each row followed until "
            f"{tick(d.time_column)}, at the event or when follow-up ended without it")
    if d.entry_column:
        text += (f"; a row was at risk only after its {tick(d.entry_column)}, on the same time "
                 f"scale")
    return text


def stated_grain_reason(column: str) -> str:
    """The grain question's stated skip (M2_CONTRACT §10), as the clause after "Not asked:"."""
    return f"every {tick(column)} appears once, so each person is one row."


@register_sentence("set_grain")
def _set_grain(d: Any, state: Any, ctx: Any) -> str:
    if d.grain == "unknown":
        return ("Whether a participant can appear in more than one row was answered as not known, "
                "so the held-out rows are drawn row by row and their scores are exploratory")
    if d.grain == "one_row_per_unit":
        if d.id_column:
            return (f"Each row was declared a different participant: no {tick(d.id_column)} "
                    f"appears in more than one row")
        return "Each row was declared a different participant: no one appears in more than one row"
    if not d.id_column:
        return ("Participants were declared to appear in more than one row; no column was named "
                "that identifies them, so their rows cannot be kept together")
    text = f"Participants were declared to appear in more than one row, identified by {tick(d.id_column)}"
    n_rows, n_units, most = _unit_counts(ctx, d.id_column)
    if n_rows and n_units:
        text += f": {count(n_rows)} rows from {count(n_units)} of them"
        if most and most > 1:
            text += f", at most {count(most)} each"
    return text


@register_sentence("set_repeat_kind")
def _set_repeat_kind(d: Any, state: Any, ctx: Any) -> str:
    whose = _whose(state)
    if d.repeat_kind == "repeats":
        return (f"{whose[:1].upper()}{whose[1:]} rows were taken as repeated measurements of the "
                f"same quantity, not different time points")
    text = f"{whose[:1].upper()}{whose[1:]} rows were taken as different time points"
    if d.time_column:
        text += f", ordered by {tick(d.time_column)}"
    return text


@register_sentence("set_unit")
def _set_unit(d: Any, state: Any, ctx: Any) -> str:
    unit = _unit_of(state)
    who = tick(unit) if unit else "participant"
    if d.unit == "unit":
        return f"The analysis was set at one row per {who}: each one's rows are combined into one"
    return (f"The analysis was kept at one row per record; {_whose(state)} rows stay together on "
            f"one side of the split")


_AGGREGATE_NOUN = {"mean": "mean", "first": "first row", "last": "last row",
                   "change": "change from first to last"}
_AGGREGATE_HOW = {
    "mean": "by their mean",
    "first": "by keeping the first row",
    "last": "by keeping the last row",
    "change": "as the change from the first to the last",
}
_OUTCOME_HOW = {"mean": "mean", "first": "first value", "last": "last value"}


def _time_order(state: Any) -> str | None:
    for slot in ("repeat_kind", "temporal"):
        value = getattr(state, slot, None)
        column = _attr(value, "time_column") if value is not None else None
        if column:
            return column
    return None


@register_sentence("set_aggregation")
def _set_aggregation(d: Any, state: Any, ctx: Any) -> str:
    unit = _unit_of(state)
    whose = _whose(state)
    text = f"{whose[:1].upper()}{whose[1:]} rows were combined into one {_AGGREGATE_HOW[d.method]}"
    if d.method != "mean":
        order = _time_order(state)
        text += f", in order of {tick(order)}" if order else ", in file order"
    n_rows, n_units, _ = _unit_counts(ctx, unit)
    if n_rows and n_units and n_units < n_rows:
        text += f": {count(n_rows)} rows became {count(n_units)}"
    target = getattr(state, "target", None)
    if d.outcome and target:
        text += f"; the outcome {tick(target)} was taken as their {_OUTCOME_HOW[d.outcome]}"
    # Codes and unchanging columns are never averaged or differenced (stages.working.column_rule).
    if d.method == "mean":
        text += "; any codes took their most frequent value"
    elif d.method == "change":
        text += "; any codes and unchanging columns kept their first value"
    return text


@register_sentence("set_temporal")
def _set_temporal(d: Any, state: Any, ctx: Any) -> str:
    unit = _unit_of(state)
    together = f", with each {tick(unit)}'s rows kept together" if unit else ""
    if d.temporal:
        by = f" by {tick(d.time_column)}" if d.time_column else ""
        return (f"The model was declared to predict a later outcome from earlier measurements: the "
                f"held-out rows are the latest{by}{together}")
    return (f"The model was declared not to predict forward in time: rows are held out at "
            f"random{together}")


@register_sentence("open_seal")
def _open_seal(d: Any, state: Any, ctx: Any) -> str:
    n = _get(ctx, "n_holdout")
    rows = f"The {count(n)} held-out rows were" if n else "The held-out rows were"
    return (f"{rows} opened once and scored; those scores are fixed in the record, and any later "
            f"change is marked as made after the seal was opened")


# ── M2: findings, answered (M2_CONTRACT §4) ──────────────────────────────────

_QUESTION_NAME = {
    "lens": "the lens question",
    "orientation": "the question of which way round the table is",
    "target": "the outcome question",
    "event": "the event question",
    "task": "the task question",
    "purpose": "the purpose question",
    "grain": "the question of whether people repeat",
    "repeat_kind": "the question of what repeats",
    "unit": "the unit-of-analysis question",
    "aggregation": "the question of how rows are combined",
    "temporal": "the temporal question",
    "roles": "the column roles",
    "exclusions": "the eligibility question",
    "missing": "the missing-values question",
    "split": "the held-out rows question",
    "energy_adjustment": "the energy-adjustment question",
    "models": "the model families question",
    "substitution": "the substitution question",
    "open_seal": "opening the seal",
}


def question_name(key: str) -> str:
    """How the app names a question in a sentence: ``the outcome question``."""
    return _QUESTION_NAME.get(key, f"the {key.replace('_', ' ')} question")


def _finding_name(finding_id: str | None, finding: Any) -> str:
    """``the finding on `bp_di``` (its columns), else its title, else plainly ``a finding``."""
    columns = list(_attr(finding, "affected_columns") or []) if finding is not None else []
    if columns:
        return f"the finding on {listing(columns, limit=3)}"
    title = finish(str(_attr(finding, "title") or ""), terminal=False) if finding is not None else ""
    if title:
        return f"the finding “{title}”"
    return "a finding"


RepairSentence = Callable[[Any, Any, Any], str]
_REPAIR_SENTENCES: dict[str, RepairSentence] = {}


def register_repair_sentence(family: str, fn: RepairSentence) -> RepairSentence:
    """A finding family's own sentence for ``apply_repair`` (``fn(decision, state, ctx)``)."""
    _REPAIR_SENTENCES[family] = fn
    return fn


def _humanized(key: str) -> str:
    return key.replace("_", " ").strip()


@register_sentence("apply_repair")
def _apply_repair(d: Any, state: Any, ctx: Any) -> str:
    from turbotab.core.stages.finding_words import family

    own = _REPAIR_SENTENCES.get(family(d.finding_id))
    if own is not None:
        return own(d, state, ctx)
    finding, repair = _get(ctx, "finding"), _get(ctx, "repair")
    offered = _offered_sentence(d, finding)
    if offered:
        return offered  # the repair registry's own methods sentence for this option (M2 §4)
    label = finish(str(_attr(repair, "label") or ""), terminal=False) if repair is not None else ""
    label = label or _humanized(d.option)
    label = label[:1].upper() + label[1:]
    columns = list(_attr(finding, "affected_columns") or []) if finding is not None else []
    if columns:
        text = f"The repair “{label}” was applied to {listing(columns, limit=3)}"
    elif finding is not None and _attr(finding, "title"):
        text = f"The repair “{label}” was applied for {_finding_name(d.finding_id, finding)}"
    else:
        text = f"The repair “{label}” was applied"
    if d.params:
        text += " (" + ", ".join(f"{_humanized(str(k))} {tick(number(v))}"
                                 for k, v in sorted(d.params.items())) + ")"
    consequence = finish(str(_attr(repair, "consequence") or ""), terminal=False) if repair is not None else ""
    if consequence:
        text += f": {consequence[:1].lower()}{consequence[1:]}"
    row_local = _attr(repair, "row_local") if repair is not None else None
    if row_local is True:
        text += "; it rewrote the working table before anything was counted"
    elif row_local is False:
        text += "; it is recorded now and runs inside each training fold"
    return text


def _offered_sentence(d: Any, finding: Any) -> str | None:
    """The methods sentence the finding's offered option carries (``repairs.RepairOption``), for
    the option and params this decision applies; None when the finding does not offer it."""
    for option in list(_attr(finding, "repairs") or []) if finding is not None else []:
        if _attr(option, "key") != d.option:
            continue
        decision = _attr(option, "decision") or {}
        params = _attr(decision, "params") or {}
        if d.params and dict(params) != dict(d.params):
            continue
        said = _attr(option, "sentence")
        if isinstance(said, str) and said.strip():
            return said.strip().rstrip(".")
    return None


@register_sentence("defer_finding")
def _defer_finding(d: Any, state: Any, ctx: Any) -> str:
    name = _finding_name(d.finding_id, _get(ctx, "finding"))
    where = _QUESTION_NAME.get(d.to, f"the {_humanized(d.to)} question")
    return f"{name[:1].upper()}{name[1:]} was set aside for {where}, where it will be raised again"


@register_sentence("dismiss_finding")
def _dismiss_finding(d: Any, state: Any, ctx: Any) -> str:
    name = _finding_name(d.finding_id, _get(ctx, "finding"))
    head = f"{name[:1].upper()}{name[1:]} was dismissed"
    if d.reason and d.reason.strip():
        return f"{head}: {_as_reason(d.reason)}"
    return f"{head}, with no reason given; it stays in the record"


def kinds() -> list[str]:
    """The decision kinds with a registered sentence."""
    return sorted(_SENTENCES)


__all__ = [
    "count", "exclusion_counts", "finish", "kinds", "listing", "machinery", "number", "plural",
    "register_repair_sentence", "register_sentence", "sentence_for", "strip_markdown", "tick",
    "words",
]

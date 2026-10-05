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

from types import SimpleNamespace

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
    if list(d.lenses) == ["other"]:  # audit RO-11 (WP18): a first-class answer, stated
        return ("The measurements were described as none of the offered fields (\"Something else, "
                "or not sure\"), so only the generic checks ran and no field's defaults were "
                "applied")
    noun = plural(len(d.lenses), "lens", "lenses")
    return f"The table was read through the {listing(d.lenses, limit=5)} {noun}"


def _outcome_unit(ctx: Any, column: str, state: Any = None) -> str | None:
    """The outcome's unit a sentence may state: the one the user recorded
    (``set_outcome_unit``) or ``ctx["outcome_unit"]`` (the target stage's stated unit, itself only
    a recorded one); None otherwise, and the sentence quotes the header verbatim. A unit is never
    guessed from the values (audit IN-05; CLINICAL_SURVEY_PACK §A1.1: "TurboTab will not guess"),
    nor read from a header's letters (BLUEPRINT §14.3: ``WBC (x10^3/uL)`` is no U/L)."""
    from turbotab.core.units import outcome_unit, recorded_unit

    recorded = recorded_unit(state, column) if state is not None else None
    if recorded:
        return recorded
    given = _get(ctx, "outcome_unit")
    if given:
        return str(given)
    # BLUEPRINT §14.1, §14.3: a settled outcome-unit reading only, which a name never is (the
    # gates: "`bmi_kg` was chosen as the outcome, in kg"; "`WBC (x10^3/uL)` …, in U/L").
    return outcome_unit(column)[0]


@register_sentence("set_target")
def _set_target(d: Any, state: Any, ctx: Any) -> str:
    text = f"{tick(d.column)} was chosen as the outcome"
    unit = _outcome_unit(ctx, d.column, state)
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
    article = "an" if str(d.task)[:1] in "aeiou" else "a"  # an `ordinal` task
    text = f"{tick(d.column)} was modeled as {article} {tick(d.task)} task"
    detected = _get(ctx, "detected_task")
    if detected and detected != d.task:
        text += f", overriding the detected {tick(detected)}"
    follow_up = getattr(state, "follow_up", None)
    if follow_up is not None and d.task != "time_to_event" and getattr(state, "target", None) == d.column:
        # The follow-up recorded for a time-to-event analysis no longer applies (WP12b).
        text += (f"; the follow-up time {tick(follow_up.time_column)} recorded for it is not used, "
                 f"so it is no longer analyzed as a time to event")
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
    "survey": "the survey answer",
    "exposure_forms": "the exposure forms",
    "outcome_order": "the order of the outcome's levels",
    "outcome_unit": "the outcome's unit",
    "column_units": "the columns' units",
    "censoring": "the follow-up answer",
    "clusters": "the grouping answer",
    "estimand": "the exposure and effect",
    "adjustment": "the adjustment answers",
}
_PLURAL_SUBJECTS = {"roles", "exclusions", "models", "adjustment"}


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
        if method == "residual_energy_dropped":
            return "the residual method with total energy left out of the outcome model"
        return "no adjustment" if method == "none" else f"the {_METHOD_NAME.get(method, tick(method))}"
    if slot == "exclusions":
        n = len(value)
        return "no exclusions" if n == 0 else f"{count(n)} exclusion {plural(n, 'rule')}"
    if slot == "missing":
        strategy = value if isinstance(value, str) else _attr(value, "strategy")
        how = {"complete_case": "complete cases",
               "multiple_imputation": "multiple imputation"}.get(strategy, "imputation")
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
        kind = _attr(value, "repeat_kind")
        return ("repeated measurements of one quantity" if kind == "repeats"
                else "imputed copies of one record" if kind == "imputed_copies"
                else "different time points")
    if slot == "unit":
        return "one row per unit" if value == "unit" else "one row per record"
    if slot == "aggregation":
        return f"by their {_AGGREGATE_NOUN.get(_attr(value, 'method'), tick(_attr(value, 'method')))}"
    if slot == "temporal":
        return "temporal" if _attr(value, "temporal") else "not temporal"
    if slot == "survey":
        if _attr(value, "estimand") == "sample":
            return "these participants, unweighted"
        return f"the surveyed population, weighted by {tick(_attr(value, 'weight'))}"
    if slot == "censoring":
        return "the same follow-up for everyone"
    if slot == "clusters":
        column = _attr(value, "column")
        return f"grouped by {tick(column)}" if column else "no grouping above the person"
    if slot == "estimand":
        if _attr(value, "family"):
            return f"the {_attr(value, 'effect')} effect of each exposure in turn"
        return f"the {_attr(value, 'effect')} effect of {tick(_attr(value, 'exposure'))}"
    if slot == "adjustment":
        return f"answers for {count(len(value))} {plural(len(value), 'covariate')}"
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
    if slot == "seal_opened" and target.decision.kind == "reseal" and after:  # WP16
        return f"{label}, so the re-seal is withdrawn and the held-out rows are open again"
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

_ROLE_ORDER = ("energy", "exposure", "covariate", "identifier", "cluster", "design", "flag", "time",
               "excluded")
_ROLE_NOUN = {
    "energy": ("energy", "energy"),
    "exposure": ("exposure", "exposures"),
    "covariate": ("covariate", "covariates"),
    "identifier": ("identifier", "identifiers"),
    "cluster": ("cluster", "clusters"),
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
    "cluster": "a cluster",
    "design": "a design column",
    "flag": "a flag",
    "time": "a time column",
    "excluded": "excluded",
}


@register_sentence("set_roles")
def _set_roles(d: Any, state: Any, ctx: Any) -> str:
    return _set_roles_body(d, state) + _waiting_clause(list(getattr(d, "unconfirmed", None) or []))


def _waiting_clause(waiting: list[str]) -> str:
    """BLUEPRINT §14 rule 2: what a bulk confirm recorded without confirming, said in the record."""
    if not waiting:
        return ""
    one = len(waiting) == 1
    return (f"; {listing(waiting, limit=3)} {'was' if one else 'were'} proposed below high "
            f"confidence and {'waits' if one else 'wait'} for {'its' if one else 'their'} own "
            f"confirmation before any default reads {'it' if one else 'them'}")


def _set_roles_body(d: Any, state: Any) -> str:
    roles: dict[str, str] = dict(d.roles)
    before: dict[str, str] = dict(state.roles or {})
    if before and set(before) == set(roles):
        changed = [c for c in roles if roles[c] != before[c]]
        if not changed:
            return f"The roles of all {count(len(roles))} columns were recorded unchanged"
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


@register_sentence("confirm_role")
def _confirm_role(d: Any, state: Any, ctx: Any) -> str:
    """BLUEPRINT §14 rule 2: an individual confirmation, recorded as one."""
    return (f"{tick(d.column)} was confirmed as {_ROLE_AS.get(d.role, tick(d.role))} on its own, "
            f"after the evidence for its proposal was read")


_UNIT_WORDS = {"kj": "kJ", "pct_energy": "percent of energy", "m": "meters", "in": "inches"}


@register_sentence("confirm_reading")
def _confirm_reading(d: Any, state: Any, ctx: Any) -> str:
    """BLUEPRINT §14.1: one reading of the data, confirmed on its own and recorded as one."""
    return f"{_confirmed(d.reading, d.column, d.value)}, on its own, after the evidence for its reading was read"


@register_sentence("confirm_readings")
def _confirm_readings(d: Any, state: Any, ctx: Any) -> str:
    """BLUEPRINT §14.2: a block confirmation, each listed reading with the value it showed; a
    homogeneous family (one kind, one value) said once with its count."""
    groups: dict[tuple[str, str], list[str]] = {}
    for item in d.items:
        groups.setdefault((item.reading, item.value), []).append(item.column)
    parts = []
    for (reading, value), columns in groups.items():
        if len(columns) >= 5:
            parts.append(_confirmed(reading, f"__{len(columns)}__", value).replace(
                f"`__{len(columns)}__` was", f"{tick(f'{len(columns):,}')} columns (from "
                f"{tick(columns[0])}) were", 1))
        else:
            parts.extend(_confirmed(reading, c, value) for c in columns)
    shown = "; ".join(parts[:6]) + (f"; and {len(parts) - 6} more" if len(parts) > 6 else "")
    return f"Confirmed together, each as the question showed it: {shown}"


def _confirmed(reading: str, column: str, value: Any) -> str:
    col = tick(column)
    value = str(value)
    d = SimpleNamespace(reading=reading)
    if d.reading == "role":
        said = f"{col} was confirmed as {_ROLE_AS.get(value, tick(value))}"
    elif d.reading == "cluster":
        said = (f"{col} was confirmed as marking rows that belong together, so the intervals "
                f"cluster by it" if value == "yes" else
                f"{col} was confirmed as not marking rows that belong together, so the intervals "
                f"do not cluster by it")
    elif d.reading == "unit":
        from turbotab.core.readings import parse_drinks, unit_words

        words = unit_words(value) if parse_drinks(value) is not None else \
            _UNIT_WORDS.get(value, value)
        said = f"{col} was confirmed to be in {words}"
    elif d.reading == "day_count":
        days = int(value) if value.isdigit() else value
        said = (f"{col} was confirmed as one day's intake" if days == 1 else
                f"{col} was confirmed as a total over {tick(str(days))} days")
    elif d.reading == "code_or_count":
        said = (f"{col} was confirmed to hold codes for categories" if value == "code" else
                f"{col} was confirmed to hold amounts or counts")
    elif d.reading == "nested_in":
        from turbotab.core.readings import NOT_NESTED

        said = (f"{col} was confirmed as part of no total" if value == NOT_NESTED else
                f"{col} was confirmed as part of {tick(value)}")
    elif d.reading == "time_column":
        said = f"{col} was confirmed as the column that orders each unit's rows"
    elif d.reading == "sex_coding":
        from turbotab.core.readings import parse_sex_coding

        coding = parse_sex_coding(value) or {}
        said = (f"{col} was confirmed to code "
                + " and ".join(f"{sex} as {tick(level)}" for level, sex in coding.items()))
    else:
        said = f"{col}'s {str(d.reading).replace('_', ' ')} was confirmed as {tick(value)}"
    return said


# set_exclusions

def _range_phrase(low: Any, high: Any) -> str:
    if low is not None and high is not None:
        return f"outside {tick(number(low))}–{tick(number(high))}"
    if low is not None:
        return f"below {tick(number(low))}"
    if high is not None:
        return f"above {tick(number(high))}"
    return "with no bound"


def _goldberg_phrase(rule: Any) -> str:
    """"`kcal` over Schofield BMR outside the Goldberg cut-offs for PAL `1.55` and `2` days"."""
    from turbotab.core.methods.misreporting import EQUATIONS

    side = {"both": "outside", "under": "below", "over": "above"}[rule.exclude]
    pal = (f"a PAL by {tick(rule.pal_by.column)}" if rule.pal_by is not None
           else f"PAL {tick(number(rule.pal))}")
    days = (f"the days in {tick(rule.days_column)}" if rule.days_column else
            f"{tick(number(rule.days))} {'day' if float(rule.days) == 1 else 'days'}")
    unconfirmed = " or not screened" if rule.missing == "exclude" else ""
    return (f"{tick(rule.column)} over {EQUATIONS[rule.equation].label} BMR {side} the Goldberg "
            f"cut-offs for {pal} and {days}{unconfirmed}")


def _rule_phrase(rule: Any) -> str:
    if getattr(rule, "kind", "range") == "goldberg":
        return _goldberg_phrase(rule)
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
        columns.update(rule.reads())
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
    return text + _domain_clause(state)


def _domain_clause(state: Any) -> str:
    """Under a population survey design, rows leave the estimate and stay in the variance
    (audit ME-06; NHANES Analytic Guidelines 2011–2016 §3.2.3): said where rows are excluded."""
    survey = getattr(state, "survey", None)
    if survey is None or getattr(survey, "estimand", None) != "population":
        return ""
    return ("; under the survey design they leave the estimate but keep their strata and PSUs in "
            "the variance (a domain analysis)")


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
                return text[0].upper() + text[1:]
            text += f": {count(kept)} of {count(before)} rows remain"
        if getattr(state, "purpose", None) == "inference":
            from turbotab.core.methods.missing import COMPLETE_CASE_ASSUMPTION

            text += f"; {COMPLETE_CASE_ASSUMPTION}"
        text += _domain_clause(state)
        return text[0].upper() + text[1:]
    others = "the other predictors' missing values" if dropped or levels else "missing predictor values"
    if d.strategy == "multiple_imputation":
        energy = _energy_column(state)
        with_energy = f" and total energy ({tick(energy)})" if energy else ""
        m = tick(getattr(d, "m", 20))
        if getattr(d, "imputation_model", "compatible") == "passive":
            # MS1 (MODELING_SEQUENCE §4): the customary chained equations, terms derived per copy
            how = ("by multiple imputation by chained equations, any nonlinear term derived in "
                   "each completed copy (passive imputation)")
        else:
            how = ("by multiple imputation compatible with the analysis model (SMC-FCS where the "
                   "model holds a spline, a log, a ratio, or a logistic or Cox outcome; chained "
                   "equations where it is linear in the imputed values)")
        text = (f"{first}{others} were imputed {how}, m = {m} or the percentage of rows with an "
                f"imputed value if that is larger, with the outcome{with_energy} in the imputation "
                f"model, and every estimate pooled over the imputations by Rubin's rules; no row "
                f"was dropped for a missing predictor")
        single = getattr(d, "imputation_levels", "clustered") == "single_level"
        passive = getattr(d, "imputation_model", "compatible") == "passive"
        if single:
            text += "; each row was imputed on its own, the clustering of its rows left out"
        if (getattr(d, "acknowledged", False) and getattr(state, "purpose", None) == "inference"
                and (passive or single)):
            from turbotab.core.methods.missing import PASSIVE_CAUTION, SINGLE_LEVEL_CAUTION

            text += _below_detection_clause(d, state)
            text += (f"; it was kept under inference as a recorded limitation: "
                     f"{PASSIVE_CAUTION if passive else SINGLE_LEVEL_CAUTION}")
            return text[0].upper() + text[1:]
    else:
        fill = _energy_fill_words(state)
        text = (f"{first}{others} were imputed in each training fold without the outcome: the "
                f"median for numbers{fill}, the most frequent value for categories; no row was "
                f"dropped for a missing predictor")
        if getattr(d, "indicators", False):
            text += ", and each imputed number carries a missing indicator"
    text += _below_detection_clause(d, state)
    if getattr(d, "acknowledged", False) and getattr(state, "purpose", None) == "inference":
        from turbotab.core.methods.missing import INDICATOR_CAUTION, SINGLE_FILL_CAUTION

        indicator = getattr(d, "indicators", False) or levels
        text += (f"; it was kept under inference as a recorded limitation: "
                 f"{INDICATOR_CAUTION if indicator else SINGLE_FILL_CAUTION}")
    return text[0].upper() + text[1:]


def _energy_column(state: Any) -> str | None:
    adj = getattr(state, "energy_adjustment", None)
    if adj is not None and getattr(adj, "energy_column", None):
        return str(adj.energy_column)
    return next((c for c, r in (getattr(state, "roles", None) or {}).items() if r == "energy"), None)


def _energy_fill_words(state: Any) -> str:
    """", an energy-bearing nutrient from its line on `kcal`" when the single fill is energy-aware
    (``methods.missing.energy_fill``), else nothing."""
    from turbotab.core.methods.missing import energy_fill

    roles = dict(getattr(state, "roles", None) or {})
    adj = getattr(state, "energy_adjustment", None)
    try:
        predictors = [c for c, r in roles.items() if r in ("exposure", "covariate", "energy")]
        fill = energy_fill(adj.model_dump() if adj is not None else None, predictors, roles, predictors)
    except Exception:  # a sentence never fails a decision; it says less
        fill = None
    if not fill:
        return ""
    return f", an energy-bearing nutrient from its line on total energy ({tick(fill['energy'])})"


def _below_detection_clause(d: Any, state: Any) -> str:
    """How values below a detection limit were filled (audit ME-08)."""
    columns = list(getattr(d, "censored_columns", None) or [])
    method = getattr(d, "below_detection", None)
    if not columns or method not in ("half_minimum", "censoring_aware"):
        reason = getattr(d, "reason", None)
        return f"; non-detections were filled as any other blank, for the recorded reason: {reason}" \
            if reason else ""
    named = listing(columns, limit=3)
    if method == "half_minimum":
        return (f"; values below the detection limit in {named} were set to half the column's "
                f"smallest detected value")
    if getattr(d, "strategy", None) == "multiple_imputation":
        return (f"; values below the detection limit in {named} were drawn below the limit from a "
                f"censored-normal (Tobit) model within the multiple imputation, given the outcome")
    return (f"; values below the detection limit in {named} were set to their expected value below "
            f"the limit under a censored-normal fit, in each training fold")


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
    # Folds that are a cluster's levels are not stratified; a held-out draw still is.
    by_cluster = getattr(d, "validation", "kfold") == "internal_external" and d.holdout == 0
    if (task in ("binary", "multiclass", "ordinal", "time_to_event") and target and not latest
            and not by_cluster):
        how.append(f"stratified by {tick(target)}")
    manner = " (" + ", ".join([f"seed {tick(d.seed)}", *how]) + ")"
    folds = f"{tick(d.folds)}-fold cross-validation"
    compared = f"models were compared by {folds} on the rest"
    if plan is not None and _attr(plan, "time_ordered_folds"):  # audit MA-11: folds follow time
        folds = (f"cross-validation over {tick(d.folds)} time-ordered folds, each scored by models "
                 f"fit on earlier units")
        compared = f"models were compared on the rest by {folds}"
    # How the training rows validated the models (audit ME-11, E16; models/validation.py).
    validation = getattr(d, "validation", "kfold")
    if validation == "repeated_kfold":
        folds = (f"{tick(d.folds)}-fold cross-validation repeated {tick(d.repeats)} times, each "
                 f"score the mean over the repeats")
        compared = f"models were compared on the rest by {folds}"
    elif validation == "internal_external" and getattr(d, "cluster", None):
        folds = (f"internal–external validation, each level of {tick(d.cluster)} held out in turn "
                 f"and scored by models fit on the others")
        compared = f"models were compared on the rest by {folds}"
    boot = ""
    if validation == "bootstrap":
        boot = (f"; the optimism of each model's apparent performance was estimated by Harrell's "
                f"bootstrap ({tick(d.n_boot)} resamples, the whole pipeline refit on each) and "
                f"subtracted, except for a family that nearly memorizes its rows (boosted trees), "
                f"whose cross-validated score stands because the bootstrap overstates it")
    # MS6 (MODELING_SEQUENCE ruling 4): under prediction the comparisons run on repeated k-fold,
    # at least 10 × K, whatever validation gives the score (models/folds.py).
    time_ordered = plan is not None and bool(_attr(plan, "time_ordered_folds"))
    if getattr(state, "purpose", None) != "inference" and not time_ordered:
        from turbotab.core.models.folds import COMPARISON_REPEATS

        r = max(int(d.repeats) if validation == "repeated_kfold" else 1, COMPARISON_REPEATS)
        compared = f"performance was estimated on the rest by {folds}"
        boot += (f"; models were compared with each other and with the no-predictor baseline on "
                 f"{tick(d.folds)}-fold cross-validation repeated {tick(r)} times, by the corrected "
                 f"repeated k-fold t (Nadeau & Bengio 2003; Bouckaert & Frank 2004)")
        if getattr(d, "nested_cv", False):
            from turbotab.core.models.validation import NESTED_REPS

            boot += (f"; each interval is the nested cross-validation interval "
                     f"({tick(NESTED_REPS)} repetitions of {tick(d.folds)} folds; Bates, Hastie & "
                     f"Tibshirani 2023)")
    # How a cross-validated or held-out R² is measured (audit MA-09; models/metrics.py).
    r2 = ("; R² was measured against the training rows' mean and pooled over every out-of-fold "
          "prediction" if task == "regression" else "")
    if d.holdout == 0:
        return f"No rows were held out; performance was estimated by {folds}{manner}{boot}{r2}"
    share = tick(f"{d.holdout:.0%}")
    # No count: the held-out rows are drawn over every row with the outcome recorded, and the
    # analysis count changes with any later exclusion or missing-values answer; the banner and
    # the Rows view say how many, as they stand.
    pool = f" of the rows with {tick(target)} recorded" if target else " of the rows"
    if latest:
        # Audit IN-24: what was drawn, not "later data". Whole units are held out by their last
        # observation, so with repeated rows their earlier rows can predate the training rows; the
        # share is the seal plan's count for this holdout and seed (seal.Chronology.earlier).
        time = tick(_attr(chron, "time_column"))
        if group:
            found = next((e for e in _attr(chron, "earlier") or []
                          if math.isclose(float(_attr(e, "holdout")), float(d.holdout))
                          and int(_attr(e, "seed")) == int(d.seed)), None)
            n_rows, n_before = ((int(_attr(found, "n_held_rows")), int(_attr(found, "n_earlier")))
                                if found is not None else (0, 0))
            if found is None:
                before = ", so some held-out rows can predate the latest training row"
            elif n_before:
                before = (f": {tick(f'{n_before / n_rows:.0%}')} of the held-out rows "
                          f"({count(n_before)} of {count(n_rows)}) were observed before the latest "
                          f"training row")
            else:
                before = ", and none of the held-out rows predates the latest training row"
            among = f" among those with {tick(target)} recorded" if target else ""
            text = (f"The {share} of {tick(group)} units seen last{among} were held out whole for "
                    f"one final score, ranked by their last {time}, each unit's earlier rows "
                    f"included{before}; {compared}")
        else:
            text = (f"The latest {share}{pool} by {time} were held out for one final score, so no "
                    f"held-out row is dated earlier than a training row; {compared}")
    else:
        text = f"A random {share}{pool}{manner} was held out for one final score; {compared}"
    text += boot + r2
    if basis is not None and _attr(basis, "exploratory"):
        text += f"; the held-out score is exploratory, as the split's basis is {_attr(basis, 'label')}"
    return text


# set_energy_adjustment

_METHOD_NAME = {
    "standard": "standard (multivariate) model",
    "residual": "residual method",
    "residual_energy_dropped": "residual method",
    "density_multivariate": "multivariate nutrient density model",
    "density": "nutrient density model",
    "partition": "energy partition model",
    "all_components": "all-components model",
}


def _energy_role_columns(state: Any) -> list[str]:
    roles = getattr(state, "roles", None) or {}
    return [c for c, r in roles.items() if r == "energy"]


@register_sentence("set_energy_adjustment")
def _set_energy_adjustment(d: Any, state: Any, ctx: Any) -> str:
    # Every adjusted nutrient by name: a methods sentence never says "and 3 more". Each sentence
    # says whether total energy stayed in the outcome model, because that decides the estimand
    # (audit WP6: ME-02, ME-03).
    nutrients = listing(d.nutrients, limit=len(d.nutrients)) if d.nutrients else ""
    energy = tick(d.energy_column) if d.energy_column else "total energy"
    if d.method == "none":
        gone = [c for c in dict.fromkeys([d.energy_column, *_energy_role_columns(state)]) if c]
        if gone:
            verb = plural(len(gone), "was", "were")
            return (f"No energy adjustment was applied: {listing(gone, limit=len(gone))} {verb} "
                    f"left out of the models, so nutrients enter as absolute intakes")
        return ("No energy adjustment was applied: no total-energy column is among the "
                "predictors, so nutrients enter as absolute intakes")
    each = f"{nutrients} were each" if len(d.nutrients) > 1 else (f"{nutrients} was" if nutrients else "each nutrient was")
    where = f" within levels of {tick(d.strata)}" if d.strata else ""
    name = _METHOD_NAME[d.method]
    if d.method in ("residual", "residual_energy_dropped"):
        # Under inference the table is refit on every analyzed row, the residual regression with
        # it (BLUEPRINT §12 ruling 3); under prediction it is learned on training rows only.
        on = ("on every analyzed row" if getattr(state, "purpose", None) == "inference"
              else "on training rows")
        scope = ("all analyzed rows" if getattr(state, "purpose", None) == "inference"
                 else "all training rows")
        many = len(d.nutrients) > 1
        who = f"{nutrients} were each" if many else (f"{nutrients} was" if nutrients else "each nutrient was")
        if d.log_transform:
            # The log variant adds back the predicted log nutrient at the mean log energy, then
            # back-transforms: the nutrient at the geometric-mean energy (audit G17).
            pooled = f"{scope}' " if d.strata else ""
            how = (f"regressed on {energy} with both logged{where} {on} and replaced "
                   f"by exp(the log residual plus the predicted log nutrient at {pooled}mean log "
                   f"energy), the nutrient at the geometric-mean energy")
        else:
            # Under strata, one constant for every level (StratifiedEnergyAdjuster), not each level's.
            mean = f"the nutrient's mean over {scope}" if d.strata else "the nutrient's mean"
            how = (f"regressed on {energy}{where} {on} and replaced by the residual "
                   f"plus {mean}")
        if d.method == "residual":
            # The log variant rescales each row by its own energy, so its coefficient is per unit
            # of the rescaled nutrient and not the standard model's (its own estimand, audit ME-03).
            tail = (", so a coefficient is per unit of the rescaled nutrient at fixed energy, not "
                    "the standard model's" if d.log_transform else "")
            return (f"Energy was adjusted by the {name} with total energy kept in the outcome model "
                    f"(the Willett–Stampfer variant): {who} {how}, and {energy} enters the models "
                    f"beside the adjusted values{tail}")
        tail = ("so a coefficient reads like a rescaled nutrient density's, not the standard model's"
                if d.log_transform else
                "so a coefficient equals the standard model's only when no other covariate "
                "correlates with energy")
        return (f"Energy was adjusted by the {name} with total energy left out of the outcome "
                f"model: {who} {how}, and {energy} then left the models, {tail}")
    if d.method == "standard":
        who = nutrients or "the nutrients"
        return (f"Energy was adjusted by the {name}: {energy} enters the models beside {who}, so "
                f"each nutrient's effect is at fixed total energy")
    # Audit D15 (IN-25): a density is the same ratio within any level, so strata change nothing and
    # the sentence does not claim a stratification (models/pipeline.py warns of the same).
    unused = (f"; strata apply to the residual method only, so {tick(d.strata)} changed nothing"
              if d.strata else "")
    if d.method == "density_multivariate":
        return (f"Energy was adjusted by the {name}: {each} divided by {energy}, which stays in "
                f"the models as its own term{unused}")
    if d.method == "density":
        return (f"Energy was adjusted by the {name}: {each} divided by {energy}, which leaves the "
                f"models{unused}")
    who = nutrients or "the chosen nutrients"
    if d.method == "all_components":
        return (f"Energy was adjusted by the {name} (Tomova et al. 2022): {energy} was split into "
                f"kcal from {who}, each its own term, and kcal from everything else; each "
                f"nutrient's average relative effect is its coefficient less the other sources' "
                f"coefficients weighted by their share of the remaining energy")
    return (f"Energy was partitioned: {energy} was split into kcal from {who} and kcal from "
            f"everything else, each its own term")


# select_models

_FAMILY_LABEL = {
    "elastic_net": "elastic net",
    "boosted_trees": "gradient-boosted trees",
    # WP11: one least-squares test per exposure, q-values by Benjamini–Hochberg
    "featurewise": "feature-wise least-squares tests with Benjamini–Hochberg false-discovery control",
    "proportional_odds": "a proportional-odds (cumulative logit) model",
    "mixed": "a random-intercept mixed model",
    "gee": "generalized estimating equations",
    "cox": "Cox proportional hazards",
}
_LINEAR_LABEL = {
    "regression": "linear regression",
    "binary": "logistic regression",
    "multiclass": "multinomial logistic regression",
    "ordinal": "multinomial logistic regression, which ignores the levels' order",
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
    chosen = (f"{head} model {plural(n, 'family', 'families')} {plural(n, 'was', 'were')} "
              f"chosen: {listing(labels, limit=8, ticked=False)}")
    # MS6 (MODELING_SEQUENCE §1 row 12 (b)): with no rows held out, what the result is.
    split = getattr(state, "split", None)
    if (n > 1 and split is not None and float(getattr(split, "holdout", 0) or 0) == 0
            and getattr(state, "purpose", None) != "inference"):
        chosen += ("; with no rows held out, the choice among them is corrected by bootstrap "
                   "bias-corrected cross-validation (Tsamardinos et al. 2018), and that "
                   "selection-corrected estimate is the reported result, not the best family's "
                   "own score")
    # MS4: under the surveyed population, each family's design-based estimator, or its block.
    from turbotab.core.models.survey import models_sentence

    population = models_sentence(state, d.models, task)
    if population:
        chosen = f"{chosen}. {population}"
    # BLUEPRINT §14.3 (amendment): the readings the values settled, which the fit reads, are
    # stated in the record ("read from the values"), each with its evidence.
    read = _get(ctx, "read_from_values")
    return f"{chosen}. {read}" if read else chosen


# set_substitution

@register_sentence("set_substitution")
def _set_substitution(d: Any, state: Any, ctx: Any) -> str:
    energy = getattr(getattr(state, "energy_adjustment", None), "energy_column", None)
    fixed = f"with {tick(energy)} held fixed" if energy else "at the same total energy"
    if getattr(d, "scale", "kcal") == "percent_energy":
        # "5% of energy from X replaced by Y" (NUTRITION_PACK §05; audit B24, D19)
        text = (f"The substitution studied is {tick(d.donor)} replaced by {tick(d.recipient)}, in "
                f"steps of {tick(number(d.step_percent))}% of each participant's own total energy "
                f"{fixed}")
    else:
        text = (f"The substitution studied is {tick(d.donor)} replaced by {tick(d.recipient)}, in "
                f"steps of {tick(number(d.step_kcal))} kcal {fixed}")
    n_boot = int(getattr(d, "n_boot", 0) or 0)
    from turbotab.core.models.survey import substitution_clause

    population = substitution_clause(state)  # MS4: the design's band replaces the refits
    if population:
        text += f"; {population}"
    elif n_boot:
        # Under inference the curve and its refits read every analyzed row (BLUEPRINT §12 ruling 3).
        rows = ("every analyzed row" if getattr(state, "purpose", None) == "inference"
                else "training rows")
        text += (f"; its band comes from {count(n_boot)} refits of each model on bootstrap "
                 f"resamples of {rows}")
    if getattr(d, "acknowledged", False):
        # The recorded attestation of the omitted-sources block under inference (audit ME-05).
        text += ("; it was kept although energy sources are missing from the model, so the curve "
                 "carries the confounding of the sources total energy holds as one composite")
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


@register_sentence("set_exposure_form")
def _set_exposure_form(d: Any, state: Any, ctx: Any) -> str:
    """How one predictor entered the models (WP12a; NUTRITION_PACK §07G and §08)."""
    from turbotab.core.methods.exposure_form import DEFAULT_KNOTS, KNOT_PERCENTILES

    adj = getattr(state, "energy_adjustment", None)
    # The residual and density methods replace the nutrient's column before the form step; the
    # standard model keeps it as recorded (energy enters beside it), and the partition refuses.
    adjusted = (adj is not None and _attr(adj, "method") in ("residual", "density",
                                                             "density_multivariate")
                and d.column in (_attr(adj, "nutrients") or []))
    values = "energy-adjusted values" if adjusted else "values"
    tested = getattr(state, "purpose", None) == "inference"
    if d.form == "linear":
        return f"{tick(d.column)} entered the models as a straight line"
    if d.form == "spline":
        k = d.knots or DEFAULT_KNOTS
        pct = [f"{100 * p:g}" for p in KNOT_PERCENTILES.get(k, ())]
        where = (f" at the {', '.join(pct[:-1])} and {pct[-1]} percentiles of its {values} in the "
                 f"rows each model was fit on (Harrell's placement)" if pct
                 else f" placed on its {values}")
        test = ("; nonlinearity was tested by a Wald test that its nonlinear terms are zero"
                if tested else "")
        return (f"{tick(d.column)} entered the models as a restricted cubic spline with {tick(k)} "
                f"knots{where}{test}")
    trend = ("; the p for trend scored each quintile by its median, entered as one continuous term"
             if tested else "")
    return (f"{tick(d.column)} entered the models as quintiles of its {values} in the rows each "
            f"model was fit on, the lowest the reference{trend}")


@register_sentence("set_outcome_unit")
def _set_outcome_unit(d: Any, state: Any, ctx: Any) -> str:
    unit = "percent" if d.unit == "%" else d.unit
    return f"The unit of {tick(d.column)} was recorded as {unit}"


_UNIT_WORDS = {"kcal": "kcal", "kj": "kJ", "years": "years", "months": "months",
               "weeks": "weeks", "days": "days"}


@register_sentence("set_column_unit")
def _set_column_unit(d: Any, state: Any, ctx: Any) -> str:
    unit = _UNIT_WORDS.get(d.unit, d.unit)
    if d.unit in ("kcal", "kj") and d.days > 1:
        return (f"{tick(d.column)} was recorded as a total over {count(d.days)} days in {unit}, so "
                f"each day's intake is a value divided by {count(d.days)}")
    if d.unit in ("kcal", "kj"):
        return f"{tick(d.column)} was recorded as a day's energy in {unit}"
    return f"{tick(d.column)} was recorded as an age in {unit}"


@register_sentence("set_outcome_scale")
def _set_outcome_scale(d: Any, state: Any, ctx: Any) -> str:
    """Audit RO-10 (WP18): the scale a positive, skewed outcome is analyzed on, and what its
    coefficient then is."""
    from turbotab.core.decisions import log_outcome_name

    column = tick(d.column)
    if d.scale == "original":
        return (f"{column} was analyzed on its original scale, so a coefficient is a difference "
                f"in its mean")
    text = (f"{column} was analyzed on the natural-log scale, as {tick(log_outcome_name(d.column))}: "
            f"a coefficient is a difference in its mean log, and its exponential a ratio of "
            f"geometric means")
    if getattr(state, "purpose", None) == "prediction":
        text += ("; predictions and their scores are on the log scale, and the exponential of a "
                 "prediction estimates the geometric mean (the median when the log is normal), "
                 "not the mean")
    return text


@register_sentence("set_outcome_order")
def _set_outcome_order(d: Any, state: Any, ctx: Any) -> str:
    return (f"The levels of {tick(d.column)} were ordered {' < '.join(tick(v) for v in d.levels)}, "
            f"lowest first")


@register_sentence("set_categorical")
def _set_categorical(d: Any, state: Any, ctx: Any) -> str:
    if not d.columns:
        return "No numeric column was declared a set of categories"
    n = len(d.columns)
    return (f"{listing(d.columns)} {'was' if n == 1 else 'were'} declared "
            f"{'a code' if n == 1 else 'codes'} for categories and entered the models as one "
            f"indicator per level after the first")


@register_sentence("set_sensitivity")
def _set_sensitivity(d: Any, state: Any, ctx: Any) -> str:
    if not d.analyses:
        return "No sensitivity analysis was set beside the primary analysis"
    parts = []
    for a in d.analyses:
        if not a.rules:
            parts.append(f"{tick(a.label)}, keeping every row")
        else:
            rules = listing([_rule_phrase(r) for r in a.rules], limit=3, ticked=False)
            parts.append(f"{tick(a.label)}, excluding rows with {rules}")
    n = len(d.analyses)
    return (f"The primary analysis was set beside {count(n)} sensitivity "
            f"{plural(n, 'analysis', 'analyses')}, each the same model on its own rows: "
            + "; ".join(parts))


@register_sentence("set_measurement_error")
def _set_measurement_error(d: Any, state: Any, ctx: Any) -> str:
    if d.method == "none":
        return ("Energy-adjusted exposures were not corrected for day-to-day error in the "
                "recalls")
    which = (f"{listing(d.exposures)}" if d.exposures else "every energy-adjusted exposure")
    return (f"Univariate regression calibration was applied to {which}, with the day-to-day "
            f"variance estimated from repeated recalls and intervals from {count(d.n_boot)} "
            f"bootstrap refits over people")


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
    # The validator refuses a follow-up beside any other task, and while the task is not settled;
    # should one stand anyway, the sentence does not claim a time-to-event analysis the fit never
    # made. Only a recorded answer makes an outcome a time to event (detection never does), so
    # the claim needs that answer, not merely the absence of a detected task (the gate's race).
    task = getattr(state, "task", None) or _get(ctx, "detected_task")
    if task is None:
        return (f"A follow-up time {tick(d.time_column)} was named for {tick(d.column)}; it is used "
                f"only if the outcome is analyzed as a time to event, which was not yet declared")
    if task != "time_to_event":
        return (f"A follow-up time {tick(d.time_column)} was named for {tick(d.column)}, but the "
                f"outcome is analyzed as a {tick(task)} task, so the follow-up is not used")
    text = (f"{tick(d.column)} was analyzed as a time to event, each row followed until "
            f"{tick(d.time_column)}, at the event or when follow-up ended without it")
    if d.entry_column:
        text += (f"; a row was at risk only after its {tick(d.entry_column)}, on the same time "
                 f"scale")
    if getattr(d, "horizon", None) is not None:  # MS6: the horizon predictions are judged at
        text += (f"; predicted risks were scored and calibrated by {tick(d.time_column)} = "
                 f"{tick(f'{float(d.horizon):g}')}, the declared horizon")
    return text


@register_sentence("set_censoring")
def _set_censoring(d: Any, state: Any, ctx: Any) -> str:
    # WP17 (audit RO-03): the follow-up question's "the same for everyone".
    text = (f"{tick(d.column)} was analyzed as a yes/no outcome: everyone was followed for the "
            f"same time, as answered, so it counts events over one period")
    if d.acknowledged:
        text += (", although a column reads as a follow-up time that varies; the answer was kept "
                 "over that reading and is a stated limitation")
    return text


@register_sentence("set_clusters")
def _set_clusters(d: Any, state: Any, ctx: Any) -> str:
    # WP17 (audit RO-08): the grouping above the person.
    if d.column is None:
        text = "Nothing groups the participants above the person; each is analyzed as independent"
        if d.acknowledged:
            text += (", although a column reads as a site, centre, household or batch; the answer "
                     "was kept over that reading and is a stated limitation")
        return text
    column = tick(d.column)
    if getattr(state, "purpose", None) == "prediction" or d.adjust is None:
        return (f"Participants are grouped by {column}: validation keeps each {column}'s rows "
                f"together, and internal–external validation can hold out whole groups")
    if d.adjust == "fixed_effects":
        return (f"Participants are grouped by {column}: each {column} has its own intercept "
                f"(fixed effects), and the intervals are cluster-robust by {column}")
    return (f"Participants are grouped by {column}: the intervals are cluster-robust by {column}, "
            f"without an intercept for each, so differences between groups stay in the estimate")


@register_sentence("set_estimand")
def _set_estimand(d: Any, state: Any, ctx: Any) -> str:
    # WP17 (MODELING_SEQUENCE §1 step 2): the exposure and its effect.
    from turbotab.core.estimand import MEASURE_WORDS, NON_COLLAPSIBLE

    target = getattr(state, "target", None)
    on = f" on {tick(target)}" if target else ""
    contrast = {"substitution": " (a substitution: in place of other energy sources at fixed "
                                "total energy)",
                "addition": " (an addition: its calories added, every other energy source "
                            "fixed)"}.get(d.contrast or "", "")
    if d.family:  # every exposure in turn, adjusted for the covariates (the feature-wise family)
        scale = (f"as the {MEASURE_WORDS[d.measure]}" if d.measure == "exposure_mean_difference"
                 else f"as a {MEASURE_WORDS.get(d.measure, d.measure)} per unit of each")
        text = (f"The analysis estimates the {d.effect} effect of each exposure in turn{on}"
                f"{contrast}, {scale}, with the false-discovery rate stated")
    else:
        text = (f"The analysis estimates the {d.effect} effect of {tick(d.exposure)}{on}{contrast}, "
                f"as a {MEASURE_WORDS.get(d.measure, d.measure)} per unit of {tick(d.exposure)}")
    if d.measure in NON_COLLAPSIBLE:
        text += ", given the adjustment set"
    if d.effect == "direct":
        text += "; a direct effect holds the mediators fixed and needs their confounders adjusted too"
    return text


@register_sentence("set_adjustment")
def _set_adjustment(d: Any, state: Any, ctx: Any) -> str:
    # WP17 (MODELING_SEQUENCE §1 step 3): roles derived from the disjunctive cause criterion.
    from turbotab.core.estimand import ROLE_PLURAL, ROLE_SINGULAR, derive

    spec = getattr(state, "estimand", None)
    effect = getattr(spec, "effect", None) or "total"
    by_role: dict[tuple[str, bool, bool], list[str]] = {}
    derived = {}
    for column, answers in d.answers.items():
        found = derive(answers, effect)
        derived[column] = found
        by_role.setdefault((found.role, found.adjusted, found.secondary), []).append(column)
    parts = []
    for (role, adjusted, secondary), columns in by_role.items():
        one = len(columns) == 1
        if adjusted:
            where = "adjusted for"
        elif secondary:
            where = "left out of the primary model and adjusted for in a declared secondary one"
        else:
            where = "left out"
        noun = ROLE_SINGULAR[role] if one else ROLE_PLURAL[role]
        parts.append(f"{listing(columns, limit=5)} {'is' if one else 'are'} {noun}, {where}")
    kept = [c for c, a in d.answers.items() if a.keep and a.acknowledged and not
            (derived[c].role == "mediator" and effect == "direct")
            and derived[c].role in ("mediator", "collider", "timing_unknown")]
    from turbotab.core.decisions import EXPOSURE_FAMILY

    whose = "each exposure" if d.exposure == EXPOSURE_FAMILY else tick(d.exposure)
    text = f"For the effect of {whose}, by the disjunctive cause criterion: " + "; ".join(parts)
    if kept:
        text += (f". {listing(kept)} {plural(len(kept), 'was', 'were')} kept in the primary set "
                 f"over that reading, as recorded, so the estimate is not a total effect")
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
    if d.repeat_kind == "imputed_copies":  # audit I18 (WP18)
        from turbotab.core.structural import COPIES_CONCERN

        by = f", numbered by {tick(d.implicate_column)}" if d.implicate_column else ""
        text = f"{whose[:1].upper()}{whose[1:]} rows were taken as imputed copies of one record{by}"
        if getattr(d, "acknowledged", False):
            return (text + f"; recorded as a limitation: they were not pooled by Rubin's rules, "
                           f"and {COPIES_CONCERN}")
        if getattr(state, "purpose", None) == "inference":  # MS3: the fit pools them
            return (text + "; kept as records, each copy is analyzed as a completed dataset with "
                           "its own outcome and the estimates pooled by Rubin's rules, which "
                           "carries the imputation's uncertainty (NCHS's combining rules)")
        return text + f"; they were not pooled by Rubin's rules, and {COPIES_CONCERN}"
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
    """The column that ordered a unit's records: a settled time-column reading only (named, or
    confirmed on its own), the one the working table ordered them by (BLUEPRINT §14.1)."""
    from turbotab.core.readings import time_column_reading

    found = time_column_reading(state, None)
    return found.column if found is not None and found.settled else None


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
    # Codes and unchanging columns are never averaged or differenced (stages.working.column_rule);
    # a column is a code only as the user said (BLUEPRINT §14.1), and the sentence names each.
    from turbotab.core.readings import confirmed_codes

    codes = [c for c in confirmed_codes(state) if c not in (d.columns or {})]
    if d.method == "mean":
        one = len(codes) == 1
        text += (f"; {listing(codes)} {'holds' if one else 'hold'} codes and "
                 f"{'took' if one else 'each took'} its most frequent value" if codes else "")
    elif d.method == "change":
        text += (f"; {listing(codes)} (codes) and unchanging columns kept their first value"
                 if codes else "; unchanging columns kept their first value")
    for column, rule in (d.columns or {}).items():
        text += f"; {tick(column)} took its {_RULE_WORDS_SENTENCE.get(rule, rule)}"
    if getattr(d, "acknowledged", False):  # audit RO-09 (WP18): blocked, and recorded
        text += ("; recorded as a limitation: predictors were summarized from records later than "
                 "the outcome's, so a later value may follow from the outcome rather than cause it "
                 "(reverse causation)")
    return text


_RULE_WORDS_SENTENCE = {"mean": "mean", "first": "first record's value",
                        "last": "last record's value", "change": "last minus first",
                        "mode": "most frequent value"}


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


def _earlier_opening(d: Any, ctx: Any) -> Any | None:
    """An earlier opening of the same outcome's seal in the records before this one (a re-seal
    came between: audit WP16, RO-05), or None."""
    from turbotab.core.decisions import Refusal, reverted

    records = list(_get(ctx, "records") or [])
    try:
        cancelled = reverted(records)
    except Refusal:
        cancelled = {}
    return next((r for r in sorted(records, key=lambda r: r.seq)
                 if r.id not in cancelled and r.decision.kind == "open_seal"
                 and getattr(r.decision, "target", None) in (None, getattr(d, "target", None))),
                None)


def _scored(d: Any) -> str:
    """``: held-out log loss `0.512` (AUC `0.801`, the customary headline)`` — the kept score of
    the declared family on the primary (WP16; MS6: a strictly proper score), with the customary
    headline beside it when the record kept one, or nothing."""
    from turbotab.core.models.metrics import HEADLINE_LABEL, LABELS
    from turbotab.core.models.validation import score_words

    scores = getattr(d, "scores", None) or {}
    metric = getattr(d, "metric", None)
    family = getattr(d, "family", None)
    kept = (scores.get(family) or {}) if family else {}
    value = kept.get(metric) if metric else None
    if value is None:
        return ""
    text = f": held-out {score_words(metric)} {tick(f'{value:.3f}')}"
    headline = next((h for h in ("auc", "c_index") if h != metric and kept.get(h) is not None), None)
    if headline is not None:
        text += f" ({LABELS[headline]} {tick(f'{kept[headline]:.3f}')}, the {HEADLINE_LABEL})"
    return text


@register_sentence("open_seal")
def _open_seal(d: Any, state: Any, ctx: Any) -> str:
    n = getattr(d, "n_holdout", None) or _get(ctx, "n_holdout")
    rows = f"The {count(n)} held-out rows were" if n else "The held-out rows were"
    earlier = _earlier_opening(d, ctx)
    if earlier is not None:  # a re-seal came between (WP16, RO-05)
        family = getattr(d, "family", None)
        task = getattr(state, "task", None) or _get(ctx, "detected_task")
        named = f" with {_family_label(family, task, ctx)} declared the final model" if family else ""
        held = f"The {count(n)} held-out rows" if n else "The held-out rows"
        return (f"{held} drawn again after the opening at decision {tick(f'#{earlier.seq}')} "
                f"were opened{named} and scored{_scored(d)}; this is not an independent test, and "
                f"the scores at that opening stay the reported result")
    # AUDIT_REPORT §5 WP8 (ME-13): the final model was declared on cross-validation beforehand.
    family = getattr(d, "family", None)
    if family:
        task = getattr(state, "task", None) or _get(ctx, "detected_task")
        others = len(getattr(state, "models", None) or []) > 1
        rest = ", the other families' are secondary," if others else ","
        return (f"With {_family_label(family, task, ctx)} declared the final model on "
                f"cross-validation beforehand, {rows[0].lower()}{rows[1:]} opened once and "
                f"scored{_scored(d)}; its held-out score is the reported result{rest} and any later "
                f"change is marked as made after the seal was opened")
    return (f"{rows} opened once and scored; those scores are fixed in the record, and any later "
            f"change is marked as made after the seal was opened")


@register_sentence("reseal")
def _reseal(d: Any, state: Any, ctx: Any) -> str:
    """Recorded only after an opening, so the log leads it with "After the held-out rows were
    opened" (``decisions.disclose``)."""
    earlier = _earlier_opening(d, ctx)
    at = f" at decision {tick(f'#{earlier.seq}')}" if earlier is not None else ""
    why = f" ({d.reason})" if getattr(d, "reason", None) else ""
    return (f"The seal was withdrawn so the held-out rows could be drawn again{why}; the scores at "
            f"the opening{at} stay the reported result, and rows drawn afterwards are withheld "
            f"until opened, as a test that is not independent")


@register_sentence("lock_plan")
def _lock_plan(d: Any, state: Any, ctx: Any) -> str:
    """What was declared in the software before any estimate was displayed (MODELING_SEQUENCE §1
    row 12): never "prespecified" or "preregistered"."""
    digest = getattr(d, "digest", None)
    hashed = f" (SHA-256 {tick(digest[:12])})" if digest else ""
    return (f"The analysis plan recorded above was declared in TurboTab before any estimate was "
            f"displayed{hashed}; every later change is marked as made after the estimates were "
            f"seen")


# set_survey (audit §5 WP10)


def _design_counts(d: Any, ctx: Any) -> tuple[int, int, int] | None:
    """PSUs (nested in strata), strata, and strata with a single PSU, read from the table."""
    store = _get(ctx, "datastore")
    if store is None or not d.psu or not d.strata:
        return None
    try:
        frame = store.materialize([d.strata, d.psu], None).dropna()
    except Exception:  # noqa: BLE001 - a sentence says less rather than fail
        return None
    per = frame.groupby(d.strata)[d.psu].nunique()
    return int(per.sum()), int(len(per)), int((per == 1).sum())


@register_sentence("set_survey")
def _set_survey(d: Any, state: Any, ctx: Any) -> str:
    from turbotab.core.survey import ATTESTATION, reading_of

    if d.estimand == "sample":
        reading = reading_of(state)
        named = [*reading.weights, *reading.strata, *reading.psu]
        unused = (f"; {listing(named, limit=6)} {plural(len(named), 'was', 'were')} recorded and "
                  f"not used") if named else ""
        return (f"The estimates describe these participants, not the surveyed population: "
                f"{ATTESTATION}{unused}")
    if d.cycle and d.four_year_weight:
        weight = (f"weighted by {tick(d.four_year_weight)} on the 1999–2002 rows (doubled, then "
                  f"divided by the number of cycles pooled in {tick(d.cycle)}) and by "
                  f"{tick(d.weight)} on the others (divided by the same number), as NCHS directs "
                  f"for 1999–2000")
    elif d.cycle:
        weight = (f"weighted by {tick(d.weight)} divided by the number of cycles pooled in "
                  f"{tick(d.cycle)}")
    else:
        weight = f"weighted by {tick(d.weight)}"
    if d.strata and d.psu:
        counts = _design_counts(d, ctx)
        shape = f" ({count(counts[0])} PSUs in {count(counts[1])} strata)" if counts else ""
        over = f"over {tick(d.psu)} nested within {tick(d.strata)}{shape}"
        if counts and counts[2]:
            # MS4: the stated rule, R survey's lonely.psu = "adjust" as survey 4.5 computes it.
            over += (f"; {count(counts[2])} {plural(counts[2], 'stratum', 'strata')} with a single "
                     f"PSU {plural(counts[2], 'was', 'were')} centered at the mean PSU total of "
                     f"the strata holding analysis rows (R survey's lonely.psu \"adjust\")")
    elif d.psu:
        over = f"over {tick(d.psu)} with no strata, as recorded"
    else:
        units = ("each row, or each unit's rows where units repeat, taken as a sampling unit of "
                 "its own")
        within = f"within {tick(d.strata)}" if d.strata else "with no strata"
        over = f"{within} and no PSU column, {units}, as recorded"
    # The domain clause only where rows are already restricted (repair round: it was said with no
    # restriction at all); a later exclusion or missing-values sentence says it where it restricts.
    rules = getattr(state, "exclusions", None) or []
    missing = getattr(getattr(state, "missing", None), "strategy", None)
    restricted = bool(rules) or missing == "complete_case"
    domain = ("; the rows the eligibility rules or missing values leave out stay in the design for "
              "the variance (a domain analysis)" if restricted else "")
    return (f"The estimates describe the surveyed population: rows were {weight}, and standard "
            f"errors were estimated by Taylor series linearization {over}, first-stage units taken "
            f"as sampled with replacement, with t intervals on the PSUs minus the strata that hold "
            f"the analysis rows{domain}")


# ── M2: findings, answered (M2_CONTRACT §4) ──────────────────────────────────

_QUESTION_NAME = {
    "lens": "the lens question",
    "orientation": "the question of which way round the table is",
    "target": "the outcome question",
    "event": "the event question",
    "task": "the task question",
    "follow_up": "the follow-up question",
    "purpose": "the purpose question",
    "grain": "the question of whether people repeat",
    "repeat_kind": "the question of what repeats",
    "unit": "the unit-of-analysis question",
    "aggregation": "the question of how rows are combined",
    "temporal": "the temporal question",
    "roles": "the column roles",
    "clusters": "the grouping question",
    "survey": "the survey question",
    "estimand": "the exposure and effect question",
    "adjustment": "the adjustment-set question",
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

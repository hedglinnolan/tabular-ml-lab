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


@register_sentence("set_target")
def _set_target(d: Any, state: Any, ctx: Any) -> str:
    text = f"{tick(d.column)} was chosen as the outcome"
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
}


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
    subject = _SLOT_SUBJECT.get(slot, slot.replace("_", " "))
    verb = "are" if subject.endswith("s") and not subject.endswith("ss") else "is"
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
    if rule.by is not None and rule.by.ranges:
        parts = []
        for i, (level, (low, high)) in enumerate(rule.by.ranges.items()):
            who = f"{tick(rule.by.column)} {tick(level)}" if i == 0 else tick(level)
            parts.append(f"{_range_phrase(low, high)} for {who}")
        text = f"{column} " + listing(parts, limit=6, ticked=False)
        if rule.low is not None or rule.high is not None:
            text += f", and {_range_phrase(rule.low, rule.high)} otherwise"
        return text
    return f"{column} {_range_phrase(rule.low, rule.high)}"


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
        from turbotab.core.stages.proposals import rule_excludes

        keep = frame[target].notna() if target in frame.columns else None
        if keep is None:
            import pandas as pd

            keep = pd.Series(True, index=frame.index)
        n_before = int(keep.sum())
        out = []
        for rule in rules:
            hit = rule_excludes(frame, rule) & keep
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
    if d.strategy == "complete_case":
        other = "any other predictor" if dropped else "any predictor"
        text = f"{first}rows missing {other} were dropped (a complete-case analysis)"
        kept, before = _get(ctx, "n_complete"), _get(ctx, "n_before")
        if kept is not None and before:
            text += f": {count(kept)} of {count(before)} rows remain"
        return text[0].upper() + text[1:]
    others = "the other predictors' missing values" if dropped else "missing predictor values"
    text = (f"{first}{others} were imputed, learned from training rows only; no row was dropped "
            f"for a missing predictor")
    return text[0].upper() + text[1:]


# set_split

@register_sentence("set_split")
def _set_split(d: Any, state: Any, ctx: Any) -> str:
    task = getattr(state, "task", None) or _get(ctx, "detected_task")  # answered, else detected
    target = getattr(state, "target", None)
    repeats = _get(ctx, "repeats")
    group = _attr(repeats, "column") if repeats else None
    how = []
    if group:
        how.append(f"keeping each {tick(group)}'s rows together")
    if task in ("binary", "multiclass") and target:
        how.append(f"stratified by {tick(target)}")
    manner = " (" + ", ".join([f"seed {tick(d.seed)}", *how]) + ")"
    folds = f"{tick(d.folds)}-fold cross-validation"
    if d.holdout == 0:
        return f"No rows were held out; performance was estimated by {folds}{manner}"
    share = tick(f"{d.holdout:.0%}")
    n = _get(ctx, "n_cohort")
    pool = f" of the {count(n)} rows" if n else " of rows"
    return (f"A random {share}{pool}{manner} was held out for one final score; models were "
            f"compared by {folds} on the rest")


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
    nutrients = listing(d.nutrients) if d.nutrients else ""
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
        return (f"Energy was adjusted by the {name}: {who} regressed on {energy}{logged}{where} "
                f"on training rows and replaced by the residual plus the nutrient's mean")
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


def kinds() -> list[str]:
    """The decision kinds with a registered sentence."""
    return sorted(_SENTENCES)


__all__ = [
    "count", "exclusion_counts", "finish", "kinds", "listing", "machinery", "number", "plural",
    "register_sentence", "sentence_for", "strip_markdown", "tick", "words",
]

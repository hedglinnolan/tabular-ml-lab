"""Binary-coded text columns, and why they outrank numeric coercion.

On NHANES, `meds_hbp` and `meds_chol` are `True` / `False` with blanks. pandas
reads them as `object`, `pd.to_numeric` happily turns Python bools into 1 and 0,
so `import_doctor.check_numeric_stored_as_text` sees a 100% parse rate and
proposes **"Convert 'meds_chol' to numbers"** at *high* confidence — the tier
the interface is allowed to pre-select.

The proposal is not wrong arithmetic. It is the wrong *diagnosis*. The column is
a binary variable whose blanks are a question in their own right ("was the
medication history not asked, or not answered?"), and coercing it to numbers
answers the shape question while deleting the interesting one. So the binary
reading outranks the numeric one: where this module recognizes a binary column,
the numeric-coercion finding for that column is superseded rather than shown
beside it — two proposals for one column is the interface asking the user to
adjudicate its own internal disagreement.

`ml/import_doctor.py` is frozen as engine-move-only (`TRANSITION_PLAN.md` §05),
so this lives beside it rather than inside it, and the reconciliation happens
where the two streams already meet.

Finding: GUIDED-001.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.import_doctor import ShapeFinding

# Ordered pairs whose meaning is not in doubt: the first token is the 1.
# A pair outside this table is still binary — it is just not *this module's*
# business to decide which side is the positive one, and it says so.
KNOWN_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("true", "false"),
    ("yes", "no"),
    ("y", "n"),
    ("t", "f"),
    ("present", "absent"),
    ("positive", "negative"),
    ("pos", "neg"),
    ("1", "0"),
)

# Values that mean "no answer" rather than a level. Kept narrow on purpose:
# treating an unfamiliar token as missing is how a third level disappears.
_BLANK_TOKENS = frozenset({"", "na", "n/a", "nan", "none", "null", ".", "-", "--"})

MIN_ROWS = 5

# Levels that are conventionally the event in a clinical or trial dataset.
# Offered as a suggestion with its reasoning shown, never as a default: the
# tokens below are habits of the literature, and `alive`/`dead` is deliberately
# absent because whether the event is death or survival is the research
# question rather than a property of the data.
EVENT_CONVENTIONS = frozenset({
    "responder", "response", "case", "event", "improved", "positive", "pos",
    "yes", "true", "present", "1", "success", "relapse", "readmitted",
})


def _normalize(value: Any) -> Optional[str]:
    """One cell as a comparison token, or None when it carries no answer."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    token = str(value).strip().lower()
    return None if token in _BLANK_TOKENS else token


def _is_texty(s: pd.Series) -> bool:
    return (s.dtype == object
            or isinstance(s.dtype, pd.StringDtype)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_string_dtype(s))


def read_as_binary_plan(s: pd.Series) -> Optional[Dict[str, Any]]:
    """How this column would be read as binary, or None if it is not binary.

    Returns the mapping, the per-level counts, the missing count, and whether
    the positive level is known or merely chosen — never a silent choice.
    """
    if not _is_texty(s):
        return None
    tokens = s.map(_normalize)
    present = tokens.dropna()
    if len(present) < MIN_ROWS:
        return None
    levels = sorted(present.unique().tolist())
    if len(levels) != 2:
        return None

    positive: Optional[str] = None
    for hi, lo in KNOWN_PAIRS:
        if {hi, lo} == set(levels):
            positive = hi
            break
    known = positive is not None
    if positive is None:
        # Deterministic, and declared. Sorted order is arbitrary; saying so is
        # the difference between a choice and a guess presented as a fact.
        positive = levels[-1]
    negative = next(v for v in levels if v != positive)

    counts = {level: int((present == level).sum()) for level in levels}
    return {
        "levels": levels,
        "positive": positive,
        "negative": negative,
        "positive_known": known,
        "mapping": {positive: 1, negative: 0},
        "counts": counts,
        "n_missing": int(len(s) - len(present)),
        "n_rows": int(len(s)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HOW THE COLUMN IS WRITTEN — `GUIDED-158`
# ─────────────────────────────────────────────────────────────────────────────
#
# Every hit used to be titled *"'<col>' is a binary variable written as text"*.
# Measured on the product owner's NHANES export: **nine hits, and eight of the
# nine were not text.** Six were dtype `bool`; two were `object` holding Python
# `True`/`False`; only `gender` held strings.
#
# Nothing downstream was wrong — the repair is right for all nine and the frame
# it produces is what the user wanted — so this is cosmetic in effect and not in
# kind. It is the first card on the file, it makes nine assertions, and eight of
# them are false about something the reader checks in one glance at their own
# CSV. **The repair is the same for all of them and the CLAIM is not, and the
# claim is the part that has to be true.**
#
# The composer already holds the series, so the sentence is DERIVED from it
# rather than assumed. Where the values are not all of one type the shape is
# `mixed` and the title falls back to the weaker sentence that is certainly
# true — the same instinct as returning `None` rather than a number nobody
# computed.
BOOL_DTYPE = "bool_dtype"
OBJECT_BOOLS = "object_bools"
OBJECT_TEXT = "object_text"
OBJECT_NUMBERS = "object_numbers"
MIXED_WRITING = "mixed"


def value_shape(s: pd.Series) -> str:
    """What this column's two values actually ARE, read off the series.

    `bool_dtype` is pandas' own boolean typing — including the nullable
    `boolean` extension dtype, which `is_bool_dtype` covers. The rest are
    `object` columns told apart by the Python type of the values inside them,
    because that is the distinction a reader can check and the one the old
    sentence got wrong.
    """
    if pd.api.types.is_bool_dtype(s):
        return BOOL_DTYPE
    values = [v for v in s.dropna().tolist()]
    if not values:
        return MIXED_WRITING
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return OBJECT_BOOLS
    if all(isinstance(v, str) for v in values):
        return OBJECT_TEXT
    if all(isinstance(v, (int, float, np.integer, np.floating))
           and not isinstance(v, (bool, np.bool_)) for v in values):
        return OBJECT_NUMBERS
    return MIXED_WRITING


def written_as(column: str, shape: str, n_missing: int = 0) -> Dict[str, str]:
    """The title and the closing detail sentence for one written-as shape.

    Separated from `binary_text_finding` so the phrasing can be asserted
    directly against a shape, rather than only through a frame that happens to
    produce one.
    """
    if shape == BOOL_DTYPE:
        return {
            "title": f"'{column}' is already a true/false column",
            # A `bool` column is binary ALREADY. Calling the repair a repair
            # would be the second false claim on the same card: what it does
            # here is re-type true/false to 1/0.
            "detail": ("Its values are already true and false, so reading it "
                       "as binary re-types the column to 1 and 0 rather than "
                       "repairing anything."),
        }
    if shape == OBJECT_BOOLS:
        detail = "Its values are true and false, not text."
        if n_missing:
            # Only claimed where there ARE blanks, because "blanks are why"
            # is an explanation and an explanation of something that is not
            # there is the same class of false sentence this row is about.
            detail += (" The blanks are why pandas left the column untyped "
                       "rather than reading it as a boolean.")
        return {"title": f"'{column}' is a binary variable written as "
                         f"true/false",
                "detail": detail}
    if shape == OBJECT_NUMBERS:
        return {"title": f"'{column}' is a binary variable written as two "
                         f"numbers",
                "detail": ("Its two values are numbers, in a column pandas "
                           "left untyped.")}
    if shape == OBJECT_TEXT:
        return {"title": f"'{column}' is a binary variable written as text",
                "detail": "It is a binary variable, not a number stored as text."}
    return {
        "title": f"'{column}' is a binary variable, written more than one way",
        "detail": ("Its two values are not all of one type in this table, so "
                   "nothing more specific is claimed about how they are "
                   "written."),
    }


def _original_labels(s: pd.Series, token: str) -> List[str]:
    """The values as the file spells them, for a token we matched normalized."""
    seen: List[str] = []
    for value in s.dropna().tolist():
        if _normalize(value) == token:
            text = str(value)
            if text not in seen:
                seen.append(text)
            if len(seen) >= 3:
                break
    return seen


def positive_class_finding(column: str, s: pd.Series) -> Optional[ShapeFinding]:
    """The target's question, which is different in kind from a feature's.

    For a feature, binary-versus-numeric is a *reading*: the values mean the
    same thing either way and the question is how to store them. For the
    outcome, the reading is nearly forced — two-level text is binary
    classification — and the decision that actually matters is **which level is
    the event being predicted**.

    That choice sets the sign of every effect estimate, what sensitivity and
    specificity are the sensitivity and specificity *of*, and what the model is
    optimized to detect. So the target is never asked "is this binary?"; it is
    asked "which of these is the event you are predicting?".

    **Never pre-selected, at any confidence.** Convention may be offered as a
    suggestion with its reasoning shown — `responder`, `improved`, `case` are
    conventionally the event — but `alive`/`dead` has no correct default.
    Whether the event is death or survival is the research question, not a
    property of the data, so `auto_suggestable` is False here regardless of how
    familiar the vocabulary looks.
    """
    plan = read_as_binary_plan(s)
    if plan is None:
        return None

    levels = plan["levels"]
    spellings = {lvl: (_original_labels(s, lvl) or [lvl])[0] for lvl in levels}
    counts = plan["counts"]
    conventional = plan["positive"] if plan["positive_known"] else None
    for lvl in levels:
        if lvl in EVENT_CONVENTIONS:
            conventional = lvl
            break

    detail = (f"'{column}' is the outcome and holds two values — "
              + " and ".join(f"{spellings[lvl]!r} ({counts[lvl]:,} rows)"
                             for lvl in levels) + ".")
    if plan["n_missing"]:
        detail += f" {plan['n_missing']:,} rows have no outcome recorded."

    why = ("Which of these is the event decides the sign of every effect "
           "estimate, what sensitivity and specificity are measuring, and what "
           "the model is trained to detect. Nothing in the file says which one "
           "you are predicting.")
    if conventional:
        why += (f" By convention {spellings[conventional]!r} is usually the "
                f"event, and it is the smaller group here"
                if counts.get(conventional, 0) == min(counts.values())
                else f" By convention {spellings[conventional]!r} is usually the event")
        why += " — but convention is a suggestion, not an answer."

    return ShapeFinding(
        id=f"positive_class__{column}",
        severity="warning",
        title=f"Which of these is the event you are predicting?",
        detail=detail,
        why_it_matters=why,
        fix_label=f"Set the event for '{column}'",
        fix_kind="set_positive_class",
        # Never high: `auto_suggestable` is `confidence == "high"`, and this
        # question may not be pre-selected at any confidence.
        confidence="medium",
        params={"column": column, "is_target": True,
                "levels": levels, "spellings": spellings, "counts": counts,
                "suggested": conventional,
                "suggested_reason": (
                    f"{spellings[conventional]!r} is conventionally the event"
                    if conventional else None),
                "n_missing": plan["n_missing"], "n_rows": plan["n_rows"]},
        affected_columns=[column],
    )


def binary_text_finding(column: str, s: pd.Series) -> Optional[ShapeFinding]:
    """One finding proposing that a binary-coded column be read as binary."""
    plan = read_as_binary_plan(s)
    if plan is None:
        return None

    pos_spellings = _original_labels(s, plan["positive"]) or [plan["positive"]]
    neg_spellings = _original_labels(s, plan["negative"]) or [plan["negative"]]
    n_pos = plan["counts"][plan["positive"]]
    n_neg = plan["counts"][plan["negative"]]

    # `GUIDED-158`. WHAT THE COLUMN ACTUALLY IS, read off the series rather
    # than assumed. Everything below that used to say `written as text` now
    # says what `value_shape` found.
    shape = value_shape(s)
    said = written_as(column, shape, int(plan["n_missing"]))

    detail = (f"'{column}' holds two values — {', '.join(map(repr, pos_spellings))} "
              f"({n_pos:,} rows) and {', '.join(map(repr, neg_spellings))} "
              f"({n_neg:,} rows)")
    if plan["n_missing"]:
        detail += f", with {plan['n_missing']:,} blank"
    detail += ". " + said["detail"]

    if plan["positive_known"]:
        why = (f"Read as binary, {pos_spellings[0]} becomes 1 and "
               f"{neg_spellings[0]} becomes 0, and the column keeps its meaning: "
               "a rate, an odds ratio and a class balance are all defined for it. "
               "Coercing it to a number instead gives the same digits with no "
               "statement of what 1 means.")
    else:
        why = (f"Read as binary, {pos_spellings[0]} becomes 1 and "
               f"{neg_spellings[0]} becomes 0. Which level is the positive one "
               "is your call — nothing in the file says — and the direction of "
               "every coefficient for this variable follows from it.")

    if plan["n_missing"]:
        why += (f" The {plan['n_missing']:,} blank(s) are a separate question, "
                "asked at Preprocess: whether not-answered is itself informative.")

    return ShapeFinding(
        id=f"binary_text__{column}",
        severity="warning",
        title=said["title"],
        detail=detail,
        why_it_matters=why,
        fix_label=(f"Read '{column}' as binary "
                   f"({pos_spellings[0]} = 1, {neg_spellings[0]} = 0)"),
        fix_kind="read_as_binary",
        confidence="high" if plan["positive_known"] else "medium",
        # `written_as` is the record beside the sentence, so anything reading
        # the payload rather than the prose gets the same answer (trap #7).
        params={"column": column, "written_as": shape, **plan},
        affected_columns=[column],
    )


def detect_binary_text(df: pd.DataFrame,
                       target: Optional[str] = None) -> List[ShapeFinding]:
    """Every binary-coded column in the frame, as findings.

    The target gets a different question from a feature. For a feature the
    question is how to read the column; for the outcome the reading is nearly
    forced and the decision is which level is the event. Passing `target` routes
    that column to `positive_class_finding`; omitting it treats every column as
    a feature, which is right before a target has been chosen.
    """
    out: List[ShapeFinding] = []
    if df is None or df.empty:
        return out
    for position, column in enumerate(df.columns):
        try:
            s = df.iloc[:, position]
            if isinstance(s, pd.DataFrame):     # duplicate labels
                continue
            name = str(column)
            finding = (positive_class_finding(name, s)
                       if target is not None and name == str(target)
                       else binary_text_finding(name, s))
        except Exception:
            continue
        if finding is not None:
            out.append(finding)
    return out


def supersede_numeric_coercion(
    findings: Sequence[ShapeFinding],
    binary: Sequence[ShapeFinding],
) -> List[ShapeFinding]:
    """Merge the two streams, the binary reading winning its own columns.

    A `numeric_as_text__<col>` finding is dropped where a binary finding claims
    the same column: showing both would put two repair proposals on one column
    and make the user settle a disagreement the engine should have settled.
    """
    claimed = {c for f in binary for c in f.affected_columns}
    kept = [f for f in findings
            if not (f.fix_kind == "coerce_numeric"
                    and any(str(c) in claimed for c in f.affected_columns))]
    return list(binary) + kept


def diagnose_with_binary(df: pd.DataFrame,
                         findings: Sequence[ShapeFinding],
                         target: Optional[str] = None) -> List[ShapeFinding]:
    """`import_doctor.diagnose` output with the binary reading folded in."""
    return supersede_numeric_coercion(findings, detect_binary_text(df, target))


def apply_positive_class(df: pd.DataFrame,
                         finding: ShapeFinding,
                         event: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Encode the outcome with the chosen level as 1. Never mutates the input.

    `event` is the level the user named. It is required: there is no default,
    at any confidence, because which level is the event is the research
    question rather than a property of the data.
    """
    params = finding.params or {}
    column = params.get("column") or (finding.affected_columns or [None])[0]
    if column is None or column not in df.columns:
        raise KeyError(f"'{column}' is not a column of this table.")
    if not event:
        raise ValueError(
            "Setting the event needs the level being predicted. There is no "
            "default: whether the event is (say) death or survival is the "
            "research question, not something the file can say.")

    chosen = _normalize(event)
    levels = list(params.get("levels") or [])
    if chosen not in levels:
        raise ValueError(
            f"{event!r} is not one of the two levels of '{column}' "
            f"({', '.join(map(repr, levels))}).")
    other = next(lvl for lvl in levels if lvl != chosen)
    spellings = params.get("spellings") or {}

    out = df.copy(deep=True)
    tokens = out[column].map(_normalize)
    out[column] = tokens.map(
        lambda t: (1 if t == chosen else 0) if t is not None else None)
    out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    event_text = spellings.get(chosen, chosen)
    other_text = spellings.get(other, other)
    description = (f"'{column}' was encoded with {event_text} as the event "
                   f"(1) and {other_text} as the comparison (0).")
    n_missing = int(out[column].isna().sum())
    if n_missing:
        description += (f" {n_missing:,} row(s) have no outcome recorded and "
                        "are excluded from modeling.")
    return out, description


def apply_read_as_binary(df: pd.DataFrame,
                         finding: ShapeFinding) -> Tuple[pd.DataFrame, str]:
    """Rewrite one binary-coded text column as 0/1. Never mutates the input.

    Blanks stay blank. Whether a blank is informative is a separate decision
    that belongs to Preprocess (GUIDED-002); filling it here would answer it by
    accident.
    """
    params = finding.params or {}
    column = params.get("column") or (finding.affected_columns or [None])[0]
    if column is None or column not in df.columns:
        raise KeyError(f"'{column}' is not a column of this table.")

    plan = read_as_binary_plan(df[column]) or params
    mapping = {str(k): int(v) for k, v in (plan.get("mapping") or {}).items()}
    positive = plan.get("positive")
    negative = plan.get("negative")

    # Quote the file's own spelling in the description. The tokens above are
    # normalized for comparison; a methods sentence that says `true = 1` about a
    # column written `True` is describing a frame nobody uploaded.
    pos_text = (_original_labels(df[column], positive) or [str(positive)])[0]
    neg_text = (_original_labels(df[column], negative) or [str(negative)])[0]

    out = df.copy(deep=True)
    tokens = out[column].map(_normalize)
    out[column] = tokens.map(lambda t: mapping.get(t) if t is not None else None)
    # Int64, not float: a binary variable reads as 1 and 0, and the nullable
    # integer type keeps the blanks blank rather than turning them into NaN
    # floats that print as `1.0` beside `<NA>`.
    out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    n_missing = int(out[column].isna().sum())
    description = (f"Read '{column}' as a binary variable: {pos_text} = 1, "
                   f"{neg_text} = 0.")
    if n_missing:
        description += (f" {n_missing:,} value(s) remain missing; whether that "
                        "absence is informative is decided at Preprocess.")
    return out, description

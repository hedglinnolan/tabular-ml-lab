"""turbotab.devchecks — runtime assertions and capture, for the drive.

**Temporary development instrumentation, not a product feature.** Off unless
``TURBOTAB_DEV_CHECKS=1``, writing to ``turbotab/sessions/<timestamp>/``, which
is gitignored. Nothing here runs, imports anything expensive, or costs a
microsecond when the flag is unset.

**Not analytics.** The product owner drives this app to find bugs, not to have
their behavior studied. Nothing here measures a person: it measures the app
against its own record, and every artifact is written to their own disk. The
data is public NHANES and the committed fixtures, so there is no privacy
constraint and the capture is deliberately total — a bug you cannot reproduce is
a bug you found and then lost.

---

## Why a harness and not more tests

A test asserts a thing somebody thought to assert. A drive is somebody using the
app in an order nobody anticipated, and the interesting failures live in exactly
the states no test constructed. So the checks here run against **whatever state
the driver actually produced**, on every transition, and they are written as
invariants rather than as expectations:

* every number displayed traces to a value in the record;
* the seal's integrity, recomputed live rather than trusted;
* a decision's sentence says what its record says;
* a deferred transform leaves the working table byte-identical;
* after an edit, exactly the right things are stale — no more and no fewer;
* no post-seal operation changes a surviving row's index label;
* a finding claiming N features names N;
* ``router.audit()`` passed before this render, not only before scoring;
* every decision taken appears in the record.

**A violation records and continues.** One bug must not end a drive: the driver
is looking for the second and third bug too, and a harness that halts on the
first has cost more than it found. Violations are appended to
``violations.jsonl`` and lead ``index.md``.

## The silent wells

`FEATURE_PARITY.md`'s family of silences has a member the ledger names a dozen
times: *"and nothing surfaces it"*. A bug hunt through code whose failure mode
is silence finds nothing until the silence is removed.

Two layers, because two different problems:

1. **Code we own** calls :func:`swallowed` at the site, which carries the
   semantic context a stack trace cannot — *what was being computed and what the
   user therefore did not see*.
2. **Code we do not own** — `ml/import_doctor.py` and its neighbors are frozen
   under `TRANSITION_PLAN.md` §05, and instrumenting them would be new
   construction on a frozen path. So nothing is edited: ``sys.monitoring``'s
   ``EXCEPTION_HANDLED`` event reports every exception caught anywhere beneath
   the Guided path, filtered to this repository's own files. Zero edits, total
   coverage, and it switches off with the flag.

The second layer catches wells the first has not been taught about yet, which is
the whole point — the ones we know about are the ones we already reason about.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ENV_FLAG = "TURBOTAB_DEV_CHECKS"
SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def enabled() -> bool:
    """True when the driver asked for instrumentation. Off is the default."""
    return os.environ.get(ENV_FLAG, "").strip() not in ("", "0", "false", "no")


# ─────────────────────────────────────────────────────────────────────────────
# The session directory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """One drive. Created lazily on the first thing worth writing down."""

    root: Path
    started_at: str
    violations: List[Dict[str, Any]] = field(default_factory=list)
    swallows: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    console: List[Dict[str, Any]] = field(default_factory=list)
    audits: int = 0
    n_dom: int = 0

    def append(self, name: str, row: Dict[str, Any]) -> None:
        path = self.root / name
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


_SESSION: Optional[Session] = None


def session() -> Optional[Session]:
    """The current session, created on first use. ``None`` when disabled."""
    global _SESSION
    if not enabled():
        return None
    if _SESSION is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = SESSIONS_DIR / stamp
        root.mkdir(parents=True, exist_ok=True)
        (root / "dom").mkdir(exist_ok=True)
        _SESSION = Session(root=root, started_at=stamp)
        _SESSION.write("README.md", _SESSION_README)
    return _SESSION


def reset_for_test(root: Optional[Path] = None) -> Optional[Session]:
    """Point the harness at a throwaway directory. Tests only."""
    global _SESSION
    _SESSION = None
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    (root / "dom").mkdir(exist_ok=True)
    _SESSION = Session(root=root, started_at="test")
    # Written here too, so a test session is the same object a drive produces.
    # A fixture that differs from the thing it stands in for tests the fixture.
    _SESSION.write("README.md", _SESSION_README)
    return _SESSION


_SESSION_README = """# A drive

Written by `turbotab/devchecks.py` because `TURBOTAB_DEV_CHECKS=1` was set.
Development instrumentation; nothing here is a product feature.

* `index.md` — read this. It leads with violations.
* `violations.jsonl` — one per line, each naming the check and the action.
* `swallowed.jsonl` — exceptions caught somewhere and never surfaced.
* `actions.jsonl` — every request and response, in order. Replayable.
* `state/` — the resolved project before and after each action, plus the diff.
* `dom/` — one snapshot per render, styles inline, opens in a browser.
* `console.jsonl` — browser errors and unhandled rejections.
* `replay.py` — re-runs `actions.jsonl` against a fresh server.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Violations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Violation:
    check: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "message": self.message, "detail": self.detail}


def record_violations(vs: Sequence[Violation], action: Dict[str, Any]) -> None:
    """Write violations down and keep going. Never raises."""
    s = session()
    if s is None or not vs:
        return
    for v in vs:
        row = {"at": _now(), "action": action, **v.to_dict()}
        s.violations.append(row)
        s.append("violations.jsonl", row)


def swallowed(where: str, exc: BaseException, note: str = "") -> None:
    """A well that would otherwise have kept its contents.

    Called at the site, because the site knows what the user therefore did not
    see — which is the half a stack trace cannot supply. Safe to call when the
    harness is off: it returns immediately.
    """
    s = session()
    if s is None:
        return
    row = {"at": _now(), "layer": "explicit", "where": where,
           "type": type(exc).__name__, "message": str(exc)[:400], "note": note}
    s.swallows.append(row)
    s.append("swallowed.jsonl", row)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — every handled exception under our own files, without editing them
# ─────────────────────────────────────────────────────────────────────────────

# The app's own vocabulary for saying no. These are the REFUSE branch of the
# governing rule working correctly; reporting them as swallows would bury the
# real wells under the app speaking properly.
_DELIBERATE = {
    "ProjectError", "GrainContradiction", "EngineRefusal", "HTTPException",
    "FeatureRefusal", "SelectionRefusal", "MissingnessRefusal",
    "EligibilityRefusal", "ModelSelectionError", "RecipeError", "RouterError",
    "ObligationError", "Cancelled", "StopIteration", "StopAsyncIteration",
    "GeneratorExit", "KeyboardInterrupt",
}

# Only our own code. A pandas internal catching a TypeError to pick a fast path
# is not a well, it is how pandas works.
_WATCHED_DIRS = ("turbotab", "ml", "utils", "models")

_monitoring_on = False


def _watched(filename: str) -> bool:
    try:
        rel = Path(filename).resolve().relative_to(_REPO_ROOT)
    except (ValueError, OSError):
        return False
    parts = rel.parts
    if not parts or parts[0] not in _WATCHED_DIRS:
        return False
    # The harness must not report itself, and a test's own try/except is the
    # test doing its job.
    return not (parts[-1].startswith("test_") or parts[-1] == "devchecks.py"
                or ".venv" in parts)


def _on_exception_handled(code, offset, exc) -> None:      # pragma: no cover
    name = type(exc).__name__
    if name in _DELIBERATE or not _watched(code.co_filename):
        return
    s = session()
    if s is None:
        return
    try:
        rel = str(Path(code.co_filename).resolve().relative_to(_REPO_ROOT))
    except (ValueError, OSError):
        rel = code.co_filename
    row = {"at": _now(), "layer": "monitoring", "where": f"{rel}::{code.co_name}",
           "type": name, "message": str(exc)[:400], "note": ""}
    s.swallows.append(row)
    s.append("swallowed.jsonl", row)


def start_listening() -> bool:
    """Report every exception handled inside this repository's own code.

    Uses ``sys.monitoring`` (3.12+) so nothing in the frozen engine modules is
    edited — instrumenting them would be new construction on a frozen path, and
    the freeze permits repair only.
    """
    global _monitoring_on
    if _monitoring_on or not enabled():
        return False
    mon = getattr(sys, "monitoring", None)
    if mon is None:                                        # pragma: no cover
        return False
    try:
        mon.use_tool_id(mon.DEBUGGER_ID, "turbotab-devchecks")
        mon.register_callback(mon.DEBUGGER_ID,
                              mon.events.EXCEPTION_HANDLED, _on_exception_handled)
        mon.set_events(mon.DEBUGGER_ID, mon.events.EXCEPTION_HANDLED)
    except (ValueError, RuntimeError):                     # pragma: no cover
        # Another tool holds the id — a debugger, or a second server in-process.
        # Losing layer 2 must not take the rest of the harness with it.
        return False
    _monitoring_on = True
    return True


def stop_listening() -> None:
    global _monitoring_on
    if not _monitoring_on:
        return
    mon = getattr(sys, "monitoring", None)
    if mon is not None:                                    # pragma: no cover
        try:
            mon.set_events(mon.DEBUGGER_ID, 0)
            mon.free_tool_id(mon.DEBUGGER_ID)
        except (ValueError, RuntimeError):
            pass
    _monitoring_on = False


# ─────────────────────────────────────────────────────────────────────────────
# Numbers
# ─────────────────────────────────────────────────────────────────────────────

# Column names carry digits — `bp_1`, `mz_0001`, `item_05` — and every one of
# them would read as an unsupported number. The app puts objects in backticks
# and quotes, so those spans come out before anything is counted.
_QUOTED = re.compile(r"`[^`]*`|'[^']*'|\"[^\"]*\"")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> List[float]:
    """Every number a sentence claims, with quoted objects removed first."""
    if not text:
        return []
    stripped = _QUOTED.sub(" ", str(text))
    out: List[float] = []
    for m in _NUMBER.finditer(stripped):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:                                 # pragma: no cover
            continue
    return out


def supported_numbers(obj: Any, depth: int = 0) -> set:
    """Every number the RECORD holds, plus the forms a sentence may render it in.

    A fraction of 0.15 is displayed as `15%`; a count of 1200 as `1,200` (the
    comma is stripped before comparison). Both are the same claim in the record,
    so both are supported. Anything left over is a number with no source.
    """
    found: set = set()
    _collect(obj, found, depth)
    derived: set = set()
    for v in found:
        derived.add(v)
        derived.add(round(v))
        if 0.0 < v < 1.0:
            derived.add(round(v * 100))
            derived.add(round(v * 100, 1))
    # A sentence may say "1 column" where the record holds a list of one.
    return derived


def _collect(obj: Any, into: set, depth: int) -> None:
    if depth > 8:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        try:
            into.add(float(obj))
        except (TypeError, ValueError, OverflowError):     # pragma: no cover
            pass
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, into, depth + 1)
        # A list's LENGTH is a number the record holds as surely as a count
        # field is — "3 columns were dropped" beside `columns: [a, b, c]`.
        for v in obj.values():
            if isinstance(v, (list, tuple)):
                into.add(float(len(v)))
        return
    if isinstance(obj, (list, tuple)):
        into.add(float(len(obj)))
        for v in obj:
            _collect(v, into, depth + 1)
        return


# ─────────────────────────────────────────────────────────────────────────────
# The checks
#
# Each returns a list of violations and never raises: a check that crashes the
# drive it is instrumenting has cost more than it found. `_guard` is what makes
# that structural rather than remembered.
# ─────────────────────────────────────────────────────────────────────────────

def _guard(fn, *args, **kwargs) -> List[Violation]:
    try:
        return list(fn(*args, **kwargs) or [])
    except Exception as exc:                               # pragma: no cover
        return [Violation("check_itself_failed",
                          f"{fn.__name__} raised {type(exc).__name__}: {exc}",
                          {"traceback": traceback.format_exc()[-1500:]})]


# Which part of the record each served sentence is a claim ABOUT.
#
# THE SCOPE IS THE WHOLE CHECK, and the first draft got it wrong in a way worth
# writing down. It built one supported set from the ENTIRE project payload —
# findings, profile, every column summary — which on a real table is several
# hundred numbers, so almost any small integer was "supported" and the check
# could not fail. Driven against a deliberately planted bug (a seal disclosure
# reporting 40 held-out rows on a project that sealed 90) it reported nothing.
#
# That is this project's own governing failure inside its own instrument: a
# green line asserting something false. Scoped per claim, the same planted bug
# fires immediately, because 40 is not in {90, 600, 45, 0.15}.
_DISCLOSURE_SCOPE = {
    "seal": ("lockbox",),
    "grain": ("grain",),
    "eligibility": ("eligibility",),
    "preprocess": ("missingness",),
    "attested": ("grain",),
}


def every_number_displayed_traces_to_the_record(after: Dict[str, Any]
                                                ) -> List[Violation]:
    """A number on screen with no source in the record is an assertion.

    Scoped to the sentences the RECORD produces — decision texts and the served
    disclosures — because those are the app's claims about its own state.
    Engine findings are the engine's claims and are checked separately.

    Each claim is checked against **the part of the record it is about**, not
    against the record as a whole. A seal sentence may quote the lockbox; it may
    not quote a number that happens to appear in an unrelated finding's params.
    """
    out: List[Violation] = []
    frame = {"n_rows": after.get("n_rows"), "n_columns": after.get("n_columns"),
             "n_working_rows": after.get("n_working_rows")}

    for d in after.get("decisions") or []:
        scope = {"payload": d.get("payload") or {}, **frame}
        _check_claim(f"decision::{d.get('kind')}", d.get("text") or "",
                     scope, out)

    for key, text in (after.get("disclosures") or {}).items():
        if not isinstance(text, str):
            continue
        fields = _DISCLOSURE_SCOPE.get(key)
        if fields is None:
            # An unmapped disclosure is checked against the frame alone, which
            # is strict. A new disclosure that trips this needs a scope entry —
            # deciding what a sentence is allowed to quote is the author's job,
            # not something to be inferred from what it happens to say.
            scope = dict(frame)
        else:
            scope = {f: after.get(f) for f in fields}
            scope.update(frame)
        _check_claim(f"disclosure::{key}", text, scope, out)
    return out


def _check_claim(where: str, text: str, scope: Dict[str, Any],
                 out: List[Violation]) -> None:
    supported = supported_numbers(scope)
    for n in numbers_in(text):
        if n in supported or round(n) in supported:
            continue
        out.append(Violation(
            "number_with_no_source",
            f"{where} displays {n:g}, which is not in the part of the record "
            f"that sentence is about",
            {"sentence": text[:300], "number": n,
             "scope": sorted(k for k in scope if scope.get(k) is not None)}))


def the_seal_states_its_own_basis(project, after: Dict[str, Any]) -> List[Violation]:
    """Recomputed live, not trusted.

    Two claims, and they fail differently. A GROUPED seal claims no subject sits
    on both sides, which is checkable against the frame and is the whole reason
    the seal exists. An EXPLORATORY seal claims it is not a clean lock, and
    `IMPORT-020` is what happens when that claim renders like a confident one.
    """
    import pandas as pd
    from turbotab import grain as G

    lockbox = after.get("lockbox")
    if not lockbox:
        return []
    out: List[Violation] = []
    basis = lockbox.get("seal_basis")
    if basis not in ("grouped", "cross_sectional", "undetermined",
                     "repetition_found_grouping_abandoned"):
        out.append(Violation("seal_basis_is_not_one_of_the_four",
                             f"seal_basis is {basis!r}", {"lockbox": lockbox}))

    # The exploratory claim must survive the trip to the interface.
    served = (after.get("disclosures") or {}).get("exploratory")
    expected = G.is_exploratory_basis(basis)
    if served is not None and bool(served) != bool(expected):
        out.append(Violation(
            "an_exploratory_seal_rendered_as_a_clean_lock",
            f"basis {basis!r} is exploratory={expected} and the interface was "
            f"told exploratory={served}",
            {"basis": basis}))

    # The sentence the interface shows must be the record's own sentence.
    served_text = (after.get("disclosures") or {}).get("seal")
    if served_text:
        recomputed = G.seal_disclosure(lockbox)
        if not served_text.startswith(recomputed[:40]):
            out.append(Violation(
                "the_seal_sentence_is_not_the_records_own",
                "the served seal disclosure does not begin with the sentence "
                "the recorded basis produces",
                {"served": served_text[:200], "recomputed": recomputed[:200]}))

    # GROUPED: recompute the thing the seal is FOR.
    if basis == "grouped":
        group_col = lockbox.get("group_col")
        labels = set(lockbox.get("labels") or [])
        df = getattr(project, "df", None)
        if group_col and df is not None and group_col in df.columns:
            held = df.index.map(lambda l: l in labels)
            test_groups = set(df.loc[held, group_col].dropna().unique())
            train_groups = set(df.loc[~pd.Series(held, index=df.index), group_col]
                               .dropna().unique())
            both = test_groups & train_groups
            if both:
                out.append(Violation(
                    "a_grouped_seal_has_subjects_on_both_sides",
                    f"{len(both)} value(s) of {group_col!r} appear in both the "
                    f"held-out rows and the training rows",
                    {"examples": [str(v) for v in list(both)[:5]],
                     "group_col": group_col}))
    return out


def decision_sentences_match_their_records(after: Dict[str, Any]) -> List[Violation]:
    """A sentence is the record's claim about itself, so the two must agree.

    Structural, per kind — never a substring search. `"model" in message` is a
    wildcard wearing an assertion's clothes (`FEATURE_PARITY.md`).
    """
    from turbotab import grain as G

    out: List[Violation] = []
    for d in after.get("decisions") or []:
        kind, payload, text = d.get("kind"), d.get("payload") or {}, d.get("text") or ""

        if kind == "set_grain":
            recomputed = G.seal_basis(payload.get("answer"),
                                      payload.get("group_col"),
                                      payload.get("n_groups"))
            if payload.get("basis") != recomputed:
                out.append(Violation(
                    "a_grain_decision_records_a_basis_its_answer_does_not_produce",
                    f"answer {payload.get('answer')!r} produces {recomputed!r}, "
                    f"recorded {payload.get('basis')!r}", {"decision": d.get("id")}))
            col = payload.get("group_col")
            if col and f"'{col}'" not in text and f"`{col}`" not in text:
                out.append(Violation(
                    "a_grain_decision_does_not_name_its_identifier",
                    f"the recorded group column {col!r} is not in the sentence",
                    {"sentence": text[:200]}))

        elif kind == "seal_lockbox":
            lb = after.get("lockbox") or {}
            if payload.get("n_test") is not None and lb.get("n_test") is not None:
                if int(payload["n_test"]) != int(lb["n_test"]):
                    out.append(Violation(
                        "the_seal_decision_and_the_lockbox_disagree_on_n",
                        f"decision says {payload['n_test']}, lockbox holds "
                        f"{lb['n_test']}", {"decision": d.get("id")}))
            if payload.get("seal_basis") != lb.get("seal_basis"):
                out.append(Violation(
                    "the_seal_decision_and_the_lockbox_disagree_on_basis",
                    f"decision says {payload.get('seal_basis')!r}, lockbox holds "
                    f"{lb.get('seal_basis')!r}", {"decision": d.get("id")}))

        elif kind == "set_eligibility":
            n = payload.get("n_excluded")
            if n is not None and float(n) not in set(numbers_in(text)) and float(n) != 0:
                out.append(Violation(
                    "an_eligibility_decision_does_not_state_the_n_it_removed",
                    f"n_excluded={n} does not appear in the sentence",
                    {"sentence": text[:200]}))

        # Every kind: the sentence may not carry a number its own payload does
        # not, which is the per-decision form of the global check.
        local = supported_numbers({"payload": payload,
                                   "n_rows": after.get("n_rows"),
                                   "n_columns": after.get("n_columns")})
        for n in numbers_in(text):
            if n not in local and round(n) not in local:
                out.append(Violation(
                    "a_decision_sentence_carries_a_number_its_payload_does_not",
                    f"{kind} sentence displays {n:g}",
                    {"sentence": text[:300], "payload_keys": sorted(payload)}))
    return out


# What each action is allowed to do to the working table and to the cascade.
# `stale` is the exact number of new `stale_downstream` entries; `None` means
# the action's effect depends on data the table cannot state in advance.
@dataclass(frozen=True)
class _Expected:
    touches_table: bool
    stale: Optional[int]
    records: bool = True


ACTION_CONTRACT: Dict[str, _Expected] = {
    # Recorded and deferred: nothing may reach the working table.
    "defer_feature":         _Expected(touches_table=False, stale=0),
    "set_selection":         _Expected(touches_table=False, stale=1),
    "select_models":         _Expected(touches_table=False, stale=1),
    "set_preparation_mode":  _Expected(touches_table=False, stale=0),
    "set_model_recipe":      _Expected(touches_table=False, stale=0),
    "settle_features":       _Expected(touches_table=False, stale=0),
    "settle_preprocess":     _Expected(touches_table=False, stale=0),
    "seal":                  _Expected(touches_table=False, stale=0),
    "set_target":            _Expected(touches_table=False, stale=0),
    "set_task_type":         _Expected(touches_table=False, stale=0),
    "set_grain":             _Expected(touches_table=False, stale=0),
    "defer":                 _Expected(touches_table=False, stale=0),
    "dismiss":               _Expected(touches_table=False, stale=0),
    "undismiss":             _Expected(touches_table=False, stale=0),
    "flag":                  _Expected(touches_table=False, stale=0),
    "unflag":                _Expected(touches_table=False, stale=0),
    "note":                  _Expected(touches_table=False, stale=0),
    "acknowledge_blocker":   _Expected(touches_table=False, stale=0),
    # A read served through the decision endpoint. Records nothing on purpose.
    "eligibility_evidence":  _Expected(touches_table=False, stale=0, records=False),
    # Executed now, because they are row-local under constitution §06.
    # `GUIDED-165`: setting a physiologically impossible entry to missing uses
    # nothing but that row's own cell, so it executes and posts a receipt. It
    # needed its own kind precisely because it used to be a `note`, and `note`
    # is declared `touches_table=False` two blocks up — the contract that would
    # have caught the defect was correct and the kind was wrong.
    "set_impossible_missing": _Expected(touches_table=True, stale=1),
    "keep_impossible":       _Expected(touches_table=False, stale=0),
    "add_feature":           _Expected(touches_table=True, stale=1),
    "remove_feature":        _Expected(touches_table=True, stale=1),
    "apply":                 _Expected(touches_table=True, stale=None),
    "revert":                _Expected(touches_table=True, stale=None),
    "resolve_blocker":       _Expected(touches_table=True, stale=None),
    # Data-dependent: two of the strategies are row-local and the rest are not;
    # an eligibility criterion that excludes nobody changes nothing.
    "route_missingness":     _Expected(touches_table=None, stale=1),
    "set_eligibility":       _Expected(touches_table=None, stale=None),
    "trim_training_rows":    _Expected(touches_table=True, stale=1),
}


def a_deferred_transform_leaves_the_table_byte_identical(
        kind: str, before: Dict[str, Any], after: Dict[str, Any]) -> List[Violation]:
    """Constitution §06's litmus, watched rather than trusted.

    Materializing a stateful transform on the working table pre-split is the
    canonical preprocessing leak, and it is invisible: the table simply has
    better numbers in it. The fingerprint is a content hash of values, labels and
    dtypes, so "nothing was touched" becomes a comparison instead of a claim.
    """
    spec = ACTION_CONTRACT.get(kind)
    if spec is None or spec.touches_table is not False:
        return []
    if before.get("fingerprint") and before["fingerprint"] != after.get("fingerprint"):
        return [Violation(
            "a_deferred_action_changed_the_working_table",
            f"{kind} is recorded and not executed, and the table's content hash "
            f"changed",
            {"before": before["fingerprint"][:16], "after": (after.get("fingerprint") or "")[:16],
             "n_rows_before": before.get("n_rows"), "n_rows_after": after.get("n_rows")})]
    return []


def after_an_edit_exactly_the_right_things_are_stale(
        kind: str, before: Dict[str, Any], after: Dict[str, Any]) -> List[Violation]:
    """Too little invalidation is a wrong number; too much is a lost afternoon.

    "Exactly" is the word that matters. A cascade that over-fires trains the user
    to ignore it, which is the blocker-budget argument applied to staleness.
    """
    spec = ACTION_CONTRACT.get(kind)
    if spec is None or spec.stale is None:
        return []
    grew = len(after.get("stale_downstream") or []) - len(before.get("stale_downstream") or [])
    if grew != spec.stale:
        return [Violation(
            "the_cascade_fired_the_wrong_number_of_times",
            f"{kind} should add {spec.stale} stale entr(ies) and added {grew}",
            {"added": [e.get("why") for e in
                       (after.get("stale_downstream") or [])[len(before.get("stale_downstream") or []):]]})]
    return []


def no_post_seal_operation_changes_a_surviving_rows_label(
        project, before: Dict[str, Any], after: Dict[str, Any]) -> List[Violation]:
    """Decision A's identity barrier, checked on the frame rather than by kind.

    The preview engine reports renumbering by CONTENT rather than by fix kind,
    and this is the same reading one layer out: whatever the operation called
    itself, did any surviving row end up with a different name?
    """
    if not before.get("barrier_raised"):
        return []
    out: List[Violation] = []
    lockbox = after.get("lockbox") or {}
    df = getattr(project, "df", None)
    if df is None:
        return []
    live = set(df.index)
    missing = [l for l in (lockbox.get("labels") or []) if l not in live]
    if missing:
        out.append(Violation(
            "a_sealed_row_label_no_longer_names_a_row",
            f"{len(missing)} sealed label(s) are gone from the table; the "
            f"quarantine no longer refers to the rows it was drawn from",
            {"examples": [str(m) for m in missing[:5]]}))
    return out


def a_finding_claiming_n_features_names_n(after: Dict[str, Any]) -> List[Violation]:
    """A count in a title, against the list the finding actually carries.

    Only the unambiguous mappings are checked. A title saying "4 groups" beside
    40 affected columns is correct and would be a false alarm here, so `group`
    is read against `params.families` and never against the column list.
    """
    pattern = re.compile(
        r"(\d[\d,]*)\s+(column|feature|group|row|value|cell)s?\(?s?\)?",
        re.IGNORECASE)
    out: List[Violation] = []
    for f in after.get("findings") or []:
        params = f.get("params") or {}
        title = f.get("title") or ""
        for m in pattern.finditer(title):
            claimed = int(m.group(1).replace(",", ""))
            noun = m.group(2).lower()
            actual = None
            if noun in ("column", "feature"):
                if isinstance(params.get("columns"), list):
                    actual = len(params["columns"])
                elif f.get("affected_columns"):
                    actual = len(f["affected_columns"])
            elif noun == "group" and isinstance(params.get("families"), dict):
                actual = len(params["families"])
            elif noun == "value" and isinstance(params.get("values"), list):
                actual = len(params["values"])
            if actual is None or claimed == actual:
                continue
            out.append(Violation(
                "a_finding_claims_a_count_it_does_not_name",
                f"{f.get('id')} says {claimed} {noun}(s) and names {actual}",
                {"title": title, "finding": f.get("id")}))
    return out


def every_decision_taken_appears_in_the_record(
        kind: str, before: Dict[str, Any], after: Dict[str, Any]) -> List[Violation]:
    """An action that changes state and records nothing is an unfalsifiable
    change. The record is what the manuscript is written from."""
    spec = ACTION_CONTRACT.get(kind)
    if spec is None:
        return []
    grew = len(after.get("decisions") or []) - len(before.get("decisions") or [])
    if spec.records and grew < 1:
        return [Violation(
            "an_action_was_taken_and_nothing_was_recorded",
            f"{kind} appended no decision",
            {"n_decisions": len(after.get("decisions") or [])})]
    if not spec.records and grew:
        return [Violation(
            "a_read_recorded_a_decision",
            f"{kind} is a read and appended {grew} decision(s)", {})]
    return []


def router_audit_passed_before_this_render(questions, rendered) -> List[Violation]:
    """`router.audit()` before RENDERING, not only before scoring.

    A plan that breaks a governing rule has no number, it has a failure — and a
    plan that is rendered without being audited has neither. The audit is
    re-run here against the same list the interface receives, so what was
    audited and what is shown cannot be two different things.
    """
    from ml import router
    out: List[Violation] = []
    try:
        router.audit(questions)
    except Exception as exc:
        out.append(Violation("the_rendered_plan_fails_the_routers_own_audit",
                             f"{type(exc).__name__}: {exc}", {}))
    if len(rendered) != len(questions):
        out.append(Violation(
            "the_rendered_plan_is_not_the_audited_plan",
            f"audited {len(questions)} question(s), rendering {len(rendered)}", {}))
    s = session()
    if s is not None:
        s.audits += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The whole battery, run on one transition
# ─────────────────────────────────────────────────────────────────────────────

def check_transition(project, before: Optional[Dict[str, Any]],
                     after: Dict[str, Any], action: Dict[str, Any]
                     ) -> List[Violation]:
    """Every check, on one state transition. Records and continues."""
    if not enabled():
        return []
    kind = str(action.get("kind") or "")
    before = before or {}
    vs: List[Violation] = []
    vs += _guard(every_number_displayed_traces_to_the_record, after)
    vs += _guard(the_seal_states_its_own_basis, project, after)
    vs += _guard(decision_sentences_match_their_records, after)
    vs += _guard(a_deferred_transform_leaves_the_table_byte_identical, kind, before, after)
    vs += _guard(after_an_edit_exactly_the_right_things_are_stale, kind, before, after)
    vs += _guard(no_post_seal_operation_changes_a_surviving_rows_label, project, before, after)
    vs += _guard(a_finding_claiming_n_features_names_n, after)
    vs += _guard(every_decision_taken_appears_in_the_record, kind, before, after)
    record_violations(vs, action)
    return vs


# ─────────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def step_of(state: Dict[str, Any]) -> str:
    """The furthest step this record has reached. For labeling snapshots only."""
    if not state:
        return "upload"
    if state.get("preprocess_settled"):
        return "preprocess"
    if state.get("features_settled"):
        return "features"
    if state.get("barrier_raised"):
        return "explore"
    if state.get("target"):
        return "target"
    return "data"


def capture_action(action: Dict[str, Any], before: Optional[Dict[str, Any]],
                   after: Optional[Dict[str, Any]]) -> None:
    """Everything needed to reproduce one action: the wire, and both states.

    The state is written whole on both sides and diffed, so an unexpected change
    is a diff rather than a guess about what moved. That is the difference
    between "something went stale" and "`selected_models` emptied when the
    recipe was set".
    """
    s = session()
    if s is None:
        return
    n = len(s.actions) + 1
    row = {"seq": n, "at": _now(), **action}
    s.actions.append(row)
    s.append("actions.jsonl", row)

    state_dir = s.root / "state"
    state_dir.mkdir(exist_ok=True)
    if before is not None:
        (state_dir / f"{n:04d}-before.json").write_text(
            json.dumps(before, indent=2, default=str), encoding="utf-8")
    if after is not None:
        (state_dir / f"{n:04d}-after.json").write_text(
            json.dumps(after, indent=2, default=str), encoding="utf-8")
    if before is not None and after is not None:
        diff = diff_state(before, after)
        (state_dir / f"{n:04d}-diff.json").write_text(
            json.dumps(diff, indent=2, default=str), encoding="utf-8")


def diff_state(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level keys that moved, with both readings. Deliberately shallow.

    A deep diff of a project carrying every finding is unreadable, and the
    question a driver asks is *what changed*, not *which byte*.
    """
    out: Dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key), after.get(key)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            out[key] = {"before_len": len(a), "after_len": len(b),
                        "added": b[len(a):] if len(b) > len(a) else None}
        elif isinstance(a, dict) and isinstance(b, dict):
            out[key] = {"changed_keys": sorted(
                k for k in set(a) | set(b) if a.get(k) != b.get(k))}
        else:
            out[key] = {"before": a, "after": b}
    return out


def capture_dom(step: str, html: str) -> Optional[str]:
    """One snapshot per render. Styles are already inline in `index.html`, so
    the file opens in a browser with nothing else beside it."""
    s = session()
    if s is None:
        return None
    s.n_dom += 1
    name = f"dom/{s.n_dom:04d}-{re.sub(r'[^a-z0-9]+', '-', step.lower())}.html"
    s.write(name, html)
    return name


def capture_console(level: str, message: str, stack: str = "",
                    url: str = "") -> None:
    s = session()
    if s is None:
        return
    row = {"at": _now(), "level": level, "message": message[:2000],
           "stack": stack[:4000], "url": url}
    s.console.append(row)
    s.append("console.jsonl", row)


# ─────────────────────────────────────────────────────────────────────────────
# index.md — violations first, narrative last
# ─────────────────────────────────────────────────────────────────────────────

def write_index() -> Optional[Path]:
    """The one file a driver reads. It opens with what went wrong.

    Deliberately not a report. A drive's narrative is available in
    `actions.jsonl` and nobody needs it summarized; what nobody can reconstruct
    by reading is the list of invariants that broke, so that is what the first
    screen holds. If there are none, it says so in one line and gets out of the
    way.
    """
    s = session()
    if s is None:
        return None

    lines: List[str] = [f"# Drive {s.started_at}", ""]

    if s.violations:
        by_check: Dict[str, List[Dict[str, Any]]] = {}
        for v in s.violations:
            by_check.setdefault(v["check"], []).append(v)
        lines += [f"## {len(s.violations)} violation(s), "
                  f"{len(by_check)} distinct check(s)", ""]
        for check, rows in sorted(by_check.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"### `{check}` — {len(rows)}")
            lines.append("")
            for r in rows[:6]:
                act = r.get("action") or {}
                where = act.get("kind") or act.get("path") or "?"
                lines.append(f"- **{where}** — {r['message']}")
                detail = r.get("detail") or {}
                for k in ("sentence", "served", "recomputed", "examples", "added"):
                    if detail.get(k):
                        lines.append(f"    - `{k}`: {json.dumps(detail[k], default=str)[:300]}")
            if len(rows) > 6:
                lines.append(f"- …and {len(rows) - 6} more in `violations.jsonl`")
            lines.append("")
    else:
        lines += ["## No violations.", "",
                  "Every check passed on every transition in this drive. That is "
                  "a statement about the checks as much as about the app — see "
                  "the coverage line below.", ""]

    if s.swallows:
        by_where: Dict[str, int] = {}
        for row in s.swallows:
            by_where[f"{row['where']} · {row['type']}"] = \
                by_where.get(f"{row['where']} · {row['type']}", 0) + 1
        lines += [f"## {len(s.swallows)} swallowed exception(s)", "",
                  "Caught somewhere and never surfaced. Not all of these are "
                  "defects — some are legitimate fallbacks — but every one is a "
                  "place where the app's failure mode is silence.", ""]
        for where, n in sorted(by_where.items(), key=lambda kv: -kv[1])[:20]:
            lines.append(f"- `{where}` × {n}")
        lines.append("")

    if s.console:
        errs = [c for c in s.console if c["level"] in ("error", "unhandledrejection")]
        lines += [f"## {len(errs)} browser error(s)", ""]
        for c in errs[:10]:
            lines.append(f"- `{c['level']}` {c['message'][:200]}")
        lines.append("")

    lines += ["## Coverage", "",
              f"- {len(s.actions)} action(s) captured, with state before and after",
              f"- {s.n_dom} DOM snapshot(s)",
              f"- {s.audits} router audit(s) re-run before rendering",
              f"- {len(s.console)} console message(s)",
              "",
              "## Replay", "",
              "```bash",
              f"turbotab/.venv/bin/python {s.root}/replay.py",
              "```", ""]
    _write_replay(s)
    return s.write("index.md", "\n".join(lines))


_REPLAY = '''"""Replay this drive against a fresh server.

Reads `actions.jsonl` in this directory and re-issues every request in order,
rewriting the project id as it goes — the id is minted on upload and every later
path carries it, so a literal replay would 404 on the second request.

    turbotab/.venv/bin/python {root}/replay.py [--base http://127.0.0.1:8777]
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8777")
    args = ap.parse_args()
    try:
        import httpx
    except ImportError:
        sys.exit("replay needs httpx: turbotab/.venv/bin/pip install httpx")

    rows = [json.loads(l) for l in
            (HERE / "actions.jsonl").read_text().splitlines() if l.strip()]
    project_id = None
    with httpx.Client(base_url=args.base, timeout=120) as client:
        for row in rows:
            path, method = row["path"], row["method"]
            if row.get("project_id") and project_id:
                path = path.replace(row["project_id"], project_id)
            if row.get("upload_filename"):
                fixture = pathlib.Path(row["upload_filename"])
                if not fixture.exists():
                    fixture = (HERE.parents[2] / "sample_data" / fixture.name)
                with fixture.open("rb") as fh:
                    r = client.post(path, files={"file": (fixture.name, fh, "text/csv")})
            elif method == "GET":
                r = client.get(path)
            else:
                r = client.request(method, path, json=row.get("request_body"))
            if r.status_code == 200 and not project_id:
                try:
                    project_id = r.json().get("id")
                except Exception:
                    pass
            print(f"{row['seq']:4d} {method:5s} {path[:70]:70s} -> {r.status_code}")


if __name__ == "__main__":
    main()
'''


def _write_replay(s: Session) -> None:
    # `str.format` and not `.replace` was the first draft, and it raised
    # `KeyError: '"file"'` on the template's own dict literal — the harness
    # crashing inside the endpoint it instruments, which is the failure
    # `safely()` now makes structurally impossible.
    s.write("replay.py", _REPLAY.replace("{root}", str(s.root)))


def safely(fn, *args, **kwargs) -> None:
    """Run a harness operation. Never let it reach the app.

    The harness's own contract: *a violation records and continues*. A harness
    that raises has not merely failed to report a bug, it has ended the drive
    that would have found the next one — and it does so wearing a stack trace
    that points at the app.

    So this is the only way the API calls into the harness, and a failure here
    becomes a violation of a check named after the harness itself.
    """
    if not enabled():
        return
    try:
        fn(*args, **kwargs)
    except Exception as exc:                               # pragma: no cover
        try:
            s = session()
            if s is not None:
                row = {"at": _now(), "check": "the_harness_itself_failed",
                       "message": f"{getattr(fn, '__name__', fn)} raised "
                                  f"{type(exc).__name__}: {exc}",
                       "detail": {"traceback": traceback.format_exc()[-2000:]},
                       "action": {"kind": "harness"}}
                s.violations.append(row)
                s.append("violations.jsonl", row)
        except Exception:
            pass

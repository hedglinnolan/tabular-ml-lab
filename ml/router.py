"""
Given a project and a record, decide which question comes next.

The differentiator, and the only component in this migration with no existing
implementation to be equivalent to. `TRANSITION_PLAN.md` §02.5 is the feasibility
verdict: the coach is *a pure annotator with one accidental act of control flow*
— it can **order** questions but cannot **gate** them, it never emits a `blocker`
severity, it has no confidence tier of its own, and 100% of its trigger logic
lives in `pages/`. So this is new construction, and the triggers are lifted here
from `pages/02_EDA.py::_auto_generate_insights` (inventoried in
`docs/turbotab/data/explore-triggers.json`).

**Decision B, and it binds everything below.**

> Skip a question only where a `high`-confidence finding makes it moot. Every
> skip is visible and reversible in the transcript. Nothing below `high` ever
> auto-advances.

Auto-advancing an interview *is* pre-selection (`PRODUCT_VISION.md` §07.1), so
the confidence tier that governs pre-selection governs skipping too. The rule is
enforced in one place — :func:`_skip_is_permitted` — and asserted, not trusted:
a skip without a high-confidence finding raises rather than being quietly
emitted.

**What "moot" means, narrowly.** A high-confidence finding can settle a question
of *fact* — "is this column categorical?" — because the engine is certain and
the transcript can state it. It can never settle a question of *choice*: whether
to apply a repair is the user's, however confident the engine is, because
applying without preview is the blind consent `PRODUCT_VISION.md` §04 rules out.
So repairs are always asked; only confirmations of detected fact are skippable.

**Determinism.** `plan()` is a pure function of (findings, detection, record).
Same inputs, same questions, same order — derivable from the record alone, with
no clock, no RNG and no session state.

Headless: no Streamlit, no project object, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Steps of the exploration phase, in order. The Router asks at the earliest step
# that can act on a question, which is what makes deferral meaningful: a
# deferred item has somewhere later to go.
STEPS: Sequence[str] = ("data", "explore")

# Severity order for ranking. The engine's own vocabulary; no new tiers.
_SEVERITY_RANK = {"critical": 0, "blocker": 0, "warning": 1, "caution": 2, "info": 3,
                  "opportunity": 4}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


class RouterError(Exception):
    """The Router was asked to do something the governing rules forbid."""


@dataclass
class Question:
    """One thing the interview puts to the user, or explains why it did not."""

    key: str
    title: str
    why: str
    step: str
    kind: str                       # target | task_type | repair | explore
    triggering_finding: Optional[str] = None
    confidence: Optional[str] = None
    severity: Optional[str] = None
    options: List[str] = field(default_factory=list)

    # asked | skipped | deferred — every one of which is visible in the
    # transcript. There is no fourth state, because a question that is neither
    # asked nor accounted for is the silence this whole design removes.
    status: str = "asked"
    skip_reason: Optional[str] = None
    defer_target: Optional[str] = None
    deferred_from: Optional[str] = None

    @property
    def is_findings_driven(self) -> bool:
        return bool(self.triggering_finding)

    @property
    def is_visible(self) -> bool:
        """A skip or deferral must carry its reason to count as visible."""
        if self.status == "skipped":
            return bool(self.skip_reason)
        if self.status == "deferred":
            return bool(self.defer_target)
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "why": self.why,
            "step": self.step, "kind": self.kind,
            "triggering_finding": self.triggering_finding,
            "confidence": self.confidence, "severity": self.severity,
            "status": self.status, "skip_reason": self.skip_reason,
            "defer_target": self.defer_target, "deferred_from": self.deferred_from,
            "options": list(self.options),
        }


def _skip_is_permitted(confidence: Optional[str], kind: str) -> bool:
    """Decision B, in one place.

    Two conditions, both necessary:

    * the finding is `high` confidence — anything less may not auto-advance,
      because auto-advancing is pre-selection;
    * the question is one of *fact*, not of *choice*. A repair is a choice no
      matter how certain the engine is; skipping it would apply a change the
      user never saw.
    """
    return confidence == "high" and kind in ("task_type",)


# ─────────────────────────────────────────────────────────────────────────────
# The plan
# ─────────────────────────────────────────────────────────────────────────────

def plan(
    findings: Sequence[Dict[str, Any]],
    *,
    target: Optional[str] = None,
    detection: Optional[Dict[str, Any]] = None,
    step: str = "data",
    deferred: Optional[Dict[str, str]] = None,
    answered: Sequence[str] = (),
) -> List[Question]:
    """Every question this step asks, in order, derived from the record.

    `deferred` maps a question key to the step it was deferred *to*; a question
    deferred to this step reappears here, which is the whole promise of deferral
    as a first-class disposition.

    `answered` is the keys already settled — the record, replayed.
    """
    if step not in STEPS:
        raise RouterError(f"Unknown step {step!r}; expected one of {list(STEPS)}.")
    deferred = dict(deferred or {})
    answered = set(answered)
    out: List[Question] = []

    # ── the target, first, because nothing below it can be decided ──────────
    if step == "data" and "choose_target" not in answered:
        out.append(Question(
            key="choose_target", kind="target", step="data",
            title="What are you predicting?",
            why=("Pick the column your paper is about. Everything after this — "
                 "which findings matter, which models are viable, what the test "
                 "set is drawn against — follows from it."),
            options=["<column>"]))

    # ── the task type: a question of FACT, so skippable at high confidence ──
    if target and detection and "confirm_task_type" not in answered:
        conf = detection.get("confidence")
        q = Question(
            key="confirm_task_type", kind="task_type", step="data",
            title=f"Is {target} a {detection.get('detected')} problem?",
            why=" ".join(detection.get("reasons") or []),
            confidence=conf,
            options=["classification", "regression"])
        if _skip_is_permitted(conf, "task_type"):
            q.status = "skipped"
            q.skip_reason = (
                f"The engine reads {target} as {detection.get('detected')} at high "
                f"confidence: {' '.join(detection.get('reasons') or [])} Stated in "
                "the transcript rather than asked — change it there if it is wrong.")
        out.append(q)

    # ── one question per repairable finding, ranked by the engine ──────────
    for f in _rank(findings):
        key = f"repair::{f['id']}"
        if key in answered:
            continue
        if not _is_repairable(f):
            continue

        home = _home_step(f)
        target_step = deferred.get(key, home)
        if target_step != step:
            # Not this step's business yet. Recorded as deferred so it is
            # visible, rather than silently absent.
            if home == step:
                q = _repair_question(f, step)
                q.status = "deferred"
                q.defer_target = target_step
                out.append(q)
            continue

        q = _repair_question(f, step)
        if key in deferred:
            q.deferred_from = "data"
        out.append(q)

    return out


def _is_repairable(f: Dict[str, Any]) -> bool:
    """A finding is a question only when the engine proposes a choice.

    `fix_kind == 'none'` is the engine refusing to guess — a report, not a fork.
    `info` findings are noticings; surfacing each as a question is how an
    interview becomes the wall of prompts it exists to replace.
    """
    return (f.get("severity") in ("critical", "warning")
            and bool(f.get("fix_label"))
            and f.get("fix_kind") not in (None, "none"))


def _home_step(f: Dict[str, Any]) -> str:
    """The earliest step that can act on this finding.

    Structural repairs belong at `data`: they change what the table *is*, and
    `T0-ID-001`'s barrier means they must happen before rows acquire identities.
    """
    return "data"


def _repair_question(f: Dict[str, Any], step: str) -> Question:
    return Question(
        key=f"repair::{f['id']}", kind="repair", step=step,
        title=f["title"],
        why=f.get("why_it_matters") or f.get("detail", ""),
        triggering_finding=f["id"],
        confidence=f.get("confidence"),
        severity=f.get("severity"),
        options=["show me what changes", "remind me later", "dismiss"])


def _rank(findings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The engine's severity, then its confidence, then id. A total order."""
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_RANK.get(f.get("severity"), 99),
                       _CONFIDENCE_RANK.get(f.get("confidence"), 1),
                       str(f.get("id"))))


def next_question(
    findings: Sequence[Dict[str, Any]],
    *,
    target: Optional[str] = None,
    detection: Optional[Dict[str, Any]] = None,
    step: str = "data",
    deferred: Optional[Dict[str, str]] = None,
    answered: Sequence[str] = (),
) -> Optional[Question]:
    """The single next question, or None when the step is done.

    Derivable from the record alone: `answered` and `deferred` are the record
    replayed, and nothing else is consulted.
    """
    for q in plan(findings, target=target, detection=detection, step=step,
                  deferred=deferred, answered=answered):
        if q.status == "asked":
            return q
    return None


def audit(questions: Sequence[Question]) -> None:
    """Assert the governing rules on a plan. Raises rather than reporting.

    Called by the Router's own tests and by the measurement harness, so a
    violation cannot be scored — a run that breaks Decision B has no number, it
    has a failure.
    """
    for q in questions:
        if q.status == "skipped":
            if not _skip_is_permitted(q.confidence, q.kind):
                raise RouterError(
                    f"{q.key} was skipped at confidence {q.confidence!r} and kind "
                    f"{q.kind!r}. Decision B allows a skip only where a high-confidence "
                    "finding makes a question of fact moot.")
            if not q.skip_reason:
                raise RouterError(
                    f"{q.key} was skipped with no reason in the transcript. Every "
                    "skip must be visible and reversible.")
        if q.status == "deferred" and not q.defer_target:
            raise RouterError(
                f"{q.key} was deferred with no target step, so it would never "
                "resurface. Deferral is a disposition, not a discard.")
        if not q.is_visible:
            raise RouterError(f"{q.key} is not visible in the transcript.")

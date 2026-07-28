"""
turbotab.measure — the routing value check, as numbers.

Built **before the Router**, deliberately. `ROADMAP.md` names discovering a weak
Router after eleven step-loops as the single most expensive mistake available in
this plan, and this is the only evidence gate the Router gets — it is the one
component with no existing implementation to be equivalent to. A baseline
measured after the fact is a baseline you unconsciously fit to.

Four metrics, operationalizing `PRODUCT_VISION.md` §04:

``required_decisions``
    What the exploration phase genuinely *needs* decided for this dataset,
    derived from the engine's own findings — not from either door's UI. This is
    the denominator, and it doubles as the register pre-list for the explore
    step.

``questions_asked``
    How many decision points the door actually puts to the user. Classic:
    interactive widgets on the golden path through pages 01–02. Guided: question
    cards presented.

``findings_driven``
    Of those questions, the fraction that cite a triggering finding in the
    record. *"Push the notable, pull the rest"* is the claim; this is the
    number. A question that exists because a pipeline stage exists is not
    findings-driven.

``deferral_closes``
    Of noticings deferred, the fraction that resurface at a step able to act on
    them. Deferral is a first-class disposition only if it comes back. When a
    dataset offers nothing to defer the metric is `None` — not applicable rather
    than perfect or failing (see `VALUE_CHECK_ADJUDICATION.md`).

**Pushed questions versus pull affordances.** *"Push the notable, pull the
rest"* (`PRODUCT_VISION.md` §04) means the interview asks about what it found
and offers everything else quietly alongside. Only the first kind is a question:

* a **pushed** question is put to the user and awaits an answer — the interview
  raised it, so the thresholds bind on it;
* a **pull affordance** is offered and costs nothing to ignore — the
  distribution gallery, the correlation matrix, Table One. It is present, and it
  is *not* a question.

Counting a pull affordance as a question would make the exploration palette read
as a threshold regression at the exact moment the product starts working as
designed. `QuestionRecord.mode` carries the distinction and every metric counts
pushed questions only.

The derived claim is ``irrelevant_questions = questions_asked -
required_decisions`` when positive: questions the dataset did not call for.

Headless. Measuring Classic needs Streamlit, so that collector lives in the test
harness; this module holds the metric definitions and the scoring, and imports
nothing from either door.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = "1.0"


@dataclass
class DecisionRequirement:
    """One thing this dataset genuinely requires a human to settle."""
    key: str
    reason: str
    severity: str = "warning"
    source_finding: Optional[str] = None
    # `high` findings can make a question moot (Decision B); anything lower
    # cannot, so it stays a required decision no matter how confident the app is.
    confidence: Optional[str] = None
    # The engine's own words for the repair. Carried so a door's question can be
    # matched to the requirement it settles without either door's key naming.
    fix_label: Optional[str] = None

    @property
    def can_be_skipped(self) -> bool:
        """Decision B, as a property.

        *Skip only where a high-confidence finding makes the question moot.*
        Nothing below `high` may ever auto-advance, because auto-advancing an
        interview is pre-selection and the governing rule reserves that for
        `high` alone.
        """
        return self.confidence == "high"


@dataclass
class QuestionRecord:
    """One decision point a door actually presented."""
    key: str
    label: str
    door: str                       # classic | guided
    step: str
    triggering_finding: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Which required decision this question settles, if any. The two doors name
    # their widgets differently, so coverage has to be computed through an
    # explicit mapping rather than by comparing key strings — otherwise every
    # door scores zero against every requirement and the metric says nothing.
    covers: Optional[str] = None
    # push | pull. A pull affordance is offered, not asked: ignoring it costs
    # nothing and it never blocks. The thresholds bind on pushed questions only,
    # or the palette reads as a regression the moment it lands.
    mode: str = "push"
    # The constitution clause that requires this question, when one does. See
    # `Measurement.constitutional` for what it is counted as and what it is not.
    clause: Optional[str] = None

    @property
    def is_question(self) -> bool:
        return self.mode == "push"

    @property
    def is_findings_driven(self) -> bool:
        return bool(self.triggering_finding)


@dataclass
class DeferralRecord:
    """One noticing the user deferred, and whether it came back."""
    finding_id: str
    deferred_at: str
    target_step: str
    resurfaced_at: Optional[str] = None

    @property
    def closed(self) -> bool:
        """Closed means it resurfaced at a step that can act on it.

        Resurfacing somewhere else is not closure; it is a reminder in the wrong
        room, which is what the rail dock exists to avoid.
        """
        return self.resurfaced_at is not None and self.resurfaced_at == self.target_step


@dataclass
class Measurement:
    """One door, one dataset, four numbers."""
    door: str
    dataset: str
    n_rows: int
    n_columns: int
    required: List[DecisionRequirement] = field(default_factory=list)
    questions: List[QuestionRecord] = field(default_factory=list)
    deferrals: List[DeferralRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ── the four metrics ────────────────────────────────────────────────────

    @property
    def required_decisions(self) -> int:
        return len(self.required)

    @property
    def questions_asked(self) -> int:
        """Pushed questions actually put to the user.

        A skipped question is not asked, and a pull affordance was never a
        question.
        """
        return sum(1 for q in self.questions if q.is_question and not q.skipped)

    @property
    def pull_affordances(self) -> int:
        """Offered and ignorable. Reported, never scored."""
        return sum(1 for q in self.questions if not q.is_question)

    @property
    def findings_driven(self) -> float:
        asked = [q for q in self.questions if q.is_question and not q.skipped]
        if not asked:
            return 0.0
        return sum(1 for q in asked if q.is_findings_driven) / len(asked)

    @property
    def deferral_closes(self) -> float:
        if not self.deferrals:
            # No deferrals is not a perfect score. It is no evidence, and
            # reporting 1.0 would let a door that cannot defer at all win the
            # metric outright.
            return float("nan")
        return sum(1 for d in self.deferrals if d.closed) / len(self.deferrals)

    @property
    def irrelevant_questions(self) -> int:
        """Questions beyond what the dataset called for.

        Floored at zero: asking *fewer* questions than the data requires is a
        different failure — silence — and is caught by `coverage` below rather
        than flattered by this number.
        """
        return max(0, self.questions_asked - self.required_decisions)

    @property
    def constitutional(self) -> int:
        """Questions a constitution clause requires, with no other origin.

        **The L16 ruling on `GUIDED-018`, and it is a new category rather than a
        new key.** The pre-registration defines an irrelevant question as one
        *absent from the decision inventory and citing no finding*. Both
        conjuncts hold for the grain question, so the recorded numbers are
        correct and they stand: `irrelevant_questions` does not move, and the
        thresholds keep binding on it.

        What was wrong is not the arithmetic but an assumption underneath it —
        that every legitimate question originates from a finding. Clause §02
        introduced a fourth origin: **asked because the app cannot know.** That
        question cites no finding and has no inventory key *by design*, because
        asking it only when a detector fires is the defect the clause exists to
        prevent.

        So the category is named and reported beside the literal count, never
        instead of it. Adding a `grain::` key to `required_decisions` was the
        other option and was refused: it smuggles a new category into an old
        bucket, and eligibility (§04) and missingness routing (§07) would each
        need their own, so the denominator would move every loop.

        **Three conjuncts, all necessary.** A question counts here only if it
        names a clause AND cites no finding AND settles no inventory key. A
        clause-bearing question that also covers a required decision is already
        counted where it belongs; letting it count twice would turn this into a
        laundering mechanism for questions the harness was right about.
        """
        return sum(1 for q in self.questions
                   if q.is_question and not q.skipped
                   and q.clause and not q.triggering_finding and not q.covers)

    @property
    def irrelevant_net(self) -> int:
        """`irrelevant − constitutional`. Reported alongside, never instead.

        The second reading, published so the first one cannot quietly become the
        only one anybody quotes. Nothing binds on this number: the prereg's
        ceilings are applied to `irrelevant_questions`, and a threshold moved
        onto a metric invented after the result is a threshold fitted to it.
        """
        return max(0, self.irrelevant_questions - self.constitutional)

    @property
    def uncovered(self) -> List[str]:
        """Required decisions this door never puts to the user."""
        raised = {q.covers for q in self.questions if q.covers}
        return [r.key for r in self.required if r.key not in raised]

    @property
    def coverage(self) -> float:
        """Fraction of required decisions the door actually raised.

        The guard against winning "fewer questions" by asking nothing. A door
        that skips a required decision has not been efficient; it has been
        silent, and silence about a consequential choice is the failure this
        product exists to remove.
        """
        if not self.required:
            return float("nan")
        raised = {q.covers for q in self.questions if q.covers}
        return sum(1 for r in self.required if r.key in raised) / len(self.required)

    @property
    def covered(self) -> int:
        """How many required decisions the door raised — coverage's numerator."""
        raised = {q.covers for q in self.questions if q.covers}
        return sum(1 for r in self.required if r.key in raised)

    def coverage_ratio(self, measured_at: Optional[str] = None) -> str:
        """Coverage as `k/n @ commit`. A bare ratio is not a result.

        The denominator is derived from the engine's findings, so it rises
        every time the engine learns to see something new — while Classic's
        numerator is structurally frozen, because the import path it renders is
        frozen. A coverage figure quoted without its denominator therefore
        drifts upward on its own, loop after loop, measuring nothing new about
        routing. Carrying `n` and the commit makes two figures comparable or
        visibly not.
        """
        n = len(self.required)
        stamp = f" @ {measured_at}" if measured_at else ""
        return f"{self.covered}/{n}{stamp}" if n else f"n/a{stamp}"

    def to_dict(self, measured_at: Optional[str] = None) -> Dict[str, Any]:
        return {
            "door": self.door, "dataset": self.dataset,
            "n_rows": self.n_rows, "n_columns": self.n_columns,
            "metrics": {
                "required_decisions": self.required_decisions,
                "questions_asked": self.questions_asked,
                "irrelevant_questions": self.irrelevant_questions,
                # Both readings travel together. The first is what the prereg
                # defines and what the thresholds bind on; the second is the
                # constitutional category named at L16. Publishing only one of
                # them is how a reading becomes the reading.
                "constitutional": self.constitutional,
                "irrelevant_net": self.irrelevant_net,
                "findings_driven": _round(self.findings_driven),
                "deferral_closes": _round(self.deferral_closes),
                "coverage": _round(self.coverage),
                # Coverage never travels as a bare ratio: the numerator and the
                # denominator go with it, and so does the commit they were
                # measured at.
                "covered": self.covered,
                "coverage_ratio": self.coverage_ratio(measured_at),
                "measured_at": measured_at,
                # Reported beside the scored metrics, never inside them.
                "pull_affordances": self.pull_affordances,
            },
            "uncovered": self.uncovered,
            "required": [asdict(r) for r in self.required],
            "questions": [asdict(q) for q in self.questions],
            "deferrals": [asdict(d) for d in self.deferrals],
            "notes": self.notes,
        }


def _round(v: float) -> Optional[float]:
    return None if v != v else round(float(v), 4)      # NaN -> null on the wire


# ─────────────────────────────────────────────────────────────────────────────
# The denominator: what a dataset genuinely requires
# ─────────────────────────────────────────────────────────────────────────────

def required_decisions(findings: Sequence[Dict[str, Any]],
                       target_chosen: bool = False) -> List[DecisionRequirement]:
    """Derive the decisions this dataset needs settled, from the engine.

    Deliberately computed from `ml/` output rather than from either door's UI,
    so it is not biased toward whichever door is being measured. Both doors are
    then scored against the same denominator.

    The rule: a finding requires a decision when it is `critical` or `warning`
    and the engine proposes a repair — that is a fork the user has to take.
    `info` findings and findings with `fix_kind == 'none'` are reports, not
    questions; the engine has already said it cannot offer a choice.
    """
    out: List[DecisionRequirement] = []
    if not target_chosen:
        out.append(DecisionRequirement(
            key="choose_target",
            reason="Nothing downstream can be decided before the outcome is named.",
            severity="critical"))

    for f in findings:
        if f.get("severity") not in ("critical", "warning"):
            continue
        if not f.get("fix_label") or f.get("fix_kind") in (None, "none"):
            continue
        out.append(DecisionRequirement(
            key=f"repair::{f['id']}",
            reason=f.get("title", f["id"]),
            severity=f["severity"],
            source_finding=f["id"],
            confidence=f.get("confidence"),
            fix_label=f.get("fix_label")))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Comparing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Comparison:
    """Guided against a committed Classic baseline, on one dataset."""
    dataset: str
    classic: Dict[str, Any]
    guided: Dict[str, Any]

    def verdict(self) -> Dict[str, Any]:
        """Does Guided measurably win where the roadmap says it must?

        Two required wins — *fewer irrelevant questions* and *deferral closes* —
        and one thing that must not regress: coverage. Winning on question count
        by not asking a required question is not a win.
        """
        c, g = self.classic["metrics"], self.guided["metrics"]

        fewer = g["irrelevant_questions"] < c["irrelevant_questions"]
        defer = _gt(g["deferral_closes"], c["deferral_closes"])
        cover = _gte(g["coverage"], c["coverage"])

        return {
            "dataset": self.dataset,
            "fewer_irrelevant_questions": fewer,
            "deferral_closes_improved": defer,
            "coverage_not_regressed": cover,
            "passes": bool(fewer and defer and cover),
            "classic": c, "guided": g,
        }


def _gt(a: Optional[float], b: Optional[float]) -> bool:
    """Greater-than where None means "no evidence", never "wins"."""
    if a is None:
        return False
    if b is None:
        return True          # a door that defers at all beats one that cannot
    return a > b


def _gte(a: Optional[float], b: Optional[float]) -> bool:
    if a is None:
        return b is None
    return True if b is None else a >= b - 1e-9


def write_baseline(path, measurements: Sequence[Measurement],
                   measured_at: Optional[str] = None,
                   prereg: Optional[str] = None) -> None:
    """Write a baseline as data.

    **Not called by the test suite.** A baseline is raw data from a
    pre-registered experiment: measurement and comparison are different acts and
    must not share a code path, or every run silently re-measures the reference
    it is supposed to be judged against. `scripts/remeasure_routing_baseline.py`
    is the only invocation, and it refuses to overwrite.

    `measured_at` records the commit the numbers were taken at, and `prereg`
    the document that banks thresholds against them. Both are provenance about
    the artifact, never part of a measurement.
    """
    import pathlib

    payload: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if measured_at:
        payload["measured_at"] = measured_at
    if prereg:
        payload["prereg"] = prereg
    payload["measurements"] = [m.to_dict(measured_at) for m in measurements]
    pathlib.Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def baseline_provenance(path) -> Dict[str, Any]:
    """The `measured_at` / `prereg` stamps on a baseline file, if any."""
    import pathlib

    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {"measured_at": data.get("measured_at"), "prereg": data.get("prereg")}


def read_baseline(path) -> List[Dict[str, Any]]:
    import pathlib

    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"baseline schema {data.get('schema_version')!r} is not "
            f"{SCHEMA_VERSION!r}; re-measure rather than reinterpreting it")
    return data["measurements"]

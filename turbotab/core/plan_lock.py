"""The inference analysis-plan lock (audit WP16, RO-12; MODELING_SEQUENCE §1 row 12).

Under inference the coefficients and p-values were live while the plan was still being changed:
every change of roles, exclusions, missing values or energy model refitted and re-served them, and
nothing recorded that the analyst had seen them (RO-12). Gelman & Loken (2013, "The garden of
forking paths", abstract): "Researcher degrees of freedom can lead to a multiple comparisons
problem, even in settings where researchers perform only a single analysis on their data. The
problem is there can be a large number of potential comparisons when the details of data analysis
are highly contingent on data". The forks they name (§1.2): "choices of control variables in a
regression, transformations, and data coding and excluding rules".

So the plan is locked before any inference estimate is displayed:

* **When.** The server records ``lock_plan`` when Fit is pressed (SIZING P0.8; ``fit_press``),
  once the questions the estimates rest on are answered (WP17); no estimate stage is served before
  it (a coefficient or an inference table in the fit, a substitution curve, a sensitivity
  analysis's estimate, a calibrated one, the "further adjusted for" model's, Describe's
  usual-intake distribution: :func:`shows_estimates`). A client never posts it. Logs written before
  P0.8 recorded it the first time an estimate was served, which was likewise before that estimate
  was displayed.
* **Withdrawn only while nothing was shown** (calm/FOUNDATION §7). Cancel before any estimate is
  served withdraws the lock, and so does a change to the plan made then: the lock says the plan
  was declared before any estimate was shown, and none was. The server records the withdrawal as a
  revert of the lock, so the record keeps the withdrawn lock with its time and fingerprint, and the
  next press of Fit records a new one (:func:`current_lock`). Once an estimate has been served the
  lock stands: a client never reverts it.
* **What.** The plan is every slot the estimates read, as it stood (:func:`plan_of`): the outcome,
  the exposures and the adjustment set (the roles), the exclusions, the missing-data plan, the
  energy model, the exposure forms, the families, the secondaries, and the data coding (repairs,
  confirmed readings). It is derived from the stage graph, so a slot a later stage adds is covered
  without a list to keep. Its SHA-256 (:func:`digest`) is recorded with it, for external
  registration.
* **After.** Every later decision is marked ``after_estimates`` and its sentence says "After the
  estimates were seen, …" (``decisions.disclose``); the methods text keeps the plan as declared and
  every change after it (``turbotab.core.provenance``).
* **Wording.** The lock says what was declared in the software before estimates were displayed.
  It is never called prespecified or preregistered (MODELING_SEQUENCE §1 row 12): the analyst may
  have seen the data, and nothing was registered outside the software.

Importing this module registers the validators and the completion.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core import decisions
from turbotab.core.decisions import Refusal
# The stages whose artifacts carry estimates of the outcome model: one list with the one WP17
# withholds by, so the "further adjusted for" model (``stages/secondary.py``) is among them.
from turbotab.core.estimand import ESTIMATE_STAGES


@lru_cache(maxsize=1)
def plan_slots() -> tuple[str, ...]:
    """Every slot the estimates read: those of the estimate stages and of everything upstream."""
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    graph = load_graph(GRAPH_FACTORY)
    names: set[str] = set()
    for stage in ESTIMATE_STAGES:
        if stage in graph:
            names |= graph.upstream(stage) | {stage}
    return tuple(sorted({slot for name in names for slot in graph[name].reads}))


TRIAGE = "triage"  # the plan's key for the dispositions of the open noticings


def plan_of(state: Any) -> dict[str, Any]:
    """The plan as ``state`` holds it: each answered slot the estimates read, as JSON, and the
    triage of the open noticings before the lock once it is confirmed (SURFACING_POLICY §3.3: the
    lock's digest covers the dispositions, each with the recommendation it was recorded on)."""
    values = state.model_dump(mode="json")
    plan = {slot: values[slot] for slot in plan_slots() if values.get(slot) is not None}
    held = (values.get("sweeps") or {}).get(decisions.sweep_key("models", "noticings"))
    if held is not None:
        plan[TRIAGE] = held["lines"]
    return plan


def digest(plan: Mapping[str, Any]) -> str:
    """The plan's SHA-256 over its canonical JSON (keys sorted, no spaces, UTF-8)."""
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── the export (ESTIMAND; MODELING_SEQUENCE §1 row 12: "The plan exports with a timestamp and a
# content hash for external registration") ─────────────────────────────────────────────────────

PLAN_FORMAT = "turbotab-analysis-plan/1"
# The words the plan's text never uses: the lock says what was declared in the software before the
# estimates were displayed, never that it was registered anywhere (MODELING_SEQUENCE §1 row 12).
NEVER_SAID = ("prespecified", "pre-specified", "preregistered", "pre-registered")


class PlanExport(BaseModel):
    """The exported analysis plan: the plan, when it was declared, its hash and its text."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    format: str
    status: Literal["locked", "declared"]
    declared_at: str | None  # ISO 8601, UTC: the lock's record, else the last record the plan holds
    through_record: int | None  # the last decision the plan holds (its sequence number)
    plan: dict[str, Any]
    plan_sha256: str  # the plan's own digest, as the lock records it (:func:`digest`)
    sentences: list[str]  # the decisions the plan holds, as the record words them
    after_estimates: list[dict[str, Any]]  # each later decision: seq, kind, at, sentence
    sha256: str  # SHA-256 over the canonical JSON of every field above
    text: str


def canonical(value: Any) -> bytes:
    """Canonical JSON: keys sorted, no spaces, UTF-8 (the bytes :func:`digest` hashes)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _when(at: Any) -> str | None:
    if at is None:
        return None
    from datetime import timezone

    return at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def current_lock(records: Sequence[Any]) -> Any:
    """The lock in force: the newest ``lock_plan`` record not withdrawn (reverted), or None."""
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        withdrawn = decisions.reverted(ordered)
    except Refusal:
        withdrawn = {}
    return next((r for r in reversed(ordered)
                 if r.decision.kind == "lock_plan" and r.id not in withdrawn), None)


def plan_document(records: Sequence[Any]) -> PlanExport:
    """The analysis plan as the decision log holds it, a pure function of the records (no clock is
    read): once locked, the plan the lock recorded and the lock's own time; before, the plan in
    force and the time of the last decision it holds. The hash covers every field but itself and
    the text, which quotes it."""
    from turbotab.core.provenance import in_force

    ordered = sorted(records, key=lambda r: r.seq)
    lock = current_lock(ordered)
    if lock is not None:
        plan = dict(lock.decision.plan or {})
        held = [r for r in ordered if r.seq < lock.seq]
        status, at, through = "locked", lock.at, lock.seq
        # Each later decision made after an estimate was shown (one made while nothing had been
        # shown under the lock is not marked so; ``ProjectService.decide``).
        later = [r for r in ordered if r.seq > lock.seq and r.after_estimates and r.sentence]
    else:
        plan = plan_of(decisions.fold(ordered))
        held, later = ordered, []
        status = "declared"
        at = ordered[-1].at if ordered else None
        through = ordered[-1].seq if ordered else None
    keep = in_force(held)
    sentences = [r.sentence for r in held if r.id in keep and r.sentence]
    content = {
        "format": PLAN_FORMAT, "status": status, "declared_at": _when(at),
        "through_record": through, "plan": plan, "plan_sha256": digest(plan),
        "sentences": sentences,
        "after_estimates": [{"seq": r.seq, "kind": r.decision.kind, "at": _when(r.at),
                             "sentence": r.sentence} for r in later],
    }
    content = json.loads(canonical(content))  # every value as JSON holds it
    sha = hashlib.sha256(canonical(content)).hexdigest()
    return PlanExport(**content, sha256=sha, text=plan_text(content, sha))


def plan_text(content: Mapping[str, Any], sha: str) -> str:
    """What the export says of itself: what was declared in the software, and when; never that it
    was registered or specified before the data were seen (the analyst may have seen them)."""
    at = content.get("declared_at") or "an unrecorded time"
    when = at.replace("T", " at ").replace("Z", " UTC") if isinstance(at, str) else at
    later = len(content.get("after_estimates") or [])
    if content.get("status") == "locked":
        lead = (f"This is the analysis plan as declared in TurboTab before any estimate was "
                f"displayed; it was locked on {when}, before the first estimate was shown.")
        tail = (f" {later:,} decision{'s were' if later != 1 else ' was'} made after the estimates "
                f"were seen, listed with it and marked so in the methods." if later else
                " No decision has been made since the estimates were seen.")
    elif (content.get("plan") or {}).get("purpose") == "prediction":
        # EXPORT: under prediction no plan is locked (the held-out rows are the seal), and scores
        # may already have been shown, so the plan says neither that it was locked nor that no
        # estimate was displayed.
        lead = (f"This is the analysis as declared in TurboTab for prediction, through the decision "
                f"recorded on {when}. Under prediction no plan is locked: the held-out rows are the "
                f"seal, and a family's score is the result only as the fit declares it (the "
                f"held-out score of a family declared final before they were opened, or the "
                f"selection-corrected estimate).")
        tail = ""
    else:
        lead = (f"This is the analysis plan as declared in TurboTab so far, through the decision "
                f"recorded on {when}; no estimate has been displayed yet.")
        tail = ""
    text = (f"{lead} It records what was declared in the software and when; it is not a "
            f"registration with any outside registry, and the analyst may have seen the data. Its "
            f"SHA-256 over canonical JSON (keys sorted, no spaces, UTF-8) is {sha}: the same "
            f"decisions give the same bytes, so the hash can be cited in an external registration."
            f"{tail}")
    if any(word in text.lower() for word in NEVER_SAID):
        raise RuntimeError('the plan text must never call the plan prespecified or preregistered')
    return text


def plan_export(records: Sequence[Any]) -> bytes:
    """The export's bytes: the canonical JSON of :func:`plan_document`, byte-identical on replay."""
    return canonical(plan_document(records).model_dump(mode="json"))


def _estimated(table: Mapping[str, Any]) -> bool:
    """Whether a fitted table carries an estimate: coefficients, an exposure's test, or an
    inference table that is not a refusal. A refusal (an unanswered survey question, a held
    missing-values answer) or a table WP17 withheld (no exposure, effect or adjustment set yet)
    puts nothing in view, so it locks nothing."""
    inference = table.get("inference")
    return bool(table.get("coefficients") or table.get("exposure_tests") or (
        isinstance(inference, Mapping) and not inference.get("refused")))


def shows_estimates(stage: str, artifact: Any) -> bool:
    """Whether a stage's artifact, as served, puts an estimate of the outcome model in view."""
    if not isinstance(artifact, Mapping):
        return False
    if stage == "fit":
        return any(_estimated(m) for m in artifact.get("models") or [] if isinstance(m, Mapping))
    if stage == "substitution":
        return any(any(v is not None for v in m.get("delta") or [])
                   for m in artifact.get("models") or [] if isinstance(m, Mapping))
    if stage in ("sensitivity", "secondary"):
        return any(_estimated(f)
                   for family in artifact.get("families") or [] if isinstance(family, Mapping)
                   for f in family.get("fits") or [] if isinstance(f, Mapping))
    if stage == "calibration":
        return any(e.get("estimate") is not None or e.get("naive") is not None
                   for e in artifact.get("exposures") or [] if isinstance(e, Mapping))
    if stage == "scales":
        return any(isinstance(sc, Mapping) and sc.get("correction") is not None
                   for sc in artifact.get("scales") or [])
    if stage == "effects":  # ESTIMAND: the declared models' exposure rows, or a marginal estimate
        return any(_effects_shown(f) for f in artifact.get("families") or []
                   if isinstance(f, Mapping))
    if stage == "causal":  # the causal lane's estimate (turbotab/core/stages/causal.py)
        return any(e.get("estimate") is not None
                   for e in artifact.get("estimates") or [] if isinstance(e, Mapping))
    if stage == "time_varying":  # its diagnostics relate no exposure to the outcome; estimates lock
        estimates = artifact.get("estimates")
        return isinstance(estimates, Mapping) and bool(estimates.get("rows"))
    if stage == "explain":  # wave 2: an explanation shows what a model learned from the outcome
        return any(f.get("explained") for f in artifact.get("families") or []
                   if isinstance(f, Mapping))
    if stage == "modification":  # FORM: a declared modifier's effects, RERI or ratio of ratios
        return any(q.get("estimate") is not None
                   for m in artifact.get("modifications") or [] if isinstance(m, Mapping)
                   for f in m.get("families") or [] if isinstance(f, Mapping)
                   for q in f.get("effects") or [] if isinstance(q, Mapping))
    if stage == "evaluation":  # wave 2, EXPLORE: under inference, the selection sensitivity's tests
        estimates = artifact.get("estimates")
        return isinstance(estimates, Mapping) and bool(estimates.get("path"))
    if stage == "usual_intake":  # Describe: a usual-intake distribution, its mean or percentiles
        return any(a.get("mean") is not None or a.get("percentiles") or a.get("share") is not None
                   for a in artifact.get("analyses") or [] if isinstance(a, Mapping))
    return False


def _effects_shown(family: Mapping[str, Any]) -> bool:
    rows = [r for s in family.get("sequence") or [] if isinstance(s, Mapping)
            for r in s.get("effects") or [] if isinstance(r, Mapping)]
    contrasts = (family.get("marginal") or {}).get("contrasts") or []
    return (any(r.get("estimate") is not None for r in rows)
            or any(c.get("rd") is not None for c in contrasts if isinstance(c, Mapping)))


def _state(ctx: Any) -> Any:
    if ctx is None:
        return None
    return ctx.get("state") if isinstance(ctx, Mapping) else getattr(ctx, "state", None)


def _records(ctx: Any) -> list[Any] | None:
    found = ctx.get("records") if isinstance(ctx, Mapping) else getattr(ctx, "records", None)
    if callable(found):
        try:
            found = found()
        except Exception:  # noqa: BLE001 - no records: nothing is checked against them
            return None
    return list(found) if found is not None else None


def _locked_once_under_inference(decision: Any, ctx: Any) -> None:
    state = _state(ctx)
    if state is None:
        return
    if state.plan_locked:
        raise Refusal(
            "plan_already_locked",
            "The analysis plan is already locked, before the estimates were displayed; every "
            "change since is marked as made after the estimates were seen.",
        )
    if state.purpose != "inference":
        raise Refusal(
            "not_inference",
            "The analysis-plan lock is for inference, where the model is not chosen from the data "
            "it reports on; under prediction the held-out rows are the seal.",
            exits=[{"label": "Declare the analysis for inference first", "decision": None}],
        )


def _the_lock_records_the_plan(decision: Any, ctx: Any) -> Any:
    """Fill the plan and its digest from the state, never from the client."""
    state = _state(ctx)
    if state is None:
        return decision.model_copy(update={"plan": None, "digest": None})
    plan = plan_of(state)
    return decision.model_copy(update={"plan": plan, "digest": digest(plan)})


def _the_lock_stays(decision: Any, ctx: Any) -> None:
    records = _records(ctx)
    target = next((r for r in records or [] if r.id == decision.decision_id), None)
    if target is not None and target.decision.kind == "lock_plan":
        raise Refusal(
            "plan_stays_locked",
            "The estimates were displayed with this plan, and that cannot be undone: the lock "
            "stays in the record, and later changes stay marked as made after the estimates were "
            "seen.",
        )


decisions.register_validator("lock_plan", _locked_once_under_inference)
decisions.register_completion("lock_plan", _the_lock_records_the_plan)
decisions.register_validator("revert", _the_lock_stays, first=True)

__all__ = ["ESTIMATE_STAGES", "NEVER_SAID", "PlanExport", "canonical", "current_lock", "digest",
           "plan_document", "plan_export", "plan_of", "plan_slots", "plan_text", "shows_estimates"]

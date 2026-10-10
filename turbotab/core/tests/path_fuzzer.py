"""The offline path fuzzer (SURFACING_POLICY §7.1, U16): seeded journeys over ``decisions.fold_onto``
and ``interview.route``, headless, with mocked stage artifacts, and the policy's invariants
(§7.2) checked on every intermediate state.

A journey starts from an empty project on one of two fixtures (the committed NHANES fixture's
columns under the dietary lens; an untargeted metabolomics table) and takes up to ``max_steps``
actions, each drawn by the seed: answer the Router's open question, answer an open declaration,
dispose of a finding, "Confirm all" in a reached stage, change an earlier answer or revert a
record (the two stated reopenings), or press Fit. Compute stages that read what an answer wrote,
and that the graph can compute, are marked running and settle at random, so questions wait on
their cards as they do live; the fit is fresh, running, failed or held whether or not Fit was
pressed, as a fit under the hold's 2 minutes computes before the press.

The invariants held now (the rest need the ledger, the caps or the timing harness):

* **I1** no estimate before the lock: while ``consequences.estimates_unseen``, every estimate stage
  is withheld (``fit_press.served``) and no quest line's text holds a number from one (each
  estimate stage that can compute is mocked with its numbers and handed to the quest log raw);
  with no goal none can compute; Results is never reached before Fit is pressed; the seal never opens before it, though the fit computes before the press
  (a fit under the hold's 2 minutes is fresh, running or failed without one).
* **I2** (its half that needs no materiality) the registry and the quest log agree: every
  question, declaration and finding that fires is a line, and every line (question, declaration,
  finding, reading) is an item that fires; a stage offers "Confirm all" only where its sweep item
  fires, and lists a default wherever it does; each default the log shows as a line is listed
  exactly where it fires. (A result's ``fires`` is held to the engine's own blocking in
  ``test_surfaceable``.)
* **I4** every Decide is answerable or says so: an open line waits for nothing, the Router's open
  question sits in a reached stage, and every question a line reads is answered, stated or not
  applicable.
* **I6** the display-order rule: nothing shown settled rests on an unanswered question without
  saying so (a line answered or set for the person whose item reads one still open waits for it,
  or says why it was asked again); every answer the engine filled is a Confirm or For the record
  line; the outcome alone waits for Who's in (the explore stage requires the split); under
  Estimate, and with no goal, the outcome beside a column waits for the lock.
* **I8** determinism: the same log gives the same quest log, and the preview's fold
  (``fold_onto``) agrees with the log's fold outside the conditional slots it documents.
* **I10** tier is mode-independent: the Decide and Confirm sets are the same in every mode
  (``surfacing.disclosure``), and at most one line is drawn at level 3.
* **I11** progress is monotone except by a stated reopening, after every action, the changes and
  reverts included: a stage's answered count never falls, and a change or a revert never makes a
  complete stage ask again, without a Reopened record in it (a forward step, a first answer or a
  reading settling, can find more for a stage to ask, which §7.2's answered/required would call a
  fall; it is not a reopening; an answered line that left because it no longer applies, or is no
  longer counted, takes its count with it); a line answered before is answered still, or names
  why itself (``reopened_by``, ``changed_since``, a
  Reopened record or the sweep's ``changed`` listing it), or left the log because it no longer
  applies; a reached stage stays reached unless it, or the stage now asking, says why.
"""
from __future__ import annotations

import gzip
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from turbotab.core import decisions, fit_press, quest, surfacing
from turbotab.core.consequences import estimates_unseen
from turbotab.core.decisions import DecisionRecord, ProjectState, Refusal, parse_decision
from turbotab.core.interview import QUESTION_KEYS, InterviewStep, route
from turbotab.core.quest import Facts, QuestLog

T0 = datetime(2026, 10, 9, tzinfo=timezone.utc)
NHANES_CSV = Path(__file__).resolve().parent / "fixtures" / "nhanes.csv.gz"
SETTLED = ("answered", "skipped", "not_applicable")


# ── fixtures ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fixture:
    name: str
    lens: tuple[str, ...]
    columns: tuple[str, ...]
    targets: dict[str, str]  # outcome -> its task as the reading settles it
    identifier: str
    exposures: tuple[str, ...]
    covariates: tuple[str, ...]
    energy: str | None = None
    batch: str | None = None
    findings: tuple[dict[str, Any], ...] = ()


def _nhanes_columns() -> tuple[str, ...]:
    with gzip.open(NHANES_CSV, "rt", encoding="utf-8") as f:
        return tuple(f.readline().strip().split(","))


NHANES = Fixture(
    name="nhanes", lens=("dietary",), columns=_nhanes_columns(),
    targets={"glucose": "regression", "meds_hbp": "binary"}, identifier="SEQN",
    exposures=("sugar", "fat_total"), covariates=("age", "gender", "bmi"), energy="kcal",
    findings=(
        {"id": "pack::dietary::implausible_intake#1", "routes_to": "exclusions",
         "affected_columns": ["kcal"], "repairs": [], "summary": "Implausible reported intakes"},
        {"id": "pack::dietary::energy_adjustment#1", "routes_to": "energy_adjustment",
         "affected_columns": ["kcal"], "repairs": [], "summary": "Energy carries the nutrients"},
        {"id": "unnamed_columns__x", "routes_to": None, "affected_columns": [], "repairs": [],
         "summary": "A column has no name"},
    ))
METABOLOMICS = Fixture(
    name="metabolomics", lens=("metabolomics",),
    columns=("sample_id", "batch", "run_order", "m_001", "m_002", "m_003", "m_004", "age",
             "responder"),
    targets={"responder": "binary"}, identifier="sample_id", exposures=("m_001",),
    covariates=("age",), batch="batch",
    findings=(
        {"id": "pack::metabolomics::left_censored#1", "routes_to": "missing",
         "affected_columns": ["m_002"], "repairs": [], "summary": "Values below detection"},
        {"id": "pack::metabolomics::acquisition_design#1", "routes_to": "roles",
         "affected_columns": ["batch"], "repairs": [], "summary": "How samples were run"},
    ))
FIXTURES = (NHANES, METABOLOMICS)


# ── what each question and declaration can be answered with ──────────────────


def _exposure(state: Any, fx: Fixture) -> str:
    roles = state.roles or {}
    return next((c for c, r in roles.items() if r == "exposure"), fx.exposures[0])


def _covariates(state: Any, fx: Fixture) -> list[str]:
    roles = state.roles or {}
    return [c for c, r in roles.items() if r == "covariate"] or list(fx.covariates)


def question_options(key: str, state: Any, fx: Fixture) -> list[dict[str, Any]]:
    """Answers to the Router question ``key`` on this fixture (empty: the fuzzer cannot answer it,
    and the journey goes on with what else it can do)."""
    target = state.target
    task = fx.targets.get(target or "", "regression")
    exposure = _exposure(state, fx)
    measure = "mean_difference" if task == "regression" else "odds_ratio"
    yes = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
    roles = {fx.identifier: "identifier", **{e: "exposure" for e in fx.exposures[:1]},
             **{c: "covariate" for c in fx.covariates}}
    if fx.energy:
        roles[fx.energy] = "energy"
    other = {**roles, **{e: "covariate" for e in fx.exposures[1:2]}}
    holdout = 0.0 if state.purpose == "inference" else 0.2
    return {
        "lens": [{"kind": "set_lens", "lenses": list(fx.lens)}],
        "orientation": [{"kind": "set_orientation", "orientation": "sample_major"}],
        "target": [{"kind": "set_target", "column": t} for t in fx.targets],
        "event": [{"kind": "set_event", "column": target, "level": "1"}] if target else [],
        "task": [{"kind": "set_task", "column": target, "task": task}] if target else [],
        "follow_up": [{"kind": "set_censoring", "column": target}] if target else [],
        "design": [{"kind": "set_design", "design": "observational"}],
        "purpose": [{"kind": "set_purpose", "purpose": p} for p in ("inference", "prediction")],
        "grain": [{"kind": "set_grain", "grain": "one_row_per_unit"},
                  {"kind": "set_grain", "grain": "repeated", "id_column": fx.identifier}],
        "repeat_kind": [{"kind": "set_repeat_kind", "repeat_kind": "repeats"}],
        "unit": [{"kind": "set_unit", "unit": u} for u in ("unit", "row")],
        "aggregation": [{"kind": "set_aggregation", "method": "mean"}],
        "temporal": [{"kind": "set_temporal", "temporal": False}],
        "roles": [{"kind": "set_roles", "roles": roles}, {"kind": "set_roles", "roles": other}],
        "clusters": [{"kind": "set_clusters", "column": None}],
        "survey": [{"kind": "set_survey", "estimand": "sample"}],
        "exclusions": [{"kind": "set_exclusions", "rules": []},
                       {"kind": "set_exclusions", "rules": [
                           {"column": fx.covariates[0], "low": 18, "high": 80,
                            "reason": "adults"}]}],
        "missing": [{"kind": "set_missing", "strategy": s} for s in ("complete_case", "impute")],
        "split": [{"kind": "set_split", "holdout": holdout, "seed": 0}],
        "estimand": [{"kind": "set_estimand", "exposure": exposure, "measure": measure}],
        "adjustment": [{"kind": "set_adjustment", "exposure": exposure,
                        "answers": {c: yes for c in _covariates(state, fx)}}],
        "energy_adjustment": [{"kind": "set_energy_adjustment", "method": "none"}],
        "form": [{"kind": "set_forms", "forms": {exposure: {"form": "linear"}}}],
        "modification": [],
        "causal": [{"kind": "set_causal", "exposure": exposure, "method": "none"}],
        "time_varying": [],
        "models": [{"kind": "select_models", "models": ["linear"]},
                   {"kind": "select_models", "models": ["linear", "elastic_net"]}],
        "substitution": [],
        "open_seal": [{"kind": "open_seal"}],
    }.get(key, [])


def declaration_options(kind: str, state: Any, fx: Fixture) -> list[dict[str, Any]]:
    exposure = _exposure(state, fx)
    return {
        "set_intended_use": [{"kind": kind, "use": "risk_estimation"}],
        "set_batch": [{"kind": kind, "column": fx.batch, "method": m}
                      for m in ("reference_combat", "none") if fx.batch],
        "set_selection": [{"kind": kind, "method": "none"}],
        "set_model_sequence": [{"kind": kind, "exposure": exposure,
                                "model_1": list(fx.covariates[:1])}],
        "set_sensitivity": [{"kind": kind, "analyses": [{"label": "Every row", "rules": []}]}],
        "set_measurement_error": [{"kind": kind, "method": "none"}],
        "set_multiplicity": [{"kind": kind, "method": "bh"}],
        "set_levers": [{"kind": kind}],
        "set_updating": [{"kind": kind, "method": "none"}],
        "set_explain": [{"kind": kind, "reseeds": 0}],
        "set_validation": [{"kind": kind, "folds": 4}],
    }.get(kind, [])


# ── mocked stage artifacts and statuses ──────────────────────────────────────


def artifacts_for(state: Any, fx: Fixture, records: list[DecisionRecord], steps_hint: Any = None,
                  *, confidence: str = "high") -> dict[str, Any]:
    from turbotab.core.repairs import annotate

    out: dict[str, Any] = {}
    if state.target is not None:
        out["target_info"] = {"column": state.target, "task": fx.targets.get(state.target),
                              "confidence": confidence,
                              "reason": "The outcome's values settle its kind."}
    if any(lens in ("metabolomics", "genomics") for lens in state.lens or ()):
        out["oriented"] = {"reading": {"reading": "sample_major", "confidence": "medium"}}
    if state.purpose == "inference":
        out["forms"] = {"purpose": "inference", "ready": True, "needs": [], "waiting": []}
    if state.lens is not None:
        out["findings"] = annotate({"findings": [dict(f) for f in fx.findings]}, state, records)
    return out


# How the fit stands, drawn per state: a fit expected to take under about 2 minutes computes before
# Fit is pressed (``fit_press.holds``), so the fit is fresh, running or failed whether or not it was
# pressed; only a held fit is idle until the press.
FIT_STATUSES = (("fresh", 5.0), ("running", 2.0), ("error", 1.0), ("idle", 2.0))


# What an estimate stage's artifact holds, raw (as the server hands the quest log the usual-intake
# stage's newest artifact, served or not): numbers no quest line may repeat before the lock (I1).
ESTIMATE_NUMBERS = ("7.3131", "6.1717", "8.4545", "0.0137")


def estimates_for(state: Any) -> dict[str, Any]:
    """Each estimate stage the graph can compute here, with its estimate, interval and p-value."""
    from turbotab.core.estimand import ESTIMATE_STAGES

    row = {"term": "exposure", "estimate": 7.3131, "ci_low": 6.1717, "ci_high": 8.4545,
           "p": 0.0137, "sentence": "The estimate is 7.3131 (6.1717 to 8.4545), p = 0.0137."}
    computable = surfacing.computable(state)
    return {name: {"estimates": [dict(row)], "summary": row["sentence"]}
            for name in ESTIMATE_STAGES if name in computable}


def stages_for(running: Iterable[str], fit: str) -> dict[str, dict[str, Any]]:
    out = {name: {"status": "running"} for name in running}
    out["fit"] = {"status": fit}
    return out


def fit_status(rng: random.Random, state: Any, pressed: bool) -> str:
    """The fit's status on this state: it computes once the graph can compute it, held or not."""
    if "fit" not in surfacing.computable(state):
        return "blocked"
    statuses = [s for s, _w in FIT_STATUSES if not (pressed and s == "idle")]
    weights = [w for s, w in FIT_STATUSES if not (pressed and s == "idle")]
    return rng.choices(statuses, weights=weights)[0]


def answered_through(fx: Fixture, purpose: str, stop: str = "open_seal"
                     ) -> tuple[ProjectState, dict[str, Any], list[DecisionRecord]]:
    """The log answering every Router question before ``stop`` by its first option on this
    fixture (the goal ``purpose``), Fit pressed and the fit fresh along the way: its state, the
    artifacts mocked for it and its records."""
    state, records = ProjectState(), []
    for _ in range(3 * len(QUESTION_KEYS)):
        artifacts = artifacts_for(state, fx, records)
        steps = route(state, {"fit": {"status": "fresh"}}, artifacts, records, pressed=True)
        first = next((s for s in steps if s.status == "open"), None)
        if first is None or first.key == stop:
            break
        options = ([{"kind": "set_purpose", "purpose": purpose}] if first.key == "purpose"
                   else question_options(first.key, state, fx))
        if not options:
            break
        records.append(_record(records, _validate(options[0], state, fx)))
        state = decisions.fold(records)
    return state, artifacts_for(state, fx, records), records


# ── a journey ────────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    fixture: Fixture
    index: int
    action: str
    reopening: bool
    kind: str | None
    state: ProjectState
    records: list[DecisionRecord]
    steps: list[InterviewStep]
    stages: dict[str, Any]
    artifacts: dict[str, Any]
    pressed: bool
    pressed_ever: bool
    log: QuestLog
    onto: ProjectState | None = None  # the preview's fold, beside the log's
    shown_at: dict[str, Any] = field(default_factory=dict)  # results shown for earlier answers


@dataclass
class Journey:
    seed: int
    fixture: str
    snapshots: list[Snapshot] = field(default_factory=list)
    refused: int = 0


def quest_of(state: Any, records: list[DecisionRecord], steps: list[InterviewStep],
             stages: dict[str, Any], artifacts: dict[str, Any], fx: Fixture, pressed: bool,
             pressed_ever: bool, shown_at: dict[str, datetime | None] | None = None) -> QuestLog:
    fit = fit_press.fit_lock(state, records, pressed=pressed, held=False, estimate=None,
                             opened=pressed_ever)
    return quest.quest_log(state, records, steps, stages, findings=artifacts.get("findings"),
                           columns=fx.columns, artifacts={**estimates_for(state), **artifacts},
                           fit=fit, shown_at=shown_at or {})


def _record(records: list[DecisionRecord], decision: Any) -> DecisionRecord:
    seq = len(records) + 1
    return DecisionRecord(id=f"r{seq}", seq=seq, at=T0 + timedelta(minutes=seq), decision=decision)


def _validate(payload: dict[str, Any], state: Any, fx: Fixture) -> Any | None:
    try:
        return decisions.validate(payload, {"columns": list(fx.columns), "target": state.target})
    except (Refusal, ValidationError, ValueError, KeyError, TypeError):
        return None


def run_journey(seed: int, max_steps: int = 40) -> Journey:
    rng = random.Random(seed)
    fx = FIXTURES[seed % len(FIXTURES)]
    confidence = rng.choice(("high", "medium"))
    journey = Journey(seed=seed, fixture=fx.name)
    state = ProjectState()
    records: list[DecisionRecord] = []
    running: set[str] = set()
    pressed_for: str | None = None
    pressed_ever = False
    press_at: datetime | None = None  # when Fit last served the estimates

    def snap(index: int, action: str, reopening: bool, kind: str | None,
             onto: ProjectState | None = None) -> Snapshot:
        pressed = pressed_for is not None and pressed_for == state.target
        fit = fit_status(rng, state, pressed)
        stages = stages_for(running, fit)
        artifacts = artifacts_for(state, fx, records, confidence=confidence)
        steps = route(state, stages, artifacts, records, pressed=pressed)
        # As the server reads it: the fit served after a press, not fresh for the answers now,
        # is a result shown for earlier ones.
        shown_at = ({"fit": press_at} if press_at is not None and not (pressed and fit == "fresh")
                    else {})
        log = quest_of(state, records, steps, stages, artifacts, fx, pressed, pressed_ever,
                       shown_at)
        return Snapshot(fx, index, action, reopening, kind, state, list(records), steps, stages,
                        artifacts, pressed, pressed_ever, log, onto, shown_at)

    journey.snapshots.append(snap(0, "start", False, None))
    for index in range(1, max_steps + 1):
        cur = journey.snapshots[-1]
        running = {s for s in sorted(running) if rng.random() < 0.35}
        choices = _choices(cur, fx)
        if running:  # waiting for a card to compute is an action too
            choices.append((3.0, "wait", None, False))
        if not choices:
            if running:
                running = set()
                journey.snapshots.append(snap(index, "settle", False, None))
                continue
            break
        weights = [w for w, *_ in choices]
        _w, action, payload, reopening = rng.choices(choices, weights=weights)[0]
        onto = None
        if action == "wait":
            journey.snapshots.append(snap(index, "wait", False, None))
            continue
        if action == "fit":
            if state.purpose == "inference" and not state.plan_locked:
                decision = parse_decision({"kind": "lock_plan", "seen_target": state.target})
                records.append(_record(records, decision))
                state = decisions.fold(records)
            pressed_for, pressed_ever = state.target, True
            press_at = T0 + timedelta(minutes=len(records), seconds=30)
            kind = "fit"
        elif action == "revert":
            decision = parse_decision({"kind": "revert", "decision_id": payload})
            records.append(_record(records, decision))
            state = decisions.fold(records)
            kind = "revert"
        else:
            decision = _validate(payload, state, fx)
            if decision is None:
                journey.refused += 1
                continue
            records.append(_record(records, decision))
            onto = decisions.fold_onto(state, decision)
            state = decisions.fold(records)
            kind = decision.kind
            written = quest.written_slots(decision)
            # the engine computes only what its requires allow (``graph.unmet_requires``)
            computable = surfacing.computable(state)
            running |= {name for name in quest.COMPUTE
                        if name != "fit" and written & quest.stage_reads(name)
                        and name in computable and rng.random() < 0.5}
        journey.snapshots.append(snap(index, action, reopening, kind, onto))
    return journey


def _choices(cur: Snapshot, fx: Fixture) -> list[tuple[float, str, Any, bool]]:
    """Each action the journey can take now: (weight, action, payload, a stated reopening)."""
    out: list[tuple[float, str, Any, bool]] = []
    state, steps, log = cur.state, cur.steps, cur.log
    first = next((s for s in steps if s.status in ("open", "waiting")), None)
    if first is not None and first.status == "open":
        for payload in question_options(first.key, state, fx):
            out.append((10.0, "answer", payload, False))
    for stage in log.stages:
        for line in stage.lines:
            if line.source == "declaration" and line.status in ("open", "set_for_you"):
                for payload in declaration_options(line.key, state, fx):
                    out.append((3.0, "declare", payload, False))
            if line.source == "finding" and line.status == "open" and stage.reached:
                out.append((1.0, "dispose", {"kind": "dismiss_finding", "finding_id": line.key},
                            False))
        sweep = stage.sweep
        if stage.reached and sweep is not None and not sweep.answered:
            from turbotab.core.sweep import sweep_lines

            lines = [l.model_dump() for l in sweep_lines(stage)]
            out.append((2.0, "sweep", {"kind": "confirm_sweep", "stage": stage.key, "lines": lines},
                        False))
    changes = [payload for step in steps if step.status == "answered"
               for payload in question_options(step.key, state, fx)]
    for payload in changes:  # a change, or a revert, about one action in twenty
        out.append((0.5 / len(changes), "change", payload, True))
    live = [r for r in cur.records if r.decision.kind not in ("revert", "lock_plan")]
    if live:
        out.append((0.25, "revert", live[-1].id, True))
    models = next(s for s in log.stages if s.key == "models")
    if (state.purpose is not None and state.models and models.reached
            and not cur.pressed and models.progress is not None and models.progress.complete):
        out.append((4.0, "fit", None, False))
    return out


# ── the invariants ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Violation:
    invariant: str
    seed: int
    index: int
    message: str

    def __str__(self) -> str:
        return f"{self.invariant} (seed {self.seed}, step {self.index}): {self.message}"


def _lines(log: QuestLog) -> Iterable[tuple[Any, Any]]:
    for stage in log.stages:
        for line in stage.lines:
            yield stage, line


def _status(steps: list[InterviewStep]) -> dict[str, str]:
    return {s.key: s.status for s in steps}


def i1_no_estimate_before_the_lock(snap: Snapshot) -> list[str]:
    from turbotab.core.estimand import ESTIMATE_STAGES

    out = []
    unseen = estimates_unseen(snap.state, snap.pressed)
    if unseen and fit_press.serving_gate(snap.state, snap.pressed) is None:
        out.append("estimates are unseen but the serving gate lets one through")
    if unseen:
        text = snap.log.model_dump_json(exclude={"fit"})
        leaked = [n for n in ESTIMATE_NUMBERS if n in text]
        if leaked:
            out.append(f"the quest log repeats estimate numbers {leaked} before the lock")
        for stage in ESTIMATE_STAGES:
            artifact = ({"models": [{"coefficients": [1.0], "inference": {"p": 0.01}}]}
                        if stage == "fit" else {"estimate": 1.0})
            served = fit_press.served(stage, artifact, snap.state, pressed=snap.pressed)
            if "withheld" not in served:
                out.append(f"{stage} is served an estimate before the lock")
    if snap.state.purpose is None:
        early = sorted(set(ESTIMATE_STAGES) & surfacing.computable(snap.state))
        if early:
            out.append(f"with no goal, {early} can compute")
    results = next(s for s in snap.log.stages if s.key == "results")
    if results.reached and not snap.pressed_ever:
        out.append("Results is reached before Fit was ever pressed")
    # The seal sits in Results: under Predict it opens on the press for this outcome, under
    # Estimate on the lock a press recorded, whatever the fit's own status (it computes before
    # the press).
    status = _status(snap.steps)
    fitted = (snap.pressed if snap.state.purpose == "prediction"
              else bool(snap.state.plan_locked) and snap.pressed_ever)
    if status.get("open_seal") == "open" and not fitted:
        out.append(f"open_seal opens before Fit (fit {snap.stages['fit']['status']})")
    return out


def facts_of(snap: Snapshot) -> Facts:
    return Facts(columns=snap.fixture.columns, artifacts=snap.artifacts)


def i2_registry_agrees_with_the_quest_log(snap: Snapshot) -> list[str]:
    from turbotab.core.stages.finding_words import family

    out = []
    facts = facts_of(snap)
    reg = surfacing.registry()
    listed = {(line.source, line.key) for _stage, line in _lines(snap.log)}
    status = _status(snap.steps)
    for key in QUESTION_KEYS:
        fires = reg[f"question:{key}"].fires(snap.state, facts)
        if fires != (status[key] != "not_applicable"):
            out.append(f"question {key} fires={fires} but the Router says {status[key]}")
        if fires and ("question", key) not in listed:
            out.append(f"question {key} fires but is not listed")
    for decl in quest.DECLARATIONS:
        if decl.only_recorded:
            continue
        fires = reg[f"decision:{decl.kind}"].fires(snap.state, facts)
        listed_now = ("declaration", decl.kind) in listed
        if fires != listed_now:
            out.append(f"declaration {decl.kind} fires={fires} but listed={listed_now}")
    findings = surfacing._listed(snap.artifacts.get("findings"))
    for f in findings:
        if ("finding", str(f.get("id"))) not in listed:
            out.append(f"finding {f.get('id')} fires but is not listed")
        fam = reg.get(f"noticing:{family(str(f.get('id')))}")
        if fam is None or not fam.fires(snap.state, facts):
            out.append(f"finding {f.get('id')} has no registry item that fires")
    # Every line is an item that fires (the other direction, for each source the log lists).
    findings_listed = {str(f.get("id")) for f in findings}
    for stage, line in _lines(snap.log):
        if line.source == "question" and line.key in status and status[line.key] == "not_applicable":
            out.append(f"{line.id} is listed but its question does not fire")
        if line.source == "finding" and line.key not in findings_listed:
            out.append(f"{line.id} is listed but no finding of the artifact is it")
        if line.source == "reading" and not reg["default:read-from-data"].fires(snap.state, facts):
            out.append(f"{line.id} is listed but no reading fires")
    # "Confirm all": a stage's sweep is offered only where its item fires, and where it fires the
    # stage lists the default it holds (a Confirm line, or For the record once weighed).
    for stage in snap.log.stages:
        item = reg.get(f"decision:confirm_sweep:{stage.key}")
        fires = item is not None and item.fires(snap.state, facts)
        if stage.sweep is not None and not fires:
            out.append(f"{stage.key} offers Confirm all but its sweep item does not fire")
        stated = [l for l in stage.lines if l.label in ("Confirm", "For the record")]
        if fires and not stated:
            out.append(f"{stage.key}'s sweep item fires but the stage states nothing")
    # The defaults the log shows as lines: listed exactly where they fire.
    for key, (source, line_key) in surfacing.DEFAULT_LINES.items():
        item = reg[key]
        fires = item.fires(snap.state, facts)
        shown = [l for st, l in _lines(snap.log) if st.key == item.stage and l.source == source
                 and (line_key is None or l.key == line_key)
                 and l.label in ("Confirm", "For the record")]  # stated, or confirmed as stated
        if fires != bool(shown):
            out.append(f"{key} fires={fires} but the log shows it {len(shown)} times")
    return out


def i4_every_decide_is_answerable(snap: Snapshot) -> list[str]:
    out = []
    status = _status(snap.steps)
    reg = surfacing.registry()
    for stage, line in _lines(snap.log):
        if line.status != "open":
            continue
        if line.source == "question" and not stage.reached:
            # The Router's open question is the Decide asked now; a later stage's declarations
            # and noticings are listed open ahead of it, as what that stage will ask.
            out.append(f"{line.id} is the open question in {stage.key}, which is not reached")
        if line.waiting_for or line.computing:
            out.append(f"{line.id} is open and waits for {[w.key for w in line.waiting_for]} "
                       f"{line.computing}")
        item = reg.get(f"question:{line.key}") if line.source == "question" else (
            reg.get(f"decision:{line.key}") if line.source == "declaration" else None)
        if item is None:
            continue
        unsettled = [k for k in item.reads if k in status and status[k] not in SETTLED]
        if unsettled:
            out.append(f"{line.id} is open but reads unanswered {unsettled}")
    opened = [s.key for s in snap.steps if s.status == "open"]
    if len(opened) > 1:
        out.append(f"the Router opens {opened} at once")
    return out


def i6_display_order(snap: Snapshot) -> list[str]:
    out = []
    # Nothing shown depends on an unanswered decision without saying so: a line shown settled
    # (answered, or set for the person) whose item reads a question still open or waiting waits,
    # or says why it was asked again.
    reg = surfacing.registry()
    # a stated line rests on what its default reads (the form stated under Predict reads the goal
    # alone), not on what the question reads when it is asked
    stated_by = {line: reg[key] for key, line in surfacing.DEFAULT_LINES.items()}
    # the questions still asking, as the quest log shows them (an answer that holds while its own
    # reading recomputes stands answered meanwhile)
    asking = {l.key for _s, l in _lines(snap.log)
              if l.source == "question" and l.status in ("open", "waiting")}
    for _stage, line in _lines(snap.log):
        if line.status not in ("answered", "set_for_you"):
            continue
        item = reg.get(f"question:{line.key}") if line.source == "question" else (
            reg.get(f"decision:{line.key}") if line.source == "declaration" else None)
        if line.label != "Decide" and (line.source, line.key) in stated_by:
            item = stated_by[(line.source, line.key)]  # stated, or confirmed as stated
        if item is None:
            continue
        unsettled = [k for k in item.reads if k in asking]
        if unsettled and not (line.waiting_for or line.reopened_by or line.changed_since):
            out.append(f"{line.id} is shown {line.status} but rests on unanswered {unsettled} "
                       f"without saying so")
    listed = {line.key: line for _stage, line in _lines(snap.log) if line.source == "question"}
    for step in snap.steps:
        if step.status == "skipped":
            line = listed.get(step.key)
            if line is None or line.label not in ("Confirm", "For the record"):
                out.append(f"{step.key} was filled by the engine but is not a Confirm or For the "
                           f"record line ({line.label if line else 'not listed'})")
    if "explore" in surfacing.computable(snap.state) and snap.state.split is None:
        out.append("the outcome is shown before Who's in is answered")
    explore = {"findings": [{"kind": "outcome_relationship", "points": [{"x": 1.0, "y": 2.0}],
                             "record": {"kind": "view_outcome"}}]}
    served = fit_press.relationships_served(explore, snap.state, snap.pressed)
    shown = bool(served["findings"][0]["points"])
    if snap.state.purpose != "prediction" and not snap.state.plan_locked and shown:
        out.append("the outcome is shown beside a column before the lock")
    return out


def i8_determinism(snap: Snapshot) -> list[str]:
    out = []
    if snap.index % 4 == 0:  # every fourth state: recomputing the quest log is the costly half
        again = quest_of(snap.state, snap.records, snap.steps, snap.stages, snap.artifacts,
                         snap.fixture, snap.pressed, snap.pressed_ever, snap.shown_at)
        if again.model_dump() != snap.log.model_dump():
            out.append("the same log gave another quest log")
    if snap.onto is not None:
        conditional = {decisions.SLOTS[k] for k in decisions._HOLDS}
        for name in ProjectState.model_fields:
            if name in conditional:
                continue
            if getattr(snap.onto, name) != getattr(snap.state, name):
                out.append(f"fold_onto and fold disagree on {name}")
    return out


def i10_mode_independence(snap: Snapshot) -> list[str]:
    out = []
    remembered = tuple(line.key for _s, line in _lines(snap.log))[::2]
    views = {mode: surfacing.disclosure(snap.log, mode, remembered) for mode in surfacing.MODES}
    tiers = {mode: {(s.stage, s.id, s.key, s.label) for s in shown
                    if s.label in ("Decide", "Confirm")} for mode, shown in views.items()}
    base = tiers[surfacing.MODES[0]]
    for mode, tier in tiers.items():
        if tier != base:
            out.append(f"the Decide and Confirm sets differ under {mode}")
        if sum(s.level == 3 for s in views[mode]) > 1:
            out.append(f"more than one line is drawn at level 3 under {mode}")
    return out


def _fires_now(snap: Snapshot, line: Any) -> bool:
    """Whether the item behind a line still fires on this state (a line that left the log is
    accounted for only when it no longer applies)."""
    facts = facts_of(snap)
    reg = surfacing.registry()
    if line.source == "question":
        status = _status(snap.steps).get(line.key)
        return status is not None and status != "not_applicable"
    if line.source == "declaration":
        item = reg.get(f"decision:{line.key}")
        return item is not None and item.fires(snap.state, facts)
    if line.source == "finding":
        return any(str(f.get("id")) == line.key
                   for f in surfacing._listed(snap.artifacts.get("findings")))
    return True


def i11_monotone_progress(prev: Snapshot, snap: Snapshot) -> list[str]:
    """§7.2 I11: a stage's answered/required never falls without a Reopened record naming the
    cause; after a change or a revert as after any other action. Each line answered before is
    answered still, or says itself why (``reopened_by``, ``changed_since``, its stage's sweep
    listing it as changed), or no longer applies; a reached stage stays reached unless it, or the
    stage now asking, holds the Reopened record that says why."""
    out = []
    after = f"after {snap.action} {snap.kind}"
    # a line is its stage, source and key (its card can change with the goal: the split under
    # Estimate is For the record)
    now = {(stage.key, line.source, line.key): (stage, line) for stage, line in _lines(snap.log)}
    for stage, line in _lines(prev.log):
        if line.status != "answered":
            continue
        found = now.get((stage.key, line.source, line.key))
        if found is None:
            if _fires_now(snap, line):
                out.append(f"{line.id} ({stage.key}) was answered and left the log while it "
                           f"still applies ({after})")
            continue
        new_stage, new_line = found
        if new_line.status == "answered":
            continue
        said = (new_line.reopened_by is not None or new_line.changed_since is not None
                or any(new_line.id in r.questions for r in new_stage.reopened)
                or (new_stage.sweep is not None and new_line.key in new_stage.sweep.changed))
        if not said:
            out.append(f"{line.id} ({stage.key}) was answered and is {new_line.status} now, "
                       f"with nothing saying why ({after})")
    stages_now = {s.key: s for s in snap.log.stages}
    for old in prev.log.stages:
        new = stages_now[old.key]
        if old.progress is None or new.progress is None:
            continue
        # Work done is never undone silently: the answered count falls only with a Reopened
        # record (or lines that left because they no longer apply, which take their count with
        # them), and a complete stage asks again only with one. (A stage still asking can count
        # more as its readings settle: a question waiting on its reading is not counted yet.)
        # an answered Decide that left (it no longer applies) or is no longer counted (the split
        # is For the record under Estimate) takes its count with it
        counted = {(l.source, l.key) for l in new.lines if l.label == "Decide" and l.counted}
        gone = sum(1 for l in old.lines if l.label == "Decide" and l.counted
                   and l.status == "answered" and (l.source, l.key) not in counted)
        lost = (old.progress.answered - new.progress.answered) - gone > 0
        # A forward step (a first answer, a reading settling) can find more for a stage to ask;
        # a change or a revert that does is a reopening, and says so.
        before = {(l.source, l.key): l for l in old.lines}
        asks = [l for l in new.lines if l.counted and l.status != "answered"
                and ((l.source, l.key) not in before
                     or before[(l.source, l.key)].status == "answered")]
        reopened = (snap.reopening and old.progress.complete and not new.progress.complete
                    and bool(asks))
        if (lost or reopened) and not new.reopened and not (
                new.sweep is not None and new.sweep.changed):
            out.append(f"{old.key}'s progress fell from {old.progress.answered}/"
                       f"{old.progress.required} to {new.progress.answered}/"
                       f"{new.progress.required} with no Reopened record ({after})")
    asking = next((s for s in snap.log.stages
                   if any(l.source == "question" and l.status in ("open", "waiting")
                          and (l.reopened_by is not None or not l.waiting_for)
                          for l in s.lines)), None)
    for old in prev.log.stages:
        if not old.reached or stages_now[old.key].reached:
            continue
        why = bool(stages_now[old.key].reopened) or (
            asking is not None and (bool(asking.reopened) or any(
                l.reopened_by is not None for l in asking.lines)))
        if not why:
            out.append(f"{old.key} stopped being reached with nothing saying why ({after})")
    return out


ONE_STATE: tuple[tuple[str, Callable[[Snapshot], list[str]]], ...] = (
    ("I1", i1_no_estimate_before_the_lock),
    ("I2", i2_registry_agrees_with_the_quest_log),
    ("I4", i4_every_decide_is_answerable),
    ("I6", i6_display_order),
    ("I8", i8_determinism),
    ("I10", i10_mode_independence),
)
def check(journey: Journey) -> list[Violation]:
    out: list[Violation] = []
    prev = None
    for snap in journey.snapshots:
        for name, fn in ONE_STATE:
            out += [Violation(name, journey.seed, snap.index, m) for m in fn(snap)]
        if prev is not None:
            out += [Violation("I11", journey.seed, snap.index, m)
                    for m in i11_monotone_progress(prev, snap)]
        prev = snap
    return out


@dataclass
class Report:
    journeys: int = 0
    states: int = 0
    refused: int = 0
    fits: int = 0
    reopenings: int = 0
    decide_load: int = 0  # the most counted Decide lines one stage held in one state (§4.2)
    locked: int = 0  # states with the plan locked
    reached: dict[str, int] = field(default_factory=dict)  # states that reached each stage
    goals: dict[str, int] = field(default_factory=dict)  # journeys by their last goal
    violations: list[Violation] = field(default_factory=list)

    def summary(self) -> str:
        by: dict[str, int] = {}
        for v in self.violations:
            by[v.invariant] = by.get(v.invariant, 0) + 1
        return (f"{self.journeys} journeys ({self.goals}), {self.states} states, {self.fits} "
                f"fits, {self.locked} locked states, {self.reopenings} reopenings, "
                f"{self.refused} refused answers, Decide load {self.decide_load}, reached "
                f"{self.reached}; violations {by or 'none'}")


def fuzz(n: int, seed: int = 0, max_steps: int = 40) -> Report:
    report = Report()
    for s in range(seed, seed + n):
        journey = run_journey(s, max_steps)
        report.journeys += 1
        report.states += len(journey.snapshots)
        report.refused += journey.refused
        report.fits += sum(snap.action == "fit" for snap in journey.snapshots)
        report.reopenings += sum(snap.reopening for snap in journey.snapshots)
        goal = str(journey.snapshots[-1].state.purpose)
        report.goals[goal] = report.goals.get(goal, 0) + 1
        for snap in journey.snapshots:
            report.locked += bool(snap.state.plan_locked)
            for stage in snap.log.stages:
                load = sum(l.label == "Decide" and l.counted for l in stage.lines)
                report.decide_load = max(report.decide_load, load)
                if stage.reached:
                    report.reached[stage.key] = report.reached.get(stage.key, 0) + 1
        report.violations += check(journey)
    return report

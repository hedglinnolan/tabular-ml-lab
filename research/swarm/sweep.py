"""The sweep and the crash sweep (SWARM_PLAN §6 "Mechanical arms" and "Order" step 1, §7 "Material, in
the loss's units" and "Refused refits", §8 "The sweep's refits under the full pipeline").

    python -m research.swarm.sweep crash --dataset <id> --out <dir>
    python -m research.swarm.sweep plan  --dataset <id> --bundle <export.zip> --name <plan> --out <dir>
                                         [--data <file>]

**Around a named plan.** The default plan (the default path's, ``research/swarm/arms.py``, walked
here), or a named final plan (an export bundle's decision log rebuilt with ``core/export/replay.py``
truncated before ``open_seal``, under Estimate before ``lock_plan``, in the sweep's home). The plan
is rebuilt once and kept as it stands, its stages computed, before any press of Fit (the
*template*); every refit, the plan's own included, runs on a copy of it (``arms.Local.copy``: the
truncated log byte for byte and the stages it computed), so each starts from the plan with no seal
and no lock, revisits its one decision, presses Fit (under Estimate that press records the refit's
own lock, which no later change could withdraw once its estimate was shown) and is read. A copy is
removed once read, but for one that met an engine defect, which is kept for its reproduction.

**What is swept** (:func:`points_of`): each Router question the plan's log answers in the person's
own words (``recorded_by: you``) but those the card and its table fix (``arms.FIXED``: the
outcome, its event level, the goal, the outcome's kind and its follow-up, the table's layout; under
Estimate the exposure the estimand names), each finding it acted on,
and each declaration line (``quest.DECLARATIONS``, ``research/swarm/declarations.py``) the plan
answered or left open or set for it; the options are the driver's own (``options.compose`` for the
question the revisit opens again, a finding's acts as its line offers them,
``declarations.compose`` for a line), never a hand list, and an option that records nothing by
itself (a field of one's own, its ``needs``) is listed, not answered. **Readings with one true
answer** (a column's code or amount, its units: each ``confirm_reading`` but a role's, and each
``set_column_unit``, in force) are graded against the dataset's declared readings, not swept
(:func:`graded`). Under Predict the validation scheme is the sweep's own (below), so its line is
not swept; the crash sweep answers and fits its options all the same.

**Each refit** answers the option on its copy (a question, a finding or a reading by its id after
the revisit, ``driver.chosen``; a declaration's option as its line composed it, which is what the
driver's ``answer`` posts for it); any question the change opens again is answered by the default
path's policy; refits are deduplicated by the plan's digest (``plan_lock.digest`` of the log as
served, folded: the digest every fit record carries), so an option that leaves a plan already
fitted is not fitted again (``duplicate_of``); then Fit is pressed and the refit read: under
Predict each family's cross-validated scores from its fit record, under Estimate the estimate the
app reports (``assemble.result_of`` over the log, the fit records and ``GET …/results``). **Scored**
under Predict with random repeated 5-fold cross-validation, 5 repeats, within the training rows,
whatever folds the plan chose: the template records that scheme (``set_validation``) before it is
copied, answered random where a cut-point seal holds the rows out (it leaves the folds' order
free); where the held-out draw orders the folds by time (a draw by share), random folds cannot be
had without moving the held-out rows, so the plan's own folds score it, and ``scoring`` says so.

**Each refit is classed** (:func:`classed`): ``scored``; ``refused`` in words, with its typed code,
its words and the option refused: by the server at the option's answer, at an answer the walk gave
after it with no way forward and no other option left, or at the sweep's own scoring, or by the
engine at a stage or at the fit; ``failed``, an HTTP 500 anywhere in the refit, an answer that
failed, or a stage or fit error the engine did not type (the driver's typing, ``driver.failures``,
whether the walk met it waiting or it stopped at a limit of its own with a stage of its copy in
error that was not before the option); ``unreached``, an option that records nothing by itself,
or a path that stopped at a limit of the walk's or the revisit's own with no stage failed (its
reason in its words: no engine defect); ``duplicate``.

**Materiality** (§7, :func:`materiality`): under Predict package 1's signed change of each family's
primary score against the plan's (``assemble.signed_change`` over the two fit records, each arm's
n_test/n_train read from the facts of the split its folds were drawn by): paired over the same
folds, or unpaired where the decision changes the analyzed rows; material where its corrected
interval excludes zero and its size is at least half the plan's own interval half-width
(:data:`HALF`). Under Estimate the shift of each estimate against the plan's, material where it is
at least half the plan's interval half-width. A decision's refused share and the refusals' codes
sit beside its materiality (:func:`decisions_of`), the sweep's own, kept apart from any analyst's.
Which decisions are material for the pilot's scoring is ``prereg.json``'s to fix and
``score_pilot.py``'s to apply; this table holds the numbers and §7's reading of them.

**The table** (``sweep.json``, and ``sweep.md`` to read): the plan, its scoring, each refit (its
decision, option, class, digest, seconds, scores or estimates, and change), each decision's
summary, and the graded readings: what §7's scoring and §9's theorist read.

**The crash sweep** (``crash``; §6 step 1), which is also the sizing and the default-plan half of
the sweep: on one dataset, every option the driver offers at every decision of the default plan,
the declarations and the readings included, each previewed, then answered and fitted once; after
each change every later decision's default is previewed again at the plan's own answer to it (a
question's, a finding's, a declaration's or a reading's; a line the plan left open has none), the
later decisions being those after a question in the plan's order, and every other for a finding, a
declaration or a reading, which the Router does not order among its questions (:func:`later_of`;
under ``only``, among the decisions swept); and each question the change opens again is previewed
at the default path's answer before the walk answers it. An HTTP 500, or a fit that failed with an
error the engine did not type, is an engine defect, reported with its reproduction (``defects/``:
the decision log as it stood, the data file, the requests that failed, and the copy of the project
kept); each refusal is kept with its reason; each fit's run time is measured and summed into the
dataset's fit budget and the sweep's compute estimate (``crash.json``). It is resumable: its home
with the template is kept under ``--out``, and ``progress.jsonl`` holds the plan and every option
finished, written as it goes, so a second run takes up where the first stopped without refitting
what it finished. It passes (:func:`passed`) when every decision of the plan was swept (``only``,
said in the report, is none), every option of each was visited (a decision whose options could not
be listed, or that offered none though the plan answered it, was not: :func:`unlisted_of`) and no
engine defect was met, on the engine commit it ran on; the report names the table it ran on by its
SHA-256, which the launcher holds to the dataset's manifest.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from research.swarm import arms

REPO = Path(__file__).resolve().parents[2]
FOLDS, REPEATS = 5, 5  # §7: random repeated 5-fold cross-validation, 5 repeats
SCORING = {"kind": "set_validation", "validation": "repeated_kfold", "folds": FOLDS,
           "repeats": REPEATS}
HALF = 0.5  # §7: material at least half the plan's own interval half-width
SCHEME = "set_validation"  # the sweep's own scoring under Predict (the module docstring)
# Records that answer no question an option of the plan's changes: the seal and the lock (cut away
# or the sweep's own), reverts, and the role confirmations (the roles question's), the outcome
# views and the Confirm sweeps.
NOT_A_POINT = ("open_seal", "lock_plan", "reseal", "revert", "confirm_role", "confirm_readings",
               "view_outcome", "confirm_sweep")
READING_KINDS = ("confirm_reading", "set_column_unit")
FINDING_KINDS = ("apply_repair", "dismiss_finding", "defer_finding")
REFIT_WAIT = 3600.0
PLAN = "plan"  # the refit of the plan itself, every other refit's comparison


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def engine_version() -> dict[str, Any]:
    """The commit the sweep ran on, the tree id of ``turbotab/`` there (the engine the freeze
    names: the same for every commit that leaves it unchanged), and whether tracked files under it
    differ from that commit."""
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                                  timeout=30, check=True).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no", "--", "turbotab")
    return {"commit": git("rev-parse", "HEAD"), "engine_tree": git("rev-parse", "HEAD:turbotab"),
            "engine_modified": bool(status)}


# ── the log ──────────────────────────────────────────────────────────────────


def records_of(view: Mapping[str, Any]) -> list[Any]:
    from turbotab.core import decisions as D

    return [D.DecisionRecord.model_validate(r) for r in view.get("decisions") or []]


def live(view: Mapping[str, Any]) -> list[Any]:
    """The records in force: every record no later revert withdrew, the reverts left out."""
    from turbotab.core import decisions as D

    records = records_of(view)
    cancelled = D.reverted(records)
    return [r for r in records if r.id not in cancelled and r.decision.kind != "revert"]


def plan_digest(view: Mapping[str, Any]) -> str:
    """The plan's digest of the log as served, folded (``plan_lock.digest(plan_of(state))``, the
    digest each fit record carries)."""
    from turbotab.core import decisions as D, plan_lock

    return plan_lock.digest(plan_lock.plan_of(D.fold(records_of(view))))


def last_seq(view: Mapping[str, Any]) -> int:
    return max((int(r.get("seq") or 0) for r in view.get("decisions") or []), default=0)


def errored(view: Mapping[str, Any]) -> set[tuple[str, Any]]:
    """Each stage in error in ``view``, with the key it failed at."""
    return {(name, (status or {}).get("key")) for name, status in (view.get("stages") or {}).items()
            if (status or {}).get("status") == "error"}


# ── what is swept ────────────────────────────────────────────────────────────


@dataclass
class Point:
    """One decision of the plan the sweep changes: a Router question (``key``), a finding acted on
    (``key`` its id), a declaration line (``key`` its kind), or (the crash sweep) a reading
    (``key`` ``<reading>:<column>``); the plan's record of it, where it has one."""

    kind: str  # "question", "finding", "declaration", "reading"
    key: str
    seq: int | None = None
    record: str | None = None
    decision: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.key}"

    def json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "key": self.key, "seq": self.seq,
                "record": self.record, "decision": self.decision}

    @classmethod
    def of(cls, data: Mapping[str, Any]) -> "Point":
        return cls(**{k: data.get(k) for k in ("kind", "key", "seq", "record", "decision")})


def question_key(kind: str) -> str | None:
    from turbotab.core.sequence import question_of

    return question_of(kind)


def points_of(view: Mapping[str, Any], ask: Mapping[str, Any], *, purpose: str,
              crash: bool) -> tuple[list[Point], list[dict[str, Any]]]:
    """The decisions swept around the plan (the module docstring), in the log's order with the
    declarations it never answered after; and those not swept, each with why."""
    from research.swarm import declarations as decl

    points: list[Point] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    fixed = set(arms.FIXED) | ({"estimand"} if purpose == "inference" else set())
    for r in reversed(live(view)):  # the latest record of each decision is the one in force
        kind = r.decision.kind
        body = r.decision.model_dump(mode="json")
        if kind in NOT_A_POINT:
            continue
        if kind in READING_KINDS:
            if kind == "confirm_reading" and body.get("reading") == "role":
                continue
            reading = body.get("reading") if kind == "confirm_reading" else "unit"
            point = Point("reading", f"{reading}:{body.get('column')}", r.seq, r.id, body)
        elif kind in FINDING_KINDS:
            point = Point("finding", str(body.get("finding_id")), r.seq, r.id, body)
        elif kind in decl.kinds():
            point = Point("declaration", kind, r.seq, r.id, body)
        else:
            key = question_key(kind)
            if key is None:
                skipped.append({"record": r.seq, "kind": kind,
                                "why": "It answers no Router question."})
                continue
            point = Point("question", key, r.seq, r.id, body)
        if point.id in seen:
            continue
        seen.add(point.id)
        if point.kind == "question" and point.key in fixed:
            skipped.append({**point.json(), "why": "The card fixes it: another answer is another "
                                                   "question."})
        elif getattr(r, "recorded_by", "you") != "you":
            skipped.append({**point.json(), "why": "TurboTab recorded it itself."})
        elif point.kind == "reading" and not crash:
            skipped.append({**point.json(), "why": "A reading with one true answer: graded, not "
                                                   "swept."})
        elif point.kind == "declaration" and point.key == SCHEME and purpose == "prediction" \
                and not crash:
            skipped.append({**point.json(), "why": SCHEME_IS_THE_SWEEPS})
        else:
            points.append(point)
    points.sort(key=lambda p: p.seq or 0)
    answered = {p.key for p in points if p.kind == "declaration"} | {
        s.get("key") for s in skipped if s.get("kind") == "declaration"}
    for line in ask.get("also_open") or []:
        kind = line.get("key")
        if line.get("source") != "declaration" or kind not in decl.kinds() or kind in answered:
            continue
        answered.add(kind)
        if kind == SCHEME and purpose == "prediction" and not crash:
            skipped.append({"id": f"declaration:{kind}", "kind": "declaration", "key": kind,
                            "why": SCHEME_IS_THE_SWEEPS})
            continue
        points.append(Point("declaration", str(kind)))
    return points, skipped


def later_of(points: Sequence[Point], point: Point) -> list[Point]:
    """The decisions whose default the crash sweep previews again after an option of ``point``
    (the module docstring), in the plan's order (:func:`points_of`): for a Router question, every
    decision after it; for a finding, a declaration or a reading, which the Router does not order
    among its questions, every other decision. Each is previewed at the plan's own answer to it,
    so one the plan left open (a declaration line it never answered) is listed but has none."""
    if point.kind != "question":
        return [p for p in points if p.id != point.id]
    at = next(i for i, p in enumerate(points) if p.id == point.id)
    return list(points[at + 1:])


SCHEME_IS_THE_SWEEPS = (f"Every refit is scored with random repeated {FOLDS}-fold "
                        f"cross-validation, {REPEATS} repeats, so the scheme is the sweep's, not "
                        f"the plan's: a refit of it is scored on other folds and moves no model.")


def readings_truth(dataset: str | None) -> dict[str, str]:
    """The dataset's declared readings (``answers.json``: the truth naming the readings a user who
    knows the table gives, ``<reading>:<column>`` → value), or none."""
    if not dataset:
        return {}
    try:
        answers = json.loads((arms.DATASETS / dataset / "answers.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    for truth in answers.get("truths") or []:
        value = truth.get("value")
        if str(truth.get("name", "")).startswith("the readings") and isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items()}
    return {}


def graded(view: Mapping[str, Any], truth: Mapping[str, str]) -> list[dict[str, Any]]:
    """Each reading with one true answer the plan records (a ``confirm_reading`` but a role's, a
    column's unit), against the dataset's declared one: right, wrong, or not declared (None)."""
    out = []
    for r in live(view):
        if r.decision.kind not in READING_KINDS:
            continue
        body = r.decision.model_dump(mode="json")
        if r.decision.kind == "set_column_unit":
            key, value = f"unit:{body.get('column')}", body.get("unit")
        else:
            if body.get("reading") == "role":
                continue
            key, value = f"{body.get('reading')}:{body.get('column')}", body.get("value")
        declared = truth.get(key)
        out.append({"reading": key, "seq": r.seq, "value": value, "declared": declared,
                    "right": None if declared is None else str(value) == str(declared)})
    return out


# ── one refit ────────────────────────────────────────────────────────────────


@dataclass
class Refit:
    """One refit: the decision (``point``) and option, its class (:func:`classed`) and why, the
    plan's digest it fitted (or the refit it duplicates), the fit's key and record, its seconds
    (``seconds``, the press to the fit's end; ``fit_seconds``, the server's own; ``took``, the
    whole refit), what it scored or estimated, its change against the plan and whether §7 reads it
    material; the previews and answers made on the way; the copy it ran on, while kept."""

    point: str
    option: str | None
    decision: dict[str, Any] | None
    kind: str = "unreached"  # scored, refused, failed, unreached, duplicate
    why: str | None = None
    refused: dict[str, Any] | None = None
    failed: dict[str, Any] | None = None
    digest: str | None = None
    duplicate_of: str | None = None
    project: str | None = None
    fit_key: str | None = None
    fit_record: str | None = None
    seconds: float | None = None
    fit_seconds: float | None = None
    took: float | None = None
    scores: dict[str, Any] | None = None
    estimates: list[dict[str, Any]] | None = None
    estimate_why: str | None = None
    split: dict[str, Any] | None = None
    change: list[dict[str, Any]] = field(default_factory=list)
    change_why: str | None = None
    material: bool | None = None
    previews: list[dict[str, Any]] = field(default_factory=list)
    answered: list[dict[str, Any]] = field(default_factory=list)
    record: dict[str, Any] | None = None  # the fit record as served (Predict: the change's arm)

    @property
    def id(self) -> str:
        return self.point if self.option is None else f"{self.point}={self.option}"

    def json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["id"] = self.id
        return out

    @classmethod
    def of(cls, data: Mapping[str, Any]) -> "Refit":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def said_json(said: Any) -> dict[str, Any]:
    return {"kind": said.kind, "by": said.by, "code": said.code, "message": said.message,
            "status": said.status}


def classed(fit: Any) -> tuple[str, dict[str, Any] | None]:
    """A press of Fit (``arms.Said``), classed (the module docstring): ``scored`` where its status
    turned fresh; ``failed`` for an HTTP 500 or a fit error the engine did not type; ``unreached``
    where no fit was made (it is blocked, or held); else ``refused`` in words, by the engine at the
    fit (its typed code, words, stage and ways forward) or by the server at the press. With what
    the class rests on."""
    if fit.kind == "ok":
        return "scored", None
    if fit.kind == "failed":
        return "failed", said_json(fit) | {"error": (fit.payload.get("fit") or {}).get("error")}
    if fit.by == "driver":
        return "unreached", said_json(fit)
    return "refused", said_json(fit) | {
        "at": "fit", "stage": (fit.payload.get("refused") or {}).get("stage"),
        "exits": [{k: e.get(k) for k in ("label", "decision")} for e in fit.exits]}


def fit_record(session: Any, key: str | None) -> dict[str, Any] | None:
    """The latest fit record ``GET …/fits`` serves for the fit ``key``."""
    fits = session.read("fits") or {}
    found = [r for r in fits.get("records") or [] if key and r.get("fit_key") == key]
    return found[-1] if found else None


def split_facts(session: Any) -> dict[str, Any] | None:
    """The facts of the folds the current answers draw (``signed.SPLIT_FACTS``), from the split
    stage while it is fresh at its current key, or None."""
    from research.swarm.signed import split_facts as facts

    key = ((session.view().get("stages") or {}).get("split") or {}).get("key")
    served = session.read("stages/split") or {}
    if not (served.get("fresh") and key and served.get("key") == key):
        return None
    return facts(served.get("artifact"))


def split_facts_when_fresh(session: Any, wait: float = 120.0) -> dict[str, Any] | None:
    end = time.monotonic() + wait
    while time.monotonic() < end:
        facts = split_facts(session)
        if facts is not None:
            return facts
        status = ((session.view().get("stages") or {}).get("split") or {}).get("status")
        if status in ("error", "blocked"):
            return None
        time.sleep(0.5)
    return None


def estimate_of(session: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """The estimate the app reports for the plan's estimand, as the record reads it
    (``assemble.result_of`` under Estimate, over the log, the fit records and the results), or why
    there is none."""
    from research.swarm import assemble as A

    view = session.view()
    fits = (session.read("fits") or {}).get("records") or []
    r = session.client.get(f"/projects/{session.pid}/results")
    results = A.Read(r.status == 200, r.json() if r.status == 200 else None)
    got = A.result_of("inference", view.get("decisions") or [], fits, results)
    if got.get("refused"):
        return None, str(got["refused"])
    return list(got.get("estimates") or []), None


def read_refit(session: Any, refit: Refit, fit: Any, purpose: str) -> None:
    """The refit's class (:func:`classed`) and, where it scored, its numbers."""
    refit.kind, detail = classed(fit)
    if refit.kind == "failed":
        refit.failed = detail
    elif refit.kind == "refused":
        refit.refused = (detail or {}) | {"option": refit.option}
    elif refit.kind == "unreached":
        refit.why = (detail or {}).get("message")
    status = fit.payload.get("fit") or {}
    refit.fit_key = status.get("key")
    refit.seconds = fit.payload.get("seconds")
    if refit.kind != "scored":
        return
    record = fit_record(session, refit.fit_key)
    if record is None:
        refit.kind, refit.why = "unreached", "No fit record holds the fit's key."
        return
    refit.fit_record, refit.fit_seconds = record.get("id"), record.get("seconds")
    if purpose == "prediction":
        refit.record = record
        refit.scores = {"metric": record.get("primary_metric"), "folds": record.get("folds"),
                        "families": record.get("scores") or {}}
        refit.split = split_facts(session)
    else:
        refit.estimates, refit.estimate_why = estimate_of(session)


# ── materiality ──────────────────────────────────────────────────────────────


def predict_change(after: Mapping[str, Any], before: Mapping[str, Any],
                   after_split: Mapping[str, Any] | None,
                   before_split: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], str | None]:
    """Package 1's signed change of each family's primary score, ``after`` (the refit's fit record)
    against ``before`` (the plan's), each arm's share read from the facts of the split its folds
    were drawn by (``assemble.signed_change``)."""
    from research.swarm import assemble as A

    splits: dict[str, Any] = {}
    for record, facts in ((after, after_split), (before, before_split)):
        if record.get("folds") and facts is not None:
            splits[str(record["folds"])] = facts
    return A.signed_change(after, before, splits)


def material_predict(change: Mapping[str, Any]) -> bool | None:
    """§7 under Predict: the corrected interval excludes zero and the change is at least half the
    plan's own interval half-width (:data:`HALF`); None where the change has no number."""
    diff, half = change.get("difference"), change.get("half_width")
    if diff is None or half is None or change.get("excludes_zero") is None:
        return None
    return bool(change["excludes_zero"]) and abs(float(diff)) >= HALF * float(half)


def estimate_shift(after: Sequence[Mapping[str, Any]], before: Sequence[Mapping[str, Any]]
                   ) -> list[dict[str, Any]]:
    """Each estimate's shift against the plan's for the same family, feature and setting, beside
    half the plan's interval half-width (§7 under Estimate), or why it has none."""
    def key(e: Mapping[str, Any]) -> tuple[Any, ...]:
        return (e.get("family"), str(e.get("feature")), e.get("setting"))

    then = {key(e): e for e in before}
    out = []
    for e in after:
        b = then.get(key(e))
        row: dict[str, Any] = {"family": e.get("family"), "feature": e.get("feature"),
                               "setting": e.get("setting"), "shift": None, "half_width": None,
                               "material": None, "why": None}
        if b is None:
            row["why"] = "The plan reports no estimate for this family, feature and setting."
        elif e.get("estimate") is None or b.get("estimate") is None:
            row["why"] = "One of the two has no estimate."
        elif b.get("ci_low") is None or b.get("ci_high") is None:
            row["why"] = "The plan's estimate has no interval."
        else:
            shift = float(e["estimate"]) - float(b["estimate"])
            half = (float(b["ci_high"]) - float(b["ci_low"])) / 2
            row.update(shift=shift, half_width=half, material=abs(shift) >= HALF * half)
        out.append(row)
    return out


def materiality(refit: Refit, plan: Refit, purpose: str) -> None:
    """The refit's change against the plan's (§7, the module docstring), and whether §7 reads it
    material: any family's (Predict) or any estimate's (Estimate); None where no number does."""
    if refit.kind != "scored" or plan.kind != "scored":
        return
    if purpose == "prediction":
        if refit.point == f"declaration:{SCHEME}":
            refit.change_why = SCHEME_IS_THE_SWEEPS
            return
        if refit.record is None or plan.record is None:
            refit.change_why = "A fit record was not read."
            return
        refit.change, refit.change_why = predict_change(refit.record, plan.record, refit.split,
                                                        plan.split)
        for c in refit.change:
            c["material"] = material_predict(c)
    else:
        if refit.estimates is None or plan.estimates is None:
            refit.change_why = refit.estimate_why or plan.estimate_why
            return
        refit.change = estimate_shift(refit.estimates, plan.estimates)
    known = [c["material"] for c in refit.change if c.get("material") is not None]
    refit.material = any(known) if known else None


def decisions_of(points: Sequence[Point], refits: Sequence[Refit]) -> list[dict[str, Any]]:
    """Each decision's summary: its options, how its refits were classed, its refused share with
    the refusals' codes beside its materiality (§7, "Refused refits"). The share is of the options
    refitted, each refusal of the engine's among them wherever on the refit's path it came: a
    duplicate is the plan it duplicates, and an option that records nothing by itself, or whose
    path stopped at a limit of the walk's own (``unreached``), was not refitted."""
    out = []
    for p in points:
        mine = [r for r in refits if r.point == p.id]
        counted = [r for r in mine if r.kind in ("scored", "refused", "failed")]
        refused = [r for r in counted if r.kind == "refused"]
        codes: dict[str, int] = {}
        for r in refused:
            code = str((r.refused or {}).get("code"))
            codes[code] = codes.get(code, 0) + 1
        material = [r.material for r in counted if r.material is not None]
        out.append({
            "point": p.id, "options": len(mine),
            "classes": {k: sum(1 for r in mine if r.kind == k)
                        for k in ("scored", "refused", "failed", "unreached", "duplicate")},
            "refused_share": (len(refused) / len(counted)) if counted else None,
            "refused_codes": codes, "refused_options": [r.option for r in refused],
            "material": (any(material) if material else None),
            "material_options": [r.option for r in counted if r.material],
        })
    return out


def unlisted_of(points: Sequence[Point], offered: Mapping[str, Mapping[str, Any]]
                ) -> list[dict[str, Any]]:
    """Each decision whose options were not listed, so none of them was visited: not yet listed;
    listed with why they could not be (a revisit refused, a question that did not open again); or
    listed with none, where the plan records an answer to it (a decision the plan answered has at
    least that answer to offer)."""
    out = []
    for p in points:
        listed = offered.get(p.id)
        if listed is None:
            out.append({"point": p.id, "why": "Its options were not listed."})
        elif listed.get("why"):
            out.append({"point": p.id, "why": str(listed["why"])})
        elif not listed.get("options") and p.decision is not None:
            out.append({"point": p.id, "why": "No option was offered for a decision the plan "
                                              "answered."})
    return out


def passed(report: Mapping[str, Any]) -> bool:
    """Whether the crash sweep passed (the module docstring): every decision of the plan swept
    (no ``only``), every option visited, no engine defect met."""
    return bool(report.get("complete") and not report.get("defects")
                and "only" in report and report["only"] is None)


def timed_of(refit: Refit) -> float | None:
    """A refit's fit's run time: the server's own seconds for its fit record (from the fit's job
    running to its key turning fresh, however long before the press it ran: the scheduler fits a
    quick fit before Fit is pressed), else the press's wait for a fit that made no record (refused
    or failed); None where no fit was pressed (an answer refused, a duplicate, a path that did not
    reach Fit)."""
    if refit.kind not in ("scored", "refused", "failed"):
        return None
    if refit.fit_seconds is not None:
        return float(refit.fit_seconds)
    return None if refit.seconds is None else float(refit.seconds)


def seconds_of(timed: Sequence[float]) -> dict[str, Any]:
    """The fits' run times: how many, their sum, and their median, 90th percentile (the nearest
    rank) and largest."""
    if not timed:
        return {"fits": 0, "total": 0.0, "median": None, "p90": None, "max": None}
    ordered = sorted(float(t) for t in timed)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {"fits": len(ordered), "total": round(sum(ordered), 3),
            "median": round(statistics.median(ordered), 3), "p90": round(p90, 3),
            "max": round(ordered[-1], 3)}


# ── the sweep ────────────────────────────────────────────────────────────────


class Sweep:
    """One sweep around one plan in ``out`` (the module docstring), resumable from its
    ``progress.jsonl``. ``crash``: the crash sweep's mode (previews, readings, the scheme).
    ``only``: the decisions swept, by point id (every one where None), said in the report.
    ``data``: the table the plan was made from, named in each defect's reproduction."""

    def __init__(self, out: Path, *, dataset: str | None, question: arms.Question,
                 crash: bool, plan: str, only: Iterable[str] | None = None,
                 log: Callable[[str], None] | None = None, data: Path | None = None):
        self.out = Path(out).absolute()
        self.dataset, self.question, self.crash, self.plan = dataset, question, crash, plan
        self.data = None if data is None else str(Path(data).absolute())  # the table, for a defect
        self.only = set(only) if only is not None else None
        self.progress = self.out / "progress.jsonl"
        self.policy = arms.Default(question)
        self.done: dict[str, Refit] = {}
        self.base: dict[str, Any] | None = None
        self.points: list[Point] = []
        self.skipped: list[dict[str, Any]] = []
        self.defects: list[dict[str, Any]] = []
        self.offered: dict[str, dict[str, Any]] = {}
        self.made: set[str] = set()
        self.log = log or (lambda text: None)
        self._read_progress()

    # progress

    def _read_progress(self) -> None:
        try:
            lines = self.progress.read_text("utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a line an interruption cut
            if "template" in entry:
                self.base = entry["template"]
            elif "refit" in entry:
                refit = Refit.of(entry["refit"])
                self.done[refit.id] = refit
            elif "offered" in entry:
                self.offered[entry["offered"]["point"]] = entry["offered"]
            elif "defect" in entry:
                self.defects.append(entry["defect"])

    def _write(self, entry: Mapping[str, Any]) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with self.progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # the template and the plan's own refit

    def template(self, session: Any, build: Callable[[Any], str]) -> None:
        """The plan rebuilt once (``build(session)`` gives its project's id), its scoring recorded
        under Predict, its stages left to compute, and its decisions read; kept in the progress
        file, and on a second run read from it."""
        if self.base is None:
            pid = build(session)
            session.use(pid)
            # The plan's decisions, read before the sweep records its own scoring on it.
            points, skipped = points_of(session.view(), session.ask(),
                                        purpose=self.question.purpose, crash=self.crash)
            scoring = self._scoring(session)
            if not session.idle(pid):
                raise arms.Stuck("The plan kept computing.")
            view = session.view()
            self.base = {"project": pid, "digest": plan_digest(view), "seq": last_seq(view),
                         "scoring": scoring, "points": [p.json() for p in points],
                         "skipped": skipped}
            self._write({"template": self.base})
        self.points = [Point.of(p) for p in self.base["points"]]
        self.skipped = list(self.base["skipped"])
        if self.only is not None:
            self.points = [p for p in self.points if p.id in self.only]

    def _scoring(self, session: Any) -> dict[str, Any]:
        """Random repeated 5-fold cross-validation, 5 repeats, within the training rows (the module
        docstring), recorded on the template under Predict; or why not."""
        if self.question.purpose != "prediction":
            return {"set": False, "why": "Under Estimate the estimate is read, not a score."}
        split = (session.view().get("state") or {}).get("split") or {}
        body = dict(SCORING)
        if split.get("cut_point"):
            body["fold_order"] = "random"
        said = session.decide(body, f"The sweep's scoring: random repeated {FOLDS}-fold "
                                    f"cross-validation, {REPEATS} repeats.")
        if not said.ok:
            return {"set": False, "why": f"{said.code}: {said.message}", "decision": body}
        facts = split_facts_when_fresh(session)
        out: dict[str, Any] = {"set": True, "decision": body, "facts": facts}
        if facts is not None and facts.get("fold_scheme") == "time_ordered":
            out["why"] = ("The held-out draw orders the folds by time: they cannot be random "
                          "without moving the held-out rows, so the plan's own folds score it.")
        return out

    def plan_refit(self, session: Any) -> Refit:
        """The plan's own refit, on a copy of the template: Fit pressed and read."""
        if PLAN in self.done:
            return self.done[PLAN]  # its fit record kept with it (``Refit.record``)
        plan = Refit(PLAN, None, None)
        copy = self._copy(session)
        plan.project = copy
        read_refit(session, plan, session.fit(REFIT_WAIT), self.question.purpose)
        plan.digest = plan_digest(session.view())
        if plan.kind != "scored":
            raise arms.Stuck(f"The plan itself was not fitted: {plan.kind}: "
                             f"{json.dumps(plan.refused or plan.failed or plan.why)[:600]}")
        self.done[PLAN] = plan
        self._write({"refit": plan.json()})
        return plan

    def _copy(self, session: Any) -> str:
        copy = session.copy(str(self.base["project"]))
        self.made.add(copy)
        session.use(copy)
        return copy

    # the options

    def options(self, session: Any, point: Point) -> list[dict[str, Any]]:
        """The options the driver offers for ``point`` on the session's project, the plan's own
        among them, once they are on offer: after the revisit, for a question, a finding or a
        reading."""
        if point.kind == "declaration":
            from research.swarm import declarations as decl, driver

            composed = decl.compose(point.key, session.view(),
                                    driver.served(session.run, session.client))
            return list(composed.get("options") or [])
        said = session.revisit(str(point.record), f"The sweep revisits {point.id}.")
        if not said.ok:
            raise arms.Stuck(f"Revisiting {point.id} was {said.kind}: {said.code}: "
                             f"{said.message}")
        q = self._reopened(session, point)
        if point.kind == "question":
            return list(q.get("options") or [])
        if point.kind == "finding":
            line = next((ln for ln in q.get("also_open") or [] if ln.get("source") == "finding"
                         and str(ln.get("key")) == point.key), {})
            return list(line.get("options") or [])
        reading, column = point.key.split(":", 1)
        if reading == "unit":
            return [w for w in arms.ways_of(q) if w.get("id") and (w.get("decision") or {}).get(
                "kind") == "set_column_unit" and (w.get("decision") or {}).get("column") == column]
        return list(arms.reading_ways(q).get((reading, column)) or [])

    def _reopened(self, session: Any, point: Point) -> dict[str, Any]:
        """The ask once the revisited question is open again (a reading or a role the server asks
        on the way settled by the default policy), or the ask for a finding or a reading."""
        if point.kind != "question":
            return session.ask()
        for _ in range(arms.STEPS):
            q = session.ask()
            question = q.get("question") or {}
            if question.get("key") == point.key and question.get("status") == "open":
                return q
            settled = self.policy.settle(q)
            if settled is not None:
                session.answer([settled[0]], [], f"{self.policy.name}: {settled[1]}.")
                continue
            if question.get("status") == "waiting":
                if q.get("failed"):
                    raise arms.met_stages(arms.failed_stages(session, q),
                                          f"Revisiting {point.id}, the {question.get('key')} "
                                          f"question waits on stages that failed: "
                                          f"{json.dumps(q['failed'])[:800]}")
                time.sleep(0.5)
                continue
            raise arms.Stuck(f"Revisiting {point.id} left the {question.get('key')} question "
                             f"open first.")
        raise arms.Stuck(f"The {point.key} question did not open again.")

    # one option

    def refit(self, session: Any, point: Point, option: Mapping[str, Any],
              later: Sequence[Point], plan: Refit) -> Refit:
        """One option refitted on a copy of the template, classed, read, and the copy removed but
        for an engine defect's (the module docstring)."""
        refit = Refit(point.id, str(option.get("id")), dict(option.get("decision") or {}))
        started = time.monotonic()
        errors = len(session.errors)
        before: set[tuple[str, Any]] | None = None
        try:
            refit.project = self._copy(session)
            before = self._errored(session)
            if point.kind != "declaration":
                offered = {str(o.get("id")) for o in self.options(session, point)}
                if refit.option not in offered:
                    raise arms.Stuck(f"The revisit offers no option {refit.option} now.")
            self._refit(session, point, option, refit, later, plan)
        except arms.Met as met:
            if refit.kind not in ("failed", "refused"):
                self._met(refit, met)
        except arms.Stuck as exc:
            if refit.kind not in ("failed", "refused"):
                refit.kind, refit.why = "unreached", str(exc)
                if before is not None:
                    self._failed_anew(session, refit, before)
        new_errors = list(session.errors[errors:])
        if new_errors and refit.kind != "failed":
            refit.kind = "failed"
            refit.failed = {"kind": "failed", "by": "server", "code": "http_500",
                            "message": new_errors[-1].get("said")}
        if refit.kind == "failed":
            refit.failed = (refit.failed or {}) | {"option": refit.option}
            self._defect(session, refit, new_errors)
        else:
            materiality(refit, plan, self.question.purpose)
            if refit.project is not None and session.drop(refit.project, self.made):
                refit.project = None
        refit.took = round(time.monotonic() - started, 3)
        return refit

    @staticmethod
    def _errored(session: Any) -> set[tuple[str, Any]] | None:
        """The copy's stages in error before the option (:func:`errored`), or None where the
        project could not be read (a 500, which classes the refit failed)."""
        from research.swarm import driver

        try:
            return errored(session.view())
        except driver.DriverError:
            return None

    @staticmethod
    def _met(refit: Refit, met: arms.Met) -> None:
        """The refit classed as the engine's words that stopped it (:class:`arms.Met`): failed or
        refused, with what the class rests on and the option."""
        detail = dict(met.detail) | {"option": refit.option}
        refit.kind, refit.why = met.kind, str(met)
        if met.kind == "failed":
            refit.failed = detail
        else:
            refit.refused = detail

    def _failed_anew(self, session: Any, refit: Refit, before: set[tuple[str, Any]]) -> None:
        """A refit whose path stopped at a limit of the walk's own (``refit.why``), where stages
        of its copy are in error that were not before the option (a question opened over a stage
        that failed, say): classed by the driver's typing of them (``driver.failures``; the
        module docstring), failed where any is an error the engine did not type, else refused.
        Unreached where none is."""
        from research.swarm import driver

        try:
            view = session.view()
        except driver.DriverError:
            return  # the project unreadable is a 500, which the refit is classed failed by
        anew = [name for name, status in (view.get("stages") or {}).items()
                if (status or {}).get("status") == "error"
                and (name, (status or {}).get("key")) not in before]
        if anew:
            self._met(refit, arms.met_stages(driver.failures(view, anew),
                                             f"{refit.why} Stages failed on the way: "
                                             f"{', '.join(anew)}."))

    def _post(self, session: Any, point: Point, option: Mapping[str, Any], reason: str,
              preview: bool = False) -> Any:
        """The option previewed or answered as the driver does: a question's, a finding's or a
        reading's by its id; a declaration's by the decision its line composed, which is what the
        driver's ``answer`` posts for it (``declarations.resolve`` with no field set)."""
        if point.kind == "declaration":
            decision = dict(option["decision"])
            return session.preview_of(decision) if preview else session.decide(decision, reason)
        ids = [str(option["id"])]
        if preview:
            return session.preview(ids, [], reason)
        return session.answer(ids, [], reason, arms.MARKER_OF.get(point.key, arms.DEFAULT_MARKER))

    def _refit(self, session: Any, point: Point, option: Mapping[str, Any], refit: Refit,
               later: Sequence[Point], plan: Refit) -> None:
        if self.crash:
            seen = self._post(session, point, option, f"The crash sweep previews {refit.id}.",
                              preview=True)
            refit.previews.append({"of": refit.id, **said_json(seen)})
        said = self._post(session, point, option, f"The sweep's refit: {refit.id}.")
        refit.answered.append({"answer": refit.id, **said_json(said)})
        if said.kind == "failed":
            refit.kind, refit.failed = "failed", said_json(said)
            return
        if said.kind == "refused":
            refit.kind = "refused"
            refit.refused = said_json(said) | {
                "at": "answer", "option": refit.option,
                "exits": [{k: e.get(k) for k in ("label", "decision")} for e in said.exits]}
            return
        if self.crash:
            for p in later:  # every later decision's default (:func:`later_of`)
                if p.decision is not None:  # a line the plan left open has no answer to preview
                    seen = session.preview_of(self._later(p))
                    refit.previews.append({"of": f"{p.id} (the plan's answer)",
                                           **said_json(seen)})
        steps: list[arms.Step] = []
        before = (lambda key, q: self._preview_reopened(session, refit, key, q)) if self.crash \
            else None
        arms.walk(session, self.policy, steps=steps, before=before)
        refit.answered += [{"answer": f"{s.at}:{s.key}={','.join(s.chosen)}", "kind": s.said,
                            "code": s.code} for s in steps]
        if point.id != f"declaration:{SCHEME}":
            self._rescore(session, refit)
        refit.digest = plan_digest(session.view())
        seen_plan = next((r for r in [plan, *self.done.values()] if r.digest == refit.digest
                          and r.kind in ("scored", "refused", "failed")), None)
        if seen_plan is not None:
            refit.kind, refit.duplicate_of = "duplicate", seen_plan.id
            return
        read_refit(session, refit, session.fit(REFIT_WAIT), self.question.purpose)

    def scheme(self, split: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """The sweep's scoring as the split ``split`` takes it (the folds' order random under a
        cut-point), where the template recorded it; else None."""
        scoring = (self.base or {}).get("scoring") or {}
        if not scoring.get("set"):
            return None
        body = dict(SCORING)
        if (split or {}).get("cut_point"):
            body["fold_order"] = "random"
        return body

    def _rescore(self, session: Any, refit: Refit) -> None:
        """The sweep's scoring recorded again where the refit's answers changed the scheme (a split
        answer records its own): every refit is scored on the same scheme (the module
        docstring)."""
        split = (session.view().get("state") or {}).get("split") or {}
        body = self.scheme(split)
        if body is None:
            return
        if all(split.get(k) == v for k, v in body.items() if k != "kind"):
            return
        said = session.decide(body, f"The sweep's scoring again: random repeated {FOLDS}-fold "
                                    f"cross-validation, {REPEATS} repeats.")
        refit.answered.append({"answer": "the sweep's scoring", **said_json(said)})
        if not said.ok:
            # Refused, the sweep's own folds are what the engine refuses (§7: the sweep's share).
            raise arms.Met(f"The sweep's scoring could not be recorded after {refit.id}: "
                           f"{said.code}: {said.message}",
                           "failed" if said.kind == "failed" else "refused",
                           said_json(said) | {"at": "scoring", "decision": body,
                                              "exits": arms.exit_fields(said.exits)})

    def _later(self, p: Point) -> dict[str, Any]:
        """The plan's answer to ``p`` as the template holds it: a split answer with the sweep's
        scheme on it, since recorded as it was it would move the scheme only, which the engine
        refuses as the scheme's own answer (``seal._a_changed_scheme_is_its_own_kind``)."""
        decision = dict(p.decision or {})
        if decision.get("kind") == "set_split":
            body = self.scheme(decision)
            if body is not None:
                decision.update({k: v for k, v in body.items() if k != "kind"})
        return decision

    def _preview_reopened(self, session: Any, refit: Refit, key: str,
                          q: Mapping[str, Any]) -> None:
        ids, sets, _ = self.policy.answer(key, q, session.read)
        seen = session.preview(ids, sets, f"The crash sweep previews {key}'s default again.")
        refit.previews.append({"of": f"question:{key} (opened again, its default)",
                               **said_json(seen)})

    def _defect(self, session: Any, refit: Refit, errors: Sequence[Mapping[str, Any]]) -> None:
        """An engine defect, with its reproduction (``defects/<id>.json``): the decision log as it
        stood, the data file, the requests that failed (or the untyped error), each stage of the
        copy in error with its own words (``driver.failures``: a fit that failed on a stage
        upstream says only that it needs it), and the copy of the project it met it on, kept."""
        from research.swarm import driver

        log: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        try:
            view = session.view()
            log = [{k: r.get(k) for k in ("seq", "decision", "recorded_by", "note")}
                   for r in view.get("decisions") or []]
            stages = driver.failures(view, list(view.get("stages") or {}))  # in the graph's order
        except Exception as exc:  # noqa: BLE001 - the project itself fails to read: said
            log = [{"unread": f"{type(exc).__name__}: {str(exc)[:300]}"}]
        folder = self.out / "defects"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{hashlib.sha256(refit.id.encode('utf-8')).hexdigest()[:12]}.json"
        stage = (refit.failed or {}).get("stage") if (refit.failed or {}).get("at") == "stage" \
            else None
        then = ("repeat each request of `requests`" if errors else
                f"read the {stage} stage, in error as `failed` says" if stage else "press Fit")
        defect = {"refit": refit.id, "point": refit.point, "option": refit.option,
                  "failed": refit.failed, "requests": list(errors), "stages": stages,
                  "log": log,
                  "project": refit.project, "data": self.data,
                  "reproduce": f"Make a project from the data file, post each decision of `log` "
                               f"in order, then {then}; the project the sweep met it on is kept "
                               f"in its home."}
        path.write_text(json.dumps(defect, indent=1, default=str) + "\n", "utf-8")
        entry = {"refit": refit.id, "point": refit.point, "option": refit.option,
                 "failed": refit.failed, "path": str(path), "project": refit.project}
        self.defects = [d for d in self.defects if d.get("refit") != refit.id] + [entry]
        self._write({"defect": entry})

    # the whole

    def run(self, session: Any, build: Callable[[Any], str]) -> dict[str, Any]:
        started = time.monotonic()
        self.template(session, build)
        plan = self.plan_refit(session)
        for point in self.points:
            later = later_of(self.points, point)
            listed = self.offered.get(point.id)
            if listed is None:
                copy = self._copy(session)
                errors = len(session.errors)
                met: arms.Met | None = None
                try:
                    options = self.options(session, point)
                    why = None
                except arms.Stuck as exc:
                    options, why = [], str(exc)
                    met = exc if isinstance(exc, arms.Met) and exc.kind == "failed" else None
                failed = list(session.errors[errors:])
                if failed or met is not None:
                    # A 500, or a stage failed untyped, met on the revisit itself: an engine
                    # defect, kept with it.
                    revisit = Refit(point.id, "(the revisit)", None, kind="failed",
                                    failed={"kind": "failed", "by": "server", "code": "http_500",
                                            "message": failed[-1].get("said")} if failed
                                    else dict(met.detail), project=copy)
                    self._defect(session, revisit, failed)
                    self.done[revisit.id] = revisit
                    self._write({"refit": revisit.json()})
                else:
                    session.drop(copy, self.made)
                listed = {"point": point.id, "why": why, "options": [
                    {"id": o.get("id"), "label": o.get("label"), "decision": o.get("decision"),
                     "needs": o.get("needs")} for o in options]}
                self.offered[point.id] = listed
                self._write({"offered": listed})
            for option in listed["options"]:
                rid = f"{point.id}={option.get('id')}"
                if rid in self.done:
                    continue
                if option.get("needs") or not isinstance(option.get("decision"), Mapping):
                    needs = ", ".join(option.get("needs") or []) or "a field of its own"
                    r = Refit(point.id, str(option.get("id")), None, kind="unreached",
                              why=f"It records nothing by itself: it takes {needs}.")
                else:
                    self.log(rid)
                    r = self.refit(session, point, option, later, plan)
                self.done[rid] = r
                self._write({"refit": r.json()})
        return self.report(plan, time.monotonic() - started)

    def report(self, plan: Refit, seconds: float) -> dict[str, Any]:
        refits = [r for k, r in self.done.items() if k != PLAN]
        timed = [timed_of(r) for r in [plan, *refits] if timed_of(r) is not None]
        unvisited = [f"{p.id}={o.get('id')}" for p in self.points
                     for o in (self.offered.get(p.id) or {}).get("options") or []
                     if f"{p.id}={o.get('id')}" not in self.done]
        unlisted = unlisted_of(self.points, self.offered)
        unvisited += [u["point"] for u in unlisted]
        return {
            "dataset": self.dataset, "plan": self.plan, "purpose": self.question.purpose,
            **engine_version(), "written": now(), "seconds": round(seconds, 1),
            "template": {k: (self.base or {}).get(k) for k in ("project", "digest", "seq")},
            "scoring": (self.base or {}).get("scoring"),
            "baseline": plan.json(),
            "only": sorted(self.only) if self.only is not None else None,
            "points": [p.json() for p in self.points], "not_swept": self.skipped,
            "offered": [self.offered[p.id] for p in self.points if p.id in self.offered],
            "refits": [r.json() for r in refits],
            "decisions": decisions_of(self.points, refits),
            "fit_seconds": seconds_of(timed),
            "complete": not unvisited, "unvisited": unvisited, "unlisted": unlisted,
            "defects": list(self.defects),
        }


# ── the two commands ─────────────────────────────────────────────────────────


def default_build(policy: arms.Default, data: Path) -> Callable[[Any], str]:
    """The default plan, walked in a new project on the session's server (``arms.walk``)."""
    def build(session: Any) -> str:
        pid = session.open(data)
        arms.walk(session, policy)
        return pid

    return build


REPLAYED = "replayed.json"  # the rebuilt plan's project, kept beside the sweep's home


def replayed(bundle: Path, data: Path | None, out: Path) -> str:
    """A named plan, rebuilt with ``core/export/replay.py`` from its log truncated before the
    opening (under Estimate before the lock) in the sweep's home, which the replay needs fresh;
    its project's id, kept in ``replayed.json`` so a second run serves the same project."""
    from turbotab.core.export import replay

    kept = out / REPLAYED
    if kept.exists():
        return str(json.loads(kept.read_text("utf-8"))["project"])
    report = replay.run(bundle, data=str(data) if data else None, home=str(out / "home"),
                        keep=True, cut=True)
    if report.refused or not report.finished or report.project is None:
        raise arms.Stuck(f"The plan was not rebuilt: {report.refused or report.messages}")
    out.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps({"project": report.project, "bundle": str(bundle),
                                "cut": report.cut.model_dump() if report.cut else None}) + "\n",
                    "utf-8")
    return str(report.project)


def write_outputs(out: Path, name: str, report: Mapping[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", "utf-8")
    (out / f"{name}.md").write_text(table_md(report), "utf-8")
    return path


def table_md(report: Mapping[str, Any]) -> str:
    lines = [f"# The sweep of {report.get('dataset')} around the {report.get('plan')} plan", "",
             f"Engine commit `{report.get('commit')}` (engine tree `{report.get('engine_tree')}`"
             f"{', modified' if report.get('engine_modified') else ''}); "
             f"{len(report.get('refits') or [])} options; complete: {report.get('complete')}"
             + (f"; passed: {report['passed']}" if "passed" in report else "")
             + (f"; swept only {', '.join(report['only'])}" if report.get("only") is not None
                else "") + ".", "",
             "| Decision | Options | Scored | Refused | Failed | Unreached | Duplicate | Refused "
             "share | Material |", "|---|---|---|---|---|---|---|---|---|"]
    for d in report.get("decisions") or []:
        c = d["classes"]
        share = "" if d["refused_share"] is None else f"{d['refused_share']:.2f}"
        lines.append(f"| {d['point']} | {d['options']} | {c['scored']} | {c['refused']} | "
                     f"{c['failed']} | {c['unreached']} | {c['duplicate']} | {share} | "
                     f"{d['material']} |")
    if report.get("unlisted"):
        lines += ["", "## Decisions whose options were not listed", ""]
        lines += [f"- `{u['point']}`: {str(u['why'])[:300]}" for u in report["unlisted"]]
    if report.get("defects"):
        lines += ["", "## Engine defects", ""]
        lines += [f"- `{d['refit']}`: {json.dumps(d.get('failed'))[:300]} ({d.get('path')})"
                  for d in report["defects"]]
    return "\n".join(lines) + "\n"


def crash(dataset: str, out: Path, *, only: Iterable[str] | None = None,
          data: Path | None = None, question: arms.Question | None = None,
          log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The crash sweep on ``dataset`` (the module docstring), into ``out``: ``crash.json`` and
    ``crash.md``. A test's small fixture gives ``data`` (a table in place of the one the dataset's
    manifest builds) and ``question`` (in place of its card's); ``only``: the decisions swept, by
    point id."""
    from research.swarm.dry_run import table_of

    out = Path(out).absolute()
    question = question or arms.Question.of_card(arms.card_of(dataset))
    table = Path(data).absolute() if data is not None else table_of(dataset, out / "data")
    sweep = Sweep(out, dataset=dataset, question=question, crash=True, plan="default",
                  only=only, log=log, data=table)
    with arms.Local(out / "home") as session:
        report = sweep.run(session, default_build(sweep.policy, table))
        session.use(str(sweep.base["project"]))
        report["readings"] = graded(session.view(), readings_truth(dataset))
    report["data"] = {"path": str(table), "sha256": hashlib.sha256(table.read_bytes()).hexdigest()}
    report["passed"] = passed(report)
    write_outputs(out, "crash", report)
    return report


def sweep_plan(dataset: str, bundle: Path, name: str, out: Path, *, data: Path | None = None,
               only: Iterable[str] | None = None, question: arms.Question | None = None,
               log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The sweep around the named plan in ``bundle`` (the module docstring), into ``out``:
    ``sweep.json`` and ``sweep.md`` (``question`` in place of the dataset card's, a test's)."""
    out = Path(out).absolute()
    question = question or arms.Question.of_card(arms.card_of(dataset))
    sweep = Sweep(out, dataset=dataset, question=question, crash=False, plan=name, only=only,
                  log=log, data=data)
    pid = sweep.base["project"] if sweep.base else replayed(Path(bundle), data, out)
    with arms.Local(out / "home") as session:
        report = sweep.run(session, lambda s: pid)
        session.use(pid)
        report["readings"] = graded(session.view(), readings_truth(dataset))
    write_outputs(out, "sweep", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.sweep",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("crash", help="the crash sweep around the default plan (SWARM_PLAN §6)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("plan", help="the sweep around a named final plan")
    p.add_argument("--dataset", required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    def log(text: str) -> None:
        sys.stderr.write(f"{now()} {text}\n")
        sys.stderr.flush()

    if args.command == "crash":
        report = crash(args.dataset, args.out, log=log)
        sys.stdout.write(json.dumps({k: report[k] for k in ("dataset", "passed", "complete",
                                                            "fit_seconds")}
                                    | {"defects": len(report["defects"])}) + "\n")
        return 0 if report["passed"] else 1
    report = sweep_plan(args.dataset, args.bundle, args.name, args.out, data=args.data, log=log)
    sys.stdout.write(json.dumps({k: report[k] for k in ("dataset", "plan", "complete")}) + "\n")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The sweep and the crash sweep (SWARM_PLAN §6 "Mechanical arms" and "Order" step 1, §7 "Material, in
the loss's units" and "Refused refits", §8 "The sweep's refits under the full pipeline").

    python -m research.swarm.sweep crash --dataset <id> --out <dir> [--refit-wait <seconds>]
    python -m research.swarm.sweep plan  --dataset <id> --bundle <export.zip> --name <plan> --out <dir>
                                         [--data <file>] [--sweeps <dir>]

**Around a named plan.** The default plan (the default path's, ``research/swarm/arms.py``, walked
here), or a named final plan (an export bundle's decision log rebuilt with ``core/export/replay.py``
truncated before ``open_seal``, under Estimate before ``lock_plan``, in the sweep's home). The plan
is rebuilt once and kept as it stands, its stages computed, before any press of Fit (the
*template*); every refit, the plan's own included, runs on a copy of it (``arms.Local.copy``: the
truncated log byte for byte and the stages it computed), so each starts from the plan with no seal
and no lock, revisits its one decision, presses Fit (under Estimate that press records the refit's
own lock, which no later change could withdraw once its estimate was shown) and is read. A copy is
removed once read, but for one that met an engine defect, which is kept for its reproduction. A
refit waits the run's refit wait from its press of Fit (the crash sweep's ``--refit-wait``,
:data:`REFIT_WAIT` seconds where it names none; the report says the wait each run ran under), for
its fit to be done and then for
every other stage of its copy to end; where that wait runs out, the sweep cancels the copy's fit
through the server's Cancel (``POST …/fit/cancel``: every estimate stage's job stops), waits for
its jobs to end (cancelling each job the copy still holds where it does not go idle), and removes
the copy, so that no refit's work holds the server's workers past its own wait.

**What is swept** (:func:`points_of`): each Router question the plan's log answers in the person's
own words (``recorded_by: you``) but those the card and its table fix (``arms.FIXED``: the
outcome, its event level, the goal, the outcome's kind and its follow-up, the table's layout; under
Estimate the exposure the estimand names), each finding it acted on,
and each declaration line (``quest.DECLARATIONS``, ``research/swarm/declarations.py``) the plan
answered or left open or set for it; the options are the driver's own (``options.compose`` for the
question the revisit opens again, a finding's acts as its line offers them,
``declarations.compose`` for a line), never a hand list; an option that records nothing by
itself (a field of one's own, its ``needs``) is listed, answered by the crash sweep with values
the card states or the sweep builds where it has them (below), and left unreached around a named
plan, but for the adjustment's answers option under Estimate. **The adjustment's answers,
toggled**: around every Estimate plan, the crash sweep's and each named one's, the answers
option of the adjustment question is answered once per column of the plan's answers in force:
every column its adjustment records still in force answer, merged by column as the engine merges
them, for the exposure, each one the question asks (:func:`adjustment_in_force`, read from the
template), so a column the plan answered in an earlier record than its latest is among them,
and each column the option offers (its ``columns``, as the driver composes them: on the copy, the
columns the revisit left unanswered) that the plan's record of the question answers. Each is
the plan's answers in force with that one column toggled, in (a confounder's answers) where the
engine's criterion reads the plan's answers as leaving it out of the primary model and out (a
cause of neither's) where they adjust for it, every other column as the plan's answers in force
have it, all posted together as the decision it composes, so the columns the revisit withdrew
are answered again (:func:`toggled`, ``answers@toggle:<column>:<in|out>``); so a column the roles
do not call a covariate (a confounder or a collider the default roles call an exposure) is swept
too. **Readings with one true
answer** (a column's code or amount, its units: each ``confirm_reading`` but a role's, and each
``set_column_unit``, in force) are graded against the dataset's declared readings, not swept
(:func:`graded`). Under Predict the validation scheme is the sweep's own (below), so its line is
not swept; the crash sweep answers and fits its options all the same.

**The revisit** reverts the decision's record. Where the engine refuses that and names an answer
to withdraw first, the sweep takes the way it names (the orchestrator's ruling of 2026-10-23):
where a recorded answer reads a column the revisit would leave on a reading nobody settled
(``role_unconfirmed``, whose words name that answer: the models answer, the exclusion rules or the
combining answer, :data:`READERS`), that answer; where the split is drawn and the decision changes
what a row is (``sealed``, "Withdraw the split first, then draw it again", :data:`SEALED_REFUSAL`),
the split, by the revert of its record its way forward names. It withdraws that answer first
(reverts its record in force), revisits the decision, answers the option, and answers each answer
withdrawn again as the plan had it, in the order of the plan's log (so where the log answers one
decision twice, the plan's latest answer is again the one in force; a split answered again is the
draw drawn again), then previews and fits as for any option; each answer withdrawn and answered
again is recorded (``withdrawn``, in the order withdrawn, in the listing and the refit). Where that
way is itself refused (the withdrawing revert, the revisit after it, or the answer again), the
refusal is recorded with the engine's reason and the option counts visited and refused; at the
listing, the decision's revisit itself does (:data:`REVISIT`), never unlisted. A refusal that
names no answer to withdraw (another code, or ``sealed`` with no revert among its ways) leaves the
decision unlisted, in the engine's words.

**Each refit** answers the option on its copy (a question, a finding or a reading by its id after
the revisit, ``driver.chosen``; a declaration's option as its line composed it, which is what the
driver's ``answer`` posts for it); any question the change opens again is answered by the default
path's policy, but a refusal the walk meets after the option takes a way forward only where the
plan's log in force before the option, or the option itself, names it (each field of the way's
decision the same, :class:`Named`), recorded with who named it (``ways``): a way the refusal lists
in its answer, or one the driver serves on the asks after it (the last refusal's, which the
default path would settle as a reading or a role); where none is named, the walk goes on as the
default path does with no way forward (another option of the question, else a stop classed
refused); the option's own refusal takes none. A reading or a role the ask card asks itself is no
refusal's way, and the default path's policy answers it. A way the walk answered (a reading, a
role, a finding's act, a refusal's way) and the engine refused is not answered again until the
walk records an answer: nothing recorded, the same answer meets the same refusal. The revisit's own walk, to the question
it opens again, takes a refusal's way under the same rule. Refits are deduplicated by the
plan's digest (``plan_lock.digest`` of the log as served, folded: the digest every fit record
carries), so an option that leaves a plan already fitted is not fitted again (``duplicate_of``);
then Fit is pressed and the refit read: under Predict each family's cross-validated scores from
its fit record, under Estimate the estimate the app reports (``assemble.result_of`` over the log,
the fit records and ``GET …/results``), each beside the q-value its locked fit's exposure row
serves, where it serves one (the feature-wise family's: the measurement fixture's discovery set
is read from it, ``prereg.json``; :func:`with_q`). **Scored**
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
whether the walk met it waiting, it stopped at a limit of its own with a stage of its copy in
error that was not before the option, or the fit was done and an estimate stage was in error
after it); ``unreached``, an option that records nothing by itself
and that has no value of the crash sweep's (around a named plan, every such option but the
adjustment's answers option, toggled under Estimate where the plan answered it), one whose
fields as the crash sweep answered them the driver does not take, a path that stopped at a
limit of the walk's or the revisit's own with no stage failed (its reason in its words: no engine
defect), a fit made whose record the server had not served :data:`RECORD_WAIT` seconds after
it turned fresh (it writes a record after the fit, :func:`fit_record`), or a refit whose wait ran
out, cancelled (its why names what it waited on and for how long, and whether its copy was
removed; ``cancelled``); ``duplicate``. After each fitted refit, its copy idle, every estimate
stage is read, those the fit does not wait on among them (the effects stage, say): an error
there the engine did not type classes the refit failed, an engine defect, and a typed refusal
there classes it refused, with its stage and code.

**Materiality** (§7, :func:`materiality`): under Predict package 1's signed change of each family's
primary score against the plan's (``assemble.signed_change`` over the two fit records, each arm's
n_test/n_train read from the facts of the split its folds were drawn by): paired over the same
folds, or unpaired where the decision changes the analyzed rows; material where its corrected
interval excludes zero and its size is at least half the plan's own interval half-width
(:data:`HALF`); a refit scored as the plan in every one of the same folds is not material, its
change zero though with no spread it has no interval (:func:`unchanged_of`). Under Estimate the
shift of each estimate against the plan's, material where it is
at least half the plan's interval half-width. A decision's refused share and the refusals' codes
sit beside its materiality (:func:`decisions_of`), the sweep's own, kept apart from any analyst's.
Which decisions are material for the pilot's scoring is ``prereg.json``'s to fix and
``score_pilot.py``'s to apply; this table holds the numbers and §7's reading of them.

**The table** (``sweep.json``, and ``sweep.md`` to read): the plan, its scoring, each refit (its
decision, option, class, digest, seconds, scores or estimates, and change), each decision's
summary, each option left unreached with why, and the graded readings: what §7's scoring and §9's
theorist read.

**The compute cap.** The sweeps of a dataset around named plans stop before a fit once its
compute cap is spent: ``prereg.json``'s ``filled.compute_cap.by_dataset`` (the crash fill's, in
fit seconds; the command refuses to sweep without it, :func:`compute_cap`), counting the fit
seconds the dataset's earlier sweeps in the same folder spent (:func:`spent_by`). The report says
so (``stopped``, with the cap and what was spent), and the options left are not visited.

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
what it finished; on a report finished before it answered fields of its own (below), it fits
only the refits that answer them, which supersede each such option's unreached line. It passes
(:func:`passed`) when every decision of the plan was swept (``only``, said in the report, is
none), every option of each was visited (a decision whose options could not be listed, or that
offered none though the plan answered it, was not: :func:`unlisted_of`) and no engine defect was
met, on one engine from its start to its end: **the engine is recorded** (:func:`engine_version`:
the commit, the ``turbotab/`` tree, whether a tracked file under it is modified) as each run of
the sweep starts (``progress.jsonl``' ``engine`` lines, ``engine_starts`` in the report) and as it
ends (the report's ``commit``, ``engine_tree`` and ``engine_modified``), and a report where any
start's tree or modified files are not the end's, or that holds work of a run that recorded no
start, is refused (``engine_changed``, in words, and said in its ``.md``): it does not pass, and
a sweep around a named plan exits as not done (:func:`engine_changed`; a commit that leaves the
tree as it was is the same engine). The report names the table it ran on by its SHA-256, which
the launcher holds to the dataset's manifest, and counts the options it left unreached, each with
why (``unreached``; the summary line of ``crash.md`` beside complete and passed).

**Options that take fields of their own** (:class:`Fields`, the crash sweep only, but for the
adjustment's toggles above, ``toggle``, around every Estimate plan): each is
answered with values the dataset's card states where it states them, else with one value built
from the table or the plan, each value's source in its refit (``answered_with``: ``card`` or
``built``, in words) and the report (``fielded``), and each answer previewed, answered (the
driver's ``answer``, its ids and ``--set`` fields, ``options.resolve``) and fitted once as every
other option, a refit of its own (``<option>@<source>:<what>``), the participant flow it fitted
kept (``flow``). From the card (:func:`card_rules`, its eligibility sentences as the intake writes
them): each range rule as the eligibility range, alone; each level rule as the levels of one's
own, with its refused and don't-know codes, alone under each handling of a row it cannot confirm
that its sentence names (it leaves out such a row unless it keeps them all: both); its whole
population, two rules or more, as one answer of them all in its order, once under each such
handling; and its held-out rows (``held_out``) as the cut-point of one's own, the folds' order
the sweep's scoring's (:data:`FOLD_ORDER`). Built, where the card states none: a range of
plausible values on the first numeric predictor the roles name, from its 1st to its 99th
percentile (:data:`RANGE_PERCENTILES`); an eligibility range on the table's age column from the
engine's adult age (``plausibility.ADULT_AGE``) in the column's unit; a level rule excluding the
rarest level of the first categorical predictor; and an adjustment set of one's own, each
covariate the roles name among the columns the plan's adjustment answers, answered as a
confounder of the declared exposure (:data:`CONFOUNDER`), beside the toggles of every column the
question offers (above). No built value is ever on the outcome
or the declared exposure (:meth:`Fields.fixed`; the plan's adjustment answers neither,
``estimand.asked_covariates``). An option with nothing to answer it from stays unreached, with
why.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
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
FOLD_ORDER = "random"  # the scoring's folds' order where a cut-point seal leaves it free
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
# A refit's wait, in seconds from its press of Fit: for its fit to be done and then, where it is
# scored, for every other stage of its copy to end (the estimate stages the fit does not wait on
# among them, read after it). Where it runs out first, the copy's fit is cancelled, its jobs waited
# for and the copy removed, and the refit left unreached (``Sweep._cancel``): a fit left running
# would hold the server's workers, and every later refit would wait behind it. This is the wait
# where a run names none; the crash sweep's ``--refit-wait SECONDS`` sets it for that run
# (``Sweep.refit_wait``), as for a dataset whose own plan's fit runs past an hour, and the report
# says which wait each run ran under.
# The server writes a fit's record on its recorder's thread after the fit's key turns fresh
# (``service._fit_event``), so a read just after the fit can come before it: it is read again
# every RECORD_POLL seconds, for RECORD_WAIT seconds at most, before the refit is left unreached.
RECORD_WAIT, RECORD_POLL = 120.0, 0.5
PLAN = "plan"  # the refit of the plan itself, every other refit's comparison
REVISIT = "(the revisit)"  # a decision's revisit itself, refitted as an option where it failed
# A revisit the engine refuses because a recorded answer reads a column the revisit would leave on
# a reading nobody settled (``decisions._answers_keep_settled_readings``): its code, the words that
# name that answer ("The recorded <answer> reads `<column>`. Change the <answer> first, …"), and
# the kinds of record each answer it names is. The sweep takes the way those words name (the
# module docstring): that answer withdrawn first, and answered again as the plan had it.
READER_REFUSAL = "role_unconfirmed"
READER_WORDS = re.compile(r"The recorded (?P<what>.+?) reads `(?P<column>[^`]+)`")
READERS = {"models answer": ("select_models",),
           "exclusion rules": ("set_exclusions", "set_sensitivity"),
           "combining answer": ("set_aggregation",)}
# A revisit the engine refuses because the split is drawn and the decision changes what a row is
# (``seal._sealed_refusal``: "Withdraw the split first, then draw it again"): its code, and the
# kind of record its way forward withdraws, a revert of the split's record (``seal._reseal_exit``).
# The sweep takes that way (the module docstring): the split withdrawn first, and drawn again as
# the plan had it.
SEALED_REFUSAL = "sealed"
SPLIT_KINDS = ("set_split",)
WITHDRAWN_AT_MOST = 4  # the most answers one revisit withdraws before the sweep says it stopped


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


ENGINE_KEYS = ("engine_tree", "engine_modified")  # what is the engine: its tree, and whether
# tracked files under it differ from it (the freeze's, ``launcher.crash_passed``); a commit that
# leaves the tree as it was is the same engine.


def engine_words(engine: Mapping[str, Any]) -> str:
    """An engine as the report names it: ``engine tree <tree> (commit <commit>)``, and whether
    tracked files under it were modified."""
    return (f"engine tree {engine.get('engine_tree')} (commit {engine.get('commit')})"
            + (", its tracked files modified" if engine.get("engine_modified") else ""))


def engine_changed(starts: Sequence[Mapping[str, Any]], end: Mapping[str, Any], *,
                   unrecorded: bool = False) -> str | None:
    """Why a sweep's results are not of one engine (the module docstring), or None: a run of it
    (``starts``, each run's engine as it started, the first run's first) started on another
    engine than the one it ended on (``end``, :data:`ENGINE_KEYS`), or it holds work of a run
    that recorded no engine when it started (``unrecorded``), or none is recorded."""
    if unrecorded or not starts:
        return ("The sweep holds work of a run that recorded no engine when it started, so the "
                "engine that work ran on is not known to be the one it ended on: the report is "
                "refused, not passed.")
    for start in starts:
        if any(start.get(k) != end.get(k) for k in ENGINE_KEYS):
            return (f"The engine changed while the sweep ran: a run of it started on "
                    f"{engine_words(start)} at {start.get('at')}, and it ended on "
                    f"{engine_words(end)}: its results are not of one engine, so the report is "
                    f"refused, not passed.")
    return None


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


def adjustment_in_force(view: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The plan's adjustment answers in force on the log ``view`` serves: every live
    ``set_adjustment`` record's answers as the engine folds them (``decisions.fold``: each
    covariate's answer kept under its column, a later record's replacing an earlier's), those
    given for the exposure its estimand names (``estimand.current_answers``), for each column the
    adjustment question asks (``estimand.asked_covariates``, the columns a ``set_adjustment`` may
    answer); in the order the plan first answered them, each without its exposure."""
    from turbotab.core import decisions as D, estimand as E

    state = D.fold(records_of(view))
    asked = set(E.asked_covariates(state))
    return {str(c): a.model_dump(mode="json", exclude={"exposure"})
            for c, a in E.current_answers(state).items() if c in asked}


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


# ── the options that take fields of their own (the crash sweep) ─────────────

# A card's eligibility rule as the intake writes it (``intake.age_line``, ``intake.level_line``):
# "a range rule on <column> from <low>", or "a level rule on <column> excluding level <level>
# (<its meaning>)" naming "<code> as its refused code" and "<code> as its don't-know code" where
# it has them, and saying what a level rule does with a row it cannot confirm in the words of
# :data:`HANDLING_WORDS`.
RULE_ON = (" range rule on ", " level rule on ")
CARD_RANGE = re.compile(r"\ba range rule on (?P<column>[^\s,;]+) from (?P<low>-?\d+(?:\.\d+)?)\b")
CARD_LEVELS = re.compile(r"\ba level rule on (?P<column>[^\s,;]+) excluding level "
                         r"(?P<exclude>[^\s,;()]+) \(")
CARD_REFUSED = re.compile(r"\b(?P<code>[^\s,;]+) as its refused code\b")
CARD_DONT_KNOW = re.compile(r"\b(?P<code>[^\s,;]+) as its don['’]t-know code\b")
HANDLING_WORDS = {"exclude": "leaves out a row it cannot confirm",
                  "keep": "unless it keeps them all"}
RANGE_PERCENTILES = (1, 99)  # a range of one's own built from the table: these percentiles
# A covariate adjusted for, by the adjustment card's three questions (``options.ADJUSTMENT_ASKED``):
# it causes the exposure, it causes the outcome, and it is not after the exposure; and one left
# out, a cause of neither and not after it (``estimand.derive``: the criterion leaves it out). The
# random path draws its adjustment answers from the same two (``arms.CONFOUNDER``).
CONFOUNDER, NEITHER = arms.CONFOUNDER, arms.NEITHER
NOTHING = "It records nothing by itself: "  # an option that takes fields, left unreached
TOGGLE = "toggle"  # the source of an adjustment answer toggled one column at a time (``toggled``)


@dataclass(frozen=True)
class CardRule:
    """One eligibility rule a card states (:func:`card_rules`): its place among the card's
    eligibility sentences (from 1), a range (its low as the card writes it) or a level rule (the
    levels it excludes and its refused and don't-know codes, each as written), the handlings of a
    row it cannot confirm that its sentence names (:data:`HANDLING_WORDS`), and the sentence."""

    place: int
    kind: str  # "range" or "levels"
    column: str
    low: str | None = None
    exclude: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    dont_know: tuple[str, ...] = ()
    handlings: tuple[str, ...] = ()
    words: str = ""


def card_rules(card: Mapping[str, Any] | None) -> tuple[list[CardRule], list[dict[str, Any]]]:
    """The eligibility rules ``card`` states, in its order, read from its sentences as the intake
    writes them (the comment above :data:`RULE_ON`); and each sentence that names a rule
    (:data:`RULE_ON`) but does not read as one, with its place: said, never dropped. A sentence
    that names no rule is the card's prose, not a rule."""
    from research.swarm import options as opts

    rules: list[CardRule] = []
    unread: list[dict[str, Any]] = []
    for place, words in enumerate((card or {}).get("eligibility") or [], 1):
        words = str(words)
        if not any(on in f" {words}" for on in RULE_ON):
            continue
        found = CARD_RANGE.search(words)
        if found:
            rules.append(CardRule(place, "range", found["column"], low=found["low"], words=words))
            continue
        found = CARD_LEVELS.search(words)
        if not found:
            unread.append({"place": place, "words": words})
            continue
        named = tuple(h for h in opts.handlings() if HANDLING_WORDS.get(h, "\0") in words)
        rules.append(CardRule(
            place, "levels", found["column"], exclude=(found["exclude"],),
            refused=tuple(m["code"] for m in CARD_REFUSED.finditer(words)),
            dont_know=tuple(m["code"] for m in CARD_DONT_KNOW.finditer(words)),
            handlings=named, words=words))
    return rules, unread


@dataclass
class Fielded:
    """An option that takes fields of its own, answered: the refit's option (``<option>@<source>:
    <what>``), the options named and their ``--set`` fields as the driver's ``answer`` takes them
    (``options.resolve``), where the values came from (``card``, ``built`` or ``toggle``) and in
    words; an answer composed whole (``decision``: the plan's adjustment answers in force with one
    column toggled, :func:`toggled`) is posted as composed, its ids naming the option it
    answers."""

    option: str
    ids: list[str]
    sets: list[str]
    source: str
    words: str
    decision: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def toggled(point: Point, option: Mapping[str, Any], effect: str = "total",
            in_force: Mapping[str, Any] | None = None) -> list[Fielded]:
    """Around an Estimate plan (the module docstring), the adjustment question's answers option
    answered once per column of the plan's answers: each of ``in_force`` (the plan's answers in
    force, merged over its records, :func:`adjustment_in_force`), then each ``option`` offers
    (its ``columns``, as the driver composes them) that the plan's record of the question
    (``point``'s, its latest) answers, each with its answer in force where it has one, else its
    record's. Each is the plan's answers with that column toggled, in (a confounder's answers,
    :data:`CONFOUNDER`) where the plan's answers leave it out of the primary model by the
    engine's criterion (``estimand.derive`` under the plan's ``effect``) and out (a cause of
    neither's, :data:`NEITHER`) where they adjust for it, every other of those columns answered
    as the plan's answers have it, in that order; none where the plan's record holds no answers.
    Each is posted as composed (:class:`Fielded`'s ``decision``): the engine keeps each answer
    under its column, so the columns the revisit withdrew are answered again with the rest."""
    from turbotab.core.estimand import derive

    if (point.kind, point.key) != ("question", "adjustment"):
        return []
    record = dict((point.decision or {}).get("answers") or {})
    exposure = (point.decision or {}).get("exposure")
    if not record or not exposure:
        return []
    given = {str(c): a for c, a in (in_force or {}).items()}
    offered = [str(c) for c in option.get("columns") or []]
    plan = {c: given[c] if c in given else record[c]
            for c in dict.fromkeys([*given, *offered]) if c in given or c in record}
    out = []
    for column in plan:
        if derive(plan[column], effect).adjusted:
            way, answer = "out", NEITHER
            said = (f"a cause of neither `{exposure}` nor the outcome, and not after "
                    f"`{exposure}`")
        else:
            way, answer = "in", CONFOUNDER
            said = (f"a confounder of `{exposure}` (it causes it, it causes the outcome, and it "
                    f"is not after it)")
        decision = {"kind": "set_adjustment", "exposure": exposure,
                    "answers": {c: (dict(answer) if c == column else dict(a))
                                for c, a in plan.items()}}
        oid = str(option.get("id"))
        out.append(Fielded(f"{oid}@{TOGGLE}:{column}:{way}", [oid], [], TOGGLE,
                           f"the plan's adjustment answers with `{column}` toggled {way}: "
                           f"answered as {said}, every other column as the plan's answers in "
                           f"force have it",
                           decision))
    return out


def rule_answer(rules: Sequence[tuple[CardRule, str | None]]) -> tuple[list[str], list[str]]:
    """The options named and the fields set for ``rules``, each with its handling of a row it
    cannot confirm (None: none set, the option's own initial handling), one rule an option in
    their order: a range as the eligibility range (the cards' rules are who the study is about), a
    level rule as the levels of one's own; each field by its option's place where several are
    named."""
    ids: list[str] = []
    sets: list[str] = []
    for i, (rule, handling) in enumerate(rules, 1):
        at = f"{i}." if len(rules) > 1 else ""
        if rule.kind == "range":
            ids.append("eligibility")
            sets += [f"{at}column={rule.column}", f"{at}low={rule.low}"]
        else:
            ids.append("levels")
            sets += [f"{at}level_column={rule.column}", f"{at}exclude={','.join(rule.exclude)}",
                     *(f"{at}{name}={','.join(codes)}" for name, codes in (
                         ("refused", rule.refused), ("dont_know", rule.dont_know)) if codes)]
        if handling is not None:
            sets.append(f"{at}missing={handling}")
    return ids, sets


def spelled(value: Any) -> str:
    """A level as the driver writes it for the server (``2.0`` as ``2``; the server compares
    levels in one spelling, ``stages.rows._level_key``)."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(value) \
            and float(value).is_integer():
        return str(int(value))
    return str(value)


def ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``4th``, ``11th``, ``99th``."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def table_frame(path: str | Path | None) -> Any:
    """The table at ``path`` as pandas reads its file (a Parquet or a delimited text file), or
    None where there is none or it cannot be read."""
    import pandas as pd

    if path is None:
        return None
    try:
        if Path(path).suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except (OSError, ValueError, ImportError):
        return None


@dataclass
class Fields:
    """What the crash sweep answers an option that takes fields of its own from (the module
    docstring): the dataset's card (its eligibility rules, its held-out cut-point), the table as
    its file holds it (``frame``), and the plan's own answers, read from the template: its roles,
    the columns its readings or the engine read as codes (``codes``), each column's unit
    (``units``), the profile's reading of each column (``profile``: its ``dtype`` and
    ``n_unique``; else read off the table) and, under Estimate, the effect its estimand names
    (``effect``, total unless it names a direct one: what the adjustment's toggles read the plan's
    answers under, :func:`toggled`) and the plan's adjustment answers in force (``adjustment``,
    :func:`adjustment_in_force`: what the toggles toggle)."""

    question: arms.Question
    card: Mapping[str, Any] | None = None
    frame: Any = None
    roles: dict[str, str] = field(default_factory=dict)
    codes: set[str] = field(default_factory=set)
    profile: dict[str, dict[str, Any]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    effect: str = "total"
    adjustment: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def of_template(cls, session: Any, question: arms.Question, dataset: str | None,
                    data: str | None, project: str) -> "Fields":
        """The fields' sources for the plan on ``session``'s template ``project``."""
        session.use(project)
        view = session.view()
        state = view.get("state") or {}
        codes = {str(c) for c in state.get("categorical") or []}
        units: dict[str, str] = {}
        for r in live(view):  # in the log's order: a later reading of a column is the one in force
            body = r.decision.model_dump(mode="json")
            column = str(body.get("column"))
            if r.decision.kind == "confirm_reading" and body.get("reading") == "code_or_count":
                (codes.add if body.get("value") == "code" else codes.discard)(column)
            elif r.decision.kind == "set_column_unit" and body.get("unit"):
                units[column] = str(body["unit"])
        served = (session.read("stages/profile") or {}).get("artifact") or {}
        profile = {str(c["name"]): dict(c) for c in served.get("columns") or []
                   if isinstance(c, Mapping) and c.get("name")}
        card = None
        if dataset and (arms.DATASETS / dataset / "card.json").exists():
            card = arms.card_of(dataset)
        return cls(question, card=card, frame=table_frame(data),
                   roles={str(k): str(v) for k, v in (state.get("roles") or {}).items()},
                   codes=codes, profile=profile, units=units,
                   effect=str((state.get("estimand") or {}).get("effect") or "total"),
                   adjustment=adjustment_in_force(view))

    # the table, as the plan reads it

    def _columns(self) -> list[str]:
        if self.frame is not None:
            return [str(c) for c in self.frame.columns]
        return list(self.profile)

    def _dtype(self, column: str) -> tuple[str | None, int | None]:
        import pandas as pd

        served = self.profile.get(column) or {}
        dtype, n_unique = served.get("dtype"), served.get("n_unique")
        if self.frame is not None and column in self.frame.columns:
            s = self.frame[column]
            if dtype is None:
                dtype = ("boolean" if pd.api.types.is_bool_dtype(s) else
                         "integer" if pd.api.types.is_integer_dtype(s) else
                         "numeric" if pd.api.types.is_numeric_dtype(s) else "text")
            if n_unique is None:
                n_unique = int(s.nunique(dropna=True))
        return dtype, n_unique

    def fixed(self) -> dict[str, str]:
        """The columns no value is built on, each with what it is: the outcome and, under
        Estimate, the declared exposure (the question the card fixes)."""
        out = {self.question.target: "the outcome"}
        if self.question.exposure:
            out.setdefault(self.question.exposure, "the declared exposure")
        return out

    def predictors(self) -> list[str]:
        """The columns the plan's roles name predictors (``decisions.PREDICTOR_ROLES``), in the
        table's order, never the outcome or the declared exposure (:meth:`fixed`)."""
        from turbotab.core.decisions import PREDICTOR_ROLES

        fixed = self.fixed()
        return [c for c in self._columns() if self.roles.get(c) in PREDICTOR_ROLES
                and c not in fixed]

    def categorical(self, column: str) -> bool:
        """Whether the plan reads ``column``'s values as levels: a code by its reading, or a
        column the engine reads as levels (``pipeline.takes_level``: text, categorical or
        boolean, or two values at most)."""
        from turbotab.core.models.pipeline import takes_level

        dtype, n_unique = self._dtype(column)
        return column in self.codes or takes_level(dtype, n_unique)

    def numeric(self, column: str) -> bool:
        from research.swarm.options import NUMERIC

        return self._dtype(column)[0] in NUMERIC and not self.categorical(column)

    def _no_table(self) -> str:
        return " (the crash sweep read no table)" if self.frame is None else ""

    # the values built from the table

    def built_range(self) -> Fielded | None:
        """A range of plausible values of one's own: the first numeric predictor's 1st and 99th
        percentiles (:data:`RANGE_PERCENTILES`; numpy's, by linear interpolation) over its
        recorded values."""
        import numpy as np
        import pandas as pd

        if self.frame is None:
            return None
        for c in self.predictors():
            if not self.numeric(c):
                continue
            values = pd.to_numeric(self.frame[c], errors="coerce").dropna().to_numpy(float)
            if not values.size:
                continue
            low, high = (float(v) for v in np.percentile(values, RANGE_PERCENTILES))
            first, last = (ordinal(p) for p in RANGE_PERCENTILES)
            return Fielded(f"custom@built:{c}", ["custom"],
                           [f"column={c}", f"low={low!r}", f"high={high!r}"], "built",
                           f"built from the table: `{c}`, the first numeric predictor the roles "
                           f"name, from its {first} percentile, {low:g}, to its {last}, "
                           f"{high:g}, over its {values.size:,} recorded values")
        return None

    def built_floor(self) -> Fielded | None:
        """A range of who the study is about of one's own: the table's age column (the engine's
        reading of its name, ``plausibility.age_reading``, in the unit the plan records for it,
        else the one its name spells), among its columns but the outcome and the declared
        exposure (:meth:`fixed`), from the engine's adult age (``plausibility.ADULT_AGE``, in
        years), in that unit."""
        from types import SimpleNamespace

        from turbotab.core.detectors import plausibility as P

        if self.frame is None:
            return None
        fixed = self.fixed()
        others = self.frame[[c for c in self.frame.columns if str(c) not in fixed]]
        reading = P.age_reading(others, {c: SimpleNamespace(unit=u)
                                         for c, u in self.units.items()})
        if reading is None or reading["unit"] not in P.MONTHS_PER:
            return None
        c, unit = reading["column"], reading["unit"]
        floor = P.ADULT_AGE * P.MONTHS_PER["years"] / P.MONTHS_PER[unit]
        return Fielded(f"eligibility@built:{c}", ["eligibility"], [f"column={c}", f"low={floor!r}"],
                       "built", f"built from the table: `{c}`, the table's age column read in "
                                f"{unit}, from the engine's adult age, {P.ADULT_AGE:g} years"
                                + (f" ({floor:g} {unit})" if unit != "years" else ""))

    def fixed_ages(self) -> str:
        """The outcome and the declared exposure that the engine reads as an age column
        (``plausibility.age_reading``), each by its name and what it is, for the words of an
        eligibility range left unreached: `` but `age`, the declared exposure, on which no value
        is built``; else nothing."""
        from turbotab.core.detectors import plausibility as P

        if self.frame is None:
            return ""
        ages = [f"`{c}`, {what}" for c, what in self.fixed().items()
                if c in self.frame.columns and P.age_reading(self.frame[[c]]) is not None]
        return f" but {' and '.join(ages)}, on which no value is built" if ages else ""

    def built_level(self) -> Fielded | None:
        """Levels of one's own: the rarest level of the first categorical predictor with two
        levels or more (a tie to the level first in its spelling's order), its blanks left out as
        the option's own handling does."""
        if self.frame is None:
            return None
        for c in self.predictors():
            if not self.categorical(c):
                continue
            counts: dict[str, int] = {}
            for v in self.frame[c].dropna():
                counts[spelled(v)] = counts.get(spelled(v), 0) + 1
            counts = {k: n for k, n in counts.items() if "," not in k}  # a list is by commas
            if len(counts) < 2:
                continue
            rarest, n = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
            return Fielded(f"levels@built:{c}", ["levels"], [f"level_column={c}",
                                                            f"exclude={rarest}"], "built",
                           f"built from the table: `{c}`, the first categorical predictor the "
                           f"roles name, its rarest level `{rarest}` ({n:,} of its "
                           f"{sum(counts.values()):,} recorded values) excluded")
        return None

    def built_adjustment(self, point: Point) -> Fielded | None:
        """An adjustment set of one's own: the covariates the plan's roles name among the columns
        its adjustment answers, each answered as a confounder of the declared exposure
        (:data:`CONFOUNDER`), and every other column it answers as a cause of neither
        (:data:`NEITHER`), so that the set is those covariates and the question asks no more
        (``estimand.adjustment_answer`` waits for every covariate's answers); in the plan's
        order."""
        from research.swarm import options as opts

        answered = list(((point.decision or {}).get("answers") or {}).keys())
        covariates = [c for c in answered if self.roles.get(c) == "covariate"]
        if not covariates:
            return None
        said = {name: ",".join(answers[k] for k in opts.ADJUSTMENT_ASKED)
                for name, answers in (("in", CONFOUNDER), ("out", NEITHER))}
        others = [c for c in answered if c not in covariates]
        exposure = (point.decision or {}).get("exposure") or self.question.exposure
        return Fielded("answers@built:covariates", ["answers"],
                       [f"{c}={said['in' if c in covariates else 'out']}" for c in answered],
                       "built", f"built from the roles: {', '.join(f'`{c}`' for c in covariates)}, "
                                f"the covariates the roles name, each answered as a confounder "
                                f"of `{exposure}` (it causes it, it causes the outcome, and it "
                                f"is not after it)"
                                + (f"; {', '.join(f'`{c}`' for c in others)} a cause of neither"
                                   if others else ""))

    # the values the card states

    def card_cut(self) -> Fielded | None:
        """A cut-point of one's own: the card's held-out rows (``held_out``: every row at or after
        its value of its column), the folds' order random, the sweep's own scoring's (the module
        docstring)."""
        held = (self.card or {}).get("held_out") or {}
        column, at = held.get("column"), held.get("at")
        if not column or at is None or isinstance(at, bool):
            return None
        written = str(at) if not isinstance(at, float) else repr(at)
        return Fielded(f"cut@card:{column}", ["cut"], [f"cut_column={column}", f"cut_at={written}",
                                                       f"fold_order={FOLD_ORDER}"], "card",
                       f"the card's held-out rows: every row at or after {written} on `{column}`; "
                       f"the folds' order {FOLD_ORDER}, the sweep's own scoring's")

    def card_answers(self, kind: str) -> list[Fielded]:
        """The card's rules of ``kind``, each answered alone: a range at the option's own handling
        of a row it cannot confirm, a level rule once under each handling its sentence names (else
        the option's own)."""
        rules, _ = card_rules(self.card)
        out = []
        for rule in rules:
            if rule.kind != kind:
                continue
            for handling in rule.handlings or (None,):
                ids, sets = rule_answer([(rule, handling)])
                tag = f"card:{rule.place}:{rule.column}" + (f":{handling}" if handling else "")
                out.append(Fielded(f"{ids[0]}@{tag}", ids, sets, "card",
                                   f"the card's eligibility rule {rule.place}: “{rule.words}”"
                                   + (f" A row it cannot confirm: {handling}." if handling
                                      else "")))
        return out

    def population(self, point: Point, offered: Sequence[str]) -> list[Fielded]:
        """The card's whole population as one answer of its rules in its order (the exclusions
        question, where the card states two rules or more and the options they answer are
        offered), once under each handling of a row it cannot confirm that every level rule's
        sentence names, each range at the option's own."""
        rules, _ = card_rules(self.card)
        if (point.kind, point.key) != ("question", "exclusions") or len(rules) < 2:
            return []
        levels = [r for r in rules if r.kind == "levels"]
        handlings = [h for h in (levels[0].handlings if levels else (None,))
                     if all(h in r.handlings for r in levels)] or [None]
        out = []
        for handling in handlings:
            ids, sets = rule_answer([(r, handling if r.kind == "levels" else None) for r in rules])
            if not set(ids) <= set(offered):
                return []
            out.append(Fielded(f"population@card" + (f":{handling}" if handling else ""), ids, sets,
                               "card", f"the card's whole population, its {len(rules)} rules "
                                       f"(its eligibility sentences "
                                       f"{', '.join(str(r.place) for r in rules)}) as one answer"
                               + (f", each level rule's row it cannot confirm: {handling}"
                                  if handling else "")))
        return out

    def answers(self, point: Point, option: Mapping[str, Any]) -> tuple[list[Fielded], str | None]:
        """``option`` of ``point``, which takes fields of its own, answered: with the values the
        card states where it states them, else with one built from the table or the plan (the
        module docstring); none, with why, where nothing is stated or can be built."""
        oid = str(option.get("id"))
        where = (point.kind, point.key, oid)
        none = self._no_table()
        if where == ("question", "exclusions", "custom"):
            got = self.built_range()
            return ([got], None) if got else ([], f"{NOTHING}it takes a range's column and "
                                                  f"bounds, and the roles name no numeric "
                                                  f"predictor to build one on{none}.")
        if where == ("question", "exclusions", "eligibility"):
            stated = self.card_answers("range")
            got = stated or [g for g in [self.built_floor()] if g]
            return (got, None) if got else ([], f"{NOTHING}it takes a range's column and bounds; "
                                                f"the card states no range rule, and the table "
                                                f"has no age column to build an adult floor "
                                                f"on{self.fixed_ages()}{none}.")
        if where == ("question", "exclusions", "levels"):
            stated = self.card_answers("levels")
            got = stated or [g for g in [self.built_level()] if g]
            return (got, None) if got else ([], f"{NOTHING}it takes a column and the levels it "
                                                f"excludes; the card states no level rule, and "
                                                f"the roles name no categorical predictor to "
                                                f"build one on{none}.")
        if where == ("question", "adjustment", "answers"):
            got = self.built_adjustment(point)
            return ([got], None) if got else ([], f"{NOTHING}it takes each covariate's answers, "
                                                  f"and the roles name no covariate among the "
                                                  f"columns the plan's adjustment answers.")
        if where == ("question", "split", "cut"):
            got = self.card_cut()
            return ([got], None) if got else ([], f"{NOTHING}it takes a cut-point's column and "
                                                  f"value; the card states no held-out "
                                                  f"cut-point, and the crash sweep builds none "
                                                  f"from the table.")
        return [], (f"{NOTHING}it takes {needs_of(option)}, and the crash sweep builds no value "
                    f"for it.")

    def card_json(self) -> dict[str, Any]:
        """The card's eligibility rules the sweep read (each by its place, kind and column) and the
        sentences naming a rule it could not read, for the report."""
        rules, unread = card_rules(self.card)
        return {"read": [{"place": r.place, "kind": r.kind, "column": r.column} for r in rules],
                "not_read": unread}


def takes_fields(option: Mapping[str, Any]) -> bool:
    """Whether ``option`` records nothing by itself: it takes a field of its own (its ``needs``,
    or no decision)."""
    return bool(option.get("needs")) or not isinstance(option.get("decision"), Mapping)


def needs_of(option: Mapping[str, Any]) -> str:
    return ", ".join(option.get("needs") or []) or "a field of its own"


def flow_of(session: Any) -> list[dict[str, Any]] | None:
    """The participant flow the session's project serves while it is fresh at its current key:
    each step's key, rows left and rows it dropped; else None."""
    from research.swarm import driver

    try:
        key = ((session.view().get("stages") or {}).get("cohort") or {}).get("key")
        served = session.read("stages/cohort") or {}
    except driver.DriverError:
        return None
    if not (served.get("fresh") and key and served.get("key") == key):
        return None
    return [{k: s.get(k) for k in ("key", "n", "dropped")}
            for s in (served.get("artifact") or {}).get("steps") or [] if isinstance(s, Mapping)]


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
    material; the previews and answers made on the way, the plan's answers the revisit withdrew
    first (``withdrawn``, in the order withdrawn) and answered again after the option in the
    plan's order, and each refusal's way forward the revisit and the walk took with what named it
    (``ways``); the copy it ran on, while kept. An option that takes fields of its own, answered by
    the crash sweep (:class:`Fields`), keeps what it was answered with and where each value came
    from (``answered_with``, :class:`Fielded`), the decision the driver resolved it to (from its
    preview), and, where its answer was recorded, the participant flow of the plan it then fitted
    (``flow``: each step's rows). A refit whose wait ran out keeps what the sweep cancelled
    (``cancelled``: the wait, the stages still computing then, and ``arms.Local.cancel``'s
    account)."""

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
    withdrawn: list[dict[str, Any]] = field(default_factory=list)  # answered again after the option
    ways: list[dict[str, Any]] = field(default_factory=list)  # refusals' ways forward taken
    record: dict[str, Any] | None = None  # the fit record as served (Predict: the change's arm)
    answered_with: dict[str, Any] | None = None  # an option's own fields, and their source
    flow: list[dict[str, Any]] | None = None  # the participant flow, for such an option
    cancelled: dict[str, Any] | None = None  # where its wait ran out: what the sweep cancelled

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
    where no fit was made (it is blocked, or held, or not done when its wait ran out, which
    :meth:`Sweep._fit` then cancels); else ``refused`` in words, by the engine at the
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
    """The latest fit record ``GET …/fits`` serves for the fit ``key``, read again every
    :data:`RECORD_POLL` seconds until it is served, for :data:`RECORD_WAIT` seconds at most (the
    server writes it after the fit turns fresh); None where none is served by then."""
    if not key:
        return None
    end = time.monotonic() + RECORD_WAIT
    while True:
        fits = session.read("fits") or {}
        found = [r for r in fits.get("records") or [] if r.get("fit_key") == key]
        if found:
            return found[-1]
        if time.monotonic() >= end:
            return None
        time.sleep(RECORD_POLL)


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
    there is none; each estimate read from the locked fit's exposure rows beside its q-value where
    the row serves one (:func:`with_q`)."""
    from research.swarm import assemble as A

    view = session.view()
    fits = (session.read("fits") or {}).get("records") or []
    r = session.client.get(f"/projects/{session.pid}/results")
    results = A.Read(r.status == 200, r.json() if r.status == 200 else None)
    got = A.result_of("inference", view.get("decisions") or [], fits, results)
    if got.get("refused"):
        return None, str(got["refused"])
    return with_q(got, fits), None


def with_q(got: Mapping[str, Any], fits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``got``'s estimates (``assemble.result_of``), each read from the locked fit's exposure rows
    (``source`` ``fits``) given that row's q-value (``q``, the feature-wise family's
    Benjamini–Hochberg q, as ``GET …/fits`` serves it) where the row serves one, so a discovery set
    can be read from the refit (the measurement fixture's materiality, ``prereg.json``); a row that
    serves none, and an estimate read from the results, are as ``got`` holds them."""
    rows = [dict(e) for e in got.get("estimates") or []]
    if got.get("source") != "fits":
        return rows
    locked = next((f for f in fits if f.get("id") == got.get("attempt")), None) or {}
    q = {(str(family), str(row.get("feature"))): row["q"]
         for family, served in (locked.get("effects") or {}).items() for row in served or ()
         if isinstance(row, Mapping) and "q" in row}
    for e in rows:
        key = (str(e.get("family")), str(e.get("feature")))
        if key in q:
            e["q"] = q[key]
    return rows


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
        refit.kind = "unreached"
        refit.why = (f"No fit record holds the fit's key: none was served in the {RECORD_WAIT:g} s "
                     f"the sweep waited after the fit turned fresh.")
        return
    refit.fit_record, refit.fit_seconds = record.get("id"), record.get("seconds")
    if purpose == "prediction":
        refit.record = record
        refit.scores = {"metric": record.get("primary_metric"), "folds": record.get("folds"),
                        "families": record.get("scores") or {}}
        refit.split = split_facts(session)
    else:
        refit.estimates, refit.estimate_why = estimate_of(session)


def in_flight(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each stage of ``view`` queued or running, in the graph's order, with its status."""
    return [{"stage": name, "status": (status or {}).get("status")}
            for name, status in (view.get("stages") or {}).items()
            if (status or {}).get("status") in arms.Local.BUSY]


def joined(names: Iterable[Any]) -> str:
    return ", ".join(str(n) for n in names)


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


def material_predict(change: Mapping[str, Any], unchanged: bool = False) -> bool | None:
    """§7 under Predict: the corrected interval excludes zero and the change is at least half the
    plan's own interval half-width (:data:`HALF`); not material where the refit scored as the plan
    on every one of the same folds (``unchanged``, :func:`unchanged_of`: the change is zero, though
    with no spread it has no interval); None where the change has no number otherwise."""
    diff, half = change.get("difference"), change.get("half_width")
    if diff is None and unchanged:
        return False
    if diff is None or half is None or change.get("excludes_zero") is None:
        return None
    return bool(change["excludes_zero"]) and abs(float(diff)) >= HALF * float(half)


def fold_scores_of(families: Mapping[str, Any] | None, family: Any, metric: Any) -> Any:
    """A family's fold scores of ``metric`` in a fit's scores by family (the fit record's
    ``scores``, which ``Refit.scores`` keeps as its ``families``), or None."""
    return ((((families or {}).get(family) or {}).get(metric)) or {}).get("folds")


def unchanged_of(change: Mapping[str, Any], after: Mapping[str, Any] | None,
                 before: Mapping[str, Any] | None) -> bool:
    """Whether a refit whose ``change`` has no number scored as the plan in every fold: paired
    (the same folds, one digest) and every fold's score the plan's but for float64's rounding
    (``signed.same_scores``), its family's and metric's scores read from ``after`` (the refit's
    scores by family) and ``before`` (the plan's)."""
    from research.swarm.signed import same_scores

    if change.get("difference") is not None or not change.get("paired"):
        return False
    family, metric = change.get("family"), change.get("metric")
    return same_scores(fold_scores_of(after, family, metric),
                       fold_scores_of(before, family, metric))


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
            c["material"] = material_predict(c, unchanged_of(c, refit.record.get("scores"),
                                                             plan.record.get("scores")))
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
    duplicate is the plan it duplicates, and an option left unreached (one that records nothing by
    itself with nothing to answer its fields, or whose path stopped at a limit of the walk's own)
    was not refitted."""
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
    least that answer to offer). A decision whose revisit took the way the engine names and was
    refused on it (``refused``, :meth:`Sweep._revisit`) was visited: its revisit is a refit,
    refused, with the engine's reason."""
    out = []
    for p in points:
        listed = offered.get(p.id)
        if listed is None:
            out.append({"point": p.id, "why": "Its options were not listed."})
        elif listed.get("why"):
            out.append({"point": p.id, "why": str(listed["why"])})
        elif listed.get("refused"):
            continue
        elif not listed.get("options") and p.decision is not None:
            out.append({"point": p.id, "why": "No option was offered for a decision the plan "
                                              "answered."})
    return out


def passed(report: Mapping[str, Any]) -> bool:
    """Whether the crash sweep passed (the module docstring): every decision of the plan swept
    (no ``only``), every option visited, no engine defect met, on one engine from its start to its
    end (``engine_changed`` None, the report saying so)."""
    return bool(report.get("complete") and not report.get("defects")
                and "only" in report and report["only"] is None
                and "engine_changed" in report and report["engine_changed"] is None)


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


def names_way(decision: Mapping[str, Any], way: Mapping[str, Any]) -> bool:
    """Whether ``decision`` names the way forward whose decision is ``way``: every field the way
    records, its kind among them, is the decision's own."""
    return bool(way) and all(decision.get(k) == v for k, v in way.items())


def refused_since_recorded(steps: Sequence[Any] | None) -> set[str]:
    """The ways the walk answered (``arms.Step`` at ``way``: a reading, a role, a finding's act or
    a refusal's way forward) that the engine refused since the walk's last answer it recorded:
    with nothing recorded since, the same answer meets the same refusal again."""
    out: set[str] = set()
    for step in reversed(list(steps or [])):
        if step.said == "ok":
            break
        if step.at == "way" and step.said == "refused":
            out.update(str(c) for c in step.chosen)
    return out


def without_ways(q: Mapping[str, Any], ids: set[str]) -> dict[str, Any]:
    """The ask ``q`` with the ways forward whose ids are ``ids`` left out, wherever it lists one
    (``arms.ways_of``: the ask card's, the last refusal's, a quest line's and its items', and a
    reading read from the data's), and each finding line's acts."""
    def kept(holder: Mapping[str, Any], key: str) -> dict[str, Any]:
        if holder.get(key) is None:
            return dict(holder)
        return {**holder, key: [w for w in holder[key] if str((w or {}).get("id")) not in ids]}

    out = kept(q, "last_refusal_exits")
    if isinstance(q.get("readings"), Mapping):
        readings = kept(q["readings"], "exits")
        if readings.get("read_from_data") is not None:
            readings["read_from_data"] = [kept(s, "change") for s in readings["read_from_data"]]
        out["readings"] = readings
    if q.get("also_open") is not None:
        lines = [kept(line, "options") for line in q["also_open"]]
        for line in lines:
            if line.get("items") is not None:
                line["items"] = [kept(item, "options") for item in line["items"]]
        out["also_open"] = lines
    return out


@dataclass
class Named(arms.Default):
    """The default path's answers for the sweep's walks (the module docstring), but a refusal's
    way forward taken only where the plan or the option names it (:func:`names_way`): a way whose
    decision is one the plan's log holds in force before the option, or the option's own
    (``names``, by who names it); none other, so a refusal the plan never met is not answered by a
    way it never chose. That holds for the ways a refusal lists in its answer (:meth:`way`) and for
    those the driver serves on the asks after it (``last_refusal_exits``), which the default path
    settles as a reading or a role (:meth:`settle`); a reading or a role the ask card asks itself
    is no refusal's way and is settled as the default path settles it. Each way taken is kept in
    ``taken``, with who named it. A way the walk answered and the engine refused is not answered
    again while the walk has recorded nothing since (``steps``, the walk's own: the same answers
    meet the same refusal, :func:`refused_since_recorded`), so a refused act is not answered over
    and over until the walk's limit (the plasma panel's ``question:lens=metabolomics``, whose
    walk answered one refused repair 298 times)."""

    names: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    taken: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Any] | None = None  # the walk's steps (``arms.walk``'s), where it is given them

    def named_by(self, e: Mapping[str, Any]) -> str | None:
        """Who names the way forward ``e`` (the plan or the option), or None."""
        decision = e.get("decision")
        if not (e.get("id") and isinstance(decision, Mapping)):
            return None
        return next((who for who, decisions in self.names.items()
                     if any(names_way(d, decision) for d in decisions)), None)

    def _take(self, e: Mapping[str, Any], by: str) -> str:
        self.taken.append({"id": str(e["id"]), "label": e.get("label"),
                           "decision": dict(e["decision"]), "named_by": by})
        return str(e["id"])

    def way(self, exits: Sequence[Mapping[str, Any]]) -> str | None:
        for e in exits:
            by = self.named_by(e)
            if by is not None:
                return self._take(e, by)
        return None

    def settle(self, q: Mapping[str, Any]) -> tuple[str, str] | None:
        """The default path's settling of ``q`` with the last refusal's ways forward that neither
        the plan nor the option names left out, and each way refused since the walk last recorded
        an answer."""
        futile = refused_since_recorded(self.steps)
        if futile:
            q = without_ways(q, futile)
        kept = [e for e in q.get("last_refusal_exits") or [] if self.named_by(e) is not None]
        return super().settle({**q, "last_refusal_exits": kept})

    def settled(self, q: Mapping[str, Any], way: str) -> None:
        """``way``, which :meth:`settle` gave on ``q`` and the walk answered, kept in ``taken``
        where it is a way the last refusal left (``last_refusal_exits``) that the plan or the
        option names; a reading or a role the ask card asks itself is not kept."""
        e = next((e for e in q.get("last_refusal_exits") or [] if str(e.get("id")) == way), None)
        by = None if e is None else self.named_by(e)
        if by is not None:
            self._take(e, by)


class Sweep:
    """One sweep around one plan in ``out`` (the module docstring), resumable from its
    ``progress.jsonl``. ``crash``: the crash sweep's mode (previews, readings, the scheme).
    ``only``: the decisions swept, by point id (every one where None), said in the report.
    ``data``: the table the plan was made from, named in each defect's reproduction. ``cap``: the
    dataset's compute cap in fit seconds (prereg.json's, for the sweeps around named plans), with
    ``spent`` the fit seconds the dataset's earlier sweeps spent; the sweep stops before a fit once
    they are spent, and says so (``stopped``). ``refit_wait``: each refit's wait in seconds from
    its press of Fit (:data:`REFIT_WAIT` where None), held for the run and said in its report."""

    def __init__(self, out: Path, *, dataset: str | None, question: arms.Question,
                 crash: bool, plan: str, only: Iterable[str] | None = None,
                 log: Callable[[str], None] | None = None, data: Path | None = None,
                 cap: float | None = None, spent: float = 0.0,
                 refit_wait: float | None = None):
        self.out = Path(out).absolute()
        self.refit_wait = float(REFIT_WAIT if refit_wait is None else refit_wait)
        self.cap, self.spent_before = (None if cap is None else float(cap)), float(spent)
        self.stopped: dict[str, Any] | None = None
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
        # The crash sweep's sources for an option's fields of its own (:class:`Fields`), read from
        # the template once one is met; and each decision's refits this run plans, by id.
        self.fields: Fields | None = None
        self.planned: dict[str, list[str]] = {}
        # The engine each run of the sweep started on (``progress.jsonl``' ``engine`` lines, the
        # first run's first), and whether the progress holds work of a run that recorded none.
        self.engines: list[dict[str, Any]] = []
        self.unrecorded = False
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
            if "engine" in entry:
                self.engines.append(dict(entry["engine"]))
                continue
            if not self.engines and any(k in entry for k in ("template", "refit", "offered",
                                                             "defect")):
                self.unrecorded = True  # work of a run that recorded no engine as it started
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
                         "skipped": skipped, "dataset": self.dataset, "plan": self.plan,
                         "crash": self.crash}
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
            body["fold_order"] = FOLD_ORDER
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
        """The plan's own refit, on a copy of the template: Fit pressed and read as every refit's
        is (:meth:`_fit`). A plan not scored stops the sweep, in words; one that failed is first
        reported as an engine defect with its reproduction."""
        if PLAN in self.done:
            return self.done[PLAN]  # its fit record kept with it (``Refit.record``)
        plan = Refit(PLAN, None, None)
        copy = self._copy(session)
        plan.project = copy
        self._fit(session, plan)
        if plan.kind != "scored":
            if plan.kind == "failed":
                self._defect(session, plan, [])
            raise arms.Stuck(f"The plan itself was not scored: {plan.kind}: "
                             f"{json.dumps(plan.refused or plan.failed or plan.why)[:600]}")
        plan.digest = plan_digest(session.view())
        self.done[PLAN] = plan
        self._write({"refit": plan.json()})
        return plan

    def _fit(self, session: Any, refit: Refit) -> None:
        """Fit pressed on the refit's copy and read (:func:`read_refit`) within the refit's wait,
        the sweep's :attr:`refit_wait` seconds from the press: for the fit to be done and, where
        it is scored, for every other stage of the copy to end, after which its estimate stages
        are read (:meth:`_estimate_stages`). Where the wait runs out first, the copy's fit is
        cancelled and the refit left unreached (:meth:`_cancel`)."""
        started = time.monotonic()
        fit = session.fit(self.refit_wait)
        read_refit(session, refit, fit, self.question.purpose)
        if arms.ran_out(fit):  # the arms press such a fit again; the sweep cancels it
            self._cancel(session, refit, (fit.payload.get("fit") or {}).get("status"))
            return
        if refit.kind != "scored":
            return
        if not session.idle(wait=max(0.0, self.refit_wait - (time.monotonic() - started))):
            self._cancel(session, refit, None)
            return
        self._estimate_stages(session, refit)

    def _cancel(self, session: Any, refit: Refit, fit: str | None) -> None:
        """A refit whose wait ran out (:meth:`_fit`), its fit's status then ``fit`` (None where the
        fit was done and its copy's other stages were not): the copy's fit cancelled through the
        server's Cancel and the copy waited for, each job it still holds then cancelled too
        (``arms.Local.cancel``), and the copy removed once none of its stages is computing. The
        refit is unreached, its why naming what it waited on and for how long, what was cancelled
        and whether the copy was removed; what was cancelled is kept (``cancelled``). A 500 met on
        the cancel keeps the copy, for the defect the refit is then classed by."""
        busy = in_flight(session.view())
        errors = len(session.errors)
        done = session.cancel(str(refit.project))
        refit.kind, refit.cancelled = "unreached", {"after": self.refit_wait, "waiting_on": busy,
                                                    **done}
        computing = (" (still computing: " + joined(f"{b['stage']} {b['status']}" for b in busy)
                     + ")" if busy else "")
        waited = (f"{self.refit_wait:g} s after the sweep pressed Fit, as long as it waits for a "
                  f"refit")
        why = (f"The fit was {fit} {waited}{computing}." if fit is not None else
               f"The fit was done, but {waited}, its other stages were not{computing}.")
        why += " The sweep cancelled the copy's fit"
        if done["jobs"]:
            why += (f"; its stages still computing {done['wait']:g} s later, it cancelled their "
                    f"jobs too ({joined(j['stage'] for j in done['jobs'])})")
        if len(session.errors) > errors:
            why += (f"; the server failed on the cancel (HTTP {session.errors[-1].get('status')}), "
                    f"and the copy is kept.")
        elif not done["idle"]:
            why += "; the copy was still computing and is kept."
        elif session.drop(str(refit.project), self.made):
            refit.project = None
            why += "; its jobs ended and the copy was removed."
        else:
            why += "; its jobs ended, but the copy could not be removed and is kept."
        refit.why = why

    def _estimate_stages(self, session: Any, refit: Refit) -> None:
        """A fitted refit's estimate stages read once its copy is idle (:meth:`_fit`), those the
        fit does not wait on among them: where any is in error, the refit is classed by them as
        the walk's stages are (``arms.met_stages`` over ``driver.failures``): failed where any is
        an error the engine did not type, an engine defect; else refused, with the first typed
        refusal's stage, code, words and ways forward."""
        from research.swarm import driver
        from turbotab.core import estimand

        view = session.view()
        names = [n for n in view.get("stages") or {} if n in estimand.ESTIMATE_STAGES]
        failures = driver.failures(view, names)
        if failures:
            stages = joined(f["stage"] for f in failures)
            self._met(refit, arms.met_stages(failures, (
                f"The fit was done, but after it an estimate stage was in error: {stages}."
                if len(failures) == 1 else
                f"The fit was done, but after it estimate stages were in error: {stages}.")))

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
        return self._listed(session, point, self._named(session))[0]

    def _named(self, session: Any, option: Mapping[str, Any] | None = None) -> Named:
        """The sweep's policy on the session's project as it stands (the plan's log, before the
        revisit and the option): a refusal's way forward taken only where that log or ``option``
        names it (:class:`Named`)."""
        names = {"the plan": [r.decision.model_dump(mode="json") for r in live(session.view())]}
        if option is not None:
            names["the option"] = [dict(option.get("decision") or {})]
        return Named(self.question, proposed=dict(self.policy.proposed), names=names)

    def _listed(self, session: Any, point: Point, policy: Named
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """:meth:`options`, and the plan's answers the revisit withdrew first (:meth:`_revisit`);
        a reading or a role asked on the way settled by ``policy``."""
        if point.kind == "declaration":
            from research.swarm import declarations as decl, driver

            composed = decl.compose(point.key, session.view(),
                                    driver.served(session.run, session.client))
            return list(composed.get("options") or []), []
        withdrawn = self._revisit(session, point)
        q = self._reopened(session, point, policy)
        if point.kind == "question":
            return list(q.get("options") or []), withdrawn
        if point.kind == "finding":
            line = next((ln for ln in q.get("also_open") or [] if ln.get("source") == "finding"
                         and str(ln.get("key")) == point.key), {})
            return list(line.get("options") or []), withdrawn
        reading, column = point.key.split(":", 1)
        if reading == "unit":
            return [w for w in arms.ways_of(q) if w.get("id") and (w.get("decision") or {}).get(
                "kind") == "set_column_unit" and (w.get("decision") or {}).get(
                "column") == column], withdrawn
        return list(arms.reading_ways(q).get((reading, column)) or []), withdrawn

    @staticmethod
    def _withdrawal(session: Any, point: Point, said: Any
                    ) -> tuple[str, str | None, Any] | str | None:
        """The answer the engine's refusal ``said`` of ``point``'s revisit names to withdraw first
        (the module docstring), as ``(what, column, record)``: the latest record in force of the
        answer whose words name the column it reads (``role_unconfirmed``, :data:`READERS`), or
        the split's record the way forward reverts (``sealed``, :data:`SPLIT_KINDS`); words where
        the refusal names one the log does not hold in force; None where it names none."""
        message = str(said.message or "")
        if said.kind != "refused":
            return None
        if said.code == READER_REFUSAL:
            named = READER_WORDS.search(message)
            kinds = READERS.get(named.group("what")) if named else None
            if kinds is None:
                return None
            what = named.group("what")
            reader = next((r for r in reversed(live(session.view())) if r.decision.kind in kinds
                           and r.id != point.record), None)
            if reader is None:
                return f"The log holds no record of the {what} in force."
            return what, named.group("column"), reader
        if said.code == SEALED_REFUSAL:
            ways = [str((e.get("decision") or {}).get("decision_id")) for e in said.exits
                    if (e.get("decision") or {}).get("kind") == "revert"
                    and (e.get("decision") or {}).get("decision_id")]
            if not ways:
                return None
            split = next((r for r in live(session.view()) if r.id in ways
                          and r.decision.kind in SPLIT_KINDS and r.id != point.record), None)
            if split is None:
                return (f"Its way forward withdraws record {ways[0]}, which the log does not "
                        f"hold in force as a split.")
            return "split", None, split
        return None

    def _revisit(self, session: Any, point: Point) -> list[dict[str, Any]]:
        """``point``'s record reverted (the revisit), each answer the engine names to withdraw
        first withdrawn first (:meth:`_withdrawal`, the module docstring): an answer that reads a
        column the revisit would leave on a reading nobody settled (:data:`READERS`), or the
        split a decision that changes what a row is waits on (:data:`SEALED_REFUSAL`); the answers
        withdrawn, in the order withdrawn, each with its record, its decision as the plan had it
        and what its revert said. A revisit refused otherwise, or naming an answer the log does
        not hold in force, stops in its words (:class:`arms.Stuck`); once an answer is withdrawn,
        a refusal on the way stops classed (:class:`arms.Met`, at ``revisit``), as does a revert
        of the answer named that is refused or fails."""
        withdrawn: list[dict[str, Any]] = []

        def stop(words: str, said: Any) -> Exception:
            if not withdrawn:
                return arms.Stuck(words)
            return arms.Met(words, "failed" if said.kind == "failed" else "refused",
                            said_json(said) | {"at": "revisit",
                                               "exits": arms.exit_fields(said.exits),
                                               "withdrawn": [dict(w) for w in withdrawn]})

        said = session.revisit(str(point.record), f"The sweep revisits {point.id}.")
        while not said.ok:
            words = f"Revisiting {point.id} was {said.kind}: {said.code}: {said.message}"
            named = self._withdrawal(session, point, said)
            if named is None:
                raise stop(words, said)
            if isinstance(named, str):
                raise stop(f"{words} {named}", said)
            what, column, reader = named
            if len(withdrawn) >= WITHDRAWN_AT_MOST:
                raise stop(f"{words} The sweep withdrew {len(withdrawn)} answers first already.",
                           said)
            took = session.revisit(reader.id, f"The sweep withdraws the plan's {what} first, as "
                                              f"the engine asks, to revisit {point.id}.")
            withdrawn.append({"what": what, "column": column, "seq": reader.seq,
                              "record": reader.id,
                              "decision": reader.decision.model_dump(mode="json"),
                              "reverted": said_json(took)})
            if not took.ok:
                raise arms.Met(f"Revisiting {point.id} asks the plan's {what} (record "
                               f"{reader.seq}) withdrawn first, and its revert was {took.kind}: "
                               f"{took.code}: {took.message}",
                               "failed" if took.kind == "failed" else "refused",
                               said_json(took) | {"at": "revisit",
                                                  "exits": arms.exit_fields(took.exits),
                                                  "withdrawn": [dict(w) for w in withdrawn]})
            said = session.revisit(str(point.record), f"The sweep revisits {point.id}.")
        return withdrawn

    def _reopened(self, session: Any, point: Point, policy: Named) -> dict[str, Any]:
        """The ask once the revisited question is open again (a reading or a role the server asks
        on the way settled by ``policy``, :class:`Named`), or the ask for a finding or a
        reading."""
        if point.kind != "question":
            return session.ask()
        for _ in range(arms.STEPS):
            q = session.ask()
            question = q.get("question") or {}
            if question.get("key") == point.key and question.get("status") == "open":
                return q
            settled = policy.settle(q)
            if settled is not None:
                session.answer([settled[0]], [], f"{policy.name}: {settled[1]}.")
                policy.settled(q, settled[0])
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
              later: Sequence[Point], plan: Refit, fielded: Fielded | None = None) -> Refit:
        """One option refitted on a copy of the template, classed, read, and the copy removed but
        for an engine defect's (the module docstring); an option that takes fields of its own
        answered as ``fielded`` says (:class:`Fields`), its participant flow kept."""
        if fielded is None:
            refit = Refit(point.id, str(option.get("id")), dict(option.get("decision") or {}))
        else:
            refit = Refit(point.id, fielded.option,
                          None if fielded.decision is None else dict(fielded.decision),
                          answered_with=fielded.json())
        started = time.monotonic()
        errors = len(session.errors)
        before: set[tuple[str, Any]] | None = None
        policy: Named | None = None
        try:
            refit.project = self._copy(session)
            before = self._errored(session)
            composed = fielded is not None and fielded.decision is not None
            policy = self._named(session, {"decision": fielded.decision} if composed else option)
            if point.kind != "declaration":
                options, withdrawn = self._listed(session, point, policy)
                refit.withdrawn = withdrawn
                refit.answered += [{"answer": f"withdrawn first: the plan's {w['what']} (record "
                                              f"{w['seq']}), as the engine asks",
                                    **w["reverted"]} for w in withdrawn]
                wanted = fielded.ids if fielded is not None else [refit.option]
                gone = [i for i in dict.fromkeys(wanted)
                        if i not in {str(o.get("id")) for o in options}]
                if gone:
                    raise arms.Stuck(f"The revisit offers no option {', '.join(gone)} now.")
            self._refit(session, point, option, refit, later, plan, policy, fielded)
            if fielded is not None and any(a.get("answer") == refit.id and a.get("kind") == "ok"
                                           for a in refit.answered):
                refit.flow = flow_of(session)  # the flow of what was fitted, the option in it
        except arms.Met as met:
            if refit.kind not in ("failed", "refused"):
                self._met(refit, met)
        except arms.Stuck as exc:
            if refit.kind not in ("failed", "refused"):
                refit.kind, refit.why = "unreached", str(exc)
                if before is not None:
                    self._failed_anew(session, refit, before)
        if policy is not None:
            refit.ways = list(policy.taken)  # the revisit's and the walk's, with who named each
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
            # A cancelled refit's copy was removed or kept by its cancel (``_cancel``), said so.
            if refit.project is not None and refit.cancelled is None \
                    and session.drop(refit.project, self.made):
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
              preview: bool = False, fielded: Fielded | None = None) -> Any:
        """The option previewed or answered as the driver does: a question's, a finding's or a
        reading's by its id (an option that takes fields of its own by the ids and fields it was
        answered with, ``fielded``, or as the decision it composed, a toggle's); a declaration's by
        the decision its line composed, which is what the driver's ``answer`` posts for it
        (``declarations.resolve`` with no field set)."""
        if point.kind == "declaration" or (fielded is not None and fielded.decision is not None):
            decision = dict(option["decision"] if fielded is None else fielded.decision)
            return session.preview_of(decision) if preview else session.decide(decision, reason)
        ids, sets = ([str(option["id"])], []) if fielded is None else (fielded.ids, fielded.sets)
        if preview:
            return session.preview(ids, sets, reason)
        return session.answer(ids, sets, reason,
                              arms.MARKER_OF.get(point.key, arms.DEFAULT_MARKER))

    def _refit(self, session: Any, point: Point, option: Mapping[str, Any], refit: Refit,
               later: Sequence[Point], plan: Refit, policy: Named,
               fielded: Fielded | None = None) -> None:
        """The option previewed (the crash sweep) and answered on the copy; each of the plan's
        answers the revisit withdrew answered again as the plan had it, in its log's order
        (``refit.withdrawn``); every later decision's plan answer previewed (the crash sweep); the
        walk to Fit by ``policy``, a refusal's way forward taken only where the plan's log in force
        before the option or the option names it (:class:`Named`); the sweep's scoring again; Fit
        pressed and read within the refit's wait (:meth:`_fit`), but for the plan's duplicate. An
        option answered with fields of its own
        (``fielded``) records the decision its preview resolved, which names the option's ways
        forward; where the driver does not take its fields, it stops in the driver's words."""
        if self.crash:
            seen = self._post(session, point, option, f"The crash sweep previews {refit.id}.",
                              preview=True, fielded=fielded)
            refit.previews.append({"of": refit.id, **said_json(seen)})
            if fielded is not None:
                if seen.kind == "refused" and seen.by == "driver":
                    raise arms.Stuck(f"{NOTHING}the driver does not take the fields the crash "
                                     f"sweep answered it with: {seen.code}: {seen.message}")
                decision = (seen.payload or {}).get("previewed")
                if isinstance(decision, Mapping):
                    refit.decision = dict(decision)
                    policy.names["the option"] = [dict(decision)]
        said = self._post(session, point, option, f"The sweep's refit: {refit.id}.",
                          fielded=fielded)
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
        # The plan's answers withdrawn first, as the plan had them and in its log's order, so that
        # of two answers to one decision the plan's latest is again the one in force.
        for w in sorted(refit.withdrawn, key=lambda w: int(w["seq"])):
            again = session.decide(w["decision"], f"The sweep answers the plan's {w['what']} "
                                                  f"again after {refit.id}, as the plan had it.")
            refit.answered.append({"answer": f"the plan's {w['what']} again (record {w['seq']})",
                                   **said_json(again)})
            if not again.ok:
                detail = said_json(again) | {"at": "answer_again", "decision": w["decision"]}
                if again.kind == "failed":
                    refit.kind, refit.failed = "failed", detail
                else:
                    refit.kind = "refused"
                    refit.refused = detail | {"option": refit.option,
                                              "exits": arms.exit_fields(again.exits)}
                return
        if self.crash:
            for p in later:  # every later decision's default (:func:`later_of`)
                if p.decision is not None:  # a line the plan left open has no answer to preview
                    seen = session.preview_of(self._later(p))
                    refit.previews.append({"of": f"{p.id} (the plan's answer)",
                                           **said_json(seen)})
        steps: list[arms.Step] = []
        policy.steps = steps  # a way refused is not answered again until one is recorded
        before = (lambda key, q: self._preview_reopened(session, refit, key, q)) if self.crash \
            else None
        try:
            arms.walk(session, policy, steps=steps, before=before)
        finally:
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
        self._fit(session, refit)

    def scheme(self, split: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """The sweep's scoring as the split ``split`` takes it (the folds' order random under a
        cut-point), where the template recorded it; else None."""
        scoring = (self.base or {}).get("scoring") or {}
        if not scoring.get("set"):
            return None
        body = dict(SCORING)
        if (split or {}).get("cut_point"):
            body["fold_order"] = FOLD_ORDER
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
        # The engine this run starts on, kept before any work (the report holds it to the end's),
        # and the refit wait it runs under (a resumed sweep's runs may each have had their own).
        start = {**engine_version(), "at": now(), "refit_wait": self.refit_wait}
        self.engines.append(start)
        self._write({"engine": start})
        self.template(session, build)
        plan = self.plan_refit(session)
        for point in self.points:
            if self.stopped is not None:
                break
            later = later_of(self.points, point)
            listed = self.offered.get(point.id)
            if listed is None:
                listed = self._list(session, point)
                self.offered[point.id] = listed
                self._write({"offered": listed})
            planned = self._planned(session, point, listed)
            self.planned[point.id] = [p["id"] for p in planned]
            for p in planned:
                rid, kept = p["id"], self.done.get(p["id"])
                if p.get("why") is not None:  # unreached: recorded again only where its why moved
                    if kept is not None and (kept.kind != "unreached" or kept.why == p["why"]):
                        continue
                    r = Refit(point.id, p["option"], None, kind="unreached", why=p["why"])
                else:
                    if kept is not None:
                        continue
                    if self._capped():
                        break
                    self.log(rid)
                    fielded = {"fielded": p["fielded"]} if p.get("fielded") else {}
                    r = self.refit(session, point, p["offered"], later, plan, **fielded)
                self.done[rid] = r
                self._write({"refit": r.json()})
        return self.report(plan, time.monotonic() - started)

    def _planned(self, session: Any, point: Point, listed: Mapping[str, Any]
                 ) -> list[dict[str, Any]]:
        """``point``'s refits, in the order of its options: each option that records something by
        itself, refitted as listed; each that takes fields of its own, under the crash sweep,
        answered with each value :class:`Fields` gives it (each its own refit, named by its
        source), else unreached with why, as every such option is around a named plan; under
        Estimate, around every plan, the adjustment's answers option also answered once per
        column of the plan's answers in force, toggled with the plan's other answers fixed
        (:func:`toggled`, :attr:`Fields.adjustment`); and the
        card's whole population after them (:meth:`Fields.population`). Each with its id
        (``point=option``), the option as offered and, for an option's fields, what answers it."""
        out: list[dict[str, Any]] = []
        fields = None
        for option in listed.get("options") or []:
            oid = str(option.get("id"))
            bare = {"id": f"{point.id}={oid}", "option": oid, "offered": option}
            if not takes_fields(option):
                out.append(bare)
                continue
            toggles: list[Fielded] = []
            if self.question.purpose == "inference" and (point.kind, point.key) == (
                    "question", "adjustment"):
                fields = fields or self._fields(session)
                toggles = toggled(point, option, fields.effect, fields.adjustment)
            if not self.crash:
                out += [{"id": f"{point.id}={a.option}", "option": a.option, "offered": option,
                         "fielded": a} for a in toggles] or [
                    bare | {"why": f"{NOTHING}it takes {needs_of(option)}."}]
                continue
            fields = fields or self._fields(session)
            answers, why = fields.answers(point, option)
            out += [{"id": f"{point.id}={a.option}", "option": a.option, "offered": option,
                     "fielded": a} for a in [*answers, *toggles]] or [bare | {"why": why}]
        if self.crash and any(takes_fields(o) for o in listed.get("options") or []):
            fields = fields or self._fields(session)
            offered = [str(o.get("id")) for o in listed.get("options") or []]
            out += [{"id": f"{point.id}={a.option}", "option": a.option,
                     "offered": {"id": a.option, "decision": None}, "fielded": a}
                    for a in fields.population(point, offered)]
        return out

    def _fields(self, session: Any) -> Fields:
        """The crash sweep's sources for an option's fields of its own (:class:`Fields`), read once
        from the template, the dataset's card and the table."""
        if self.fields is None:
            self.fields = Fields.of_template(session, self.question, self.dataset, self.data,
                                             str((self.base or {}).get("project")))
        return self.fields

    def _reported(self, rid: str, refit: Refit) -> bool:
        """Whether a refit of the progress file is one of this run's: of a decision this run did
        not plan, each; of one it planned, those it planned and its revisit's own. An option
        unreached before the crash sweep answered fields of its own is now the refits that answer
        it, which supersede it."""
        ids = self.planned.get(refit.point)
        return ids is None or rid in ids or refit.option == REVISIT

    def _list(self, session: Any, point: Point) -> dict[str, Any]:
        """``point``'s options listed on a copy of the template (:meth:`_listed`), with the plan's
        answers its revisit withdrew first and each refusal's way forward taken on the way, with
        who named it (``ways``, :class:`Named`). A 500 met on the way, or a stage failed untyped,
        is an engine defect (its revisit refitted failed, the copy kept); the way the engine names
        for the revisit, refused (:meth:`_revisit`), is the decision visited and refused (its
        revisit refitted refused, never unlisted); any other stop lists it with why it could not
        be."""
        copy = self._copy(session)
        errors = len(session.errors)
        met: arms.Met | None = None
        withdrawn: list[dict[str, Any]] = []
        refused: dict[str, Any] | None = None
        policy: Named | None = None
        try:
            policy = self._named(session)
            options, withdrawn = self._listed(session, point, policy)
            why = None
        except arms.Stuck as exc:
            options, why = [], str(exc)
            met = exc if isinstance(exc, arms.Met) else None
        failed = list(session.errors[errors:])
        if failed or (met is not None and met.kind == "failed"):
            # A 500, or a stage failed untyped, met on the revisit itself: an engine defect, kept
            # with it.
            revisit = Refit(point.id, REVISIT, None, kind="failed",
                            failed={"kind": "failed", "by": "server", "code": "http_500",
                                    "message": failed[-1].get("said")} if failed
                            else dict(met.detail), project=copy)
            self._defect(session, revisit, failed)
            self.done[revisit.id] = revisit
            self._write({"refit": revisit.json()})
        elif met is not None and met.detail.get("at") == "revisit":
            withdrawn = list(met.detail.get("withdrawn") or [])
            refused = dict(met.detail) | {"option": REVISIT}
            revisit = Refit(point.id, REVISIT, None, kind="refused", why=str(met), refused=refused)
            self.done[revisit.id] = revisit
            self._write({"refit": revisit.json()})
            why = None
            session.drop(copy, self.made)
        else:
            session.drop(copy, self.made)
        return {"point": point.id, "why": why, "refused": refused,
                "withdrawn": [{k: w.get(k) for k in ("what", "column", "seq", "record")}
                              for w in withdrawn],
                "ways": list(policy.taken) if policy is not None else [],
                "options": [{"id": o.get("id"), "label": o.get("label"),
                             "decision": o.get("decision"), "needs": o.get("needs"),
                             **({"columns": list(o["columns"])} if o.get("columns") else {})}
                            for o in options]}

    def spent(self) -> float:
        """The fit seconds this dataset's sweeps have spent: the earlier sweeps' (``spent_before``)
        and every fit of this one's (:func:`timed_of`)."""
        return self.spent_before + sum(t for t in (timed_of(r) for r in self.done.values())
                                       if t is not None)

    def _capped(self) -> bool:
        """Whether the dataset's compute cap (``cap``, prereg.json's) is spent, so the sweep stops
        before its next fit, said in ``stopped``."""
        if self.cap is None or self.stopped is not None:
            return self.stopped is not None
        spent = self.spent()
        if spent < self.cap:
            return False
        self.stopped = {"cap": self.cap, "spent": round(spent, 3),
                        "spent_before": round(self.spent_before, 3),
                        "why": (f"The sweeps of {self.dataset} stopped at its compute cap: "
                                f"{spent:,.0f} of {self.cap:,.0f} fit seconds spent "
                                f"(prereg.json's compute_cap), before the next fit.")}
        return True

    def report(self, plan: Refit, seconds: float) -> dict[str, Any]:
        """The sweep's table (the module docstring): beside what it swept and how each refit was
        classed, each option left unreached with why (``unreached``), each answered with fields of
        its own and where their values came from (``fielded``), and the card's rules read
        (``card_rules``, where the crash sweep read the card)."""
        refits = [r for k, r in self.done.items() if k != PLAN and self._reported(k, r)]
        timed = [timed_of(r) for r in [plan, *refits] if timed_of(r) is not None]
        unvisited = [rid for p in self.points if p.id in self.offered
                     for rid in self.planned.get(p.id) or [
                         f"{p.id}={o.get('id')}" for o in self.offered[p.id].get("options") or []]
                     if rid not in self.done]
        unlisted = unlisted_of(self.points, self.offered)
        unvisited += [u["point"] for u in unlisted]
        end = engine_version()  # the engine as the sweep ends, held to each run's start
        return {
            "dataset": self.dataset, "plan": self.plan, "purpose": self.question.purpose,
            **end, "engine_starts": [dict(e) for e in self.engines],
            "engine_changed": engine_changed(self.engines, end, unrecorded=self.unrecorded),
            "refit_wait": self.refit_wait,  # this run's (each run's is on its start)
            "written": now(), "seconds": round(seconds, 1),
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
            "unreached": [{"refit": r.id, "why": r.why} for r in refits if r.kind == "unreached"],
            "fielded": [{"refit": r.id, "kind": r.kind, **r.answered_with} for r in refits
                        if r.answered_with],
            "card_rules": self.fields.card_json() if self.fields is not None else None,
            "defects": list(self.defects),
            "cap": None if self.cap is None else {
                "seconds": self.cap, "spent": round(self.spent(), 3),
                "spent_before": round(self.spent_before, 3)},
            "stopped": self.stopped,
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
             + (f"; unreached: {len(report['unreached'] or [])}" if "unreached" in report else "")
             + (f"; swept only {', '.join(report['only'])}" if report.get("only") is not None
                else "") + "."
             + (f" Each refit waited at most {report['refit_wait']:g} s from its press of Fit."
                if report.get("refit_wait") is not None else ""), "",
             "| Decision | Options | Scored | Refused | Failed | Unreached | Duplicate | Refused "
             "share | Material |", "|---|---|---|---|---|---|---|---|---|"]
    for d in report.get("decisions") or []:
        c = d["classes"]
        share = "" if d["refused_share"] is None else f"{d['refused_share']:.2f}"
        lines.append(f"| {d['point']} | {d['options']} | {c['scored']} | {c['refused']} | "
                     f"{c['failed']} | {c['unreached']} | {c['duplicate']} | {share} | "
                     f"{d['material']} |")
    if report.get("engine_changed"):
        lines += ["", "## The engine changed while the sweep ran", "",
                  str(report["engine_changed"])]
    if report.get("stopped"):
        lines += ["", "## Stopped at the compute cap", "", str(report["stopped"]["why"])]
    withdrawn = [o for o in report.get("offered") or [] if o.get("withdrawn")]
    if withdrawn:
        lines += ["", "## Answers withdrawn first to revisit a decision, as the engine asks", ""]
        lines += [f"- `{o['point']}`: " + ", ".join(f"the {w['what']} (record {w['seq']})"
                                                    + (f", which reads `{w['column']}`"
                                                       if w.get("column") else "")
                                                    for w in o["withdrawn"])
                  + ("; refused on the way" if o.get("refused") else
                     "; answered again after each option, as the plan had them")
                  for o in withdrawn]
    if report.get("unlisted"):
        lines += ["", "## Decisions whose options were not listed", ""]
        lines += [f"- `{u['point']}`: {str(u['why'])[:300]}" for u in report["unlisted"]]
    if report.get("fielded"):
        lines += ["", "## Options answered with values the crash sweep supplied", ""]
        lines += [f"- `{f['refit']}` ({f['kind']}), {f['source']}: {f['words']}"
                  for f in report["fielded"]]
    if report.get("unreached"):
        lines += ["", "## Options not reached", ""]
        lines += [f"- `{u['refit']}`: {str(u['why'])[:400]}" for u in report["unreached"]]
    unread = (report.get("card_rules") or {}).get("not_read") or []
    if unread:
        lines += ["", "## The card's sentences naming a rule the crash sweep could not read", ""]
        lines += [f"- {u['place']}: {u['words']}" for u in unread]
    if report.get("defects"):
        lines += ["", "## Engine defects", ""]
        lines += [f"- `{d['refit']}`: {json.dumps(d.get('failed'))[:300]} ({d.get('path')})"
                  for d in report["defects"]]
    return "\n".join(lines) + "\n"


def crash(dataset: str, out: Path, *, only: Iterable[str] | None = None,
          data: Path | None = None, question: arms.Question | None = None,
          log: Callable[[str], None] | None = None,
          refit_wait: float | None = None) -> dict[str, Any]:
    """The crash sweep on ``dataset`` (the module docstring), into ``out``: ``crash.json`` and
    ``crash.md``. A test's small fixture gives ``data`` (a table in place of the one the dataset's
    manifest builds) and ``question`` (in place of its card's); ``only``: the decisions swept, by
    point id; ``refit_wait``: each refit's wait in seconds from its press of Fit
    (:data:`REFIT_WAIT` where None), which the report records."""
    from research.swarm.dry_run import table_of

    out = Path(out).absolute()
    question = question or arms.Question.of_card(arms.card_of(dataset))
    table = Path(data).absolute() if data is not None else table_of(dataset, out / "data")
    sweep = Sweep(out, dataset=dataset, question=question, crash=True, plan="default",
                  only=only, log=log, data=table, refit_wait=refit_wait)
    with arms.Local(out / "home") as session:
        report = sweep.run(session, default_build(sweep.policy, table))
        session.use(str(sweep.base["project"]))
        report["readings"] = graded(session.view(), readings_truth(dataset))
    report["data"] = {"path": str(table), "sha256": hashlib.sha256(table.read_bytes()).hexdigest()}
    report["passed"] = passed(report)
    write_outputs(out, "crash", report)
    return report


def compute_cap(dataset: str, prereg: Mapping[str, Any] | None = None) -> float:
    """The compute cap of ``dataset``'s sweeps around named plans, in fit seconds: prereg.json's
    ``filled.compute_cap.by_dataset[dataset].seconds`` (``prereg_fill``'s crash fill; ``prereg``
    in place of the file's, a test's); refused in words (:class:`arms.Stuck`) where it holds
    none."""
    if prereg is None:
        from research.swarm import prereg_fill

        prereg = prereg_fill.load()
    by = (((prereg.get("filled") or {}).get("compute_cap") or {}).get("by_dataset") or {})
    seconds = (by.get(dataset) or {}).get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
            or not math.isfinite(float(seconds)):
        raise arms.Stuck(f"prereg.json holds no compute cap for the sweeps of {dataset}: fill it "
                         f"from the crash sweeps (python -m research.swarm.prereg_fill crash …).")
    return float(seconds)


def spent_by(folder: Path, dataset: str, *, but: Path | None = None) -> float:
    """The fit seconds ``dataset``'s sweeps around named plans in ``folder`` spent, each in a
    folder of its own there (its ``progress.jsonl``: :func:`timed_of` of each refit, its plan's
    own among them), but the sweep in ``but``; a crash sweep's are the default plan's, paid
    before the cap, and are not counted."""
    total = 0.0
    for progress in sorted(Path(folder).glob("*/progress.jsonl")):
        if but is not None and progress.parent.resolve() == Path(but).resolve():
            continue
        base: dict[str, Any] = {}
        refits: dict[str, Refit] = {}
        for line in progress.read_text("utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a line an interruption cut
            if "template" in entry:
                base = entry["template"]
            elif "refit" in entry:
                refit = Refit.of(entry["refit"])
                refits[refit.id] = refit
        if base.get("dataset") != dataset or base.get("crash", True):
            continue
        total += sum(t for t in (timed_of(r) for r in refits.values()) if t is not None)
    return total


def sweep_plan(dataset: str, bundle: Path, name: str, out: Path, *, data: Path | None = None,
               only: Iterable[str] | None = None, question: arms.Question | None = None,
               log: Callable[[str], None] | None = None, cap: float | None = None,
               spent: float = 0.0) -> dict[str, Any]:
    """The sweep around the named plan in ``bundle`` (the module docstring), into ``out``:
    ``sweep.json`` and ``sweep.md`` (``question`` in place of the dataset card's, a test's).
    ``cap``: the dataset's compute cap in fit seconds, ``spent`` its earlier sweeps' fit seconds
    (the command reads both: :func:`compute_cap`, :func:`spent_by`); None, no cap (a test's)."""
    out = Path(out).absolute()
    question = question or arms.Question.of_card(arms.card_of(dataset))
    sweep = Sweep(out, dataset=dataset, question=question, crash=False, plan=name, only=only,
                  log=log, data=data, cap=cap, spent=spent)
    pid = sweep.base["project"] if sweep.base else replayed(Path(bundle), data, out)
    with arms.Local(out / "home") as session:
        report = sweep.run(session, lambda s: pid)
        session.use(pid)
        report["readings"] = graded(session.view(), readings_truth(dataset))
    write_outputs(out, "sweep", report)
    return report


def wait_of(text: str) -> float:
    """A command line's number of seconds: a positive finite number, else refused in words."""
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of seconds")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.sweep",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("crash", help="the crash sweep around the default plan (SWARM_PLAN §6)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--refit-wait", type=wait_of, default=REFIT_WAIT, metavar="SECONDS",
                   help=f"how long each refit waits from its press of Fit before it is "
                        f"cancelled and left unreached (default {REFIT_WAIT:g})")
    p = sub.add_parser("plan", help="the sweep around a named final plan")
    p.add_argument("--dataset", required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sweeps", type=Path, help="the folder of the dataset's sweeps around named "
                                               "plans, each in its own (default: --out's)")
    args = parser.parse_args(argv)

    def log(text: str) -> None:
        sys.stderr.write(f"{now()} {text}\n")
        sys.stderr.flush()

    if args.command == "crash":
        report = crash(args.dataset, args.out, log=log, refit_wait=args.refit_wait)
        sys.stdout.write(json.dumps({k: report[k] for k in ("dataset", "passed", "complete",
                                                            "fit_seconds", "engine_changed")}
                                    | {"unreached": len(report["unreached"]),
                                       "defects": len(report["defects"])}) + "\n")
        return 0 if report["passed"] else 1
    try:
        cap = compute_cap(args.dataset)
    except arms.Stuck as exc:
        sys.stdout.write(json.dumps({"refused": str(exc)}) + "\n")
        return 1
    out = Path(args.out).absolute()
    spent = spent_by(args.sweeps or out.parent, args.dataset, but=out)
    report = sweep_plan(args.dataset, args.bundle, args.name, out, data=args.data, log=log,
                        cap=cap, spent=spent)
    sys.stdout.write(json.dumps({k: report[k] for k in ("dataset", "plan", "complete",
                                                        "stopped", "engine_changed")}) + "\n")
    return 0 if report["complete"] and report["engine_changed"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The walker (SWARM_PLAN §6 "Mechanical arms"; the owner's ruling of 2026-10-23, "Go with D"): seeded
walks through a run of the driver's own, on real fits, along the paths the crash sweep never takes.

    python -m research.swarm.walk --dataset <id> --seeds <a-b> --out <dir>

**Why.** The crash sweep (``research/swarm/sweep.py``) changes one decision at a time around the
default plan, before any press of Fit; it never seals and never exports. A walk takes the path
classes it cannot reach: several changes in force at once, revisits after Fit, after the lock and
after the seal, repeated revisits and the undoing of a revert, stale ids, a second seal, and the
export end.

**One walk** (:func:`walk_one`) is one seed on the dataset, in a run of the driver's own
(``arms.Driven``: every command through ``driver.main``, each kept with what it said, its
``transcript``, :class:`Walked`). It is the random arm's walk to the first Fit (``arms.Random`` at
the seed: the card's question, every question, reading and finding answered by its draws, each
open declaration line once, Fit as ``arms.fit_through`` presses it, the results), then 2 to 4
rounds (:data:`ROUNDS`; the count and every draw of the walker's own from numpy's generator at
``(STREAM, seed)``, :func:`walker_rng`, so the random arm's draws stay its own). Each round draws
one action uniformly among those the driver offers at that point (:func:`offered_actions`), then
one of its candidates uniformly:

* ``revisit`` a decision in force (not the card's own answers, :data:`arms.FIXED` and the
  exposure the estimand names, nor the seal, the lock or a revert);
* ``undo`` a revert in force (the revert itself revisited);
* ``finding``: act on an open finding as its line offers;
* ``declaration``: answer an open declaration line;
* ``preview`` an option of the open question, of a declaration line or of a finding;
* ``fit``: press Fit;
* ``stale``: send an id no longer good, a decision already reverted (revisited again) or an
  option id an earlier ask offered and this one does not (answered); the kind is drawn first,
  uniformly among those offered, then an id of that kind (:func:`stale_of`), so the reverted
  kind is not lost among the many options earlier asks offered.

An option that takes values of its own takes them from the crash sweep's ``sweep.Fields`` where it
builds them (the first it builds), and is otherwise skipped and counted (``skipped``, with why;
:func:`answer_of`). Where the walker's own answer or revisit is refused with ways forward that
record something, it takes one, drawn. After each change the walk goes on as the random arm does
(``arms.walk``, Fit, the results).

**The end of the path** (:meth:`Walker.end`). Under Predict: what the fit raises answered and Fit
pressed again (the random arm's), the held-out rows opened on a family drawn from those fitted
(``arms.seal_through``), the seal again, a decision revisited after it, and the export. Under
Estimate the plan's lock is recorded by the press of Fit itself (the driver posts none), and the
seal comes only where the engine asks for it (the ask's ``open_seal``), sealed again and revisited
after as under Predict; then a decision is revisited after the lock (where it is recorded, the walk
goes on and fits), and the export. Where the export is refused with ways forward that record
something, one is taken, the walk goes on, and the export is asked again, :data:`EXPORT_TRIES`
times at most.

**The oracle** (:func:`classed`): a walk is classed ``defect`` where it met any of

1. an HTTP 500, the driver's own failure (``driver_failed``, a server it could not read or reach),
   or a stage or fit error the engine did not type as a refusal (:func:`failures_of`; every stage
   of the project after each press of Fit and each read of the results, with the round it was
   met in, and as the walk ends, :func:`stages_of`; a question that waited on one);
2. a refusal with no code or no words (:func:`wordless_of`);
3. a dead end: a way forward that records something which, taken (the next command after its
   refusal that records something answers it), is refused again with the same code, or, where
   the refusal was of a press of Fit, the seal or the export, is recorded and the next of that
   command is refused again with the same code (:func:`dead_ends_of`; so a walk the export's or
   the seal's tries end, each way taken and refused again alike, is a defect);
4. a fit record whose plan's digest is not the log's: the decision log (``decisions.jsonl``, read
   with the engine's own reader) refolded offline as it stood when the fit completed (the records
   appended by the record's time, the longest prefix whose records in force are those it names)
   gives another digest, or no prefix holds those records (:func:`digest_checks`; no refit is run);
5. a second seal or a revisit after the seal that is accepted (expected refused, with the codes the
   engine's validator of an opening and the driver's guard raise, :func:`expected_codes`; a refusal
   with another code is noted, not a defect, :func:`noted_of`).

Else it is ``clean`` where it reached the export and the export was given, and ``refused`` where it
stopped short (``stops``, each by whom and in its words): a refusal in words that left it no way on
(the engine's, the server's or the driver's), or a limit of the walk's own. A fault in the walker's
own code ends that walk too, kept as the walker's (``by: walker``): evidence neither way, so a report
holding one does not pass. A refusal is expected, never a defect, when it is typed and worded and
its way forward works, or it has none and its words say why. That no held-out row is read before the seal is not checked: the driver cannot observe it,
and a known deferred exception exists (Harrell's rule under prediction reads every cohort row).

**Each defect keeps its reproduction** (``defects/<dataset>-<seed>-<n>.json``): the seed, the
walk's actions and steps, the decision log as it stood, the request that failed (its command line
and what it said), the engine's words, the run's folder (kept, with its project) and the table.

**The engine** is stamped at each walk's start and end (``sweep.engine_version``: the commit, the
``turbotab/`` tree, whether a tracked file under it is modified), as the sweep's are; a report
whose engines differ, any walk's start or end from the report's end, is refused
(``sweep.engine_changed``, in words, and said in ``walk.md``): it does not pass.

**Resumable by seed**: each walk is written as it ends (``walks/<dataset>-<seed>.json``), and a
seed already written is not walked again. The report (``walk.json``, and ``walk.md`` to read; the
crash sweep's shape) holds each walk's class and summary, the defects with their reproductions, the
count of each class and of the options skipped, the fit records held to the log, and whether it
passed: every seed reported, none ended by the walker's own fault, no defect, one engine.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from research.swarm import arms, sweep

REPO = Path(__file__).resolve().parents[2]
STREAM = 1  # the walker's own draws: numpy's generator at (STREAM, seed), apart from the arm's
ROUNDS = (2, 4)  # the fewest and the most rounds of offered actions after the first Fit
ACTIONS = ("revisit", "undo", "finding", "declaration", "preview", "fit", "stale")
CLASSES = ("clean", "refused", "defect")
EXPORT_TRIES = 3
BUDGET_FITS, BUDGET_SECONDS = 400, 4 * 3600.0  # the run's budget: the walk's own, never the cap
# Records a revisit never takes: the seal and the lock (the engine's events), and the reverts
# (undone as their own action); and the card's own answers (the question, ``arms.FIXED``, and the
# exposure the estimand names), which another answer would make another question.
UNREVISITED = ("open_seal", "lock_plan", "reseal", "revert")
CARDS = (*arms.FIXED, "estimand")
# The commands that record something (a press of Fit records the lock, or a press), so the first
# of them after a refusal is the one that took its way forward or did not (:func:`dead_ends_of`).
RECORDING = ("answer", "revisit", "seal", "fit")
# The commands the walk gives again once a way forward from their refusal is recorded (Fit as
# ``arms.fit_through`` presses it, the seal as ``arms.seal_through`` asks it, the export), so a
# way recorded is a dead end where the next of them is refused with the same code.
RETRIED = ("fit", "seal", "export")
STALE = ("reverted", "option")  # the kinds of stale id, drawn first (:func:`stale_of`)
# Where a stop came from a command the transcript holds (``arms.walk``'s answers, the walker's
# own, a press of Fit, the export), so its refusal is read there, not again from the stop.
COMMANDED = ("walk", "walker", "fit", "export")
NEEDS = re.compile(r"^Needs '([a-z_]+)', which failed\.?$")


def walker_rng(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng([STREAM, int(seed)])


def rounds_of(rng: Any) -> int:
    return int(rng.integers(ROUNDS[0], ROUNDS[1] + 1))


def draw(rng: Any, items: Sequence[Any]) -> Any:
    return items[int(rng.integers(len(items)))]


def stale_of(rng: Any, candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """A stale id drawn from ``candidates`` (``offered_actions``' ``stale``): its kind uniformly
    among the kinds of :data:`STALE` offered, then one of that kind uniformly; so a decision
    already reverted, which the driver's ``revisit`` posts to the server, is sent as often as an
    option id an earlier ask offered, which the driver's own resolver refuses, however many of
    each are offered."""
    kinds = [k for k in STALE if any(c.get("kind") == k for c in candidates)]
    kind = draw(rng, kinds)
    return draw(rng, [c for c in candidates if c.get("kind") == kind])


def stamp() -> dict[str, Any]:
    """The engine as it stands now (``sweep.engine_version``), with when."""
    return {**sweep.engine_version(), "at": sweep.now()}


def seeds_of(text: str) -> list[int]:
    """``--seeds``: ``a-b`` (each from a to b, both included) or one seed."""
    low, _, high = str(text).partition("-")
    a, b = int(low), int(high or low)
    if b < a:
        raise ValueError(f"--seeds {text!r} runs backwards.")
    return list(range(a, b + 1))


# ── the session ──────────────────────────────────────────────────────────────


def offered_ids(q: Mapping[str, Any]) -> set[str]:
    """Every id the ask offers: its question's options, each line's, and every way forward."""
    ids = {str(o["id"]) for o in q.get("options") or [] if o.get("id") is not None}
    ids |= {str(o["id"]) for line in q.get("also_open") or [] for o in line.get("options") or []
            if o.get("id")}
    ids |= {str(w["id"]) for w in arms.ways_of(q) if w.get("id")}
    ids |= {str(w["id"]) for f in q.get("fit_failed") or []
            for w in (f.get("refusal") or {}).get("exits") or [] if w.get("id")}
    return ids


def entry_of(argv: Sequence[str], code: int, payload: Mapping[str, Any], *, n: int
             ) -> dict[str, Any]:
    """What the walk keeps of one command: its name, its arguments but the run's id, the ids it
    named, its exit status, what it said (``arms.said_of``: its kind, by whom, its code, words and
    status, and its ways forward with their ids and decisions), the record it made, and for an ask
    the question and every id it offered; for one that did not say ok, its JSON (trimmed as the
    driver trims), the request a reproduction repeats."""
    from research.swarm import driver

    command = str(argv[0]) if argv else None
    rest = [str(a) for a in argv[1:]]
    if rest[:1] == ["--run"]:
        rest = rest[2:]
    said = arms.said_of(code, dict(payload))
    out: dict[str, Any] = {"n": n, "command": command, "args": rest,
                           "ids": list(itertools.takewhile(lambda a: not a.startswith("--"), rest)),
                           "exit": code, **arms.said_fields(said),
                           "exits": [{k: e.get(k) for k in ("id", "label", "decision")}
                                     for e in said.exits if isinstance(e, Mapping)]}
    recorded = payload.get("recorded") if said.ok else None
    if isinstance(recorded, Mapping):
        out["recorded"] = {k: recorded.get(k) for k in ("id", "seq", "decision")}
    if command == "ask" and said.ok:
        q = payload.get("question") or {}
        out["asked"] = {"question": q.get("key"), "status": q.get("status"),
                        "offered": sorted(offered_ids(payload))}
    elif not said.ok:
        out["payload"] = driver.trim(dict(payload))
    if command == "fit" and isinstance(payload.get("fit"), Mapping):
        out["fit"] = dict(payload["fit"])
    return out


class Walked(arms.Driven):
    """A run of the driver's own (``arms.Driven``) that keeps every command it ran, with what it
    said (:func:`entry_of`), in ``transcript``; and after each press of Fit and each read of the
    results, the project's stages in error (:func:`stages_of`, one view read), each kept once in
    ``met`` with where the walk was (``at``: ``forward``, a round's number, ``end``) and the
    command it followed, since a later refit computes the stage again at another key and the
    project's view as the walk ends no longer holds it; with the driver's ``revisit``."""

    def __init__(self, runs: Path):
        super().__init__(runs)
        self.transcript: list[dict[str, Any]] = []
        self.met: list[dict[str, Any]] = []
        self.at: Any = "forward"

    def command(self, *argv: str) -> tuple[int, dict[str, Any]]:
        code, payload = super().command(*argv)
        self.transcript.append(entry_of(argv, code, payload, n=len(self.transcript)))
        return code, payload

    def held(self) -> None:
        """The stages in error now, each not met before (its defect, stage and key) kept."""
        seen = {(d.get("defect"), d.get("stage"), d.get("key")) for d in self.met}
        after = self.transcript[-1]["n"] if self.transcript else None
        for d in stages_of(self.view()):
            if (d.get("defect"), d.get("stage"), d.get("key")) not in seen:
                self.met.append(d | {"round": self.at, "after": after})

    def fit(self) -> arms.Said:
        said = super().fit()
        self.held()
        return said

    def results(self) -> arms.Said:
        said = super().results()
        self.held()
        return said

    def use(self, pid: str) -> None:
        """A run has one project (``sweep.Fields.of_template`` names it): nothing to do."""

    def revisit(self, record: str, reason: str) -> arms.Said:
        return self.cmd("revisit", record, "--reason", reason)

    def project_folder(self) -> Path:
        return self.runs / str(self.run) / "home" / "projects" / str(self.meta()["project"])


class LazyFields:
    """``sweep.Fields`` read from the run's project as it stands when an option needs values."""

    def __init__(self, session: Any, question: arms.Question, dataset: str | None, table: Path):
        self.session, self.question, self.dataset, self.table = session, question, dataset, table

    def answers(self, point: sweep.Point, option: Mapping[str, Any]
                ) -> tuple[list[sweep.Fielded], str | None]:
        fields = sweep.Fields.of_template(self.session, self.question, self.dataset,
                                          str(self.table), "")
        return fields.answers(point, option)


def answer_of(option: Mapping[str, Any], point: sweep.Point | None, fields: Any,
              skipped: list[dict[str, Any]]) -> tuple[list[str], list[str]] | None:
    """``option`` as the driver's ``answer`` takes it: its id, and where it takes values of its
    own, the first answer ``fields`` builds for it (``sweep.Fields.answers``: ids and ``--set``
    fields); None where none is built, the option kept in ``skipped`` with why."""
    if not sweep.takes_fields(option):
        return [str(option["id"])], []
    got, why = fields.answers(point, option) if point is not None else (
        [], f"{sweep.NOTHING}it takes {sweep.needs_of(option)}.")
    if got:
        return list(got[0].ids), list(got[0].sets)
    skipped.append({"option": str(option.get("id")), "point": point.id if point else None,
                    "why": why})
    return None


# ── the actions offered ──────────────────────────────────────────────────────


def offered_actions(q: Mapping[str, Any], view: Mapping[str, Any],
                    seen: Iterable[str] = ()) -> dict[str, list[dict[str, Any]]]:
    """Each action's candidates where the ask ``q`` and the project ``view`` stand (the module
    docstring), each list in the log's or the ask's order; ``seen``, every id an earlier ask
    offered."""
    from research.swarm import declarations as decl
    from turbotab.core import decisions as D

    records = sweep.records_of(view)
    try:
        cancelled = D.reverted(records)
    except D.Refusal:
        cancelled = {}
    out: dict[str, list[dict[str, Any]]] = {a: [] for a in ACTIONS}
    for r in sorted(records, key=lambda r: r.seq):
        kind = r.decision.kind
        if r.id in cancelled:
            continue
        if kind == "revert":
            out["undo"].append({"id": r.id, "reverts": r.decision.decision_id, "seq": r.seq})
        elif kind not in UNREVISITED and sweep.question_key(kind) not in CARDS:
            out["revisit"].append({"id": r.id, "kind": kind, "seq": r.seq})
    declarable = set(decl.kinds())
    for line in q.get("also_open") or []:
        source, key = line.get("source"), line.get("key")
        if source == "finding" or (source == "declaration" and key in declarable):
            out[str(source)] += [{"id": str(o["id"]), "line": key, "source": source,
                                  "option": dict(o)}
                                 for o in line.get("options") or [] if o.get("id")]
    key = (q.get("question") or {}).get("key")
    out["preview"] = [{"id": str(o["id"]), "question": key, "option": dict(o)}
                      for o in q.get("options") or [] if o.get("id") is not None] \
        + out["finding"] + out["declaration"]
    out["fit"] = [{}]
    by_seq = {r.id: r.seq for r in records}
    out["stale"] = [{"kind": "reverted", "id": rid}
                    for rid in sorted(cancelled, key=lambda i: by_seq.get(i, 0))] + [
        {"kind": "option", "id": oid} for oid in sorted(set(seen) - offered_ids(q))]
    return out


def point_of(candidate: Mapping[str, Any], view: Mapping[str, Any]) -> sweep.Point:
    """The crash sweep's point an option belongs to (``sweep.Fields.answers`` reads it): the open
    question with its record in force, a declaration line, or a finding."""
    if candidate.get("question") is not None:
        key = str(candidate["question"])
        record = next((r for r in reversed(sweep.live(view))
                       if sweep.question_key(r.decision.kind) == key), None)
        return sweep.Point("question", key, seq=record.seq if record else None,
                           record=record.id if record else None,
                           decision=record.decision.model_dump(mode="json") if record else None)
    return sweep.Point(str(candidate.get("source")), str(candidate.get("line")))


# ── the oracle ───────────────────────────────────────────────────────────────


def failures_of(transcript: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """(1) Each command that failed (``arms.said_of``: an HTTP 500 or above, the driver's own
    failure, a server it could not read or reach, a fit in error the engine did not type)."""
    return [{"defect": "failed", "command": e.get("command"), "code": e.get("code"),
             "status": e.get("status"), "words": e.get("message"), "request": dict(e)}
            for e in transcript if e.get("kind") == "failed"]


def stages_of(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each stage of ``view`` in error, as the driver types it (``driver.failures``): (1) one the
    engine did not type as a refusal (``stage_failed``; not a stage that only names one upstream
    already said, "Needs 'x', which failed."), and (2) a typed refusal with no code or no words
    (``wordless``, once for the stage that refused)."""
    from research.swarm import driver

    found = driver.failures(dict(view), list(view.get("stages") or {}))
    failed = {f["stage"] for f in found}
    out: list[dict[str, Any]] = []
    worded: set[str] = set()
    for f in found:
        if f.get("kind") != "refused":
            upstream = NEEDS.match(str(f.get("error") or ""))
            if upstream and upstream.group(1) in failed:
                continue
            out.append({"defect": "stage_failed", "stage": f["stage"], "key": f.get("key"),
                        "words": f.get("error")})
            continue
        refusal = f.get("refusal") or {}
        if refusal.get("code") and refusal.get("message"):
            continue
        at = str(refusal.get("stage") or f["stage"])
        if at not in worded:
            worded.add(at)
            out.append({"defect": "wordless", "stage": at, "key": f.get("key"),
                        "code": refusal.get("code"), "words": refusal.get("message")})
    return out


def wordless_of(transcript: Sequence[Mapping[str, Any]],
                stops: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """(2) Each refusal with no code or no words: a command's, and a stop's that no command of the
    transcript said (a question waiting on a stage that refused, :data:`COMMANDED`)."""
    out = [{"defect": "wordless", "command": e.get("command"), "code": e.get("code"),
            "words": e.get("message"), "request": dict(e)}
           for e in transcript if e.get("kind") == "refused"
           and not (e.get("code") and e.get("message"))]
    for s in stops:
        d = s.get("detail") or {}
        if s.get("kind") == "refused" and d.get("at") not in COMMANDED \
                and not (d.get("code") and d.get("message")):
            out.append({"defect": "wordless", "command": None, "stage": d.get("stage"),
                        "code": d.get("code"), "words": d.get("message"), "stop": dict(s)})
    return out


def dead_ends_of(transcript: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """(3) Each way forward that records something which, taken, is refused again with the same
    code: a refusal whose ways have ids and decisions, and the next command after it that records
    something (:data:`RECORDING`) an answer naming one of them (the way taken), which is refused
    with the refusal's code; or, where the refusal was of a command the walk gives again
    (:data:`RETRIED`: a press of Fit, the seal, the export), the way recorded and the next of that
    command after it refused with the refusal's code. Each keeps the refusal, the way taken and
    the request refused again."""
    out = []
    for i, e in enumerate(transcript):
        if e.get("kind") != "refused":
            continue
        ways = {str(w["id"]) for w in e.get("exits") or [] if w.get("id") and w.get("decision")}
        if not ways:
            continue
        j, taken = next(((j, f) for j, f in enumerate(transcript[i + 1:], start=i + 1)
                         if f.get("command") in RECORDING), (len(transcript), None))
        if taken is None or taken.get("command") != "answer" or not taken.get("ids") \
                or taken["ids"][0] not in ways:
            continue
        again: Mapping[str, Any] | None = taken
        if taken.get("kind") == "ok" and e.get("command") in RETRIED:
            again = next((g for g in transcript[j + 1:] if g.get("command") == e.get("command")),
                         None)
        if again is not None and again.get("kind") == "refused" \
                and again.get("code") == e.get("code"):
            out.append({"defect": "dead_end", "command": e.get("command"), "code": e.get("code"),
                        "way": taken["ids"][0], "words": again.get("message"),
                        "refused": dict(e), "taken": dict(taken), "request": dict(again)})
    return out


def _when(text: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def digest_checks(log_path: Path, fits_path: Path) -> list[dict[str, Any]]:
    """(4) Each fit record of ``fits_path`` (``fits.jsonl``, each line as written) held to the
    decision log ``log_path`` refolded offline (the module docstring): the records appended by the
    record's time (``at``), the longest prefix of them whose records in force
    (``provenance.in_force``) are the seqs the record names (``decisions``), and the plan's digest
    of its fold (``plan_lock.digest(plan_lock.plan_of(fold))``) beside the record's
    (``plan_digest``). A record that names no plan (no prefix keyed its fit, as the server read
    it) is not compared; one whose records no prefix holds is said not to match."""
    from turbotab.core import decisions as D, plan_lock, provenance

    try:
        lines = Path(fits_path).read_text("utf-8").splitlines()
    except OSError:
        return []
    records = sorted(D.DecisionLog(log_path).records(), key=lambda r: r.seq) \
        if Path(log_path).exists() else []
    out = []
    for line in lines:
        try:
            fit = json.loads(line)
        except ValueError:
            continue  # a line an interruption cut
        check: dict[str, Any] = {"record": fit.get("id"), "seq": fit.get("seq"),
                                 "fit_key": fit.get("fit_key"), "at": fit.get("at"),
                                 "plan_digest": fit.get("plan_digest"),
                                 "decisions": fit.get("decisions"), "prefix": None,
                                 "refolded": None}
        if fit.get("plan_digest") is None or fit.get("decisions") is None:
            out.append(check | {"checked": False, "matched": None,
                                "why": "The fit record names no plan: no prefix of the log keyed "
                                       "its fit, as the server read it."})
            continue
        at = _when(fit.get("at"))
        appended = [r for r in records if at is None or r.at <= at]
        named = sorted(int(s) for s in fit["decisions"])
        prefix = next((n for n in range(len(appended), -1, -1) if sorted(
            r.seq for r in appended[:n] if r.id in provenance.in_force(appended[:n])) == named),
            None)
        if prefix is None:
            out.append(check | {"checked": True, "matched": False,
                                "why": f"No prefix of the {len(appended)} records the log held "
                                       f"by the fit's time holds the records in force its record "
                                       f"names ({', '.join(map(str, named)) or 'none'})."})
            continue
        try:
            refolded = plan_lock.digest(plan_lock.plan_of(D.fold(appended[:prefix])))
        except D.Refusal as exc:
            out.append(check | {"checked": True, "matched": False, "prefix": prefix,
                                "why": f"The log's first {prefix} records do not fold: "
                                       f"{exc.code}: {exc.message}"})
            continue
        matched = refolded == fit["plan_digest"]
        out.append(check | {"checked": True, "matched": matched, "prefix": prefix,
                            "refolded": refolded,
                            "why": None if matched else (
                                f"The log's first {prefix} records, those in force the ones the "
                                f"record names, fold to the plan {refolded}, not the record's "
                                f"{fit['plan_digest']}.")})
    return out


def digest_defects(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"defect": "digest", "record": c.get("record"), "seq": c.get("seq"),
             "record_digest": c.get("plan_digest"), "refolded": c.get("refolded"),
             "prefix": c.get("prefix"), "words": c.get("why")}
            for c in checks if c.get("checked") and not c.get("matched")]


def expected_codes() -> dict[str, str]:
    """The codes a second seal and a revisit after the seal are refused with, read from the code
    that raises them: the engine's validator of an opening (``seal._open_seal_once_on_a_fresh_fit``,
    once the seal is open) and the driver's guard of every change after it
    (``driver.guard_sealed``)."""
    from types import SimpleNamespace

    from research.swarm import driver
    from turbotab.core import seal
    from turbotab.core.decisions import Refusal

    codes: dict[str, str] = {}
    try:
        seal._open_seal_once_on_a_fresh_fit(None, {"state": SimpleNamespace(seal_opened=True)})
    except Refusal as exc:
        codes["seal"] = exc.code
    try:
        driver.guard_sealed({"state": {"seal_opened": True}})
    except driver.DriverError as exc:
        codes["revisit"] = exc.code
    return codes


def sealed_twice(session: Any, family: str, revisitable: Sequence[str], rng: Any, reason: str
                 ) -> list[dict[str, Any]]:
    """Once the held-out rows are open: the seal again on ``family``, then a decision of
    ``revisitable`` (drawn) revisited; each with what it said and the code it is expected to be
    refused with (:func:`expected_codes`)."""
    codes = expected_codes()
    again = session.seal(family, f"{reason}: the seal again, once it is open.")
    out = [{"what": "second_seal", "family": family, "expected": codes.get("seal"),
            **arms.said_fields(again)}]
    if revisitable:
        record = draw(rng, list(revisitable))
        said = session.revisit(record, f"{reason}: a decision revisited after the seal.")
        out.append({"what": "revisit_after_seal", "record": record,
                    "expected": codes.get("revisit"), **arms.said_fields(said)})
    return out


def sealed_defects(after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """(5) A second seal or a revisit after the seal that was accepted."""
    return [{"defect": "accepted_after_seal", "what": a.get("what"), "record": a.get("record"),
             "words": f"The {str(a.get('what')).replace('_', ' ')} was accepted; it is refused "
                      f"({a.get('expected')}) once the held-out rows are open."}
            for a in after if a.get("kind") == "ok"]


def noted_of(after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A second seal or a revisit after the seal refused with another code than expected."""
    return [{"what": a.get("what"), "code": a.get("code"), "expected": a.get("expected"),
             "words": a.get("message")}
            for a in after if a.get("kind") == "refused" and a.get("code") != a.get("expected")]


def _key(d: Mapping[str, Any]) -> str:
    return json.dumps([d.get(k) for k in ("defect", "command", "stage", "code", "record", "what",
                                          "way")] + [(d.get("request") or {}).get("n")],
                      default=str)


def classed(walk: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The walk's class and its defects (the module docstring), from what it kept: its
    ``transcript``, its ``stops``, the stages in error met after each press of Fit and each read
    of the results (``stages_met``, :class:`Walked`) and as it ended (``stages``,
    :func:`stages_of`), its fit records held to the log (``checks``), what the second seal and the
    revisit after it said (``after_seal``), and whether the export was given (``reached``). A
    defect met more than once is kept as first met."""
    transcript, stops = walk.get("transcript") or [], walk.get("stops") or []
    stages = [*(walk.get("stages_met") or []), *(walk.get("stages") or [])]
    said = {d.get("stage") for d in stages if d.get("defect") == "stage_failed"}
    for s in stops:
        d = s.get("detail") or {}
        if s.get("kind") == "failed" and d.get("at") == "stage" and d.get("stage") not in said:
            stages.append({"defect": "stage_failed", "stage": d.get("stage"), "key": None,
                           "words": d.get("message"), "stop": dict(s)})
    found = [*failures_of(transcript), *stages, *wordless_of(transcript, stops),
             *dead_ends_of(transcript), *digest_defects(walk.get("checks") or []),
             *sealed_defects(walk.get("after_seal") or [])]
    first: dict[str, dict[str, Any]] = {}
    for d in found:
        first.setdefault(_key(d), d)
    defects = list(first.values())
    kind = "defect" if defects else "clean" if walk.get("reached") else "refused"
    return kind, defects


def stop_of(exc: arms.Stuck) -> dict[str, Any]:
    """Where the walk stopped: by whom (the engine, the server or the driver, as what it said
    names; else the walk's own limit), its class (``arms.Met``'s, else ``stuck``), its words and
    what the class rests on."""
    if isinstance(exc, arms.Met):
        return {"by": exc.detail.get("by") or "engine", "kind": exc.kind, "words": str(exc),
                "detail": dict(exc.detail)}
    return {"by": "walk", "kind": "stuck", "words": str(exc), "detail": None}


# ── one walk ─────────────────────────────────────────────────────────────────


class Walker:
    """One walk's moves on its session (the module docstring), each kept in ``walk``."""

    def __init__(self, session: Walked, policy: arms.Random, rng: Any, walk: dict[str, Any],
                 fields: Any):
        self.session, self.policy, self.rng, self.walk = session, policy, rng, walk
        self.fields = fields
        self.steps: list[arms.Step] = []
        self.reason = f"walk:{policy.seed}"

    def check(self, said: arms.Said, at: str = "walker") -> None:
        """A command that failed ends the walk (the transcript holds it, a defect)."""
        if said.kind == "failed":
            raise arms.Met(f"A command failed: {said.code}: {said.message}", "failed",
                           arms.said_fields(said) | {"at": at})

    def fitted(self) -> arms.Said:
        """Fit pressed as the random arm presses it (``arms.fit_through``), and the results read
        where it is fresh."""
        said = arms.fit_through(self.session, self.policy, self.steps)
        self.check(said, "fit")
        if said.ok:
            self.session.results()
        return said

    def go_on(self) -> arms.Said:
        """After a change: the walk goes on as the random arm walks, and Fit is pressed."""
        arms.walk(self.session, self.policy, steps=self.steps)
        return self.fitted()

    def forward(self) -> None:
        """The random arm's walk to the first Fit, its declarations, Fit and the results."""
        q = arms.walk(self.session, self.policy, steps=self.steps)
        for ids, sets, why in self.policy.declarations(q):
            said = self.session.answer(ids, sets, f"{self.policy.name}: {why}.")
            self.steps.append(arms.Step("declaration", ids[0].split(":", 1)[0], [], ids, sets, why,
                                        said.kind, said.code))
            self.check(said)
        self.fitted()

    # the rounds

    def round(self, n: int) -> None:
        self.session.at = n
        q = self.session.ask()
        view = self.session.view()
        seen = {i for e in self.session.transcript[:-1] for i in (e.get("asked") or {}).get(
            "offered") or []}
        offers = offered_actions(q, view, seen)
        names = [a for a in ACTIONS if offers[a]]
        action = draw(self.rng, names)
        act: dict[str, Any] = {"round": n, "action": action,
                               "offered": {a: len(offers[a]) for a in ACTIONS}}
        self.walk["actions"].append(act)
        reason = f"{self.reason}: round {n}, {action}"
        if action == "fit":
            act["said"] = arms.said_fields(self.fitted())
        elif action in ("revisit", "undo"):
            pick = draw(self.rng, offers[action])
            act["id"] = pick["id"]
            self.after(self.session.revisit(pick["id"], f"{reason}."), act)
        elif action == "stale":
            act["stale_offered"] = {k: sum(1 for c in offers[action] if c["kind"] == k)
                                    for k in STALE}
            pick = stale_of(self.rng, offers[action])
            act.update(id=pick["id"], stale=pick["kind"])
            said = self.session.revisit(pick["id"], f"{reason}: a decision already reverted.") \
                if pick["kind"] == "reverted" else self.session.answer(
                    [pick["id"]], [], f"{reason}: an option an earlier ask offered.")
            act["accepted"] = said.ok
            self.after(said, act)
        else:
            got = self.answered(offers[action], act, view)
            if got is None:
                return
            ids, sets = got
            if action == "preview":
                said = self.session.preview(ids, sets, f"{reason}.")
                act["said"] = arms.said_fields(said)
                self.check(said)
            else:
                self.after(self.session.answer(ids, sets, f"{reason}."), act)

    def answered(self, candidates: Sequence[Mapping[str, Any]], act: dict[str, Any],
                 view: Mapping[str, Any]) -> tuple[list[str], list[str]] | None:
        """One candidate drawn, answered as the driver takes it (:func:`answer_of`); one that
        needs values the sweep's fields do not build is skipped and counted, and another drawn;
        None where none is left (``act["skipped"]``)."""
        left = list(candidates)
        while left:
            pick = draw(self.rng, left)
            got = answer_of(pick["option"], point_of(pick, view), self.fields,
                            self.walk["skipped"])
            if got is not None:
                act.update(id=pick["id"], sets=got[1])
                return got
            left.remove(pick)
        act["skipped"] = True
        return None

    def after(self, said: arms.Said, act: dict[str, Any]) -> None:
        """What the walker's own answer or revisit said: recorded, the walk goes on; refused with
        ways forward that record something, one drawn is taken, and where it is recorded the walk
        goes on."""
        act["said"] = arms.said_fields(said)
        self.check(said)
        if said.ok:
            act["fit"] = arms.said_fields(self.go_on())
            return
        ways = [w for w in said.exits if w.get("id") and w.get("decision")]
        if said.kind != "refused" or not ways:
            return
        way = draw(self.rng, ways)
        took = self.session.answer([str(way["id"])], [], f"{self.reason}: the refusal's way "
                                                         f"forward, {way.get('label')}.")
        act["way"] = {"id": way["id"], "label": way.get("label"), **arms.said_fields(took)}
        self.check(took)
        if took.ok:
            act["fit"] = arms.said_fields(self.go_on())

    # the end of the path

    def settle(self) -> dict[str, Any]:
        """What the fit raises answered and Fit pressed again, three times at most (the random
        arm's), and Fit pressed where the fit is not fresh; the last ask."""
        q: dict[str, Any] = {}
        for _ in range(3):
            before = len(self.steps)
            q = arms.walk(self.session, self.policy, steps=self.steps)
            if len(self.steps) == before:
                break
            said = self.fitted()
            if not said.ok:
                raise arms.Met(f"Fit was not made again: {said.code}: {said.message}",
                               "refused", arms.said_fields(said) | {"at": "fit"})
        status = ((self.session.view().get("stages") or {}).get("fit") or {}).get("status")
        if status != "fresh":
            said = self.fitted()
            if not said.ok:
                raise arms.Met(f"Fit was not made for the end of the path: {said.code}: "
                               f"{said.message}", "refused", arms.said_fields(said) | {"at": "fit"})
            q = self.session.ask()
        return q

    def revisitable(self) -> list[str]:
        return [c["id"] for c in offered_actions(self.session.ask(),
                                                 self.session.view())["revisit"]]

    def sealed(self, q: Mapping[str, Any]) -> None:
        """The held-out rows opened on a family drawn from those fitted, the seal again and a
        decision revisited after it (:func:`sealed_twice`)."""
        families = arms.families_of(q, self.session.view())
        if not families:
            raise arms.Stuck("No family was fitted to declare final.")
        family = self.policy.family(families)
        self.walk["family"] = family
        arms.seal_through(self.session, self.policy, family, self.steps)
        self.walk["end"].append({"what": "seal", "family": family})
        self.session.results()
        self.walk["after_seal"] = sealed_twice(self.session, family, self.revisitable(),
                                               self.rng, self.reason)

    def end(self) -> None:
        self.session.at = "end"
        q = self.settle()
        if self.policy.question.purpose == "prediction" \
                or (q.get("question") or {}).get("key") == "open_seal":
            self.sealed(q)
        if self.policy.question.purpose != "prediction":
            locked = any(r.decision.kind == "lock_plan" for r in sweep.live(self.session.view()))
            self.walk["end"].append({"what": "lock", "locked": locked})
            revisitable = self.revisitable()
            if revisitable:
                act: dict[str, Any] = {"what": "revisit_after_lock", "id": draw(self.rng,
                                                                                revisitable)}
                self.walk["end"].append(act)
                self.after(self.session.revisit(act["id"], f"{self.reason}: a decision "
                                                            f"revisited after the lock."), act)
        self.exported()

    def exported(self) -> None:
        """The export; refused with ways forward that record something, one drawn is taken, the
        walk goes on and the export is asked again (:data:`EXPORT_TRIES`); once a way is
        recorded, the export refused again with the same code is a dead end
        (:func:`dead_ends_of`), and the walk a defect however it ends."""
        for n in range(1, EXPORT_TRIES + 1):
            said = self.session.export()
            act: dict[str, Any] = {"what": "export", "try": n, **arms.said_fields(said)}
            self.walk["end"].append(act)
            self.check(said, "export")
            if said.ok:
                self.walk["reached"] = True
                return
            ways = [w for w in said.exits if w.get("id") and w.get("decision")]
            if not ways:
                raise arms.Met(f"The export was refused: {said.code}: {said.message}", "refused",
                               arms.said_fields(said) | {"at": "export"})
            way = draw(self.rng, ways)
            took = self.session.answer([str(way["id"])], [], f"{self.reason}: the export's way "
                                                             f"forward, {way.get('label')}.")
            act["way"] = {"id": way["id"], "label": way.get("label"), **arms.said_fields(took)}
            self.check(took, "export")
            if not took.ok:
                raise arms.Met(f"The export's way forward was refused: {took.code}: "
                               f"{took.message}", "refused",
                               arms.said_fields(took) | {"at": "export"})
            act["fit"] = arms.said_fields(self.go_on())
        raise arms.Stuck(f"The export was refused {EXPORT_TRIES} times, each way forward taken.")


def walk_one(dataset: str, seed: int, out: Path, *, data: Path | None = None,
             question: arms.Question | None = None) -> dict[str, Any]:
    """One walk (the module docstring) of ``dataset`` at ``seed``, its run under ``out/runs``;
    ``data`` and ``question`` in place of the dataset's table and card (a test's). What it kept,
    its class and its defects, each defect's reproduction written under ``out/defects``."""
    from research.swarm.dry_run import table_of

    out = Path(out).absolute()
    question = question or arms.Question.of_card(arms.card_of(dataset))
    table = Path(data).absolute() if data is not None else table_of(dataset, out / "data" / dataset)
    started = time.monotonic()
    rng = walker_rng(seed)
    walk: dict[str, Any] = {
        "dataset": dataset, "seed": int(seed), "purpose": question.purpose,
        "engine": {"start": stamp()},
        "data": {"path": str(table), "sha256": hashlib.sha256(table.read_bytes()).hexdigest()},
        "run": None, "rounds": rounds_of(rng), "actions": [], "end": [], "after_seal": [],
        "stops": [], "skipped": [], "stages_met": [], "stages": [], "checks": [],
        "reached": False}
    session = Walked(out / "runs")
    policy = arms.Random(question, int(seed))
    walker = Walker(session, policy, rng, walk, LazyFields(session, question, dataset, table))
    try:
        try:
            session.start(table, budget_fits=BUDGET_FITS, budget_seconds=BUDGET_SECONDS,
                          model=f"mechanical-walk-{seed}")
            walk["run"] = session.run
            walker.forward()
            for n in range(1, walk["rounds"] + 1):
                walker.round(n)
            walker.end()
        except arms.Stuck as exc:
            walk["stops"].append(stop_of(exc))
        except Exception as exc:  # noqa: BLE001 - a fault of the walker's own: kept, in its words
            walk["stops"].append({"by": "walker", "kind": "error",
                                  "words": f"{type(exc).__name__}: {str(exc)[:600]}",
                                  "detail": None})
        if session.run is not None:
            try:
                walk["stages"] = stages_of(session.view())
            except Exception as exc:  # noqa: BLE001 - the project could not be read: said
                walk["stages_unread"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        session.stop()
    log: list[Any] = []
    if session.run is not None:
        folder = session.project_folder()
        walk["checks"] = digest_checks(folder / "decisions.jsonl", folder / "fits.jsonl")
        try:
            log = [json.loads(x) for x in (folder / "decisions.jsonl").read_text(
                "utf-8").splitlines() if x.strip()]
        except (OSError, ValueError):
            log = []
    walk["steps"] = [s.json() for s in walker.steps]
    walk["stages_met"] = session.met
    walk["transcript"] = session.transcript
    walk["engine"]["end"] = stamp()
    walk["seconds"] = round(time.monotonic() - started, 1)
    walk["class"], walk["defects"] = classed(walk)
    folder = out / "defects"
    for n, d in enumerate(walk["defects"], start=1):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{dataset}-{seed}-{n}.json"
        d["path"] = str(path)
        path.write_text(json.dumps({
            "dataset": dataset, "seed": int(seed), "defect": d, "words": d.get("words"),
            "request": d.get("request"), "actions": walk["actions"], "end": walk["end"],
            "steps": walk["steps"], "log": log, "data": walk["data"],
            "run": str(session.runs / str(session.run)) if session.run else None,
            "engine": walk["engine"],
            "reproduce": f"python -m research.swarm.walk --dataset {dataset} --seeds {seed} --out "
                         f"<a new folder> walks the same seed; the run's folder keeps its "
                         f"commands and its project."}, indent=1, ensure_ascii=False,
            default=str) + "\n", "utf-8")
    return walk


# ── the report ───────────────────────────────────────────────────────────────


def summary_of(walk: Mapping[str, Any]) -> dict[str, Any]:
    stops = walk.get("stops") or []
    return {"seed": walk.get("seed"), "run": walk.get("run"), "class": walk.get("class"),
            "rounds": walk.get("rounds"), "actions": [a.get("action") for a in
                                                      walk.get("actions") or []],
            "reached": walk.get("reached"), "stopped": stops[0] if stops else None,
            "defects": [d.get("defect") for d in walk.get("defects") or []],
            "skipped": len(walk.get("skipped") or []), "seconds": walk.get("seconds"),
            "record": walk.get("record")}


def report_of(dataset: str, purpose: str, seeds: Sequence[int], walks: Sequence[Mapping[str, Any]],
              end: Mapping[str, Any], seconds: float) -> dict[str, Any]:
    """The walks' report (the module docstring): refused where any walk's engine at its start or
    its end is not ``end``'s (``sweep.engine_changed``); a walk the walker's own fault ended
    (``walker_errors``) is evidence neither way, so a report holding one does not pass."""
    walks = sorted(walks, key=lambda w: int(w["seed"]))
    starts = [dict(s) for w in walks for s in (w["engine"]["start"], w["engine"]["end"])]
    changed = sweep.engine_changed(starts, end) if walks else None
    reported = {int(w["seed"]) for w in walks}
    unreported = [int(s) for s in seeds if int(s) not in reported]
    classes = {k: sum(1 for w in walks if w.get("class") == k) for k in CLASSES}
    checks = [c for w in walks for c in w.get("checks") or []]
    errors = [int(w["seed"]) for w in walks
              if any(s.get("by") == "walker" for s in w.get("stops") or [])]
    return {
        "dataset": dataset, "purpose": purpose, "seeds": [int(s) for s in seeds],
        **{k: end.get(k) for k in ("commit", "engine_tree", "engine_modified")},
        "engine_starts": starts, "engine_changed": changed, "written": sweep.now(),
        "seconds": round(float(seconds), 1),
        "walks": [summary_of(w) for w in walks], "classes": classes,
        "defects": [{"seed": w["seed"], **{k: d.get(k) for k in (
            "defect", "command", "stage", "round", "code", "what", "record", "way", "words",
            "path")}}
            for w in walks for d in w.get("defects") or []],
        "skipped": [{"seed": w["seed"], **s} for w in walks for s in w.get("skipped") or []],
        "noted": [{"seed": w["seed"], **n} for w in walks for n in noted_of(
            w.get("after_seal") or [])],
        "checks": {"records": len(checks), "checked": sum(1 for c in checks if c.get("checked")),
                   "matched": sum(1 for c in checks if c.get("matched"))},
        "walker_errors": errors, "unreported": unreported, "complete": not unreported,
        "passed": not unreported and not errors and not classes["defect"] and changed is None,
    }


def walk_md(report: Mapping[str, Any]) -> str:
    c = report.get("classes") or {}
    lines = [f"# The walks of {report.get('dataset')}", "",
             f"Engine commit `{report.get('commit')}` (engine tree `{report.get('engine_tree')}`"
             f"{', modified' if report.get('engine_modified') else ''}); "
             f"{len(report.get('walks') or [])} walks of {len(report.get('seeds') or [])} seeds; "
             f"clean {c.get('clean', 0)}, refused {c.get('refused', 0)}, defect "
             f"{c.get('defect', 0)}; options skipped for want of values: "
             f"{len(report.get('skipped') or [])}; fit records held to the log: "
             f"{(report.get('checks') or {}).get('checked', 0)} of "
             f"{(report.get('checks') or {}).get('records', 0)}; passed: {report.get('passed')}.",
             "", "| Seed | Rounds | Actions | Class | Defects | Stopped |", "|---|---|---|---|---|---|"]
    for w in report.get("walks") or []:
        stop = w.get("stopped") or {}
        said = f"{stop.get('by')}: {str(stop.get('words'))[:160]}" if stop else ""
        lines.append(f"| {w.get('seed')} | {w.get('rounds')} | {', '.join(w.get('actions') or [])} "
                     f"| {w.get('class')} | {', '.join(w.get('defects') or [])} | "
                     f"{said.replace('|', '/')} |")
    if report.get("engine_changed"):
        lines += ["", "## The engine changed while the walks ran", "", str(report["engine_changed"])]
    if report.get("walker_errors"):
        lines += ["", "## Walks the walker's own fault ended (evidence neither way)", "",
                  ", ".join(str(s) for s in report["walker_errors"])]
    if report.get("unreported"):
        lines += ["", "## Seeds not reported", "",
                  ", ".join(str(s) for s in report["unreported"])]
    if report.get("defects"):
        lines += ["", "## Defects", ""]
        lines += [f"- seed {d['seed']}, `{d['defect']}`"
                  + (f" ({d['command']})" if d.get("command") else "")
                  + (f" at the {d['stage']} stage" if d.get("stage") else "")
                  + (f" (met in round {d['round']})" if isinstance(d.get("round"), int) else "")
                  + (f", `{d['code']}`" if d.get("code") else "")
                  + f": {str(d.get('words'))[:300]} ({d.get('path')})" for d in report["defects"]]
    if report.get("noted"):
        lines += ["", "## Refused after the seal with another code than expected", ""]
        lines += [f"- seed {n['seed']}, the {str(n['what']).replace('_', ' ')}: `{n['code']}`, "
                  f"not `{n['expected']}`: {str(n.get('words'))[:200]}" for n in report["noted"]]
    if report.get("skipped"):
        lines += ["", "## Options skipped for want of values", ""]
        lines += [f"- seed {s['seed']}, `{s.get('point')}={s.get('option')}`: "
                  f"{str(s.get('why'))[:200]}" for s in report["skipped"]]
    return "\n".join(lines) + "\n"


def _kept(path: Path) -> dict[str, Any] | None:
    try:
        walk = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return walk if isinstance(walk, dict) and walk.get("class") in CLASSES else None


def run(dataset: str, seeds: Sequence[int], out: Path, *,
        walker: Callable[..., dict[str, Any]] = walk_one, purpose: str | None = None,
        data: Path | None = None, question: arms.Question | None = None,
        log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The walks of ``dataset`` at ``seeds`` into ``out`` (the module docstring), each seed already
    written skipped: ``walk.json`` and ``walk.md``. ``walker`` in place of :func:`walk_one`,
    ``purpose`` of the card's, ``data`` and ``question`` of the dataset's (a test's)."""
    out = Path(out).absolute()
    if purpose is None:
        purpose = (question or arms.Question.of_card(arms.card_of(dataset))).purpose
    started = time.monotonic()
    walks = []
    for seed in seeds:
        path = out / "walks" / f"{dataset}-{seed}.json"
        kept = _kept(path)
        if kept is None:
            if log is not None:
                log(f"{dataset} seed {seed}")
            kept = dict(walker(dataset, int(seed), out, data=data, question=question))
            kept["record"] = str(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(kept, indent=1, ensure_ascii=False, default=str) + "\n",
                            "utf-8")
        walks.append(kept)
    report = report_of(dataset, purpose, seeds, walks, stamp(), time.monotonic() - started)
    out.mkdir(parents=True, exist_ok=True)
    (out / "walk.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str)
                                   + "\n", "utf-8")
    (out / "walk.md").write_text(walk_md(report), "utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.walk",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", required=True, help="a-b, each from a to b, or one seed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        seeds = seeds_of(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    def log(text: str) -> None:
        sys.stderr.write(f"{sweep.now()} {text}\n")
        sys.stderr.flush()

    report = run(args.dataset, seeds, args.out, log=log)
    sys.stdout.write(json.dumps({k: report[k] for k in ("dataset", "classes", "passed", "complete",
                                                        "engine_changed")}
                                | {"defects": len(report["defects"])}) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

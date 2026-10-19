"""Assemble one run's corpus record from the server alone (DRIVER_PLAN D3; SWARM_PLAN §4).

After a run, with its server still up, ``python -m research.swarm.assemble research/runs/<run id>``
reads the run directory the driver wrote and the server's own data, builds the run's record
(:class:`record.Record`), validates it and writes it once:

* ``research/corpus/<run id>.json`` when it validates;
* ``research/corpus/failed/<run id>.json``, the error beside what was assembled, when it does not:
  a failed run, never an edited record. Neither file is ever written over.

**The run directory** (``research/runs/<run id>/``, written by the driver, ``research.swarm.driver``):

* ``driver.json`` (``driver.cmd_start``): ``run`` (its id), ``pid`` (the server's process),
  ``port``, ``project``, ``budget`` (the fit budget), ``started`` and ``ended``; and, where the
  driver or its launcher writes them, the run's ``design``, ``analyst_model``, ``effort``,
  ``prompt_version``, ``transcript`` (the analyst agent's transcript JSONL, relative to the run
  directory or absolute), ``dataset`` (the intake's ``id``, ``control``, ``fixture``, ``known``
  answer, ``codebook`` and ``card``) and ``help`` (each intervention a person made). An argument
  given to :func:`assemble` takes the place of the file's.
* ``commands.jsonl`` (``driver.main``): one line per command the driver ran, oldest first, each
  with ``command`` (its name), ``at`` and ``summary`` (``driver.summary`` of the JSON it printed):
  a summary holding ``refused`` (``code``, ``message``, ``exits``) is a refusal met, the server's
  or the driver's, and one holding ``exported`` with exit status 0 is the export returned. No other
  key is read.
* ``home/``: the server's home.

**The server's reads.** The analyst log comes first, since any later read can render a claim and
each render is logged. Then the project with its decision log, the fit records, the results, which
decisions mattered (waiting while that stage computes), the materiality ledger, the ingest, the
outcome and the participant flow, and the export. A read of the project, the analyst log, the fit
records, the results or the materiality ledger that fails leaves a failed run
(:class:`RunUnreadable`), never a record that reads as if the route had served nothing. Every
number in the record is a served value; nothing is recomputed. These reads count what they serve
as shown, as the stage routes do; the run is over, so no decision of the run is marked by them.

**A fit's change from the fit before** (:func:`change_of`) is read both ways from the records in
force at each (``GET …/fits``' ``decisions``), against the decision log. A record that left or came
into force is credited to the one revert of its chain (a revert of it, a revert of that revert, …)
that this fit read and the fit before did not; a revert appended by the time a fit completed
counts as read by it wherever reading it leaves the record as that fit holds it, and one before
the fit's last record in force always does, since a fit's records are the longest prefix of the
log that keys its fit. Where its chain accounts for no shift, a record that left force is credited
to a new record answering its question, and one that came into force to the revert that withdrew
what had replaced it; any other came into force new. A record no record of the log is found to
credit, or that several may be (more than one revert of its chain, or a new answer beside a
revert of its chain this fit may not have read), is kept with none. A revert is never in force,
so a fit whose only change is a revert has that revert as its change and its note as its reason.

Each credit is then held to the server's own rule (:class:`Prefixes`: ``provenance.in_force``
over ``decisions.fold``, for the longest prefix of the log appended by each fit's completion that
holds its records in force, as ``runs.keyed_prefix`` reads a fit's answers, where every prefix
appended by then that holds them folds to the same answers): a record the fit before had read
already is credited with nothing of this fit's, and a fit's whole change is one record's only
where that record alone, after the records the fit before read, holds this fit's records in force
and folds to its answers. Otherwise the records this fit read since the fit before moved together
and are its change, each with its note among the reasons. Where those are not known (no prefix
holds a fit's records, or two that do fold to different answers, so which one keyed it is not
known), neither is the change: ``changed`` is null (:data:`UNACCOUNTED`) and no shift is credited
to any record.

**A decision's materiality** is the engine's (:class:`record.Materiality`):

* under Predict, the paired movement ``GET …/fits`` serves for the first fit the record was in
  force in (a revert: the first whose change holds it), where the fit's whole change from the fit
  before is that one record; with others, or with a record credited to none, it moved together, so
  it has no measure of its own, and the why names them;
* under Estimate, for the exclusion rules or covariates' adjustment answers a record holds in
  force (each covariate's answer its own question, as the server keeps it), ``GET …/mattered``'
  specifications that change the choices those answers recorded, beside the primary;
* otherwise none, and why: the route's own words where it gave them.

**The result** under Estimate is the fit the lock in force was recorded on: the one fit record
``GET …/fits`` marks so (``locked``: its key is the one the answers at the lock give, the server's
own rule, whatever its plan's digest, which differs when an answer the fit does not read came
between the fit and the press). Where no record, or more than one, is marked, the result is
refused and says so, as a result the record cannot know (``refused_kind``), never as one the run
did not report; it is never the last fit before the lock, nor a fit matched by its digest.

**Examined options.** A decision answers one or more questions (:func:`answers_of`: each answer
the server keeps apart is its own question, so a preview of another covariate's adjustment answer
is not an option examined for this covariate's). For each, an option is examined for the decision
when it was previewed (the analyst log's previews) while the decision was being chosen or while it
stood, after the question's previous answer and before its next (a preview followed by keeping the
answer examined it too), or when it was an earlier answer in force in a fit that completed before
the decision (SWARM_PLAN §3: reading a card does not count). It is an alternative when it is not
the option chosen and, for a preview, the preview was not refused. On the split, which the held-out
draw and the validation scheme both answer, an option is the one chosen where it leaves the slot
as the chosen answer does, each as the log stood where it was given: a scheme answer leaves the
draw then in force with its scheme (:func:`same_answer`), so another draw previewed while a scheme
answer stands is an alternative to it.

**Spend** (:func:`spend`): the transcript's assistant lines, once per message id. A message
streams as several lines with one id, the same input counts and a growing output count, so each
count is the largest its lines give: fresh is input plus cache creation plus output, cached is
cache reads.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import ValidationError

from research.swarm import record as R

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "research" / "corpus"
FAILED = "failed"
API = "/api"

DRIVER_FILE = "driver.json"
COMMANDS_FILE = "commands.jsonl"
META_KEYS = ("design", "analyst_model", "effort", "prompt_version", "transcript", "dataset", "help")

# A repair of values below a detection limit answers that reading (``truths.answers``).
BELOW_DETECTION = "below_detection__"
# The kinds that dispose of a finding: each writes the finding's entry, under its id.
FINDING_KINDS = ("apply_repair", "defer_finding", "dismiss_finding")
# Where the server keeps each covariate's adjustment answer (``decisions.register_kind``), and the
# question it answers, keyed as a fixture's declared truth keys it (``adjust:<column>``).
ADJUSTMENT_SLOT = "adjustment"
# The choices ``GET …/mattered`` varies, by the decision kind that recorded them (mattered v0).
MATTERED_KINDS = ("set_exclusions", "set_adjustment")
# The stage statuses after which ``GET …/mattered`` is waited for no longer.
SETTLED = ("fresh", "error", "blocked")

NOT_IN_A_FIT = "No fit completed while this record was in force."
UNACCOUNTED = ("The decision log, read by the server's own rule, does not account for this fit's "
               "records from the fit before's, so no record is credited with the change between "
               "them.")
NOT_SERVED = "its records are not served"
FIRST_FIT = "This record was in force at the first fit, which has no earlier fit to compare with."
LOOSE = "by no record found between the two fits"
NO_MOVEMENT = "The fit records serve no movement for the fit this record first took effect in."
NOT_VARIED = ("The curve of which decisions mattered varies the exclusion rules and the adjustment "
              "set (its v0), so it has no specification for this decision.")
WITHDRAWN = ("Withdrawn by a revert: the curve served is drawn for the plan in force now, so it "
             "does not vary this record's choices.")
NO_SPEC = "The curve estimated no specification that changes this decision's choices:"
NOT_HELD = "No answer this record gave is in force, so the curve served varies none of them."
NOT_CHOSEN = ("The curve served varies no choice recorded by the answers this record holds in "
              "force.")
NOT_DEPARTED = ("The curve served has no specification that changes a choice recorded by the "
                "answers this record holds in force.")


def _records(seqs: Sequence[int]) -> str:
    listed = ", ".join(f"#{s}" for s in seqs)
    return f"record {listed}" if len(seqs) == 1 else f"records {listed}"


def _together(others: Sequence[int], fit_seq: int, left: Sequence[int] = (),
              back: Sequence[int] = ()) -> str:
    """Why a record has no measure of its own: at fit ``fit_seq`` it moved with ``others``, or
    with records that left force (``left``) or came back into it (``back``) credited to none."""
    text = f"It took effect at fit #{fit_seq}"
    if others:
        text += f" together with {_records(others)}"
    loose = [f"{_records(left)} left force"] if left else []
    loose += [f"{_records(back)} returned to force"] if back else []
    if loose:
        text += f"{',' if others else ''} while {' and '.join(loose)} {LOOSE}"
    return (f"{text}: the fit records measure the change between two fits, not each record's "
            f"share of it.")


def _restored(fit_seq: int, by: int | None) -> str:
    if by is None:
        return f"It took effect at fit #{fit_seq}, back in force {LOOSE}."
    return (f"It took effect at fit #{fit_seq}, restored by record #{by}, which is credited with "
            f"that change.")


def unserved_why(fit: Mapping[str, Any]) -> str:
    """The engine's why for a fit whose records ``GET …/fits`` does not serve: the route's own
    words for withholding its numbers, else the record's for its movement."""
    return str(fit.get("withheld") or fit.get("movement_why") or NOT_SERVED)


def _not_known_in(fits: Sequence[Mapping[str, Any]], revert: bool = False) -> str:
    """Why a record found in no fit whose records are served is not said to be in none: ``fits``,
    whose records are not served, may hold it (a revert: may have it as their change)."""
    one = len(fits) == 1
    listed = ", ".join(f"#{f['seq']}" for f in fits)
    whys = " ".join(dict.fromkeys(unserved_why(f) for f in fits))
    named = f"Fit {listed}'s records are" if one else f"Fits {listed}' records are"
    what = ("whether this revert is in its change from the fit before" if revert and one else
            "whether this revert is in their changes from the fits before" if revert else
            "whether this record was in force at it" if one else
            "whether this record was in force at them")
    return f"{named} not served, so {what} is not known: {whys}"


def _before_unserved(fit: Mapping[str, Any]) -> str:
    return (f"The fit before this record's first fit, fit #{fit['seq']}, has records the server "
            f"does not serve, so the change between them is not known: {unserved_why(fit)}")


def _unseen_revert(target: int | None) -> str:
    withdrawn = f"record #{target}" if target is not None else "the record it names"
    return f"No fit's change from the fit before it is credited to this revert of {withdrawn}."


def no_locked_fit(lock: int) -> str:
    return (f"No fit record is marked as the fit the lock at record #{lock} was recorded on, so "
            f"that fit is not known from the fit records.")


def locked_twice(lock: int, ids: Sequence[str]) -> str:
    listed = ", ".join(f"`{i}`" for i in ids)
    return (f"Fit records {listed} are each marked as the fit the lock at record #{lock} was "
            f"recorded on, so which one it was is not known from the fit records.")


def _superseded(seqs: Sequence[int]) -> str:
    those = "that record's" if len(seqs) == 1 else "those records'"
    return (f"Superseded by {_records(seqs)}, which answered again each question this record "
            f"answered: the curve served is drawn for the plan in force now, so it varies {those} "
            f"choices, not this one's.")


# ── the run directory ────────────────────────────────────────────────────────


@dataclass
class RunDir:
    path: Path
    driver: dict[str, Any]
    commands: list[dict[str, Any]]


def read_run_dir(path: str | Path) -> RunDir:
    """The driver's ``driver.json`` and ``commands.jsonl`` (the module docstring), as written."""
    path = Path(path)
    driver = json.loads((path / DRIVER_FILE).read_text("utf-8"))
    commands: list[dict[str, Any]] = []
    log = path / COMMANDS_FILE
    if log.exists():
        for line in log.read_text("utf-8").splitlines():
            if line.strip():
                commands.append(json.loads(line))
    return RunDir(path, driver, commands)


# ── spend ────────────────────────────────────────────────────────────────────


class SpendRefused(ValueError):
    """The transcript's spend cannot be counted once per message."""


def spend(transcript: str | Path) -> R.Spend:
    """The analyst agent's tokens (the module docstring): once per ``message.id``, each count the
    largest its lines give. Refused when a line with usage carries no message id, since it could
    not be counted once."""
    counts: dict[str, dict[str, int]] = {}
    fields = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
              "output_tokens")
    unnamed = 0
    for line in Path(transcript).read_text("utf-8").splitlines():
        if not line.strip():
            continue
        message = json.loads(line).get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("usage"), Mapping):
            continue
        mid = message.get("id")
        if not mid:
            unnamed += 1
            continue
        usage = message["usage"]
        best = counts.setdefault(str(mid), dict.fromkeys(fields, 0))
        for f in fields:
            best[f] = max(best[f], int(usage.get(f) or 0))
    if unnamed:
        raise SpendRefused(f"{unnamed:,} transcript lines with usage carry no message id, so they "
                           f"cannot be counted once each.")
    fresh = sum(c["input_tokens"] + c["cache_creation_input_tokens"] + c["output_tokens"]
                for c in counts.values())
    cached = sum(c["cache_read_input_tokens"] for c in counts.values())
    return R.Spend(fresh=fresh, cached=cached, messages=len(counts))


# ── the server's reads ───────────────────────────────────────────────────────


@dataclass
class Read:
    status: int
    body: Any  # the JSON served, or the bytes of the export

    @property
    def ok(self) -> bool:
        return self.status == 200


@dataclass
class Served:
    analyst_log: Read
    project: Read
    fits: Read
    results: Read
    mattered: Read
    ledger: Read
    stages: dict[str, Read]
    export: Read
    health: Read


def _json(response: Any) -> Read:
    try:
        body = response.json()
    except ValueError:
        body = {"error": {"code": "unreadable", "message": response.text[:500], "exits": []}}
    return Read(response.status_code, body)


def fetch(client: Any, pid: str, *, mattered_wait: float = 600.0, poll: float = 0.25) -> Served:
    """Every route the record reads, analyst log first (the module docstring). ``client`` is an
    HTTP client on the server's root (``httpx.Client`` or FastAPI's ``TestClient``)."""
    base = f"{API}/projects/{pid}"
    log = _json(client.get(f"{base}/analyst-log"))
    project = _json(client.get(base))
    fits = _json(client.get(f"{base}/fits"))
    results = _json(client.get(f"{base}/results"))
    end = time.monotonic() + mattered_wait
    while True:
        mattered = _json(client.get(f"{base}/mattered"))
        status = (mattered.body or {}).get("status") if mattered.ok else None
        if not mattered.ok or status in SETTLED or time.monotonic() > end:
            break
        time.sleep(poll)
    ledger = _json(client.get(f"{base}/materiality"))
    stages = {s: _json(client.get(f"{base}/stages/{s}")) for s in ("ingest", "target_info",
                                                                    "cohort")}
    response = client.get(f"{base}/export")
    export = Read(response.status_code, response.content) if response.status_code == 200 \
        else _json(response)
    health = _json(client.get(f"{API}/health"))
    return Served(log, project, fits, results, mattered, ledger, stages, export, health)


# ── what a decision answers ──────────────────────────────────────────────────


class UnreadDecision(ValueError):
    """A decision of a keyed kind that the engine's own models do not read, so where the server
    keeps its answers is not known."""


def _dumped(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {k: _dumped(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dumped(v) for v in value]
    return value


def _question(slot: str, key: str | None) -> str:
    """A server write's question: the slot, and the entry's key where the slot is keyed; a
    covariate's adjustment answer as ``adjust:<column>``."""
    if key is None:
        return slot
    return f"adjust:{key}" if slot == ADJUSTMENT_SLOT else f"{slot}:{key}"


def answers_of(decision: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The questions ``decision`` answers, each with its value.

    * A reading as ``<reading>:<column>``, as a fixture's declared truth keys it: a role (by
      ``confirm_role``, ``set_roles`` or a confirmation), a unit and its days, a code or an amount,
      …; a disposition of a finding of values below a detection limit as
      ``detection_limit:<column>``; a codebook's readings, and the codebook as
      ``codebooks:<codebook>``.
    * Any other decision as the server keeps its answers (``provenance._writes``: each slot it
      writes, and the entry's key where the slot is keyed, so two answers to one entry are one
      question and answers to two entries are two): an unkeyed slot by its name (``models``,
      ``exclusions``, ``target``, …); each entry of a keyed one as ``<slot>:<key>`` (a finding's
      disposition as ``findings:<finding id>``, a column's form as ``exposure_forms:<column>``,
      a modifier as ``modifications:<modifier>``, …), a covariate's adjustment answer as
      ``adjust:<column>``. Its value, for a keyed slot, is the entry the server keeps (so a
      column's form is one value whichever kind answered it), else every field of the decision but
      its kind.
    * A revert as ``revert:<record id>``; a kind the server keeps nowhere, by its kind.

    Raises :class:`UnreadDecision` for a decision of a keyed kind that does not read as one."""
    from turbotab.core import decisions as D

    kind = str(decision.get("kind"))
    rest = {k: v for k, v in decision.items() if k != "kind"}
    if kind == "confirm_reading":
        return [(f"{decision['reading']}:{decision['column']}", decision.get("value"))]
    if kind == "confirm_readings":
        return [(f"{i['reading']}:{i['column']}", i.get("value"))
                for i in decision.get("items") or []]
    if kind == "confirm_role":
        return [(f"role:{decision['column']}", decision.get("role"))]
    if kind == "set_roles":
        return [(f"role:{c}", r) for c, r in (decision.get("roles") or {}).items()]
    if kind == "set_column_unit":
        return [(f"unit:{decision['column']}", decision.get("unit")),
                (f"day_count:{decision['column']}", str(decision.get("days")))]
    if kind == "import_codebook":
        return [*((f"{i['reading']}:{i['column']}", i.get("value"))
                  for i in decision.get("items") or []),
                (f"codebooks:{decision.get('codebook')}",
                 {k: v for k, v in rest.items() if k != "items"})]
    finding = str(decision.get("finding_id") or "")
    if kind in FINDING_KINDS and finding.startswith(BELOW_DETECTION):
        value = decision.get("option") if kind == "apply_repair" else {
            k: v for k, v in rest.items() if k != "finding_id"}
        return [(f"detection_limit:{finding[len(BELOW_DETECTION):]}", value)]
    if kind == "revert":
        return [(f"revert:{decision.get('decision_id')}", rest)]
    if kind in D._ENTRIES or kind in D._KEYS:
        try:
            model = D.parse_decision(dict(decision))
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc") or ())
            raise UnreadDecision(f"A `{kind}` decision does not read as one: `{where}`: "
                                 f"{first.get('msg')}") from exc
        if kind in D._ENTRIES:
            return [(_question(slot, key), _dumped(value))
                    for slot, key, value in D._ENTRIES[kind](model)]
        slot = D._SLOT_FOR[kind](model) if kind in D._SLOT_FOR else D.SLOTS[kind]
        return [(_question(slot, D._KEYS[kind](model)), _dumped(D._SLOT_VALUE[kind](model)))]
    return [(D.SLOTS.get(kind, kind), rest)]


def same_option(examined: Any, chosen: Any) -> bool:
    """Whether an examined value is the option chosen. A field the examined value leaves empty
    (None) is not compared: a preview logs the decision as asked, and the server fills such fields
    when it records one (an opening's outcome and scores), so the same option previewed and then
    recorded is not an alternative to itself."""
    if isinstance(examined, Mapping) and isinstance(chosen, Mapping):
        return all(examined.get(k) is None or examined.get(k) == chosen.get(k)
                   for k in set(examined) | set(chosen))
    return examined == chosen


# The slot both the held-out draw (``set_split``) and the validation scheme (``set_validation``,
# written onto the draw: ``decisions._scheme_onto_the_draw``) answer.
SPLIT_SLOT = "split"


def left_in_split(decision: Mapping[str, Any], standing: Mapping[str, Any] | None = None
                  ) -> dict[str, Any] | None:
    """What a split answer leaves in the split slot as the slot stands (``standing``: the split in
    force where it was given, as the server keeps it, or None), each field as the server fills it:
    a ``set_split`` the whole split (its draw and its scheme); a ``set_validation`` the scheme it
    writes onto the draw standing (``decisions._scheme_onto_the_draw``: the draw kept), or the
    scheme alone (``SetValidation.scheme()``) where no draw is known to stand. None for any other
    decision, or one that does not read as one."""
    from turbotab.core import decisions as D

    kind = decision.get("kind")
    if kind not in ("set_split", "set_validation"):
        return None
    try:
        model = D.parse_decision(dict(decision))
    except ValidationError:
        return None
    if kind == "set_split":
        return model.model_dump(exclude={"kind"})
    return {**standing, **model.scheme()} if standing else model.scheme()


def same_answer(question: str, examined: Mapping[str, Any], value: Any,
                chosen: Mapping[str, Any], chosen_value: Any,
                standing: tuple[Mapping[str, Any] | None, Mapping[str, Any] | None] = (None, None)
                ) -> bool:
    """Whether the examined decision's answer to ``question`` is the option chosen. On the split
    slot each is compared by what it leaves there as the slot stood where it was given
    (:func:`left_in_split`; ``standing``: the split in force there, the examined one's and the
    chosen one's): a scheme answer leaves the draw standing with its scheme, so a draw and a
    scheme answer are the same option where they leave the same split. Where a scheme answer's
    draw is not known, on the fields both write. Any other answer by :func:`same_option`."""
    if question == SPLIT_SLOT:
        a, b = left_in_split(examined, standing[0]), left_in_split(chosen, standing[1])
        if a is not None and b is not None:
            return all(a[k] == b[k] for k in set(a) & set(b))
    return same_option(value, chosen_value)


def phase_of(rec: Mapping[str, Any]) -> str:
    """The record's phase (:data:`record.Phase`), by the decision log's own marks."""
    if rec.get("post_seal"):
        return "after_seal"
    return "after_estimates" if rec.get("after_estimates") else "before"


def _at(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def cancelled(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each record a revert withdraws at the end of ``records``, by id, with its revert's id: the
    decision log's own rule (``decisions.reverted``), latest first, so a revert counts only when no
    later revert withdrew it."""
    out: dict[str, str] = {}
    for r in sorted(records, key=lambda r: r["seq"], reverse=True):
        if r["id"] not in out and r["decision"].get("kind") == "revert":
            out.setdefault(str(r["decision"].get("decision_id")), r["id"])
    return out


# ── the attempts ─────────────────────────────────────────────────────────────


CHANGE = ("changed", "withdrawn", "reinstated")
AMBIGUOUS = object()  # a shift several records made, which no one of them is credited with


class Prefixes:
    """The server's own rule over the decision log (DRV-INT): the records in force in each prefix
    of it (``provenance.in_force`` over ``decisions.fold``), and the prefix a fit read. A fit's
    records are those in force in the longest prefix of the log appended by its completion whose
    answers key its fit (``runs.keyed_prefix``). Two prefixes holding the same records in force
    can fold to different answers (``in_force`` counts a draw as superseded by the scheme written
    onto it, and ``fold`` keeps the draw, so a revert of that draw changes the split and no record
    in force). So the prefix a fit read is the longest appended by then whose records in force are
    the fit's only where every such prefix folds to the same answers; otherwise it is not known
    (:meth:`read_through`). ``log`` is None where a record does not read as the server's
    (``decisions.DecisionRecord``): then no prefix is known."""

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        from turbotab.core.decisions import DecisionRecord

        fields = DecisionRecord.model_fields
        try:
            self.log: list[Any] | None = sorted(
                (DecisionRecord.model_validate({k: v for k, v in r.items() if k in fields})
                 for r in records), key=lambda r: r.seq)
        except (ValidationError, TypeError, AttributeError):
            self.log = None
        self._held: dict[int, tuple[frozenset[int], Any] | None] = {}

    def held(self, log: Sequence[Any]) -> tuple[frozenset[int], Any] | None:
        """``log``'s records in force (their seqs) and its folded answers, or None where the log
        refuses to fold (a revert of what is not there)."""
        from turbotab.core import decisions as D, provenance

        try:
            state = D.fold(log)
        except D.Refusal:
            return None
        ids = provenance.in_force(log)
        return frozenset(r.seq for r in log if r.id in ids), state.model_dump(mode="json")

    def at(self, n: int) -> tuple[frozenset[int], Any] | None:
        if n not in self._held:
            self._held[n] = self.held(self.log[:n]) if self.log is not None else None
        return self._held[n]

    def read_through(self, fit: Mapping[str, Any]) -> int | None:
        """How many records of the log ``fit`` read (the class docstring): the longest prefix
        appended by its completion that holds its records in force, where every such prefix folds
        to the same answers. None where its records are not served, where no such prefix holds
        them, or where two fold to different answers (a draw a revert withdrew after a scheme
        superseded it), since which one keyed the fit is then not known."""
        served = fit.get("decisions")
        if self.log is None or served is None:
            return None
        done = _at(fit["at"]) if fit.get("at") else None
        appended = next((i for i, r in enumerate(self.log) if done is not None and r.at > done),
                        len(self.log))
        want = frozenset(served)
        holding = [n for n in range(appended, -1, -1)
                   if (held := self.at(n)) is not None and held[0] == want]
        if not holding or any(self.at(n)[1] != self.at(holding[0])[1] for n in holding[1:]):
            return None
        return holding[0]

    def between(self, before: Mapping[str, Any], fit: Mapping[str, Any]) -> list[int] | None:
        """The records ``fit`` read that ``before`` (the fit before it) did not, by their seqs; None
        where those are not known (either fit's prefix is not, or ``fit`` read no more of the log
        than ``before``)."""
        n_before, n_now = self.read_through(before), self.read_through(fit)
        if n_before is None or n_now is None or n_now <= n_before:
            return None
        return [r.seq for r in self.log[n_before:n_now]]

    def alone(self, before: Mapping[str, Any], fit: Mapping[str, Any], seq: int
              ) -> tuple[bool, list[int] | None]:
        """Whether ``fit``'s records differ from those of ``before`` (the fit before it) by record
        ``seq`` alone, by the server's rule: ``seq`` is among the records ``fit`` read that
        ``before`` did not, and that record alone, after those ``before`` read, holds ``fit``'s
        records in force and folds to its answers. With it, :meth:`between` (the records that
        moved together)."""
        between = self.between(before, fit)
        if between is None or seq not in between:
            return False, between
        n_before = self.read_through(before)
        alone = self.held([*self.log[:n_before], next(r for r in self.log if r.seq == seq)])
        return alone is not None and alone == self.at(self.read_through(fit)), between


def change_of(before: Mapping[str, Any] | None, fit: Mapping[str, Any],
              records: Sequence[Mapping[str, Any]], prefixes: Prefixes | None = None
              ) -> dict[str, Any]:
    """``fit``'s change from ``before``, the fit record before it (None for the first), both ways
    (the module docstring; :class:`record.Attempt`'s ``changed``, ``withdrawn`` and
    ``reinstated``).

    A fit's records are the longest prefix of the log appended by its completion whose answers
    key its fit (``runs.appended_by``, ``runs.keyed_prefix``), so a revert appended by the time the
    fit before completed may have been read by it or not. Each record that left or came into force
    has its chain of reverts (a revert of it, a revert of that revert, …), appended by this fit's
    completion: each revert of the chain read flips it, so it is in force where an even number of
    them are read. Of the chain's reverts appended by the time each fit completed, each fit read
    the most that leave the record as that fit holds it (the prefix is the longest), and at least
    those before its last record in force (the prefix reaches it). Where this fit read exactly one
    revert of the chain more than the fit before, that revert is credited; where it read several,
    or the chain branches, no one record is. Where the chain does not account for the shift (none
    of it read since), a record that left force is credited to a new record that answers its
    question (one that replaced it), and one that came into force to the revert that withdrew what
    had replaced it. Beside a new answer to its question, a record's chain is credited with its
    leaving only where this fit read every revert of the chain.

    Then the server's own rule (``prefixes``, :class:`Prefixes` over ``records`` by default):
    where the records this fit read since the fit before are not known (:meth:`Prefixes.between`),
    the change is null and no shift is credited; a shift credited to a record the fit before had
    read is credited to none; a change credited to records outside those this fit read since the
    fit before is those records; and a change credited to one record alone stands only where
    :meth:`Prefixes.alone` confirms it, else it is the records this fit read since the fit before
    (they moved together)."""
    now = fit.get("decisions")
    if before is None:
        return {"changed": sorted(now) if now is not None else None, "withdrawn": [],
                "reinstated": []}
    was = before.get("decisions")
    if now is None or was is None:
        return {"changed": None, "withdrawn": [], "reinstated": []}
    by_seq = {r["seq"]: r for r in records}
    seq_of = {r["id"]: r["seq"] for r in records}
    floor = max(was, default=0)
    done = _at(fit["at"]) if fit.get("at") else None
    done_before = _at(before["at"]) if before.get("at") else None

    def by(r: Mapping[str, Any], when: datetime | None) -> bool:
        return when is None or _at(r["at"]) <= when

    reverts: dict[int, list[Mapping[str, Any]]] = {}  # each record's reverts, by its seq
    for r in records:
        if r["decision"].get("kind") == "revert" and r["decision"].get("decision_id") in seq_of \
                and by(r, done):
            reverts.setdefault(seq_of[r["decision"]["decision_id"]], []).append(r)

    whole: set[int] = set()  # the records whose shift this fit's whole chain read accounts for

    def chained(seq: int, in_force_before: bool, in_force_now: bool) -> Any:
        """The revert of ``seq``'s chain this fit read and the fit before did not: its seq, None
        where the chain accounts for no shift, :data:`AMBIGUOUS` where it accounts for one made
        by several reverts, or where the chain branches."""
        chain: list[Mapping[str, Any]] = []
        link = seq
        while reverts.get(link):
            if len(reverts[link]) > 1:
                return AMBIGUOUS
            chain.append(reverts[link][0])
            link = chain[-1]["seq"]

        def read(appended: int, in_force: bool, last: int) -> int:
            """How many reverts of the chain a fit read: the most of those ``appended`` that leave
            the record as the fit holds it, and at least each one before ``last``, the fit's
            last record in force (its prefix reaches it); -1 where none does."""
            k = appended if (appended % 2 == 0) == in_force else appended - 1
            return k if k >= sum(1 for r in chain if r["seq"] < last) else -1

        k_before = read(sum(1 for r in chain if by(r, done_before)), in_force_before, floor)
        k_now = read(len(chain), in_force_now, max(now, default=0))
        if k_before < 0 or k_now <= k_before:
            return None
        if k_now == len(chain):
            whole.add(seq)
        return chain[k_now - 1]["seq"] if k_now == k_before + 1 else AMBIGUOUS

    def questions(seq: int) -> set[str]:
        return {q for q, _ in answers_of(by_seq[seq]["decision"])} if seq in by_seq else set()

    gone, came = sorted(set(was) - set(now)), sorted(set(now) - set(was))
    withdrawn: dict[int, Any] = {x: chained(x, True, False) for x in gone}
    reinstated: dict[int, int | None] = {}
    new: list[int] = []
    for s in came:
        found = chained(s, False, True)
        if found is None:  # restored by the revert that withdrew what had replaced it
            found = next((withdrawn[y] for y in gone if isinstance(withdrawn[y], int)
                          and y > s and questions(y) & questions(s)), None)
        if found is not None or s < floor:  # before ``floor``: in the fit before's prefix
            reinstated[s] = found if isinstance(found, int) else None
        else:
            new.append(s)
    for x in gone:
        # Replaced by a new answer to its question. A revert of the chain stays credited beside a
        # new answer only where every revert of the chain was read (it withdrew the record, the
        # answer or no answer); where it was taken as unread, the new answer may be why.
        replaced = next((s for s in new if s > x and questions(x) & questions(s)), None)
        if withdrawn[x] is None:
            withdrawn[x] = replaced
        elif withdrawn[x] is AMBIGUOUS or (replaced is not None and x not in whole):
            withdrawn[x] = None
    prefixes = prefixes or Prefixes(records)
    read = prefixes.between(before, fit)
    if read is None:
        # Which records this fit read since the fit before is not known by the server's own rule
        # (:meth:`Prefixes.between`), so none is credited with its change or with a shift of it.
        return {"changed": None,
                "withdrawn": [{"seq": x, "by": None} for x in withdrawn],
                "reinstated": [{"seq": x, "by": None} for x in reinstated]}
    # A record the fit before read already is credited with no shift of this fit's.
    for shifted in (withdrawn, reinstated):
        for x, b in shifted.items():
            if isinstance(b, int) and b not in read:
                shifted[x] = None
    credited = {b for b in (*withdrawn.values(), *reinstated.values()) if b is not None}
    changed = sorted(set(new) | credited)
    if not set(changed) <= set(read):
        changed = sorted(read)  # credited by the log's chains, not by the server's prefixes
    shifts = [*withdrawn.values(), *reinstated.values()]
    if len(changed) == 1 and None not in shifts:
        # The fit's whole change would be one record's: only where the server's own rule confirms
        # it (:meth:`Prefixes.alone`); else the records this fit read since the fit before moved
        # together, and where those are not known, neither is the change.
        ok, between = prefixes.alone(before, fit, changed[0])
        if not ok:
            changed = sorted(between) if between else None
    return {"changed": changed,
            "withdrawn": [{"seq": x, "by": b} for x, b in withdrawn.items()],
            "reinstated": [{"seq": x, "by": b} for x, b in reinstated.items()]}


def attempts_of(fits: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
                ) -> list[dict[str, Any]]:
    """Each fit record as served, with its change from the fit before it (:func:`change_of`) and
    the notes of the records credited with it (``reasons``)."""
    notes = {r["seq"]: r.get("note") for r in records}
    prefixes = Prefixes(records)
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(fits):
        change = change_of(fits[i - 1] if i else None, rec, records, prefixes)
        reasons = [{"seq": s, "note": notes[s]} for s in change["changed"] or () if notes.get(s)]
        keep = {k: rec.get(k) for k in R.Attempt.model_fields if k not in (*CHANGE, "reasons")}
        out.append({**keep, **change, "reasons": reasons})
    return out


# ── the decisions ────────────────────────────────────────────────────────────


def examined_of(rec: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                previews: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
                prefixes: Prefixes | None = None) -> list[dict[str, Any]]:
    """The options examined for each question ``rec`` answers (the module docstring), previews
    first, then fits, each in time order: each preview of the question made while this answer was
    being chosen or while it stood (after the question's previous answer, before its next), and
    each earlier answer in force in a fit that completed before this one was recorded. An option
    examined the same way again (previewed again with the same refusal, or held by another earlier
    record in a fit) is listed once, at its first. On the split, each answer is compared with the
    one chosen by what it leaves in the slot as the log stood where it was given (``prefixes``,
    :class:`Prefixes` over ``records`` by default: the split the log folds to through the records
    before it, for a preview those appended by its time)."""
    at = _at(rec["at"])
    chosen = dict(answers_of(rec["decision"]))
    stood = _standing(records, prefixes, read=SPLIT_SLOT in chosen)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    window: dict[str, tuple[datetime | None, datetime | None]] = {}
    for question in chosen:
        answering = [r for r in records if r["seq"] != rec["seq"]
                     and question in dict(answers_of(r["decision"]))]
        before = [_at(r["at"]) for r in answering if r["seq"] < rec["seq"]]
        after = [_at(r["at"]) for r in answering if r["seq"] > rec["seq"]]
        window[question] = (max(before) if before else None, min(after) if after else None)

    def once(question: str, value: Any, how: str, refused: str | None = None) -> bool:
        key = (question, how, json.dumps(value, sort_keys=True, default=str), refused)
        if key in seen:
            return False
        seen.add(key)
        return True

    def within(question: str, when: datetime) -> bool:
        start, end = window[question]
        return (start is None or when > start) and (end is None or when < end)

    for p in previews:
        if p.get("decision") is None:
            continue
        for question, value in answers_of(p["decision"]):
            if question in chosen and within(question, _at(p["at"])) \
                    and once(question, value, "preview", p.get("refused")):
                out.append({"question": question, "value": value, "how": "preview", "at": p["at"],
                            "refused": p.get("refused"),
                            "alternative": p.get("refused") is None
                            and not same_answer(question, p["decision"], value, rec["decision"],
                                                chosen[question],
                                                (stood(when=_at(p["at"])), stood(rec["seq"])))})
    for earlier in records:
        if earlier["seq"] >= rec["seq"]:
            continue
        fitted = next((a for a in attempts if earlier["seq"] in (a.get("decisions") or ())
                       and _at(a["at"]) < at), None)
        if fitted is None:
            continue
        for question, value in answers_of(earlier["decision"]):
            if question in chosen and once(question, value, "fit"):
                out.append({"question": question, "value": value, "how": "fit",
                            "at": fitted["at"], "seq": earlier["seq"], "attempt": fitted["id"],
                            "alternative": not same_answer(
                                question, earlier["decision"], value, rec["decision"],
                                chosen[question], (stood(earlier["seq"]), stood(rec["seq"])))})
    return out


def _standing(records: Sequence[Mapping[str, Any]], prefixes: Prefixes | None = None,
              read: bool = True) -> Callable[..., Any]:
    """The split in force before a record (by its seq) or at a time (a preview's: the records
    appended by then), as ``decisions.fold`` gives it over the log (``prefixes``, :class:`Prefixes`
    over ``records`` by default); None where the log does not fold there or holds no draw, and
    everywhere when ``read`` is false (no split answer is compared, so the log is not folded)."""
    prefixes = (prefixes or Prefixes(records)) if read else None

    def stood(seq: int | None = None, when: datetime | None = None) -> Any:
        if prefixes is None or prefixes.log is None:
            return None
        n = (next((i for i, r in enumerate(prefixes.log) if r.seq >= seq), len(prefixes.log))
             if seq is not None else
             next((i for i, r in enumerate(prefixes.log) if r.at > when), len(prefixes.log)))
        held = prefixes.at(n)
        return held[1].get(SPLIT_SLOT) if held is not None else None

    return stood


def fits_materiality(rec: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]],
                     records: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The paired movement of the first fit ``rec`` was in force in (a revert, which is never in
    force: the first fit whose change holds it), where the fit's whole change from the fit before
    is ``rec`` alone (:func:`change_of`); else why there is none. ``records``: the decision log, to
    name what a revert withdrew. A record is said to be in no fit (a revert: in no fit's change)
    only where every fit's records are served; otherwise whether it was is not known, and the
    engine's why says so (:func:`unserved_why`). Where the first fit's change from the fit before
    is not known, the why says which (the fit before's records are not served, or the log does not
    account for the change, :data:`UNACCOUNTED`)."""
    seq = rec["seq"]
    decision = rec.get("decision") or {}
    unserved = [a for a in attempts if a.get("decisions") is None]
    if decision.get("kind") == "revert":
        first = next((i for i, a in enumerate(attempts) if seq in (a.get("changed") or ())), None)
        if first is None:
            if unserved:
                return {"source": None, "why": _not_known_in(unserved, revert=True)}
            if any(a.get("changed") is None for a in attempts):
                return {"source": None, "why": UNACCOUNTED}
            target = next((r["seq"] for r in records if r["id"] == decision.get("decision_id")),
                          None)
            return {"source": None, "why": _unseen_revert(target)}
    else:
        first = next((i for i, a in enumerate(attempts) if seq in (a.get("decisions") or ())),
                     None)
        if first is None:
            return {"source": None, "why": _not_known_in(unserved) if unserved else NOT_IN_A_FIT}
    fit = attempts[first]
    if first == 0:
        return {"source": None, "attempt": fit["id"], "why": fit.get("movement_why") or FIRST_FIT}
    changed = fit.get("changed")
    if changed is None:
        before = attempts[first - 1]
        return {"source": None, "attempt": fit["id"],
                "why": _before_unserved(before) if before.get("decisions") is None
                else UNACCOUNTED}
    if seq not in changed:  # restored by a revert, which is credited with it
        by = next((s.get("by") for s in fit.get("reinstated") or () if s.get("seq") == seq), None)
        return {"source": None, "attempt": fit["id"], "why": _restored(fit["seq"], by)}
    others = [s for s in changed if s != seq]
    left = [s["seq"] for s in fit.get("withdrawn") or () if s.get("by") is None]
    back = [s["seq"] for s in fit.get("reinstated") or () if s.get("by") is None]
    if others or left or back:
        return {"source": None, "attempt": fit["id"],
                "why": _together(others, fit["seq"], left, back)}
    moved = fit.get("movement") or []
    if any(m.get("movement") is not None for m in moved):
        return {"source": "fits", "attempt": fit["id"], "movement": moved}
    whys = " ".join(f"`{m.get('family')}`: {m['why']}" for m in moved if m.get("why"))
    return {"source": None, "attempt": fit["id"],
            "why": whys or fit.get("movement_why") or fit.get("withheld") or NO_MOVEMENT}


def in_force_of(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Each question's record in force at the end of the log: its last answer that no revert
    withdrew (:func:`cancelled`)."""
    reverted = cancelled(records)
    out: dict[str, int] = {}
    for r in records:
        if r["id"] in reverted:
            continue
        for question, _ in answers_of(r["decision"]):
            out[question] = r["seq"]
    return out


def choice_question(key: str, kind: str) -> str | None:
    """The question whose answer recorded a choice ``GET …/mattered`` varies: the exclusion rules
    (``exclusion:<i>``) are the one ``exclusions`` answer; a covariate adjusted for or left out
    (``adjusted:<column>``, ``left_out:<column>``) is that covariate's ``adjust:<column>``."""
    if kind == "set_exclusions":
        return "exclusions"
    prefix, _, column = key.partition(":")
    if kind == "set_adjustment" and prefix in ("adjusted", "left_out") and column:
        return _question(ADJUSTMENT_SLOT, column)
    return None


def mattered_materiality(rec: Mapping[str, Any], mattered: Read, in_force: Mapping[str, int],
                         withdrawn: Sequence[str] = ()) -> dict[str, Any] | None:
    """Under Estimate, ``GET …/mattered``' specifications for ``rec``'s choices: those of the
    choices whose question ``rec`` is the answer in force for now (the curve served is drawn for
    the plan in force; each covariate's answer is its own question, :func:`choice_question`); the
    route's own refusal or status where it drew none; None for a decision the curve does not vary.
    ``withdrawn``: the ids of the records a revert withdrew (:func:`cancelled`)."""
    kind = rec["decision"].get("kind")
    if kind not in MATTERED_KINDS:
        return None
    if rec["id"] in withdrawn:
        return {"source": None, "why": WITHDRAWN}
    questions = [q for q, _ in answers_of(rec["decision"])]
    mine = {q for q in questions if in_force.get(q) == rec["seq"]}
    if not mine:
        later = sorted({in_force[q] for q in questions if in_force.get(q) is not None})
        return {"source": None, "why": _superseded(later) if later else NOT_HELD}
    if not mattered.ok:
        return {"source": None,
                "why": str(((mattered.body or {}).get("error") or {}).get("message")
                           or NOT_VARIED)}
    curve = (mattered.body or {}).get("curve")
    if not curve:
        status = (mattered.body or {}).get("status")
        error = (mattered.body or {}).get("error")
        return {"source": None, "why": str(error or f"The curve's stage is {status}, with no "
                                                    f"curve drawn.")}
    choices = [c for c in curve.get("choices") or [] if c.get("decision") == kind
               and choice_question(str(c.get("key")), kind) in mine]
    primary = next((s for s in curve.get("specs") or [] if s.get("primary")), None)
    departures: list[dict[str, Any]] = []
    for c in choices:
        for entry, estimated in ([(s, True) for s in curve.get("specs") or []]
                                 + [(n, False) for n in curve.get("not_estimated") or []]):
            pick = (entry.get("picks") or {}).get(c["key"])
            if entry.get("primary") or pick is None or pick == c.get("primary"):
                continue
            departures.append({"choice": c["key"], "label": c.get("label"), "option": pick,
                               "spec": entry if estimated else None,
                               "not_estimated": None if estimated else entry.get("reason")})
    if not departures:
        return {"source": None, "why": NOT_CHOSEN if not choices else NOT_DEPARTED}
    if all(d["spec"] is None for d in departures):
        options = {c["key"]: {o["key"]: o.get("label") for o in c.get("options") or []}
                   for c in choices}
        reasons = " ".join(f"`{d['label']}` {options[d['choice']].get(d['option'], d['option'])}: "
                           f"{d['not_estimated']}" for d in departures)
        return {"source": None, "why": f"{NO_SPEC} {reasons}"}
    return {"source": "mattered", "primary": primary, "departures": departures}


def decisions_of(records: Sequence[Mapping[str, Any]], previews: Sequence[Mapping[str, Any]],
                 attempts: Sequence[Mapping[str, Any]], *, purpose: str | None,
                 mattered: Read) -> list[dict[str, Any]]:
    in_force = in_force_of(records)
    withdrawn = set(cancelled(records))
    prefixes = Prefixes(records)
    out: list[dict[str, Any]] = []
    for rec in records:
        materiality = fits_materiality(rec, attempts, records)
        if purpose == "inference" and materiality.get("source") is None:
            drawn = mattered_materiality(rec, mattered, in_force, withdrawn)
            if drawn is not None:
                materiality = drawn
        out.append({
            "id": rec["id"], "seq": rec["seq"], "at": rec["at"],
            "kind": rec["decision"].get("kind"), "option": rec["decision"],
            "answers": [{"question": q, "value": v} for q, v in answers_of(rec["decision"])],
            "reason": rec.get("note"),
            "source": "app" if rec.get("recorded_by") == "turbotab" else "analyst",
            "phase": phase_of(rec), "post_seal": bool(rec.get("post_seal")),
            "after_estimates": bool(rec.get("after_estimates")), "early": rec.get("early"),
            "examined": examined_of(rec, records, previews, attempts, prefixes),
            "materiality": materiality,
        })
    return out


# ── the result ───────────────────────────────────────────────────────────────


def _last(records: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    """The last record of ``kind`` no revert withdrew (:func:`cancelled`; the server withdraws a
    lock no estimate was served under by a revert of its own, ``plan_lock.current_lock``)."""
    reverted = cancelled(records)
    return next((r for r in reversed(records) if r["decision"].get("kind") == kind
                 and r["id"] not in reverted), None)


def result_of(purpose: str | None, records: Sequence[Mapping[str, Any]],
              attempts: Sequence[Mapping[str, Any]], results: Read) -> dict[str, Any]:
    """The reported result, once (:class:`record.Result`), or why there is none, with its kind
    (``refused_kind``): ``not_reported``, the run reported none (the seal never opened, the plan
    never locked, no goal); ``not_served``, the engine served none (the opening recorded no
    held-out scores and its fit serves none; the locked fit serves no exposure rows and withholds
    none); ``not_known``, the record cannot say what the run reported (no fit record, or several,
    is marked the one the lock was recorded on; the route withholds the locked fit's rows now; the
    opening recorded no scores while its fit serves held-out scores)."""
    base: dict[str, Any] = {"purpose": purpose, "attempt": None, "plan_digest": None,
                            "decisions": None}
    if purpose == "prediction":
        opening = next((r for r in records if r["decision"].get("kind") == "open_seal"), None)
        if opening is None:
            return {**base, "refused": "The held-out rows were never opened.",
                    "refused_kind": "not_reported"}
        d = opening["decision"]
        target = d.get("target")
        opened = [a for a in attempts if a.get("target") == target and not a.get(
            "changed_after_seal") and (a.get("at_opening") or {}).get("seq") == opening["seq"]]
        fit = opened[-1] if opened else None
        out = {**base, "opening": opening["seq"], "family": d.get("family"),
               "metric": d.get("metric"), "n_holdout": d.get("n_holdout"),
               "scores": d.get("scores") or None, "intervals": None, "intervals_why": None}
        if fit is not None:
            out.update(attempt=fit["id"], plan_digest=fit.get("plan_digest"),
                       decisions=fit.get("decisions"))
        doc = results.body if results.ok else {}
        at_opening = (doc or {}).get("at_opening") or {}
        family = next((f for f in (doc or {}).get("families") or []
                       if f.get("family") == d.get("family")), None)
        if at_opening.get("seq") == opening["seq"] and at_opening.get("current") and family \
                and family.get("intervals"):
            out["intervals"] = family["intervals"]
        else:
            out["intervals_why"] = str(at_opening.get("note") or (doc or {}).get("withheld")
                                       or "The results serve no intervals for the opening's "
                                          "scores.")
        if not out["scores"]:
            # None recorded: none served at the opening, unless its fit serves held-out scores,
            # which the record cannot say the run did not report.
            served = fit is not None and any((fit.get("holdout") or {}).values())
            out.update(scores=None, refused="The opening recorded no held-out scores.",
                       refused_kind="not_known" if served else "not_served")
        return out
    if purpose == "inference":
        lock = _last(records, "lock_plan")
        if lock is None:
            return {**base, "refused": "The plan was never locked, so no estimate was served.",
                    "refused_kind": "not_reported"}
        digest = lock["decision"].get("digest")
        out = {**base, "lock": lock["seq"], "plan_digest": digest}
        held = [a for a in attempts if a.get("locked")]
        if not held:
            return {**out, "refused": no_locked_fit(lock["seq"]),
                    "refused_kind": "not_known"}
        if len(held) > 1:
            return {**out, "refused": locked_twice(lock["seq"], [a["id"] for a in held]),
                    "refused_kind": "not_known"}
        fit = held[0]
        out.update(attempt=fit["id"], decisions=fit.get("decisions"))
        effects = {f: rows for f, rows in (fit.get("effects") or {}).items() if rows}
        if not effects:
            # Withheld by the route's gate now: what was served during the run is not known.
            withheld = fit.get("withheld")
            return {**out, "refused": str(withheld or "The locked fit serves no exposure rows."),
                    "refused_kind": "not_known" if withheld else "not_served"}
        return {**out, "effects": effects}
    return {**base, "refused": "No goal was declared.", "refused_kind": "not_reported"}


# ── the rest of the record ───────────────────────────────────────────────────


def _bundle(data: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if not n.endswith("/")}
    checklist = next((n for n in sorted(files) if n.startswith("checklist/")
                      and n.endswith(".json")), None)
    return {"provenance": json.loads(files["provenance.json"]),
            "methods": files["methods.md"].decode("utf-8") if "methods.md" in files else None,
            "checklist": json.loads(files[checklist]) if checklist else None}


def _summary(command: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = command.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def export_of(served: Served, commands: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any],
                                                                               dict[str, Any]]:
    """The record's export, and the bundle's provenance (empty when the export is refused).
    ``reached``: an ``export`` command of the run returned the bundle (its summary's
    ``exported``, exit status 0)."""
    reached = any(c.get("command") == "export" and _summary(c).get("exit") == 0
                  and _summary(c).get("exported") and _refusal(c) is None for c in commands)
    if not served.export.ok:
        error = (served.export.body or {}).get("error") or {}
        return ({"reached": reached, "sha256": None, "methods": None, "checklist": None,
                 "refused": {"code": error.get("code"), "message": error.get("message")}}, {})
    data = served.export.body
    parts = _bundle(data)
    return ({"reached": reached, "sha256": hashlib.sha256(data).hexdigest(),
             "methods": parts["methods"], "checklist": parts["checklist"], "refused": None},
            parts["provenance"])


def _refusal(command: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The refusal a command met, as its summary keeps it (the server's or the driver's), or
    None."""
    error = _summary(command).get("refused")
    if isinstance(error, Mapping) and error.get("code") and error.get("message") is not None:
        return error
    return None


def surfacings_of(commands: Sequence[Mapping[str, Any]], ledger: Read) -> dict[str, Any]:
    """Each refusal the run met, with its ways forward by their labels (by their ids where the
    summary keeps none), and the command that followed it; the materiality ledger's rows."""
    refusals = []
    for i, c in enumerate(commands):
        error = _refusal(c)
        if error is None:
            continue
        nxt = commands[i + 1] if i + 1 < len(commands) else None
        exits = [e.get("label") or e.get("id") if isinstance(e, Mapping) else e
                 for e in error.get("exits") or []]
        refusals.append({"at": c.get("at"), "command": str(c.get("command")),
                         "code": str(error["code"]), "message": str(error["message"]),
                         "exits": [str(e) for e in exits if e is not None],
                         "then": str(nxt.get("command")) if nxt is not None else None})
    return {"refusals": refusals, "concerns": (ledger.body or {}).get("rows") or []}


def regime_of(state: Mapping[str, Any], stages: Mapping[str, Read]) -> dict[str, Any]:
    ingest = stages["ingest"].body.get("artifact") if stages["ingest"].ok else None
    target = stages["target_info"].body.get("artifact") if stages["target_info"].ok else None
    cohort = stages["cohort"].body.get("artifact") if stages["cohort"].ok else None
    event = state.get("event")
    events = None
    if target and event is not None:
        events = next((c["count"] for c in target.get("classes") or []
                       if str(c.get("value")) == str(event)), None)
    return {
        "n_rows": (ingest or {}).get("n_rows"), "n_analyzed": (cohort or {}).get("n_final"),
        "task": (target or {}).get("task"), "event": event, "events": events,
        "missing": {c["name"]: c["n_missing"] for c in (ingest or {}).get("columns") or []
                    if c.get("n_missing")},
        "design": {k: state.get(k) for k in ("survey", "clusters", "grain", "repeat_kind", "unit",
                                             "design")},
        "lenses": list(state.get("lens") or []),
    }


def dataset_of(meta: Mapping[str, Any], provenance: Mapping[str, Any],
               stages: Mapping[str, Read], summary: Mapping[str, Any]) -> dict[str, Any]:
    given = dict(meta.get("dataset") or {})
    if provenance.get("inputs"):
        files = [{k: f.get(k) for k in ("role", "name", "sha256", "bytes", "url", "release_date",
                                       "license")} for f in provenance["inputs"]]
    else:
        ingest = stages["ingest"].body.get("artifact") if stages["ingest"].ok else {}
        files = [{"role": "table", "name": summary.get("source_name"),
                  "sha256": (ingest or {}).get("source_sha256"),
                  "bytes": (ingest or {}).get("source_bytes")}]
    return {"id": given.get("id"), "control": given.get("control"),
            "fixture": given.get("fixture"), "known": given.get("known"), "files": files,
            "codebook": given.get("codebook"), "card": given.get("card")}


def question_of(state: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    card = (meta.get("dataset") or {}).get("card") or {}
    return {"purpose": state.get("purpose"), "target": state.get("target"),
            "estimand": state.get("estimand"), "intended_use": state.get("intended_use"),
            "stated": card.get("question"), "paper": card.get("paper")}


def run_of(run: RunDir, meta: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]],
           provenance: Mapping[str, Any], health: Read) -> dict[str, Any]:
    d = run.driver
    transcript = meta.get("transcript")
    spent, why = None, None
    if not transcript:
        why = "No transcript of the analyst agent was given."
    else:
        path = Path(transcript)
        path = path if path.is_absolute() else run.path / path
        try:
            spent = spend(path).model_dump()
        except (OSError, SpendRefused, ValueError) as exc:
            why = f"The transcript's spend was not counted: {exc}"
    seconds = [a["seconds"] for a in attempts if a.get("seconds") is not None]
    engine = provenance.get("engine") or {}
    h = health.body if health.ok else {}
    return {
        "id": d.get("run") or run.path.name, "design": meta.get("design"),
        "engine_commit": engine.get("commit"), "engine_modified": engine.get("modified"),
        "engine_version": (h or {}).get("version"), "analyst_model": meta.get("analyst_model"),
        "effort": meta.get("effort"), "prompt_version": meta.get("prompt_version"),
        "started": d.get("started"), "ended": d.get("ended"), "spend": spent, "spend_why": why,
        "fits": len(attempts), "fit_seconds": float(sum(seconds)),
        "fits_unseen": len(attempts) - len(seconds), "fit_budget": d.get("budget"),
        "help": [str(h_) if not isinstance(h_, Mapping) else json.dumps(h_, sort_keys=True)
                 for h_ in meta.get("help") or []],
        "server": {"pid": d.get("pid"), "port": d.get("port"), "mode": (h or {}).get("mode"),
                   "workers": (h or {}).get("workers")},
    }


class RunUnreadable(RuntimeError):
    """The server serves no full record of this run: a route that refuses only a project it does
    not hold (the project, its fit records, its results, its materiality ledger), or its analyst
    log (a server not in analyst mode refuses it), cannot be read, or a decision of the log does
    not read as one (:class:`UnreadDecision`). The run is a failed run."""


# The routes every record reads, which refuse only a project the server does not hold: a read of
# any of them that fails leaves no record (``Served``'s field, and its name in the error).
REQUIRED = (("project", "The project"), ("analyst_log", "The analyst log"),
            ("fits", "The fit records"), ("results", "The results"),
            ("ledger", "The materiality ledger"))


def build(served: Served, run: RunDir, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The record, as a dict, from what the server served and the run directory: nothing is
    validated here (:func:`write` validates)."""
    for field, name in REQUIRED:
        read = getattr(served, field)
        if not read.ok:
            raise RunUnreadable(f"{name} could not be read (HTTP {read.status}): {read.body}")
    view = served.project.body
    records = sorted(view.get("decisions") or [], key=lambda r: r["seq"])
    state = view.get("state") or {}
    purpose = state.get("purpose")
    fit_records = (served.fits.body or {}).get("records") or []
    attempts = attempts_of(fit_records, records)
    log = served.analyst_log.body
    export, provenance = export_of(served, run.commands)
    return {
        "version": R.VERSION,
        "run": run_of(run, meta, attempts, provenance, served.health),
        "dataset": dataset_of(meta, provenance, served.stages, view.get("summary") or {}),
        "question": question_of(state, meta),
        "regime": regime_of(state, served.stages),
        "decisions": decisions_of(records, log.get("previews") or [], attempts, purpose=purpose,
                                  mattered=served.mattered),
        "attempts": attempts,
        "result": result_of(purpose, records, attempts, served.results),
        "claims": list(log.get("claims") or []),
        "surfacings": surfacings_of(run.commands, served.ledger),
        "export": export,
        "replication": None,
    }


# ── writing ──────────────────────────────────────────────────────────────────


def write(raw: Mapping[str, Any], run_id: str, corpus: Path = CORPUS) -> tuple[Path, bool]:
    """Validate ``raw`` and write it once: ``<corpus>/<run id>.json`` when it validates, else
    ``<corpus>/failed/<run id>.json`` with the error beside what was assembled. Returns the path
    and whether the record validated. Refuses (``FileExistsError``) to write over either."""
    try:
        rec = R.validate(raw)
    except ValidationError as exc:
        path = corpus / FAILED / f"{run_id}.json"
        body = {"run_id": run_id, "error": str(exc),
                "errors": json.loads(exc.json(include_url=False)), "assembled": raw}
        ok = False
    else:
        path = corpus / f"{run_id}.json"
        body = rec.model_dump(mode="json", by_alias=True)
        ok = True
    for taken in (corpus / f"{run_id}.json", corpus / FAILED / f"{run_id}.json"):
        if taken.exists():
            raise FileExistsError(f"{taken} is already written; a run's record is written once.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    return path, ok


def meta_of(run: RunDir, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """The run's own fields (:data:`META_KEYS`): the given ones, else driver.json's."""
    given = {k: v for k, v in (given or {}).items() if v is not None}
    return {k: given.get(k, run.driver.get(k)) for k in META_KEYS}


def assemble(run_dir: str | Path, *, meta: Mapping[str, Any] | None = None, client: Any = None,
             corpus: Path = CORPUS, mattered_wait: float = 600.0) -> tuple[Path, bool]:
    """Assemble, validate and write the record of the run in ``run_dir`` (the module docstring).
    ``client``: an HTTP client on the server's root; by default one on ``127.0.0.1`` at the
    run's port."""
    run = read_run_dir(run_dir)
    run_id = str(run.driver.get("run") or Path(run_dir).name)
    pid = run.driver.get("project")
    own = None
    if client is None:
        import httpx

        own = client = httpx.Client(base_url=f"http://127.0.0.1:{run.driver['port']}",
                                    timeout=600.0)
    try:
        served = fetch(client, str(pid), mattered_wait=mattered_wait)
    finally:
        if own is not None:
            own.close()
    try:
        raw: dict[str, Any] = build(served, run, meta_of(run, meta))
    except (RunUnreadable, UnreadDecision) as exc:
        raw = {"unreadable": str(exc)}
    return write(raw, run_id, corpus)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.assemble",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--design", type=int, choices=(1, 2, 3))
    parser.add_argument("--analyst-model")
    parser.add_argument("--effort")
    parser.add_argument("--prompt-version")
    parser.add_argument("--transcript")
    parser.add_argument("--dataset", type=Path, help="a JSON file: the intake's id, control, "
                                                     "fixture, known answer, codebook and card")
    parser.add_argument("--helped", action="append", default=None,
                        help="an intervention a person made (repeat for each)")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args(argv)
    meta = {"design": args.design, "analyst_model": args.analyst_model, "effort": args.effort,
            "prompt_version": args.prompt_version, "transcript": args.transcript,
            "dataset": json.loads(args.dataset.read_text("utf-8")) if args.dataset else None,
            "help": args.helped}
    path, ok = assemble(args.run_dir, meta=meta, corpus=args.corpus)
    print(json.dumps({"record": str(path), "valid": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

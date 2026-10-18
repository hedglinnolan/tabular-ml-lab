"""Assemble one run's corpus record from the server alone (DRIVER_PLAN D3; SWARM_PLAN §4).

After a run, with its server still up, ``python -m research.swarm.assemble research/runs/<run id>``
reads the run directory the driver wrote and the server's own data, builds the run's record
(:class:`record.Record`), validates it and writes it once:

* ``research/corpus/<run id>.json`` when it validates;
* ``research/corpus/failed/<run id>.json``, the error beside what was assembled, when it does not:
  a failed run, never an edited record. Neither file is ever written over.

**The run directory** (``research/runs/<run id>/``, written by the driver):

* ``driver.json``: ``run_id``, ``pid`` (the server's process), ``port``, ``project_id``,
  ``fit_budget``, ``start`` and ``end``; and, where the driver or its launcher writes them, the
  run's ``design``, ``analyst_model``, ``effort``, ``prompt_version``, ``transcript`` (the analyst
  agent's transcript JSONL, relative to the run directory or absolute), ``dataset`` (the intake's
  ``id``, ``control``, ``fixture``, ``known`` answer, ``codebook`` and ``card``) and ``help`` (each
  intervention a person made). An argument given to :func:`assemble` takes the place of the file's.
* ``commands.jsonl``: one line per command the driver ran, oldest first, each with ``command``
  (its name), ``at`` and ``output`` (the JSON it printed); an output holding the server's ``error``
  object (``code``, ``message``, ``exits``) is a refusal met. No other key is read.
* ``home/``: the server's home.

**The server's reads.** The analyst log comes first, since any later read can render a claim and
each render is logged. Then the project with its decision log, the fit records, the results, which
decisions mattered (waiting while that stage computes), the materiality ledger, the ingest, the
outcome and the participant flow, and the export. Every number in the record is a served value;
nothing is recomputed. These reads count what they serve as shown, as the stage routes do; the run
is over, so no decision of the run is marked by them.

**A decision's materiality** is the engine's (:class:`record.Materiality`):

* under Predict, the paired movement ``GET …/fits`` serves for the first fit the record was in
  force in, where it is the only record that changed since the fit before; with others it moved
  together, so it has no measure of its own, and the why names them;
* under Estimate, for an exclusion rule or an adjustment set in force when the curve was drawn,
  ``GET …/mattered``' specifications that change its choices, beside the primary;
* otherwise none, and why: the route's own words where it gave them.

**Examined options.** A decision answers one or more questions (:func:`answers_of`). For each, an
option is examined for the decision when it was previewed (the analyst log's previews) while the
decision was being chosen or while it stood, after the question's previous answer and before its
next (a preview followed by keeping the answer examined it too), or when it was an earlier answer
in force in a fit that completed before the decision (SWARM_PLAN §3: reading a card does not
count). It is an alternative when it is not the option chosen and, for a preview, the preview was
not refused.

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
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from research.swarm import record as R

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "research" / "corpus"
FAILED = "failed"
API = "/api"

DRIVER_FILE = "driver.json"
COMMANDS_FILE = "commands.jsonl"
META_KEYS = ("design", "analyst_model", "effort", "prompt_version", "transcript", "dataset", "help")

# The fields that say which question a decision answers, by kind; any other kind answers one
# question, its kind's. A reading is keyed as a fixture's declared truth keys it (``answers_of``).
ADDRESS: dict[str, tuple[str, ...]] = {
    "apply_repair": ("finding_id",), "defer_finding": ("finding_id",),
    "dismiss_finding": ("finding_id",), "confirm_sweep": ("stage", "sweep"),
    "respond_diagnostic": ("exposure", "check"), "set_exposure_form": ("column",),
    "set_adjustment": ("exposure",), "set_causal": ("exposure",), "set_usual_intake": ("nutrient",),
    "set_outcome_order": ("column",), "set_outcome_scale": ("column",),
    "set_outcome_unit": ("column",), "join_files": ("file",), "import_codebook": ("codebook",),
}
# A repair of values below a detection limit answers that reading (``truths.answers``).
BELOW_DETECTION = "below_detection__"
# The choices ``GET …/mattered`` varies, by the decision kind that recorded them (mattered v0).
MATTERED_KINDS = ("set_exclusions", "set_adjustment")
# The stage statuses after which ``GET …/mattered`` is waited for no longer.
SETTLED = ("fresh", "error", "blocked")

NOT_IN_A_FIT = "No fit completed while this record was in force."
FIRST_FIT = "This record was in force at the first fit, which has no earlier fit to compare with."
NO_MOVEMENT = "The fit records serve no movement for the fit this record first took effect in."
NOT_VARIED = ("The curve of which decisions mattered varies the exclusion rules and the adjustment "
              "set (its v0), so it has no specification for this decision.")
WITHDRAWN = ("Withdrawn by a revert: the curve served is drawn for the plan in force now, so it "
             "does not vary this record's choices.")


def _together(others: Sequence[int], fit_seq: int) -> str:
    listed = ", ".join(f"#{s}" for s in others)
    noun = "record" if len(others) == 1 else "records"
    return (f"It took effect at fit #{fit_seq} together with {noun} {listed}: the fit records "
            f"measure the change between two fits, not each record's share of it.")


def _superseded(seq: int) -> str:
    return (f"Superseded by record #{seq}: the curve served is drawn for the plan in force now, so "
            f"it varies that record's choices, not this one's.")


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


def answers_of(decision: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The questions ``decision`` answers, each with its value: a reading as
    ``<reading>:<column>`` (a role, a unit and its days, a code or an amount, …; a repair of values
    below a detection limit as ``detection_limit:<column>``); any other decision as its kind and
    the fields that say which question it answers (:data:`ADDRESS`), its value every other field."""
    kind = str(decision.get("kind"))
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
    finding = str(decision.get("finding_id") or "")
    if kind == "apply_repair" and finding.startswith(BELOW_DETECTION):
        return [(f"detection_limit:{finding[len(BELOW_DETECTION):]}", decision.get("option"))]
    address = ADDRESS.get(kind, ())
    question = ":".join([kind, *(str(decision.get(f)) for f in address)])
    value = {k: v for k, v in decision.items() if k != "kind" and k not in address}
    return [(question, value)]


def same_option(examined: Any, chosen: Any) -> bool:
    """Whether an examined value is the option chosen. A field the examined value leaves empty
    (None) is not compared: a preview logs the decision as asked, and the server fills such fields
    when it records one (an opening's outcome and scores), so the same option previewed and then
    recorded is not an alternative to itself."""
    if isinstance(examined, Mapping) and isinstance(chosen, Mapping):
        return all(examined.get(k) is None or examined.get(k) == chosen.get(k)
                   for k in set(examined) | set(chosen))
    return examined == chosen


def phase_of(rec: Mapping[str, Any]) -> str:
    """The record's phase (:data:`record.Phase`), by the decision log's own marks."""
    if rec.get("post_seal"):
        return "after_seal"
    return "after_estimates" if rec.get("after_estimates") else "before"


def _at(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# ── the attempts ─────────────────────────────────────────────────────────────


def attempts_of(fits: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
                ) -> list[dict[str, Any]]:
    """Each fit record as served, with the records in force at it and not at the fit before it
    (``changed``; every record in force, for the first) and their notes (``reasons``)."""
    notes = {r["seq"]: r.get("note") for r in records}
    out: list[dict[str, Any]] = []
    before: set[int] | None = set()
    for rec in fits:
        now = rec.get("decisions")
        changed = None if now is None or before is None else sorted(set(now) - before)
        reasons = [{"seq": s, "note": notes[s]} for s in changed or () if notes.get(s)]
        keep = {k: rec.get(k) for k in R.Attempt.model_fields if k not in ("changed", "reasons")}
        out.append({**keep, "changed": changed, "reasons": reasons})
        before = set(now) if now is not None else None
    return out


# ── the decisions ────────────────────────────────────────────────────────────


def examined_of(rec: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                previews: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]
                ) -> list[dict[str, Any]]:
    """The options examined for each question ``rec`` answers (the module docstring), previews
    first, then fits, each in time order: each preview of the question made while this answer was
    being chosen or while it stood (after the question's previous answer, before its next), and
    each earlier answer in force in a fit that completed before this one was recorded. An option
    examined the same way again (previewed again with the same refusal, or held by another earlier
    record in a fit) is listed once, at its first."""
    at = _at(rec["at"])
    chosen = dict(answers_of(rec["decision"]))
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
                            and not same_option(value, chosen[question])})
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
                            "alternative": not same_option(value, chosen[question])})
    return out


def fits_materiality(rec: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]
                     ) -> dict[str, Any]:
    """The paired movement of the first fit ``rec`` was in force in, where it alone changed since
    the fit before; else why there is none."""
    seq = rec["seq"]
    first = next((i for i, a in enumerate(attempts) if seq in (a.get("decisions") or ())), None)
    if first is None:
        return {"source": None, "why": NOT_IN_A_FIT}
    fit = attempts[first]
    if first == 0:
        return {"source": None, "attempt": fit["id"], "why": fit.get("movement_why") or FIRST_FIT}
    changed = fit.get("changed")
    if changed is None:
        return {"source": None, "attempt": fit["id"],
                "why": fit.get("movement_why") or fit.get("withheld") or NO_MOVEMENT}
    others = [s for s in changed if s != seq]
    if others:
        return {"source": None, "attempt": fit["id"], "why": _together(others, fit["seq"])}
    if fit.get("movement"):
        return {"source": "fits", "attempt": fit["id"], "movement": fit["movement"]}
    return {"source": None, "attempt": fit["id"],
            "why": fit.get("movement_why") or fit.get("withheld") or NO_MOVEMENT}


def in_force_of(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Each question's record in force at the end of the log: its last answer that no revert
    withdrew."""
    reverted = {r["decision"].get("decision_id") for r in records
                if r["decision"].get("kind") == "revert"}
    out: dict[str, int] = {}
    for r in records:
        if r["id"] in reverted:
            continue
        for question, _ in answers_of(r["decision"]):
            out[question] = r["seq"]
    return out


def mattered_materiality(rec: Mapping[str, Any], mattered: Read,
                         in_force: Mapping[str, int]) -> dict[str, Any] | None:
    """Under Estimate, ``GET …/mattered``' specifications for ``rec``'s choices: those of the
    choices its kind recorded, where ``rec`` is the record in force now (the curve served is drawn
    for the plan in force); the route's own refusal or status where it drew none; None for a
    decision the curve does not vary."""
    kind = rec["decision"].get("kind")
    if kind not in MATTERED_KINDS:
        return None
    question = answers_of(rec["decision"])[0][0]
    current = in_force.get(question)
    if current is None:
        return {"source": None, "why": WITHDRAWN}
    if current != rec["seq"]:
        return {"source": None, "why": _superseded(current)}
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
    choices = [c for c in curve.get("choices") or [] if c.get("decision") == kind]
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
        return {"source": None, "why": NOT_VARIED}
    return {"source": "mattered", "primary": primary, "departures": departures}


def decisions_of(records: Sequence[Mapping[str, Any]], previews: Sequence[Mapping[str, Any]],
                 attempts: Sequence[Mapping[str, Any]], *, purpose: str | None,
                 mattered: Read) -> list[dict[str, Any]]:
    in_force = in_force_of(records)
    out: list[dict[str, Any]] = []
    for rec in records:
        materiality = fits_materiality(rec, attempts)
        if purpose == "inference" and materiality.get("source") is None:
            drawn = mattered_materiality(rec, mattered, in_force)
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
            "examined": examined_of(rec, records, previews, attempts),
            "materiality": materiality,
        })
    return out


# ── the result ───────────────────────────────────────────────────────────────


def _last(records: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    """The last record of ``kind`` no revert withdrew (the server withdraws a lock no estimate was
    served under by a revert of its own)."""
    reverted = {r["decision"].get("decision_id") for r in records
                if r["decision"].get("kind") == "revert"}
    return next((r for r in reversed(records) if r["decision"].get("kind") == kind
                 and r["id"] not in reverted), None)


def result_of(purpose: str | None, records: Sequence[Mapping[str, Any]],
              attempts: Sequence[Mapping[str, Any]], results: Read) -> dict[str, Any]:
    """The reported result, once (:class:`record.Result`)."""
    base: dict[str, Any] = {"purpose": purpose, "attempt": None, "plan_digest": None,
                            "decisions": None}
    if purpose == "prediction":
        opening = next((r for r in records if r["decision"].get("kind") == "open_seal"), None)
        if opening is None:
            return {**base, "refused": "The held-out rows were never opened."}
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
            out["scores"] = None
            out["refused"] = "The opening recorded no held-out scores."
        return out
    if purpose == "inference":
        lock = _last(records, "lock_plan")
        if lock is None:
            return {**base, "refused": "The plan was never locked, so no estimate was served."}
        locked = [a for a in attempts if a.get("decisions") is not None
                  and max(a["decisions"], default=0) <= lock["seq"]]
        fit = locked[-1] if locked else None
        out = {**base, "lock": lock["seq"], "plan_digest": lock["decision"].get("digest")}
        if fit is None:
            return {**out, "refused": "No fit record holds the answers the plan was locked on."}
        out.update(attempt=fit["id"], decisions=fit.get("decisions"))
        effects = {f: rows for f, rows in (fit.get("effects") or {}).items() if rows}
        if not effects:
            return {**out, "refused": str(fit.get("withheld")
                                          or "The locked fit serves no exposure rows.")}
        return {**out, "effects": effects}
    return {**base, "refused": "No goal was declared."}


# ── the rest of the record ───────────────────────────────────────────────────


def _bundle(data: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if not n.endswith("/")}
    checklist = next((n for n in sorted(files) if n.startswith("checklist/")
                      and n.endswith(".json")), None)
    return {"provenance": json.loads(files["provenance.json"]),
            "methods": files["methods.md"].decode("utf-8") if "methods.md" in files else None,
            "checklist": json.loads(files[checklist]) if checklist else None}


def export_of(served: Served, commands: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any],
                                                                               dict[str, Any]]:
    """The record's export, and the bundle's provenance (empty when the export is refused)."""
    reached = any(c.get("command") == "export" and _refusal(c.get("output")) is None
                  for c in commands)
    if not served.export.ok:
        error = (served.export.body or {}).get("error") or {}
        return ({"reached": reached, "sha256": None, "methods": None, "checklist": None,
                 "refused": {"code": error.get("code"), "message": error.get("message")}}, {})
    data = served.export.body
    parts = _bundle(data)
    return ({"reached": reached, "sha256": hashlib.sha256(data).hexdigest(),
             "methods": parts["methods"], "checklist": parts["checklist"], "refused": None},
            parts["provenance"])


def _refusal(output: Any) -> Mapping[str, Any] | None:
    error = output.get("error") if isinstance(output, Mapping) else None
    if isinstance(error, Mapping) and "code" in error and "message" in error:
        return error
    return None


def surfacings_of(commands: Sequence[Mapping[str, Any]], ledger: Read) -> dict[str, Any]:
    refusals = []
    for i, c in enumerate(commands):
        error = _refusal(c.get("output"))
        if error is None:
            continue
        nxt = commands[i + 1] if i + 1 < len(commands) else None
        refusals.append({"at": c.get("at"), "command": str(c.get("command")),
                         "code": str(error["code"]), "message": str(error["message"]),
                         "exits": [str(e.get("label")) for e in error.get("exits") or []
                                   if isinstance(e, Mapping)],
                         "then": str(nxt.get("command")) if nxt is not None else None})
    concerns = (ledger.body or {}).get("rows") or [] if ledger.ok else []
    return {"refusals": refusals, "concerns": concerns}


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
        "id": d.get("run_id"), "design": meta.get("design"),
        "engine_commit": engine.get("commit"), "engine_modified": engine.get("modified"),
        "engine_version": (h or {}).get("version"), "analyst_model": meta.get("analyst_model"),
        "effort": meta.get("effort"), "prompt_version": meta.get("prompt_version"),
        "started": d.get("start"), "ended": d.get("end"), "spend": spent, "spend_why": why,
        "fits": len(attempts), "fit_seconds": float(sum(seconds)),
        "fits_unseen": len(attempts) - len(seconds), "fit_budget": d.get("fit_budget"),
        "help": [str(h_) if not isinstance(h_, Mapping) else json.dumps(h_, sort_keys=True)
                 for h_ in meta.get("help") or []],
        "server": {"pid": d.get("pid"), "port": d.get("port"), "mode": (h or {}).get("mode"),
                   "workers": (h or {}).get("workers")},
    }


class RunUnreadable(RuntimeError):
    """The server serves no record of this run: its project, or its analyst log (a server not in
    analyst mode refuses it), cannot be read. The run is a failed run."""


def build(served: Served, run: RunDir, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The record, as a dict, from what the server served and the run directory: nothing is
    validated here (:func:`write` validates)."""
    if not served.project.ok:
        raise RunUnreadable(f"The project could not be read: {served.project.body}")
    if not served.analyst_log.ok:
        raise RunUnreadable(f"The analyst log could not be read: {served.analyst_log.body}")
    view = served.project.body
    records = sorted(view.get("decisions") or [], key=lambda r: r["seq"])
    state = view.get("state") or {}
    purpose = state.get("purpose")
    fit_records = (served.fits.body or {}).get("records") or [] if served.fits.ok else []
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
    run_id = str(run.driver.get("run_id") or Path(run_dir).name)
    pid = run.driver.get("project_id")
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
    except RunUnreadable as exc:
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

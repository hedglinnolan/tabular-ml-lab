"""The sentences a journey shows, indexed to the claim that rendered it, or triaged (R1_PLAN.md
R1-B; CLAIMS_REGISTRY.md §7; SWARM_PLAN §10 package 6a).

    python -m research.registry.sentences capture [--journey nhanes-glucose-predict] [--out FILE]
    python -m research.registry.sentences table [--journey nhanes-glucose-predict] [--check]

**The capture** drives a journey through the real server (a local-mode server with two workers over
a fresh home, ``server_drive.local_server``, as ``turbotab/core/reference/journeys.py`` drives the
review packets' journeys, and with its answer sources), and reads what the person is shown
(:func:`read_shown`) before every answer it posts and every press of Fit, once no stage is computing
(:func:`settled`, waiting at most :data:`SETTLE_SECONDS`), so in the state the answer is given in;
again after each one taken; and at the end, once no stage is computing:

* **the quest log** (``GET /quest``, ``GET /triage``, ``GET /record``, and of ``GET /checklist``
  what the Write-up card draws, its title and what it waits for): each line, its reason and what
  it would change, the sweeps, the triage's noticings and the For the record lines;
* **the ask card** of each open question (``GET /projects/{pid}``'s ``interview[].ask``), the
  fields the quest shell's card draws and its text, which the swarm driver's ``ask`` serves
  (:func:`ask_drawn`);
* **the stage artifacts** (``GET /stages/{stage}``, each fresh artifact once for each content it
  serves, so one that changes under its key, the fit once the held-out rows open, is read again;
  and ``GET /results`` and ``GET /fits`` once Fit is pressed): every card's words;
* **the methods** (``GET /methods``, the line drawn under each answered line, and
  ``GET /methods/sections``, the manuscript beside the log, at every read; and the export
  bundle's ``methods.md``);

and the refusals the answers met (each ``POST /decisions`` refused, its message and its ways
forward), which a card shows where the answer was given. At the end it reads **the teaching
entries** (``GET /teaching``, :func:`read_teaching`) of the questions the journey answered, the
roles card's under the project's goal, and the repairs entry where a noticing's card was drawn:
each entry as the card draws it and the swarm driver's ``ask`` serves it, its concept drawer left
out and counted (:data:`TEACHING_LEFT_OUT`). It reads nothing else: no preview of an answer before
it is given (a journey previews none) and none of the data views (the table's pages, the column
summaries, the histograms). A **sentence** is a string served there as the value of a key that is
not a name key (:data:`NAME_KEYS`: a string under one is not read, whatever it holds, and under
``source`` that is where a thing comes from, citations among them; the capture counts, by key,
those that read as sentences, ``named``), of at least :data:`MIN_WORDS` words, or of fewer that
ends a sentence or states a number (:func:`is_sentence`); a shorter string that does neither is a
**label** (:func:`is_label`), counted and not kept as a sentence. **A state that lasts only while a
stage computes** is read only where a read that does not wait falls in it, since every read before
an answer or a press of Fit, and the last, waits it out: the one just after the upload, and one
just after each answer and each press of Fit taken. The capture counts the reads that waited and
those that did not, and of the latter those that found a stage computing (``reads``); it does not
mark the sentences such a read alone served. The capture keeps each distinct string once, with
every place it was served (the source and the JSON path, its list indices dropped), the teaching
entries read and the drawer sentences left out, the labels, the name keys' counts and the reads',
the stages a read stopped waiting for, and the project's claims render log
(``claims_rendered.jsonl``: what a swarm record's ``claims`` reads), and writes them to
``research/registry/journeys/<journey>.capture.json``.

**The table** reads a capture and splits each string into its sentences (:func:`sentences_of`), and
each sentence, once, with every place it was shown, is:

* **indexed** to the claim whose card or methods sentence, or one clause of it, as the registry
  renders it now (``claims.card``, ``claims.methods`` with no context), lies in it, where one does
  (:func:`index`); the table says whether the capture's render log holds that claim (**traced**);
  a claim rendered outside the log (a static fill made at import, a read the log does not wrap, a
  teaching entry) is not;
* else **not shown** where every place it was served is the card of a Router question the
  journey's interview did not ask (not applicable, or set for you: :data:`QUESTION_OF`);
* else **triaged** by CLAIMS_REGISTRY §7's rule, through the rulings in
  ``research/registry/triage.py``: a claim to register only if it states a number the app relies
  on, sets a default, a ranking or a refusal, or teaches a method; a registered claim's rule said
  in other words (the claim named); everything else plain wording (a label, a restated answer, a
  reading of the data, a count, interface copy). A sentence no ruling covers is **untriaged**,
  and ``table`` exits 1 naming it, as it does for a ruling that names a claim no one registers and
  for two sentences stating one rule whose verdicts disagree (:func:`disagreements`).

It writes ``research/registry/journeys/<journey>.md``: what the capture read and left out, with the
counts (the labels under the floor among them, whole or split off a longer string; the name keys'
strings and the reads that found a stage computing where the capture counted them, and where it
did not, that it holds no count); what is registered (:func:`registered_words`: the registry's
claims, those that render a sentence here, the sentences a record traces, the sentences queued to
register and the rulings they fall under, each ruling with its count; the table registers none);
the claims that render a sentence and whether the record traced each, the claims the log holds
that render none, and every sentence with where it was shown and its verdict. ``--check``
writes nothing and exits 1 when the committed table is not what the capture, the registry and the
rulings give now, so the next batch that registers a claim or changes a ruling sees the table it
leaves stale. A new journey is a :class:`Spec` in :data:`SPECS`.

Run the capture with ``OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2``: it fits the journey's families once
(the NHANES glucose journey, linear and ridge, in a few minutes, with the reads that wait).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "journeys"
# Fewer words is a label or a name (CLAIMS_REGISTRY.md §1), unless the string ends a sentence or
# states a number ("5-fold cross-validation.", the reason a default is set for you).
MIN_WORDS = 4
SETTLE_SECONDS = 300.0  # the longest one read waits for the stages computing to finish
POLL_SECONDS = 0.5
BUSY = ("queued", "running")  # a stage computing (``graph.StatusName``; a held stage is idle)

# ── the journeys ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spec:
    """A journey the capture drives: the reference journey it follows (``journeys.Journey``'s
    fields), and what it declares that no Router question asks, before the question it names."""

    name: str
    words: str  # the journey in a sentence, for the table's head
    journey: Callable[[], Any]  # a journeys.Journey
    declare: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    final: str | None = None  # the family declared final at the opening


def _nhanes_glucose_predict() -> Any:
    """The walkthrough's journey (notes/walkthrough/2026-10-17-nhanes-glucose-predict.md): the
    NHANES export the product tracks (``turbotab/core/tests/fixtures/nhanes.csv.gz``), the dietary
    and clinical lenses, glucose under Predict, the roles the app proposes confirmed as proposed,
    every row kept, the missing-values card's and the seal plan's first-ranked options, linear and
    ridge."""
    from turbotab.core.reference import journeys as j

    return j.Journey(
        "nhanes-glucose-predict", "dietary", "prediction",
        "How well do the table's measures predict fasting glucose?",
        j.NHANES_FILE, "the walkthrough of 2026-10-17 (notes/walkthrough/)",
        j.nhanes_path, j.nhanes_dietary_truth, target="glucose", lenses=("dietary", "clinical"),
        answers={"exclusions": {"kind": "set_exclusions", "rules": []},
                 "models": {"kind": "select_models", "models": ["linear", "ridge"]}},
        needs=(str(j.nhanes_path()),), fixture_key=j.NHANES_KEY, timeout=540.0)


SPECS: dict[str, Spec] = {s.name: s for s in (
    Spec("nhanes-glucose-predict",
         "NHANES fasting glucose under Predict, as the walkthrough of 2026-10-17 walked it: the "
         "product's NHANES fixture, the dietary and clinical lenses, the roles as the app "
         "proposes them, every row kept, the first-ranked missing-values and seal answers, linear "
         "and ridge, Fit, the triage, ridge declared final, the opening and the bundle.",
         _nhanes_glucose_predict,
         declare={"models": ({"kind": "set_intended_use", "use": "risk_estimation"},
                             {"kind": "set_selection", "method": "none"})},
         final="ridge"),
)}

# ── reading what is shown ────────────────────────────────────────────────────

# Keys whose string values name things or carry machine values, never words a person reads.
NAME_KEYS = frozenset({
    "id", "key", "kind", "column", "columns", "hash", "sha256", "digest", "fingerprint", "path",
    "source_path", "at", "created_at", "computed_at", "seq", "decision_id", "stage", "status",
    "family", "families", "model", "models", "feature", "features", "option", "value_key",
    "testid", "url", "file", "name_key", "code", "role", "proposed", "reading", "task", "purpose",
    "lens", "lenses", "metric", "unit", "level", "levels", "target", "version", "fid", "pid",
    "job_id", "plan_digest", "digest_short", "engine", "turbotab", "method", "strategy", "source",
    "time_column", "id_column", "donor", "recipient", "exposure", "energy_column", "nutrients",
    "row_ids", "keys", "fold", "folds", "seed", "cluster", "validation", "grain",
})
_INDEX = re.compile(r"\[\d+\]")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_NAMEISH = re.compile(r"^[\w./:\-]+$")  # one token: an identifier, a path, a key
_DIGIT = re.compile(r"\d")


def _words(text: str) -> int:
    return len(_WORD.findall(text))


def _wording(t: str) -> bool:
    """Whether ``t`` (stripped) holds words at all: not one token (an identifier or a path), not
    JSON, not an ISO time, and at least one word of letters."""
    return bool(t) and not _NAMEISH.match(t) and t[0] not in "{[" \
        and not re.match(r"^\d{4}-\d{2}-\d{2}T", t) and _words(t) >= 1


def is_sentence(text: str) -> bool:
    """A string a person reads as words (:func:`_wording`) of at least :data:`MIN_WORDS` words, or
    of fewer that ends a sentence (".", "!" or "?") or states a number (a digit)."""
    t = text.strip()
    return _wording(t) and (_words(t) >= MIN_WORDS or t[-1] in ".!?" or bool(_DIGIT.search(t)))


def is_label(text: str) -> bool:
    """A string under the floor: words (:func:`_wording`), fewer than :data:`MIN_WORDS` of them,
    that neither end a sentence nor state a number ("Keep every row", "an amount")."""
    t = text.strip()
    return _wording(t) and not is_sentence(t)


def strings_in(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Each (string, path) in ``node`` under a key that is not a name (:data:`NAME_KEYS`); a path
    reads ``a.b[].c``, its indices dropped."""
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k) in NAME_KEYS and not isinstance(v, (dict, list)):
                continue
            yield from strings_in(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for v in node:
            yield from strings_in(v, f"{path}[]")
    elif isinstance(node, str):
        yield node.strip(), _INDEX.sub("[]", path)


def named_in(node: Any) -> Iterator[tuple[str, str]]:
    """Each (key, string) :func:`strings_in` skips: a string served directly under a key of
    :data:`NAME_KEYS`, whatever it holds (under ``source``, where a thing comes from, citations
    among them)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k) in NAME_KEYS and isinstance(v, str):
                yield str(k), v.strip()
            elif isinstance(v, (dict, list)):
                yield from named_in(v)
    elif isinstance(node, list):
        for v in node:
            yield from named_in(v)


def sentences_in(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Each (string, path) of :func:`strings_in` that :func:`is_sentence`."""
    return ((t, p) for t, p in strings_in(node, path) if is_sentence(t))


@dataclass
class Shown:
    """What a capture saw: each distinct string, and every place it was served; the labels under
    the floor (:func:`is_label`), counted, not kept as sentences; the strings under a name key
    that read as sentences (:func:`named_in`), by key, counted and not kept; the stages a read
    stopped waiting for (:func:`settled`), with the number of reads each was still computing at;
    and the reads, those that waited until no stage was computing and those that did not, with
    how many of the latter found a stage computing (:func:`read_shown`)."""

    strings: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    snapshots: set[str] = field(default_factory=set)  # digests of the reads kept, by source
    reached: set[str] = field(default_factory=set)  # the questions answered on their card
    findings: bool = False  # whether a noticing's card was drawn (the repairs entry words it)
    labels: set[str] = field(default_factory=set)
    unsettled: dict[str, int] = field(default_factory=dict)
    named: dict[str, set[str]] = field(default_factory=dict)
    reads: dict[str, int] = field(default_factory=dict)

    def add(self, node: Any, source: str) -> None:
        for text, path in strings_in(node):
            if is_sentence(text):
                self.strings.setdefault(text, set()).add((source, path))
            elif is_label(text):
                self.labels.add(text)
        for key, text in named_in(node):
            if is_sentence(text):
                self.named.setdefault(key, set()).add(text)

    def once(self, node: Any, source: str) -> None:
        """Add ``node`` unless the same content was read from ``source`` already: a read whose
        content changed under the same key (a stage artifact once the held-out rows open) is read
        again."""
        digest = hashlib.sha256((source + json.dumps(node, sort_keys=True, default=str))
                                .encode()).hexdigest()
        if digest not in self.snapshots:
            self.snapshots.add(digest)
            self.add(node, source)


def _json(r: Any) -> Any:
    try:
        return r.json() if r.status_code == 200 else None
    except ValueError:
        return None


# The project's reads the quest shell draws from, after every answer: the quest log, the triage,
# For the record, the methods line under each answered line (``GET /methods``, by its record's id:
# QuestScreen's ``answers``), the manuscript beside the log (``GET /methods/sections``), and the
# Write-up card's checklist, of which the card draws the title and what it waits for
# (``StageCard.Checklist``; :data:`DRAWN`).
READS = (("quest", "quest"), ("triage", "quest:triage"), ("record", "quest:record"),
         ("checklist", "quest:checklist"), ("methods", "methods:lines"),
         ("methods/sections", "methods"))
DRAWN = {"quest:checklist": ("title", "waiting")}  # a read of which the shell draws these keys


def ask_drawn(card: dict[str, Any]) -> dict[str, Any]:
    """An open step's ask card (``GET /projects/{pid}``'s ``interview[].ask``), the fields the
    quest shell's card draws (``AskFor``'s ledger card, ``record/ask/AskCard.tsx``): who reads the
    answer, each line's guessed answer and its evidence, the ways forward's labels, and each reading
    the values settled with its evidence and the labels of the changes it offers; and its text, the
    same words the card draws in pieces (its heading, then each line), which the swarm driver's
    ``ask`` serves with the rest as ``readings``. A reading's value and a line's guess are values
    the card does not draw."""
    return {
        "text": card.get("text"), "consumer": card.get("consumer"),
        "groups": [{k: g.get(k) for k in ("guess_words", "evidence")}
                   for g in card.get("groups") or [] if isinstance(g, dict)],
        "exits": [{"label": e.get("label")} for e in card.get("exits") or [] if isinstance(e, dict)],
        "read_from_data": [{"words": s.get("words"), "evidence": s.get("evidence"),
                            "change": [{"label": c.get("label")} for c in s.get("change") or []
                                       if isinstance(c, dict)]}
                           for s in card.get("read_from_data") or [] if isinstance(s, dict)],
    }


def settled(client: Any, pid: str) -> list[str]:
    """Wait until no stage of the project is computing (:data:`BUSY`), at most
    :data:`SETTLE_SECONDS`; the stages still computing when it stopped waiting, none once settled."""
    end = time.monotonic() + SETTLE_SECONDS
    while True:
        view = _json(client.get(f"/api/projects/{pid}")) or {}
        busy = sorted(st for st, s in (view.get("stages") or {}).items()
                      if (s or {}).get("status") in BUSY)
        if not busy or time.monotonic() >= end:
            return busy
        time.sleep(POLL_SECONDS)


def read_shown(client: Any, pid: str, shown: Shown, *, final: bool = False,
               settle: bool = False) -> None:
    """What the person is shown now: the quest log, the triage, For the record, the checklist and
    the methods (:data:`READS`); the ask card of each open question (:func:`ask_drawn`); each fresh
    stage artifact, once for each content it has served; once Fit is pressed (or at the end,
    ``final``), the results and the fits. With ``settle``, and at the end, it first waits until no
    stage is computing (:func:`settled`), so it reads the state an answer is given in, and notes a
    stage it stopped waiting for. It notes the questions answered (each card the person answered
    on) and whether a noticing's card is drawn, for the teaching read at the end
    (:func:`read_teaching`); a question the Router opens for a moment while a stage computes, and
    then sets for you or finds not applicable, is not one. It counts itself in ``shown.reads``: as
    one that waited, or as one that did not, and then also as one that found a stage computing
    where the project's view, read in the same pass, shows one."""
    base = f"/api/projects/{pid}"
    waits = settle or final
    if waits:
        for stage in settled(client, pid):
            shown.unsettled[stage] = shown.unsettled.get(stage, 0) + 1
    for route, source in READS:
        found = _json(client.get(f"{base}/{route}"))
        if isinstance(found, dict) and source in DRAWN:
            found = {k: found.get(k) for k in DRAWN[source]}
        if found is not None:
            shown.once(found, source)
    view = _json(client.get(base)) or {}
    kind = "waited" if waits else "unwaited"
    shown.reads[kind] = shown.reads.get(kind, 0) + 1
    if not waits:
        busy = any((s or {}).get("status") in BUSY for s in (view.get("stages") or {}).values())
        shown.reads["computing"] = shown.reads.get("computing", 0) + int(busy)
    shown.reached.update(str(s.get("key")) for s in view.get("interview") or []
                         if s.get("status") == "answered")
    for step in view.get("interview") or []:
        if step.get("status") == "open" and isinstance(step.get("ask"), dict):
            shown.once(ask_drawn(step["ask"]), f"ask:{step.get('key')}")
    for stage, status in sorted((view.get("stages") or {}).items()):
        if (status or {}).get("status") != "fresh":
            continue
        served = _json(client.get(f"{base}/stages/{stage}"))
        if served is not None:
            artifact = served.get("artifact")
            shown.once(artifact, f"artifact:{stage}")
            if stage == "findings" and (artifact or {}).get("findings"):
                shown.findings = True
    if ((view.get("state") or {}).get("fit_pressed") or final
            or any((s or {}).get("status") == "fresh" for st, s in (view.get("stages") or {}).items()
                   if st == "fit")):
        for route in ("results", "fits"):
            found = _json(client.get(f"{base}/{route}"))
            if found is not None:
                shown.once(found, f"artifact:{route}")


# A question card's teaching entry, as the card draws it and the swarm driver's ``ask`` serves it
# (``research/swarm/driver.py``): its title, question, one line, why, who reads the answer, its
# terms and its options (its evidence badge names a status and a source, no sentence). The concept
# drawer is left out: the quest shell's card offers it by a press that opens nothing (AskFor's
# plain wiring, its ``onOpenDrawer`` a no-op), and the driver serves none of it.
TEACHING_LEFT_OUT = ("drawer",)


def read_teaching(client: Any, shown: Shown, purpose: str | None) -> dict[str, int]:
    """Each reached question's teaching entry (``GET /teaching``), and the repairs entry where a
    noticing's card was drawn, as the card reads it: the roles card under the project's goal, every
    other card with none (``AskFor``); the sentences of what is left out
    (:data:`TEACHING_LEFT_OUT`), counted by entry."""
    plain = _json(client.get("/api/teaching")) or []
    goal = ((_json(client.get("/api/teaching", params={"purpose": purpose})) or []) if purpose
            else plain)
    keys = sorted(shown.reached | ({"repairs"} if shown.findings else set()))
    left_out: dict[str, int] = {}
    for key in keys:
        entry = next((e for e in (goal if key == "roles" else plain) if e.get("key") == key), None)
        if entry is None:
            continue
        shown.once({k: v for k, v in entry.items() if k not in TEACHING_LEFT_OUT},
                   f"teaching:{key}")
        n = sum(len(sentences_of(t)) for t, _ in sentences_in(
            {k: entry.get(k) for k in TEACHING_LEFT_OUT}))
        if n:
            left_out[key] = n
    return left_out


class Watching:
    """The reference journey's recording client (``journeys.Recording``), reading what is shown
    before each answer it posts and each press of Fit, once no stage is computing (the state the
    person answers in), and again after each that is taken, and keeping each refusal's words."""

    def __init__(self, recording: Any, shown: Shown, pid: Callable[[], str]):
        self.r, self.shown, self.pid = recording, shown, pid

    @property
    def app(self) -> Any:
        return self.r.app

    def get(self, url: str, **kw: Any) -> Any:
        return self.r.get(url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:  # noqa: A002
        if url.endswith(("/decisions", "/fit")):
            read_shown(self.r, self.pid(), self.shown, settle=True)
        r = self.r.post(url, json=json, **kw)
        if url.endswith("/decisions"):
            if r.status_code == 409:
                error = (_json_any(r) or {}).get("error") or {}
                if error.get("code") != "not_yet":
                    kind = (json or {}).get("kind") if isinstance(json, dict) else None
                    self.shown.add({k: v for k, v in error.items() if k != "code"},
                                   f"refusal:{kind}")
            elif r.status_code == 200:
                read_shown(self.r, self.pid(), self.shown)
        elif url.endswith("/fit") and r.status_code == 200:
            read_shown(self.r, self.pid(), self.shown)
        return r


def _json_any(r: Any) -> Any:
    try:
        return r.json()
    except ValueError:
        return None


def capture(spec: Spec, out: Path | None = None) -> dict[str, Any]:
    """Drive ``spec`` once in a fresh home (module docstring) and write its capture."""
    from turbotab.core.reference import journeys as j
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

    journey = spec.journey()
    run = j.Run(journey)
    shown = Shown()
    notes: list[str] = []
    started = time.monotonic()
    log_lines: list[dict[str, Any]] = []
    interview: dict[str, str] = {}
    teaching_left_out: dict[str, int] = {}
    export_status: int | None = None
    with tempfile.TemporaryDirectory(prefix=f"tt-sentences-{spec.name}-") as folder:
        home = Path(folder) / "home"
        try:
            with local_server(home) as client:
                run.client = Watching(j.Recording(client, run), shown, lambda: run.pid)
                r = client.post("/api/projects", json={"path": str(journey.path())})
                r.raise_for_status()
                run.pid = r.json()["id"]
                t = journey.truth()
                t.guess = j._server_guess(client, run.pid)
                t.ranked = j._server_ranked(client, run.pid)
                run.drive = Drive(run.client, run.pid, t)
                run.drive.artifact("ingest", timeout=900)
                read_shown(client, run.pid, shown)
                for key, bodies in spec.declare.items():
                    journey.before[key] = j.declare(*bodies)
                if spec.final:
                    journey.answers["open_seal"] = {"kind": "open_seal", "family": spec.final}
                if j.follow(run) and j.wait_results(run):
                    response = j.export(run)
                    export_status = getattr(response, "status_code", None)
                    if export_status == 200:
                        bundle = j.bundle_parts(response.content, home)
                        shown.add({"methods_md": bundle["methods_md"]}, "methods:bundle")
                read_shown(client, run.pid, shown, final=True)
                view = _json(client.get(f"/api/projects/{run.pid}")) or {}
                interview = {str(s.get("key")): str(s.get("status"))
                             for s in view.get("interview") or []}
                teaching_left_out = read_teaching(client, shown,
                                                  (view.get("state") or {}).get("purpose"))
                log = home / "projects" / run.pid / "claims_rendered.jsonl"
                if log.is_file():
                    log_lines = [json.loads(x) for x in log.read_text("utf-8").splitlines()
                                 if x.strip()]
        except Exception as exc:  # noqa: BLE001 - a journey that stops is kept with its reason
            notes.append(f"stopped by {type(exc).__name__}: {str(exc)[:900]}")
            traceback.print_exc()
    renders: dict[tuple[str, str], int] = {}
    for x in log_lines:
        if not x.get("refused"):
            k = (str(x.get("fallback") or x.get("id")), str(x.get("register")))
            renders[k] = renders.get(k, 0) + 1
    rendered = sorted(renders)
    result = {
        "journey": {"name": spec.name, "words": spec.words, "question": journey.question,
                    "lenses": list(journey.lenses), "target": journey.target,
                    "purpose": journey.purpose},
        "captured": {"commit": j._git_head(), "date": time.strftime("%Y-%m-%d"),
                     "seconds": round(time.monotonic() - started), "workers": 2,
                     "export": export_status},
        "answers": run.log,
        "notes": run.notes + notes,
        "interview": interview,
        "teaching": {"read": sorted(shown.reached | ({"repairs"} if shown.findings else set())),
                     "left_out": {"fields": list(TEACHING_LEFT_OUT),
                                  "sentences": teaching_left_out}},
        "labels": sorted(shown.labels),
        "named": {k: len(v) for k, v in sorted(shown.named.items())},
        "reads": dict(sorted(shown.reads.items())),
        "unsettled": dict(sorted(shown.unsettled.items())),
        "rendered": [{"id": i, "register": reg, "renders": renders[(i, reg)]}
                     for i, reg in rendered],
        "strings": [{"text": text, "shown": sorted([s, p] for s, p in places)}
                    for text, places in sorted(shown.strings.items())],
    }
    path = out or HERE / f"{spec.name}.capture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(result['strings'])} strings, {len(rendered)} claims rendered, "
          f"export {export_status}, {result['captured']['seconds']} s", flush=True)
    return result


# ── the table ────────────────────────────────────────────────────────────────

# A sentence ends at ".", "!" or "?" before a space and a capital, a quote, a parenthesis, a mark
# or a digit, but not after an abbreviation ("et al.", "e.g.", "i.e.", "vs.", "cf."); a line break
# ends one too (the methods are Markdown).
_SPLIT = re.compile(r"(?<=[.!?])(?<!\bal\.)(?<!e\.g\.)(?<!i\.e\.)(?<!\bvs\.)(?<!\bcf\.)"
                    r"\s+(?=[A-Z`“\"(*_\d])|\n+")
_MARKS = "#*_-> "  # Markdown's heading, list, emphasis and quote marks at a line's start


def _parts(text: str) -> Iterator[str]:
    for part in _SPLIT.split(text):
        part = part.strip().lstrip(_MARKS).strip()
        if part:
            yield part


def sentences_of(text: str) -> list[str]:
    """The sentences of a string (:func:`is_sentence`), in order, a Markdown line's leading marks
    dropped."""
    return [p for p in _parts(text) if is_sentence(p)]


def labels_of(text: str) -> list[str]:
    """The pieces of a string under the floor (:func:`is_label`), in order: a heading, a label."""
    return [p for p in _parts(text) if is_label(p)]


@dataclass(frozen=True)
class Said:
    """A claim's sentence as the registry renders it now, or one clause of it (a render site may
    place a claim's clauses apart: ``explore.missing_by_group``)."""

    text: str
    claim: str
    register: str


def claim_sentences(reg: Any) -> list[Said]:
    """Each built claim's card and methods sentence, as ``claims.card`` and ``claims.methods``
    render them with no context, and each of their clauses (split at "; ") of at least
    :data:`MIN_WORDS` words (a shorter clause, though it ends a sentence or states a number, would
    index any sentence that happens to hold it), longest first. A claim the frozen status file does not list as built
    renders nowhere, so it indexes nothing."""
    from turbotab.core.claims import ClaimError, fill

    out: list[Said] = []
    for c in reg:
        try:
            if reg.frozen_status(c.id) != "built":
                continue
        except ClaimError:
            continue
        for register in ("card", "methods"):
            template = getattr(c, register)
            if not template:
                continue
            text = fill(template, c.params)
            pieces = {text, *(p.strip() for p in text.split("; ")
                              if is_sentence(p) and _words(p) >= MIN_WORDS)}
            out.extend(Said(p, c.id, register) for p in pieces)
    return sorted(out, key=lambda s: (-len(s.text), s.claim, s.register, s.text))


def index(sentence: str, said: Iterable[Said]) -> list[str]:
    """The claims whose sentence, or a clause of it, lies in ``sentence``, or that hold it whole
    (a claim's sentence of several sentences) where it has at least :data:`MIN_WORDS` words, in id
    order."""
    whole = _words(sentence) >= MIN_WORDS
    return sorted({s.claim for s in said if s.text in sentence or (whole and sentence in s.text)})


# Artifact paths a card shows only where its Router question is asked: a sentence served there
# alone is not shown on a journey whose interview marks that question not applicable.
QUESTION_OF: tuple[tuple[str, str, str], ...] = (
    ("artifact:proposals", "labels.energy_adjustment", "energy_adjustment"),
    ("artifact:proposals", "energy.", "energy_adjustment"),
    ("artifact:proposals", "exposure_forms", "form"),
    ("artifact:forms", "", "form"),
)


NOT_ASKED = ("not_applicable", "skipped")  # a skipped question is set for you: its card is not shown


def not_shown(places: Iterable[tuple[str, str]], interview: dict[str, str]) -> str | None:
    """The question whose card alone serves every one of ``places``, where the journey's interview
    marks it not applicable or skipped (set for you, said For the record instead of on its card);
    None where any place is shown (:data:`QUESTION_OF`)."""
    found = set()
    for source, path in places:
        q = next((q for s, prefix, q in QUESTION_OF if s == source and path.startswith(prefix)),
                 None)
        if q is None or interview.get(q) not in NOT_ASKED:
            return None
        found.add(q)
    return ", ".join(sorted(found)) or None


@dataclass
class Row:
    sentence: str
    places: list[tuple[str, str]]
    verdict: str  # "claim", "unrendered", "register", "plain", "not shown", "untriaged"
    claims: list[str]
    why: str


def rows_of(capture_data: dict[str, Any], reg: Any, rulings: Iterable[Any]) -> list[Row]:
    """Every sentence of the capture, once, with all its places, and its verdict: indexed to the
    claims that render it, else the first ruling that matches it, else untriaged."""
    said = claim_sentences(reg)
    rulings = list(rulings)
    interview = capture_data.get("interview") or {}
    places: dict[str, set[tuple[str, str]]] = {}
    for entry in capture_data["strings"]:
        for sentence in sentences_of(entry["text"]):
            places.setdefault(sentence, set()).update((s, p) for s, p in entry["shown"])
    out = []
    for sentence in sorted(places, key=str.lower):
        where = sorted(places[sentence])
        found = index(sentence, said)
        if found:
            out.append(Row(sentence, where, "claim", found, ""))
            continue
        question = not_shown(where, interview)
        if question is not None:
            out.append(Row(sentence, where, "not shown", [],
                           f"served only on the card of a question this journey does not ask "
                           f"({question})"))
            continue
        ruling = next((r for r in rulings if re.search(r.pattern, sentence) and (
            not r.where or any(re.search(r.where, f"{s} {p}") for s, p in where))), None)
        if ruling is None:
            out.append(Row(sentence, where, "untriaged", [], ""))
        else:
            out.append(Row(sentence, where, ruling.verdict, list(ruling.claims), ruling.why))
    return out


# A sentence's values: a `ticked` name or value, and a number.
_TICKED = re.compile(r"`[^`]*`")
_NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")
CLAIMED = ("claim", "unrendered", "register")  # the verdicts that say a claim states a sentence
HELD_WORDS = 6  # a sentence held inside another is compared only from this many words up


def normal(sentence: str) -> str:
    """``sentence`` with each value one mark (a ticked span ```·```, a number ``0``) and its
    closing period dropped: two sentences that state one rule on two columns, or with two counts,
    read alike."""
    return _NUMBER.sub("0", _TICKED.sub("`·`", sentence.strip().rstrip(".")))


def disagreements(rows: Iterable[Row]) -> list[str]:
    """Each place two triaged sentences that state one rule get verdicts that disagree:

    * two sentences alike but for their values (:func:`normal`) with another verdict, or another
      registered claim named;
    * a sentence holding another whole (of at least :data:`HELD_WORDS` words, values aside) that a
      claim states (:data:`CLAIMED`), while it is plain wording itself;
    * a sentence holding another whole that a registered claim's rule says in other words
      (``unrendered``), while it neither names that claim nor is rendered by a claim.

    A sentence not shown, or untriaged, is compared with none. Each is said in a sentence."""
    kept = [r for r in rows if r.verdict not in ("not shown", "untriaged")]
    out: list[str] = []
    seen: dict[str, Row] = {}
    for r in kept:
        first = seen.setdefault(normal(r.sentence), r)
        if first is not r and (first.verdict, first.claims) != (r.verdict, r.claims):
            out.append(f"alike but for their values, “{first.sentence}” is {first.verdict} "
                       f"{first.claims} and “{r.sentence}” is {r.verdict} {r.claims}")
    held = [(normal(r.sentence), r) for r in kept if r.verdict in CLAIMED]
    held = [(n, r) for n, r in held if len(_WORD.findall(n)) >= HELD_WORDS]
    for a in kept:
        whole = normal(a.sentence)
        for n, b in held:
            if b is a or n == whole or n not in whole:
                continue
            if a.verdict == "plain":
                out.append(f"“{a.sentence}” is plain wording, and it holds “{b.sentence}”, which "
                           f"is {b.verdict} {b.claims}")
            elif (b.verdict == "unrendered" and a.verdict != "claim"
                  and not set(b.claims) <= set(a.claims)):
                out.append(f"“{a.sentence}” is {a.verdict} {a.claims}, and it holds "
                           f"“{b.sentence}”, which says {b.claims}' rule unrendered")
    return out


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _shown(places: list[tuple[str, str]]) -> str:
    by: dict[str, list[str]] = {}
    for source, path in places:
        by.setdefault(source, []).append(path)
    return "; ".join(f"{s} ({', '.join(sorted(set(p))[:2])}{', …' if len(set(p)) > 2 else ''})"
                     for s, p in sorted(by.items()))


VERDICT_WORDS = {
    "claim": "rendered by a registered claim",
    "unrendered": "a registered claim's rule, in words not its own",
    "register": "a claim, to register",
    "plain": "plain wording",
    "not shown": "not shown on this journey",
    "untriaged": "untriaged",
}


SOURCES = (
    "the quest log, the triage, For the record and the Write-up card's checklist (its title and "
    "what it waits for)",
    "the ask card of each open question, as the quest shell's card draws it (who reads the answer, "
    "each line's guessed answer and its evidence, the ways forward, and what the values settled, "
    "with its evidence and the changes it offers) and with its text, which the swarm driver's "
    "`ask` serves with the rest",
    "the methods line under each answered line and the manuscript beside the log",
    "each fresh stage artifact, once for each content it served, and the results and the fits "
    "once Fit was pressed",
    "the refusals the answers met",
    "the bundle's methods",
)


def floor_labels(capture_data: dict[str, Any]) -> set[str]:
    """The labels under the floor (:func:`is_label`): the strings the capture read and left out,
    and the pieces of the sentences' strings that split off as labels (a heading, a label)."""
    out = set(capture_data.get("labels") or [])
    for entry in capture_data["strings"]:
        out.update(labels_of(entry["text"]))
    return out


def held_words(capture_data: dict[str, Any]) -> str:
    """What the capture read and what it left out by design, with the counts, in sentences."""
    teaching = capture_data.get("teaching") or {}
    read = list(teaching.get("read") or [])
    left = (teaching.get("left_out") or {}).get("sentences") or {}
    questions = [k for k in read if k != "repairs"]
    parts = list(SOURCES)
    if read:
        parts.append(
            f"at the end, the teaching entries of the {len(questions)} questions the journey "
            f"answered "
            f"({', '.join(f'`{k}`' for k in questions)})"
            + (" and of the repairs its noticings drew" if "repairs" in read else "")
            + ": the title, question, one line, why, who reads the answer and terms, as each card "
              "draws them and the swarm driver's `ask` serves them, and the options as the entry "
              "words them (a card that composes its options from a stage's artifact shows those)")
    out = (f"The sentences the capture read, before every answer and every press of Fit once no "
           f"stage was computing (waiting at most {SETTLE_SECONDS:g} s), so in the state each was "
           f"given in, again after each one taken, and at the end once no stage was computing: "
           + "; ".join(parts[:-1]) + "; and " + parts[-1] + ". A sentence is a string, or a piece "
           f"of one split at a sentence's end, of at least {MIN_WORDS} words, or of fewer that "
           "ends a sentence or states a number, served as the value of a key that is not a name "
           "key (below).")
    labels = floor_labels(capture_data)
    named = capture_data.get("named")
    reads = capture_data.get("reads")
    taken = sum(1 for a in capture_data.get("answers") or [] if a.get("status") == 200)
    out += (" **Left out by design:** "
            + (f"those entries' concept drawers, {sum(left.values())} sentences (the quest "
               "shell's card offers a drawer by a press that opens nothing, and the swarm "
               "driver serves none of it); " if left else "")
            + (f"{len(labels)} strings of fewer than {MIN_WORDS} words that neither end a sentence "
               "nor state a number (labels and headings), whole or split off a longer string; "
               if labels else "")
            + f"the strings served as the value of a name key, one that names a thing or carries "
              f"a machine value (the {len(NAME_KEYS)} keys of `sentences.NAME_KEYS`, such as an "
              "id, a path, a code or a column's name, and `source`, whose values say where a thing "
              "comes from, citations among them, some quoting a passage), whatever they hold, "
            + (f"{sum(named.values())} strings among them that read as sentences, "
               f"{named.get('source', 0)} under `source`; " if named is not None
               else "which this capture does not count; ")
            + "a preview of an answer before it is given (this journey previews none); and the "
              "data views (the table's pages, the column summaries, the histograms), which the "
              "capture does not read. **Read only where a read falls in it:** a state that lasts "
              "only while a stage computes (a question waiting on it, a gate that reads its answer "
              "as not yet given). Every read before an answer or a press of Fit, and the last, "
              "waits until no stage is computing; the reads that do not wait are the one just "
              f"after the upload and one just after each answer taken ({taken} here) and each "
              "press of Fit taken, and "
            + (f"{reads.get('computing', 0)} of those {reads.get('unwaited', 0)} reads found a "
               "stage computing (the project's view, read in the same pass, showed one)"
               if reads is not None
               else "which of those reads found a stage computing this capture does not record")
            + ". The capture neither counts nor marks the sentences only such a state serves, so "
              "a run that reads at other moments can find ones it lacks.")
    unsettled = capture_data.get("unsettled") or {}
    if unsettled:
        out += (f" **Not read:** what a stage still computing after {SETTLE_SECONDS:g} s served "
                "when a read stopped waiting for it: "
                + ", ".join(f"`{st}` ({n} {'read' if n == 1 else 'reads'})"
                            for st, n in sorted(unsettled.items())) + ".")
    return out


def _many(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def registered_words(rows: list[Row], traced: set[str], held: int) -> str:
    """What is registered, in sentences: the registry's ``held`` claims, those that render a
    sentence here and the sentences a record traces (rendered by a claim the render log held),
    of the sentences shown; the sentences queued to register and the rulings they fall under;
    those that say a registered claim's rule outside its render. The table registers none."""
    shown = [r for r in rows if r.verdict != "not shown"]
    rendered = [r for r in rows if r.verdict == "claim"]
    claimed = {c for r in rendered for c in r.claims}
    traces = [r for r in rendered if set(r.claims) & traced]
    queued = [r for r in rows if r.verdict == "register"]
    rulings = {r.why for r in queued}
    unrendered = sum(1 for r in rows if r.verdict == "unrendered")
    return (f"**What is registered.** The registry holds {held} claims; {len(claimed)} of them "
            f"{'renders' if len(claimed) == 1 else 'render'} a sentence here "
            f"({_many(len(rendered), 'sentence', 'sentences')}), and {len(traces)} of those "
            f"sentences {'is' if len(traces) == 1 else 'are'} rendered by a claim the record "
            f"traced, so of the {len(shown)} sentences this journey showed a swarm record's "
            f"`claims` traces {len(traces)}. "
            + (f"{_many(len(queued), 'sentence is a claim', 'sentences are claims')} still to "
               f"register, under the {_many(len(rulings), 'ruling', 'rulings')} below, and no "
               "record traces them. " if queued else "No sentence is queued to register. ")
            + f"{_many(unrendered, 'sentence says', 'sentences say')} a registered claim's rule "
              "in words not its own, outside its render, so no record traces "
            + ("it" if unrendered == 1 else "them") + " either. This table registers no claim: "
              "a ruling queues a sentence, and a later batch registers its claim with its parts "
              "and tests.")


def render_table(capture_data: dict[str, Any], rows: list[Row], reg: Any) -> str:
    """The committed table: the journey, the counts, and every sentence with its verdict."""
    traced = {r["id"] for r in capture_data.get("rendered") or []}
    authors = {c.id: c.author for c in reg}
    j, cap = capture_data["journey"], capture_data["captured"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    claimed = sorted({c for r in rows if r.verdict == "claim" for c in r.claims})
    lines = [
        f"# The sentences of the {j['name']} journey",
        "",
        f"Generated by `python -m research.registry.sentences table --journey {j['name']}` from "
        f"`{j['name']}.capture.json` (captured {cap['date']} at `{cap['commit']}`, "
        f"{cap['seconds']} s, export {cap['export']}), the claims registry and the rulings in "
        "`research/registry/triage.py`; `--check` holds it current. Do not edit it by hand.",
        "",
        f"**The journey.** {j['words']}",
        "",
        "**What it holds.** " + held_words(capture_data),
        "",
        "**How to read it.** Each sentence once, with where it was shown (the source and the "
        "served path). Its verdict: the claim whose words render it; or, by CLAIMS_REGISTRY §7's "
        "rule, a claim to register (it states a number the app relies on, sets a default, a "
        "ranking or a refusal, or teaches a method) or plain wording; or a registered claim's "
        "rule said in other words. **Traced** marks a claim the project's claims render log held "
        "during the capture: what a swarm record's `claims` lists. A claim rendered outside that "
        "log is not traced: a static fill made at import (a module's constant), a read the log "
        "does not wrap, or a teaching entry, which is served to every project alike.",
        "",
        "| | sentences |",
        "|---|---|",
    ]
    for v in ("claim", "unrendered", "register", "plain", "not shown", "untriaged"):
        lines.append(f"| {VERDICT_WORDS[v]} | {counts.get(v, 0)} |")
    lines.append(f"| all | {len(rows)} |")
    lines += ["", registered_words(rows, traced, len(authors))]
    queued: dict[str, int] = {}
    for r in rows:
        if r.verdict == "register":
            queued[r.why] = queued.get(r.why, 0) + 1
    if queued:
        lines += ["", "**The sentences queued to register**, by the ruling that queues them, most "
                  "first:", "", "| ruling: what the sentences state, and where the code writes it "
                  "| sentences |", "|---|---|"]
        lines += [f"| {_cell(why)} | {n} |"
                  for why, n in sorted(queued.items(), key=lambda x: (-x[1], x[0]))]
    outside = [c for c in claimed if c not in traced]
    lines += ["", f"**The claims that render a sentence here**, and whether the record traced "
              f"them: {len(claimed) - len(outside)} of {len(claimed)} did"
              + (f"; the other {len(outside)} rendered outside the log ("
                 + ", ".join(f"`{c}`" for c in outside) + ")" if outside else "") + ".",
              "", "| claim | sentences | traced | registered by |", "|---|---|---|---|"]
    for cid in claimed:
        n = sum(1 for r in rows if r.verdict == "claim" and cid in r.claims)
        lines.append(f"| `{cid}` | {n} | {'yes' if cid in traced else 'no'} | "
                     f"{_cell(authors.get(cid, ''))} |")
    unshown = sorted(traced - set(claimed))
    if unshown:
        lines += ["", "**The claims the log holds that render no sentence here** (a swarm "
                  "record's `claims` lists them, though no sentence shown holds their words): "
                  + ", ".join(f"`{c}`" for c in unshown) + "."]
    lines += ["", "## Every sentence the capture read", "",
              "| # | sentence | shown | verdict | why |",
              "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        verdict = VERDICT_WORDS[r.verdict]
        if r.claims:
            ids = ", ".join(f"`{c}`" + (" (traced)" if r.verdict == "claim" and c in traced
                                        else "") for c in r.claims)
            verdict = f"{verdict}: {ids}"
        lines.append(f"| {i} | {_cell(r.sentence)} | {_cell(_shown(r.places))} | "
                     f"{_cell(verdict)} | {_cell(r.why)} |")
    return "\n".join(lines) + "\n"


def table(journey: str, capture_path: Path | None = None, out: Path | None = None,
          check: bool = False, *, reg: Any = None, rulings: Iterable[Any] | None = None) -> int:
    """Write (or with ``check``, compare) the journey's table; 1 when a sentence is untriaged, two
    sentences that state one rule get verdicts that disagree (:func:`disagreements`), a ruling
    names a claim that is not registered, or the committed table is stale. ``reg`` and
    ``rulings`` default to the claims registry and ``triage.RULINGS``."""
    if rulings is None:
        from research.registry.triage import RULINGS as rulings
    if reg is None:
        from turbotab.core import claims

        reg = claims.registry()
    rulings = list(rulings)
    capture_path = capture_path or HERE / f"{journey}.capture.json"
    out = out or HERE / f"{journey}.md"
    data = json.loads(capture_path.read_text(encoding="utf-8"))
    rows = rows_of(data, reg, rulings)
    untriaged = [r.sentence for r in rows if r.verdict == "untriaged"]
    for s in untriaged:
        print(f"untriaged: {s}", file=sys.stderr)
    unknown = sorted({c for r in rulings for c in r.claims if c not in reg})
    for c in unknown:
        print(f"a ruling names {c}, which no claim registers", file=sys.stderr)
    disagree = disagreements(rows)
    for d in disagree:
        print(f"verdicts disagree: {d}", file=sys.stderr)
    failed = int(bool(untriaged or unknown or disagree))
    text = render_table(data, rows, reg)
    if check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"{out} is stale: run python -m research.registry.sentences table --journey "
                  f"{journey}", file=sys.stderr)
            failed = 1
        return failed
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}: {len(rows)} sentences, {len(untriaged)} untriaged")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="drive the journey and write its capture")
    c.add_argument("--journey", default="nhanes-glucose-predict", choices=sorted(SPECS))
    c.add_argument("--out", type=Path, default=None)
    t = sub.add_parser("table", help="index and triage a capture's sentences into the table")
    t.add_argument("--journey", default="nhanes-glucose-predict", choices=sorted(SPECS))
    t.add_argument("--capture", type=Path, default=None)
    t.add_argument("--out", type=Path, default=None)
    t.add_argument("--check", action="store_true", help="write nothing; exit 1 when stale")
    args = parser.parse_args(argv)
    os.environ.setdefault("TURBOTAB_WORKERS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if args.command == "capture":
        capture(SPECS[args.journey], args.out)
        return 0
    return table(args.journey, args.capture, args.out, check=args.check)


if __name__ == "__main__":
    sys.exit(main())

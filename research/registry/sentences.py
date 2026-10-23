"""Every sentence a journey shows, indexed to the claim that rendered it, or triaged (R1_PLAN.md
R1-B; CLAIMS_REGISTRY.md §7; SWARM_PLAN §10 package 6a).

    python -m research.registry.sentences capture [--journey nhanes-glucose-predict] [--out FILE]
    python -m research.registry.sentences table [--journey nhanes-glucose-predict] [--check]

**The capture** drives a journey through the real server (a local-mode server with two workers over
a fresh home, ``server_drive.local_server``, as ``turbotab/core/reference/journeys.py`` drives the
review packets' journeys, and with its answer sources), and after every answer it posts reads what
the person is shown, from three sources:

* **the quest log** (``GET /quest``, ``GET /triage``, ``GET /record``): each line, its reason and
  what it would change, the sweeps, the triage's noticings and the For the record lines;
* **the stage artifacts** (``GET /stages/{stage}``, each fresh artifact once per key, and
  ``GET /results`` and ``GET /fits`` once Fit is pressed): every card's words;
* **the methods** (``GET /methods/sections``, and the export bundle's ``methods.md``);

and the refusals the answers met (each ``POST /decisions`` refused, its message and its ways
forward), which a card shows where the answer was given. A **sentence** is a string served there of
at least :data:`MIN_WORDS` words that is not a name (:func:`sentences_in`). The capture keeps each
distinct string once, with every place it was served (the source and the JSON path, its list
indices dropped), and the project's claims render log (``claims_rendered.jsonl``: what a swarm
record's ``claims`` reads), and writes them to ``research/registry/journeys/<journey>.capture.json``.

**The table** reads a capture and splits each string into its sentences (:func:`sentences_of`), and
each sentence, once, with every place it was shown, is:

* **indexed** to the claim whose card or methods sentence, or one clause of it, as the registry
  renders it now (``claims.card``, ``claims.methods`` with no context), lies in it, where one does
  (:func:`index`); the table says whether the capture's render log holds that claim (**traced**);
* else **not shown** where every place it was served is the card of a Router question the
  journey's interview did not ask (not applicable, or set for you: :data:`QUESTION_OF`);
* else **triaged** by CLAIMS_REGISTRY §7's rule, through the rulings in
  ``research/registry/triage.py``: a claim to register only if it states a number the app relies
  on, sets a default, a ranking or a refusal, or teaches a method; a registered claim's rule said
  in other words (the claim named); everything else plain wording (a label, a restated answer, a
  reading of the data, a count, interface copy). A sentence no ruling covers is **untriaged**,
  and ``table`` exits 1 naming it, as it does for a ruling that names a claim no one registers.

It writes ``research/registry/journeys/<journey>.md``: the counts, the claims that render a sentence
and whether the record traced each, and every sentence with where it was shown and its verdict.
``--check`` writes nothing and exits 1 when the committed table is not what the capture, the
registry and the rulings give now, so the next batch that registers a claim or changes a ruling
sees the table it leaves stale. A new journey is a :class:`Spec` in :data:`SPECS`.

Run the capture with ``OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2``: it fits the journey's families once
(the NHANES glucose journey, linear and ridge, in about two minutes).
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
MIN_WORDS = 4  # fewer words is a label or a name, never a sentence (CLAIMS_REGISTRY.md §1)

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


def is_sentence(text: str) -> bool:
    """A string a person reads as words: at least :data:`MIN_WORDS` words of letters, not one
    token (an identifier or a path), not JSON, not an ISO time."""
    t = text.strip()
    if not t or _NAMEISH.match(t) or t[0] in "{[" or re.match(r"^\d{4}-\d{2}-\d{2}T", t):
        return False
    return len(_WORD.findall(t)) >= MIN_WORDS


def sentences_in(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Each (string, path) in ``node`` that :func:`is_sentence`, under a key that is not a name
    (:data:`NAME_KEYS`); a path reads ``a.b[].c``, its indices dropped."""
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k) in NAME_KEYS and not isinstance(v, (dict, list)):
                continue
            yield from sentences_in(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for v in node:
            yield from sentences_in(v, f"{path}[]")
    elif isinstance(node, str) and is_sentence(node):
        yield node.strip(), _INDEX.sub("[]", path)


@dataclass
class Shown:
    """What a capture saw: each distinct string, and every place it was served."""

    strings: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    artifacts: set[tuple[str, str]] = field(default_factory=set)  # (stage, key) read
    snapshots: set[str] = field(default_factory=set)  # digests of quest-log reads kept

    def add(self, node: Any, source: str) -> None:
        for text, path in sentences_in(node):
            self.strings.setdefault(text, set()).add((source, path))

    def once(self, node: Any, source: str) -> None:
        """Add ``node`` unless the same content was read from ``source`` already."""
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


def read_shown(client: Any, pid: str, shown: Shown, *, final: bool = False) -> None:
    """What the person is shown now: the quest log, the triage and For the record; each fresh stage
    artifact not read at its key yet; once Fit is pressed, the results and the fits; at the end
    (``final``), the methods."""
    base = f"/api/projects/{pid}"
    for route, source in (("quest", "quest"), ("triage", "quest:triage"),
                          ("record", "quest:record")):
        found = _json(client.get(f"{base}/{route}"))
        if found is not None:
            shown.once(found, source)
    view = _json(client.get(base)) or {}
    for stage, status in sorted((view.get("stages") or {}).items()):
        key = (status or {}).get("key")
        if (status or {}).get("status") != "fresh" or not key or (stage, key) in shown.artifacts:
            continue
        served = _json(client.get(f"{base}/stages/{stage}"))
        if served is not None:
            shown.artifacts.add((stage, key))
            shown.add(served.get("artifact"), f"artifact:{stage}")
    if ((view.get("state") or {}).get("fit_pressed") or final
            or any((s or {}).get("status") == "fresh" for st, s in (view.get("stages") or {}).items()
                   if st == "fit")):
        for route in ("results", "fits"):
            found = _json(client.get(f"{base}/{route}"))
            if found is not None:
                shown.once(found, f"artifact:{route}")
    if final:
        found = _json(client.get(f"{base}/methods/sections"))
        if found is not None:
            shown.once(found, "methods")


class Watching:
    """The reference journey's recording client (``journeys.Recording``), reading what is shown
    after each answer it posts, and keeping each refusal's words."""

    def __init__(self, recording: Any, shown: Shown, pid: Callable[[], str]):
        self.r, self.shown, self.pid = recording, shown, pid

    @property
    def app(self) -> Any:
        return self.r.app

    def get(self, url: str, **kw: Any) -> Any:
        return self.r.get(url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:  # noqa: A002
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


def sentences_of(text: str) -> list[str]:
    """The sentences of a string, each of at least :data:`MIN_WORDS` words (:func:`is_sentence`),
    in order, a Markdown line's leading marks dropped."""
    out = []
    for part in _SPLIT.split(text):
        part = part.strip().lstrip(_MARKS).strip()
        if part and is_sentence(part):
            out.append(part)
    return out


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
    :data:`MIN_WORDS` words, longest first. A claim the frozen status file does not list as built
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
            pieces = {text, *(p.strip() for p in text.split("; ") if is_sentence(p))}
            out.extend(Said(p, c.id, register) for p in pieces)
    return sorted(out, key=lambda s: (-len(s.text), s.claim, s.register, s.text))


def index(sentence: str, said: Iterable[Said]) -> list[str]:
    """The claims whose sentence, or a clause of it, lies in ``sentence``, or that hold it whole
    (a claim's sentence of several sentences), in id order."""
    return sorted({s.claim for s in said if s.text in sentence or sentence in s.text})


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
        "**How to read it.** Each sentence the journey showed from the quest log, the stage "
        "artifacts, the methods and the refusals its answers met, once, with where it was shown "
        "(the source and the served path). Its verdict: the claim whose words render it; or, by "
        "CLAIMS_REGISTRY §7's rule, a claim to register (it states a number the app relies on, "
        "sets a default, a ranking or a refusal, or teaches a method) or plain wording; or a "
        "registered claim's rule said in other words. **Traced** marks a claim the project's "
        "claims render log held during the capture: what a swarm record's `claims` lists.",
        "",
        "| | sentences |",
        "|---|---|",
    ]
    for v in ("claim", "unrendered", "register", "plain", "not shown", "untriaged"):
        lines.append(f"| {VERDICT_WORDS[v]} | {counts.get(v, 0)} |")
    lines.append(f"| all | {len(rows)} |")
    lines += ["", "**The claims that render a sentence here**, and whether the record traced "
              "them:", "", "| claim | sentences | traced | registered by |", "|---|---|---|---|"]
    for cid in claimed:
        n = sum(1 for r in rows if r.verdict == "claim" and cid in r.claims)
        lines.append(f"| `{cid}` | {n} | {'yes' if cid in traced else 'no'} | "
                     f"{_cell(authors.get(cid, ''))} |")
    lines += ["", "## Every sentence", "", "| # | sentence | shown | verdict | why |",
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
    """Write (or with ``check``, compare) the journey's table; 1 when a sentence is untriaged, a
    ruling names a claim that is not registered, or the committed table is stale. ``reg`` and
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
    failed = int(bool(untriaged or unknown))
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

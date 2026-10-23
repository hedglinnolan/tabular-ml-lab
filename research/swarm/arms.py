"""The swarm's mechanical arms (SWARM_PLAN §6): the default path and a seeded random path through the
driver, each to its record, and the walker and sessions the sweep (``research/swarm/sweep.py``)
shares with them. No arm spends a token.

    python -m research.swarm.arms default --dataset <id> [--runs DIR] [--corpus DIR]
    python -m research.swarm.arms random  --dataset <id> --seed N [--runs DIR] [--corpus DIR]

**The question is the card's** (:class:`Question`): the outcome, its event level, the purpose and,
under Estimate, the exposure, as the dataset's question card gives them (``card.json``; the event
level read from the outcome's own words on the card, "(1 yes, 0 no)" or "(1)", else ``1`` where
the outcome's levels are 0 and 1, else the first level offered). An arm never answers these
otherwise, and neither sweep changes them, nor the facts of the outcome and the table the Router
asks (its kind, its follow-up, the table's layout; :data:`FIXED`): another outcome is another
question.

**The default path** (:class:`Default`) answers each open question with the app's own first-ranked
option as the driver offers it: journeys.py's answer source (``turbotab/core/reference/journeys.py``,
the dry run's, ``research/swarm/dry_run.py``): the first option the seal plan, the shelf, the
missing-values and exclusions cards rank, the app's detection of the outcome's kind, the proposed
roles confirmed, the acceptance drivers' grouping, time-varying and causal answers, the form card's
proposal, the lens the profile hints (with none, the lens that claims no field: the lenses are
listed, not ranked); each reading the server asks at the app's own guess (its ask card's), else its
first way; each role line confirmed as the roles answer proposed it; each finding's first repair.
It records no declaration (each open declaration line stays as the app sets it), and declares final
the first family fitted. **The random path** (:class:`Random`, seeded by
``numpy.random.default_rng``) answers the same questions, readings and findings with an option drawn
uniformly from those the driver offers that record something by themselves (a field of one's own is
never invented), answers each open declaration line once the same way before Fit, and declares
final a family drawn from those fitted. Where an answer or a fit is refused, either takes a way
forward the refusal offers (the first, or one drawn); where an answer is refused with none, it takes
another option (the next offered, or another drawn), never another of the card's answers. Where
none is left, an answer fails, or a question waits on stages that failed, the walk stops with what
the engine said, classed refused or failed (:class:`Met`), which the sweep classes its refit by.
Under Predict, what the fit raises is answered and Fit pressed again before the held-out rows open,
so they open on a current fit. Each record is given the dataset as the launcher gives an analyst's
run (``launcher.dataset_meta``: its control kind and known answer).

**Two sessions** (:class:`Driven`, :class:`Local`), one walker (:func:`walk`). ``Driven`` is a run
of the driver's own: every command goes through ``driver.main`` in this process, so the run's
``commands.jsonl`` and its record (``research/swarm/assemble.py``) are an analyst's; it is how the
arms run. ``Local`` is a project on an in-process analyst-mode server (FastAPI's ``TestClient``,
whose unhandled exceptions are served as a 500, as a server serves them), driven by the driver's own
functions (``the_question``, ``chosen``, ``post_decision``, ``wait_fit``): how the sweep runs, many
projects on one server, with every HTTP 500 kept (:attr:`Local.errors`).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "research" / "datasets"
# The card's question and the facts of its outcome and its table, never changed by a sweep: another
# outcome, another event level, another goal, another kind of outcome or another follow-up (one
# that varies makes the outcome a time to event) is another question, and the table's layout is
# how it was written.
FIXED = ("target", "event", "purpose", "task", "follow_up", "orientation")
WAIT = 240.0  # seconds a command waits for what is computing
STEPS = 300  # the most answers a walk gives before it says it never reached Fit
PROMPT_VERSION = "arms-1"
# The tag each answer carries (record version 2's ``Tag``): the marker of the vocabulary the
# question's answer rests on, as the dry run tags its scripted analyst's (``dry_run.MARKER_OF``).
DEFAULT_MARKER = "design"


# ── the card's question ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    """What the card fixes: the outcome, the goal, the exposure under Estimate, and the words the
    event level is read from."""

    target: str
    purpose: str  # "prediction" or "inference"
    exposure: str | None = None
    meaning: str | None = None  # the outcome's meaning, in the card's words

    @classmethod
    def of_card(cls, card: Mapping[str, Any]) -> "Question":
        purpose = {"Predict": "prediction", "Estimate": "inference"}.get(str(card.get("purpose")))
        if purpose is None:
            raise ValueError(f"The card's purpose is {card.get('purpose')!r}, not Predict or "
                             f"Estimate.")
        outcome = card.get("outcome") or {}
        exposure = (card.get("exposure") or {}).get("column") if purpose == "inference" else None
        return cls(target=str(outcome["column"]), purpose=purpose, exposure=exposure,
                   meaning=outcome.get("meaning"))

    def event(self, levels: Sequence[str]) -> str | None:
        """The event level among ``levels`` (the class docstring), or None when none is offered."""
        levels = [str(v) for v in levels]
        if not levels:
            return None
        for pattern in (r"\((\w+) yes", r"\((\w+)\)"):
            found = re.search(pattern, self.meaning or "")
            if found and found.group(1) in levels:
                return found.group(1)
        return "1" if set(levels) == {"0", "1"} else levels[0]


def card_of(dataset: str) -> dict[str, Any]:
    return json.loads((DATASETS / dataset / "card.json").read_text("utf-8"))


# ── what a command said ──────────────────────────────────────────────────────


@dataclass
class Said:
    """What a command returned, classed: ``ok``; ``refused`` in words (by the server at the answer,
    by the engine at the fit, or by the driver), with its code, words and ways forward; ``failed``,
    an HTTP 500 or an error no refusal typed."""

    kind: str  # "ok", "refused", "failed"
    payload: dict[str, Any] = field(default_factory=dict)
    by: str | None = None
    code: str | None = None
    message: str | None = None
    exits: list[dict[str, Any]] = field(default_factory=list)
    status: int | None = None

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def said_of(code: int, payload: dict[str, Any]) -> Said:
    """A driver command's exit status and JSON, classed (:class:`Said`): a refusal whose status is
    500 or above, or the driver's own report that it failed, is a failure, never a refusal; a fit
    whose status is an error the engine did not type (which the driver's ``fit`` says with exit
    status 0, as it failed) is a failure, and one that is neither fresh nor an error (blocked, or
    held) was not reached."""
    refused = payload.get("refused")
    # A press of Fit's answer (``pressed`` beside the fit's status; an ask's ``fit`` is another).
    fit = payload.get("fit") if "pressed" in payload and isinstance(payload.get("fit"), Mapping) \
        else None
    if code == 0 and not refused and fit is not None and fit.get("status") == "error":
        return Said("failed", payload, by="engine", code="fit_failed", message=fit.get("error"))
    if code == 0 and not refused and fit is not None and fit.get("status") != "fresh":
        held = f", held for {fit['held']}" if fit.get("held") else ""
        return Said("refused", payload, by="driver", code="fit_not_reached",
                    message=f"The fit is {fit.get('status')}{held}.")
    if code == 0 and not refused:
        return Said("ok", payload)
    refused = refused or {}
    status = refused.get("status")
    server = refused.get("server") if isinstance(refused.get("server"), Mapping) else {}
    crashed = (isinstance(status, int) and status >= 500) or refused.get("code") in (
        "driver_failed", "server_unreadable", "server_unreachable") or str(
        server.get("code") or "").startswith("http_5")
    return Said("failed" if crashed else "refused", payload, by=refused.get("by"),
                code=refused.get("code"), message=refused.get("message"),
                exits=list(refused.get("exits") or []), status=status)


class Stuck(RuntimeError):
    """A walk that cannot go on: what it met, in words."""


class Met(Stuck):
    """A walk stopped by what the engine said, classed: ``kind`` ``refused`` (in words: an answer
    refused with no way forward, or a stage's typed refusal) or ``failed`` (an answer that failed,
    or a stage in error the engine did not type); ``detail``, what the class rests on (its code,
    words and ways forward, where it stopped, and each stage that failed)."""

    def __init__(self, words: str, kind: str, detail: Mapping[str, Any]):
        super().__init__(words)
        self.kind, self.detail = kind, dict(detail)


def said_fields(said: Said) -> dict[str, Any]:
    return {"kind": said.kind, "by": said.by, "code": said.code, "message": said.message,
            "status": said.status}


def exit_fields(exits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: e.get(k) for k in ("label", "decision")} for e in exits]


def failed_stages(session: Any, q: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each stage the ask's question waits on that failed (``q["failed"]``), typed as the driver
    types it (``driver.failures``: ``refused`` where the stage serves the engine's typed refusal,
    else ``failed``): the ask's own ``fit_failed``, the driver's typing over the view the ask read,
    else the driver's typing over the project's view. A stage neither types is ``failed``."""
    from research.swarm import driver

    names = [str(n) for n in q.get("failed") or {}]
    typed = {f.get("stage"): f for f in q.get("fit_failed") or [] if f.get("stage") in names}
    missing = [n for n in names if n not in typed]
    if missing:
        try:
            typed.update({f["stage"]: f for f in driver.failures(session.view(), missing)})
        except driver.DriverError:
            pass  # the project could not be read: each such stage is said untyped, below
    return [typed.get(n) or {"stage": n, "key": None, "error": (q.get("failed") or {}).get(n),
                             "kind": "failed", "refusal": None} for n in names]


def met_stages(failures: Sequence[Mapping[str, Any]], words: str) -> Met:
    """Stages in error (``driver.failures``' entries), classed: ``failed`` where any is an error
    the engine did not type (the first such, its words), else ``refused`` with the first typed
    refusal's stage, code, words and ways forward; each stage kept (``stages``)."""
    untyped = [f for f in failures if f.get("kind") != "refused"] or (
        [] if failures else [{"stage": None, "error": words}])  # none named: said untyped
    if untyped:
        first = untyped[0]
        return Met(words, "failed", {"kind": "failed", "by": "engine", "code": "stage_failed",
                                     "message": first.get("error"), "status": None,
                                     "at": "stage", "stage": first.get("stage"),
                                     "stages": [dict(f) for f in failures]})
    first = failures[0]
    refusal = first.get("refusal") or {}
    return Met(words, "refused", {"kind": "refused", "by": "engine", "code": refusal.get("code"),
                                  "message": refusal.get("message"), "status": None,
                                  "at": "stage",
                                  "stage": refusal.get("stage") or first.get("stage"),
                                  "exits": exit_fields(refusal.get("exits") or []),
                                  "stages": [dict(f) for f in failures]})


def met_answer(words: str, key: str | None, said: Said, answered: Sequence[str] = ()) -> Met:
    """An answer the walk gave that the engine refused with no way forward (``refused``) or that
    failed (``failed``), classed as ``said`` is, with the question and the options answered."""
    kind = "failed" if said.kind == "failed" else "refused"
    return Met(words, kind, said_fields(said) | {"at": "walk", "question": key,
                                                 "answered": list(answered),
                                                 "exits": exit_fields(said.exits)})


# ── the sessions ─────────────────────────────────────────────────────────────


class Driven:
    """A run of the driver's own (the module docstring): ``start --scripted`` and every later
    command through ``driver.main`` in this process, with the runs' home ``runs``."""

    def __init__(self, runs: Path):
        self.runs = Path(runs).absolute()
        self.run: str | None = None

    @contextlib.contextmanager
    def _home(self) -> Iterator[None]:
        from research.swarm import driver

        before = os.environ.get(driver.RUNS_ENV)
        os.environ[driver.RUNS_ENV] = str(self.runs)
        try:
            yield
        finally:
            if before is None:
                os.environ.pop(driver.RUNS_ENV, None)
            else:
                os.environ[driver.RUNS_ENV] = before

    def command(self, *argv: str) -> tuple[int, dict[str, Any]]:
        from research.swarm import driver

        out: list[str] = []
        with self._home():
            code = driver.main(list(argv), out=out.append)
        return code, (json.loads(out[-1]) if out else {})

    def cmd(self, name: str, *args: str) -> Said:
        if self.run is None:
            raise Stuck("The run has not started.")
        return said_of(*self.command(name, "--run", self.run, *args))

    def start(self, data: Path, *, budget_fits: int, budget_seconds: float,
              model: str, effort: str | None = None,
              prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
        args = ["start", "--data", str(data), "--budget-fits", str(budget_fits),
                "--budget-seconds", str(budget_seconds), "--scripted", "--model", model,
                "--prompt-version", prompt_version]
        if effort is not None:
            args += ["--effort", effort]
        code, payload = self.command(*args)
        if code != 0:
            raise Stuck(f"The run did not start: {json.dumps(payload)[:800]}")
        self.run = payload["started"]["run"]
        return payload["started"]

    def stop(self) -> None:
        if self.run is not None:
            self.command("stop", "--run", self.run)

    def meta(self) -> dict[str, Any]:
        return json.loads((self.runs / str(self.run) / "driver.json").read_text("utf-8"))

    def view(self) -> dict[str, Any]:
        return self.read(None) or {}

    def read(self, path: str | None) -> Any:
        """``GET /projects/{pid}/{path}`` on the run's server (the arm's own read, as the dry run
        reads an artifact; the project itself where ``path`` is None), or None when it refuses."""
        import urllib.error
        import urllib.request

        meta = self.meta()
        url = f"http://127.0.0.1:{meta['port']}/api/projects/{meta['project']}" + (
            f"/{path}" if path else "")
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read() or b"null")
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def ask(self) -> dict[str, Any]:
        s = self.cmd("ask", "--wait", "90")
        if not s.ok:
            raise Stuck(f"ask was refused: {json.dumps(s.payload)[:800]}")
        return s.payload

    def answer(self, ids: Sequence[str], sets: Sequence[str], reason: str,
               marker: str = DEFAULT_MARKER) -> Said:
        return self.cmd("answer", *ids, *[a for s in sets for a in ("--set", s)], "--reason",
                        reason, "--marker", marker)

    def preview(self, ids: Sequence[str], sets: Sequence[str], reason: str) -> Said:
        return self.cmd("preview", ids[0], *[a for s in sets for a in ("--set", s)], "--reason",
                        reason)

    def fit(self) -> Said:
        return self.cmd("fit", "--wait", str(int(WAIT)))

    def results(self) -> Said:
        return self.cmd("results", "--wait", str(int(WAIT)))

    def seal(self, family: str, reason: str) -> Said:
        return self.cmd("seal", family, "--reason", reason)

    def export(self) -> Said:
        return self.cmd("export", "--wait", str(int(WAIT)))


class _Client:
    """The driver's ``Client`` on FastAPI's ``TestClient``: the same paths under ``/api``, each
    answer as the driver's ``Response``; every answer of status 500 or above kept in ``errors``."""

    def __init__(self, client: Any, errors: list[dict[str, Any]]):
        self.c, self.errors = client, errors

    def request(self, method: str, path: str, *, body: bytes | None = None,
                headers: dict[str, str] | None = None, query: dict[str, Any] | None = None) -> Any:
        from research.swarm import driver

        q = {k: v for k, v in (query or {}).items() if v is not None}
        r = self.c.request(method, "/api" + path, content=body, headers=headers or {}, params=q)
        if r.status_code >= 500:
            self.errors.append({"method": method, "path": path, "status": r.status_code,
                                "body": (body or b"").decode("utf-8", "replace")[:2000],
                                "said": r.text[:2000]})
        return driver.Response(r.status_code, r.content, dict(r.headers))

    def get(self, path: str, **query: Any) -> Any:
        return self.request("GET", path, query=query)

    def post(self, path: str, data: Any = None, **query: Any) -> Any:
        body = None if data is None else json.dumps(data).encode("utf-8")
        return self.request("POST", path, body=body, query=query,
                            headers={"Content-Type": "application/json"} if body else {})


class Local:
    """Projects on an in-process analyst-mode server in ``home`` (the module docstring), each
    driven as the driver drives its run's: a ``with`` block holds the server; :meth:`use` names the
    project the commands act on."""

    def __init__(self, home: Path, *, workers: int = 2):
        self.home = Path(home).absolute()
        self.workers = workers
        self.errors: list[dict[str, Any]] = []
        self.pid: str | None = None
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> "Local":
        from fastapi.testclient import TestClient

        from turbotab.core.config import Settings
        from turbotab.server.app import create_app

        self.home.mkdir(parents=True, exist_ok=True)
        settings = dataclasses.replace(Settings.from_env(), home=self.home, mode="local",
                                       analyst=True, workers=self.workers)
        app = create_app(settings, frontend_dist=self.home / "no-frontend")
        tc = self._stack.enter_context(TestClient(app, base_url="http://127.0.0.1",
                                                  raise_server_exceptions=False))
        self.client = _Client(tc, self.errors)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()

    # the project

    def open(self, data: Path) -> str:
        r = self.client.post("/projects", {"path": str(Path(data).absolute())})
        if r.status != 200:
            raise Stuck(f"The project was not made: {r.status} {r.body[:600]!r}")
        self.use(r.json()["id"])
        end = time.monotonic() + WAIT
        while time.monotonic() < end:
            status = ((self.view().get("stages") or {}).get("ingest") or {}).get("status")
            if status in ("fresh", "error", "blocked"):
                break
            time.sleep(0.3)
        return str(self.pid)

    def use(self, pid: str) -> None:
        from research.swarm import driver

        self.pid = pid
        folder = self.home / "driver" / pid
        folder.mkdir(parents=True, exist_ok=True)
        self.run = driver.Run(folder, {"project": pid, "budget": {"fits": 10 ** 6,
                                                                  "seconds": 1e12}})

    def view(self) -> dict[str, Any]:
        from research.swarm import driver

        return driver.view_of(self.run, self.client)

    # copies of a project (the sweep's refits, ``research/swarm/sweep.py``)

    BUSY = ("queued", "running")

    def idle(self, pid: str | None = None, wait: float = 600.0) -> bool:
        """Whether no stage of ``pid`` (the project in use where None) is computing, after waiting
        up to ``wait`` seconds for those that are."""
        from research.swarm import driver

        pid = pid or self.pid
        run = driver.Run(self.home / "driver" / str(pid), {"project": pid})
        end = time.monotonic() + wait
        while True:
            stages = (driver.view_of(run, self.client).get("stages") or {}).values()
            if not any((s or {}).get("status") in self.BUSY for s in stages):
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.5)

    def copy(self, pid: str) -> str:
        """A copy of the project ``pid`` under a new id: its folder as it stands once none of its
        stages is computing (the table, the decision log byte for byte, the stages computed and
        the fit records), its ``project.json`` naming the new id; the copy's id."""
        import secrets
        import shutil

        if not self.idle(pid):
            raise Stuck(f"The project {pid} kept computing, so it was not copied.")
        root = self.home / "projects"
        while True:
            new = "p" + secrets.token_hex(5)
            if not (root / new).exists():
                break
        shutil.copytree(root / pid, root / new)
        meta_path = root / new / "project.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        meta["id"] = new
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        return new

    def drop(self, pid: str, made: set[str]) -> bool:
        """The copy ``pid`` removed, once none of its stages is computing; only a copy this session
        made (``made``), never a project it did not. Whether it was removed."""
        import shutil

        if pid not in made or not self.idle(pid, wait=120.0):
            return False
        shutil.rmtree(self.home / "projects" / pid)
        shutil.rmtree(self.home / "driver" / pid, ignore_errors=True)  # its driver folder (``use``)
        made.discard(pid)
        return True

    def read(self, path: str) -> Any:
        r = self.client.get(f"/projects/{self.pid}/{path}")
        return r.json() if r.status == 200 else None

    # the driver's commands, by its own functions

    def _said(self, call: Callable[[], tuple[int, dict[str, Any]]]) -> Said:
        from research.swarm import driver

        before = len(self.errors)
        try:
            code, payload = call()
        except driver.DriverError as exc:
            code, payload = 1, {"refused": {"by": "driver", "code": exc.code,
                                            "message": exc.message, **exc.extra}}
        said = said_of(code, driver.private_free(payload))
        if len(self.errors) > before and said.kind != "failed":
            said = Said("failed", said.payload, by="server", code="http_500",
                        message=self.errors[-1]["said"], status=self.errors[-1]["status"])
        return said

    def ask(self, wait: float = 90.0) -> dict[str, Any]:
        from research.swarm import driver

        said = self._said(lambda: (0, driver.the_question(self.run, self.client, wait)))
        if not said.ok:
            raise Stuck(f"ask failed: {json.dumps(said.payload)[:800]}")
        return said.payload

    def answer(self, ids: Sequence[str], sets: Sequence[str], reason: str,
               marker: str = DEFAULT_MARKER) -> Said:
        from research.swarm import driver

        def call() -> tuple[int, dict[str, Any]]:
            decision, _ = driver.chosen(self.run, self.client, list(ids), list(sets))
            driver.guard_kind(decision)
            return self.post(decision, reason)

        return self._said(call)

    def post(self, decision: Mapping[str, Any], reason: str) -> tuple[int, dict[str, Any]]:
        """``decision`` posted with its reason as the driver's ``answer`` (a revert: its
        ``revisit``) posts it, the server's refusal said as the driver says it: its ways forward
        kept for the next ask (``last_refusal_exits``) until an answer is recorded, which clears
        them, as the driver's do."""
        from research.swarm import driver

        command = "revisit" if decision.get("kind") == "revert" else "answer"
        r = driver.post_decision(self.run, self.client, dict(decision), reason)
        if r.status != 200:
            return 1, {"posted": dict(decision)} | driver.refused(r, self.run, command)
        self.run.keep_exits([], None, command)
        return 0, driver.recorded(r.json(), str(decision["kind"]))

    def decide(self, decision: Mapping[str, Any], reason: str) -> Said:
        """A decision posted as composed (a revert, a recorded option again)."""
        return self._said(lambda: self.post(decision, reason))

    def revisit(self, record: str, reason: str) -> Said:
        return self.decide({"kind": "revert", "decision_id": record}, reason)

    def seal(self, family: str, reason: str) -> Said:
        """The held-out rows opened on ``family``, as the driver's ``seal`` posts it."""
        return self.decide({"kind": "open_seal", "family": family}, reason)

    def preview(self, ids: Sequence[str], sets: Sequence[str], reason: str) -> Said:
        from research.swarm import driver

        def call() -> tuple[int, dict[str, Any]]:
            decision, _ = driver.chosen(self.run, self.client, list(ids), list(sets))
            return self.preview_decision(decision)

        return self._said(call)

    def preview_decision(self, decision: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        from research.swarm import driver

        r = self.client.post(f"/projects/{self.pid}/preview", dict(decision))
        if r.status != 200:
            return 1, {"previewed": dict(decision)} | driver.refused(r, self.run, "preview")
        return 0, {"previewed": dict(decision), "preview": driver.trim(r.json())}

    def preview_of(self, decision: Mapping[str, Any]) -> Said:
        return self._said(lambda: self.preview_decision(decision))

    def fit(self, wait: float = 3600.0) -> Said:
        """Fit pressed and waited for, as the driver's ``fit`` presses it (no budget: the sweep's
        compute is its own), its status classed: fresh, refused by the engine in words, or failed
        with an error it did not type (``said.payload["fit"]``, the stage status; ``seconds``, the
        wall time from the press)."""
        from research.swarm import driver

        def call() -> tuple[int, dict[str, Any]]:
            started = time.monotonic()
            r = self.client.post(f"/projects/{self.pid}/fit")
            if r.status != 200:
                return 1, driver.refused(r, self.run, "fit")
            status = driver.wait_fit(self.run, self.client, wait)
            out = {"pressed": r.json() or {}, "fit": status,
                   "seconds": round(time.monotonic() - started, 3)}
            refusal = status.get("refusal")
            if status.get("status") == "error" and refusal:
                return 1, out | {"refused": {"by": "engine", "stage": refusal.get("stage"),
                                             "code": refusal.get("code"),
                                             "message": refusal.get("message"),
                                             "exits": opts.exits(refusal.get("exits"))}}
            return 0, out  # fresh, an untyped error, or not reached: classed by ``said_of``

        from research.swarm import options as opts

        return self._said(call)


# ── the policies ─────────────────────────────────────────────────────────────


def offered(q: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The open question's options that record something by themselves (no field of one's own)."""
    return [o for o in q.get("options") or [] if o.get("id") is not None
            and isinstance(o.get("decision"), Mapping) and not o.get("needs")]


def ways_of(q: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every way forward the ask lists: the ask card's, the last refusal's, and each quest line
    item's press."""
    readings = q.get("readings") or {}
    return [*(readings.get("exits") or []), *(q.get("last_refusal_exits") or []),
            *(w for line in q.get("also_open") or [] for item in line.get("items") or []
              for w in item.get("options") or []),
            *(w for s in readings.get("read_from_data") or [] for w in s.get("change") or [])]


def reading_ways(q: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """The ways that confirm a reading the server asks (not a role), by (reading, column)."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for way in ways_of(q):
        d = way.get("decision") or {}
        if way.get("id") and d.get("kind") == "confirm_reading" and d.get("reading") != "role":
            out.setdefault((str(d["reading"]), str(d["column"])), []).append(way)
    return out


def guess_of(q: Mapping[str, Any], reading: str, column: str) -> str | None:
    """The app's own guess for a reading, from its ask card's groups (journeys' ``GuessingTruth``)."""
    for g in (q.get("readings") or {}).get("groups") or []:
        if g.get("kind") == reading and column in (g.get("columns") or []) \
                and g.get("guess") is not None:
            return str(g["guess"])
    return None


@dataclass
class Default:
    """The default path's answers (the module docstring)."""

    question: Question
    proposed: dict[str, str] = field(default_factory=dict)
    name: str = "default"

    # what each draws from; the random path draws, the default takes the first
    def pick(self, items: Sequence[Any]) -> Any:
        return items[0]

    def reading(self, q: Mapping[str, Any]) -> tuple[str, str] | None:
        for (reading, column), ways in reading_ways(q).items():
            guess = guess_of(q, reading, column)
            way = next((w for w in ways if guess is not None
                        and str((w.get("decision") or {}).get("value")) == guess), None)
            way = way or self.pick(ways)
            return way["id"], f"the {reading} reading of `{column}`, at the app's guess" if guess \
                else f"the {reading} reading of `{column}`, the first way offered"
        unit = next((w for w in ways_of(q) if (w.get("decision") or {}).get("kind")
                     == "set_column_unit" and w.get("id")), None)
        if unit is not None:
            return unit["id"], "a column's unit, as the server reads it"
        return None

    def role(self, q: Mapping[str, Any]) -> tuple[str, str] | None:
        for way in ways_of(q):
            d = way.get("decision") or {}
            if not way.get("id"):
                continue
            if d.get("kind") == "confirm_role" and self.proposed.get(d.get("column")) == d.get("role"):
                return way["id"], "a role confirmed as the roles answer proposed it"
            if d.get("kind") == "confirm_readings" and d.get("items") and all(
                    i.get("reading") == "role" and self.proposed.get(i.get("column")) == i.get("value")
                    for i in d["items"]):
                return way["id"], "roles confirmed as the roles answer proposed them"
        return None

    def finding(self, q: Mapping[str, Any]) -> tuple[str, str] | None:
        for line in q.get("also_open") or []:
            if line.get("source") != "finding":
                continue
            acts = [w for w in line.get("options") or [] if w.get("id")]
            repairs = [w for w in acts if (w.get("decision") or {}).get("kind") == "apply_repair"]
            way = self._finding_act(repairs, acts)
            if way is not None:
                return way["id"], f"the finding {line.get('key')} acted on: {way.get('label')}"
        return None

    def _finding_act(self, repairs: list[dict[str, Any]], acts: list[dict[str, Any]]
                     ) -> dict[str, Any] | None:
        return repairs[0] if repairs else None

    def settle(self, q: Mapping[str, Any]) -> tuple[str, str] | None:
        return self.reading(q) or self.role(q) or self.finding(q)

    def settled(self, q: Mapping[str, Any], way: str) -> None:
        """Told that the walk answered ``way``, which :meth:`settle` gave on ``q``: nothing to
        keep here (the sweep's policy keeps the refusals' ways it takes, ``sweep.Named``)."""

    def answer(self, key: str, q: Mapping[str, Any], read: Callable[[str], Any]
               ) -> tuple[list[str], list[str], str]:
        """The answer to the open question ``key``: ids, ``--set`` fields and where it came from."""
        options = offered(q)
        ids = [str(o["id"]) for o in options]
        data = q.get("data") or {}
        fixed = self.fixed(key, q)
        if fixed is not None:
            return fixed
        if key == "roles":
            self.proposed = {c["column"]: c["proposed"] for c in data.get("columns") or []}
            return ["proposed"], [], "the app's proposed roles, confirmed"
        if key in ("exclusions", "missing", "split", "energy_adjustment", "models"):
            body = journeys_body(key, read, self.question.purpose)
            pick = self.choose_body(key, body, q)
            if pick is not None:
                return pick
        if key == "follow_up":
            return ["same"], [], "one period for everyone, the acceptance drivers' answer"
        if key == "grain":
            return ["one_row_per_unit"], [], "one row per unit, the card's population"
        if key == "temporal":
            return ["false"], [], "the journeys' answer: no time order asked"
        if key == "clusters":
            named = [g["column"] for g in data.get("groupings") or []]
            pick = next((o["id"] for o in q.get("options") or [] if o.get("column") in named[:1]
                         and (self.question.purpose == "prediction"
                              or str(o["id"]).startswith("cluster_only:"))), "none")
            return [pick], [], "the roles' named grouping, else none (the acceptance drivers)"
        if key == "adjustment":
            sets = []
            for o in q.get("options") or []:
                for column in o.get("columns") or []:
                    guess = o.get("guess") or {}
                    said = ",".join(str(guess.get(k) or "unknown") for k in
                                    ("causes_exposure", "causes_outcome", "after_exposure"))
                    if f"{column}=" not in "".join(sets):
                        sets.append(f"{column}={said}")
            return ["answers"], sets, "each covariate's causal place at the card's guess"
        if key == "time_varying":
            return ["standard"], ["ordering=exposure_precedes_outcome"], \
                "standard regression, the exposure before the outcome (the acceptance drivers)"
        if key == "form":
            return ["proposed"], [], "the form card's proposal"
        if key == "causal":
            return ["none"], [], "the primary model alone"
        if key == "substitution":
            return [ids[0]], ["step_kcal=100"], "the first pair the design offers, 100 kcal a step"
        if not ids:
            raise Stuck(f"No option the driver offers for {key!r} records anything by itself: "
                        f"{json.dumps(q.get('options'))[:600]}")
        return [ids[0]], [], "the app's first-ranked option"

    def fixed(self, key: str, q: Mapping[str, Any]) -> tuple[list[str], list[str], str] | None:
        """The card's answers (:data:`FIXED`, and the exposure under Estimate)."""
        options = [str(o["id"]) for o in q.get("options") or [] if o.get("id") is not None]
        if key == "target":
            return [self.question.target], [], "the card's outcome"
        if key == "event":
            level = self.question.event(options)
            if level is None:
                raise Stuck("The event question offers no level.")
            return [level], [], "the card's outcome level"
        if key == "purpose":
            return [self.question.purpose], [], "the card's purpose"
        if key == "task":
            detected = (q.get("data") or {}).get("detected_task")
            pick = next((str(o["id"]) for o in offered(q)
                         if (o.get("decision") or {}).get("task") == detected), None)
            if pick is None and offered(q):
                pick = str(offered(q)[0]["id"])
            if pick is None:
                raise Stuck("The task question offers nothing to record.")
            return [pick], [], "the app's detection of the outcome's kind"
        if key == "lens":
            # The lenses are listed, not ranked: the profile's hints are the app's own evidence,
            # and with none, the lens that claims no field ("other").
            hints = [h.get("lens") for h in (q.get("data") or {}).get("lens_hints") or []]
            pick = next((o for o in options if o in hints),
                        "other" if "other" in options else (options[0] if options else None))
            if pick is None:
                raise Stuck("The lens question offers no lens.")
            return [pick], [], "the profile's lens hint, else the lens that claims no field"
        if key == "estimand" and self.question.exposure is not None:
            if self.question.exposure not in options:
                raise Stuck(f"The card's exposure {self.question.exposure} is not offered: "
                            f"{options}")
            return [self.question.exposure], [], "the card's exposure, its first-ranked measure"
        return None

    def choose_body(self, key: str, body: Mapping[str, Any] | None, q: Mapping[str, Any]
                    ) -> tuple[list[str], list[str], str] | None:
        """journeys' answer for ``key`` as the driver offers it (``body``), by its id."""
        from research.swarm.options import canonical

        if body is None:
            return None
        if key == "models":
            return list(body["models"]), [], "the shelf's first-ranked families (journeys)"
        if key == "split":
            fields = {f.get("name"): f for f in q.get("fields") or []}
            initial = (fields.get("validation") or {}).get("initial")
            sets = [] if body["validation"] == initial else [f"validation={body['validation']}"]
            return [format(float(body["holdout"]), "g")], sets, \
                "the seal plan's first option (journeys)"
        pick = next((str(o["id"]) for o in offered(q)
                     if canonical(o["decision"]) == canonical(body)), None)
        if pick is None:
            return None
        return [pick], [], "the app's first-ranked option (journeys)"

    def declarations(self, q: Mapping[str, Any]) -> list[tuple[list[str], list[str], str]]:
        """The declaration lines this path answers before Fit: none (each stays as the app sets
        it)."""
        return []

    def instead(self, key: str, q: Mapping[str, Any], refused: set[str]
                ) -> tuple[list[str], list[str], str] | None:
        """Another option of ``key`` where those in ``refused`` were refused with no way forward:
        the first offered that records something by itself and was not refused (the random path
        draws one); None where none is left, and for the card's own answers, which no other
        option stands for."""
        if key in FIXED or key in ("target", "estimand"):
            return None
        left = [o for o in offered(q) if str(o["id"]) not in refused]
        if not left:
            return None
        pick = self.pick(left)
        return [str(pick["id"])], [], (f"the next option offered, {len(refused)} refused with no "
                                       f"way forward")

    def way(self, exits: Sequence[Mapping[str, Any]]) -> str | None:
        """A refusal's way forward to take: its first that records something."""
        ways = [str(w["id"]) for w in exits if w.get("id") and w.get("decision")]
        return self.pick(ways) if ways else None

    def family(self, families: Sequence[str]) -> str:
        return self.pick(list(families))


class Random(Default):
    """The random path's answers (the module docstring), each draw from ``numpy``'s generator at
    ``seed``."""

    def __init__(self, question: Question, seed: int):
        import numpy as np

        super().__init__(question=question, name=f"random:{seed}")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def pick(self, items: Sequence[Any]) -> Any:
        return items[int(self.rng.integers(len(items)))]

    def _finding_act(self, repairs: list[dict[str, Any]], acts: list[dict[str, Any]]
                     ) -> dict[str, Any] | None:
        return self.pick(acts) if acts else None

    def answer(self, key: str, q: Mapping[str, Any], read: Callable[[str], Any]
               ) -> tuple[list[str], list[str], str]:
        fixed = self.fixed(key, q)
        if fixed is not None:
            return fixed
        if key == "roles":
            return super().answer(key, q, read)
        options = offered(q)
        if not options:
            return super().answer(key, q, read)  # a question answered only with fields
        pick = self.pick(options)
        return [str(pick["id"])], [], f"drawn at random from the {len(options)} options offered"

    def declarations(self, q: Mapping[str, Any]) -> list[tuple[list[str], list[str], str]]:
        from research.swarm import declarations as decl

        out = []
        for line in q.get("also_open") or []:
            if line.get("source") != "declaration" or line.get("key") not in decl.kinds():
                continue
            options = [o for o in line.get("options") or [] if o.get("id")
                       and isinstance(o.get("decision"), Mapping) and not o.get("needs")]
            if options:
                pick = self.pick(options)
                out.append(([str(pick["id"])], [], f"the {line['key']} line, drawn at random "
                                                   f"from its {len(options)} options"))
        return out


def journeys_body(key: str, read: Callable[[str], Any], purpose: str) -> dict[str, Any] | None:
    """journeys.py's own answer to ``key`` where it reads the app's first-ranked option from an
    artifact (``first_exclusions``, ``first_missing``, ``first_split``, ``first_energy``,
    ``first_models``), over ``read`` (``stages/<stage>`` as the session serves it)."""
    from types import SimpleNamespace

    from turbotab.core.reference import journeys as J
    from turbotab.server import app  # noqa: F401 - every family the server registers

    def artifact(stage: str, timeout: Any = None) -> dict[str, Any]:
        served = read(f"stages/{stage}") or {}
        return served.get("artifact") or {}

    run = SimpleNamespace(drive=SimpleNamespace(artifact=artifact), note=lambda text: None)
    firsts = {"exclusions": J.first_exclusions, "missing": J.first_missing,
              "split": J.first_split, "energy_adjustment": J.first_energy}
    if key in firsts:
        return firsts[key](run)
    if key == "models":
        return J.first_models(run, {"state": {"purpose": purpose}})
    return None


# ── the walk ─────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """One answer a walk gave: where (a question, a reading, a role line, a finding, a refusal's
    way forward, a declaration line), the options the driver offered, what was chosen and why, and
    what the answer said."""

    at: str
    key: str | None
    offered: list[dict[str, Any]]
    chosen: list[str]
    sets: list[str]
    why: str
    said: str
    code: str | None = None
    record: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _offered_of(q: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{k: o.get(k) for k in ("id", "label", "decision", "needs")} for o in q.get("options") or []]


def walk(session: Any, policy: Default, *, stop: Callable[[dict[str, Any]], bool] | None = None,
         steps: list[Step] | None = None, before: Callable[[str, dict[str, Any]], None] | None = None,
         limit: int = STEPS) -> dict[str, Any]:
    """Answer the Router through ``session`` by ``policy`` until Fit is reached, the held-out rows
    are what is left to open, or ``stop(ask)``; returns that ask. Each answer is kept in ``steps``;
    ``before(key, ask)`` runs before an open question is answered (the crash sweep's preview of the
    answer). An answer refused in words takes the way forward the policy picks (a refusal that only
    asks for a reading takes the reading). What the engine said stops the walk classed
    (:class:`Met`): an answer that failed, an answer refused with no way forward and no other
    option left, a question waiting on stages that failed (typed as the driver types them,
    :func:`failed_stages`); any other limit stops it in its words (:class:`Stuck`)."""
    steps = steps if steps is not None else []
    read = session.read
    waits = 0
    n = len(steps)
    refused: dict[str, set[str]] = {}  # each question's options refused with no way forward
    last: dict[str, Said] = {}  # each question's last such refusal
    for _ in range(limit):
        q = session.ask()
        settled = policy.settle(q)
        if settled is not None:
            way, why = settled
            said = session.answer([way], [], f"{policy.name}: {why}.")
            policy.settled(q, way)
            steps.append(Step("way", None, [], [way], [], why, said.kind, said.code,
                              (said.payload.get("recorded") if said.ok else None)))
            if said.kind == "failed":
                raise met_answer(f"An answer failed: {json.dumps(said.payload)[:800]}", None,
                                 said, [way])
            continue
        if stop is not None and stop(q):
            return q
        question = q.get("question")
        if question is None or q.get("fit_reached") or question.get("key") == "open_seal":
            return q
        if question.get("status") != "open":
            if q.get("failed"):
                raise met_stages(failed_stages(session, q),
                                 f"The {question.get('key')} question waits on stages that "
                                 f"failed: {json.dumps(q['failed'])[:800]}")
            waits += 1
            if waits > 40:
                raise Stuck(f"The {question.get('key')} question kept waiting.")
            time.sleep(0.5)
            continue
        waits = 0
        key = str(question["key"])
        ids, sets, why = policy.answer(key, q, read)
        if ids and ids[0] in refused.get(key, set()):
            other = policy.instead(key, q, refused[key])
            if other is None:
                raise met_answer(f"Every option of the {key} question this path takes was "
                                 f"refused with no way forward "
                                 f"({', '.join(sorted(refused[key]))}).", key, last[key],
                                 sorted(refused[key]))
            ids, sets, why = other
        if before is not None:
            before(key, q)
        said = session.answer(ids, sets, f"{policy.name}: {key} answered from {why}.",
                              MARKER_OF.get(key, DEFAULT_MARKER))
        steps.append(Step("question", key, _offered_of(q), list(ids), list(sets), why, said.kind,
                          said.code, said.payload.get("recorded") if said.ok else None))
        if said.kind == "failed":
            raise met_answer(f"The {key} answer failed: {json.dumps(said.payload)[:800]}", key,
                             said, ids)
        if said.kind == "refused":
            way = policy.way(said.exits)
            if way is None:
                # A refusal that asks for a reading is settled on the next ask; any other leaves
                # this option out, and the path takes another (``Default.instead``).
                q2 = session.ask()
                if policy.settle(q2) is None:
                    refused.setdefault(key, set()).add(ids[0])
                    last[key] = said
                    if policy.instead(key, q, refused[key]) is None:
                        raise met_answer(f"The {key} answer was refused with no way forward: "
                                         f"{said.code}: {said.message}", key, said,
                                         sorted(refused[key]))
                continue
            again = session.answer([way], [], f"{policy.name}: the refusal's way forward.")
            steps.append(Step("way", key, list(said.exits), [way], [], "the refusal's way forward",
                              again.kind, again.code,
                              again.payload.get("recorded") if again.ok else None))
            if again.kind == "failed":
                raise met_answer(f"The {key} refusal's way forward failed: "
                                 f"{json.dumps(again.payload)[:800]}", key, again, [way])
    raise Stuck(f"The Router never reached Fit in {limit} answers ({len(steps) - n} given).")


# The dry run's tag of each question's answer (``dry_run.MARKER_OF``), read from it so the two
# never disagree.
def _markers() -> dict[str, str]:
    from research.swarm.dry_run import MARKER_OF as markers

    return dict(markers)


MARKER_OF = _markers()


def fit_through(session: Any, policy: Default, steps: list[Step], *, tries: int = 3) -> Said:
    """Fit pressed; a fit the engine refused in words takes the way forward the policy picks and the
    walk goes on, up to ``tries`` presses; the last press's answer."""
    said = session.fit()
    for _ in range(tries - 1):
        if said.kind != "refused" or said.by != "engine":
            return said
        way = policy.way(said.exits)
        steps.append(Step("fit", "fit", list(said.exits), [way] if way else [], [],
                          "the fit's refusal", said.kind, said.code))
        if way is None:
            return said
        took = session.answer([way], [], f"{policy.name}: the fit's way forward.")
        if took.kind == "failed":
            return took
        walk(session, policy, steps=steps)
        said = session.fit()
    return said


def seal_through(session: Any, policy: Default, family: str, steps: list[Step], *,
                 tries: int = 6) -> Said:
    """The held-out rows opened on ``family``; where the opening is refused in words (it waits for
    the triage of the open noticings, say), the way forward that confirms the triage, else the
    first that is not the opening itself, then the opening again (the dry run's ``seal``)."""
    reason = f"{policy.name}: {family} declared final."
    for _ in range(tries):
        opened = session.seal(family, reason)
        if opened.ok:
            return opened
        if opened.code == "fit_not_fresh":  # the opening waits for a current fit: press it
            pressed = session.fit()
            steps.append(Step("seal", "open_seal", [], [], [], "the opening waits for the fit",
                              pressed.kind, pressed.code))
            if not pressed.ok:
                raise Stuck(f"The fit the opening waits for was not made: {pressed.code}: "
                            f"{pressed.message}")
            continue
        ways = [w for w in opened.exits if w.get("decision") and w.get("id")]
        way = next((w for w in ways if w["decision"].get("kind") == "confirm_sweep"), None) or \
            next((w for w in ways if w["decision"].get("kind") != "open_seal"), None)
        steps.append(Step("seal", "open_seal", list(opened.exits),
                          [way["id"]] if way else [], [], "the opening's refusal", opened.kind,
                          opened.code))
        if way is None:
            raise Stuck(f"The seal was not opened: {opened.code}: {opened.message}")
        took = session.answer([way["id"]], [], f"{policy.name}: the opening's way forward, "
                                               f"{way.get('label')}.")
        if took.kind == "failed":
            raise Stuck(f"The opening's way forward failed: {json.dumps(took.payload)[:800]}")
    raise Stuck("The seal never opened.")


# ── one arm, to its record ───────────────────────────────────────────────────


def record_meta(dataset: str, model: str) -> dict[str, Any]:
    """What an arm's record is given (``assemble.assemble``'s ``meta``): design 2, the arm as its
    model, no persona, and the dataset as the launcher gives an analyst's run
    (``launcher.dataset_meta``: its control kind, fixture and known answer, and its card)."""
    from research.swarm.launcher import dataset_meta

    known = {k: v for k, v in dataset_meta(dataset).items() if k != "known_source"}
    return {"design": 2, "analyst_model": model, "persona": None, "effort": None,
            "prompt_version": PROMPT_VERSION, "dataset": known, "help": []}


def families_of(q: Mapping[str, Any], view: Mapping[str, Any]) -> list[str]:
    if (q.get("question") or {}).get("key") == "open_seal":
        found = [str(o["id"]) for o in q.get("options") or [] if o.get("id")]
        if found:
            return found
    return [str(f) for f in (view.get("state") or {}).get("models") or []]


def run_arm(dataset: str, policy_name: str, *, seed: int | None, runs: Path, corpus: Path,
            budget_fits: int = 40, budget_seconds: float = 4 * 3600.0,
            data: Path | None = None, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One mechanical arm on ``dataset`` (the module docstring), through a run of the driver's own,
    to its record in ``corpus`` (``assemble.assemble``): its card's question, the default or the
    random path, Fit, under Predict the seal on the family the path declares final, the export.
    ``meta`` (the launcher's ``dataset`` and the rest) is given to the record; without it, the
    dataset's id and card. Returns what was done: the run, its steps, the record and whether it
    validated, or where the arm stopped."""
    from research.swarm import assemble as A
    from research.swarm.dry_run import table_of

    card = card_of(dataset)
    question = Question.of_card(card)
    policy = Default(question) if policy_name == "default" else Random(question, int(seed or 0))
    table = data if data is not None else table_of(dataset, Path(runs) / "data" / dataset)
    session = Driven(runs)
    model = "mechanical-default" if policy_name == "default" else f"mechanical-random-{seed}"
    session.start(table, budget_fits=budget_fits, budget_seconds=budget_seconds, model=model)
    steps: list[Step] = []
    error = None
    try:
        try:
            q = walk(session, policy, steps=steps)
            for ids, sets, why in policy.declarations(q):
                said = session.answer(ids, sets, f"{policy.name}: {why}.")
                steps.append(Step("declaration", ids[0].split(":", 1)[0], [], ids, sets, why,
                                  said.kind, said.code))
            said = fit_through(session, policy, steps)
            if not said.ok:
                raise Stuck(f"Fit was not made: {said.code}: {said.message}")
            session.results()
            if question.purpose == "prediction":
                # What the fit raises (a finding, a reading) is answered, and Fit pressed again
                # for the answers as they then stand, so the seal opens on a current fit.
                for _ in range(3):
                    before = len(steps)
                    q = walk(session, policy, steps=steps)
                    if len(steps) == before:
                        break
                    said = fit_through(session, policy, steps)
                    if not said.ok:
                        raise Stuck(f"Fit was not made again: {said.code}: {said.message}")
                families = families_of(q, session.view())
                if not families:
                    raise Stuck("No family was fitted to declare final.")
                seal_through(session, policy, policy.family(families), steps)
                session.results()
            exported = session.export()
            if not exported.ok:
                raise Stuck(f"The export was refused: {exported.code}: {exported.message}")
        except Stuck as exc:
            error = str(exc)
        given = dict(meta) if meta is not None else record_meta(dataset, model)
        path, ok = A.assemble(session.runs / str(session.run), meta=given, corpus=Path(corpus))
    finally:
        session.stop()
    return {"dataset": dataset, "arm": policy.name, "run": session.run, "record": str(path),
            "valid": ok, "error": error, "steps": [s.json() for s in steps]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.arms",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("arm", choices=("default", "random"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runs", type=Path, default=REPO / "research" / "runs" / "arms")
    parser.add_argument("--corpus", type=Path, default=REPO / "research" / "corpus" / "arms")
    args = parser.parse_args(argv)
    if args.arm == "random" and args.seed is None:
        parser.error("the random path takes --seed")
    out = run_arm(args.dataset, args.arm, seed=args.seed, runs=args.runs, corpus=args.corpus)
    sys.stdout.write(json.dumps({k: out[k] for k in ("dataset", "arm", "run", "record", "valid",
                                                     "error")}) + "\n")
    return 0 if out["valid"] and not out["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

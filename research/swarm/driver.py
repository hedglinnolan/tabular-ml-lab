"""The swarm's driver (DRIVER_PLAN D1; SWARM_PLAN §2): an analyst works TurboTab through the app's
own API, one small command at a time, each printing JSON.

    python -m research.swarm.driver start --data <file> [--provenance url,date,license]
                                         --budget-fits N --budget-seconds S [--run <id>]
    python -m research.swarm.driver ask      --run <id>
    python -m research.swarm.driver preview  --run <id> <option> [--set name=value ...]
    python -m research.swarm.driver answer   --run <id> <option> [<option> ...] [--set ...] --reason <why>
    python -m research.swarm.driver revisit  --run <id> <decision id> --reason <why>
    python -m research.swarm.driver fit      --run <id>
    python -m research.swarm.driver results  --run <id>
    python -m research.swarm.driver seal     --run <id> <family> --reason <why>
    python -m research.swarm.driver export   --run <id>
    python -m research.swarm.driver stop     --run <id>

**One run** is one directory, ``research/runs/<run id>/`` (``$TURBOTAB_SWARM_RUNS`` in its place
for a test): ``home/`` is the server's ``TURBOTAB_HOME``; ``driver.json`` holds the run's id, the
PID and port of the server the driver started, the project's id, the fit budget, the data file
and where it came from, and when the run started and ended; ``commands.jsonl`` holds every command
with its arguments and a summary of its response; ``server.log`` is the server's output;
``exits.json`` the ways forward of the last refusal, so ``answer`` can take one; ``export.zip``
the bundle. ``start`` runs ``python -m turbotab.server --analyst`` on a free local port with two
workers and uploads the file; ``stop`` stops that PID and no other.

**A run refuses to start where the analyst cannot be held to the driver** (DRIVER_PLAN D2): the
analyst agent type (``.claude/agents/analyst.md``, ``tools: Bash``) and its PreToolUse hook
(``research/swarm/analyst_hook.py``) wired for every tool in a Claude Code settings file
(:data:`SETTINGS_FILES`, read and never written). ``start --scripted`` starts a run whose analyst
is a script, not an agent (the dry run, the tests), which needs neither; ``driver.json`` says
which (``analyst``).

**What the analyst reads is the server's.** ``ask`` serves the Router's open question as the quest
log and its ask card serve it (its words from the question's teaching entry, its quest line, its
options with the ids the server serves, the readings it rests on, and the ways forward on hand),
read as ``turbotab/core/reference/journeys.py`` reads open questions (:data:`journeys.OPEN`,
:func:`journeys.reaches_fit`); none of journeys.py's answer sources is used. Options are composed
by :mod:`research.swarm.options`.

**What the driver refuses**, whatever the analyst asks (SWARM_PLAN §2; DRIVER_PLAN D1):

* a ``reseal`` (the server refuses one in analyst mode too, and offers none), a ``lock_plan`` (the
  server's own), a ``revert`` other than by ``revisit``, and an ``open_seal`` other than by ``seal``;
* ``fit`` past the run's fit budget: its count of fits and their seconds, read from
  ``GET /fits``'s records (SWARM_ENGINE_PLAN E1), with the fit under way counted among them;
* once those meet the budget, an ``answer`` or ``revisit`` that would make a new fit, since the
  server makes and serves it without another press, and one the driver cannot tell makes none
  (the fit budget's section below); one it can tell makes none (the triage, which the fit does
  not read; a revisit that leaves the models unanswered; going back to answers already fitted)
  is still recorded;
* once the held-out rows are opened, any change or attempt (``preview``, ``answer``, ``revisit``,
  ``fit``): the reported result is the opening's score, and nothing more is tried (SWARM_PLAN §3).
  ``seal`` again goes to the server, which refuses a second opening.

Exit status: 0 when the command did what it says, 1 when the server or the driver refused it (the
JSON says why, with the ways forward), 2 when the command line does not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research.swarm import options as opts

REPO = Path(__file__).resolve().parents[2]
RUNS_ENV = "TURBOTAB_SWARM_RUNS"
WORKERS = 2  # DRIVER_PLAN D1: two workers per run's server
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
REQUEST_SECONDS = 120.0
HEALTH_SECONDS = 180.0
INGEST_SECONDS = 600.0
STOP_SECONDS = 30.0
WAIT_DEFAULT = 90.0  # under the analyst's shell's own two-minute limit
WAIT_MAX = 540.0
NOT_YET_SECONDS = 60.0
HOOK_SCRIPT = "research/swarm/analyst_hook.py"
AGENT_FILE = REPO / ".claude" / "agents" / "analyst.md"
# Where Claude Code reads hooks from, project first (the owner wires the hook locally).
SETTINGS_FILES = (REPO / ".claude" / "settings.local.json", REPO / ".claude" / "settings.json",
                  Path.home() / ".claude" / "settings.json")
EVERY_TOOL = (None, "", "*")
# The driver's own refusals: code and message (the only words the driver writes).
NOT_THIS_WAY = {
    "reseal": "A re-seal is never recorded in a run: the held-out rows open once, and the reported "
              "result is the score at that opening.",
    "lock_plan": "The plan's lock is recorded by TurboTab itself when Fit is pressed: run fit.",
    "revert": "An earlier decision is changed with revisit, which records why.",
    "open_seal": "The held-out rows are opened with seal, which records why.",
}
AFTER_SEAL = ("The held-out rows are open: the run's result is the score at the opening, and "
              "nothing more is tried. ask, results and export still read the run.")


class DriverError(Exception):
    """A refusal of the driver's own: a code, a plain message, and what it says beside them."""

    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def runs_root() -> Path:
    return Path(os.environ.get(RUNS_ENV) or REPO / "research" / "runs").expanduser().absolute()


# ── HTTP: the stdlib only, so every command starts fast ──────────────────────


@dataclass
class Response:
    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8")) if self.body else None
        except ValueError:
            return None

    def error(self) -> dict[str, Any]:
        """The server's refusal (``{"error": {code, message, exits}}``), or a 422's detail."""
        data = self.json()
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            return data["error"]
        if isinstance(data, dict) and "detail" in data:
            return {"code": "invalid", "message": "The server could not read the request.",
                    "detail": data["detail"]}
        return {"code": f"http_{self.status}", "message": self.body.decode("utf-8", "replace")[:600]}


class Client:
    def __init__(self, port: int, timeout: float = REQUEST_SECONDS):
        self.port = port
        self.base = f"http://127.0.0.1:{port}/api"
        self.timeout = timeout

    def request(self, method: str, path: str, *, body: bytes | None = None,
                headers: dict[str, str] | None = None, query: dict[str, Any] | None = None) -> Response:
        url = self.base + path
        q = {k: v for k, v in (query or {}).items() if v is not None}
        if q:
            url += "?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return Response(r.status, r.read(), dict(r.headers))
        except urllib.error.HTTPError as e:
            return Response(e.code, e.read(), dict(e.headers or {}))
        except OSError as e:  # refused, reset or timed out: no answer at all
            raise DriverError("server_unreachable", f"The run's server did not answer on port "
                                                    f"{self.port} ({type(e).__name__}).") from None

    def get(self, path: str, **query: Any) -> Response:
        return self.request("GET", path, query=query)

    def post(self, path: str, data: Any = None, **query: Any) -> Response:
        body = None if data is None else json.dumps(data).encode("utf-8")
        return self.request("POST", path, body=body, query=query,
                            headers={"Content-Type": "application/json"} if body else {})

    def upload(self, path: str, file: Path, **query: Any) -> Response:
        boundary = "turbotab-driver-" + secrets.token_hex(8)
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{file.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
        body = head.encode("utf-8") + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", path, body=body, query=query,
                            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


# ── the run ──────────────────────────────────────────────────────────────────


@dataclass
class Run:
    folder: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.folder.name

    @classmethod
    def load(cls, run_id: str) -> "Run":
        if not RUN_ID.match(run_id or ""):
            raise DriverError("no_such_run", f"{run_id!r} is not a run id.")
        folder = runs_root() / run_id
        try:
            meta = json.loads((folder / "driver.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise DriverError("no_such_run", f"There is no run {run_id}.") from None
        return cls(folder, meta)

    def save(self) -> None:
        path = self.folder / "driver.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    def log(self, entry: dict[str, Any]) -> None:
        with (self.folder / "commands.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    @property
    def project(self) -> str:
        return str(self.meta["project"])

    def client(self) -> Client:
        if self.meta.get("ended"):
            raise DriverError("run_stopped", f"Run {self.id} was stopped at {self.meta['ended']}.")
        return Client(int(self.meta["port"]))

    def keep_exits(self, ways: list[dict[str, Any]], code: str | None, command: str) -> None:
        (self.folder / "exits.json").write_text(json.dumps(
            {"command": command, "code": code, "exits": ways}, ensure_ascii=False) + "\n",
            encoding="utf-8")

    def kept_exits(self) -> list[dict[str, Any]]:
        try:
            return list(json.loads((self.folder / "exits.json").read_text("utf-8"))["exits"])
        except (OSError, ValueError, KeyError):
            return []


# ── the server's process ─────────────────────────────────────────────────────


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ps(*args: str) -> str:
    try:
        return subprocess.run(["ps", *args], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def alive(pid: int) -> bool:
    """Whether ``pid`` runs (a zombie does not: it is reaped here when it is this process's)."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return _ps("-o", "stat=", "-p", str(pid)).strip()[:1] not in ("Z", "")


def command_of(pid: int) -> str:
    return _ps("-o", "command=", "-p", str(pid)).strip()


def descendants(pid: int) -> list[int]:
    """The processes ``pid`` started and theirs (its job workers), from ``ps``."""
    children: dict[int, list[int]] = {}
    for line in _ps("-A", "-o", "pid=,ppid=").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    out, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in out:
                out.append(child)
                todo.append(child)
    return out


def ours(run: Run) -> bool:
    """Whether the recorded PID is still the server this run started (a PID can be reused)."""
    pid = int(run.meta.get("pid") or 0)
    return pid > 0 and alive(pid) and "turbotab.server" in command_of(pid) \
        and f"--port {run.meta.get('port')}" in command_of(pid)


def wait_gone(pids: list[int], seconds: float) -> list[int]:
    end = time.monotonic() + seconds
    left = [p for p in pids if alive(p)]
    while left and time.monotonic() < end:
        time.sleep(0.2)
        left = [p for p in left if alive(p)]
    return left


def start_server(folder: Path, port: int) -> subprocess.Popen:
    env = {**os.environ, "TURBOTAB_HOME": str(folder / "home"), "TURBOTAB_WORKERS": str(WORKERS),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS") or "2", "TURBOTAB_MODE": "local",
           "TURBOTAB_ANALYST": "1", "PYTHONUNBUFFERED": "1"}
    log = (folder / "server.log").open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "turbotab.server", "--analyst", "--port", str(port)],
            cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    finally:
        log.close()


def stop_pid(pid: int, children: list[int]) -> dict[str, Any]:
    """Stop the server ``pid`` (SIGTERM, then SIGKILL after :data:`STOP_SECONDS`), then make sure
    the job workers it started went with it; one still running is stopped by its own PID."""
    killed = False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if wait_gone([pid], STOP_SECONDS):
        os.kill(pid, signal.SIGKILL)
        killed = True
        wait_gone([pid], 10.0)
    left = wait_gone(children, 10.0)
    for child in left:
        try:
            os.kill(child, signal.SIGTERM)
        except ProcessLookupError:
            pass
    left_after = wait_gone(left, 10.0)
    for child in left_after:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"pid": pid, "killed": killed, "workers": children, "workers_stopped_by_pid": left,
            "running": [p for p in [pid, *children] if alive(p)]}


# ── reading the project ──────────────────────────────────────────────────────


def view_of(run: Run, client: Client) -> dict[str, Any]:
    r = client.get(f"/projects/{run.project}")
    if r.status != 200:
        raise DriverError("server_unreadable", f"The project could not be read ({r.status}).",
                          server=r.error())
    return r.json()


def served(run: Run, client: Client) -> Callable[[str], Any]:
    cache: dict[str, Any] = {}

    def artifact(stage: str) -> Any:
        if stage not in cache:
            r = client.get(f"/projects/{run.project}/stages/{stage}")
            cache[stage] = ((r.json() or {}).get("artifact") if r.status == 200 else None)
        return cache[stage]

    return artifact


def open_step(view: dict[str, Any]) -> dict[str, Any] | None:
    from turbotab.core.reference.journeys import OPEN

    return next((s for s in view.get("interview") or [] if s.get("status") in OPEN), None)


def fit_reached(view: dict[str, Any]) -> bool:
    from turbotab.core.reference.journeys import reaches_fit

    return reaches_fit(view)


def failed_on(view: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    stages = view.get("stages") or {}
    return {w: stages[w].get("error") for w in step.get("waiting_on") or []
            if (stages.get(w) or {}).get("status") == "error"}


def wait_open(run: Run, client: Client, seconds: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The view once its first unsettled question is open, Fit is reached, a stage it waits on
    failed, or ``seconds`` pass."""
    end = time.monotonic() + seconds
    while True:
        view = view_of(run, client)
        step = open_step(view)
        if step is None or step["status"] == "open" or fit_reached(view) or failed_on(view, step) \
                or time.monotonic() >= end:
            return view, step
        time.sleep(0.5)


def teaching(client: Client, purpose: str | None, key: str) -> dict[str, Any] | None:
    r = client.get("/teaching", purpose=purpose if purpose in ("prediction", "inference") else None)
    for entry in (r.json() or []) if r.status == 200 else []:
        if entry.get("key") == key:
            return entry
    return None


def quest_lines(run: Run, client: Client) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    r = client.get(f"/projects/{run.project}/quest")
    quest = r.json() if r.status == 200 else {}
    lines = [dict(ln, stage=st.get("key")) for st in (quest or {}).get("stages") or []
             for ln in st.get("lines") or []]
    return lines, (quest or {}).get("fit")


def line_items(line: dict[str, Any]) -> list[dict[str, Any]]:
    """A quest line's readings, each press with its id (``options.exits``)."""
    out = []
    for item in line.get("items") or []:
        out.append({k: item.get(k) for k in ("kind", "column", "value", "words", "evidence")}
                   | {"options": opts.exits(item.get("options"))})
    return out


def finding_options(run: Run, client: Client, purpose: str | None) -> dict[str, list[dict[str, Any]]]:
    """Each finding's ways to act on it, as its card offers them: its repairs (the findings
    artifact's, each with the ``apply_repair`` it records) and dismissing it, worded by the repairs
    question's teaching entry; by finding id."""
    r = client.get(f"/projects/{run.project}/stages/findings")
    art = ((r.json() or {}).get("artifact") or {}) if r.status == 200 else {}
    entry = teaching(client, purpose, "repairs") or {}
    dismiss = next((o for o in entry.get("options") or [] if o.get("value") == "dismiss"), {})
    out: dict[str, list[dict[str, Any]]] = {}
    for f in art.get("findings") or []:
        if not isinstance(f, dict) or not f.get("id") or f.get("disposition"):
            continue
        acts = [w | {"consequence": rep.get("consequence")}
                for rep in f.get("repairs") or [] for w in opts.exits([rep])]
        acts += opts.exits([{"label": dismiss.get("label"),
                             "decision": {"kind": "dismiss_finding", "finding_id": f["id"],
                                          "reason": None}}],
                           consequence=dismiss.get("consequence"))
        out[str(f["id"])] = acts
    return out


LINE_KEEP = ("id", "key", "source", "label", "name", "status", "reason", "waiting_for", "computing",
             "reopened_by", "would_change", "changes_nothing", "changed_since", "stage")


def describe_line(line: dict[str, Any]) -> dict[str, Any]:
    out = {k: line.get(k) for k in LINE_KEEP if line.get(k) not in (None, [], {})}
    items = line_items(line)
    if items:
        out["items"] = items
    return out


def decision_log(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [{k: r.get(k) for k in ("id", "seq", "sentence", "note", "post_seal", "after_estimates")
             if r.get(k) not in (None, False)} | {"kind": (r.get("decision") or {}).get("kind")}
            for r in view.get("decisions") or []]


def the_question(run: Run, client: Client, wait: float) -> dict[str, Any]:
    """What ``ask`` serves, with the private parts ``preview`` and ``answer`` resolve against."""
    view, step = wait_open(run, client, wait)
    state = view.get("state") or {}
    lines, fit = quest_lines(run, client)
    out: dict[str, Any] = {"run": run.id, "purpose": state.get("purpose"),
                           "seal_opened": bool(state.get("seal_opened"))}
    if fit:
        out["fit"] = {k: fit.get(k) for k in ("pressed", "locked", "waits", "sentence", "why")
                      if fit.get(k) not in (None, "")}
    ways: list[dict[str, Any]] = []
    composed: dict[str, Any] = {"options": [], "fields": []}
    if step is None:
        out["question"] = None
        # Every question answered: Fit is what is left until it is pressed (under Estimate the
        # plan's lock), as the flowchart shows it.
        out["fit_reached"] = not (fit or {}).get("pressed")
    else:
        key = step["key"]
        out["question"] = {k: step.get(k) for k in ("key", "status", "reason", "waiting_on",
                                                    "followup", "deferred_findings", "decision_id")
                           if step.get(k) not in (None, [], "")}
        if step["status"] == "waiting":
            out["fit_reached"] = fit_reached(view)
            failed = failed_on(view, step)
            if failed:
                out["failed"] = failed
        entry = teaching(client, state.get("purpose"), key)
        if entry:
            out["words"] = {k: entry.get(k) for k in ("title", "question", "one_liner", "why",
                                                      "consumer", "terms") if entry.get(k)}
        own = next((ln for ln in lines if ln.get("source") == "question" and ln.get("key") == key),
                   None)
        if own:
            out["line"] = describe_line(own)
        if step["status"] == "open":
            composed = opts.compose(key, step, view, served(run, client), entry)
            out["options"] = composed["options"]
            if composed.get("fields"):
                out["fields"] = composed["fields"]
            if composed.get("multi"):
                out["multi"] = True
            if composed.get("data"):
                out["data"] = composed["data"]
            if composed.get("seal"):
                out["by"] = "seal"
        card = step.get("ask")
        if card:
            card_exits = opts.exits(card.get("exits"))
            out["readings"] = {"text": card.get("text"), "consumer": card.get("consumer"),
                               "groups": card.get("groups"), "exits": card_exits}
            settled = [{k: s.get(k) for k in ("kind", "column", "value", "words", "evidence")}
                       | {"change": opts.exits(s.get("change"))} for s in card.get("read_from_data") or []]
            if settled:
                out["readings"]["read_from_data"] = settled
            ways += card_exits + [w for s in settled for w in s["change"]]
    others = [ln for ln in lines if ln.get("status") == "open" and ln.get("counted")
              and not (step is not None and ln.get("source") == "question" and ln.get("key") == step["key"])]
    if others:
        described = [describe_line(ln) for ln in others]
        acts = finding_options(run, client, state.get("purpose"))
        for line in described:
            if line.get("source") == "finding" and line.get("key") in acts:
                line["options"] = acts[line["key"]]
                ways += line["options"]
        out["also_open"] = described
        ways += [w for ln in others for item in line_items(ln) for w in item["options"]]
    kept = run.kept_exits()
    if kept:
        out["last_refusal_exits"] = kept
        ways += kept
    out["decisions"] = decision_log(view)
    out["_composed"], out["_ways"], out["_view"] = composed, ways, view
    return out


def private_free(payload: dict[str, Any]) -> dict[str, Any]:
    return opts.public(payload)


# ── posting ──────────────────────────────────────────────────────────────────


def refused(r: Response, run: Run, command: str) -> dict[str, Any]:
    """A server refusal as the analyst reads it, its ways forward kept for ``answer``."""
    error = r.error()
    ways = opts.exits(error.get("exits"))
    run.keep_exits(ways, error.get("code"), command)
    out = {"refused": {"by": "server", "status": r.status, "code": error.get("code"),
                       "message": error.get("message"), "exits": ways}}
    if error.get("detail") is not None:
        out["refused"]["detail"] = error["detail"]
    return out


def post_decision(run: Run, client: Client, decision: dict[str, Any], note: str) -> Response:
    """Post ``decision`` with its reason; while the Router holds its question behind one still
    computing (``not_yet``), post again (``journeys``'s own wait, mechanically)."""
    end = time.monotonic() + NOT_YET_SECONDS
    while True:
        r = client.post(f"/projects/{run.project}/decisions", decision, note=note)
        if r.status != 409 or r.error().get("code") != "not_yet" or time.monotonic() > end:
            return r
        time.sleep(0.5)


def the_record(view: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for record in reversed(view.get("decisions") or []):
        if (record.get("decision") or {}).get("kind") == kind:
            return record
    return None


def recorded(view: dict[str, Any], kind: str) -> dict[str, Any]:
    record = the_record(view, kind) or {}
    step = open_step(view)
    return {"recorded": {k: record.get(k) for k in ("id", "seq", "sentence", "note", "post_seal",
                                                    "after_estimates", "decision")
                         if record.get(k) is not None},
            "next": {"key": step["key"], "status": step["status"]} if step else None}


def reason_of(text: str | None) -> str:
    text = (text or "").strip()
    if not text:
        raise DriverError("reason_required", "Say why: --reason takes the hypothesis this tests, or "
                                             "why this option.")
    return text


def guard_sealed(view: dict[str, Any]) -> None:
    if (view.get("state") or {}).get("seal_opened"):
        raise DriverError("after_seal", AFTER_SEAL)


def guard_kind(decision: dict[str, Any], allowed: tuple[str, ...] = ()) -> None:
    kind = str(decision.get("kind"))
    if kind in NOT_THIS_WAY and kind not in allowed:
        raise DriverError("no_reseal" if kind == "reseal" else "not_this_way", NOT_THIS_WAY[kind])


# ── the analyst's restriction (DRIVER_PLAN D2) ───────────────────────────────


def hook_wired(files: tuple[Path, ...] | None = None) -> str | None:
    """The settings file whose PreToolUse hooks run the analyst's hook for every tool, or None."""
    for path in files if files is not None else SETTINGS_FILES:
        try:
            settings = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        groups = ((settings.get("hooks") or {}).get("PreToolUse") or []) \
            if isinstance(settings, dict) and isinstance(settings.get("hooks"), dict) else []
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict) or group.get("matcher") not in EVERY_TOOL:
                continue
            for hook in group.get("hooks") or []:
                if isinstance(hook, dict) and hook.get("type", "command") == "command" \
                        and HOOK_SCRIPT in str(hook.get("command") or ""):
                    return str(path)
    return None


def agent_restricted(path: Path | None = None) -> bool:
    """Whether the analyst agent type is defined with Bash as its only tool."""
    try:
        text = Path(path or AGENT_FILE).read_text(encoding="utf-8")
    except OSError:
        return False
    head = text.split("---", 2)[1] if text.startswith("---") and text.count("---") >= 2 else ""
    fields = dict(line.split(":", 1) for line in head.splitlines() if ":" in line)
    return fields.get("name", "").strip() == "analyst" and fields.get("tools", "").strip() == "Bash"


def guarded(scripted: bool) -> dict[str, Any]:
    """What holds the run's analyst to the driver, or a refusal where nothing does."""
    if scripted:
        return {"analyst": "scripted"}
    wired = hook_wired()
    if not agent_restricted() or wired is None:
        missing = [w for w, ok in (("the analyst agent type with Bash as its only tool "
                                    f"({AGENT_FILE.relative_to(REPO)})", agent_restricted()),
                                   (f"the analyst's hook ({HOOK_SCRIPT}) wired for every tool in "
                                    "a settings file", wired is not None)) if not ok]
        raise DriverError("unguarded", "The run does not start: an analyst could not be held to the "
                                       "driver's commands without " + " and ".join(missing) + ".")
    return {"analyst": "agent", "hook": wired}


# ── the commands ─────────────────────────────────────────────────────────────


def cmd_start(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    guard = guarded(bool(args.scripted))
    data = Path(args.data).expanduser().absolute()
    if not data.is_file():
        raise DriverError("no_data", f"There is no file at {data}.")
    if args.budget_fits < 1 or args.budget_seconds <= 0:
        raise DriverError("no_budget", "A run's fit budget is at least one fit and some seconds.")
    provenance: dict[str, str] = {}
    if args.provenance:
        parts = args.provenance.rsplit(",", 2)
        if len(parts) != 3:
            raise DriverError("bad_provenance", "--provenance takes url,date,license.")
        provenance = dict(zip(("url", "release_date", "license"), (p.strip() for p in parts)))
    run_id = args.run or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    if not RUN_ID.match(run_id):
        raise DriverError("bad_run_id", f"{run_id!r} is not a run id (letters, digits, . _ -).")
    folder = runs_root() / run_id
    if folder.exists():
        raise DriverError("run_exists", f"Run {run_id} exists already.")
    (folder / "home").mkdir(parents=True)
    args.run = run_id  # a start that fails is logged in its own run
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    port = free_port()
    run = Run(folder, {
        "run": run_id, "pid": None, "port": port, "project": None,
        "budget": {"fits": int(args.budget_fits), "seconds": float(args.budget_seconds)},
        "started": now(), "ended": None, "workers": WORKERS, **guard,
        "data": {"path": str(data), "name": data.name, "bytes": data.stat().st_size, "sha256": digest},
        "provenance": provenance, "home": str(folder / "home"), "python": sys.executable,
        "commit": _git_head()})
    proc = start_server(folder, port)
    run.meta["pid"] = proc.pid
    run.save()
    client = Client(port)
    end = time.monotonic() + HEALTH_SECONDS
    healthy = False
    while time.monotonic() < end and proc.poll() is None:
        try:
            healthy = client.get("/health").status == 200
        except DriverError:
            healthy = False
        if healthy:
            break
        time.sleep(0.3)

    def fail(code: str, message: str, **extra: Any) -> None:
        stopped = stop_pid(proc.pid, descendants(proc.pid)) if alive(proc.pid) else {}
        run.meta["ended"] = now()
        run.save()
        raise DriverError(code, message, stopped=stopped, **extra)

    if not healthy:
        tail = (folder / "server.log").read_text("utf-8", "replace")[-1500:]
        fail("server_did_not_start", "The server did not answer.", log=tail)
    r = client.upload("/projects/upload", data, **provenance)
    if r.status != 200:
        fail("upload_refused", "The server refused the file.", server=r.error())
    run.meta["project"] = r.json()["id"]
    run.save()
    if client.get(f"/projects/{run.project}/analyst-log").status != 200:
        fail("not_analyst_mode", "The server is not in analyst mode, so the run cannot start.")
    end = time.monotonic() + INGEST_SECONDS
    status: dict[str, Any] = {}
    while time.monotonic() < end:
        status = (view_of(run, client).get("stages") or {}).get("ingest") or {}
        if status.get("status") in ("fresh", "error", "blocked"):
            break
        time.sleep(0.5)
    if status.get("status") != "fresh":
        fail("ingest_failed", "The file was not read.", ingest=status)
    return 0, {"started": run.meta}, run


def cmd_stop(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    pid = int(run.meta.get("pid") or 0)
    if run.meta.get("ended") and not ours(run):
        return 0, {"stopped": {"run": run.id, "already": run.meta["ended"]}}, run
    if not ours(run):
        run.meta["ended"] = now()
        run.save()
        return 0, {"stopped": {"run": run.id, "pid": pid, "running": False}}, run
    result = stop_pid(pid, descendants(pid))
    run.meta["ended"] = now()
    run.meta["stopped"] = result
    run.save()
    return (0 if not result["running"] else 1), {"stopped": {"run": run.id, **result}}, run


def cmd_ask(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    q = the_question(run, run.client(), args.wait)
    return 0, private_free(q), run


def chosen(run: Run, client: Client, ids: list[str], sets: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    # What is offered is read once the question is open: a reading just confirmed rebuilds the
    # working table, and the question waits on it for a moment (journeys' ``not_yet`` wait).
    q = the_question(run, client, NOT_YET_SECONDS)
    guard_sealed(q["_view"])
    try:
        decision = opts.resolve(q["_composed"], ids, sets, q["_ways"])
    except opts.Unanswerable as exc:
        raise DriverError("not_offered", str(exc)) from None
    return decision, q


def cmd_preview(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    decision, _ = chosen(run, client, [args.option], args.set or [])
    guard_kind(decision, allowed=("open_seal",))  # what opening would show, without opening
    r = client.post(f"/projects/{run.project}/preview", decision)
    if r.status != 200:
        return 1, {"previewed": decision} | refused(r, run, "preview"), run
    return 0, {"previewed": decision, "preview": trim(r.json())}, run


def cmd_answer(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    note = reason_of(args.reason)
    decision, q = chosen(run, client, list(args.option), args.set or [])
    guard_kind(decision)
    guard_budget(run, client, q["_view"], decision)
    r = post_decision(run, client, decision, note)
    if r.status != 200:
        return 1, {"posted": decision} | refused(r, run, "answer"), run
    run.keep_exits([], None, "answer")
    return 0, recorded(r.json(), str(decision["kind"])), run


def cmd_revisit(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    note = reason_of(args.reason)
    view = view_of(run, client)
    guard_sealed(view)
    ids = {str(r.get("id")) for r in view.get("decisions") or []}
    if args.decision not in ids:
        raise DriverError("no_such_decision", f"No decision {args.decision} is in the log; ask "
                                              f"lists each decision's id.")
    revert = {"kind": "revert", "decision_id": args.decision}
    guard_budget(run, client, view, revert)
    r = post_decision(run, client, revert, note)
    if r.status != 200:
        return 1, refused(r, run, "revisit"), run
    run.keep_exits([], None, "revisit")
    return 0, recorded(r.json(), "revert"), run


# ── the fit budget ───────────────────────────────────────────────────────────
#
# The budget bounds the fits made for the analyst, not only the presses of Fit: under Predict,
# once Fit is pressed for the outcome, every fit the scheduler makes is served without another
# press (``fit_press.serving_gate``), and a fit expected under about two minutes runs without one
# (``fit_press.holds``); under Estimate a change after the lock refits too. So once the fits made
# meet the budget, an ``answer`` or ``revisit`` is recorded only when the driver can tell it makes
# no new fit: each fit it may leave is none, one already made, or the one under way. Which fits
# it may leave is told by the engine's own key rule over the answer as the server would record it
# (:func:`fits_after`), trusted only where it gives the server's keys; where the driver cannot
# tell, it says so, and nothing is recorded.

FIT_UNDER_WAY_NOT = ("error", "blocked")
RECORD_WAIT = 10.0  # a fit just made is recorded by the server's recorder a moment later


def spent_of(records: list[dict[str, Any]], fit: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """What the run has spent: ``GET /fits``' records, their seconds, and the fit under way, the
    current fit (``fit``, its stage status) when no record holds its key and it is neither held for
    Fit, failed, blocked nor cancelled: the scheduler is making it."""
    keys = [x.get("fit_key") for x in records]
    key = fit.get("key")
    running = key if (key and key not in keys and fit.get("held") != "fit" and not fit.get("cancelled")
                      and fit.get("status") not in FIT_UNDER_WAY_NOT) else None
    return {"fits": len(records), "running": int(running is not None),
            "seconds": round(sum(float(x.get("seconds") or 0.0) for x in records), 3),
            "budget_fits": budget["fits"], "budget_seconds": budget["seconds"], "keys": keys,
            "running_key": running}


def budget_now(run: Run, client: Client, view: dict[str, Any] | None = None) -> dict[str, Any]:
    """:func:`spent_of` as the server stands now. A fit just turned fresh is waited for, up to
    :data:`RECORD_WAIT` seconds, until its record is written."""
    view = view if view is not None else view_of(run, client)
    end = time.monotonic() + RECORD_WAIT
    while True:
        r = client.get(f"/projects/{run.project}/fits")
        if r.status != 200:
            raise DriverError("server_unreadable", f"The fit records could not be read ({r.status}).")
        fit = (view.get("stages") or {}).get("fit") or {}
        spent = spent_of((r.json() or {}).get("records") or [], fit, run.meta["budget"])
        if not (spent["running"] and fit.get("status") == "fresh") or time.monotonic() >= end:
            return spent
        time.sleep(0.5)
        view = view_of(run, client)


def budget_met(spent: dict[str, Any]) -> bool:
    """Whether the fits made, the one under way among them, meet the budget's count or seconds."""
    return (spent["fits"] + spent["running"] >= int(spent["budget_fits"])
            or spent["seconds"] >= float(spent["budget_seconds"]))


def budget_words(spent: dict[str, Any]) -> str:
    """The budget and what was spent of it, from ``GET /fits``, as a refusal says them."""
    budget, made = int(spent["budget_fits"]), int(spent["fits"])
    if made + spent["running"] >= budget:
        done = (f"{made} {'was' if made == 1 else 'were'} made and 1 is under way" if spent["running"]
                else f"{made} {'was' if made == 1 else 'were'} made already")
        return f"the run's fit budget is {budget} {'fit' if budget == 1 else 'fits'}, and {done}"
    return (f"the run's fit budget is {float(spent['budget_seconds']):g} seconds of fitting, and "
            f"{spent['seconds']:g} were spent already")


def over_budget(spent: dict[str, Any], fit_key: str | None) -> str | None:
    """Why pressing Fit now would pass the run's fit budget, or None. A fit already recorded for
    the answers now (one the scheduler ran before the press) is within the count it was made in."""
    counted = spent["fits"] if fit_key in spent["keys"] else spent["fits"] + 1
    if counted > int(spent["budget_fits"]) or spent["seconds"] >= float(spent["budget_seconds"]):
        return f"Fit is not pressed: {budget_words(spent)}."
    return None


AFTER_BUDGET = ("An answer the driver can tell makes no new fit is still recorded, and results, "
                "seal and export still run.")


def attempt_over_budget(spent: dict[str, Any], after: str | None) -> str | None:
    """Why recording an answer that leaves the fit keyed ``after`` would pass the budget, or None:
    within it every answer passes; once it is met, only one that leaves no fit (``None``), a fit
    already made, or the one under way."""
    if not budget_met(spent) or after is None or after in spent["keys"] or after == spent["running_key"]:
        return None
    return (f"Not recorded: this would make a new fit if recorded, and {budget_words(spent)}. "
            f"{AFTER_BUDGET}")


class CannotTell(Exception):
    """The fit an answer would leave cannot be told here, with why."""


# What the server records is not the answer as posted. ``decisions.validate`` runs the kind's
# completions under the server's decision context (``service.DecisionContext``): a roles answer
# records which proposals below high rode along unconfirmed, a fill the not-asked level, a triage
# its lines. And ``ProjectService.decide`` records with it what TurboTab then records itself
# (``ProjectService._complete``, P0.6): the roles once every one was confirmed on its own, and the
# split under Estimate. The driver runs the completions under :func:`served_context`, that context
# as the server serves it; a part of it the driver cannot read as the server holds it, once a
# completion reads it, means the driver cannot tell.

# The stages whose artifact ``GET …/stages/{stage}`` serves as the server's context reads it: the
# public artifact (``graph.read_artifact``), and the roles' with the predictor groups the route
# adds (``ProjectService._served_roles``) taken off again.
AS_KEPT = ("roles", "proposals", "target_info")
# The parts of the server's context the driver cannot read as the server holds it, as a refusal
# names them. The quest log, the triage and the stages' statuses are read only for the log as it
# stands.
UNSERVED = {
    "store": "the table's values", "bundle": "a stage's files", "sealed": "the held-out rows",
    "analyzed": "the rows analyzed", "interview": "the Router's steps",
    "settled": "a stage as it settles", "fit_gate": "the gate on the estimates",
    "sealed_scores": "the held-out scores", "project_dir": "the project's folder",
    "columns": "the table's columns", "column_info": "the table's columns",
    "n_rows": "the table's rows", "ingest_status": "the table's reading",
    "ingest_error": "the table's reading", "quest": "the quest log", "triage": "the triage",
    "stage": "a stage's status",
}
# What TurboTab records itself under Estimate when the Router reaches the split with none recorded
# (``ProjectService._completion_due``): no rows held out.
INFERENCE_SPLIT = {"kind": "set_split", "holdout": 0.0, "seed": 0}


def reader(run: Run, client: Client) -> Callable[[str], Any]:
    """``read(path)``: the JSON ``GET /projects/{pid}/{path}`` serves, or None when it refuses."""
    def read(path: str) -> Any:
        r = client.get(f"/projects/{run.project}/{path}")
        return r.json() if r.status == 200 else None

    return read


def served_context(view: dict[str, Any], records: list[Any], state: Any, keys: dict[str, Any],
                   read: Callable[[str], Any] | None, *, now: bool) -> tuple[Any, list[str]]:
    """The server's decision context for the log ``records`` (``ProjectService.decision_context``)
    as the driver reads it from the server, and the list in which each read of a part it cannot
    read as the server holds it is named (:data:`UNSERVED`; a stage's artifact outside
    :data:`AS_KEPT`). ``keys``: the stage keys for ``records``; a stage's artifact is fresh only at
    its key. ``now``: ``records`` is the log as the server holds it (``view``'s), so the stages'
    statuses, the quest log and the triage are read; for a log with an answer folded in, they are
    not the server's yet."""
    from turbotab.core.graph import StageStatus
    from turbotab.core.quest import QuestLog
    from turbotab.core.sweep import Triage
    from turbotab.server.service import DecisionContext

    unread: list[str] = []
    got: dict[str, Any] = {}
    stages = view.get("stages") or {}

    def result(path: str) -> Any:
        if path not in got:
            got[path] = read(path) if read is not None else None
        return got[path]

    def fresh(stage: str) -> dict[str, Any] | None:
        """``GET …/stages/{stage}`` when it serves the stage fresh at its key here, else None."""
        served = result(f"stages/{stage}") or {}
        ok = served.get("fresh") and served.get("key") and served.get("key") == keys.get(stage)
        return served if ok else None

    def kept(stage: str, only_fresh: bool) -> Any:
        if stage not in AS_KEPT or read is None:
            unread.append(f"the {stage} stage's artifact")
            return None
        served = fresh(stage) if only_fresh else result(f"stages/{stage}")
        artifact = (served or {}).get("artifact")
        if stage == "roles" and isinstance(artifact, dict):
            artifact = {k: v for k, v in artifact.items() if k != "predictor_groups"}
        return artifact

    def settled(stage: str) -> Any:
        """The stage's artifact once it settles (``ProjectService._settled``): fresh at its key, it
        is that artifact; blocked, none. One still computing the server waits for, unread here."""
        if stage in AS_KEPT and read is not None and fresh(stage) is not None:
            return kept(stage, True)
        if now and (stages.get(stage) or {}).get("status") == "blocked":
            return None
        unread.append(UNSERVED["settled"])
        return None

    def facts() -> tuple[Any, Any, Any, Any, Any] | None:
        """``(ingest status, its error, columns, column info, rows)`` as the server reads them
        (``ProjectService._table_facts``): none before the table is read; the working table's once
        it is fresh. Otherwise the server's table is one the driver does not read."""
        if "facts" not in got:
            ingest = stages.get("ingest") or {}
            same = ingest.get("key") == keys.get("ingest")
            if read is None or not same or (not now and ingest.get("status") != "fresh"):
                got["facts"] = None
            elif ingest.get("status") != "fresh":
                got["facts"] = (ingest.get("status"), ingest.get("error"), None, None, None)
            else:
                table = (fresh("working") or {}).get("artifact")
                cols = table.get("columns") if isinstance(table, dict) else None
                got["facts"] = None if not isinstance(cols, list) else (
                    "fresh", None, [str(c["name"]) for c in cols],
                    {str(c["name"]): {"dtype": c["dtype"], "n_unique": int(c["n_unique"]),
                                      "n_missing": int(c["n_missing"])} for c in cols},
                    int(table["n_rows"]))
        return got["facts"]

    def task() -> Any:
        if state.task is not None:
            return state.task
        info = kept("target_info", True)
        return info.get("task") if isinstance(info, dict) and info.get("column") == state.target \
            else None

    def fact(i: int, name: str) -> Callable[[], Any]:
        def value() -> Any:
            found = facts()
            if found is None:
                unread.append(UNSERVED[name])
                return None
            return found[i]

        return value

    lazy: dict[str, Callable[[], Any]] = {
        "ingest_status": fact(0, "ingest_status"), "ingest_error": fact(1, "ingest_error"),
        "columns": fact(2, "columns"), "column_info": fact(3, "column_info"),
        "n_rows": fact(4, "n_rows"), "task": task}
    unserved = {name for name in UNSERVED if name not in lazy and name != "settled"}
    if now:
        unserved -= {"stage", "quest", "triage"}

    class Served(DecisionContext):
        def __getattribute__(self, name: str) -> Any:
            if name in lazy:
                return lazy[name]()
            if name in unserved:
                unread.append(UNSERVED[name])
            return super().__getattribute__(name)

    def stage(name: str) -> Any:
        found = stages.get(name)
        return StageStatus.model_validate(found) if isinstance(found, dict) else None

    def model(path: str, cls: Any) -> Callable[[], Any]:
        def value() -> Any:
            found = result(path)
            return cls.model_validate(found) if found is not None else None

        return value

    ctx = Served(columns=None, ingest_status="", target=state.target, state=state,
                 records=lambda: list(records), artifact=lambda s: kept(s, True),
                 shown=lambda s: kept(s, False), settled=settled, analyst=True, early=False,
                 stage=stage if now else None, quest=model("quest", QuestLog) if now else None,
                 triage=model("triage", Triage) if now else None)
    return ctx, unread


def completed(decision: Any, ctx: Any, unread: list[str]) -> Any | None:
    """``decision`` as the server records it: its kind's completions run under ``ctx``
    (``decisions.validate`` runs them after the validators, which only refuse). None when a
    completion refuses it, as the server then does: nothing is recorded."""
    from turbotab.core import decisions

    try:
        for fn in decisions._COMPLETIONS.get(decision.kind, ()):
            decision = fn(decision, ctx)
    except decisions.Refusal:
        decision = None
    except Exception as exc:  # noqa: BLE001 - a completion that fails here keys nothing
        raise CannotTell(f"what TurboTab records for this answer does not compute here "
                         f"({type(exc).__name__})") from None
    if unread:
        raise CannotTell("what TurboTab records for this answer reads "
                         + ", ".join(dict.fromkeys(unread)) + ", which the driver cannot read as "
                         "the server holds it")
    return decision


def recorded_itself(state: Any, ctx: Any, keys: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """What TurboTab may record itself with an answer that leaves ``state``
    (``ProjectService._complete``, P0.6), in the order it would, each with its name: nothing once
    the roles are answered, unless under Estimate no split is; the roles once every predictor's
    was confirmed on its own (``readings.roles_completion`` over the roles' proposals fresh at
    ``keys``), and then the split under Estimate. Whether the Router has reached them is not read
    here, so each may or may not be recorded."""
    from turbotab.core.readings import roles_completion

    split = state.purpose == "inference" and state.split is None
    if state.roles and not split:
        return []
    if not state.roles:
        roles = ctx.artifact("roles")
        if roles is None and keys.get("roles") is not None:
            raise CannotTell("with no roles answered, TurboTab records the roles itself where each "
                             "was confirmed on its own, read from the roles' proposals as they are "
                             "computed again after this answer, which the driver cannot read yet")
        body = roles_completion(state, roles) if roles is not None else None
        if body is None:
            return []  # the Router holds at the roles
        return [("the roles", body)] + ([("the split", dict(INFERENCE_SPLIT))] if split else [])
    return [("the split", dict(INFERENCE_SPLIT))]


def fits_after(run: Run, view: dict[str, Any], decision: dict[str, Any],
               read: Callable[[str], Any] | None = None) -> list[tuple[str | None, str | None]]:
    """Each fit key the server may leave once ``decision`` is recorded (None: the fit would not
    compute then), with what TurboTab records itself before it leaves that key (None: nothing);
    an empty list when the answer is refused, as the server refuses it (:func:`keys_after`)."""
    return [(keys.get("fit"), by) for keys, by in keys_after(run, view, decision, read)]


def keys_after(run: Run, view: dict[str, Any], decision: dict[str, Any],
               read: Callable[[str], Any] | None = None) -> list[tuple[dict[str, Any], str | None]]:
    """Each set of stage keys the server may leave once ``decision`` is recorded, with what
    TurboTab records itself before it leaves them (None: nothing); an empty list when the answer
    is refused, as the server refuses it.

    Told by the engine's own rule, as the server keys every stage (``graph.stage_keys`` over the
    stage graph the server runs, ``workspace.stage_graph``): the decision log as ``GET
    /projects/{pid}`` serves it, folded with ``decision`` as the server records it
    (:func:`completed`) and with what TurboTab records itself then (:func:`recorded_itself`), on
    the data file's fingerprint (``datastore.fingerprint_file``). ``read`` reads the server
    (:func:`reader`). The rule is trusted only where the keys it gives the log as it stands are
    every key the server serves for it now; otherwise, or when ``decision`` does not read or fold,
    or what the server records for it is not read here, :class:`CannotTell` says why. A plan's
    lock no estimate was served under, which a change to the plan withdraws first
    (``ProjectService._append``), is not folded: of the stages only ``mattered`` reads the lock,
    and the fit does not."""
    from turbotab.server import app  # noqa: F401 - what the server imports registers every completion
    from turbotab.core import decisions
    from turbotab.core.datastore import fingerprint_file
    from turbotab.core.graph import stage_keys
    from turbotab.core.workspace import stage_graph

    fingerprint = run.meta.get("fingerprint")
    if not fingerprint:
        try:
            fingerprint = fingerprint_file(Path(run.meta["data"]["path"]))
        except (OSError, KeyError, TypeError):
            raise CannotTell("the run's data file cannot be read") from None
        run.meta["fingerprint"] = fingerprint
        if (run.folder / "driver.json").exists():
            run.save()
    order = stage_graph().order()

    def keyed(log: list[Any]) -> tuple[Any, dict[str, Any]]:
        state = decisions.fold(log)
        return state, stage_keys(order, state.model_dump(mode="json"), fingerprint)

    try:
        records = [decisions.DecisionRecord.model_validate(r) for r in view.get("decisions") or []]
        state, now_keys = keyed(records)
    except Exception as exc:  # noqa: BLE001 - a log this cannot read is a log it cannot key
        raise CannotTell(f"the decision log does not read here ({type(exc).__name__})") from None
    served = {name: (s or {}).get("key") for name, s in (view.get("stages") or {}).items()}
    differ = sorted(name for name, key in served.items() if now_keys.get(name) != key)
    if differ or not served:
        raise CannotTell("the keys it computes are not the server's, for "
                         + (", ".join(differ[:4]) if differ else "any stage"))
    try:
        parsed = decisions.parse_decision(decision)
    except Exception as exc:  # noqa: BLE001 - not a decision
        raise CannotTell(f"the answer does not read as a decision ({type(exc).__name__})") from None
    ctx, unread = served_context(view, records, state, now_keys, read, now=True)
    answer = completed(parsed, ctx, unread)
    if answer is None:
        return []
    seq = max((r.seq for r in records), default=0)
    at = datetime.now(timezone.utc)

    def probe(d: Any, n: int) -> Any:
        return decisions.DecisionRecord(id=f"__probe{n}__", seq=seq + n + 1, at=at, decision=d)

    try:
        log = [*records, probe(answer, 0)]
        state, keys = keyed(log)
    except Exception as exc:  # noqa: BLE001 - one the log refuses
        raise CannotTell(f"the answer does not fold into the log ({type(exc).__name__})") from None
    out: list[tuple[dict[str, Any], str | None]] = [(keys, None)]
    ctx, unread = served_context(view, log, state, keys, read, now=False)
    names: list[str] = []
    for n, (name, body) in enumerate(recorded_itself(state, ctx, keys), start=1):
        own = completed(decisions.parse_decision(body), ctx, unread)
        if own is None:
            break
        log = [*log, probe(own, n)]
        names.append(name)
        out.append((keyed(log)[1], " and ".join(names)))
    return out


def guard_budget(run: Run, client: Client, view: dict[str, Any], decision: dict[str, Any]) -> None:
    """Refuse ``decision`` when the budget is met and it would make a new fit, or the driver cannot
    tell (the section's rule); a decision is told only once the budget is met."""
    spent = budget_now(run, client, view)
    if not budget_met(spent):
        return
    try:
        afters = fits_after(run, view, decision, reader(run, client))
    except CannotTell as exc:
        raise DriverError("fit_budget_unknown", f"Not recorded: {budget_words(spent)}, and the driver "
                                                f"cannot tell whether this would make a new fit: "
                                                f"{exc}.", spent=public_budget(spent)) from None
    new = [key for key, _ in afters if attempt_over_budget(spent, key)]
    if not new:
        return
    if len(new) < len(afters):  # some of the ways the server may record it make a new fit
        raise DriverError("fit_budget_unknown", f"Not recorded: {budget_words(spent)}, and the driver "
                                                f"cannot tell whether this would make a new fit: "
                                                f"that depends on whether TurboTab records "
                                                f"{afters[-1][1]} itself with it.",
                          spent=public_budget(spent))
    raise DriverError("fit_budget", attempt_over_budget(spent, new[0]), spent=public_budget(spent))


def wait_fit(run: Run, client: Client, seconds: float) -> dict[str, Any]:
    end = time.monotonic() + seconds
    while True:
        status = (view_of(run, client).get("stages") or {}).get("fit") or {}
        if status.get("status") in ("fresh", "error", "blocked") or status.get("held") \
                or time.monotonic() >= end:
            return {k: status.get(k) for k in ("status", "key", "error", "held", "progress")
                    if status.get(k) is not None}
        time.sleep(0.5)


def cmd_fit(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    view = view_of(run, client)
    guard_sealed(view)
    spent = budget_now(run, client, view)
    why = over_budget(spent, ((view.get("stages") or {}).get("fit") or {}).get("key"))
    if why:
        raise DriverError("fit_budget", why, spent=public_budget(spent))
    r = client.post(f"/projects/{run.project}/fit")
    if r.status != 200:
        return 1, refused(r, run, "fit") | {"spent": public_budget(spent)}, run
    run.keep_exits([], None, "fit")
    lock = r.json() or {}
    status = wait_fit(run, client, args.wait)
    return 0, {"pressed": lock, "fit": status, "spent": public_budget(budget_now(run, client))}, run


def public_budget(spent: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spent.items() if k not in ("keys", "running_key")}


def cmd_results(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    end = time.monotonic() + args.wait
    while True:
        view = view_of(run, client)
        stages = view.get("stages") or {}
        busy = [s for s in ("fit", "effects", "mattered")
                if (stages.get(s) or {}).get("status") in ("queued", "running")]
        if not busy or time.monotonic() >= end:
            break
        time.sleep(0.5)
    r = client.get(f"/projects/{run.project}/results")
    if r.status != 200:
        return 1, refused(r, run, "results"), run
    out: dict[str, Any] = {"results": r.json(),
                           "stages": {s: (stages.get(s) or {}).get("status")
                                      for s in ("fit", "effects", "mattered") if s in stages}}
    state = view.get("state") or {}
    if state.get("purpose") == "inference" and state.get("plan_locked"):
        m = client.get(f"/projects/{run.project}/mattered")
        out["mattered"] = m.json() if m.status == 200 else {"refused": m.error()}
    return 0, out, run


def cmd_seal(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    note = reason_of(args.reason)
    view = view_of(run, client)
    families = list((view.get("state") or {}).get("models") or [])
    if args.family not in families:
        raise DriverError("not_offered", f"{args.family} is not a fitted family; the fitted families "
                                         f"are {', '.join(families) or 'none yet'}.")
    r = post_decision(run, client, {"kind": "open_seal", "family": args.family}, note)
    if r.status != 200:
        return 1, refused(r, run, "seal"), run
    run.keep_exits([], None, "seal")
    return 0, recorded(r.json(), "open_seal"), run


def cmd_export(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    import io
    import zipfile

    run = Run.load(args.run)
    client = run.client()
    end = time.monotonic() + args.wait
    while True:
        r = client.get(f"/projects/{run.project}/export")
        if r.status == 200 or r.error().get("code") != "result_not_ready" or time.monotonic() >= end:
            break
        time.sleep(1.0)
    if r.status != 200:
        return 1, refused(r, run, "export"), run
    path = run.folder / "export.zip"
    path.write_bytes(r.body)
    run.keep_exits([], None, "export")
    with zipfile.ZipFile(io.BytesIO(r.body)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        methods = next((z.read(n).decode("utf-8") for n in names if n.endswith("/methods.md")), None)
    return 0, {"exported": {"path": str(path), "bytes": len(r.body),
                            "sha256": hashlib.sha256(r.body).hexdigest(),
                            "files": [n.split("/", 1)[-1] for n in names], "methods": methods}}, run


# ── output ───────────────────────────────────────────────────────────────────

TRIM = 40


def trim(obj: Any, limit: int = TRIM) -> Any:
    """``obj`` with each list longer than ``limit`` cut to its first ``limit`` items and a last
    item saying how many more were served (a preview's points)."""
    if isinstance(obj, dict):
        return {k: trim(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        head = [trim(v, limit) for v in obj[:limit]]
        return head + ([{"more": len(obj) - limit}] if len(obj) > limit else [])
    return obj


def summary(name: str, code: int, payload: dict[str, Any]) -> dict[str, Any]:
    """What ``commands.jsonl`` keeps of a response: enough to follow the run, not the whole."""
    out: dict[str, Any] = {"exit": code}
    if "refused" in payload:
        r = payload["refused"]
        out["refused"] = {k: r.get(k) for k in ("by", "status", "code", "message")}
        out["refused"]["exits"] = [e.get("id") for e in r.get("exits") or []]
    if name == "ask":
        q = payload.get("question") or {}
        out.update({"question": q.get("key"), "status": q.get("status"),
                    "options": [o.get("id") for o in payload.get("options") or []],
                    "fit_reached": payload.get("fit_reached")})
    for key in ("posted", "previewed"):
        if key in payload:
            out[key] = payload[key]
    if "recorded" in payload:
        rec = payload["recorded"]
        out["recorded"] = {k: rec.get(k) for k in ("id", "seq", "note", "post_seal")} | {
            "decision": rec.get("decision")}
        out["next"] = payload.get("next")
    if "preview" in payload:
        p = payload["preview"] or {}
        out["preview"] = {"kind": p.get("kind"), "views": len(p.get("views") or []),
                          "basis": p.get("basis"), "note": p.get("note")}
    if "pressed" in payload:
        out.update({"pressed": True, "fit": payload.get("fit"), "spent": payload.get("spent")})
    if "spent" in payload and "pressed" not in payload:
        out["spent"] = payload["spent"]
    if "results" in payload:
        res = payload["results"] or {}
        out["results"] = {"purpose": res.get("purpose"), "withheld": res.get("withheld"),
                          "holdout_sealed": res.get("holdout_sealed"),
                          "families": [f.get("family") for f in res.get("families") or []],
                          "stages": payload.get("stages")}
        if "mattered" in payload:
            out["mattered"] = bool(payload["mattered"]) and "refused" not in payload["mattered"]
    if "exported" in payload:
        out["exported"] = {k: payload["exported"].get(k) for k in ("path", "bytes", "sha256")}
    if "stopped" in payload:
        out["stopped"] = payload["stopped"]
    if "started" in payload:
        out["started"] = {k: payload["started"].get(k) for k in ("pid", "port", "project")}
    return out


COMMANDS: dict[str, Callable[[argparse.Namespace], tuple[int, dict[str, Any], Run | None]]] = {
    "start": cmd_start, "stop": cmd_stop, "ask": cmd_ask, "preview": cmd_preview,
    "answer": cmd_answer, "revisit": cmd_revisit, "fit": cmd_fit, "results": cmd_results,
    "seal": cmd_seal, "export": cmd_export,
}
ANALYST_COMMANDS = ("ask", "preview", "answer", "revisit", "fit", "results", "seal", "export")


def _wait(text: str) -> float:
    value = float(text)
    if not 0 <= value <= WAIT_MAX:
        raise argparse.ArgumentTypeError(f"--wait is 0 to {WAIT_MAX:g} seconds")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.driver",
                                     description="The swarm's driver (DRIVER_PLAN D1).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="start a run's server and upload its data file")
    p.add_argument("--data", required=True)
    p.add_argument("--provenance", help="url,release date (YYYY-MM-DD),license")
    p.add_argument("--budget-fits", type=int, required=True)
    p.add_argument("--budget-seconds", type=float, required=True)
    p.add_argument("--run", help="the run's id (default: the time and a random suffix)")
    p.add_argument("--scripted", action="store_true",
                   help="the analyst is a script, not an agent (the dry run, the tests)")

    def with_run(name: str, help_: str, wait: bool = False) -> argparse.ArgumentParser:
        q = sub.add_parser(name, help=help_)
        q.add_argument("--run", required=True)
        if wait:
            q.add_argument("--wait", type=_wait, default=WAIT_DEFAULT,
                           help=f"seconds to wait for what is computing (default {WAIT_DEFAULT:g})")
        return q

    with_run("stop", "stop the run's server")
    with_run("ask", "the open question", wait=True)
    p = with_run("preview", "what recording an option would do; nothing is recorded")
    p.add_argument("option")
    p.add_argument("--set", action="append", metavar="NAME=VALUE")
    p = with_run("answer", "record an option, with why")
    p.add_argument("option", nargs="+")
    p.add_argument("--set", action="append", metavar="NAME=VALUE")
    p.add_argument("--reason", required=True)
    p = with_run("revisit", "change an earlier decision (the log keeps it), with why")
    p.add_argument("decision")
    p.add_argument("--reason", required=True)
    with_run("fit", "press Fit, within the run's fit budget", wait=True)
    with_run("results", "the results the app shows now", wait=True)
    p = with_run("seal", "open the held-out rows, once, declaring the final family, with why")
    p.add_argument("family")
    p.add_argument("--reason", required=True)
    with_run("export", "the methods, the checklist and the bundle", wait=True)
    return parser


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None, out: Callable[[str], None] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else 2
    write = out or (lambda text: print(text, flush=True))
    run: Run | None = None
    started = now()
    try:
        try:
            code, payload, run = COMMANDS[args.command](args)
        except DriverError:
            raise
        except Exception as exc:  # noqa: BLE001 - a driver defect still answers in JSON, logged
            raise DriverError("driver_failed", f"The driver failed: {type(exc).__name__}: "
                                               f"{str(exc)[:600]}") from exc
    except DriverError as exc:
        code = 1
        payload = {"refused": {"by": "driver", "code": exc.code, "message": exc.message,
                               **exc.extra}}
        if run is None and getattr(args, "run", None):
            try:
                run = Run.load(args.run)
            except DriverError:
                run = None
    payload = private_free(payload)
    write(json.dumps(payload, ensure_ascii=False, default=str))
    if run is not None:
        given = {k: v for k, v in vars(args).items() if k not in ("command",) and v is not None}
        run.log({"at": started, "command": args.command, "args": given,
                 "summary": summary(args.command, code, payload)})
    return code


if __name__ == "__main__":
    raise SystemExit(main())

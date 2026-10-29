"""The swarm's driver (DRIVER_PLAN D1; SWARM_PLAN §2): an analyst works TurboTab through the app's
own API, one small command at a time, each printing JSON.

    python -m research.swarm.driver start --data <file> [--provenance url,date,license]
                                         --budget-fits N --budget-seconds S [--scripted]
                                         [--model M] [--persona P] [--effort E] [--prompt-version V]
    python -m research.swarm.driver ask      --run <id> [--decisions | --line <line id>]
    python -m research.swarm.driver preview  --run <id> <option> [<option> ...] [--set ...] --reason <why>
    python -m research.swarm.driver answer   --run <id> <option> [<option> ...] [--set ...] --reason <why>
                                             --marker <marker> [--rejected <option>]
    python -m research.swarm.driver revisit  --run <id> <decision id> --reason <why>
    python -m research.swarm.driver fit      --run <id>
    python -m research.swarm.driver results  --run <id>
    python -m research.swarm.driver seal     --run <id> <family> --reason <why>
    python -m research.swarm.driver export   --run <id>
    python -m research.swarm.driver want     --run <id> <what> --kind <kind> --reason <why> --marker <marker>
    python -m research.swarm.driver account  --run <id> <the account>
    python -m research.swarm.driver stop     --run <id>
    python -m research.swarm.driver <analyst's command> --run <id> --part N

**One run** is one directory, ``research/runs/<run id>/`` (``$TURBOTAB_SWARM_RUNS`` in its place
for a test), named by the id ``start`` gives and prints (:func:`new_run_id`): a neutral label
``start`` makes itself, ``run_`` and one more than the runs in the runs' home (``run_01``,
``run_02``, …), never one taken from the data or chosen by the caller, since the analyst is shown
it and no label may name a control kind; and a secret part, so no analyst reaches another's run
by guessing its id. ``--run`` names only an id of that form whose folder is in the runs' home
itself (:meth:`Run.load`). ``home/`` is the server's ``TURBOTAB_HOME``; ``driver.json`` holds the
run's id, the PID and port of the server the driver started, the project's id, the fit budget, the
data file and where it came from, and when the run started and, once ``stop`` ran, ended;
``commands.jsonl`` holds every command with when it began (``at``; ``began`` to the microsecond)
and returned (``done``; ``returned``), its arguments and a summary of its response;
``markers.jsonl`` the live markers read at each command that reads the run's results (``fit``,
``results``, ``seal``, ``export``), by the current fit's key (``research/swarm/markers.py``);
``server.log`` is the server's output;
``exits.json`` the ways forward of the last refusal, so ``answer`` can take one; ``parts/`` the
last output of each command too long to print whole, as its parts; ``export.zip`` the bundle.
``start`` runs ``python -m turbotab.server --analyst`` on a free local port with two workers and
uploads the file; ``stop`` stops that PID and no other.

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
by :mod:`research.swarm.options`. Each open declaration line (a decision with no Router question:
the selection, the batch handling, the intended use, the scales, the levers, the shrinkage, the
validation scheme, the explanations, the model sequence, the measurement error, the usual intake,
the multiplicity and the sensitivity analyses; ``quest.DECLARATIONS``) is listed under
``also_open`` with its options and fields, composed by :mod:`research.swarm.declarations` from the
engine's own definitions, and ``preview`` and ``answer`` take them by id like a question's. A fit
the engine refused in words (its stage status's typed ``refusal``) is said with its code and its
ways forward, which ``answer`` takes by id; a fit that failed with an error the engine did not type
as a refusal is said as it failed. So is the effects stage once it stopped, in ``ask`` and
``results`` (``effects_failed``), since the results serve no words for a stage in error. A way
forward whose decision the driver records only through a command of its own (a revert, the
opening of the held-out rows), which ``answer`` refuses, is listed with no id and with the command
that records it (``options.exits``: ``revisit <decision id>``, ``seal <family>``).

**Several options in one answer.** Where the open question takes several (the lens, the models,
the exclusions' rules: ``options.resolve``), ``ask``'s ``multi`` says how in the form ``answer``
reads (:func:`several`): each id its own word, separated by spaces; and ``answer`` refuses ids
joined by commas in words that name that form (``options.resolve``), unless the joined word is
itself an id offered.

**What a command prints** (the driver-output package, 2026-10-28): Claude Code shows a Bash
call's output inline only up to about 30,000 characters, and past that an analyst sees a 2 KB
preview of it and nothing more, since it runs only the driver's commands. So every analyst's
command prints at most :data:`OUTPUT_CAP` (:func:`to_cap`), weighed in UTF-8 bytes, which no count
of its characters exceeds, and what it leaves out it names with the command that shows it. An
``ask`` that would pass the cap lists each decision without its note, then each other open line
by its id, key, source, name and status alone, then only the newest decisions there is room for,
and ``left_out`` names each command that shows the rest: ``ask --decisions``, the whole log, and
``ask --line <line id>``, one other open line whole. The question's own words, options, fields,
readings, ways forward and refusals are never left out. Where an output still passes the cap
(``ask``'s question alone, the log, the results, any other), it is printed in parts: the first by
the command itself, each with a ``part`` that says which of how many, where it goes on from the
part before (``continues``) and the command that prints the next, ``<command> --run <id> --part
N``, which runs nothing else and prints part N as kept in ``parts/<command>.json``; an output
that printed whole drops the kept parts of the same output, and an ``answer``, ``revisit``,
``fit`` or ``seal`` that may have changed the run (any but one the driver or the server refused)
marks every other command's no longer shown. A scripted analyst (``start --scripted``) reads the
JSON in its own process, past no harness, so its run's outputs print whole (:func:`held_to_cap`).

**The analyst's tags** (SWARM_PLAN §2, §4): ``preview`` and ``answer`` take a reason; ``answer``
takes a marker of the controlled vocabulary (``research/swarm/vocabulary.py``) and may name the
alternative it rejected, an option offered for the same question; ``want`` records an option, a
check, a reading or an eligibility rule the engine lacks, with its kind, its reason and its
marker; ``account`` records the analyst's end-of-run account, once, marked whether the results had
been read and whether the held-out rows were open. None of these is posted to the server: the
run's record reads them from ``commands.jsonl``, so their source is the driver.

**What the driver refuses**, whatever the analyst asks (SWARM_PLAN §2; DRIVER_PLAN D1):

* a ``reseal`` (the server refuses one in analyst mode too, and offers none), a ``lock_plan`` (the
  server's own), a ``revert`` other than by ``revisit``, and an ``open_seal`` other than by ``seal``;
* ``fit`` past the run's fit budget: once the fits made reach its count, or their seconds its
  seconds, read from ``GET /fits``'s records (SWARM_ENGINE_PLAN E1), a press that would make a
  fit is refused (the fit under way counts as made by it); a press for the answers as they stand
  whose fit was made already makes none, so it goes through;
* once those meet the budget, an ``answer`` or ``revisit`` that would make a new fit, since the
  server makes and serves it without another press, and one the driver cannot tell makes none
  (the fit budget's section below); one it can tell makes none (the triage, which the fit does
  not read; a revisit that leaves the models unanswered; going back to answers already fitted)
  is still recorded;
* once the held-out rows are opened, any change or attempt (``preview``, ``answer``, ``revisit``,
  ``fit``): the reported result is the opening's score, and nothing more is tried (SWARM_PLAN §3).
  ``seal`` again goes to the server, which refuses a second opening.

**What an analyst's command never reads** (:func:`analyst_sealed`): while ``ask``, ``preview``,
``answer``, ``revisit``, ``fit``, ``results``, ``seal``, ``export``, ``want`` or ``account``
runs (:data:`ANALYST_COMMANDS`), opening anything a run
is judged against (a dataset's declared answers and its manifest under ``research/datasets/``, the
corpus records and their materiality under ``research/corpus/``, the launches with their known
answers in the main checkout's, which every checkout shares (``launcher.LAUNCHES``), the fixtures'
declared truths, the decoy's generator, the glucose benchmark) or importing a module that holds or builds them
(:data:`UNREAD_MODULES`), by
an import statement or by ``importlib``, is refused, so nothing it prints can come from them. The
driver itself imports none of those modules, so none is loaded before a command runs.

Exit status: 0 when the command did what it says, 1 when the server, the engine (a fit it refused
in words) or the driver refused it (the JSON says why under ``refused``: a refusal by the server or
the engine lists its ways forward under ``exits``, one by the driver none), 2 when the command line
does not read.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research.swarm import declarations as decl
from research.swarm import launcher
from research.swarm import options as opts
from research.swarm import vocabulary as V

REPO = Path(__file__).resolve().parents[2]
RUNS_ENV = "TURBOTAB_SWARM_RUNS"
WORKERS = 2  # DRIVER_PLAN D1: two workers per run's server
# A run's id, as ``start`` gives it (:func:`new_run_id`): a label (a plain word) and a secret part,
# 32 hex digits from ``secrets``, so no analyst reaches another's run by guessing its id (three
# analysts run on one dataset at once, SWARM_PLAN §6). Each is matched whole (``fullmatch``: a
# ``$`` would match before a final newline).
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{0,39}")
RUN_ID = re.compile(rf"{LABEL.pattern}-[0-9a-f]{{32}}")
LABEL_STEM = "run_"  # the neutral label ``start`` makes: ``run_`` and the run's count
REQUEST_SECONDS = 120.0
HEALTH_SECONDS = 180.0
INGEST_SECONDS = 600.0
STOP_SECONDS = 30.0
WAIT_DEFAULT = 90.0  # under the analyst's shell's own two-minute limit
WAIT_MAX = 540.0
NOT_YET_SECONDS = 60.0
HOOK_SCRIPT = "research/swarm/analyst_hook.py"
MARKERS_FILE = "markers.jsonl"
PERSONAS = ("conventional", "skeptical", "explorer")  # SWARM_PLAN §6
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
# What an analyst's command prints at once (the module docstring, **What a command prints**):
# Claude Code shows a Bash call's output inline only up to about 30,000 characters, so this keeps
# a margin under it. Weighed in UTF-8 bytes (:func:`printed_size`).
OUTPUT_CAP = 24_000
INVOKE = "scripts/py -m research.swarm.driver"  # the form an analyst runs a command in
PARTS_DIR = "parts"  # a run's outputs too long to print whole, one file per command
# The commands that may change the run (a decision posted, Fit pressed): once one has, another
# command's kept parts no longer show the run as it stands.
RECORDING = ("answer", "revisit", "fit", "seal")


class DriverError(Exception):
    """A refusal of the driver's own: a code, a plain message, and what it says beside them."""

    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


# ── what an analyst's command never reads (the module docstring) ─────────────

# The modules that hold or build what a run is judged against: the fixtures' declared truths, the
# intake (each dataset's answers), the decoy's generator, the record with its materiality, its
# checks and its referee, and the glucose project's benchmark (Bang et al. 2009's reported AUC and
# the comparator's score on the held-out participants).
UNREAD_MODULES = ("turbotab.core.tests.truths", "research.swarm.intake", "research.swarm.decoys",
                  "research.swarm.record", "research.swarm.assemble", "research.swarm.checks",
                  "research.swarm.referee", "research.swarm.benchmark")
# The dry run's folder holds corpus-shaped records with known answers for three pilot datasets;
# the launches, each with its run's known answer, are the main checkout's, whichever checkout
# this is (``launcher.LAUNCHES``, None where git cannot say which that is).
UNREAD_FOLDERS = (REPO / "research" / "datasets", REPO / "research" / "corpus",
                  REPO / "research" / "swarm" / "dryrun") + (
    (launcher.LAUNCHES,) if launcher.LAUNCHES is not None else ())
UNREAD_FILES = (REPO / "turbotab" / "core" / "tests" / "truths.py",
                REPO / "research" / "swarm" / "decoys.py", REPO / "research" / "swarm" / "intake.py",
                REPO / "research" / "swarm" / "benchmark.py")
_SEALED = {"on": False, "in_hook": False}


def unread(path: Any) -> bool:
    """Whether ``path`` is one an analyst's command never opens (:data:`UNREAD_FOLDERS`, their
    contents, and :data:`UNREAD_FILES`)."""
    if isinstance(path, int):
        return False  # an open file descriptor: what it names was opened already
    try:
        p = Path(os.fsdecode(path)).resolve()
    except (OSError, TypeError, ValueError, RuntimeError):
        return False
    return p in UNREAD_FILES or any(p == f or f in p.parents for f in UNREAD_FOLDERS)


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _SEALED["on"] or _SEALED["in_hook"] or event not in ("open", "import"):
        return
    _SEALED["in_hook"] = True
    try:
        name = args[0] if args else None
        if event == "open" and unread(name):
            raise DriverError("not_for_the_analyst", "An analyst's command reads nothing a run is "
                                                     "judged against.")
        if event == "import" and isinstance(name, str) and name in UNREAD_MODULES:
            raise DriverError("not_for_the_analyst", "An analyst's command reads nothing a run is "
                                                     "judged against.")
    finally:
        _SEALED["in_hook"] = False


class _Unimportable:
    """A finder first on ``sys.meta_path`` that refuses, while sealed, to find a module of
    :data:`UNREAD_MODULES`: an import by ``importlib.import_module`` raises no ``import`` audit
    event, and its compiled file is read in place of the source the audit hook watches."""

    @staticmethod
    def find_spec(name: str, path: Any = None, target: Any = None) -> None:
        if _SEALED["on"] and not _SEALED["in_hook"] and name in UNREAD_MODULES:
            raise DriverError("not_for_the_analyst", "An analyst's command reads nothing a run is "
                                                     "judged against.")
        return None


class analyst_sealed:
    """While an analyst's command runs, refuse every read of what a run is judged against (the
    module docstring): an audit hook (``sys.addaudithook``, added once per process) on each file
    opened and each module imported by an import statement, and a finder first on
    ``sys.meta_path`` (:class:`_Unimportable`) for any other way of importing one, each refusing
    only while one of these is entered. A module already imported before the command is not
    imported again; the driver imports none of them (its module docstring)."""

    hooked = False

    def __enter__(self) -> "analyst_sealed":
        if not analyst_sealed.hooked:
            sys.addaudithook(_audit)
            analyst_sealed.hooked = True
        if not any(f is _Unimportable for f in sys.meta_path):
            sys.meta_path.insert(0, _Unimportable)
        self.before = _SEALED["on"]
        _SEALED["on"] = True
        return self

    def __exit__(self, *exc: Any) -> None:
        _SEALED["on"] = self.before


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_us() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_run_id(home: Path) -> str:
    """A new run's id: a neutral label made here, ``run_`` and one more than the runs in ``home``
    (the folders named by a run's id), two digits at least, never taken from the data or chosen
    by the caller (the module docstring); and 32 hex digits from ``secrets``, so an id cannot be
    guessed from another's. Two runs started at once can share a label; never an id."""
    try:
        made = sum(1 for p in home.iterdir() if RUN_ID.fullmatch(p.name))
    except OSError:
        made = 0  # no runs' home yet
    label = f"{LABEL_STEM}{made + 1:02d}"
    return f"{label}-{secrets.token_hex(16)}"


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
        """The run ``run_id`` names: an id of the form ``start`` gives (:data:`RUN_ID`), whose
        folder is in the runs' home itself, not a link to a folder elsewhere."""
        if not RUN_ID.fullmatch(run_id or ""):
            raise DriverError("no_such_run", f"{run_id!r} is not a run id.")
        home = runs_root()
        folder = home / run_id
        if folder.is_symlink() or folder.resolve().parent != home.resolve():
            raise DriverError("no_such_run", f"Run {run_id} is not in the runs' home.")
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

    def _parts_file(self, command: str) -> Path:
        return self.folder / PARTS_DIR / f"{command}.json"

    def _write_parts(self, command: str, kept: dict[str, Any]) -> None:
        path = self._parts_file(command)
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(kept, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    def keep_parts(self, command: str, of: str, made: str, texts: list[str]) -> None:
        """``command``'s last output, too long to print whole, as the parts it prints in."""
        self._write_parts(command, {"command": command, "of": of, "made": made, "stale": None,
                                    "parts": texts})

    def kept_parts(self, command: str) -> dict[str, Any] | None:
        try:
            return dict(json.loads(self._parts_file(command).read_text("utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def drop_parts(self, command: str, of: str) -> None:
        """``command``'s kept parts, dropped once its output printed whole, where they are of the
        same output (``of``: ``ask``, ``ask --decisions`` or ``ask --line <id>`` for ``ask``)."""
        kept = self.kept_parts(command)
        if kept is not None and kept.get("of") == of:
            self._parts_file(command).unlink()

    def stale_parts(self, by: str, at: str) -> None:
        """Every other command's kept parts, marked no longer the run as it stands: ``by`` ran at
        ``at`` and may have changed it (:data:`RECORDING`)."""
        try:
            files = [p for p in (self.folder / PARTS_DIR).iterdir() if p.suffix == ".json"]
        except FileNotFoundError:
            return
        for path in files:
            kept = self.kept_parts(path.stem)
            if path.stem == by or kept is None or kept.get("stale"):
                continue
            self._write_parts(path.stem, kept | {"stale": {"command": by, "at": at}})


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


def failures(view: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """Each stage of ``names`` in error, classed by its status alone: ``refused`` where it serves
    the engine's typed refusal (its code, words and ways forward, each way with its id), else
    ``failed`` (an error the engine did not type as a refusal)."""
    stages = view.get("stages") or {}
    out = []
    for name in dict.fromkeys(names):
        status = stages.get(name) or {}
        if status.get("status") != "error":
            continue
        refusal = status.get("refusal")
        out.append({"stage": name, "key": status.get("key"), "error": status.get("error"),
                    "kind": "refused" if refusal else "failed",
                    "refusal": None if not refusal else {
                        "stage": refusal.get("stage"), "code": refusal.get("code"),
                        "message": refusal.get("message"),
                        "exits": opts.exits(refusal.get("exits"))}})
    return out


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


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def several(composed: dict[str, Any], run_id: str) -> str:
    """How the open question takes several of its options in one answer, in the form ``answer``
    reads (``options.resolve``): each id its own word, each option adding its own items to the
    answer's list (``multi``: its lenses, models or rules) in the order named; a rule of one's own
    (an exclusions option with fields of its own) named more than once, a rule each time; a
    setting's place among the options named where more than one takes it; and an option whose
    answer lists none of them, answered alone. ``ask``'s ``multi``, in the driver's words."""
    multi = str(composed.get("multi"))
    offered = [o for o in composed.get("options") or [] if o.get("id") is not None]

    def items(option: dict[str, Any]) -> Any:
        decision = option.get("decision") or option.get("_base")
        return decision.get(multi) if isinstance(decision, dict) else None

    adds = [str(o["id"]) for o in offered if items(o)]
    again = [str(o["id"]) for o in offered if o.get("_range") or o.get("_levels")]
    alone = [str(o["id"]) for o in offered
             if isinstance(o.get("decision"), dict) and not o["decision"].get(multi)]
    example = [again[0], again[-1], again[-1]] if again else adds[:2]
    words = (f"This question takes several of its options in one answer, each adding its own to "
             f"the {multi} recorded, in the order named: write each option's id after answer, "
             f"each id its own word, separated by spaces, never joined by commas, as in {INVOKE} "
             f"answer --run {run_id} {' '.join(example)} "
             f"{'--set ... ' if again else ''}--marker MARKER --reason \"WHY\".")
    if again:
        words += (f" {_and(again)} may {'each ' if len(again) > 1 else ''}be named more than "
                  f"once, a rule of your own each time, set with --set.")
    placed = [f for f in composed.get("fields") or [] if f.get("applies_to")]
    if placed:
        second = example[1] if len(example) > 1 else None
        name = next((str(f["name"]) for f in placed if second in f["applies_to"]),
                    str(placed[0]["name"]))
        words += (f" A setting goes to the one option named that takes it; where more than one "
                  f"option named takes it, write the place of the one it is for, among those "
                  f"named, before its name: --set 2.{name}=VALUE sets the second option's {name}.")
    if alone:
        words += f" {_and(alone)} {'is' if len(alone) == 1 else 'are each'} answered alone."
    return words


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
                out["multi"] = several(composed, run.id)
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
    declarable = set(decl.kinds())
    others = [ln for ln in lines
              if (ln.get("status") == "open" and ln.get("counted")
                  and not (step is not None and ln.get("source") == "question"
                           and ln.get("key") == step["key"]))
              or (ln.get("source") == "declaration" and ln.get("key") in declarable
                  and ln.get("status") in ("open", "set_for_you"))]
    declared: dict[str, Any] = {}
    if others:
        described = [describe_line(ln) for ln in others]
        acts = finding_options(run, client, state.get("purpose"))
        artifact = served(run, client)
        for line in described:
            if line.get("source") == "finding" and line.get("key") in acts:
                line["options"] = acts[line["key"]]
                ways += line["options"]
            if line.get("source") == "declaration" and line.get("key") in declarable:
                try:
                    offered = decl.compose(line["key"], view, artifact)
                except Exception as exc:  # noqa: BLE001 - a driver defect on one line, said there
                    line["options_why"] = (f"The driver could not compose this line's options "
                                           f"({type(exc).__name__}: {str(exc)[:200]}).")
                    continue
                declared[line["key"]] = offered
                line.update({k: offered[k] for k in ("words", "options", "fields", "data")
                             if offered.get(k)})
        out["also_open"] = described
        ways += [w for ln in others for item in line_items(ln) for w in item["options"]]
    failed = failures(view, [*((step or {}).get("waiting_on") or []), "fit"])
    if failed:
        out["fit_failed"] = failed
        ways += [w for f in failed for w in (f["refusal"] or {}).get("exits") or []]
    effects = failures(view, ["effects"])
    if effects:  # the effects stage stopped (under Estimate): its own words and ways forward
        out["effects_failed"] = effects
        ways += [w for f in effects for w in (f["refusal"] or {}).get("exits") or []]
    kept = run.kept_exits()
    if kept:
        out["last_refusal_exits"] = kept
        ways += kept
    out["decisions"] = decision_log(view)
    out["_composed"], out["_ways"], out["_view"] = composed, ways, view
    out["_declared"] = declared
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
    run_id = new_run_id(runs_root())
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
        "commit": _git_head(), "vocabulary": V.DIGEST,
        **{k: v for k, v in (("analyst_model", args.model), ("persona", args.persona),
                             ("effort", args.effort), ("prompt_version", args.prompt_version))
           if v is not None}})
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
    """The open question; with ``--decisions`` the decision log alone, every decision with its
    note; with ``--line <id>`` one of the other open lines ``also_open`` lists, whole."""
    run = Run.load(args.run)
    client = run.client()
    if getattr(args, "decisions", False):
        return 0, {"run": run.id, "decisions": decision_log(view_of(run, client))}, run
    q = the_question(run, client, args.wait)
    wanted = getattr(args, "line", None)
    if wanted is None:
        return 0, private_free(q), run
    lines = q.get("also_open") or []
    line = next((ln for ln in lines if ln.get("id") == wanted), None)
    if line is None:
        listed = ", ".join(str(ln.get("id")) for ln in lines) or "none"
        own = (q.get("line") or {}).get("id")
        raise DriverError("no_such_line", (
            f"{wanted} is the open question's own line, which ask shows itself; the other open "
            f"lines are {listed}." if wanted == own else
            f"No other open line {wanted} is listed now; the other open lines ask lists under "
            f"also_open are {listed}."))
    return 0, {"run": run.id, "also_open": [private_free(line)]}, run


def cmd_part(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    """``<command> --part N``: part N of ``command``'s last output too long to print whole, as
    kept when it printed its first part (:func:`to_cap`); nothing else runs."""
    run = Run.load(args.run)
    kept = run.kept_parts(args.command)
    if kept is None:
        raise DriverError("no_such_part", f"No output of {args.command} is kept in parts for "
                                          f"this run: its last output printed whole, or none "
                                          f"was too long to, so there is no part {args.part}.")
    stale = kept.get("stale")
    if stale:
        again = (f" {INVOKE} {args.command} --run {run.id} shows the run as it stands."
                 if args.command in ("ask", "results") else "")
        raise DriverError("no_such_part", f"The output of {kept['of']} at {kept['made']} is no "
                                          f"longer the run as it stands: {stale['command']} ran "
                                          f"after it, at {stale['at']}, so its parts are not "
                                          f"shown.{again}")
    texts = list(kept.get("parts") or [])
    if args.part > len(texts):
        raise DriverError("no_such_part", f"The output of {kept['of']} at {kept['made']} is in "
                                          f"{len(texts)} parts; there is no part {args.part}.")
    return 0, json.loads(texts[args.part - 1]), run


def chosen(run: Run, client: Client, ids: list[str], sets: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    # What is offered is read once the question is open: a reading just confirmed rebuilds the
    # working table, and the question waits on it for a moment (journeys' ``not_yet`` wait).
    q = the_question(run, client, NOT_YET_SECONDS)
    guard_sealed(q["_view"])
    kind = decl.owner(ids[0], q["_declared"]) if ids else None
    try:
        if kind is not None:
            decision = decl.resolve(q["_declared"][kind], ids, sets)
        else:
            decision = opts.resolve(q["_composed"], ids, sets, q["_ways"])
    except opts.Unanswerable as exc:
        raise DriverError("not_offered", str(exc)) from None
    return decision, q


def marker_of(name: str | None) -> str:
    """``--marker``'s name, one of the controlled vocabulary's (SWARM_PLAN §2)."""
    if name not in V.NAMES:
        raise DriverError("marker_unknown", "--marker names a marker of the vocabulary: "
                                            + V.unknown(str(name), V.NAMES))
    return str(name)


def rejected_of(q: dict[str, Any], chosen_id: str, rejected: str | None) -> dict[str, Any] | None:
    """The alternative ``--rejected`` names: an option offered for the same question as
    ``chosen_id`` (the open question's own options, or the same declaration line's), not the one
    chosen, with the decision it would record; None when none is named."""
    if rejected is None:
        return None
    kind = decl.owner(chosen_id, q["_declared"])
    offered = (q["_declared"][kind] if kind is not None else q["_composed"]).get("options") or []
    option = next((o for o in offered if str(o.get("id")) == rejected), None)
    if option is None or rejected == chosen_id:
        ids = [str(o.get("id")) for o in offered if str(o.get("id")) != chosen_id]
        raise DriverError("not_offered", f"--rejected names another option offered for the same "
                                         f"question: one of {', '.join(ids) or 'none'}; not "
                                         f"{rejected!r}.")
    return {"id": rejected, "decision": option.get("decision")}


def cmd_preview(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    reason_of(args.reason)  # SWARM_PLAN §2: a preview says why, as an answer does
    decision, _ = chosen(run, client, list(args.option), args.set or [])
    guard_kind(decision, allowed=("open_seal",))  # what opening would show, without opening
    r = client.post(f"/projects/{run.project}/preview", decision)
    if r.status != 200:
        return 1, {"previewed": decision} | refused(r, run, "preview"), run
    return 0, {"previewed": decision, "preview": trim(r.json())}, run


def cmd_answer(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    run = Run.load(args.run)
    client = run.client()
    note = reason_of(args.reason)
    marker = marker_of(args.marker)
    decision, q = chosen(run, client, list(args.option), args.set or [])
    tag = {"marker": marker, "rejected": rejected_of(q, args.option[0], args.rejected)}
    guard_kind(decision)
    guard_budget(run, client, q["_view"], decision)
    r = post_decision(run, client, decision, note)
    if r.status != 200:
        return 1, {"posted": decision, "tag": tag} | refused(r, run, "answer"), run
    run.keep_exits([], None, "answer")
    return 0, recorded(r.json(), str(decision["kind"])) | {"tag": tag}, run


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
    return count_met(spent) or spent["seconds"] >= float(spent["budget_seconds"])


def budget_words(spent: dict[str, Any], count: bool) -> str:
    """The budget and what was spent of it, from ``GET /fits``, as a refusal says them: the
    count's words where the count's rule fired (``count``), else the seconds'."""
    budget, made = int(spent["budget_fits"]), int(spent["fits"])
    if count:
        done = (f"{made} {'was' if made == 1 else 'were'} made and 1 is under way" if spent["running"]
                else f"{made} {'was' if made == 1 else 'were'} made already")
        return f"the run's fit budget is {budget} {'fit' if budget == 1 else 'fits'}, and {done}"
    return (f"the run's fit budget is {float(spent['budget_seconds']):g} seconds of fitting, and "
            f"{spent['seconds']:g} were spent already")


def count_met(spent: dict[str, Any]) -> bool:
    """:func:`budget_met`'s count: the fits made, the one under way among them, meet the budget's."""
    return spent["fits"] + spent["running"] >= int(spent["budget_fits"])


PAST_BUDGET = ("Past the budget, Fit is pressed only for answers already fitted, which makes no "
               "new fit.")


def over_budget(spent: dict[str, Any], fit_key: str | None) -> str | None:
    """Why pressing Fit now would pass the run's fit budget, or None. A press for the answers as
    they stand whose fit was made already (its key among ``GET /fits``' records: one the
    scheduler ran before the press) makes no new fit, so neither rule refuses it; any other press
    would make one (the fit under way counts as that one), so it is refused once the fits made
    reach the budget's count or their seconds its seconds. It says the rule that fired: the
    count's where the fits made reach it, else the seconds'."""
    if fit_key in spent["keys"]:
        return None
    count = spent["fits"] + 1 > int(spent["budget_fits"])
    if count or spent["seconds"] >= float(spent["budget_seconds"]):
        return f"Fit is not pressed: {budget_words(spent, count)}. {PAST_BUDGET}"
    return None


AFTER_BUDGET = ("An answer the driver can tell makes no new fit is still recorded, and results, "
                "seal and export still run.")


def attempt_over_budget(spent: dict[str, Any], after: str | None) -> str | None:
    """Why recording an answer that leaves the fit keyed ``after`` would pass the budget, or None:
    within it every answer passes; once it is met, only one that leaves no fit (``None``), a fit
    already made, or the one under way."""
    if not budget_met(spent) or after is None or after in spent["keys"] or after == spent["running_key"]:
        return None
    return (f"Not recorded: this would make a new fit if recorded, and "
            f"{budget_words(spent, count_met(spent))}. {AFTER_BUDGET}")


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
    (``decisions.complete``, which ``decisions.validate`` runs after the validators, which only
    refuse). None when a completion refuses it, as the server then does: nothing is recorded."""
    from turbotab.core import decisions

    try:
        decision = decisions.complete(decision, ctx)
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
        raise DriverError("fit_budget_unknown", f"Not recorded: "
                                                f"{budget_words(spent, count_met(spent))}, and "
                                                f"the driver cannot tell whether this would make "
                                                f"a new fit: {exc}.",
                          spent=public_budget(spent)) from None
    new = [key for key, _ in afters if attempt_over_budget(spent, key)]
    if not new:
        return
    if len(new) < len(afters):  # some of the ways the server may record it make a new fit
        raise DriverError("fit_budget_unknown", f"Not recorded: "
                                                f"{budget_words(spent, count_met(spent))}, and "
                                                f"the driver cannot tell whether this would make "
                                                f"a new fit: that depends on whether TurboTab "
                                                f"records {afters[-1][1]} itself with it.",
                          spent=public_budget(spent))
    raise DriverError("fit_budget", attempt_over_budget(spent, new[0]), spent=public_budget(spent))


def wait_fit(run: Run, client: Client, seconds: float) -> dict[str, Any]:
    """The fit's status once it is fresh, failed or blocked, once it waits for a press of Fit
    (``held: "fit"``), or after ``seconds``. A fit held while the shelf measures its time
    (``held: "estimate"``) is waited for: the hold ends by itself (``ProjectService._hold``)."""
    end = time.monotonic() + seconds
    while True:
        status = (view_of(run, client).get("stages") or {}).get("fit") or {}
        if status.get("status") in ("fresh", "error", "blocked") or status.get("held") == "fit" \
                or time.monotonic() >= end:
            return {k: status.get(k) for k in ("status", "key", "error", "held", "progress",
                                               "refusal")
                    if status.get(k) is not None}
        time.sleep(0.5)


def keep_markers(run: Run, client: Client, command: str) -> None:
    """The live markers as the project stands, and the facts of the folds its fit is scored on
    (``markers.split_now``), kept by the current fit's key in ``markers.jsonl``
    (``research/swarm/markers.py``). It never fails the command: a read that fails keeps
    nothing."""
    from research.swarm import markers

    try:
        view = view_of(run, client)
        read = reader(run, client)
        snapshot = {"at": now_us(), "command": command, "fit_key": markers.fit_key(view),
                    "markers": markers.live(view, read), "split": markers.split_now(view, read)}
        with (run.folder / MARKERS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - the markers are the record's, never the command's
        return


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
    keep_markers(run, client, "fit")
    out = {"pressed": lock, "fit": status, "spent": public_budget(budget_now(run, client))}
    refusal = status.get("refusal")
    if status.get("status") == "error" and refusal:
        # The engine refused the fit in words: a refusal with its ways forward, as the server's.
        ways = opts.exits(refusal.get("exits"))
        run.keep_exits(ways, refusal.get("code"), "fit")
        return 1, out | {"refused": {"by": "engine", "stage": refusal.get("stage"),
                                     "code": refusal.get("code"),
                                     "message": refusal.get("message"), "exits": ways}}, run
    return 0, out, run


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
    keep_markers(run, client, "results")
    if r.status != 200:
        return 1, refused(r, run, "results"), run
    out: dict[str, Any] = {"results": r.json(),
                           "stages": {s: (stages.get(s) or {}).get("status")
                                      for s in ("fit", "effects", "mattered") if s in stages}}
    effects = failures(view, ["effects"])
    if effects:  # the results serve no words for a stage that stopped: its status's, as served
        out["effects_failed"] = effects
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
    keep_markers(run, client, "seal")
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
    keep_markers(run, client, "export")
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


def cmd_want(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    """An option, a check, a reading or an eligibility rule the engine lacks (SWARM_PLAN §2):
    its kind, what, why and the marker it answers to, kept in ``commands.jsonl``; nothing is
    posted."""
    run = Run.load(args.run)
    run.client()  # a stopped run takes no more commands
    what = (args.what or "").strip()
    if not what:
        raise DriverError("want_unsaid", "Say what the engine lacks: want takes its words.")
    if args.kind not in V.WANT_KINDS:
        raise DriverError("want_kind_unknown", "--kind names what is wanted: "
                                               + V.unknown(str(args.kind), V.WANT_KINDS))
    reason = reason_of(args.reason)
    marker = marker_of(args.marker)
    return 0, {"wanted": {"at": now_us(), "kind": args.kind, "what": what, "reason": reason,
                          "marker": marker}}, run


def account_given(run: Run) -> str | None:
    """When the run's account was given, from ``commands.jsonl``, or None: an ``account`` that
    kept its account (a read of its output's part keeps none)."""
    for entry in _commands(run):
        given = entry.get("summary") or {}
        if entry.get("command") == "account" and given.get("exit") == 0 and "account" in given:
            return str(entry.get("at"))
    return None


def _commands(run: Run) -> list[dict[str, Any]]:
    try:
        lines = (run.folder / "commands.jsonl").read_text("utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def cmd_account(args: argparse.Namespace) -> tuple[int, dict[str, Any], Run | None]:
    """The analyst's end-of-run account (SWARM_PLAN §4), once per run, kept in
    ``commands.jsonl`` and marked whether a ``results`` command had returned the results before
    it and whether the held-out rows were open; nothing is posted."""
    run = Run.load(args.run)
    client = run.client()
    text = (args.text or "").strip()
    if not text:
        raise DriverError("account_unsaid", "The account takes the analyst's words.")
    given = account_given(run)
    if given is not None:
        raise DriverError("account_given", f"The run's account was given at {given}; it is "
                                           f"given once.")
    after_results = any(e.get("command") == "results"
                        and (e.get("summary") or {}).get("exit") == 0 for e in _commands(run))
    after_seal = bool((view_of(run, client).get("state") or {}).get("seal_opened"))
    return 0, {"account": {"at": now_us(), "text": text, "after_results": after_results,
                           "after_seal": after_seal}}, run


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


# ── what a command prints: under the cap (the module docstring) ─────────────

HEADER_ROOM = 2_000  # what a part's own header (its ``part``) may take of the cap
MIN_ROOM = 400  # below this much room, a value is not cut further
LINE_BRIEF = ("id", "key", "source", "name", "status")  # an other open line, listed briefly
PIECE = "piece"  # a path's mark of one piece of a text cut across parts


def printed_size(text: str) -> int:
    """What ``text`` weighs against :data:`OUTPUT_CAP`: its UTF-8 bytes, which no count of its
    characters exceeds (code points, or UTF-16 units as a JavaScript string counts them)."""
    return len(text.encode("utf-8"))


def _dumped(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _weight(value: Any) -> int:
    return printed_size(_dumped(value))


def held_to_cap(run: Run | None) -> bool:
    """Whether a run's analyst reads what its commands print through Claude Code's Bash tool,
    whose limit :data:`OUTPUT_CAP` keeps under: every analyst but a script (``start
    --scripted``), which reads the JSON in its own process."""
    return run is None or run.meta.get("analyst") != "scripted"


def log_words(n: int, k: int, run_id: str) -> str:
    """What a shortened ``ask`` says of the decision log it lists: the ``k`` newest of its ``n``
    decisions, each without its note, and the command that lists them all."""
    command = f"{INVOKE} ask --run {run_id} --decisions"
    if k == n:
        lead = ("The decision is listed without its note" if n == 1 else
                f"Each of the {n} decisions is listed without its note")
    elif k == 0:
        lead = f"None of the {n} decisions is listed here"
    else:
        before = n - k
        one = k == 1
        lead = (f"{'The newest' if one else f'The {k} newest'} of the {n} decisions "
                f"{'is' if one else 'are'} listed, without {'its note' if one else 'their notes'}, "
                f"and the {before} before {'it' if one else 'them'} "
                f"{'is' if before == 1 else 'are'} left out")
    return f"{lead}; {command} lists every decision with its note."


def lines_words(m: int, run_id: str) -> str:
    """What a shortened ``ask`` says of the other open lines it lists by :data:`LINE_BRIEF`."""
    lead = "The other open line is" if m == 1 else f"Each of the {m} other open lines is"
    return (f"{lead} listed by its {', '.join(LINE_BRIEF[:-1])} and {LINE_BRIEF[-1]} alone; "
            f"{INVOKE} ask --run {run_id} --line ID shows one in full, with its options, "
            f"fields and readings, ID its id as listed here.")


def short_ask(payload: dict[str, Any], run_id: str) -> tuple[dict[str, Any], bool]:
    """``ask``'s output shortened toward the cap, and whether it fits: first each decision without
    its note; then, too, each other open line by :data:`LINE_BRIEF` alone; then only the newest
    decisions there is room for. The question's own words, options, fields, readings, ways
    forward and refusals are kept whole, and ``left_out`` says what is left out and the command
    that shows it. Where none fits, the last, with no decision listed, is the one printed in
    parts."""
    log = list(payload.get("decisions") or [])
    lines = list(payload.get("also_open") or [])
    bare = [{k: v for k, v in d.items() if k != "note"} for d in log]
    brief = [{k: ln[k] for k in LINE_BRIEF if k in ln} for ln in lines]
    noted = any("note" in d for d in log)

    def built(k: int, short_lines: bool) -> dict[str, Any]:
        left: dict[str, str] = {}
        if log and (k < len(log) or noted):
            left["decisions"] = log_words(len(log), k, run_id)
        if short_lines:
            left["also_open"] = lines_words(len(lines), run_id)
        out: dict[str, Any] = {"run": payload.get("run"), "left_out": left}
        out.update({key: value for key, value in payload.items() if key != "run"})
        if "decisions" in payload:
            out["decisions"] = bare[len(bare) - k:] if k else []
        if short_lines:
            out["also_open"] = brief
        return out

    def fits(shown: dict[str, Any]) -> bool:
        return _weight(shown) <= OUTPUT_CAP

    n = len(log)
    for shown in [built(n, False)] + ([built(n, True)] if lines else []):
        if fits(shown):
            return shown, True
    low, high = 0, n - 1  # the most decisions there is room for, if any: their weight grows with k
    if not fits(built(0, bool(lines))):
        return built(0, bool(lines)), False
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if fits(built(mid, bool(lines))) else (low, mid - 1)
    return built(low, bool(lines)), True


def _cost(path: tuple) -> int:
    """The most a value at ``path`` adds to a part beside its own weight: each key's or list
    place's wrapping."""
    return sum(6 + (_weight(p) if isinstance(p, str) else 0) for p in path)


def _pieces(text: str, room: int) -> list[str]:
    """``text`` cut into pieces, in order, each weighing at most ``room`` printed."""
    out, i = [], 0
    while i < len(text):
        n = min(len(text) - i, room)
        while n > 1 and _weight(text[i:i + n]) > room:
            n = max(1, min(n - 1, n * room // _weight(text[i:i + n])))
        out.append(text[i:i + n])
        i += n
    return out


def _units(value: Any, path: tuple, room: int) -> Any:
    """``value`` (at ``path``) as the units a part takes whole: itself where it fits in ``room``
    with its path's wrapping, else an object by its keys, a list by its items and a text by its
    pieces, each in order."""
    left = room - _cost(path)
    if left < MIN_ROOM or _weight(value) <= left:
        yield path, value
    elif isinstance(value, dict) and value:
        for k, v in value.items():
            yield from _units(v, path + (k,), room)
    elif isinstance(value, list) and value:
        for i, v in enumerate(value):
            yield from _units(v, path + (i,), room)
    elif isinstance(value, str):  # pieces a quarter of a part at most, so a part ends fuller
        for j, piece in enumerate(_pieces(value, min(left - 6, room // 4))):
            yield path + ((PIECE, j),), piece
    else:
        yield path, value


class _Seq(dict):
    """A list cut across parts: its items by their place in the whole."""


class _Text(list):
    """A text cut across parts: its pieces in order."""


def _assemble(units: list[tuple[tuple, Any]]) -> dict[str, Any]:
    """The part that ``units`` make: each value at its path, a list's items and a text's pieces
    in order."""
    root: dict[str, Any] = {}
    for path, value in units:
        node: Any = root
        for depth, head in enumerate(path):
            if depth == len(path) - 1:
                if isinstance(head, tuple):
                    node.append(value)
                else:
                    node[head] = value
                break
            nxt = path[depth + 1]
            if head not in node:
                node[head] = (_Text() if isinstance(nxt, tuple) else _Seq() if isinstance(nxt, int)
                              else {})
            node = node[head]

    def plain(node: Any) -> Any:
        if isinstance(node, _Text):
            return "".join(node)
        if isinstance(node, _Seq):
            return [plain(v) for v in node.values()]
        if isinstance(node, dict):
            return {k: plain(v) for k, v in node.items()}
        return node

    return plain(root)


def _groups(payload: dict[str, Any], room: int) -> list[list[tuple[tuple, Any]]]:
    """``payload``'s units packed in order into parts of at most ``room`` printed each."""
    groups: list[list[tuple[tuple, Any]]] = []
    current: list[tuple[tuple, Any]] = []
    bound = 2  # an upper bound of what ``current`` weighs assembled; exact where it matters
    for unit in _units(payload, (), room):
        add = _weight(unit[1]) + _cost(unit[0])
        if current and bound + add > room:
            exact = _weight(_assemble(current + [unit]))
            if exact > room:
                groups.append(current)
                current, bound = [unit], _weight(_assemble([unit]))
                continue
            bound = exact
        else:
            bound += add
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def _common(a: tuple, b: tuple) -> tuple:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def part_head(k: int, n: int, of: str, command: str, run_id: str, made: str,
              continues: tuple) -> dict[str, Any]:
    """Part ``k`` of ``n``'s own words: what it is, where it goes on from the part before, and
    the command that prints the next."""
    head: dict[str, Any] = {"this": k, "of": n, "made": made, "words": (
        f"The output of {of} at {made} is too long to print whole, so it is printed in {n} "
        f"parts, and this is part {k}. The parts hold its keys in order; where one ends inside a "
        f"list, a text or an object, the next goes on with it and names where under continues "
        f"(keys, and list places counted from 0)."
        + (f" {INVOKE} {command} --run {run_id} --part {k + 1} prints part {k + 1}." if k < n
           else " This is the last part."))}
    if continues:
        head["continues"] = list(continues)
    if k < n:
        head["next"] = f"{INVOKE} {command} --run {run_id} --part {k + 1}"
    return head


def in_parts(payload: dict[str, Any], of: str, command: str, run_id: str, made: str) -> list[str]:
    """``payload`` printed in parts, each at most :data:`OUTPUT_CAP` with its header
    (:func:`part_head`); a refusal where it cannot be."""
    plain = json.loads(_dumped(payload))
    for room in (HEADER_ROOM, 4 * HEADER_ROOM):
        groups = _groups(plain, OUTPUT_CAP - room)
        texts = []
        for k, group in enumerate(groups, start=1):
            continues = _common(groups[k - 2][-1][0], group[0][0]) if k > 1 else ()
            head = part_head(k, len(groups), of, command, run_id, made, continues)
            texts.append(_dumped({"part": head} | _assemble(group)))
        if all(printed_size(t) <= OUTPUT_CAP for t in texts):
            return texts
    raise DriverError("output_unsplit", "This output is too long to print whole, and the driver "
                                        "could not print it in parts.")


def to_cap(args: argparse.Namespace, run: Run | None, payload: dict[str, Any],
           text: str) -> tuple[str, dict[str, Any] | None]:
    """What an analyst's command prints, at most :data:`OUTPUT_CAP`, and what ``commands.jsonl``
    keeps of how (None where it printed whole): ``text`` where it fits; else ``ask``'s question
    shortened (:func:`short_ask`) where that fits; else part 1 of the output (``ask``'s
    shortened), each part kept in the run for ``<command> --part N``. An output that prints
    whole drops the kept parts of the same output (:meth:`Run.drop_parts`): ``ask --decisions``
    printed whole leaves the question's parts kept."""
    of = args.command
    if args.command == "ask" and getattr(args, "decisions", False):
        of = "ask --decisions"
    elif args.command == "ask" and getattr(args, "line", None) is not None:
        of = f"ask --line {args.line}"
    full = printed_size(text)
    if full <= OUTPUT_CAP:
        if run is not None:
            run.drop_parts(args.command, of)
        return text, None
    if run is None:
        return _dumped({"refused": {"by": "driver", "code": "output_too_long", "message": (
            "This output is too long to print whole, and with no run named the driver keeps no "
            "parts of it.")}}), {"bytes": full, "as": "refused"}
    shown = payload
    if of == "ask" and "refused" not in payload:
        shown, fits = short_ask(payload, run.id)
        if fits:
            run.drop_parts(args.command, of)
            return _dumped(shown), {"bytes": full, "as": "shortened"}
    made = now()
    try:
        texts = in_parts(shown, of, args.command, run.id, made)
    except DriverError as exc:
        return _dumped({"refused": {"by": "driver", "code": exc.code, "message": exc.message}}), \
            {"bytes": full, "as": "refused"}
    run.keep_parts(args.command, of, made, texts)
    return texts[0], {"bytes": full, "as": "parts", "parts": len(texts)}


def part_summary(code: int, payload: dict[str, Any]) -> dict[str, Any]:
    """What ``commands.jsonl`` keeps of a read of an output's part: the part, or its refusal;
    never what the command it is a part of records."""
    if "refused" in payload:
        r = payload["refused"]
        return {"exit": code, "refused": {k: r.get(k) for k in ("by", "code", "message")}
                | {"exits": []}}
    head = payload.get("part") or {}
    return {"exit": code, "part": {k: head.get(k) for k in ("this", "of", "made")}}


def summary(name: str, code: int, payload: dict[str, Any]) -> dict[str, Any]:
    """What ``commands.jsonl`` keeps of a response: enough to follow the run, not the whole (the
    record's assembler reads its ``refused``, each way forward by its id and label, and its
    ``exported``)."""
    out: dict[str, Any] = {"exit": code}
    if "refused" in payload:
        r = payload["refused"]
        out["refused"] = {k: r.get(k) for k in ("by", "status", "code", "message")}
        out["refused"]["exits"] = [{"id": e.get("id"), "label": e.get("label")}
                                   for e in r.get("exits") or [] if isinstance(e, dict)]
    if "by" in (payload.get("refused") or {}) and payload["refused"].get("stage"):
        out["refused"]["stage"] = payload["refused"]["stage"]
    if name == "ask":
        q = payload.get("question") or {}
        out.update({"question": q.get("key"), "status": q.get("status"),
                    "options": [o.get("id") for o in payload.get("options") or []],
                    "fit_reached": payload.get("fit_reached")})
        if payload.get("fit_failed"):
            out["fit_failed"] = payload["fit_failed"]
    for key in ("tag", "wanted", "account"):
        if key in payload:
            out[key] = payload[key]
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
    "seal": cmd_seal, "export": cmd_export, "want": cmd_want, "account": cmd_account,
}
ANALYST_COMMANDS = ("ask", "preview", "answer", "revisit", "fit", "results", "seal", "export",
                    "want", "account")


def _wait(text: str) -> float:
    value = float(text)
    if not 0 <= value <= WAIT_MAX:
        raise argparse.ArgumentTypeError(f"--wait is 0 to {WAIT_MAX:g} seconds")
    return value


def _part(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError("--part is a part's number, 1 or more")
    return value


OPTION_HELP = ("the option's id, as ask lists it; where ask shows multi, several ids, each id its "
               "own word, separated by spaces, never joined by commas")
PART_HELP = ("print part N of this command's last output too long to print whole, and run "
             "nothing else; takes --run alone beside it")


def reads_a_part(argv: list[str]) -> bool:
    """Whether ``argv`` asks for an output's part (``<command> --run <id> --part N``)."""
    return bool(argv) and argv[0] in ANALYST_COMMANDS and any(
        a == "--part" or a.startswith("--part=") for a in argv[1:])


def part_parser() -> argparse.ArgumentParser:
    """The command line of a read of an output's part: an analyst's command, its run and the
    part, nothing else."""
    parser = argparse.ArgumentParser(prog="python -m research.swarm.driver",
                                     description="A part of a command's last output.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ANALYST_COMMANDS:
        q = sub.add_parser(name)
        q.add_argument("--run", required=True)
        q.add_argument("--part", type=_part, required=True, metavar="N", help=PART_HELP)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.driver",
                                     description="The swarm's driver (DRIVER_PLAN D1).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="start a run's server and upload its data file")
    p.add_argument("--data", required=True)
    p.add_argument("--provenance", help="url,release date (YYYY-MM-DD),license")
    p.add_argument("--budget-fits", type=int, required=True)
    p.add_argument("--budget-seconds", type=float, required=True)
    p.add_argument("--scripted", action="store_true",
                   help="the analyst is a script, not an agent (the dry run, the tests)")
    p.add_argument("--model", help="the analyst's model, kept for the record")
    p.add_argument("--persona", choices=PERSONAS, help="the analyst's persona (SWARM_PLAN §6)")
    p.add_argument("--effort", help="the analyst's effort, kept for the record")
    p.add_argument("--prompt-version", help="the analyst's prompt version, kept for the record")

    def with_run(name: str, help_: str, wait: bool = False) -> argparse.ArgumentParser:
        q = sub.add_parser(name, help=help_)
        q.add_argument("--run", required=True)
        if wait:
            q.add_argument("--wait", type=_wait, default=WAIT_DEFAULT,
                           help=f"seconds to wait for what is computing (default {WAIT_DEFAULT:g})")
        if name in ANALYST_COMMANDS:  # read by part_parser, which main uses where it is given
            q.add_argument("--part", type=_part, metavar="N", help=PART_HELP)
        return q

    with_run("stop", "stop the run's server")
    p = with_run("ask", "the open question", wait=True)
    shown = p.add_mutually_exclusive_group()
    shown.add_argument("--decisions", action="store_true", default=None,  # unlogged unless given
                       help="the decision log alone, every decision with its note")
    shown.add_argument("--line", metavar="ID",
                       help="one other open line in full, by its id under also_open")
    p = with_run("preview", "what recording an option would do, with why; nothing is recorded")
    p.add_argument("option", nargs="+", help=OPTION_HELP)
    p.add_argument("--set", action="append", metavar="NAME=VALUE")
    p.add_argument("--reason", required=True)
    p = with_run("answer", "record an option, with why and the marker it answers to")
    p.add_argument("option", nargs="+", help=OPTION_HELP)
    p.add_argument("--set", action="append", metavar="NAME=VALUE")
    p.add_argument("--reason", required=True)
    p.add_argument("--marker", required=True, help="a marker of the controlled vocabulary")
    p.add_argument("--rejected", help="the option offered for the same question it rejected")
    p = with_run("revisit", "change an earlier decision (the log keeps it), with why")
    p.add_argument("decision")
    p.add_argument("--reason", required=True)
    with_run("fit", "press Fit, within the run's fit budget", wait=True)
    with_run("results", "the results the app shows now", wait=True)
    p = with_run("seal", "open the held-out rows, once, declaring the final family, with why")
    p.add_argument("family")
    p.add_argument("--reason", required=True)
    with_run("export", "the methods, the checklist and the bundle", wait=True)
    p = with_run("want", "an option, a check, a reading or an eligibility rule the engine lacks")
    p.add_argument("what")
    p.add_argument("--kind", required=True, help=f"one of {', '.join(V.WANT_KINDS)}")
    p.add_argument("--reason", required=True)
    p.add_argument("--marker", required=True, help="a marker of the controlled vocabulary")
    p = with_run("account", "the analyst's end-of-run account, given once")
    p.add_argument("text")
    return parser


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None, out: Callable[[str], None] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    reading = reads_a_part(argv)
    parser = part_parser() if reading else build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else 2
    write = out or (lambda text: print(text, flush=True))
    run: Run | None = None
    started, began = now(), now_us()
    try:
        try:
            if args.command in ANALYST_COMMANDS:
                with analyst_sealed():
                    code, payload, run = (cmd_part if reading else COMMANDS[args.command])(args)
            else:
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
    text = json.dumps(payload, ensure_ascii=False, default=str)
    printed = None
    if not reading and args.command in ANALYST_COMMANDS and held_to_cap(run):
        text, printed = to_cap(args, run, payload, text)
    write(text)
    if run is not None:
        by = (payload.get("refused") or {}).get("by")
        if not reading and args.command in RECORDING and (code == 0 or by not in ("driver",
                                                                                  "server")):
            run.stale_parts(args.command, now())
        given = {k: v for k, v in vars(args).items() if k not in ("command",) and v is not None}
        kept = part_summary(code, payload) if reading else summary(args.command, code, payload)
        if printed:
            kept["printed"] = printed
        run.log({"at": started, "done": now(), "began": began, "returned": now_us(),
                 "command": args.command, "args": given, "summary": kept})
    return code


if __name__ == "__main__":
    raise SystemExit(main())

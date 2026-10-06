"""Start TurboTab on this computer: the launcher behind the one command and the double-click files.

    python3 turbotab/deploy/launch.py [--port N] [--no-open] [--smoke]

(``turbotab.sh`` on macOS and Linux and ``turbotab.ps1`` on Windows find a Python 3.12+ first and
run this; ``Start TurboTab.command`` and ``Start TurboTab.bat`` are those for a double-click.)

Every step is skipped when it is already done, so the second start takes seconds:

1. TurboTab already running for this workspace: open it in the browser and stop here.
2. A Python 3.12+ environment at ``$TURBOTAB_VENV`` (default ``$TURBOTAB_HOME/env``): created when
   missing or broken; ``turbotab/server/requirements.txt`` installed into it once, and again only
   when that file changes (its SHA-256 is kept in the environment).
3. The interface: ``turbotab/frontend/dist`` as it is; built with npm only when it is missing and
   Node.js 20+ is installed; otherwise the launcher says how to get it and starts anyway.
4. ``python -m turbotab.server --mode local --open`` on 127.0.0.1, on port 8787 when it is free
   (else any free port), with the browser opened once it answers.
5. Ctrl+C, closing the window or a stop signal shuts the server down cleanly (its job workers
   included) and frees the port.

``--smoke`` checks a start end to end instead of staying up: health, a small CSV uploaded through
the API and read by the job workers, then a clean stop. CI runs it on macOS and Windows.

Standard library only: this runs before the environment exists.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 12):  # noqa: UP036 - the point is to say so on an old interpreter
    sys.exit(f"TurboTab needs Python 3.12 or newer; this is {sys.version.split()[0]} "
             f"({sys.executable}). Install Python 3.12+ from https://www.python.org/downloads/ "
             "and start TurboTab again.")

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "turbotab" / "server" / "requirements.txt"
FRONTEND = ROOT / "turbotab" / "frontend"
DIST = FRONTEND / "dist"
STAMP = ".turbotab-requirements.sha256"
STATE = "launcher.json"
DEFAULT_PORT = 8787
WINDOWS = os.name == "nt"
MIN_NODE = (20, 19)  # Vite's floor
START_TIMEOUT = 180.0
STOP_TIMEOUT = 20.0
SMOKE_CSV = ("id,energy_kcal,protein_g,fiber_g,outcome\n"
             + "".join(f"{i},{1700 + 11 * i},{48 + i % 23},{12 + i % 9},{i % 2}\n" for i in range(60)))


def say(message: str) -> None:
    print(f"TurboTab: {message}", flush=True)


for _stream in (sys.stdout, sys.stderr):  # a folder name a Windows code page cannot print
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def home() -> Path:
    raw = os.environ.get("TURBOTAB_HOME") or "~/.turbotab"
    return Path(raw).expanduser().absolute()


def venv_dir(workspace: Path) -> Path:
    raw = os.environ.get("TURBOTAB_VENV")
    return Path(raw).expanduser().absolute() if raw else workspace / "env"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WINDOWS else "bin/python")


def run(cmd: list[str], **kwargs) -> int:
    if WINDOWS and Path(cmd[0]).suffix.lower() in (".cmd", ".bat"):
        cmd = ["cmd", "/c", *cmd]  # npm is a batch file on Windows
    return subprocess.run(cmd, **kwargs).returncode


# ── 1. already running? ──────────────────────────────────────────────────────


# Requests to this machine's own server never go through a proxy the system or the environment
# names (a university network often sets one).
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with LOCAL.open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def running(workspace: Path) -> str | None:
    """The address of a TurboTab this launcher started for this workspace, when it still answers."""
    try:
        state = json.loads((workspace / STATE).read_text("utf-8"))
        port = int(state["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    health = get_json(f"http://127.0.0.1:{port}/api/health")
    if health and health.get("mode") == "local":
        return f"http://localhost:{port}/"
    (workspace / STATE).unlink(missing_ok=True)  # left by a launcher that did not stop cleanly
    return None


# ── 2. the Python environment ────────────────────────────────────────────────


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def healthy(python: Path) -> bool:
    if not python.exists():
        return False
    check = "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
    try:
        return subprocess.run([str(python), "-c", check], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_environment(venv: Path) -> Path:
    python = venv_python(venv)
    if not healthy(python):
        if venv.exists():
            if not (venv / "pyvenv.cfg").is_file():
                sys.exit(f"TurboTab: {venv} is not a Python environment TurboTab made; set "
                         "TURBOTAB_VENV to another folder or empty this one.")
            say(f"the Python environment at {venv} no longer works; making it again.")
            shutil.rmtree(venv)
        say(f"making a Python environment at {venv} (Python {sys.version.split()[0]}; once).")
        # From the interpreter as it was found, else from the file a link points at: an
        # environment made through a link to a uv-managed Python cannot find its standard library.
        bases = list(dict.fromkeys([sys.executable, os.path.realpath(sys.executable)]))
        for base in bases:
            shutil.rmtree(venv, ignore_errors=True)
            if run([base, "-m", "venv", str(venv)], capture_output=True) == 0 and healthy(python):
                break
        else:
            shutil.rmtree(venv, ignore_errors=True)
            run([bases[-1], "-m", "venv", str(venv)])  # again, showing why
            sys.exit("TurboTab: the Python environment could not be made (see above).")
    stamp = venv / STAMP
    wanted = requirements_hash()
    if stamp.is_file() and stamp.read_text("utf-8").strip() == wanted:
        say("Python environment: up to date.")
        return python
    say("installing TurboTab's Python libraries (once; a few hundred MB, a few minutes)...")
    uv = shutil.which("uv")
    if uv:
        code = run([uv, "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS)])
    else:
        code = run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    "-r", str(REQUIREMENTS)])
    if code != 0:
        sys.exit("TurboTab: installing the libraries failed (see above). Check the internet "
                 "connection and start TurboTab again; what was installed is kept.")
    stamp.write_text(wanted, "utf-8")
    say("Python environment: ready.")
    return python


# ── 3. the interface ─────────────────────────────────────────────────────────


def node_version(node: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30).stdout
        major, minor = out.strip().lstrip("v").split(".")[:2]
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def ensure_interface() -> str:
    if (DIST / "index.html").is_file():
        say("interface: built.")
        return "built"
    node, npm = shutil.which("node"), shutil.which("npm")
    version = node_version(node) if node else None
    if not (node and npm and version and version >= MIN_NODE):
        found = f" (found Node.js {version[0]}.{version[1]})" if version else ""
        say("the interface is not built, and building it needs Node.js "
            f"{MIN_NODE[0]}.{MIN_NODE[1]} or newer{found}. Install it from https://nodejs.org "
            "and start TurboTab again; it builds the interface once. Starting without it for now: "
            "the page will say how to build it.")
        return "missing"
    say("building the interface (once; about a minute)...")
    build = [npm, "run", "build"]
    # A checkout that already has its packages (a developer's) builds with them; otherwise, or
    # when that fails, the locked packages are installed first.
    if (FRONTEND / "node_modules").is_dir() and run(build, cwd=FRONTEND) == 0:
        say("interface: built.")
        return "built"
    lock = FRONTEND / "package-lock.json"
    install = [npm, "ci" if lock.is_file() else "install", "--no-audit", "--no-fund"]
    for step in (install, build):
        if run(step, cwd=FRONTEND) != 0:
            say("the interface did not build (see above); starting without it.")
            return "missing"
    say("interface: built.")
    return "built"


# ── 4. start, 5. stop ────────────────────────────────────────────────────────


def listening(port: int) -> bool:
    """Whether something answers on 127.0.0.1:``port``."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def port_free(port: int) -> bool:
    """Whether the server could bind ``port`` (as uvicorn does: a closed connection's TIME_WAIT
    does not hold the port)."""
    if listening(port):
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if not WINDOWS:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(asked: int | None) -> int:
    if asked:
        if not port_free(asked):
            sys.exit(f"TurboTab: port {asked} is in use; choose another with --port.")
        return asked
    if port_free(DEFAULT_PORT):
        return DEFAULT_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    def __init__(self, python: Path, workspace: Path, port: int, open_browser: bool):
        self.port = port
        self.url = f"http://localhost:{port}/"
        self.api = f"http://127.0.0.1:{port}/api"
        self.workspace = workspace
        # --stop-on-eof: the server shuts down cleanly when its standard input closes, which is
        # how the launcher stops it on every OS, and why it stops when the launcher's window is
        # closed or the launcher is killed.
        cmd = [str(python), "-m", "turbotab.server", "--mode", "local", "--port", str(port),
               "--stop-on-eof"]
        if open_browser:
            cmd.append("--open")
        env = {**os.environ, "TURBOTAB_HOME": str(workspace), "TURBOTAB_MODE": "local"}
        # Its own process group, so Ctrl+C reaches the launcher alone and the launcher stops the
        # server in order (uvicorn, then its job workers) instead of everything at once.
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
                 else {"start_new_session": True})
        self.started = time.monotonic()
        self.process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.PIPE, **group)
        self.forced = False

    def wait_ready(self) -> dict:
        end = time.monotonic() + START_TIMEOUT
        while time.monotonic() < end:
            if self.process.poll() is not None:
                sys.exit(f"TurboTab: the server stopped while starting (exit code "
                         f"{self.process.returncode}); see above.")
            health = get_json(f"{self.api}/health", timeout=1.0)
            if health:
                return health
            time.sleep(0.25)
        self.stop()
        sys.exit(f"TurboTab: the server did not answer within {START_TIMEOUT:.0f} s.")

    def record(self) -> None:
        state = {"pid": self.process.pid, "launcher": os.getpid(), "port": self.port, "url": self.url}
        (self.workspace / STATE).write_text(json.dumps(state), "utf-8")

    def stop(self) -> int | None:
        """Ask the server to shut down (close its input, then a signal); force it only if it has
        not after ``STOP_TIMEOUT``."""
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(STOP_TIMEOUT)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
                except OSError:
                    pass
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.forced = True
                if WINDOWS:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                                   capture_output=True)
                else:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                self.process.wait(10)
        try:
            state = json.loads((self.workspace / STATE).read_text("utf-8"))
            if state.get("pid") == self.process.pid:
                (self.workspace / STATE).unlink()
        except (OSError, ValueError):
            pass
        return self.process.returncode

    def leftovers(self) -> bool:
        """Whether any process the server started is still alive (POSIX: its process group)."""
        if WINDOWS:
            return False
        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def serve(server: Server) -> int:
    stopping = False

    def on_signal(signum, _frame) -> None:  # noqa: ANN001
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt

    # SIGINT too: a launcher started in the background inherits Ctrl+C as ignored, and Python
    # then installs no handler of its own.
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)
    try:
        while server.process.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stopping = True
    if stopping:
        say("stopping...")
        server.stop()
        say("stopped." if not server.forced else "stopped (it had to be forced).")
        return 0
    say(f"the server stopped by itself (exit code {server.process.returncode}).")
    return server.process.returncode or 1


# ── the smoke check ──────────────────────────────────────────────────────────


def upload(api: str, name: str, content: str) -> dict:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: text/csv\r\n\r\n{content}\r\n--{boundary}--\r\n").encode("utf-8")
    request = urllib.request.Request(f"{api}/projects/upload", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with LOCAL.open(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def smoke(server: Server, health: dict, interface: str) -> int:
    failures: list[str] = []
    say(f"smoke: health answered: TurboTab {health.get('version')}, {health.get('mode')} mode, "
        f"{health.get('workers')} workers.")
    if health.get("mode") != "local":
        failures.append(f"health says mode {health.get('mode')!r}, not 'local'")
    try:
        summary = upload(server.api, "smoke.csv", SMOKE_CSV)
        pid = summary["id"]
        say(f"smoke: uploaded smoke.csv as project {pid}; waiting for the workers to read it...")
        end, stages = time.monotonic() + 300, {}
        while time.monotonic() < end:
            view = get_json(f"{server.api}/projects/{pid}", timeout=10) or {}
            stages = {k: v.get("status") for k, v in (view.get("stages") or {}).items()}
            if stages.get("ingest") == "fresh" and stages.get("profile") == "fresh":
                break
            bad = [k for k in ("ingest", "profile") if stages.get(k) == "error"]
            if bad:
                failures.append(f"{bad[0]} failed: {view['stages'][bad[0]].get('error')}")
                break
            time.sleep(0.5)
        else:
            failures.append(f"ingest and profile were not fresh after 300 s: {stages}")
        if not failures:
            n = (view.get("summary") or {}).get("n_rows")
            say(f"smoke: ingested and profiled ({n} rows).")
            if n != 60:
                failures.append(f"the project has {n} rows, not 60")
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        failures.append(f"the upload failed: {exc}")
    try:
        with LOCAL.open(server.url.replace("localhost", "127.0.0.1"), timeout=10) as r:
            page = r.read().decode("utf-8", "replace")
        built = '<div id="root">' in page
        say(f"smoke: / serves {'the interface' if built else 'the build hint'}.")
        if built != (interface == "built"):
            failures.append("the page served does not match the interface found")
    except (urllib.error.URLError, OSError) as exc:
        failures.append(f"/ did not answer: {exc}")
    code = server.stop()
    still_listening, left = listening(server.port), server.leftovers()
    say(f"smoke: stopped (exit code {code}{', forced' if server.forced else ''}); port "
        f"{server.port} {'STILL ANSWERS' if still_listening else 'free'}"
        f"{'; processes left behind' if left else ''}.")
    if server.forced:
        failures.append("the server did not stop when asked")
    if still_listening or left:
        failures.append("something was left running")
    if failures:
        for failure in failures:
            say(f"smoke FAILED: {failure}")
        return 1
    say("smoke: passed.")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="launch.py", description="Start TurboTab on this computer.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TURBOTAB_PORT") or 0) or None,
                        help=f"default: {DEFAULT_PORT} when free, else any free port")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument("--smoke", action="store_true",
                        help="start, check health and an upload, stop; exit 0 when all held")
    args = parser.parse_args(argv)

    began = time.monotonic()
    workspace = home()
    workspace.mkdir(parents=True, exist_ok=True)
    say(f"workspace {workspace}")
    url = running(workspace)
    if url:
        if args.smoke:
            say(f"already running at {url}; the smoke check needs it stopped first.")
            return 1
        say(f"already running at {url}")
        if not args.no_open:
            webbrowser.open(url)
        return 0

    python = ensure_environment(venv_dir(workspace))
    interface = ensure_interface()
    port = choose_port(args.port)
    server = Server(python, workspace, port, open_browser=not (args.no_open or args.smoke))
    try:
        health = server.wait_ready()
        server.record()
        say(f"running at {server.url} (ready in {time.monotonic() - began:.1f} s; the server "
            f"itself took {time.monotonic() - server.started:.1f} s).")
        if args.smoke:
            return smoke(server, health, interface)
        say("keep this window open while you work; press Ctrl+C or close it to stop TurboTab.")
        return serve(server)
    except KeyboardInterrupt:
        say("stopping...")
        server.stop()
        return 0
    finally:
        if server.process.poll() is None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch TurboTab: ``venv/bin/python -m turbotab.server --port 8787 [--mode local|server] [--open]``.

Local mode binds 127.0.0.1 only. Server mode listens where ``--host`` says
(default 127.0.0.1, so exposing it is a deliberate choice) and signs users in
(``turbotab.server.auth``; accounts: ``python -m turbotab.server.users``).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import sys
import threading
import time
import webbrowser
from typing import Callable

from turbotab.core.config import MODES, Settings


def _open_when_ready(host: str, port: int, url: str, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m turbotab.server", description="Run TurboTab.")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--mode", choices=MODES, default=None, help="default: $TURBOTAB_MODE, else local")
    parser.add_argument("--host", default=None, help="server mode only; local mode always binds 127.0.0.1")
    parser.add_argument("--open", action="store_true", help="open the app in a browser once it is up")
    parser.add_argument("--stop-on-eof", action="store_true",
                        help="shut down cleanly when standard input closes: how the desktop "
                             "launcher (turbotab/deploy/launch.py) stops the server on every OS, "
                             "and why closing the launcher's window stops it too")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.mode:
        settings = dataclasses.replace(settings, mode=args.mode)
    host = args.host or "127.0.0.1"
    if settings.mode == "local" and host != "127.0.0.1":
        parser.error("local mode binds 127.0.0.1 only; pass --mode server to listen elsewhere")

    import uvicorn

    from turbotab.server.app import create_app
    from turbotab.server.auth import AuthConfigError, describe

    try:
        app = create_app(settings)
    except AuthConfigError as exc:
        parser.exit(2, f"TurboTab did not start: {exc}\n")
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{args.port}/"
    print(f"TurboTab ({settings.mode} mode, {settings.workers} workers, home {settings.home})", flush=True)
    if settings.mode == "server":
        print(describe(app.state.auth), flush=True)
    print(f"Open {url}", flush=True)
    if args.open:
        probe = "127.0.0.1" if host == "0.0.0.0" else host
        threading.Thread(target=_open_when_ready, args=(probe, args.port, url), daemon=True).start()
    # Server mode reads X-Forwarded-For and X-Forwarded-Proto itself, and only from
    # TURBOTAB_TRUSTED_PROXIES (turbotab.server.auth): uvicorn's own reading would replace the
    # peer address that the proxy check rests on.
    extra = {"proxy_headers": False} if settings.mode == "server" else {}
    config = uvicorn.Config(app, host=host, port=args.port, log_level="info",
                            timeout_graceful_shutdown=2, **extra)
    server = uvicorn.Server(config)
    if args.stop_on_eof:
        def stop() -> None:
            server.should_exit = True  # the same graceful stop as Ctrl+C

        stop_on_eof(stop)
    server.run()
    return 0


def stop_on_eof(stop: Callable[[], None]) -> threading.Thread | None:
    """Call ``stop`` once standard input closes, reading it on a thread of its own.

    The pipe is moved off descriptor 0 before the read begins, and descriptor 0 (on Windows also
    the process's standard input handle) becomes the null device, so no process started from now
    on shares the pipe. On Windows that is what lets a job worker start at all: a new process is
    handed this one's standard handles, a read pending on a synchronous pipe blocks every other
    use of that pipe, and a worker holding it hung as it started, before any Python ran, until the
    read ended (that is, until TurboTab stopped). Returns the thread, or None when there is no
    standard input to watch.
    """
    try:
        private = os.dup(0)  # not inheritable
    except OSError:
        print("TurboTab: --stop-on-eof has no standard input to watch; stop it with Ctrl+C.",
              file=sys.stderr, flush=True)
        return None
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, 0)
    finally:
        os.close(null)

    def watch() -> None:
        try:
            while os.read(private, 65536):
                pass
        except OSError:
            pass  # a broken pipe is a closed one
        finally:
            os.close(private)
        stop()

    thread = threading.Thread(target=watch, name="turbotab-stop-on-eof", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    raise SystemExit(main())

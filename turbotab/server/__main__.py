"""Launch TurboTab: ``venv/bin/python -m turbotab.server --port 8787 [--mode local|server] [--open]``.

Local mode binds 127.0.0.1 only. Server mode listens where ``--host`` says
(default 127.0.0.1, so exposing it is a deliberate choice) and signs users in
(``turbotab.server.auth``; accounts: ``python -m turbotab.server.users``).
"""
from __future__ import annotations

import argparse
import dataclasses
import socket
import sys
import threading
import time
import webbrowser

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
        def stop_when_stdin_closes() -> None:
            try:
                sys.stdin.buffer.read()
            except (OSError, ValueError):
                pass
            server.should_exit = True  # the same graceful stop as Ctrl+C

        threading.Thread(target=stop_when_stdin_closes, daemon=True).start()
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

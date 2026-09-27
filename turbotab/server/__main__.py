"""Launch TurboTab: ``venv/bin/python -m turbotab.server --port 8787 [--mode local|server] [--open]``.

Local mode binds 127.0.0.1 only. Server mode listens where ``--host`` says
(default 127.0.0.1, so exposing it is a deliberate choice).
"""
from __future__ import annotations

import argparse
import dataclasses
import socket
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
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.mode:
        settings = dataclasses.replace(settings, mode=args.mode)
    host = args.host or "127.0.0.1"
    if settings.mode == "local" and host != "127.0.0.1":
        parser.error("local mode binds 127.0.0.1 only; pass --mode server to listen elsewhere")

    import uvicorn

    from turbotab.server.app import create_app

    app = create_app(settings)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{args.port}/"
    print(f"TurboTab ({settings.mode} mode, {settings.workers} workers, home {settings.home})", flush=True)
    print(f"Open {url}", flush=True)
    if args.open:
        probe = "127.0.0.1" if host == "0.0.0.0" else host
        threading.Thread(target=_open_when_ready, args=(probe, args.port, url), daemon=True).start()
    uvicorn.run(app, host=host, port=args.port, log_level="info", timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

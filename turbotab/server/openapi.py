"""Print the API's OpenAPI document, or write it to ``turbotab/server/openapi.json``.

    venv/bin/python -m turbotab.server.openapi            # print
    venv/bin/python -m turbotab.server.openapi --write    # update the committed copy

The committed copy is what the frontend's types are generated from
(``npm run gen:api``); a test fails when it drifts from the app.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from turbotab.core.config import Settings

OPENAPI_JSON = Path(__file__).resolve().parent / "openapi.json"


def build() -> dict[str, Any]:
    from turbotab.server.app import create_app

    # The app is only built, never started: nothing is read from or written to this home.
    settings = Settings(
        home=Path(tempfile.gettempdir()) / "turbotab-openapi",
        mode="local",
        workers=1,
        memory_budget_bytes=1 << 30,
    )
    return create_app(settings).openapi()


def render() -> str:
    return json.dumps(build(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m turbotab.server.openapi", description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"write {OPENAPI_JSON.name} instead of printing")
    args = parser.parse_args(argv)
    text = render()
    if args.write:
        OPENAPI_JSON.write_text(text, encoding="utf-8")
        print(f"wrote {OPENAPI_JSON}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

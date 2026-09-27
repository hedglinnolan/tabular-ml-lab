"""The FastAPI app (docs/turbotab-next/BLUEPRINT.md §6): HTTP and SSE only.

``create_app(settings)`` builds the app without touching the disk or starting
anything; the workspace, the job workers and the engine start in the lifespan,
so ``python -m turbotab.server.openapi`` can build the app just to read its
schema.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic.json_schema import models_json_schema
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from turbotab.core.config import Settings
from turbotab.server import __version__, errors
from turbotab.server.errors import ApiError
from turbotab.server.routes import build_router
from turbotab.server.schemas import ARTIFACT_MODELS

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})

NOT_BUILT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TurboTab</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:36rem;margin:4rem auto;padding:0 1rem;color:#1d1d1f}
code{font:14px ui-monospace,monospace;background:#f2f2f4;padding:.1rem .3rem;border-radius:4px}
@media (prefers-color-scheme:dark){body{background:#151517;color:#e8e8ea}code{background:#26262a}}</style>
</head><body>
<h1>TurboTab is running</h1>
<p>The interface has not been built yet. Build it once, then reload this page:</p>
<p><code>cd turbotab/frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>
<p>The API is already answering at <a href="/api/health"><code>/api/health</code></a>.</p>
</body></html>
"""


class LocalHostGuard:
    """Refuse any request whose ``Host`` is not localhost / 127.0.0.1.

    In local mode the server binds 127.0.0.1, but a web page elsewhere can still
    point a hostname it controls at 127.0.0.1 (DNS rebinding) and read the
    answers. Its requests carry that hostname, so they stop here.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and request_host(scope) not in LOCAL_HOSTS:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            body = ApiError(
                403,
                "host_not_allowed",
                "This TurboTab answers only at localhost or 127.0.0.1.",
            ).body()
            await JSONResponse(body, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def request_host(scope: Scope) -> str:
    for name, value in scope.get("headers") or ():
        if name == b"host":
            host = value.decode("latin-1").strip().lower()
            if host.startswith("["):  # [::1]:8787
                return host[: host.find("]") + 1]
            return host.rsplit(":", 1)[0] if ":" in host else host
    return ""


def install_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built frontend at ``/``, with every unknown path answered by index.html."""

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, "not_found", f"There is no API route /{path}.")
        index = dist / "index.html"
        if not index.is_file():
            return HTMLResponse(NOT_BUILT, headers={"Cache-Control": "no-cache"})
        if path:
            root = dist.resolve()
            candidate = (root / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app(settings: Settings, *, frontend_dist: Path = FRONTEND_DIST) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from turbotab.server.service import ProjectService

        service = await run_in_threadpool(ProjectService, settings)
        app.state.service = service
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    app = FastAPI(
        title="TurboTab",
        version=__version__,
        description="TurboTab Next: the HTTP and SSE contract between the engine and the frontend.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    errors.install(app)
    app.include_router(build_router())
    install_frontend(app, frontend_dist)
    if settings.mode == "local":
        app.add_middleware(LocalHostGuard)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
            )
            # Stage artifacts are typed by stage name on the client; publish their shapes.
            _, defs = models_json_schema(
                [(model, "serialization") for model in ARTIFACT_MODELS.values()],
                ref_template="#/components/schemas/{model}",
            )
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            for name, definition in defs.get("$defs", {}).items():
                components.setdefault(name, definition)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app

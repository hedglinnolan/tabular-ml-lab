"""The FastAPI app (docs/turbotab-next/BLUEPRINT.md §6): HTTP and SSE only.

``create_app(settings)`` builds the app without touching the disk or starting
anything; the workspace, the job workers and the engine start in the lifespan,
so ``python -m turbotab.server.openapi`` can build the app just to read its
schema.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator
from urllib.parse import urlsplit

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

if TYPE_CHECKING:  # pragma: no cover
    from turbotab.server.auth import AuthConfig

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
    """Refuse what a web page elsewhere could make a TurboTab on this machine do.

    * Any request whose ``Host`` is not localhost / 127.0.0.1. The server binds
      127.0.0.1, but a page can point a hostname it controls at 127.0.0.1 (DNS
      rebinding) and read the answers; its requests carry that hostname.
    * Any request that changes something (every method but GET, HEAD and
      OPTIONS) sent by a page on another site. A form post or a multipart upload
      is a CORS "simple" request, sent without asking and with Host 127.0.0.1,
      so the Host check lets it through. The browser names where it came from:
      ``Origin`` must be localhost / 127.0.0.1 (any port, so the Vite dev server
      works), and ``Sec-Fetch-Site`` must not be ``cross-site``. Clients that
      send neither (curl, scripts) are this machine's own programs.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and request_host(scope) not in LOCAL_HOSTS:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            message = "This TurboTab answers only at localhost or 127.0.0.1."
            await refuse(scope, receive, send, "host_not_allowed", message)
            return
        if scope["type"] == "http" and scope["method"] not in SAFE_METHODS and cross_site(scope):
            await refuse(
                scope,
                receive,
                send,
                "cross_site",
                "This request came from a page on another website. TurboTab takes changes "
                "only from its own pages; open it at localhost or 127.0.0.1 and do it there.",
            )
            return
        await self.app(scope, receive, send)


SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


async def refuse(scope: Scope, receive: Receive, send: Send, code: str, message: str) -> None:
    body = ApiError(403, code, message).body()
    await JSONResponse(body, status_code=403)(scope, receive, send)


def header(scope: Scope, wanted: bytes) -> str | None:
    for name, value in scope.get("headers") or ():
        if name == wanted:
            return value.decode("latin-1").strip()
    return None


def request_host(scope: Scope) -> str:
    host = (header(scope, b"host") or "").lower()
    if host.startswith("["):  # [::1]:8787
        return host[: host.find("]") + 1]
    return host.rsplit(":", 1)[0] if ":" in host else host


def cross_site(scope: Scope) -> bool:
    """True when the browser says the request came from a page on another site."""
    if (header(scope, b"sec-fetch-site") or "").lower() == "cross-site":
        return True
    origin = header(scope, b"origin")
    if origin is None:
        return False
    try:
        parts = urlsplit(origin)
        hostname = parts.hostname
    except ValueError:
        return True
    return parts.scheme not in ("http", "https") or hostname not in LOCAL_HOSTS  # "null" too


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


def create_app(settings: Settings, *, frontend_dist: Path = FRONTEND_DIST,
               auth: "AuthConfig | None" = None) -> FastAPI:
    """Local mode: one workspace, no sign-in, the localhost guards. Server mode: every request
    from a signed-in user (``auth``, else ``AuthConfig.from_env``; a setting that would leave the
    server open raises ``AuthConfigError`` here, before anything starts), each user in a
    workspace of their own (``turbotab.server.tenancy``)."""
    from turbotab.server import auth as signin

    auth_config = None
    if settings.mode == "server":
        auth_config = auth if auth is not None else signin.AuthConfig.from_env(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.mode == "server":
            from turbotab.server.tenancy import Tenants

            tenants = await run_in_threadpool(Tenants, settings)
            app.state.tenants = tenants
            try:
                yield
            finally:
                await run_in_threadpool(tenants.close)
            return
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
    if auth_config is not None:
        signin.install(app, settings, auth_config)  # before the frontend's catch-all route
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

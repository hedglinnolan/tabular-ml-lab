"""One error shape for everything the API refuses: ``{error: {code, message, exits}}``.

A refused decision is HTTP 409 (BLUEPRINT §3). The same envelope carries the
server's other refusals (403 in the wrong mode, 404 for an unknown project, …)
so the frontend renders every one of them the same way. Request bodies that do
not parse are FastAPI's own 422.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from turbotab.core import decisions
from turbotab.core.datastore import UnknownColumn
from turbotab.core.workspace import ProjectNotFound


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        exits: Iterable[Mapping[str, Any]] = (),
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.exits = [{"label": str(e["label"]), "decision": e.get("decision")} for e in exits]

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "exits": self.exits}}


def not_found(pid: str) -> ApiError:
    return ApiError(404, "unknown_project", f"There is no project {pid!r}.")


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(decisions.Refusal)
    async def _refusal(_: Request, exc: decisions.Refusal) -> JSONResponse:
        return JSONResponse(exc.to_dict(), status_code=409)

    @app.exception_handler(ProjectNotFound)
    async def _no_project(_: Request, exc: ProjectNotFound) -> JSONResponse:
        return JSONResponse(not_found(exc.pid).body(), status_code=404)

    @app.exception_handler(UnknownColumn)
    async def _no_column(_: Request, exc: UnknownColumn) -> JSONResponse:
        body = ApiError(404, "unknown_column", f"There is no column named {exc.column!r} in this dataset.")
        return JSONResponse(body.body(), status_code=404)


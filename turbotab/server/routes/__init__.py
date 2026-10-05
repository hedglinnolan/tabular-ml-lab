"""The API routes, one module per family, all mounted under ``/api``."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from turbotab.core.config import Settings
from turbotab.server.errors import ApiError
from turbotab.server.schemas import Refusal
from turbotab.server.service import ProjectService


def get_service(request: Request) -> ProjectService:
    return request.app.state.service


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_local(settings: Settings, what: str, exits: list[dict[str, Any]] | None = None) -> None:
    if settings.mode != "local":
        raise ApiError(
            403,
            "local_only",
            f"{what} works only when TurboTab runs on your own machine.",
            exits or [],
        )


def refusal(description: str) -> dict[str, Any]:
    return {"model": Refusal, "description": description}


def build_router() -> APIRouter:
    from turbotab.server.routes import (
        assembly,
        data,
        events,
        export,
        jobs,
        models,
        preview,
        projects,
        system,
        teaching,
    )

    api = APIRouter(prefix="/api")
    for module in (system, projects, preview, data, jobs, events, models, teaching, assembly,
                   export):
        api.include_router(module.router)
    return api

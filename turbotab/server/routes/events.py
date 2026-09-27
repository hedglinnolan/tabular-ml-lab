"""Server-sent events for one project (BLUEPRINT §5).

    event: decision   data: DecisionRecord
    event: stage      data: StageStatus
    event: job        data: JobView
    event: resync     data: {}   (refetch everything; always the first event)
    event: ping       data: {}   (after 15 s without another event)
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from turbotab.core.events import EventBus
from turbotab.server.routes import get_service, refusal

router = APIRouter(tags=["events"])

HEARTBEAT_SECONDS = 15.0


def frame(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'), allow_nan=False)}\n\n"


async def stream(bus: EventBus, pid: str) -> AsyncIterator[str]:
    events = bus.subscribe(pid, heartbeat=HEARTBEAT_SECONDS, resync_first=True)
    try:
        async for event_type, data in events:
            yield frame(event_type, data)
    finally:
        await events.aclose()


@router.get(
    "/projects/{pid}/events",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}, "description": "An SSE stream"},
        404: refusal("No such project"),
    },
)
async def events(request: Request, pid: str) -> StreamingResponse:
    service = get_service(request)
    await run_in_threadpool(service.workspace.get, pid)  # 404 before the stream opens
    return StreamingResponse(
        stream(service.bus, pid),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

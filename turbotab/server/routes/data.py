"""UI reads: row windows, column summaries, histograms. Each is a query over Parquet."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from turbotab.core.datastore import MAX_HISTOGRAM_BINS, MAX_WINDOW_ROWS, UnknownColumn
from turbotab.server.errors import ApiError
from turbotab.server.routes import get_service, refusal
from turbotab.server.schemas import ColumnSummary, Histogram, TableWindow

router = APIRouter(tags=["data"])

NOT_READY = refusal("The table has not been read yet (the ingest stage is not fresh)")
MAX_COMMAS_IN_A_NAME = 8


def parse_columns(raw: str | None, known: list[str]) -> list[str] | None:
    """``a,b,c`` -> names. A name that itself holds commas is matched whole."""
    if raw is None or raw == "":
        return None
    names = set(known)
    if raw in names:
        return [raw]
    parts = raw.split(",")
    out: list[str] = []
    i = 0
    while i < len(parts):
        for j in range(i + 1, min(len(parts), i + MAX_COMMAS_IN_A_NAME) + 1):
            name = ",".join(parts[i:j])
            if name in names:
                out.append(name)
                i = j
                break
        else:
            raise UnknownColumn(parts[i])
    return out


@router.get(
    "/projects/{pid}/table",
    response_model=TableWindow,
    responses={404: refusal("No such project or column"), 409: NOT_READY},
)
def table(
    request: Request,
    pid: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=0, le=MAX_WINDOW_ROWS),
    columns: str | None = Query(None, description="Comma-separated column names; default: all"),
) -> dict:
    """Rows ``[offset, offset + limit)`` in file order."""
    store = get_service(request).store(pid)
    return store.window(offset, limit, parse_columns(columns, store.columns))


@router.get(
    "/projects/{pid}/columns",
    response_model=list[ColumnSummary],
    responses={404: refusal("No such project"), 409: NOT_READY},
)
def columns(request: Request, pid: str) -> list[dict]:
    return get_service(request).store(pid).summaries()


@router.get(
    "/projects/{pid}/columns/{name:path}/histogram",
    response_model=Histogram,
    responses={
        400: refusal("Not a numeric column"),
        404: refusal("No such project or column"),
        409: NOT_READY,
    },
)
def histogram(
    request: Request, pid: str, name: str, bins: int = Query(30, ge=1, le=MAX_HISTOGRAM_BINS)
) -> dict:
    store = get_service(request).store(pid)
    try:
        return store.histogram(name, bins)
    except UnknownColumn:
        raise
    except ValueError as exc:
        raise ApiError(400, "not_numeric", str(exc)) from None

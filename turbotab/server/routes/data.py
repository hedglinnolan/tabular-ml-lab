"""UI reads: row windows, column summaries, histograms. Each is a query over Parquet."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

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
    """Rows ``[offset, offset + limit)`` in file order, of the ``columns`` asked for (the ones a
    grid shows): only those are read, so a window of a 20,000-column table costs what its
    visible columns cost. Once the outcome is chosen, a window leaves it out until the outcome
    beside a column opens, and says so in ``withheld`` (CROSSWALK disagreement 2)."""
    service = get_service(request)
    store = service.store(pid)
    return service.table_window(pid, offset, limit, parse_columns(columns, store.columns))


def matching(names: list[str], query: str | None) -> list[str]:
    """Columns whose name holds every word of ``query`` (any case), in table order."""
    words = (query or "").lower().split()
    if not words:
        return names
    return [n for n in names if all(w in n.lower() for w in words)]


@router.get(
    "/projects/{pid}/columns",
    response_model=list[ColumnSummary],
    responses={404: refusal("No such project or column"), 409: NOT_READY},
)
def columns(
    request: Request,
    response: Response,
    pid: str,
    query: str | None = Query(None, description="Words every returned column name holds, any case"),
    names: str | None = Query(None, description="Comma-separated column names; default: all"),
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=0, description="At most this many; default: every match"),
) -> list[dict]:
    """Column summaries in table order. A wide table's roles list searches with ``query`` and
    pages with ``offset``/``limit``; ``X-Total-Count`` is the number of matches before paging."""
    service = get_service(request)
    store = service.store(pid)
    if query is None and names is None and offset == 0 and limit is None:
        return service.column_summaries(pid, store, store.summaries())
    chosen = parse_columns(names, store.columns) if names else store.columns
    found = matching(chosen, query)
    response.headers["X-Total-Count"] = str(len(found))
    page = found[offset:] if limit is None else found[offset:offset + limit]
    return service.column_summaries(pid, store, store.summaries(page)) if page else []


@router.get(
    "/projects/{pid}/columns/{name:path}/histogram",
    response_model=Histogram,
    responses={
        400: refusal("Not a numeric column"),
        404: refusal("No such project or column"),
        409: refusal("The table is not read yet, or the outcome's distribution is not open yet"),
    },
)
def histogram(
    request: Request, pid: str, name: str, bins: int = Query(30, ge=1, le=MAX_HISTOGRAM_BINS)
) -> dict:
    """The column's histogram. The outcome's is the outcome alone: refused with its line until
    its gate opens, then drawn on the rows its view reads (``turbotab.core.outcome_gate``)."""
    try:
        return get_service(request).column_histogram(pid, name, bins)
    except UnknownColumn:
        raise
    except ValueError as exc:
        raise ApiError(400, "not_numeric", str(exc)) from None

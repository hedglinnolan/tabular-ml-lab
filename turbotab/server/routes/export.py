"""The export: the manuscript bundle and the reporting checklist (V2 definition of done §1, §3.6;
``turbotab/core/export``)."""
from __future__ import annotations

import re

from fastapi import APIRouter, Request, Response

from turbotab.core.export.checklists import ChecklistReport
from turbotab.server.routes import get_service, refusal

router = APIRouter(tags=["export"])


def _filename(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "project"
    return f"{slug[:60]}-turbotab-export.zip"


@router.get(
    "/projects/{pid}/export",
    response_class=Response,
    responses={200: {"content": {"application/zip": {"schema": {"type": "string",
                                                                "format": "binary"}}},
                     "description": "The bundle, a zip"},
               404: refusal("No such project"),
               409: refusal("Not ready to export: names everything missing, with a way forward "
                            "for each")},
)
def export(request: Request, pid: str) -> Response:
    """The manuscript bundle: the methods section from the record (ordered by STROBE or
    TRIPOD+AI), the participant-flow and lineage figures as journal-format SVG, the results tables
    as CSV and Markdown, the filled checklist, the analysis plan with its SHA-256 and the
    provenance record a replay checks (``python -m turbotab.replay``). Refused while a required
    question is unanswered, the plan is open, an input file changed or a result is not computed."""
    service = get_service(request)
    data = service.export(pid)
    name = service.workspace.get(pid).name
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{_filename(name)}"'})


@router.get(
    "/projects/{pid}/checklist",
    response_model=ChecklistReport,
    responses={404: refusal("No such project")},
)
def checklist(request: Request, pid: str) -> ChecklistReport:
    """The reporting checklist of the declared purpose (STROBE-nut under inference, TRIPOD+AI
    under prediction): every item quoted from its source, with where the record answers it or
    "unanswered — the author must supply this", and what the export still waits for. Reading it
    records nothing, so it shows no score that was not already shown: under prediction the
    declared result is quoted only once the fit's cross-validated scores have been shown."""
    return get_service(request).checklist(pid)

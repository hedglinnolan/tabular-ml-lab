"""Files to join and codebooks to import (DATAIN, V2 definition of done §1).

A file added to a project is read once, as the table was, and joined to the table by a
``join_files`` decision after its preview (``POST …/join-preview``) has shown the row counts. A
codebook is read and kept in the project, and its preview says what it would settle and ask; an
``import_codebook`` decision records it.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from turbotab.core.codebook import check_suffix
from turbotab.server.routes import get_service, get_settings, refusal, require_local
from turbotab.server.schemas import (
    AddedFile,
    AddFile,
    CodebookPreview,
    CodebookRequest,
    JoinPreview,
    JoinPreviewRequest,
)
from turbotab.server.uploads import FIELD, receive_upload

router = APIRouter(tags=["assembly"])

_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": [FIELD],
                    "properties": {FIELD: {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


@router.get("/projects/{pid}/files", response_model=list[AddedFile],
            responses={404: refusal("No such project")})
def list_files(request: Request, pid: str) -> list[dict]:
    """The files added to the project to join to its table, and whether each is joined."""
    return get_service(request).files(pid)


@router.post("/projects/{pid}/files", response_model=AddedFile,
             responses={400: refusal("Not a readable data file"),
                        403: refusal("Server mode: add files by uploading them"),
                        404: refusal("No file at that path")})
def add_file(request: Request, pid: str, body: AddFile) -> dict:
    """Add a file on this machine (local mode only); it is read once, as the table was."""
    require_local(get_settings(request), "Adding a file by its path",
                  [{"label": "Upload the file instead", "decision": None}])
    return get_service(request).add_file_from_path(pid, body.path)


@router.post("/projects/{pid}/files/upload", response_model=AddedFile,
             responses={400: refusal("No file, or not a readable data file")},
             openapi_extra=_UPLOAD_BODY)
async def upload_file(request: Request, pid: str) -> dict:
    """Upload a file to join to the project's table (multipart field ``file``)."""
    service = get_service(request)
    await run_in_threadpool(service.workspace.get, pid)
    staged = await receive_upload(request, service.workspace.uploads_dir())
    return await run_in_threadpool(service.add_file_from_upload, pid, staged.path,
                                   staged.client_name)


@router.post("/projects/{pid}/join-preview", response_model=JoinPreview,
             responses={404: refusal("No such project or file"),
                        409: refusal("The table is still being read")})
def join_preview(request: Request, pid: str, body: JoinPreviewRequest) -> dict:
    """The row counts a join would give, before it is committed (``join_files`` commits it)."""
    return get_service(request).join_preview(pid, body.file, body.on, body.right_on, body.how)


@router.post("/projects/{pid}/codebooks", response_model=CodebookPreview,
             responses={400: refusal("Not a codebook TurboTab reads"),
                        403: refusal("Server mode: upload the codebook"),
                        409: refusal("The table is still being read")})
def add_codebook(request: Request, pid: str, body: CodebookRequest) -> dict:
    """Read a codebook on this machine (local mode only), or the table's own XPT labels, and say
    what importing it would do (``import_codebook`` records it)."""
    if body.path:
        require_local(get_settings(request), "Reading a codebook by its path",
                      [{"label": "Upload the codebook instead", "decision": None}])
    return get_service(request).stage_codebook(pid, path=body.path, labels=body.labels)


@router.post("/projects/{pid}/codebooks/upload", response_model=CodebookPreview,
             responses={400: refusal("No file, or not a codebook TurboTab reads"),
                        409: refusal("The table is still being read")},
             openapi_extra=_UPLOAD_BODY)
async def upload_codebook(request: Request, pid: str) -> dict:
    """Upload a codebook (multipart field ``file``): a variable table, an NHANES codebook page or
    an XPT file; the answer says what importing it would do."""
    service = get_service(request)
    await run_in_threadpool(service.workspace.get, pid)
    staged = await receive_upload(request, service.workspace.uploads_dir(), accept=check_suffix)
    return await run_in_threadpool(lambda: service.stage_codebook(
        pid, staged=staged.path, client_name=staged.client_name))

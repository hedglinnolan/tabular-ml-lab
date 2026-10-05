"""``/health`` and the local file browser (``/fs/list``)."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Query, Request

from turbotab.server import __version__
from turbotab.server.errors import ApiError
from turbotab.server.routes import get_service, get_settings, refusal, require_local
from turbotab.server.schemas import FsEntry, FsListing, Health

router = APIRouter(tags=["system"])


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """The build, the mode, and in server mode who is signed in and how (a sign-out control only
    where signing out means something: under ``password``, not behind the institution's proxy)."""
    settings = get_settings(request)
    auth = getattr(request.app.state, "auth", None)
    return Health(version=__version__, mode=settings.mode, workers=get_service(request).runner.workers,
                  user=request.scope.get("turbotab.user"),
                  auth=auth.config.mode if auth is not None else "none")


@router.get(
    "/fs/list",
    response_model=FsListing,
    responses={403: refusal("Server mode, or a folder TurboTab may not read"), 404: refusal("No such folder")},
)
def fs_list(request: Request, path: str | None = Query(None, description="A folder; default: your home folder")) -> FsListing:
    """The entries of one folder on this machine (local mode only). Hidden entries are left out."""
    require_local(get_settings(request), "Browsing this computer's files")
    folder = Path(path).expanduser() if path else Path.home()
    if not folder.is_absolute():
        raise ApiError(400, "relative_path", "Give the folder's full path, starting from the top of the disk.")
    folder = folder.resolve()
    if not folder.exists():
        raise ApiError(404, "no_such_folder", f"There is no folder at {folder}.")
    if not folder.is_dir():
        raise ApiError(400, "not_a_folder", f"{folder} is a file, not a folder.")
    entries: list[FsEntry] = []
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                    size = None if is_dir else entry.stat().st_size
                except OSError:
                    continue  # a broken link or a file that vanished mid-listing
                entries.append(FsEntry(name=entry.name, path=os.path.join(folder, entry.name), is_dir=is_dir, size=size))
    except PermissionError:
        raise ApiError(403, "unreadable_folder", f"TurboTab is not allowed to read {folder}.") from None
    entries.sort(key=lambda e: (not e.is_dir, e.name.casefold()))
    parent = None if folder.parent == folder else str(folder.parent)
    return FsListing(path=str(folder), parent=parent, entries=entries)

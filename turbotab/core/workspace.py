"""The on-disk workspace (BLUEPRINT §2).

    $TURBOTAB_HOME/
      uploads/                     staging for streamed uploads
      projects/<pid>/
        project.json               ProjectMeta
        decisions.jsonl            append-only decision log
        data/raw.parquet           the ingested table (+ raw.info.json sidecar)
        cache/<stage>/<key>/       disposable stage artifacts

Project ids are ``"p"`` + 10 lowercase hex characters. Every accessor
validates the id against that shape before touching the filesystem, so an id
taken from a URL can never name a path outside ``projects/``.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from turbotab.core.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from turbotab.core.datastore import DataStore

SourceKind = Literal["path", "upload"]
SOURCE_KINDS: tuple[str, ...] = ("path", "upload")
PROJECT_ID_RE = re.compile(r"^p[0-9a-f]{10}$")


@dataclass
class ProjectMeta:
    id: str
    name: str
    created_at: datetime
    source_kind: SourceKind
    source_path: str
    source_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectMeta":
        created = data["created_at"]
        if not isinstance(created, datetime):
            created = datetime.fromisoformat(str(created))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return cls(id=str(data["id"]), name=str(data["name"]),
                   created_at=created.astimezone(timezone.utc),
                   source_kind=data["source_kind"], source_path=str(data["source_path"]),
                   source_name=str(data["source_name"]))


class ProjectNotFound(KeyError):
    def __init__(self, pid: str):
        super().__init__(pid)
        self.pid = pid

    def __str__(self) -> str:
        return f"no project with id {self.pid!r}"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class Workspace:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.home = Path(settings.home).expanduser()
        self.projects_root = self.home / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── projects ──────────────────────────────────────────────────────────────
    def create_project(self, name: str, source_path: str | os.PathLike[str],
                       source_kind: SourceKind, *, source_name: str | None = None
                       ) -> ProjectMeta:
        """Create ``projects/<pid>/`` and its ``project.json``.

        ``source_name`` defaults to the file name of ``source_path``; an upload
        passes the client's original file name here, since the staged copy's
        name is TurboTab's own.
        """
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind must be 'path' or 'upload', not {source_kind!r}")
        src = Path(source_path).expanduser()
        src_name = source_name or src.name
        with self._lock:
            while True:
                pid = "p" + secrets.token_hex(5)
                pdir = self.projects_root / pid
                try:
                    pdir.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    continue
        (pdir / "data").mkdir(exist_ok=True)
        (pdir / "cache").mkdir(exist_ok=True)
        meta = ProjectMeta(
            id=pid,
            name=(name or "").strip() or Path(src_name).stem or pid,
            created_at=datetime.now(timezone.utc),
            source_kind=source_kind,
            source_path=str(src.absolute()),
            source_name=src_name,
        )
        _write_json_atomic(pdir / "project.json", meta.to_dict())
        return meta

    def get(self, pid: str) -> ProjectMeta:
        path = self.project_dir(pid) / "project.json"
        try:
            with open(path, encoding="utf-8") as fh:
                return ProjectMeta.from_dict(json.load(fh))
        except FileNotFoundError:
            raise ProjectNotFound(pid) from None

    def list(self) -> list[ProjectMeta]:
        """Every readable project, newest first."""
        metas: list[ProjectMeta] = []
        for pdir in self.projects_root.iterdir():
            if not PROJECT_ID_RE.fullmatch(pdir.name) or not pdir.is_dir():
                continue
            try:
                with open(pdir / "project.json", encoding="utf-8") as fh:
                    metas.append(ProjectMeta.from_dict(json.load(fh)))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # half-created or damaged project: not listable
        metas.sort(key=lambda m: (m.created_at, m.id), reverse=True)
        return metas

    def exists(self, pid: str) -> bool:
        try:
            self.project_dir(pid)
        except ProjectNotFound:
            return False
        return True

    # ── paths ─────────────────────────────────────────────────────────────────
    def project_dir(self, pid: str) -> Path:
        if not isinstance(pid, str) or not PROJECT_ID_RE.fullmatch(pid):
            raise ProjectNotFound(str(pid))
        pdir = self.projects_root / pid
        if not pdir.is_dir():
            raise ProjectNotFound(pid)
        return pdir

    def data_path(self, pid: str) -> Path:
        return self.project_dir(pid) / "data" / "raw.parquet"

    def cache_dir(self, pid: str) -> Path:
        path = self.project_dir(pid) / "cache"
        path.mkdir(exist_ok=True)
        return path

    def decisions_path(self, pid: str) -> Path:
        return self.project_dir(pid) / "decisions.jsonl"

    def uploads_dir(self) -> Path:
        path = self.home / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── convenience ───────────────────────────────────────────────────────────
    def datastore(self, pid: str) -> "DataStore":
        """A DataStore over the project's ingested table, under this machine's budget."""
        from turbotab.core.datastore import DataStore
        return DataStore(self.data_path(pid), self.settings.memory_budget_bytes)

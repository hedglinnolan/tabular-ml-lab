"""The on-disk workspace (BLUEPRINT §2).

    $TURBOTAB_HOME/
      uploads/                     staging for streamed uploads
      projects/<pid>/
        project.json               ProjectMeta
        decisions.jsonl            append-only decision log
        data/raw.parquet           the ingested table (+ raw.info.json sidecar; raw.labels.json
                                   when an XPT source carries variable labels)
        data/joined-<digest>.parquet  the table with a set of files joined to it, one file per
                                   set (``datastore.table_file``; the same sidecars beside it)
        files/<fid>/               a file added to join to the table: its source, its ingested
                                   raw.parquet and file.json (DATAIN, minimal assembly)
        codebooks/<cid>/           an imported codebook: codebook.json and its source file
        cache/<stage>/<key>/       disposable stage artifacts
        claims_rendered.jsonl      each claim rendered for the project (the claims render log,
                                   ``claims.render_log``), written in every mode
        previews.jsonl             each preview asked for, in analyst mode (SWARM_ENGINE_PLAN E4)

Project ids are ``"p"`` + 10 lowercase hex characters. Every accessor
validates the id against that shape before touching the filesystem, so an id
taken from a URL can never name a path outside ``projects/``.
"""
from __future__ import annotations

import functools
import json
import os
import re
import secrets
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
from urllib.parse import urlsplit

from turbotab.core.config import Settings
from turbotab.core import atomic

if TYPE_CHECKING:  # pragma: no cover
    from turbotab.core.datastore import DataStore
    from turbotab.core.graph import Graph, StageContext

SourceKind = Literal["path", "upload"]
SOURCE_KINDS: tuple[str, ...] = ("path", "upload")
PROJECT_ID_RE = re.compile(r"^p[0-9a-f]{10}$")
CLAIMS_LOG = "claims_rendered.jsonl"
PREVIEWS_LOG = "previews.jsonl"
# Where a file came from (SWARM_ENGINE_PLAN E5), as given with it: each optional, kept only when
# given, in this order.
PROVENANCE_FIELDS: tuple[str, ...] = ("url", "release_date", "license")
_PROVENANCE_WORDS = {"url": "URL", "release_date": "release date", "license": "license"}
PROVENANCE_MAX_CHARS = 2000
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def source_provenance(url: str | None = None, release_date: str | None = None,
                      license: str | None = None) -> dict[str, str]:
    """The file's URL, release date (``YYYY-MM-DD``) and license, as given, keeping only those
    given (a blank is not given); raises ValueError, in words a client shows, for one that does
    not read: a URL without its scheme and host, a date that is not a day on the calendar written
    ``YYYY-MM-DD``, or a field over :data:`PROVENANCE_MAX_CHARS` characters."""
    given = {"url": url, "release_date": release_date, "license": license}
    out: dict[str, str] = {}
    for name in PROVENANCE_FIELDS:
        value = (given[name] or "").strip()
        if not value:
            continue
        if len(value) > PROVENANCE_MAX_CHARS:
            raise ValueError(f"The file's {_PROVENANCE_WORDS[name]} holds at most "
                             f"{PROVENANCE_MAX_CHARS:,} characters; this one has {len(value):,}.")
        if name == "url":
            parts = urlsplit(value)
            if not (parts.scheme and parts.netloc):
                raise ValueError(f"Give the file's address in full, starting with its scheme "
                                 f"(https://), not {value!r}.")
        if name == "release_date":
            try:
                if not _ISO_DATE.match(value):
                    raise ValueError
                date.fromisoformat(value)
            except ValueError:
                raise ValueError(f"Write the release date as YYYY-MM-DD (a day on the calendar), "
                                 f"not {value!r}.") from None
        out[name] = value
    return out


@dataclass
class ProjectMeta:
    id: str
    name: str
    created_at: datetime
    source_kind: SourceKind
    source_path: str
    source_name: str
    # Where the table's file came from, as given with it (``source_provenance``); empty when none
    # was, and then not written, so the file reads as every project's did before.
    provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_name": self.source_name,
        }
        if self.provenance:
            out["provenance"] = dict(self.provenance)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectMeta":
        created = data["created_at"]
        if not isinstance(created, datetime):
            created = datetime.fromisoformat(str(created))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        kept = data.get("provenance") or {}
        return cls(id=str(data["id"]), name=str(data["name"]),
                   created_at=created.astimezone(timezone.utc),
                   source_kind=data["source_kind"], source_path=str(data["source_path"]),
                   source_name=str(data["source_name"]),
                   provenance={k: str(kept[k]) for k in PROVENANCE_FIELDS if kept.get(k)})


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
        atomic.replace(tmp, path)
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
                       source_kind: SourceKind, *, source_name: str | None = None,
                       provenance: dict[str, str] | None = None) -> ProjectMeta:
        """Create ``projects/<pid>/`` and its ``project.json``.

        ``source_name`` defaults to the file name of ``source_path``; an upload
        passes the client's original file name here, since the staged copy's
        name is TurboTab's own. ``provenance``: where the file came from, as
        :func:`source_provenance` reads it.
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
            provenance=dict(provenance or {}),
        )
        _write_json_atomic(pdir / "project.json", meta.to_dict())
        return meta

    def save(self, meta: ProjectMeta) -> None:
        """Rewrite an existing project's ``project.json`` (atomically)."""
        _write_json_atomic(self.project_dir(meta.id) / "project.json", meta.to_dict())

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
        """The source file's own table, without any file joined to it."""
        return self.project_dir(pid) / "data" / "raw.parquet"

    def table_path(self, pid: str, state: Any = None) -> Path:
        """The table the ingest stage writes under ``state``: the source's own
        (:meth:`data_path`) or, with files joined to it, the file of that set of joins
        (``datastore.table_file``)."""
        from turbotab.core.datastore import table_file

        joins = getattr(state, "joins", None) if state is not None else None
        return table_file(self.project_dir(pid) / "data", joins)

    def cache_dir(self, pid: str) -> Path:
        path = self.project_dir(pid) / "cache"
        path.mkdir(exist_ok=True)
        return path

    def decisions_path(self, pid: str) -> Path:
        return self.project_dir(pid) / "decisions.jsonl"

    def files_dir(self, pid: str) -> Path:
        """``projects/<pid>/files/``: the files added to join to the table."""
        path = self.project_dir(pid) / "files"
        path.mkdir(exist_ok=True)
        return path

    def codebooks_dir(self, pid: str) -> Path:
        """``projects/<pid>/codebooks/``: the codebooks staged for import."""
        path = self.project_dir(pid) / "codebooks"
        path.mkdir(exist_ok=True)
        return path

    def uploads_dir(self) -> Path:
        path = self.home / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def claims_log_path(self, pid: str) -> Path:
        """``projects/<pid>/claims_rendered.jsonl``: the claims render log."""
        return self.project_dir(pid) / CLAIMS_LOG

    def previews_log_path(self, pid: str) -> Path:
        """``projects/<pid>/previews.jsonl``: each preview asked for in analyst mode."""
        return self.project_dir(pid) / PREVIEWS_LOG

    # ── convenience ───────────────────────────────────────────────────────────
    def datastore(self, pid: str) -> "DataStore":
        """A DataStore over the project's ingested table, under this machine's budget."""
        from turbotab.core.datastore import DataStore
        return DataStore(self.data_path(pid), self.settings.memory_budget_bytes)


# ── the engine's stage graph (SWARM_ENGINE_PLAN E4) ──────────────────────────
# The engine loads its graph by this name, in the server and again in each job worker
# (``graph.load_graph``): the stage graph itself, each stage run inside its project's claims
# render log, so a claim a stage renders is logged wherever the stage runs.
STAGE_GRAPH = "turbotab.core.workspace:stage_graph"


def logged_stage(fn: Callable[["StageContext"], Any]) -> Callable[["StageContext"], Any]:
    """``fn``, run inside the claims render log of the project folder its context names (no log
    where it names none)."""
    from turbotab.core.claims._registry import render_log

    @functools.wraps(fn)
    def run(ctx: "StageContext") -> Any:
        folder = (getattr(ctx, "paths", None) or {}).get("project_dir")
        with render_log(Path(folder) / CLAIMS_LOG if folder else None):
            return fn(ctx)

    return run


def stage_graph() -> "Graph":
    """``stages.build_graph``'s graph, each stage's function run by :func:`logged_stage`. Every
    other field of a stage, and so every stage key, is the stage graph's own."""
    from turbotab.core.graph import Graph
    from turbotab.core.stages import build_graph

    return Graph([replace(stage, fn=logged_stage(stage.fn)) for stage in build_graph().stages()])

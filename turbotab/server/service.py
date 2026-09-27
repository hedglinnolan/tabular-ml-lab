"""Project wiring: the workspace, a decision log per project, one engine.

The server orchestrates and computes nothing. Every number it serves comes from
the data layer (``DataStore``) or from a stage artifact.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turbotab.core import decisions
from turbotab.core.config import Settings
from turbotab.core.datastore import DataStore, fingerprint_file
from turbotab.core.datastore import _source_kind as source_kind  # the one list of readable types
from turbotab.core.decisions import DecisionLog, Refusal
from turbotab.core.events import EventBus
from turbotab.core.graph import (
    Engine,
    ProjectContext,
    StageStatus,
    latest_key,
    read_artifact,
)
from turbotab.core.jobs import PRELOAD, JobRunner, JobView
from turbotab.core.stages import GRAPH_FACTORY
from turbotab.core.workspace import ProjectMeta, Workspace
from turbotab.server.errors import ApiError

WORKER_PRELOAD = PRELOAD + (
    "duckdb",
    "pyarrow.parquet",
    "turbotab.core.datastore",
    "turbotab.core.stages",
    "turbotab.engine",
    "turbotab.packs",
)
SOURCE_FILE = "source.json"
MAX_REMEMBERED_JOBS = 10_000


# ── decision validation that needs the server's view of the project ─────────


@dataclass(frozen=True)
class DecisionContext:
    """What ``decisions.validate`` is told about a project.

    ``columns`` is None until the ingest stage is fresh: before that the
    dataset's columns are not known. ``target`` is the current target (None:
    none chosen), which a task answer must name.
    """

    columns: list[str] | None
    ingest_status: str
    ingest_error: str | None = None
    target: str | None = None


def _target_needs_columns(decision: Any, ctx: Any) -> None:
    if not isinstance(ctx, DecisionContext) or ctx.columns is not None:
        return
    if ctx.ingest_status == "error":
        raise Refusal(
            "table_unreadable",
            f"The table could not be read, so it has no columns to choose from. {ctx.ingest_error or ''}".strip(),
        )
    raise Refusal(
        "table_not_ready",
        "The table is still being read. Choose the target once its columns are listed.",
    )


decisions.register_validator("set_target", _target_needs_columns)


# ── events ───────────────────────────────────────────────────────────────────


class ServerBus(EventBus):
    """The event bus, remembering which project each job belongs to.

    The engine announces every job it submits with a ``job`` event for its
    project; that is how ``/projects/{pid}/jobs/{jid}`` knows ``jid`` is ``pid``'s.
    """

    def __init__(self) -> None:
        super().__init__()
        self._owners: OrderedDict[str, str] = OrderedDict()
        self._owners_lock = threading.Lock()

    def publish(self, pid: str, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "job" and data.get("job_id"):
            with self._owners_lock:
                self._owners[str(data["job_id"])] = pid
                while len(self._owners) > MAX_REMEMBERED_JOBS:
                    self._owners.popitem(last=False)
        super().publish(pid, event_type, data)

    def job_owner(self, job_id: str) -> str | None:
        with self._owners_lock:
            return self._owners.get(job_id)


# ── the service ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IngestFacts:
    key: str
    n_rows: int
    n_cols: int
    columns: list[str]


def _write_json_atomic(path: Path, payload: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ProjectService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.workspace = Workspace(settings)
        self.bus = ServerBus()
        self.runner = JobRunner(settings.workers, preload=WORKER_PRELOAD)
        try:
            self.engine = Engine(
                graph_factory=GRAPH_FACTORY,
                runner=self.runner,
                bus=self.bus,
                project_ctx=self._project_ctx,
            )
        except BaseException:
            self.runner.shutdown()
            raise
        self._lock = threading.Lock()
        self._logs: dict[str, DecisionLog] = {}
        self._fingerprints: dict[str, str] = {}
        self._facts: dict[str, IngestFacts] = {}
        self._stores: dict[str, tuple[str, DataStore]] = {}

    def close(self) -> None:
        self.engine.shutdown()
        self.runner.shutdown()
        with self._lock:
            stores = [store for _, store in self._stores.values()]
            self._stores.clear()
        for store in stores:
            store.close()

    # ── per-project plumbing ──

    def log(self, pid: str) -> DecisionLog:
        path = self.workspace.decisions_path(pid)  # raises ProjectNotFound
        with self._lock:
            log = self._logs.get(pid)
            if log is None:
                log = self._logs[pid] = DecisionLog(path)
            return log

    def fingerprint(self, pid: str) -> str:
        with self._lock:
            known = self._fingerprints.get(pid)
        if known:
            return known
        record = self.workspace.project_dir(pid) / SOURCE_FILE
        try:
            value = str(json.loads(record.read_text("utf-8"))["fingerprint"])
        except (OSError, ValueError, KeyError, TypeError):
            value = fingerprint_file(Path(self.workspace.get(pid).source_path))
            _write_json_atomic(record, {"fingerprint": value})
        with self._lock:
            self._fingerprints[pid] = value
        return value

    def _project_ctx(self, pid: str) -> ProjectContext:
        meta = self.workspace.get(pid)
        return ProjectContext(
            project_id=pid,
            state=self.log(pid).state(),
            cache_root=self.workspace.cache_dir(pid),
            paths={
                "source": meta.source_path,
                "data": str(self.workspace.data_path(pid)),
                "project_dir": str(self.workspace.project_dir(pid)),
            },
            settings={"memory_budget_bytes": int(self.settings.memory_budget_bytes)},
            fingerprint=self.fingerprint(pid),
        )

    def _record_source(self, pid: str, fingerprint: str) -> None:
        _write_json_atomic(self.workspace.project_dir(pid) / SOURCE_FILE, {"fingerprint": fingerprint})
        with self._lock:
            self._fingerprints[pid] = fingerprint

    # ── creating projects ──

    def create_from_path(self, raw: str) -> ProjectMeta:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ApiError(400, "relative_path", "Give the file's full path, starting from the top of the disk.")
        if path.is_dir():
            raise ApiError(400, "is_a_folder", f"{path} is a folder. Choose a file inside it.")
        if not path.is_file():
            raise ApiError(404, "no_such_file", f"There is no file at {path}.")
        path = path.resolve()
        try:
            source_kind(path)
        except ValueError as exc:
            raise ApiError(400, "unsupported_file", str(exc)) from None
        if not os.access(path, os.R_OK):
            raise ApiError(403, "unreadable_file", f"TurboTab is not allowed to read {path}.")
        fingerprint = fingerprint_file(path)
        meta = self.workspace.create_project("", path, "path")
        self._record_source(meta.id, fingerprint)
        self.engine.on_decision(meta.id)
        return meta

    def create_from_upload(self, staged: Path, client_name: str, fingerprint: str) -> ProjectMeta:
        """Adopt a file streamed into ``uploads/``: it moves into the project folder."""
        meta = self.workspace.create_project("", staged, "upload", source_name=client_name)
        folder = self.workspace.project_dir(meta.id) / "source"
        folder.mkdir(exist_ok=True)
        dest = folder / staged.name
        os.replace(staged, dest)
        shutil.rmtree(staged.parent, ignore_errors=True)  # the upload's own staging folder
        meta.source_path = str(dest)
        self.workspace.save(meta)
        self._record_source(meta.id, fingerprint)
        self.engine.on_decision(meta.id)
        return meta

    # ── reading projects ──

    def _ingest_facts(self, pid: str, key: str) -> IngestFacts:
        with self._lock:
            facts = self._facts.get(pid)
        if facts is not None and facts.key == key:
            return facts
        info = read_artifact(self.workspace.cache_dir(pid), "ingest", key)
        facts = IngestFacts(
            key=key,
            n_rows=int(info["n_rows"]),
            n_cols=int(info["n_cols"]),
            columns=[str(c["name"]) for c in info["columns"]],
        )
        with self._lock:
            self._facts[pid] = facts
        return facts

    def summary(self, meta: ProjectMeta, ingest: StageStatus | None = None) -> dict[str, Any]:
        """ProjectSummary. The size comes from the ingest artifact once it exists.

        Without ``ingest`` (the project list) nothing wakes the engine: the size
        is read from disk, and the ingest status is the engine's only if it
        already holds the project.
        """
        facts: IngestFacts | None = None
        if ingest is not None:
            if ingest.status == "fresh" and ingest.key:
                facts = self._ingest_facts(meta.id, ingest.key)
        else:
            held = self.engine.peek(meta.id)
            ingest = held.get("ingest") if held else None
            with self._lock:
                facts = self._facts.get(meta.id)
            if facts is None:
                key = latest_key(self.workspace.cache_dir(meta.id), "ingest")
                if key is not None:
                    try:
                        facts = self._ingest_facts(meta.id, key)
                    except (OSError, ValueError, KeyError):
                        facts = None
        return {
            "id": meta.id,
            "name": meta.name,
            "created_at": meta.created_at,
            "source_kind": meta.source_kind,
            "source_name": meta.source_name,
            "n_rows": facts.n_rows if facts else None,
            "n_cols": facts.n_cols if facts else None,
            "ingest": ingest,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.summary(meta) for meta in self.workspace.list()]

    def view(self, pid: str) -> dict[str, Any]:
        meta = self.workspace.get(pid)
        stages = self.engine.status(pid)
        records = self.log(pid).records()
        return {
            "summary": self.summary(meta, stages["ingest"]),
            "state": decisions.fold(records),
            "decisions": records,
            "stages": stages,
        }

    # ── deciding ──

    def decide(self, pid: str, decision: Any) -> dict[str, Any]:
        self.workspace.get(pid)
        ingest = self.engine.status(pid)["ingest"]
        columns = None
        if ingest.status == "fresh" and ingest.key:
            columns = self._ingest_facts(pid, ingest.key).columns
        ctx = DecisionContext(
            columns=columns,
            ingest_status=ingest.status,
            ingest_error=ingest.error,
            target=self.log(pid).state().target,
        )
        parsed = decisions.validate(decision, ctx)  # raises Refusal
        record = self.log(pid).append(parsed)  # raises Refusal for a revert it cannot make
        self.bus.publish(pid, "decision", record.model_dump(mode="json"))
        self.engine.on_decision(pid)
        return self.view(pid)

    # ── stages and jobs ──

    def stage_result(self, pid: str, stage: str) -> dict[str, Any]:
        self.workspace.get(pid)
        if stage not in self.engine.graph:
            raise ApiError(404, "unknown_stage", f"There is no stage named {stage!r}.")
        result = self.engine.get(pid, stage)
        artifact = result.artifact
        if artifact is not None and not isinstance(artifact, dict):
            artifact = {"value": artifact}
        return {
            "stage": result.stage,
            "key": result.key,
            "fresh": result.fresh,
            "status": result.status,
            "artifact": artifact,
        }

    def run_stage(self, pid: str, stage: str) -> StageStatus:
        """Compute ``stage`` for the current answers, retrying a failure or a cancel upstream too."""
        self.workspace.get(pid)
        if stage not in self.engine.graph:
            raise ApiError(404, "unknown_stage", f"There is no stage named {stage!r}.")
        self.engine.ensure(pid, stage)
        return self.engine.status(pid)[stage]

    def job(self, pid: str, job_id: str) -> JobView:
        self.workspace.get(pid)
        if self.bus.job_owner(job_id) != pid:
            raise ApiError(404, "unknown_job", f"Project {pid!r} has no job {job_id!r}.")
        try:
            return self.runner.get(job_id)
        except KeyError:
            raise ApiError(404, "unknown_job", f"Job {job_id!r} is no longer remembered.") from None

    def cancel_job(self, pid: str, job_id: str) -> JobView:
        self.job(pid, job_id)
        self.runner.cancel(job_id)
        return self.runner.get(job_id)

    # ── the data layer ──

    def store(self, pid: str) -> DataStore:
        """The project's DataStore; answers only once the ingest stage is fresh."""
        self.workspace.get(pid)
        ingest = self.engine.status(pid)["ingest"]
        if ingest.status != "fresh" or not ingest.key:
            if ingest.status == "error":
                raise ApiError(409, "ingest_failed", f"The table could not be read. {ingest.error or ''}".strip())
            raise ApiError(409, "table_not_ready", "The table is still being read.")
        with self._lock:
            cached = self._stores.get(pid)
            if cached is not None and cached[0] == ingest.key:
                return cached[1]
        store = DataStore(self.workspace.data_path(pid), int(self.settings.memory_budget_bytes))
        with self._lock:
            self._stores[pid] = (ingest.key, store)
        return store

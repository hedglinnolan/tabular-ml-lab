"""The hash-keyed stage graph and the engine that keeps it live
(docs/turbotab-next/BLUEPRINT.md §4).

A :class:`Stage` is a pure function of its upstream artifacts and the decision
slots it reads. Its **key** is a sha256 over its name, version, its deps' keys,
the values of the slots it reads, and (for a stage with no deps) the dataset
fingerprint. So a changed decision changes keys exactly downstream of the slot
it wrote, and nothing ever keeps a manual invalidation list.

Artifacts live at ``cache_root/<stage>/<key>/`` (``meta.json`` plus the
artifact: dict/list/model -> JSON, DataFrame -> Parquet, anything else ->
joblib). A per-stage ``latest`` pointer names the newest artifact, which is
served with ``fresh: false`` while the current key has none: stale, never
deleted.

The :class:`Engine` recomputes keys on every decision, publishes a ``stage``
event for each stage whose status changed, cancels in-flight work whose key is
no longer current, and schedules every unblocked stage whose current key lacks
an artifact once its deps are fresh. Light stages run on a thread pool; heavy
stages run as jobs in worker processes, which rebuild the graph from
``graph_factory``, read their inputs from the cache by key, write their
artifact into the cache themselves, and send back only metadata.

Status rules (per stage, for its current key):

* ``blocked`` — a slot in ``requires`` is unset here or upstream; ``missing``
  lists every such slot, transitively; ``key`` is null.
* ``fresh`` — the artifact for the current key exists.
* ``queued`` / ``running`` — work for the current key is in flight.
* ``error`` — the stage raised for this key (the message is kept), or a dep is
  in error (``"Needs 'x', which failed."``). An errored key is not retried
  automatically; :meth:`Engine.ensure` retries it.
* ``stale`` — none of the above, but an older artifact exists.
* ``idle`` — never computed.

A job cancelled by request (not superseded) is not rerun automatically either;
``ensure`` reruns it. Until then the stage (and every stage waiting on it) is
``stale`` or ``idle`` with ``cancelled: true``, so the UI can say who stopped it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import ProjectState
from turbotab.core.events import EventBus
from turbotab.core.jobs import Cancelled, JobRunner, JobView
from turbotab.core.jobs import current as current_job

log = logging.getLogger(__name__)

StatusName = Literal["idle", "queued", "running", "fresh", "stale", "blocked", "error"]
ArtifactFormat = Literal["json", "parquet", "joblib", "bundle"]

PROGRESS_EVENT_INTERVAL = 0.25  # seconds between progress-only stage events


# ── stages and the graph ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Stage:
    name: str
    version: int
    deps: tuple[str, ...]
    reads: tuple[str, ...]
    fn: Callable[["StageContext"], Any]
    heavy: bool = False
    requires: tuple[str, ...] = ()
    # What a job chip calls this work, in plain language; not part of the key.
    label: str | None = None
    # What it serves that a gate withholds (``"estimate"``: withheld while a question the estimate
    # rests on is open, and the first one served locks the plan); not part of the key. The gate
    # lists are derived from it, never kept by hand (V2X_SEAMS rule 6).
    serves: str | None = None

    def __post_init__(self) -> None:
        for attr in ("deps", "reads", "requires"):  # accept lists, store tuples
            object.__setattr__(self, attr, tuple(getattr(self, attr)))


class GraphError(ValueError):
    """The graph cannot be ordered: a duplicate, an unknown dep, or a cycle."""


class Graph:
    def __init__(self, stages: tuple[Stage, ...] | list[Stage] = ()):
        self._stages: dict[str, Stage] = {}
        self._order: list[Stage] | None = None
        for stage in stages:
            self.register(stage)

    def register(self, stage: Stage) -> Stage:
        if stage.name in self._stages:
            raise GraphError(f"stage {stage.name!r} is registered twice")
        self._stages[stage.name] = stage
        self._order = None
        return stage

    def stages(self) -> list[Stage]:
        """Stages in registration order."""
        return list(self._stages.values())

    def get(self, name: str) -> Stage | None:
        return self._stages.get(name)

    def __getitem__(self, name: str) -> Stage:
        return self._stages[name]

    def __contains__(self, name: object) -> bool:
        return name in self._stages

    def order(self) -> list[Stage]:
        """Topological order, ties broken by registration order."""
        if self._order is not None:
            return list(self._order)
        for stage in self._stages.values():
            for dep in stage.deps:
                if dep not in self._stages:
                    raise GraphError(f"stage {stage.name!r} depends on unknown stage {dep!r}")
        remaining = {name: set(stage.deps) for name, stage in self._stages.items()}
        done: set[str] = set()
        order: list[Stage] = []
        while remaining:
            ready = [name for name, deps in remaining.items() if deps <= done]
            if not ready:
                raise GraphError(f"stages {sorted(remaining)} form a dependency cycle")
            for name in ready:  # dict order = registration order
                order.append(self._stages[name])
                done.add(name)
                del remaining[name]
        self._order = order
        return list(order)

    def upstream(self, name: str) -> set[str]:
        """Every stage ``name`` depends on, transitively."""
        seen: set[str] = set()
        todo = list(self._stages[name].deps)
        while todo:
            dep = todo.pop()
            if dep not in seen:
                seen.add(dep)
                todo.extend(self._stages[dep].deps)
        return seen


_GRAPHS: dict[str, Graph] = {}
_GRAPHS_LOCK = threading.Lock()


def load_graph(factory: str) -> Graph:
    """Import ``"module:function"`` and call it, once per process."""
    with _GRAPHS_LOCK:
        graph = _GRAPHS.get(factory)
        if graph is None:
            module_name, sep, attr = factory.partition(":")
            if not sep or not module_name or not attr:
                raise ValueError(f"graph_factory must look like 'module:function', not {factory!r}")
            graph = getattr(importlib.import_module(module_name), attr)()
            if not isinstance(graph, Graph):
                raise TypeError(f"{factory} returned {type(graph).__name__}, not a Graph")
            graph.order()  # fail here, not on first use
            _GRAPHS[factory] = graph
        return graph


# ── contexts and wire models ─────────────────────────────────────────────────


@dataclass
class StageContext:
    """What a stage function is handed."""

    project_id: str
    state: ProjectState
    inputs: dict[str, Any]
    paths: dict[str, str]
    settings: dict[str, Any]
    on_progress: Callable[[float, str], Any] | None = field(default=None, repr=False)
    is_cancelled: Callable[[], bool] | None = field(default=None, repr=False)

    def progress(self, fraction: float, message: str = "") -> None:
        """Report progress in [0, 1]; raises :class:`Cancelled` once cancelled."""
        if self.cancelled():
            raise Cancelled()
        if self.on_progress is not None:
            self.on_progress(min(1.0, max(0.0, float(fraction))), message)

    def cancelled(self) -> bool:
        return bool(self.is_cancelled is not None and self.is_cancelled())


@dataclass
class ProjectContext:
    """Everything the engine needs to know about a project, from the workspace."""

    project_id: str
    state: ProjectState
    cache_root: Path
    paths: dict[str, str]
    settings: dict[str, Any]
    fingerprint: str


class StageStatus(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    stage: str
    status: StatusName
    key: str | None = None
    fresh: bool = False
    missing: list[str] = []
    error: str | None = None
    job_id: str | None = None
    progress: float | None = None
    updated_at: datetime | None = None
    # The work for the current key (or for a stage it waits on) was cancelled by
    # request: it restarts only through Engine.ensure (POST .../stages/{stage}/run).
    cancelled: bool = False
    # Why the scheduler holds it (``Engine``'s ``hold``): ``"fit"``, a fit expected to take over
    # about 2 minutes waits for Fit to be pressed; ``"estimate"``, it waits while that estimate is
    # measured (RECIPES_AND_TUNING §4.4). None when nothing holds it.
    held: str | None = None


class StageResult(BaseModel):
    """A stage's artifact as served.

    ``key`` is the key of the artifact returned (null when there is none), and
    ``fresh`` says whether that is the current key. ``artifact`` is the stored
    object: a dict/list for JSON artifacts, a DataFrame for Parquet ones.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    stage: str
    key: str | None = None
    fresh: bool = False
    status: StatusName
    artifact: Any = None


# ── keys ─────────────────────────────────────────────────────────────────────

# The part of a slot's value a stage's key depends on, where it is not the whole value: the
# ``findings`` slot holds every finding's disposition, but only an applied repair changes data, so
# deferring or dismissing a finding recomputes nothing (turbotab.core.repairs registers it).
KEY_VIEWS: dict[str, Callable[[Any], Any]] = {}


def register_key_view(slot: str, view: Callable[[Any], Any]) -> None:
    """``view(json_value) -> value`` replaces ``slot``'s value in every stage key that reads it."""
    KEY_VIEWS[slot] = view


def key_value(slot: str, value: Any) -> Any:
    view = KEY_VIEWS.get(slot)
    return value if view is None or value is None else view(value)


def unmet_requires(order: Sequence[Stage], values: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each stage (in topological ``order``) -> the slots it or a stage upstream ``requires`` that
    are unset in ``values`` (a state's JSON dump), deduplicated in order; empty when it can
    compute. The scheduler's rule (:meth:`Engine._compute_keys`), read by the surfacing registry
    too (``surfacing.blocked``), so the two cannot drift."""
    out: dict[str, list[str]] = {}
    for stage in order:
        missing = [slot for slot in stage.requires if values.get(slot) is None]
        for dep in stage.deps:
            missing.extend(out[dep])
        out[stage.name] = list(dict.fromkeys(missing))
    return out


def stage_key(
    stage: Stage,
    dep_keys: Mapping[str, str],
    reads: Mapping[str, Any],
    fingerprint: str | None,
) -> str:
    payload: dict[str, Any] = {
        "name": stage.name,
        "version": stage.version,
        "deps": dict(dep_keys),
        "reads": dict(reads),
    }
    if not stage.deps:
        payload["fingerprint"] = fingerprint
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── the artifact cache ───────────────────────────────────────────────────────

_FILES: dict[str, str] = {
    "json": "artifact.json",
    "parquet": "artifact.parquet",
    "joblib": "artifact.joblib",
    "bundle": "artifact.json",
}


@dataclass
class Bundle:
    """A stage artifact in parts, for stages whose output is more than JSON.

    ``data`` is plain JSON data and is the only part a client ever sees
    (``StageResult.artifact``). ``frames`` are DataFrames stored as Parquet —
    row ids, fold assignments, predictions. ``objects`` are anything else,
    stored with joblib — fitted pipelines. Downstream stages receive the whole
    Bundle as their input; the cache stays disposable, so fitted objects never
    leave it (BLUEPRINT §2).
    """

    data: Any
    frames: dict[str, Any] = field(default_factory=dict)
    objects: dict[str, Any] = field(default_factory=dict)
    # Files the stage wrote itself (e.g. a working table written out-of-core by DuckDB), moved
    # into the artifact folder on write; on read, absolute paths inside it. For data too large to
    # pass through pandas — BLUEPRINT §2.
    files: dict[str, Any] = field(default_factory=dict)
LATEST = "latest"


def artifact_dir(cache_root: str | os.PathLike[str], stage: str, key: str) -> Path:
    return Path(cache_root) / stage / key


def has_artifact(cache_root: str | os.PathLike[str], stage: str, key: str) -> bool:
    return (artifact_dir(cache_root, stage, key) / "meta.json").is_file()


def read_meta(cache_root: str | os.PathLike[str], stage: str, key: str) -> dict[str, Any]:
    return json.loads((artifact_dir(cache_root, stage, key) / "meta.json").read_text("utf-8"))


def read_artifact(
    cache_root: str | os.PathLike[str], stage: str, key: str, *, public: bool = False
) -> Any:
    """The artifact for ``stage`` at ``key``.

    ``public=True`` returns only what a client may see: a Bundle's ``data``.
    """
    folder = artifact_dir(cache_root, stage, key)
    fmt = read_meta(cache_root, stage, key)["format"]
    path = folder / _FILES[fmt]
    if fmt == "json":
        return json.loads(path.read_text("utf-8"))
    if fmt == "bundle":
        data = json.loads(path.read_text("utf-8"))
        if public:
            return data
        import pandas as pd

        frames = {p.stem: pd.read_parquet(p) for p in sorted((folder / "frames").glob("*.parquet"))}
        objects: dict[str, Any] = {}
        if (folder / "objects").is_dir():
            import joblib

            objects = {p.stem: joblib.load(p) for p in sorted((folder / "objects").glob("*.joblib"))}
        files: dict[str, Any] = {}
        if (folder / "files").is_dir():
            files = {p.name: p.resolve() for p in sorted((folder / "files").iterdir())}
        return Bundle(data=data, frames=frames, objects=objects, files=files)
    if fmt == "parquet":
        import pandas as pd

        return pd.read_parquet(path)
    import joblib

    return joblib.load(path)


def _jsonable(obj: Any) -> Any:
    """Plain JSON data, or TypeError. Non-finite floats become null."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                raise TypeError(f"a {type(k).__name__} cannot be a JSON key")
            out[k if isinstance(k, str) else json.dumps(k)] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_jsonable(v) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(obj, BaseModel):
        return _jsonable(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    pd = sys.modules.get("pandas")
    if pd is not None and (obj is pd.NaT or obj is pd.NA):
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(obj, np.generic):
            return _jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return _jsonable(obj.tolist())
    raise TypeError(f"a {type(obj).__name__} is not JSON data")


def _json_like(obj: Any) -> bool:
    return (
        obj is None
        or isinstance(obj, (Mapping, list, tuple, str, bool, int, float, BaseModel))
        or (dataclasses.is_dataclass(obj) and not isinstance(obj, type))
    )


def _dump(obj: Any, folder: Path) -> ArtifactFormat:
    if isinstance(obj, Bundle):
        text = json.dumps(_jsonable(obj.data), ensure_ascii=False, allow_nan=False)
        (folder / _FILES["bundle"]).write_text(text, "utf-8")
        for kind, parts in (("frames", obj.frames), ("objects", obj.objects)):
            for name in parts:
                if not name or "/" in name or name.startswith("."):
                    raise ValueError(f"a Bundle part cannot be named {name!r}")
        if obj.frames:
            (folder / "frames").mkdir()
            for name, frame in obj.frames.items():
                frame.to_parquet(folder / "frames" / f"{name}.parquet")
        if obj.objects:
            import joblib

            (folder / "objects").mkdir()
            for name, thing in obj.objects.items():
                joblib.dump(thing, folder / "objects" / f"{name}.joblib")
        if obj.files:
            (folder / "files").mkdir()
            for name, source in obj.files.items():
                if not name or "/" in name or name.startswith("."):
                    raise ValueError(f"a Bundle file cannot be named {name!r}")
                shutil.move(os.fspath(source), folder / "files" / name)
        return "bundle"
    if _json_like(obj):
        try:
            text = json.dumps(_jsonable(obj), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            pass  # not plain data after all: keep it whole with joblib
        else:
            (folder / _FILES["json"]).write_text(text, "utf-8")
            return "json"
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(obj, pd.DataFrame):
        try:
            obj.to_parquet(folder / _FILES["parquet"])
            return "parquet"
        except Exception:  # e.g. non-string column names; joblib keeps it whole
            (folder / _FILES["parquet"]).unlink(missing_ok=True)
    import joblib

    joblib.dump(obj, folder / _FILES["joblib"])
    return "joblib"


def write_artifact(
    cache_root: str | os.PathLike[str], stage: str, key: str, version: int, obj: Any
) -> ArtifactFormat:
    """Write atomically (temp dir + rename) and move the ``latest`` pointer."""
    final = artifact_dir(cache_root, stage, key)
    if (final / "meta.json").is_file():
        return read_meta(cache_root, stage, key)["format"]
    stage_dir = final.parent
    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = stage_dir / f".tmp-{key}-{uuid.uuid4().hex}"
    assert tmp is not None
    tmp.mkdir()
    try:
        fmt = _dump(obj, tmp)
        meta = {
            "stage": stage,
            "key": key,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format": fmt,
        }
        (tmp / "meta.json").write_text(json.dumps(meta), "utf-8")
        try:
            os.rename(tmp, final)
        except OSError:
            if (final / "meta.json").is_file():  # someone else wrote the same key
                return read_meta(cache_root, stage, key)["format"]
            shutil.rmtree(final, ignore_errors=True)  # a half-written leftover
            os.rename(tmp, final)
        tmp = None
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    pointer = stage_dir / f".{LATEST}-{uuid.uuid4().hex}"
    pointer.write_text(key, "utf-8")
    os.replace(pointer, stage_dir / LATEST)
    return fmt


def latest_key(
    cache_root: str | os.PathLike[str], stage: str, exclude: str | None = None
) -> str | None:
    """The newest artifact's key for ``stage``, other than ``exclude``."""
    stage_dir = Path(cache_root) / stage
    try:
        key = (stage_dir / LATEST).read_text("utf-8").strip()
    except OSError:
        key = ""
    if key and key != exclude and has_artifact(cache_root, stage, key):
        return key
    best: tuple[str, str] | None = None  # (created_at, key)
    try:
        entries = list(stage_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        if entry.name.startswith(".") or entry.name == exclude or not entry.is_dir():
            continue
        try:
            meta = json.loads((entry / "meta.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        candidate = (str(meta.get("created_at", "")), entry.name)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None


# ── the heavy-stage job (runs in a worker process) ──────────────────────────


def run_stage_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute one stage for one key inside a worker; returns metadata only."""
    job = current_job()
    graph = load_graph(spec["graph_factory"])
    stage = graph[spec["stage"]]
    if stage.version != spec["version"]:
        raise RuntimeError(
            f"stage {stage.name!r} is version {stage.version} in the worker but "
            f"{spec['version']} in the engine; restart the server"
        )
    cache_root = Path(spec["cache_root"])
    inputs = {dep: read_artifact(cache_root, dep, key) for dep, key in spec["dep_keys"].items()}
    ctx = StageContext(
        project_id=spec["project_id"],
        state=spec["state"],
        inputs=inputs,
        paths=dict(spec["paths"]),
        settings=dict(spec["settings"]),
        on_progress=job.progress if job is not None else None,
        is_cancelled=job.cancelled if job is not None else None,
    )
    started = time.perf_counter()
    result = stage.fn(ctx)
    if ctx.cancelled():
        raise Cancelled()  # whatever came back may be partial: never cache it
    fmt = write_artifact(cache_root, stage.name, spec["key"], stage.version, result)
    return {
        "stage": stage.name,
        "key": spec["key"],
        "format": fmt,
        "seconds": time.perf_counter() - started,
    }


# ── the engine ───────────────────────────────────────────────────────────────


@dataclass(eq=False)
class _Run:
    """Work in flight for one stage and one key."""

    stage: str
    key: str
    heavy: bool
    job_id: str | None = None
    started: bool = False
    progress: float | None = None
    message: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


class _Project:
    def __init__(self, pid: str):
        self.pid = pid
        self.lock = threading.RLock()
        self.ctx: ProjectContext | None = None
        self.keys: dict[str, str | None] = {}
        self.missing: dict[str, list[str]] = {}
        self.runs: dict[str, _Run] = {}
        self.errors: dict[tuple[str, str], str] = {}
        self.held: set[tuple[str, str]] = set()  # cancelled by request: not rerun by itself
        self.known: set[tuple[str, str]] = set()  # artifacts known to exist
        self.has_any: set[str] = set()  # stages with at least one artifact
        self.waiting: dict[str, str] = {}  # held by the scheduler's hold, and why
        self.published: dict[str, tuple[Any, ...]] = {}
        self.updated_at: dict[str, datetime] = {}
        self.progress_sent: dict[str, float] = {}


@dataclass(frozen=True)
class HoldView:
    """A project as a hold reads it, under the scheduler's lock: its state, and each stage's
    artifact at its current key (None when not computed) and whether it is still to come."""

    pid: str
    state: Any
    artifact: Callable[[str], Any]
    pending: Callable[[str], bool]


# Whether the scheduler holds a stage that is ready to start, and why (``StageStatus.held``); None
# when it starts. Called under the project's lock: it must not wait on the engine.
Hold = Callable[[str, HoldView], "str | None"]


class Engine:
    """Keeps every project's stage graph live. Thread-safe.

    The engine does not own ``runner`` or ``bus``: :meth:`shutdown` stops the
    engine's own threads and cancels its jobs, and the caller shuts the runner.

    ``hold``: the scheduler's hold (RECIPES_AND_TUNING §4.4): a stage it holds is not started
    until a later refresh finds it released. It is the scheduler's, not a stage requirement, so
    keys and artifacts stay pure functions of the log; with none, every stage computes live.
    """

    def __init__(
        self,
        *,
        graph_factory: str,
        runner: JobRunner,
        bus: EventBus,
        project_ctx: Callable[[str], ProjectContext],
        threads: int = 4,
        hold: Hold | None = None,
    ):
        self.graph_factory = graph_factory
        self.graph = load_graph(graph_factory)
        self._order = self.graph.order()
        self._runner = runner
        self._bus = bus
        self._project_ctx = project_ctx
        self._hold = hold
        self._pool = ThreadPoolExecutor(threads, thread_name_prefix="turbotab-stage")
        self._lock = threading.Lock()
        self._projects: dict[str, _Project] = {}
        self._closed = False

    # ── public API ──

    def keys(self, pid: str) -> dict[str, str | None]:
        p = self._project(pid)
        with p.lock:
            return dict(p.keys)

    def status(self, pid: str) -> dict[str, StageStatus]:
        p = self._project(pid)
        with p.lock:
            return self._statuses(p)

    def get(self, pid: str, stage: str, allow_stale: bool = True) -> StageResult:
        if stage not in self.graph:
            raise KeyError(stage)
        p = self._project(pid)
        with p.lock:
            status = self._statuses(p)[stage]
            assert p.ctx is not None
            cache_root = p.ctx.cache_root
        if status.status == "fresh" and status.key is not None:
            artifact = read_artifact(cache_root, stage, status.key, public=True)
            return StageResult(stage=stage, key=status.key, fresh=True, status="fresh", artifact=artifact)
        if allow_stale:
            old = latest_key(cache_root, stage, exclude=status.key)
            if old is not None:
                return StageResult(
                    stage=stage,
                    key=old,
                    fresh=False,
                    status=status.status,
                    artifact=read_artifact(cache_root, stage, old, public=True),
                )
        return StageResult(stage=stage, key=None, fresh=False, status=status.status, artifact=None)

    def on_decision(self, pid: str) -> None:
        """Bring ``pid`` up to date with its (new) state; call after every decision."""
        self._project(pid, refresh=True)

    def peek(self, pid: str) -> dict[str, StageStatus] | None:
        """Statuses for a project the engine already holds; None, without loading it, otherwise."""
        with self._lock:
            p = self._projects.get(pid)
        if p is None:
            return None
        with p.lock:
            return self._statuses(p) if p.ctx is not None else None

    def ensure(self, pid: str, stage: str) -> str | None:
        """Compute ``stage`` for its current key, retrying errors and cancellations.

        Returns the stage's job id when it is a heavy stage now in flight.
        """
        if stage not in self.graph:
            raise KeyError(stage)
        p = self._project(pid)
        with p.lock:
            for name in self.graph.upstream(stage) | {stage}:
                key = p.keys.get(name)
                if key is not None:
                    p.errors.pop((name, key), None)
                    p.held.discard((name, key))
            self._schedule(p)
            self._publish(p)
            run = p.runs.get(stage)
            return run.job_id if run is not None else None

    def shutdown(self) -> None:
        self._closed = True
        with self._lock:
            projects = list(self._projects.values())
        for p in projects:
            with p.lock:
                runs = list(p.runs.values())
                p.runs.clear()
            for run in runs:
                run.cancel.set()
                if run.job_id is not None:
                    try:
                        self._runner.cancel(run.job_id)
                    except Exception:
                        pass
        self._pool.shutdown(wait=True, cancel_futures=True)

    # ── bookkeeping (all under p.lock) ──

    def _project(self, pid: str, refresh: bool = False) -> _Project:
        with self._lock:
            p = self._projects.get(pid)
            if p is None:
                p = self._projects[pid] = _Project(pid)
        with p.lock:
            if refresh or p.ctx is None:
                # First touch after a restart counts as a decision: work that is
                # missing for the current keys starts now, not at the next click.
                try:
                    self._refresh(p)
                except BaseException:
                    if p.ctx is None:  # e.g. an unknown project: remember nothing
                        with self._lock:
                            if self._projects.get(pid) is p:
                                del self._projects[pid]
                    raise
        return p

    def _refresh(self, p: _Project) -> None:
        p.ctx = self._project_ctx(p.pid)
        self._compute_keys(p)
        self._cancel_superseded(p)
        self._publish(p)
        self._schedule(p)
        self._publish(p)

    def _compute_keys(self, p: _Project) -> None:
        assert p.ctx is not None
        state = p.ctx.state
        fields = type(state).model_fields
        values = state.model_dump(mode="json")
        unmet = unmet_requires(self._order, values)
        for stage in self._order:
            for slot in (*stage.reads, *stage.requires):
                if slot not in fields:
                    raise GraphError(
                        f"stage {stage.name!r} reads slot {slot!r}, which "
                        f"{type(state).__name__} does not have"
                    )
            missing = unmet[stage.name]
            p.missing[stage.name] = missing
            if missing:
                p.keys[stage.name] = None
                continue
            p.keys[stage.name] = stage_key(
                stage,
                {dep: p.keys[dep] for dep in stage.deps},  # type: ignore[misc]
                {slot: key_value(slot, values.get(slot)) for slot in stage.reads},
                p.ctx.fingerprint,
            )

    def _cancel_superseded(self, p: _Project) -> None:
        for name, run in list(p.runs.items()):
            if p.keys.get(name) != run.key:
                del p.runs[name]
                run.cancel.set()
                if run.job_id is not None:
                    self._runner.cancel(run.job_id)

    def _has(self, p: _Project, stage: str, key: str | None) -> bool:
        if key is None:
            return False
        if (stage, key) in p.known:
            return True
        assert p.ctx is not None
        if has_artifact(p.ctx.cache_root, stage, key):
            p.known.add((stage, key))
            p.has_any.add(stage)
            return True
        return False

    def _has_older(self, p: _Project, stage: str, key: str) -> bool:
        if stage in p.has_any:
            return True
        assert p.ctx is not None
        if latest_key(p.ctx.cache_root, stage, exclude=key) is not None:
            p.has_any.add(stage)
            return True
        return False

    def _schedule(self, p: _Project) -> None:
        if self._closed or p.ctx is None:
            return
        p.waiting = {}
        for stage in self._order:
            key = p.keys.get(stage.name)
            if key is None or stage.name in p.runs:
                continue
            if (stage.name, key) in p.errors or (stage.name, key) in p.held:
                continue
            if self._has(p, stage.name, key):
                continue
            if all(self._has(p, dep, p.keys.get(dep)) for dep in stage.deps):
                why = self._held(p, stage.name)
                if why is not None:
                    p.waiting[stage.name] = why
                    continue
                self._start(p, stage, key)

    def _held(self, p: _Project, stage: str) -> str | None:
        """Why the hold keeps ``stage`` from starting now, or None. A hold that fails holds
        nothing: computing stays live, and the failure is logged."""
        if self._hold is None:
            return None
        assert p.ctx is not None
        cache_root = p.ctx.cache_root

        def artifact(name: str) -> Any:
            key = p.keys.get(name)
            if key is None or not self._has(p, name, key):
                return None
            return read_artifact(cache_root, name, key, public=True)

        def pending(name: str) -> bool:
            """Computing, or bound to be: not blocked, failed or stopped, nor waiting on a stage
            that is."""
            key = p.keys.get(name)
            if key is None or self._has(p, name, key):
                return False
            if name in p.runs:
                return True
            if (name, key) in p.errors or (name, key) in p.held:
                return False
            return all(self._has(p, dep, p.keys.get(dep)) or pending(dep)
                       for dep in self.graph[name].deps)

        try:
            return self._hold(stage, HoldView(p.pid, p.ctx.state, artifact, pending))
        except Exception:  # noqa: BLE001 - a hold that cannot decide holds nothing
            log.exception("the hold could not decide on %s for project %s", stage, p.pid)
            return None

    def _start(self, p: _Project, stage: Stage, key: str) -> None:
        ctx = p.ctx
        assert ctx is not None
        dep_keys = {dep: p.keys[dep] for dep in stage.deps}
        run = _Run(stage.name, key, stage.heavy)
        p.runs[stage.name] = run
        if not stage.heavy:
            try:
                self._pool.submit(self._run_light, p, stage, run, dep_keys, ctx)
            except RuntimeError:  # the pool shut down under us
                del p.runs[stage.name]
            return
        spec = {
            "graph_factory": self.graph_factory,
            "stage": stage.name,
            "version": stage.version,
            "key": key,
            "dep_keys": dep_keys,
            "cache_root": str(ctx.cache_root),
            "project_id": ctx.project_id,
            "state": ctx.state,
            "paths": dict(ctx.paths),
            "settings": dict(ctx.settings),
        }
        try:
            run.job_id = self._runner.submit(
                run_stage_job,
                spec,
                label=stage.label or stage.name,
                stage=stage.name,
                on_progress=partial(self._on_job_progress, p, run),
                on_done=partial(self._on_job_done, p, run),
            )
        except Exception as exc:
            del p.runs[stage.name]
            p.errors[(stage.name, key)] = f"{type(exc).__name__}: {exc}"
            log.exception("could not submit stage %s", stage.name)
            return
        self._bus.publish(p.pid, "job", self._runner.get(run.job_id).model_dump(mode="json"))

    def _run_light(
        self, p: _Project, stage: Stage, run: _Run, dep_keys: dict[str, str], ctx: ProjectContext
    ) -> None:
        with p.lock:
            if p.runs.get(stage.name) is not run or run.cancel.is_set():
                return  # superseded before it started
            run.started = True
            self._publish(p)
        outcome, error = "done", None
        try:
            inputs = {dep: read_artifact(ctx.cache_root, dep, key) for dep, key in dep_keys.items()}
            sctx = StageContext(
                project_id=ctx.project_id,
                state=ctx.state,
                inputs=inputs,
                paths=dict(ctx.paths),
                settings=dict(ctx.settings),
                on_progress=partial(self._on_light_progress, p, run),
                is_cancelled=run.cancel.is_set,
            )
            result = stage.fn(sctx)
            if run.cancel.is_set():
                raise Cancelled()
            write_artifact(ctx.cache_root, stage.name, run.key, stage.version, result)
        except Cancelled:
            outcome = "cancelled"
        except BaseException as exc:  # noqa: BLE001 - a stage failure is a status
            outcome, error = "error", f"{type(exc).__name__}: {exc}"
            log.exception("stage %s failed for project %s", stage.name, p.pid)
        self._finish_run(p, run, outcome, error)

    def _on_light_progress(self, p: _Project, run: _Run, fraction: float, message: str) -> None:
        with p.lock:
            if p.runs.get(run.stage) is run:
                run.progress = fraction
                run.message = message or None
                self._publish(p, progress_of=run.stage)

    def _on_job_progress(self, p: _Project, run: _Run, view: JobView) -> None:
        with p.lock:
            if p.runs.get(run.stage) is run:
                run.started = run.started or view.state == "running"
                run.progress = view.progress
                run.message = view.message
                self._publish(p, progress_of=run.stage)
            self._bus.publish(p.pid, "job", view.model_dump(mode="json"))

    def _on_job_done(self, p: _Project, run: _Run, view: JobView, result: Any) -> None:
        with p.lock:
            self._bus.publish(p.pid, "job", view.model_dump(mode="json"))
            self._finish_run(p, run, view.state, view.error)

    def _finish_run(self, p: _Project, run: _Run, outcome: str, error: str | None) -> None:
        with p.lock:
            current = p.runs.get(run.stage) is run
            if current:
                del p.runs[run.stage]
            if outcome == "done":
                p.known.add((run.stage, run.key))
                p.has_any.add(run.stage)
            elif current and outcome == "error":
                p.errors[(run.stage, run.key)] = error or "The stage failed."
            elif current and outcome == "cancelled":
                p.held.add((run.stage, run.key))
            self._schedule(p)
            self._publish(p)

    def _statuses(self, p: _Project) -> dict[str, StageStatus]:
        out: dict[str, StageStatus] = {}
        for stage in self._order:
            out[stage.name] = self._status(p, stage, out)
        return out

    def _status(self, p: _Project, stage: Stage, upstream: dict[str, StageStatus]) -> StageStatus:
        name = stage.name
        key = p.keys.get(name)
        common: dict[str, Any] = {"stage": name, "key": key, "updated_at": p.updated_at.get(name)}
        if key is None:
            return StageStatus(status="blocked", missing=list(p.missing.get(name, [])), **common)
        if self._has(p, name, key):
            return StageStatus(status="fresh", fresh=True, **common)
        run = p.runs.get(name)
        if run is not None:
            return StageStatus(
                status="running" if run.started else "queued",
                job_id=run.job_id,
                progress=run.progress,
                **common,
            )
        error = p.errors.get((name, key))
        if error is None:
            failed = next((d for d in stage.deps if upstream[d].status == "error"), None)
            if failed is not None:
                error = f"Needs {failed!r}, which failed."
        if error is not None:
            return StageStatus(status="error", error=error, **common)
        cancelled = (name, key) in p.held or any(upstream[d].cancelled for d in stage.deps)
        return StageStatus(
            status="stale" if self._has_older(p, name, key) else "idle",
            cancelled=cancelled,
            held=p.waiting.get(name),
            **common,
        )

    def _publish(self, p: _Project, progress_of: str | None = None) -> None:
        """Publish a ``stage`` event for each stage whose status changed."""
        now = datetime.now(timezone.utc)
        tick = time.monotonic()
        for name, status in self._statuses(p).items():
            signature = (
                status.status,
                status.key,
                status.fresh,
                tuple(status.missing),
                status.error,
                status.job_id,
                status.cancelled,
            )
            if p.published.get(name) != signature:
                p.published[name] = signature
                p.updated_at[name] = now
            elif name != progress_of or tick - p.progress_sent.get(name, 0.0) < PROGRESS_EVENT_INTERVAL:
                continue
            p.progress_sent[name] = tick
            status.updated_at = p.updated_at[name]
            self._bus.publish(p.pid, "stage", status.model_dump(mode="json"))

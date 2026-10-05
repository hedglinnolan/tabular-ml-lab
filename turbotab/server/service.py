"""Project wiring: the workspace, a decision log per project, one engine.

The server orchestrates and computes nothing. Every number it serves comes from
the data layer (``DataStore``) or from a stage artifact.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

from turbotab.core import consequences, decisions, estimand, seal, voice  # seal: its validators and preview
from turbotab.core import plan_lock  # noqa: F401 - the analysis-plan lock's validators (WP16)
from turbotab.core.config import Settings
from turbotab.core.datastore import DataStore, fingerprint_file
from turbotab.core.datastore import _source_kind as source_kind  # the one list of readable types
from turbotab.core.decisions import DecisionLog, ProjectState, Refusal
from turbotab.core.events import EventBus
from turbotab.core.graph import (
    Engine,
    ProjectContext,
    StageStatus,
    artifact_dir,
    latest_key,
    read_artifact,
)
from turbotab.core.interview import InterviewStep, route
from turbotab.core.jobs import PRELOAD, JobRunner, JobView
from turbotab.core.stages import GRAPH_FACTORY
from turbotab.core.workspace import ProjectMeta, Workspace
from turbotab.server.errors import ApiError

log = logging.getLogger(__name__)

WORKER_PRELOAD = PRELOAD + (
    "duckdb",
    "pyarrow.parquet",
    "turbotab.core.datastore",
    "turbotab.core.stages",
    "turbotab.engine",
    "turbotab.packs",
)
SOURCE_FILE = "source.json"
# Audit WP16 (the routing gate): where an estimate was first displayed under prediction, kept beside
# the project until the purpose becomes inference, when the plan lock records it.
SHOWN_UNDER_PREDICTION = "estimates_shown.json"
# Decision kinds the server records itself and a client never posts: the analysis-plan lock.
SYSTEM_KINDS = ("lock_plan",)
MAX_REMEMBERED_JOBS = 10_000


# ── decision validation that needs the server's view of the project ─────────


@dataclass(frozen=True)
class DecisionContext:
    """What ``decisions.validate`` and ``voice.sentence_for`` are told about a project.

    ``columns`` is None until the ingest stage is fresh: before that the
    dataset's columns are not known. ``target`` is the current target (None:
    none chosen), which a task answer must name. ``column_info`` maps each
    column to ``{dtype, n_unique, n_missing}`` (from the ingest artifact);
    ``state`` is the state before the decision; ``task`` the answered task, else
    the detected one. ``store()`` returns the DataStore (None before ingest) and
    ``artifact(stage)`` a stage's fresh public artifact (None when not fresh):
    both are for sentences that count rows, and cost nothing until called.
    """

    columns: list[str] | None
    ingest_status: str
    ingest_error: str | None = None
    target: str | None = None
    column_info: dict[str, dict[str, Any]] | None = None
    state: ProjectState | None = None
    task: str | None = None
    n_rows: int | None = None
    store: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    artifact: Callable[[str], Any] | None = field(default=None, repr=False, compare=False)
    # The newest split's held-out row ids (None before any split): what a check on the data
    # made while validating (a partition's units) must not read.
    sealed: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    # Every row the analysis reads under inference (the split's rows on both sides of the seal, or
    # the cohort's before a split): what an inference check on the data reads (ruling 3).
    analyzed: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    # The decision log as it stands (the seal's checks: who drew the seal, what a revert undoes).
    records: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    # The Router's steps as the project stands (answers in order, M2_CONTRACT §12.2).
    interview: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    # A stage's newest public artifact, fresh or stale: what a client may have been shown (the
    # readings ledger reads which roles a bulk answer carried against it; BLUEPRINT §14.1).
    shown: Callable[[str], Any] | None = field(default=None, repr=False, compare=False)
    # The fresh fit's held-out scores, set only while a decision is being recorded, never for a
    # preview: what the opening keeps in the record as the reported result (audit WP16, RO-05).
    sealed_scores: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    # The project's folder: a join reads its added files, a codebook import its staged codebook
    # (DATAIN, V2 definition of done §1).
    project_dir: str | None = None


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


# first: a table that cannot be read (or is still being read) is the reason before any other,
# including an answer the Router has not reached yet (M2_CONTRACT §12.2)
for _kind in ("set_target", "set_roles", "set_exclusions", "set_energy_adjustment", "set_substitution"):
    decisions.register_validator(_kind, _target_needs_columns, first=True)


class SentenceFacts:
    """The facts ``voice.sentence_for`` reads (its documented ``ctx`` keys), gathered lazily.

    Each is read only when the decision's sentence asks for it. Row counts come from the same
    participant flow the ``cohort`` stage runs (``stages.rows.compute_cohort``), so a sentence
    and the Rows panel never disagree. Anything that cannot be known yet is None: the sentence
    then says less.
    """

    def __init__(self, ctx: DecisionContext, decision: Any, records: list[Any]):
        self._ctx = ctx
        self._decision = decision
        self.records = records
        self.n_rows = ctx.n_rows
        self._flow: tuple[list[dict[str, Any]], list[str]] | None | bool = False

    @cached_property
    def columns(self) -> list[dict[str, Any]] | None:
        if self._ctx.columns is None:
            return None
        info = self._ctx.column_info or {}
        return [{"name": c, "n_missing": (info.get(c) or {}).get("n_missing")}
                for c in self._ctx.columns]

    def _artifact(self, stage: str) -> Any:
        try:
            return self._ctx.artifact(stage) if self._ctx.artifact else None
        except Exception:  # noqa: BLE001 - a sentence never fails a decision
            return None

    @cached_property
    def datastore(self) -> Any:
        return self._ctx.store() if self._ctx.store else None

    @cached_property
    def detected_task(self) -> str | None:
        info = self._artifact("target_info")
        state = self._ctx.state
        if isinstance(info, dict) and state is not None and info.get("column") == state.target:
            return info.get("detected_task") or info.get("task")
        return None

    @cached_property
    def read_from_values(self) -> str | None:
        """The methods record's line for the readings the values settled, no question asked
        (BLUEPRINT §14.3, amendment: settlement is visible in the methods record)."""
        try:
            from turbotab.core.readings import read_from_data, read_from_values_sentence

            items = read_from_data(self._ctx.state, roles=self._artifact("roles"),
                                   target_info=self._artifact("target_info"),
                                   proposals=self._artifact("proposals"), store=self.datastore)
        except Exception:  # noqa: BLE001 - a sentence never fails a decision; it says less
            return None
        return read_from_values_sentence(items) or None

    @cached_property
    def repeats(self) -> dict[str, Any] | None:
        roles = self._artifact("roles")
        repeats = roles.get("repeats") if isinstance(roles, dict) else None
        state = self._ctx.state
        if repeats and state is not None and (state.roles or {}).get(repeats["column"]) == "identifier":
            return repeats
        return None

    @cached_property
    def n_cohort(self) -> int | None:
        cohort = self._artifact("cohort")
        return int(cohort["n_final"]) if isinstance(cohort, dict) and "n_final" in cohort else None

    def _steps(self) -> list[dict[str, Any]] | None:
        """The flow's steps with this decision's exclusions or missing-value answer in place."""
        if self._flow is not False:
            return self._flow  # type: ignore[return-value]
        self._flow = None
        state, store = self._ctx.state, self.datastore
        info = self._ctx.column_info or {}
        # the columns of the table the cohort stage reads (the working table), as it records them:
        # dtype and distinct counts too, which decide the columns whose blanks become a level
        ingest = ({"columns": [{"name": c, "n_missing": (info.get(c) or {}).get("n_missing"),
                                **{k: v for k, v in (info.get(c) or {}).items() if k != "n_missing"}}
                               for c in self._ctx.columns]}
                  if self._ctx.columns is not None else None)
        d = self._decision
        if state is None or store is None or not isinstance(ingest, dict):
            return None
        if d.kind == "set_exclusions":
            state = state.model_copy(update={"exclusions": d.rules, "missing": None})
        elif d.kind == "set_missing":  # all of it: a blank kept as a level drops no row
            state = state.model_copy(update={"missing": decisions.MissingSpec(
                **d.model_dump(exclude={"kind"}))})
        else:
            return None
        try:
            from turbotab.core.stages.rows import compute_cohort

            steps, _, _ = compute_cohort(store, state, ingest)
        except Exception:  # noqa: BLE001 - a sentence never fails a decision; it says less
            return None
        self._flow = steps
        return steps

    @property
    def exclusion_counts(self) -> list[int] | None:
        steps = self._steps() if self._decision.kind == "set_exclusions" else None
        if steps is None:
            return None
        from turbotab.core.stages.rows import rule_drops  # a rule's lines: not recorded, then range

        return rule_drops(steps)

    @property
    def n_before(self) -> int | None:
        steps = self._steps()
        if not steps:
            return None
        if self._decision.kind == "set_missing":
            done = [s for s in steps if s["key"] != "complete_cases"]
            return int(done[-1]["n"]) if done else None
        first = next((i for i, s in enumerate(steps) if str(s["key"]).startswith("exclusion:")), None)
        return int(steps[first - 1]["n"]) if first else None

    # M2 facts (voice.py documents them): each read only when the decision's sentence asks.

    @cached_property
    def finding(self) -> dict[str, Any] | None:
        fid = getattr(self._decision, "finding_id", None)
        found = self._artifact("findings") if fid else None
        for f in (found or {}).get("findings", []) if isinstance(found, dict) else []:
            if f.get("id") == fid:
                return f
        return None

    @cached_property
    def levels(self) -> list[Any] | None:
        info = self._artifact("target_info")
        column = getattr(self._decision, "column", None)
        if isinstance(info, dict) and info.get("column") == column and info.get("classes"):
            return [c["value"] for c in info["classes"]]
        return None

    @cached_property
    def seal_plan(self) -> dict[str, Any] | None:
        """The seal's basis and chronological draw for the current answers (the split sentence)."""
        plan = self._artifact("seal_plan")
        return plan if isinstance(plan, dict) else None

    @cached_property
    def n_holdout(self) -> int | None:
        split = self._artifact("split")
        return int(split["n_holdout"]) if isinstance(split, dict) and "n_holdout" in split else None

    @cached_property
    def outcome_unit(self) -> str | None:
        info = self._artifact("target_info")
        column = getattr(self._decision, "column", None)
        if isinstance(info, dict) and info.get("column") == column:
            return info.get("unit")
        return None

    @property
    def n_complete(self) -> int | None:
        steps = self._steps() if self._decision.kind == "set_missing" else None
        last = steps[-1] if steps else None
        return int(last["n"]) if last is not None and last["key"] == "complete_cases" else None


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
    column_info: dict[str, dict[str, Any]] = field(default_factory=dict)


PREVIEW_CELLS = 20_000_000  # the before-frame's budget: rows x columns
MAX_CACHED_ARTIFACTS = 16
MAX_CACHED_FRAMES = 4


def _write_json_atomic(path: Path, payload: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


POOL_WORDS = {
    "training": "training rows",
    "analyzed": "analyzed rows",
    "cohort": "rows in the analysis",
    "unsealed": "rows not held out",
    "all": "rows",
    "given": "rows",
}


def preview_basis(ctx: consequences.PreviewContext, result: consequences.PreviewResult,
                  n_rows: int) -> str:
    """The rows a preview was computed on, in words, from what its builders actually read.

    An exact count over the pool ("Counts on the 17,479 rows not held out") and a sample
    ("values on a sample of 5,000 of the 17,084 training rows") are named separately; a scatter
    that draws fewer points than it sampled says so.
    """
    if ctx.read.get("basis"):  # a builder that read every row says so itself (a repair's preview)
        return str(ctx.read["basis"])
    sealed = ctx.sealed_row_ids is not None and len(ctx.sealed_row_ids) > 0
    parts: list[str] = []
    counted = ctx.read.get("counted")
    if counted is not None:
        parts.append(f"Counts on the {counted:,} rows not held out" if sealed
                     else f"Counts on all {counted:,} rows")
    sample = ctx.read.get("sample")
    if sample is not None:
        kind, pool, drawn = sample
        words = POOL_WORDS.get(kind, "rows")
        if drawn < pool:
            parts.append(f"values on a sample of {drawn:,} of the {pool:,} {words}")
        else:
            parts.append(f"values on all {pool:,} {words}")
        drawn_points = max((len(v.points_before) for v in result.views
                            if getattr(v, "kind", None) == "relationship"), default=0)
        if 0 < drawn_points < drawn:
            parts.append(f"the scatter draws {drawn_points:,} of them")
    if not parts:
        return ("Read from the column names and summaries; no rows were read." if not sealed else
                "Read from the column names and summaries; held-out rows stay sealed.")
    if sealed:
        parts.append("held-out rows stay sealed")
    text = "; ".join(parts) + "."
    return text[:1].upper() + text[1:]


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
        self._artifacts: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
        self._frames: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._locking = threading.Lock()  # one analysis-plan lock per project, however many reads

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
        state = self.log(pid).state()
        return ProjectContext(
            project_id=pid,
            state=state,
            cache_root=self.workspace.cache_dir(pid),
            paths={
                "source": meta.source_path,
                # The table the ingest stage writes for these joins (each set its own file).
                "data": str(self.workspace.table_path(pid, state)),
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

    # ── files to join, codebooks to import (DATAIN, V2 definition of done §1) ──

    def add_file_from_path(self, pid: str, raw: str) -> dict[str, Any]:
        """Add a file on this machine to the project, to join to its table: it is read where it
        is, once, into ``files/<id>/raw.parquet`` (as the table's own file is read)."""
        self.workspace.get(pid)
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ApiError(400, "relative_path", "Give the file's full path, starting from the top of the disk.")
        if path.is_dir():
            raise ApiError(400, "is_a_folder", f"{path} is a folder. Choose a file inside it.")
        if not path.is_file():
            raise ApiError(404, "no_such_file", f"There is no file at {path}.")
        try:
            source_kind(path)
        except ValueError as exc:
            raise ApiError(400, "unsupported_file", str(exc)) from None
        if not os.access(path, os.R_OK):
            raise ApiError(403, "unreadable_file", f"TurboTab is not allowed to read {path}.")
        return self._add_file(pid, path.resolve(), path.name, "path", copy=False)

    def add_file_from_upload(self, pid: str, staged: Path, client_name: str) -> dict[str, Any]:
        """Adopt a file streamed into ``uploads/`` as a file to join to the project's table."""
        try:
            return self._add_file(pid, staged, client_name, "upload", copy=True)
        finally:
            shutil.rmtree(staged.parent, ignore_errors=True)

    def _add_file(self, pid: str, source: Path, name: str, kind: str, *, copy: bool) -> dict[str, Any]:
        """``copy``: an upload moves into ``files/<id>/source/``; a file on this machine is read
        where it is, as the table's own file is. Either is read once, into the file's own
        ``raw.parquet``, which every join reads."""
        import secrets

        from turbotab.core.datastore import ingest

        files = self.workspace.files_dir(pid)
        while True:
            fid = "f" + secrets.token_hex(5)
            folder = files / fid
            try:
                folder.mkdir()
                break
            except FileExistsError:
                continue
        dest = source
        if copy:
            (folder / "source").mkdir()
            dest = folder / "source" / source.name
            os.replace(source, dest)
        try:
            info = ingest(dest, folder / "raw.parquet")
        except (ValueError, OSError) as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise ApiError(400, "unreadable_file", f"{name} could not be read: {exc}") from None
        meta = {"id": fid, "name": name, "source_kind": kind, "source_path": str(dest),
                "fingerprint": info.fingerprint,
                "n_rows": info.n_rows, "n_cols": info.n_cols,
                "columns": [c.name for c in info.columns], "warnings": list(info.warnings)}
        _write_json_atomic(folder / "file.json", meta)
        return meta

    def files(self, pid: str) -> list[dict[str, Any]]:
        """The files added to the project, oldest first, and whether each is joined."""
        from turbotab.core.assembly import file_meta

        self.workspace.get(pid)
        state = self.log(pid).state()
        joined = set((state.joins or {}).keys())
        folder = self.workspace.files_dir(pid)
        out = []
        for entry in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime):
            meta = file_meta(self.workspace.project_dir(pid), entry.name)
            if meta is not None:
                out.append({**meta, "joined": entry.name in joined})
        return out

    def _table_ready(self, pid: str) -> Path:
        ingest = self.engine.status(pid)["ingest"]
        if ingest.status != "fresh" or not ingest.key:
            raise ApiError(409, "not_yet", "The table is still being read; try again once it is ready.")
        return self._ingested_table(pid, ingest.key)

    def _ingested_table(self, pid: str, key: str) -> Path:
        """The file the ingest artifact at ``key`` describes: the source's own table, or the file
        of the set of joins it was read with (it names it, ``data/<table>``). Read from the artifact,
        never from the log: this is reached while a decision is being recorded."""
        info = read_artifact(self.workspace.cache_dir(pid), "ingest", key, public=True)
        name = info.get("table") if isinstance(info, dict) else None
        if not name:
            return self.workspace.data_path(pid)
        return self.workspace.data_path(pid).with_name(Path(str(name)).name)

    def join_preview(self, pid: str, file: str, on: str, right_on: str | None, how: str) -> dict[str, Any]:
        """What joining ``file`` on ``on`` would do: the row counts on each side, the rows with no
        partner, the relation, the joined table's rows, and the sentence it would record; a
        many-to-many join carries its refusal. Nothing is recorded."""
        from turbotab.core.assembly import file_meta, file_parquet, join_sentence, preview

        self.workspace.get(pid)
        pdir = self.workspace.project_dir(pid)
        meta = file_meta(pdir, file)
        if meta is None:
            raise ApiError(404, "unknown_file", f"This project has no added file {file!r}.")
        table = self._table_ready(pid)
        plan = preview(table, file_parquet(pdir, file), on=on, right_on=right_on, how=how,
                       left_name=self.workspace.get(pid).source_name, right_name=str(meta["name"]))
        out = plan.to_dict()
        out["file_side"] = out.pop("file")
        out["file"] = file
        out["sentence"] = (voice.finish(join_sentence(str(meta["name"]), on, right_on, how,
                                                      plan.counts()))
                           if plan.refusal is None else "")
        return out

    def stage_codebook(self, pid: str, *, path: str | None = None, labels: bool = False,
                       staged: Path | None = None, client_name: str | None = None) -> dict[str, Any]:
        """Read a codebook (a file on this machine, an upload, or the table's own XPT labels), keep
        it in the project, and say what importing it would do: the readings it settles, the fields
        the values contradict (asked), the answers that stand, and the sentence it would record.
        Nothing is recorded until ``import_codebook``."""
        from turbotab.core import codebook as cb

        self.workspace.get(pid)
        pdir = self.workspace.project_dir(pid)
        table = self._table_ready(pid)

        def book_source(raw: str | None, up: Path | None) -> Path | None:
            if up is not None:
                return up
            return Path(raw).expanduser() if raw else None

        try:
            book = self._read_codebook(table, path, labels, staged, client_name)
            cb.stage(book, pdir, book_source(path, staged))
        finally:
            if staged is not None:
                shutil.rmtree(staged.parent, ignore_errors=True)
        a = cb.assessment_for(book, pdir, self.log(pid).state())
        out = a.to_dict()
        out["sentence"] = voice.finish(out["sentence"])
        return out

    def _read_codebook(self, table: Path, path: str | None, labels: bool, staged: Path | None,
                       client_name: str | None) -> Any:
        from turbotab.core import codebook as cb
        from turbotab.core.datastore import read_labels

        try:
            if labels:
                sidecar = read_labels(table) or {}
                sources = sidecar.get("sources") or []
                files = [str(s.get("file")) for s in sources if s.get("file")]
                if not files:
                    raise ApiError(400, "no_labels", "The table's files carry no variable labels "
                                                     "(only SAS transport files do).")
                book = cb.from_labels(sources, voice.listing(files, limit=50, ticked=False))
            else:
                if staged is not None:
                    source: Path = staged
                    name = client_name or staged.name
                else:
                    if not path:
                        raise ApiError(400, "no_codebook", "Name the codebook's file, or ask for the table's own labels.")
                    source = Path(path).expanduser()
                    if not source.is_absolute():
                        raise ApiError(400, "relative_path", "Give the file's full path, starting from the top of the disk.")
                    if not source.is_file():
                        raise ApiError(404, "no_such_file", f"There is no file at {source}.")
                    name = source.name
                book = cb.read(source, name)
        except cb.CodebookError as exc:
            raise ApiError(400, "unreadable_codebook", str(exc)) from None
        return book

    # ── reading projects ──

    def _ingest_facts(self, pid: str, key: str, stage: str = "ingest") -> IngestFacts:
        """Rows and columns of a table artifact (``ingest``, ``oriented`` or ``working``) by key."""
        cache = f"{pid}#{stage}"
        with self._lock:
            facts = self._facts.get(cache if stage != "ingest" else pid)
        if facts is not None and facts.key == key:
            return facts
        info = read_artifact(self.workspace.cache_dir(pid), stage, key, public=True)
        facts = IngestFacts(
            key=key,
            n_rows=int(info["n_rows"]),
            n_cols=int(info["n_cols"]),
            columns=[str(c["name"]) for c in info["columns"]],
            column_info={
                str(c["name"]): {"dtype": c["dtype"], "n_unique": int(c["n_unique"]),
                                 "n_missing": int(c["n_missing"])}
                for c in info["columns"]
            },
        )
        with self._lock:
            self._facts[cache if stage != "ingest" else pid] = facts
        return facts

    def table_source(self, pid: str, stages: dict[str, StageStatus] | None = None
                     ) -> tuple[str, str] | None:
        """``(stage, key)`` of the table the analysis reads now (M2_CONTRACT §2); None: the raw file.

        The working table when it is fresh. While it recomputes, its newest artifact — the row space
        the newest split and previews were drawn in — unless the table was turned around since, when
        the oriented table names the columns. Before any working table, the oriented one.
        """
        stages = self.engine.status(pid) if stages is None else stages
        cache = self.workspace.cache_dir(pid)

        def current(stage: str) -> str | None:
            status = stages.get(stage)
            return status.key if status is not None and status.status == "fresh" and status.key else None

        working = current("working")
        if working:
            return ("working", working)
        oriented = current("oriented") or latest_key(cache, "oriented")
        older = latest_key(cache, "working")
        if older is not None:
            turned = self._artifact(pid, "working", older, public=True).get("transposed")
            if oriented is None or turned == self._artifact(pid, "oriented", oriented,
                                                             public=True).get("transposed"):
                return ("working", older)
        return ("oriented", oriented) if oriented else None

    def _table_facts(self, pid: str, stages: dict[str, StageStatus], ingest_key: str) -> IngestFacts:
        source = self.table_source(pid, stages)
        if source is None:
            return self._ingest_facts(pid, ingest_key)
        return self._ingest_facts(pid, source[1], stage=source[0])

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
        state = decisions.fold(records)
        return {
            "summary": self.summary(meta, stages["ingest"]),
            "state": state,
            "decisions": records,
            "stages": stages,
            "interview": self.interview(pid, state, stages, records),
        }

    def readings(self, pid: str) -> dict[str, Any]:
        """The readings the values settled with no question asked ("read from your data",
        BLUEPRINT §14.3): read from the roles, outcome and proposals stages (fresh, else the
        newest computed) and the working table's values."""
        from turbotab.core.readings import read_from_data, read_from_values_sentence

        self.workspace.get(pid)
        state = self.log(pid).state()
        items = read_from_data(state, store=self._store_or_none(pid),
                               **{name: self._shown(pid, stage) for name, stage in
                                  (("roles", "roles"), ("target_info", "target_info"),
                                   ("proposals", "proposals"))})
        return {"read_from_data": items, "sentence": read_from_values_sentence(items)}

    def _store_or_none(self, pid: str) -> Any:
        try:
            return self.store(pid)
        except ApiError:
            return None

    def interview(self, pid: str, state: ProjectState, stages: dict[str, StageStatus],
                  records: list[Any]) -> list[InterviewStep]:
        """The Router's answer (turbotab/core/interview.py) for this project now."""
        artifacts: dict[str, Any] = {}
        # What the Router reads, when fresh (WP17: the roles stage's proposals name the groupings
        # the cluster question asks about; WP18: the proposals for the ask card, whose "read from
        # your data" reads the roles, the outcome and the proposals as GET /readings does).
        # The causal lane: its card says whether the causal question is asked or stated; the
        # time-varying lane (V2 causal row) reads whether the exposure changes within units.
        for stage in ("target_info", "oriented", "structure", "roles", "proposals", "causal_design",
                      "time_varying"):
            status = stages.get(stage)
            if status is not None and status.status == "fresh" and status.key:
                artifacts[stage] = self._artifact(pid, stage, status.key, public=True)

        def column_info() -> Any:
            ingest = stages.get("ingest")
            if ingest is None or ingest.status != "fresh" or not ingest.key:
                return None
            return self._table_facts(pid, stages, ingest.key).column_info

        def store() -> Any:
            try:
                return self.store(pid)
            except ApiError:
                return None

        # BLUEPRINT §14.2 (audit WP18): the open question's ask card reads the table's summaries
        # and its whole numbers (cached per column), only for the question it sits in.
        from turbotab.core.ask import AskContext

        ask = AskContext(state, artifacts, column_info=column_info, store=store)
        return route(state, stages, artifacts, records, ask=ask)

    # ── deciding ──

    def decision_context(self, pid: str, stages: dict[str, StageStatus] | None = None) -> DecisionContext:
        stages = self.engine.status(pid) if stages is None else stages
        ingest = stages["ingest"]
        facts = None
        if ingest.status == "fresh" and ingest.key:
            facts = self._table_facts(pid, stages, ingest.key)
        state = self.log(pid).state()
        task = state.task
        info = stages.get("target_info")
        if task is None and info is not None and info.status == "fresh" and info.key:
            detected = self._artifact(pid, "target_info", info.key, public=True)
            if isinstance(detected, dict) and detected.get("column") == state.target:
                task = detected.get("task")

        def store() -> Any:
            try:
                return self.store(pid)
            except ApiError:
                return None

        return DecisionContext(
            columns=facts.columns if facts else None,
            ingest_status=ingest.status,
            ingest_error=ingest.error,
            target=state.target,
            column_info=facts.column_info if facts else None,
            state=state,
            task=task,
            n_rows=facts.n_rows if facts else None,
            store=store,
            artifact=lambda stage: self._fresh(pid, stage, public=True),
            sealed=lambda: self.sealed_rows(pid, stages),
            analyzed=lambda: self.analyzed_rows(pid, stages, self._cohort_ids(pid, stages)),
            records=lambda: self.log(pid).records(),
            interview=lambda: self.interview(pid, state, stages, self.log(pid).records()),
            shown=lambda stage: self._shown(pid, stage),
            project_dir=str(self.workspace.project_dir(pid)),
        )

    def decide(self, pid: str, decision: Any, *, system: bool = False) -> dict[str, Any]:
        """Validate and record ``decision``. ``system``: recorded by the server itself (the
        analysis-plan lock, when an estimate is first displayed), never by a client."""
        self.workspace.get(pid)
        ctx = replace(self.decision_context(pid), sealed_scores=lambda: self._fresh_sealed_scores(pid))
        parsed = decisions.validate(decision, ctx)  # raises Refusal
        if parsed.kind in SYSTEM_KINDS and not system:
            # Audit WP16 (the routing gate's p14): a lock posted by hand before any estimate was
            # displayed marked every later answer as made after the estimates were seen, which was
            # false. The plan locks itself when an estimate is first displayed.
            raise Refusal(
                "plan_locks_itself",
                "The analysis plan is locked by TurboTab itself, the first time an estimate is "
                "displayed: the plan in force then is recorded with its SHA-256, and every later "
                "change is marked as made after the estimates were seen. Locked by hand before "
                "that, answers made before any estimate would be marked as made after one.",
                exits=[{"label": "Answer the remaining questions; the first estimate shown locks "
                                 "the plan", "decision": None}])
        log = self.log(pid)
        facts = SentenceFacts(ctx, parsed, log.records())
        # The log leads the sentence with what had been seen: held-out scores, or the inference
        # estimates (``decisions.disclose``; audit WP16).
        record = log.append(  # raises Refusal for a revert it cannot make
            parsed, sentence=lambda d, before: voice.sentence_for(d, before, facts))
        self.bus.publish(pid, "decision", record.model_dump(mode="json"))
        self.engine.on_decision(pid)
        self._restart_stopped(pid, parsed.kind)
        if parsed.kind == "set_purpose" and parsed.purpose == "inference":
            self._lock_after_prediction(pid)
        return self.view(pid)

    def _restart_stopped(self, pid: str, kind: str) -> None:
        """Recording an answer again is asking for its work again.

        An unchanged answer leaves the stage keys as they were, so a stage the user stopped (or
        one that failed) for that answer would stay stopped; recording it again restarts every
        stage that reads its slot, as ``POST …/stages/{stage}/run`` would.
        """
        slot = decisions.SLOTS.get(kind)
        if slot is None:
            return
        statuses = self.engine.status(pid)
        for stage in self.engine.graph.order():
            status = statuses.get(stage.name)
            if status is None or slot not in stage.reads:
                continue
            if status.cancelled or status.status == "error":
                self.engine.ensure(pid, stage.name)

    # ── previews ──

    def preview(self, pid: str, decision: Any) -> consequences.PreviewResult:
        """What recording ``decision`` would do to this project's data. Nothing is recorded.

        A decision that would be refused is refused here too (409), so an option that
        cannot be taken says why instead of previewing.
        """
        from turbotab.core import (  # noqa: F401 - register the builders
            fact_previews,
            row_previews,
            structure_previews,
        )

        self.workspace.get(pid)
        stages = self.engine.status(pid)
        ctx = self.decision_context(pid, stages)
        parsed = decisions.validate(decision, ctx)
        store = self.store(pid)
        state = ctx.state if ctx.state is not None else self.log(pid).state()

        def artifact(stage: str) -> Any:
            status = stages.get(stage)
            if status is None or status.status != "fresh" or not status.key:
                return None
            return self._artifact(pid, stage, status.key)

        sealed = self.sealed_rows(pid, stages)
        cohort = artifact("cohort")
        cohort_ids = cohort.frames["rows"]["row_id"].to_numpy(dtype="int64") if cohort is not None else None
        training = self.training_rows(pid, stages, sealed, cohort_ids)
        kind = "training"
        if getattr(state, "purpose", None) == "inference" and training is not None:
            # The seal is purpose-scoped (BLUEPRINT §12 ruling 3): under inference the coefficient
            # table is estimated from every analyzed row, so a preview of a modeling choice reads
            # them too, and no row is sealed from it (the methods gate: previews read training rows).
            training, sealed, kind = self.analyzed_rows(pid, stages, cohort_ids), None, "analyzed"
        used: dict[str, int] = {}
        pctx = consequences.PreviewContext(
            project_id=pid,
            state=state,
            datastore=store,
            artifact=artifact,
            training_row_ids=training,
            cohort_row_ids=cohort_ids,
            sealed_row_ids=sealed,
            training_kind=kind,
        )
        pctx.before = lambda: self._before_frame(pid, pctx, stages, used)
        result = consequences.plan(parsed, pctx, basis="")
        result.basis = preview_basis(pctx, result, int(store.n_rows))
        return result

    def sealed_rows(self, pid: str, stages: dict[str, StageStatus]) -> Any:
        """Every held-out row of the newest split, fresh or still recomputing; None before any.

        A changed exclusion or missing-values answer re-runs the split, but the held-out rows are
        drawn over every row with the outcome measured, so they do not move: the newest split's
        sealed rows hold while it recomputes, and no preview reads them meanwhile.
        """
        from turbotab.core import row_previews

        status = stages.get("split")
        key = status.key if status is not None and status.status == "fresh" and status.key else None
        if key is None:
            key = latest_key(self.workspace.cache_dir(pid), "split")
        if key is None:
            return None
        try:
            return row_previews.sealed_rows(self._artifact(pid, "split", key))
        except (OSError, ValueError, KeyError):
            return None

    def training_rows(self, pid: str, stages: dict[str, StageStatus], sealed: Any,
                      cohort_ids: Any) -> Any:
        """The rows models train on: the current cohort's rows outside the seal (None before a
        split exists, or while the cohort itself recomputes)."""
        import numpy as np

        status = stages.get("split")
        if status is not None and status.status == "fresh" and status.key:
            frame = self._artifact(pid, "split", status.key).frames["assignment"]
            return frame.loc[frame["partition"] == "train", "row_id"].to_numpy(dtype="int64")
        if sealed is None or cohort_ids is None:
            return None
        return np.setdiff1d(cohort_ids, np.asarray(sealed, dtype="int64"))

    def _cohort_ids(self, pid: str, stages: dict[str, StageStatus]) -> Any:
        """The fresh cohort's row ids, or None."""
        status = stages.get("cohort")
        if status is None or status.status != "fresh" or not status.key:
            return None
        try:
            frame = self._artifact(pid, "cohort", status.key).frames["rows"]
        except (OSError, ValueError, KeyError):
            return None
        return frame["row_id"].to_numpy(dtype="int64")

    def analyzed_rows(self, pid: str, stages: dict[str, StageStatus], cohort_ids: Any) -> Any:
        """Every row the analysis reads under inference: the split's rows on both sides of the seal
        (the current cohort's while the split recomputes)."""
        import numpy as np

        status = stages.get("split")
        if status is not None and status.status == "fresh" and status.key:
            frame = self._artifact(pid, "split", status.key).frames["assignment"]
            return np.sort(frame["row_id"].to_numpy(dtype="int64"))
        return cohort_ids

    def _before_frame(self, pid: str, ctx: consequences.PreviewContext,
                      stages: dict[str, StageStatus], used: dict[str, int]) -> Any:
        """The sampled working frame: the columns the models would get now, on sampled pool rows."""
        from turbotab.core.stages.rows import predictors

        store = ctx.datastore
        state = ctx.state
        order = [c for c in store.columns if c != decisions.ROW_ID]
        columns = (predictors(state.roles, order, drop=decisions.left_out(state))
                   or [c for c in order if c != state.target])
        n = max(200, min(ctx.sample_size, PREVIEW_CELLS // max(1, len(columns))))
        pool_key = tuple((stages[s].key, stages[s].status) for s in ("split", "cohort") if s in stages)
        # The pool is purpose-scoped (training rows, or every analyzed row under inference).
        key = (pid, str(store.parquet), pool_key, ctx.training_kind, tuple(columns), n)
        with self._lock:
            frame = self._frames.get(key)
            if frame is not None:
                self._frames.move_to_end(key)
        if frame is None:
            frame = store.materialize(columns, ctx.sample_row_ids(n=n))
            with self._lock:
                self._frames[key] = frame
                while len(self._frames) > MAX_CACHED_FRAMES:
                    self._frames.popitem(last=False)
        used["rows"] = len(frame)
        return frame

    # ── finding evidence ──

    def evidence(self, pid: str, finding_id: str) -> consequences.PreviewResult:
        """The views that show why a finding was raised (M1_CONTRACT §12.3). Nothing is recorded."""
        from turbotab.core import evidence

        self.workspace.get(pid)
        stages = self.engine.status(pid)
        status = stages.get("findings")
        if status is None or status.status != "fresh" or not status.key:
            raise ApiError(409, "findings_not_ready",
                           "The findings are still being worked out; their evidence follows them.")
        found = self._artifact(pid, "findings", status.key, public=True)
        finding = next((f for f in (found or {}).get("findings", []) if f.get("id") == finding_id), None)
        if finding is None:
            raise ApiError(404, "unknown_finding", f"There is no finding {finding_id!r} now.")
        store = self.store(pid)
        sealed = self.sealed_rows(pid, stages)
        cohort = stages.get("cohort")
        cohort_ids = None
        if cohort is not None and cohort.status == "fresh" and cohort.key:
            cohort_ids = self._artifact(pid, "cohort", cohort.key).frames["rows"]["row_id"].to_numpy(
                dtype="int64")
        training = self.training_rows(pid, stages, sealed, cohort_ids)
        ctx = evidence.EvidenceContext(state=self.log(pid).state(), datastore=store, sealed=sealed,
                                       training=training)
        return evidence.evidence(finding, ctx)

    def _artifact(self, pid: str, stage: str, key: str, public: bool = False) -> Any:
        """A stage's artifact by key, remembered (an artifact at a key never changes)."""
        cache_key = (pid, f"{stage}#public" if public else stage, key)
        with self._lock:
            if cache_key in self._artifacts:
                self._artifacts.move_to_end(cache_key)
                return self._artifacts[cache_key]
        value = read_artifact(self.workspace.cache_dir(pid), stage, key, public=public)
        with self._lock:
            self._artifacts[cache_key] = value
            while len(self._artifacts) > MAX_CACHED_ARTIFACTS:
                self._artifacts.popitem(last=False)
        return value

    def _shown(self, pid: str, stage: str) -> Any:
        """The stage's newest artifact, fresh or stale, or None when none was ever computed."""
        try:
            return self.engine.get(pid, stage).artifact
        except Exception:  # noqa: BLE001 - nothing computed: nothing was shown
            return None

    def _fresh(self, pid: str, stage: str, public: bool = False) -> Any:
        status = self.engine.status(pid).get(stage)
        if status is None or status.status != "fresh" or not status.key:
            return None
        found = self._artifact(pid, stage, status.key, public=public)
        if stage == "fit" and public and isinstance(found, dict):
            return self._served_fit(pid, found, status.key)
        return found

    def _served_fit(self, pid: str, data: dict[str, Any], key: str | None) -> dict[str, Any]:
        """The fit as a client may see it (M2_CONTRACT §3).

        Held-out scores are withheld (``holdout: null``, ``holdout_sealed: true``) until
        ``open_seal`` is recorded, then served exactly as the fit computed them, with whether this
        fit changed after the opening and the post-seal decisions that changed what it reads.
        """
        records = self.log(pid).records()
        folded = decisions.fold(records)
        opened = bool(folded.seal_opened)
        cache = self.workspace.cache_dir(pid)
        out = seal.serve_fit(data, opened=opened,
                             scores=lambda: seal.read_sealed_scores(cache, key) if key else None,
                             details=lambda: seal.read_sealed_detail(cache, key) if key else None)
        out["changed_after_seal"] = bool(opened and seal.changed_after_seal(
            self.engine.graph, records, self.fingerprint(pid), key))
        out["post_seal_decisions"] = (seal.post_seal_changes(
            records, seal.slots_read_by(self.engine.graph, "fit")) if opened else [])
        # AUDIT_REPORT §5 WP8 (ME-13): the family declared final at the opening is the result.
        from turbotab.core.models.selection import declared_family, mark_final

        out = mark_final(out, opened=opened, family=declared_family(records) if opened else None)
        # Audit WP16 (RO-05): the scores the first opening kept stay the reported result; once the
        # served ones are not those (a fit changed after it, rows drawn again), the note says so.
        at = seal.reported_result(records, opened=opened, changed=out["changed_after_seal"])
        out["at_opening"] = at
        if at is not None and not at["current"] and out.get("final_note"):
            out["final_note"] = at["note"]
        # MS6 (MODELING_SEQUENCE §1 row 12 (b), §4): with no rows held out, the result stands only
        # when every family whose score was shown for this outcome is fitted; the only family
        # fitted is declared before any score was seen only when no other's was ever shown. A stale
        # fit (veiled, inert) may answer another outcome, so only the current one is checked.
        status = self.engine.status(pid).get("fit")
        if status is None or status.status != "fresh" or status.key != key:
            return out
        from turbotab.core.models.selection import read_seen, vouch

        target = folded.target
        return vouch(out, read_seen(self.workspace.project_dir(pid)).get(target or "", []), target)

    def _fresh_sealed_scores(self, pid: str) -> Any:
        """The fresh fit's held-out scores from its sealed frame (None: no fresh fit)."""
        status = self.engine.status(pid).get("fit")
        if status is None or status.status != "fresh" or not status.key:
            return None
        return seal.read_sealed_scores(self.workspace.cache_dir(pid), status.key)

    def _lock_when_shown(self, pid: str, stage: str, artifact: Any) -> None:
        """Under inference, the first estimate served to a client locks the analysis plan: the
        plan in force is recorded as declared before any estimate was displayed, and every later
        decision is marked as made after the estimates were seen (audit WP16, RO-12).

        Under prediction an estimate shown locks nothing (no coefficient is read as an effect),
        but that it was shown is kept beside the project (:data:`SHOWN_UNDER_PREDICTION`): should
        the purpose become inference, the plan is locked at once, as declared after estimates
        were seen (the routing gate's p02: coefficients seen under prediction, the adjustment set
        then chosen under inference, and a lock that said nothing had been displayed)."""
        from turbotab.core import plan_lock

        if stage not in plan_lock.ESTIMATE_STAGES or not plan_lock.shows_estimates(stage, artifact):
            return
        with self._locking:
            records = self.log(pid).records()
            state = decisions.fold(records)
            if state.plan_locked:
                return
            if state.purpose == "prediction":
                path = self.workspace.project_dir(pid) / SHOWN_UNDER_PREDICTION
                if not path.exists():
                    _write_json_atomic(path, {
                        "stage": stage, "target": state.target,
                        "seq": max((r.seq for r in records), default=0)})
                return
            if state.purpose != "inference":
                return
            try:
                self.decide(pid, {"kind": "lock_plan"}, system=True)
            except Refusal:
                log.exception("the analysis plan of %s could not be locked", pid)

    def _lock_after_prediction(self, pid: str) -> None:
        """The purpose just became inference: if estimates were displayed under prediction, the
        plan is locked now, its record saying where they were shown, so every answer after it is
        marked as made after the estimates were seen (Gelman & Loken 2013's forking paths)."""
        path = self.workspace.project_dir(pid) / SHOWN_UNDER_PREDICTION
        try:
            shown = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        with self._locking:
            state = self.log(pid).state()
            if state.purpose != "inference" or state.plan_locked:
                return
            try:
                self.decide(pid, {"kind": "lock_plan", "seen": "prediction",
                                  "seen_target": shown.get("target"),
                                  "seen_at": shown.get("seq")}, system=True)
            except Refusal:
                log.exception("the analysis plan of %s could not be locked", pid)

    def plan(self, pid: str) -> bytes:
        """The analysis-plan export (ESTIMAND; MODELING_SEQUENCE §1 row 12): canonical JSON of the
        plan as the decision log holds it, with its timestamp and SHA-256 (``plan_lock.plan_export``)."""
        from turbotab.core.plan_lock import plan_export

        self.workspace.get(pid)
        return plan_export(self.log(pid).records())

    def methods(self, pid: str) -> Any:
        """The methods text built from the decision log (``turbotab.core.provenance``)."""
        from turbotab.core.provenance import methods_text

        self.workspace.get(pid)
        records = self.log(pid).records()
        # A restated sentence reads what its record's sentence read besides the answers: the
        # detected task, for the families' estimators under the survey answer (MS4).
        facts = SentenceFacts(self.decision_context(pid), None, records)
        return methods_text(records, {"detected_task": facts.detected_task})

    # ── stages and jobs ──

    def stage_result(self, pid: str, stage: str) -> dict[str, Any]:
        self.workspace.get(pid)
        if stage not in self.engine.graph:
            raise ApiError(404, "unknown_stage", f"There is no stage named {stage!r}.")
        result = self.engine.get(pid, stage)
        artifact = result.artifact
        if artifact is not None and not isinstance(artifact, dict):
            artifact = {"value": artifact}
        if stage == "fit" and artifact is not None:
            artifact = self._served_fit(pid, artifact, result.key)
        if artifact is not None and stage in estimand.ESTIMATE_STAGES:
            # WP17: no estimate is served while a question it rests on is unanswered (the follow-up;
            # under inference the grouping, the exposure and its effect, the adjustment set), as
            # held-out scores are withheld until the seal is opened; once answered, the fit is
            # captioned from the estimand.
            records = self.log(pid).records()
            state = decisions.fold(records)
            gate = estimand.served_gate(state, self.interview(pid, state, self.engine.status(pid),
                                                              records))
            if gate is not None:
                artifact = estimand.withhold(stage, artifact, gate)
            elif stage == "fit":
                artifact = estimand.annotate_fit(artifact, state)
        if stage == "findings" and artifact is not None:  # M2 §4: each finding's disposition
            from turbotab.core import repairs

            records = self.log(pid).records()
            artifact = repairs.annotate(artifact, decisions.fold(records), records)
        self._lock_when_shown(pid, stage, artifact)
        if stage in ("fit", "explain") and result.fresh and isinstance(artifact, dict):
            # MS6: the families whose cross-validated scores this client now sees, for its outcome,
            # kept beside the project; a revert or a new seed cannot unsee them. The explanations'
            # floors quote the same scores (wave 2a's EXPLAIN), so serving them counts too.
            from turbotab.core.models.selection import explained_in, note_seen, scored_in

            note_seen(self.workspace.project_dir(pid), self.log(pid).state().target,
                      scored_in(artifact) if stage == "fit" else explained_in(artifact))
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
        """The DataStore over the table the analysis reads (``table_source``): the working table,
        else the oriented one, else the raw file. Answers only once the ingest stage is fresh."""
        self.workspace.get(pid)
        ingest = self.engine.status(pid)["ingest"]
        if ingest.status != "fresh" or not ingest.key:
            if ingest.status == "error":
                raise ApiError(409, "ingest_failed", f"The table could not be read. {ingest.error or ''}".strip())
            raise ApiError(409, "table_not_ready", "The table is still being read.")
        source = self.table_source(pid)
        if source is not None:
            path = (artifact_dir(self.workspace.cache_dir(pid), *source) / "files" / "table.parquet").resolve()
        else:  # the table the fresh ingest describes (a joined table has a file of its own)
            path = self._ingested_table(pid, ingest.key)
        tag = f"{ingest.key}:{path}"
        with self._lock:
            cached = self._stores.get(pid)
            if cached is not None and cached[0] == tag:
                return cached[1]
        store = DataStore(path, int(self.settings.memory_budget_bytes))
        with self._lock:  # a request still reading the previous table keeps its own reference
            self._stores[pid] = (tag, store)
        return store

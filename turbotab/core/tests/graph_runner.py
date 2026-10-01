"""Run the real stage graph in-process, the way the workers do, for one state.

Every stage whose required slots are set runs in dependency order: its inputs are read back from a
real artifact cache (so Bundle files are moved and resolved exactly as in a worker) and its result
is written there. No engine, no processes: cheap enough for a test to call per fixture.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from turbotab.core.datastore import ingest
from turbotab.core.decisions import ProjectState
from turbotab.core.graph import StageContext, read_artifact, write_artifact

BUDGET = 2 << 30


class GraphRun:
    """A project folder (``data/raw.parquet`` + ``cache/``) over one source file."""

    def __init__(self, source: Path, folder: Path):
        from turbotab.core.stages import build_graph

        self.source = Path(source)
        self.folder = Path(folder)
        (self.folder / "data").mkdir(parents=True, exist_ok=True)
        self.cache = self.folder / "cache"
        self.graph = build_graph()
        self.raw = self.folder / "data" / "raw.parquet"
        self.info = ingest(self.source, self.raw).to_dict()
        self.seconds: dict[str, float] = {}
        self._n = 0

    def paths(self) -> dict[str, str]:
        return {"data": str(self.raw), "source": str(self.source), "project_dir": str(self.folder)}

    def run(self, state: ProjectState, upto: Iterable[str] | None = None,
            before: Callable[[str], Any] | None = None) -> dict[str, Any]:
        """Each runnable stage's full artifact (Bundles whole), keyed by stage name.

        ``before(stage)`` is called just before each stage runs (a spy's hook).
        """
        self._n += 1
        tag = f"r{self._n}"
        wanted = set(upto) if upto is not None else None
        if wanted is not None:
            for name in list(wanted):
                wanted |= self.graph.upstream(name)
        out: dict[str, Any] = {"ingest": self.info}
        write_artifact(self.cache, "ingest", tag, 1, self.info)
        values = state.model_dump()
        for stage in self.graph.order():
            if stage.name == "ingest" or (wanted is not None and stage.name not in wanted):
                continue
            if any(values.get(slot) is None for slot in stage.requires):
                continue
            if any(dep not in out for dep in stage.deps):
                continue
            ctx = StageContext(
                project_id="test", state=state,
                inputs={dep: read_artifact(self.cache, dep, tag) for dep in stage.deps},
                paths=self.paths(), settings={"memory_budget_bytes": BUDGET})
            if before is not None:
                before(stage.name)
            started = time.perf_counter()
            result = stage.fn(ctx)
            self.seconds[stage.name] = time.perf_counter() - started
            write_artifact(self.cache, stage.name, tag, stage.version, result)
            out[stage.name] = read_artifact(self.cache, stage.name, tag)
        return out

    def public(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        """What a client (and the Router) sees of each artifact."""
        return {k: getattr(v, "data", v) for k, v in artifacts.items()}

    def close(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

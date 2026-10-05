"""A preview the way the server plans one, over a project run by the real stage graph.

``Project`` ingests a table and runs the stage graph in-process (``graph_runner.GraphRun``) for the
recorded state; :meth:`Project.preview` builds the ``PreviewContext`` exactly as
``ProjectService.preview`` does (the working table's store, the purpose-scoped pool: the training
rows under prediction, every analyzed row under inference, held-out rows sealed only under
prediction) and plans the decision; :meth:`Project.after` runs the graph again with the decision
recorded, so a test can hold the preview's numbers to the stage's (``test_previews_3_truth.py``).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from turbotab.core import consequences, decisions
from turbotab.core.datastore import DataStore
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.graph_runner import BUDGET, GraphRun


def working_table(artifacts: dict[str, Any]) -> Path:
    from turbotab.core.stages.working import _bundle_table

    for stage in ("working", "oriented"):
        if stage in artifacts:
            found = _bundle_table(artifacts[stage])
            if found is not None:
                return Path(found)
    raise RuntimeError("no working table")


class Project:
    def __init__(self, frame: pd.DataFrame | Path, folder: Path, state: ProjectState,
                 upto: Iterable[str] | None = None, project_dir: Path | None = None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if isinstance(frame, pd.DataFrame):
            source = folder / "table.csv"
            frame.to_csv(source, index=False)
        else:
            source = Path(frame)
        self.run = GraphRun(source, folder / "project")
        self.upto = list(upto) if upto is not None else None
        self.state = state
        self.project_dir = project_dir
        self.artifacts = self.run.run(state, upto=self.upto)
        self.store = DataStore(working_table(self.artifacts), BUDGET)

    # ── the context, as ProjectService.preview builds it ──

    def context(self, state: ProjectState | None = None, artifacts: dict[str, Any] | None = None,
                sample_size: int = 5000) -> consequences.PreviewContext:
        from turbotab.core import row_previews

        state = state or self.state
        arts = artifacts if artifacts is not None else self.artifacts
        split = arts.get("split")
        cohort = arts.get("cohort")
        sealed = row_previews.sealed_rows(split) if split is not None else None
        cohort_ids = (cohort.frames["rows"]["row_id"].to_numpy(dtype="int64")
                      if cohort is not None else None)
        training, kind = None, "training"
        if split is not None:
            a = split.frames["assignment"]
            training = a.loc[a["partition"] == "train", "row_id"].to_numpy(dtype="int64")
            if state.purpose == "inference":
                training = np.sort(a["row_id"].to_numpy(dtype="int64"))
                sealed, kind = None, "analyzed"
        settings: dict[str, Any] = {}
        if self.project_dir is not None:
            settings["project_dir"] = str(self.project_dir)
        return consequences.PreviewContext(
            project_id="test", state=state, datastore=self.store,
            artifact=lambda stage: arts.get(stage), training_row_ids=training,
            cohort_row_ids=cohort_ids, sealed_row_ids=sealed, training_kind=kind,
            sample_size=sample_size, settings=settings)

    def preview(self, decision: Any, *, sample_size: int = 5000,
                ctx: consequences.PreviewContext | None = None) -> tuple[Any, Any]:
        """``(result, ctx)``: the planned preview, its basis worded as the server words it."""
        from turbotab.server.service import preview_basis

        ctx = ctx or self.context(sample_size=sample_size)
        result = consequences.plan(decision, ctx, basis="")
        result.basis = preview_basis(ctx, result, int(self.store.n_rows))
        return result, ctx

    def timed(self, decision: Any, repeats: int = 20, *, sample_size: int = 5000) -> list[float]:
        """Seconds per preview over ``repeats`` plans, each with a fresh context (a hover)."""
        out = []
        for _ in range(repeats):
            ctx = self.context(sample_size=sample_size)
            started = time.perf_counter()
            consequences.plan(decision, ctx, basis="")
            out.append(time.perf_counter() - started)
        return out

    # ── the decision recorded ──

    def after(self, decision: Any, upto: Iterable[str] | None = None) -> tuple[ProjectState, dict[str, Any]]:
        state = decisions.fold_onto(self.state, decision)
        return state, self.run.run(state, upto=list(upto) if upto is not None else self.upto)

    def close(self) -> None:
        self.store.close()
        self.run.close()


def p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def view(result: Any, kind: str, index: int = 0) -> Any:
    found = [v for v in result.views if v.kind == kind]
    assert len(found) > index, (kind, [v.kind for v in result.views], result.note)
    return found[index]

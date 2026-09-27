"""The M0 stage graph (docs/turbotab-next/BLUEPRINT.md §4).

``build_graph`` is the engine's ``graph_factory``
(``"turbotab.core.stages:build_graph"``): the server builds the graph from it,
and so does every worker process that runs a heavy stage.

    ingest       heavy   deps: —        DatasetInfo
    profile      heavy   deps: ingest   column summaries + lens hints
    target_info  light   deps: ingest   reads target, task; requires target
    findings     heavy   deps: ingest   reads lens, target; requires lens

Each stage is a pure function of its inputs and the slots it reads. The
statistics are the data layer's and the legacy domain code's; the stages only
call them and shape the result into the contract's artifact.
"""
from __future__ import annotations

from turbotab.core.graph import Graph, Stage
from turbotab.core.stages.data import ingest_stage, profile_stage
from turbotab.core.stages.findings import findings_stage
from turbotab.core.stages.target import target_info_stage

GRAPH_FACTORY = "turbotab.core.stages:build_graph"


def build_graph() -> Graph:
    return Graph(
        [
            Stage("ingest", 1, (), (), ingest_stage, heavy=True, label="Reading the file"),
            Stage(
                "profile",
                1,
                ("ingest",),
                (),
                profile_stage,
                heavy=True,
                label="Summarizing every column",
            ),
            Stage(
                "target_info",
                1,
                ("ingest",),
                ("target", "task"),
                target_info_stage,
                requires=("target",),
                label="Reading the outcome column",
            ),
            Stage(
                "findings",
                1,
                ("ingest",),
                ("lens", "target"),
                findings_stage,
                heavy=True,
                requires=("lens",),
                label="Checking the table against the chosen lenses",
            ),
        ]
    )


__all__ = ["GRAPH_FACTORY", "build_graph"]

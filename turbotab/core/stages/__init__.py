"""The M0 stage graph (docs/turbotab-next/BLUEPRINT.md §4).

``build_graph`` is the engine's ``graph_factory``
(``"turbotab.core.stages:build_graph"``): the server builds the graph from it,
and so does every worker process that runs a heavy stage.

    ingest       heavy   deps: —        DatasetInfo
    profile      heavy   deps: ingest   column summaries + lens hints
    target_info  light   deps: ingest   reads target, task; requires target
    findings     heavy   deps: ingest   reads lens, target; requires lens

M1 (docs/turbotab-next/M1_CONTRACT.md):

    roles        heavy   deps: ingest, profile                reads lens, target
    proposals    light   deps: ingest, profile, roles         reads lens, roles, target
    cohort       heavy   deps: ingest, target_info            reads target, roles, exclusions, missing; requires target
    split        heavy   deps: cohort, target_info            reads split, roles, task; requires split
    shelf        light   deps: cohort, target_info            reads purpose, task, roles; requires roles
    design       heavy   deps: split, target_info             reads roles, energy_adjustment, missing, models, purpose; requires models, roles
    fit          heavy   deps: design, split, target_info     reads models, purpose, task; requires models
    substitution heavy   deps: fit, design                    reads substitution; requires substitution

Each stage is a pure function of its inputs and the slots it reads. The
statistics are the data layer's and the legacy domain code's; the stages only
call them and shape the result into the contract's artifact.
"""
from __future__ import annotations

from turbotab.core.graph import Graph, Stage
from turbotab.core.stages.data import ingest_stage, profile_stage
from turbotab.core.stages.findings import findings_stage
from turbotab.core.stages.modeling import design_stage, fit_stage, shelf_stage, substitution_stage
from turbotab.core.stages.proposals import proposals_stage
from turbotab.core.stages.rows import cohort_stage, roles_stage, split_stage
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
            # ── M1 ──
            Stage("roles", 1, ("ingest", "profile"), ("lens", "target"), roles_stage,
                  heavy=True, label="Reading what each column is"),
            Stage("proposals", 1, ("ingest", "profile", "roles"), ("lens", "roles", "target"),
                  proposals_stage, label="Looking up what the field usually does"),
            Stage("cohort", 1, ("ingest", "target_info"),
                  ("target", "roles", "exclusions", "missing"), cohort_stage,
                  heavy=True, requires=("target",), label="Counting who is in the analysis"),
            Stage("split", 1, ("cohort", "target_info"), ("split", "roles", "task"), split_stage,
                  heavy=True, requires=("split",), label="Drawing the held-out rows"),
            Stage("shelf", 1, ("cohort", "target_info"), ("purpose", "task", "roles"), shelf_stage,
                  requires=("roles",), label="Ranking the model families for this table"),
            Stage("design", 1, ("split", "target_info"),
                  ("roles", "energy_adjustment", "missing", "models", "purpose"), design_stage,
                  heavy=True, requires=("models", "roles"),
                  label="Building each model's pipeline"),
            Stage("fit", 1, ("design", "split", "target_info"), ("models", "purpose", "task"),
                  fit_stage, heavy=True, requires=("models",), label="Fitting the models"),
            Stage("substitution", 1, ("fit", "design"), ("substitution",), substitution_stage,
                  heavy=True, requires=("substitution",), label="Drawing the substitution curves"),
        ]
    )


__all__ = ["GRAPH_FACTORY", "build_graph"]

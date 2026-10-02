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
    cohort       heavy   deps: ingest, target_info            reads target, roles, exclusions, missing, findings; requires target
    split        heavy   deps: cohort, target_info            reads split, roles, task; requires split
    shelf        light   deps: cohort, target_info            reads purpose, task, roles; requires roles
    design       heavy   deps: split, target_info             reads roles, energy_adjustment, missing, models, purpose,
                                                              exposure_forms; requires models, roles
    fit          heavy   deps: design, split, target_info     reads models, purpose, task, outcome_order; requires models
    substitution heavy   deps: fit, design                    reads substitution; requires substitution

M2 (docs/turbotab-next/M2_CONTRACT.md §2) — the table the analysis reads:

    oriented     heavy   deps: ingest                         reads orientation, feature_table
    structure    heavy   deps: oriented                       reads grain, target, lens, repeat_kind,
                                                              findings (the date-reading repair)
    working      heavy   deps: oriented, findings, structure  reads findings, target, grain, unit,
                                                              aggregation, repeat_kind

    findings and profile read the oriented table; target_info, roles, proposals, cohort, split,
    shelf, seal_plan, design, fit and substitution read the working table
    (``stages.working.table_path``).

M2, the seal (M2_CONTRACT.md §3): ``split`` also reads the grain, unit, aggregation, temporal and
repeat_kind answers (its basis and the chronological draw); ``shelf`` ranks on the training rows,
so it waits for the split; ``fit`` keeps its held-out scores out of its public data.

    seal_plan    light   deps: working, cohort, target_info, structure   reads roles, task + the seal's; requires target

M2 part 2 (M2_CONTRACT.md §12): ``split`` and ``seal_plan`` also depend on ``structure``, whose
``grain.stated`` is the grain when the Router states it rather than asks (a unique person
identifier); the seal needs a grain, answered or stated. ``shelf`` is heavy: it times one fit of
each family on a sample of the training rows, so each family carries ``estimate_seconds``.

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
from turbotab.core.stages.seal import SEAL_READS, seal_plan_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.stages.working import oriented_stage, structure_stage, working_stage

GRAPH_FACTORY = "turbotab.core.stages:build_graph"


def build_graph() -> Graph:
    return Graph(
        [
            Stage("ingest", 1, (), (), ingest_stage, heavy=True, label="Reading the file"),
            # ── M2: what the table is (M2_CONTRACT §2) ──
            Stage("oriented", 2, ("ingest",), ("orientation", "feature_table"), oriented_stage,
                  heavy=True, label="Reading which way round the table is"),
            Stage(
                "profile",
                1,
                ("oriented",),
                (),
                profile_stage,
                heavy=True,
                label="Summarizing every column",
            ),
            Stage(
                "findings",
                4,
                ("oriented",),
                ("lens", "target"),
                findings_stage,
                heavy=True,
                requires=("lens",),
                label="Checking the table against the chosen lenses",
            ),
            # structure reads ``findings`` for the date-reading repair: a date column that reads
            # both month-first and day-first is read only once that is answered (audit MA-05).
            Stage("structure", 4, ("oriented",),
                  ("grain", "target", "lens", "repeat_kind", "findings"),
                  structure_stage, heavy=True, label="Reading how the rows repeat"),
            Stage("working", 2, ("oriented", "findings", "structure"),
                  ("findings", "target", "grain", "unit", "aggregation", "repeat_kind"),
                  working_stage, heavy=True, label="Building the working table"),
            Stage(
                "target_info",
                2,
                ("working",),
                ("target", "task"),
                target_info_stage,
                requires=("target",),
                label="Reading the outcome column",
            ),
            # ── M1 (each reads the working table) ──
            Stage("roles", 1, ("working",), ("lens", "target"), roles_stage,
                  heavy=True, label="Reading what each column is"),
            Stage("proposals", 1, ("working", "roles"), ("lens", "roles", "target"),
                  proposals_stage, label="Looking up what the field usually does"),
            Stage("cohort", 1, ("working", "target_info"),
                  ("target", "roles", "exclusions", "missing", "findings"), cohort_stage,
                  heavy=True, requires=("target",), label="Counting who is in the analysis"),
            Stage("split", 3, ("working", "cohort", "target_info", "structure"),
                  ("split", "roles", "task", *SEAL_READS), split_stage, heavy=True,
                  requires=("split",), label="Drawing the held-out rows"),
            Stage("shelf", 4, ("working", "cohort", "target_info", "split"),
                  ("purpose", "task", "roles", "missing", "categorical", "outcome_order",
                   "exposure_forms"), shelf_stage, heavy=True,
                  requires=("roles",), label="Ranking the model families for this table"),
            Stage("design", 3, ("working", "split", "target_info"),
                  ("roles", "energy_adjustment", "missing", "models", "purpose", "categorical",
                   "exposure_forms"),
                  design_stage,
                  heavy=True, requires=("models", "roles"),
                  label="Building each model's pipeline"),
            Stage("fit", 5, ("working", "design", "split", "target_info"),
                  ("models", "purpose", "task", "event", "outcome_order"), fit_stage, heavy=True,
                  requires=("models",),
                  label="Fitting the models"),
            Stage("substitution", 3, ("working", "fit", "design"),
                  ("substitution", "event", "outcome_order"),
                  substitution_stage, heavy=True, requires=("substitution",),
                  label="Drawing the substitution curves"),
            # ── M2: the seal (docs/turbotab-next/M2_CONTRACT.md §3) ──
            Stage("seal_plan", 2, ("working", "cohort", "target_info", "structure"),
                  ("roles", "task", *SEAL_READS), seal_plan_stage, requires=("target",),
                  label="Reading what a held-out set can measure"),
        ]
    )

__all__ = ["GRAPH_FACTORY", "build_graph"]

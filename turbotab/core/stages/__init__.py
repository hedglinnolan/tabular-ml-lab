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
    proposals    light   deps: ingest, profile, roles         reads lens, roles, target, purpose
    cohort       heavy   deps: ingest, target_info            reads target, roles, exclusions, missing, findings; requires target
    split        heavy   deps: cohort, target_info            reads split, roles, task; requires split
    shelf        light   deps: cohort, target_info            reads purpose, task, roles; requires roles
    design       heavy   deps: split, target_info             reads roles, energy_adjustment, missing, models, purpose, event,
                                                              exposure_forms; requires models, roles
    fit          heavy   deps: design, split, target_info     reads models, purpose, task, survey, outcome_order,
                                                              follow_up; requires models
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

Audit WP12 (AUDIT_REPORT §5): a time-to-event outcome's follow-up (``set_follow_up``) is read by
``design`` (a follow-up column is never a predictor) and ``fit`` (the outcome is the event with
its follow-up); ``shelf`` and ``seal_plan`` read the event, which they count for such an outcome.

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
            # proposals 2: the declared purpose orders the energy methods by soundness (audit WP6).
            Stage("proposals", 2, ("working", "roles"), ("lens", "roles", "target", "purpose"),
                  proposals_stage, label="Looking up what the field usually does"),
            Stage("cohort", 1, ("working", "target_info"),
                  ("target", "roles", "exclusions", "missing", "findings"), cohort_stage,
                  heavy=True, requires=("target",), label="Counting who is in the analysis"),
            Stage("split", 3, ("working", "cohort", "target_info", "structure"),
                  ("split", "roles", "task", *SEAL_READS), split_stage, heavy=True,
                  requires=("split",), label="Drawing the held-out rows"),
            Stage("shelf", 6, ("working", "cohort", "target_info", "split"),
                  ("purpose", "task", "roles", "missing", "categorical", "lens", "findings", "event",
                   "outcome_order", "exposure_forms"),
                  shelf_stage, heavy=True,
                  requires=("roles",), label="Ranking the model families for this table"),
            # design 6: the estimand and coefficient meanings are read off the matrix, and the
            # energy-dropped residual's gap reads the outcome and its event (audit WP6); an omics
            # normalization step reads the lens and the findings (audit WP11); each formed
            # exposure's spline or quintiles (audit WP12a); a follow-up is no predictor (WP12b).
            Stage("design", 6, ("working", "split", "target_info"),
                  ("roles", "energy_adjustment", "missing", "models", "purpose", "categorical",
                   "event", "lens", "findings", "exposure_forms", "follow_up"),
                  design_stage,
                  heavy=True, requires=("models", "roles"),
                  label="Building each model's pipeline"),
            # fit 8: the merged fit (WP8's every-row table, WP9's validation, WP10's survey design,
            # WP11's feature-wise tests, WP12a's ordinal outcome and exposure tests, WP12b's
            # follow-up and the families that model the unit).
            Stage("fit", 8, ("working", "design", "split", "target_info"),
                  ("models", "purpose", "task", "event", "survey", "outcome_order", "follow_up"),
                  fit_stage, heavy=True, requires=("models",),
                  label="Fitting the models"),
            # substitution 5: a swap can move a share of energy (WP12a); a random intercept's band
            # refits one intercept per resampled unit (WP12b).
            Stage("substitution", 5, ("working", "fit", "design"),
                  ("substitution", "event", "outcome_order"),
                  substitution_stage, heavy=True, requires=("substitution",),
                  label="Drawing the substitution curves"),
            # ── M2: the seal (docs/turbotab-next/M2_CONTRACT.md §3) ──
            Stage("seal_plan", 2, ("working", "cohort", "target_info", "structure"),
                  ("roles", "task", "event", *SEAL_READS), seal_plan_stage, requires=("target",),
                  label="Reading what a held-out set can measure"),
        ]
    )

__all__ = ["GRAPH_FACTORY", "build_graph"]

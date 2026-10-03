"""The M0 stage graph (docs/turbotab-next/BLUEPRINT.md §4).

``build_graph`` is the engine's ``graph_factory``
(``"turbotab.core.stages:build_graph"``): the server builds the graph from it,
and so does every worker process that runs a heavy stage.

    ingest       heavy   deps: —        DatasetInfo
    profile      heavy   deps: ingest   column summaries + lens hints
    target_info  light   deps: ingest   reads target, task, outcome_unit; requires target
    findings     heavy   deps: ingest   reads lens, target; requires lens

M1 (docs/turbotab-next/M1_CONTRACT.md):

    roles        heavy   deps: ingest, profile                reads lens, target, purpose
    proposals    light   deps: ingest, profile, roles         reads lens, roles, target, purpose
    cohort       heavy   deps: ingest, target_info            reads target, roles, exclusions, missing, findings; requires target
    split        heavy   deps: cohort, target_info            reads split, roles, task; requires split
    shelf        light   deps: cohort, target_info            reads purpose, task, roles; requires roles
    design       heavy   deps: split, target_info             reads roles, energy_adjustment, missing, models, purpose, event,
                                                              exposure_forms; requires models, roles
    fit          heavy   deps: design, split, target_info,    reads models, purpose, task, survey, outcome_order,
                               cohort                         follow_up; requires models
    substitution heavy   deps: fit, design                    reads substitution, purpose; requires substitution

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
repeat_kind answers (its basis and the chronological draw); ``shelf`` ranks on the training rows
(every analyzed row under inference, BLUEPRINT §12 ruling 3), so it waits for the split; ``fit``
keeps its held-out scores out of its public data.

    seal_plan    light   deps: working, cohort, target_info, structure   reads roles, task + the seal's; requires target

WP12 (AUDIT_REPORT §5, methods a reviewer expects):

    sensitivity  heavy   deps: working, design, split, target_info          reads sensitivity, exclusions + the
                                                                             cohort's and fit's; requires sensitivity, models
    calibration  heavy   deps: oriented, findings, structure, working,      reads measurement_error, energy_adjustment,
                               cohort, design, target_info                   aggregation, purpose …; requires measurement_error, models

    ``sensitivity`` refits each chosen family on the rows each analysis's exclusion rules keep (the
    primary beside every-row and any other screen; Banna et al. 2017). ``calibration`` corrects
    energy-adjusted exposures for day-to-day error in the recalls each person's row averages
    (univariate regression calibration; Freedman et al. 2011): it reads the oriented table's rows
    behind each working row through the working table's row map.

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

# The roles and what the leash records beside them (BLUEPRINT §14): a stage that reads a
# number-changing default from the roles reads which of them are settled too.
# The readings ledger (BLUEPRINT §14.1) adds each reading's own confirmation (``confirm_reading``).
ROLE_READS: tuple[str, ...] = ("roles", "roles_unconfirmed", "role_confirmations",
                               "reading_confirmations", "shape_confirmations")
from turbotab.core.stages.calibration import CALIBRATION_READS, calibration_stage
from turbotab.core.stages.sensitivity import SENSITIVITY_READS, sensitivity_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.stages.working import oriented_stage, structure_stage, working_stage

GRAPH_FACTORY = "turbotab.core.stages:build_graph"


def build_graph() -> Graph:
    return Graph(
        [
            Stage("ingest", 1, (), (), ingest_stage, heavy=True, label="Reading the file"),
            # ── M2: what the table is (M2_CONTRACT §2) ──
            # oriented 3 (WP14): the names are read before the shape, and the shape is scale-aware.
            Stage("oriented", 3, ("ingest",), ("orientation", "feature_table"), oriented_stage,
                  heavy=True, label="Reading which way round the table is"),
            # profile 2 (WP14): lens hints need positive evidence (turbotab.core.detectors.lenses).
            # profile 3 (integration): the dietary hint reads total energy by the one recognizer
            # (audit IN-08: DR2TKCAL, ENERC_KCAL, TotalKcal …).
            # profile 4 (repair round): a wide table hints genomics only by a library-size
            # signature, never by "log-scale, normalization not recoverable" (IN-13).
            # profile 5 (gate repair): whole numbers hint genomics only when their features' mean
            # counts span orders of magnitude, as genes' do (an FFQ's codes, portions, minute
            # counts do not).
            # profile 6 (recognition's leash, BLUEPRINT §14): a count matrix holds the small counts
            # a sequencer records and never a column named in grams; the dietary hint needs a day's
            # energy in the median.
            Stage(
                "profile",
                6,
                ("oriented",),
                (),
                profile_stage,
                heavy=True,
                label="Summarizing every column",
            ),
            # findings 5 (WP13): identifiers, kJ and NHANES weights read by the one recognizer;
            # the pack's energy findings for energy names only the recognizer reads.
            # findings 5 (WP14): the detectors that fired on clean data are read by
            # turbotab.core.detectors (codes, survey scales, drift, redundancy, plausibility,
            # genomics data type).
            # findings 6 (WP13 + WP14 merged): both of the above in one stage.
            # findings 5 (audit WP15): the energy finding states the estimand choice and, for an
            # energy-related outcome, the dispute; TNTC reads as right-censored, not a failure.
            # findings 7 (WP13 + WP14 + WP15 merged): all of the above in one stage.
            # findings 8 (integration): under the genomics lens an expression matrix the data-type
            # card reads keeps its wide shape (IN-14); outcome names by the one tokenizer.
            # findings 9 (repair round): nutrients corroborated by their values, total energy read
            # as intake only, a screen that would remove most rows asks the unit, the OGTT and
            # pre-pandemic NHANES weights, codes beside the answers, ages by whole words, voom, and
            # the current BAM edition.
            # findings 10 (gate repair): the energy findings are about the column whose values
            # follow the macronutrients; an energy unit only proposed, or an age's unit the body
            # sizes contradict, is asked, not judged in; NHANES subsample variables by their
            # missingness; an outcome nothing places states the dispute as a condition; the
            # recorded column units (``set_column_unit``) are read.
            # findings 11 (recognition's leash, BLUEPRINT §14): the energy finding names only
            # nutrients the values corroborate (r ≥ 0.3, one per nutrient and occasion); a day count
            # in an energy name is asked; the outcome's dispute reads value-corroborated readings.
            # findings 12 (the readings ledger, BLUEPRINT §14.1): an energy column only its name reads
            # counts no misreport, and a day count the values do not settle is asked.
            Stage(
                "findings",
                12,
                ("oriented",),
                ("lens", "target", "column_units"),
                findings_stage,
                heavy=True,
                requires=("lens",),
                label="Checking the table against the chosen lenses",
            ),
            # structure reads ``findings`` for the date-reading repair: a date column that reads
            # both month-first and day-first is read only once that is answered (audit MA-05).
            # structure 5 (WP13): the grain question never suggests a measurement as the unit.
            # structure 5 (WP14): repeats are stated only when unambiguous.
            # structure 6 (WP13 + WP14 merged): both of the above in one stage.
            # structure 7 (repair round): spacing alone never states repeats; a two-record change
            # is time-point evidence (IN-12).
            # structure 8 (gate repair): same-date records ordered within the day, and a regular
            # schedule nothing names as visits (or intakes under the dietary lens), are asked.
            # structure 9 (recognition's leash, BLUEPRINT §14): a date constant within units is never
            # spacing evidence; a recall index beside an occasion that changes within units is asked.
            # structure 10 (the readings ledger, BLUEPRINT §14.1): the time column is the settled
            # one (the reading's is proposed); the whole numbers that may be codes or counts.
            Stage("structure", 10, ("oriented",),
                  ("grain", "target", "lens", "repeat_kind", "findings", "temporal",
                   "shape_confirmations"),
                  structure_stage, heavy=True, label="Reading how the rows repeat"),
            # working 3 (the readings ledger): a code or a count is combined only as the user said,
            # and first, last and change only by a settled time column.
            Stage("working", 3, ("oriented", "findings", "structure"),
                  ("findings", "target", "grain", "unit", "aggregation", "repeat_kind", "temporal",
                   "shape_confirmations", "categorical"),
                  working_stage, heavy=True, label="Building the working table"),
            # target_info 3 (WP13, audit IN-05): the unit is stated only as recorded or spelled out
            # by the name; the clinical pack's reading is a proposal.
            # target_info 4 (gate repair): a unit is stated only from a whole suffix (``protein_g_kg``
            # is g/kg, not kg; ``wbc_k_ul`` thousands per µL, not U/L).
            # target_info 5 (the readings ledger, BLUEPRINT §14.1): three or more labels are asked
            # (ordered or not); a bare amount the quantity does not take is no stated unit.
            Stage(
                "target_info",
                5,
                ("working",),
                ("target", "task", "outcome_unit"),
                target_info_stage,
                requires=("target",),
                label="Reading the outcome column",
            ),
            # ── M1 (each reads the working table) ──
            # roles 2 (WP13): whole-token recognizers; a study's arms are exposures, a site or
            # household a cluster, and a batch's proposed role follows the declared purpose.
            # roles 3 (repair round): a nutrient name the values contradict is no nutrient; an arm
            # named ``arm_id`` is an exposure; acquisition and survey-weight names read whole.
            # roles 4 (gate repair): confidence says how much was checked; total energy by its values
            # against the macronutrients; ratios (g/kg, g/1000 kcal) no day's amount; an ``_id``
            # with a few values a code for groups; acquisition and sampling-weight names corroborated
            # by the table (an assay lens, a survey design).
            # roles 5 (recognition's leash, BLUEPRINT §14): "high" only where the values corroborate,
            # codebook names included; every proposal below high carries ``attention`` and the
            # payload lists ``needs_confirmation``.
            # roles 6 (the readings ledger, BLUEPRINT §14.1): a repeating ``*_id`` read by its name
            # (a stratum, a PSU, an interviewer) is medium, unless the grain answer names it the unit.
            Stage("roles", 6, ("working",), ("lens", "target", "purpose", "grain"), roles_stage,
                  heavy=True, label="Reading what each column is"),
            # proposals 3: the declared purpose orders the energy methods by soundness (audit WP6);
            # the survey question (WP10) and the Goldberg screen's recall days (WP12c).
            # proposals 4: the missing-data methods ordered by purpose (audit WP7).
            # proposals 5 (repair round): the Goldberg screen reads body measures left out of the
            # model; the exposure-form options with their two labels (WP12a).
            # proposals 6 (methods gate): Willett's sex-specific screen reads sex left out of the
            # model, as the Goldberg screen does.
            # proposals 7 (WP13): nutrients and energy by the one recognizer; the energy unit by
            # its suffix, the Atwater reconstruction, or the pack's magnitude prior (audit IN-07).
            # proposals 7 (audit WP15): the sex-specific screens attributed to their sources
            # (Willett 2013, NHS/HPFS); an energy-related outcome's DISPUTED note on the energy card.
            # proposals 8 (WP13 + WP15 merged): both of the above in one stage.
            # proposals 9 (integration): the energy card reads the outcome by the one tokenizer.
            # proposals 10 (repair round): the energy column is intake by name and median; nutrients
            # are corroborated; a screen removing most rows is refused (IN-07).
            # proposals 11 (gate repair): total energy corroborated by its values; a unit only
            # proposed refuses the screens until ``set_column_unit`` records it; an outcome nothing
            # places states the dispute as a condition; subsample weights read by missingness.
            # proposals 12 (recognition's leash, BLUEPRINT §14): the energy card, the screens and the
            # survey options read settled roles only; a day count in an energy name is asked.
            # proposals 13 (the readings ledger, BLUEPRINT §14.1): a day count is settled by the
            # values or recorded (the Atwater identity says nothing of days); the Goldberg screen's
            # body measures and the sex-specific screens' sex column wait for settled units and
            # roles; a design the user set by role is offered, placed by its values.
            Stage("proposals", 13, ("working", "roles"),
                  ("lens", *ROLE_READS, "target", "purpose", "column_units", "repeat_kind"),
                  proposals_stage, label="Looking up what the field usually does"),
            # cohort 2: the rows complete cases drop beside those they keep (audit WP7, E14).
            # cohort 3 (the readings ledger, BLUEPRINT §14.1): complete cases read settled roles.
            Stage("cohort", 3, ("working", "target_info"),
                  ("target", *ROLE_READS, "exclusions", "missing", "findings"), cohort_stage,
                  heavy=True, requires=("target",), label="Counting who is in the analysis"),
            # split 4 (WP13): a measurement named as the unit groups the draw but is exploratory.
            # split 4 (audit WP15, IN-24): the chronology counts held-out rows that predate training.
            # split 5 (WP13 + WP15 merged): both of the above in one stage.
            # split 6 (recognition's leash): the draw groups by a settled identifier only.
            Stage("split", 6, ("working", "cohort", "target_info", "structure"),
                  ("split", *ROLE_READS, "task", *SEAL_READS), split_stage, heavy=True,
                  requires=("split",), label="Drawing the held-out rows"),
            # shelf 7 (methods gate): under inference it ranks for every analyzed row and its basis
            # says so (BLUEPRINT §12 ruling 3); timing stays on the training rows.
            # shelf 8 (the readings ledger): its predictors are the settled roles'.
            Stage("shelf", 8, ("working", "cohort", "target_info", "split"),
                  ("purpose", "task", *ROLE_READS, "missing", "categorical", "lens", "findings",
                   "event",
                   "outcome_order", "exposure_forms"),
                  shelf_stage, heavy=True,
                  requires=("roles",), label="Ranking the model families for this table"),
            # design 6: the estimand and coefficient meanings are read off the matrix, and the
            # energy-dropped residual's gap reads the outcome and its event (audit WP6); an omics
            # normalization step reads the lens and the findings (audit WP11); each formed
            # exposure's spline or quintiles (audit WP12a); a follow-up is no predictor (WP12b).
            # design 7: the energy-aware single fill and the below-detection step (WP7).
            # design 8 (repair round): total energy kept as a covariate reads as the standard model
            # (ME-02); exposures in percent of energy carry their own meaning and pairs (B24).
            # design 9 (methods gate): under inference the estimand's fitted elasticity, the energy
            # step's warnings, the residual gap, the lineage and the matrix read every analyzed row
            # (ruling 3); the pipelines are still sized for the training rows.
            # design 10 (WP13): energy sources, parts and total energy read by the one recognizer
            # (whole words, NHANES and INFOODS codes; ``alc_kcal`` is alcohol, not total energy).
            # design 10 (audit WP15, IN-25): the lineage attributes each operation to the columns it
            # touched, marks pass-throughs kept, and names the geometric mean under log.
            # design 11 (WP13 + WP15 merged): both of the above in one stage.
            # design 12 (repair round): total energy read as intake only (no expenditure or score).
            # design 13 (gate repair): with an energy role named, a second energy-named predictor is
            # what the user said it is, not total energy.
            # design 14 (recognition's leash): the fit's clusters and its intake line read settled
            # roles only (BLUEPRINT §14).
            # design 15 (the readings ledger, BLUEPRINT §14.1): the fit reads settled readings only;
            # a role that rode along, or a whole-number predictor's code-or-amount reading, is asked.
            Stage("design", 15, ("working", "split", "target_info"),
                  (*ROLE_READS, "energy_adjustment", "missing", "models", "purpose", "categorical",
                   "event", "lens", "findings", "exposure_forms", "follow_up"),
                  design_stage,
                  heavy=True, requires=("models", "roles"),
                  label="Building each model's pipeline"),
            # fit 8: the merged fit (WP8's every-row table, WP9's validation, WP10's survey design,
            # WP11's feature-wise tests, WP12a's ordinal outcome and exposure tests, WP12b's
            # follow-up and the families that model the unit).
            # fit 9: missing data by purpose (WP7): under inference the table is pooled over
            # multiple imputations with the outcome, or held, or carries complete cases' cost.
            # fit 10 (repair round): under inference each family's every-row refit is kept for the
            # substitution curve; the Cox, mixed and GEE tables declare their scale; Harrell's
            # bootstrap is not applied to a family that declares it unsound for it.
            # fit 11 (methods gate): the outcome keeps the values the table spells, so a True/False
            # outcome's event is coded and named as declared; under inference the event's share
            # is of every analyzed row.
            # fit 12 (audit WP15, IN-22): under inference the table carries its measurement-error line.
            # fit 13 (gate repair): a design whose strata or PSUs are read with no weight asks the
            # survey question too, so the inference table waits for its answer.
            Stage("fit", 13, ("working", "design", "split", "target_info", "cohort"),
                  ("models", "purpose", "task", "event", "survey", "outcome_order", "follow_up"),
                  fit_stage, heavy=True, requires=("models",),
                  label="Fitting the models"),
            # substitution 6: a swap can move a share of energy (WP12a); a random intercept's band
            # refits one intercept per resampled unit (WP12b); a curve says it is not pooled over
            # multiple imputations (WP7).
            # substitution 7 (repair round): under inference the curve reads every analyzed row and
            # the families refit on them (BLUEPRINT §12 ruling 3), as the coefficient table does.
            # substitution 8 (methods gate): the outcome keeps its own values (a True/False event).
            # substitution 9 (WP13, audit IN-05): the estimand states the outcome's unit only as
            # recorded or spelled out by its name.
            # substitution 10 (the readings ledger, BLUEPRINT §14.1): each kcal-per-unit factor is
            # read settled only.
            Stage("substitution", 10, ("working", "fit", "design"),
                  ("substitution", "event", "outcome_order", "purpose", "outcome_unit",
                   *ROLE_READS),
                  substitution_stage, heavy=True, requires=("substitution",),
                  label="Drawing the substitution curves"),
            # ── M2: the seal (docs/turbotab-next/M2_CONTRACT.md §3) ──
            # seal_plan 3 (repair round): the declared purpose orders the split question (under
            # inference no holdout leads; BLUEPRINT §12 ruling 3), and the validation options.
            # seal_plan 4 (WP13): the basis reads a measurement named as the unit as exploratory.
            # seal_plan 4 (audit WP15, IN-24): the chronology's held-out rows that predate training.
            # seal_plan 5 (WP13 + WP15 merged): both of the above in one stage.
            # seal_plan 6 (recognition's leash): as split 6.
            Stage("seal_plan", 6, ("working", "cohort", "target_info", "structure"),
                  (*ROLE_READS, "task", "event", "purpose", *SEAL_READS), seal_plan_stage,
                  requires=("target",),
                  label="Reading what a held-out set can measure"),
            # ── WP12: methods a reviewer expects (AUDIT_REPORT §5) ──
            # sensitivity 2: each analysis fit as the fit stage fits the primary (scale, survey
            # design, units, ordinal order, follow-up).
            # sensitivity 3: under inference each analysis pools its own multiple imputations.
            # sensitivity 4 (methods gate): the outcome keeps its own values (a True/False event).
            # sensitivity 5 (gate repair): as fit 13, a half-read survey design waits for its answer.
            # sensitivity 6 (recognition's leash): its clusters read settled roles only.
            Stage("sensitivity", 7, ("working", "design", "split", "target_info"),
                  SENSITIVITY_READS, sensitivity_stage, heavy=True,
                  requires=("sensitivity", "models"),
                  label="Refitting the model on each analysis's rows"),
            # calibration 4 (methods gate): the outcome keeps its own values (a True/False event).
            # calibration 5, sensitivity 7 (the readings ledger): each reads the readings' own
            # confirmations; calibration applies only on an answered repeat kind.
            Stage("calibration", 5,
                  ("oriented", "findings", "structure", "working", "cohort", "design", "target_info"),
                  CALIBRATION_READS, calibration_stage, heavy=True,
                  requires=("measurement_error", "models"),
                  label="Correcting energy-adjusted intakes for day-to-day error"),
        ]
    )

__all__ = ["GRAPH_FACTORY", "build_graph"]

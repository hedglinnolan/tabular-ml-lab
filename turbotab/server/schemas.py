"""The HTTP contract (docs/turbotab-next/BLUEPRINT.md §6), as pydantic models.

The OpenAPI document generated from these is what the frontend's types come
from (``python -m turbotab.server.openapi``). Decision, DecisionRecord,
ProjectState, StageStatus and JobView are the runtime's own models, re-exported
so there is one definition of each.

Stage artifacts travel as ``StageResult.artifact`` (an object). Their shapes are
declared here too (``ARTIFACT_MODELS``) and added to the OpenAPI components, so
the frontend can type an artifact by its stage name.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from turbotab.core.consequences import CoachNote, PreviewResult, RowStep  # noqa: F401 - re-exported
from turbotab.core.decisions import (  # noqa: F401 - re-exported contract models
    Decision,
    DecisionRecord,
    EnergyMethod,
    ExclusionRule,
    FindingDisposition,
    Lens,
    ProjectState,
    Purpose,
    Role,
    Task,
)
from turbotab.core.graph import StageStatus, StatusName  # noqa: F401
from turbotab.core.interview import InterviewStep  # noqa: F401
# A finding routes to a question the Router asks, so its route is typed by the Router's keys and
# widens exactly when the interview does (the teaching also covers the repairs and seal cards).
from turbotab.core.interview import QuestionKey  # noqa: F401
from turbotab.core.jobs import JobView  # noqa: F401
from turbotab.core.repairs import RepairOption  # noqa: F401 - re-exported contract model
from turbotab.core.seal import Chronology, SealBasis, SealPlan  # noqa: F401 - the seal's shapes
from turbotab.core.teaching import TeachingEntry  # noqa: F401 - re-exported contract model

Mode = Literal["local", "server"]
SourceKind = Literal["path", "upload"]
Dtype = Literal["numeric", "integer", "boolean", "categorical", "datetime", "text"]
Severity = Literal["info", "warning", "critical"]
FindingSource = Literal["profile", "pack", "structural"]
Confidence = Literal["high", "medium", "low"]


class Model(BaseModel):
    """Every field is present in a response, so generated types mark none optional."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


# ── system ───────────────────────────────────────────────────────────────────


class Health(Model):
    version: str
    mode: Mode
    workers: int


# ── projects ─────────────────────────────────────────────────────────────────


class ProjectSummary(Model):
    id: str
    name: str
    created_at: datetime
    source_kind: SourceKind
    source_name: str
    n_rows: int | None = None
    n_cols: int | None = None
    # Whether the file is being read, was read, failed or was stopped. Null in the
    # project list for a project not opened since the server started.
    ingest: StageStatus | None = None


class CreateProject(BaseModel):
    """Open a file on this machine by its path (local mode only)."""

    model_config = ConfigDict(extra="forbid")

    path: str


class ProjectView(Model):
    summary: ProjectSummary
    state: ProjectState
    decisions: list[DecisionRecord]
    stages: dict[str, StageStatus]
    # The questions in asking order, with which one is open now (turbotab/core/interview.py).
    interview: list[InterviewStep]


class StageResult(Model):
    stage: str
    key: str | None
    fresh: bool
    status: StatusName
    artifact: dict[str, Any] | None


# ── data reads ───────────────────────────────────────────────────────────────


class ValueCount(Model):
    value: Any
    count: int


class ColumnSummary(Model):
    name: str
    dtype: Dtype
    n: int
    n_missing: int
    n_unique: int
    mean: float | None
    std: float | None
    min: float | None
    q25: float | None
    median: float | None
    q75: float | None
    max: float | None
    top: list[ValueCount] | None


class TableWindow(Model):
    columns: list[str]
    rows: list[list[Any]]
    total_rows: int
    offset: int


class Histogram(Model):
    column: str
    edges: list[float]
    counts: list[int]
    n_missing: int


# ── errors ───────────────────────────────────────────────────────────────────


class Exit(Model):
    label: str
    decision: Decision | None


class RefusalDetail(Model):
    code: str
    message: str
    exits: list[Exit]


class Refusal(Model):
    """Every error the API answers itself; HTTP 409 when a decision is refused."""

    error: RefusalDetail


# ── the file browser ─────────────────────────────────────────────────────────


class FsEntry(Model):
    name: str
    path: str
    is_dir: bool
    size: int | None


class FsListing(Model):
    path: str
    parent: str | None
    entries: list[FsEntry]


# ── stage artifacts (StageResult.artifact, by stage) ─────────────────────────


class ColumnInfo(Model):
    name: str
    dtype: Dtype
    physical_type: str
    n_missing: int
    n_unique: int
    sample: list[Any]


class DatasetInfo(Model):
    """The ``ingest`` artifact."""

    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    source_bytes: int
    parquet_bytes: int
    ingest_seconds: float
    fingerprint: str
    warnings: list[str]


class LensHint(Model):
    lens: Lens
    because: str


class ProfileArtifact(Model):
    """The ``profile`` artifact."""

    columns: list[ColumnSummary]
    lens_hints: list[LensHint]
    basis: str


class TargetInfo(Model):
    """The ``target_info`` artifact."""

    column: str
    task: Task
    detected_task: Task
    confidence: Confidence
    reason: str
    histogram: Histogram | None
    classes: list[ValueCount] | None
    # M2 (M2_CONTRACT §6): the outcome's unit (mg/dL…), read from its name or the clinical pack's
    # analytes; None when neither says. ``unit_source`` is "name" or "pack".
    unit: str | None = None
    unit_source: Literal["name", "pack"] | None = None


class FindingEvidence(Model):
    status: str
    source: str


class Finding(Model):
    id: str
    severity: Severity
    title: str
    detail: str
    why_it_matters: str | None
    affected_columns: list[str]
    source: FindingSource
    lens: Lens | None
    evidence: FindingEvidence | None
    # M1 (M1_CONTRACT §6): the one-line claim (≤ 20 words), the question that acts on it and
    # what that question will do (≤ 5 words), and a pager key shared by same-kind findings.
    # With no lever, routes_to and lever_label are null and the summary says so.
    summary: str
    routes_to: QuestionKey | None
    lever_label: str | None
    group: str | None
    # M2 (M2_CONTRACT §4): the repairs it offers, each previewed before it is applied; and, as
    # served, its disposition and the record that answered it (its own disposition, the answer to
    # its question, or another finding's repair that already did this one's work), else null.
    repairs: list[RepairOption] = []
    disposition: FindingDisposition | None = None
    answered_by: str | None = None


class FindingsArtifact(Model):
    """The ``findings`` artifact."""

    findings: list[Finding]
    basis: str


class RoleProposal(Model):
    column: str
    proposed: Role
    confidence: Confidence
    reason: str  # ≤ 16 words, plain language
    linked_to: str | None  # a flag's base column
    unit: str | None
    # The column this one is part of (fat_sat in fat_total, sugar in carb): the names say so and
    # the part never exceeds the total on 99% of rows. Substitution moves parts with their total.
    nested_in: str | None


class Repeats(Model):
    column: str
    n_units: int
    max_rows_per_unit: int


class RolesArtifact(Model):
    """The ``roles`` artifact: a proposed role for every column but the outcome."""

    columns: list[RoleProposal]
    repeats: Repeats | None


class CohortArtifact(Model):
    """The ``cohort`` artifact (a Bundle's ``data``): the participant flow."""

    steps: list[RowStep]
    n_final: int
    predictors: list[str]


class SplitArtifact(Model):
    """The ``split`` artifact (a Bundle's ``data``): held-out rows and folds."""

    n_train: int
    n_holdout: int
    holdout: float
    seed: int
    folds: int
    grouped_by: str | None
    n_groups: int | None
    stratified: bool
    note: str
    # M2 (M2_CONTRACT §3): how the held-out rows were drawn, in one of four recorded states, the
    # chronological draw when the answers asked for one, and whether either makes held-out scores
    # exploratory (never drawn as a clean lock).
    basis: SealBasis
    chronology: Chronology | None
    exploratory: bool


class ExclusionProposal(Model):
    """One of the pack's exclusion rules, with the rows it would remove. Never pre-selected."""

    key: str
    rule: ExclusionRule
    label: str
    affected: int
    evidence: FindingEvidence


class MethodVerdict(Model):
    ok: bool
    reason: str


class NotAdjusted(Model):
    """An exposure energy adjustment leaves as it is, and why (it carries no energy, it is a share)."""

    column: str
    reason: str


class EnergyReading(Model):
    """What the energy-adjustment question can offer on this table (NUTRITION_PACK §04)."""

    energy_column: str | None
    nutrients: list[str]
    strata_candidates: list[str]
    applicability: dict[str, MethodVerdict]
    usual: EnergyMethod | None
    usual_evidence: FindingEvidence | None
    r_with_energy: dict[str, float]
    notes: list[str]
    not_adjusted: list[NotAdjusted]


class MissingColumn(Model):
    """A predictor with blanks, counted among rows with the outcome measured."""

    column: str
    n_missing: int
    share: float
    # A mostly blank (≥ 50%) yes/no or medication-like column: a blank is a question not asked.
    likely_not_asked: bool
    reason: str


class LeaveOut(Model):
    """The offer: leave the likely-not-asked columns out, then handle the rest."""

    columns: list[str]
    n_rows: int  # rows blank in at least one of them
    share: float


class MissingReading(Model):
    """What the missing-values question can offer on this table (M1_CONTRACT §12.4)."""

    columns: list[MissingColumn]  # most blank first
    leave_out: LeaveOut | None


class ProposalsArtifact(Model):
    """The ``proposals`` artifact: offered for the exclusions, missing-values and energy questions."""

    exclusions: list[ExclusionProposal]
    energy: EnergyReading | None
    missing: MissingReading
    n_base: int  # the rows every count here is made among (the outcome recorded), all rows as loaded
    basis: str
    # M2: at most one coach line per decision card, keyed by question ("exclusions", "missing").
    coach: dict[str, CoachNote] = {}


# ── the table the analysis reads (M2_CONTRACT §2; turbotab/core/stages/working.py) ──────────


class OrientationReading(Model):
    """Question 1.5's shape reading: row-mean spread over column-mean spread, on a log scale."""

    reading: Literal["sample_major", "feature_major", "undetermined"]
    ratio: float | None
    s_rows: float | None
    s_cols: float | None
    n_rows: int
    n_numeric: int
    sentence: str
    confidence: Literal["medium", "low"]


class TurnCheck(Model):
    """Whether the table can be turned around: the feature-name column, and why not."""

    label_column: str | None
    n_features: int
    n_samples: int
    refusal: str | None
    code: str | None


class OrientedArtifact(DatasetInfo):
    """The ``oriented`` artifact: the raw table or its transpose (DatasetInfo of the result)."""

    transposed: bool
    reading: OrientationReading
    turn: TurnCheck


class RepetitionEvidence(Model):
    column: str
    n_distinct: int
    n_rows: int
    rows_per: float
    modal_rows_per: int
    regular_share: float


class GrainContradiction(Model):
    columns: list[str]  # the columns whose shape says rows repeat, most regular first
    message: str


class StatedGrain(Model):
    """The grain stated rather than asked (M2_CONTRACT §10): a recognized person identifier that is
    unique on every row, with nothing else repeating like a roster. "Ask me anyway" reopens it."""

    column: str
    n_rows: int
    sentence: str  # the skip's reason, after the client's "Not asked:" label


class GrainReading(Model):
    suggested: list[str]  # offered under "rows repeat", best first; never an answer
    evidence: list[RepetitionEvidence]  # name-blind repetition, the most regular first
    if_one_row: GrainContradiction | None  # what "one row per unit" would contradict
    stated: StatedGrain | None = None  # the grain the Router states rather than asks


class UnitCounts(Model):
    column: str
    n_units: int
    max_rows_per_unit: int
    min_rows_per_unit: int
    n_missing: int


class Spacing(Model):
    column: str
    n_people: int
    n_gaps: int
    min_days: float
    max_days: float
    median_days: float
    cv: float
    all_identical: bool


class RepeatsReading(Model):
    """Repeats or time points, read from date spacing or a record index (turbotab/repeats.py)."""

    reading: Literal["repeats", "time_points"] | None
    stated: bool  # strong enough to state as a skip ("Ask me anyway" reopens it)
    confidence: Literal["high", "medium"] | None
    evidence: list[str]
    sentence: str
    spacing: Spacing | None
    replicate_index: str | None
    n_units_read: int  # the reading walks at most 5,000 units, a fixed sample of a longer table


class OutcomeWithinUnit(Model):
    column: str
    varies: bool  # then combining rows asks which outcome to keep
    n_units_varying: int
    numeric: bool


class AggregationMenu(Model):
    """The domain-shaped menu: repeats → mean recommended with its reason; time points → none."""

    kind: Literal["repeats", "time_points"]
    recommended: Literal["mean", "first", "last", "change"] | None
    reason: str | None
    marker: str | None
    from_pack: str | None
    options: list[Literal["mean", "first", "last", "change"]]


class StructureArtifact(Model):
    """The ``structure`` artifact: what the grain, repeats, unit and aggregation questions offer."""

    grain: GrainReading
    units: UnitCounts | None
    repeats: RepeatsReading | None
    outcome: OutcomeWithinUnit | None
    aggregation: AggregationMenu | None
    time_columns: list[str]
    time_column: str | None


class Repair(Model):
    column: str
    expression: str


class OutcomeRule(Model):
    column: str
    varies: bool
    rule: Literal["constant", "mean", "first", "last"]


class AggregationReceipt(Model):
    """What combining did: rows → units, how the outcome was kept, what lost information."""

    id_column: str
    method: Literal["mean", "first", "last", "change"]
    outcome: OutcomeRule | None
    time_column: str | None
    ordered_by: str
    n_source_rows: int
    n_units: int
    single_record_units: int
    varying: list[str]  # non-numeric columns that differed within a unit and took the first record's
    combined_numeric: int


class WorkingArtifact(DatasetInfo):
    """The ``working`` artifact: DatasetInfo of the table every later stage reads."""

    pass_through: bool  # nothing structural recorded: the oriented table, referenced, not copied
    transposed: bool
    n_source_rows: int
    repairs: list[Repair]
    aggregation: AggregationReceipt | None
    row_map: Literal["identity", "row_map.parquet"]


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "ingest": DatasetInfo,
    "oriented": OrientedArtifact,
    "structure": StructureArtifact,
    "working": WorkingArtifact,
    "profile": ProfileArtifact,
    "target_info": TargetInfo,
    "findings": FindingsArtifact,
    "roles": RolesArtifact,
    "cohort": CohortArtifact,
    "split": SplitArtifact,
    "proposals": ProposalsArtifact,
    "seal_plan": SealPlan,
}

# The model-side artifacts (shelf, design, fit, substitution) are defined beside the stages that
# build them; they join the published shapes here.
from turbotab.core.models.artifacts import MODELING_ARTIFACTS  # noqa: E402

ARTIFACT_MODELS.update(MODELING_ARTIFACTS)

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

from turbotab.core.decisions import (  # noqa: F401 - re-exported contract models
    Decision,
    DecisionRecord,
    EnergyMethod,
    ExclusionRule,
    Lens,
    ProjectState,
    Purpose,
    Task,
)
from turbotab.core.graph import StageStatus, StatusName  # noqa: F401
from turbotab.core.jobs import JobView  # noqa: F401
from turbotab.core.teaching import (  # noqa: F401 - re-exported contract models
    QuestionKey,
    TeachingEntry,
)

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


class FindingsArtifact(Model):
    """The ``findings`` artifact."""

    findings: list[Finding]
    basis: str


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


class ProposalsArtifact(Model):
    """The ``proposals`` artifact: offered for the exclusions and energy questions."""

    exclusions: list[ExclusionProposal]
    energy: EnergyReading | None
    basis: str


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "ingest": DatasetInfo,
    "profile": ProfileArtifact,
    "target_info": TargetInfo,
    "findings": FindingsArtifact,
    "proposals": ProposalsArtifact,
}

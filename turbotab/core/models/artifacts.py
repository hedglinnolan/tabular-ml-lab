"""The client-visible shapes of the model-side stage artifacts (M1_CONTRACT §3).

The stages build these models and store their JSON; the server publishes them in the OpenAPI
document (``ARTIFACT_MODELS``), so the frontend types each artifact by its stage name.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from turbotab.core.consequences import Lineage
from turbotab.core.decisions import Task
from turbotab.core.models.base import Fit


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class ShelfFamily(_Model):
    key: str
    label: str
    rank: int
    fit: Fit
    concerns: list[str]
    inductive_bias: str


class ShelfArtifact(_Model):
    """The ``shelf`` artifact: every family that can model the task, best first."""

    families: list[ShelfFamily]
    basis: str


class MatrixShape(_Model):
    n_rows: int
    n_cols: int


class DesignStep(_Model):
    key: str
    label: str
    detail: str


class DesignModel(_Model):
    family: str
    label: str
    steps: list[DesignStep]


class SubstitutionPair(_Model):
    donor: str
    recipient: str


class DesignArtifact(_Model):
    """The ``design`` artifact: each model's pipeline and the columns it will see."""

    lineage: Lineage
    matrix: MatrixShape
    models: list[DesignModel]
    estimand: str | None
    substitution_pairs: list[SubstitutionPair]
    warnings: list[str]


class MetricSummary(_Model):
    mean: float | None
    sd: float | None
    folds: list[float | None]


class Coefficient(_Model):
    feature: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p: float | None


class FittedModel(_Model):
    family: str
    label: str
    cv: dict[str, MetricSummary]
    holdout: dict[str, float | None] | None
    coefficients: list[Coefficient] | None
    fit_seconds: float
    concerns: list[str]


class FitArtifact(_Model):
    """The ``fit`` artifact: cross-validated and held-out performance per model."""

    task: Task
    primary_metric: str
    metric_labels: dict[str, str]
    n_train: int
    n_holdout: int
    models: list[FittedModel]


class SubstitutionModel(_Model):
    family: str
    label: str
    delta: list[float | None]
    ci_low: list[float | None] | None
    ci_high: list[float | None] | None
    on_support_fraction: list[float]
    stopped_at: float | None
    effect_label: str | None


class SubstitutionArtifact(_Model):
    """The ``substitution`` artifact: one curve per fitted model."""

    donor: str
    recipient: str
    step_kcal: float
    ks: list[float]
    total_kind: Literal["fixed", "variable"]
    estimand: str | None
    note: str
    basis: str
    models: list[SubstitutionModel]


MODELING_ARTIFACTS: dict[str, type[BaseModel]] = {
    "shelf": ShelfArtifact,
    "design": DesignArtifact,
    "fit": FitArtifact,
    "substitution": SubstitutionArtifact,
}

__all__ = [
    "Coefficient", "DesignArtifact", "DesignModel", "DesignStep", "FitArtifact", "FittedModel",
    "MODELING_ARTIFACTS", "MatrixShape", "MetricSummary", "ShelfArtifact", "ShelfFamily",
    "SubstitutionArtifact", "SubstitutionModel", "SubstitutionPair",
]

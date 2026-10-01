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
from turbotab.core.models.baseline import VersusBaseline


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


class NestedColumn(_Model):
    """A predictor that is part of another (``fat_sat`` of ``fat_total``), confirmed on training rows."""

    column: str
    parent: str


class DesignArtifact(_Model):
    """The ``design`` artifact: each model's pipeline and the columns it will see."""

    lineage: Lineage
    matrix: MatrixShape
    models: list[DesignModel]
    estimand: str | None
    substitution_pairs: list[SubstitutionPair]  # never a total paired with its own part
    warnings: list[str]
    nested: list[NestedColumn] = []
    left_out: list[str] = []  # predictors the missing-values answer left out


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


class Baseline(_Model):
    """What predicting without the predictors scores on the same folds (M1_CONTRACT §12.6).

    The outcome's training-fold mean for regression; the training-fold class prior for
    classification. ``value`` is the CV mean of ``metric`` (the primary metric).
    """

    metric: str
    value: float | None
    label: str  # "the outcome's average" | "the class prior"


class FittedModel(_Model):
    family: str
    label: str
    cv: dict[str, MetricSummary]
    # Held-out scores: null while the seal is closed (M2_CONTRACT §3). The fit computes them once
    # and keeps them out of its public data; the server fills them in after ``open_seal``.
    holdout: dict[str, float | None] | None
    coefficients: list[Coefficient] | None
    fit_seconds: float
    concerns: list[str]  # a family that scores worse than, or no better than, its baseline says so first
    baseline: Baseline
    # The primary metric against the baseline's, paired over the same folds, within a stated
    # tolerance (turbotab/core/models/baseline.py). Null only in artifacts from before M2.
    versus_baseline: VersusBaseline | None = None


class FitArtifact(_Model):
    """The ``fit`` artifact: cross-validated and held-out performance per model."""

    task: Task
    primary_metric: str
    metric_labels: dict[str, str]
    n_train: int
    n_holdout: int
    models: list[FittedModel]
    # True while held-out scores exist and the seal is not opened: every ``holdout`` is null.
    holdout_sealed: bool = False
    # Set by the server once the seal is opened: whether this fit differs from the one the seal was
    # opened on, and the decisions made after the opening that changed what it reads.
    changed_after_seal: bool = False
    post_seal_decisions: list[str] = []


class SubstitutionModel(_Model):
    family: str
    label: str
    delta: list[float | None]
    ci_low: list[float | None] | None
    ci_high: list[float | None] | None
    on_support_fraction: list[float]
    stopped_at: float | None
    effect_label: str | None


class SubstitutionBand(_Model):
    """How the band was made: refits of every family on bootstrap resamples of training rows."""

    n_boot: int
    n_rows: int  # training rows each resample is drawn from (at most 2,000)
    grouped_by: str | None  # resampled by this identifier's units, when rows repeat
    seconds: float
    failed: int  # refits that could not be made (a resample with one class), left out


class BandEstimate(_Model):
    """A measured estimate of what adding a band would cost: one refit per family, timed."""

    n_boot: int
    seconds: float


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
    carried: list[str] = []  # parts or totals that moved with the donor or the recipient
    band: SubstitutionBand | None = None  # set when the substitution asked for n_boot > 0
    band_estimate: BandEstimate | None = None  # "Add an uncertainty band (about N s)"


MODELING_ARTIFACTS: dict[str, type[BaseModel]] = {
    "shelf": ShelfArtifact,
    "design": DesignArtifact,
    "fit": FitArtifact,
    "substitution": SubstitutionArtifact,
}

__all__ = [
    "BandEstimate", "Baseline", "NestedColumn", "SubstitutionBand", "Coefficient", "DesignArtifact", "DesignModel", "DesignStep", "FitArtifact", "FittedModel",
    "MODELING_ARTIFACTS", "MatrixShape", "MetricSummary", "ShelfArtifact", "ShelfFamily",
    "SubstitutionArtifact", "SubstitutionModel", "SubstitutionPair",
]

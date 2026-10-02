"""The client-visible shapes of the model-side stage artifacts (M1_CONTRACT §3).

The stages build these models and store their JSON; the server publishes them in the OpenAPI
document (``ARTIFACT_MODELS``), so the frontend types each artifact by its stage name.
"""
from __future__ import annotations

from typing import Any, Literal

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
    # What fitting this family takes here (M2_CONTRACT §12.6): one fit timed on a sample of the
    # training rows, scaled to the whole table and to the folds the fit makes, as the band estimate
    # is measured. None when the timing fit could not run. ``estimate`` says it in words.
    estimate_seconds: float | None = None
    estimate: str | None = None  # "about 5 minutes at 20,004 columns"


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
    """One metric's cross-validated score (``turbotab/core/models/metrics.py``).

    ``estimate`` is the score the app reports: pooled over every out-of-fold prediction for R²,
    RMSE and MAE (``estimator: "pooled"``), the mean over folds otherwise (``"fold_mean"``).
    ``mean`` and ``sd`` describe the per-fold values, which show the spread.
    """

    mean: float | None
    sd: float | None
    folds: list[float | None]
    estimate: float | None = None  # null only in artifacts from before the pooled estimator
    estimator: Literal["pooled", "fold_mean"] = "fold_mean"


class Coefficient(_Model):
    feature: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p: float | None
    se: float | None = None  # the standard error the interval rests on (inference only)
    # The t reference distribution's degrees of freedom; null for a normal or likelihood-based one.
    df: float | None = None
    # On a ratio scale (the table's ``inference.scale``): exp(estimate), the odds ratio or the
    # relative-risk ratio, and exp of the interval's ends. Null for the intercept and on the
    # difference scale.
    ratio: float | None = None
    ratio_low: float | None = None
    ratio_high: float | None = None


class InferenceExit(_Model):
    """A way forward when no interval can be reported, as a refusal's exits are shaped."""

    label: str
    decision: dict[str, Any] | None = None


class Inference(_Model):
    """How the coefficient table's intervals were made (AUDIT_REPORT §5 WP2;
    ``turbotab/core/models/inference.py``): the estimator, the covariance, and the clusters."""

    estimator: str  # "ordinary least squares", "Firth-penalized logistic regression", …
    # HC3 · CR2 (cluster-robust, Bell–McCaffrey df) · model (Wald, from the information) ·
    # profile (penalized likelihood) · none (refused: ``refused`` says why).
    covariance: Literal["HC3", "CR2", "model", "profile", "none"]
    # One or two lines: the scale (on a ratio scale), the covariance, the clusters, the reference
    # distribution and the rows the table was estimated from.
    caption: str
    grouped_by: str | None = None  # the identifier the intervals are clustered by
    n_clusters: int | None = None
    n_missing_ids: int = 0  # rows with no identifier, each counted as a unit of its own
    separated: list[str] = []  # columns that separate a binary outcome
    refused: str | None = None  # why no interval or p-value is reported
    exits: list[InferenceExit] = []
    # The scale a row's effect is read on (AUDIT_REPORT §5 WP8, ME-07): a difference in the mean
    # outcome (drawn on a linear axis), or an odds ratio or relative-risk ratio (the rows' ``ratio``
    # fields, drawn on a log axis). ``effect`` says it in a sentence that names the outcome, and
    # for a ratio the ``event`` (binary) and the ``reference`` level it is against.
    scale: Literal["difference", "odds_ratio", "relative_risk_ratio"] = "difference"
    axis: Literal["linear", "log"] = "linear"
    effect: str | None = None
    event: str | None = None
    reference: str | None = None
    # The rows the table was estimated from: every analyzed row under inference (BLUEPRINT §12
    # ruling 3), the training rows otherwise. Null only in artifacts from before WP8.
    n_rows: int | None = None
    rows: Literal["all", "training"] | None = None


class Baseline(_Model):
    """What predicting without the predictors scores on the same folds (M1_CONTRACT §12.6).

    The outcome's training-fold mean for regression; the training-fold class prior for
    classification. ``value`` is its cross-validated ``metric`` (the primary metric), estimated
    as the models' is: a pooled R² of the training-fold mean is 0 by construction.
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
    # How the intervals were made, under inference; null under prediction and for families
    # without an inference table.
    inference: Inference | None = None
    # How many rows the coefficients were estimated from (every analyzed row under inference, the
    # training rows under prediction); null without coefficients.
    coefficients_n: int | None = None
    # Once the seal is opened: "final" for the family declared before the opening (its held-out
    # score is the reported result), "secondary" for the others. Null while sealed.
    role: Literal["final", "secondary"] | None = None


class Selection(_Model):
    """What choosing among families on cross-validation costs (``models/selection.py``): the best
    family's CV score flatters it, by about ``optimism`` (Tibshirani & Tibshirani 2009)."""

    metric: str
    families: list[str]
    best: str  # the family with the best cross-validated score
    cv: float  # its cross-validated score
    optimism: float  # how much that score overstates its performance, in the score's units
    optimism_se: float | None
    corrected: float  # the score with the optimism taken off
    folds: int
    method: str
    text: str


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
    # How the folds were used: "random" (each fold scored by a model fit on the others) or
    # "time_ordered" (each fold scored by a model fit on the earlier ones, forward chaining).
    fold_scheme: Literal["random", "time_ordered"] = "random"
    cv_definition: str | None = None  # what a cross-validated score is, in one or two sentences
    # With two or more families: the optimism of picking the best of them by cross-validation.
    selection: Selection | None = None
    # Set by the server once the seal is opened: the family declared final at the opening (its
    # held-out score is the reported result), and the sentence that says so.
    final_model: str | None = None
    final_note: str | None = None


class SubstitutionModel(_Model):
    family: str
    label: str
    delta: list[float | None]  # each k over its own on-support rows
    ci_low: list[float | None] | None
    ci_high: list[float | None] | None
    on_support_fraction: list[float]
    stopped_at: float | None
    effect_label: str | None
    # The same curve over one fixed population: the rows on support at every k the curve reached.
    fixed_delta: list[float | None] = []
    fixed_ci_low: list[float | None] | None = None
    fixed_ci_high: list[float | None] | None = None
    band_ok: int | None = None  # refits of this family's band that succeeded


class SubstitutionSupport(_Model):
    """Which rows each k averages over, and how many each support check left out (per k)."""

    total: str | None  # the total-energy column shares are taken of; None: composition unchecked
    n_rows: int  # rows the curve averages over before any check
    not_recorded: int  # donor, recipient or total energy not recorded: off support at every k
    off_amount: list[int]  # a shifted amount outside its observed range or below 0
    off_share: list[int]  # further rows: a shifted share of energy outside its observed range
    fixed_rows: int  # the fixed population: rows on support at every k the curve reached
    fixed_through: float | None  # the largest k the fixed population is on support through


class SubstitutionBand(_Model):
    """How the band was made: refits of every family on bootstrap resamples of training rows."""

    n_boot: int
    n_rows: int  # training rows the resamples are drawn from: every row the models were fit on
    grouped_by: str | None  # resampled by this identifier's units, when rows repeat
    seconds: float
    failed: int  # refits that could not be made (a resample with one class), left out
    n_units: int | None = None  # units resampled: rows, or whole grouped_by units
    resample_units: int | None = None  # units each refit draws; fewer than n_units: rescaled
    scale: float | None = None  # sqrt(resample_units / n_units), 1 for the ordinary bootstrap
    interval: Literal["normal", "percentile"] | None = None
    level: float | None = None
    min_ok_share: float | None = None  # a family's band needs this share of its refits to succeed
    caption: str | None = None  # the saved figure's: rows, refits, how many succeeded


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
    support: SubstitutionSupport | None = None  # the support checks' counts (data, not model)


MODELING_ARTIFACTS: dict[str, type[BaseModel]] = {
    "shelf": ShelfArtifact,
    "design": DesignArtifact,
    "fit": FitArtifact,
    "substitution": SubstitutionArtifact,
}

__all__ = [
    "BandEstimate", "Baseline", "NestedColumn", "SubstitutionBand", "Coefficient", "DesignArtifact", "Inference", "InferenceExit", "DesignModel", "DesignStep", "FitArtifact", "FittedModel",
    "MODELING_ARTIFACTS", "MatrixShape", "MetricSummary", "Selection", "ShelfArtifact", "ShelfFamily",
    "SubstitutionArtifact", "SubstitutionModel", "SubstitutionPair", "SubstitutionSupport",
]

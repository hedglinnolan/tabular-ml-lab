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
from turbotab.core.models.performance import Calibration, Interval
from turbotab.core.models.validation import FamilyDifference, InternalExternal, Optimism


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
    # What each model-matrix column's coefficient means, read off the matrix (audit WP6:
    # ``methods.energy.describe_model``): the energy sources a swap is "in place of", a total
    # beside its parts as the remainder. The fit copies it onto each coefficient row.
    terms: dict[str, str] = {}
    # The energy model the matrix actually holds (one of the energy methods), whatever was named;
    # null when the table has no energy question.
    energy_form: str | None = None


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
    # The estimate's standard error and 95% interval (turbotab/core/models/performance.py): LeDell
    # et al.'s for a fold mean, the delta method for a pooled score; null for macro-F1, and in
    # artifacts from before WP9.
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    repeats: int = 1  # repeated k-fold: the estimate is the mean of this many repeats' estimates
    repeat_sd: float | None = None  # the spread of the repeats' estimates (null for one run)


class Coefficient(_Model):
    feature: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p: float | None
    se: float | None = None  # the standard error the interval rests on (inference only)
    # The t reference distribution's degrees of freedom; null for a normal or likelihood-based one.
    df: float | None = None
    # What the coefficient means in this model (the design's ``terms``), e.g. "fat in place of
    # alcohol and other energy, total energy fixed"; the all-components model's average relative
    # effects (``<nutrient>_relative`` rows) say their weights here.
    meaning: str | None = None
    # On a ratio scale (the table's ``inference.scale``): exp(estimate), the odds ratio or the
    # relative-risk ratio, and exp of the interval's ends. Null for the intercept and on the
    # difference scale.
    ratio: float | None = None
    ratio_low: float | None = None
    ratio_high: float | None = None
    # The Benjamini–Hochberg adjusted p-value across the features a feature-wise family tested.
    q: float | None = None
    # Under multiple imputation (WP7): the fraction of missing information, Rubin's γ.
    fmi: float | None = None
    # Under multiple imputation (MS2): the Monte Carlo error of the pooled estimate, √(B/m).
    mc_se: float | None = None


class InferenceExit(_Model):
    """A way forward when no interval can be reported, as a refusal's exits are shaped."""

    label: str
    decision: dict[str, Any] | None = None


class SurveyInference(_Model):
    """The survey design a design-based table rests on (AUDIT_REPORT §5 WP10;
    ``turbotab/core/models/survey.py``): what weighted it, its PSUs and strata, the domain."""

    weight: str | None
    strata: str | None
    psu: str | None
    weight_note: str | None = None  # how a pooled-cycle weight was built
    psu_note: str | None = None  # no PSU column: what stood for one
    n_design: int  # rows in the design (every row of the table with a stratum and PSU)
    n_domain: int  # the analysis rows the estimate uses
    n_psu: int
    n_strata: int
    domain_psu: int | None = None  # PSUs and strata holding analysis rows (the df count them)
    domain_strata: int | None = None
    df: int | None = None  # design degrees of freedom: domain PSUs minus domain strata
    lonely_strata: list[str] = []  # strata with a single PSU, centered at the mean PSU total
    lonely_method: str = "centered"


class MissingData(_Model):
    """How the inference table handled missing predictor values (AUDIT_REPORT §5 WP7, ME-01;
    ``turbotab/core/methods/missing.py``): multiple imputation with the outcome (m imputations by
    chained equations, pooled by Rubin's rules), complete cases with their assumption, or a single
    fill kept with its recorded attestation."""

    method: Literal["multiple_imputation", "complete_case", "single_fill"]
    assumption: str
    m: int | None = None
    iterations: int | None = None
    n_rows: int | None = None
    n_incomplete_rows: int | None = None  # rows with at least one imputed value
    imputed: dict[str, int] = {}  # column -> cells imputed
    variables: list[str] = []  # the imputation model's variables ("the outcome" among them)
    outcome_in_model: bool = False
    energy: str | None = None  # total energy, in the imputation model
    censored: list[str] = []  # left-censored columns drawn below their detection limit
    below_detection: str | None = None
    n_dropped: int | None = None  # complete cases: rows dropped for a missing predictor
    recorded: bool = False  # a blocked answer kept with its attestation
    note: str | None = None
    # MS1–MS3 (methods/smcfcs.py, methods/missing.py): how the copies were drawn and what the
    # imputation model held, the m rule, the Monte Carlo error, and the methods sentence.
    model: Literal["chained_equations", "smcfcs", "supplied"] | None = None
    compatible: bool | None = None  # compatible with the analysis model (False: passive, recorded)
    substantive: Literal["linear", "logistic", "cox"] | None = None  # SMC-FCS's analysis model
    terms: list[str] = []  # the declared nonlinear or derived terms the imputation respects
    logged: list[str] = []  # imputed on the log scale, as the analysis logs them
    identity_energy: str | None = None  # total energy, derived as its sources plus the rest
    identity_sources: list[str] = []
    identity_infeasible: int = 0  # rows whose recorded sources already reach their recorded total
    design_strata: str | None = None  # the survey design's columns in the imputation model
    design_psu: str | None = None
    design_weight: str | None = None
    df_com: float | None = None  # the complete-data df Rubin's rules used (the design's)
    unit: str | None = None  # clustered imputation: the unit column
    unit_level: list[str] = []  # imputed once per unit
    knots: dict[str, list[float]] = {}  # placed once on the observed values, held in every copy
    cuts: dict[str, list[float]] = {}
    m_asked: int | None = None
    percent_incomplete: float | None = None  # rows with any imputed value, as a percentage
    rejection_failures: int = 0
    mc_max_ratio: float | None = None  # the largest Monte Carlo error, as a share of its SE
    mc_feature: str | None = None
    copies: list[str] | None = None  # imputed copies the data carried, by their numbers
    implicate: str | None = None
    tests_d1: int = 0  # multi-parameter tests pooled by D1
    sentence: str | None = None  # the methods sentence


class BrantColumn(_Model):
    statistic: float
    df: int
    p: float


class BrantCheck(_Model):
    """Brant's (1990) Wald test of the proportional-odds assumption: overall, and per column."""

    statistic: float
    df: int
    p: float
    columns: dict[str, BrantColumn]


class Inference(_Model):
    """How the coefficient table's intervals were made (AUDIT_REPORT §5 WP2;
    ``turbotab/core/models/inference.py``): the estimator, the covariance, and the clusters."""

    estimator: str  # "ordinary least squares", "Firth-penalized logistic regression", …
    # HC3 · CR2 (cluster-robust, Bell–McCaffrey df) · CR0 (the Cox model's Lin–Wei cluster
    # sandwich, on t(G − 1)) · model (Wald from the information, or a mixed model's REML
    # covariance) · profile (penalized likelihood) · design (Taylor linearization over a survey
    # design, WP10) · none (refused: ``refused`` says why).
    # CR1 (cluster sandwich, G/(G − 1), on t(G − 1)): the proportional-odds family's.
    covariance: Literal["HC3", "CR2", "CR1", "CR0", "model", "profile", "design", "none"]
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
    # outcome (drawn on a linear axis), or an odds ratio, relative-risk ratio or (a Cox model's,
    # WP12b) hazard ratio (the rows' ``ratio`` fields, drawn on a log axis). ``effect`` says it in a sentence that names the outcome, and
    # for a ratio the ``event`` (binary) and the ``reference`` level it is against.
    scale: Literal["difference", "odds_ratio", "relative_risk_ratio", "hazard_ratio"] = "difference"
    axis: Literal["linear", "log"] = "linear"
    effect: str | None = None
    event: str | None = None
    reference: str | None = None
    # The rows the table was estimated from: every analyzed row under inference (BLUEPRINT §12
    # ruling 3), the training rows otherwise. Null only in artifacts from before WP8.
    n_rows: int | None = None
    rows: Literal["all", "training"] | None = None
    survey: SurveyInference | None = None  # the design, when the table is design-based
    brant: BrantCheck | None = None  # the proportional-odds family's check (independent rows)
    missing: MissingData | None = None  # how missing predictor values were handled (WP7)


class ExposureTest(_Model):
    """A test an exposure's form carries under inference (``methods/exposure_form.py``): a
    spline's overall and nonlinear Wald tests, or quintiles' trend across their medians."""

    column: str
    form: Literal["spline", "quintiles"]
    # global: every quintile indicator at once (a multi-df Wald test, D1 under multiple imputation)
    test: Literal["overall", "nonlinear", "trend", "global"]
    statistic: float | None
    df_num: int | None
    df_den: float | None  # the F or t reference's degrees of freedom; null for χ² and z
    distribution: Literal["F", "chi2", "t", "z"]
    p: float | None
    estimate: float | None = None  # trend: the coefficient per unit, scored at quintile medians
    ci_low: float | None = None
    ci_high: float | None = None
    knots: list[float] | None = None  # the spline's knots, learned on the fitting rows
    medians: list[float] | None = None  # each quintile's median, learned on the fitting rows
    caption: str


class Baseline(_Model):
    """What predicting without the predictors scores on the same folds (M1_CONTRACT §12.6).

    The outcome's training-fold mean for regression; the training-fold class prior for
    classification. ``value`` is its cross-validated ``metric`` (the primary metric), estimated
    as the models' is: a pooled R² of the training-fold mean is 0 by construction.
    """

    metric: str
    value: float | None
    label: str  # "the outcome's average" | "the class prior"


class HoldoutDetail(_Model):
    """What the held-out rows say beyond their scores: an interval on each, and calibration."""

    intervals: dict[str, Interval]
    calibration: Calibration | None = None


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
    # Out-of-fold calibration (the first repeat's predictions): intercept, slope, smoothed curve
    # (audit ME-10). Null for a multiclass outcome.
    calibration: Calibration | None = None
    # Held-out intervals and calibration: sealed like ``holdout`` until the seal is opened.
    holdout_detail: HoldoutDetail | None = None
    # Bootstrap optimism correction, when the split asked for it (audit ME-11).
    optimism: Optimism | None = None
    # Internal–external validation by a cluster column, when the split asked for it (audit E16).
    internal_external: InternalExternal | None = None
    # Under inference, the tests each spline or quintile exposure carries (WP12a).
    exposure_tests: list[ExposureTest] = []


class Selection(_Model):
    """What choosing among families on cross-validation costs (``models/selection.py``): the best
    family's CV score flatters it, by about ``optimism``, estimated by bootstrap bias-corrected
    cross-validation over the families' out-of-fold predictions (Tsamardinos et al. 2018)."""

    metric: str
    families: list[str]
    best: str  # the family with the best cross-validated score
    cv: float  # its cross-validated score
    # How much that score overstates the chosen family's performance, in the score's units (positive
    # flatters): ``cv`` less ``corrected`` for a score where higher is better.
    optimism: float
    corrected: float  # the performance expected on new rows once the choice is accounted for
    corrected_low: float  # its 95% percentile interval over the resamples
    corrected_high: float
    replicates: int  # resamples of the out-of-fold predictions
    wins: dict[str, int]  # how often each family was the one chosen on a resample
    method: str
    text: str


class AtOpening(_Model):
    """The held-out scores recorded when the current outcome's seal was first opened: the reported
    result, whatever is drawn, fitted or opened later (audit WP16, RO-05). Set by the server."""

    seq: int  # the record that opened it
    family: str | None  # the final model declared at the opening
    metric: str | None  # the primary metric
    n_holdout: int | None
    scores: dict[str, dict[str, float | None]]  # family -> metric -> held-out score
    # Whether the held-out scores served beside it are these (the same opening, an unchanged fit).
    current: bool
    note: str  # one sentence: the reported result, and what the scores shown now are


class EstimandAnnotation(_Model):
    """WP17: the served fit under a declared estimand (``turbotab/core/estimand.py``): the caption
    worded from it, and which table rows are the exposure's effect; the rest are adjustment terms."""

    exposure: str
    effect: Literal["total", "direct"]
    measure: str
    contrast: Literal["substitution", "addition"] | None = None
    caption: str | None
    adjusted: list[str]
    left_out: dict[str, str]  # column -> its derived role
    secondary: list[str]  # the declared "further adjusted for" model's added columns
    features: list[str]  # the model-matrix columns carrying the exposure's effect


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
    # Set by the server: the scores the first opening of this outcome's seal recorded (WP16).
    at_opening: AtOpening | None = None
    # How the training rows validated the models (the split's answer; audit ME-11, E16).
    validation: Literal["kfold", "repeated_kfold", "bootstrap", "internal_external"] = "kfold"
    repeats: int = 1
    # How the families are ranked, in words: "highest AUC", "highest R²", "lowest log loss".
    ranking: str | None = None
    se_definition: str | None = None  # what the standard errors count and leave out
    # On the user's own rows: how far apart the families are, pair by pair, with intervals
    # corrected for shared training rows (replaces the unsourced "below about 50 rows" claim).
    comparisons: list[FamilyDifference] = []
    precision: str | None = None  # one sentence: each family's standard error on these rows
    # Binary outcomes: the event's share and the resampling tension in one line (audit E15).
    imbalance: str | None = None
    # An ordinal outcome's levels in their order, lowest first: code k is levels[k] (WP12a).
    levels: list[str] | None = None
    # Set by the server (WP17): why every estimate is withheld while a question it rests on is
    # unanswered; else, under inference, the declared estimand the table is captioned from.
    withheld: str | None = None
    estimand: EstimandAnnotation | None = None


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
    # Under multiple imputation (MS3): how the curve was pooled over the copies — "contrast" (a
    # linear all-components model: the exact contrast of the pooled coefficients with their pooled
    # covariance) or "per_k" (each copy's curve, pooled at each k by Rubin's rules) — and the
    # Barnard–Rubin degrees of freedom of each k's interval.
    pooled: Literal["contrast", "per_k"] | None = None
    df: list[float | None] | None = None


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


class OmittedEnergy(_Model):
    """How much of total energy a model's energy-bearing columns leave out (audit ME-05)."""

    energy_column: str
    columns: list[str]  # the model's energy-bearing exposures, a part beside its total counted once
    sources: list[str]  # the energy sources they carry (protein, carbohydrate, fat, alcohol)
    omitted: list[str]  # the sources not in the model, "other" last
    mean_share: float | None  # the mean share of total energy they leave out
    rows_over: int  # rows whose remainder exceeds 10% of their total energy
    n_rows: int


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
    # The energy sources the model leaves to total energy's composite, and how much of total
    # energy they make up on the training rows (audit ME-05; ``methods.energy.omitted_energy``).
    omitted_energy: OmittedEnergy | None = None
    # What k measures (WP12a; audit B24, D19): kcal on every row, or percentage points of each
    # row's own total energy ("5% of energy from X replaced by Y"; ks then step by step_percent).
    scale: Literal["kcal", "percent_energy"] = "kcal"
    step_percent: float | None = None


MODELING_ARTIFACTS: dict[str, type[BaseModel]] = {
    "shelf": ShelfArtifact,
    "design": DesignArtifact,
    "fit": FitArtifact,
    "substitution": SubstitutionArtifact,
}

__all__ = [
    "AtOpening", "BandEstimate", "Baseline", "BrantCheck", "BrantColumn", "ExposureTest", "NestedColumn", "SubstitutionBand", "Coefficient", "DesignArtifact", "Inference", "InferenceExit", "DesignModel", "DesignStep", "FitArtifact", "FittedModel", "HoldoutDetail",
    "MODELING_ARTIFACTS", "MatrixShape", "MetricSummary", "MissingData", "OmittedEnergy", "Selection", "ShelfArtifact", "ShelfFamily",
    "SubstitutionArtifact", "SubstitutionModel", "SubstitutionPair", "SubstitutionSupport",
]

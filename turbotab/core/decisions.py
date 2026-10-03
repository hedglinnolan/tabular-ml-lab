"""Decisions, event-sourced (docs/turbotab-next/BLUEPRINT.md §3).

Every user answer is a :data:`Decision`. Decisions are appended to
``decisions.jsonl`` as :class:`DecisionRecord` lines and never edited; the
project's :class:`ProjectState` is a fold over that log.

Adding a decision kind is two steps in one place, never an if-chain:

1. define its model (a ``kind`` literal plus its payload) and add it to the
   :data:`Decision` union below, so the API and the log can parse it;
2. call :func:`register_kind` with the slot it writes (and, when the payload has
   more than one field, a ``value=`` function; when the answer is about another
   slot's value, a ``holds=`` condition), plus :func:`register_validator` for
   any refusal that needs project context.

Revert semantics (Tier A, pinned by ``tests/test_decisions.py``): a reverted
record is folded as though it had never been made, so ``revert(X)`` restores
X's slot to the value it had before X (when X was that slot's latest write);
reverting a revert reinstates what it reverted. A revert may target only an
earlier record that writes a slot, or an earlier revert; anything else is
refused with ``unknown_decision`` and nothing is recorded.

Conditional writes (Tier A, same file): a ``set_task`` answers for the column
it names, so the ``task`` slot holds the latest ``set_task`` for the *current*
target and is unset while no answer names it. Choosing another target never
carries one column's task over to the next.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal, Mapping, Sequence, Union

from pydantic import (AwareDatetime, BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter,
                      field_validator, model_validator)

try:  # POSIX only; on Windows the in-process lock is the whole of it.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

Lens = Literal["metabolomics", "genomics", "dietary", "clinical", "survey"]
# "ordinal": ordered levels (a 1–5 rating, none < mild < severe), modeled by a cumulative-link
# family that keeps the order (audit ME-19, RO-10). The order is declared, never inferred.
# time_to_event: the outcome column is the event; its follow-up is named by ``set_follow_up``.
Task = Literal["regression", "binary", "multiclass", "ordinal", "time_to_event"]
Purpose = Literal["prediction", "inference"]

ROW_ID = "__row_id"  # the stable row identity the datastore adds (BLUEPRINT §2)


def _kind_is_required(schema: dict[str, Any]) -> None:
    """Keep ``kind`` required in the JSON schema even though it has a default.

    The default lets Python write ``SetLens(lenses=[...])``; the schema must
    still say ``kind`` is always present, or generated TypeScript types make the
    discriminant optional and narrowing breaks.
    """
    required = schema.setdefault("required", [])
    if "kind" not in required:
        required.insert(0, "kind")


class _DecisionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra=_kind_is_required,
        json_schema_serialization_defaults_required=True,
    )


class SetLens(_DecisionModel):
    kind: Literal["set_lens"] = "set_lens"
    lenses: list[Lens] = Field(min_length=1)

    @field_validator("lenses")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each lens may be named only once")
        return value


class SetTarget(_DecisionModel):
    kind: Literal["set_target"] = "set_target"
    column: str = Field(min_length=1)


class SetTask(_DecisionModel):
    """An override of task detection, for the outcome column it names.

    It stands only while ``column`` is the target (see :func:`fold`).
    """

    kind: Literal["set_task"] = "set_task"
    column: str = Field(min_length=1)
    task: Task


class SetPurpose(_DecisionModel):
    kind: Literal["set_purpose"] = "set_purpose"
    purpose: Purpose


class Revert(_DecisionModel):
    kind: Literal["revert"] = "revert"
    decision_id: str = Field(min_length=1)


# ── M1 kinds (docs/turbotab-next/M1_CONTRACT.md) ─────────────────────────────
# The shapes are fixed here so every M1 agent builds on one definition; their
# validators, sentences and the stages that read them are the agents' work.

Role = Literal["identifier", "exposure", "energy", "covariate", "design", "flag", "time", "excluded"]
# "residual" keeps total energy in the outcome model; "residual_energy_dropped" lets it leave
# (BLUEPRINT §12 ruling 1); "all_components" gives every energy source its own term (audit WP6).
EnergyMethod = Literal["none", "standard", "residual", "residual_energy_dropped",
                       "density_multivariate", "density", "partition", "all_components"]
# "multiple_imputation" is inference's: chained equations with the outcome and total energy in the
# imputation model, pooled by Rubin's rules (BLUEPRINT §12 ruling 4; turbotab/core/methods/missing.py).
MissingStrategy = Literal["complete_case", "impute", "multiple_imputation"]
# How a left-censored column's blanks (values below a detection limit) are filled (audit ME-08).
BelowDetection = Literal["half_minimum", "censoring_aware", "as_missing"]


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True,
                              json_schema_serialization_defaults_required=True)


class RangeByLevel(_Value):
    """Different plausible ranges per level of another column (e.g. by sex)."""

    column: str = Field(min_length=1)
    ranges: dict[str, tuple[float | None, float | None]]


class ExclusionRule(_Value):
    kind: Literal["range"] = "range"
    column: str = Field(min_length=1)
    low: float | None = None
    high: float | None = None
    by: RangeByLevel | None = None
    reason: str = Field(min_length=1)
    # A row whose value is not recorded (or, with ``by``, whose level has no range) cannot be
    # confirmed eligible. "exclude" (the default, STROBE item 13a's "confirmed eligible") leaves it
    # out on a line of its own, "`age` not recorded"; "keep" keeps it and the step says so.
    missing: Literal["exclude", "keep"] = "exclude"

    def reads(self) -> list[str]:
        """Every column the rule reads (the outcome guard refuses a rule that reads the outcome)."""
        return [self.column] + ([self.by.column] if self.by is not None else [])


# WP12 (audit ME-16): the Goldberg screen. The arithmetic and its sources are in
# turbotab/core/methods/misreporting.py; this is the shape the answer records.
BmrEquation = Literal["schofield", "schofield_height", "henry", "henry_height", "mifflin"]


class LevelValues(_Value):
    """A number per level of another column (a PAL per activity category)."""

    column: str = Field(min_length=1)
    values: dict[str, float] = Field(min_length=1)


class GoldbergRule(_Value):
    """Keep rows whose reported energy intake over estimated BMR lies within the Goldberg cut-offs
    (Goldberg et al. 1991, revised by Black 2000): ``PAL × exp(±2 S / 100)`` at n = 1, with
    ``S = √(CV²_wEI / d + CV²_wB + CV²_tP)``. Every input is stated: the BMR equation, the PAL (one
    value, or one per level of an activity column), and the days of intake the energy averages."""

    kind: Literal["goldberg"] = "goldberg"
    column: str = Field(min_length=1)  # reported energy intake, a mean over ``days`` days
    energy_unit: Literal["kcal", "kj"] = "kcal"
    days: float | None = Field(default=None, ge=1)
    days_column: str | None = None  # each row's own number of days, when they differ
    sex: str = Field(min_length=1)
    female: list[str] = Field(default_factory=list)  # levels of ``sex`` read as female
    male: list[str] = Field(default_factory=list)
    age: str = Field(min_length=1)  # years
    weight: str = Field(min_length=1)  # kg
    height: str | None = None
    height_unit: Literal["cm", "m"] = "cm"
    equation: BmrEquation
    pal: float | None = Field(default=None, ge=1.0, le=3.0)
    pal_by: LevelValues | None = None
    exclude: Literal["both", "under", "over"] = "both"  # which reporters leave
    cv_wei: float = Field(default=23.0, gt=0, le=100)  # Black 2000's values
    cv_wb: float = Field(default=8.5, gt=0, le=100)
    cv_tp: float = Field(default=15.0, gt=0, le=100)
    reason: str = Field(min_length=1)
    missing: Literal["exclude", "keep"] = "exclude"

    @model_validator(mode="after")
    def _stated(self) -> "GoldbergRule":
        if self.pal is None and self.pal_by is None:
            raise ValueError("a Goldberg screen needs a PAL: one value, or one per activity level")
        if self.days is None and not self.days_column:
            raise ValueError("a Goldberg screen needs the days of intake the energy averages")
        if not self.female and not self.male:
            raise ValueError("say which levels of the sex column are female and which are male")
        if set(map(str, self.female)) & set(map(str, self.male)):
            raise ValueError("a level cannot be both female and male")
        if self.equation in ("schofield_height", "henry_height", "mifflin") and not self.height:
            raise ValueError(f"the {self.equation} equation needs a height column")
        if self.pal_by is not None and any(not 1.0 <= v <= 3.0 for v in self.pal_by.values.values()):
            raise ValueError("a PAL is total over basal expenditure, between 1.0 and 3.0")
        return self

    def reads(self) -> list[str]:
        from turbotab.core.methods.misreporting import rule_columns

        return rule_columns(self)


def _rule_kind(value: Any) -> str:
    """A rule's kind; a rule written before the Goldberg screen existed is a range."""
    if isinstance(value, Mapping):
        return str(value.get("kind") or "range")
    return str(getattr(value, "kind", None) or "range")


EligibilityRule = Annotated[
    Union[Annotated[ExclusionRule, Tag("range")], Annotated[GoldbergRule, Tag("goldberg")]],
    Discriminator(_rule_kind),
]
RULE_ADAPTER: TypeAdapter[Any] = TypeAdapter(EligibilityRule)


def as_rule(value: Any) -> "ExclusionRule | GoldbergRule":
    """An eligibility rule from a model or its JSON form."""
    if isinstance(value, (ExclusionRule, GoldbergRule)):
        return value
    return RULE_ADAPTER.validate_python(value)


class SensitivityAnalysis(_Value):
    """One analysis beside the primary: the same model on the rows these rules keep instead."""

    label: str = Field(min_length=1)
    rules: list[EligibilityRule] = Field(default_factory=list)  # [] keeps every row


MeasurementErrorMethod = Literal["none", "regression_calibration"]


class MeasurementErrorSpec(_Value):
    method: MeasurementErrorMethod
    exposures: list[str] = Field(default_factory=list)  # [] = every energy-adjusted exposure
    n_boot: int = Field(default=200, ge=50, le=2000)


class EnergyAdjustment(_Value):
    method: EnergyMethod
    energy_column: str | None = None
    nutrients: list[str] = Field(default_factory=list)
    log_transform: bool = False
    strata: str | None = None


# How the training rows validate a model (audit ME-11; turbotab/core/models/validation.py):
# k-fold once; repeated k-fold; Harrell's bootstrap optimism correction (alongside one k-fold run);
# or internal–external validation, one fold per level of a cluster column. Independent of the
# holdout, which seals rows for one final score.
Validation = Literal["kfold", "repeated_kfold", "bootstrap", "internal_external"]
REPEATS = 10  # repeated k-fold's default: 10 × 5
MAX_REPEATS = 50
OPTIMISM_BOOT = 200  # bootstrap optimism correction's default resamples (audit WP9: B ≈ 200)


def _validation_fields(validation: str, cluster: str | None) -> None:
    if validation == "internal_external" and not cluster:
        raise ValueError("internal–external validation needs the cluster column whose levels are "
                         "the folds")
    if cluster and validation != "internal_external":
        raise ValueError("a cluster column is read only by internal–external validation")


class SplitSpec(_Value):
    holdout: float = Field(ge=0.0, le=0.4)
    seed: int = 0
    folds: int = Field(default=5, ge=2, le=10)
    validation: Validation = "kfold"
    repeats: int = Field(default=REPEATS, ge=2, le=MAX_REPEATS)  # repeated_kfold
    n_boot: int = Field(default=OPTIMISM_BOOT, ge=20, le=2000)  # bootstrap
    cluster: str | None = None  # internal_external: each level is a fold

    @field_validator("holdout")
    @classmethod
    def _holdout(cls, value: float) -> float:
        if value != 0.0 and value < 0.1:
            raise ValueError("a holdout is 0 (cross-validation only) or between 0.1 and 0.4")
        return value

    @model_validator(mode="after")
    def _scheme(self) -> "SplitSpec":
        _validation_fields(self.validation, self.cluster)
        return self


# A percentile band needs 1,000 to 2,000 refits (Carpenter & Bithell 2000, Stat Med 19:1141).
MAX_BOOT = 2000


# The scale a substitution moves energy on: kcal (the same amount on every row), or a share of each
# row's own total energy, "5% of energy from X replaced by Y", the field's expected figure
# (NUTRITION_PACK §05; audit B24, D19).
SubstitutionScale = Literal["kcal", "percent_energy"]


class SubstitutionSpec(_Value):
    donor: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    step_kcal: float = Field(default=100.0, gt=0)
    n_boot: int = Field(default=0, ge=0, le=MAX_BOOT)
    # Kept under inference although energy sources are missing from the model (audit ME-05): the
    # recorded attestation that the curve carries their confounding.
    acknowledged: bool = False
    scale: SubstitutionScale = "kcal"
    step_percent: float = Field(default=5.0, gt=0, le=50)  # percentage points of energy per step


class MissingSpec(_Value):
    """The missing-values answer: columns left out of the predictors, then a strategy for the rest.

    ``m``: the imputations under multiple imputation. ``below_detection`` and ``censored_columns``:
    how the blanks of columns whose values lie below a detection limit are filled. ``acknowledged``
    and ``reason``: the recorded attestation that keeps a blocked answer (a single fill or the
    missing-indicator method under inference; a median fill of non-detections, with its reason).
    """

    strategy: MissingStrategy
    drop_columns: list[str] = Field(default_factory=list)
    categorical: Literal["missing_category", "impute"] = "impute"
    indicators: bool = False
    m: int = Field(default=20, ge=20, le=200)
    below_detection: BelowDetection | None = None
    censored_columns: list[str] = Field(default_factory=list)
    acknowledged: bool = False
    reason: str | None = None


class SetRoles(_DecisionModel):
    """The confirmed reading of every column except the outcome."""

    kind: Literal["set_roles"] = "set_roles"
    roles: dict[str, Role] = Field(min_length=1)


class SetEnergyAdjustment(_DecisionModel):
    kind: Literal["set_energy_adjustment"] = "set_energy_adjustment"
    method: EnergyMethod
    energy_column: str | None = None
    nutrients: list[str] = Field(default_factory=list)
    log_transform: bool = False
    strata: str | None = None


class SetExclusions(_DecisionModel):
    """Row exclusions; an empty list is the answer "keep every row"."""

    kind: Literal["set_exclusions"] = "set_exclusions"
    rules: list[EligibilityRule]


class SetSensitivity(_DecisionModel):
    """Analyses beside the primary, each on the rows its own exclusion rules keep (audit ME-16).

    Banna et al. 2017: "analyses in the total sample without exclusion of participants should
    also be conducted and reported". An empty list is the answer "no sensitivity analysis".
    """

    kind: Literal["set_sensitivity"] = "set_sensitivity"
    analyses: list[SensitivityAnalysis]

    @field_validator("analyses")
    @classmethod
    def _unique(cls, value: list[SensitivityAnalysis]) -> list[SensitivityAnalysis]:
        labels = [a.label for a in value]
        if len(set(labels)) != len(labels):
            raise ValueError("each sensitivity analysis needs a label of its own")
        return value


class SetMeasurementError(_DecisionModel):
    """Whether energy-adjusted exposures are corrected for day-to-day error in the recalls
    (univariate regression calibration; audit IN-22, Freedman et al. 2011)."""

    kind: Literal["set_measurement_error"] = "set_measurement_error"
    method: MeasurementErrorMethod
    exposures: list[str] = Field(default_factory=list)
    n_boot: int = Field(default=200, ge=50, le=2000)

    @field_validator("exposures")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each exposure may be named only once")
        return value


class SetMissing(_DecisionModel):
    """How missing predictor values are handled.

    ``drop_columns`` leave the predictors first (a mostly blank column that means "not asked"),
    and the strategy applies to the predictors that remain; the cohort and the design read both.
    """

    kind: Literal["set_missing"] = "set_missing"
    strategy: MissingStrategy
    drop_columns: list[str] = Field(default_factory=list)
    # M2 — routed by dtype and mechanism (ROADMAP lockbox constitution §07): binary/categorical
    # blanks may become their own "Missing" level, which keeps the signal when a blank means
    # "not asked"; numeric columns may carry a missing indicator beside the in-fold imputation.
    categorical: Literal["missing_category", "impute"] = "impute"
    indicators: bool = False
    # WP7 (audit §5, ME-01, ME-08): missing data by purpose. Under inference, multiple imputation
    # with ``m`` imputations; a single fill or the missing-indicator method only with the recorded
    # attestation (``acknowledged``); non-detections filled by ``below_detection``, and a median
    # fill of them only with a ``reason``.
    m: int = Field(default=20, ge=20, le=200)
    below_detection: BelowDetection | None = None
    censored_columns: list[str] = Field(default_factory=list)
    acknowledged: bool = False
    reason: str | None = None

    @field_validator("drop_columns", "censored_columns")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each column may be named only once")
        return value

    @field_validator("reason")
    @classmethod
    def _said(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("a reason must say something")
        return value.strip() if value is not None else None


class SetSplit(_DecisionModel):
    kind: Literal["set_split"] = "set_split"
    holdout: float = Field(ge=0.0, le=0.4)
    seed: int = 0
    folds: int = Field(default=5, ge=2, le=10)
    validation: Validation = "kfold"
    repeats: int = Field(default=REPEATS, ge=2, le=MAX_REPEATS)
    n_boot: int = Field(default=OPTIMISM_BOOT, ge=20, le=2000)
    cluster: str | None = None

    @model_validator(mode="after")
    def _scheme(self) -> "SetSplit":
        _validation_fields(self.validation, self.cluster)
        return self


class SelectModels(_DecisionModel):
    kind: Literal["select_models"] = "select_models"
    models: list[str] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each model family may be named only once")
        return value


class SetSubstitution(_DecisionModel):
    """The substitution to draw; ``n_boot > 0`` adds a band from that many bootstrap refits."""

    kind: Literal["set_substitution"] = "set_substitution"
    donor: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    step_kcal: float = Field(default=100.0, gt=0)
    n_boot: int = Field(default=0, ge=0, le=MAX_BOOT)
    # The attestation exit of the omitted-sources block under inference (audit ME-05): "keep this
    # swap; the curve carries the confounding of the energy sources not in the model".
    acknowledged: bool = False
    # "percent_energy": each step moves ``step_percent`` % of each row's total energy.
    scale: SubstitutionScale = "kcal"
    step_percent: float = Field(default=5.0, gt=0, le=50)


# ── M2 kinds (docs/turbotab-next/M2_CONTRACT.md) ─────────────────────────────
# The opening sequence (OPENING_SEQUENCE.md), the seal, and findings with dispositions.

Orientation = Literal["sample_major", "feature_major"]
RepeatKind = Literal["repeats", "time_points"]
AggregationMethod = Literal["mean", "first", "last", "change"]
FindingAction = Literal["applied", "deferred", "dismissed"]


# "unknown" is the user's explicit "I don't know" (OPENING_SEQUENCE §03 grain; M2_CONTRACT §12.2):
# the only answer that makes the seal's basis ``undetermined``. Never inferred from silence.
GrainAnswer = Literal["one_row_per_unit", "repeated", "unknown"]


class GrainSpec(_Value):
    grain: GrainAnswer
    id_column: str | None = None  # the column naming the unit (person, sample) when repeated
    # The user kept this answer over the data's contradiction (turbotab/grain.py): a noted
    # disagreement the methods carry as a stated limitation.
    acknowledged: bool = False


CombineRule = Literal["mean", "first", "last", "change", "mode"]


class RepeatSpec(_Value):
    repeat_kind: RepeatKind
    time_column: str | None = None
    # The order of a text time column's levels ("baseline", "month_6", "month_12"): what puts a
    # unit's records in order when the labels do not say it themselves (audit MA-03).
    levels: list[str] | None = None


class AggregationSpec(_Value):
    method: AggregationMethod
    outcome: Literal["mean", "first", "last"] | None = None  # when the outcome varies within a unit
    # A column's own rule, over the one the method and the column's kind give it (audit MA-14):
    # e.g. age at every visit kept at baseline under "change".
    columns: dict[str, CombineRule] = Field(default_factory=dict)


class FeatureTableSpec(_Value):
    """How a features-in-rows table's columns read before it is turned (audit MA-04)."""

    label: str | None = None  # the column naming the features (None: rows are named row_<i>)
    annotations: list[str] = Field(default_factory=list)  # columns describing features, never samples


class TemporalSpec(_Value):
    temporal: bool
    time_column: str | None = None


class FindingDisposition(_Value):
    action: FindingAction
    option: str | None = None  # the repair option applied
    params: dict[str, Any] = Field(default_factory=dict)
    to: str | None = None  # the question key a deferral resurfaces at
    reason: str | None = None  # why it was dismissed, when given


class SetOrientation(_DecisionModel):
    """Which way round an assay table is; feature-major is transposed before any diagnosis."""

    kind: Literal["set_orientation"] = "set_orientation"
    orientation: Orientation


class SetEvent(_DecisionModel):
    """Which level of a binary outcome is the event. Stands only while ``column`` is the target."""

    kind: Literal["set_event"] = "set_event"
    column: str = Field(min_length=1)
    level: str = Field(min_length=1)


class FollowUpSpec(_Value):
    """How long each row of a time-to-event outcome was observed (WP12,
    ``turbotab/core/models/survival.py``): at risk from ``entry`` (0 when not named) to ``time``,
    when the event happened or follow-up ended without it, both on one time scale."""

    time_column: str
    entry_column: str | None = None  # staggered or delayed entry: when each row came under observation


class SetFollowUp(_DecisionModel):
    """The follow-up of a time-to-event outcome. Stands only while ``column`` is the target."""

    kind: Literal["set_follow_up"] = "set_follow_up"
    column: str = Field(min_length=1)
    time_column: str = Field(min_length=1)
    entry_column: str | None = None


class SetGrain(_DecisionModel):
    kind: Literal["set_grain"] = "set_grain"
    grain: GrainAnswer
    id_column: str | None = None
    # The attestation exit of a grain contradiction: "my answer is right, the data is like this".
    acknowledged: bool = False


class SetRepeatKind(_DecisionModel):
    kind: Literal["set_repeat_kind"] = "set_repeat_kind"
    repeat_kind: RepeatKind
    time_column: str | None = None
    levels: list[str] | None = None  # the declared order of a text time column's levels

    @field_validator("levels")
    @classmethod
    def _distinct(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("each level may be placed only once")
        return value


class SetUnit(_DecisionModel):
    """When a unit repeats: is one row of the analysis a unit (combined) or a record?"""

    kind: Literal["set_unit"] = "set_unit"
    unit: Literal["unit", "row"]


class SetAggregation(_DecisionModel):
    kind: Literal["set_aggregation"] = "set_aggregation"
    method: AggregationMethod
    outcome: Literal["mean", "first", "last"] | None = None
    columns: dict[str, CombineRule] = Field(default_factory=dict)  # a column's own rule


class SetFeatureTable(_DecisionModel):
    """How a features-in-rows table reads: the column naming the features, and the columns that
    describe them (m/z, retention time, IDs). Turning it makes every other column a sample."""

    kind: Literal["set_feature_table"] = "set_feature_table"
    label: str | None = None
    annotations: list[str] = Field(default_factory=list)


class SetCategorical(_DecisionModel):
    """Columns whose numbers are codes for categories (RIDRETH3: 1, 2, 3, 4, 6, 7): each enters
    the models as indicators, one per level after the first, never as one straight line."""

    kind: Literal["set_categorical"] = "set_categorical"
    columns: list[str]

    @field_validator("columns")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each column may be named only once")
        return value


# WP12a (audit ME-17, ME-19): the form an exposure takes, and the order of an ordinal outcome.

ExposureFormKind = Literal["linear", "spline", "quintiles"]


class ExposureFormSpec(_Value):
    """How one numeric predictor enters the models: a straight line, a restricted cubic spline
    (``knots`` 3–5, at Harrell's percentiles), or quintile indicators with a trend test."""

    form: ExposureFormKind
    knots: int | None = Field(default=None, ge=3, le=5)


class SetExposureForm(_DecisionModel):
    """The form of one predictor (``turbotab.core.methods.exposure_form``); one entry per column."""

    kind: Literal["set_exposure_form"] = "set_exposure_form"
    column: str = Field(min_length=1)
    form: ExposureFormKind
    knots: int | None = Field(default=None, ge=3, le=5)


class SetOutcomeOrder(_DecisionModel):
    """The order of an ordinal outcome's levels, lowest first. Stands only while ``column`` is the
    target. Numbers are ordered by value without it; text levels need it (asked, never inferred)."""

    kind: Literal["set_outcome_order"] = "set_outcome_order"
    column: str = Field(min_length=1)
    levels: list[str] = Field(min_length=2)

    @field_validator("levels")
    @classmethod
    def _distinct(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each level may be placed only once")
        return value


class SetTemporal(_DecisionModel):
    kind: Literal["set_temporal"] = "set_temporal"
    temporal: bool
    time_column: str | None = None


# WP10 (audit §5, ME-06): under inference with survey design columns, whose estimate it is.
SurveyEstimand = Literal["population", "sample"]


class SurveySpec(_Value):
    """The survey answer: the surveyed population (design-based, ``weight`` and the design named),
    or these participants (unweighted, recorded as such; ``turbotab/core/methods/survey.py``)."""

    estimand: SurveyEstimand
    weight: str | None = None
    strata: str | None = None
    psu: str | None = None
    cycle: str | None = None  # the cycle column, when the table pools survey cycles
    four_year_weight: str | None = None  # 1999–2002's four-year weight, when 1999–2000 is pooled
    # A weight with no strata or no PSU named: the user attests the intervals' sampling units.
    acknowledged: bool = False


class SetSurvey(_DecisionModel):
    """Whose estimate it is under a survey design (``SurveySpec``)."""

    kind: Literal["set_survey"] = "set_survey"
    estimand: SurveyEstimand
    weight: str | None = None
    strata: str | None = None
    psu: str | None = None
    cycle: str | None = None
    four_year_weight: str | None = None
    acknowledged: bool = False


class OpenSeal(_DecisionModel):
    """Open the held-out rows: once, at the end. Held-out scores are withheld until then.

    ``family`` is the final model, declared on cross-validation before any held-out score is seen
    (AUDIT_REPORT §5 WP8, ME-13; ``turbotab/core/models/selection.py``): its held-out score is the
    reported result. Required when several families are fitted; with one, the opening declares it.
    Null only in records from before the rule.
    """

    kind: Literal["open_seal"] = "open_seal"
    family: str | None = None


class ApplyRepair(_DecisionModel):
    """Apply one of a finding's repair options. Row-local repairs rewrite the working table now;
    statistical ones are recorded and executed in-fold (ROADMAP lockbox constitution §06)."""

    kind: Literal["apply_repair"] = "apply_repair"
    finding_id: str = Field(min_length=1)
    option: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class DeferFinding(_DecisionModel):
    """Hold a finding until the question it belongs to; it resurfaces there, attributed."""

    kind: Literal["defer_finding"] = "defer_finding"
    finding_id: str = Field(min_length=1)
    to: str = Field(min_length=1)


class DismissFinding(_DecisionModel):
    kind: Literal["dismiss_finding"] = "dismiss_finding"
    finding_id: str = Field(min_length=1)
    reason: str | None = None


Decision = Annotated[
    Union[
        SetLens, SetTarget, SetTask, SetPurpose, Revert,
        SetRoles, SetEnergyAdjustment, SetExclusions, SetMissing, SetSplit,
        SelectModels, SetSubstitution,
        SetOrientation, SetEvent, SetGrain, SetRepeatKind, SetUnit, SetAggregation, SetTemporal,
        OpenSeal, ApplyRepair, DeferFinding, DismissFinding,
        SetFeatureTable, SetCategorical, SetSurvey,
        SetExposureForm, SetOutcomeOrder, SetFollowUp,
        SetSensitivity, SetMeasurementError,
    ],
    Field(discriminator="kind"),
]
DECISION_ADAPTER: TypeAdapter[Any] = TypeAdapter(Decision)


def parse_decision(obj: Any) -> Any:
    """Validate a dict (or a decision model) into a :data:`Decision`."""
    return DECISION_ADAPTER.validate_python(obj)


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    id: str
    seq: int = Field(ge=1)
    at: AwareDatetime
    note: str | None = None
    # The sentence the Record shows, authored by the server when the decision is
    # recorded (DESIGN_LANGUAGE §05.1: the receipt is a quotation, never a
    # composition). Backticks mark data values. None only for M0-era records.
    sentence: str | None = None
    # Recorded after the held-out rows were opened (``open_seal``; M2_CONTRACT §3). Set when the
    # record is appended, from the log as it stood; never by rewriting a past line, so every
    # record from before the opening (and every pre-M2 line) reads False.
    post_seal: bool = False
    decision: Decision

    @field_validator("at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)


class ProjectState(BaseModel):
    """The fold of the decision log. Later milestones add slots as fields."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    lens: list[Lens] | None = None
    target: str | None = None
    task: Task | None = None
    purpose: Purpose | None = None
    roles: dict[str, Role] | None = None
    energy_adjustment: EnergyAdjustment | None = None
    exclusions: list[EligibilityRule] | None = None
    missing: MissingSpec | None = None
    split: SplitSpec | None = None
    models: list[str] | None = None
    substitution: SubstitutionSpec | None = None
    # M2
    orientation: Orientation | None = None
    event: str | None = None  # the event level of a binary target (holds while its column is the target)
    grain: GrainSpec | None = None
    repeat_kind: RepeatSpec | None = None
    unit: Literal["unit", "row"] | None = None
    aggregation: AggregationSpec | None = None
    temporal: TemporalSpec | None = None
    seal_opened: bool | None = None
    findings: dict[str, FindingDisposition] | None = None  # finding id -> its disposition
    # WP1 (audit §5): what values mean
    feature_table: FeatureTableSpec | None = None  # a features-in-rows table's label and annotations
    categorical: list[str] | None = None  # integer columns that are codes for categories
    # WP10 (audit §5): the surveyed population or these participants, under a survey design
    survey: SurveySpec | None = None
    # WP12a (audit §5): each formed predictor's form, and an ordinal outcome's declared order
    exposure_forms: dict[str, ExposureFormSpec] | None = None
    outcome_order: list[str] | None = None  # holds while its column is the target
    # WP12: a time-to-event outcome's follow-up (holds while its column is the target)
    follow_up: FollowUpSpec | None = None
    # WP12 (audit §5): methods a reviewer expects
    sensitivity: list[SensitivityAnalysis] | None = None  # analyses beside the primary's rows
    measurement_error: MeasurementErrorSpec | None = None  # regression calibration, or none

    @field_validator("missing", mode="before")
    @classmethod
    def _strategy_alone(cls, value: Any) -> Any:
        """A bare strategy (``"impute"``) is the answer with no column left out."""
        return {"strategy": value} if isinstance(value, str) else value


def missing_strategy(state: Any) -> MissingStrategy | None:
    """The missing-values strategy the state holds, or None while unanswered."""
    spec = getattr(state, "missing", None)
    return None if spec is None else spec.strategy


def left_out(state: Any) -> list[str]:
    """Columns left out of the predictors: by the missing-values answer, then by a repair that
    marked a column unusable (``turbotab.core.repairs``)."""
    spec = getattr(state, "missing", None)
    out = list(spec.drop_columns) if spec is not None else []
    from turbotab.core.repairs import unusable_columns  # repairs imports this module

    return out + [c for c in unusable_columns(state) if c not in out]


class Refusal(Exception):
    """The "refuse" rung: HTTP 409 ``{error: {code, message, exits}}``, never recorded.

    ``exits`` are ways forward, each ``{label, decision|None}``; a decision may
    be given as a model or a dict and is stored as its JSON form.
    """

    def __init__(self, code: str, message: str, exits: Iterable[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exits: list[dict[str, Any]] = [_normalize_exit(e) for e in exits]

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "exits": self.exits}}

    def __repr__(self) -> str:
        return f"Refusal({self.code!r}, {self.message!r}, exits={self.exits!r})"


def _normalize_exit(item: Mapping[str, Any]) -> dict[str, Any]:
    decision = item.get("decision")
    if decision is not None:
        decision = parse_decision(decision).model_dump(mode="json")
    return {"label": str(item["label"]), "decision": decision}


# ── the registry: one handler per kind ──────────────────────────────────────

SLOTS: dict[str, str] = {}
"""kind -> the ProjectState slot it writes. ``revert`` writes no slot."""

_SLOT_VALUE: dict[str, Callable[[Any], Any]] = {}
_HOLDS: dict[str, Callable[[Any, Mapping[str, Any]], bool]] = {}
_KEYS: dict[str, Callable[[Any], str]] = {}
_VALIDATORS: dict[str, list[Callable[[Any, Any], None]]] = {}
_COMPLETIONS: dict[str, list[Callable[[Any, Any], Any]]] = {}


def kind_of(model_cls: type[BaseModel]) -> str:
    field = model_cls.model_fields.get("kind")
    if field is None or not isinstance(field.default, str):
        raise TypeError(f"{model_cls.__name__} needs a `kind: Literal[...] = ...` field")
    return field.default


def register_kind(
    model_cls: type[BaseModel],
    slot: str,
    *,
    value: Callable[[Any], Any] | None = None,
    holds: Callable[[Any, Mapping[str, Any]], bool] | None = None,
    key: Callable[[Any], str] | None = None,
) -> type[BaseModel]:
    """Declare that decisions of ``model_cls`` write ``slot``.

    ``value(decision)`` gives the slot's new value; by default it is the model's
    single payload field. ``holds(decision, slots)``, when given, makes the write
    conditional: it stands only while it is true of the slots that
    unconditional kinds wrote, and the slot holds its latest write that stands.
    ``key(decision)``, when given, makes the slot a mapping: each decision writes
    one entry (the latest write per key wins, and a revert restores that key's
    earlier entry). Several kinds may write one keyed slot — apply, defer and
    dismiss all write a finding's disposition under its finding id.
    Returns the class, so it also works as a decorator via ``functools.partial``.
    """
    kind = kind_of(model_cls)
    if kind == "revert":
        raise ValueError("revert is built in and writes no slot")
    if slot not in ProjectState.model_fields:
        raise ValueError(f"ProjectState has no slot {slot!r}; add the field first")
    if value is None:
        payload = [name for name in model_cls.model_fields if name != "kind"]
        if len(payload) != 1:
            raise ValueError(f"{model_cls.__name__} has fields {payload}; pass value=")
        name = payload[0]

        def value(decision: Any, _name: str = name) -> Any:
            return getattr(decision, _name)

    SLOTS[kind] = slot
    _SLOT_VALUE[kind] = value
    if key is not None:
        _KEYS[kind] = key
    else:
        _KEYS.pop(kind, None)
    if holds is not None:
        _HOLDS[kind] = holds
    else:
        _HOLDS.pop(kind, None)
    return model_cls


def register_validator(kind: str, fn: Callable[[Any, Any], None], *, first: bool = False) -> None:
    """Add a check for ``kind``. ``fn(decision, ctx)`` raises :class:`Refusal`.

    ``first`` runs it before the checks already registered: a refusal that outranks every other
    reason (the seal's Decision A) is the one the user should read.
    """
    checks = _VALIDATORS.setdefault(kind, [])
    if first:
        checks.insert(0, fn)
    else:
        checks.append(fn)


def register_completion(kind: str, fn: Callable[[Any, Any], Any]) -> None:
    """Add a completion for ``kind``: ``fn(decision, ctx) -> decision`` fills in what the server
    knows and the client may leave out (a repair's parameters), after the validators pass. The
    completed decision is what is previewed and recorded."""
    _COMPLETIONS.setdefault(kind, []).append(fn)


def validate(decision: Any, ctx: Any = None) -> Any:
    """Parse ``decision``, run its kind's validators, then its completions; returns the decision.

    ``ctx`` is whatever the caller knows about the project (for M0: an object or
    mapping with ``columns`` and ``target``); what it does not name is not checked. Revert targets are checked by
    :meth:`DecisionLog.append`, which holds the log.
    """
    decision = parse_decision(decision)
    for fn in _VALIDATORS.get(decision.kind, ()):
        fn(decision, ctx)
    for fn in _COMPLETIONS.get(decision.kind, ()):
        decision = fn(decision, ctx)
    return decision


def _columns_of(ctx: Any) -> set[str] | None:
    if ctx is None:
        return None
    columns = ctx.get("columns") if isinstance(ctx, Mapping) else getattr(ctx, "columns", None)
    if columns is None:
        return None
    names: set[str] = set()
    for column in columns:
        if isinstance(column, str):
            names.add(column)
        elif isinstance(column, Mapping):
            names.add(str(column["name"]))
        else:
            names.add(str(getattr(column, "name")))
    return names


def _target_is_a_column(decision: SetTarget, ctx: Any) -> None:
    if decision.column == ROW_ID:
        raise Refusal(
            "row_identity",
            "The row identity column numbers the rows; it cannot be the target.",
            exits=[{"label": "Choose one of the dataset's own columns", "decision": None}],
        )
    columns = _columns_of(ctx)
    if columns is not None and decision.column not in columns:
        raise Refusal(
            "unknown_column",
            f"There is no column named {decision.column!r} in this dataset.",
            exits=[{"label": "Choose one of the dataset's columns", "decision": None}],
        )


_UNKNOWN = object()


def _target_of(ctx: Any) -> Any:
    """The current target as ``ctx`` knows it (None: unset), or ``_UNKNOWN``."""
    if ctx is None:
        return _UNKNOWN
    if isinstance(ctx, Mapping):
        return ctx.get("target", _UNKNOWN)
    return getattr(ctx, "target", _UNKNOWN)


def _task_is_for_the_target(decision: SetTask, ctx: Any) -> None:
    target = _target_of(ctx)
    if target is _UNKNOWN:
        return
    if target is None:
        raise Refusal(
            "no_target",
            "Choose the outcome first; the task describes it.",
            exits=[{"label": "Choose the outcome", "decision": None}],
        )
    if decision.column != target:
        raise Refusal(
            "not_the_target",
            f"The outcome is {target!r}, not {decision.column!r}; a task answers for the outcome.",
            exits=[{"label": f"Answer the task question for {target}", "decision": None}],
        )


register_kind(SetLens, "lens")
register_kind(SetTarget, "target")
register_kind(
    SetTask,
    "task",
    value=lambda decision: decision.task,
    holds=lambda decision, slots: slots.get("target") == decision.column,
)
register_kind(SetPurpose, "purpose")
register_kind(SetRoles, "roles")
register_kind(SetEnergyAdjustment, "energy_adjustment",
              value=lambda d: EnergyAdjustment(**d.model_dump(exclude={"kind"})))
register_kind(SetExclusions, "exclusions")
register_kind(SetMissing, "missing", value=lambda d: MissingSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetSplit, "split", value=lambda d: SplitSpec(**d.model_dump(exclude={"kind"})))
register_kind(SelectModels, "models")
register_kind(SetSubstitution, "substitution",
              value=lambda d: SubstitutionSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetOrientation, "orientation")
register_kind(SetEvent, "event", value=lambda d: d.level,
              holds=lambda d, slots: slots.get("target") == d.column)
register_kind(SetGrain, "grain", value=lambda d: GrainSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetRepeatKind, "repeat_kind",
              value=lambda d: RepeatSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetUnit, "unit")
register_kind(SetAggregation, "aggregation",
              value=lambda d: AggregationSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetTemporal, "temporal",
              value=lambda d: TemporalSpec(**d.model_dump(exclude={"kind"})))
register_kind(OpenSeal, "seal_opened", value=lambda d: True)
register_kind(ApplyRepair, "findings", key=lambda d: d.finding_id,
              value=lambda d: FindingDisposition(action="applied", option=d.option, params=d.params))
register_kind(DeferFinding, "findings", key=lambda d: d.finding_id,
              value=lambda d: FindingDisposition(action="deferred", to=d.to))
register_kind(DismissFinding, "findings", key=lambda d: d.finding_id,
              value=lambda d: FindingDisposition(action="dismissed", reason=d.reason))
register_kind(SetFeatureTable, "feature_table",
              value=lambda d: FeatureTableSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetCategorical, "categorical")
register_kind(SetSurvey, "survey", value=lambda d: SurveySpec(**d.model_dump(exclude={"kind"})))
register_kind(SetExposureForm, "exposure_forms", key=lambda d: d.column,
              value=lambda d: ExposureFormSpec(form=d.form, knots=d.knots))
register_kind(SetOutcomeOrder, "outcome_order", value=lambda d: list(d.levels),
              holds=lambda d, slots: slots.get("target") == d.column)
register_kind(SetFollowUp, "follow_up",
              value=lambda d: FollowUpSpec(time_column=d.time_column, entry_column=d.entry_column),
              holds=lambda d, slots: slots.get("target") == d.column)
register_kind(SetSensitivity, "sensitivity")
register_kind(SetMeasurementError, "measurement_error",
              value=lambda d: MeasurementErrorSpec(**d.model_dump(exclude={"kind"})))
register_validator("set_target", _target_is_a_column)
register_validator("set_task", _task_is_for_the_target)
register_validator("set_split", lambda d, ctx: _cluster_is_a_column_with_levels(d, ctx))


# ── M1 validators (M1_CONTRACT.md §2) ────────────────────────────────────────
# Each reads only what ``ctx`` names; what it does not name is not checked. The
# server's ctx (turbotab/server/service.py DecisionContext) carries ``columns``,
# ``column_info`` ({name: {dtype, n_unique, n_missing}}), ``state`` (the state
# before this decision), ``task`` (the answered or detected task) and ``target``.

NUMERIC_DTYPES = ("numeric", "integer")
MAX_CLASSES_MULTICLASS = 20
MAX_STRATA_LEVELS = 10
PREDICTOR_ROLES = ("exposure", "covariate", "energy")


def _ctx(ctx: Any, name: str, default: Any = None) -> Any:
    if ctx is None:
        return default
    if isinstance(ctx, Mapping):
        return ctx.get(name, default)
    return getattr(ctx, name, default)


def _info(ctx: Any, column: str) -> Mapping[str, Any] | None:
    info = _ctx(ctx, "column_info")
    if not info:
        return None
    return info.get(column)


def _state(ctx: Any) -> "ProjectState | None":
    return _ctx(ctx, "state")


def _and(items: Sequence[str]) -> str:
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _cluster_is_a_column_with_levels(decision: "SetSplit", ctx: Any) -> None:
    """Internal–external validation folds by a column's levels (audit E16): it must be one of the
    table's columns, not the outcome, with at least two levels."""
    cluster = decision.cluster
    if not cluster:
        return
    columns = _columns_of(ctx)
    if cluster == ROW_ID or (columns is not None and cluster not in columns):
        raise Refusal("unknown_column", f"This table has no column named `{cluster}`.",
                      exits=[{"label": "Choose the column that names each site, study or period",
                              "decision": None}])
    target = _ctx(ctx, "target") or getattr(_state(ctx), "target", None)
    if cluster == target:
        raise Refusal("cluster_is_outcome",
                      f"`{cluster}` is the outcome; the folds of internal–external validation are "
                      f"the levels of a column that says where or when a row came from.",
                      exits=[{"label": "Choose another column", "decision": None}])
    info = _info(ctx, cluster)
    if info is not None and int(info.get("n_unique") or 0) < 2:
        raise Refusal("one_cluster",
                      f"`{cluster}` has one level, so no cluster could be held out.",
                      exits=[{"label": "Choose a column with two or more levels", "decision": None}])


def _task_fits_the_outcome(decision: SetTask, ctx: Any) -> None:
    info = _info(ctx, decision.column)
    if info is None:
        return
    n_unique = int(info.get("n_unique") or 0)
    numeric = info.get("dtype") in NUMERIC_DTYPES
    fits: list[str] = []
    if n_unique == 2:
        fits.append("binary")
    if 2 < n_unique <= MAX_CLASSES_MULTICLASS:
        fits.extend(["multiclass", "ordinal"])
    if numeric and n_unique > 2:
        fits.append("regression")
    exits = [{"label": f"Treat it as {task}", "decision": SetTask(column=decision.column, task=task)}
             for task in fits]
    if decision.task == "binary" and n_unique != 2:
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` has {n_unique:,} distinct values; a binary outcome has exactly 2.",
            exits=exits or [{"label": "Choose another outcome", "decision": None}],
        )
    if decision.task == "multiclass" and n_unique > MAX_CLASSES_MULTICLASS:
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` has {n_unique:,} distinct values; multiclass takes at most "
            f"{MAX_CLASSES_MULTICLASS}.",
            exits=exits or [{"label": "Choose another outcome", "decision": None}],
        )
    if decision.task == "ordinal" and not 2 < n_unique <= MAX_CLASSES_MULTICLASS:
        why = ("two levels are a binary outcome" if n_unique == 2 else
               f"an ordinal outcome takes 3 to {MAX_CLASSES_MULTICLASS} levels")
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` has {n_unique:,} distinct values; {why}.",
            exits=[e for e in exits if e["decision"].task != "ordinal"]
            or [{"label": "Choose another outcome", "decision": None}],
        )
    if decision.task == "time_to_event" and n_unique != 2:
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` has {n_unique:,} distinct values; a time-to-event outcome's "
            f"column says whether the event happened, so it has exactly 2.",
            exits=exits or [{"label": "Choose another outcome", "decision": None}],
        )


def _follow_up_belongs_to_the_outcome(decision: SetFollowUp, ctx: Any) -> None:
    """A follow-up answers for the outcome it names, with numeric columns other than it."""
    target = _target_of(ctx)
    if target is not _UNKNOWN:
        if target is None:
            raise Refusal("no_target", "Choose the outcome first; the follow-up belongs to it.",
                          exits=[{"label": "Choose the outcome", "decision": None}])
        if decision.column != target:
            raise Refusal(
                "not_the_target",
                f"The outcome is `{target}`, not `{decision.column}`; a follow-up answers for the "
                f"outcome.",
                exits=[{"label": f"Answer the follow-up question for `{target}`", "decision": None}])
    # A follow-up is read only by a time-to-event model: recorded beside a yes/no task, the record
    # would say the outcome "was analyzed as a time to event" over a logistic fit (the verifier's
    # RO-03 repro). The task must be time to event first, and only an answer makes it one:
    # detection reads a 0/1 column as binary and never as a time to event
    # (``stages.target.target_info_stage``). So the recorded answer decides, and while none is
    # recorded the follow-up is refused whether detection has finished or not (the gate's race:
    # posted before ``target_info`` was fresh, the task read as None and the follow-up was
    # accepted, then the task step was skipped as binary and the fit was logistic).
    state = _state(ctx)
    answered = getattr(state, "task", None) if state is not None else None
    task = answered if answered is not None else _ctx(ctx, "task", _UNKNOWN)
    if task is _UNKNOWN and state is not None:
        task = None  # the state is known and holds no answer: the task is not settled
    if task is not _UNKNOWN and task != "time_to_event":
        if task is None:
            message = (f"`{decision.column}`'s task is not settled yet, and only a time-to-event "
                       f"model reads a follow-up time; a yes/no column is read as binary unless it "
                       f"is declared a time to event. Analyze it as a time to event first.")
        else:
            message = (f"`{decision.column}` is analyzed as a {str(task).replace('_', '-')} outcome, "
                       f"and only a time-to-event model reads a follow-up time. Analyze it as a "
                       f"time to event first.")
        raise Refusal(
            "not_time_to_event", message,
            exits=[{"label": f"Analyze `{decision.column}` as a time to event",
                    "decision": SetTask(column=decision.column, task="time_to_event")},
                   {"label": f"Keep `{decision.column}` as it is, without a follow-up",
                    "decision": None}])
    named = [decision.time_column] + ([decision.entry_column] if decision.entry_column else [])
    columns = _columns_of(ctx)
    if columns is not None:
        unknown = [c for c in named if c not in columns or c == ROW_ID]
        if unknown:
            raise Refusal("unknown_column", f"This dataset has no column named {_and(unknown)}.",
                          exits=[{"label": "Name one of the dataset's columns", "decision": None}])
    if decision.column in named:
        raise Refusal("follow_up_is_the_outcome",
                      f"`{decision.column}` is the event; the follow-up is another column, the time "
                      f"each row was observed until.",
                      exits=[{"label": "Name the follow-up time column", "decision": None}])
    if decision.entry_column is not None and decision.entry_column == decision.time_column:
        raise Refusal("entry_is_the_end",
                      f"`{decision.time_column}` cannot be both where follow-up starts and where it "
                      f"ends.",
                      exits=[{"label": "Leave the entry column out",
                              "decision": SetFollowUp(column=decision.column,
                                                      time_column=decision.time_column)}])
    for column in named:
        info = _info(ctx, column)
        if info is not None and info.get("dtype") not in NUMERIC_DTYPES:
            raise Refusal("not_numeric",
                          f"`{column}` holds {info.get('dtype')} values; a follow-up time is a "
                          f"number on one time scale.",
                          exits=[{"label": "Name a numeric time column", "decision": None}])


def _roles_name_real_columns(decision: SetRoles, ctx: Any) -> None:
    columns = _columns_of(ctx)
    if columns is not None:
        unknown = [c for c in decision.roles if c not in columns or c == ROW_ID]
        if unknown:
            rest = {c: r for c, r in decision.roles.items() if c not in unknown}
            exits = [{"label": "Keep the roles of the real columns", "decision": SetRoles(roles=rest)}] \
                if rest else []
            raise Refusal(
                "unknown_column",
                f"This dataset has no column named {_and(unknown)}.",
                exits=exits + [{"label": "Choose roles for the dataset's columns", "decision": None}],
            )
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is not None and target in decision.roles:
        rest = {c: r for c, r in decision.roles.items() if c != target}
        raise Refusal(
            "target_has_role",
            f"`{target}` is the outcome, so it cannot also be {decision.roles[target]!s}.",
            exits=[{"label": "Leave the outcome out of the roles", "decision": SetRoles(roles=rest)}]
            if rest else [],
        )
    if not any(role in PREDICTOR_ROLES for role in decision.roles.values()):
        raise Refusal(
            "no_predictors",
            "No column is an exposure, a covariate or energy, so the models would have nothing to use.",
            exits=[{"label": "Mark at least one column as an exposure or a covariate", "decision": None}],
        )


def _rule_without(decision: SetExclusions, index: int) -> SetExclusions:
    return SetExclusions(rules=[r for i, r in enumerate(decision.rules) if i != index])


def _goldberg_reads_numbers(rule: GoldbergRule, ctx: Any, drop: Mapping[str, Any]) -> None:
    """A Goldberg screen's columns exist, and the ones it does arithmetic on are numbers."""
    columns = _columns_of(ctx)
    for name in rule.reads():
        if columns is not None and (name not in columns or name == ROW_ID):
            raise Refusal("unknown_column", f"This dataset has no column named `{name}`.", exits=[drop])
    numbers = [rule.column, rule.age, rule.weight, *([rule.height] if rule.height else []),
               *([rule.days_column] if rule.days_column else [])]
    for name in numbers:
        info = _info(ctx, name)
        if info is not None and info.get("dtype") not in NUMERIC_DTYPES:
            raise Refusal("not_numeric",
                          f"`{name}` is {info.get('dtype')}, not numbers, so the Goldberg screen "
                          f"cannot compute with it.", exits=[drop])


def _exclusions_are_ranges_on_numbers(decision: SetExclusions, ctx: Any) -> None:
    columns = _columns_of(ctx)
    for i, rule in enumerate(decision.rules):
        drop = {"label": f"Drop the rule on `{rule.column}`", "decision": _rule_without(decision, i)}
        if isinstance(rule, GoldbergRule):
            _goldberg_reads_numbers(rule, ctx, drop)
            continue
        for name in [rule.column] + ([rule.by.column] if rule.by is not None else []):
            if columns is not None and (name not in columns or name == ROW_ID):
                raise Refusal("unknown_column", f"This dataset has no column named `{name}`.", exits=[drop])
        info = _info(ctx, rule.column)
        if info is not None and info.get("dtype") not in NUMERIC_DTYPES:
            raise Refusal(
                "not_numeric",
                f"`{rule.column}` is {info.get('dtype')}, not numbers, so it has no range to keep.",
                exits=[drop],
            )
        if rule.low is None and rule.high is None and not (rule.by and rule.by.ranges):
            raise Refusal("no_bounds", f"The rule on `{rule.column}` sets no lower or upper bound.",
                          exits=[drop])
        bounds = [(None, rule.low, rule.high)]
        if rule.by is not None:
            bounds += [(level, lo, hi) for level, (lo, hi) in rule.by.ranges.items()]
        for level, low, high in bounds:
            if low is not None and high is not None and low >= high:
                where = f" for `{level}`" if level is not None else ""
                exits = [drop]
                if level is None and low > high:
                    swapped = list(decision.rules)
                    swapped[i] = rule.model_copy(update={"low": high, "high": low})
                    exits.insert(0, {"label": "Swap the bounds", "decision": SetExclusions(rules=swapped)})
                raise Refusal(
                    "empty_range",
                    f"The range on `{rule.column}`{where} runs from {low:g} to {high:g}; "
                    f"the lower bound must be below the upper.",
                    exits=exits,
                )


# ── no eligibility rule on the outcome (audit RO-01) ─────────────────────────
# Keeping rows by the outcome's own value selects on the value the model explains: with a true
# fiber slope of −0.25, a `bmi 18.5–30` rule on a `bmi` outcome gave −0.092 (−0.104, −0.080),
# a tight interval around a truncated answer (docs/turbotab-next/audit, RO-01). Under prediction
# the rule could not be applied to anyone whose outcome is still unknown. Refused under every
# purpose; impossible outcome values have their own repair, which the exits offer.

OUTCOME_RULE = ("Keeping rows by their outcome selects on the value being explained, which biases "
                "the estimates, and no one whose outcome is still unknown could be screened by it.")


def _on_the_outcome(rule: Any, target: str) -> bool:
    """The rule reads the outcome: as its column, as the column its ranges are set by, or (a
    Goldberg screen) as any input of the screen, such as the body weight its BMR reads."""
    return target in as_rule(rule).reads()


def _records_in(ctx: Any) -> list["DecisionRecord"] | None:
    found = _ctx(ctx, "records")
    if callable(found):
        try:
            found = found()
        except Exception:  # noqa: BLE001 - no log: nothing is checked against it
            return None
    return list(found) if found is not None else None


def state_after(decision: Any, ctx: Any) -> "ProjectState | None":
    """The state recording ``decision`` would leave: the log ``ctx`` names, folded with it.

    None when ``ctx`` names no log, or the log would refuse the decision itself (an unknown
    revert): what cannot be folded is not checked here.
    """
    records = _records_in(ctx)
    if records is None:
        return None
    probe = DecisionRecord(id="__probe__", seq=max((r.seq for r in records), default=0) + 1,
                           at=datetime.now(timezone.utc), decision=decision)
    try:
        return fold([*records, probe])
    except Refusal:
        return None


def _outcome_repair_exits(target: str, ctx: Any) -> list[dict[str, Any]]:
    """The repairs a finding offers for impossible or coded values of the outcome, as exits: only
    those that would be accepted now, and none for a finding already repaired."""
    from turbotab.core import repairs

    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return []
    try:
        findings = artifact("findings")
    except Exception:  # noqa: BLE001 - no findings: no repair to offer
        return []
    state = _state(ctx)
    done = {fid for fid, d in ((state.findings or {}) if state is not None else {}).items()
            if d.action == "applied"}
    exits: list[dict[str, Any]] = []
    for finding in repairs.findings_of(findings):
        if finding.get("id") in done:
            continue
        for option in finding.get("repairs") or []:
            decision = option.get("decision") or {}
            touched = repairs.option_columns(str(finding["id"]), str(option.get("key")),
                                             decision.get("params") or {})
            if target not in touched:
                continue
            try:
                validate(decision, ctx)
            except Refusal:
                continue
            exits.append({"label": f"Impossible or coded `{target}` values: "
                                   f"{str(option.get('label') or option.get('key')).lower()}",
                          "decision": decision})
    return exits


def _exclusions_leave_the_outcome_alone(decision: SetExclusions, ctx: Any) -> None:
    target = _target_of(ctx)
    if target is _UNKNOWN or target is None:
        return
    on = [i for i, rule in enumerate(decision.rules) if _on_the_outcome(rule, target)]
    if not on:
        return
    kept = SetExclusions(rules=[r for i, r in enumerate(decision.rules) if i not in on])
    raise Refusal(
        "rule_on_outcome",
        f"`{target}` is the outcome. {OUTCOME_RULE} Say who is studied by what was known before "
        f"the outcome.",
        exits=[{"label": f"Drop the rule on `{target}`", "decision": kept},
               *_outcome_repair_exits(target, ctx),
               {"label": "Restrict by a variable measured before the outcome", "decision": None}],
    )


def _new_outcome_has_no_rule(decision: SetTarget, ctx: Any) -> None:
    """A rule already on the column chosen as the outcome would become a rule on the outcome."""
    state = _state(ctx)
    rules = list(state.exclusions or []) if state is not None else []
    on = [r for r in rules if _on_the_outcome(r, decision.column)]
    if not on or decision.column == getattr(state, "target", None):
        return
    raise Refusal(
        "rule_on_outcome",
        f"An eligibility rule reads `{decision.column}`, so making it the outcome would keep rows by "
        f"their outcome. {OUTCOME_RULE}",
        exits=[{"label": f"Drop the rule on `{decision.column}` first",
                "decision": SetExclusions(rules=[r for r in rules if r not in on])}],
    )


def _revert_leaves_no_rule_on_the_outcome(decision: "Revert", ctx: Any) -> None:
    """A revert that would bring back an outcome with a rule on it, or a rule on the outcome."""
    after = state_after(decision, ctx)
    if after is None or after.target is None:
        return
    if not any(_on_the_outcome(r, after.target) for r in after.exclusions or []):
        return
    now = _state(ctx)
    if now is not None and now.target == after.target and \
            any(_on_the_outcome(r, now.target) for r in now.exclusions or []):
        return  # already so before this revert: not this revert's doing
    raise Refusal(
        "rule_on_outcome",
        f"Undoing that decision would leave an eligibility rule on the outcome `{after.target}`. "
        f"{OUTCOME_RULE}",
        exits=[{"label": "Keep the answers as they are", "decision": None}],
    )


def _energy_adjustment_fits_the_roles(decision: SetEnergyAdjustment, ctx: Any) -> None:
    if decision.log_transform and decision.method not in ("residual", "residual_energy_dropped"):
        # The log variant is a residual method's (its own estimand, audit WP6); any other method
        # would fail at the design, after the answer was recorded.
        raise Refusal(
            "log_needs_residual",
            "Logging nutrient and energy applies to the residual methods only.",
            exits=[{"label": "Without the log",
                    "decision": SetEnergyAdjustment(**{**decision.model_dump(exclude={"kind"}),
                                                       "log_transform": False})}],
        )
    if decision.method == "none":
        return
    state = _state(ctx)
    columns = _columns_of(ctx)
    if state is None:
        return
    roles = state.roles or {}
    base = decision.model_dump(exclude={"kind"})

    def with_(**changes: Any) -> SetEnergyAdjustment:
        return SetEnergyAdjustment(**{**base, **changes})

    if not roles:
        raise Refusal(
            "roles_first",
            "Energy adjustment works on the column roles; confirm them first.",
            exits=[{"label": "Confirm the column roles", "decision": None},
                   {"label": "Do not adjust for energy", "decision": with_(method="none")}],
        )
    energy_columns = [c for c, r in roles.items() if r == "energy"]
    if decision.energy_column is None or roles.get(decision.energy_column) != "energy":
        exits = [{"label": f"Adjust against `{c}`", "decision": with_(energy_column=c)} for c in energy_columns]
        named = f"`{decision.energy_column}` does not" if decision.energy_column else "No column was named that"
        raise Refusal(
            "energy_role",
            f"{named} has the energy role, and adjustment is computed against total energy.",
            exits=exits + [{"label": "Give a column the energy role", "decision": None}],
        )
    exposures = [n for n in decision.nutrients if roles.get(n) == "exposure"]
    if not decision.nutrients or len(exposures) != len(decision.nutrients):
        others = [n for n in decision.nutrients if n not in exposures]
        message = ("Name the nutrients to adjust." if not decision.nutrients else
                   f"{_and(others)} {'is' if len(others) == 1 else 'are'} not an exposure; only exposures are adjusted.")
        exits = [{"label": "Adjust only the exposures", "decision": with_(nutrients=exposures)}] if exposures else []
        raise Refusal("nutrients_not_exposures", message,
                      exits=exits + [{"label": "Choose the nutrients to adjust", "decision": None}])
    gone = [c for c in (decision.energy_column, *decision.nutrients) if c in left_out(state)]
    if gone:
        kept = [n for n in decision.nutrients if n not in gone]
        exits = ([{"label": "Adjust only the nutrients still in the model", "decision": with_(nutrients=kept)}]
                 if kept and decision.energy_column not in gone else [])
        raise Refusal(
            "left_out",
            f"{_and(gone)} {'was' if len(gone) == 1 else 'were'} left out of the predictors with the "
            f"missing values, so energy adjustment cannot use {'it' if len(gone) == 1 else 'them'}.",
            exits=exits + [{"label": "Do not adjust for energy", "decision": with_(method="none")}],
        )
    from turbotab.core.methods.energy import METHOD_TABLE, applicable_methods

    table = applicable_methods(sorted(columns or roles), decision.energy_column, decision.nutrients)
    verdict = table.get(decision.method, {"ok": True})
    if not verdict["ok"]:
        exits = [{"label": METHOD_TABLE[m]["label"], "decision": with_(method=m)}
                 for m, v in table.items() if v["ok"] and m != decision.method]
        raise Refusal("method_not_applicable", str(verdict["reason"]), exits=exits)
    if decision.method in ("partition", "all_components"):
        _partition_runs_on_the_data(decision, ctx, with_, table)
    if decision.strata is not None:
        info = _info(ctx, decision.strata)
        known = columns is None or decision.strata in columns
        levels = int(info.get("n_unique") or 0) if info else None
        categorical = info is None or info.get("dtype") in ("categorical", "boolean", "integer")
        if not known or not categorical or (levels is not None and levels > MAX_STRATA_LEVELS):
            raise Refusal(
                "strata_not_categorical",
                f"Strata need a category with at most {MAX_STRATA_LEVELS} levels; "
                f"`{decision.strata}` is not one.",
                exits=[{"label": "Adjust without strata", "decision": with_(strata=None)}],
            )


PARTITION_CHECK_ROWS = 5_000


def _partition_runs_on_the_data(decision: SetEnergyAdjustment, ctx: Any,
                                with_: Callable[..., SetEnergyAdjustment],
                                table: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a partition the fit would refuse, or one that counts a total and its parts twice.

    The same checks the fit makes (energy.partition_refusal), on a sample of the rows outside the
    held-out set: a unit question, asked of the data before the answer is recorded, so the design
    never fails on it afterwards.
    """
    import numpy as np

    from turbotab.core.methods.energy import METHOD_TABLE, partition_refusal
    from turbotab.core.methods.nesting import nested_components

    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data to check: the fit decides
        store = None
    E, nutrients = decision.energy_column, list(decision.nutrients)
    if store is None or E is None or not {E, *nutrients} <= set(store.columns):
        return
    pool = np.arange(int(store.n_rows), dtype=np.int64)
    state = _ctx(ctx, "state")
    if getattr(state, "purpose", None) == "inference":
        # BLUEPRINT §12 ruling 3: under inference no row is sealed from the estimate, so the
        # unit check reads the rows the table will (the omitted-energy check does the same).
        analyzed_of = _ctx(ctx, "analyzed")
        try:
            analyzed = analyzed_of() if callable(analyzed_of) else None
        except Exception:  # noqa: BLE001 - not known yet: every row
            analyzed = None
        if analyzed is not None and len(analyzed):
            pool = np.asarray(analyzed, dtype=np.int64)
    else:
        sealed_of = _ctx(ctx, "sealed")
        sealed = sealed_of() if callable(sealed_of) else None
        if sealed is not None and len(sealed):
            pool = np.setdiff1d(pool, np.asarray(sealed, dtype=np.int64), assume_unique=True)
    if len(pool) > PARTITION_CHECK_ROWS:
        pool = np.sort(np.random.default_rng(0).choice(pool, size=PARTITION_CHECK_ROWS, replace=False))
    frame = store.materialize([E, *nutrients], pool)
    nested = nesting_of(ctx) or nested_components(frame, nutrients)
    refused = partition_refusal(frame, E, nutrients, nested=nested)
    if refused is None:
        return
    exits: list[dict[str, Any]] = []
    if refused["nutrients"]:
        exits.append({"label": f"Partition {_and(refused['nutrients'])}",
                      "decision": with_(nutrients=refused["nutrients"])})
    exits += [{"label": METHOD_TABLE[m]["label"], "decision": with_(method=m)}
              for m in ("residual", "standard") if table.get(m, {}).get("ok")]
    reason = str(refused["reason"])
    if decision.method == "all_components":
        reason = reason.replace("a partition would", "the all-components model would", 1)
    raise Refusal("method_not_applicable", _ticked(reason, [E, *nutrients]), exits=exits)


def _ticked(text: str, columns: Sequence[str]) -> str:
    """Wrap these column names in backticks (data chips), longest first, whole words only."""
    import re

    for c in sorted(set(columns), key=len, reverse=True):
        text = re.sub(rf"(?<![`\w]){re.escape(c)}(?![`\w])", f"`{c}`", text)
    return text


def model_families() -> dict[str, set[str]]:
    """Model family key -> the tasks it can model, from ``turbotab.core.models`` when present.

    The registry is the modeling agent's (M1_CONTRACT.md §7); until it is importable the M1
    families stand in, so a validator never refuses a family that exists.
    """
    fallback = {key: {"regression", "binary", "multiclass"}
                for key in ("linear", "elastic_net", "boosted_trees")}
    try:
        import importlib

        module = importlib.import_module("turbotab.core.models")
    except Exception:
        return fallback
    found: Any = None
    for name in ("FAMILIES", "REGISTRY", "families", "all_families", "registry"):
        value = getattr(module, name, None)
        if value is None:
            continue
        found = value() if callable(value) and not isinstance(value, Mapping) else value
        break
    if found is None:
        return fallback
    items = found.values() if isinstance(found, Mapping) else found
    out: dict[str, set[str]] = {}
    for family in items:
        key = _ctx(family, "key")
        if key:
            out[str(key)] = {str(t) for t in (_ctx(family, "tasks") or ())}
    return out or fallback


def _models_can_fit_the_task(decision: SelectModels, ctx: Any) -> None:
    families = model_families()
    unknown = [m for m in decision.models if m not in families]
    known = [m for m in decision.models if m in families]
    if unknown:
        raise Refusal(
            "unknown_model",
            f"There is no model family called {_and(unknown)}.",
            exits=([{"label": "Keep the families that exist", "decision": SelectModels(models=known)}]
                   if known else []) + [{"label": "Choose from the shelf", "decision": None}],
        )
    task = _ctx(ctx, "task")
    if task is None:
        return
    unable = [m for m in decision.models if families[m] and task not in families[m]]
    if unable:
        able = [m for m in decision.models if m not in unable]
        raise Refusal(
            "model_cannot_fit_task",
            f"{_and(unable)} cannot model a {str(task).replace('_', '-')} outcome.",
            exits=([{"label": "Keep the families that can", "decision": SelectModels(models=able)}]
                   if able else []) + [{"label": "Choose from the shelf", "decision": None}],
        )


def _substitution_swaps_energy(decision: SetSubstitution, ctx: Any) -> None:
    if decision.donor == decision.recipient:
        raise Refusal(
            "same_nutrient",
            f"A substitution swaps one nutrient for another; `{decision.donor}` cannot replace itself.",
            exits=[{"label": "Choose a different recipient", "decision": None}],
        )
    state = _state(ctx)
    if state is None:
        return
    from turbotab.core.methods.percent_energy import is_percent_of_energy
    from turbotab.core.stages.rows import energy_bearing

    roles = state.roles or {}
    # A share of energy (``fat_pct_kcal``) moves in percentage points of energy (audit B24).
    percent = [c for c in (decision.donor, decision.recipient)
               if roles.get(c) == "exposure" and is_percent_of_energy(c)]
    if percent and decision.scale == "kcal":
        base = decision.model_dump(exclude={"kind"})
        raise Refusal(
            "percent_of_energy",
            f"{_and(percent)} {'is' if len(percent) == 1 else 'are'} already a share of energy, so "
            f"energy moves through {'it' if len(percent) == 1 else 'them'} in percentage points of "
            f"energy, not in kcal.",
            exits=[{"label": "Move 5% of energy at a time",
                    "decision": SetSubstitution(**{**base, "scale": "percent_energy"})}])
    if decision.scale == "percent_energy":
        amounts = [c for c in (decision.donor, decision.recipient) if c not in percent]
        if amounts and not any(r == "energy" for r in roles.values()):
            raise Refusal(
                "no_total_energy",
                f"Moving a share of energy through {_and(amounts)} needs each row's total energy; "
                f"no column has the energy role.",
                exits=[{"label": "Give the total-energy column the energy role", "decision": None}])
    candidates = [c for c, r in roles.items() if r == "exposure"
                  and (energy_bearing(c) or (decision.scale == "percent_energy"
                                             and is_percent_of_energy(c)))]
    bad = [c for c in (decision.donor, decision.recipient) if c not in candidates]
    if bad:
        base = decision.model_dump(exclude={"kind"})
        exits = []
        for other in candidates:
            if other not in (decision.donor, decision.recipient) and len(exits) < 3:
                fixed = {**base, **({"donor": other} if decision.donor in bad else {"recipient": other})}
                if fixed["donor"] != fixed["recipient"] and fixed["donor"] in candidates \
                        and fixed["recipient"] in candidates:
                    exits.append({"label": f"Swap `{fixed['donor']}` for `{fixed['recipient']}`",
                                  "decision": SetSubstitution(**fixed)})
        raise Refusal(
            "not_energy_bearing",
            f"{_and(bad)} {'is' if len(bad) == 1 else 'are'} not an energy-bearing nutrient in the "
            f"model; a substitution moves kcal between two of them.",
            exits=exits + [{"label": "Choose two energy-bearing nutrients", "decision": None}],
        )


def nesting_of(ctx: Any) -> dict[str, str]:
    """Child column -> the column it is part of, as the design (else the roles stage) found it."""
    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return {}
    for stage in ("design", "roles"):
        try:
            data = artifact(stage)
        except Exception:  # noqa: BLE001 - a missing artifact checks nothing
            data = None
        if not isinstance(data, Mapping):
            continue
        if stage == "design" and data.get("nested") is not None:
            return {str(e["column"]): str(e["parent"]) for e in data["nested"]}
        if stage == "roles":
            return {str(e["column"]): str(e["nested_in"]) for e in data.get("columns") or []
                    if e.get("nested_in")}
    return {}


def _substitution_moves_between_separate_nutrients(decision: SetSubstitution, ctx: Any) -> None:
    nested = nesting_of(ctx)
    d, r = decision.donor, decision.recipient
    if nested.get(d) == r or nested.get(r) == d:
        child, parent = (d, r) if nested.get(d) == r else (r, d)
        raise Refusal(
            "part_of_the_other",
            f"`{child}` is part of `{parent}`, so moving kcal between them moves nothing; a "
            f"substitution swaps two separate nutrients.",
            exits=[{"label": "Choose two separate nutrients", "decision": None}],
        )


OMITTED_CHECK_ROWS = 5_000


def _substitution_has_every_energy_source(decision: SetSubstitution, ctx: Any) -> None:
    """Under inference, block and record a swap whose model leaves a large share of total energy
    to the composite of the sources not in it (audit ME-05).

    A substitution curve with only some energy sources in the model is confounded through the
    implicit "other" composite total energy carries (Tomova, Gilthorpe & Tennant 2022). Above
    :data:`~turbotab.core.methods.energy.MAX_OMITTED_SHARE` of total energy on average, on every
    analyzed row (BLUEPRINT §12 ruling 3: under inference no row is sealed from the estimate), the
    swap is refused with its exits: add the missing sources to the model, or keep it with the
    recorded attestation that the curve carries their confounding. Under prediction the curve is a
    model contrast and the substitution artifact states the concern instead.

    Shares of energy (``fat_pct_kcal``) sum to 100%, so they cannot all be in the model beside its
    intercept: the field's model in percent of energy leaves exactly one source out as the
    reference, and each coefficient is a point of energy from its source in place of that one (Hu
    et al. 1997; NUTRITION_PACK §05, "all components except one, plus total energy"). So, for
    shares (the methods gate, item C):

    * every share in the model, summing to 100% on every row, leaves no reference (a rank-deficient
      model): refused, with one exit per source that could be left out as the reference;
    * one named source left out, with nothing beyond it (the shares with it sum to within
      ``MAX_OMITTED_SHARE`` of 100%), is that leave-one-out model: accepted, whatever the reference's
      own share;
    * two or more left out: the exits add the rest *but one*, one exit per choice of reference,
      never every share.
    """
    import numpy as np

    from turbotab.core.methods.energy import MAX_OMITTED_SHARE, energy_factor, nutrient_role, omitted_energy
    from turbotab.core.methods.nesting import compositions, nested_components
    from turbotab.core.methods.percent_energy import is_percent_of_energy

    state = _state(ctx)
    if state is None or state.purpose != "inference" or decision.acknowledged:
        return
    roles = state.roles or {}
    adj = state.energy_adjustment
    E = (adj.energy_column if adj is not None and adj.energy_column else
         next((c for c, r in roles.items() if r == "energy"), None))
    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data to check: the curve states the concern
        store = None
    gone = set(left_out(state))
    exposures = [c for c, r in roles.items() if r == "exposure" and c not in gone]
    if store is None or E is None or E not in store.columns:
        return
    # Every analyzed row when the project knows them (the split's rows on both sides of the seal),
    # else every row: under inference the estimate reads held-out rows too (ruling 3).
    analyzed_of = _ctx(ctx, "analyzed")
    try:
        analyzed = analyzed_of() if callable(analyzed_of) else None
    except Exception:  # noqa: BLE001 - not known yet: every row
        analyzed = None
    pool = (np.asarray(analyzed, dtype=np.int64) if analyzed is not None and len(analyzed)
            else np.arange(int(store.n_rows), dtype=np.int64))
    if len(pool) > OMITTED_CHECK_ROWS:
        pool = np.sort(np.random.default_rng(0).choice(pool, size=OMITTED_CHECK_ROWS, replace=False))

    def source(column: str) -> str | None:
        try:
            return nutrient_role(column)
        except ValueError:
            return None

    target = getattr(state, "target", None)
    present = [c for c in exposures if c in store.columns]
    # The table's other share-of-energy columns, by source: the candidates for a reference.
    spare_shares = [c for c in store.columns if c not in exposures and c not in (E, target)
                    and roles.get(c) != "energy" and is_percent_of_energy(c)]
    frame = store.materialize(list(dict.fromkeys([E, *present, *spare_shares])), pool)
    base = decision.model_dump(exclude={"kind"})
    another = {"label": "Choose another swap", "decision": None}
    moved = (decision.donor, decision.recipient)

    def by_share(columns: Sequence[str]) -> list[str]:
        """Largest mean share first: the roles stage's reference (``nesting.reference_share``)."""
        means = {c: float(np.nanmean(frame[c].to_numpy(dtype=float))) for c in columns}
        return sorted(columns, key=lambda c: (-means[c], c))

    shares_in = [c for c in present if is_percent_of_energy(c)]
    composed = compositions(frame, shares_in) if len(shares_in) >= 2 else []
    if composed:
        candidates = by_share([c for c in composed if c not in moved])
        raise Refusal(
            "no_reference_share",
            f"{_and(composed)} sum to 100% of energy on every row, so beside the intercept each is "
            f"fixed by the others and the model cannot estimate them all. The field's model leaves "
            f"one source out as the reference; each coefficient is then a point of energy from its "
            f"source in place of that one (Hu et al. 1997).",
            exits=[{"label": f"Leave `{c}` out as the reference",
                    "decision": SetRoles(roles={**roles, c: "excluded"})} for c in candidates]
            + [another])

    nested = nesting_of(ctx) or nested_components(frame[present], present)
    reading = omitted_energy(frame[[E, *present]], E, present, nested=nested)
    share = None if reading is None else reading["mean_share"]
    if share is None or share <= MAX_OMITTED_SHARE:
        return

    # A model holding its sources as shares of energy adds a missing one as a share
    # (``alcohol_pct_kcal``), a model holding amounts adds an amount (audit B24).
    in_shares = any(is_percent_of_energy(c) for c in reading["columns"])
    in_amounts = any(not is_percent_of_energy(c) for c in reading["columns"]) or not in_shares
    missing = [s for s in reading["omitted"] if s != "other"]
    if in_shares and not in_amounts and len(missing) == 1:
        # The leave-one-out model: the one source left out is the reference, when the shares with
        # it leave nothing beyond (its column measures that; without one, the remainder stands).
        reference = [c for c in spare_shares if source(c) == missing[0]]
        if reference:
            held = np.zeros(len(frame))
            for c in [*reading["columns"], reference[0]]:
                held = held + frame[c].to_numpy(dtype=float) / 100.0
            beyond = 1.0 - held[np.isfinite(held)]
            if beyond.size and float(beyond.mean()) <= MAX_OMITTED_SHARE:
                return

    def carries(column: str) -> bool:
        if is_percent_of_energy(column):
            return in_shares
        return in_amounts and energy_factor(column).factor is not None

    addable = [c for c in store.columns if c not in exposures and c not in (E, target)
               and roles.get(c) != "energy" and source(c) in reading["omitted"] and carries(c)]
    held = _and(reading["columns"]) if reading["columns"] else "no energy source"
    named = ", ".join([*missing, "other energy"]) if missing else "other energy"
    exits: list[dict[str, Any]] = []
    add_shares = [c for c in addable if is_percent_of_energy(c)]
    if add_shares and len(add_shares) == len(addable) \
            and compositions(frame, [*shares_in, *add_shares]):
        # Every share added would sum to 100%: leave one of them out as the reference.
        for reference in by_share(add_shares):
            rest = [c for c in add_shares if c != reference]
            exits.append({
                "label": (f"Add {_and(rest)} to the model, with `{reference}` left out as the "
                          f"reference"),
                "decision": SetRoles(roles={**roles, **{c: "exposure" for c in rest},
                                            reference: "excluded"})})
    elif addable:
        exits.append({"label": f"Add {_and(addable)} to the model",
                      "decision": SetRoles(roles={**roles, **{c: "exposure" for c in addable}})})
    exits += [
        {"label": "Keep this swap; the curve carries their confounding",
         "decision": SetSubstitution(**{**base, "acknowledged": True})},
        another,
    ]
    raise Refusal(
        "omitted_energy_sources",
        f"Under inference this curve is read as an effect, but the model holds {held}: "
        f"{named} make up {share:.0%} of total energy on average, more than the "
        f"{MAX_OMITTED_SHARE:.0%} the Atwater factors' own error explains. Total energy carries "
        f"them as one composite, so the curve carries their confounding.",
        exits=exits,
    )


def _left_out_columns_are_predictors(decision: SetMissing, ctx: Any) -> None:
    if not decision.drop_columns:
        return
    columns = _columns_of(ctx)
    keep = {"label": "Keep every predictor", "decision": SetMissing(strategy=decision.strategy)}
    if columns is not None:
        unknown = [c for c in decision.drop_columns if c not in columns or c == ROW_ID]
        if unknown:
            raise Refusal("unknown_column", f"This dataset has no column named {_and(unknown)}.",
                          exits=[keep])
    state = _state(ctx)
    roles = (state.roles if state is not None else None) or {}
    if not roles:
        return
    preds = [c for c, r in roles.items() if r in PREDICTOR_ROLES]
    outside = [c for c in decision.drop_columns if c not in preds]
    if outside:
        rest = [c for c in decision.drop_columns if c not in outside]
        exits = ([{"label": f"Leave out only {_and(rest)}",
                   "decision": SetMissing(strategy=decision.strategy, drop_columns=rest)}]
                 if rest else []) + [keep]
        raise Refusal(
            "not_a_predictor",
            f"{_and(outside)} {'is' if len(outside) == 1 else 'are'} not a predictor, so there is "
            f"nothing to leave out.",
            exits=exits,
        )
    if not [c for c in preds if c not in decision.drop_columns]:
        raise Refusal(
            "no_predictors",
            "Leaving out every predictor would leave the models nothing to use.",
            exits=[keep],
        )


def _missing_base(decision: SetMissing, **change: Any) -> SetMissing:
    """``decision`` with ``change``: an exit that keeps the answer's other parts."""
    return SetMissing(**{**decision.model_dump(exclude={"kind"}), **change})


def _missing_fits_the_purpose(decision: SetMissing, ctx: Any) -> None:
    """Missing data by purpose (BLUEPRINT §12 ruling 4; AUDIT_REPORT §5 WP7, ME-01).

    Multiple imputation puts the outcome in the imputation model, so under prediction it is
    refused: a held-out row's own outcome would fill its predictors, and a new row has none. Under
    inference a single fill and the missing-indicator method (a missing indicator, or blanks as a
    level) are blocked and recorded: refused with their exits, multiple imputation first, and kept
    only with the attestation (``acknowledged``) the methods sentence carries.
    """
    from turbotab.core.methods.missing import INDICATOR_CAUTION, SINGLE_FILL_CAUTION

    state = _state(ctx)
    purpose = getattr(state, "purpose", None)
    keep = {"drop_columns": list(decision.drop_columns)}
    if decision.strategy == "multiple_imputation":
        if purpose != "inference":
            raise Refusal(
                "imputation_with_the_outcome",
                "Multiple imputation puts the outcome in the imputation model. That is right for "
                "inference, but a prediction model must impute a new row without its outcome, so "
                "under prediction the imputation is fit in each training fold without it (Sisk et "
                "al. 2023).",
                exits=[{"label": "Fill in each training fold, without the outcome",
                        "decision": SetMissing(strategy="impute", **keep)},
                       {"label": "Complete cases", "decision": SetMissing(strategy="complete_case", **keep)},
                       {"label": "Change the purpose to inference", "decision": None}])
        if decision.indicators:
            raise Refusal(
                "indicators_with_imputation",
                "Multiple imputation fills every blank from the other variables; a missing "
                "indicator beside it is the other method, not an addition to it.",
                exits=[{"label": "Multiple imputation without indicators",
                        "decision": _missing_base(decision, indicators=False)}])
    if purpose != "inference" or decision.acknowledged:
        return
    single = decision.strategy == "impute"
    indicator = decision.indicators or decision.categorical == "missing_category"
    if not (single or indicator):
        return
    exits: list[dict[str, Any]] = [
        {"label": "Multiple imputation with the outcome and energy (m = 20)",
         "decision": _missing_base(decision, strategy="multiple_imputation", indicators=False,
                                   categorical="impute", acknowledged=False)},
        {"label": "Complete cases, with their assumption stated",
         "decision": _missing_base(decision, strategy="complete_case", indicators=False,
                                   categorical="impute", acknowledged=False)},
    ]
    if indicator:
        what = ("blanks as their own level" if decision.categorical == "missing_category"
                and not decision.indicators else "missing indicators")
        exits.append({"label": "Keep it, recorded: a blank here means not asked, or the "
                               "covariates are baseline ones in a randomized trial",
                      "decision": _missing_base(decision, acknowledged=True)})
        raise Refusal(
            "indicator_under_inference",
            f"Under inference {what} are the missing-indicator method: {INDICATOR_CAUTION}. "
            f"Multiple imputation with the outcome keeps every row without that bias.",
            exits=exits)
    exits.append({"label": "Keep the single fill, recorded: intervals too narrow, estimates "
                           "possibly biased",
                  "decision": _missing_base(decision, acknowledged=True)})
    raise Refusal(
        "single_fill_under_inference",
        f"Under inference {SINGLE_FILL_CAUTION}. With a confounder 40% missing at random, a median "
        f"fill's 95% intervals never covered the truth (audit ME-01). Multiple imputation with the "
        f"outcome and energy in the imputation model, pooled by Rubin's rules, is the sound "
        f"answer.",
        exits=exits)


def _censored_named(ctx: Any) -> list[str]:
    """Left-censored columns the findings (and the zeros-as-non-detections repair) name."""
    from turbotab.core.methods.missing import censored_columns

    artifact = _ctx(ctx, "artifact")
    findings = None
    if callable(artifact):
        try:
            findings = artifact("findings")
        except Exception:  # noqa: BLE001 - no findings yet: nothing is known to be censored
            findings = None
    return censored_columns(findings, _state(ctx))


def _non_detections_are_not_filled_by_the_median(decision: SetMissing, ctx: Any) -> None:
    """Values below a detection limit are small, not unknown (audit ME-08): when the left-censoring
    finding names predictors (or the user recoded zeros as non-detections), a fill that treats
    their blanks as any other blank is refused unless the answer gives a reason. Half the minimum
    (customary) and a censoring-aware fill are offered. Complete cases keep only the rows with every
    value detected, and say what they lose under inference."""
    columns = _columns_of(ctx)
    named = list(decision.censored_columns)
    if columns is not None:
        unknown = [c for c in named if c not in columns or c == ROW_ID]
        if unknown:
            raise Refusal("unknown_column", f"This dataset has no column named {_and(unknown)}.",
                          exits=[{"label": "Name no censored column",
                                  "decision": _missing_base(decision, censored_columns=[])}])
    state = _state(ctx)
    roles = (getattr(state, "roles", None) or {}) if state is not None else {}
    gone = set(decision.drop_columns)
    found = [c for c in _censored_named(ctx)
             if (not roles or roles.get(c) in PREDICTOR_ROLES) and c not in gone]
    censored = list(dict.fromkeys([*named, *found]))
    if decision.below_detection in ("half_minimum", "censoring_aware") and not named:
        if not censored:
            raise Refusal(
                "no_censored_columns",
                "No column is known to hold values below a detection limit: name the columns "
                "whose blanks are non-detections.",
                exits=[{"label": "Treat blanks as any other blank",
                        "decision": _missing_base(decision, below_detection=None)}])
        raise Refusal(
            "censored_columns_unnamed",
            f"Name the columns whose blanks are non-detections; the findings name "
            f"{_and(censored[:6])}{' and more' if len(censored) > 6 else ''}.",
            exits=[{"label": "Those columns", "decision": _missing_base(decision, censored_columns=censored)}])
    if not censored or decision.strategy == "complete_case" or decision.reason:
        return
    if decision.below_detection in ("half_minimum", "censoring_aware"):
        return
    inference = getattr(state, "purpose", None) == "inference"
    how = ("multiple imputation reads them as missing at random" if decision.strategy ==
           "multiple_imputation" else "the median fill places them in the middle of the distribution")
    aware = ("a censored-normal draw below the limit, given the outcome" if inference else
             "the expected value below the limit, fit in each training fold")
    raise Refusal(
        "median_below_detection",
        f"{_and(censored[:6])}{' and others' if len(censored) > 6 else ''} hold values below a "
        f"detection limit: each blank is known to be small, but {how}. Choose how non-detections "
        f"are filled, or keep this fill with your reason.",
        exits=[{"label": f"Censoring-aware: {aware}",
                "decision": _missing_base(decision, below_detection="censoring_aware",
                                          censored_columns=censored)},
               {"label": "Half the smallest detected value (customary)",
                "decision": _missing_base(decision, below_detection="half_minimum",
                                          censored_columns=censored)},
               {"label": "Keep this fill: give your reason", "decision": None}])


def _raw_columns(ctx: Any) -> set[str] | None:
    """The file's own columns (the ingest's), which a features-in-rows table is declared over."""
    fn = _ctx(ctx, "artifact")
    if callable(fn):
        try:
            found = fn("ingest")
        except Exception:  # noqa: BLE001 - no artifact: the context's columns stand in
            found = None
        found = getattr(found, "data", found)
        if isinstance(found, Mapping) and found.get("columns") is not None:
            return {str(c["name"]) for c in found["columns"]}
    return _columns_of(ctx)


def _feature_table_names_the_files_columns(decision: SetFeatureTable, ctx: Any) -> None:
    columns = _raw_columns(ctx)
    named = [c for c in [decision.label, *decision.annotations] if c]
    if columns is not None:
        unknown = [c for c in named if c not in columns or c == ROW_ID]
        if unknown:
            raise Refusal("unknown_column", f"This file has no column named {_and(unknown)}.",
                          exits=[{"label": "Name the file's own columns", "decision": None}])
    if decision.label is not None and decision.label in decision.annotations:
        rest = [c for c in decision.annotations if c != decision.label]
        raise Refusal(
            "label_is_annotation",
            f"`{decision.label}` names the features, so it is not also one of their annotations.",
            exits=[{"label": f"`{decision.label}` names the features",
                    "decision": SetFeatureTable(label=decision.label, annotations=rest)}])


def _categorical_names_predictors(decision: SetCategorical, ctx: Any) -> None:
    columns = _columns_of(ctx)
    if columns is not None:
        unknown = [c for c in decision.columns if c not in columns or c == ROW_ID]
        if unknown:
            rest = [c for c in decision.columns if c not in unknown]
            raise Refusal(
                "unknown_column", f"This dataset has no column named {_and(unknown)}.",
                exits=[{"label": "Keep the real columns", "decision": SetCategorical(columns=rest)}])
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is not None and target in decision.columns:
        rest = [c for c in decision.columns if c != target]
        raise Refusal(
            "target_categorical",
            f"`{target}` is the outcome; whether it is a category is the task question's answer.",
            exits=[{"label": "Leave the outcome out", "decision": SetCategorical(columns=rest)}])


def _outcome_values(ctx: Any, column: str) -> list[Any] | None:
    """The distinct recorded values of ``column``, read from the store ``ctx`` opens; None when it
    opens none."""
    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data to check: the fit decides
        store = None
    if store is None or column not in set(store.columns):
        return None
    return list(store.materialize([column])[column].dropna().unique())


def _order_names_the_outcome(decision: SetOutcomeOrder, ctx: Any) -> None:
    """The order answers for the outcome and places each of its levels exactly once."""
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is None:
        raise Refusal("no_target", "Choose the outcome first; the order describes its levels.",
                      exits=[{"label": "Choose the outcome", "decision": None}])
    if target is not _UNKNOWN and decision.column != target:
        raise Refusal(
            "not_the_target",
            f"The outcome is `{target}`, not `{decision.column}`; the order answers for the outcome.",
            exits=[{"label": f"Order the levels of `{target}`", "decision": None}])
    from turbotab.core.stages.rows import _level_key

    values = _outcome_values(ctx, decision.column)
    if values is None:
        info = _info(ctx, decision.column)
        known = info.get("n_unique") if info is not None else None
        if known is not None and int(known) != len(decision.levels):
            raise Refusal(
                "levels_mismatch",
                f"`{decision.column}` has {int(known):,} distinct values, but "
                f"{len(decision.levels):,} levels were placed in order.",
                exits=[{"label": "Place every level, each once", "decision": None}])
        return
    have = {_level_key(v) for v in values}
    named = {_level_key(v) for v in decision.levels}
    missing = sorted(str(v) for v in values if _level_key(v) not in named)
    extra = [lv for lv in decision.levels if _level_key(lv) not in have]
    if missing or extra:
        said = []
        if missing:
            said.append(f"{_and(missing)} {'is' if len(missing) == 1 else 'are'} not placed")
        if extra:
            said.append(f"{_and(extra)} {'is' if len(extra) == 1 else 'are'} not a level of it")
        raise Refusal(
            "levels_mismatch",
            f"The order must place every level of `{decision.column}` once: {'; '.join(said)}.",
            exits=[{"label": "Place every level, each once", "decision": None}])


def _form_fits_the_column(decision: SetExposureForm, ctx: Any) -> None:
    """A spline or quintiles need a numeric predictor with enough distinct values; knots belong
    to a spline only."""
    linear = {"label": f"Keep `{decision.column}` a straight line",
              "decision": SetExposureForm(column=decision.column, form="linear")}
    if decision.knots is not None and decision.form != "spline":
        raise Refusal(
            "knots_without_spline",
            f"Knots belong to a spline; the {decision.form} form has none.",
            exits=[{"label": f"{decision.form.capitalize()} without knots",
                    "decision": SetExposureForm(column=decision.column, form=decision.form)}])
    columns = _columns_of(ctx)
    if columns is not None and (decision.column not in columns or decision.column == ROW_ID):
        raise Refusal("unknown_column", f"This dataset has no column named `{decision.column}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is not None and decision.column == target:
        raise Refusal("form_of_outcome",
                      f"`{decision.column}` is the outcome; a form shapes a predictor.",
                      exits=[{"label": "Choose a predictor", "decision": None}])
    if decision.form == "linear":
        return
    state = _state(ctx)
    roles = (state.roles if state is not None else None) or {}
    if roles and roles.get(decision.column) not in PREDICTOR_ROLES:
        raise Refusal(
            "not_a_predictor",
            f"`{decision.column}` is not an exposure, a covariate or energy, so it does not enter "
            f"the models.",
            exits=[{"label": "Choose a predictor", "decision": None}])
    declared = set((state.categorical if state is not None else None) or [])
    info = _info(ctx, decision.column)
    if decision.column in declared or (info is not None and info.get("dtype") not in NUMERIC_DTYPES):
        raise Refusal(
            "not_numeric",
            f"`{decision.column}` is a category, so it has no curve or quintiles; its levels "
            f"already enter as indicators.",
            exits=[linear])
    n_unique = int(info.get("n_unique") or 0) if info is not None else None
    if n_unique is not None and n_unique < 5:
        raise Refusal(
            "too_few_values",
            f"`{decision.column}` has {n_unique} distinct values; a {decision.form} needs at least "
            f"5. Declared a category, each value gets its own coefficient.",
            exits=[linear, {"label": f"`{decision.column}` is a category",
                            "decision": SetCategorical(columns=[*sorted(declared), decision.column])}])
    adj = state.energy_adjustment if state is not None else None
    if adj is not None and adj.method == "partition" and decision.column in (adj.nutrients or []):
        raise Refusal(
            "replaced_by_energy_step",
            f"The energy partition replaces `{decision.column}` by its kcal, so it leaves the "
            f"model before any form could shape it.",
            exits=[linear])


# ── WP12: sensitivity analyses and measurement error ─────────────────────────


def _sensitivity_rules_are_eligibility_rules(decision: SetSensitivity, ctx: Any) -> None:
    """Each analysis's rules pass the checks the primary's do: real numeric columns, bounds that
    keep something, and never the outcome (RO-01 holds for a sensitivity analysis too)."""
    for i, analysis in enumerate(decision.analyses):
        rest = SetSensitivity(analyses=[a for j, a in enumerate(decision.analyses) if j != i])
        probe = SetExclusions(rules=list(analysis.rules))
        try:
            _exclusions_leave_the_outcome_alone(probe, ctx)
            _exclusions_are_ranges_on_numbers(probe, ctx)
        except Refusal as refused:
            raise Refusal(refused.code, f"In “{analysis.label}”: {refused.message}",
                          exits=[{"label": f"Leave out “{analysis.label}”", "decision": rest}]) from None


def _calibrated_exposures_are_columns(decision: SetMeasurementError, ctx: Any) -> None:
    columns = _columns_of(ctx)
    if columns is None or not decision.exposures:
        return
    unknown = [c for c in decision.exposures if c not in columns or c == ROW_ID]
    if unknown:
        rest = [c for c in decision.exposures if c not in unknown]
        raise Refusal(
            "unknown_column", f"This dataset has no column named {_and(unknown)}.",
            exits=[{"label": "Calibrate every energy-adjusted exposure",
                    "decision": decision.model_copy(update={"exposures": rest})}])


def _calibration_is_for_inference(decision: SetMeasurementError, ctx: Any) -> None:
    """Regression calibration corrects a coefficient; under prediction the model is used on recalls
    measured the same way, so there is nothing to correct (audit IN-22: right for prediction)."""
    state = _state(ctx)
    if decision.method == "none" or state is None or state.purpose != "prediction":
        return
    raise Refusal(
        "not_for_prediction",
        "Under prediction the model is used on recalls measured the same way as these, so its "
        "predictions need no correction; regression calibration corrects an exposure's coefficient, "
        "which is an inference question.",
        exits=[{"label": "Keep the recalls' mean uncorrected",
                "decision": SetMeasurementError(method="none")},
               {"label": "Change the purpose to inference", "decision": None}])


register_validator("set_sensitivity", _sensitivity_rules_are_eligibility_rules)
register_validator("set_measurement_error", _calibrated_exposures_are_columns)
register_validator("set_measurement_error", _calibration_is_for_inference)
register_validator("set_task", _task_fits_the_outcome)
register_validator("set_outcome_order", _order_names_the_outcome)
register_validator("set_exposure_form", _form_fits_the_column)
register_validator("set_follow_up", _follow_up_belongs_to_the_outcome)
register_validator("set_roles", _roles_name_real_columns)
register_validator("set_feature_table", _feature_table_names_the_files_columns)
register_validator("set_categorical", _categorical_names_predictors)
register_validator("set_missing", _left_out_columns_are_predictors)
register_validator("set_missing", _missing_fits_the_purpose)
register_validator("set_missing", _non_detections_are_not_filled_by_the_median)
register_validator("set_substitution", _substitution_moves_between_separate_nutrients)
register_validator("set_exclusions", _exclusions_are_ranges_on_numbers)
# first: a rule on the outcome is refused for that reason, whatever else is wrong with its bounds
register_validator("set_exclusions", _exclusions_leave_the_outcome_alone, first=True)
register_validator("set_target", _new_outcome_has_no_rule)
register_validator("revert", _revert_leaves_no_rule_on_the_outcome)
register_validator("set_energy_adjustment", _energy_adjustment_fits_the_roles)
register_validator("select_models", _models_can_fit_the_task)
register_validator("set_substitution", _substitution_swaps_energy)
register_validator("set_substitution", _substitution_has_every_energy_source)


# ── the fold ─────────────────────────────────────────────────────────────────

def _check_revertable(target_id: str, earlier: Mapping[str, DecisionRecord]) -> None:
    target = earlier.get(target_id)
    if target is None or not (
        isinstance(target.decision, Revert) or target.decision.kind in SLOTS
    ):
        raise Refusal(
            "unknown_decision",
            f"There is no earlier decision {target_id!r} that can be reverted.",
        )


def reverted(records: Sequence[DecisionRecord]) -> dict[str, str]:
    """Map each currently reverted record id to the id of the revert cancelling it.

    Raises :class:`Refusal` (``unknown_decision``) if a revert names a record
    that does not precede it or writes no slot.
    """
    ordered = sorted(records, key=lambda r: r.seq)
    earlier: dict[str, DecisionRecord] = {}
    for record in ordered:
        if isinstance(record.decision, Revert):
            _check_revertable(record.decision.decision_id, earlier)
        earlier[record.id] = record
    # Latest first: a revert only counts if nothing later has reverted it.
    cancelled: dict[str, str] = {}
    for record in reversed(ordered):
        if record.id in cancelled:
            continue
        if isinstance(record.decision, Revert):
            cancelled.setdefault(record.decision.decision_id, record.id)
    return cancelled


def fold(records: Sequence[DecisionRecord]) -> ProjectState:
    """The project state: each slot holds its latest write that is not reverted.

    A conditional write (``register_kind(..., holds=)``) counts only while its
    condition holds on the unconditional slots: the ``task`` slot is the latest
    ``set_task`` naming the current target, or unset.
    """
    ordered = sorted(records, key=lambda r: r.seq)
    cancelled = reverted(ordered)
    live = [r.decision for r in ordered if r.id not in cancelled and r.decision.kind in SLOTS]
    slots: dict[str, Any] = {}
    for decision in live:
        if decision.kind in _HOLDS:
            continue
        keyed = _KEYS.get(decision.kind)
        if keyed is not None:
            entries = dict(slots.get(SLOTS[decision.kind]) or {})
            entries[keyed(decision)] = _SLOT_VALUE[decision.kind](decision)
            slots[SLOTS[decision.kind]] = entries
        else:
            slots[SLOTS[decision.kind]] = _SLOT_VALUE[decision.kind](decision)
    base = dict(slots)
    for decision in live:
        holds = _HOLDS.get(decision.kind)
        if holds is not None and holds(decision, base):
            slots[SLOTS[decision.kind]] = _SLOT_VALUE[decision.kind](decision)
    return ProjectState(**slots)


def _check_revert(existing: Sequence[DecisionRecord], decision: Revert) -> None:
    _check_revertable(decision.decision_id, {r.id: r for r in existing})
    cancelled = reverted(existing)
    by = cancelled.get(decision.decision_id)
    if by is not None:
        raise Refusal(
            "already_reverted",
            "That decision has already been reverted.",
            exits=[{"label": "Undo the earlier revert instead", "decision": Revert(decision_id=by)}],
        )


# ── the log ──────────────────────────────────────────────────────────────────

class DecisionLog:
    """Append-only ``decisions.jsonl``. Safe across threads and processes.

    Each append is one ``write`` of one line under an exclusive ``flock``,
    followed by ``fsync``. A torn line left by a crash is skipped on read (with
    a warning) rather than making the project unreadable.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: tuple[tuple[int, int], list[DecisionRecord]] | None = None

    def records(self) -> list[DecisionRecord]:
        with self._lock:
            return list(self._load())

    def state(self) -> ProjectState:
        return fold(self.records())

    def append(
        self,
        decision: Any,
        note: str | None = None,
        sentence: Callable[[Any, ProjectState], str | None] | None = None,
    ) -> DecisionRecord:
        """Append ``decision``; ``sentence(decision, state_before)`` authors the record's sentence.

        ``state_before`` is the fold of the log as it stands under the lock, just before this
        record. A sentence that fails is logged and left unset: the answer is still recorded.
        """
        decision = parse_decision(decision)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                existing = self._load()
                if isinstance(decision, Revert):
                    _check_revert(existing, decision)
                before = fold(existing)
                text: str | None = None
                if sentence is not None:
                    try:
                        text = sentence(decision, before)
                    except Exception:  # noqa: BLE001 - a missing sentence never loses an answer
                        log.exception("no sentence for a %s decision", decision.kind)
                        text = None
                record = DecisionRecord(
                    id=uuid.uuid4().hex,
                    seq=max((r.seq for r in existing), default=0) + 1,
                    at=datetime.now(timezone.utc),
                    note=note,
                    sentence=text if isinstance(text, str) and text.strip() else None,
                    post_seal=bool(before.seal_opened),
                    decision=decision,
                )
                data = (record.model_dump_json() + "\n").encode("utf-8")
                size = os.fstat(fd).st_size
                if size and os.pread(fd, 1, size - 1) != b"\n":
                    data = b"\n" + data  # never glue onto a torn line
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self._cache = None
            return record

    def _load(self) -> list[DecisionRecord]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return []
        stamp = (st.st_size, st.st_mtime_ns)
        if self._cache is not None and self._cache[0] == stamp:
            return self._cache[1]
        records: list[DecisionRecord] = []
        text = self.path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d is a torn line; skipped", self.path, number)
                continue
            records.append(DecisionRecord.model_validate(raw))
        records.sort(key=lambda r: r.seq)
        self._cache = (stamp, records)
        return records


# The opening sequence's validators (orientation, event, grain, repeats, unit, aggregation,
# temporal) live beside the Router's rules; importing them registers them.
from turbotab.core import sequence as _sequence  # noqa: E402,F401
# The survey question's refusals (``set_survey``; audit §5 WP10) live with its name reading.
from turbotab.core import survey as _survey  # noqa: E402,F401

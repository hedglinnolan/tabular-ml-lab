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

# "other" is "Something else, or not sure" (audit RO-11, WP18): a first-class answer that runs the
# generic checks only, recorded and stated; it stands alone (``turbotab.core.structural``).
Lens = Literal["metabolomics", "genomics", "dietary", "clinical", "survey", "other"]
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

# "cluster" (audit WP13, IN-06): a column that groups participants (a site, a household, a family)
# is neither a person's identifier nor a trait; the seal never groups by it as if it named people.
Role = Literal["identifier", "exposure", "energy", "covariate", "design", "flag", "time", "excluded",
               "cluster"]
# "residual" keeps total energy in the outcome model; "residual_energy_dropped" lets it leave
# (BLUEPRINT §12 ruling 1); "all_components" gives every energy source its own term (audit WP6).
EnergyMethod = Literal["none", "standard", "residual", "residual_energy_dropped",
                       "density_multivariate", "density", "partition", "all_components"]
# "multiple_imputation" is inference's: chained equations with the outcome and total energy in the
# imputation model, pooled by Rubin's rules (BLUEPRINT §12 ruling 4; turbotab/core/methods/missing.py).
MissingStrategy = Literal["complete_case", "impute", "multiple_imputation"]
# How a left-censored column's blanks (values below a detection limit) are filled (audit ME-08);
# "qrilc" (MS7): quantile regression imputation of left-censored data, per sample.
BelowDetection = Literal["half_minimum", "censoring_aware", "as_missing", "qrilc"]


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
    weight: str = Field(min_length=1)
    # The unit ``weight`` is recorded in; pounds are converted to kilograms exactly (1 lb =
    # 0.45359237 kg) before the BMR equation reads them (BLUEPRINT §14.3: a recorded unit is
    # honored, never read as kg).
    weight_unit: Literal["kg", "lb"] = "kg"
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


# MS8 (MODELING_SEQUENCE §0 ruling 8): a multi-item scale scored as one predictor, its reliability
# (ω; α labeled customary), and the correction of its coefficient for measurement error. The
# arithmetic and its sources are in turbotab/core/methods/scales.py; the leash in turbotab/core/scales.py.
ScaleKind = Literal["reflective", "formative"]
ScaleStructure = Literal["unidimensional", "multidimensional"]
ScaleCorrection = Literal["none", "regression_calibration"]
ReliabilitySource = Literal["internal_consistency", "test_retest", "calibration_substudy"]
DEFAULT_GROUP_FACTORS = 3  # psych::omega's default number of group factors


class ScaleSpec(_Value):
    """One scale: its items, the instrument's key (the reverse-coded items and the response scale
    ``low``–``high`` they are turned over on), how the score is formed, what kind of construct it
    measures (``reflective``: the items are caused by it; ``formative``: an index defined by its
    components, such as a diet-quality score), its structure, the role the score takes in the
    models, and whether its coefficient is corrected for measurement error, from which reliability."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    items: list[str] = Field(min_length=3)
    reverse: list[str] = Field(default_factory=list)
    low: int
    high: int
    scoring: Literal["sum", "mean"] = "sum"
    kind: ScaleKind
    structure: ScaleStructure = "unidimensional"
    factors: int | None = Field(default=None, ge=2, le=10)  # group factors (multidimensional)
    role: Literal["exposure", "covariate"] = "exposure"
    correction: ScaleCorrection = "none"
    reliability: ReliabilitySource = "internal_consistency"
    # A repeat administration: its items in the order of ``items`` (scored with the same key), or
    # one column holding its score as recorded.
    retest: list[str] = Field(default_factory=list)
    reference: str | None = None  # a calibration substudy's reference measure (blank outside it)
    n_boot: int = Field(default=200, ge=50, le=2000)
    instrument: str | None = Field(default=None, max_length=80, pattern=r"^[^`\n\r]+$")

    @model_validator(mode="after")
    def _keyed(self) -> "ScaleSpec":
        if len(set(self.items)) != len(self.items):
            raise ValueError("each item may be listed only once")
        if not self.high > self.low:
            raise ValueError("the response scale's highest answer must exceed its lowest")
        stray = [c for c in self.reverse if c not in self.items]
        if stray:
            raise ValueError(f"reverse-coded items must be items of the scale: {', '.join(stray)}")
        if len(set(self.reverse)) != len(self.reverse):
            raise ValueError("each reverse-coded item may be listed only once")
        if self.factors is not None and self.structure != "multidimensional":
            raise ValueError("group factors belong to a multidimensional scale")
        if self.reliability == "test_retest" and len(self.retest) not in (1, len(self.items)):
            raise ValueError("a repeat administration is its items, one per item of the scale, or "
                             "one column holding its score")
        if self.reliability != "test_retest" and self.retest:
            raise ValueError("a repeat administration is read only for a test–retest reliability")
        if self.reliability == "calibration_substudy" and not self.reference:
            raise ValueError("a calibration substudy needs its reference measure's column")
        if self.reliability != "calibration_substudy" and self.reference:
            raise ValueError("a reference measure is read only for a calibration substudy")
        return self

    def group_factors(self) -> int:
        """The factors the reliability's factor analysis extracts: 1 for a unidimensional scale."""
        if self.structure != "multidimensional":
            return 1
        return int(self.factors or DEFAULT_GROUP_FACTORS)

    def columns(self) -> list[str]:
        """Every column the scale reads: its items, a repeat administration, a reference."""
        return [*self.items, *self.retest, *([self.reference] if self.reference else [])]


# The NCI usual-intake method (V2 definition of done, "Dietary, extended"; turbotab/core/usual_intake.py
# routes it, turbotab/core/methods/usual_intake.py fits it): one dietary component's distribution of
# usual intake, its own estimand beside any association analysis.
UsualIntakeModel = Literal["none", "amount_only", "two_part"]
UsualIntakePopulation = Literal["whole", "consumers"]  # STROBE-nut nut-14
CutoffKind = Literal["EAR", "AI", "UL", "other"]
# How a weekend column is coded: 1 on a Friday–Sunday recall (MIXTRAN's "weekend (Fri.-Sun.)
# indicator"), or NHANES's day of the week (``DR1DAY``: 1 Sunday … 7 Saturday).
WeekendCoding = Literal["indicator", "nhanes_day"]


class UsualIntakeSpec(_Value):
    model: UsualIntakeModel
    days: list[str] = Field(default_factory=list)  # a wide table's recall-day columns, first first
    order_column: str | None = None  # a long table's column ordering each person's recalls
    weekend: list[str] = Field(default_factory=list)
    weekend_coding: WeekendCoding = "indicator"
    population: UsualIntakePopulation = "whole"
    consumer_column: str | None = None
    cutoff: float | None = None
    cutoff_kind: CutoffKind | None = None
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
    """The confirmed reading of every column except the outcome.

    ``unconfirmed`` is the server's, never the client's (BLUEPRINT §14 rule 2): the proposals below
    high confidence this answer records exactly as proposed, read from the roles the server showed.
    A bulk confirm records them, and no number-changing default reads them until each has its own
    ``confirm_role``."""

    kind: Literal["set_roles"] = "set_roles"
    roles: dict[str, Role] = Field(min_length=1)
    unconfirmed: list[str] = Field(default_factory=list)


class ConfirmRole(_DecisionModel):
    """One column's role, confirmed on its own after its evidence was read (BLUEPRINT §14 rule 2:
    a proposal below high "needs its own confirmation; it never rides along in a bulk 'confirm
    all'"). One column per record, so an individual confirmation is individual by construction."""

    kind: Literal["confirm_role"] = "confirm_role"
    column: str = Field(min_length=1)
    role: Role


# The readings ledger (BLUEPRINT §14.1): the kinds of reading ``confirm_reading`` records. The
# others are answered by their own decisions (``set_outcome_unit``, ``set_repeat_kind``, …).
ReadingKind = Literal["role", "cluster", "unit", "day_count", "code_or_count", "time_column",
                      "nested_in", "sex_coding"]


class ConfirmReading(_DecisionModel):
    """One reading of the data, confirmed on its own after its evidence was read (BLUEPRINT §14.1:
    "Confirmation is one reading at a time, never a bulk confirm of uncertain readings"). One
    column, one kind and one value per record: a role (``exposure``), whether a column's
    repeating values cluster the rows (``yes``/``no``), a unit (``kg``), the days a total-energy
    value spans (``2``), whether whole numbers are codes or amounts, or that a column orders a
    unit's rows. ``confirm_role`` records stay valid: each is this decision's role kind."""

    kind: Literal["confirm_reading"] = "confirm_reading"
    reading: ReadingKind
    column: str = Field(min_length=1)
    value: str = Field(min_length=1, max_length=200, pattern=r"^[^`\n\r]+$")


class ReadingItem(BaseModel):
    """One reading a block confirmation lists, with the value it shows."""

    model_config = ConfigDict(extra="forbid")

    reading: ReadingKind
    column: str = Field(min_length=1)
    value: str = Field(min_length=1, max_length=200, pattern=r"^[^`\n\r]+$")


class ConfirmReadings(_DecisionModel):
    """A block confirmation (BLUEPRINT §14.2: "A block confirmation settles exactly the readings
    it lists, each with the value it shows. It never settles a reading it does not list."): the
    readings a question listed, each with its best guess as shown (or as the user changed it),
    recorded exactly as ``confirm_reading`` records each one. One record, so one revert undoes it."""

    kind: Literal["confirm_readings"] = "confirm_readings"
    items: list[ReadingItem] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def _each_reading_once(self) -> "ConfirmReadings":
        seen: set[tuple[str, str]] = set()
        for item in self.items:
            pair = (item.reading, item.column)
            if pair in seen:
                raise ValueError(f"the {item.reading.replace('_', ' ')} reading of "
                                 f"{item.column!r} is listed twice")
            seen.add(pair)
        return self


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


# MS7 (MODELING_SEQUENCE §2, "Batch"; turbotab/core/methods/batch.py): how a batch column is
# handled. "covariate": a term of the outcome model (inference's first); "reference_combat": ComBat
# with a reference batch fitted in each training fold without the outcome (prediction's first);
# "outcome_combat": ComBat with the outcome protected (refused for testing and under prediction);
# "none": left alone; "not_a_batch": the column is not a batch at all (the reading was wrong).
BatchMethod = Literal["covariate", "reference_combat", "outcome_combat", "none", "not_a_batch"]


class BatchSpec(_Value):
    column: str = Field(min_length=1)
    method: BatchMethod
    figures: bool = False  # ComBat with the outcome protected, for figures only (never the tests)


class SetBatch(_DecisionModel):
    """How a batch column is handled, by purpose (``turbotab.core.methods.batch``). ``figures``
    adds ComBat with the outcome protected for figures only: never the matrix the tests read."""

    kind: Literal["set_batch"] = "set_batch"
    column: str = Field(min_length=1)
    method: BatchMethod
    figures: bool = False


# MS7 (MODELING_SEQUENCE §2, "An exposure family"): the multiplicity method of an exposure family.
MultiplicityMethod = Literal["bh", "stated_count", "none"]


class MultiplicitySpec(_Value):
    method: MultiplicityMethod
    acknowledged: bool = False  # the attestation that keeps a blocked answer


class SetMultiplicity(_DecisionModel):
    """How an exposure family's tests are adjusted (``turbotab.core.methods.omics``):
    Benjamini–Hochberg q-values (implied when unanswered), unadjusted with the number of tests
    stated (a few prespecified hypotheses), or none; the last two beyond a few tests only with the
    recorded attestation (``acknowledged``)."""

    kind: Literal["set_multiplicity"] = "set_multiplicity"
    method: MultiplicityMethod
    acknowledged: bool = False


class SetScales(_DecisionModel):
    """The multi-item scales scored as predictors (MS8): each one's items become one score in the
    models. An empty list is the answer "no scale is scored"."""

    kind: Literal["set_scales"] = "set_scales"
    scales: list[ScaleSpec]

    @model_validator(mode="after")
    def _apart(self) -> "SetScales":
        names = [s.name for s in self.scales]
        if len(set(names)) != len(names):
            raise ValueError("each scale needs a name of its own")
        seen: dict[str, str] = {}
        for s in self.scales:
            for c in s.items:
                if c in seen:
                    raise ValueError(f"{c} is an item of both {seen[c]} and {s.name}")
                seen[c] = s.name
        return self


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
# "imputed_copies" (audit I18, WP18): each unit's rows are multiply-imputed copies of one record
# (NHANES DXA ships five, numbered by ``_MULT_``), neither repeats nor time points.
RepeatKind = Literal["repeats", "time_points", "imputed_copies"]
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
    # Imputed copies (audit I18): the column numbering each unit's copies (``_MULT_``), and the
    # attestation under inference that they are analyzed without Rubin's rules (block and record).
    implicate_column: str | None = None
    acknowledged: bool = False


class AggregationSpec(_Value):
    method: AggregationMethod
    outcome: Literal["mean", "first", "last"] | None = None  # when the outcome varies within a unit
    # A column's own rule, over the one the method and the column's kind give it (audit MA-14):
    # e.g. age at every visit kept at baseline under "change".
    columns: dict[str, CombineRule] = Field(default_factory=dict)
    # Audit RO-09 (WP18): under inference, predictors summarized from records later than the
    # outcome's are blocked and recorded; this is the attestation the methods sentence carries.
    acknowledged: bool = False


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
    implicate_column: str | None = None  # imputed copies: the column numbering them (``_MULT_``)
    acknowledged: bool = False  # imputed copies under inference: analyzed without Rubin's rules

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
    acknowledged: bool = False  # predictors summarized after the outcome, kept under inference


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


# Audit RO-10 (WP18): the scale a positive, markedly skewed outcome is analyzed on. "log" analyzes
# its natural logarithm, a column of its own (``ln_<outcome>``, :func:`log_outcome_name`) derived
# row by row in the working table: a coefficient is then a difference in mean log outcome, and its
# exponential a ratio of geometric means.
OutcomeScale = Literal["original", "log"]


def log_outcome_name(column: str) -> str:
    """The column a log-scale outcome is analyzed as."""
    return f"ln_{column}"


class OutcomeScaleSpec(_Value):
    column: str  # the outcome as the table spells it
    scale: OutcomeScale


class SetOutcomeScale(_DecisionModel):
    """The scale the outcome ``column`` is analyzed on. "log" makes ``ln_<column>`` the outcome
    (the target slot), so every stage, score and sentence reads the column it names; "original"
    keeps ``column``, its estimand a difference in means."""

    kind: Literal["set_outcome_scale"] = "set_outcome_scale"
    column: str = Field(min_length=1)
    scale: OutcomeScale


class SetOutcomeUnit(_DecisionModel):
    """The outcome's unit, as the user reads it from the source's data dictionary. Stands only
    while ``column`` is the target. A unit the name does not spell out is never guessed into a
    sentence (audit IN-05; CLINICAL_SURVEY_PACK §A1.1: "TurboTab will not guess")."""

    kind: Literal["set_outcome_unit"] = "set_outcome_unit"
    column: str = Field(min_length=1)
    unit: str = Field(min_length=1, max_length=24, pattern=r"^[^`\n\r]+$")


# Audit WP13 gate repair: a unit TurboTab could only propose (total energy's, by its median or by
# nothing; an age's, by a bare ``age``) is recorded by the user before any number is read in it.
ENERGY_UNITS = ("kcal", "kj")
AGE_UNITS = ("years", "months", "weeks", "days")
ColumnUnit = Literal["kcal", "kj", "years", "months", "weeks", "days"]
# Every unit a reading of a column may be confirmed in (``confirm_reading`` unit; readings.UNIT_VALUES).
RecordedUnit = Literal["kcal", "kj", "g", "kg", "lb", "cm", "m", "in", "years", "months", "weeks",
                       "days", "pct_energy", "drinks"]


class ColumnUnitSpec(_Value):
    """A column's recorded unit and, for total energy, the number of days each value totals: the
    one place both readings are kept (BLUEPRINT §14.3, every confirmation is honored), written by
    ``set_column_unit`` (both in one answer) and by ``confirm_reading`` / ``confirm_readings`` for a
    unit or a day count (each merged into what was recorded before). ``None``: not recorded."""

    unit: RecordedUnit | None = None
    days: int | None = 1
    # Alcohol counted in standard drinks (``confirm_reading`` unit ``drinks:<grams>``): the grams of
    # ethanol one drink holds, by the country's definition (8–20 g; readings.DRINK_GRAMS).
    grams_per_drink: float | None = None


class SetColumnUnit(_DecisionModel):
    """A column's unit, as the user reads it from the source's data dictionary, where TurboTab
    could only propose one: total energy in kcal or kJ (a day's, or a total over ``days`` days),
    or an age in years, months, weeks or days. Leash (BLUEPRINT §11.3): a unit not read from a
    stated suffix, a codebook or the values' own agreement is a proposal, and no screen, band or
    count is applied in it until it is recorded here."""

    kind: Literal["set_column_unit"] = "set_column_unit"
    column: str = Field(min_length=1)
    unit: ColumnUnit
    days: int = Field(default=1, ge=1, le=366)


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


# ── WP17 (AUDIT_REPORT §5): the declared purpose routes the questions ──────────
# The answers live here so the API and the log can parse them; their refusals, the roles the
# adjustment answers derive and the gates are in ``turbotab/core/estimand.py``.


class SetCensoring(_DecisionModel):
    """The follow-up question's "no" (audit RO-03): nobody's follow-up ended before the event could
    be seen, so the yes/no outcome counts events over one period for everyone. Stands only while
    ``column`` is the target. Follow-up that varies is a time to event instead (``set_task``, then
    ``set_follow_up``). ``acknowledged``: kept although a column reads as a follow-up time that
    varies (block and record)."""

    kind: Literal["set_censoring"] = "set_censoring"
    column: str = Field(min_length=1)
    acknowledged: bool = False


# How a grouping above the person enters an inference model (audit RO-08): each group its own
# intercept with intervals clustered by it, or the intervals clustered by it alone.
ClusterAdjustment = Literal["fixed_effects", "cluster_only"]


class ClusterSpec(_Value):
    """The cluster answer: the column grouping participants (a site, a centre, a household, a
    batch), or None when nothing groups them; under inference how the model takes it."""

    column: str | None = None
    adjust: ClusterAdjustment | None = None
    acknowledged: bool = False  # "no grouping" kept although a column reads as one


class SetClusters(_DecisionModel):
    kind: Literal["set_clusters"] = "set_clusters"
    column: str | None = None
    adjust: ClusterAdjustment | None = None
    acknowledged: bool = False


EffectKind = Literal["total", "direct"]
EnergyContrast = Literal["substitution", "addition"]
# The effect measure (MODELING_SEQUENCE §0 ruling 9). The engine fits the conditional measures of
# each family; the marginal risk difference and risk ratio (g-computation) are named so that a
# request for them is refused with the reason, never silently answered with an odds ratio.
EffectMeasure = Literal["mean_difference", "odds_ratio", "hazard_ratio", "cumulative_odds_ratio",
                        "relative_risk_ratio", "risk_difference", "risk_ratio",
                        "exposure_mean_difference"]
# The exposure family's sentinel in the adjustment answers: answered for every exposure of the family.
EXPOSURE_FAMILY = "*"


class EstimandSpec(_Value):
    """The exposure and the effect the inference reports (MODELING_SEQUENCE §1 step 2): one
    exposure, or an exposure family (``family``: every exposure-role column, each reported, with
    its multiplicity method; ``exposure`` is then None)."""

    exposure: str | None = None
    family: bool = False
    effect: EffectKind = "total"
    contrast: EnergyContrast | None = None  # an energy-bearing exposure: substitution or addition
    measure: EffectMeasure


class SetEstimand(_DecisionModel):
    kind: Literal["set_estimand"] = "set_estimand"
    exposure: str | None = None
    family: bool = False
    effect: EffectKind = "total"
    contrast: EnergyContrast | None = None
    measure: EffectMeasure

    @model_validator(mode="after")
    def _one_or_a_family(self) -> "SetEstimand":
        if self.family == (self.exposure is not None and self.exposure != ""):
            raise ValueError("name one exposure, or declare the exposure family, not both")
        return self


Answer3 = Literal["yes", "no", "unknown"]


class CovariateAnswers(_Value):
    """The modified disjunctive cause criterion, asked as questions (VanderWeele 2019; MODELING_
    SEQUENCE §1 step 3). The covariate's role is derived from these, never chosen directly."""

    causes_exposure: Answer3
    causes_outcome: Answer3
    after_exposure: Answer3  # could the exposure have changed it, or was it measured after it?
    instrument: bool = False  # known to affect the outcome only through the exposure
    proxy: bool = False  # stands in for an unmeasured cause of both
    # A mediator or collider kept in a total-effect set: block and record (MODELING_SEQUENCE §4):
    # refused until ``acknowledged``, which the record and the fit then state.
    keep: bool = False
    acknowledged: bool = False
    # "Further adjusted for" it: a labeled secondary model beside the primary.
    further: bool = False


class AdjustmentAnswer(CovariateAnswers):
    exposure: str  # the exposure these answers were given for


class SetAdjustment(_DecisionModel):
    """Answers for some covariates, against one exposure: one tap per group of covariates that
    share their answers (BLUEPRINT §14.2). Each covariate's answer is kept under its column, so a
    later group's answers never undo an earlier group's."""

    kind: Literal["set_adjustment"] = "set_adjustment"
    exposure: str = Field(min_length=1)
    answers: dict[str, CovariateAnswers] = Field(min_length=1)


class OpenSeal(_DecisionModel):
    """Open the held-out rows: once, at the end. Held-out scores are withheld until then.

    ``family`` is the final model, declared on cross-validation before any held-out score is seen
    (AUDIT_REPORT §5 WP8, ME-13; ``turbotab/core/models/selection.py``): its held-out score is the
    reported result. Required when several families are fitted; with one, the opening declares it.
    Null only in records from before the rule.
    """

    kind: Literal["open_seal"] = "open_seal"
    family: str | None = None
    # Filled by the server when the opening is recorded, never taken from a client (audit WP16,
    # RO-05): the outcome whose seal this opens (the opening holds only while it is the outcome: a
    # new outcome starts its own seal), how many rows were held out, the fit's primary metric, and
    # every family's held-out scores at this moment, ``{family: {metric: value}}``: the scores
    # the record keeps as the reported result, whatever is fitted later.
    target: str | None = None
    n_holdout: int | None = None
    metric: str | None = None
    scores: dict[str, dict[str, float | None]] | None = None


class Reseal(_DecisionModel):
    """Draw the held-out rows again after they were opened (audit WP16, RO-05).

    Once opened, the held-out rows hold still: a new seed or holdout, or anything else that would
    draw them again, waits for this recorded re-seal. It withdraws the opening of the current
    outcome's seal, so the rows drawn next are withheld until they are opened in turn; the scores
    at the first opening stay the reported result, and later held-out scores are not an
    independent test. ``target`` is filled by the server. It stands while its outcome is the
    target, as the opening does.
    """

    kind: Literal["reseal"] = "reseal"
    reason: str | None = None
    target: str | None = None

    @field_validator("reason")
    @classmethod
    def _said(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("a reason must say something")
        return value.strip() if value is not None else None


class LockPlan(_DecisionModel):
    """The inference analysis-plan lock (audit WP16, RO-12; MODELING_SEQUENCE §1 row 12).

    Recorded by the server when inference estimates are first displayed (or by the user, before
    that): the plan in force then is what was declared in the software before any estimate was
    displayed, and every later decision is marked as made after the estimates were seen. ``plan``
    (every slot the estimates read, as it stood) and ``digest`` (its SHA-256) are filled by the
    server. The lock is never undone.
    """

    kind: Literal["lock_plan"] = "lock_plan"
    plan: dict[str, Any] | None = None
    digest: str | None = None


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


class SetUsualIntake(_DecisionModel):
    """One dietary component's usual-intake distribution by the NCI method, or ``model="none"``
    (turbotab/core/usual_intake.py). ``nutrient`` names it: a long table's column, or the label of
    a wide table's ``days``."""

    kind: Literal["set_usual_intake"] = "set_usual_intake"
    nutrient: str = Field(min_length=1)
    model: UsualIntakeModel
    days: list[str] = Field(default_factory=list)
    order_column: str | None = None
    weekend: list[str] = Field(default_factory=list)
    weekend_coding: WeekendCoding = "indicator"
    population: UsualIntakePopulation = "whole"
    consumer_column: str | None = None
    cutoff: float | None = Field(default=None, gt=0)
    cutoff_kind: CutoffKind | None = None
    n_boot: int = Field(default=200, ge=50, le=2000)

    @model_validator(mode="after")
    def _shape(self) -> "SetUsualIntake":
        if len(set(self.days)) != len(self.days):
            raise ValueError("each recall day's column may be named only once")
        if len(self.days) == 1:
            raise ValueError("a wide table names at least two recall-day columns")
        if self.days and self.order_column:
            raise ValueError("a wide table's recalls are ordered by its day columns")
        if self.weekend and self.days and len(self.weekend) != len(self.days):
            raise ValueError("a wide table names one weekend column per recall day")
        if self.weekend and not self.days and len(self.weekend) != 1:
            raise ValueError("a long table names one weekend column")
        if self.cutoff_kind is not None and self.cutoff is None:
            raise ValueError("a cut-off's kind needs the cut-off")
        return self


class DismissFinding(_DecisionModel):
    kind: Literal["dismiss_finding"] = "dismiss_finding"
    finding_id: str = Field(min_length=1)
    reason: str | None = None


# V2 definition of done §1 (package DATAIN): minimal multi-file assembly and codebook import.
# Their logic lives in ``turbotab.core.assembly`` and ``turbotab.core.codebook``, which register
# their validators, completions and sentences when this module imports them (at its end).

FILE_ID = r"^f[0-9a-f]{10}$"
CODEBOOK_ID = r"^c[0-9a-f]{10}$"
JoinHow = Literal["left", "inner"]
JoinRelation = Literal["one-to-one", "one-to-many", "many-to-one"]


class JoinCounts(_Value):
    """What the join's preview counted, recorded with the answer (the server's): the rows on each
    side, the identifier values they share, the rows with no partner on each side, and the rows
    the joined table holds."""

    relation: JoinRelation
    table_rows: int = Field(ge=0)
    file_rows: int = Field(ge=0)
    matched_keys: int = Field(ge=0)
    table_unmatched: int = Field(ge=0)
    file_unmatched: int = Field(ge=0)
    rows: int = Field(ge=0)
    added_columns: int = Field(ge=0)
    renamed: dict[str, str] = Field(default_factory=dict)


class JoinSpec(_Value):
    """One file joined to the table (``joins`` slot, keyed by the file's id, in answer order)."""

    file: str = Field(pattern=FILE_ID)
    name: str
    on: str
    right_on: str | None = None
    how: JoinHow = "left"
    counts: JoinCounts | None = None


class JoinFiles(_DecisionModel):
    """Join a file added to the project to the table on a shared identifier (NHANES ships each
    component as its own file, joined on ``SEQN``). ``how``: ``left`` keeps every row of the
    table (a row with no partner holds blanks in the file's columns), ``inner`` only the rows with
    one. One-to-one, one-to-many and many-to-one are joined; many-to-many is refused with its
    reason. ``name`` and ``counts`` are the server's, never the client's: the preview's counts,
    recorded with the answer."""

    kind: Literal["join_files"] = "join_files"
    file: str = Field(pattern=FILE_ID)
    on: str = Field(min_length=1)
    right_on: str | None = None
    how: JoinHow = "left"
    name: str | None = None
    counts: JoinCounts | None = None


CodebookForm = Literal["table", "nhanes", "xpt"]
CodebookField = Literal["unit", "codes", "type", "range"]


class CodebookConflict(_Value):
    """A codebook field the values contradict: asked, never applied (BLUEPRINT §14.2–§14.3)."""

    column: str
    field: CodebookField
    says: str      # what the codebook documents
    values: str    # what the values show instead


class CodebookSpec(_Value):
    """One imported codebook as the state keeps it (``codebooks`` slot, keyed by its id):
    ``settled`` the readings its structured fields settled (``"<kind>:<column>"`` -> value),
    ``units`` the documented units no reading kind takes (``mg/dL``; a sentence may state them),
    ``labels`` its free-text labels of the table's columns (they only strengthen a guess),
    ``asked`` its fields the values contradict, ``kept`` the readings the user had answered
    otherwise before (the user's answer stands)."""

    name: str
    form: CodebookForm
    settled: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    asked: list[CodebookConflict] = Field(default_factory=list)
    kept: list[str] = Field(default_factory=list)
    n_entries: int = 0
    n_matched: int = 0


class ImportCodebook(_DecisionModel):
    """Import the researcher's own data dictionary (Nolan, 2026-10-03; BLUEPRINT §14.2): a
    variable/label/unit/codes table, an NHANES codebook page, or the labels an XPT file carries.
    Its structured fields (units, value-code tables, the variable type) settle readings as the
    user's own documentation, through the confirmation path ``confirm_readings`` writes; its
    free-text labels only strengthen the guesses the ask leads with; a field the values contradict
    is asked, never applied. The client names the staged codebook; everything else is the
    server's, read from the codebook and the values."""

    kind: Literal["import_codebook"] = "import_codebook"
    codebook: str = Field(pattern=CODEBOOK_ID)
    name: str | None = None
    form: CodebookForm | None = None
    items: list[ReadingItem] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    asked: list[CodebookConflict] = Field(default_factory=list)
    kept: list[str] = Field(default_factory=list)
    n_entries: int = 0
    n_matched: int = 0


Decision = Annotated[
    Union[
        SetLens, SetTarget, SetTask, SetPurpose, Revert,
        SetRoles, SetEnergyAdjustment, SetExclusions, SetMissing, SetSplit,
        SelectModels, SetSubstitution,
        SetOrientation, SetEvent, SetGrain, SetRepeatKind, SetUnit, SetAggregation, SetTemporal,
        OpenSeal, ApplyRepair, DeferFinding, DismissFinding,
        SetFeatureTable, SetCategorical, SetSurvey,
        SetExposureForm, SetOutcomeOrder, SetFollowUp,
        SetSensitivity, SetMeasurementError, SetOutcomeUnit, SetColumnUnit, ConfirmRole,
        ConfirmReading, ConfirmReadings, SetOutcomeScale,
        Reseal, LockPlan,
        SetCensoring, SetClusters, SetEstimand, SetAdjustment,
        JoinFiles, ImportCodebook, SetBatch, SetMultiplicity, SetScales,
        SetUsualIntake,
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
    # record from before the opening (and every pre-M2 line) reads False. A seal opened for an
    # earlier outcome, or before a re-seal, counts: the held-out scores were seen (audit RO-05).
    post_seal: bool = False
    # Recorded after the inference estimates were first displayed (``lock_plan``; audit WP16,
    # RO-12), set the same way: the decision was made with the estimates in view.
    after_estimates: bool = False
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
    # Whether the current outcome's seal is open: its latest opening stands and no re-seal came
    # after it (audit WP16, RO-05). None for a new outcome's seal, or after a re-seal.
    seal_opened: bool | None = None
    # The inference analysis plan was locked when its estimates were first displayed (RO-12).
    plan_locked: bool | None = None
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
    # MS8: the multi-item scales scored as predictors, their reliability and correction
    scales: list[ScaleSpec] | None = None
    # The NCI usual-intake method: each dietary component's distribution, keyed by its name
    usual_intake: dict[str, UsualIntakeSpec] | None = None
    # WP13 (audit IN-05): the outcome's unit as the user recorded it (holds while its column is
    # the target); a unit the name does not spell out is proposed, never stated, until then
    outcome_unit: str | None = None
    # WP13 gate repair: a column's unit the user recorded where TurboTab could only propose one
    # (total energy in kcal or kJ, an age in months): column -> its unit
    column_units: dict[str, ColumnUnitSpec] | None = None
    # BLUEPRINT §14 (Recognition's leash): the proposals below high the roles answer recorded as
    # proposed (written by ``set_roles``), and each column's own confirmation (``confirm_role``):
    # a number-changing default reads a role only once it is settled (turbotab.core.readings)
    roles_unconfirmed: list[str] | None = None
    role_confirmations: dict[str, Role] | None = None
    # BLUEPRINT §14.1 (the readings ledger): each reading the user confirmed on its own
    # (``confirm_reading``), keyed ``"<kind>:<column>"`` -> the value recorded for it
    reading_confirmations: dict[str, str] | None = None
    # ... and the two that shape the working table (whether whole numbers are codes or amounts, the
    # column that orders a unit's records), kept apart so that confirming a unit or a role never
    # rebuilds it
    shape_confirmations: dict[str, str] | None = None
    # BLUEPRINT §14.3: which level of a sex column is female and which male, as the user confirmed
    # it (``"female=2,male=1"``, by column), kept apart so the detectors that read it (the CDC
    # growth charts' z-scores) recompute alone
    sex_codings: dict[str, str] | None = None
    # WP17 (audit §5): the declared purpose routes the questions. The follow-up question's "the same
    # for everyone" (holds while its column is the target); the grouping above the person; under
    # inference the exposure and its effect, and each covariate's answers to the disjunctive cause
    # criterion, by column (``turbotab/core/estimand.py``)
    censoring: Literal["same", "same_attested"] | None = None
    clusters: ClusterSpec | None = None
    estimand: EstimandSpec | None = None
    adjustment: dict[str, AdjustmentAnswer] | None = None
    # Audit RO-10 (WP18): the scale the user chose for a positive, markedly skewed outcome; under
    # "log" the target slot names the derived ``ln_<column>`` (``SetOutcomeScale``)
    outcome_scale: OutcomeScaleSpec | None = None
    # V2 definition of done §1 (DATAIN): the files joined to the table, by file id in answer order
    # (``join_files``; the ingest stage reads it), and each imported codebook by its id
    # (``import_codebook``): what its structured fields settled, its units and labels, and what
    # the values contradicted
    joins: dict[str, JoinSpec] | None = None
    codebooks: dict[str, CodebookSpec] | None = None
    # MS7 (MODELING_SEQUENCE §2): how a batch column is handled, and an exposure family's
    # multiplicity method (``set_batch``, ``set_multiplicity``)
    batch: BatchSpec | None = None
    multiplicity: MultiplicitySpec | None = None

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

    out += [c for c in unusable_columns(state) if c not in out]
    # WP17: under inference, the covariates the adjustment answers leave out of the primary model
    # (a mediator in a total-effect set, a collider, an instrument, a non-cause; MODELING_SEQUENCE
    # §1 step 3).
    from turbotab.core.estimand import adjustment_left_out

    return out + [c for c in adjustment_left_out(state) if c not in out]


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
# kind -> {another slot it writes: its value}: one answer that also records a fact beside its slot
# (``set_roles`` writes the roles and which of them were recorded unconfirmed).
_ALSO: dict[str, dict[str, Callable[[Any], Any]]] = {}
_HOLDS: dict[str, Callable[[Any, Mapping[str, Any]], bool]] = {}
_KEYS: dict[str, Callable[[Any], str]] = {}
_SLOT_FOR: dict[str, Callable[[Any], str]] = {}
# kind -> its writes as ``[(slot, key, value), …]``: one answer that writes several keyed entries
# (``confirm_readings``: each listed reading where ``confirm_reading`` would write it).
_ENTRIES: dict[str, Callable[[Any], list[tuple[str, str, Any]]]] = {}
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
    also: Mapping[str, Callable[[Any], Any]] | None = None,
    slot_for: Callable[[Any], str] | None = None,
    entries: Callable[[Any], list[tuple[str, str, Any]]] | None = None,
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
    ``also`` (``{slot: value(decision)}``) names further slots an unconditional,
    unkeyed kind writes with its own: ``set_roles`` records which proposals it
    carried unconfirmed (BLUEPRINT §14) beside the roles themselves.
    ``slot_for(decision)``, for a keyed kind, names the slot one decision writes where
    it is not ``slot`` (``confirm_reading`` keeps a role's confirmation beside the
    others ``confirm_role`` wrote, so confirming a role reshapes no table).
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
    if also:
        for extra in also:
            if extra not in ProjectState.model_fields:
                raise ValueError(f"ProjectState has no slot {extra!r}; add the field first")
        _ALSO[kind] = dict(also)
    else:
        _ALSO.pop(kind, None)
    if key is not None:
        _KEYS[kind] = key
    else:
        _KEYS.pop(kind, None)
    if holds is not None:
        _HOLDS[kind] = holds
    else:
        _HOLDS.pop(kind, None)
    if slot_for is not None:
        _SLOT_FOR[kind] = slot_for
    else:
        _SLOT_FOR.pop(kind, None)
    if entries is not None:
        _ENTRIES[kind] = entries
    else:
        _ENTRIES.pop(kind, None)
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
register_kind(SetRoles, "roles", value=lambda d: d.roles,
              also={"roles_unconfirmed": lambda d: list(d.unconfirmed)})
register_kind(ConfirmRole, "role_confirmations", value=lambda d: d.role, key=lambda d: d.column)
# A role's confirmation is kept where ``confirm_role`` keeps it (by column), so the stages that read
# roles read it and the table-shaping stages, which read the other confirmations, never recompute.
def reading_slot(reading: str, column: str) -> tuple[str, str]:
    """Where a reading's confirmation is kept, ``(slot, key)``: a role's beside ``confirm_role``'s
    (by column), a unit's and a day count's with ``set_column_unit``'s (by column: one store, so
    the screens, the findings, the coach and the substitution read one answer, the latest), the
    two that shape the working table apart, every other in ``reading_confirmations``
    (``"<kind>:<column>"``)."""
    if reading == "role":
        return "role_confirmations", column
    if reading == "sex_coding":
        return "sex_codings", column
    if reading in ("unit", "day_count"):
        return "column_units", column
    if reading in ("code_or_count", "time_column"):
        return "shape_confirmations", f"{reading}:{column}"
    return "reading_confirmations", f"{reading}:{column}"


def _codebook_spec(d: "ImportCodebook") -> "CodebookSpec":
    return CodebookSpec(name=d.name or d.codebook, form=d.form or "table",
                        settled={f"{i.reading}:{i.column}": i.value for i in d.items},
                        units=dict(d.units), labels=dict(d.labels), asked=list(d.asked),
                        kept=list(d.kept), n_entries=d.n_entries, n_matched=d.n_matched)


def reading_entry(reading: str, column: str, value: str) -> tuple[str, str, Any]:
    """One confirmation's write, ``(slot, key, value)``: a unit or a day count is merged into the
    column's recorded spec (a callable the fold applies to the entry before it), every other
    reading written as its value."""
    slot, key = reading_slot(reading, column)
    if reading == "unit":
        from turbotab.core.readings import DRINKS, parse_drinks

        grams = parse_drinks(value)
        return slot, key, lambda prev: ColumnUnitSpec(
            unit=DRINKS if grams is not None else value, grams_per_drink=grams,
            days=getattr(prev, "days", None) if prev is not None else None)
    if reading == "day_count":
        return slot, key, lambda prev: ColumnUnitSpec(
            unit=getattr(prev, "unit", None) if prev is not None else None, days=int(value),
            grams_per_drink=getattr(prev, "grams_per_drink", None) if prev is not None else None)
    return slot, key, value


register_kind(ConfirmReading, "reading_confirmations", value=lambda d: d.value,
              entries=lambda d: [reading_entry(d.reading, d.column, d.value)])
# A block confirmation writes each listed reading exactly where its own confirmation would, and
# nothing else (BLUEPRINT §14.2).
register_kind(ConfirmReadings, "reading_confirmations", value=lambda d: None,
              entries=lambda d: [reading_entry(i.reading, i.column, i.value) for i in d.items])
register_kind(JoinFiles, "joins", key=lambda d: d.file,
              value=lambda d: JoinSpec(file=d.file, name=d.name or d.file, on=d.on,
                                       right_on=d.right_on, how=d.how, counts=d.counts))
# A codebook's structured fields write each reading exactly where its own confirmation would
# (BLUEPRINT §14.2, "let the codebook answer"), and the codebook itself under its id: the
# readings ledger names it as their evidence (``readings.codebook_source``).
register_kind(ImportCodebook, "codebooks", value=lambda d: None,
              entries=lambda d: [*(reading_entry(i.reading, i.column, i.value) for i in d.items),
                                 ("codebooks", d.codebook, _codebook_spec(d))])
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
# An opening, and a re-seal, stand only while the outcome they name is the target (audit WP16,
# RO-05): a new outcome starts its own seal. The latest that stands decides: opened, or (after a
# re-seal) sealed again. Records from before the rule name no outcome and always stand.
def _for_this_outcome(decision: Any, slots: Mapping[str, Any]) -> bool:
    return decision.target is None or slots.get("target") == decision.target


register_kind(OpenSeal, "seal_opened", value=lambda d: True, holds=_for_this_outcome)
register_kind(Reseal, "seal_opened", value=lambda d: None, holds=_for_this_outcome)
register_kind(LockPlan, "plan_locked", value=lambda d: True)
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
register_kind(SetOutcomeUnit, "outcome_unit", value=lambda d: d.unit,
              holds=lambda d, slots: slots.get("target") == d.column)
register_kind(SetColumnUnit, "column_units", key=lambda d: d.column,
              value=lambda d: ColumnUnitSpec(unit=d.unit, days=d.days))
register_kind(SetSensitivity, "sensitivity")
# The outcome's scale writes the target: "log" makes the derived log column the outcome, so every
# consumer of the target reads the column it names (and the task, event and unit answered for the
# original column stop holding); the answer itself is kept beside it.
register_kind(SetOutcomeScale, "target",
              value=lambda d: log_outcome_name(d.column) if d.scale == "log" else d.column,
              also={"outcome_scale": lambda d: OutcomeScaleSpec(column=d.column, scale=d.scale)})
register_kind(SetMeasurementError, "measurement_error",
              value=lambda d: MeasurementErrorSpec(**d.model_dump(exclude={"kind"})))
# WP17: the follow-up, the grouping above the person, the estimand and the adjustment answers
register_kind(SetCensoring, "censoring",
              value=lambda d: "same_attested" if d.acknowledged else "same",
              holds=lambda d, slots: slots.get("target") == d.column)
register_kind(SetClusters, "clusters", value=lambda d: ClusterSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetEstimand, "estimand", value=lambda d: EstimandSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetAdjustment, "adjustment", value=lambda d: None,
              entries=lambda d: [("adjustment", column, AdjustmentAnswer(
                  exposure=d.exposure, **answers.model_dump()))
                  for column, answers in d.answers.items()])
register_kind(SetBatch, "batch", value=lambda d: BatchSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetMultiplicity, "multiplicity",
              value=lambda d: MultiplicitySpec(**d.model_dump(exclude={"kind"})))
register_kind(SetScales, "scales")
register_kind(SetUsualIntake, "usual_intake", key=lambda d: d.nutrient,
              value=lambda d: UsualIntakeSpec(**d.model_dump(exclude={"kind", "nutrient"})))
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


def _store_of(ctx: Any) -> Any:
    """The table's DataStore when ``ctx`` can open it (the server's context), else None."""
    opener = _ctx(ctx, "store")
    if callable(opener):
        try:
            return opener()
        except Exception:  # noqa: BLE001 - no store: the stage that reads the values asks
            return None
    return opener


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


def task_fits(n_unique: int, numeric: bool) -> list[str]:
    """The tasks an outcome with ``n_unique`` distinct values may be answered as: the one rule
    the explicit answer (:func:`_task_fits_the_outcome`) and the detection's skip
    (``stages.target``) share, so a task the Router would state is one the answer accepts (audit
    RO-10: a text outcome above 20 classes was skipped as multiclass, an explicit multiclass
    refused)."""
    fits: list[str] = []
    if n_unique == 2:
        fits.extend(["binary", "time_to_event"])
    if 2 < n_unique <= MAX_CLASSES_MULTICLASS:
        fits.extend(["multiclass", "ordinal"])
    if numeric and n_unique >= 2:
        fits.append("regression")
    return fits


def _task_fits_the_outcome(decision: SetTask, ctx: Any) -> None:
    info = _info(ctx, decision.column)
    if info is None:
        return
    n_unique = int(info.get("n_unique") or 0)
    numeric = info.get("dtype") in NUMERIC_DTYPES
    fits = task_fits(n_unique, numeric)
    exits = [{"label": f"Treat it as {task}", "decision": SetTask(column=decision.column, task=task)}
             for task in fits if task != "time_to_event"
             and not (task == "regression" and n_unique <= 2)]
    if decision.task == "regression" and not numeric:
        # WP18 (audit RO-10): the answer refuses what the skip never states, so the two agree.
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` holds labels, not numbers, so it has no mean to model as a "
            f"regression outcome.",
            exits=exits or [{"label": "Choose another outcome", "decision": None}],
        )
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


# ── Recognition's leash (BLUEPRINT §14) ─────────────────────────────────────


def _roles_record_what_rode_along(decision: SetRoles, ctx: Any) -> SetRoles:
    """The proposals below high that this answer records exactly as proposed, read from the roles
    the server showed (the fresh ``roles`` artifact): what a bulk "confirm all" carried without the
    user's own look. The server's, never the client's; with no roles artifact to read, the answer
    is the user's own and nothing rode along."""
    from turbotab.core.readings import proposals_of, rode_along

    opener = _ctx(ctx, "artifact")
    try:
        proposals = proposals_of(opener("roles")) if callable(opener) else []
    except Exception:  # noqa: BLE001 - no proposals to read
        proposals = []
    if not proposals:
        # BLUEPRINT §14.1 (the census: "a set_roles recorded with no fresh roles artifact settles
        # everything"): while the roles stage recomputes, the proposals a client was shown are the
        # newest it computed, stale or not, and a bulk answer is read against them.
        shown = _ctx(ctx, "shown")
        try:
            proposals = proposals_of(shown("roles")) if callable(shown) else []
        except Exception:  # noqa: BLE001 - nothing was shown
            proposals = []
    if proposals:
        return decision.model_copy(update={"unconfirmed": rode_along(decision.roles, proposals)})
    # No proposal was ever computed, so none was shown and none could ride along: the roles are
    # the user's own answer (a script, a unit test, or a client that names them itself).
    return decision.model_copy(update={"unconfirmed": []})


def _reading_names_a_column(decision: "ConfirmReading", ctx: Any) -> None:
    """A confirmation names one of the dataset's columns, never the outcome (whose unit and kind
    have their own answers), with a value its kind can take."""
    from turbotab.core.readings import VALUES

    columns = _columns_of(ctx)
    if columns is not None and (decision.column not in columns or decision.column == ROW_ID):
        raise Refusal("unknown_column", f"This dataset has no column named `{decision.column}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is not None and decision.column == target:
        raise Refusal("target_has_role",
                      f"`{target}` is the outcome; its unit and kind have their own questions.",
                      exits=[{"label": "Confirm another column", "decision": None}])
    allowed = VALUES.get(decision.reading)
    if decision.reading == "nested_in":
        # Any column of the table is a total the user may name, or none (BLUEPRINT §14.3: the
        # sixth gate's confirmation naming another total than the design's was accepted and read
        # nowhere; every consumer of the nesting now reads it, ``readings.nesting``).
        from turbotab.core.readings import NOT_NESTED, nesting_parents

        if decision.value == NOT_NESTED:
            return
        eligible = (None if columns is None else
                    nesting_parents(decision.column, sorted(columns),
                                    None if target is _UNKNOWN else target))
        if decision.value == decision.column or (eligible is not None
                                                 and decision.value not in eligible):
            what = ("itself" if decision.value == decision.column else
                    "the outcome" if decision.value == target else
                    f"`{decision.value}`, which is no column of this dataset")
            raise Refusal("unknown_column",
                          f"`{decision.column}` cannot be part of {what}: name another column of "
                          f"the table as its total, or none.",
                          exits=[{"label": "Name the total it is part of", "decision": None},
                                 {"label": f"`{decision.column}` is part of no total",
                                  "decision": ConfirmReading(reading="nested_in",
                                                             column=decision.column,
                                                             value=NOT_NESTED)}])
        state = _state(ctx)
        if state is not None:
            from turbotab.core.readings import nesting

            if nesting(state).get(decision.value) == decision.column:
                raise Refusal("part_of_the_other",
                              f"`{decision.value}` is recorded as part of `{decision.column}`, so "
                              f"`{decision.column}` cannot also be part of it.",
                              exits=[{"label": f"Change what `{decision.value}` is part of",
                                      "decision": None}])
        return
    if decision.reading == "code_or_count" and decision.value == "amount":
        store = _store_of(ctx)
        reader = getattr(store, "text_numbers", None)
        try:
            found = (reader([decision.column]) if callable(reader) else {}).get(decision.column)
        except Exception:  # noqa: BLE001 - no reading of its values: the fit's design asks
            found = None
        if found is not None and found.get("labels"):
            raise Refusal(
                "not_numbers",
                f"`{decision.column}` holds labels, not numbers ({found.get('evidence')}): read as "
                f"amounts, every value would be blank.",
                exits=[{"label": f"`{decision.column}` holds codes for categories",
                        "decision": ConfirmReading(reading="code_or_count", column=decision.column,
                                                   value="code")},
                       {"label": f"Leave `{decision.column}` out of the model",
                        "decision": ConfirmReading(reading="role", column=decision.column,
                                                   value="excluded")}])
    if decision.reading == "day_count":
        ok = decision.value.isdigit() and 1 <= int(decision.value) <= 366
        allowed_words = "a whole number of days from 1 to 366"
    elif decision.reading == "unit":
        from turbotab.core.readings import parse_drinks

        ok = decision.value in (allowed or ()) or parse_drinks(decision.value) is not None
        allowed_words = (", ".join(f"`{v}`" for v in allowed or ()) + ", or `drinks:<grams>` for "
                         "standard drinks of 8 to 20 g of alcohol")
    elif decision.reading == "sex_coding":
        from turbotab.core.readings import parse_sex_coding

        ok = parse_sex_coding(decision.value) is not None
        allowed_words = "`female=<level>,male=<level>`, two different levels"
    else:
        ok = allowed is None or decision.value in allowed
        allowed_words = ", ".join(f"`{v}`" for v in allowed or ())
    if not ok:
        raise Refusal("not_a_value",
                      f"`{decision.value}` is not a value a {decision.reading.replace('_', ' ')} "
                      f"reading takes; it takes {allowed_words}.",
                      exits=[{"label": "Choose one of its values", "decision": None}])


def _readings_name_columns(decision: "ConfirmReadings", ctx: Any) -> None:
    """Each reading a block confirmation lists passes the checks its own confirmation would; a
    reading that fails is named, and the block without it offered."""
    for i, item in enumerate(decision.items):
        try:
            _reading_names_a_column(ConfirmReading(reading=item.reading, column=item.column,
                                                   value=item.value), ctx)
        except Refusal as refused:
            rest = [it for j, it in enumerate(decision.items) if j != i]
            exits = ([{"label": f"Confirm the others, leaving `{item.column}` out",
                       "decision": ConfirmReadings(items=rest)}] if rest else [])
            raise Refusal(refused.code, refused.message,
                          exits=[*exits, *refused.exits]) from None


def _models_read_settled_readings(decision: Any, ctx: Any) -> None:
    """BLUEPRINT §14.1: a fit is a number-changing consumer of every role (a column enters the
    model or leaves it by its role) and of each whole-number predictor's code-or-amount reading.
    While one rode along unconfirmed, the fit asks, one exit per reading."""
    from turbotab.core.readings import Unsettled, predictors_or_ask

    state = _state(ctx)
    if state is None or not state.roles:
        return
    try:
        predictors_or_ask(state, _ctx(ctx, "column_info"), drop=left_out(state),
                          store=_store_of(ctx))
    except Unsettled as waiting:
        raise Refusal("reading_unsettled", str(waiting), exits=[
            *waiting.exits,
            {"label": "Change the roles (the roles question)", "decision": None}]) from None


def _confirmed_role_names_a_column(decision: ConfirmRole, ctx: Any) -> None:
    columns = _columns_of(ctx)
    if columns is not None and (decision.column not in columns or decision.column == ROW_ID):
        raise Refusal("unknown_column", f"This dataset has no column named `{decision.column}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is not None and decision.column == target:
        raise Refusal("target_has_role",
                      f"`{target}` is the outcome, so it has no role among the predictors.",
                      exits=[{"label": "Confirm another column", "decision": None}])


def _defaults_and_their_columns(state: Any) -> dict[str, list[str]]:
    """The recorded number-changing answers, each with the columns whose roles it reads."""
    out: dict[str, list[str]] = {}
    adj = getattr(state, "energy_adjustment", None)
    if adj is not None and adj.method != "none":
        out["energy adjustment"] = [c for c in (adj.energy_column, *adj.nutrients) if c]
    survey = getattr(state, "survey", None)
    if survey is not None and survey.estimand == "population":
        out["survey answer"] = [c for c in (survey.weight, survey.strata, survey.psu,
                                            survey.four_year_weight) if c]
    roles = getattr(state, "roles", None) or {}
    on_energy = [r.column for r in (getattr(state, "exclusions", None) or [])
                 if roles.get(r.column) == "energy"]
    if on_energy:
        out["exclusion rules"] = list(dict.fromkeys(on_energy))
    return out


def _answers_keep_settled_roles(decision: Any, ctx: Any) -> None:
    """BLUEPRINT §14 rule 2 (and §13's "invalidates"): an answer that would un-settle a role a
    recorded number-changing answer reads (reverting its confirmation, or recording the roles
    again so that it rides along unconfirmed) is refused until that answer changes; the default
    is never kept silently on a role nobody confirmed."""
    from turbotab.core.readings import unsettled

    if decision.kind == "set_roles":
        decision = _roles_record_what_rode_along(decision, ctx)
    after = state_after(decision, ctx)
    now = _state(ctx)
    if after is None or not after.roles:
        return
    newly: dict[str, list[str]] = {}
    for what, columns in _defaults_and_their_columns(after).items():
        before = set(unsettled(now, columns)) if now is not None else set()
        waiting = [c for c in unsettled(after, columns) if c not in before]
        if waiting:
            newly[what] = waiting
    if not newly:
        return
    what, columns = next(iter(newly.items()))
    one = len(columns) == 1
    raise Refusal(
        "role_unconfirmed",
        f"The recorded {what} reads {_and(columns)}, which would then be a role nobody confirmed. "
        f"Change the {what} first, or keep {'its' if one else 'their'} confirmation.",
        exits=[{"label": "Keep the answers as they are", "decision": None}])


def _energy_reads_settled_roles(decision: SetEnergyAdjustment, ctx: Any) -> None:
    """BLUEPRINT §14 rule 2: the energy column and the nutrients adjusted are number-changing
    defaults, so each must be a settled role: high and confirmed, or confirmed on its own."""
    if decision.method == "none":
        return
    state = _state(ctx)
    if state is None or not state.roles:
        return
    from turbotab.core.readings import confirm_exits, unsettled, unsettled_message

    named = [c for c in (decision.energy_column, *decision.nutrients) if c]
    waiting = unsettled(state, list(dict.fromkeys(named)))
    if not waiting:
        return
    base = decision.model_dump(exclude={"kind"})
    kept = [n for n in decision.nutrients if n not in waiting]
    exits = confirm_exits(state, waiting)
    if decision.energy_column not in waiting and kept:
        exits.append({"label": "Adjust only the confirmed nutrients",
                      "decision": SetEnergyAdjustment(**{**base, "nutrients": kept})})
    exits.append({"label": "Do not adjust for energy",
                  "decision": SetEnergyAdjustment(**{**base, "method": "none", "energy_column": None,
                                                     "nutrients": [], "strata": None,
                                                     "log_transform": False})})
    raise Refusal("role_unconfirmed", unsettled_message(waiting, "the energy adjustment"),
                  exits=exits)


def _screens_read_a_settled_energy_column(decision: SetExclusions, ctx: Any) -> None:
    """BLUEPRINT §14 rule 2: an intake screen's bounds are read in the energy column; a column
    whose energy role rode along unconfirmed sets no exclusion until it is confirmed."""
    state = _state(ctx)
    if state is None or not state.roles:
        return
    from turbotab.core.readings import confirm_exits, unsettled, unsettled_message

    on_energy = [r.column for r in decision.rules if state.roles.get(r.column) == "energy"]
    waiting = unsettled(state, list(dict.fromkeys(on_energy)))
    if not waiting:
        return
    kept = [r for r in decision.rules if r.column not in waiting]
    raise Refusal("role_unconfirmed", unsettled_message(waiting, "an exclusion rule"),
                  exits=[*confirm_exits(state, waiting),
                         {"label": "Keep the rules on confirmed columns only",
                          "decision": SetExclusions(rules=kept)}])


_BODY_EXPECTED = (("age", "age", "years"), ("weight", "weight", None), ("height", "height", None))
_SEX_WORDS = ("sex", "gender")


def _sex_named(column: str) -> bool:
    from turbotab.core.recognizers import tokens

    return column.lower() == "riagendr" or bool(set(tokens(column)) & set(_SEX_WORDS))


def _screens_read_settled_body_measures(decision: SetExclusions, ctx: Any) -> None:
    """BLUEPRINT §14.1, §14.3: the Goldberg screen reads a weight in the unit its rule records (kg,
    or lb converted exactly), a height in cm or m and an age in years, and each screen that reads
    a sex column reads which of its levels is female. Each is a reading the screen changes numbers
    by. A unit nobody recorded (a header's ``kg`` is a name; a weight's median tells kg from lb no
    better than a heavy cohort from a light one: the gate's US women's ``weight`` in pounds, 136 of
    500 rows excluded against 5), a sex column's numeric codes nobody confirmed (NHANES codes 1
    male, 2 female; other studies 1 female), or a role that rode along unconfirmed is asked first,
    one exit per reading; a rule whose unit or levels contradict the recorded reading is refused."""
    from turbotab.core.readings import (
        BODY_CONVERSIONS, body_unit_reading, confirm_exit, confirm_exits, listing,
        parse_sex_coding, sex_coding_exits, sex_coding_reading, unsettled,
    )

    state = _state(ctx)
    store = _store_of(ctx)

    def values(column: str) -> Any:
        if store is None or column not in set(store.columns):
            return None
        return store.materialize([column])[column]

    exits: list[dict[str, Any]] = []
    units: list[str] = []
    codings: list[str] = []
    wrong: list[str] = []
    roles_waiting: list[str] = []

    def sex_levels_settled(column: str, female: Sequence[Any], male: Sequence[Any]) -> None:
        found = sex_coding_reading(state, column, values(column))
        if not found.settled:
            if column not in codings:
                codings.append(column)
                exits.extend(sex_coding_exits(column, values(column)))
            return
        coding = parse_sex_coding(found.value) or {}
        said = {**{str(v): "female" for v in female}, **{str(v): "male" for v in male}}
        clash = [lv for lv, sx in said.items() if coding.get(lv) not in (None, sx)]
        if clash:
            wrong.append(f"`{column}`'s level `{clash[0]}` is {coding.get(clash[0])}, as "
                         f"recorded, and the rule reads it as {said[clash[0]]}")

    for rule in decision.rules:
        columns: list[str] = []
        if getattr(rule, "kind", "range") == "goldberg":
            for attr, measure, expected in _BODY_EXPECTED:
                column = getattr(rule, attr, None)
                if not column:
                    continue
                columns.append(column)
                want = expected or (rule.weight_unit if measure == "weight" else rule.height_unit)
                found = body_unit_reading(column, measure, state,
                                          values(column) if measure == "height" else None)
                if found.settled and found.value != want:
                    if found.value in BODY_CONVERSIONS.get(measure, {}) and measure == "weight":
                        exits.append({"label": f"Read `{column}` in {found.value}, as recorded",
                                      "decision": SetExclusions(rules=[
                                          r.model_copy(update={"weight_unit": found.value})
                                          if r is rule else r for r in decision.rules])})
                    wrong.append(f"`{column}` is in {found.value}, and the rule reads {want}")
                elif not found.settled and column not in units:
                    units.append(column)
                    choices = ([want] + [u for u in BODY_CONVERSIONS.get(measure, {})
                                         if u != want and (measure != "height" or u != "in")])
                    exits.extend(confirm_exit("unit", column, u, f"`{column}` is in {u}")
                                 for u in choices)
            columns.append(rule.sex)
            sex_levels_settled(rule.sex, rule.female, rule.male)
        elif getattr(rule, "by", None) is not None:
            columns.append(rule.by.column)
            if _sex_named(rule.by.column):
                # Which level takes women's range and which men's rests on the sex coding.
                sex_levels_settled(rule.by.column, [], [])
        if state is not None and state.roles:
            roles_waiting += [c for c in unsettled(state, columns) if c not in roles_waiting]
    if not (units or codings or wrong or roles_waiting):
        return
    kept = [r for r in decision.rules if getattr(r, "kind", "range") != "goldberg"
            and not (getattr(r, "by", None) is not None
                     and (r.by.column in roles_waiting or r.by.column in codings))]
    leave = {"label": "Keep only the screens that read settled columns",
             "decision": SetExclusions(rules=kept)}
    if wrong:
        raise Refusal("reading_unsettled",
                      f"The screen cannot read these columns as the rule says: {'; '.join(wrong)}.",
                      exits=[*exits, leave])
    said = []
    if units:
        one = len(units) == 1
        said.append(f"{listing(units)}'s {'unit is' if one else 'units are'} not recorded: a "
                    f"header or a median tells kg from lb, or years from months, no better than a "
                    f"heavy cohort from a light one")
    if codings:
        one = len(codings) == 1
        said.append(f"which level of {listing(codings)} is female is not confirmed: studies code "
                    f"1 and 2 either way")
    if roles_waiting:
        one = len(roles_waiting) == 1
        said.append(f"{listing(roles_waiting)} {'was' if one else 'were'} proposed below high "
                    f"confidence and not confirmed on {'its' if one else 'their'} own")
    raise Refusal("reading_unsettled",
                  f"The screen reads {'; and '.join(said)}. Confirm each first.",
                  exits=[*exits, *confirm_exits(state, roles_waiting), leave])


def _answers_keep_settled_readings(decision: Any, ctx: Any) -> None:
    """BLUEPRINT §14.1 (and §13's "invalidates"): undoing a reading's confirmation while a recorded
    answer reads it (a screen's body measure, the order a unit's records were combined in, a
    predictor's code-or-amount reading) would leave a number resting on a reading nobody settled;
    refused until that answer changes, never kept silently."""
    from turbotab.core.readings import (
        Unsettled, body_unit_reading, predictors_or_ask, time_column_reading,
    )

    now = _state(ctx)
    after = state_after(decision, ctx)
    if now is None or after is None:
        return
    readers: list[tuple[str, str]] = []
    rules = [*(now.exclusions or []), *(r for a in (now.sensitivity or []) for r in a.rules)]
    for rule in rules:
        if getattr(rule, "kind", "range") != "goldberg":
            continue
        for attr, measure, _ in _BODY_EXPECTED:
            column = getattr(rule, attr, None)
            if column and body_unit_reading(column, measure, now).settled \
                    and not body_unit_reading(column, measure, after).settled:
                readers.append(("exclusion rules", column))
    agg = now.aggregation
    if agg is not None:
        from turbotab.core.stages.working import needs_order

        if needs_order(agg.method, agg.outcome, agg.columns):
            before_t, after_t = time_column_reading(now, None), time_column_reading(after, None)
            if before_t is not None and before_t.settled and (after_t is None or not after_t.settled):
                readers.append(("combining answer", before_t.column))
    if now.models:
        info = _ctx(ctx, "column_info")
        store = _store_of(ctx)
        try:
            predictors_or_ask(now, info, drop=left_out(now), store=store)
        except Unsettled:
            pass  # already waiting: the fit asks for it
        else:
            try:
                predictors_or_ask(after, info, drop=left_out(after), store=store)
            except Unsettled as waiting:
                column = (waiting.exits[0].get("decision") or {}).get("column", "a predictor")
                readers.append(("models answer", str(column)))
    if not readers:
        return
    what, column = readers[0]
    raise Refusal(
        "role_unconfirmed",
        f"The recorded {what} reads `{column}`, which would then rest on a reading nobody "
        f"settled. Change the {what} first, or keep the confirmation.",
        exits=[{"label": "Keep the answers as they are", "decision": None}])


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


# ── an intake screen never removes most rows (audit IN-07) ───────────────────
# "Refuse a screen that would remove more than half the rows, with a units exit": a kJ energy
# column read as kcal lost 764–989 of 800–1,000 rows to the 500–5,000 kcal screen. The exits are
# the same rule read in kJ, or as a weekly total, when that reading keeps most rows.

_UNIT_READINGS = (("in kJ", 4.184), ("as a weekly total", 7.0))


def _scaled_rule(rule: Any, factor: float, words: str) -> Any:
    def scale(v: float | None) -> float | None:
        return None if v is None else round(float(v) * factor, 1)

    update: dict[str, Any] = {"low": scale(rule.low), "high": scale(rule.high),
                              "reason": f"{rule.reason}, read {words}"}
    if rule.by is not None:
        update["by"] = rule.by.model_copy(update={"ranges": {
            k: (scale(lo), scale(hi)) for k, (lo, hi) in rule.by.ranges.items()}})
    return rule.model_copy(update=update)


def _screens_keep_most_rows(decision: SetExclusions, ctx: Any) -> None:
    import pandas as pd

    from turbotab.core.recognizers import reads_as_total_energy
    from turbotab.core.stages.proposals import MOST_ROWS, rule_excludes

    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data to check: nothing is refused for it
        store = None
    if store is None:
        return
    state = _state(ctx)
    roles = (getattr(state, "roles", None) or {}) if state is not None else {}
    names = set(store.columns)
    for i, rule in enumerate(decision.rules):
        column = rule.column
        if column not in names or not (roles.get(column) == "energy"
                                       or reads_as_total_energy(column)):
            continue
        reads = [c for c in as_rule(rule).reads() if c in names]
        if len(reads) < len(as_rule(rule).reads()):
            continue
        frame = store.materialize(reads)
        present = pd.to_numeric(frame[column], errors="coerce").notna()
        n = int(present.sum())
        removed = int((rule_excludes(frame, rule) & present).sum())
        if not n or removed <= MOST_ROWS * n:
            continue
        exits: list[dict[str, Any]] = []
        if not isinstance(rule, GoldbergRule):
            for words, factor in _UNIT_READINGS:
                scaled = _scaled_rule(rule, factor, words)
                if int((rule_excludes(frame, scaled) & present).sum()) <= MOST_ROWS * n:
                    rules = list(decision.rules)
                    rules[i] = scaled
                    exits.append({"label": f"Read `{column}` {words}",
                                  "decision": SetExclusions(rules=rules)})
        exits.append({"label": f"Drop the rule on `{column}`",
                      "decision": _rule_without(decision, i)})
        raise Refusal(
            "screen_removes_most_rows",
            f"The rule on `{column}` would remove {removed:,} of {n:,} rows ({removed / n:.0%}). An "
            f"intake screen removes the implausible few; one that removes most rows says the bounds "
            f"are not in `{column}`'s unit. Check the unit first.",
            exits=exits)


# ── a screen on total energy waits for its unit (audit WP13 gate repair) ───────
# The leash (BLUEPRINT §11.3): a unit read from a median-magnitude prior, or from nothing, is a
# proposal; a screen's bounds are numbers in a unit, so no screen on total energy is recorded
# until that unit is (a toddler's 4,040 kJ read as kcal lost 15% of healthy children to the
# 500–5,000 kcal screen, offered unrefused). The exits record the unit.


def _names_a_macro_total(column: str) -> bool:
    from turbotab.core.recognizers import AmbiguousNutrient, read_nutrient

    try:
        reading = read_nutrient(column)
    except AmbiguousNutrient:
        return False
    return reading is not None and reading.part is None and reading.macro in (
        "protein", "carbohydrate", "fat", "alcohol")


def _screens_wait_for_the_unit(decision: SetExclusions, ctx: Any) -> None:
    from turbotab.core.recognizers import macro_totals, reads_as_total_energy
    from turbotab.core.stages.proposals import energy_unit_reading, recorded_energy_unit

    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data to check: nothing is refused for it
        store = None
    if store is None:
        return
    state = _state(ctx)
    if state is not None and "dietary" not in (getattr(state, "lens", None) or []):
        return
    roles = (getattr(state, "roles", None) or {}) if state is not None else {}
    names = set(store.columns)
    for rule in decision.rules:
        column = rule.column
        if column not in names or not (roles.get(column) == "energy"
                                       or reads_as_total_energy(column)):
            continue
        totals = [c for c in names if c != column and _names_a_macro_total(c)]
        frame = store.materialize([column, *totals])
        macros = macro_totals(frame, exclude=[column])
        reading = energy_unit_reading(frame[[column, *macros.values()]], column,
                                      recorded_energy_unit(state, column))
        if reading.get("confirmed", True):
            continue
        proposed = str(reading["unit"])
        other = "kcal" if proposed == "kj" else "kj"
        word = {"kj": "kJ", "kcal": "kcal"}
        recorded_days = (recorded_energy_unit(state, column) or ColumnUnitSpec(days=None)).days
        if reading.get("basis") == "atwater_ambiguous":
            # BLUEPRINT §14.3 (amendment after the fifth gate): the reconstruction ratio fits more
            # than one reading (4.00: kJ, or a 4-day kcal total beside daily-mean
            # macronutrients); the unit and the days are asked in one answer, each fit offered.
            pairs = [(str(u), int(d)) for u, d in reading.get("candidates") or []]
            raise Refusal(
                "energy_unit_unconfirmed",
                f"The rule on `{column}` reads its bounds in its unit and over the days each value "
                f"spans, and its values fit more than one reading: "
                f"{_unit_reading_words(reading)}. Record which first; then the screen's bounds "
                f"are read in it.",
                exits=[*({"label": f"`{column}` is "
                                   + (f"a total over {d} days, in {word[u]}" if d > 1
                                      else f"one day's intake, in {word[u]}"),
                          "decision": SetColumnUnit(column=column, unit=u, days=d)}
                         for u, d in pairs),
                       {"label": "Another unit or day count (record it with the column's unit)",
                        "decision": None}])
        if reading.get("basis") == "not_energy":
            raise Refusal(
                "energy_unit_unconfirmed",
                f"`{column}` is recorded in {reading.get('recorded_unit')}, which is no unit of "
                f"energy, so the rule's bounds cannot be read in it. Record kcal or kJ first.",
                exits=[{"label": f"`{column}` is a day's energy in {word[u]}",
                        "decision": SetColumnUnit(column=column, unit=u,
                                                  days=int(recorded_days or 1))}
                       for u in ("kcal", "kj")])
        spanned = reading.get("days_in_name")
        if reading.get("basis") == "days" and spanned:
            # BLUEPRINT §14 rule 3: a day count in the name is asked, never read as one day.
            raise Refusal(
                "energy_unit_unconfirmed",
                f"The rule on `{column}` reads its bounds over the days each value spans, and the "
                f"name's `{spanned}` days can mean a total over them or a mean of them. Record "
                f"which first; then the screen's bounds are read in it.",
                exits=[{"label": f"`{column}` is a total over {spanned} days, in {word[proposed]}",
                        "decision": SetColumnUnit(column=column, unit=proposed, days=int(spanned))},
                       {"label": f"`{column}` is a day's energy (a mean over {spanned} days), in "
                                 f"{word[proposed]}",
                        "decision": SetColumnUnit(column=column, unit=proposed)},
                       {"label": f"`{column}` is a total over {spanned} days, in {word[other]}",
                        "decision": SetColumnUnit(column=column, unit=other, days=int(spanned))}])
        if reading.get("days_unsettled") and reading.get("basis") in ("name", "atwater",
                                                                       "decision"):
            # BLUEPRINT §14.1: the unit is settled, the days are not (the gate's
            # ``energy_kcal_day1_day2``: the Atwater identity held for 2-day totals).
            evidence = (reading.get("days_reading") or {}).get("evidence") or "nothing settles it"
            spans = [d for d in reading.get("days_candidates") or [1, 2] if d > 1] or [2]
            named = (f" (only its name says {word[proposed]})" if reading.get("basis") == "name"
                     else "")
            raise Refusal(
                "energy_unit_unconfirmed",
                f"The rule on `{column}` reads its bounds in a day's {word[proposed]}{named}, but "
                f"how many days each value spans is not settled: {evidence}. Record it first; then "
                f"the screen's bounds are read in it.",
                exits=[{"label": f"`{column}` is one day's intake, in {word[proposed]}",
                        "decision": SetColumnUnit(column=column, unit=proposed)},
                       *({"label": f"`{column}` is a total over {d} days, in {word[proposed]}",
                          "decision": SetColumnUnit(column=column, unit=proposed, days=int(d))}
                         for d in spans[:3])])
        raise Refusal(
            "energy_unit_unconfirmed",
            f"The rule on `{column}` reads its bounds in {word[proposed]}, but "
            + ("only the column's median says so" if reading.get("basis") == "magnitude"
               else "nothing says what unit the column is in")
            + ". Record the unit first; then the screen's bounds are read in it.",
            exits=([{"label": f"`{column}` is a total over {recorded_days} days (as recorded), "
                              f"in {word[u]}",
                     "decision": SetColumnUnit(column=column, unit=u, days=int(recorded_days))}
                    for u in (proposed, other)] if recorded_days and int(recorded_days) > 1 else
                   [{"label": f"`{column}` is a day's energy in {word[proposed]}",
                     "decision": SetColumnUnit(column=column, unit=proposed)},
                    {"label": f"`{column}` is a day's energy in {word[other]}",
                     "decision": SetColumnUnit(column=column, unit=other)},
                    *([{"label": f"`{column}` is a week's total, in {word[proposed]}",
                        "decision": SetColumnUnit(column=column, unit=proposed, days=7)}]
                      if _names_a_week(column) else []),
                    {"label": f"`{column}` is a total over two days, in {word[proposed]}",
                     "decision": SetColumnUnit(column=column, unit=proposed, days=2)}]))


def _unit_reading_words(reading: Mapping[str, Any]) -> str:
    """An ambiguous energy reading's evidence, as the proposals' sentence states it."""
    return str(reading.get("sentence") or "").split("; record")[0].rstrip(".")


def _names_a_week(column: str) -> bool:
    from turbotab.core.recognizers import tokens

    return bool(set(tokens(column)) & {"week", "weeks", "weekly", "wk", "wks"})


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
    if not decision.nutrients:
        # BLUEPRINT §14 rule 2: the card pre-fills only settled nutrients; when every candidate is
        # waiting for its own confirmation, say which, one exit each.
        from turbotab.core.readings import confirm_exits, unsettled, unsettled_message
        from turbotab.core.stages.proposals import energy_bearing

        waiting = unsettled(state, [c for c, r in roles.items()
                                    if r == "exposure" and energy_bearing(c)])
        if waiting:
            raise Refusal("role_unconfirmed", unsettled_message(waiting, "the energy adjustment"),
                          exits=[*confirm_exits(state, waiting),
                                 {"label": "Do not adjust for energy",
                                  "decision": with_(method="none", energy_column=None,
                                                    nutrients=[], strata=None,
                                                    log_transform=False)}])
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
    nested = effective_nesting(ctx, frame, nutrients)
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


def _substitution_reads_settled_readings(decision: SetSubstitution, ctx: Any) -> None:
    """BLUEPRINT §14.1 (the readings ledger): a substitution moves energy between two columns by
    three readings of each, every one settled before a curve reads it: its exposure role, its kcal
    per unit (an Atwater factor needs the column in grams: stated by the name or its codebook,
    agreed by the Atwater identity, or confirmed), and, on the share-of-energy scale, that its
    values are percentages of energy (0 to 100, not fractions)."""
    from turbotab.core.methods.percent_energy import check_percent_values, is_percent_of_energy
    from turbotab.core.readings import (
        confirm_exits, confirmation, factor_exits, factor_verdicts, kcal_per_unit, nested_exits,
        unsettled,
    )

    state = _state(ctx)
    if state is None or not state.roles:
        return
    moved = [decision.donor, decision.recipient]
    waiting = unsettled(state, moved)
    if waiting:
        raise Refusal("role_unconfirmed",
                      f"{_and(waiting)} {'was' if len(waiting) == 1 else 'were'} proposed below "
                      f"high confidence and recorded with the other roles; a substitution moves "
                      f"energy through {'it' if len(waiting) == 1 else 'them'} only once confirmed "
                      f"on {'its' if len(waiting) == 1 else 'their'} own.",
                      exits=[*confirm_exits(state, waiting),
                             {"label": "Choose two confirmed nutrients", "decision": None}])
    shares = [c for c in moved if is_percent_of_energy(c)]
    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data: only stated or confirmed units settle
        store = None
    # BLUEPRINT §14.3: each kcal per unit is derived from the recorded unit (g, kg, kcal, kJ,
    # alcohol's standard drinks), or from grams the registry's Atwater test reads (never for
    # alcohol or a minor source: the sixth gate); a recorded share of energy or another unit is
    # refused with its route, and an unrecorded one is asked, every unit it may be in offered.
    try:
        verdicts = factor_verdicts(state, store, [c for c in moved if c not in shares])
    except Exception:  # noqa: BLE001 - no values to read: only a recorded unit settles
        verdicts = {}
    readings = [kcal_per_unit(state, c, verdict=verdicts.get(c))
                for c in moved if c not in shares]
    refused = [r for r in readings if not r.settled and confirmation(state, "unit", r.column)]
    if refused:
        r = refused[0]
        exits = [e for e in factor_exits(r.column) if e.get("decision")]
        if r.route:
            exits.append({"label": r.route, "decision": None})
        exits.append({"label": "Choose two nutrients whose units carry energy", "decision": None})
        raise Refusal("reading_unsettled",
                      f"{r.why}, so moving energy through it has no kcal per unit to read. "
                      f"Record its unit as an amount, or move energy another way.", exits=exits)
    factors = [r.column for r in readings if not r.settled]
    if factors:
        raise Refusal(
            "reading_unsettled",
            f"{_and(factors)} {'reads' if len(factors) == 1 else 'read'} as an energy-bearing "
            f"nutrient, but {'its' if len(factors) == 1 else 'their'} unit is not recorded, so the "
            f"kcal each unit carries is a guess (a name never says it). Record the unit first.",
            exits=[*(e for c in factors for e in factor_exits(c)),
                   {"label": "Choose two nutrients whose units are stated", "decision": None}])
    # A total moves with its parts (and a part with its total) by the nesting reading: the values
    # agree (a part never exceeds its total), which is necessary, not sufficient; confirmed once,
    # naming the total the guess found, another column, or none (``readings.nesting``: the user's
    # answer stands over the guess for every consumer that moves energy).
    guessed = nesting_of(ctx)
    pairs = [(child, parent) for child, parent in guessed.items()
             if child in moved or parent in moved]
    open_pairs = [(c, p) for c, p in pairs if confirmation(state, "nested_in", c) is None]
    if open_pairs:
        raise Refusal(
            "reading_unsettled",
            "Moving energy here moves parts with their totals: "
            + "; ".join(f"`{c}` reads as part of `{p}`" for c, p in open_pairs)
            + ". Their values agree (a part never exceeds its total), which a part must, but does "
              "not prove it. Confirm each before the curve moves them together.",
            exits=[*(e for c, p in open_pairs for e in nested_exits(c, p) if e.get("decision")),
                   {"label": "Choose two nutrients that are not parts of each other",
                    "decision": None}])
    if decision.scale == "percent_energy" and shares:
        if store is not None:
            present = [c for c in shares if c in set(store.columns)]
            try:
                check_percent_values(store.materialize(present), present)
            except ValueError as wrong:
                raise Refusal("reading_unsettled",
                              f"{wrong} A share of energy moves in percentage points, so its "
                              f"values are read before any curve.",
                              exits=[{"label": "Choose two columns that hold percentages of energy",
                                      "decision": None}]) from None


def effective_nesting(ctx: Any, frame: Any = None, columns: Any = None) -> dict[str, str]:
    """Child column -> the total it is part of, as the ledger holds it (``readings.nesting``): the
    design's (else the roles stage's, else ``frame``'s names and values) guess, each column's own
    confirmation standing over it."""
    from turbotab.core.readings import nesting

    found = nesting_of(ctx)
    if found or frame is None:
        return nesting(_state(ctx), found, columns=columns)
    return nesting(_state(ctx), frame=frame, columns=columns)


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
    nested = effective_nesting(ctx)
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
    from turbotab.core.methods.nesting import compositions
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

    nested = effective_nesting(ctx, frame[present], present)
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
        # BLUEPRINT §14.1: each column the names read as an energy source is added on its own,
        # one reading per answer, never several at once.
        for c in addable[:4]:
            exits.append({"label": f"Add `{c}` to the model as an exposure",
                          "decision": SetRoles(roles={**roles, c: "exposure"})})
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
    inference = getattr(state, "purpose", None) == "inference"
    # MS7 (MODELING_SEQUENCE §4): under inference a reason does not keep a fill that reads values
    # below a limit as missing at random; under prediction it does, ranked lower.
    if not censored or decision.strategy == "complete_case" or (decision.reason and not inference):
        return
    if decision.below_detection in ("half_minimum", "censoring_aware", "qrilc"):
        return
    how = ("multiple imputation reads them as missing at random" if decision.strategy ==
           "multiple_imputation" else "the median fill places them in the middle of the distribution")
    aware = ("a censored-normal draw below the limit, given the outcome" if inference else
             "the expected value below the limit, fit in each training fold")
    raise Refusal(
        "median_below_detection",
        f"{_and(censored[:6])}{' and others' if len(censored) > 6 else ''} hold values below a "
        f"detection limit: each blank is known to be small, but {how}. Choose how non-detections "
        f"are filled" + ("; under inference they are never filled as missing at random." if inference
                         else ", or keep this fill with your reason."),
        exits=[{"label": f"Censoring-aware: {aware}",
                "decision": _missing_base(decision, below_detection="censoring_aware",
                                          censored_columns=censored)},
               {"label": "Half the smallest detected value (customary)",
                "decision": _missing_base(decision, below_detection="half_minimum",
                                          censored_columns=censored)},
               *([] if inference else [{"label": "Keep this fill: give your reason",
                                        "decision": None}])])


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


def _unit_names_the_outcome(decision: SetOutcomeUnit, ctx: Any) -> None:
    """The unit answers for the outcome: a recorded unit is what every outcome quantity is then
    stated in (audit IN-05)."""
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is None:
        raise Refusal("no_target", "Choose the outcome first; the unit describes it.",
                      exits=[{"label": "Choose the outcome", "decision": None}])
    if target is not _UNKNOWN and decision.column != target:
        raise Refusal(
            "not_the_target",
            f"The outcome is `{target}`, not `{decision.column}`; the unit answers for the outcome.",
            exits=[{"label": f"Record the unit of `{target}`", "decision": None}])


def _column_unit_fits(decision: SetColumnUnit, ctx: Any) -> None:
    """The unit names a numeric column of this table, and a span of days belongs to energy."""
    columns = _columns_of(ctx)
    if columns is not None and decision.column not in columns:
        raise Refusal("unknown_column", f"This dataset has no column named `{decision.column}`.",
                      exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    info = _info(ctx, decision.column)
    if info is not None and info.get("dtype") not in NUMERIC_DTYPES:
        raise Refusal("not_numeric",
                      f"`{decision.column}` does not hold numbers, so it has no unit to record.",
                      exits=[{"label": "Choose a numeric column", "decision": None}])
    if decision.days != 1 and decision.unit not in ENERGY_UNITS:
        raise Refusal(
            "days_need_energy",
            f"A span of days says how many days' intake a value totals; an age in "
            f"{decision.unit} has none.",
            exits=[{"label": f"Record `{decision.column}` in {decision.unit}",
                    "decision": SetColumnUnit(column=decision.column, unit=decision.unit)}])


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
    from turbotab.core.readings import confirmed_codes

    declared = set(confirmed_codes(state) if state is not None else [])
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
            _screens_wait_for_the_unit(probe, ctx)
            _screens_read_a_settled_energy_column(probe, ctx)
            _screens_read_settled_body_measures(probe, ctx)
        except Refusal as refused:
            # A unit to record or a role to confirm is answered where it is asked (its own exits),
            # or the analysis left out.
            kept = ([e for e in refused.exits if (e.get("decision") or {}).get("kind")
                     in ("set_column_unit", "confirm_role", "confirm_reading")]
                    if refused.code in ("energy_unit_unconfirmed", "role_unconfirmed",
                                        "reading_unsettled") else [])
            raise Refusal(refused.code, f"In “{analysis.label}”: {refused.message}",
                          exits=[*kept, {"label": f"Leave out “{analysis.label}”",
                                         "decision": rest}]) from None


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
register_validator("set_outcome_unit", _unit_names_the_outcome)
register_validator("set_column_unit", _column_unit_fits)
register_validator("set_exposure_form", _form_fits_the_column)
register_validator("set_follow_up", _follow_up_belongs_to_the_outcome)
register_validator("set_roles", _roles_name_real_columns)
register_completion("set_roles", _roles_record_what_rode_along)
register_validator("confirm_role", _confirmed_role_names_a_column)
register_validator("confirm_reading", _reading_names_a_column)
register_validator("confirm_readings", _readings_name_columns)
register_validator("set_exclusions", _screens_read_a_settled_energy_column)
register_validator("set_feature_table", _feature_table_names_the_files_columns)
register_validator("set_categorical", _categorical_names_predictors)
register_validator("set_missing", _left_out_columns_are_predictors)
register_validator("set_missing", _missing_fits_the_purpose)
register_validator("set_missing", _non_detections_are_not_filled_by_the_median)
register_validator("set_substitution", _substitution_moves_between_separate_nutrients)
register_validator("set_exclusions", _exclusions_are_ranges_on_numbers)
register_validator("set_exclusions", _screens_wait_for_the_unit)
register_validator("set_exclusions", _screens_read_settled_body_measures)
register_validator("set_exclusions", _screens_keep_most_rows)
# first: a rule on the outcome is refused for that reason, whatever else is wrong with its bounds
register_validator("set_exclusions", _exclusions_leave_the_outcome_alone, first=True)
register_validator("set_target", _new_outcome_has_no_rule)
register_validator("revert", _revert_leaves_no_rule_on_the_outcome)
register_validator("revert", _answers_keep_settled_roles)
register_validator("revert", _answers_keep_settled_readings)
register_validator("set_roles", _answers_keep_settled_roles)
register_validator("set_energy_adjustment", _energy_adjustment_fits_the_roles)
register_validator("set_energy_adjustment", _energy_reads_settled_roles)
register_validator("select_models", _models_can_fit_the_task)
register_validator("select_models", _models_read_settled_readings)
register_validator("set_substitution", _substitution_swaps_energy)
register_validator("set_substitution", _substitution_has_every_energy_source)
register_validator("set_substitution", _substitution_reads_settled_readings)


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


def _confirmed_role_is_the_role(slots: dict[str, Any], slot: str, column: str, role: Any) -> None:
    """BLUEPRINT §14.3, every confirmation is honored: a role confirmed for a column the roles
    answer recorded (``confirm_reading`` / ``confirm_readings`` / ``confirm_role``) is that column's
    role from then on, whatever the answer recorded before; a confirmation of the same role only
    settles it. A later roles answer is the user's newer word and stands over it (the column is
    then settled only if it was recorded as the user's own, or confirmed as the same role)."""
    if slot != "role_confirmations":
        return
    roles = slots.get("roles")
    if not roles or column not in roles or roles[column] == role:
        return
    slots["roles"] = {**roles, column: role}


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
        written = _ENTRIES.get(decision.kind)
        if written is not None:
            for slot, entry, value in written(decision):
                entries = dict(slots.get(slot) or {})
                entries[entry] = value(entries.get(entry)) if callable(value) else value
                slots[slot] = entries
                _confirmed_role_is_the_role(slots, slot, entry, entries[entry])
            continue
        keyed = _KEYS.get(decision.kind)
        if keyed is not None:
            slot = _SLOT_FOR[decision.kind](decision) if decision.kind in _SLOT_FOR \
                else SLOTS[decision.kind]
            entries = dict(slots.get(slot) or {})
            entries[keyed(decision)] = _SLOT_VALUE[decision.kind](decision)
            slots[slot] = entries
            _confirmed_role_is_the_role(slots, slot, keyed(decision), entries[keyed(decision)])
        else:
            slots[SLOTS[decision.kind]] = _SLOT_VALUE[decision.kind](decision)
            for extra, fn in _ALSO.get(decision.kind, {}).items():
                slots[extra] = fn(decision)
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


# ── what had been seen when a decision was made (audit WP16) ────────────────

def opened_ever(records: Sequence[DecisionRecord]) -> bool:
    """Whether held-out rows were ever opened in this log: for any outcome, before any re-seal.
    An opening is never undone, so its scores were seen from then on."""
    try:
        cancelled = reverted(records)
    except Refusal:
        cancelled = {}
    return any(r.decision.kind == "open_seal" and r.id not in cancelled for r in records)


SEEN_ESTIMATES = "After the estimates were seen"
SEEN_HELD_OUT = "After the held-out rows were opened"
SEEN_BOTH = "After the estimates were seen and the held-out rows were opened"


def disclose(text: str | None, *, post_seal: bool, after_estimates: bool) -> str | None:
    """A record's sentence, led by what had been seen when it was made: the held-out scores
    (``post_seal``) or the inference estimates (``after_estimates``; Gelman & Loken 2013's forking
    paths). Never applied twice."""
    if not text or not (post_seal or after_estimates):
        return text
    lead = SEEN_BOTH if post_seal and after_estimates else (
        SEEN_ESTIMATES if after_estimates else SEEN_HELD_OUT)
    if text.startswith((SEEN_BOTH, SEEN_ESTIMATES, SEEN_HELD_OUT)):
        return text
    body = text[0].lower() + text[1:] if text[:1].isupper() and not text[1:2].isupper() else text
    return f"{lead}, {body}"


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
        The record is marked, and its sentence led, by what had been seen when it was made: held-out
        scores (``post_seal``) and the inference estimates (``after_estimates``; :func:`disclose`).
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
                post_seal, after_estimates = opened_ever(existing), bool(before.plan_locked)
                if isinstance(text, str):
                    text = disclose(text, post_seal=post_seal, after_estimates=after_estimates)
                record = DecisionRecord(
                    id=uuid.uuid4().hex,
                    seq=max((r.seq for r in existing), default=0) + 1,
                    at=datetime.now(timezone.utc),
                    note=note,
                    sentence=text if isinstance(text, str) and text.strip() else None,
                    post_seal=post_seal,
                    after_estimates=after_estimates,
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
# WP17's refusals (the follow-up, the clusters, the estimand and the adjustment answers).
from turbotab.core import estimand as _estimand  # noqa: E402,F401
# The structural questions that ask where evidence is thin (audit §5 WP18): the lens that is none
# of the five, predictors summarized after the outcome, the outcome's order and scale, reference
# rows, and imputed copies.
from turbotab.core import structural as _structural  # noqa: E402,F401
# Joins and codebook import (DATAIN): their validators, completions and sentences.
from turbotab.core import assembly as _assembly  # noqa: E402,F401
from turbotab.core import codebook as _codebook  # noqa: E402,F401
# The scales question's refusals (``set_scales``; MS8) live with its routing and contract.
from turbotab.core import scales as _scales  # noqa: E402,F401
# The NCI usual-intake method's refusals, contract and sentence (``set_usual_intake``).
from turbotab.core import usual_intake as _usual_intake  # noqa: E402,F401

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
    EligibilityRule,
    EnergyMethod,
    ExclusionRule,
    FindingDisposition,
    GoldbergRule,
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
from turbotab.core.custom_sound import LabeledQuestion  # noqa: F401 - WP17's two labels
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
    # Infinite values (a ratio over zero) are counted here and left out of every statistic above.
    n_infinite: int


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


class ReadFromData(Model):
    """One reading the values settled with no question asked (BLUEPRINT §14.3, amendment:
    settlement is visible), with its evidence and the answers that change it."""

    kind: str
    column: str
    value: Any
    words: str  # what was read, as a predicate of the column ("is an amount (one slope)")
    evidence: str
    change: list[Exit]


# ── files to join and codebooks to import (DATAIN, V2 definition of done §1) ─────


class AddFile(BaseModel):
    """Add a file on this machine to the project, to join to its table (local mode only)."""

    model_config = ConfigDict(extra="forbid")

    path: str


class AddedFile(Model):
    id: str
    name: str
    source_kind: SourceKind
    fingerprint: str
    n_rows: int
    n_cols: int
    columns: list[str]
    warnings: list[str]
    joined: bool = False


class JoinPreviewRequest(BaseModel):
    """The join to count: the added file, the identifier (``right_on`` when the file names it
    otherwise), and which rows the joined table keeps."""

    model_config = ConfigDict(extra="forbid")

    file: str
    on: str
    right_on: str | None = None
    how: Literal["left", "inner"] = "left"


class JoinSide(Model):
    name: str
    rows: int
    keys: int
    blank_keys: int
    repeats: bool
    max_repeat: int
    example: Any
    example_count: int


class JoinRefusal(Model):
    code: str
    message: str
    exits: list[Exit]


class JoinPreview(Model):
    """What a join would do before it is committed: each side's rows and identifier values, the
    relation (one-to-one, one-to-many, many-to-one; many-to-many is refused), the rows with no
    partner on each side, the joined table's rows, its new and renamed columns, and the sentence
    the join would record."""

    file: str
    on: str
    right_on: str
    how: Literal["left", "inner"]
    table: JoinSide
    file_side: JoinSide
    relation: str
    matched_keys: int
    table_unmatched: int
    file_unmatched: int
    rows: int
    added_columns: list[str]
    renamed: dict[str, str]
    refusal: JoinRefusal | None
    sentence: str


class CodebookRequest(BaseModel):
    """A codebook to read: a file on this machine (local mode only), or the variable labels the
    table's own SAS transport files carry (``labels``)."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    labels: bool = False


class CodebookSettlement(Model):
    reading: str
    column: str
    value: str
    field: str


class CodebookAsk(Model):
    column: str
    field: str
    says: str
    values: str
    exits: list[Exit]


class CodebookPreview(Model):
    """What importing a codebook would do (``import_codebook`` with this ``id`` records it): the
    readings its structured fields settle, its documented units, how many labels it gives, the
    fields the values contradict (asked, never applied), the readings the user answered
    otherwise (kept), and the sentence it would record."""

    id: str
    name: str
    form: Literal["table", "nhanes", "xpt"]
    n_entries: int
    n_matched: int
    unmatched: list[str]
    n_unmatched: int
    settles: list[CodebookSettlement]
    units: dict[str, str]
    labels: int
    asked: list[CodebookAsk]
    kept: list[str]
    sentence: str


class ReadingsCard(Model):
    """The readings card's "read from your data": every reading the values settled, never a
    required tap, and the methods record's line for them."""

    read_from_data: list[ReadFromData]
    sentence: str


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


class FollowUpCandidate(Model):
    """A column that reads as how long each row was followed (``turbotab/core/estimand.py``)."""

    column: str
    min: float
    max: float
    varies: bool  # its values span more than 5% of its largest: follow-up ended at different times


class OrderQuestion(Model):
    """"Are these levels ordered?" (``turbotab.core.structural.order_question``)."""

    question: str
    levels: list[str]
    numeric: bool
    proposed_order: list[str] | None
    evidence: str
    options: list[Exit]  # ordered (ordinal) or unordered (multiclass)
    order_options: list[Exit]  # the proposed order, lowest first (``set_outcome_order``)


class ScaleQuestion(Model):
    """The scale a positive, markedly skewed outcome is analyzed on
    (``turbotab.core.structural.scale_question``)."""

    question: str
    skewness: float  # the outcome's (adjusted Fisher–Pearson); above 2 (West et al.)
    log_skewness: float  # its natural log's; within ±2, or a log would not answer it
    n: int
    min: float
    median: float
    max: float
    evidence: str
    log_column: str
    options: list[Exit]  # the original scale, or the log scale (``set_outcome_scale``)


class TargetInfo(Model):
    """The ``target_info`` artifact."""

    column: str
    task: Task
    detected_task: Task
    confidence: Confidence
    reason: str
    histogram: Histogram | None
    classes: list[ValueCount] | None
    # M2 (M2_CONTRACT §6): the outcome's unit (mg/dL…) as sentences may state it: recorded by
    # ``set_outcome_unit`` ("decision") or spelled out by the name's suffix ("name"); else None.
    # WP13 (audit IN-05): the clinical pack's reading is only proposed (``proposed_unit``, with its
    # ``unit_candidates``) for the user's decision; it is never stated.
    unit: str | None = None
    unit_source: Literal["name", "decision"] | None = None
    proposed_unit: str | None = None
    unit_candidates: list[str] = []
    # WP17 (audit RO-03): columns that read as a follow-up time, the follow-up question's options.
    follow_up: list[FollowUpCandidate] = []
    # WP18 (audit RO-10): the tasks the explicit answer accepts (one rule for the answer and the
    # skip); "are these levels ordered?" for 3–10 levels; the scale of a positive, markedly skewed
    # outcome. The Router keeps the task question open while either waits (InterviewStep.followup).
    fits: list[Task] = []
    order_question: OrderQuestion | None = None
    scale_question: ScaleQuestion | None = None


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
    # WP13 (audit IN-02): "acquisition" for a batch, plate or run-order column, whose proposed role
    # follows the purpose (a covariate under inference, left out under prediction).
    kind: Literal["acquisition"] | None = None


class Repeats(Model):
    column: str
    n_units: int
    max_rows_per_unit: int


class CategoricalProposal(Model):
    """A predictor whose numbers may be codes for groups (``set_categorical`` declares it)."""

    column: str
    levels: int
    confidence: Literal["high", "medium"]
    reason: str


class RolesArtifact(Model):
    """The ``roles`` artifact: a proposed role for every column but the outcome."""

    columns: list[RoleProposal]
    repeats: Repeats | None
    categorical: list[CategoricalProposal]


class RowComparison(Model):
    """One column on the rows complete cases kept and on those they dropped (audit E14)."""

    column: str
    kept: float
    dropped: float
    smd: float  # the standardized difference, dropped minus kept
    kind: str  # "mean", or "share <level>" for a category
    n_dropped_observed: int


class CompleteCaseLoss(Model):
    """What complete cases cost: the rows dropped beside the rows kept (audit WP7, E14)."""

    n_before: int
    n_kept: int
    n_dropped: int
    share: float
    outcome: RowComparison | None
    columns: list[RowComparison]  # largest standardized difference first


class CohortArtifact(Model):
    """The ``cohort`` artifact (a Bundle's ``data``): the participant flow."""

    steps: list[RowStep]
    n_final: int
    predictors: list[str]
    complete_case_loss: CompleteCaseLoss | None = None


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
    # How the folds were drawn (audit MA-11, A17): "time_ordered" forward-chains by whole unit (each
    # fold scored by models fit on the folds before it); fold stratification is decided on its own.
    fold_scheme: Literal["random", "time_ordered"] = "random"
    folds_stratified: bool = False
    time_ordered_folds: bool = False


class ExclusionProposal(Model):
    """One of the pack's exclusion rules, with the rows it would remove. Never pre-selected."""

    key: str
    rule: EligibilityRule  # a range, or the Goldberg screen (WP12)
    label: str
    affected: int
    evidence: FindingEvidence
    # Audit IN-07: why the screen is refused (it would remove more than half the rows that hold
    # energy, which says the unit is wrong, not the people); None when it may be chosen.
    refused: str | None = None


class MethodVerdict(Model):
    ok: bool
    reason: str


class NotAdjusted(Model):
    """An exposure energy adjustment leaves as it is, and why (it carries no energy, it is a share)."""

    column: str
    reason: str


class EnergyRanking(Model):
    """The energy methods in order of soundness for the declared purpose, applicable ones first
    (audit WP6; BLUEPRINT §12 ruling 2): the all-components model first under inference, the
    energy-keeping models first under prediction. ``line`` names the tension with custom."""

    purpose: Purpose | None
    order: list[EnergyMethod]
    line: str | None


class OutcomeDispute(Model):
    """An energy-related outcome's DISPUTED note for the energy card (audit IN-20; NUTRITION_PACK
    §04: "escalate the mediation/collider warning"): what the outcome's name reads as, the card
    line (also in ``notes``) and its badge."""

    outcome: str
    # "body weight" · "BMI" · "waist size" · "adiposity" · "diabetes" · "hip size" · "body size" ·
    # "child growth"; None when nothing places the outcome (``basis`` "unconfirmed")
    kind: str | None
    note: str
    evidence: FindingEvidence
    # WP13 gate repair: how the outcome was read: its name, its values (it tracks a body-size
    # column), or nothing (the dispute stated as a condition the researcher answers).
    basis: Literal["name", "values", "unconfirmed"] = "name"


class EnergyReading(Model):
    """What the energy-adjustment question can offer on this table (NUTRITION_PACK §04)."""

    energy_column: str | None
    nutrients: list[str]
    strata_candidates: list[str]
    applicability: dict[str, MethodVerdict]
    usual: EnergyMethod | None  # the customary method (the field's), whatever the purpose
    usual_evidence: FindingEvidence | None
    ranking: EnergyRanking | None = None
    r_with_energy: dict[str, float]
    notes: list[str]
    not_adjusted: list[NotAdjusted]
    outcome_dispute: OutcomeDispute | None = None


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


class MissingMethodOption(Model):
    """A way to handle missing values (or values below detection), with its two labels (north star
    5) and its rung for the declared purpose (audit WP7; ``turbotab/core/methods/missing.py``)."""

    key: str
    label: str
    customary: str  # customary in the field, with a source
    sound: str  # sound for the declared purpose, with the reason
    rung: Literal["recommended", "available", "block_and_record", "refused"]
    decision: dict[str, Any] = {}  # the set_missing fields that choose it (below detection: none)


class MissingReading(Model):
    """What the missing-values question can offer on this table (M1_CONTRACT §12.4)."""

    columns: list[MissingColumn]  # most blank first
    leave_out: LeaveOut | None
    # WP7: the methods, soundest first for the declared purpose; and the below-detection fills.
    methods: list[MissingMethodOption] = []
    below_detection: list[MissingMethodOption] = []


class SurveyOption(Model):
    """One answer the survey question offers, and the ``set_survey`` it records (audit §5 WP10)."""

    key: str  # "population:<weight>" or "sample"
    label: str
    consequence: str
    decision: dict[str, Any]


class SurveyProposal(Model):
    """What the survey question offers on this table: the design the column names read as, the
    cycles a cycle column pools, and one option per recognized weight plus "these participants"."""

    weights: list[str]
    strata: list[str]
    psu: list[str]
    cycle: str | None  # a cycle column holding more than one cycle
    cycles: list[str]
    four_year: dict[str, str]  # 2-year weight -> its 1999–2002 four-year weight
    options: list[SurveyOption]


class ExposureFormOption(Model):
    """One form an exposure can enter the model in (WP12a; ``turbotab/core/methods/exposure_form.py``
    ``options``), with north star 5's two labels, in the order of soundness for the purpose."""

    value: Literal["spline", "linear", "quintiles"]
    label: str
    customary: str  # customary in the field, with a source
    sound: str  # sound for the declared purpose, with the reason
    consequence: str


class EnergyUnitReading(Model):
    """The energy column's unit and how it was read (audit IN-07)."""

    unit: Literal["kcal", "kj"]
    # decision: recorded with ``set_column_unit``; name: a suffix or codebook; atwater: the
    # reconstruction from the macronutrients; magnitude: the pack's median-magnitude prior;
    # assumed: nothing said, kcal the reading counts are made in.
    basis: Literal["decision", "name", "atwater", "magnitude", "assumed"]
    sentence: str
    # WP13 gate repair: the days each value totals (a recorded total over several days), and
    # whether the unit is settled; a magnitude or an assumption is a proposal, and every screen
    # on the column is refused until ``set_column_unit`` records it.
    days: int = 1
    confirmed: bool = True


class QuestionLabels(Model):
    """WP17 (north star 5): each question's options with "customary in <field>" and "sound for
    <purpose>", soundest first (``turbotab/core/custom_sound.py``)."""

    missing: LabeledQuestion | None = None
    exclusions: LabeledQuestion | None = None
    energy_adjustment: LabeledQuestion | None = None


class EstimandExposure(Model):
    column: str
    energy_contrast: bool  # an energy-bearing exposure beside total energy: substitution or addition


class EstimandChoice(Model):
    effect: Literal["total", "direct"] | None = None
    contrast: Literal["substitution", "addition"] | None = None
    label: str
    consequence: str


class EstimandMeasure(Model):
    measure: str
    label: str
    fitted: bool  # False: named, and refused with the reason (not fitted for this outcome)
    reason: str
    # ESTIMAND (ruling 9): a difference or a ratio; conditional, marginal, or both (collapsible);
    # and its rank on the card (None: named and refused)
    scale: Literal["difference", "ratio"] = "ratio"
    conditioning: str = "conditional"
    collapsible: bool = False
    rank: int | None = None


class LabeledChoice(Model):
    """An option with north star 5's two labels (as ``custom_sound`` shapes them)."""

    key: str
    label: str
    customary: dict[str, str]  # field, text, source
    sound: dict[str, str]  # purpose, verdict, reason


class MultiplicityQuestion(Model):
    """ESTIMAND (MODELING_SEQUENCE §2): an exposure family's multiplicity method."""

    question: Literal["multiplicity"]
    options: list[LabeledChoice]
    customary_first: str
    tension: str | None
    n_tests: int


class EstimandFamily(Model):
    """Every exposure reported in turn (the feature-wise family), with its multiplicity method."""

    n: int
    energy_contrast: bool
    measures: list[EstimandMeasure]
    multiplicity: MultiplicityQuestion | None = None
    consequence: str


class EstimandCard(Model):
    """WP17 (MODELING_SEQUENCE §1 step 2): what the exposure and effect question offers."""

    exposures: list[EstimandExposure]
    effects: list[EstimandChoice]
    contrasts: list[EstimandChoice]
    measures: list[EstimandMeasure]
    family: EstimandFamily | None = None
    prevalence: float | None = None  # ESTIMAND: the event's share, which ranks the measures


class AdjustmentGroup(Model):
    """Covariates the pack guesses alike, confirmed with one tap (``decision``); or the unguessed."""

    key: str
    label: str
    columns: list[str]
    guess: dict[str, str] | None
    reason: str
    derived: str | None  # the role the guess derives
    derived_words: str | None
    decision: dict[str, Any] | None
    # ESTIMAND (MODELING_SEQUENCE §2): under an odds or hazard ratio, a cause of the outcome only
    # changes the conditional estimand, and the card says so
    estimand_note: str | None = None


class DerivedRole(Model):
    role: str
    words: str
    adjusted: bool  # in the primary model
    secondary: bool  # in the declared "further adjusted for" model
    why: str
    estimand_note: str | None = None


class AdjustmentCard(Model):
    """WP17 (MODELING_SEQUENCE §1 step 3): the disjunctive cause criterion, asked per covariate."""

    exposure: str
    effect: str
    questions: dict[str, str]
    groups: list[AdjustmentGroup]
    answered: dict[str, DerivedRole]
    adjusted: list[str]
    left_out: list[str]
    secondary: list[str]
    source: str
    estimand_note: str | None = None


class ModelSequenceCard(Model):
    """ESTIMAND (MODELING_SEQUENCE §1 row 11): Model 1's declaration, the pack's guess leading."""

    declared: list[str] | None
    guess: list[str]
    allowed: list[str]
    decision: dict[str, Any]
    reason: str


class ProposalsArtifact(Model):
    """The ``proposals`` artifact: offered for the exclusions, missing-values and energy questions."""

    exclusions: list[ExclusionProposal]
    energy: EnergyReading | None
    missing: MissingReading
    n_base: int  # the rows every count here is made among (the outcome recorded), all rows as loaded
    basis: str
    # M2: at most one coach line per decision card, keyed by question ("exclusions", "missing").
    coach: dict[str, CoachNote] = {}
    # WP10: the survey question's options, when a column reads as a survey weight.
    survey: SurveyProposal | None = None
    # WP12a: the exposure-form options, soundest first for the declared purpose.
    exposure_forms: list[ExposureFormOption] = []
    # WP13: the energy column's unit and how it was read; None without an energy column.
    energy_unit: EnergyUnitReading | None = None
    # WP17: customary and sound on every option; under inference the estimand and adjustment cards.
    labels: QuestionLabels | None = None
    estimand: EstimandCard | None = None
    adjustment: AdjustmentCard | None = None
    # ESTIMAND: Model 1 of the declared sequence, offered beside the adjustment answers so it is
    # declared before any estimate is shown
    model_sequence: ModelSequenceCard | None = None


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


class TurnExit(Model):
    label: str
    decision: dict[str, Any] | None


class TurnCheck(Model):
    """Whether the table can be turned around: the feature-name column, the columns that describe
    the features (kept beside them, never turned into samples), and why not."""

    label_column: str | None
    label_kind: Literal["text", "number"] | None
    annotations: list[str]
    n_features: int
    n_samples: int
    refusal: str | None
    code: str | None
    exits: list[TurnExit]
    declared: bool  # the user's ``feature_table`` answer, not the reading


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

    reading: Literal["repeats", "time_points", "imputed_copies"] | None
    stated: bool  # strong enough to state as a skip ("Ask me anyway" reopens it)
    confidence: Literal["high", "medium"] | None
    evidence: list[str]
    sentence: str
    spacing: Spacing | None
    replicate_index: str | None
    # WP18 (audit I18): the column numbering imputed copies (``_MULT_``), when read as such
    implicate_column: str | None = None
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


class DateExample(Model):
    text: str
    month_first: str
    day_first: str


class TimeOrder(Model):
    """Whether the time column can put a unit's records in order (stages.working.time_order)."""

    column: str
    kind: Literal["numbers", "dates", "levels", "ambiguous", "mixed", "none"]
    n: int  # values present
    placed: int  # values it orders; the rest are undated
    orderable: bool
    levels: list[str]  # a text column's levels, first seen first
    proposed: list[str]  # a natural order of them, to declare
    examples: list[DateExample]  # dates that read two ways: both readings


class StructureArtifact(Model):
    """The ``structure`` artifact: what the grain, repeats, unit and aggregation questions offer."""

    grain: GrainReading
    units: UnitCounts | None
    repeats: RepeatsReading | None
    outcome: OutcomeWithinUnit | None
    aggregation: AggregationMenu | None
    time_columns: list[str]
    time_column: str | None
    unread_dates: list[str]  # text dates that read month-first and day-first alike: not read yet
    time_order: TimeOrder | None


class Repair(Model):
    column: str
    expression: str


class OutcomeRule(Model):
    column: str
    varies: bool
    rule: Literal["constant", "mean", "first", "last"]
    n_missing: int  # units whose kept outcome is blank (the first or last record had none)


class CombinedColumn(Model):
    """One column that varied within a unit (or was given its own rule), and how it was combined."""

    column: str
    kind: Literal["amount", "code", "category", "constant"]
    rule: Literal["mean", "first", "last", "change", "mode", "constant"]
    varied: bool
    chosen: bool  # the aggregation answer set this column's rule


class AggregationReceipt(Model):
    """What combining did: rows → units, how the outcome was kept, what lost information."""

    id_column: str
    method: Literal["mean", "first", "last", "change"]
    outcome: OutcomeRule | None
    time_column: str | None
    ordered_by: str  # the time column when it ordered the records, else "file order"
    order: Literal["numbers", "dates", "declared levels", "file order"]
    undated_records: int  # records the time column could not place: never first, last or change
    units_without_dated_records: int
    n_source_rows: int
    n_units: int
    single_record_units: int
    varying: list[str]  # non-numeric columns that differed within a unit
    combined_numeric: int
    constant_columns: int
    columns: list[CombinedColumn]  # every column that varied within a unit, with its rule


class ReferenceRowsLeft(Model):
    """Rows a recorded repair excluded as reference rows (pooled QC injections; audit RO-13)."""

    column: str
    levels: list[str]
    finding: str
    n: int


class DerivedColumn(Model):
    """A column the working table derives row by row (the log-scale outcome; audit RO-10)."""

    column: str
    expression: Literal["ln"]
    source: str  # the column it is derived from


class WorkingArtifact(DatasetInfo):
    """The ``working`` artifact: DatasetInfo of the table every later stage reads."""

    pass_through: bool  # nothing structural recorded: the oriented table, referenced, not copied
    transposed: bool
    n_source_rows: int
    repairs: list[Repair]
    aggregation: AggregationReceipt | None
    row_map: Literal["identity", "row_map.parquet"]
    # WP18: the reference rows that left before anything read the table, and the derived columns
    reference_rows: list[ReferenceRowsLeft] = []
    derived: list[DerivedColumn] = []


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

# WP12 (AUDIT_REPORT §5): the primary analysis beside its sensitivity analyses, and regression
# calibration of energy-adjusted intakes. Defined beside their stages.
from turbotab.core.stages.calibration import CalibrationArtifact  # noqa: E402
from turbotab.core.stages.sensitivity import SensitivityArtifact  # noqa: E402

ARTIFACT_MODELS.update({"sensitivity": SensitivityArtifact, "calibration": CalibrationArtifact})

# WP17 (AUDIT_REPORT §5): the declared "further adjusted for" model beside the primary.
from turbotab.core.stages.secondary import SecondaryArtifact  # noqa: E402

ARTIFACT_MODELS.update({"secondary": SecondaryArtifact})

# The NCI usual-intake method: its offer and each component's distribution.
from turbotab.core.stages.usual_intake import UsualIntakeArtifact  # noqa: E402

ARTIFACT_MODELS.update({"usual_intake": UsualIntakeArtifact})
# ESTIMAND (MODELING_SEQUENCE §1 rows 2, 11, 12): the exposure's effect across the declared models.
from turbotab.core.stages.effects import EffectsArtifact  # noqa: E402

ARTIFACT_MODELS.update({"effects": EffectsArtifact})
# The causal lane: its card (outcome-free) and its estimate (turbotab/core/stages/causal.py).
from turbotab.core.stages.causal import CausalArtifact, CausalDesignArtifact  # noqa: E402

ARTIFACT_MODELS.update({"causal_design": CausalDesignArtifact, "causal": CausalArtifact})
# V2 causal row: a time-varying exposure by g-methods, diagnostics before estimates.
from turbotab.core.stages.time_varying import TimeVaryingArtifact  # noqa: E402

ARTIFACT_MODELS.update({"time_varying": TimeVaryingArtifact})

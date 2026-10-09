"""The model-family protocol and its registry (M1_CONTRACT §7; MODEL_FAMILY_CONTRACT §1).

A family is a plug-in: it declares what it needs (``needs_scaling``,
``handles_missing``), what it assumes (``inductive_bias``), and how it judges a
situation (``assess``), so the shelf, the pipeline, the lineage and the
consequence previews all derive from the declaration. A later milestone adds a
family by calling :func:`register_family`.

**The contract's declarations** (MODEL_FAMILY_CONTRACT §1, package MC-1): what the family is
(:class:`Identity`), what it gives under inference (:class:`InferenceDecl`), its quiet names
(:class:`Named`) with their sources (:class:`Source`), the transformations its predictions do not
change under, the shape its curves take, the knobs that set its complexity (:class:`Knob`), the
checks it reports about its own fit, its output and raw scales, its explanation paths, its review
lenses and its replay tolerance. :func:`register_family` refuses a family that leaves out a member
of the protocol or declares a value outside its vocabulary, naming what is wrong. RECIPES §2.4's
``recipe``, ``tuning`` and ``defaults_version`` join these with RECIPES RT-2.

Code reads these declarations instead of switching on a family's key or its estimator's class.
The switches that remain are listed, with the package that retires each, in
``tests/acceptance/test_mc2_no_family_switches.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, get_args, runtime_checkable

from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import Lens, Purpose, Task

Fit = Literal["good", "fair", "poor"]
TASKS: tuple[Task, ...] = ("regression", "binary", "multiclass", "ordinal", "time_to_event")
PURPOSES: tuple[Purpose, ...] = ("prediction", "inference")
# What a family models unless it says otherwise: an ordered outcome is modeled as unordered
# classes by a family that does not declare ``ordered_levels = True`` (on that shelf it says so and
# ranks a step lower: audit ME-19, RO-10); a time-to-event outcome is an event with its follow-up,
# which a family takes only by declaring it (``survival.Cox``).
DEFAULT_TASKS: tuple[Task, ...] = ("regression", "binary", "multiclass", "ordinal")
ORDER_BLIND = "Treats the ordered levels as unordered classes, so it ignores their order."
ORDER_BLIND_COST = 1.0
INDUCTIVE_BIAS_WORDS = 20
PLAIN_WORDS = 22  # a Named's plain sentence and a Prior's (MODEL_FAMILY_CONTRACT C5)
KNOWN_AS_WORDS = 6  # a Named's quiet name

# ── the contract's vocabularies (MODEL_FAMILY_CONTRACT §1) ───────────────────

IdentityKind = Literal["estimator", "trained_network", "pretrained_prior"]
InferenceTable = Literal["intervals", "shrunk_no_intervals", "description_only"]
PriorKind = Literal["bound", "empirical", "convention"]
Direction = Literal["simpler", "more_flexible"]
CurveShape = Literal["straight", "piecewise_constant", "any"]
Output = Literal["value", "margin", "probability"]
Attribution = Literal["linear", "trees", "none"]
# How one fit's time grows with an n × p matrix (C6's cost model; ``models.cost.fit_cost``): one
# pass over the cells per iteration (n·p), or forming and factoring the p × p cross-product
# (n·p·min(n, p)), as least squares and Newton steps do.
CostModel = Literal["cells", "cross_product"]
# The interval kinds an inference table may carry (C2), in the words of ``models.inference``'s
# covariance labels: model-based (classical t, Wald, likelihood), HC3, CR2, a cluster sandwich (CR1,
# Lin–Wei's CR0, GEE's), t on Satterthwaite degrees of freedom, Firth's profile penalized
# likelihood, and Taylor linearization over a survey design.
INTERVAL_KINDS = ("model", "HC3", "CR2", "sandwich", "Satterthwaite", "profile", "Taylor")
# The input transformations a family's predictions may not change under (C3), each probed two-sided
# by the reference tests (C13). A family declares the largest that holds; the smaller are implied.
INVARIANCES = ("linear_maps", "rotation_after_scaling", "monotone_per_column", "column_scale")
# The scale ``models.explain.Anatomy.raw_score`` draws a task's curves on (C10), or "not drawn: "
# and the reason said.
RAW_SCALES = ("value", "margin", "latent", "log_hazard")
NOT_DRAWN = "not drawn: "
# Several classes would need a curve each, which v2.0 does not draw; a family that models an
# ordered outcome as unordered classes says the same of it. The raw scales of a family that models
# DEFAULT_TASKS: the prediction for a number, the log-odds for a yes/no outcome.
CLASSES_NOT_DRAWN = f"{NOT_DRAWN}several classes would need a curve each, which v2.0 does not draw"
CLASS_SCALES: Mapping[str, str] = MappingProxyType({
    "regression": "value", "binary": "margin", "multiclass": CLASSES_NOT_DRAWN,
    "ordinal": CLASSES_NOT_DRAWN})
ARCHITECTURES = ("equation", "trees", "shrinkage", "spectrum")
# The checks a family reports about its own fit (C8). They are keys MC-18's diagnostics registry
# will hold, each with a fixture where it fires and one where it stays silent.
DIAGNOSTICS = ("separation", "collinearity", "residual_spread", "influence", "proportional_odds",
               "proportional_hazards", "boundary", "convergence")
UPDATING = ("shrinkage",)  # recalibration paths (C9): shrinkage by the calibration slope
SOLVABLE: tuple[str, ...] = ()  # the solvable-settings harness (C13, MC-14) holds none yet
REVIEW_LENSES = ("shared", *get_args(Lens))


@dataclass(frozen=True)
class Source:
    """A source a declaration cites: a key into ``models.sources.SOURCES`` (MODEL_FAMILY_CONTRACT
    §7's verified list, until SIZING X4's citation registry), and where in it."""

    key: str
    where: str = ""  # "§3.4.1, eqs. 3.47 and 3.50"


@dataclass(frozen=True)
class Named:
    """A term or phenomenon in two registers (C5): the card's plain sentence, and the quiet name
    with its source, shown on point or focus, never as a second label."""

    plain: str  # ≤ 22 words
    known_as: str  # ≤ 6 words: "double descent"
    source: Source


@dataclass(frozen=True)
class Identity:
    """What the family is (C1). Its library's version is recorded at every fit; two families that
    differ here are two versions."""

    kind: IdentityKind
    library: str
    estimator: str  # class names, for provenance only: nothing switches on them
    learning_rule: str = ""  # trained_network: optimizer, schedule, batch size, epochs or stopping
    initialization: str = ""  # trained_network: scheme and scale
    # trained_network: "standard", "NTK" or "muP", and the output multiplier. It implies no regime:
    # lazy or rich training is measured after the fit (C8).
    parameterization: str = ""
    seed_policy: str = ""  # how its seeds derive from the split's seed (RECIPES §4.7)
    prior: str = ""  # pretrained_prior: the checkpoint's name and SHA-256


@dataclass(frozen=True)
class InferenceDecl:
    """What the family gives under inference (C2). A family that declares none is not offered as an
    inference table."""

    table: InferenceTable
    intervals: tuple[str, ...] = ()  # INTERVAL_KINDS, each with its reference test (C13)
    design_based: bool = False  # it estimates over a survey design (``models.survey``)
    product_terms: bool = False  # it tests product terms (``methods.interaction``)
    matrix_table: bool = False  # its table is made from a model matrix alone (``stages.effects``)
    default_for: tuple[Task, ...] = ()  # the tasks it is the default inference family for


@dataclass(frozen=True)
class Prior:
    """How data-hungry the family is (C4): a sourced caution under "More angles", never a forecast,
    and never a move of the score by itself."""

    says: str  # ≤ 22 words
    kind: PriorKind
    source: Source | None  # None only for a convention


@dataclass(frozen=True)
class Knob:
    """A setting that moves the family's effective complexity (C7), and which way."""

    setting: str  # the estimator's parameter, or "time" for early stopping
    more_means: Direction
    # A key into ``models.formulas.FORMULAS``, whose callable takes the estimator's own parameters;
    # empty where no formula exists, and the card says so instead of borrowing one.
    formula: str = ""
    source: Source | None = None


@dataclass(frozen=True)
class Situation:
    """What the shelf knows about the analysis when it ranks families."""

    task: Task
    purpose: Purpose | None
    n_rows: int
    n_features: int
    n_events: int | None = None  # binary: rows in the rarer class; time to event: rows with the event
    n_classes: int | None = None
    # Candidate predictor parameters (a category with k levels is k − 1 of them), which sample-size
    # criteria count; None: one per predictor. The outcome's mean and SD on the rows ranked
    # (regression), which Riley et al.'s intercept criterion needs.
    n_parameters: int | None = None
    outcome_mean: float | None = None
    outcome_sd: float | None = None
    lenses: tuple[str, ...] = ()  # the declared lenses (an omics lens changes what is sound)
    class_counts: tuple[int, ...] | None = None  # rows per class or level
    # Units when an identifier repeats in these rows (``inference.resolve_clusters``); None when
    # every row is a unit of its own, or nothing says which rows belong together.
    n_units: int | None = None


@dataclass(frozen=True)
class Assessment:
    """A family's own judgment of a situation: order (``score``, higher first) and stated concern."""

    score: float
    fit: Fit
    concerns: tuple[str, ...] = ()


@runtime_checkable
class ModelFamily(Protocol):
    key: str
    label: str
    tasks: tuple[Task, ...]
    inductive_bias: str  # ≤ 20 words
    strengths: tuple[str, ...]
    cautions: tuple[str, ...]
    needs_scaling: bool
    handles_missing: bool
    # MODEL_FAMILY_CONTRACT §1 (:class:`FamilyBase` says what each one means).
    identity: Identity
    purposes: tuple[Purpose, ...]
    predicts: bool
    flexible: bool
    bootstrap_optimism: bool
    ordered_levels: bool
    linear_in_values: bool
    pools_imputations: bool
    inference_decl: InferenceDecl | None
    reads: tuple[str, ...]
    sample_efficiency: tuple[Prior, ...]
    same_kind_as: tuple[str, float] | None
    bias_terms: tuple[Named, ...]
    invariances: tuple[str, ...]
    curve_shape: CurveShape
    complexity: tuple[Knob, ...]
    diagnostics: tuple[str, ...]
    output: Output
    updating: tuple[str, ...]
    raw_scale: Mapping[str, str]
    attribution: Attribution
    architecture: tuple[str, ...]
    review_lenses: tuple[str, ...]
    solvable: tuple[str, ...]
    replay_tolerance: float
    sources: tuple[Source, ...]
    cost_model: CostModel

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        """An unfitted sklearn estimator; the pipeline's last step."""

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        """(label, detail) of the model step, in the app's voice."""

    def methods_label(self, task: Task | None) -> str:
        """What the methods text calls the family for ``task``, mid-sentence ("linear regression",
        "gradient-boosted trees"); ``voice``'s ``select_models`` sentence reads it."""

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        """Coefficient rows ``{feature, estimate, ci_low, ci_high, p}`` from the pipeline fit on
        ``X``/``y``, or None when the family has none. Under prediction those are the training
        rows; under inference, every analyzed row (BLUEPRINT §12 ruling 3)."""

    def assess(self, situation: Situation) -> Assessment:
        """How well this family suits the situation, with every concern stated."""


class FamilyInfo(BaseModel):
    """What ``GET /api/models`` says about one family."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: str
    label: str
    tasks: list[Task]
    inductive_bias: str
    strengths: list[str]
    cautions: list[str]
    needs_scaling: bool
    handles_missing: bool
    # The purposes the family serves, and whether it predicts (a family that only tests has no
    # cross-validated score).
    purposes: list[Purpose] = ["prediction", "inference"]
    predicts: bool = True
    # MODEL_FAMILY_CONTRACT §1's user-facing declarations: whether it is a flexible learner (ranked
    # after the regression families below Riley's minimum), whether Harrell's bootstrap is sound for
    # it, the transformations its predictions do not change under, the table it gives under
    # inference (none: it is not offered as one), and its quiet names with their sources.
    flexible: bool = False
    bootstrap_optimism: bool = True
    invariances: list[str] = []
    inference_table: InferenceTable | None = None
    bias_terms: list[Named] = []


# Every member of the protocol, which :func:`register_family` asks for by name (a family missing
# one would otherwise fail ``isinstance`` without saying what it lacks).
MEMBERS = ("key", "label", "tasks", "inductive_bias", "strengths", "cautions", "needs_scaling",
           "handles_missing", "identity", "purposes", "predicts", "flexible", "bootstrap_optimism",
           "ordered_levels", "linear_in_values", "pools_imputations", "inference_decl", "reads",
           "sample_efficiency", "same_kind_as", "bias_terms", "invariances", "curve_shape",
           "complexity", "diagnostics", "output", "updating", "raw_scale", "attribution",
           "architecture", "review_lenses", "solvable", "replay_tolerance", "sources", "cost_model",
           "build", "describe", "methods_label", "coefficients", "assess")

_REGISTRY: dict[str, ModelFamily] = {}


def register_family(family: ModelFamily) -> ModelFamily:
    """Add ``family`` to the shelf. Its order of registration breaks ties in the ranking.

    The family must declare every member of the protocol and keep each declaration inside its
    vocabulary (MODEL_FAMILY_CONTRACT §1); otherwise the error names each thing that is wrong."""
    name = getattr(family, "key", None) or repr(family)
    missing = [m for m in MEMBERS if not hasattr(family, m)]
    if missing:
        raise TypeError(f"{name}: the model-family contract asks for {', '.join(missing)}, which "
                        f"it does not declare (MODEL_FAMILY_CONTRACT §1)")
    if not isinstance(family, ModelFamily):
        raise TypeError(f"{family!r} does not implement ModelFamily")
    words = len(family.inductive_bias.split())
    if words > INDUCTIVE_BIAS_WORDS:
        raise ValueError(f"{family.key}: inductive_bias has {words} words; the budget is "
                         f"{INDUCTIVE_BIAS_WORDS}")
    unknown = set(family.tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"{family.key}: unknown tasks {sorted(unknown)}")
    problems = contract_problems(family)
    if problems:
        raise ValueError(f"{family.key} does not meet the model-family contract: "
                         + "; ".join(problems))
    if family.key in _REGISTRY and _REGISTRY[family.key] is not family:
        raise ValueError(f"a model family {family.key!r} is already registered")
    _REGISTRY[family.key] = family
    return family


def contract_problems(family: ModelFamily) -> list[str]:
    """What in ``family``'s declarations breaks MODEL_FAMILY_CONTRACT §1, one plain clause each,
    with its clause; empty when nothing does."""
    from turbotab.core.models.formulas import FORMULAS
    from turbotab.core.models.sources import SOURCES

    out: list[str] = []

    def outside(what: str, values: Any, vocabulary: Sequence[str], clause: str) -> None:
        odd = [v for v in values if v not in vocabulary]
        if odd:
            out.append(f"{what} {odd} are not among {list(vocabulary)} ({clause})")

    def cited(what: str, source: Any, clause: str) -> None:
        if not isinstance(source, Source):
            out.append(f"{what} cites {source!r}, not a Source ({clause})")
        elif source.key not in SOURCES:
            out.append(f"{what} cites {source.key!r}, which is not a verified source in "
                       f"models.sources ({clause})")

    def budget(what: str, text: str, words: int, clause: str) -> None:
        if not text.strip() or len(text.split()) > words:
            out.append(f"{what} must say something in at most {words} words ({clause})")

    tasks = set(family.tasks)
    identity = family.identity
    if not isinstance(identity, Identity):
        out.append("identity must be an Identity (C1)")
    else:
        outside("identity kind", [identity.kind], get_args(IdentityKind), "C1")
        if not identity.library or not identity.estimator:
            out.append("identity must name its library and estimator (C1)")
        if identity.kind == "trained_network":
            empty = [f for f in ("learning_rule", "initialization", "parameterization",
                                 "seed_policy") if not getattr(identity, f)]
            if empty:
                out.append(f"a trained network's identity must declare {', '.join(empty)} (C1)")
        if identity.kind == "pretrained_prior" and not _has_sha256(identity.prior):
            out.append("a pretrained prior's identity must name its checkpoint and SHA-256 (C1)")
    if not family.purposes:
        out.append("purposes must name at least one purpose (C2)")
    outside("purposes", family.purposes, PURPOSES, "C2")
    for flag in ("predicts", "flexible", "bootstrap_optimism", "ordered_levels",
                 "linear_in_values", "pools_imputations"):
        if not isinstance(getattr(family, flag), bool):
            out.append(f"{flag} must be True or False (C2, C11)")
    decl = family.inference_decl
    if decl is not None:
        if not isinstance(decl, InferenceDecl):
            out.append("inference_decl must be an InferenceDecl or None (C2)")
        else:
            outside("the inference table", [decl.table], get_args(InferenceTable), "C2")
            outside("interval kinds", decl.intervals, INTERVAL_KINDS, "C2")
            if (decl.table == "intervals") != bool(decl.intervals):
                out.append("a table with intervals names their kinds, and only such a table "
                           "names any (C2)")
            outside("default_for", decl.default_for, sorted(tasks), "C2")
            if "inference" not in family.purposes:
                out.append("an inference table is declared, but inference is not among its "
                           "purposes (C2)")
    if family.same_kind_as is not None:
        kind_of = family.same_kind_as
        if (not isinstance(kind_of, tuple) or len(kind_of) != 2
                or not isinstance(kind_of[1], (int, float))):
            out.append("same_kind_as must be (family key, rank offset) (C4)")
        elif kind_of[0] == family.key or kind_of[0] not in _REGISTRY:
            out.append(f"same_kind_as names {kind_of[0]!r}, which is not another registered "
                       f"family; register that family first (C4)")
    for prior in family.sample_efficiency:
        budget("a sample-efficiency prior", prior.says, PLAIN_WORDS, "C4")
        outside("a prior's kind", [prior.kind], get_args(PriorKind), "C4")
        if prior.source is not None or prior.kind != "convention":
            cited("a sample-efficiency prior", prior.source, "C4")
    for term in family.bias_terms:
        budget("a bias term's plain sentence", term.plain, PLAIN_WORDS, "C5")
        budget("a bias term's quiet name", term.known_as, KNOWN_AS_WORDS, "C5")
        cited(f"the bias term {term.known_as!r}", term.source, "C5")
    outside("invariances", family.invariances, INVARIANCES, "C3")
    outside("curve_shape", [family.curve_shape], get_args(CurveShape), "C5")
    for knob in family.complexity:
        outside(f"the knob {knob.setting!r}'s direction", [knob.more_means], get_args(Direction),
                "C7")
        if knob.formula and knob.formula not in FORMULAS:
            out.append(f"the knob {knob.setting!r} names the formula {knob.formula!r}, which "
                       f"models.formulas does not hold (C7)")
        if knob.source is not None:
            cited(f"the knob {knob.setting!r}", knob.source, "C7")
    outside("diagnostics", family.diagnostics, DIAGNOSTICS, "C8")
    outside("output", [family.output], get_args(Output), "C9")
    outside("updating", family.updating, UPDATING, "C9")
    outside("raw_scale's tasks", list(family.raw_scale), sorted(tasks), "C10")
    for task, scale in family.raw_scale.items():
        if scale not in RAW_SCALES and not (scale.startswith(NOT_DRAWN)
                                            and scale[len(NOT_DRAWN):].strip()):
            out.append(f"raw_scale[{task!r}] is {scale!r}: one of {list(RAW_SCALES)}, or "
                       f"{NOT_DRAWN!r} and the reason (C10)")
    outside("attribution", [family.attribution], get_args(Attribution), "C10")
    outside("architecture", family.architecture, ARCHITECTURES, "C10")
    if family.predicts:
        unscaled = sorted(tasks - set(family.raw_scale))
        if unscaled:
            out.append(f"a predicting family declares a raw scale for each of its tasks; "
                       f"{unscaled} have none (C10)")
        if "binary" in tasks:
            if family.output == "value":
                out.append("a predicting family for a yes/no outcome outputs a margin or a "
                           "probability (C9)")
            model = family.build("binary", "prediction", 100, 2)
            if not hasattr(model, "decision_function"):
                out.append("its yes/no model step has no decision_function, which "
                           "Anatomy.raw_score reads (C9)")
    else:
        if family.invariances or family.raw_scale or family.attribution != "none":
            out.append("a family that makes no predictions declares no invariances, raw scale or "
                       "attribution: they are not applicable (C3, C10)")
    outside("review_lenses", family.review_lenses, REVIEW_LENSES, "C14")
    outside("solvable settings", family.solvable, SOLVABLE, "C13")
    tolerance = family.replay_tolerance
    if not isinstance(tolerance, (int, float)) or not 0 < tolerance < math.inf:
        out.append("replay_tolerance must be a positive number (C12)")
    for source in family.sources:
        cited("a primary source", source, "C1")
    outside("cost_model", [family.cost_model], get_args(CostModel), "C6")
    return out


def _has_sha256(text: str) -> bool:
    import re

    return re.search(r"\b[0-9a-f]{64}\b", text or "") is not None


def unregister_family(key: str) -> None:
    """Remove a family (for plug-ins that are unloaded, and for tests)."""
    _REGISTRY.pop(key, None)


def families(task: Task | None = None) -> list[ModelFamily]:
    """Registered families (those that can model ``task``, when given), in registration order."""
    return [f for f in _REGISTRY.values() if task is None or task in f.tasks]


def get_family(key: str) -> ModelFamily:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"there is no model family {key!r}; known: {', '.join(_REGISTRY)}") from None


def info(family: ModelFamily) -> FamilyInfo:
    return FamilyInfo(
        key=family.key, label=family.label, tasks=list(family.tasks),
        inductive_bias=family.inductive_bias, strengths=list(family.strengths),
        cautions=list(family.cautions), needs_scaling=family.needs_scaling,
        handles_missing=family.handles_missing, purposes=list(family.purposes),
        predicts=family.predicts, flexible=family.flexible,
        bootstrap_optimism=family.bootstrap_optimism, invariances=list(family.invariances),
        inference_table=family.inference_decl.table if family.inference_decl else None,
        bias_terms=list(family.bias_terms),
    )


def inference_default(task: Task) -> ModelFamily | None:
    """The registered family that is ``task``'s default inference family (its ``inference_decl``
    names the task in ``default_for``), or None."""
    return next((f for f in families(task)
                 if f.inference_decl is not None and task in f.inference_decl.default_for), None)


def rank(situation: Situation) -> list[tuple[ModelFamily, Assessment]]:
    """Every family that can model the task, best first. The shelf is never shortened."""
    order = {f.key: i for i, f in enumerate(families())}
    judged = []
    for family in families(situation.task):
        judged_one = family.assess(situation)
        if situation.task == "ordinal" and not family.ordered_levels:
            judged_one = Assessment(judged_one.score - ORDER_BLIND_COST, judged_one.fit,
                                    (ORDER_BLIND, *judged_one.concerns))
        judged.append((family, judged_one))

    def place(fa: tuple[ModelFamily, Assessment]) -> tuple[float, bool, int]:
        # Under prediction, a family that makes no predictions (WP11's feature-wise tests) goes
        # after every family that does and scores as well: it has nothing to offer the purpose.
        family, judged_one = fa
        silent = situation.purpose == "prediction" and not family.predicts
        return (-judged_one.score, silent, order[family.key])

    return sorted(judged, key=place)


# ── shared helpers for families ──────────────────────────────────────────────


class FamilyBase:
    """Defaults a concrete family can inherit; it still declares everything the protocol asks.

    The members with no default here (``identity``, ``purposes``, ``predicts``, ``flexible``,
    ``bootstrap_optimism`` and ``methods_label``) are the ones every family states itself
    (MODEL_FAMILY_CONTRACT §1); :func:`register_family` names any that is missing."""

    key: str = ""
    label: str = ""
    tasks: tuple[Task, ...] = DEFAULT_TASKS
    inductive_bias: str = ""
    strengths: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    needs_scaling: bool = False
    handles_missing: bool = False
    identity: Identity  # C1: what it is, by library and estimator
    purposes: tuple[Purpose, ...]  # C2: the purposes it serves
    predicts: bool  # C2. False: it tests and makes no predictions (no cross-validated score)
    # C11: a flexible learner, ranked after the regression families below Riley's minimum
    # (``models.selection.shelf_order``) and compared with the interpretable model as one.
    flexible: bool
    # C11: whether Harrell's bootstrap optimism correction is sound for it (``models.validation``).
    # A learner that nearly memorizes its rows scores the original rows inside each resample almost
    # perfectly, so the bootstrap understates its optimism (Coley et al. 2023): it declares False,
    # and the fit keeps its cross-validated score as its internal validation.
    bootstrap_optimism: bool
    ordered_levels: bool = False  # it models an ordered outcome's order (``rank`` reads this)
    # Its model is a weighted sum of the values as given, so their scale (raw counts or log) is part
    # of what it assumes; raw omics values wait for a normalization (``methods.omics``).
    linear_in_values: bool = False
    # False: it pools no multiple imputations, and the fit holds its table under that answer.
    pools_imputations: bool = True
    inference_decl: InferenceDecl | None = None  # C2; None: not offered as an inference table
    reads: tuple[str, ...] = ()  # C4: the input-profile fields its assess reads (MC-4)
    sample_efficiency: tuple[Prior, ...] = ()  # C4
    # C4: (family key, rank offset). It reads that family's assessment, less the offset, instead
    # of keeping one of its own (XGBoost beside boosted trees, RECIPES §2.2).
    same_kind_as: tuple[str, float] | None = None
    bias_terms: tuple[Named, ...] = ()  # C5: the inductive bias's quiet names
    invariances: tuple[str, ...] = ()  # C3: INVARIANCES, the largest that holds
    curve_shape: CurveShape = "any"  # C5: straight, piecewise constant, or any
    complexity: tuple[Knob, ...] = ()  # C7
    diagnostics: tuple[str, ...] = ()  # C8: DIAGNOSTICS
    output: Output = "value"  # C9: the scale its yes/no predictions come on, if it makes any
    updating: tuple[str, ...] = ()  # C9: UPDATING
    raw_scale: Mapping[str, str] = MappingProxyType({})  # C10: task -> RAW_SCALES or NOT_DRAWN
    attribution: Attribution = "none"  # C10: exact linear SHAP, path-dependent TreeSHAP, or none
    architecture: tuple[str, ...] = ()  # C10: ARCHITECTURES
    review_lenses: tuple[str, ...] = ()  # C14: the packets that review it in full, or "shared"
    solvable: tuple[str, ...] = ()  # C13: SOLVABLE
    replay_tolerance: float = 1e-12  # C12: DoD gate 6's; a looser one needs Nolan's approval
    sources: tuple[Source, ...] = ()  # its primary sources, as a method contract has
    cost_model: CostModel = "cells"  # C6: how one fit's time grows (``models.cost.fit_cost``)

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        return None

    def __repr__(self) -> str:
        return f"<model family {self.key}>"


def reports_coefficients(family: Any) -> bool:
    """Whether ``family`` has a coefficient table: it defines ``coefficients`` itself rather than
    inheriting :class:`FamilyBase`'s, which has none."""
    method = getattr(type(family), "coefficients", None)
    return method is not None and method is not FamilyBase.coefficients


def coefficient_rows(features: Sequence[str], estimates: Any, *, intercept: Any = None,
                     classes: Sequence[Any] | None = None, ci_low: Any = None, ci_high: Any = None,
                     p: Any = None) -> list[dict[str, Any]]:
    """Coefficient rows in the fit artifact's shape.

    ``estimates`` is (p,) or (k, p) for k class rows; with class rows each feature is labeled
    ``feature [class]``. ``intercept`` is a scalar or (k,). CIs and p-values align with estimates.
    """
    import numpy as np

    est = np.atleast_2d(np.asarray(estimates, dtype=float))
    lo = None if ci_low is None else np.atleast_2d(np.asarray(ci_low, dtype=float))
    hi = None if ci_high is None else np.atleast_2d(np.asarray(ci_high, dtype=float))
    pv = None if p is None else np.atleast_2d(np.asarray(p, dtype=float))
    icpt = None if intercept is None else np.atleast_1d(np.asarray(intercept, dtype=float))
    rows: list[dict[str, Any]] = []
    for k in range(est.shape[0]):
        suffix = f" [{classes[k]}]" if classes is not None and est.shape[0] > 1 else ""
        names = (["(intercept)"] if icpt is not None else []) + list(features)
        values = ([icpt[k]] if icpt is not None else []) + list(est[k])
        for j, (name, value) in enumerate(zip(names, values)):
            rows.append({
                "feature": f"{name}{suffix}",
                "estimate": _finite(value),
                "ci_low": None if lo is None else _finite(lo[k][j]),
                "ci_high": None if hi is None else _finite(hi[k][j]),
                "p": None if pv is None else _finite(pv[k][j]),
            })
    return rows


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


__all__ = [
    "ARCHITECTURES", "Assessment", "CLASSES_NOT_DRAWN", "CLASS_SCALES", "DEFAULT_TASKS",
    "DIAGNOSTICS", "FamilyBase", "FamilyInfo", "Fit", "INTERVAL_KINDS", "INVARIANCES", "Identity",
    "InferenceDecl", "Knob", "MEMBERS", "ModelFamily", "NOT_DRAWN", "Named", "ORDER_BLIND",
    "PURPOSES", "Prior", "RAW_SCALES", "Situation", "Source", "TASKS", "coefficient_rows",
    "contract_problems", "families", "get_family", "inference_default", "info", "rank",
    "register_family", "reports_coefficients", "unregister_family",
]

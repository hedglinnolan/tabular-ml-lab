"""Tuning: a family's declaration, and the one plan every fit of it follows (RECIPES_AND_TUNING §4;
MODEL_FAMILY_CONTRACT C6; WAVE_C6A_PLAN §2).

**What is here (RT-1a): the types and the pure functions.**

* A family's declaration: :class:`Dimension` and :class:`TuningDecl` (RECIPES §4.1), with the
  contract's ``structural`` settings (C6). A declaration is checked when it is made
  (:func:`tuning_problems`), so a family module whose tuning cannot run fails at import with every
  problem named.
* The plan: :class:`TuningPlan`, computed once by :func:`make_plan` from the plan's size
  (:func:`effective_size`, :func:`plan_size`, :func:`inner_k`), recorded and stated. Every fit of a
  family uses the same plan unchanged: every outer fold of every repeat, bootstrap resamples, the
  nested cross-validation's folds, internal–external refits, evaluation's design-based folds and
  the final refit (RECIPES §4.2). Only a setting declared as a share of a fit's units rescales.
* The candidates: the standard settings first, once per combination of "Try both" options, then a
  scrambled Sobol sample mapped through each dimension's scale (:func:`sobol_sample`,
  :func:`map_unit`), or, for a path family, its grid (:func:`path_grid`). The candidate list is
  fixed before any score is seen, so replay never depends on the order fits finish in.
* The strategy: :data:`STRATEGIES`, keyed by the strategy's name and never by a family (V2X_SEAMS
  seam guard 3). v2 has one, ``"sobol"``; it owns the candidate generator and the fit count
  (:meth:`TuningPlan.fits`), so the time estimate asks the strategy instead of keeping a formula of
  its own. A v2.x strategy (halving, TPE) is one more name.
* The choice: :func:`pooled_loss` over every inner validation row, then :func:`choose`, the lowest
  loss after rounding to :data:`LOSS_PRECISION` of the smallest, a tie going to the lower index:
  the standard candidates first, and on a path the stronger penalty.
* Seeds: :func:`derive_seed`, the SHA-256 of canonical JSON, never Python's salted ``hash()``.
* What a fit records: :class:`TuningRecord` and, for a path, :class:`PathCurve`.

**What is not here yet (RT-1b): the engine** that reads a plan: the head fitted once per inner
split, the inner splits drawn as the outer ones are, ``TunedPipeline``, the time estimate's
center candidate, cancel and the test seam that observes every split.

**How a candidate becomes the estimator's parameters** (:func:`estimator_params`): the
declaration's ``fixed`` settings, then the candidate's ``values``, passed through the family's
``settings`` member when it has one (shares of units to rows, symbolic standards such as
``"default"``, a cap on the leaf size, XGBoost's hessian scaling, ``alpha = n·λ``). A family without
``settings`` declares only names that are its estimator's own parameters; ``register_family``
checks both (``models.base``). The engine adds the early-stopping parameters itself, from the plan.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, get_args, runtime_checkable

import numpy as np

TuningKind = Literal["none", "path", "search"]
Scale = Literal["log", "linear", "int", "log_int", "choice", "share_of_units"]
StrategyName = Literal["sobol"]  # seam guard 3: v2.x adds names, never renames one
Mode = Literal["automatic", "lighter", "standard", "manual"]  # what SetTuning (RT-6) records
Loss = Literal["mse", "log_loss", "rps"]  # validation.PER_ROW_LOSSES' strictly proper primaries
SizeUnit = Literal["units", "events", "rarest_class"]
Activity = Literal["always", "without_early_stopping"]

SMALL_N = 300  # below this effective size a search keeps its standard candidates (§4.6)
MID_N = 1_000  # from this size the Sobol sample doubles, 8 to 16 (§4.2)
STOP_FROM_N = 1_500  # Sobol candidates stop early from this effective size (§4.2)
STANDARD_STOP_ROWS = 10_000  # standard candidates stop early above this many plan rows (sklearn)
SEARCHED_K = 3  # inner folds of a searched family (a convention: every candidate meets the same)
STOP_SHARE = 0.1  # the stopping set: a tenth of a fit's units
PER_FOLD_FLOOR = 2  # units of the rarest class, and PSUs, that every inner fold holds at least
LOSS_PRECISION = 1e-9  # pooled losses are rounded to this share of the smallest before the argmin
IMBALANCE_FITS = 5  # w: one candidate fit under the imbalance correction (1 + 5 folds × 4/5)
SOBOL_SIZES = ((SMALL_N, 8), (MID_N, 16))  # (from n_plan, S): the Sobol sample by plan size
# The keys an early-stopping declaration states. ``param``: the estimator's parameter holding the
# most rounds; ``rounds``: its value under early stopping; ``patience_param`` / ``patience``: the
# rounds without improvement before stopping; ``share``: the stopping set's share of a fit's units.
EARLY_STOPPING_KEYS = ("param", "rounds", "patience_param", "patience", "share")
# Optional: ``standard_rows``, the plan rows above which the *standard* candidates stop early too
# (scikit-learn's own rule, :data:`STANDARD_STOP_ROWS`, when absent; None: they never do, as
# XGBoost's defaults never stop early).
STANDARD_ROWS_KEY = "standard_rows"
PATH_SCALES = ("log", "linear")  # the scales a path dimension's points are spaced on


# ═════════════════════════════════════════════════════════════════════════════
# the declaration
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Dimension:
    """One setting a family's tuning may move (RECIPES §4.1), searched (``TuningDecl.dimensions``)
    or settable by hand (``TuningDecl.by_hand``).

    ``name`` is the estimator's own parameter, or a name the family's ``settings`` member resolves
    into parameters (a ridge's ``lambda`` into ``alpha = n·λ``). ``low``/``high`` bound the range on
    ``scale`` (:func:`map_unit` says how each scale maps; a ``choice`` reads ``choices`` instead, and
    ``share_of_units`` runs from one unit of the plan to ``high``). ``source`` cites where the range
    comes from, as the citation registry resolves it ("Probst et al. 2019").

    Added by RT-1a, each defaulted:

    * ``points``: a path family's grid points on this dimension (:func:`path_grid`); 0 for a
      searched dimension;
    * ``active``: ``"without_early_stopping"`` for the number of trees or rounds, which is searched
      only when the plan does not stop early; it must be the early-stopping ``param``;
    * ``tunability``: how much tuning this setting gained on benchmark data, where measured (C6;
      Probst, Boulesteix & Bischl 2019), so the tuning line can name the one or two that matter.
    """

    name: str  # the estimator's parameter (or one ``settings`` resolves)
    label: str  # plain: "how big each correction step is"
    term: str  # quiet: "learning rate"
    low: float
    high: float
    scale: Scale
    choices: tuple = ()
    source: str = ""
    points: int = 0  # a path grid's points (0 for a searched dimension)
    active: Activity = "always"
    tunability: float | None = None  # C6 (Probst et al. 2019), where measured


@dataclass(frozen=True)
class TuningDecl:
    """How a family is tuned (RECIPES §4.1), with the contract's ``structural`` settings (C6).

    * ``kind``: ``"none"`` (nothing searched; by-hand settings only, as Huber's threshold),
      ``"path"`` (one path per inner split over a grid, :func:`path_grid`: ridge, the elastic net)
      or ``"search"`` (candidates from the plan's strategy).
    * ``standard``: the standard settings, by name; every searched and by-hand dimension has one.
      A value may be symbolic (``"default"``), resolved per task and per fit by ``settings``.
    * ``fixed``: stated and never searched (``{"n_estimators": 500}``); merged under the values.
    * ``early_stopping``: :data:`EARLY_STOPPING_KEYS`, plus the optional
      :data:`STANDARD_ROWS_KEY`; None when the family never stops early.
    * ``out_of_bag``: candidates may be scored on their out-of-bag predictions (the forest), where
      the engine allows it.
    * ``max_drawn``: the most Sobol candidates any plan draws for this family, a power of two (the
      forest's 8: the least tunable, RECIPES §4.2); None: the plan's size alone decides.
    * ``space_version``: ``"boosted_trees/1"``, bumped whenever the space changes; it seeds the
      candidates (:func:`make_plan`).
    """

    kind: TuningKind
    dimensions: tuple[Dimension, ...] = ()  # searched
    by_hand: tuple[Dimension, ...] = ()  # settable by hand, never searched (Huber's threshold)
    standard: Mapping[str, Any] = field(default_factory=dict)  # the standard settings
    standard_source: str = ""  # "scikit-learn's defaults", "ranger's defaults"
    fixed: Mapping[str, Any] = field(default_factory=dict)  # stated, never searched
    early_stopping: Mapping[str, Any] | None = None
    out_of_bag: bool = False
    space_version: str = ""  # "boosted_trees/1"
    reason: str = ""  # ≤ 22 words
    # MODEL_FAMILY_CONTRACT C6: the settings that are identity (C1), never searched: the
    # architecture, the parameterization and output multiplier, the loss, the booster type.
    structural: tuple[str, ...] = ()
    max_drawn: int | None = None

    def __post_init__(self) -> None:
        problems = tuning_problems(self)
        if problems:
            raise ValueError("the tuning declaration breaks the model-family contract: "
                             + "; ".join(problems))

    def names(self) -> tuple[str, ...]:
        """Every dimension's name, searched then by hand."""
        return tuple(d.name for d in (*self.dimensions, *self.by_hand))


def tuning_problems(decl: TuningDecl) -> list[str]:
    """What in ``decl`` breaks MODEL_FAMILY_CONTRACT C6 or cannot run, one plain clause each; empty
    when nothing does."""
    out: list[str] = []
    if decl.kind not in get_args(TuningKind):
        out.append(f"its kind {decl.kind!r} is not among {list(get_args(TuningKind))}")
    both = sorted({d.name for d in decl.dimensions} & set(decl.structural))
    if both:
        out.append(f"{both} are structural, part of what the family is, so they are never "
                   f"searched dimensions (C6)")
    names = [d.name for d in (*decl.dimensions, *decl.by_hand)]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        out.append(f"{twice} are declared more than once (C6)")
    fixed = sorted(set(decl.fixed) & set(names))
    if fixed:
        out.append(f"{fixed} are fixed, so they are not dimensions too (C6)")
    stopping = decl.early_stopping
    if stopping is not None:
        lacks = [k for k in EARLY_STOPPING_KEYS if k not in stopping]
        if lacks:
            out.append(f"early_stopping lacks {lacks}: it states {list(EARLY_STOPPING_KEYS)} "
                       f"(C6)")
    for d in (*decl.dimensions, *decl.by_hand):
        out.extend(_dimension_problems(d, decl))
    if decl.kind in ("search", "none"):
        unstated = [d.name for d in (decl.dimensions if decl.kind == "search" else decl.by_hand)
                    if d.name not in decl.standard]
        if unstated:
            out.append(f"no standard value for {unstated}: the standard settings are always a "
                       f"candidate (RECIPES §4.1)")
    if decl.max_drawn is not None and (not isinstance(decl.max_drawn, int) or decl.max_drawn < 1
                                       or decl.max_drawn & (decl.max_drawn - 1)):
        out.append(f"max_drawn is {decl.max_drawn!r}: a Sobol sample is a power of two")
    return out


def _dimension_problems(d: Dimension, decl: TuningDecl) -> list[str]:
    out: list[str] = []
    lacks = [what for what, given in (("a plain label", d.label), ("a quiet term", d.term),
                                      ("a source", d.source)) if not str(given).strip()]
    if d.scale not in get_args(Scale):
        lacks.append(f"a scale among {list(get_args(Scale))}")
    if lacks:
        out.append(f"the dimension {d.name!r} needs {', '.join(lacks)} (C6)")
        return out
    if d.scale == "choice":
        if not d.choices:
            out.append(f"the dimension {d.name!r} is a choice with no choices")
    elif d.scale == "share_of_units":
        if not 0 < d.high <= 1:
            out.append(f"the dimension {d.name!r} is a share of units, so its high end is in "
                       f"(0, 1]")
    elif not d.low < d.high:
        out.append(f"the dimension {d.name!r} runs from {d.low} to {d.high}: low is below high")
    elif d.scale in ("log", "log_int") and d.low <= 0:
        out.append(f"the dimension {d.name!r} is on a log scale, so its low end is above 0")
    elif d.scale in ("int", "log_int") and (d.low != int(d.low) or d.high != int(d.high)):
        out.append(f"the dimension {d.name!r} is an integer, so its ends are integers")
    if d.active not in get_args(Activity):
        out.append(f"the dimension {d.name!r} is active {d.active!r}, not among "
                   f"{list(get_args(Activity))}")
    elif d.active == "without_early_stopping":
        stop = decl.early_stopping
        if stop is None or stop.get("param") != d.name:
            out.append(f"the dimension {d.name!r} is searched only without early stopping, so it "
                       f"is the early-stopping param of a declaration that stops early")
    if decl.kind == "path" and d in decl.dimensions:
        if not d.choices and (d.points < 2 or d.scale not in PATH_SCALES):
            out.append(f"the dimension {d.name!r} needs points or choices on a path: at least 2 "
                       f"points on a scale among {list(PATH_SCALES)}, or its choices")
    elif d.points:
        out.append(f"the dimension {d.name!r} has points, which only a path's dimensions have")
    return out


def tuning_for(family: Any, task: str) -> TuningDecl | None:
    """The family's declaration for ``task``: its ``tuning`` when that is one :class:`TuningDecl`
    (for every task it models), the entry for ``task`` when it is a per-task mapping (the elastic
    net's logistic grid), or None: nothing to tune for that task."""
    tuning = family.tuning
    if tuning is None:
        return None
    if isinstance(tuning, TuningDecl):
        return tuning if task in family.tasks else None
    return tuning.get(task)


# ═════════════════════════════════════════════════════════════════════════════
# the plan's parts
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class PathFit:
    """One split's whole grid, as a path family's ``path`` member returns it.

    The grid is :func:`path_grid`'s: its last dimension is the path itself (G points, the strongest
    penalty first); the dimensions before it multiply into M outer settings (the elastic net's
    mixes), in order, the first slowest. M is 1 for a single dimension. K is the number of outputs
    (1 for a number or a yes/no outcome; the classes otherwise), p the matrix's columns."""

    values: np.ndarray  # (M, G): the penalties used on these rows (a ratio times λ_max, say)
    coefs: np.ndarray  # (M, G, K, p)
    intercepts: np.ndarray  # (M, G, K)


@dataclass(frozen=True)
class Candidate:
    """One setting of every dimension the plan tries.

    * ``index``: its place in the plan; a tie in the pooled loss goes to the lower (:func:`choose`).
    * ``values``: unit-free, as the declaration states them: a share of units stays a share, a
      symbolic standard (``"default"``) stays symbolic, a path's penalty is its grid value (a ridge
      λ per row, an elastic net's ratio of λ_max). :func:`estimator_params` turns them into the
      estimator's parameters for one fit. Under the plan's early stopping, a Sobol candidate holds
      no value for the dimension active only without it: the engine sets the early-stopping
      ``param`` to its ``rounds``.
    * ``options``: each "Try both" slot's option for this candidate (RT-3; empty until then).
    * ``standard``: one of the standard candidates, which come first.

    A sequence value is held as a tuple (a list given becomes one), so the canonical form, where
    it is a list, reads back exactly; equal candidates hash alike (by their canonical JSON).
    """

    index: int
    values: Mapping[str, Any]
    options: Mapping[str, str] = field(default_factory=dict)
    standard: bool = False

    def __post_init__(self) -> None:
        # plain dicts, so the plan pickles (a MappingProxyType does not)
        object.__setattr__(self, "values", _held(self.values))
        object.__setattr__(self, "options", dict(self.options))

    def __hash__(self) -> int:
        return hash(_canonical(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {"index": int(self.index), "values": _plain(dict(self.values)),
                "options": _plain(dict(self.options)), "standard": bool(self.standard)}


@dataclass(frozen=True, eq=False)
class FitDesign:
    """The survey design of one fit's rows (fixes F15): each row's stratum, PSU and weight, aligned
    with the rows the fit is given. Under the population answer only the inner loss is weighted;
    models are fit unweighted, and inner splits keep whole PSUs."""

    strata: np.ndarray | None = None
    psu: np.ndarray | None = None
    weights: np.ndarray | None = None

    def take(self, rows: np.ndarray) -> "FitDesign":
        """The design of ``rows`` (positions into this design's rows), in that order."""
        rows = np.asarray(rows)
        return FitDesign(*(None if a is None else np.asarray(a)[rows]
                           for a in (self.strata, self.psu, self.weights)))


@dataclass(frozen=True)
class TuningPlan:
    """The one plan every fit of a family follows (RECIPES §4.2), computed once by
    :func:`make_plan` in the design stage, recorded and stated.

    * ``n_plan`` and ``unit``: the effective size of one outer training fold (:func:`plan_size`),
      in units, events (a yes/no outcome) or units of the rarest class.
    * ``plan_rows``: the rows of one outer training fold, for scikit-learn's own early-stopping
      rule on the standard candidates (``standard_stops``).
    * ``inner_k``: the inner folds every fit draws, after the floor (:func:`inner_k`); 0: no fit
      draws inner folds, and the first standard candidate is used (§4.6). With one candidate
      there is nothing to choose and no fit draws them either; :meth:`chooses` says which. A fit
      holding fewer than the floor keeps this K and draws its folds as evenly as its units allow.
    * ``early_stopping``: the Sobol candidates stop early (decided once: ``n_plan`` ≥
      :data:`STOP_FROM_N`); ``standard_stops``: the standard candidates do (plan rows above the
      declaration's ``standard_rows``, scikit-learn's 10,000 unless it says otherwise).
    * ``stop_share``: the stopping set's share of a fit's units, whole units, drawn first.
    * ``seed``: :func:`derive_seed` of (``split_seed``, family, ``space_version``); it seeds the
      candidates. The inner splits and stopping sets derive their own seeds from ``split_seed``
      and each fit's content keys (RT-1b).
    * ``options``: each "Try both" slot and the options it tries, in order (empty until RT-3).
    * ``manual``: values set by hand, held in every candidate (``mode="manual"``: the only
      candidate is the standard settings with them).
    * ``out_of_bag``: candidates are scored out of bag (fit once each, then the refit);
      ``weighted``: the inner loss is survey-weighted; ``imbalance``: every candidate is the
      wrapped, recalibrated model (each fit costs :data:`IMBALANCE_FITS`); ``threads``: the thread
      count every fit runs at.

    It is frozen, holds only plain values (it pickles; a sequence value as a tuple), and
    :meth:`to_dict` / :meth:`from_dict` give its canonical JSON-safe form, an exact round trip;
    equal plans hash alike (by that form's canonical JSON)."""

    family: str
    kind: TuningKind
    task: str
    loss: Loss
    strategy: StrategyName = "sobol"
    mode: Mode = "automatic"
    n_plan: int = 0
    unit: SizeUnit = "units"
    plan_rows: int = 0  # rows of one outer training fold (the standard 10,000 rule)
    inner_k: int = 0  # after the floor; 0 = no inner folds in any fit (§4.6)
    early_stopping: bool = False  # Sobol candidates, decided once (n_plan >= 1,500)
    standard_stops: bool = False  # standard candidates: plan_rows > 10,000
    stop_share: float = STOP_SHARE
    split_seed: int = 0
    seed: int = 0  # derive_seed(split_seed, family, space_version)
    space_version: str = ""
    defaults_version: str = "1"
    options: tuple[tuple[str, tuple[str, ...]], ...] = ()  # empty until RT-3
    manual: Mapping[str, Any] = field(default_factory=dict)
    candidates: tuple[Candidate, ...] = ()
    out_of_bag: bool = False
    weighted: bool = False
    imbalance: bool = False
    threads: int = 1

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"the tuning plan's strategy {self.strategy!r} is not among "
                             f"{list(STRATEGIES)} (seam guard 3: a strategy is added by name)")
        for name, value, allowed in (("kind", self.kind, TuningKind), ("mode", self.mode, Mode),
                                     ("loss", self.loss, Loss), ("unit", self.unit, SizeUnit)):
            if value not in get_args(allowed):
                raise ValueError(f"the tuning plan's {name} {value!r} is not among "
                                 f"{list(get_args(allowed))}")
        object.__setattr__(self, "options", tuple((str(slot), tuple(str(o) for o in opts))
                                                  for slot, opts in self.options))
        object.__setattr__(self, "manual", _held(self.manual))
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def __hash__(self) -> int:
        return hash(_canonical(self.to_dict()))

    def fits(self) -> int:
        """F: the fits one outer fit makes, in fits on that fit's rows, as the plan's strategy
        counts them (RECIPES §4.2)."""
        return STRATEGIES[self.strategy].fit_count(self)

    def chooses(self) -> bool:
        """Whether a fit scores candidates before its refit: more than one candidate, and inner
        folds (``inner_k`` ≥ 2) or out-of-bag scoring. When False there is nothing to choose (one
        candidate) or nothing to choose with (§4.6): no fit draws inner folds or scores out of
        bag, whatever ``inner_k`` holds, and the fit is the first candidate's refit alone
        (``fits()`` = w)."""
        return len(self.candidates) > 1 and (self.inner_k >= 2 or self.out_of_bag)

    def to_dict(self) -> dict[str, Any]:
        """The plan as canonical JSON-safe values: every field, candidates as dicts, options as
        ``[[slot, [option, …]], …]``; ``json.dumps(…, sort_keys=True)`` of it is canonical."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "candidates":
                out[f.name] = [c.to_dict() for c in value]
            elif f.name == "options":
                out[f.name] = [[slot, list(opts)] for slot, opts in value]
            else:
                out[f.name] = _plain(value)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TuningPlan":
        """The plan :meth:`to_dict` gave (after a JSON round trip too); an unknown key is
        refused, naming it."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"a tuning plan has no fields {unknown}")
        kw = dict(d)
        kw["candidates"] = tuple(Candidate(int(c["index"]), dict(c["values"]),
                                           dict(c.get("options") or {}), bool(c["standard"]))
                                 for c in d.get("candidates", ()))
        kw["options"] = tuple((slot, tuple(opts)) for slot, opts in d.get("options", ()))
        return cls(**kw)


def _tupled(value: Any) -> Any:
    """A setting's value with every list, tuple or array in it as a tuple (recursively), so a
    value read back from its canonical form (lists) equals the one written."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_tupled(v) for v in (value.tolist() if isinstance(value, np.ndarray)
                                           else value))
    return value


def _held(values: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _tupled(v) for k, v in dict(values).items()}


def _canonical(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _plain(value: Any) -> Any:
    """``value`` as JSON-safe Python: numpy scalars to numbers, tuples and arrays to lists."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ═════════════════════════════════════════════════════════════════════════════
# seeds, sizes and folds
# ═════════════════════════════════════════════════════════════════════════════


def derive_seed(*parts: Any) -> int:
    """A 32-bit seed from ``parts``, the same in every process and on every platform: the SHA-256
    of their canonical JSON (``json.dumps(list(parts), sort_keys=True, separators=(",", ":"),
    ensure_ascii=True)``, encoded UTF-8), its first four bytes read big-endian. Never Python's
    ``hash()``, which is salted per process. numpy scalars count as the numbers they hold."""
    text = json.dumps(_plain(list(parts)), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _unit_names(units: Any, n: int) -> np.ndarray:
    if units is None:
        return np.asarray([f"{i:012d}" for i in range(n)], dtype=object)
    from turbotab.core.models.inner_cv import unit_labels

    return unit_labels(units)


def effective_size(task: str, y: Any, units: Any = None) -> tuple[int, SizeUnit]:
    """n_eff (RECIPES §4.2) and its unit.

    A number (or a time to an event): the units, each repeated row of a unit counted once
    (``units`` per row; None: every row is its own unit). A yes/no outcome: the units in the rarer
    class, in ``"events"``; several classes: the units in the rarest class, in
    ``"rarest_class"``. Each unit counts once, by its most common class, as the stratified splits
    count it (``inner_cv``: pandas' ``mode``, a tie going to the smaller label as text). A class
    that no unit holds most often counts zero units."""
    y = np.asarray(y)
    names = _unit_names(units, len(y))
    if task in ("regression", "time_to_event"):
        return int(len(set(names.tolist()))), "units"
    from turbotab.core.models.inner_cv import _unit_label_of

    modal = _unit_label_of(names, y)
    counts = {str(c): 0 for c in np.unique(y.astype(str))} if len(y) else {}
    for cls in modal.values():
        counts[str(cls)] = counts.get(str(cls), 0) + 1
    rarest = min(counts.values()) if counts else 0
    return int(rarest), ("events" if task == "binary" else "rarest_class")


def plan_size(task: str, y: Any, units: Any, *, folds: int, order: Any = None
              ) -> tuple[int, int, SizeUnit]:
    """(n_plan, plan_rows, unit) of the headline split's training rows ``y`` (with ``units`` per
    row, or None) cut into ``folds`` outer folds (RECIPES §4.2).

    Without ``order``: n_plan is ⌊n_eff·(K − 1)/K⌋ and plan_rows ⌊rows·(K − 1)/K⌋, which is the
    smallest training fold's. With ``order`` (each row's time), the folds follow time as the split
    stage draws them (``stages.rows._assign_folds``: ``folds.forward_blocks`` over K + 1 blocks,
    each scored fold fit on every earlier block, a tie in time broken by the unit's key as text,
    the key being the row's position when ``units`` is None and the unit otherwise, so pass the
    unit keys the split stage was given): n_plan and plan_rows are the median training fold's, the
    lower median when the count is even, so each is a fold's own."""
    y = np.asarray(y)
    k = int(folds)
    if k < 2:
        raise ValueError(f"a plan needs at least 2 outer folds, not {k}")
    if order is None:
        n_eff, unit = effective_size(task, y, units)
        return n_eff * (k - 1) // k, len(y) * (k - 1) // k, unit
    from turbotab.core.models.folds import forward_blocks, forward_pairs

    # Keyed as the split stage keys them (``stages.rows._assign_folds``): each row by its position
    # when rows are units, else the unit itself. ``forward_blocks`` breaks a tie in time by the key
    # as text, so another keying (padded text) would draw other blocks when times tie.
    keys = np.arange(len(y)) if units is None else np.asarray(units, dtype=object)
    classify = task not in ("regression", "time_to_event")
    blocks, _, _ = forward_blocks(order, keys, k + 1, labels=y if classify else None)
    sizes = []
    for train, _ in forward_pairs(blocks):
        n_eff, unit = effective_size(task, y[train], None if units is None
                                     else np.asarray(units, dtype=object)[train])
        sizes.append((n_eff, len(train)))
    if not sizes:
        return 0, 0, effective_size(task, y[:0])[1]
    middle = (len(sizes) - 1) // 2
    n_plan = sorted(s[0] for s in sizes)[middle]
    rows = sorted(s[1] for s in sizes)[middle]
    return int(n_plan), int(rows), effective_size(task, y[:0])[1]


def path_folds(n_plan: int) -> int:
    """A path family's inner folds before the floor: 5 from 100 to 5,000 units, otherwise 3 (the
    elastic net's ``inner_folds`` rule, read on the plan's size)."""
    return 5 if 100 <= int(n_plan) <= 5000 else 3


def inner_k(kind: str, n_plan: int, *, rarest: int | None = None, psus: int | None = None) -> int:
    """The inner folds every fit draws (RECIPES §4.2): :data:`SEARCHED_K` for a search,
    :func:`path_folds` for a path, 0 for ``"none"``; then the one floor, every inner fold holding
    at least :data:`PER_FOLD_FLOOR` units of the rarest class (``rarest``: n_plan for a class
    outcome; None for a number) and, under the population answer, that many PSUs (``psus``: the
    fewest in any outer training fold the stages draw), so K = min(K, ⌊m/2⌋). Below 2 it is 0: no
    fit draws inner folds."""
    if kind == "search":
        k = SEARCHED_K
    elif kind == "path":
        k = path_folds(n_plan)
    else:
        return 0
    for m in (rarest, psus):
        if m is not None:
            k = min(k, int(m) // PER_FOLD_FLOOR)
    return k if k >= 2 else 0


def searched_size(decl: TuningDecl, *, n_plan: int, mode: str) -> int:
    """S, the Sobol candidates a search draws (RECIPES §4.2, §4.6): none below an effective size
    of :data:`SMALL_N`, 8 from there, 16 from :data:`MID_N`, never more than the declaration's
    ``max_drawn``; ``"lighter"`` halves it; ``"standard"`` and ``"manual"`` draw none. A path or a
    declaration of kind ``"none"`` draws none either."""
    if decl.kind != "search" or mode in ("standard", "manual"):
        return 0
    size = 0
    for start, s in SOBOL_SIZES:
        if n_plan >= start:
            size = s
    if decl.max_drawn is not None:
        size = min(size, decl.max_drawn)
    return size // 2 if mode == "lighter" else size


def sobol_sample(d: int, n: int, seed: int) -> np.ndarray:
    """(n, d) points of scipy's scrambled Sobol sequence: ``qmc.Sobol(d, scramble=True,
    rng=seed).random_base2(log2 n)`` (``seed=seed`` before scipy 1.15, the same sample). ``n`` is
    0 or a power of two (the sequence's balance holds at powers of two)."""
    n, d = int(n), int(d)
    if n == 0 or d == 0:
        return np.zeros((n, d))
    if n < 0 or n & (n - 1):
        raise ValueError(f"a Sobol sample is a power of two, not {n}")
    import inspect

    from scipy.stats import qmc

    # rng= from scipy 1.15, seed= before (RECIPES §4.7); both seed numpy.random.default_rng(seed)
    key = "rng" if "rng" in inspect.signature(qmc.Sobol).parameters else "seed"
    return qmc.Sobol(d, scramble=True, **{key: int(seed)}).random_base2(int(n).bit_length() - 1)


def map_unit(dim: Dimension, u: float, *, n_plan: int) -> Any:
    """``u`` in [0, 1) on ``dim``'s scale (RECIPES §4.1), as a plain Python number or choice:

    * ``linear``: ``low + u·(high − low)``;
    * ``log``: ``exp(log low + u·(log high − log low))``;
    * ``int``: ``⌊low + u·(high − low + 1)⌋``, at most ``high``: every integer equally likely;
    * ``log_int``: ``⌊exp(log low + u·(log(high + 1) − log low))⌋``, at most ``high``: the integer
      k with weight log((k + 1)/k);
    * ``choice``: ``choices[⌊u·m⌋]`` of its m choices (the last at most);
    * ``share_of_units``: log-uniform from one unit of the plan, ``1/n_plan``, to ``high``, kept as
      a share (``low`` is not read); the family's ``settings`` turns it into rows on each fit's own
      units.
    """
    u = float(u)
    lo, hi = float(dim.low), float(dim.high)
    if dim.scale == "linear":
        return lo + u * (hi - lo)
    if dim.scale == "log":
        return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    if dim.scale == "int":
        return min(int(hi), int(math.floor(lo + u * (hi - lo + 1))))
    if dim.scale == "log_int":
        return min(int(hi), int(math.floor(math.exp(
            math.log(lo) + u * (math.log(hi + 1) - math.log(lo))))))
    if dim.scale == "choice":
        m = len(dim.choices)
        return dim.choices[min(int(u * m), m - 1)]
    if dim.scale == "share_of_units":
        one = min(1.0 / max(int(n_plan), 1), hi)
        return math.exp(math.log(one) + u * (math.log(hi) - math.log(one)))
    raise ValueError(f"the dimension {dim.name!r} has no scale {dim.scale!r}")


def path_grid(decl: TuningDecl) -> dict[str, np.ndarray]:
    """A path family's grid, by dimension in declaration order: a choice's ``choices`` in order;
    otherwise ``points`` points spaced on its scale from ``high`` down to ``low``, so the strongest
    penalty comes first and wins a tie. The last dimension is the path (G points); the ones before
    it multiply into the M outer settings :class:`PathFit` is shaped by."""
    out: dict[str, np.ndarray] = {}
    for d in decl.dimensions:
        if d.choices:
            out[d.name] = np.asarray(d.choices)
        elif d.scale == "log":
            out[d.name] = np.geomspace(d.high, d.low, int(d.points))
        else:
            out[d.name] = np.linspace(d.high, d.low, int(d.points))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# the strategy
# ═════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Strategy(Protocol):
    """A way of choosing candidates (seam hook 1; V2X_SEAMS rows 1–2). The plan names its strategy
    by ``name``; :data:`STRATEGIES` finds it. A strategy owns its candidate list, its fit count and
    the order its candidates are evaluated in, so a v2.x strategy whose candidates depend on
    earlier scores (halving, TPE) records that order, and pinned replay needs no search."""

    name: str

    def candidates(self, decl: TuningDecl, *, task: str, n_plan: int, seed: int, mode: str,
                   manual: Mapping[str, Any], options: Sequence[tuple[str, Sequence[str]]],
                   early_stopping: bool, searched_size: int) -> tuple[Candidate, ...]:
        """Every candidate, in the plan's order (the standard ones first)."""

    def fit_count(self, plan: TuningPlan) -> int:
        """F, the fits one outer fit makes under ``plan``, in fits on that fit's rows."""

    def order(self, plan: TuningPlan) -> Sequence[int]:
        """The candidates' indices in the order they are evaluated."""


class SobolStrategy:
    """v2's strategy: plain random search over a scrambled Sobol sample, fixed before any score
    (RECIPES §4.2), and a path family's grid.

    **Candidates.** The standard settings come first, once per combination of the "Try both"
    options (the first slot slowest), with the values set by hand held. A search then adds
    ``searched_size`` Sobol candidates: one Sobol coordinate per free dimension (in declaration
    order, leaving out those set by hand and, under the plan's early stopping, those active only
    without it) and then one per "Try both" slot, which picks ``options[⌊u·m⌋]``. Each holds the
    standard settings that are not dimensions, its mapped values (:func:`map_unit`) and the values
    set by hand. A path's candidates are its grid (:func:`path_grid`, the first dimension slowest),
    once per combination of options, the options slowest.

    **Fit count** (RECIPES §4.2), with w = :data:`IMBALANCE_FITS` under the imbalance correction
    and 1 otherwise, C candidates and K inner folds: nothing to choose (C = 1, whatever the kind,
    out of bag or a path whose every dimension is set by hand; or K = 0) F = w, the refit alone,
    and no fit draws inner folds or scores out of bag; otherwise a search F = w·[(K − 1)·C + 1],
    out of bag F = C + 1, and a path F = w·[(K − 1)·r + 1], r the combinations of options (1
    without "Try both")."""

    name = "sobol"

    def candidates(self, decl: TuningDecl, *, task: str, n_plan: int, seed: int, mode: str,
                   manual: Mapping[str, Any], options: Sequence[tuple[str, Sequence[str]]],
                   early_stopping: bool, searched_size: int) -> tuple[Candidate, ...]:
        manual = dict(manual or {})
        slots = [str(slot) for slot, _ in options]
        combos = list(itertools.product(*[tuple(opts) for _, opts in options]))
        out: list[Candidate] = []
        if decl.kind == "path":
            grid = {k: v for k, v in path_grid(decl).items() if k not in manual}
            points = list(itertools.product(*[[_plain(v) for v in values]
                                              for values in grid.values()]))
            for combo in combos:
                for point in points:
                    out.append(Candidate(len(out), {**dict(zip(grid, point)), **manual},
                                         dict(zip(slots, combo))))
            return tuple(out)
        standard = {**decl.standard, **manual}
        for combo in combos:
            out.append(Candidate(len(out), dict(standard), dict(zip(slots, combo)), True))
        free = [d for d in decl.dimensions if d.name not in manual
                and (d.active == "always" or not early_stopping)]
        width = len(free) + len(slots)
        if decl.kind != "search" or not searched_size or not width:
            return tuple(out)
        searched = {d.name for d in decl.dimensions}
        rest = {k: v for k, v in decl.standard.items() if k not in searched}
        for row in sobol_sample(width, searched_size, seed):
            values = {d.name: map_unit(d, u, n_plan=n_plan) for d, u in zip(free, row)}
            chosen = {}
            for (slot, opts), u in zip(options, row[len(free):]):
                opts = tuple(opts)
                chosen[str(slot)] = opts[min(int(float(u) * len(opts)), len(opts) - 1)]
            out.append(Candidate(len(out), {**rest, **values, **manual}, chosen, False))
        return tuple(out)

    def fit_count(self, plan: TuningPlan) -> int:
        w = IMBALANCE_FITS if plan.imbalance else 1
        c = len(plan.candidates)
        if c <= 1:
            return w  # nothing to choose, whatever the kind: the refit alone
        if plan.out_of_bag:
            return c + 1
        if plan.inner_k < 2:
            return w
        if plan.kind == "path":
            r = math.prod(len(opts) for _, opts in plan.options) if plan.options else 1
            return w * ((plan.inner_k - 1) * r + 1)
        return w * ((plan.inner_k - 1) * c + 1)

    def order(self, plan: TuningPlan) -> Sequence[int]:
        return tuple(range(len(plan.candidates)))


STRATEGIES: Mapping[str, Strategy] = MappingProxyType({"sobol": SobolStrategy()})


# ═════════════════════════════════════════════════════════════════════════════
# the plan
# ═════════════════════════════════════════════════════════════════════════════


class NoInnerFolds(ValueError):
    """A path family's plan with too few units to draw inner folds: it cannot choose its penalty
    (RECIPES §4.6), so it is refused, saying why."""


_NOUNS = {"units": "units", "events": "events", "rarest_class": "units of the rarest class"}


def make_plan(family: Any, *, task: str, loss: str, n_plan: int, plan_rows: int, unit: str,
              split_seed: int, rarest: int | None = None, psus: int | None = None,
              mode: str = "automatic", manual: Mapping[str, Any] | None = None,
              options: Sequence[tuple[str, Sequence[str]]] = (), out_of_bag: bool = False,
              weighted: bool = False, imbalance: bool = False, threads: int = 1
              ) -> TuningPlan | None:
    """The one plan for ``family`` on ``task`` (RECIPES §4.2), or None when the family declares no
    tuning for the task (:func:`tuning_for`).

    ``n_plan``, ``plan_rows`` and ``unit`` come from :func:`plan_size`; ``rarest`` defaults to
    ``n_plan`` for a yes/no or class outcome; ``psus`` is the fewest PSUs in any outer training
    fold under the population answer. ``manual`` holds values set by hand (dimension or by-hand
    names only); ``options`` the "Try both" slots (RT-3). ``out_of_bag`` must be allowed by the
    declaration, and never comes with the imbalance correction.

    With no inner folds (K = 0, out of bag aside) the plan keeps only the first candidate: the
    first standard one, the first option of each slot. A path family there raises
    :class:`NoInnerFolds`, a stated refusal, unless every dimension is set by hand (nothing to
    choose)."""
    decl = tuning_for(family, task)
    if decl is None:
        return None
    if mode not in get_args(Mode):
        raise ValueError(f"the tuning mode {mode!r} is not among {list(get_args(Mode))}")
    manual = dict(manual or {})
    unknown = sorted(set(manual) - set(decl.names()))
    if unknown:
        raise ValueError(f"a value set by hand for {', '.join(map(repr, unknown))}, which "
                         f"{family.key} does not tune")
    if out_of_bag and not decl.out_of_bag:
        raise ValueError(f"{family.key} declares no out-of-bag scoring")
    if out_of_bag and imbalance:
        raise ValueError("out-of-bag scoring is never used with the imbalance correction")
    options = tuple((str(slot), tuple(str(o) for o in opts)) for slot, opts in options)
    if unit not in get_args(SizeUnit):
        raise ValueError(f"the plan's unit {unit!r} is not among {list(get_args(SizeUnit))}")
    if rarest is None and unit != "units":
        rarest = n_plan
    k = inner_k(decl.kind, n_plan, rarest=rarest, psus=psus)
    # a path whose every dimension is set by hand has nothing to choose, so needs no inner folds
    if decl.kind == "path" and k == 0 and any(d.name not in manual for d in decl.dimensions):
        short = (int(psus), "PSUs") if psus is not None and (
            rarest is None or int(psus) < int(rarest)) else (int(rarest or 0), _NOUNS[unit])
        raise NoInnerFolds(
            f"Too few to choose a penalty: each inner fold needs at least {PER_FOLD_FLOOR} "
            f"{short[1]}, and the plan's fit holds {short[0]:,}.")
    early = decl.kind == "search" and decl.early_stopping is not None and n_plan >= STOP_FROM_N
    standard_rows = (decl.early_stopping or {}).get(STANDARD_ROWS_KEY, STANDARD_STOP_ROWS)
    standard_stops = (decl.kind == "search" and decl.early_stopping is not None
                      and standard_rows is not None and plan_rows > int(standard_rows))
    scored = k >= 2 or out_of_bag
    size = searched_size(decl, n_plan=n_plan, mode=mode) if scored else 0
    seed = derive_seed(split_seed, family.key, decl.space_version)
    strategy = STRATEGIES["sobol"]
    candidates = strategy.candidates(decl, task=task, n_plan=n_plan, seed=seed, mode=mode,
                                     manual=manual, options=options, early_stopping=early,
                                     searched_size=size)
    if not scored:
        candidates = candidates[:1]
    return TuningPlan(
        family=family.key, kind=decl.kind, task=task, loss=loss, strategy=strategy.name,
        mode=mode, n_plan=int(n_plan), unit=unit, plan_rows=int(plan_rows),
        inner_k=0 if out_of_bag else k, early_stopping=early, standard_stops=standard_stops,
        stop_share=float((decl.early_stopping or {}).get("share", STOP_SHARE)),
        split_seed=int(split_seed), seed=seed, space_version=decl.space_version,
        defaults_version=str(family.defaults_version), options=options, manual=manual,
        candidates=candidates, out_of_bag=out_of_bag, weighted=weighted, imbalance=imbalance,
        threads=int(threads))


def estimator_params(family: Any, task: str, values: Mapping[str, Any], *, n_units: int,
                     n_rows: int, y: Any = None, Z: Any = None,
                     plan: TuningPlan | None = None) -> dict[str, Any]:
    """The estimator's own parameters for one fit at a candidate's ``values``: the declaration's
    ``fixed`` settings, then ``values``, passed through the family's ``settings`` member when it
    has one, called as ``settings(merged, task=, n_units=, n_rows=, y=, Z=, plan=)`` on the fit's
    own rows (its units and rows, its outcome, its model matrix after the head, and the plan). The
    early-stopping parameters are not here: the engine sets them from the plan."""
    decl = tuning_for(family, task)
    merged = {**(decl.fixed if decl is not None else {}), **dict(values)}
    if family.settings is None:
        return merged
    return dict(family.settings(merged, task=task, n_units=n_units, n_rows=n_rows, y=y, Z=Z,
                                plan=plan))


# ═════════════════════════════════════════════════════════════════════════════
# the score and the choice
# ═════════════════════════════════════════════════════════════════════════════


def pooled_loss(task: str, loss: str, y: Any, prediction: Any, *, classes: Sequence[Any] | None,
                weights: Any = None) -> float:
    """The pooled inner loss: Σ w·loss / Σ w over every inner validation row (``y`` and
    ``prediction`` concatenated across the inner folds), each row's loss from
    ``validation.loss_rows`` (squared error, log loss, ranked probability score). ``weights``: the
    survey weights under the population answer; None weighs every row alike."""
    from turbotab.core.models.validation import loss_rows

    rows = loss_rows(task, loss, y, prediction, classes=classes)
    if rows is None:
        raise ValueError(f"{loss!r} is not a mean of per-row losses")
    rows = np.asarray(rows, dtype=float)
    if weights is None:
        return float(np.mean(rows))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * rows) / np.sum(w))


def choose(losses: Any, precision: float = LOSS_PRECISION) -> int:
    """The flat index of the lowest loss, each rounded first to a multiple of ``precision`` times
    the smallest; equal rounded losses go to the lowest index (RECIPES_AND_TUNING §4.2, "Choice").
    A loss that is None or not finite never wins unless none is finite (then 0).

    Rounding makes a near-tie a tie, and a tie the earlier candidate's: the standard settings
    first, and on a penalty path the larger penalty. Without it, two losses closer than
    floating-point noise trade places with the order of a sum. (Moved here from
    ``elastic_net.lowest_rounded``, which re-exports it.)"""
    values = np.asarray(losses, dtype=float).ravel()
    finite = np.isfinite(values)
    if not finite.any():
        return 0
    scale = float(np.min(np.abs(values[finite]))) or 1.0
    rounded = np.where(finite, np.round(values / (precision * scale)), np.inf)
    return int(np.argmin(rounded))  # the first of equal minima


# ═════════════════════════════════════════════════════════════════════════════
# what a fit records
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class PathCurve:
    """A path family's pooled inner curve in one fit.

    ``names``: the grid's dimensions (:func:`path_grid`, the path last); ``grid``: (C, D), each
    grid point's values in candidate order (the first dimension slowest); ``losses``: (C,) pooled
    inner losses; ``chosen``: the chosen point's index on each dimension; ``at_edge``: the path's
    own dimension was chosen at its first or last point (a concern when it is so in more than half
    the folds, RECIPES §4.6)."""

    names: tuple[str, ...]
    grid: np.ndarray
    losses: np.ndarray
    chosen: tuple[int, ...]
    at_edge: bool


@dataclass(frozen=True)
class TuningRecord:
    """What one tuned fit records (``tuning_`` on a fitted ``TunedPipeline``; RECIPES §4.7).

    ``order``: the candidates in the order they were evaluated (V2X_SEAMS row 2); ``losses``: each
    candidate's pooled inner loss, by candidate index, None where not scored (V2X_SEAMS row 4, so
    the tuning curve needs no refit); ``chosen``: the chosen candidate's index; ``chosen_params``:
    the estimator's own parameters at the refit (what pinned replay holds); ``inner_k_used``: the
    inner folds drawn; ``below_floor``: this fit held fewer than the floor and drew its folds as
    evenly as its units allowed; ``libraries`` and ``threads``: versions and thread counts."""

    plan: TuningPlan
    libraries: Mapping[str, str]
    threads: Mapping[str, int]
    order: tuple[int, ...]
    losses: tuple[float | None, ...]
    chosen: int
    chosen_params: Mapping[str, Any]
    chosen_options: Mapping[str, str]
    path: PathCurve | None
    inner_k_used: int
    below_floor: bool
    n_fits: int
    seconds: float


__all__ = [
    "Activity", "Candidate", "Dimension", "EARLY_STOPPING_KEYS", "FitDesign", "IMBALANCE_FITS",
    "LOSS_PRECISION", "Loss", "MID_N", "Mode", "NoInnerFolds", "PER_FOLD_FLOOR", "PathCurve",
    "PathFit", "SEARCHED_K", "SMALL_N", "STANDARD_ROWS_KEY", "STANDARD_STOP_ROWS", "STOP_FROM_N",
    "STOP_SHARE", "STRATEGIES", "Scale", "SizeUnit", "SobolStrategy", "Strategy", "StrategyName",
    "TuningDecl", "TuningKind", "TuningPlan", "TuningRecord", "choose", "derive_seed",
    "effective_size", "estimator_params", "inner_k", "make_plan", "map_unit", "path_folds",
    "path_grid", "plan_size", "pooled_loss", "searched_size", "sobol_sample", "tuning_for",
    "tuning_problems",
]

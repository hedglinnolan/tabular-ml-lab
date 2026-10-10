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

**The engine (RT-1b)** reads a plan, nested in every fit (RECIPES §4.3):

* :func:`fit_parts`, the one per-fit helper, shared with the plain path of
  ``inner_cv.fit_pipeline``: on the rows it is given (the rows stay an argument, V2X_SEAMS row 1)
  it draws the stopping units first, fits the steps before the model on the other rows
  (:func:`fit_head`), draws every nested split those rows need and fits the model
  (:func:`fit_model`).
* :func:`inner_splits_for`: the inner splits, drawn inside each fit from its own rows as the
  outer ones are: whole units; forward chaining by unit under time; ``GroupKFold`` over
  stratum × PSU under the population answer, never stratified within strata; otherwise content
  keys shuffled by the seed, stratified by class where the splitter allows.
* :class:`TunedPipeline`: a ``Pipeline`` whose ``fit`` always searches when it carries a plan. One
  head per inner split is shared by every candidate; each candidate is scored on the pooled inner
  loss (survey-weighted under the population answer); the chosen one is refit on every row of the
  fit; ``tuning_`` holds the :class:`TuningRecord`. ``at`` returns a plain ``Pipeline`` at chosen
  settings (pinned refits, timing).
* :func:`center` (the candidate the time estimate times), :func:`cancel_scope` (checked before
  every candidate fit) and :func:`observing` (the test seam: every set the search draws).

Seeds (RECIPES §4.7): the candidates' from :func:`derive_seed` of (split seed, family, space
version); the inner splits' from :func:`derive_seed` of (split seed, ``"inner splits"``) over each
fit's own content keys or units; the stopping sets and every split nested in a candidate (a step's
``cv``, the imbalance correction's recalibration) are drawn at the split's seed, as
``inner_cv.fit_pipeline`` draws them, so pinned replay (``fit_pipeline(tuned.at(record
.chosen_params), …, seed=split seed)``) reproduces the deployed fit without searching.

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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import (Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence, get_args,
                    runtime_checkable)

import numpy as np
from sklearn.pipeline import Pipeline

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



# ═════════════════════════════════════════════════════════════════════════════
# the engine (RT-1b): the search, nested in every fit
# ═════════════════════════════════════════════════════════════════════════════

# The early-stopping interface's switch and stopping share (``inner_cv.stopping_setting``): set on
# the model step itself, a wrapper's own when it wraps the model (``ImbalanceCorrected``), so the
# fit hands it its stopping rows. Every other parameter reaches the wrapped estimator through the
# wrapper's ``inner_param``.
STOP_FLAG = "early_stopping"
STOP_SHARE_PARAM = "validation_fraction"
_STOP_PARAMS = (STOP_FLAG, STOP_SHARE_PARAM)
INNER_SPLITS = "inner splits"  # derive_seed(split seed, INNER_SPLITS): the inner splits' seed


class MissingPlan(ValueError):
    """A tuned family's pipeline asked for with no plan: every fit of a tuned family follows the
    one plan the design stage made (RECIPES §4.2), so no call site can skip the search."""


def _take(data: Any, rows: Any) -> Any:
    """``rows`` (positions) of ``data``: by position for a frame or series, else of the array."""
    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[np.asarray(rows)]
    return np.asarray(data)[np.asarray(rows)]


def _every(rows: np.ndarray, n: int) -> bool:
    return len(rows) == n and bool(np.array_equal(rows, np.arange(n)))


def _row_labels(data: Any, rows: Any) -> np.ndarray:
    """The rows' index labels (row ids) when ``data`` is a frame, else their positions."""
    rows = np.asarray(rows, dtype=np.int64)
    index = getattr(data, "index", None)
    return np.asarray(index)[rows] if index is not None else rows


@dataclass
class Head:
    """One fit's steps before the model (RECIPES §4.3, steps 1–3), fit on its rows less the
    stopping units.

    ``fitted``: those steps, fitted (None when the pipeline is the model alone). ``Z``, ``y``,
    ``groups``, ``order``: the rows the model trains on, after the steps. ``Z_stop``, ``y_stop``:
    the stopping rows, transformed by the fitted steps; None when the fit draws none. ``keys``: the
    training rows' content keys, when the rows carry no units or times (the nested splits are
    drawn over them). ``rows`` and ``stopping``: the positions, in the data the fit was handed, of
    the training rows and of the stopping rows."""

    fitted: Any | None
    Z: Any
    y: Any
    groups: Any
    order: Any
    Z_stop: Any | None
    y_stop: np.ndarray | None
    keys: np.ndarray | None = None
    rows: np.ndarray | None = None
    stopping: np.ndarray | None = None

    def transform(self, X: Any) -> Any:
        """``X`` through the fitted steps; as it is when there are none."""
        return X if self.fitted is None else self.fitted.transform(X)

    @property
    def n_units(self) -> int:
        """The units the model trains on: each unit once, every row its own without units."""
        if self.groups is None:
            return int(len(self.y))
        from turbotab.core.models.inner_cv import unit_labels

        return int(len(np.unique(unit_labels(self.groups))))


def fit_head(pipeline: Any, X: Any, y: Any, rows: Any, *, groups: Any = None, order: Any = None,
             seed: int, stop_share: float | None, classify: bool) -> Head:
    """RECIPES §4.3 steps 1–3 on ``rows`` (positions into ``X``; ``y``, ``groups`` and ``order``
    align with ``X``), with ``pipeline``'s own steps, fitted in place.

    1. The stopping units are drawn first: ``stop_share`` of the units, whole units, stratified by
       class when ``classify``, the latest when ``order`` is given (``inner_cv.validation_rows``,
       at ``seed``); none when ``stop_share`` is None or 0.
    2. Each step's own inner splits (a step with ``cv``) are drawn on the remaining rows, and the
       steps are fitted on them.
    3. The stopping rows are transformed by the fitted steps, so a step that reads the outcome
       never sees them (F11)."""
    from sklearn.pipeline import Pipeline

    from turbotab.core.models.inner_cv import (_steps_with_cv, row_keys, validation_rows,
                                               with_step_cv)

    rows = np.asarray(rows, dtype=np.int64)
    whole = _every(rows, len(y))
    X_r = X if whole else _take(X, rows)
    y_r = y if whole else _take(y, rows)
    y_arr = np.asarray(y_r)
    g_r = None if groups is None else np.asarray(groups)[rows]
    o_r = None if order is None else np.asarray(order)[rows]
    model = pipeline.steps[-1][1]
    share = float(stop_share or 0.0)
    nested = bool(_steps_with_cv(pipeline)) or "cv" in model.get_params(deep=False)
    keys = (row_keys(X_r, y_arr) if (share > 0 or nested) and g_r is None and o_r is None
            else None)
    held = rest = None
    if share > 0:
        mask = validation_rows(share, groups=g_r, keys=keys, order=o_r,
                               y=y_arr if classify else None, seed=seed)
        held, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
    if held is None:
        X_f, y_f, g_f, o_f, k_f = X_r, y_r, g_r, o_r, keys
    else:
        X_f, y_f = _take(X_r, rest), _take(y_r, rest)
        g_f, o_f, k_f = (None if a is None else a[rest] for a in (g_r, o_r, keys))
    with_step_cv(pipeline, groups=g_f, keys=k_f, order=o_f, y=np.asarray(y_f), seed=seed)
    head = (Pipeline(pipeline.steps[:-1], memory=pipeline.memory, verbose=pipeline.verbose)
            if len(pipeline.steps) > 1 else None)
    Z = X_f if head is None else head.fit_transform(X_f, y_f)
    Z_stop = y_stop = None
    if held is not None:
        X_s = _take(X_r, held)
        Z_stop = X_s if head is None else head.transform(X_s)
        y_stop = y_arr[held]
    return Head(fitted=head, Z=Z, y=y_f, groups=g_f, order=o_f, Z_stop=Z_stop, y_stop=y_stop,
                keys=k_f, rows=rows if held is None else rows[rest],
                stopping=None if held is None else rows[held])


def fit_model(model: Any, head: Head, *, seed: int) -> Any:
    """RECIPES §4.3 steps 3–4: the model's own nested splits (its ``cv``: the imbalance
    correction's recalibration, an inner cross-validation) drawn on the head's rows, as the outer
    folds are (whole units, time order, else content keys, at ``seed``), then the model fitted;
    a model that stops early (its switch on) is handed the head's stopping rows. Returns it."""
    from turbotab.core.models.inner_cv import stopping_setting, with_inner_cv

    with_inner_cv(Pipeline([("model", model)]), groups=head.groups, keys=head.keys,
                  order=head.order, y=np.asarray(head.y), seed=seed)
    if head.Z_stop is not None and stopping_setting(model)[0] is True:
        return model.fit(head.Z, head.y, X_val=head.Z_stop, y_val=head.y_stop)
    return model.fit(head.Z, head.y)


def fit_parts(pipeline: Any, X: Any, y: Any, rows: Any, *, groups: Any = None, order: Any = None,
              design: FitDesign | None = None, seed: int, stop_share: float | None = None) -> Any:
    """Fit ``pipeline`` (in place) on ``rows`` of ``X`` (RECIPES §4.3 steps 1–4): the stopping units
    first, the steps on the other rows, every nested split those rows need, then the model
    (:func:`fit_head`, :func:`fit_model`). The rows stay an argument (V2X_SEAMS row 1), so a
    strategy can hand it a subsample.

    ``stop_share``: None lets the model's own setting decide (``early_stopping`` True, or
    ``"auto"`` above 10,000 of these rows, scikit-learn's rule) at its ``validation_fraction``; a
    share draws that share and switches the model's early stopping on; 0 draws none. ``design``
    is accepted so that every fit receives it: the steps and the model are fit unweighted (only a
    search's inner loss is weighted), and a plain pipeline's own nested splits are drawn as the
    outer folds are, by units, time or content keys. Returns ``pipeline``."""
    from sklearn.base import is_classifier

    from turbotab.core.models.inner_cv import _stops_early, stopping_setting

    rows = np.asarray(rows, dtype=np.int64)
    model = pipeline.steps[-1][1]
    if stop_share is None:
        share = float(stopping_setting(model)[1]) if _stops_early(model, len(rows)) else 0.0
    else:
        share = float(stop_share)
    head = fit_head(pipeline, X, y, rows, groups=groups, order=order, seed=seed,
                    stop_share=share, classify=is_classifier(model))
    if head.Z_stop is not None:
        model.set_params(**{STOP_FLAG: True})
    fit_model(model, head, seed=seed)
    return pipeline


def inner_splits_for(plan: TuningPlan, X: Any, y: Any, rows: Any, *, groups: Any = None,
                     order: Any = None, design: FitDesign | None = None, seed: int
                     ) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """The plan's ``inner_k`` inner splits of ``rows`` (positions into ``X``; ``y``, ``groups``,
    ``order`` and ``design`` align with ``X``), as (training, validation) positions into ``X``;
    None when fewer than two units can be told apart.

    Drawn as the outer folds are (``inner_cv.inner_splits``): under the population answer
    (``design`` with PSUs) whole PSUs, ``GroupKFold`` over stratum × PSU, never stratified within
    strata, forward-chained when ``order`` is given; otherwise whole units when ``groups`` is given;
    forward chaining by whole unit when ``order`` is; else content keys (``inner_cv.row_keys``)
    shuffled by ``seed``. A class outcome keeps its classes in proportion where the splitter
    allows, except under the design. A fit with fewer units than ``inner_k`` draws as many folds
    as it has units."""
    from turbotab.core.models.inner_cv import inner_splits, row_keys

    rows = np.asarray(rows, dtype=np.int64)
    y_r = np.asarray(y)[rows]
    o_r = None if order is None else np.asarray(order)[rows]
    k = int(plan.inner_k)
    if design is not None and design.psu is not None:
        labels = _psu_labels(design)[rows]
        splits = inner_splits(labels, k, seed, order=o_r)
    else:
        g_r = None if groups is None else np.asarray(groups)[rows]
        keys = row_keys(_take(X, rows), y_r) if g_r is None and o_r is None else None
        classify = plan.task not in ("regression", "time_to_event")
        splits = inner_splits(g_r, k, seed, keys=keys, order=o_r, y=y_r if classify else None)
    if splits is None:
        return None
    return [(rows[np.asarray(a)], rows[np.asarray(b)]) for a, b in splits]


def _psu_labels(design: FitDesign) -> np.ndarray:
    """Each row's stratum × PSU label (a PSU is its label within its stratum)."""
    psu = np.asarray(design.psu, dtype=object)
    strata = (np.zeros(len(psu), dtype=object) if design.strata is None
              else np.asarray(design.strata, dtype=object))
    return np.asarray([f"{s}|{p}" for s, p in zip(strata, psu)], dtype=object)


def _below_floor(plan: TuningPlan, y: Any, groups: Any, design: FitDesign | None) -> bool:
    """Whether this fit holds fewer than :data:`PER_FOLD_FLOOR` per inner fold at the plan's K:
    of the rarest class's units (a class outcome), of PSUs (under the design), or fewer units than
    folds (RECIPES §4.2: it keeps the plan's K, and the record counts it)."""
    k = int(plan.inner_k)
    if k < 2:
        return False
    from turbotab.core.models.inner_cv import unit_labels

    n_units = len(np.unique(unit_labels(groups))) if groups is not None else len(np.asarray(y))
    if n_units < k:
        return True
    if design is not None and design.psu is not None:
        if len(np.unique(_psu_labels(design))) < PER_FOLD_FLOOR * k:
            return True
    if plan.task not in ("regression", "time_to_event"):
        rarest, _ = effective_size(plan.task, y, groups)
        if rarest < PER_FOLD_FLOOR * k:
            return True
    return False


# ── cancel and the test seam ──────────────────────────────────────────────────

_local = threading.local()
_OBSERVERS: list[Callable[["Drawn"], None]] = []
_OBSERVERS_LOCK = threading.Lock()


def _cancel_checks() -> list[Callable[[], bool]]:
    checks = getattr(_local, "cancel", None)
    if checks is None:
        checks = _local.cancel = []
    return checks


@contextmanager
def cancel_scope(check: Callable[[], bool]) -> Iterator[None]:
    """While open, every fit the search makes in this thread first asks ``check()`` (before each
    candidate fit, each head and the refit) and raises ``jobs.Cancelled`` once it says so: a
    pressed Cancel stops a search within one candidate fit (RECIPES §4.3: about 2 seconds)."""
    checks = _cancel_checks()
    checks.append(check)
    try:
        yield
    finally:
        checks.pop()


def _check_cancelled() -> None:
    for check in tuple(_cancel_checks()):
        if check():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()


@dataclass(frozen=True, eq=False)
class Drawn:
    """One set of rows the search drew (the test seam, :func:`observing`): ``kind`` is
    ``"inner"`` (an inner split, ``split`` its index), ``"out_of_bag"`` (the candidates' one fit)
    or ``"refit"`` (the chosen candidate on every row of the fit). ``train``, ``validation`` and
    ``stopping`` are the rows' index labels (row ids) when the fit was handed a frame, else their
    positions; ``validation`` is None outside an inner split, ``stopping`` when none was drawn."""

    plan: TuningPlan
    kind: str
    split: int | None
    train: np.ndarray
    validation: np.ndarray | None
    stopping: np.ndarray | None


@contextmanager
def observing(callback: Callable[[Drawn], None]) -> Iterator[None]:
    """The test seam (RECIPES T3, T17): while open, ``callback`` receives a :class:`Drawn` for
    every (training, validation, stopping) set every search draws, in any thread."""
    with _OBSERVERS_LOCK:
        _OBSERVERS.append(callback)
    try:
        yield
    finally:
        with _OBSERVERS_LOCK:
            _OBSERVERS.remove(callback)


def _emit(plan: TuningPlan, kind: str, split: int | None, X: Any, head: Head,
          validation: np.ndarray | None) -> None:
    with _OBSERVERS_LOCK:
        observers = tuple(_OBSERVERS)
    if not observers:
        return
    drawn = Drawn(plan, kind, split, _row_labels(X, head.rows),
                  None if validation is None else _row_labels(X, validation),
                  None if head.stopping is None else _row_labels(X, head.stopping))
    for callback in observers:
        callback(drawn)


# ── a candidate's parameters ──────────────────────────────────────────────────


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _stops(plan: TuningPlan, decl: TuningDecl, candidate: Candidate) -> bool:
    """Whether ``candidate`` stops early under ``plan``: a standard one by scikit-learn's rule
    (``standard_stops``), a Sobol one by the plan's ``early_stopping``, decided once."""
    if decl.early_stopping is None or plan.kind != "search":
        return False
    return plan.standard_stops if candidate.standard else plan.early_stopping


def _candidate_params(family: Any, plan: TuningPlan, decl: TuningDecl, candidate: Candidate, *,
                      n_units: int, n_rows: int, y: Any, Z: Any) -> dict[str, Any]:
    """The model step's parameters for one fit at ``candidate``: :func:`estimator_params` on the
    fit's own rows, then the early-stopping settings the plan decides. A Sobol candidate that
    stops runs to the declaration's ``rounds`` with its ``patience``; a standard one keeps its own
    (scikit-learn's ``"auto"``, decided by the plan's rows instead of each fit's: F13)."""
    params = estimator_params(family, plan.task, candidate.values, n_units=n_units,
                              n_rows=n_rows, y=y, Z=Z, plan=plan)
    stop = decl.early_stopping
    if stop is not None and plan.kind == "search":
        stops = _stops(plan, decl, candidate)
        if stops and not candidate.standard:
            params[stop["param"]] = stop["rounds"]
            params[stop["patience_param"]] = stop["patience"]
        params[STOP_FLAG] = bool(stops)
        if stops:
            params[STOP_SHARE_PARAM] = float(plan.stop_share)
    return {str(k): _scalar(v) for k, v in params.items()}


def _set_params(model: Any, params: Mapping[str, Any]) -> None:
    """``params`` on the model step: the early-stopping switch and share on the step itself (a
    wrapper's own), every other one on the estimator it wraps (``inner_param``: the wrapper's
    parameter holding it), else on the step. A switch the model does not have is left out: such a
    model is never handed stopping rows."""
    own = model.get_params(deep=False)
    inner = getattr(type(model), "inner_param", None)
    wrapped = own.get(inner) if inner else None
    wrapped_own = wrapped.get_params(deep=False) if wrapped is not None else {}
    direct, rest = {}, {}
    for k, v in params.items():
        if k in _STOP_PARAMS and k in own:
            direct[k] = v
        elif k in _STOP_PARAMS and not (inner and k in wrapped_own) and k not in own:
            continue
        else:
            rest[k] = v
    if inner:
        rest = {f"{inner}__{k}": v for k, v in rest.items()}
    if rest:
        model.set_params(**rest)
    if direct:
        model.set_params(**direct)


# ── the search ───────────────────────────────────────────────────────────────


def _predicted(task: str, model: Any, Z: Any, classes: np.ndarray | None) -> np.ndarray:
    """ŷ for a number; the probability of each of ``classes``, in that order, otherwise."""
    if task == "regression":
        return np.asarray(model.predict(Z), dtype=float)
    proba = np.asarray(model.predict_proba(Z), dtype=float)
    return _aligned(proba, list(model.classes_), classes)


def _aligned(proba: np.ndarray, have: Sequence[Any], classes: np.ndarray) -> np.ndarray:
    if len(have) == len(classes) and all(a == b for a, b in zip(have, classes)):
        return proba
    out = np.zeros((len(proba), len(classes)))
    where = {c: j for j, c in enumerate(classes.tolist())}
    for j, c in enumerate(have):
        out[:, where[_scalar(c)]] = proba[:, j]
    return out


def _from_scores(task: str, eta: np.ndarray, classes: np.ndarray | None) -> np.ndarray:
    """A path's linear scores (n, K) as predictions: the value for a number; for a yes/no outcome
    the log-odds of the second class in sorted order (as scikit-learn's ``coef_``); softmax over
    the classes otherwise."""
    if task == "regression":
        return eta[:, 0]
    if eta.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-eta[:, 0]))
        return np.column_stack([1.0 - p, p])
    e = np.exp(eta - eta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _oob(task: str, model: Any, classes: np.ndarray | None) -> np.ndarray:
    """The fitted model's out-of-bag predictions (scikit-learn's ``oob_prediction_`` or
    ``oob_decision_function_``), aligned with ``classes``."""
    if task == "regression":
        return np.asarray(model.oob_prediction_, dtype=float)
    return _aligned(np.asarray(model.oob_decision_function_, dtype=float),
                    list(model.classes_), classes)


@dataclass
class _Run:
    """What one tuned fit is doing: its plan, family and declaration, its data, and its count."""

    pipe: Any
    plan: TuningPlan
    family: Any
    decl: TuningDecl
    X: Any
    y: Any
    groups: Any
    order: Any
    design: FitDesign | None
    classes: np.ndarray | None
    n_fits: int = 0

    @property
    def classify(self) -> bool:
        return self.plan.task not in ("regression", "time_to_event")

    def weights(self) -> np.ndarray | None:
        if not self.plan.weighted or self.design is None or self.design.weights is None:
            return None
        return np.nan_to_num(np.asarray(self.design.weights, dtype=float), nan=0.0)

    def template(self) -> Any:
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline

        p = self.pipe
        return clone(Pipeline(p.steps, memory=p.memory, verbose=p.verbose))

    def head(self, pipeline: Any, rows: np.ndarray, stops: bool) -> Head:
        _check_cancelled()
        return fit_head(pipeline, self.X, self.y, rows, groups=self.groups, order=self.order,
                        seed=self.plan.split_seed,
                        stop_share=self.plan.stop_share if stops else None,
                        classify=self.classify)

    def fit(self, model: Any, candidate: Candidate, head: Head) -> dict[str, Any]:
        """``model`` fitted at ``candidate`` on ``head``; returns the parameters it was given."""
        _check_cancelled()
        params = _candidate_params(self.family, self.plan, self.decl, candidate,
                                   n_units=head.n_units, n_rows=len(head.y),
                                   y=np.asarray(head.y), Z=head.Z)
        _set_params(model, params)
        fit_model(model, head, seed=self.plan.split_seed)
        self.n_fits += 1
        return params


def _searched(run: _Run, splits: list[tuple[np.ndarray, np.ndarray]],
              order: Sequence[int]) -> list[float | None]:
    """Each candidate's pooled inner loss: one head per inner split, shared by every candidate,
    fit on the split's training rows less its stopping units (drawn whenever any candidate stops
    early; a candidate that does not stop trains on the same rows and ignores them)."""
    from sklearn.base import clone

    plan, cands = run.plan, run.plan.candidates
    stops = any(_stops(plan, run.decl, c) for c in cands)
    weights = run.weights()
    y_all = np.asarray(run.y)
    pooled_y, pooled_w = [], []
    predictions: dict[int, list[np.ndarray]] = {c: [] for c in order}
    for i, (train, validation) in enumerate(splits):
        template = run.template()
        head = run.head(template, train, stops)
        _emit(plan, "inner", i, run.X, head, validation)
        Z_val = head.transform(_take(run.X, validation))
        pooled_y.append(y_all[validation])
        if weights is not None:
            pooled_w.append(weights[validation])
        base = template.steps[-1][1]
        for c in order:
            model = clone(base)
            run.fit(model, cands[c], head)
            predictions[c].append(_predicted(plan.task, model, Z_val, run.classes))
    return _pooled(run, order, pooled_y, pooled_w, predictions)


def _pooled(run: _Run, order: Sequence[int], pooled_y: list, pooled_w: list,
            predictions: Mapping[int, list[np.ndarray]]) -> list[float | None]:
    plan = run.plan
    y = np.concatenate(pooled_y)
    w = np.concatenate(pooled_w) if pooled_w else None
    classes = None if run.classes is None else list(run.classes)
    losses: list[float | None] = [None] * len(plan.candidates)
    for c in order:
        losses[c] = pooled_loss(plan.task, plan.loss, y, np.concatenate(predictions[c]),
                                classes=classes, weights=w)
    return losses


def _path_grid(plan: TuningPlan, decl: TuningDecl) -> dict[str, np.ndarray]:
    """The path's grid with each value set by hand held at it (one point), in declaration order,
    so the grid's flat order is the candidates' (the first dimension slowest)."""
    return {name: (np.asarray([plan.manual[name]]) if name in plan.manual else values)
            for name, values in path_grid(decl).items()}


def _path_searched(run: _Run, splits: list[tuple[np.ndarray, np.ndarray]]
                   ) -> tuple[list[float | None], tuple[int, ...]]:
    """Each grid point's pooled inner loss: per inner split, the head refit on its training rows,
    the family's whole path on them (``family.path``), and every point scored on the split's
    validation rows."""
    plan = run.plan
    grid = _path_grid(plan, run.decl)
    shape = tuple(len(v) for v in grid.values())
    if math.prod(shape) != len(plan.candidates):
        raise ValueError(f"the plan holds {len(plan.candidates)} candidates for a grid of "
                         f"{math.prod(shape)} points: it was made for another declaration")
    weights = run.weights()
    y_all = np.asarray(run.y)
    pooled_y, pooled_w = [], []
    order = tuple(range(len(plan.candidates)))
    predictions: dict[int, list[np.ndarray]] = {c: [] for c in order}
    for i, (train, validation) in enumerate(splits):
        head = run.head(run.template(), train, False)
        _emit(plan, "inner", i, run.X, head, validation)
        _check_cancelled()
        fitted = run.family.path(head.Z, np.asarray(head.y), grid, task=plan.task, weights=None)
        run.n_fits += 1
        Z_val = np.asarray(head.transform(_take(run.X, validation)), dtype=float)
        coefs = np.asarray(fitted.coefs, dtype=float)
        k, p = coefs.shape[-2], coefs.shape[-1]
        eta = (Z_val @ coefs.reshape(-1, p).T).reshape(len(Z_val), -1, k)
        eta = eta + np.asarray(fitted.intercepts, dtype=float).reshape(1, -1, k)
        pooled_y.append(y_all[validation])
        if weights is not None:
            pooled_w.append(weights[validation])
        for c in order:
            predictions[c].append(_from_scores(plan.task, eta[:, c, :], run.classes))
    return _pooled(run, order, pooled_y, pooled_w, predictions), shape


def _out_of_bag(run: _Run, order: Sequence[int]) -> list[float | None]:
    """Each candidate fit once on every row of the fit and scored on its out-of-bag predictions
    (tuneRanger's method; the plan allows it only where no step reads the outcome)."""
    from sklearn.base import clone

    plan = run.plan
    template = run.template()
    rows = np.arange(len(np.asarray(run.y)))
    head = run.head(template, rows, False)
    _emit(plan, "out_of_bag", None, run.X, head, None)
    y = np.asarray(head.y)
    classes = None if run.classes is None else list(run.classes)
    losses: list[float | None] = [None] * len(plan.candidates)
    base = template.steps[-1][1]
    for c in order:
        model = clone(base)
        run.fit(model, plan.candidates[c], head)
        oob = _oob(plan.task, model, run.classes)
        ok = np.isfinite(oob) if oob.ndim == 1 else np.isfinite(oob).all(axis=1)
        if ok.any():
            losses[c] = pooled_loss(plan.task, plan.loss, y[ok], oob[ok], classes=classes)
    return losses


def _path_curve(plan: TuningPlan, decl: TuningDecl, losses: Sequence[float | None],
                shape: tuple[int, ...], chosen: int) -> PathCurve:
    grid = _path_grid(plan, decl)
    names = tuple(grid)
    rows = [[c.values[n] for n in names] for c in plan.candidates]
    try:
        values = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        values = np.asarray(rows, dtype=object)
    at = tuple(int(i) for i in np.unravel_index(int(chosen), shape))
    return PathCurve(names=names, grid=values,
                     losses=np.asarray([np.nan if v is None else v for v in losses], dtype=float),
                     chosen=at, at_edge=shape[-1] > 1 and at[-1] in (0, shape[-1] - 1))


def _libraries(model: Any) -> dict[str, str]:
    import sys

    import scipy
    import sklearn

    out = {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    root = type(model).__module__.split(".")[0]
    version = getattr(sys.modules.get(root), "__version__", None)
    if root not in ("sklearn", "numpy", "scipy", "turbotab") and isinstance(version, str):
        out[root] = version
    return out


def _threads(plan: TuningPlan) -> dict[str, int]:
    out = {"plan": int(plan.threads)}
    try:
        from threadpoolctl import threadpool_info

        for pool in threadpool_info():
            out[str(pool.get("internal_api") or pool.get("user_api"))] = int(pool["num_threads"])
    except Exception:  # noqa: BLE001 - the record still holds the plan's count
        pass
    return out


def _tuned_fit(pipe: "TunedPipeline", X: Any, y: Any, *, groups: Any, order: Any,
               design: FitDesign | None) -> "TunedPipeline":
    """The search and the refit (RECIPES §4.2–§4.3, §4.7), in place on ``pipe``."""
    from turbotab.core.models.base import get_family

    started = time.perf_counter()
    plan = pipe.search
    family = get_family(plan.family)
    decl = tuning_for(family, plan.task)
    if decl is None:
        raise ValueError(f"{family.key} declares no tuning for {plan.task}, so it has no plan")
    if plan.options:
        raise ValueError("a \"Try both\" slot needs one head per option, which the recipes bring "
                         "(RT-3): this plan's options cannot be searched yet")
    y_arr = np.asarray(y)
    n = len(y_arr)
    if design is not None:
        for name in ("strata", "psu", "weights"):
            values = getattr(design, name)
            if values is not None and len(values) != n:
                raise ValueError(f"the fit's design gives {len(values)} {name} for {n} rows")
    classify = plan.task not in ("regression", "time_to_event")
    run = _Run(pipe, plan, family, decl, X, y, groups, order, design,
               np.unique(y_arr) if classify else None)
    order_eval = tuple(int(i) for i in STRATEGIES[plan.strategy].order(plan))
    losses: list[float | None] = [None] * len(plan.candidates)
    chosen, k_used, below, curve = 0, 0, False, None
    every = np.arange(n)
    if plan.chooses():
        if plan.out_of_bag:
            losses = _out_of_bag(run, order_eval)
        else:
            below = _below_floor(plan, y_arr, groups, design)
            splits = inner_splits_for(plan, X, y_arr, every, groups=groups, order=order,
                                      design=design,
                                      seed=derive_seed(plan.split_seed, INNER_SPLITS))
            if splits is None or len(splits) < 2:
                below = True
            else:
                k_used = len(splits)
                if plan.kind == "path":
                    losses, shape = _path_searched(run, splits)
                else:
                    losses = _searched(run, splits, order_eval)
        if any(v is not None for v in losses):
            chosen = choose(losses)
        if plan.kind == "path" and k_used:
            curve = _path_curve(plan, decl, losses, shape, chosen)
    candidate = plan.candidates[chosen]
    plain = _plain_pipeline(pipe)
    head = run.head(plain, every, _stops(plan, decl, candidate))
    _emit(plan, "refit", None, X, head, None)
    params = run.fit(pipe.steps[-1][1], candidate, head)
    pipe.tuning_ = TuningRecord(
        plan=plan, libraries=_libraries(pipe.steps[-1][1]), threads=_threads(plan),
        order=order_eval, losses=tuple(losses), chosen=int(chosen), chosen_params=params,
        chosen_options=dict(candidate.options), path=curve, inner_k_used=int(k_used),
        below_floor=bool(below), n_fits=int(run.n_fits),
        seconds=float(time.perf_counter() - started))
    return pipe


def _plain_pipeline(pipe: Any) -> Any:
    """A plain ``Pipeline`` over ``pipe``'s own step objects (fitting it fits them)."""
    from sklearn.pipeline import Pipeline

    return Pipeline(pipe.steps, memory=pipe.memory, verbose=pipe.verbose)


class TunedPipeline(Pipeline):
    """A family's pipeline with its plan (RECIPES §4.3): ``search``, the one
    :class:`TuningPlan` every fit of the family follows.

    ``fit`` always searches when ``search`` is set: the candidates scored on inner splits drawn from
    the rows it is given (:func:`inner_splits_for`; out of bag where the plan says), the lowest
    pooled inner loss chosen (:func:`choose`), and the chosen candidate refit on every row
    (:func:`fit_parts`' steps), with the record in ``tuning_``. ``search=None`` (the default, so
    scikit-learn's slicing and cloning, which rebuild with ``self.__class__(steps, …)``, work)
    behaves as a plain ``Pipeline``. A direct ``.fit(X, y)`` draws its splits from row keys;
    ``inner_cv.fit_pipeline`` passes the units, times and design. ``at`` returns a plain
    ``Pipeline`` at chosen settings."""

    def __init__(self, steps: Any, *, search: TuningPlan | None = None, transform_input: Any = None,
                 memory: Any = None, verbose: bool = False):
        super().__init__(steps, transform_input=transform_input, memory=memory, verbose=verbose)
        self.search = search

    def fit(self, X: Any, y: Any = None, *, groups: Any = None, order: Any = None,
            design: FitDesign | None = None, **params: Any) -> "TunedPipeline":
        if self.search is None:
            return super().fit(X, y, **params)
        if params:
            raise TypeError(f"a tuned fit takes no fit parameters, not {sorted(params)}")
        return _tuned_fit(self, X, y, groups=groups, order=order, design=design)

    def at(self, chosen: Candidate | Mapping[str, Any], *, units: int | None = None) -> Any:
        """A plain, unfitted ``Pipeline`` of fresh copies of these steps with the model step at
        ``chosen``: a fitted search's ``tuning_.chosen_params`` (the estimator's own parameters at
        the refit, held as they are: pinned refits, RECIPES §4.3), or a :class:`Candidate`, whose
        values are resolved for ``units`` units (the plan's own size by default) with no outcome
        or matrix (``settings`` is called with ``y`` and ``Z`` None), as the time estimate times
        :func:`center`."""
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline

        plan = self.search
        if plan is None:
            raise ValueError("this pipeline carries no plan, so it has no settings to fix")
        steps = [(name, clone(step) if hasattr(step, "get_params") else step)
                 for name, step in self.steps]
        if isinstance(chosen, Candidate):
            from turbotab.core.models.base import get_family

            family = get_family(plan.family)
            decl = tuning_for(family, plan.task)
            n_units = int(units) if units is not None else int(plan.n_plan)
            n_rows = int(units) if units is not None else int(plan.plan_rows or plan.n_plan)
            params = _candidate_params(family, plan, decl, chosen, n_units=n_units,
                                       n_rows=n_rows, y=None, Z=None)
        else:
            params = dict(chosen)
        _set_params(steps[-1][1], params)
        return Pipeline(steps, transform_input=self.transform_input, memory=self.memory,
                        verbose=self.verbose)


def center(plan: TuningPlan) -> Candidate:
    """The candidate the time estimate times (RECIPES §4.4): every searched dimension at the
    midpoint of its scale (:func:`map_unit` at ½; a dimension active only without early stopping
    is left to the plan's rounds when it stops early), a path's middle grid point, the standard
    settings for a declaration of kind ``"none"``, values set by hand held, and the first option
    of each "Try both" slot. Its index is −1: it is not one of the plan's candidates."""
    from turbotab.core.models.base import get_family

    decl = tuning_for(get_family(plan.family), plan.task)
    if decl is None:
        raise ValueError(f"{plan.family} declares no tuning for {plan.task}")
    options = {slot: opts[0] for slot, opts in plan.options if opts}
    manual = dict(plan.manual)
    if plan.kind == "path":
        values = {name: _plain(v[len(v) // 2]) for name, v in path_grid(decl).items()
                  if name not in manual}
        return Candidate(-1, {**values, **manual}, options, False)
    if plan.kind == "none":
        return Candidate(-1, {**decl.standard, **manual}, options, True)
    searched = {d.name for d in decl.dimensions}
    rest = {k: v for k, v in decl.standard.items() if k not in searched}
    values = {d.name: map_unit(d, 0.5, n_plan=plan.n_plan) for d in decl.dimensions
              if d.name not in manual and (d.active == "always" or not plan.early_stopping)}
    return Candidate(-1, {**rest, **values, **manual}, options, False)


__all__ = [
    "Activity", "Candidate", "Dimension", "Drawn", "EARLY_STOPPING_KEYS", "FitDesign", "Head",
    "IMBALANCE_FITS", "INNER_SPLITS", "LOSS_PRECISION", "Loss", "MID_N", "MissingPlan", "Mode",
    "NoInnerFolds", "PER_FOLD_FLOOR", "PathCurve", "PathFit", "SEARCHED_K", "SMALL_N",
    "STANDARD_ROWS_KEY", "STANDARD_STOP_ROWS", "STOP_FLAG", "STOP_FROM_N", "STOP_SHARE",
    "STOP_SHARE_PARAM", "STRATEGIES", "Scale", "SizeUnit", "SobolStrategy", "Strategy",
    "StrategyName", "TunedPipeline", "TuningDecl", "TuningKind", "TuningPlan", "TuningRecord",
    "cancel_scope", "center", "choose", "derive_seed", "effective_size", "estimator_params",
    "fit_head", "fit_model", "fit_parts", "inner_k", "inner_splits_for", "make_plan", "map_unit",
    "observing", "path_folds", "path_grid", "plan_size", "pooled_loss", "searched_size",
    "sobol_sample", "tuning_for", "tuning_problems",
]

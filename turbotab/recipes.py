"""turbotab.recipes — per-model preprocessing defaults, as a table.

**model × operation → variant + reason, resolved at runtime.**

Built as data rather than as conditionals for one reason, stated before any
domain pack exists so it can be tested before it is needed: *the first pack that
lands would otherwise mean rewriting the branches.* `DOMAIN_PACKS.md` §02 says a
pack supplies detectors, reference data, conventions and prose, and **never
interface** — which only holds if the defaults it adjusts are a lookup.

A pack must be able to add **operations**, not merely override variants.
Sample-level normalization (PQN), batch correction, detection-limit imputation
and energy adjustment are operations the generic catalogue does not have. Each
still passes clause §06's litmus to be classified row-local or deferred; a pack
does not get to skip that.

──────────────────────────────────────────────────────────────────────────────
THE SPLIT, from the constitution
──────────────────────────────────────────────────────────────────────────────

**Model-determined is a FACT.** A linear model, an SVM, a KNN and a neural net
need scaled inputs; a tree does not. That is a property of the model, true
regardless of the dataset — so it may be **pre-selected at high confidence with
its reason shown and a rendered skip**, which is exactly what the routing
constitution permits for a question of fact.

**Data-determined is a CHOICE and stays asked.** *"Your outcome is skewed,
transform it?"* is a judgment about this data, and no confidence in the engine
makes a judgment moot.

The determinacy lives on the **operation**, because it is a property of the
question rather than of the answer. Scaling is model-determined whatever variant
you pick; a power transform is data-determined whatever model you pick.

──────────────────────────────────────────────────────────────────────────────
ASK ONLY WHEN THE CHOICE CHANGES THE ANSWER
──────────────────────────────────────────────────────────────────────────────

Standard versus robust scaling on well-behaved data produces near-identical
matrices — asking is ceremony. On heavy-tailed data they diverge — asking is
essential. That is computable, and `divergence()` computes it.

The measure for scaling is deliberately not "are the numbers different" (they
always are, by a constant) but **do the two rescale the columns differently
RELATIVE TO EACH OTHER.** Standard divides by σ, robust by IQR. If σ/IQR is
near-constant across columns, the two differ by one global factor and no
scale-equivariant model can tell them apart. If it varies, they reweight the
features against one another and the choice is real. The spread of σ/IQR across
columns is therefore the statistic, and it is reported as evidence rather than
asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Determinacy — the routing constitution's two kinds, named here so a new
# operation has to declare which it is.
MODEL_DETERMINED = "model_determined"      # FACT: pre-selectable, rendered skip
DATA_DETERMINED = "data_determined"        # CHOICE: always asked

# Clause §06's two dispositions, carried on the operation so a pack cannot add
# one without answering the litmus.
ROW_LOCAL = "row_local"
STATEFUL = "stateful"

CORE = "core"


class RecipeError(Exception):
    """The recipe table was asked for something it cannot honestly answer."""


@dataclass(frozen=True)
class Operation:
    """One preprocessing operation and its named variants."""
    key: str
    label: str
    variants: Tuple[str, ...]
    determinacy: str
    scope: str                      # clause §06
    because: str                    # the litmus answer, in words
    applies_to: str = "numeric"     # numeric | categorical | any
    origin: str = CORE              # `core`, or the pack that contributed it
    # Ordered pairs `(variant, alternative)`: given that the table resolved to
    # `variant`, which alternative is worth PUSHING. This is `measure.py`'s own
    # distinction — *push the notable, pull the rest* — applied to variants.
    # Every variant stays selectable from the pull side; this names the one the
    # app raises on its own initiative, and only when the data says the two
    # would differ.
    pushed_alternatives: Tuple[Tuple[str, str], ...] = ()

    def alternatives_for(self, variant: str) -> Tuple[str, ...]:
        return tuple(b for a, b in self.pushed_alternatives if a == variant)

    def __post_init__(self) -> None:
        if self.determinacy not in (MODEL_DETERMINED, DATA_DETERMINED):
            raise RecipeError(
                f"{self.key}: determinacy must be {MODEL_DETERMINED!r} or "
                f"{DATA_DETERMINED!r}. An operation that does not say which is "
                f"an operation nobody decided how to route.")
        if self.scope not in (ROW_LOCAL, STATEFUL):
            raise RecipeError(
                f"{self.key}: scope must be {ROW_LOCAL!r} or {STATEFUL!r}. "
                f"Clause §06's litmus is not optional for a pack.")
        if len(self.because) <= 40:
            raise RecipeError(
                f"{self.key}: `because` must state the litmus answer. A scope "
                f"with no reason is a classification nobody can check.")


@dataclass(frozen=True)
class Default:
    """One row of the table: for this selector, this operation gets this variant."""
    operation: str
    variant: str
    reason: str
    # `*` (every model), a registry group (`Linear`, `Trees`, …), a capability
    # flag (`caps:requires_scaled_numeric`), or an exact model key. Specificity
    # decides; see `_SPECIFICITY`.
    selector: str = "*"
    origin: str = CORE


# Most specific wins. A pack overriding one model must not be outranked by a
# core rule about every model, and a core rule about an exact model must not be
# silently replaced by a pack's rule about a whole group.
_SPECIFICITY = {"*": 0, "group": 1, "caps": 2, "model": 3}


def _selector_kind(selector: str) -> str:
    if selector == "*":
        return "*"
    if selector.startswith("caps:"):
        return "caps"
    if selector.startswith("group:"):
        return "group"
    return "model"


# ─────────────────────────────────────────────────────────────────────────────
# The core catalogue
# ─────────────────────────────────────────────────────────────────────────────

_OPERATIONS: Dict[str, Operation] = {}
_DEFAULTS: List[Default] = []


def register_operation(op: Operation, *, replace_existing: bool = False) -> None:
    """Add an operation. The extension point a pack uses.

    Refuses a silent overwrite: a pack that shadows a core operation without
    saying so is a pack changing behavior nobody can see, which is
    `DOMAIN_PACKS.md` §05's *"a pack that fires on the wrong data asserts
    something false"* in its quietest form.
    """
    if op.key in _OPERATIONS and not replace_existing:
        raise RecipeError(
            f"operation {op.key!r} already exists (from "
            f"{_OPERATIONS[op.key].origin!r}). Pass replace_existing=True to "
            f"shadow it deliberately.")
    _OPERATIONS[op.key] = op


def register_default(d: Default) -> None:
    """Add a table row. A pack overriding a variant does it here."""
    if d.operation not in _OPERATIONS:
        raise RecipeError(
            f"default for unknown operation {d.operation!r}. Register the "
            f"operation first — a default with no operation is a row nothing "
            f"will ever read.")
    op = _OPERATIONS[d.operation]
    if d.variant not in op.variants:
        raise RecipeError(
            f"{d.operation}: {d.variant!r} is not one of {list(op.variants)}.")
    if len(d.reason) <= 30:
        raise RecipeError(
            f"{d.operation}/{d.selector}: a default states its reason. That "
            f"reason is what the rendered skip shows the user, so a default "
            f"without one cannot be skipped honestly.")
    _DEFAULTS.append(d)


def defaults(origin: Optional[str] = None) -> List[Default]:
    """Every row of the table, in registration order.

    Exists so a caller can ask WHO contributed a row rather than keeping a
    second copy of what it registered. `packs.recipe_origins` reads this; a
    mirror in `packs.py` would be the drift `GUIDED-025` is about, one level
    down from the drift it names.
    """
    return [d for d in _DEFAULTS if origin is None or d.origin == origin]


def operations(origin: Optional[str] = None) -> List[Operation]:
    out = sorted(_OPERATIONS.values(), key=lambda o: o.key)
    return [o for o in out if origin is None or o.origin == origin]


def operation(key: str) -> Operation:
    if key not in _OPERATIONS:
        raise RecipeError(f"no operation {key!r}. Known: "
                          f"{', '.join(sorted(_OPERATIONS))}.")
    return _OPERATIONS[key]


def _install_core() -> None:
    _OPERATIONS.clear()
    _DEFAULTS.clear()

    register_operation(Operation(
        key="scale", label="Feature scaling",
        variants=("standard", "robust", "minmax", "none"),
        determinacy=MODEL_DETERMINED, scope=STATEFUL,
        because=("Stateful: a mean, a standard deviation, a median and an IQR "
                 "are all facts about the whole column, so each is fitted "
                 "inside the training folds."),
        applies_to="numeric",
        # WHETHER to scale is the model's property and the table settles it.
        # WHICH scaling, once it happens, is a judgment about this data — so
        # standard and robust are pushed against each other, and only when they
        # would put the columns on different relative footings. `minmax` stays
        # available and is never raised unprompted; nothing here measures it.
        pushed_alternatives=(("standard", "robust"), ("robust", "standard"))))
    register_operation(Operation(
        key="encode", label="Categorical encoding",
        variants=("onehot", "ordinal", "target", "none"),
        determinacy=MODEL_DETERMINED, scope=STATEFUL,
        because=("Stateful: the set of levels, and their order or target means, "
                 "are properties of the rows the encoder saw."),
        applies_to="categorical"))
    register_operation(Operation(
        key="power", label="Power transform",
        variants=("none", "log1p", "yeo_johnson"),
        determinacy=DATA_DETERMINED, scope=STATEFUL,
        because=("Yeo-Johnson fits lambda from the column, so it is stateful. "
                 "log1p is row-local and lives in the Features catalogue; here "
                 "it is offered as the null-lambda case for continuity."),
        applies_to="numeric"))
    register_operation(Operation(
        key="outliers", label="Outlier handling",
        variants=("none", "winsorize", "mad"),
        determinacy=DATA_DETERMINED, scope=STATEFUL,
        because=("Stateful: percentile cut-points and a median absolute "
                 "deviation are both computed across rows."),
        applies_to="numeric"))

    # ── the model-determined layer, seeded FROM THE REGISTRY ────────────────
    # Generated rather than typed out, because `requires_scaled_numeric` is
    # already a declared capability on every model spec. Hand-listing 22 models
    # would be a second copy of a fact that already exists, and second copies
    # are what this project keeps finding to have drifted.
    register_default(Default(
        operation="scale", variant="none", selector="*",
        reason=("Tree-based and rule-based models split on order rather than "
                "on distance, so rescaling a column changes nothing they can "
                "see. Scaling them is harmless and pointless.")))
    register_default(Default(
        operation="scale", variant="standard",
        selector="caps:requires_scaled_numeric",
        reason=("This model measures distances or penalizes coefficients, so a "
                "column measured in thousands would dominate one measured in "
                "units purely because of its scale. The registry records this "
                "as a property of the model, not of your data.")))
    register_default(Default(
        operation="encode", variant="onehot", selector="*",
        reason=("One column per level, so no ordering is implied between "
                "categories that have none.")))
    register_default(Default(
        operation="encode", variant="ordinal", selector="group:Trees",
        reason=("A tree can split an integer-coded category at any point, so it "
                "recovers groupings one-hot would have to spend a column each "
                "on — and at high cardinality one-hot starves the splits.")))
    register_default(Default(
        operation="encode", variant="ordinal", selector="group:Boosting",
        reason=("Same as trees: boosted ensembles split on order and handle "
                "integer-coded categories directly, so one-hot costs width "
                "without buying separation.")))
    register_default(Default(
        operation="power", variant="none", selector="*",
        reason=("Transforming is a judgment about THIS data — whether the "
                "skew is a measurement artifact or the phenomenon — so nothing "
                "is applied unless you say so.")))
    register_default(Default(
        operation="outliers", variant="none", selector="*",
        reason=("Whether an extreme value is an error or a finding is a "
                "question about your study, not about the number.")))


_install_core()


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Resolved:
    """What the table says for one (model, operation), and how it may be asked."""
    model: str
    operation: str
    variant: str
    reason: str
    determinacy: str
    scope: str
    origin: str
    selector: str
    variants: Tuple[str, ...]

    @property
    def is_fact(self) -> bool:
        return self.determinacy == MODEL_DETERMINED

    @property
    def may_be_preselected(self) -> bool:
        """A FACT may be pre-selected with a rendered skip. A CHOICE may not."""
        return self.is_fact

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "operation": self.operation,
                "variant": self.variant, "reason": self.reason,
                "determinacy": self.determinacy, "scope": self.scope,
                "origin": self.origin, "selector": self.selector,
                "variants": list(self.variants),
                "may_be_preselected": self.may_be_preselected}


def _matches(selector: str, model_key: str, spec: Any) -> bool:
    kind = _selector_kind(selector)
    if kind == "*":
        return True
    if kind == "model":
        return selector == model_key
    if kind == "group":
        return str(getattr(spec, "group", "")) == selector.split(":", 1)[1]
    flag = selector.split(":", 1)[1]
    return bool(getattr(spec.capabilities, flag, False))


def resolve(model_key: str, operation_key: str,
            registry: Optional[Dict[str, Any]] = None) -> Resolved:
    """The table's answer for one model and one operation.

    Most specific selector wins, ties broken by registration order so a pack
    registered after core overrides core at equal specificity — which is the
    behavior a pack author expects and the reason `register_default` appends
    rather than inserts.
    """
    if registry is None:
        from ml.model_registry import get_registry
        registry = get_registry()
    if model_key not in registry:
        raise RecipeError(f"no model {model_key!r} in the registry.")
    op = operation(operation_key)
    spec = registry[model_key]

    best: Optional[Default] = None
    best_rank = -1
    for d in _DEFAULTS:
        if d.operation != operation_key or not _matches(d.selector, model_key, spec):
            continue
        rank = _SPECIFICITY[_selector_kind(d.selector)]
        if rank >= best_rank:              # >= so later registration wins ties
            best, best_rank = d, rank
    if best is None:
        raise RecipeError(
            f"the table has no default for {model_key}/{operation_key}. Every "
            f"operation needs at least a `*` row, or a model falls through to "
            f"nothing and the interface has to invent one.")
    return Resolved(model=model_key, operation=operation_key,
                    variant=best.variant, reason=best.reason,
                    determinacy=op.determinacy, scope=op.scope,
                    origin=best.origin, selector=best.selector,
                    variants=op.variants)


def candidates(model_key: str, operation_key: str,
               registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every default row that MATCHED this cell, ranked as `resolve` ranks them,
    with the winner marked.

    `resolve` returns the answer and discards the reasoning. `GUIDED-074` is
    that discard: the precedence lattice is the app's model of the decision
    space, and a cell rendered as one sentence says *this is what happens*
    where the structure says *four rows matched, this one is the most specific,
    and here is what the others would have done*.

    Computed HERE rather than in the consumer, because the ranking rule and the
    tie-break are `resolve`'s and a second implementation of them is a second
    thing to drift — the prototype's capture script reached into `_matches`
    and `_SPECIFICITY` to do exactly this, which is what made this the right
    place to put it.
    """
    if registry is None:
        from ml.model_registry import get_registry
        registry = get_registry()
    if model_key not in registry:
        raise RecipeError(f"no model {model_key!r} in the registry.")
    spec = registry[model_key]

    rows: List[Dict[str, Any]] = []
    for d in _DEFAULTS:
        if d.operation != operation_key or not _matches(d.selector, model_key, spec):
            continue
        kind = _selector_kind(d.selector)
        rows.append({"selector": d.selector, "kind": kind,
                     "rank": _SPECIFICITY[kind], "variant": d.variant,
                     "reason": d.reason, "origin": d.origin, "wins": False})
    if rows:
        # `resolve` scans in registration order and takes `rank >= best`, so the
        # winner is the LAST row of maximal rank. Marked, never recomputed by a
        # consumer.
        best = max(r["rank"] for r in rows)
        winner = max(i for i, r in enumerate(rows) if r["rank"] == best)
        rows[winner]["wins"] = True
    return rows


def recipe(model_key: str, registry: Optional[Dict[str, Any]] = None,
           operations_: Optional[Sequence[str]] = None) -> List[Resolved]:
    """Every operation resolved for one model, in a stable order."""
    keys = list(operations_ or sorted(_OPERATIONS))
    return [resolve(model_key, k, registry) for k in keys]


# ─────────────────────────────────────────────────────────────────────────────
# Ask only when the choice changes the answer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Divergence:
    """Whether two variants of one operation would produce different answers."""
    operation: str
    a: str
    b: str
    material: bool
    statistic: float
    threshold: float
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return {"operation": self.operation, "a": self.a, "b": self.b,
                "material": self.material, "statistic": round(self.statistic, 4),
                "threshold": self.threshold, "evidence": self.evidence}


# How different is different enough to be worth a question. Not tuned: chosen as
# a round number and reported alongside the statistic so a reader can disagree
# with it, which is the only honest way to ship a threshold nobody has
# calibrated.
SCALE_THRESHOLD = 0.25


def _scale_divergence(df: pd.DataFrame, cols: Sequence[str],
                      a: str, b: str) -> Divergence:
    """Do `standard` and `robust` rescale the columns differently RELATIVE to
    each other?

    Both are affine, so asking "are the numbers different" always answers yes
    and answers nothing. What matters downstream is whether the two put the
    columns on DIFFERENT RELATIVE FOOTINGS: a penalized linear model sees the
    ratio between columns, not their absolute units.

    Standard divides by σ, robust by IQR. On approximately Gaussian columns
    σ ≈ 1.349·IQR, so the ratio is near-constant across columns and the two
    scalings differ by one global factor that no scale-equivariant model can
    detect. Heavy tails inflate σ without inflating IQR, and only for the
    columns that have them — so the ratio VARIES, and the two scalings reweight
    the features against one another.

    The statistic is therefore the coefficient of variation of σ/IQR across
    columns. Reported with the threshold so the reader can disagree.
    """
    ratios: List[float] = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 8:
            continue
        sigma = float(s.std(ddof=0))
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        if sigma <= 0 or iqr <= 0:
            continue
        ratios.append(sigma / iqr)

    if len(ratios) < 2:
        return Divergence(
            operation="scale", a=a, b=b, material=False, statistic=0.0,
            threshold=SCALE_THRESHOLD,
            evidence=("Fewer than two columns have usable spread, so the two "
                      "scalings cannot put them on different footings."))
    arr = np.asarray(ratios, dtype=float)
    cv = float(arr.std(ddof=0) / arr.mean()) if arr.mean() else 0.0
    material = cv > SCALE_THRESHOLD
    return Divergence(
        operation="scale", a=a, b=b, material=material, statistic=cv,
        threshold=SCALE_THRESHOLD,
        evidence=(
            f"σ/IQR varies by {cv:.0%} across {len(ratios)} numeric columns "
            + ("— heavy tails in some columns and not others, so standard and "
               "robust scaling would weight the features differently against "
               "one another and the choice changes the fit."
               if material else
               "— close to the constant 1.35 a Gaussian column gives, so the "
               "two scalings differ by roughly one global factor and no "
               "scale-equivariant model can tell them apart.")))


# Per-operation divergence tests. An operation with no entry is always asked:
# not knowing whether a choice matters is not evidence that it does not, and
# suppressing on ignorance is the failure mode this whole mechanism must avoid.
_DIVERGENCE: Dict[str, Callable[..., Divergence]] = {"scale": _scale_divergence}


def register_divergence(operation_key: str, fn: Callable[..., Divergence]) -> None:
    """A pack may teach the mechanism about its own operation."""
    _DIVERGENCE[operation_key] = fn


def divergence(df: pd.DataFrame, cols: Sequence[str], operation_key: str,
               a: str, b: str) -> Optional[Divergence]:
    """Would these two variants produce materially different answers?

    `None` when nothing knows how to compare them — which the caller must treat
    as *ask*, never as *suppress*.
    """
    fn = _DIVERGENCE.get(operation_key)
    if fn is None:
        return None
    return fn(df, cols, a, b)


def worth_asking(df: pd.DataFrame, cols: Sequence[str],
                 resolved: Resolved) -> Tuple[bool, Optional[Divergence]]:
    """Is there a live variant question here, or would asking be ceremony?

    Weighs only the operation's **pushed** alternatives. A variant with none is
    `(False, None)` — not a suppression but an absence: there is no question to
    put. That distinction is why the caller counts suppressions on the returned
    `Divergence` rather than on the boolean.

    Returns `(True, None)` when a pushed alternative exists and nothing can
    compare it, because an unmeasured difference is not a measured sameness.
    """
    alternatives = operation(resolved.operation).alternatives_for(resolved.variant)
    if not alternatives:
        return False, None
    worst: Optional[Divergence] = None
    for alt in alternatives:
        d = divergence(df, cols, resolved.operation, resolved.variant, alt)
        if d is None:
            return True, None
        if worst is None or d.statistic > worst.statistic:
            worst = d
    return (bool(worst and worst.material), worst)


# ─────────────────────────────────────────────────────────────────────────────
# Testing support
# ─────────────────────────────────────────────────────────────────────────────

def snapshot() -> Tuple[Dict[str, Operation], List[Default], Dict[str, Any]]:
    """The whole table, for a test that installs a pack and puts it back."""
    return dict(_OPERATIONS), list(_DEFAULTS), dict(_DIVERGENCE)


def restore(state: Tuple[Dict[str, Operation], List[Default], Dict[str, Any]]) -> None:
    ops, defs, divs = state
    _OPERATIONS.clear(); _OPERATIONS.update(ops)
    _DEFAULTS.clear(); _DEFAULTS.extend(defs)
    _DIVERGENCE.clear(); _DIVERGENCE.update(divs)

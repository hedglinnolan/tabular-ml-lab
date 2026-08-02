"""turbotab.pipeline_plan — the fold-fitted pipeline, composed from the record.

`GUIDED-095`, and it is the class rather than one defect:

> The app records **36 kinds of decision.** `training.py` read **six** project
> attributes and contained no reference to `recipes`, `model_recipes`, the
> missingness declarations, `deferred_transforms`, `selection_spec`,
> `engineered`, `purpose` or `lens`.

**The dividing line was never *which* decision. It was whether the decision
executed immediately.** Clause §06 splits preprocessing in two: row-local
operations, which may run now, and stateful ones, *recorded now and fitted
inside each training fold* so the held-out rows never inform them. The first
class reached the model because executing rewrites `working_table` and the
trainer read `working_table`. The second class reached nothing — the record was
written, the receipt counted it, the sentence was composed, and **the executor
did not exist.**

This module is the executor.

## What was and was not broken, because it decides what this must preserve

The **leakage** safety was real and stays real: the pipeline is constructed
inside the estimator, so `fit(X_train, y_train)` is the only place a parameter
can come from, and `turbotab/test_the_seal_holds.py` asserts bitwise equality of
every fitted parameter after moving the held-out rows. What did not exist was
**fidelity** — the fold fitted the app's defaults rather than the user's
choices. The app was *safe and unfaithful*, which is the harder failure to see,
because every number it printed was honest about the seal and none of them was
about the analysis the user specified.

So nothing here materializes anything on the working table, and nothing here is
fitted. `compose()` returns a **plan**; `Plan.build(estimator)` returns an
unfitted `Pipeline`.

## Two rules that are not negotiable, and where they live

**1 · The sentence and the pipeline are one object, not two that agree.**
A `Step` carries the sentence, and for an imputation the sentence is *the
declaration's own string, taken from the record* — `Step.sentence is
declaration["sentence"]`, asserted as identity, not as equality. Two strings
that happen to agree today are two strings, and drift between them is exactly
the defect this module closes: the recorded methods line said the blank was left
and the fit filled it with 27.15.

**2 · Where a declaration cannot be honored for a given model, the run says
which and why.** `leave` on a linear model is the case: gradient boosting reads
a blank natively and linear and neural models do not. That is a `Divergence` —
stated, per model, carrying the recorded request AND what was actually fitted —
never a silent substitution. **The plan is never shortened**, exactly as the
shelf is never shortened: a model that cannot honor a declaration still fits and
still says so.

Which is why a diverged step's sentence is NOT the declaration's. The recorded
sentence has become false for that model, and reprinting it would be this
module's own governing failure.

## Whether a column has a blank is read from the WHOLE frame, and that is not a leak

The *presence* of a blank decides whether an imputer exists at all, and a
held-out row with a blank still has to be scored — a pipeline with no imputer
would raise at `predict` rather than produce a number. What is fitted from the
training rows and only from them is the imputer's STATISTIC, which is what the
seal is about. Structure from the whole frame, parameters from the fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from turbotab import missingness as _miss

#: The percentile pair a `winsorize` variant clips at, and the number of median
#: absolute deviations a `mad` variant clips at. Chosen as the conventional
#: values and REPORTED IN THE SENTENCE rather than buried, the same posture
#: `recipes.SCALE_THRESHOLD` takes: a threshold nobody has calibrated for this
#: app ships stated, so a reader can disagree with it.
WINSOR_LIMITS = (0.01, 0.99)
MAD_MULTIPLE = 3.0


class PlanRefusal(Exception):
    """The recorded plan asks for something this module will not fake."""


# ─────────────────────────────────────────────────────────────────────────────
# The two transformers sklearn does not ship, fitted in-fold like everything else
# ─────────────────────────────────────────────────────────────────────────────

def _sklearn_base():
    from sklearn.base import BaseEstimator, TransformerMixin
    return BaseEstimator, TransformerMixin


_BASE, _MIXIN = _sklearn_base()


class Winsorizer(_BASE, _MIXIN):
    """Clip each column at fitted percentiles. Stateful by clause §06's litmus.

    The cut-points are quantiles of the column, so they are facts about the rows
    the transformer saw — which is why this is a pipeline step and not something
    the Preprocess step could have executed.
    """

    def __init__(self, lower: float = WINSOR_LIMITS[0],
                 upper: float = WINSOR_LIMITS[1]):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.lower_ = frame.quantile(self.lower).to_numpy(dtype=float)
        self.upper_ = frame.quantile(self.upper).to_numpy(dtype=float)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        return frame.clip(lower=pd.Series(self.lower_, index=frame.columns),
                          upper=pd.Series(self.upper_, index=frame.columns),
                          axis=1).to_numpy(dtype=float)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class MADClipper(_BASE, _MIXIN):
    """Clip at median ± `multiple` median absolute deviations.

    Robust where a percentile clip is not: a column where a fifth of the values
    are extreme has an extreme 99th percentile, and its median absolute
    deviation is still about the bulk.
    """

    def __init__(self, multiple: float = MAD_MULTIPLE):
        self.multiple = multiple

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.center_ = frame.median().to_numpy(dtype=float)
        deviation = (frame - frame.median()).abs().median().to_numpy(dtype=float)
        # A column with zero deviation cannot be clipped by this rule at all,
        # and clipping it to a point would replace the column with a constant.
        # Left alone, stated in the sentence rather than silently widened.
        self.deviation_ = np.where(deviation > 0, deviation, np.inf)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        low = self.center_ - self.multiple * self.deviation_
        high = self.center_ + self.multiple * self.deviation_
        return frame.clip(lower=pd.Series(low, index=frame.columns),
                          upper=pd.Series(high, index=frame.columns),
                          axis=1).to_numpy(dtype=float)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class ParetoScaler(_BASE, _MIXIN):
    """Center, then divide by the SQUARE ROOT of the standard deviation.

    The metabolomics pack registers this as the `scale` variant for any model
    that needs scaled inputs, with its own reason and its own
    `evidence_status`: auto-scaling gives every feature equal weight including
    the noise-dominated low-abundance ones, and dividing by √σ retains some
    magnitude information. `research/METABOLOMICS_PACK.md` is the source; the
    pack states it as a CONVENTION rather than a fact, and offers auto-scaling
    beside it.

    **It had no fitted form until `GUIDED-099`.** A pack could add a variant to
    the recipe table, the lattice would render it, and the pipeline would raise
    a `KeyError` on the first fit — which is `GUIDED-095`'s shape arriving
    through the extension point rather than through the trainer.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.mean_ = frame.mean().to_numpy(dtype=float)
        spread = np.sqrt(frame.std(ddof=0).to_numpy(dtype=float))
        # A constant column has no spread to divide by, and dividing by zero
        # would make it infinite rather than flat. Left at unit scale, which is
        # what every sklearn scaler does with the same case.
        self.scale_ = np.where(spread > 0, spread, 1.0)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        return ((pd.DataFrame(X).to_numpy(dtype=float) - self.mean_)
                / self.scale_)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class FrequencyEncoder(_BASE, _MIXIN):
    """Encode each level by how common it is. `features.ordinal_frequency`.

    The order comes from counts across the rows the encoder saw, which is why
    the catalogue marks it stateful and why it can only be fitted here. A level
    absent from the training fold encodes to 0 — never seen is not the same as
    seen once, and 0 says so.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.ranks_ = [
            {level: rank for rank, level
             in enumerate(frame[c].value_counts().index, start=1)}
            for c in frame.columns]
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        out = np.zeros((len(frame), frame.shape[1]), dtype=float)
        for j, c in enumerate(frame.columns):
            out[:, j] = frame[c].map(self.ranks_[j]).fillna(0).to_numpy(dtype=float)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# The plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Step:
    """One thing the fold-fitted pipeline does, and the sentence that IS it."""

    key: str
    #: `missingness` | `recipe` | `deferred_transform` | `default`
    source: str
    columns: Tuple[str, ...]
    #: The methods-prose line. For an honored declaration this IS the string on
    #: the record — the same object, not a copy that agrees.
    sentence: str
    #: Clause §06's litmus answer, in words.
    because: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "source": self.source,
                "columns": list(self.columns), "sentence": self.sentence,
                "because": self.because, "params": dict(self.params)}


@dataclass(frozen=True)
class Divergence:
    """A recorded decision this model could not honor, and what happened instead.

    Never a silent substitution: `requested` is what the record says, `applied`
    is what was fitted, and `why` is the property of the model that forced it.
    """

    subject: str
    source: str
    requested: str
    applied: str
    why: str
    #: The sentence the record carries, kept beside the one that is now true of
    #: this fit — a reader has to be able to see both to know they differ.
    recorded_sentence: str = ""
    fitted_sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "source": self.source,
                "requested": self.requested, "applied": self.applied,
                "why": self.why,
                "recorded_sentence": self.recorded_sentence,
                "fitted_sentence": self.fitted_sentence}


@dataclass
class Plan:
    """What one model's fold-fitted pipeline will do, and why."""

    model: str
    task_type: str
    steps: List[Step] = field(default_factory=list)
    divergences: List[Divergence] = field(default_factory=list)
    #: Columns with blanks and no recorded decision. The run says so rather than
    #: letting a default read as a choice.
    undeclared: List[str] = field(default_factory=list)
    #: Set by `build`. Kept so a caller can report the shape without rebuilding.
    _blocks: Any = None

    def sentences(self) -> List[str]:
        return [s.sentence for s in self.steps if s.sentence]

    def step_for(self, column: str) -> Optional[Step]:
        for s in self.steps:
            if column in s.columns:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "task_type": self.task_type,
            "steps": [s.to_dict() for s in self.steps],
            "divergences": [d.to_dict() for d in self.divergences],
            "undeclared": list(self.undeclared),
            "sentences": self.sentences(),
        }

    def build(self, estimator: Any) -> Any:
        """The unfitted `Pipeline`. Nothing here has seen a value."""
        from sklearn.pipeline import Pipeline

        prep, shape = self._blocks
        return Pipeline([("prep", prep), ("shape", shape), ("model", estimator)])


# ─────────────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────────────

def _accepts_nan(estimator: Any) -> Tuple[bool, str]:
    """Does this estimator read a blank natively, and how do we know?

    Asked of the estimator rather than answered from a list, because a list is
    a second copy of a fact sklearn already publishes. A wrapper that does not
    publish tags is asked about the model it wraps; one that answers neither is
    treated as NOT accepting a blank, and the reason says the tolerance was
    undetermined rather than claiming it was refused.
    """
    from sklearn.utils import get_tags

    for candidate, how in ((estimator, "the estimator's own sklearn tags"),
                           (getattr(estimator, "model", None),
                            "the sklearn estimator it wraps")):
        if candidate is None:
            continue
        try:
            return bool(get_tags(candidate).input_tags.allow_nan), how
        except Exception:
            continue
    return False, ("this model does not advertise whether it accepts a blank, "
                   "so the safe reading was taken")


def _tolerates_nan(transformer: Any) -> bool:
    from sklearn.utils import get_tags

    try:
        return bool(get_tags(transformer).input_tags.allow_nan)
    except Exception:
        return False


_DEFAULT_FILL = {"numeric": _miss.IMPUTE_MEDIAN,
                 "categorical": _miss.IMPUTE_MODE}

#: The stateful half of a compound strategy. `indicator_and_impute` puts the
#: was-it-missing column on the table at Preprocess and leaves the fill to
#: here, which is the only place it can be fitted without seeing held-out rows.
_MIXED_FILL = dict(_DEFAULT_FILL)

_SIMPLE_STRATEGY = {_miss.IMPUTE_MEDIAN: "median",
                    _miss.IMPUTE_MEAN: "mean",
                    _miss.IMPUTE_MODE: "most_frequent"}


def _imputer(strategy_key: str, seed: int) -> Any:
    from sklearn.impute import SimpleImputer

    if strategy_key in _SIMPLE_STRATEGY:
        # `keep_empty_features` so an all-blank column survives as a column.
        # Dropping it here would silently change the feature set between the
        # plan and the fit, and the plan is what the methods section quotes.
        return SimpleImputer(strategy=_SIMPLE_STRATEGY[strategy_key],
                             keep_empty_features=True)
    if strategy_key == _miss.IMPUTE_MICE:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(random_state=seed, keep_empty_features=True)
    raise PlanRefusal(
        f"{strategy_key!r} has no fold-fitted form here. A strategy the record "
        f"accepts and the pipeline cannot execute is the gap this module was "
        f"built to close, so it refuses rather than substituting a default.")


def _scaler(variant: str) -> Optional[Any]:
    from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                       StandardScaler)

    made = {"standard": StandardScaler(), "robust": RobustScaler(),
            "minmax": MinMaxScaler(), "pareto": ParetoScaler(),
            "none": None}
    if variant not in made:
        raise PlanRefusal(
            f"no scaling named {variant!r}. A pack may add a variant to the "
            f"recipe table and the lattice will render it; if nothing here can "
            f"fit it, the choice is one the app offers and cannot perform.")
    return made[variant]


def _power(variant: str) -> Optional[Any]:
    from sklearn.preprocessing import FunctionTransformer, PowerTransformer

    if variant == "none":
        return None
    if variant == "log1p":
        return FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    if variant == "yeo_johnson":
        return PowerTransformer(method="yeo-johnson", standardize=False)
    raise PlanRefusal(f"no power transform named {variant!r}.")


def _outliers(variant: str) -> Optional[Any]:
    if variant == "none":
        return None
    if variant == "winsorize":
        return Winsorizer()
    if variant == "mad":
        return MADClipper()
    raise PlanRefusal(f"no outlier handling named {variant!r}.")


def _encoder(variant: str, seed: int) -> Optional[Any]:
    from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                       TargetEncoder)

    if variant == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if variant == "ordinal":
        # An unseen level and a blank are DIFFERENT absences and get different
        # codes, so a model can tell "not in the training fold" from "nobody
        # recorded it". Collapsing them would be the app asserting they are the
        # same thing.
        return OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1, encoded_missing_value=-2)
    if variant == "target":
        return TargetEncoder(random_state=seed)
    if variant == "none":
        return None
    raise PlanRefusal(f"no encoding named {variant!r}.")


_DEFERRED_BUILDERS = {
    "bin_quantile": ("quantile", "equal-sized"),
    "bin_uniform": ("uniform", "equal-width"),
    "bin_kmeans": ("kmeans", "clustered"),
}


def _deferred_transformer(spec: Dict[str, Any], seed: int) -> Any:
    key = spec["key"]
    params = spec.get("params") or {}
    if key in _DEFERRED_BUILDERS:
        from sklearn.preprocessing import KBinsDiscretizer

        strategy, _ = _DEFERRED_BUILDERS[key]
        return KBinsDiscretizer(n_bins=int(params.get("n_bins") or 4),
                                encode="ordinal", strategy=strategy,
                                quantile_method="averaged_inverted_cdf"
                                if strategy == "quantile" else "linear")
    if key == "ordinal_frequency":
        return FrequencyEncoder()
    if key == "standardize":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    if key == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=int(params.get("n_components") or 2),
                   random_state=seed)
    raise PlanRefusal(
        f"{key!r} was recorded as a deferred transform and this module has no "
        f"fold-fitted form for it. A decision the record accepts and nothing "
        f"executes is the defect this module exists to remove, so it is "
        f"refused loudly rather than dropped.")


def _resolved_variants(project: Any, model_key: str) -> Dict[str, Dict[str, Any]]:
    """The recipe row per operation for this model, user overrides applied."""
    # THE PACKS ARE LOADED INTO THE TABLE FIRST, and forgetting this was
    # `GUIDED-095`'s own shape recreated inside its fix — caught by
    # `test_a_recorded_decision_changes_something`, which is what that probe is
    # for. `/recipes` calls `packs.load` before resolving, so a pack's variant
    # preference reached the lattice a user READS; `compose` did not, so it
    # reached nothing that was FITTED. The recipe table is canonical for a
    # pack's preferences (`GUIDED-025`) and resolution has to read the loaded
    # table on both paths. Idempotent.
    from turbotab import packs as _packs, recipes as _rec

    lens = getattr(project, "lens", None) or []
    _packs.load(lens)
    rows = project.resolved_recipes().get(model_key)
    if rows:
        return {r["operation"]: r for r in rows}
    # No model was selected at Preprocess, so the table's own defaults stand.
    # Read from `recipes` rather than assumed, so an unselected model is still
    # fitted to the table's answer instead of to this module's — and SCOPED to
    # this project's lens, or it would be fitted to another project's.
    return {r.operation: r.to_dict()
            for r in _rec.recipe(model_key,
                                 origins=_rec.allowed_origins(lens))}


def compose(project: Any, model_key: str, frame: pd.DataFrame, *,
            task_type: Optional[str] = None, seed: int = 42) -> Plan:
    """The fold-fitted pipeline for one model, composed from what was recorded.

    `frame` is the feature frame — the outcome and the grouping key are already
    out of it. Nothing is fitted and nothing is materialized.
    """
    from ml.model_registry import get_registry
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    registry = get_registry()
    if model_key not in registry:
        raise PlanRefusal(f"{model_key!r} is not in the model registry.")
    spec = registry[model_key]
    task = task_type or project.task_type or "regression"
    estimator = spec.factory(task, int(seed))
    accepts_nan, nan_evidence = _accepts_nan(estimator)

    plan = Plan(model=model_key, task_type=task)

    numeric = [str(c) for c in frame.columns
               if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [str(c) for c in frame.columns if str(c) not in numeric]
    has_blank = {str(c): bool(frame[c].isna().any()) for c in frame.columns}

    declared = {str(d["column"]): d for d in (project.missingness or [])
                if str(d["column"]) in set(map(str, frame.columns))}
    variants = _resolved_variants(project, model_key)

    # ── stage one · the fill, per column, from the record ───────────────────
    fill_groups: Dict[str, List[str]] = {}
    defaulted: Dict[str, List[str]] = {}
    keep_blank: List[str] = []

    for column in [*numeric, *categorical]:
        branch = "numeric" if column in numeric else "categorical"
        decision = declared.get(column)
        if decision is not None:
            strategy = str(decision["strategy"])
        elif has_blank[column]:
            strategy = _DEFAULT_FILL[branch]
            plan.undeclared.append(column)
        else:
            strategy = _miss.LEAVE

        if strategy == _miss.INDICATOR_AND_IMPUTE:
            # The indicator already landed at Preprocess; the fill is this
            # module's half. Recorded as one step carrying the record's own
            # sentence, because the user made one decision.
            fill = _MIXED_FILL[branch]
            plan.steps.append(Step(
                key=strategy, source="missingness", columns=(column,),
                sentence=decision["sentence"],
                because=_miss.strategy(strategy)["because"]))
            fill_groups.setdefault(fill, []).append(column)
            continue

        if strategy in (_miss.LEAVE, _miss.INDICATOR, _miss.EXPLICIT_CATEGORY):
            # Row-local or nothing: whatever it did already happened at
            # Preprocess, and the value stays as it is.
            if not has_blank[column]:
                if decision is not None:
                    plan.steps.append(Step(
                        key=strategy, source="missingness", columns=(column,),
                        sentence=decision["sentence"],
                        because=decision["because"]))
                keep_blank.append(column)
                continue
            honored = accepts_nan if branch == "numeric" else (
                variants.get("encode", {}).get("variant") in ("onehot", "ordinal"))
            if honored:
                keep_blank.append(column)
                if decision is not None:
                    plan.steps.append(Step(
                        key=strategy, source="missingness", columns=(column,),
                        # IDENTITY, not equality: the record's own string.
                        sentence=decision["sentence"],
                        because=decision["because"]))
                continue
            # RULE 2. The declaration cannot be honored for THIS model.
            fallback = _DEFAULT_FILL[branch]
            why = (f"{spec.name} cannot be fitted around a blank — "
                   f"{nan_evidence}. Gradient boosting reads a blank natively; "
                   f"linear, distance and neural models do not."
                   if branch == "numeric" else
                   f"the {variants.get('encode', {}).get('variant', 'chosen')} "
                   f"encoding cannot represent a blank as a level.")
            fitted = (
                f"For {spec.name} only, missing values in `{column}` were "
                f"filled with the training-fold "
                f"{'median' if fallback == _miss.IMPUTE_MEDIAN else 'most frequent level'}"
                f", because {why}")
            plan.divergences.append(Divergence(
                subject=column, source="missingness",
                requested=strategy, applied=fallback, why=why,
                recorded_sentence=(decision or {}).get("sentence", ""),
                fitted_sentence=fitted))
            plan.steps.append(Step(
                key=fallback, source="missingness", columns=(column,),
                sentence=fitted,
                because=_miss.strategy(fallback)["because"]))
            fill_groups.setdefault(fallback, []).append(column)
            continue

        # An imputation. Fitted in the fold; the sentence is the record's.
        #
        # WHERE THERE IS NO RECORD THE SENTENCE SAYS SO, and the columns are
        # collected into ONE step rather than one each. Composing the
        # declaration's own prose per column would make a default read as a
        # choice — the same substitution this module exists to stop, arriving
        # from the other side — and on an assay table it would also print 290
        # sentences, which is a methods section nobody can read.
        if decision is not None:
            plan.steps.append(Step(
                key=strategy, source="missingness", columns=(column,),
                sentence=decision["sentence"],
                because=_miss.strategy(strategy)["because"]))
        else:
            defaulted.setdefault(strategy, []).append(column)
        fill_groups.setdefault(strategy, []).append(column)

    for strategy, columns in sorted(defaulted.items()):
        plan.steps.append(Step(
            key=strategy, source="default", columns=tuple(columns),
            sentence=_undeclared_sentence(columns, strategy),
            because=_miss.strategy(strategy)["because"]))

    fill_blocks: List[Any] = []
    for strategy, columns in sorted(fill_groups.items()):
        fill_blocks.append((f"fill_{strategy}", _imputer(strategy, seed),
                            list(columns)))
    if keep_blank:
        fill_blocks.append(("keep_blank", "passthrough", list(keep_blank)))
    if not fill_blocks:
        raise PlanRefusal(
            "Every column except the outcome was dropped, so there is nothing "
            "to fit on.")

    prep = ColumnTransformer(fill_blocks, remainder="drop",
                             verbose_feature_names_out=False)
    prep.set_output(transform="pandas")

    # ── stage two · the shape, per the recipe table and the deferred record ──
    still_blank = set(keep_blank) & {c for c, blank in has_blank.items() if blank}

    def numeric_shape_steps() -> List[Any]:
        """A FRESH transformer per block, every time.

        `ColumnTransformer` clones its transformers before fitting, so sharing
        one instance across two blocks happens to be safe today — and
        `TRANSITION_PLAN.md` §02.1 records what this project already paid for
        relying on that reasoning: the global pipeline slot handed two models
        the same live instance, page 06 fitted it in place, and their fitted
        pipelines aliased. Building fresh costs nothing and cannot alias.
        """
        steps: List[Any] = []
        for operation in ("outliers", "power", "scale"):
            row = variants.get(operation)
            if row is None:
                continue
            made = {"outliers": _outliers, "power": _power,
                    "scale": _scaler}[operation](str(row["variant"]))
            if made is None:
                continue
            steps.append((operation, made))
        return steps

    numeric_shape = numeric_shape_steps()

    numeric_blank = sorted(c for c in numeric if c in still_blank)
    numeric_filled = sorted(c for c in numeric if c not in still_blank)
    # A transformer that cannot read a blank cannot sit over a column that kept
    # one. Rather than dropping the transform or the column, the blank is
    # filled for THIS model and the divergence says so.
    intolerant = [name for name, t in numeric_shape if not _tolerates_nan(t)]
    if numeric_blank and intolerant:
        for column in numeric_blank:
            recorded = declared.get(column, {})
            why = (f"the {', '.join(intolerant)} step this model's recipe "
                   f"resolves to cannot be fitted around a blank.")
            fitted = (f"For {spec.name} only, missing values in `{column}` "
                      f"were filled with the training-fold median before "
                      f"{intolerant[0]}, because {why}")
            plan.divergences.append(Divergence(
                subject=column, source="recipe",
                requested=str(recorded.get("strategy") or _miss.LEAVE),
                applied=_miss.IMPUTE_MEDIAN, why=why,
                recorded_sentence=recorded.get("sentence", ""),
                fitted_sentence=fitted))
        numeric_filled = sorted(numeric_filled + numeric_blank)
        numeric_blank = []

    shape_blocks: List[Any] = []
    for label, columns, tolerant_only in (("num", numeric_filled, False),
                                          ("num_blank", numeric_blank, True)):
        if not columns:
            continue
        steps = [(name, t) for name, t in numeric_shape_steps()
                 if not tolerant_only or _tolerates_nan(t)]
        shape_blocks.append(
            (label, Pipeline(steps) if steps else "passthrough", list(columns)))

    for operation in ("outliers", "power", "scale"):
        row = variants.get(operation)
        if row is None or not numeric:
            continue
        plan.steps.append(Step(
            key=f"{operation}:{row['variant']}", source="recipe",
            columns=tuple(numeric),
            sentence=_recipe_sentence(operation, str(row["variant"]),
                                      spec.name, len(numeric)),
            because=row.get("reason", ""),
            params={"variant": row["variant"], "selector": row.get("selector")}))

    if categorical:
        row = variants.get("encode") or {}
        variant = str(row.get("variant") or "onehot")
        made = _encoder(variant, seed)
        if made is None:
            # `encode: none` hands text to an estimator that cannot read it.
            # The plan is not shortened and neither is the shelf: one-hot is
            # applied and the divergence says the recorded variant was not.
            why = (f"{spec.name} cannot read a text column, and `encode: none` "
                   f"would hand it {len(categorical)} of them.")
            fitted = (f"For {spec.name} only, the {len(categorical)} "
                      f"categorical column(s) were one-hot encoded, because "
                      f"{why}")
            plan.divergences.append(Divergence(
                subject=", ".join(categorical), source="recipe",
                requested="none", applied="onehot", why=why,
                recorded_sentence=str(row.get("reason") or ""),
                fitted_sentence=fitted))
            made = _encoder("onehot", seed)
            variant = "onehot"
        cat_steps: List[Any] = []
        if variant not in ("onehot", "ordinal"):
            cat_steps.append(("impute", _imputer(_miss.IMPUTE_MODE, seed)))
        cat_steps.append(("encode", made))
        shape_blocks.append(("cat", Pipeline(cat_steps), list(categorical)))
        plan.steps.append(Step(
            key=f"encode:{variant}", source="recipe",
            columns=tuple(categorical),
            sentence=_recipe_sentence("encode", variant, spec.name,
                                      len(categorical)),
            because=row.get("reason", ""),
            params={"variant": variant, "selector": row.get("selector")}))

    # ── the deferred transforms, at last executed ───────────────────────────
    for i, deferred in enumerate(project.deferred_transforms or []):
        columns = [c for c in deferred.get("columns", [])
                   if c in set(map(str, frame.columns))]
        if not columns:
            continue
        transformer = _deferred_transformer(deferred, seed)
        branch = ("numeric" if all(c in numeric for c in columns)
                  else "categorical")
        block = Pipeline([("fill", _imputer(_DEFAULT_FILL[branch], seed)),
                          ("t", transformer)])
        shape_blocks.append((f"deferred_{i}_{deferred['key']}", block,
                             list(columns)))
        plan.steps.append(Step(
            key=deferred["key"], source="deferred_transform",
            columns=tuple(columns),
            # IDENTITY again: the sentence `features.declare` composed when the
            # user recorded the decision, not a second one written here.
            sentence=deferred["sentence"],
            because=(deferred.get("because", "")
                     + " Any blank in the source column is filled with the "
                       "training-fold "
                     + ("median" if branch == "numeric" else "most frequent "
                        "level")
                     + " first, because this transform cannot be fitted around "
                       "one; the source column itself keeps whatever the "
                       "record said it keeps."),
            params=dict(deferred.get("params") or {})))

    if not shape_blocks:
        shape_blocks.append(("all", "passthrough", list(frame.columns)))
    shape = ColumnTransformer(shape_blocks, remainder="drop",
                              verbose_feature_names_out=False)
    plan._blocks = (prep, shape)
    return plan


_RECIPE_PROSE = {
    ("scale", "standard"): ("{n} numeric column(s) were centered and scaled "
                            "using the mean and standard deviation of each "
                            "training fold, as {model} requires."),
    ("scale", "robust"): ("{n} numeric column(s) were scaled using the median "
                          "and interquartile range of each training fold."),
    ("scale", "minmax"): ("{n} numeric column(s) were rescaled to the range "
                          "observed in each training fold."),
    ("scale", "pareto"): ("{n} numeric column(s) were Pareto scaled — centered "
                          "and divided by the square root of the standard "
                          "deviation of each training fold — which retains "
                          "some magnitude information rather than giving every "
                          "feature equal weight."),
    ("scale", "none"): ("No scaling was applied: {model} splits on order "
                        "rather than distance, so rescaling changes nothing "
                        "it can see."),
    ("encode", "onehot"): ("{n} categorical column(s) were one-hot encoded "
                           "from the levels present in each training fold; a "
                           "level absent from a fold is encoded as all-zero."),
    ("encode", "ordinal"): ("{n} categorical column(s) were integer-coded from "
                            "the levels present in each training fold, with "
                            "unseen levels and blanks given distinct codes."),
    ("encode", "target"): ("{n} categorical column(s) were target-encoded "
                           "within each training fold, cross-fitted so a row's "
                           "own outcome does not set its own code."),
    ("encode", "none"): "Categorical columns were passed through unencoded.",
    ("power", "none"): "",
    ("power", "log1p"): ("{n} numeric column(s) were log(1+x) transformed."),
    ("power", "yeo_johnson"): ("{n} numeric column(s) were Yeo-Johnson "
                               "transformed, with lambda fitted inside each "
                               "training fold."),
    ("outliers", "none"): "",
    ("outliers", "winsorize"): (
        "{n} numeric column(s) were winsorized at the {low:.0%} and {high:.0%} "
        "percentiles of each training fold."),
    ("outliers", "mad"): (
        "{n} numeric column(s) were clipped at {k:g} median absolute "
        "deviations from the median of each training fold."),
}


_UNDECLARED_FILL = {
    _miss.IMPUTE_MEDIAN: "the training-fold median",
    _miss.IMPUTE_MODE: "the most frequent level of each training fold",
}


def _undeclared_sentence(columns: Sequence[str], strategy: str) -> str:
    """The line for the columns with blanks that nobody answered for.

    Deliberately not the declaration's prose. *"Missing values in `x` will be
    filled using the median within each training fold"* reads as a decision
    somebody made; this says the app defaulted, which is what happened.

    One sentence for the set rather than one each, and not only for length: a
    methods section that lists 290 metabolite columns one at a time is a
    methods section nobody reads, and the first eight plus a count is what a
    reader can check.
    """
    named = list(columns)
    shown = ", ".join(f"`{c}`" for c in named[:8])
    if len(named) > 8:
        shown += f", and {len(named) - 8} more"
    fill = _UNDECLARED_FILL.get(strategy, "this app's default")
    return (f"No handling was recorded for the missing values in "
            f"{len(named)} column(s) ({shown}), so they were filled with "
            f"{fill} — the app's default rather than a choice made at "
            f"Preprocess.")


def _recipe_sentence(operation: str, variant: str, model_name: str,
                     n: int) -> str:
    """The methods line for one resolved recipe cell.

    **Composed here and nowhere else.** There is no second string to agree with:
    the plan holds this sentence, `training` reports it, and a manuscript quotes
    it — which is the whole of rule 1 for the half of the plan that has no
    user-written sentence on the record.
    """
    template = _RECIPE_PROSE.get((operation, variant))
    if template is None:
        raise PlanRefusal(
            f"no methods sentence for {operation}:{variant}. A step with no "
            f"sentence is a step the manuscript cannot describe, and this "
            f"module refuses to fit one.")
    return template.format(n=n, model=model_name, low=WINSOR_LIMITS[0],
                           high=WINSOR_LIMITS[1], k=MAD_MULTIPLE)

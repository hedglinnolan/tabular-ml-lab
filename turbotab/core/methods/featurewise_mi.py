"""Feature-wise regression pooled over multiple imputations drawn one study factor at a time (the
plasma scope's item 1, under the orchestrator's methods ruling of 2026-11-01).

**Why.** The engine's other route to multiple imputation is one chained-equations model over every
input (``missing.imputation_frame``). A feature-wise table tests each study factor on its own, often
more of them than rows (134 plasma features on 100 participants), so that joint model is beyond what
chained equations hold, and the table used to be refused at the Fit press. Its way forward then, one
censoring-aware fill per feature, is intercept-only and ignores the outcome, though under the limma
design the feature is the response: simulated by construction (the plasma scope's ``fill_sim.py``,
2,000 replicates of 58 + 42 rows, a true difference 0.5), the fill shrank it to 0.459 at 30%
censored and 0.357 at 60%, with 95% coverage 0.865 at 60%.

**What is drawn.** Each study factor's complete-data model is the one its test fits: the factor on
the event and the adjustment columns ``Z`` (the limma design), or under a numeric outcome the
outcome on the factor and ``Z``. So each study factor ``x_j``'s blanks are drawn from its own model
given the outcome's term and ``Z``, on the scale its test reads it (after the normalization and the
log the design applies): below a detection limit (the smallest detected value, one per column) from a
censored-normal (Tobit) model, its parameters drawn from N(θ̂, I(θ̂)⁻¹) before each copy's draws,
exactly as ``imputation``'s censored draws are; a blank not below a limit from a normal linear
model, mice's ``norm``. van Buuren (FIMD 2nd ed. §6.3.2): "Include all variables that appear in
the complete-data model … including the outcome". Where ``Z`` has blanks, or a unit's rows repeat,
the copies are drawn by chained equations on {``x_j``, ``Z``, the outcome} for each study factor
(``smcfcs.impute`` in its ``"fcs"`` mode, the engine's one sampler; clustered rows with the unit
means), so ``Z``'s draws see the factor its test treats as the response. A study factor whose
blanks are its own alone, on rows that do not repeat, is a monotone pattern of one variable: its
censored-normal model is fit once and its m copies drawn from that fit, which is what m chains of
one sweep each would do (each would refit the same model on the same observed values).

**What is pooled.** Each copy is tested by the family's own Frisch–Waugh–Lovell tests
(``featurewise.feature_tests``); each study factor is pooled by Rubin's rules with Barnard and
Rubin's degrees of freedom, ν_com = n − rank[1, Z] − 1 (the Bell–McCaffrey df under CR2), then
Benjamini–Hochberg runs on the pooled p-values (``featurewise.featurewise_table``). m is the
engine's rule (``imputation.m_rule``) over the most rows any one study factor's model draws a value
in.

**Its truths** (``turbotab/core/tests/test_featurewise_mi.py``): the pooled rows against
``mice::pool`` (3.19.0) on lm fits to these copies, ``dfcom`` from ``df.residual``, and
``p.adjust(p, "BH")``; the draws against ``survival::survreg``'s left-censored Gaussian fit at
m = 200; and a simulation by construction of coverage, null rejection and the false-discovery
proportion.

**Its budget.** At most :data:`MAX_FEATURE_COPIES` draws (the study factors drawn × m, a chained
one counted once per sweep), refused beyond it in words that give the counts
(:func:`budget_refusal`), at the fit and before the lock (:func:`plan_refusal`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# The most draws one feature-wise table makes: each study factor drawn, m times, a chained one
# (covariates with blanks, or rows that repeat) once per sweep of its chain. On 100 rows a chained
# draw refits its censored-normal model (about 15 ms with the chain's other models), so 20,000
# chained draws take about five minutes; a study factor drawn alone fits its model once (about
# 5 ms) and its m copies cost little more. The plasma panel's 40 censored features × m 100 make
# 4,000 draws.
MAX_FEATURE_COPIES = 20_000
OUTCOME_TERM = "__outcome"  # the outcome's term among each study factor's imputation variables
UNPOOLED = "featurewise_unpooled"  # the table's, and the plan's, refusal of what it cannot pool
SINGLE_FILL = "featurewise_single_fill"  # … and of one censoring-aware fill under inference


class Unpooled(ValueError):
    """The pooled feature-wise table cannot be drawn here: ``reason`` says why and ``exits`` where
    to go instead."""

    def __init__(self, reason: str, exits: Sequence[dict[str, Any]] = ()):
        super().__init__(reason)
        self.reason = reason
        self.exits = [dict(e) for e in exits]


# ── the steps that keep the blanks ───────────────────────────────────────────


class KeepBelowDetection(TransformerMixin, BaseEstimator):
    """The detect step of the pooled feature-wise table: each value below detection is left blank,
    for each copy to draw (a fill here would hand every copy the same value). ``positive``: the
    step logs next (``omics.LogScaleCensoredFill``), so a value at or below zero counts as below
    detection, as that fill counts it; otherwise only a blank does
    (``missing.BelowDetectionFill``). ``limit_``: each column's smallest detected value on the
    fitting rows, the limit the copies are drawn below. Every other column passes through."""

    def __init__(self, columns: Sequence[str] = (), positive: bool = False):
        self.columns = columns
        self.positive = positive

    def _below(self, x: np.ndarray) -> np.ndarray:
        found = ~np.isfinite(x)
        if self.positive:
            found |= np.isfinite(x) & (x <= 0)
        return found

    def fit(self, X: pd.DataFrame, y: Any = None) -> "KeepBelowDetection":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.limit_: dict[str, float] = {}
        for c in self.columns:
            if c in X.columns:
                x = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
                detected = x[~self._below(x)]
                if len(detected):
                    self.limit_[c] = float(detected.min())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.columns:
            if c in out.columns:
                x = pd.to_numeric(out[c], errors="coerce")
                out[c] = x.where(~self._below(x.to_numpy(dtype=float)))
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        mine = set(self.columns)
        return [{"output": c, "inputs": [c],
                 "operation": "below detection: drawn in each imputation" if c in mine else "kept"}
                for c in self.feature_names_in_]


class KeepBlanks(TransformerMixin, BaseEstimator):
    """The impute step of the pooled feature-wise table: nothing is filled; each blank is drawn in
    the imputations, from the model of the study factor whose table reads it."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "KeepBlanks":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


# ── the matrix with its blanks ───────────────────────────────────────────────

# The steps a pooled feature-wise table's pipeline may hold: each acts on a column alone or drops
# columns, so a study factor's blanks reach the model matrix as blanks of that one column. ComBat
# (``batch``) adjusts every feature together, and a scale's score, an energy model or a form forms
# new columns from others; a blank in a column they read is refused (:func:`prepare`).
ALONE = ("d_ratio", "detect", "normalize", "log", "impute", "levels", "onehot", "model")


def blank_pipeline(template: Any) -> tuple[Any, list[str], bool]:
    """(``template`` unfitted with its blanks kept, the columns its detect step reads as below
    detection, whether the step logs next). A censoring-aware detect step becomes
    :class:`KeepBelowDetection`; half the minimum stays (the answer fills those once, before any
    draw, as ``imputation_plan`` does); the impute step becomes :class:`KeepBlanks`."""
    from sklearn.base import clone

    from turbotab.core.methods.missing import BelowDetectionFill
    from turbotab.core.methods.omics import LogScaleCensoredFill

    pipe = clone(template)
    names = [n for n, _ in pipe.steps]
    detect = pipe.named_steps.get("detect") if "detect" in names else None
    censored: list[str] = []
    logged = False
    if isinstance(detect, LogScaleCensoredFill):
        censored, logged = [str(c) for c in detect.columns], True
        pipe.set_params(detect=KeepBelowDetection(censored, positive=True))
    elif isinstance(detect, BelowDetectionFill) and detect.method == "censoring_aware":
        censored = [str(c) for c in detect.columns]
        pipe.set_params(detect=KeepBelowDetection(censored, positive=False))
    if "impute" in names:
        pipe.set_params(impute=KeepBlanks())
    return pipe, censored, logged


@dataclass
class Prepared:
    """The analyzed rows as each study factor's test reads them, blanks kept: the fitted
    pipeline, its model matrix, the study factors tested and the adjustment columns, each censored
    study factor's limit on the matrix's scale, and which columns hold a blank."""

    fitted: Any
    matrix: pd.DataFrame
    tested: list[str]
    adjust: list[str]
    censored: dict[str, Any]
    drawn: list[str]
    adjust_blank: list[str]


def _another_family(state: Any = None) -> dict[str, Any]:
    from turbotab.core.models.featurewise import ANOTHER_FAMILY

    return dict(ANOTHER_FAMILY)


def ways_forward(state: Any, frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Where a feature-wise table goes when its imputations cannot be pooled: complete cases,
    where they keep rows enough for the design (``omics.featurewise_missing_exits``), and another
    family. Never one censoring-aware fill: under inference it is the pooled route's to replace
    (the orchestrator's ruling on M2)."""
    from turbotab.core.methods.omics import featurewise_missing_exits

    exits = [{k: v for k, v in e.items() if k != "way"}
             for e in featurewise_missing_exits(state, frame) if e["way"] == "complete_case"]
    return [*exits, _another_family(state)]


def single_censoring_fill(answer: Mapping[str, Any] | None) -> bool:
    """Whether the missing-values answer fills each value below detection once, censoring-aware
    (strategy ``impute`` with ``below_detection`` ``censoring_aware``), recorded or not."""
    return bool(answer) and answer.get("strategy") == "impute" \
        and answer.get("below_detection") == "censoring_aware"


# The single fill's refusal under inference (the orchestrator's ruling on M2): the pooled route
# replaces it. ``missing.BelowDetectionFill`` and ``omics.LogScaleCensoredFill`` give each column
# one value, E[X | X < L] under a censored normal fitted to the column alone; simulated by
# construction with the limma design (``test_featurewise_mi_simulation.py``, 1,000 replicates), a
# true difference of 0.5 was covered by 86.8% of its 95% intervals at 60% censored, where the
# pooled route's covered it in 94.8%.
SINGLE_FILL_WORDS = (
    "One censoring-aware fill gives every value below detection in a feature the same number, from "
    "that feature's own censored-normal fit alone, without the outcome or the covariates its test "
    "reads. Under a two-level outcome feature-wise regression tests each feature as the response, "
    "so such a fill biases its difference toward zero, and under either outcome one value per "
    "blank makes its intervals too narrow. Under inference its tests are pooled over "
    "censoring-aware multiple imputations instead: each feature's values below detection are drawn "
    "from its own model given the outcome and the covariates.")
POOLED_ROUTE = "Censored-normal: a draw below the limit within the multiple imputation"


def pooled_answer(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The pooled route as a ``set_missing`` decision: multiple imputation with the values below
    detection drawn censoring-aware, the answer's columns, drops and count kept."""
    keep = {k: v for k, v in dict(answer).items() if k not in ("kind", "reason")}
    return {**keep, "kind": "set_missing", "strategy": "multiple_imputation",
            "below_detection": "censoring_aware", "acknowledged": False, "indicators": False,
            "categorical": "impute"}


def single_fill_refusal(answer: Mapping[str, Any], state: Any,
                        frame: pd.DataFrame | None) -> tuple[str, list[dict[str, Any]]]:
    """(reason, exits) of the feature-wise table under inference where the answer fills the values
    below detection once, censoring-aware: the pooled route, then complete cases where they keep
    rows enough (``omics.featurewise_missing_exits``)."""
    from turbotab.core.methods.omics import featurewise_missing_exits

    cases = [{k: v for k, v in e.items() if k != "way"}
             for e in featurewise_missing_exits(state, frame) if e["way"] == "complete_case"]
    return SINGLE_FILL_WORDS, [{"label": POOLED_ROUTE, "decision": pooled_answer(answer)}, *cases]


def _onehot_blank(fitted: Any) -> list[str]:
    """The categories whose one-hot encoder took a blank as a level of its own (its categories
    hold a missing value): a covariate category with blanks, which no copy here draws."""
    steps = dict(getattr(fitted, "named_steps", None) or {})
    onehot = steps.get("onehot")
    encoder = (getattr(onehot, "named_transformers_", {}) or {}).get("onehot") if onehot else None
    if encoder is None or not hasattr(encoder, "categories_"):
        return []
    out = []
    for column, levels in zip(encoder.feature_names_in_, encoder.categories_):
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in levels):
            out.append(str(column))
    return out


def prepare(template: Any, X: pd.DataFrame, y: Any, *, state: Any = None,
            logged_scale: bool | None = None) -> Prepared:
    """The pooled table's matrix: ``template`` (the family's unfitted pipeline) fitted on ``X``
    (every analyzed row) with its blanks kept (:func:`blank_pipeline`), each censored study
    factor's limit its smallest detected value on the matrix's scale and its scale the one its test
    reads (already the log where the design logged after the fill; else the scale a censored normal
    fits better, ``imputation.censoring_of``'s rule). Raises :class:`Unpooled` where a blank would
    reach the matrix through ComBat, through a step that forms columns from others (a score, an
    energy model, a form), or through a category's one-hot encoding."""
    from turbotab.core.methods.imputation import Censoring, censoring_of
    from turbotab.core.models.featurewise import split_columns
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.voice import listing

    pipe, censored_inputs, logged = blank_pipeline(template)
    blank_inputs = [str(c) for c in X.columns if X[c].isna().any()]
    frame = X[[c for c in X.columns]]
    others = [n for n, _ in pipe.steps if n not in ALONE]
    if blank_inputs and "batch" in others:
        raise Unpooled(
            "Feature-wise regression is pooled over multiple imputations one study factor at a "
            "time, and ComBat adjusts every feature together, so no study factor's copies can be "
            "drawn and adjusted on their own.", ways_forward(state, frame))
    fitted = pipe.fit(frame, y)
    hidden = _onehot_blank(fitted)
    if hidden:
        raise Unpooled(
            f"Feature-wise regression is pooled over multiple imputations one study factor at a "
            f"time, drawing numbers; {listing(hidden)} "
            f"{'is a category' if len(hidden) == 1 else 'are categories'} among the covariates "
            f"with blanks, which these copies do not draw.",
            ways_forward(state, frame))
    matrix = model_matrix(fitted, frame)
    model = fitted[-1]
    tested, adjust = split_columns(list(matrix.columns), getattr(model, "features", None))
    names = set(str(c) for c in X.columns)
    holding = [str(c) for c in matrix.columns if matrix[c].isna().any()]
    formed = [c for c in holding if c not in set(tested) and c not in names]
    if formed:
        raise Unpooled(
            f"Feature-wise regression is pooled over multiple imputations one study factor at a "
            f"time; {listing(formed)} {'holds' if len(formed) == 1 else 'hold'} blanks after a step "
            f"that forms {'it' if len(formed) == 1 else 'them'} from other columns, which these "
            f"copies do not draw.", ways_forward(state, frame))
    limits = dict(getattr(fitted.named_steps.get("detect"), "limit_", None) or {}) \
        if "detect" in fitted.named_steps else {}
    log_scale = logged if logged_scale is None else logged_scale
    censoring: dict[str, Any] = {}
    for c in [*tested, *adjust]:  # a censored covariate's blanks are drawn below its limit too
        if c in set(censored_inputs) and matrix[c].isna().any() and c in limits:
            series = matrix[c]
            if log_scale:
                censoring[c] = Censoring(limit=float(np.nanmin(series.to_numpy(dtype=float))),
                                         log=False)
            else:
                censoring[c] = censoring_of(series)
    drawn = [c for c in tested if matrix[c].isna().any()]
    adjust_blank = [c for c in adjust if matrix[c].isna().any()]
    return Prepared(fitted=fitted, matrix=matrix, tested=tested, adjust=adjust,
                    censored=censoring, drawn=drawn, adjust_blank=adjust_blank)


# ── m, its budget and the copies ─────────────────────────────────────────────


def incomplete_rows(matrix: pd.DataFrame, tested: Sequence[str], adjust: Sequence[str]) -> int:
    """The most rows any one study factor's model draws a value in: rows with a blank in the
    factor or in an adjustment column."""
    if not len(matrix):
        return 0
    covariates = (matrix[list(adjust)].isna().any(axis=1).to_numpy() if adjust
                  else np.zeros(len(matrix), dtype=bool))
    most = int(covariates.sum())
    for c in tested:
        most = max(most, int((matrix[c].isna().to_numpy() | covariates).sum()))
    return most


def imputation_count(matrix: pd.DataFrame, tested: Sequence[str], adjust: Sequence[str],
                     asked: int) -> int:
    """m: the engine's rule (``imputation.m_rule``: at least ``asked``, 20 at least, and at least
    the percentage of rows with a value drawn) over the most incomplete study factor's model; 0
    where nothing is drawn."""
    from turbotab.core.methods.imputation import m_rule

    n_incomplete = incomplete_rows(matrix, tested, adjust)
    if not n_incomplete:
        return 0
    return m_rule(n_incomplete, len(matrix), asked)


def sweeps_of(own_blank: bool, adjust_blank: Sequence[str], units: Any) -> int:
    """The sweeps of one study factor's chain, as the engine's sampler runs it
    (``smcfcs.impute``): one where a single variable of its model is blank (its own blanks, or one
    covariate's) and the rows do not repeat, a monotone pattern; else the engine's chained
    equations' count."""
    from turbotab.core.methods.imputation import ITERATIONS

    incomplete = int(bool(own_blank)) + len(adjust_blank)
    return ITERATIONS if (incomplete > 1 or units is not None) else 1


def drawn_factors(tested: Sequence[str], drawn: Sequence[str], adjust_blank: Sequence[str]
                  ) -> list[str]:
    """The study factors whose copies are drawn: every one where an adjustment column has a blank
    (its copies hold that column's draws), else those with a blank of their own."""
    return list(tested) if adjust_blank else [c for c in tested if c in set(drawn)]


def draws_of(tested: Sequence[str], drawn: Sequence[str], adjust_blank: Sequence[str],
             units: Any) -> list[int]:
    """Each drawn study factor's sweeps (:func:`drawn_factors`, :func:`sweeps_of`), in order."""
    own = set(drawn)
    return [sweeps_of(c in own, adjust_blank, units)
            for c in drawn_factors(tested, drawn, adjust_blank)]


def budget_refusal(sweeps: Sequence[int], m: int) -> str | None:
    """Why the copies are not drawn, where the draws, m × the sum of each drawn study factor's
    ``sweeps``, exceed :data:`MAX_FEATURE_COPIES`; None within it. A draw is one copy of one study
    factor, or under chained equations one sweep of it: each sweep refits the study factor's
    models, where a study factor drawn alone fits its model once for all m copies."""
    draws = int(m) * int(sum(sweeps))
    if draws <= MAX_FEATURE_COPIES:
        return None
    n = len(sweeps)
    chained = [s for s in sweeps if s > 1]
    if not chained:
        each = ""
    elif len(chained) == n:
        each = f" × {chained[0]} sweeps of chained equations"
    else:
        each = f", {len(chained):,} of them over {chained[0]} sweeps of chained equations,"
    return (f"Feature-wise regression is pooled over multiple imputations drawn one study factor "
            f"at a time: here {n:,} study factors × {m} imputations{each} make {draws:,} draws, "
            f"more than the {MAX_FEATURE_COPIES:,} it makes.")


def plan_frame(state: Any, ctx: Any) -> pd.DataFrame | None:
    """The predictors the plan's models get (by settled role, less the columns left out), on the
    participant rows of the table the analysis reads (the pooled QCs and other reference rows
    leave), among the analyzed rows where the context counts them: what the check before the lock
    counts blanks on. None where the context cannot open the table."""
    from turbotab.core.decisions import _ctx, _store_of
    from turbotab.core.methods.omics import _reference_rules_of, participant_rows
    from turbotab.core.models.pipeline import model_predictors

    store = _store_of(ctx)
    if store is None:
        return None
    try:
        present = set(store.columns)
        predictors = [c for c in model_predictors(state) if c in present]
        rules = [str(r["column"]) for r in _reference_rules_of(state) if str(r["column"]) in present]
        if not predictors:
            return None
        frame = store.materialize(list(dict.fromkeys([*predictors, *rules])))
    except Exception:  # noqa: BLE001 - an unreadable table: the fit counts them
        return None
    keep = participant_rows(frame, state)
    frame = frame.loc[np.asarray(keep, dtype=bool), predictors]
    analyzed = _ctx(ctx, "analyzed")
    if callable(analyzed):
        try:
            ids = analyzed()
        except Exception:  # noqa: BLE001 - no count of the analyzed rows: every participant row
            ids = None
        if ids is not None:
            frame = frame.loc[frame.index.isin(np.asarray(ids))]
    return frame


def plan_refusal(state: Any, answer: Mapping[str, Any], ctx: Any
                 ) -> tuple[str, list[dict[str, Any]]] | None:
    """Why the plan's feature-wise table cannot pool the copies its multiple-imputation answer
    draws, counted on the plan's rows (:func:`plan_frame`): a category among the covariates with
    blanks, or draws beyond :data:`MAX_FEATURE_COPIES` (the study factors as the roles answer
    gives them, before the design's filters, so never fewer than the fit draws). With its ways
    forward (:func:`ways_forward`). None where it can, or where the rows cannot be read: the fit
    then counts them and refuses in the same words."""
    from turbotab.core.methods.imputation import M_DEFAULT
    from turbotab.core.models.pipeline import is_categorical
    from turbotab.core.readings import settled_roles
    from turbotab.core.voice import listing

    frame = plan_frame(state, ctx)
    if frame is None or not len(frame):
        return None
    gone = set(answer.get("drop_columns") or [])
    roles = settled_roles(state)
    columns = [c for c in frame.columns if c not in gone]
    tested = [c for c in columns if roles.get(c) == "exposure"]
    adjust = [c for c in columns if c not in set(tested)]
    if not tested:
        return None
    frame = frame[columns]
    exits = ways_forward(state, frame)
    hidden = [c for c in adjust if frame[c].isna().any() and is_categorical(frame[c])]
    if hidden:
        return (f"Feature-wise regression is pooled over multiple imputations one study factor at a "
                f"time, drawing numbers; {listing(hidden)} "
                f"{'is a category' if len(hidden) == 1 else 'are categories'} among the covariates "
                f"with blanks, which these copies do not draw.", exits)
    blank_z = [c for c in adjust if frame[c].isna().any()]
    drawn = [c for c in tested if frame[c].isna().any()]
    if not drawn and not blank_z:
        return None
    m = imputation_count(frame, tested, adjust, int(answer.get("m") or M_DEFAULT))
    grain = getattr(state, "grain", None)
    repeats = getattr(grain, "grain", None) == "repeated" and getattr(grain, "id_column", None)
    over = budget_refusal(draws_of(tested, drawn, blank_z, True if repeats else None), m)
    return (over, exits) if over is not None else None


MARGINAL_UNPOOLED_CODE = "marginal_risks_unpooled"


def marginal_refusal(state: Any, models: Sequence[str], answer: Mapping[str, Any], ctx: Any
                     ) -> tuple[str, list[dict[str, Any]]] | None:
    """The check before the lock's refusal of the marginal risk difference or ratio under multiple
    imputation: where the declared estimand is one, a chosen family standardizes it
    (``effects.standardizes_risks``) and a predictor the plan's models get is blank on its rows
    (:func:`plan_frame`), the effects stage would draw copies and refuse the risks
    (``effects.MARGINAL_UNPOOLED``); so the conditional odds ratio, or complete cases. None
    otherwise, or where the rows cannot be read."""
    from turbotab.core import estimand as est
    from turbotab.core.decisions import _ctx
    from turbotab.core.models import get_family
    from turbotab.core.stages.effects import MARGINAL_UNPOOLED, standardizes_risks

    spec = est.current_estimand(state)
    if spec is None or str(spec.measure) not in est.MARGINAL:
        return None
    task = getattr(state, "task", None) or _ctx(ctx, "task")
    if not any(standardizes_risks(get_family(k), str(task)) for k in models):
        return None
    frame = plan_frame(state, ctx)
    gone = set(answer.get("drop_columns") or [])
    if frame is None or not any(frame[c].isna().any() for c in frame.columns if c not in gone):
        return None
    cases = [{"label": "Complete cases, with their assumption stated",
              "decision": {**{k: v for k, v in dict(answer).items() if k != "kind"},
                           "kind": "set_missing", "strategy": "complete_case",
                           "indicators": False, "categorical": "impute"}}]
    return MARGINAL_UNPOOLED, [
        {"label": "Report the conditional odds ratio",
         "decision": {**spec.model_dump(mode="json"), "kind": "set_estimand",
                      "measure": "odds_ratio"}}, *cases]


@dataclass
class FeatureCopies:
    """``m`` completed copies of each drawn study factor (``values``: its column in each copy, n ×
    m), and of the adjustment columns where they were drawn with it (``adjust``: m arrays, n ×
    the ``adjust_names``), with what was drawn: each study factor's blanks (``imputed``), the most
    rows any one model drew a value in (``n_incomplete``), the censored ones' limits, the most sweeps
    any chain ran, the seed, and the unit the rows repeat by (None where they do not)."""

    m: int
    sweeps: int
    seed: int
    n_rows: int
    values: dict[str, np.ndarray]
    adjust: dict[str, list[np.ndarray]]
    adjust_names: list[str]
    imputed: dict[str, int]
    n_incomplete: int
    censored: dict[str, Any] = field(default_factory=dict)
    unit: str | None = None
    asked: int = 20


def _design(term: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """A study factor's imputation design as the engine's chains build one: the intercept and
    every other variable standardized (the outcome's term, then the adjustment columns)."""
    from turbotab.core.methods.imputation import _standardize

    block = np.column_stack([term, Z]) if Z.size else term.reshape(-1, 1)
    return np.column_stack([np.ones(len(term)), _standardize(block)])


def _censored_copies(D: np.ndarray, x: np.ndarray, miss: np.ndarray, censoring: Any, m: int,
                     rng: np.random.Generator, name: str) -> np.ndarray:
    """The m draws of a censored study factor's blanks (n_blank × m) from one censored-normal fit
    on ``D``: θ̂ and its inverse observed information (``imputation.tobit_fit``), then for each copy
    θ* ~ N(θ̂, I⁻¹) and each blank from N(Dβ*, σ*²) truncated above at the limit, as
    ``imputation``'s censored draw makes one copy."""
    from scipy import stats

    from turbotab.core.methods.imputation import ImputationRefused, tobit_fit

    z = np.log(np.clip(x, 1e-300, None)) if censoring.log else x.astype(float)
    limit = math.log(censoring.limit) if censoring.log else float(censoring.limit)
    theta, V = tobit_fit(D, np.where(miss, limit, z), miss, limit)
    try:
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(V))):
            raise np.linalg.LinAlgError
        root = np.linalg.cholesky(V + np.eye(len(V)) * 1e-12)
    except np.linalg.LinAlgError:
        refused = ImputationRefused(f"`{name}`'s censored-normal model has no finite fit on these "
                                    f"rows, so its values below detection cannot be drawn.")
        refused.columns = [name]
        raise refused from None
    rows = D[miss]
    # Each copy's parameters (one column each), then every copy's blanks in one draw.
    stars = theta[:, None] + root @ rng.standard_normal((len(theta), m))
    mu = rows @ stars[:-1]
    sigma = np.broadcast_to(np.exp(stars[-1])[None, :], mu.shape)
    upper = (limit - mu) / sigma
    draws = stats.truncnorm.rvs(-np.inf, upper, loc=mu, scale=sigma, random_state=rng)
    draws = np.minimum(np.asarray(draws, dtype=float).reshape(mu.shape), limit)
    return np.exp(draws) if censoring.log else draws


def draw_copies(matrix: pd.DataFrame, outcome: Any, tested: Sequence[str],
                adjust: Sequence[str], censored: Mapping[str, Any], m: int, *, seed: int = 0,
                units: np.ndarray | None = None, unit: str | None = None, asked: int = 20,
                progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> FeatureCopies:
    """``m`` copies of each study factor whose model draws a value (:func:`drawn_factors`): its
    blanks from its own model given ``outcome`` (the outcome's term: the 0/1 event, or the numeric
    outcome) and the ``adjust`` columns, a ``censored`` one below its limit. Where the adjustment
    columns have blanks, or ``units`` (integer codes) says the rows repeat, each study factor's
    copies are drawn by the engine's chained equations on {factor, outcome, adjustment columns}
    (``smcfcs.impute``, ``"fcs"``); otherwise from one fit of its own model (see the module).
    Raises ``imputation.ImputationRefused`` where a study factor has fewer than two observed values
    or no finite censored-normal fit. Deterministic for a given ``seed``: each study factor draws
    from its own stream, by its place among ``tested``."""
    from turbotab.core.methods.imputation import ImputationRefused, column_kind
    from turbotab.core.methods.smcfcs import Variable
    from turbotab.core.methods.smcfcs import impute as fcs

    n = len(matrix)
    term = np.asarray(outcome, dtype=float).reshape(-1)
    adjust = [str(c) for c in adjust]
    Zframe = matrix[adjust] if adjust else pd.DataFrame(index=matrix.index)
    adjust_blank = [c for c in adjust if Zframe[c].isna().any()]
    chained = bool(adjust_blank) or units is not None
    sweeps = 1  # the most sweeps any study factor's chain ran
    drawn = drawn_factors(tested, [c for c in tested if matrix[c].isna().any()], adjust_blank)
    Z = Zframe.to_numpy(dtype=float) if adjust else np.zeros((n, 0))
    values: dict[str, np.ndarray] = {}
    covariates: dict[str, list[np.ndarray]] = {}
    imputed: dict[str, int] = {}
    D = None if chained else _design(term, Z)
    position = {c: j for j, c in enumerate(tested)}
    for done, name in enumerate(drawn):
        if cancelled is not None and cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()
        x = matrix[name].to_numpy(dtype=float)
        miss = ~np.isfinite(x)
        if miss.any() and int((~miss).sum()) < 2:
            refused = ImputationRefused(f"`{name}` has fewer than two observed values, so it cannot "
                                        f"be imputed.")
            refused.columns = [name]
            raise refused
        stream = np.random.SeedSequence([int(seed), int(position[name])])
        if not chained and name in censored:
            rng = np.random.default_rng(stream)
            copies = np.repeat(x[:, None], m, axis=1)
            copies[miss] = _censored_copies(D, x, miss, censored[name], m, rng, name)
            values[name] = copies
        else:
            frame = pd.DataFrame({name: x, OUTCOME_TERM: term}, index=matrix.index)
            for c in adjust:
                frame[c] = Zframe[c].to_numpy(dtype=float)
            variables = []
            if miss.any():
                kind = "censored" if name in censored else column_kind(frame[name])
                variables.append(Variable(name, kind, censoring=censored.get(name)))
            variables += [Variable(c, "censored", censoring=censored[c]) if c in censored
                          else Variable(c, column_kind(frame[c])) for c in adjust_blank]
            drawn_set = fcs(frame, variables, mode="fcs", units=units, m=m,
                            seed=int(stream.generate_state(1)[0]), cancelled=cancelled)
            values[name] = np.column_stack([f[name].to_numpy(dtype=float)
                                            for f in drawn_set.frames])
            if adjust_blank:
                covariates[name] = [f[adjust].to_numpy(dtype=float) for f in drawn_set.frames]
            sweeps = max(sweeps, int(drawn_set.iterations))
        imputed[name] = int(miss.sum())
        if progress is not None:
            progress(done + 1, len(drawn))
    return FeatureCopies(m=int(m), sweeps=int(sweeps), seed=int(seed), n_rows=int(n),
                         values=values, adjust=covariates, adjust_names=adjust, imputed=imputed,
                         n_incomplete=incomplete_rows(matrix, tested, adjust),
                         censored={c: censored[c] for c in [*drawn, *adjust_blank]
                                   if c in censored}, unit=unit,
                         asked=int(asked))


# ── the copies tested and pooled ─────────────────────────────────────────────


@dataclass
class PooledTests:
    """Each tested study factor's pooled test: the estimate, its standard error √T, Barnard and
    Rubin's ν, whether nothing of it is left once the covariates are projected out (in some copy),
    the fraction of missing information and the Monte Carlo error √(B/m); ``m`` the copies."""

    estimate: np.ndarray
    se: np.ndarray
    df: np.ndarray
    flat: np.ndarray
    fmi: np.ndarray
    mc_se: np.ndarray
    m: int


def pooled_tests(copies: FeatureCopies, matrix: pd.DataFrame, other: np.ndarray, binary: bool,
                 tested: Sequence[str], adjust: Sequence[str], clusters: Any,
                 refusal: Any = None) -> PooledTests:
    """Each copy of each study factor tested as the family tests the complete data
    (``featurewise.feature_tests``: least squares by Frisch–Waugh–Lovell, CR2 with Bell–McCaffrey
    df where the rows repeat; estimates alone under ``refusal``), on ``adjust`` (the columns this
    table adjusts for, a subset of those the copies were drawn with), then pooled per study factor
    by Rubin's rules with Barnard and Rubin's df (``imputation.pool_scalar``, ν_com each copy's
    df, their mean). A study factor no copy drew is tested once: its pooled row is its
    complete-data test (B = 0, so T = Ū and ν = ν_com)."""
    from turbotab.core.methods.imputation import pool_scalar
    from turbotab.core.models.featurewise import LEVEL, _basis, feature_tests

    n = len(matrix)
    adjust = [str(c) for c in adjust]
    k = len(tested)
    est = np.full(k, np.nan)
    se = np.full(k, np.nan)
    df = np.full(k, np.nan)
    flat = np.zeros(k, dtype=bool)
    fmi = np.full(k, np.nan)
    mc = np.full(k, np.nan)

    def design(Zv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        Zfull = np.column_stack([np.ones(n), Zv]) if Zv.size else np.ones((n, 1))
        Q = _basis(Zfull)
        return Zfull, Q, float(n - Q.shape[1] - 1)

    base = matrix[adjust].to_numpy(dtype=float) if adjust else np.zeros((n, 0))
    fixed = design(base) if np.isfinite(base).all() else None
    per: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    # The study factors whose copies share the adjustment columns: tested together, a block of
    # columns per factor (each copy one column), as the complete data's tests run.
    shared = [j for j, c in enumerate(tested) if c not in copies.adjust]
    if shared:
        if fixed is None:
            raise ValueError("The adjustment columns hold blanks no copy drew.")
        Zfull, Q, dfr = fixed
        blocks, spans = [], []
        for j in shared:
            values = copies.values.get(tested[j])
            block = (values if values is not None
                     else matrix[tested[j]].to_numpy(dtype=float).reshape(-1, 1))
            spans.append((j, block.shape[1]))
            blocks.append(block)
        X = np.column_stack(blocks)
        e, s, d, f = feature_tests(X, other, Q, Zfull, dfr, binary, clusters, refusal)
        at = 0
        for j, width in spans:
            per[j] = (e[at:at + width], s[at:at + width], d[at:at + width], f[at:at + width])
            at += width
    where = {c: i for i, c in enumerate(copies.adjust_names)}
    for j, c in enumerate(tested):
        if j in per:
            continue
        cols = [where[a] for a in adjust]
        out = [[], [], [], []]
        for kk, Zk in enumerate(copies.adjust[c]):
            Zfull, Q, dfr = design(Zk[:, cols] if cols else np.zeros((n, 0)))
            x = copies.values[c][:, kk].reshape(-1, 1)
            for slot, value in zip(out, feature_tests(x, other, Q, Zfull, dfr, binary, clusters,
                                                      refusal)):
                slot.append(value[0])
        per[j] = tuple(np.asarray(v) for v in out)  # type: ignore[assignment]
    for j in range(k):
        e, s, d, f = per[j]
        flat[j] = bool(np.any(f))
        if flat[j] or not np.all(np.isfinite(e)):
            continue
        if not (np.all(np.isfinite(s)) and np.all(s > 0)):
            est[j] = float(np.mean(e))  # estimates alone (a refused table's rows): their mean
            continue
        dfs = np.asarray(d, dtype=float)
        pooled = pool_scalar(e, s ** 2, float(np.mean(dfs)) if np.all(np.isfinite(dfs)) else None,
                             LEVEL)
        est[j] = pooled.estimate
        if pooled.total > 0 and math.isfinite(pooled.total):
            se[j] = math.sqrt(pooled.total)
            df[j] = pooled.df if pooled.df is not None else np.inf
        fmi[j] = pooled.fmi if pooled.fmi is not None else np.nan
        mc[j] = pooled.mc_se if pooled.mc_se is not None else np.nan
    return PooledTests(estimate=est, se=se, df=df, flat=flat, fmi=fmi, mc_se=mc, m=copies.m)


def first_copy(matrix: pd.DataFrame, copies: FeatureCopies) -> pd.DataFrame:
    """The matrix with each drawn study factor's first copy, and the adjustment columns' first
    copy where they were drawn (any study factor's: each holds its own draws of them): the
    complete matrix the table's names, rows, rank and refusals are read from."""
    out = matrix.copy()
    for c, values in copies.values.items():
        if c in out.columns:
            out[c] = values[:, 0]
    if copies.adjust:
        first = next(iter(copies.adjust.values()))[0]
        for i, c in enumerate(copies.adjust_names):
            if c in out.columns and out[c].isna().any():
                out[c] = first[:, i]
    return out


def caption_of(copies: FeatureCopies) -> str:
    """The pooled table's first sentence: how its copies were drawn and pooled."""
    censored = bool(copies.censored)
    plain = [c for c in copies.values if c not in copies.censored and copies.imputed.get(c)]
    how = []
    if censored:
        how.append("below a detection limit from a censored-normal (Tobit) model")
    if plain:
        how.append("otherwise from a normal linear model" if censored else
                   "from a normal linear model")
    drawn = f" ({', '.join(how)})" if how else ""
    chained = []
    if copies.adjust:
        chained.append("the covariates' blanks drawn with it by chained equations")
    if copies.unit:
        chained.append(f"the rows clustered by `{copies.unit}`, so by chained equations with the "
                       f"`{copies.unit}` means of the other variables and its own mean over the "
                       f"`{copies.unit}`'s other rows")
    beside = f", {' and '.join(chained)}" if chained else ""
    given = "the outcome and the covariates" if copies.adjust_names else "the outcome"
    return (f"Multiple imputation, m = {copies.m}, one study factor at a time: each study "
            f"factor's blanks were drawn from its own model given {given}{drawn}{beside}; each "
            f"completed copy was analyzed as follows, and each study factor pooled by Rubin's "
            f"rules.")


def as_imputations(copies: FeatureCopies, adjust: Sequence[str]) -> Any:
    """The copies as an ``imputation.Imputations`` record (no frames: each study factor's copies are
    its own), so the pooled table's record and concerns are written as every pooled table's
    (``missing.multiple_imputation_info``, ``missing.mi_concerns``). Its method is chained
    equations, on each study factor's own variables (one sweep where the factor's blanks are the
    only ones: the monotone case), and its variables lead with
    ``missing.PER_FEATURE_VARIABLE``, which the record's readers take it by."""
    from turbotab.core.methods.imputation import Imputations
    from turbotab.core.methods.missing import PER_FEATURE_VARIABLE

    return Imputations(
        frames=[], m=copies.m, iterations=copies.sweeps,
        imputed={c: v for c, v in copies.imputed.items() if v},
        n_incomplete_rows=int(copies.n_incomplete),
        variables=[PER_FEATURE_VARIABLE, OUTCOME_TERM, *[str(a) for a in adjust]], kinds={},
        censored=dict(copies.censored), seed=copies.seed, method="chained_equations",
        plan={"m_asked": copies.asked, "unit": copies.unit,
              "row_level": [PER_FEATURE_VARIABLE] if copies.unit else None})


# The record's note: what uses these copies (the joint imputation's note speaks of scores, which
# the feature-wise tests have none of).
NOTE = ("Feature-wise regression makes no predictions; its tests alone use these imputations, drawn "
        "one study factor at a time.")


def outcome_term(y: Any, task: str) -> np.ndarray:
    """The outcome as each study factor's imputation model and test read it: the 0/1 event of a
    two-level outcome (the limma design's regressor), or the numeric outcome."""
    from turbotab.core.models.featurewise import event_indicator

    if task == "binary":
        return event_indicator(y)[0]
    if task == "regression":
        return np.asarray(y, dtype=float)
    raise ValueError("Feature-wise regression models a numeric or a two-level outcome.")


def copies_for(matrix: pd.DataFrame, y: Any, task: str, features: Sequence[str] | None,
               clusters: Any = None, *, censored: Mapping[str, Any], m: int | None = None,
               asked: int = 20, seed: int = 0,
               progress: Callable[[int, int], None] | None = None,
               cancelled: Callable[[], bool] | None = None) -> FeatureCopies | None:
    """The copies the pooled table draws on ``matrix`` (the model matrix with its blanks,
    :func:`prepare`; ``censored`` each censored column's limit on it): m by the engine's rule over
    ``asked`` (:func:`imputation_count`) unless given, within :data:`MAX_FEATURE_COPIES`
    (:class:`Unpooled` beyond it). None where no study factor is tested or nothing is blank."""
    from turbotab.core.models.featurewise import split_columns
    from turbotab.core.models.inference import INDEPENDENT

    clusters = INDEPENDENT if clusters is None else clusters
    tested, adjust = split_columns(list(matrix.columns), features)
    if not tested or not any(matrix[c].isna().any() for c in [*tested, *adjust]):
        return None
    other = outcome_term(y, task)
    units = clusters.codes if clusters.clustered else None
    if m is None:
        m = imputation_count(matrix, tested, adjust, asked)
    blank_z = [c for c in adjust if matrix[c].isna().any()]
    own = [c for c in tested if matrix[c].isna().any()]
    over = budget_refusal(draws_of(tested, own, blank_z, units), int(m))
    if over is not None:
        raise Unpooled(over)
    return draw_copies(matrix, other, tested, adjust, censored, int(m), seed=seed, units=units,
                       unit=clusters.column if clusters.clustered else None, asked=asked,
                       progress=progress, cancelled=cancelled)


def pooled_featurewise_table(matrix: pd.DataFrame, y: Any, task: str,
                             features: Sequence[str] | None, clusters: Any = None,
                             event: str | None = None, *, censored: Mapping[str, Any],
                             m: int | None = None, asked: int = 20, seed: int = 0,
                             named: Sequence[str] | None = None,
                             forms: Mapping[str, Any] | None = None,
                             entered: Mapping[str, Sequence[str]] | None = None,
                             copies: FeatureCopies | None = None, spec: Any = None,
                             progress: Callable[[int, int], None] | None = None,
                             cancelled: Callable[[], bool] | None = None
                             ) -> tuple[Any, FeatureCopies | None]:
    """The feature-wise table pooled over imputations drawn one study factor at a time, and the
    copies: ``matrix`` is the model matrix with its blanks (:func:`prepare`), ``censored`` each
    censored column's limit on it, ``m`` the imputations (default the engine's rule over
    ``asked``, :func:`copies_for`); ``copies``, when given (Table 2's models share the primary's),
    are tested on this matrix's own adjustment columns instead of drawing anew. The table's
    refusals before any test (a formed study factor, none in the matrix) are made before any copy
    is drawn; draws beyond :data:`MAX_FEATURE_COPIES` raise :class:`Unpooled`. Where this matrix
    holds no blank the table is the complete data's (no copy is drawn, and none is said)."""
    from turbotab.core.methods.missing import mi_concerns, multiple_imputation_info
    from turbotab.core.models.featurewise import (CR2_MAX_FEATURES, _basis, featurewise_table,
                                                  split_columns)
    from turbotab.core.models.inference import INDEPENDENT, floor_refusal

    clusters = INDEPENDENT if clusters is None else clusters
    tested, adjust = split_columns(list(matrix.columns), features)
    plain = forms or not tested or not any(matrix[c].isna().any() for c in [*tested, *adjust])
    if plain:
        return featurewise_table(matrix, y, task, features, clusters, event, named=named,
                                 forms=forms, entered=entered), None
    if copies is None:
        copies = copies_for(matrix, y, task, features, clusters, censored=censored, m=m,
                            asked=asked, seed=seed, progress=progress, cancelled=cancelled)
    other = outcome_term(y, task)
    complete = first_copy(matrix, copies)
    Zfirst = complete[adjust].to_numpy(dtype=float) if adjust else np.zeros((len(complete), 0))
    Q = _basis(np.column_stack([np.ones(len(complete)), Zfirst]) if adjust
               else np.ones((len(complete), 1)))
    # The table's own refusals of intervals (too few clusters, more study factors than CR2 is
    # computed for, no residual df): each copy then gives its least-squares estimates alone.
    refusal = floor_refusal(clusters)
    if refusal is None and clusters.clustered and len(tested) > CR2_MAX_FEATURES:
        refusal = ("beyond", ())
    if refusal is None and len(complete) - Q.shape[1] - 1 < 1:
        refusal = ("no df", ())
    pooled = pooled_tests(copies, complete, other, task == "binary", tested, adjust, clusters,
                          refusal)
    table = featurewise_table(complete, y, task, features, clusters, event, named=named,
                              forms=forms, entered=entered, pooled=pooled)
    if table.rows and any(r.get("estimate") is not None for r in table.rows):
        table.info["caption"] = f"{caption_of(copies)} {table.info.get('caption', '')}".strip()
        record_spec = spec if spec is not None else _Spec(list(matrix.columns))
        imputations = as_imputations(copies, adjust)
        table.info["missing"] = {**multiple_imputation_info(imputations, record_spec,
                                                            len(complete), rows=table.rows),
                                 "note": NOTE}
        for c in mi_concerns(imputations, table.rows, len(complete)):
            if c not in table.concerns:
                table.concerns.append(c)
    return table, copies


class _Spec:
    """What ``missing.multiple_imputation_info`` reads of a design where none is at hand (Table 2's
    refits on a matrix): the columns, no roles, no energy fill, no censored record."""

    def __init__(self, inputs: Sequence[str]):
        self.inputs = list(inputs)
        self.roles: dict[str, str] = {}
        self.energy_fill = None
        self.censored = None


__all__ = ["FeatureCopies", "KeepBelowDetection", "KeepBlanks", "MARGINAL_UNPOOLED_CODE",
           "MAX_FEATURE_COPIES", "NOTE", "POOLED_ROUTE", "PooledTests", "Prepared",
           "SINGLE_FILL", "SINGLE_FILL_WORDS", "UNPOOLED", "Unpooled", "as_imputations",
           "blank_pipeline", "budget_refusal", "caption_of", "copies_for", "draw_copies",
           "draws_of", "drawn_factors", "first_copy", "imputation_count", "incomplete_rows",
           "marginal_refusal", "outcome_term", "plan_frame", "plan_refusal", "pooled_answer",
           "pooled_featurewise_table", "pooled_tests", "prepare", "single_censoring_fill",
           "single_fill_refusal", "sweeps_of", "ways_forward"]

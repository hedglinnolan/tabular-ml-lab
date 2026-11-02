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
one sweep each would do (each would refit the same model on the same observed values). A censored
column is drawn only where its model has one finite maximum-likelihood fit, read from the design
and never from the solver (:func:`identification_refusal`; Olsen 1978): one detected only in rows
of one outcome, or whose detected values are all equal, is refused before any draw, and one with
fewer detected values than terms is drawn where the rows below detection identify the rest. A
chained model the sampler cannot fit is refused, never drawn (:func:`chain_refusal`). On repeated
rows a censored column's chained model holds the other variables' unit means, so beside a complete
column that is one value per unit two of its terms are one: whether the solver still fits it rests
on rounding (in the second repair round's check, 4 of 10 panels; their pooled estimates were those
of the same chain without the repeated term within one Monte Carlo error), and where it does not
the refusal says why.

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
    from turbotab.core.voice import plural

    n = len(sweeps)
    lead = ("Feature-wise regression is pooled over multiple imputations drawn one study factor at "
            "a time")
    tail = (f"more than the {MAX_FEATURE_COPIES:,} one table may draw, a limit that bounds how "
            f"long its fit takes.")
    groups: dict[int, int] = {}
    for s in sweeps:
        groups[int(s)] = groups.get(int(s), 0) + 1
    if len(groups) == 1:
        (s,) = groups
        each = f" × {s} sweeps of chained equations" if s > 1 else ""
        return (f"{lead}: here {n:,} {plural(n, 'study factor', 'study factors')} × {m} "
                f"imputations{each} make {draws:,} draws, {tail}")
    # Some by chained equations, some once in each imputation (the verifier's FWMI-R1-4): each
    # group's count with its sweeps and m, so the sum can be followed.
    parts = []
    for s, k in sorted(groups.items(), reverse=True):
        who = (f"{k:,} {plural(k, 'study factor is', 'study factors are')} drawn" if not parts
               else f"{k:,}")
        imputations = "" if parts else " imputations"
        parts.append(f"{who} by chained equations, {s} sweeps in each of {m} imputations" if s > 1
                     else f"{who} once in each of {m}{imputations}")
    return f"{lead}: here {', and '.join(parts)}, {draws:,} draws in all, {tail}"


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


def _split_grouping(ctx: Any) -> str | None:
    """The column the split grouped a unit's rows by, where the context serves the split (the
    fit reads it from the split it is handed, ``stages.modeling.fit_stage``)."""
    from turbotab.core.decisions import _ctx

    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return None
    try:
        split = artifact("split")
    except Exception:  # noqa: BLE001 - no split yet: the rows are grouped by none
        return None
    found = split.get("grouped_by") if isinstance(split, Mapping) else None
    return str(found) if found else None


def plan_units(state: Any, ctx: Any, rows: Any) -> np.ndarray | None:
    """Each of the plan's ``rows`` (row ids) by the unit the fit clusters them by, as it does
    (``inference.resolve_clusters`` over ``inference.cluster_columns``: the grain's column, the
    identifiers and clusters, the cluster answer and the split's grouping, on the rows the
    analysis reads), so the copies a check before the lock counts are the ones the fit draws;
    None where the rows do not cluster or the table cannot be read."""
    from turbotab.core.decisions import _store_of
    from turbotab.core.models.inference import cluster_columns, resolve_clusters

    store = _store_of(ctx)
    if store is None:
        return None
    try:
        grouped_by = _split_grouping(ctx)
        columns = cluster_columns(state, list(store.columns), [grouped_by])
        if not columns:
            return None
        frame = store.materialize(columns)
    except Exception:  # noqa: BLE001 - an unreadable table: the fit counts them
        return None
    frame = frame.loc[frame.index.isin(np.asarray(rows))]
    clusters = resolve_clusters(state, frame, [grouped_by])
    return clusters.codes if clusters.clustered else None


def plan_categories(state: Any, frame: pd.DataFrame) -> set[str]:
    """The plan's columns the design one-hot encodes (``pipeline.category_inputs``: text, and a
    number declared as codes), which the copies, drawing numbers, do not draw."""
    from turbotab.core.models.pipeline import category_inputs

    return set(category_inputs(state, frame, list(frame.columns)))


def plan_levels(answer: Mapping[str, Any], ctx: Any, frame: pd.DataFrame) -> set[str]:
    """The plan's columns whose blanks the answer keeps as a level of their own
    (``categorical == "missing_category"``), as the design reads them
    (``pipeline.level_columns``: categorical or two-valued, over the table's column summaries
    where the context's table serves them, else ``pipeline.frame_level_columns`` on ``frame``): no
    blank of theirs reaches the model matrix, so none is drawn."""
    from types import SimpleNamespace

    from turbotab.core.decisions import _store_of
    from turbotab.core.models.pipeline import frame_level_columns, level_columns

    if answer.get("categorical") != "missing_category":
        return set()
    # The design reads the answer from the state's slot; here the answer is the one being checked.
    reading = SimpleNamespace(missing=SimpleNamespace(categorical="missing_category"))
    columns = list(frame.columns)
    store = _store_of(ctx)
    info = None
    try:
        info = {c.name: c for c in store.info().columns} if hasattr(store, "info") else None
    except Exception:  # noqa: BLE001 - no column summaries: the plan's rows stand in
        info = None
    found = (level_columns(reading, columns, info) if info is not None
             else frame_level_columns(reading, frame, columns))
    return set(found)


def plan_refusal(state: Any, answer: Mapping[str, Any], ctx: Any
                 ) -> tuple[str, list[dict[str, Any]]] | None:
    """Why the plan's feature-wise table cannot pool the copies its multiple-imputation answer
    draws, counted on the plan's rows (:func:`plan_frame`): a category among the covariates with
    blanks (as the design reads categories, :func:`plan_categories`), or draws beyond
    :data:`MAX_FEATURE_COPIES` (the study factors as the roles answer gives them, before the
    design's filters, so never fewer than the fit draws; chained where the rows cluster as the fit
    clusters them, :func:`plan_units`). A column whose blanks the answer keeps as a level of their
    own is neither (:func:`plan_levels`): the fit encodes them, and draws none. With its ways
    forward (:func:`ways_forward`). None where it can, or where the rows cannot be read: the fit
    then counts them and refuses in the same words."""
    from turbotab.core.methods.imputation import M_DEFAULT
    from turbotab.core.models.pipeline import MISSING_LEVEL
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
    levels = plan_levels(answer, ctx, frame)
    if levels:
        # Their blanks are a level the design encodes: counted as values, never as blanks.
        frame = frame.assign(**{c: frame[c].astype(object).where(frame[c].notna(), MISSING_LEVEL)
                                for c in levels})
    categories = plan_categories(state, frame)
    hidden = [c for c in adjust if frame[c].isna().any() and c in categories]
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
    over = budget_refusal(draws_of(tested, drawn, blank_z, plan_units(state, ctx, frame.index)), m)
    return (over, exits) if over is not None else None


MARGINAL_UNPOOLED_CODE = "marginal_risks_unpooled"


def marginal_refusal(state: Any, models: Sequence[str], answer: Mapping[str, Any], ctx: Any
                     ) -> tuple[str, list[dict[str, Any]]] | None:
    """The check before the lock's refusal of the marginal risk difference or ratio under multiple
    imputation: where the declared estimand is one, a chosen family standardizes it
    (``effects.standardizes_risks``) and a predictor the plan's models get is blank on its rows
    (:func:`plan_frame`), the effects stage would draw copies and refuse the risks
    (``effects.MARGINAL_UNPOOLED``); so the conditional odds ratio, or complete cases where the
    record would accept them (``row_floor.accepted_exit``). None otherwise, or where the rows
    cannot be read."""
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
    from turbotab.core.row_floor import accepted_exit

    frame = plan_frame(state, ctx)
    gone = set(answer.get("drop_columns") or [])
    if frame is None or not any(frame[c].isna().any() for c in frame.columns if c not in gone):
        return None
    # Complete cases only where the record accepts them now (``decisions._accepted_ways``'s rule):
    # ones that would leave too few rows, or one value of the outcome, are refused when taken.
    whole = {**{k: v for k, v in dict(answer).items() if k != "kind"}, "kind": "set_missing",
             "strategy": "complete_case", "indicators": False, "categorical": "impute"}
    cases = ([{"label": "Complete cases, with their assumption stated", "decision": whole}]
             if accepted_exit(whole, ctx) is not None else [])
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
    any chain ran, the seed, the unit the rows repeat by (None where they do not), and the model
    each drawn column's blanks came from (``kinds``: ``"censored"``, the censored normal;
    ``"binary"``, a logistic regression; ``"numeric"``, a normal linear model)."""

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
    kinds: dict[str, str] = field(default_factory=dict)


def _design(term: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """A study factor's imputation design as the engine's chains build one: the intercept and
    every other variable standardized (the outcome's term, then the adjustment columns)."""
    from turbotab.core.methods.imputation import _standardize

    block = np.column_stack([term, Z]) if Z.size else term.reshape(-1, 1)
    return np.column_stack([np.ones(len(term)), _standardize(block)])


# A fit of the detected values whose residual sum of squares is this small a share of their sum of
# squares fits them exactly (the share is about σ²/(σ² + μ²) for values with any spread; exactly 0
# for equal ones).
EXACT_FIT = 1e-20
# An exact fit of the detected values puts a row below detection at or below the limit where it
# lies above it by no more than this share of the values' scale (rounding: 0 for equal values).
AT_LIMIT = 1e-9


def _on_scale(x: np.ndarray, censoring: Any) -> np.ndarray:
    """A censored column's values on the scale its censored normal is fitted on."""
    return np.log(np.clip(x, 1e-300, None)) if censoring.log else x.astype(float)


def _rank(A: np.ndarray) -> tuple[int, np.ndarray]:
    """(the rank of ``A`` at numpy's tolerance, ``matrix_rank``'s, an orthonormal basis of its null
    space as columns)."""
    k = A.shape[1]
    if not A.shape[0]:
        return 0, np.eye(k)
    _, s, Vt = np.linalg.svd(A, full_matrices=A.shape[0] < k)  # Vt is k × k either way
    tol = (s.max() if len(s) else 0.0) * max(A.shape) * np.finfo(float).eps
    r = int(np.sum(s > tol))
    return r, Vt[r:].T


def _rises_along_unestimated(Dd: np.ndarray, Dc: np.ndarray) -> bool:
    """Whether some combination v of the terms the detected rows leave unestimated (D_d v = 0)
    moves no row below detection up and some down (D_c v ≤ 0, Σ D_c v < 0): along it each such
    row's likelihood rises toward a bound it never reaches, so the fit has no finite maximum. A
    linear program, never the Tobit solver."""
    from scipy.optimize import linprog

    _, N = _rank(Dd)
    if not N.shape[1]:
        return False
    C = Dc @ N
    k = N.shape[1]
    found = linprog(np.zeros(k), A_ub=np.vstack([C, C.sum(axis=0)]),
                    b_ub=np.r_[np.zeros(len(C)), -1.0], bounds=[(None, None)] * k,
                    method="highs")
    return found.status != 2  # 2: infeasible, no such combination


def _exact_fit_at_or_below(Dd: np.ndarray, zd: np.ndarray, Dc: np.ndarray, limit: float) -> bool:
    """Whether some exact fit of the detected values (D_d β = z_d) puts every row below detection
    at or below the limit (D_c β ≤ L): then σ → 0 raises the likelihood without bound. A linear
    program over the exact fits (one, or a line or more where D_d is rank-deficient)."""
    from scipy.optimize import linprog

    beta = np.linalg.lstsq(Dd, zd, rcond=None)[0]
    resid = zd - Dd @ beta
    if float(resid @ resid) > EXACT_FIT * max(float(zd @ zd), np.finfo(float).tiny):
        return False
    scale = max(1.0, abs(float(limit)), float(np.max(np.abs(zd))) if len(zd) else 0.0)
    excess = Dc @ beta - limit
    _, N = _rank(Dd)
    if not N.shape[1]:
        worst = float(excess.max())
    else:
        # The least worst excess over the exact fits β + N w: min s with D_c N w − s ≤ −excess.
        C = Dc @ N
        k = N.shape[1]
        found = linprog(np.r_[np.zeros(k), 1.0], A_ub=np.column_stack([C, -np.ones(len(C))]),
                        b_ub=-excess, bounds=[(None, None)] * k + [(-scale, None)],
                        method="highs")
        worst = float(found.fun) if found.status == 0 else -scale
    return worst <= AT_LIMIT * scale


def identification_refusal(name: str, z: np.ndarray, miss: np.ndarray, term: np.ndarray,
                           Z: np.ndarray, limit: float | None = None) -> str | None:
    """Why ``name``'s censored-normal (Tobit) model of ``z`` (on its fitted scale; ``miss`` its
    values below detection, ``limit`` theirs on that scale, else the smallest detected value) on
    the outcome's ``term`` and the complete columns ``Z`` has no one maximum-likelihood fit; None
    where it has one. Under Olsen's (1978) reparametrization (γ, τ) = (β/σ, 1/σ) the
    log-likelihood is concave, strictly where the design D has full column rank on every row, and
    a concave function has a maximum, and one only, where no direction but zero leaves it
    non-decreasing (Rockafellar 1970, Theorem 27.1). Here such a direction is a combination of
    terms D leaves unestimated on every row (refused: no one fit is best), a combination the
    detected rows leave unestimated (D_d v = 0) along which no row below detection moves up
    (:func:`_rises_along_unestimated`; with a two-level outcome, every detected value in rows of
    one outcome is one), or an exact fit of the detected values putting every row below
    detection at or below the limit, along which σ → 0 (:func:`_exact_fit_at_or_below`; equal
    detected values are one). Fewer detected values than terms is none of these by itself: the
    rows below detection can carry the rest (survreg fits such a model to the same maximum).
    Each is read from the design by linear algebra and linear programs, never from the Tobit
    solver. Where the copies are drawn by chained equations ``Z`` is the complete part of the
    model's predictors: each reason found there holds beside any other column (a direction with
    zero on the others)."""
    detected = ~miss
    n_d = int(detected.sum())
    if not miss.any():
        return None
    D = _design(term, Z)
    p = D.shape[1]
    lead = f"`{name}`'s censored-normal model has no finite fit on these rows: "
    tail = "its values below detection cannot be drawn."
    terms = "the intercept and the outcome" + (" and the covariates" if Z.shape[1] else "")
    if _rank(D)[0] < p:
        return (f"`{name}`'s censored-normal model has no single fit on these rows: the {p} terms "
                f"of its model ({terms}) are tied exactly to one another on them, so no one fit is "
                f"best, and {tail}")
    Dd, zd, Dc = D[detected], z[detected], D[miss]
    at = float(limit) if limit is not None else (float(zd.min()) if n_d else 0.0)
    if n_d and _exact_fit_at_or_below(Dd, zd, Dc, at):
        how = ("are all equal" if np.ptp(zd) == 0
               else "are fitted exactly by the outcome and the covariates, by a fit that puts "
                    "every value below detection at or below the limit")
        return (f"{lead}its {n_d:,} detected values {how}, so their spread cannot be estimated, "
                f"and {tail}")
    if _rises_along_unestimated(Dd, Dc):
        if n_d and np.ptp(term) > 0 and np.ptp(term[detected]) == 0:
            return (f"{lead}all {n_d:,} of its detected values are in rows with the same outcome, "
                    f"so how far below the limit the other rows lie cannot be estimated, and "
                    f"{tail}")
        return (f"{lead}its {n_d:,} detected values leave a combination of the {p} terms of its "
                f"model ({terms}) unestimated, and the rows below detection fit ever better along "
                f"it, so how far below the limit they lie cannot be estimated, and {tail}")
    return None


def unit_repeats(complete: Mapping[str, np.ndarray], units: np.ndarray) -> list[str]:
    """The complete columns of a chained model on repeated rows (``complete``, keyed by the words
    that name them: the outcome's term, the complete covariates) that are the same on every row of
    their unit (``units``: integer codes), so the unit mean the engine's chained equations hold
    beside each (``smcfcs.impute``) repeats it: two terms of the model are one, and a censored
    model's information then has no inverse (the verifier's FWMI-R1-1). Whether the solver still
    returns a fit rests on rounding, so this only says why, where it did not."""
    codes = np.asarray(units, dtype=np.int64)
    G = int(codes.max()) + 1 if len(codes) else 0
    size = np.bincount(codes, minlength=G).astype(float)
    size[size == 0] = 1.0
    out = []
    for label, values in complete.items():
        v = np.asarray(values, dtype=float)
        mean = (np.bincount(codes, weights=v, minlength=G) / size)[codes]
        spread = float(np.max(np.abs(v))) if len(v) else 0.0
        if np.ptp(v) > 0 and np.all(np.abs(v - mean) <= 1e-12 * max(1.0, spread)):
            out.append(label)
    return out


def chain_refusal(name: str, frame: pd.DataFrame, units: np.ndarray | None,
                  unit: str | None) -> str:
    """The refusal of a study factor whose chained equations (on ``frame``, its chain's variables)
    could not fit their models: where the rows repeat, with the complete columns its chain holds
    beside unit means that repeat them (:func:`unit_repeats`: the outcome's term, the complete
    covariates, and the study factor itself where it has no blank)."""
    from turbotab.core.voice import listing

    why = ""
    if units is not None:
        complete = {("the outcome" if c == OUTCOME_TERM else f"`{c}`"):
                    frame[c].to_numpy(dtype=float)
                    for c in frame.columns if not frame[c].isna().any()}
        repeated = unit_repeats(complete, units)
        if repeated:
            by = f"`{unit}`" if unit else "unit"
            one = len(repeated) == 1
            repeats = "that mean repeats it" if one else "those means repeat them"
            why = (f": on rows that repeat by {by} each of their models holds each other "
                   f"variable's mean over the {by}'s rows beside it, and "
                   f"{listing(repeated, ticked=False)} {'is' if one else 'are'} the same on every "
                   f"row of a {by}, so {repeats}")
    return (f"The chained equations that draw `{name}`'s copies could not fit their models on "
            f"these rows, so its copies cannot be drawn{why}.")


def _refuse(name: str, reason: str) -> Any:
    from turbotab.core.methods.imputation import ImputationRefused

    refused = ImputationRefused(reason)
    refused.columns = [name]
    return refused


def _censored_copies(D: np.ndarray, x: np.ndarray, miss: np.ndarray, censoring: Any, m: int,
                     rng: np.random.Generator, name: str) -> np.ndarray:
    """The m draws of a censored study factor's blanks (n_blank × m) from one censored-normal fit
    on ``D``: θ̂ and its inverse observed information (``imputation.tobit_fit``), then for each copy
    θ* ~ N(θ̂, I⁻¹) and each blank from N(Dβ*, σ*²) truncated above at the limit, as
    ``imputation``'s censored draw makes one copy. The caller has checked that the model has a
    finite fit (:func:`identification_refusal`); a solver that still returns none is refused."""
    from scipy import stats

    from turbotab.core.methods.imputation import tobit_fit

    z = _on_scale(x, censoring)
    limit = math.log(censoring.limit) if censoring.log else float(censoring.limit)
    try:
        theta, V = tobit_fit(D, np.where(miss, limit, z), miss, limit)
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(V))):
            raise np.linalg.LinAlgError
        root = np.linalg.cholesky(V + np.eye(len(V)) * 1e-12)
    except (np.linalg.LinAlgError, ValueError):
        raise _refuse(name, f"`{name}`'s censored-normal model could not be fit on these rows, so "
                            f"its values below detection cannot be drawn.") from None
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
    Raises ``imputation.ImputationRefused`` where a study factor has fewer than two observed values,
    or a censored column's model no one finite fit (:func:`identification_refusal`), before any
    is drawn, and where the chained equations cannot fit a model (a LinAlgError, never a copy:
    :func:`chain_refusal`). Deterministic for a given ``seed``: each study factor draws from its
    own stream, by its place among ``tested``. ``kinds`` records the model each drawn column's
    blanks came from, as the sampler was handed it."""
    from turbotab.core.methods.imputation import column_kind
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
    kinds: dict[str, str] = {}
    D = None if chained else _design(term, Z)
    position = {c: j for j, c in enumerate(tested)}
    whole = [c for c in adjust if c not in set(adjust_blank)]

    def identified(column: str, x: np.ndarray, others: Sequence[str]) -> None:
        # A censored column's model is fit only where it has a finite fit (Olsen 1978): its
        # complete predictors are the outcome's term and ``others``.
        Zc = Zframe[list(others)].to_numpy(dtype=float) if others else np.zeros((n, 0))
        c = censored[column]
        reason = identification_refusal(column, _on_scale(x, c), ~np.isfinite(x), term, Zc,
                                        limit=math.log(c.limit) if c.log else float(c.limit))
        if reason is not None:
            raise _refuse(column, reason)

    for c in adjust_blank:
        if c in censored:
            identified(c, Zframe[c].to_numpy(dtype=float), [a for a in whole if a != c])
    for done, name in enumerate(drawn):
        if cancelled is not None and cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()
        x = matrix[name].to_numpy(dtype=float)
        miss = ~np.isfinite(x)
        if miss.any() and int((~miss).sum()) < 2:
            raise _refuse(name, f"`{name}` has fewer than two observed values, so it cannot be "
                                f"imputed.")
        if miss.any() and name in censored:
            identified(name, x, whole)
        stream = np.random.SeedSequence([int(seed), int(position[name])])
        if not chained and name in censored:
            rng = np.random.default_rng(stream)
            copies = np.repeat(x[:, None], m, axis=1)
            copies[miss] = _censored_copies(D, x, miss, censored[name], m, rng, name)
            values[name] = copies
            kinds[name] = "censored"
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
            kinds.update({v.name: str(v.kind) for v in variables})
            try:
                drawn_set = fcs(frame, variables, mode="fcs", units=units, m=m,
                                seed=int(stream.generate_state(1)[0]), cancelled=cancelled)
            except np.linalg.LinAlgError:
                # A model the chain cannot fit (the verifier's FWMI-R1-1: a censored model's
                # information with no inverse, raised from the sampler) is refused, never a copy;
                # where rows repeat and a complete column is one value per unit, that says why.
                raise _refuse(name, chain_refusal(name, frame, units, unit)) from None
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
                         asked=int(asked), kinds=kinds)


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
    """The pooled table's first sentence: how its copies were drawn and pooled, each study
    factor's own blanks by the model the sampler drew them from (``FeatureCopies.kinds``)."""
    own = {copies.kinds.get(c) for c in copies.values if copies.imputed.get(c)}
    how = []
    if "censored" in own:
        how.append("below a detection limit from a censored-normal (Tobit) model")
    if "binary" in own:
        how.append("a two-valued study factor's from a logistic regression")
    if "numeric" in own:
        how.append("otherwise from a normal linear model" if how else "from a normal linear model")
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
    only ones: the monotone case), its variables lead with ``missing.PER_FEATURE_VARIABLE``,
    which the record's readers take it by, and its ``kinds`` are the models each drawn column's
    blanks came from, which its methods sentence names (``missing.mi_sentence``)."""
    from turbotab.core.methods.imputation import Imputations
    from turbotab.core.methods.missing import PER_FEATURE_VARIABLE

    return Imputations(
        frames=[], m=copies.m, iterations=copies.sweeps,
        imputed={c: v for c, v in copies.imputed.items() if v},
        n_incomplete_rows=int(copies.n_incomplete),
        variables=[PER_FEATURE_VARIABLE, OUTCOME_TERM, *[str(a) for a in adjust]],
        kinds=dict(copies.kinds),
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

"""Explanations of fitted models: SHAP, interaction ranking, inductive-bias curves and the
architecture lane (V2 definition of done §2, "Explainability"; MODELING_SEQUENCE §1 row 9, "each
family's inductive bias stated").

**What is explained.** A fitted pipeline runs ``detect → normalize → impute → energy`` (the
lineage's *adjusted* lane, :data:`~turbotab.core.models.pipeline.ADJUST_STEPS`) and then
``form → levels → one-hot → scale → model``. Every explanation here is of the model's inputs as
the adjusted lane names them (``protein_adj`` under the residual method, ``kcal``, ``gender``): the
exposure on its *final* scale, the scale MODELING_SEQUENCE §1 row 5 declares a form on. SHAP
attributions of the model-matrix columns are summed over the columns one adjusted input became
(a one-hot's indicators, a spline's basis); curves and interactions move an adjusted input and
push it through the rest of the pipeline. Everything is on the model's own scale: the outcome's
units for a numeric outcome, the log-odds of the event for a yes/no one.

**SHAP** (Lundberg & Lee 2017, NeurIPS; Lundberg et al. 2020, *Nat Mach Intell* 2:56–67):

* a linear model (least squares, logistic, the elastic net): the exact closed form under
  independent features, ``φ_j(x) = β_j (z_j − E[z_j])`` with ``E`` over the rows the model was fit
  on (Lundberg & Lee 2017, Linear SHAP);
* boosted trees: path-dependent TreeSHAP, computed exactly from the trees' leaf paths. Each leaf's
  path-dependent value function is a product over the distinct features on its path,
  ``v(S) = value · Π_{d∈S} o_d · Π_{d∉S} z_d`` (``o_d``: the row satisfies every split on ``d``
  along the path; ``z_d``: the product of the training-cover fractions of those splits), so its
  Shapley values have a closed form, ``φ_j = value · (o_j − z_j) · Σ_s w(s) e_s``, with ``e_s`` the
  coefficient of ``t^s`` in ``Π_{i≠j} (z_i + o_i t)`` and ``w(s) = s!(k−1−s)!/k!``. This is the
  decomposition GPU TreeSHAP uses (Mitchell et al. 2022); it gives Lundberg's Algorithm 2 values.

Stability: the family is refit on :data:`RESEEDS` bootstrap resamples of its training rows (whole
units when rows repeat, MODELING_SEQUENCE §2), each with its own seed, and the mean |SHAP| of the
same rows is recomputed; the report is the Spearman correlation of those importances between every
pair of refits, and each input's rank across them.

**Interactions**: Friedman & Popescu's H statistic (2008, *Ann Appl Stat* 2:916–954) between the
top inputs, from partial dependence evaluated at the rows themselves and centered,
``H²_jk = Σ_i [F_jk − F_j − F_k]² / Σ_i F_jk²``. A pair whose joint effect is weak can have a large
H² (Molnar, *Interpretable Machine Learning*, "the H-statistic will be very large"), so pairs are
ranked by the interaction's own size on the model's scale, the root mean square of
``F_jk − F_j − F_k``, with H² beside it.

**Inductive-bias curves**: accumulated local effects (Apley & Zhu 2020, *JRSS B* 82:1059–1086),
``ĝ(x) = Σ_{k ≤ k(x)} mean_{i ∈ N(k)} [f(z_k, x_i,\\j) − f(z_{k−1}, x_i,\\j)]``, centered by
subtracting ``(1/n) Σ_i ĝ(x_ij)``, on a grid of the input's observed quantiles shared by every
family. A segment with fewer than :data:`MIN_SEGMENT_ROWS` rows is masked: no curve is drawn where
the data are absent. A family whose cross-validated score does not beat the no-predictor baseline
draws no curve, with that reason (Molnar et al. 2022, LNCS 13200: "interpreting models that do not
generalize well").

**What they are not.** Every explanation describes the fitted model's predictions. None is a
causal effect (Molnar et al. 2022: "making unjustified causal interpretations"), and under
inference none is offered as an effect estimate: the effect is the declared estimand's estimate
(BLUEPRINT §11.3; MODELING_SEQUENCE §4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# ── the leash's words (asserted verbatim) ────────────────────────────────────

DESCRIBES = ("These explanations describe how each fitted model turns its inputs into "
             "predictions. They are not causal effects.")
UNDER_INFERENCE = ("Under inference the effect is the declared estimate in the coefficient table. "
                   "These explanations describe each model and are never offered as effect "
                   "estimates.")
SINGLE_FILL = ("They describe each model as fitted on a single in-fold fill of the missing values, "
               "not the analysis pooled over the multiple imputations, and estimate nothing.")
ADJUSTMENT_TERM = "adjustment term, not an effect estimate"
# MODELING_SEQUENCE §0 ruling 3 and §4: a lever pulled by hand from what the analyst saw sits
# outside the resampling. Choosing predictors from these explanations is such a lever.
SELECTION_NOTE = ("Choosing predictors from these explanations and refitting is a choice made "
                  "outside the cross-validation: the score would not be corrected for it.")

# ── sizes ────────────────────────────────────────────────────────────────────

RESEEDS = 5
EXPLAIN_ROWS = 1_000  # rows whose predictions are explained (a seeded sample above this)
BEESWARM_ROWS = 300  # points per input in the beeswarm
OBSERVATION_ROWS = 200  # rows whose full attributions are listed
SHOWN_INPUTS = 20  # inputs listed by importance; the rest are summed into one column
STABILITY_INPUTS = 200  # inputs whose refit importances are listed (ρ is over every input)
TOP_EXPOSURES = 3
H_ROWS = 150  # rows the partial dependence of the H statistic is evaluated at (n² predictions)
H_INPUTS = 6  # the top inputs whose pairs are ranked
TOP_PAIRS = 3  # a pair "stays in the top" across reseeds when it ranks in the first TOP_PAIRS
ADDITIVE = 1e-9  # every pair's interaction below this share of the predictions' SD: additive
ALE_INTERVALS = 20
MIN_SEGMENT_ROWS = 5
EQUATION_TERMS = 12
PATH_POINTS = 40
NARROW_PATH = 200  # columns up to which the shrinkage path runs past the chosen penalty
TREE_LEVELS = 3
CELLS = 2_000_000  # explained rows × inputs held at once (a 20,000-gene table explains 100 rows)

# ═════════════════════════════════════════════════════════════════════════════
# SHAP
# ═════════════════════════════════════════════════════════════════════════════


def linear_shap(coef: Any, Z: Any, mean: Any) -> np.ndarray:
    """Exact SHAP values of a linear model under independent features (Lundberg & Lee 2017):
    ``φ_ij = β_j (z_ij − E[z_j])``.

    ``coef``: (p,) or (K, p) for K outputs; ``Z``: (n, p); ``mean``: (p,). Returns (n, p, K)."""
    beta = np.atleast_2d(np.asarray(coef, dtype=float))
    Z = np.asarray(Z, dtype=float)
    centered = Z - np.asarray(mean, dtype=float)[None, :]
    return centered[:, :, None] * beta.T[None, :, :]


@dataclass(frozen=True)
class LeafPaths:
    """Every leaf of one tree as a path: for leaf ``l`` and its ``k`` distinct split features,
    ``features[l]`` (k,), the cover fractions ``zero[l]`` (k,), and the interval ``(lower, upper]``
    a row's value must fall in to follow every split on that feature (``missing[l]``: whether a
    blank follows them all)."""

    values: np.ndarray  # (L,)
    features: list[np.ndarray]
    zero: list[np.ndarray]
    lower: list[np.ndarray]
    upper: list[np.ndarray]
    missing: list[np.ndarray]


@dataclass(frozen=True)
class TreeEnsemble:
    """An additive ensemble of trees: ``raw(x) = base[k] + Σ_t tree_t,k(x)`` for output ``k``."""

    base: np.ndarray  # (K,)
    trees: list[list[LeafPaths]]  # [output][tree]


def leaf_paths(nodes: np.ndarray) -> LeafPaths:
    """The leaf paths of a scikit-learn histogram gradient-boosting tree (``TreePredictor.nodes``).

    A row goes left when its value is ``<= num_threshold``, and a blank goes where
    ``missing_go_to_left`` says (scikit-learn's ``_predictor.pyx``). A feature split twice on one
    path keeps the product of its cover fractions and the intersection of its intervals, as
    TreeSHAP's unwinding of a repeated feature does."""
    values, features, zero, lower, upper, missing = [], [], [], [], [], []
    stack: list[tuple[int, dict[int, tuple[float, float, float, bool]]]] = [(0, {})]
    while stack:
        i, conditions = stack.pop()
        node = nodes[i]
        if bool(node["is_leaf"]):
            keys = sorted(conditions)
            values.append(float(node["value"]))
            features.append(np.asarray(keys, dtype=np.int64))
            zero.append(np.asarray([conditions[f][0] for f in keys], dtype=float))
            lower.append(np.asarray([conditions[f][1] for f in keys], dtype=float))
            upper.append(np.asarray([conditions[f][2] for f in keys], dtype=float))
            missing.append(np.asarray([conditions[f][3] for f in keys], dtype=bool))
            continue
        if "is_categorical" in nodes.dtype.names and bool(node["is_categorical"]):
            raise ValueError("a categorical split: the pipeline one-hot encodes categories first")
        feature = int(node["feature_idx"])
        threshold = float(node["num_threshold"])
        to_left = bool(node["missing_go_to_left"])
        parent = float(node["count"])
        for child, left in ((int(node["left"]), True), (int(node["right"]), False)):
            fraction = float(nodes[child]["count"]) / parent
            z, lo, hi, blank = conditions.get(feature, (1.0, -math.inf, math.inf, True))
            if left:
                hi, blank = min(hi, threshold), blank and to_left
            else:
                lo, blank = max(lo, threshold), blank and not to_left
            nxt = dict(conditions)
            nxt[feature] = (z * fraction, lo, hi, blank)
            stack.append((child, nxt))
    return LeafPaths(np.asarray(values, dtype=float), features, zero, lower, upper, missing)


def hgb_ensemble(model: Any) -> TreeEnsemble:
    """A fitted ``HistGradientBoostingRegressor`` / ``…Classifier`` as leaf paths, on its raw
    (margin) scale: the outcome's for a regression, the log-odds for a binary classifier."""
    base = np.atleast_1d(np.asarray(model._baseline_prediction, dtype=float).ravel())
    outputs = len(model._predictors[0])
    trees: list[list[LeafPaths]] = [[] for _ in range(outputs)]
    for iteration in model._predictors:
        for k, predictor in enumerate(iteration):
            trees[k].append(leaf_paths(predictor.nodes))
    return TreeEnsemble(base=base, trees=trees)


def _shapley_weights(k: int) -> np.ndarray:
    return np.asarray([math.factorial(s) * math.factorial(k - 1 - s) / math.factorial(k)
                       for s in range(k)], dtype=float)


def _path_shapley(one: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """Shapley values of the product games of leaves that share a path length ``k``.

    ``one``: (n, L, k) 0/1; ``zero``: (L, k). Returns (n, L, k): ``(o_j − z_j) · Σ_s w(s) e_s^(−j)``
    with ``e^(−j)`` the coefficients of ``Π_{i≠j} (z_i + o_i t)``."""
    n, L, k = one.shape
    weights = _shapley_weights(k)
    out = np.empty((n, L, k))
    for j in range(k):
        poly = np.zeros((n, L, k))
        poly[:, :, 0] = 1.0
        degree = 0
        for i in range(k):
            if i == j:
                continue
            nxt = poly * zero[None, :, i, None]
            nxt[:, :, 1:degree + 2] += poly[:, :, :degree + 1] * one[:, :, i, None]
            poly, degree = nxt, degree + 1
        out[:, :, j] = (one[:, :, j] - zero[None, :, j]) * (poly @ weights)
    return out


def tree_shap(ensemble: TreeEnsemble, Z: Any, *, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Path-dependent TreeSHAP values of every row of ``Z`` (n, p): ``(phi (n, p, K), expected
    (K,))``, with ``expected + phi.sum(axis=1) == raw prediction`` for every row."""
    X = np.asarray(Z, dtype=float)
    n, p = X.shape
    blank = np.isnan(X)
    K = len(ensemble.trees)
    phi = np.zeros((n, p, K))
    expected = ensemble.base.astype(float).copy()
    for k, trees in enumerate(ensemble.trees):
        by_length: dict[int, list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray]]] = {}
        for tree in trees:
            for leaf in range(len(tree.values)):
                feats = tree.features[leaf]
                zero = tree.zero[leaf]
                value = float(tree.values[leaf])
                expected[k] += value * float(np.prod(zero))
                if feats.size:
                    by_length.setdefault(int(feats.size), []).append(
                        (value, feats, zero, tree.lower[leaf], tree.upper[leaf],
                         tree.missing[leaf]))
        flat = phi[:, :, k]
        for leaves in by_length.values():
            for start in range(0, len(leaves), chunk):
                part = leaves[start:start + chunk]
                values = np.asarray([leaf[0] for leaf in part])
                feats = np.stack([leaf[1] for leaf in part])  # (L, k)
                zero = np.stack([leaf[2] for leaf in part])
                lower = np.stack([leaf[3] for leaf in part])
                upper = np.stack([leaf[4] for leaf in part])
                missing = np.stack([leaf[5] for leaf in part])
                x = X[:, feats]  # (n, L, k)
                nan = blank[:, feats]
                with np.errstate(invalid="ignore"):
                    inside = (x > lower[None]) & (x <= upper[None])
                one = np.where(nan, missing[None], inside).astype(float)
                contrib = _path_shapley(one, zero) * values[None, :, None]
                np.add.at(flat.T, feats.ravel(), contrib.reshape(n, -1).T)
    return phi, expected


# ═════════════════════════════════════════════════════════════════════════════
# the pipeline's anatomy: inputs (the adjusted lane), the model matrix, the model's scale
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Anatomy:
    """A fitted pipeline split where the lineage's adjusted lane ends.

    ``adjust`` maps the raw columns to the model's inputs (``A``); ``rest`` maps the inputs through
    the form, levels, one-hot and scale steps to the model (``rest[:-1]`` gives the matrix ``Z``).
    ``sources[a]``: the raw columns input ``a`` was made from. ``group[z]``: the input each matrix
    column belongs to."""

    adjust: Any
    rest: Any
    model: Any
    sources: dict[str, list[str]]
    group: dict[str, str]
    task: str
    # The inputs the adjusted lane passed on as they were (kept, or only imputed): the raw
    # column's unit is theirs. An energy-adjusted, normalized or derived input has no stated unit.
    as_raw: frozenset[str] = frozenset()

    def inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.adjust is None:
            return X
        out = self.adjust.transform(X)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out, index=X.index)

    def matrix(self, A: pd.DataFrame) -> pd.DataFrame:
        if len(self.rest.steps) == 1:
            return A
        out = self.rest[:-1].transform(A)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, index=A.index, columns=self.rest[:-1].get_feature_names_out())
        return out

    def raw_score(self, A: pd.DataFrame) -> np.ndarray:
        """The model's own scale: the prediction for a numeric outcome, the log-odds of the class
        coded 1 for a yes/no one."""
        if self.task == "binary":
            return np.asarray(self.rest.decision_function(A), dtype=float).ravel()
        return np.asarray(self.rest.predict(A), dtype=float).ravel()


UNCHANGED_OPERATIONS = ("kept", "imputed")


def _trace(steps: Sequence[tuple[str, Any]], columns: Sequence[str]
           ) -> tuple[dict[str, list[str]], set[str]]:
    """Each output of fitted ``steps``: the columns of ``columns`` it was made from (the lineage's
    own tracer, step by step, without its wide-table collapse), and the outputs that are one of
    them under its own name, only kept or imputed on the way."""
    from turbotab.core.models.lineage import trace_step

    origin = {str(c): [str(c)] for c in columns}
    unchanged = set(origin)
    for _, step in steps:
        mapping = trace_step(step, list(origin))
        unchanged = {out for out, (parents, op, _f) in mapping.items()
                     if parents == [out] and out in unchanged and op in UNCHANGED_OPERATIONS}
        origin = {out: list(dict.fromkeys(s for p in parents for s in origin.get(p, [p])))
                  for out, (parents, _op, _formula) in mapping.items()}
    return origin, unchanged


def anatomy(pipeline: Any, raw_columns: Sequence[str], task: str) -> Anatomy:
    """The fitted ``pipeline`` split at the end of its adjusted lane."""
    from turbotab.core.models.pipeline import ADJUST_STEPS

    steps = list(pipeline.steps)
    k = 0
    while k < len(steps) - 1 and steps[k][0] in ADJUST_STEPS:
        k += 1
    raw = [str(c) for c in raw_columns]
    sources, unchanged = _trace(steps[:k], raw) if k else ({c: [c] for c in raw}, set(raw))
    later = _trace(steps[k:-1], list(sources))[0] if len(steps) - k > 1 else {a: [a] for a in sources}
    group = {z: (parents[0] if len(parents) == 1 else z) for z, parents in later.items()}
    return Anatomy(adjust=pipeline[:k] if k else None, rest=pipeline[k:], model=steps[-1][1],
                   sources=sources, group=group, task=task, as_raw=frozenset(unchanged))


LINEAR_MODELS = ("LinearRegression", "LogisticRegression", "ElasticNetCV", "Float32ElasticNetCV",
                 "LogisticRegressionCV")


def model_kind(model: Any) -> str | None:
    """How SHAP is computed for this model step: ``linear``, ``trees``, or None (not built)."""
    name = type(model).__name__
    if name.startswith("HistGradientBoosting"):
        return "trees"
    if name in LINEAR_MODELS and hasattr(model, "coef_"):
        return "linear"
    return None


def attributions(anat: Anatomy, A: pd.DataFrame, background: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, float] | None:
    """SHAP values of every row of ``A`` (the model's inputs), one column per model-matrix column,
    on the model's own scale, and the expected value they are measured from; None when the family
    has no SHAP computation here or several outputs.

    ``background``: the inputs of the rows the model was fit on; its mean matrix row is ``E[z]``
    in the linear closed form (TreeSHAP's covers are the trees' own training counts)."""
    kind = model_kind(anat.model)
    if kind is None:
        return None
    Z = anat.matrix(A)
    names = [str(c) for c in Z.columns]
    if kind == "trees":
        phi, expected = tree_shap(hgb_ensemble(anat.model), Z.to_numpy(dtype=float))
        if phi.shape[2] != 1:
            return None
        return pd.DataFrame(phi[:, :, 0], index=A.index, columns=names), float(expected[0])
    coef = np.atleast_2d(np.asarray(anat.model.coef_, dtype=float))
    if coef.shape[0] != 1:
        return None
    Zb = anat.matrix(background) if background is not None else Z
    mean = Zb.to_numpy(dtype=float).mean(axis=0)
    phi = linear_shap(coef, Z.to_numpy(dtype=float), mean)[:, :, 0]
    intercept = float(np.atleast_1d(np.asarray(anat.model.intercept_, dtype=float))[0])
    return pd.DataFrame(phi, index=A.index, columns=names), intercept + float(coef[0] @ mean)


def grouped(phi: pd.DataFrame, group: Mapping[str, str]) -> pd.DataFrame:
    """SHAP values of the model-matrix columns summed over the input each was made from (a
    category's indicators, a spline's basis)."""
    names = [group.get(str(c), str(c)) for c in phi.columns]
    order = list(dict.fromkeys(names))
    values = phi.to_numpy(dtype=float)
    out = np.zeros((len(phi), len(order)))
    where = {name: i for i, name in enumerate(order)}
    for j, name in enumerate(names):
        out[:, where[name]] += values[:, j]
    return pd.DataFrame(out, index=phi.index, columns=order)


def importance(phi: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| of each input over the explained rows."""
    return phi.abs().mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# stability across reseeds
# ═════════════════════════════════════════════════════════════════════════════


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rank correlation, ties at their average rank; None when either side is constant
    (there is no ranking to compare)."""
    from scipy.stats import rankdata

    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pairwise_spearman(rows: Sequence[Sequence[float]]) -> list[float | None]:
    """Spearman's ρ between every pair of the given vectors, in (i, j) order with i < j."""
    return [spearman(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]


def resample(n: int, seed: int, units: Any = None) -> np.ndarray:
    """A bootstrap resample's row positions: ``n`` rows drawn with replacement, or, when ``units``
    labels the rows, as many whole units drawn with replacement, every row of a drawn unit taken
    each time it is drawn (MODELING_SEQUENCE §2: a bootstrap by unit)."""
    rng = np.random.default_rng(seed)
    if units is None:
        return rng.integers(0, n, n)
    codes, labels = pd.factorize(pd.Series(np.asarray(units, dtype=object)), use_na_sentinel=False)
    rows_of = [np.flatnonzero(codes == u) for u in range(len(labels))]
    drawn = rng.integers(0, len(labels), len(labels))
    return np.concatenate([rows_of[u] for u in drawn])


def reseeded(model: Any, seed: int) -> Any:
    """``model`` (an unfitted pipeline) with every integer ``random_state`` it declares set to
    ``seed``."""
    params = {k: seed for k, v in model.get_params(deep=True).items()
              if k.endswith("random_state") and (v is None or isinstance(v, (int, np.integer)))}
    return model.set_params(**params) if params else model


def ranks(values: Sequence[float]) -> list[int]:
    """1 for the largest; ties keep their order of appearance."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    out = np.empty(len(order), dtype=int)
    out[order] = np.arange(1, len(order) + 1)
    return out.tolist()


# ═════════════════════════════════════════════════════════════════════════════
# partial dependence and Friedman's H statistic
# ═════════════════════════════════════════════════════════════════════════════

Score = Callable[[pd.DataFrame], np.ndarray]


def partial_dependence_at_rows(score: Score, A: pd.DataFrame, columns: Sequence[str], *,
                               batch: int = 60_000) -> np.ndarray:
    """``F_S(x_iS) = (1/m) Σ_l f(x_iS, x_l,\\S)`` at each of the ``m`` rows of ``A``: the model with
    ``columns`` set to row ``i``'s values, averaged over every row's other values (Friedman 2001)."""
    base = A.reset_index(drop=True)
    m = len(base)
    out = np.empty(m)
    per = max(1, batch // max(m, 1))
    for start in range(0, m, per):
        rows = np.arange(start, min(m, start + per))
        block = base.iloc[np.tile(np.arange(m), len(rows))].reset_index(drop=True)
        for c in columns:
            block[c] = np.repeat(base[c].to_numpy()[rows], m)
        out[rows] = np.asarray(score(block), dtype=float).reshape(len(rows), m).mean(axis=1)
    return out


def centered(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


@dataclass(frozen=True)
class HStat:
    a: str
    b: str
    h2: float | None  # Friedman & Popescu's H²; None when the pair's joint effect is nil
    strength: float  # the root mean square of F_ab − F_a − F_b, on the model's scale


def h_statistics(score: Score, A: pd.DataFrame, inputs: Sequence[str]) -> list[HStat]:
    """H² and the interaction's size for every pair of ``inputs``, at the rows of ``A``."""
    single = {c: centered(partial_dependence_at_rows(score, A, [c])) for c in inputs}
    out = []
    for i, a in enumerate(inputs):
        for b in inputs[i + 1:]:
            joint = centered(partial_dependence_at_rows(score, A, [a, b]))
            part = joint - single[a] - single[b]
            denominator = float(np.sum(joint ** 2))
            nil = denominator <= 1e-24 * max(1.0, float(np.sum(single[a] ** 2 + single[b] ** 2)))
            out.append(HStat(a, b, None if nil else float(np.sum(part ** 2) / denominator),
                             float(np.sqrt(np.mean(part ** 2)))))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# accumulated local effects, on a shared grid with support masks
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Grid:
    """``edges`` z_0 < … < z_K (observed values); ``counts[k]``: rows in segment k + 1,
    ``(z_k, z_{k+1}]`` (the first also takes z_0); ``supported[k]``: at least
    :data:`MIN_SEGMENT_ROWS` of them."""

    edges: np.ndarray
    counts: np.ndarray
    supported: np.ndarray

    def segment_of(self, x: np.ndarray) -> np.ndarray:
        """Each value's segment, 1-based (z_0 belongs to segment 1)."""
        return np.clip(np.searchsorted(self.edges, x, side="left"), 1, len(self.edges) - 1)


def ale_grid(values: Any, intervals: int = ALE_INTERVALS, min_rows: int = MIN_SEGMENT_ROWS) -> Grid:
    """The input's quantiles at ``k/intervals`` taken as observed values (the inverse of the
    empirical distribution), or every distinct value when there are no more than
    ``intervals + 1``; then both ends of every gap between neighboring observed values wider than
    one ``intervals``-th of the range, so that no segment spans a stretch without data."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    distinct = np.unique(x)
    if len(distinct) < 2:
        return Grid(distinct, np.zeros(0, dtype=int), np.zeros(0, dtype=bool))
    if len(distinct) <= intervals + 1:
        edges = distinct
    else:
        q = np.quantile(x, np.linspace(0.0, 1.0, intervals + 1), method="inverted_cdf")
        width = (distinct[-1] - distinct[0]) / intervals
        gaps = np.flatnonzero(np.diff(distinct) > width)
        edges = np.unique(np.concatenate([q, distinct[gaps], distinct[gaps + 1]]))
    bare = Grid(edges, np.zeros(len(edges) - 1, dtype=int), np.zeros(len(edges) - 1, dtype=bool))
    counts = np.bincount(bare.segment_of(x) - 1, minlength=len(edges) - 1)
    return Grid(edges, counts, counts >= min_rows)


def _with(A: pd.DataFrame, column: str, values: np.ndarray) -> pd.DataFrame:
    out = A.copy()
    out[column] = values
    return out


def ale_curve(score: Score, A: pd.DataFrame, column: str, grid: Grid) -> np.ndarray:
    """The centered first-order ALE at the grid's edges (Apley & Zhu 2020): the local effects
    ``mean_{i in segment k} [f(z_k, x_i,\\j) − f(z_{k−1}, x_i,\\j)]`` accumulated from z_0, less the
    mean over the rows of each row's accumulated value at its own segment."""
    x = A[column].to_numpy(dtype=float)
    keep = np.isfinite(x)
    rows = A.loc[keep]
    seg = grid.segment_of(x[keep])
    upper = score(_with(rows, column, grid.edges[seg]))
    lower = score(_with(rows, column, grid.edges[seg - 1]))
    K = len(grid.edges) - 1
    total = np.bincount(seg - 1, weights=np.asarray(upper) - np.asarray(lower), minlength=K)
    counts = np.bincount(seg - 1, minlength=K)
    local = np.divide(total, counts, out=np.zeros(K), where=counts > 0)
    g = np.concatenate([[0.0], np.cumsum(local)])
    return g - g[seg].mean()


def pd_curve(score: Score, A: pd.DataFrame, column: str, grid: Grid) -> np.ndarray:
    """Partial dependence at the grid's edges (Friedman 2001): the mean prediction with the input
    set to each edge for every row, centered as the ALE is (less its mean over the rows at each
    row's own segment), so both draw on one axis."""
    x = A[column].to_numpy(dtype=float)
    keep = np.isfinite(x)
    rows = A.loc[keep]
    values = np.array([float(np.mean(score(_with(rows, column, np.full(len(rows), z)))))
                       for z in grid.edges])
    return values - values[grid.segment_of(x[keep])].mean()


def masked(values: np.ndarray, grid: Grid) -> list[float | None]:
    """The curve at the grid's edges; None at an edge that touches no supported segment."""
    K = len(grid.edges) - 1
    return [float(v) if ((i > 0 and grid.supported[i - 1]) or (i < K and grid.supported[i]))
            else None for i, v in enumerate(values)]


# ═════════════════════════════════════════════════════════════════════════════
# the artifact
# ═════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

Role = str  # "exposure" · "adjustment" · "predictor"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class InputImportance(_Model):
    """One input's mean |SHAP|, its rank, and its rank in each reseeded refit."""

    input: str
    mean_abs: float
    rank: int
    role: Role
    reseed_ranks: list[int] = Field(default_factory=list)
    in_top: int | None = None  # refits that keep it among the TOP_INPUTS most important


class BeeswarmInput(_Model):
    """One input's points: its SHAP value, its value (None for a category or a blank), and its
    value's percentile among the explained rows for the color (a category: its level's place)."""

    input: str
    role: Role
    phi: list[float]
    value: list[float | None]
    level: list[str | None] | None = None
    color: list[float | None]


class Observations(_Model):
    """Per-row attributions: ``base + Σ phi + rest == prediction`` on the model's scale."""

    row_ids: list[int]
    base: float
    prediction: list[float]
    inputs: list[str]
    phi: list[list[float]]  # rows × inputs
    rest: list[float]  # the inputs not listed, summed


class Stability(_Model):
    """Mean |SHAP| over the same rows, refit on bootstrap resamples each with its own seed."""

    reseeds: int
    resampled_by: str  # "row", or the unit column in backticks
    inputs: list[str]
    importance: list[list[float]]  # per refit, aligned with ``inputs``
    pairwise: list[float | None]  # Spearman ρ between refits i < j
    versus_fit: list[float | None]  # each refit against the reported fit
    rho_mean: float | None
    rho_min: float | None
    top: int  # the "top" whose membership is counted (``InputImportance.in_top``)


class Interaction(_Model):
    a: str
    b: str
    rank: int
    strength: float  # root mean square of F_ab − F_a − F_b, on the model's scale
    h2: float | None
    reseed_strength: list[float] = Field(default_factory=list)
    reseed_h2: list[float | None] = Field(default_factory=list)
    in_top: int | None = None  # refits that rank it among the first TOP_PAIRS


class Interactions(_Model):
    inputs: list[str]
    rows: int
    pairs: list[Interaction]
    rho_mean: float | None  # Spearman ρ of the pairs' strengths, each refit against the fit
    additive: bool  # every pair's interaction is nil: the model adds its inputs' effects
    method: str


class Floor(_Model):
    """The held-out performance floor: the family's cross-validated score against the
    no-predictor baseline's on the same folds (``models/baseline.py``)."""

    passed: bool
    verdict: str
    metric: str
    model: float | None
    baseline: float | None
    reason: str | None


class Curve(_Model):
    family: str
    label: str
    drawn: bool
    values: list[float | None]  # at the grid's edges; None where the data are absent
    reason: str | None = None


class InputCurves(_Model):
    """One input's curve for every family, on one grid (shared axes)."""

    input: str
    sources: list[str]
    role: Role
    method: str  # "ale" · "partial_dependence"
    grid: list[float]
    counts: list[int]  # rows per segment (z_k, z_k+1]
    supported: list[bool]  # segments with at least MIN_SEGMENT_ROWS rows
    unit: str | None
    curves: list[Curve]


class Term(_Model):
    column: str  # the model-matrix column
    input: str
    kind: str  # "slope" · "indicator" · "basis" · "other"
    coefficient: float  # per unit of the column as the model reads it, scaling undone
    unit: str | None  # the column's settled unit (a slope of an input kept as recorded)
    coefficient_unit: str | None
    level: str | None = None
    role: Role = "predictor"


class Equation(_Model):
    outcome: str
    scale: str
    outcome_unit: str | None
    intercept: float
    terms: list[Term]  # the largest first (|coefficient| × the column's SD), up to EQUATION_TERMS
    n_terms: int  # the model's columns
    zeros: int = 0  # columns whose coefficient is zero (shrunk to zero by a penalty)
    text: str


class TreeNode(_Model):
    id: int
    depth: int
    column: str | None
    input: str | None
    threshold: float | None
    blanks: str | None  # "left" · "right"
    n: int
    value: float | None  # a leaf's value; None for a split
    left: int | None
    right: int | None


class SplitCount(_Model):
    input: str
    root: int  # trees whose first split is on it
    splits: int  # its splits in the top TREE_LEVELS levels of every tree
    median_threshold: float | None  # of its root splits


class TreeStructure(_Model):
    n_trees: int
    levels: int
    first_tree: list[TreeNode]
    splits: list[SplitCount]


class PathLine(_Model):
    column: str
    input: str
    coefficients: list[float]


class ShrinkagePath(_Model):
    penalty_name: str  # "alpha" (least squares) · "C" (logistic, an inverse strength)
    penalties: list[float]  # from the strongest penalty to the weakest
    chosen: float
    chosen_index: int
    l1_ratio: float
    scale: str
    lines: list[PathLine]
    nonzero: list[int]
    n_columns: int


class Architecture(_Model):
    kind: str  # "equation" · "trees" · "shrinkage"
    equation: Equation | None = None
    trees: TreeStructure | None = None
    path: ShrinkagePath | None = None


class FamilyExplanation(_Model):
    family: str
    label: str
    explained: bool
    reason: str | None = None
    method: str | None = None  # how its SHAP values were computed
    scale: str | None = None
    base: float | None = None
    floor: Floor | None = None
    importance: list[InputImportance] = Field(default_factory=list)
    beeswarm: list[BeeswarmInput] = Field(default_factory=list)
    observations: Observations | None = None
    stability: Stability | None = None
    interactions: Interactions | None = None
    architecture: Architecture | None = None


class ExplainArtifact(_Model):
    purpose: str | None
    task: str
    describes: str
    under_inference: str | None = None
    single_fill: str | None = None
    rows: int  # rows explained
    rows_of: int  # rows the models were fit on
    rows_basis: str
    row_ids: list[int] = Field(default_factory=list)  # the rows explained, in the order used
    curve_method: str
    families: list[FamilyExplanation]
    curves: list[InputCurves]
    methods: str
    adjustment_terms: str | None = None  # under inference: what a covariate's attribution is
    relations: list[str] = Field(default_factory=list)  # the contract's relations that fired here
    notes: list[str] = Field(default_factory=list)
    withheld: str | None = None  # set by the server while a question an estimate rests on is open


# ═════════════════════════════════════════════════════════════════════════════
# the architecture lane
# ═════════════════════════════════════════════════════════════════════════════


def _num(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    text = f"{abs(value):.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa} × 10^{int(exponent)}"
    return text


def equation_units(state: Any, target: str | None, columns: Sequence[str]
                   ) -> tuple[str | None, dict[str, str]]:
    """The units an equation may state (BLUEPRINT §14.3: a sentence states a unit only once it is
    settled; until then it quotes the header verbatim): the outcome's recorded unit, and each
    column's unit as the user recorded or confirmed it."""
    from turbotab.core import readings
    from turbotab.core.units import outcome_unit, recorded_unit

    out = outcome_unit(target, recorded=recorded_unit(state, target))[0] if target else None
    units: dict[str, str] = {}
    for column in columns:
        value = readings.confirmation(state, "unit", column)
        if value is not None:
            units[str(column)] = readings.unit_words(value)
    return out, units


_ONE = {"years": "year", "months": "month", "weeks": "week", "days": "day",
        "% of energy": "percentage point of energy"}


def _one_unit(words: str) -> str:
    """One of a unit, for "per": ``years`` → ``year``, a standard drink for drinks."""
    if words.startswith("standard drinks "):
        return "standard drink " + words[len("standard drinks "):]
    return _ONE.get(words, words)


def _undo_scaling(anat: Anatomy, coef: np.ndarray, intercept: float) -> tuple[np.ndarray, float]:
    """Coefficients per unit of each matrix column as it entered the scaler (the elastic net's
    own ``coefficients`` does the same)."""
    scaler = dict(anat.rest.steps).get("scale")
    if scaler is None:
        return coef, intercept
    per_unit = coef / scaler.scale_
    return per_unit, intercept - float(per_unit @ scaler.mean_)


def linear_equation(anat: Anatomy, A_fit: pd.DataFrame, *, target: str, scale: str,
                    outcome_unit: str | None, units: Mapping[str, str],
                    roles: Mapping[str, Role]) -> Equation:
    """The fitted equation of a linear model, on the columns the model reads (its scaling undone),
    with units where they are settled; the largest terms first."""
    model = anat.model
    coef = np.atleast_2d(np.asarray(model.coef_, dtype=float))[0]
    intercept = float(np.atleast_1d(np.asarray(model.intercept_, dtype=float))[0])
    coef, intercept = _undo_scaling(anat, coef, intercept)
    Z = anat.rest[:-1].transform(A_fit) if len(anat.rest.steps) > 1 else A_fit
    scaler = dict(anat.rest.steps).get("scale")
    columns = [str(c) for c in (scaler.feature_names_in_ if scaler is not None else Z.columns)]
    raw = Z.to_numpy(dtype=float)
    if scaler is not None:
        raw = raw * scaler.scale_ + scaler.mean_
    sd = raw.std(axis=0)
    order = [j for j in np.argsort(-np.abs(coef * sd), kind="stable") if coef[j] != 0]
    zeros = int(np.sum(coef == 0))
    terms: list[Term] = []
    for j in order[:EQUATION_TERMS]:
        column = columns[j]
        source = anat.group.get(column, column)
        numeric = source in A_fit.columns and pd.api.types.is_numeric_dtype(A_fit[source])
        if column == source:
            kind = "slope" if numeric else "other"
        elif column.startswith(f"{source}_") and not numeric:
            kind = "indicator"
        else:
            kind = "basis"
        unit = units.get(column) if kind == "slope" and column in anat.as_raw else None
        per = None
        if unit:
            head = "log-odds" if scale.startswith("log-odds") else outcome_unit
            one = _one_unit(unit)
            per = f"{head} per {one}" if head else f"per {one}"
        terms.append(Term(column=column, input=source, kind=kind, coefficient=float(coef[j]),
                          unit=unit, coefficient_unit=per,
                          level=column[len(source) + 1:] if kind == "indicator" else None,
                          role=roles.get(source, "predictor")))
    left = f"{scale}" if scale.startswith("log-odds") else (
        f"`{target}`" + (f" ({outcome_unit})" if outcome_unit else ""))
    parts = [f"{left} = {'−' if intercept < 0 else ''}{_num(intercept)}"]
    for t in terms:
        if t.kind == "indicator":
            name = f"[`{t.input}` = `{t.level}`]"
        else:
            name = f"`{t.column}`" + (f" ({t.unit})" if t.unit else "")
        parts.append(f"{'−' if t.coefficient < 0 else '+'} {_num(t.coefficient)} × {name}")
    more = len(order) - len(terms)
    if more > 0:
        parts.append(f"+ {more:,} more term{'s' if more != 1 else ''}")
    text = " ".join(parts)
    if zeros:
        text += (f"; {zeros:,} coefficient{'s' if zeros != 1 else ''} shrunk to zero"
                 if hasattr(model, "alpha_") or hasattr(model, "C_") else
                 f"; {zeros:,} coefficient{'s' if zeros != 1 else ''} of zero")
    return Equation(outcome=target, scale=scale, outcome_unit=outcome_unit, intercept=intercept,
                    terms=terms, n_terms=len(columns), zeros=zeros, text=text)


def tree_structure(anat: Anatomy, columns: Sequence[str], levels: int = TREE_LEVELS
                   ) -> TreeStructure:
    """The first tree's top ``levels`` levels, and which inputs the trees split on near the root."""
    from collections import defaultdict

    model = anat.model
    trees = [p[0].nodes for p in model._predictors]
    first = trees[0]
    shown: list[TreeNode] = []
    queue = [0]
    while queue:
        i = queue.pop(0)
        node = first[i]
        depth = int(node["depth"])
        leaf = bool(node["is_leaf"])
        expand = not leaf and depth + 1 < levels  # its children are within the levels shown
        column = None if leaf else str(columns[int(node["feature_idx"])])
        shown.append(TreeNode(
            id=i, depth=depth, column=column,
            input=None if column is None else anat.group.get(column, column),
            threshold=None if leaf else float(node["num_threshold"]),
            blanks=None if leaf else ("left" if bool(node["missing_go_to_left"]) else "right"),
            n=int(node["count"]), value=float(node["value"]) if leaf else None,
            left=int(node["left"]) if expand else None,
            right=int(node["right"]) if expand else None))
        if expand:
            queue.extend([int(node["left"]), int(node["right"])])
    root: dict[str, list[float]] = defaultdict(list)
    splits: dict[str, int] = defaultdict(int)
    for nodes in trees:
        for node in nodes:
            if bool(node["is_leaf"]) or int(node["depth"]) >= levels:
                continue
            name = anat.group.get(str(columns[int(node["feature_idx"])]),
                                  str(columns[int(node["feature_idx"])]))
            splits[name] += 1
            if int(node["depth"]) == 0:
                root[name].append(float(node["num_threshold"]))
    counts = [SplitCount(input=name, root=len(root.get(name, [])), splits=n,
                         median_threshold=float(np.median(root[name])) if root.get(name) else None)
              for name, n in splits.items()]
    counts.sort(key=lambda s: (-s.root, -s.splits, s.input))
    return TreeStructure(n_trees=len(trees), levels=levels, first_tree=shown, splits=counts)


def shrinkage_path(anat: Anatomy, A_fit: pd.DataFrame, y: Any, *, lines: int = EQUATION_TERMS,
                   points: int = PATH_POINTS) -> ShrinkagePath | None:
    """The elastic net's coefficients along its penalty grid at the mixing it chose, on the
    standardized columns it was fit on, with the chosen penalty marked.

    Least squares: scikit-learn's ``enet_path`` over the chosen mix's own grid (``alphas_``), the
    rows centered as the fit centers them. Logistic: a refit at each ``C`` of its grid (``Cs_``)
    at the chosen mix. On a matrix of more than :data:`NARROW_PATH` columns the path runs from the
    strongest penalty down to the chosen one: the weaker end, which the fit did not choose, costs
    minutes at 20,000 columns (seconds down to the chosen penalty)."""
    model = anat.model
    Z = anat.matrix(A_fit)
    columns = [str(c) for c in Z.columns]
    X = Z.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    l1 = float(np.atleast_1d(model.l1_ratio_)[0])
    narrow = len(columns) <= NARROW_PATH
    if hasattr(model, "alpha_"):
        from sklearn.linear_model import enet_path

        grid = np.atleast_2d(np.asarray(model.alphas_, dtype=float))
        mixes = np.atleast_1d(np.asarray(model.l1_ratio, dtype=float))
        row = grid[int(np.argmin(np.abs(mixes - l1)))] if grid.shape[0] > 1 else grid[0]
        chosen = float(model.alpha_)
        row = row if narrow else row[row >= chosen]
        pick = np.unique(np.concatenate([row[np.linspace(0, len(row) - 1, min(points, len(row)))
                                             .round().astype(int)], [chosen]]))[::-1]
        _, coefs, _ = enet_path(X - X.mean(axis=0), y - y.mean(), l1_ratio=l1, alphas=pick,
                                tol=1e-12 if narrow else 1e-8, max_iter=200_000)
        name, penalties = "alpha", pick
    elif hasattr(model, "C_"):
        import warnings

        from sklearn.linear_model import LogisticRegression

        chosen = float(np.atleast_1d(model.C_)[0])
        grid = np.asarray(model.Cs_, dtype=float)
        grid = grid if narrow else grid[grid <= chosen]
        penalties = np.unique(np.concatenate([grid, [chosen]]))
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c in penalties:
                fit = LogisticRegression(C=float(c), l1_ratio=l1, solver="saga",
                                         tol=1e-8 if narrow else 1e-6,
                                         max_iter=20_000).fit(X, y)
                rows.append(np.ravel(fit.coef_))
        coefs = np.asarray(rows).T
        name = "C"
    else:
        return None
    index = int(np.argmin(np.abs(np.asarray(penalties) - chosen)))
    reach = np.max(np.abs(coefs), axis=1)
    order = np.argsort(-reach, kind="stable")[:lines]
    return ShrinkagePath(
        penalty_name=name, penalties=[float(p) for p in penalties], chosen=chosen,
        chosen_index=index, l1_ratio=l1, scale="per standard deviation of each column",
        lines=[PathLine(column=columns[j], input=anat.group.get(columns[j], columns[j]),
                        coefficients=[float(v) for v in coefs[j]]) for j in order],
        nonzero=[int(v) for v in (np.abs(coefs) > 0).sum(axis=0)], n_columns=len(columns))


# ═════════════════════════════════════════════════════════════════════════════
# explaining the fitted families
# ═════════════════════════════════════════════════════════════════════════════

STABLE_TOP = 5  # an input "stays in the top" across refits when it ranks in the first five
CURVE_WORDS = {"ale": "accumulated local effects", "partial_dependence": "partial dependence"}


@dataclass
class FamilyFit:
    """A family as the fit stage left it: its pipeline fit on the rows explained (the training
    rows; every analyzed row under inference), the design's unfitted pipeline for the refits, and
    its cross-validated primary score beside the no-predictor baseline's."""

    key: str
    label: str
    fitted: Any
    unfitted: Any
    versus: Mapping[str, Any] | None
    score: float | None
    baseline: float | None


@dataclass
class Setting:
    task: str
    purpose: str | None
    target: str
    event: str | None
    X: pd.DataFrame  # the raw inputs of the rows the models were fit on, indexed by row id
    y: np.ndarray  # their coded outcome
    state: Any = None
    units: np.ndarray | None = None  # each row's unit when a unit's rows repeat
    unit_name: str | None = None
    exposures: Sequence[str] = ()  # raw columns asked for; empty: the top ones
    declared: Sequence[str] = ()  # under inference: the declared exposure(s)
    candidates: Sequence[str] = ()  # raw columns that may be a top exposure
    curve_method: str = "ale"
    reseeds: int = RESEEDS
    seed: int = 0
    metric_label: str = "score"
    baseline_label: str = "the no-predictor baseline"
    higher_is_better: bool = True
    multiple_imputation: bool = False


Refit = Callable[[Any, pd.DataFrame, np.ndarray, Any], Any]
Progress = Callable[[float, str], None]


@dataclass
class _Work:
    fam: FamilyFit
    anat: Anatomy
    A_all: pd.DataFrame
    A: pd.DataFrame
    phi: pd.DataFrame  # grouped, the explained rows × inputs
    base: float
    floor: Floor
    refits: list[tuple[Anatomy, pd.DataFrame]]  # each refit and its inputs for the explained rows


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}".replace("-", "−")


def floor_of(fam: FamilyFit, s: Setting) -> Floor:
    """Whether the family's held-out (cross-validated) score beats the no-predictor baseline."""
    versus = dict(fam.versus or {})
    verdict = str(versus.get("verdict") or "unscored")
    passed = verdict == "better"
    how = {"worse": "is worse than", "no_better": "is not shown to beat"}.get(verdict)
    if passed:
        reason = None
    elif how is None:
        reason = (f"{fam.label} draws no curve: it has no cross-validated score to show that it "
                  f"beats {s.baseline_label}.")
    else:
        reason = (f"{fam.label} draws no curve: its cross-validated {s.metric_label} of "
                  f"{_fmt(fam.score)} {how} the {s.metric_label} of {s.baseline_label}, "
                  f"{_fmt(fam.baseline)}, so a curve would describe noise.")
    return Floor(passed=passed, verdict=verdict, metric=s.metric_label, model=fam.score,
                 baseline=fam.baseline, reason=reason)


def scale_of(s: Setting, unit: str | None) -> str:
    if s.task == "binary":
        return f"log-odds of `{s.event}`" if s.event is not None else "log-odds of the event"
    return f"predicted `{s.target}`" + (f" ({unit})" if unit else "")


def input_of(raw: str, sources: Mapping[str, Sequence[str]]) -> str | None:
    """The model input a raw column became: itself when kept, else the input made first from it
    (``protein_adj`` from ``protein`` and ``kcal``)."""
    hits = [a for a, src in sources.items() if raw in src]
    if raw in hits:
        return raw
    led = [a for a in hits if list(sources[a])[:1] == [raw]]
    return (led or hits or [None])[0]


def _roles(s: Setting, sources: Mapping[str, Sequence[str]]) -> dict[str, Role]:
    if s.purpose != "inference":
        return {a: "predictor" for a in sources}
    declared = set(s.declared)
    return {a: ("exposure" if declared & set(src) else "adjustment") for a, src in sources.items()}


def _colors(values: pd.Series) -> tuple[list[float | None], list[float | None], list[str | None] | None]:
    from scipy.stats import rankdata

    if pd.api.types.is_numeric_dtype(values):
        x = values.to_numpy(dtype=float)
        ok = np.isfinite(x)
        color: list[float | None] = [None] * len(x)
        if ok.sum() > 1:
            r = (rankdata(x[ok]) - 1) / (ok.sum() - 1)
            for i, v in zip(np.flatnonzero(ok), r):
                color[i] = float(v)
        return [float(v) if np.isfinite(v) else None for v in x], color, None
    levels = sorted({str(v) for v in values.dropna()})
    place = {lv: (i / (len(levels) - 1) if len(levels) > 1 else 0.5) for i, lv in enumerate(levels)}
    names = [None if pd.isna(v) else str(v) for v in values]
    return [None] * len(names), [None if v is None else place[v] for v in names], names


def _work(fam: FamilyFit, s: Setting, sample: np.ndarray) -> _Work | FamilyExplanation:
    anat = anatomy(fam.fitted, list(s.X.columns), s.task)
    if model_kind(anat.model) is None:
        return FamilyExplanation(family=fam.key, label=fam.label, explained=False,
                                 reason=f"{fam.label}: its SHAP values are not built here.")
    A_all = anat.inputs(s.X)
    A = A_all.iloc[sample]
    found = attributions(anat, A, A_all)
    if found is None:
        return FamilyExplanation(family=fam.key, label=fam.label, explained=False,
                                 reason=f"{fam.label}: explanations are built for one output "
                                        f"(a numeric or yes/no outcome).")
    phi, base = found
    return _Work(fam=fam, anat=anat, A_all=A_all, A=A, phi=grouped(phi, anat.group), base=base,
                 floor=floor_of(fam, s), refits=[])


def _refits(w: _Work, s: Setting, sample: np.ndarray, refit: Refit,
            say: Callable[[int], None] = lambda r: None) -> list[pd.Series]:
    """Mean |SHAP| of the explained rows under each reseeded refit, aligned with the fit's inputs."""
    from sklearn.base import clone

    out = []
    X_ex = s.X.iloc[sample]
    for r in range(1, s.reseeds + 1):
        say(r)
        idx = resample(len(s.X), s.seed + r, s.units)
        units = None if s.units is None else np.asarray(s.units)[idx]
        model = refit(reseeded(clone(w.fam.unfitted), r), s.X.iloc[idx], np.asarray(s.y)[idx], units)
        anat = anatomy(model, list(s.X.columns), s.task)
        A = anat.inputs(X_ex)
        found = attributions(anat, A, anat.inputs(s.X.iloc[idx]))
        if found is None:
            continue
        w.refits.append((anat, A))
        out.append(importance(grouped(found[0], anat.group)).reindex(w.phi.columns, fill_value=0.0))
    return out


def _stability(w: _Work, s: Setting, importances: list[pd.Series], main: pd.Series
               ) -> Stability | None:
    if not importances:
        return None
    rows = [imp.to_numpy(dtype=float) for imp in importances]
    pairwise = pairwise_spearman(rows)
    rho_mean, rho_min = _mean_min(pairwise)
    shown = list(main.index[:STABILITY_INPUTS])
    return Stability(
        reseeds=len(rows), resampled_by=f"`{s.unit_name}`" if s.units is not None and s.unit_name
        else "row", inputs=shown,
        importance=[[float(imp[c]) for c in shown] for imp in importances],
        pairwise=pairwise, versus_fit=[spearman(main.reindex(imp.index).to_numpy(), row)
                                       for imp, row in zip(importances, rows)],
        rho_mean=rho_mean, rho_min=rho_min, top=min(STABLE_TOP, len(main)))


def _mean_min(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    known = [v for v in values if v is not None]
    return (float(np.mean(known)), float(np.min(known))) if known else (None, None)


def _interactions(w: _Work, s: Setting, order: Sequence[str]) -> Interactions | None:
    inputs = list(order[:H_INPUTS])
    if len(inputs) < 2:
        return None
    rows = w.A.iloc[:H_ROWS]
    method = ("Friedman and Popescu's H statistic, the pairs ranked by the root mean square of "
              "their interaction part on the model's scale")
    stats = h_statistics(w.anat.raw_score, rows, inputs)
    strength = np.asarray([h.strength for h in stats])
    spread = float(np.std(w.anat.raw_score(rows))) or 1.0
    if strength.max() <= ADDITIVE * spread:
        # The model adds its inputs' effects (a linear model with no product terms): every pair's
        # interaction part is zero up to rounding, so there is no ranking to report.
        return Interactions(inputs=inputs, rows=len(rows), pairs=[], rho_mean=None, additive=True,
                            method=method)
    again = [h_statistics(anat.raw_score, A.iloc[:H_ROWS], inputs) for anat, A in w.refits]
    rank = ranks(strength)
    again_rank = [ranks([h.strength for h in hs]) for hs in again]
    rho = [spearman(strength, [h.strength for h in hs]) for hs in again]
    pairs = [Interaction(a=h.a, b=h.b, rank=rank[i], strength=h.strength, h2=h.h2,
                         reseed_strength=[hs[i].strength for hs in again],
                         reseed_h2=[hs[i].h2 for hs in again],
                         in_top=sum(r[i] <= TOP_PAIRS for r in again_rank) if again else None)
             for i, h in enumerate(stats)]
    pairs.sort(key=lambda p: p.rank)
    return Interactions(inputs=inputs, rows=len(rows), pairs=pairs, rho_mean=_mean_min(rho)[0],
                        additive=False, method=method)


def _architecture(w: _Work, s: Setting, outcome_unit: str | None, units: Mapping[str, str],
                  roles: Mapping[str, Role], scale: str) -> Architecture:
    kind = model_kind(w.anat.model)
    if kind == "trees":
        columns = [str(c) for c in w.anat.matrix(w.A.iloc[:1]).columns]
        return Architecture(kind="trees", trees=tree_structure(w.anat, columns))
    equation = linear_equation(w.anat, w.A_all, target=s.target, scale=scale,
                               outcome_unit=outcome_unit, units=units, roles=roles)
    if hasattr(w.anat.model, "alpha_") or hasattr(w.anat.model, "C_"):
        return Architecture(kind="shrinkage", equation=equation,
                            path=shrinkage_path(w.anat, w.A_all, s.y))
    return Architecture(kind="equation", equation=equation)


def _exposure_inputs(s: Setting, works: Sequence[_Work]) -> tuple[list[str], list[str]]:
    """The inputs whose curves are drawn, and notes on the columns that get none."""
    sources = works[0].anat.sources
    A = works[0].A
    notes: list[str] = []
    if s.purpose == "inference" and not s.exposures and not s.declared:
        return [], ["Under inference a curve is drawn for the declared exposure, and none is "
                    "declared."]
    asked = list(s.exposures) or (list(s.declared) if s.purpose == "inference" else [])
    if asked:
        found = []
        for raw in asked:
            a = input_of(raw, sources)
            if a is None:
                notes.append(f"`{raw}` is not among the model's inputs, so it has no curve.")
            elif not pd.api.types.is_numeric_dtype(A[a]):
                notes.append(f"`{raw}` holds categories: a curve is drawn for a number only.")
            elif a not in found:
                found.append(a)
        return found[:TOP_EXPOSURES] if not s.exposures else found, notes
    passing = [w for w in works if w.floor.passed] or list(works)
    sign = 1.0 if s.higher_is_better else -1.0
    best = max(passing, key=lambda w: (w.fam.score is not None,
                                       sign * w.fam.score if w.fam.score is not None else 0.0))
    pool = [input_of(raw, sources) for raw in (s.candidates or list(s.X.columns))]
    pool = [a for a in dict.fromkeys(pool) if a is not None and pd.api.types.is_numeric_dtype(A[a])]
    imp = importance(best.phi)
    pool.sort(key=lambda a: -float(imp.get(a, 0.0)))
    return pool[:TOP_EXPOSURES], notes


def _curves(s: Setting, works: Sequence[_Work], inputs: Sequence[str], roles: Mapping[str, Role],
            units: Mapping[str, str]) -> list[InputCurves]:
    out = []
    for a in inputs:
        grid = ale_grid(works[0].A[a])
        curves = []
        for w in works:
            if len(grid.edges) < 2:
                curves.append(Curve(family=w.fam.key, label=w.fam.label, drawn=False, values=[],
                                    reason=f"`{a}` takes one value in these rows."))
            elif not w.floor.passed:
                curves.append(Curve(family=w.fam.key, label=w.fam.label, drawn=False, values=[],
                                    reason=w.floor.reason))
            else:
                draw = ale_curve if s.curve_method == "ale" else pd_curve
                values = draw(w.anat.raw_score, w.A, a, grid)
                curves.append(Curve(family=w.fam.key, label=w.fam.label, drawn=True,
                                    values=masked(values, grid)))
        out.append(InputCurves(
            input=a, sources=list(works[0].anat.sources.get(a, [a])), role=roles.get(a, "predictor"),
            method=s.curve_method, grid=[float(z) for z in grid.edges],
            counts=[int(c) for c in grid.counts], supported=[bool(v) for v in grid.supported],
            unit=units.get(a) if a in works[0].anat.as_raw else None, curves=curves))
    return out


def explain(families: Sequence[FamilyFit], s: Setting, refit: Refit,
            progress: Progress | None = None) -> ExplainArtifact:
    """Every family's SHAP values with their stability, its interaction ranking, its architecture,
    and each top exposure's curve per family on one grid, gated by the performance floor."""
    say = progress or (lambda f, m: None)
    n = len(s.X)
    rng = np.random.default_rng(s.seed)
    size = min(n, EXPLAIN_ROWS, max(100, CELLS // max(1, s.X.shape[1])))
    sample = np.sort(rng.choice(n, size=size, replace=False))
    rows_word = "analyzed rows" if s.purpose == "inference" else "training rows"
    outcome_unit, units = equation_units(s.state, s.target, list(s.X.columns))
    scale = scale_of(s, outcome_unit)
    explained: list[FamilyExplanation] = []
    works: list[_Work] = []
    for i, fam in enumerate(families):
        say(0.05 + 0.85 * i / max(1, len(families)), f"{fam.label}: SHAP values")
        w = _work(fam, s, sample)
        if isinstance(w, FamilyExplanation):
            explained.append(w)
            continue
        roles = _roles(s, w.anat.sources)
        main = importance(w.phi).sort_values(ascending=False, kind="stable")
        def refit_progress(r: int, _i: int = i, _label: str = fam.label) -> None:
            say(0.05 + 0.85 * (_i + 0.1 + 0.5 * (r - 1) / max(1, s.reseeds)) / len(families),
                f"{_label}: refit {r} of {s.reseeds}, for stability")

        again = _refits(w, s, sample, refit, refit_progress) if s.reseeds > 0 else []
        again_ranks = [dict(zip(main.index, ranks(imp.reindex(main.index).to_numpy())))
                       for imp in again]
        shown = list(main.index[:SHOWN_INPUTS])
        importance_rows = [InputImportance(
            input=a, mean_abs=float(main[a]), rank=k + 1, role=roles.get(a, "predictor"),
            reseed_ranks=[int(r[a]) for r in again_ranks],
            in_top=sum(r[a] <= STABLE_TOP for r in again_ranks) if again_ranks else None)
            for k, a in enumerate(main.index[:STABILITY_INPUTS])]
        points = w.phi.iloc[:BEESWARM_ROWS]
        beeswarm = []
        for a in shown:
            value, color, level = _colors(w.A[a].iloc[:BEESWARM_ROWS])
            beeswarm.append(BeeswarmInput(input=a, role=roles.get(a, "predictor"),
                                          phi=[float(v) for v in points[a]], value=value,
                                          level=level, color=color))
        listed = w.phi.iloc[:OBSERVATION_ROWS]
        prediction = w.anat.raw_score(w.A.iloc[:OBSERVATION_ROWS])
        observations = Observations(
            row_ids=[int(v) for v in listed.index], base=w.base,
            prediction=[float(v) for v in prediction], inputs=shown,
            phi=[[float(v) for v in row] for row in listed[shown].to_numpy()],
            rest=[float(v) for v in listed.drop(columns=shown).sum(axis=1)])
        say(0.05 + 0.85 * (i + 0.6) / max(1, len(families)), f"{fam.label}: interactions")
        interactions = _interactions(w, s, list(main.index)) if w.floor.passed else None
        explained.append(FamilyExplanation(
            family=fam.key, label=fam.label, explained=True,
            method=METHOD_WORDS[model_kind(w.anat.model) or "linear"], scale=scale, base=w.base,
            floor=w.floor, importance=importance_rows, beeswarm=beeswarm,
            observations=observations, stability=_stability(w, s, again, main),
            interactions=interactions,
            architecture=_architecture(w, s, outcome_unit, units, roles, scale)))
        works.append(w)
    notes: list[str] = []
    curves: list[InputCurves] = []
    if works:
        say(0.92, "Drawing each exposure's curve per family")
        inputs, notes = _exposure_inputs(s, works)
        curves = _curves(s, works, inputs, _roles(s, works[0].anat.sources), units)
    inference = s.purpose == "inference"
    if not inference and works:
        notes.append(SELECTION_NOTE)
    fired = []  # each relation of the contract, said where it took effect
    if curves:
        fired.append(FLOOR_SAYS)
    if works and s.units is not None and s.reseeds > 0:
        fired.append(UNITS_SAYS)
    if any(c.input not in works[0].anat.as_raw for c in curves):
        fired.append(SCALE_SAYS)
    if inference:
        fired += [LOCK_SAYS, TERMS_SAYS] + ([FILL_SAYS] if s.multiple_imputation else [])
    artifact = ExplainArtifact(
        purpose=s.purpose, task=s.task, describes=DESCRIBES,
        under_inference=UNDER_INFERENCE if inference else None,
        single_fill=SINGLE_FILL if inference and s.multiple_imputation else None,
        adjustment_terms=ADJUSTMENT_TERM if inference else None, relations=fired,
        rows=len(sample), rows_of=n, row_ids=[int(v) for v in s.X.index[sample]],
        rows_basis=(f"SHAP values and curves of all {n:,} {rows_word} the models were fit on."
                    if len(sample) == n else
                    f"SHAP values and curves of a seeded sample of {len(sample):,} of the {n:,} "
                    f"{rows_word} the models were fit on."),
        curve_method=s.curve_method, families=explained, curves=curves, methods="", notes=notes)
    artifact.methods = methods_sentence(artifact, s)
    say(1.0, "Done")
    return artifact


METHOD_WORDS = {"linear": "exact linear SHAP", "trees": "path-dependent TreeSHAP"}


def _listing(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def methods_sentence(a: ExplainArtifact, s: Setting) -> str:
    """The methods paragraph the explanations write, from what was computed."""
    done = [f for f in a.families if f.explained]
    if not done:
        return ""
    rows_word = "analyzed rows" if s.purpose == "inference" else "training rows"
    by_method: dict[str, list[str]] = {}
    for f in done:
        by_method.setdefault(str(f.method), []).append(f"the {f.label.lower()}")
    how = "; ".join(f"{m} values for {_listing(names)}" for m, names in by_method.items())
    sample = f"all {a.rows:,} {rows_word}" if a.rows == a.rows_of else \
        f"a seeded sample of {a.rows:,} of the {a.rows_of:,} {rows_word}"
    out = [f"SHAP values were computed for {sample} on each model's own scale: {how}."]
    stable = [f for f in done if f.stability is not None]
    if stable:
        st = stable[0].stability
        by = "rows" if st.resampled_by == "row" else f"whole {st.resampled_by} units"
        rhos = _listing([f"{_fmt(f.stability.rho_mean)} for the {f.label.lower()}" for f in stable])
        out.append(f"Their stability was measured over {st.reseeds} refits of each model on "
                   f"bootstrap resamples of {by}, each with its own seed: the mean Spearman "
                   f"correlation of the inputs' mean absolute SHAP values between refits was "
                   f"{rhos}.")
    ranked = [f for f in done if f.interactions is not None and not f.interactions.additive]
    additive = [f for f in done if f.interactions is not None and f.interactions.additive]
    if ranked:
        it = ranked[0].interactions
        out.append(f"Pairwise interactions among each model's {len(it.inputs)} most important "
                   f"inputs were ranked by the root mean square of their interaction part, with "
                   f"Friedman and Popescu's H² beside it, on {it.rows:,} rows.")
    if additive:
        verb = "adds its" if len(additive) == 1 else "add their"
        out.append(f"The {_listing([f.label.lower() for f in additive])} {verb} inputs' effects, "
                   f"so no interaction was found to rank.")
    if a.curves:
        names = _listing([f"`{c.input}`" for c in a.curves])
        words = ("Accumulated local effects (Apley and Zhu 2020)" if a.curve_method == "ale" else
                 "Partial dependence curves (Friedman 2001), the customary choice, which average "
                 "over combinations of the inputs the data may not contain,")
        out.append(f"{words} of {names} were drawn for each family on one grid of the inputs' "
                   f"quantiles, with no curve where an interval held fewer than "
                   f"{MIN_SEGMENT_ROWS} rows.")
        dropped = [f for f in done if f.floor is not None and not f.floor.passed]
        if dropped:
            out.append(f"No curve was drawn for "
                       f"{_listing([f'the {f.label.lower()}' for f in dropped])}, whose "
                       f"cross-validated {s.metric_label} did not beat {s.baseline_label}.")
    out.append("These explanations describe each model's predictions, not causal effects.")
    if s.purpose == "inference":
        out.append("Under inference they were not used as effect estimates.")
    return " ".join(out)


# ═════════════════════════════════════════════════════════════════════════════
# the method contract (BLUEPRINT §13)
# ═════════════════════════════════════════════════════════════════════════════

from turbotab.core.methods.contract import (  # noqa: E402
    ContractOption,
    MethodContract,
    Relation,
    register_contract,
)

APLEY_ZHU = "Apley & Zhu 2020, J R Stat Soc B 82:1059–1086"
FRIEDMAN_PD = "Friedman 2001, Ann Stat 29:1189–1232"
FRIEDMAN_POPESCU = "Friedman & Popescu 2008, Ann Appl Stat 2:916–954"
LUNDBERG_LEE = "Lundberg & Lee 2017, NeurIPS 30:4765–4774"
LUNDBERG_TREES = "Lundberg et al. 2020, Nat Mach Intell 2:56–67"
MOLNAR_PITFALLS = "Molnar et al. 2022, xxAI, LNCS 13200:39–68"

# The relations' words, each said where it fires (asserted verbatim by the chain test).
FLOOR_SAYS = ("A family whose cross-validated score does not beat the no-predictor baseline draws "
              "no curve, and says why.")
UNITS_SAYS = ("Rows that repeat within a unit are resampled as whole units in the refits that "
              "measure stability.")
SCALE_SAYS = ("Each curve is drawn on the exposure's final scale, as the model reads it after "
              "energy adjustment or normalization, the scale its form is declared on.")
EFFECT_SAYS = ("An explanation describes a fitted model's predictions, not what changing an "
               "intake would do, so it is never reported as an effect.")
LOCK_SAYS = ("Under inference the explanations wait until the exposure, its effect and the "
             "adjustment set are answered, and their first display locks the analysis plan.")
TERMS_SAYS = ("Under inference the covariates' attributions are labeled adjustment terms, not "
              "effect estimates.")
FILL_SAYS = ("Under multiple imputation the explanations describe the model fitted on one in-fold "
             "fill and estimate nothing.")

CONTRACT = register_contract(MethodContract(
    key="explain",
    label="Model explanations",
    slot="evaluation",
    # Row i's explanation reads the fitted model, so it moves with the outcome (the scope test).
    scope="model",
    run_order=1.0,
    needs=("a fitted family that predicts", "the rows it was fit on",
           "its cross-validated score against the no-predictor baseline",
           "under inference, the declared exposure"),
    question="How should the fitted models be described?",
    options=(
        ContractOption(
            key="ale", label="Accumulated local effects",
            customary=f"Less used in applied papers than partial dependence ({APLEY_ZHU}).",
            sound={"prediction": "Sound with correlated intakes: it averages local changes among "
                                 f"rows that hold such values ({APLEY_ZHU}).",
                   "inference": "Describes the fitted model beside the declared estimate; it is "
                                "not an effect estimate."},
            rung={"prediction": "recommended", "inference": "recommended"},
            order={"prediction": 0, "inference": 0}),
        ContractOption(
            key="partial_dependence", label="Partial dependence",
            customary=f"The usual curve in applied machine-learning papers ({FRIEDMAN_PD}).",
            sound={p: ("Averages predictions over every row's other values, including intake "
                       "combinations the data never hold when intakes are correlated "
                       f"({APLEY_ZHU}; {MOLNAR_PITFALLS}: \"ignoring feature dependencies\").")
                   for p in ("prediction", "inference")},
            rung={"prediction": "rank_lower", "inference": "rank_lower"},
            order={"prediction": 1, "inference": 1}),
        ContractOption(
            key="as_effect", label="Report the explanations as each exposure's effect",
            customary=(f"Common enough to be a named pitfall ({MOLNAR_PITFALLS}: \"making "
                       f"unjustified causal interpretations\")."),
            sound={"prediction": "A prediction model's explanations describe its predictions; an "
                                 "effect needs a declared exposure and effect under inference.",
                   "inference": "The effect is the declared estimand's estimate; an explanation "
                                "of a fitted model is not an effect estimate."},
            rung={"prediction": "refused", "inference": "refused"},
            order={"prediction": 2, "inference": 2}),
    ),
    storyboard=("Each family fit on the same rows",
                "Each prediction split into its inputs' SHAP values",
                "Refit on resamples, for stability",
                "Pairs ranked by their interaction",
                "Each exposure's curve per family, where it beats the baseline"),
    relations=(
        Relation("implies", "performance_floor", FLOOR_SAYS),
        Relation("implies", "unit_resampling", UNITS_SAYS),
        Relation("implies", "domain_transform", SCALE_SAYS),
        Relation("conflicts", "effect_estimate", EFFECT_SAYS, rung="refused", when=("as_effect",)),
        Relation("implies", "plan_lock", LOCK_SAYS, purposes=("inference",)),
        Relation("implies", "adjustment_terms", TERMS_SAYS, purposes=("inference",)),
        Relation("implies", "multiple_imputation", FILL_SAYS, purposes=("inference",)),
    ),
    sources=(LUNDBERG_LEE, LUNDBERG_TREES, FRIEDMAN_POPESCU, APLEY_ZHU, FRIEDMAN_PD,
             MOLNAR_PITFALLS),
    clause=lambda run: run.get("explain_methods") or None,
))


# ═════════════════════════════════════════════════════════════════════════════
# the decision: its leash and its sentence
# ═════════════════════════════════════════════════════════════════════════════

from turbotab.core import decisions as _decisions  # noqa: E402


def _state_of(ctx: Any) -> Any:
    if ctx is None:
        return None
    return ctx.get("state") if isinstance(ctx, Mapping) else getattr(ctx, "state", None)


def _keep(decision: Any, **change: Any) -> dict[str, Any]:
    out = decision.model_dump(mode="json")
    out.update(change)
    return out


def _explanations_are_never_effects(decision: Any, ctx: Any) -> None:
    """The leash (MODELING_SEQUENCE §4; BLUEPRINT §11.3): an explanation is never reported as an
    effect, under either purpose; the explanations themselves stay, described as the models'."""
    if not decision.as_effect:
        return
    state = _state_of(ctx)
    keep = {"label": "Keep the explanations, described as the models'",
            "decision": _keep(decision, as_effect=False)}
    if getattr(state, "purpose", None) == "prediction":
        raise _decisions.Refusal(
            "explanation_is_not_an_effect",
            "A prediction model's explanations describe its predictions, not what changing an "
            "intake would do; an effect needs a declared exposure and effect under inference.",
            exits=[keep, {"label": "Declare an inference analysis with an exposure and its effect",
                          "decision": {"kind": "set_purpose", "purpose": "inference"}}])
    raise _decisions.Refusal(
        "explanation_is_not_an_effect",
        "Under inference the effect is the declared estimate in the coefficient table; an "
        "explanation describes the fitted model's predictions and is never reported as an effect "
        f"estimate ({MOLNAR_PITFALLS}: \"making unjustified causal interpretations\").",
        exits=[keep, {"label": "Report the declared estimate from the coefficient table",
                      "decision": None}])


def _explained_columns_are_predictors(decision: Any, ctx: Any) -> None:
    """A curve is drawn only for one of the model's predictors (the settled roles')."""
    state = _state_of(ctx)
    if not decision.exposures or state is None:
        return
    from turbotab.core.models.pipeline import model_predictors

    predictors = set(model_predictors(state))
    absent = [c for c in decision.exposures if c not in predictors]
    if absent:
        named = ", ".join(f"`{c}`" for c in absent)
        raise _decisions.Refusal(
            "not_a_predictor",
            f"{named} {'is' if len(absent) == 1 else 'are'} not among the model's predictors, so "
            f"there is no curve to draw.",
            exits=[{"label": "Draw the top exposures' curves",
                    "decision": _keep(decision, exposures=[])}])


def _stability_compares_refits(decision: Any, ctx: Any) -> None:
    """Stability is a comparison between refits: none (0) or at least two."""
    if decision.reseeds == 1:
        raise _decisions.Refusal(
            "one_refit",
            "One refit has no other to be compared with; stability needs at least two refits.",
            exits=[{"label": f"Measure stability over {RESEEDS} refits",
                    "decision": _keep(decision, reseeds=RESEEDS)},
                   {"label": "Skip the stability check",
                    "decision": _keep(decision, reseeds=0)}])


_decisions.register_validator("set_explain", _explanations_are_never_effects, first=True)
_decisions.register_validator("set_explain", _stability_compares_refits)
_decisions.register_validator("set_explain", _explained_columns_are_predictors)


def decision_sentence(d: Any, state: Any) -> str:
    """The sentence ``set_explain`` records (``voice``): what the explanations are and what they
    are not."""
    curves = ("accumulated local effects curves (Apley and Zhu 2020)" if d.curves == "ale" else
              "partial dependence curves (Friedman 2001), the customary choice, which average "
              "over combinations of the inputs the data may not contain,")
    if d.exposures:
        of = " and ".join(f"`{c}`" for c in d.exposures) if len(d.exposures) <= 2 else \
            ", ".join(f"`{c}`" for c in d.exposures[:-1]) + f" and `{d.exposures[-1]}`"
    else:
        of = ("the declared exposure" if getattr(state, "purpose", None) == "inference" else
              "the top exposures")
    stability = (f", with their stability over {d.reseeds} refits on bootstrap resamples"
                 if d.reseeds else ", without a check of their stability across refits")
    tail = "these describe the models' predictions, not causal effects"
    if getattr(state, "purpose", None) == "inference":
        tail += ", and are not effect estimates"
    return (f"The fitted models were described by their SHAP values{stability}, by a ranking of "
            f"pairwise interactions (Friedman and Popescu's H statistic), and by {curves} of {of}, "
            f"drawn only for families whose cross-validated score beats the no-predictor "
            f"baseline; {tail}.")


__all__ = [
    "ADJUSTMENT_TERM", "CONTRACT", "DESCRIBES", "ExplainArtifact", "FamilyFit", "Setting",
    "SINGLE_FILL", "UNDER_INFERENCE", "ale_curve", "ale_grid", "attributions", "anatomy",
    "decision_sentence", "equation_units", "explain", "h_statistics", "hgb_ensemble",
    "linear_equation", "linear_shap", "partial_dependence_at_rows", "pd_curve", "resample",
    "shrinkage_path", "spearman", "tree_shap", "tree_structure",
]

"""Dietary patterns (V2 definition of done, amendment of 2026-10-07; crosswalk D3): the engine core.

Four ways to find patterns in food-group intakes, each as its own method contract option:

* **principal components** of the food groups' correlation matrix (``pca``);
* **exploratory factor analysis** by principal axis factoring with iterated communalities
  (``factor_analysis``);
* **k-means cluster analysis** of people by their intakes (``cluster_analysis``);
* **reduced rank regression** on intermediate responses (``reduced_rank_regression``; Hoffmann et
  al. 2004): the food-group combinations that explain the most variation in responses on the
  pathway (biomarkers, nutrients), never the outcome being studied, which is refused.

Components and factors are rotated by varimax (Kaiser 1958, with Kaiser's row normalization), and
their number is retained by a declared rule: parallel analysis (Horn 1965; the 95th percentile of
random data of the same size, Glorfeld 1995), the scree rule (the number read from the plot of
eigenvalues, stated by the analyst; Cattell 1966) or eigenvalues above 1 (Kaiser 1960, customary
and known to keep too many; Zwick & Velicer 1986). Clusters number by the largest average
silhouette width (Rousseeuw 1987) or a declared k.

Before any of them, the food groups are made comparable as declared: energy-adjusted by the
residual method (Willett et al. 1997), expressed per 1,000 kcal, or left in their units; each is
then standardized, so components and factors come from the correlation matrix.

**Survey weights.** Every one of the four has a design-based point estimate, so a weight column is
honored throughout: weighted means, standard deviations and correlations (the reliability-weights
denominator, so equal weights give the unweighted answer exactly), weighted least squares for the
residual method and reduced rank regression, weighted k-means centers and a weighted average
silhouette width. Parallel analysis draws its random data under the same weights. No standard
error is reported for a loading, so the strata and PSUs never enter a number here.

**Under Predict** every quantity is learned from the training rows only (:class:`PatternTransformer`
fits on the fold and scores the held-out rows with the fold's means, standard deviations, energy
coefficients, loadings and centers); the outcome never enters, so the scope is ``training_fold``.

Nothing here is wired to a stage or a question yet: the Describe goal and the stages take it up in
crosswalk D1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PatternMethod = Literal["pca", "factor_analysis", "cluster_analysis", "reduced_rank_regression"]
InputForm = Literal["residual", "density", "standardized"]
CountRule = Literal["parallel_analysis", "scree", "eigenvalue_over_one"]
ClusterRule = Literal["silhouette", "declared"]

METHODS: tuple[str, ...] = ("pca", "factor_analysis", "cluster_analysis", "reduced_rank_regression")
INPUT_FORMS: tuple[str, ...] = ("residual", "density", "standardized")
COUNT_RULES: tuple[str, ...] = ("parallel_analysis", "scree", "eigenvalue_over_one")
CLUSTER_RULES: tuple[str, ...] = ("silhouette", "declared")

CONTRACT = "dietary_patterns"
INPUTS_CONTRACT = "pattern_inputs"
COUNT_CONTRACT = "pattern_count"
CLUSTERS_CONTRACT = "pattern_clusters"

# Primary sources, cited by the contracts and the methods sentence.
HU_2002 = "Hu 2002, Curr Opin Lipidol 13:3"
NEWBY_TUCKER = "Newby & Tucker 2004, Nutr Rev 62:177"
NEWBY_2003 = "Newby et al. 2003, Am J Clin Nutr 77:1417"
HOFFMANN = "Hoffmann et al. 2004, Am J Epidemiol 159:935"
HORN = "Horn 1965, Psychometrika 30:179"
GLORFELD = "Glorfeld 1995, Educ Psychol Meas 55:377"
KAISER_1958 = "Kaiser 1958, Psychometrika 23:187"
KAISER_1960 = "Kaiser 1960, Educ Psychol Meas 20:141"
CATTELL = "Cattell 1966, Multivariate Behav Res 1:245"
ZWICK_VELICER = "Zwick & Velicer 1986, Psychol Bull 99:432"
ROUSSEEUW = "Rousseeuw 1987, J Comput Appl Math 20:53"
WILLETT = "Willett, Howe & Kushi 1997, Am J Clin Nutr 65:1220S"
FABRIGAR = "Fabrigar et al. 1999, Psychol Methods 4:272"

_TOL = 1e-12


class PatternRefused(ValueError):
    """A pattern the data or the declaration cannot support, with its ways forward."""

    def __init__(self, reason: str, exits: Sequence[str] = ()):
        super().__init__(reason)
        self.reason = reason
        self.exits = tuple(exits)


@dataclass(frozen=True)
class PatternSpec:
    """What the analyst declared.

    ``foods`` are the food-group intake columns. ``inputs`` says how they are made comparable;
    ``energy`` is the total-energy column (kcal) the residual and density forms need. ``count_rule``
    and ``n_patterns`` decide how many components or factors are kept (the scree rule needs the
    stated number); ``cluster_rule``, ``k`` and ``k_range`` decide the clusters. ``responses`` are
    reduced rank regression's intermediate responses and ``outcome`` the outcome being studied,
    which may never be one of them. ``weights`` is a survey weight column.
    """

    method: PatternMethod
    foods: tuple[str, ...]
    inputs: InputForm = "standardized"
    energy: str | None = None
    count_rule: CountRule = "parallel_analysis"
    n_patterns: int | None = None
    cluster_rule: ClusterRule = "silhouette"
    k: int | None = None
    k_range: tuple[int, int] = (2, 8)
    responses: tuple[str, ...] = ()
    outcome: str | None = None
    weights: str | None = None
    iterations: int = 1000  # parallel analysis's random data sets
    quantile: float = 0.95
    starts: int = 25  # k-means random starts
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "foods", tuple(self.foods))
        object.__setattr__(self, "responses", tuple(self.responses))
        object.__setattr__(self, "k_range", tuple(self.k_range))
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r} is not one of {METHODS}")
        if self.inputs not in INPUT_FORMS:
            raise ValueError(f"inputs {self.inputs!r} is not one of {INPUT_FORMS}")
        if self.count_rule not in COUNT_RULES:
            raise ValueError(f"count_rule {self.count_rule!r} is not one of {COUNT_RULES}")
        if self.cluster_rule not in CLUSTER_RULES:
            raise ValueError(f"cluster_rule {self.cluster_rule!r} is not one of {CLUSTER_RULES}")


# ── weighted moments ─────────────────────────────────────────────────────────


def _denominator(w: np.ndarray) -> float:
    """The reliability-weights denominator, sum(w) - sum(w²)/sum(w): n - 1 under equal weights."""
    s = float(w.sum())
    return s - float((w ** 2).sum()) / s


def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w @ X) / w.sum()


def weighted_covariance(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted covariance matrix (R's ``cov.wt(method = "unbiased")``)."""
    D = X - weighted_mean(X, w)
    return (D * w[:, None]).T @ D / _denominator(w)


def weighted_correlation(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted correlation matrix: a design-consistent estimate of the population's."""
    C = weighted_covariance(X, w)
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return (R + R.T) / 2


def _standardize(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = weighted_mean(X, w)
    scale = np.sqrt(np.diag(weighted_covariance(X, w)))
    return center, scale


# ── making the food groups comparable ─────────────────────────────────────────


@dataclass(frozen=True)
class Inputs:
    """The fitted pre-processing: the energy coefficients (residual form), then each food group's
    center and scale, all learned from the rows it was fitted on."""

    form: InputForm
    foods: tuple[str, ...]
    energy: str | None
    center: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray | None = None
    slope: np.ndarray | None = None

    def adjusted(self, frame: pd.DataFrame) -> np.ndarray:
        """The food groups in the declared form, before standardizing."""
        X = _numeric(frame, self.foods, "food group")
        if self.form == "standardized":
            return X
        e = _numeric(frame, (self.energy,), "total energy")[:, 0]  # type: ignore[arg-type]
        if self.form == "density":
            if np.any(e <= 0):
                raise PatternRefused(
                    "A row's total energy is zero or below, so its intake per 1,000 kcal is "
                    "undefined.", exits=("leave those rows out", "energy-adjust by the residual "
                                         "method instead"))
            return X / e[:, None] * 1000.0
        return X - (self.intercept + np.outer(e, self.slope))  # type: ignore[operator]

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        """The standardized food groups, with the fitted rows' center and scale."""
        return (self.adjusted(frame) - self.center) / self.scale


def _numeric(frame: pd.DataFrame, columns: Sequence[str], what: str) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PatternRefused(f"No {what} column named {', '.join(map(str, missing))}.")
    X = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        bad = [c for c, ok in zip(columns, np.isfinite(X).all(axis=0)) if not ok]
        raise PatternRefused(
            f"The {what} column{'s' if len(bad) > 1 else ''} {', '.join(map(str, bad))} "
            f"{'have' if len(bad) > 1 else 'has'} blanks or values that are not numbers.",
            exits=("fill the blanks first (missing values are handled before patterns)",
                   "leave those rows or columns out"))
    return X


def fit_inputs(frame: pd.DataFrame, form: InputForm, foods: Sequence[str], energy: str | None,
               w: np.ndarray) -> Inputs:
    """Fit the declared pre-processing on ``frame``'s rows with weights ``w``."""
    foods = tuple(foods)
    if form != "standardized" and not energy:
        raise PatternRefused(
            "Energy adjustment needs the total-energy column.",
            exits=("name the total-energy column", "standardize the food groups without "
                   "energy adjustment"))
    if energy and energy in foods:
        raise PatternRefused("Total energy cannot also be one of the food groups.",
                             exits=("leave total energy out of the food groups",))
    intercept = slope = None
    if form == "residual":
        X = _numeric(frame, foods, "food group")
        e = _numeric(frame, (energy,), "total energy")[:, 0]  # type: ignore[arg-type]
        D = np.column_stack([np.ones_like(e), e])
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(D * sw[:, None], X * sw[:, None], rcond=None)
        intercept, slope = coef[0], coef[1]
    staged = Inputs(form, foods, energy, np.zeros(len(foods)), np.ones(len(foods)), intercept,
                    slope)
    A = staged.adjusted(frame)
    center, scale = _standardize(A, w)
    floor = 1e-12 * np.maximum(1.0, np.abs(A).max(axis=0))
    flat = [f for f, s, lo in zip(foods, scale, floor) if not s > lo]
    if flat:
        raise PatternRefused(
            f"The food group{'s' if len(flat) > 1 else ''} {', '.join(flat)} "
            f"{'do' if len(flat) > 1 else 'does'} not vary, so it cannot share a pattern.",
            exits=("leave it out of the food groups",))
    return Inputs(form, foods, energy, center, scale, intercept, slope)


# ── rotation and retention ────────────────────────────────────────────────────


def varimax(L: np.ndarray, normalize: bool = True, tol: float = 1e-13,
            max_iter: int = 100000) -> tuple[np.ndarray, np.ndarray]:
    """Varimax rotation (Kaiser 1958, with Kaiser's row normalization), by the iteration of R's
    ``stats::varimax``, run until the rotation itself stops moving. R stops when the criterion
    stops rising (by 1e-5 by default); the criterion is flat at its maximum, so that stop leaves
    the loadings off in the third decimal, and even a stop at machine precision in the seventh.

    Returns the rotated loadings and the orthogonal rotation matrix ``T`` (``L @ T``)."""
    L = np.asarray(L, dtype=float)
    p, m = L.shape
    if m < 2:
        return L.copy(), np.eye(m)
    sc = np.sqrt((L ** 2).sum(axis=1)) if normalize else np.ones(p)
    sc = np.where(sc > 0, sc, 1.0)
    x = L / sc[:, None]
    T = np.eye(m)
    for _ in range(max_iter):
        z = x @ T
        B = x.T @ (z ** 3 - z @ np.diag((z ** 2).sum(axis=0)) / p)
        u, _, vt = np.linalg.svd(B)
        T, T_past = u @ vt, T
        if np.max(np.abs(T - T_past)) < tol:
            break
    return (x @ T) * sc[:, None], T


def eigen(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (largest first) and their eigenvectors of a symmetric matrix."""
    values, vectors = np.linalg.eigh((R + R.T) / 2)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def parallel_analysis(n: int, p: int, w: np.ndarray | None = None, *, iterations: int = 1000,
                      quantile: float = 0.95, seed: int = 0) -> np.ndarray:
    """The reference eigenvalues of parallel analysis (Horn 1965): for each position, the
    ``quantile`` (Glorfeld 1995's 95th percentile) of the correlation-matrix eigenvalues of
    ``iterations`` sets of independent normal data of ``n`` rows and ``p`` columns, each correlated
    under the same weights ``w`` as the data were."""
    rng = np.random.default_rng(seed)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    out = np.empty((iterations, p))
    for i in range(iterations):
        out[i] = eigen(weighted_correlation(rng.standard_normal((n, p)), w))[0]
    return np.quantile(out, quantile, axis=0)


def retained(eigenvalues: np.ndarray, rule: CountRule, *, reference: np.ndarray | None = None,
             declared: int | None = None) -> int:
    """How many components or factors the declared rule keeps.

    Parallel analysis keeps the leading run of eigenvalues above the reference; eigenvalues above 1
    count every one above 1; the scree rule keeps the number the analyst read from the plot."""
    if rule == "scree":
        if declared is None:
            raise PatternRefused("The scree rule needs the number of patterns read from the plot.",
                                 exits=("state the number of patterns from the scree plot",
                                        "retain by parallel analysis instead"))
        return int(declared)
    if rule == "eigenvalue_over_one":
        return int(np.sum(np.asarray(eigenvalues) > 1.0))
    if reference is None:
        raise ValueError("parallel analysis needs its reference eigenvalues")
    above = np.asarray(eigenvalues) > np.asarray(reference)
    return int(len(above) if above.all() else np.argmin(above))


def _orient(L: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Columns ordered by their sum of squared loadings (largest first), each signed so that its
    largest loading in absolute value is positive; ``others`` get the same order and signs."""
    ss = (L ** 2).sum(axis=0)
    order = np.argsort(-ss, kind="stable")
    L = L[:, order]
    sign = np.sign(L[np.argmax(np.abs(L), axis=0), np.arange(L.shape[1])])
    sign[sign == 0] = 1.0
    return (L * sign, *(o[:, order] * sign for o in others))


# ── the four methods ──────────────────────────────────────────────────────────


def principal_axis(R: np.ndarray, m: int, *, tol: float = _TOL,
                   max_iter: int = 100000) -> tuple[np.ndarray, np.ndarray, int]:
    """Principal axis factoring with iterated communalities, started at the squared multiple
    correlations (R's ``psych::fa(fm = "pa")``). Returns the unrotated loadings, the
    communalities and the iterations taken."""
    h = 1.0 - 1.0 / np.diag(np.linalg.inv(R))
    L = np.zeros((R.shape[0], m))
    for i in range(1, max_iter + 1):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        values, vectors = eigen(Rr)
        L = vectors[:, :m] * np.sqrt(np.clip(values[:m], 0.0, None))
        new = (L ** 2).sum(axis=1)
        if np.max(np.abs(new - h)) < tol:
            return L, new, i
        h = new
    raise PatternRefused(
        f"The factor analysis did not settle in {max_iter:,} iterations.",
        exits=("retain fewer factors", "use principal components instead"))


def kmeans(Z: np.ndarray, k: int, w: np.ndarray, *, starts: int = 25, seed: int = 0,
           init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted k-means (Lloyd's algorithm, the best of ``starts`` k-means++ starts, or one start
    at ``init``). Returns the labels (0..k-1, largest weighted cluster first), the centers and the
    weighted within-cluster sum of squares."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, init=init if init is not None else "k-means++",
                n_init=1 if init is not None else starts, random_state=seed, algorithm="lloyd",
                max_iter=10000, tol=0.0)
    km.fit(Z, sample_weight=w)
    labels, centers = km.labels_, km.cluster_centers_
    size = np.bincount(labels, weights=w, minlength=k)
    order = np.lexsort((centers[:, 0], -size))
    relabel = np.empty(k, dtype=int)
    relabel[order] = np.arange(k)
    labels, centers = relabel[labels], centers[order]
    within = float(np.sum(w * ((Z - centers[labels]) ** 2).sum(axis=1)))
    return labels, centers, within


def silhouette_widths(Z: np.ndarray, labelings: Sequence[np.ndarray], w: np.ndarray | None = None,
                      *, block: int = 1024) -> list[float]:
    """The (weighted) average silhouette width (Rousseeuw 1987) of each labeling of the rows of
    ``Z``. A person's a(i) is the weighted mean distance to the other members of their cluster,
    b(i) the smallest weighted mean distance to another cluster's members, s(i) = (b - a) /
    max(a, b) (0 in a cluster of one), averaged with the weights. Equal weights give R's
    ``cluster::silhouette``. The distances are computed in blocks, once for every labeling."""
    n = Z.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    ks = [int(lab.max()) + 1 for lab in labelings]
    offsets = np.concatenate([[0], np.cumsum(ks)])
    M = np.zeros((n, int(offsets[-1])))
    for j, lab in enumerate(labelings):
        M[np.arange(n), offsets[j] + lab] = w
    mass = M.sum(axis=0)
    sq = (Z ** 2).sum(axis=1)
    s = np.zeros((len(labelings), n))
    for start in range(0, n, block):
        rows = slice(start, min(n, start + block))
        D = np.sqrt(np.clip(sq[rows, None] + sq[None, :] - 2 * Z[rows] @ Z.T, 0.0, None))
        sums = D @ M  # rows × every cluster of every labeling
        for j, lab in enumerate(labelings):
            cols = slice(offsets[j], offsets[j + 1])
            own = lab[rows]
            S = sums[:, cols]
            Mj = mass[cols]
            idx = np.arange(S.shape[0])
            own_mass = Mj[own] - w[rows]  # the person is not their own neighbor
            a = np.where(own_mass > 0, S[idx, own] / np.where(own_mass > 0, own_mass, 1.0), 0.0)
            other = S / Mj
            other[idx, own] = np.inf
            b = other.min(axis=1)
            width = np.where(own_mass > 0, (b - a) / np.maximum(np.maximum(a, b), 1e-300), 0.0)
            s[j, rows] = width
    return [float(np.sum(w * row) / w.sum()) for row in s]


def check_responses(spec: PatternSpec, y: Any = None, frame: pd.DataFrame | None = None) -> None:
    """Reduced rank regression's responses must be intermediate ones: never the outcome being
    studied (by name, or a response column that equals the outcome passed to the fit), never a
    food group, and at least one."""
    exits = ("choose intermediate responses on the pathway, such as biomarkers or nutrients",
             "derive the patterns by principal components or factor analysis instead")
    if not spec.responses:
        raise PatternRefused("Reduced rank regression needs at least one intermediate response.",
                             exits=exits)
    if spec.outcome is not None and spec.outcome in spec.responses:
        raise PatternRefused(
            f"The outcome being studied, {spec.outcome}, cannot be a response: patterns chosen to "
            f"explain the outcome would make the association with it circular (Hoffmann et al. "
            f"2004 use intermediate responses).", exits=exits)
    overlap = [r for r in spec.responses if r in spec.foods]
    if overlap:
        raise PatternRefused(f"{', '.join(overlap)} cannot be both a food group and a response.",
                             exits=("leave it out of one of the two lists",))
    if y is not None and frame is not None:
        target = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce").to_numpy(float)
        for r in spec.responses:
            if r in frame.columns and len(target) == len(frame):
                col = pd.to_numeric(frame[r], errors="coerce").to_numpy(float)
                if np.allclose(col, target, equal_nan=True):
                    raise PatternRefused(
                        f"The response {r} holds the outcome being studied, so it cannot be a "
                        f"response.", exits=exits)


# ── the fitted patterns ───────────────────────────────────────────────────────


@dataclass
class Patterns:
    """Dietary patterns fitted on a set of rows, and how to score any row with them.

    ``loadings`` (food group × pattern) are the rotated loadings of components and factors, the
    correlations of each food group with each reduced rank regression score, and each cluster's
    standardized center. ``coefficients`` turn standardized food groups into scores.
    """

    spec: PatternSpec
    inputs: Inputs
    n: int
    weighted: bool
    count: int
    eigenvalues: np.ndarray | None = None
    reference: np.ndarray | None = None
    loadings: pd.DataFrame | None = None
    coefficients: np.ndarray | None = None
    explained: np.ndarray | None = None  # share of the food groups' variation, per pattern
    communalities: np.ndarray | None = None
    iterations: int | None = None
    centers: np.ndarray | None = None
    profile: pd.DataFrame | None = None  # each cluster's weighted mean intake, in the inputs' form
    shares: np.ndarray | None = None  # each cluster's weighted share of the rows
    labels: np.ndarray | None = None
    silhouettes: dict[int, float] = field(default_factory=dict)
    response_loadings: pd.DataFrame | None = None
    response_explained: np.ndarray | None = None
    response_center: np.ndarray | None = None
    response_scale: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        if self.spec.method == "cluster_analysis":
            return [f"cluster_{i + 1}" for i in range(self.count)]
        return [f"pattern_{i + 1}" for i in range(self.count)]

    def scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Each row's scores (or its cluster, 1..k), with the fitted rows' pre-processing."""
        Z = self.inputs.apply(frame)
        if self.spec.method == "cluster_analysis":
            d = ((Z[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)  # type: ignore[index]
            return pd.DataFrame({"pattern_cluster": d.argmin(axis=1) + 1}, index=frame.index)
        return pd.DataFrame(Z @ self.coefficients, index=frame.index, columns=self.names)

    def defining(self, threshold: float = 0.3) -> dict[str, list[str]]:
        """The food groups that load on each pattern at ``threshold`` or above in absolute value,
        strongest first (a customary cut; Newby & Tucker 2004 report 0.2 to 0.3)."""
        if self.loadings is None or self.spec.method == "cluster_analysis":
            return {}
        out = {}
        for name in self.loadings.columns:
            col = self.loadings[name]
            keep = col[col.abs() >= threshold]
            out[name] = list(keep.reindex(keep.abs().sort_values(ascending=False).index).index)
        return out

    def summary(self) -> dict[str, Any]:
        """What the record keeps: the declaration and every number the sentence states."""
        out: dict[str, Any] = {
            "method": self.spec.method, "inputs": self.spec.inputs, "foods": list(self.spec.foods),
            "n": self.n, "weighted": self.weighted, "count": self.count}
        if self.spec.method in ("pca", "factor_analysis"):
            out.update(count_rule=self.spec.count_rule, iterations=self.spec.iterations,
                       quantile=self.spec.quantile)
        if self.spec.method == "cluster_analysis":
            out.update(cluster_rule=self.spec.cluster_rule, k_range=list(self.spec.k_range),
                       silhouettes={str(k): v for k, v in self.silhouettes.items()})
        if self.spec.method == "reduced_rank_regression":
            out.update(responses=list(self.spec.responses),
                       response_explained=[float(x) for x in self.response_explained])  # type: ignore[union-attr]
        if self.explained is not None:
            out["explained"] = [float(x) for x in self.explained]
        return out

    def sentence(self, purpose: str = "inference") -> str:
        return methods_sentence(self, purpose)


def _weights(frame: pd.DataFrame, spec: PatternSpec, sample_weight: Any = None) -> np.ndarray:
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float).ravel()
    elif spec.weights:
        w = _numeric(frame, (spec.weights,), "survey weight")[:, 0]
    else:
        return np.ones(len(frame))
    if len(w) != len(frame) or not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
        raise PatternRefused("The survey weights must be finite, zero or above, and not all zero.",
                             exits=("fix the weight column", "derive the patterns unweighted, "
                                    "for the sample only"))
    return w


def fit_patterns(frame: pd.DataFrame, spec: PatternSpec, *, y: Any = None,
                 sample_weight: Any = None) -> Patterns:
    """Derive the declared dietary patterns from ``frame``'s rows (a training fold under Predict,
    every analyzed row otherwise). ``y`` is the outcome, read only to refuse it as a response."""
    if spec.method == "reduced_rank_regression":
        check_responses(spec, y, frame)
    if len(spec.foods) < 2:
        raise PatternRefused("Patterns need at least two food groups.",
                             exits=("add food groups",))
    w = _weights(frame, spec, sample_weight)
    weighted = bool(spec.weights) or sample_weight is not None
    inputs = fit_inputs(frame, spec.inputs, spec.foods, spec.energy, w)
    Z = inputs.apply(frame)
    n, p = Z.shape
    if int(np.sum(w > 0)) <= p:
        raise PatternRefused(f"{p} food groups need more than {p} rows with a weight.",
                             exits=("combine food groups into fewer", "add rows"))
    fit = {"pca": _fit_components, "factor_analysis": _fit_factors,
           "cluster_analysis": _fit_clusters, "reduced_rank_regression": _fit_rrr}[spec.method]
    return fit(frame, spec, inputs, Z, w, weighted)


def _count(spec: PatternSpec, values: np.ndarray, n_rows: int, w: np.ndarray, p: int,
           cap: int) -> tuple[int, np.ndarray | None]:
    reference = None
    if spec.count_rule == "parallel_analysis":
        reference = parallel_analysis(n_rows, p, w, iterations=spec.iterations,
                                      quantile=spec.quantile, seed=spec.seed)
    m = retained(values, spec.count_rule, reference=reference, declared=spec.n_patterns)
    if m < 1:
        raise PatternRefused(
            "No pattern stands out: no eigenvalue is above what random data of the same size "
            "give." if spec.count_rule == "parallel_analysis" else
            "No eigenvalue is above 1, so the rule keeps no pattern.",
            exits=("report that the food groups share no clear pattern",
                   "state the number of patterns from the scree plot"))
    if m > cap:
        raise PatternRefused(f"{m} patterns cannot be drawn from {p} food groups (at most {cap}).",
                             exits=(f"retain {cap} or fewer",))
    return m, reference


def _fit_components(frame: pd.DataFrame, spec: PatternSpec, inputs: Inputs, Z: np.ndarray,
                    w: np.ndarray, weighted: bool) -> Patterns:
    R = weighted_correlation(Z, w)
    values, vectors = eigen(R)
    p = Z.shape[1]
    m, reference = _count(spec, values, Z.shape[0], w, p, p)
    L = vectors[:, :m] * np.sqrt(values[:m])
    Lr, T = varimax(L)
    A = vectors[:, :m] / np.sqrt(values[:m]) @ T  # unit-variance, uncorrelated scores
    Lr, A = _orient(Lr, A)
    names = [f"pattern_{i + 1}" for i in range(m)]
    return Patterns(spec, inputs, Z.shape[0], weighted, m, eigenvalues=values, reference=reference,
                    loadings=pd.DataFrame(Lr, index=list(spec.foods), columns=names),
                    coefficients=A, explained=(Lr ** 2).sum(axis=0) / p,
                    communalities=(Lr ** 2).sum(axis=1))


def _fit_factors(frame: pd.DataFrame, spec: PatternSpec, inputs: Inputs, Z: np.ndarray,
                 w: np.ndarray, weighted: bool) -> Patterns:
    R = weighted_correlation(Z, w)
    values, _ = eigen(R)
    p = Z.shape[1]
    m, reference = _count(spec, values, Z.shape[0], w, p, p - 1)
    L, h, iterations = principal_axis(R, m)
    Lr, _ = varimax(L)
    A = np.linalg.solve(R, Lr)  # regression (Thurstone) scores
    Lr, A = _orient(Lr, A)
    notes = []
    if np.any(h >= 1.0):
        notes.append("A food group's communality reached 1 or above (a Heywood case): its "
                     "uniqueness is not positive, so that factor solution is suspect.")
    names = [f"pattern_{i + 1}" for i in range(m)]
    return Patterns(spec, inputs, Z.shape[0], weighted, m, eigenvalues=values, reference=reference,
                    loadings=pd.DataFrame(Lr, index=list(spec.foods), columns=names),
                    coefficients=A, explained=(Lr ** 2).sum(axis=0) / p, communalities=h,
                    iterations=iterations, notes=notes)


def _fit_clusters(frame: pd.DataFrame, spec: PatternSpec, inputs: Inputs, Z: np.ndarray,
                  w: np.ndarray, weighted: bool) -> Patterns:
    n = Z.shape[0]
    if spec.cluster_rule == "declared":
        if not spec.k or spec.k < 2:
            raise PatternRefused("A declared number of clusters must be 2 or more.",
                                 exits=("state the number of clusters",
                                        "choose it by the silhouette width instead"))
        candidates = [int(spec.k)]
    else:
        lo, hi = spec.k_range
        candidates = [k for k in range(max(2, lo), min(hi, n - 1) + 1)]
        if not candidates:
            raise PatternRefused("The range of cluster numbers is empty.",
                                 exits=("widen the range",))
    fits = {k: kmeans(Z, k, w, starts=spec.starts, seed=spec.seed) for k in candidates}
    widths: dict[int, float] = {}
    if spec.cluster_rule == "silhouette":
        widths = dict(zip(candidates, silhouette_widths(Z, [fits[k][0] for k in candidates], w)))
        k = max(candidates, key=lambda c: (widths[c], -c))
    else:
        k = candidates[0]
    labels, centers, within = fits[k]
    A = inputs.adjusted(frame)
    profile = pd.DataFrame(
        [weighted_mean(A[labels == j], w[labels == j]) for j in range(k)],
        index=[f"cluster_{j + 1}" for j in range(k)], columns=list(spec.foods)).T
    total = float(np.sum(w * (Z ** 2).sum(axis=1)))
    shares = np.bincount(labels, weights=w, minlength=k) / w.sum()
    return Patterns(spec, inputs, n, weighted, k,
                    loadings=pd.DataFrame(centers.T, index=list(spec.foods),
                                          columns=[f"cluster_{j + 1}" for j in range(k)]),
                    explained=np.array([1.0 - within / total]), centers=centers, profile=profile,
                    shares=shares, labels=labels + 1, silhouettes=widths)


def _fit_rrr(frame: pd.DataFrame, spec: PatternSpec, inputs: Inputs, Z: np.ndarray,
             w: np.ndarray, weighted: bool) -> Patterns:
    """Reduced rank regression (Hoffmann et al. 2004): the weighted least-squares fit of the
    standardized responses on the standardized food groups, then the principal components of the
    fitted responses. Each factor score is a combination of food groups with unit variance; the
    k-th explains the k-th largest share of the responses' variation."""
    Yraw = _numeric(frame, spec.responses, "response")
    yc, ys = _standardize(Yraw, w)
    if np.any(ys <= 0):
        raise PatternRefused("A response does not vary, so it cannot be explained.",
                             exits=("leave it out of the responses",))
    Y = (Yraw - yc) / ys
    q, p = Y.shape[1], Z.shape[1]
    den = _denominator(w)
    Zw = Z * w[:, None]
    B = np.linalg.solve(Zw.T @ Z, Zw.T @ Y)
    F = Z @ B
    C = (F * w[:, None]).T @ F / den
    values, vectors = eigen(C)
    m = q if spec.n_patterns is None else int(spec.n_patterns)
    if not 1 <= m <= min(q, p):
        raise PatternRefused(f"Reduced rank regression gives at most {min(q, p)} factors here "
                             f"(one per response).", exits=(f"retain {min(q, p)} or fewer",))
    if values[m - 1] <= 1e-12:
        raise PatternRefused("The food groups explain none of a response's variation.",
                             exits=("retain fewer factors", "choose other responses"))
    A = B @ vectors[:, :m] / np.sqrt(values[:m])
    T = Z @ A
    L = (Zw.T @ T) / den  # correlations of the food groups with the scores
    Ly = ((Y * w[:, None]).T @ T) / den
    sign = np.sign(L[np.argmax(np.abs(L), axis=0), np.arange(m)])
    sign[sign == 0] = 1.0
    L, A, Ly = L * sign, A * sign, Ly * sign
    names = [f"pattern_{i + 1}" for i in range(m)]
    return Patterns(spec, inputs, Z.shape[0], weighted, m, eigenvalues=values,
                    loadings=pd.DataFrame(L, index=list(spec.foods), columns=names),
                    coefficients=A, explained=(L ** 2).sum(axis=0) / p,
                    response_loadings=pd.DataFrame(Ly, index=list(spec.responses), columns=names),
                    response_explained=values[:m] / q, response_center=yc, response_scale=ys)


# ── in each training fold ─────────────────────────────────────────────────────


class PatternTransformer(TransformerMixin, BaseEstimator):
    """The dietary patterns as an in-fold step: ``fit`` derives them from the training rows only
    (refusing the outcome as a reduced rank regression response), ``transform`` scores any rows
    with what the training rows taught it. The food groups are replaced by the pattern scores (or
    the cluster) unless ``keep_foods``."""

    def __init__(self, spec: PatternSpec | None = None, keep_foods: bool = False):
        self.spec = spec
        self.keep_foods = keep_foods

    def fit(self, X: pd.DataFrame, y: Any = None, sample_weight: Any = None) -> "PatternTransformer":
        if self.spec is None:
            raise ValueError("PatternTransformer needs a PatternSpec")
        self.patterns_ = fit_patterns(X, self.spec, y=y, sample_weight=sample_weight)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scores = self.patterns_.scores(X)
        if self.keep_foods:
            return pd.concat([X, scores], axis=1)
        return pd.concat([X.drop(columns=list(self.spec.foods)), scores], axis=1)  # type: ignore[union-attr]


# ── the methods sentence ──────────────────────────────────────────────────────

_FORM_WORDS = {
    "residual": "each adjusted for total energy by the residual method and standardized",
    "density": "each expressed per 1,000 kcal and standardized",
    "standardized": "each standardized to a mean of 0 and a standard deviation of 1",
}


def _count_words(s: Mapping[str, Any]) -> str:
    m, what = s["count"], ("component" if s["method"] == "pca" else "factor")
    plural = f"{m} {what}{'s' if m != 1 else ''}"
    rule = s.get("count_rule")
    if rule == "parallel_analysis":
        pct = f"{100 * float(s.get('quantile', 0.95)):g}th percentile"
        return (f"{plural} {'were' if m != 1 else 'was'} retained by parallel analysis (Horn "
                f"1965), keeping those whose eigenvalue exceeded the {pct} of "
                f"{int(s.get('iterations', 1000)):,} random data sets of the same size")
    if rule == "scree":
        return (f"{plural} {'were' if m != 1 else 'was'} retained by the scree rule, read from "
                f"the plot of eigenvalues (Cattell 1966)")
    return (f"{plural} {'were' if m != 1 else 'was'} retained as those with an eigenvalue above 1 "
            f"(Kaiser 1960)")


def clause(summary: Mapping[str, Any] | Patterns, purpose: str = "inference") -> str:
    """The method's clause, lowercase and without a final period, from a fit or its summary."""
    s = summary.summary() if isinstance(summary, Patterns) else dict(summary)
    p = len(s["foods"])
    form = _FORM_WORDS[s["inputs"]]
    weighted = ", weighted by the survey weights" if s.get("weighted") else ""
    fold = ("; the patterns were derived within each training fold and applied to the "
            "held-out fold" if purpose == "prediction" else "")
    method = s["method"]
    if method in ("pca", "factor_analysis"):
        how = ("principal component analysis" if method == "pca" else
               "exploratory factor analysis (principal axis factoring with iterated "
               "communalities)")
        rotated = "rotated " if s["count"] > 1 else ""
        scores = (f"each person's pattern scores were computed from the {rotated}components" if
                  method == "pca" else "each person's factor scores were computed by the "
                  "regression method")
        rotate = (", rotated by varimax (Kaiser 1958)" if s["count"] > 1 else "")
        return (f"dietary patterns were derived by {how} of the correlation matrix of {p} food "
                f"groups ({form}){weighted}; {_count_words(s)}{rotate}, and {scores}{fold}")
    if method == "cluster_analysis":
        k = s["count"]
        if s.get("cluster_rule") == "silhouette":
            lo, hi = s["k_range"]
            rule = (f"the number of clusters, {k}, had the largest average silhouette width "
                    f"among {lo} to {hi} (Rousseeuw 1987)")
        else:
            rule = f"the number of clusters, {k}, was declared"
        return (f"participants were grouped by k-means cluster analysis of {p} food groups "
                f"({form}){weighted}; {rule}{fold}")
    q = len(s["responses"])
    share = 100 * float(sum(s["response_explained"]))
    m = s["count"]
    return (f"dietary patterns were derived by reduced rank regression (Hoffmann et al. 2004) of "
            f"{p} food groups ({form}) on {q} intermediate response{'s' if q != 1 else ''} "
            f"({', '.join(s['responses'])}){weighted}; {m} factor{'s' if m != 1 else ''} "
            f"{'were' if m != 1 else 'was'} retained, explaining {share:.1f}% of the responses' "
            f"variation{fold}")


def methods_sentence(summary: Mapping[str, Any] | Patterns, purpose: str = "inference") -> str:
    """The dietary-pattern methods sentence: the method, the food groups' form, the weights, the
    retention rule and, under Predict, that the patterns were derived in each training fold."""
    text = clause(summary, purpose)
    return text[:1].upper() + text[1:] + "."


def _clause(run: Mapping[str, Any]) -> str | None:
    s = run.get(CONTRACT)
    if not s:
        return None
    return clause(s, str(run.get("purpose", "inference")))


# ── the method contracts ──────────────────────────────────────────────────────

_HERE = "turbotab.core.methods.dietary_patterns"
_SCOPE = ("Fitted on the training rows only under prediction and applied to the held-out rows "
          "with their means, standard deviations, energy coefficients, loadings and centers; "
          "under inference, on every analyzed row. The outcome never enters (a reduced rank "
          "regression response that is the outcome is refused).")


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS as REGISTRY, ContractOption, MethodContract,
                                         Relation, register_contract)

    if CONTRACT in REGISTRY:
        return
    both = ("prediction", "inference")

    def option(key: str, label: str, customary: str, prediction: str, inference: str,
               rung: tuple[str, str]) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"prediction": prediction, "inference": inference},
                              {"prediction": rung[0], "inference": rung[1]})  # type: ignore[dict-item]

    register_contract(MethodContract(
        key=CONTRACT, label="Dietary patterns: foods eaten together",
        slot="in_fold", scope="training_fold", scope_note=_SCOPE, run_order=0.41,
        needs=("two or more food-group intake columns",
               "intermediate responses on the pathway, for reduced rank regression",
               "the survey weights, when the rows are a weighted sample"),
        question="How should eating patterns be found in the food groups?",
        options=(
            option("pca", "Patterns of foods eaten together (principal components)",
                   f"The most used method for empirical dietary patterns ({HU_2002}; "
                   f"{NEWBY_TUCKER})",
                   "Sound: derived from the food groups alone, in each training fold, and scored "
                   "on the held-out rows with the fold's loadings",
                   "Sound for describing how foods go together; the patterns are derived "
                   "without the outcome, so their association with it is not chosen to be large",
                   ("recommended", "recommended")),
            option("factor_analysis",
                   "Patterns from shared variation only (exploratory factor analysis)",
                   f"Common beside principal components, often reported as factor analysis "
                   f"({NEWBY_TUCKER}); {FABRIGAR} on factors versus components",
                   "Sound: derived from the food groups alone, in each training fold",
                   "Sound when the patterns are taken as underlying habits that the food groups "
                   "measure with error; it models each food group's own variation apart",
                   ("available", "available")),
            option("cluster_analysis", "Groups of people who eat alike (k-means clusters)",
                   f"Used to place each person in one eating group ({NEWBY_2003}; "
                   f"{NEWBY_TUCKER})",
                   "Sound: the centers are learned in each training fold and held-out rows join "
                   "the nearest one",
                   "Sound for describing groups of people; membership is all-or-nothing, so it "
                   "carries less information than pattern scores",
                   ("available", "available")),
            option("reduced_rank_regression",
                   "Patterns that explain intermediate responses (reduced rank regression)",
                   f"{HOFFMANN}: food-group combinations that explain the most variation in "
                   f"responses on the pathway (biomarkers, nutrients)",
                   "Sound with intermediate responses measured in the training rows; held-out "
                   "rows need only their food groups",
                   "Sound with intermediate responses on the pathway, never the outcome being "
                   "studied, which is refused",
                   ("available", "available")),
        ),
        storyboard=("Read the food groups (and the responses, for reduced rank regression)",
                    "Make the food groups comparable as declared, then standardize them",
                    "Compute their correlations, weighted by the survey weights when there are "
                    "any",
                    "Find the patterns: components, factors, clusters or response-explaining "
                    "factors",
                    "Rotate components and factors by varimax and order them by variance",
                    "Score every person, the held-out rows with the training rows' patterns"),
        relations=(
            Relation("conflicts", "the outcome as a response",
                     "Refused: the outcome being studied cannot be a reduced rank regression "
                     "response, so the patterns are never chosen to explain it.", purposes=both,
                     rung="refused", when=("reduced_rank_regression",),
                     exits=("choose intermediate responses on the pathway, such as biomarkers or "
                            "nutrients", "derive the patterns by principal components or factor "
                            "analysis instead"),
                     condition="the outcome, or a column equal to it, among the responses",
                     enforced_by=f"{_HERE}:check_responses", id="outcome_as_response"),
            Relation("implies", "patterns derived in each training fold",
                     "The patterns are derived from the training rows of each fold and applied "
                     "to the held-out rows, never from every row.", purposes=("prediction",),
                     condition="Predict", enforced_by=f"{_HERE}:PatternTransformer",
                     id="in_each_fold"),
            Relation("implies", "survey_population",
                     "The correlations, cluster centers and reduced rank regression are weighted "
                     "by the survey weights, so the patterns are the surveyed population's.",
                     purposes=both, condition="a survey weight column",
                     enforced_by=f"{_HERE}:weighted_correlation", id="weighted_covariance"),
            Relation("implies", "varimax rotation",
                     "Components and factors are rotated by varimax, so each food group loads "
                     "mainly on one pattern.", purposes=both,
                     when=("pca", "factor_analysis"), condition="two or more patterns retained",
                     enforced_by=f"{_HERE}:varimax", id="varimax"),
            Relation("enables", COUNT_CONTRACT,
                     "How many components or factors to keep is asked next.", purposes=both,
                     when=("pca", "factor_analysis")),
            Relation("enables", CLUSTERS_CONTRACT,
                     "How many groups of people to form is asked next.", purposes=both,
                     when=("cluster_analysis",)),
            Relation("implies", "one factor at most per response",
                     "Reduced rank regression gives at most one factor per response; each "
                     "factor's share of the responses' variation is reported.", purposes=both,
                     when=("reduced_rank_regression",),
                     enforced_by=f"{_HERE}:fit_patterns", id="factors_per_response"),
        ),
        sources=(HU_2002, NEWBY_TUCKER, NEWBY_2003, HOFFMANN, KAISER_1958, FABRIGAR,
                 "R stats::prcomp, stats::varimax; psych::fa (fm = \"pa\")"),
        place="Crosswalk D3: engine only; asked by the Describe goal and the stages in D1",
        short="the dietary patterns", clause=_clause,
        sentence=f"{_HERE}:methods_sentence", package="PATTERNS"))

    register_contract(MethodContract(
        key=INPUTS_CONTRACT, label="Dietary patterns: how the food groups are made comparable",
        slot="in_fold", scope="training_fold", scope_note=_SCOPE, run_order=0.40,
        needs=("the food-group intake columns", "total energy, for the energy-adjusted forms"),
        question="Before finding patterns, should the food groups be adjusted for how much "
                 "people eat overall?",
        options=(
            option("residual", "Adjusted for total energy (residual method), then standardized",
                   f"{WILLETT}: the residual method, standard for energy-adjusting intakes in "
                   f"cohort studies",
                   "Sound: the energy regression is fitted in each training fold",
                   "Sound: the patterns describe what people eat, not how much, so the first "
                   "pattern does not just track total intake",
                   ("available", "recommended")),
            option("density", "Per 1,000 kcal, then standardized",
                   f"Common in dietary-pattern reports ({NEWBY_TUCKER})",
                   "Sound: a per-row ratio, then the fold's center and scale",
                   "Sound: the patterns describe the diet's make-up; the ratio keeps some "
                   "dependence on energy", ("available", "available")),
            option("standardized", "Standardized as eaten, no energy adjustment",
                   f"The most common input: intakes standardized so the correlation matrix is "
                   f"used ({HU_2002})",
                   "Sound: the fold's center and scale",
                   "Sound with total energy kept in the outcome model; without it, the first "
                   "pattern often tracks how much people eat", ("available", "available")),
        ),
        storyboard=("Fit each food group on total energy (residual method), or divide by it "
                    "(per 1,000 kcal)",
                    "Center and scale each food group with the training rows' weighted mean and "
                    "standard deviation"),
        relations=(
            Relation("precedes", CONTRACT, "The food groups are made comparable before the "
                     "patterns are found.", purposes=both),
            Relation("implies", "total energy in the outcome model",
                     "With intakes not energy-adjusted, keep total energy as a covariate in the "
                     "outcome model, since the first pattern often tracks how much people eat.",
                     purposes=("inference",), when=("standardized",), id="energy_in_outcome"),
        ),
        sources=(WILLETT, NEWBY_TUCKER, HU_2002),
        place="Crosswalk D3: engine only; asked by the Describe goal and the stages in D1",
        sentence=f"{_HERE}:methods_sentence", package="PATTERNS"))

    register_contract(MethodContract(
        key=COUNT_CONTRACT, label="Dietary patterns: how many to keep",
        slot="in_fold", scope="training_fold", scope_note=_SCOPE, run_order=0.42,
        needs=("the food groups' correlation matrix",
               "the number read from the scree plot, for the scree rule"),
        question="How many patterns should be kept?",
        options=(
            option("parallel_analysis",
                   "Those that stand out from random data (parallel analysis)",
                   f"{HORN}; {GLORFELD} (the 95th percentile); recommended by {ZWICK_VELICER}",
                   "Sound: a rule fixed in advance, rerun in each training fold",
                   "Sound: keeps only patterns larger than random data of the same size would "
                   "give; under survey weights the random data are weighted the same way",
                   ("recommended", "recommended")),
            option("scree", "The number read from the scree plot (the scree rule)",
                   f"{CATTELL}; most dietary-pattern reports combine it with interpretability "
                   f"({NEWBY_TUCKER})",
                   "Sound if the number is fixed before the folds; read per fold it is not "
                   "repeatable",
                   "Sound when the number and the plot are reported; it is a judgment",
                   ("available", "available")),
            option("eigenvalue_over_one", "Every pattern with an eigenvalue above 1",
                   f"{KAISER_1960}: the most used default in software and in reports",
                   f"Keeps too many with many food groups ({ZWICK_VELICER}); rank lower",
                   f"Keeps too many with many food groups ({ZWICK_VELICER}); rank lower",
                   ("rank_lower", "rank_lower")),
        ),
        storyboard=("Compute the eigenvalues of the food groups' correlation matrix",
                    "Simulate random data of the same size (under the same weights) and take "
                    "the 95th percentile of their eigenvalues",
                    "Keep the leading patterns whose eigenvalue is above that"),
        relations=(
            Relation("implies", "survey_population",
                     "Parallel analysis draws its random data under the same survey weights as "
                     "the food groups' correlations.", purposes=both,
                     when=("parallel_analysis",), condition="a survey weight column",
                     enforced_by=f"{_HERE}:parallel_analysis", id="weighted_reference"),
        ),
        sources=(HORN, GLORFELD, CATTELL, KAISER_1960, ZWICK_VELICER,
                 "R psych::fa.parallel"),
        place="Crosswalk D3: engine only; asked by the Describe goal and the stages in D1",
        sentence=f"{_HERE}:methods_sentence", package="PATTERNS"))

    register_contract(MethodContract(
        key=CLUSTERS_CONTRACT, label="Dietary patterns: how many groups of people",
        slot="in_fold", scope="training_fold", scope_note=_SCOPE, run_order=0.42,
        needs=("the standardized food groups", "a range of group numbers, or the declared one"),
        question="How many groups of people who eat alike should be formed?",
        options=(
            option("silhouette", "The number whose groups are most distinct (silhouette width)",
                   f"{ROUSSEEUW}: the average silhouette width",
                   "Sound: a rule fixed in advance, rerun in each training fold",
                   "Sound: chooses the number whose people sit most clearly in their own group; "
                   "under survey weights the widths are weighted", ("recommended", "recommended")),
            option("declared", "A number stated in advance",
                   f"Common: the number chosen for interpretability and group size "
                   f"({NEWBY_2003}; {NEWBY_TUCKER})",
                   "Sound if stated before the folds",
                   "Sound when stated with its reason; it is a judgment",
                   ("available", "available")),
        ),
        storyboard=("Run k-means from many random starts for each number of groups",
                    "Compute each person's silhouette width and average it, weighted",
                    "Keep the number with the largest average width"),
        relations=(
            Relation("implies", "survey_population",
                     "The cluster centers are weighted means and the silhouette widths are "
                     "averaged with the survey weights.", purposes=both,
                     condition="a survey weight column",
                     enforced_by=f"{_HERE}:silhouette_widths", id="weighted_widths"),
        ),
        sources=(ROUSSEEUW, NEWBY_2003, "R stats::kmeans; R cluster::silhouette"),
        place="Crosswalk D3: engine only; asked by the Describe goal and the stages in D1",
        sentence=f"{_HERE}:methods_sentence", package="PATTERNS"))


_register_contracts()


__all__ = [
    "CLUSTERS_CONTRACT", "CLUSTER_RULES", "CONTRACT", "COUNT_CONTRACT", "COUNT_RULES",
    "INPUTS_CONTRACT", "INPUT_FORMS", "METHODS", "Inputs", "PatternRefused", "PatternSpec",
    "PatternTransformer", "Patterns", "check_responses", "clause", "eigen", "fit_inputs",
    "fit_patterns", "kmeans", "methods_sentence", "parallel_analysis", "principal_axis", "retained",
    "silhouette_widths", "varimax", "weighted_correlation", "weighted_covariance", "weighted_mean",
]

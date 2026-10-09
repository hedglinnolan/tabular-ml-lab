"""Subgroups of similar people (cluster analysis), as an engine method core (SIZING D4).

What it does, in the order it runs:

* **The rule is declared first.** :func:`declare` fixes the method, the rule that picks the number
  of subgroups *k*, the range of *k* it searches, the seed and the number of starts, and returns a
  :class:`Plan` whose ``statement`` says so in plain words. :func:`run` takes only a plan, so *k*
  is never chosen by looking at the groups.
* **Three methods**, each with its own rule:

  - ``kmeans``: k-means on standardized numbers (z-scores), *k* by the average silhouette width
    (Rousseeuw 1987) or the gap statistic (Tibshirani, Walther & Hastie 2001, the paper's own
    "smallest k with Gap(k) ≥ Gap(k+1) − s(k+1)" rule, squared Euclidean distances, reference
    data drawn along the principal components);
  - ``gmm``: a Gaussian mixture, *k* and the covariance shape by BIC (Schwarz 1978; Fraley &
    Raftery 2002), the shapes named as mclust names them (VII, VVI, EEE, VVV). EM finds local
    optima: the chosen model and its BIC agree with mclust's best of its randomized starts, but a
    cell far from the chosen one (a larger *k*) may sit a few BIC units either side of mclust's,
    each program stopping in a different optimum;
  - ``pam_gower``: k-medoids (PAM, Kaufman & Rousseeuw 1990, BUILD then SWAP as R's
    ``cluster::pam`` runs them) on the Gower distance (Gower 1971), the one method here that reads
    categories: a category is a match or a mismatch, an ordered one its rank, a number its
    distance over its range. *k* by the average silhouette width.

* **Categories are never one-hot encoded for k-means or a mixture.** A column that is a category
  (text, yes/no, two values, or named by the caller from its settled reading) is refused there
  with two exits: drop it, or use ``pam_gower``.
* **Stability** by bootstrap (Hennig 2007, as R's ``fpc::clusterboot`` computes it): each
  resample's distinct people are clustered again with the same method and *k*; each original
  subgroup's Jaccard similarity with its closest resampled subgroup, restricted to the resampled
  people, is averaged over the resamples, and the resamples where it dissolved (≤ 0.5) or was
  recovered (> 0.75) are counted.
* **Survey weights** are honored where the method has a weighted form and refused with an exit
  where it does not. A survey weight is how many people in the population a participant stands
  for, so every weighted number here is the same when every weight is multiplied by one constant
  (the weights are read relative to their mean). k-means takes the weights in its standardization
  (the weighted mean and SD, as ``survey::svyvar`` estimates them), its objective and its
  silhouette (each person's mean distance to the *others* in their subgroup, weighted: equal
  weights give Rousseeuw's silhouette exactly), and its stability uses the rescaled bootstrap for
  complex surveys (Rao, Wu & Yue 1992): n_h − 1 PSUs drawn within each stratum, the weights
  rescaled, and the Jaccard counted in weights. The gap statistic's structureless reference, the
  mixture's BIC and the medoids have no design-based form: each is refused under weights, with
  the weighted k-means and the sample-only description as the exits.
* **Every exit runs.** Each refusal's exits are built for the plan at hand, so following one
  (:func:`follow_exit`) never walks into another refusal of the same run: an exit that keeps the
  weights never offers a method the weights refuse.
* **Whole-number codes are asked about.** A column of 3 to 10 whole-number values (race coded
  1–5, a Likert score) may be codes or amounts; it is refused until its reading is settled
  (``numeric=`` or ``categorical=``, from the readings ledger), never silently read as amounts.
* **No outcome is used.** The outcome named to :func:`run` is refused as a feature. Under Predict,
  membership as a feature is :class:`SubgroupMembership`, an estimator fitted inside each training
  fold (rule, standardization, centers and all) and applied to the held-out fold.
* **Deterministic.** Every random part draws from the declared seed (the starts from it; the gap
  reference from ``[seed, 1]``; the bootstrap from ``[seed, 2]``), and the subgroups are numbered
  largest first, so the same plan on the same rows gives the same numbers.

The reference tests (``turbotab/core/tests/acceptance/test_subgroups.py``) hold every number to R:
``stats::kmeans``, ``cluster::silhouette``, ``cluster::clusGap`` and ``cluster::maxSE``,
``cluster::daisy`` and ``cluster::pam``, ``mclust::mclustBIC``, and ``fpc::clusterboot``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# ── sources and words ────────────────────────────────────────────────────────

MACQUEEN = "MacQueen 1967, Proc 5th Berkeley Symp 1:281"
HARTIGAN_WONG = "Hartigan & Wong 1979, Appl Stat 28:100"
HARTIGAN = "Hartigan 1975, Clustering Algorithms, Wiley, ch. 4"
ROUSSEEUW = "Rousseeuw 1987, J Comput Appl Math 20:53"
TIBSHIRANI = "Tibshirani, Walther & Hastie 2001, J R Stat Soc B 63:411"
SCHWARZ = "Schwarz 1978, Ann Stat 6:461"
FRALEY_RAFTERY = "Fraley & Raftery 2002, J Am Stat Assoc 97:611"
SCRUCCA = "Scrucca et al. 2016, R J 8:289 (mclust 5)"
KAUFMAN_ROUSSEEUW = "Kaufman & Rousseeuw 1990, Finding Groups in Data, Wiley, ch. 2"
GOWER = "Gower 1971, Biometrics 27:857"
HENNIG = "Hennig 2007, Comput Stat Data Anal 52:258"
HUANG = "Huang 1998, Data Min Knowl Discov 2:283"
NEWBY_TUCKER = "Newby & Tucker 2004, Nutr Rev 62:177"
RAO_WU_YUE = "Rao, Wu & Yue 1992, Surv Methodol 18:209"
LUMLEY = "Lumley 2004, J Stat Softw 9(8):1"

METHODS: tuple[str, ...] = ("kmeans", "gmm", "pam_gower")
RULES: dict[str, tuple[str, ...]] = {"kmeans": ("silhouette", "gap"), "gmm": ("bic",),
                                     "pam_gower": ("silhouette",)}
# sklearn's covariance types and the mclust model each one is
SHAPES: dict[str, str] = {"spherical": "VII", "diag": "VVI", "tied": "EEE", "full": "VVV"}
SHAPE_WORDS: dict[str, str] = {
    "spherical": "round, each its own size", "diag": "stretched along the axes, each its own",
    "tied": "one shared shape and tilt", "full": "each its own shape and tilt"}

METHOD_WORDS: dict[str, str] = {
    "kmeans": "groups around average profiles, on standardized numbers (k-means)",
    "gmm": "overlapping bell-shaped groups (a Gaussian mixture model)",
    "pam_gower": "groups around typical members, for numbers and categories together "
                 "(k-medoids with the Gower distance)",
}
RULE_WORDS: dict[str, str] = {
    "silhouette": "how clearly each person sits in their own group rather than the next "
                  "(the average silhouette width)",
    "gap": "how much tighter the groups are than in data with no groups at all (the gap "
           "statistic)",
    "bic": "fit against complexity (the Bayesian information criterion, BIC)",
}
DISSOLVED = 0.5  # Hennig 2007: a Jaccard of 0.5 or less, the subgroup has dissolved
RECOVERED = 0.75  # above 0.75 it was found again
PAM_MAX_ROWS = 4000  # PAM holds every pair's distance; past this it would not fit in memory

CODE_VALUES = (3, 10)  # whole numbers with this many values may be codes: asked, not assumed


class SubgroupsRefused(ValueError):
    """The plan cannot run on these data as declared; the message says why, ``exits`` the ways
    forward."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]


# ── the declared plan ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Plan:
    """The method, the rule for *k* and every setting that moves a number, fixed before the run."""

    method: str
    rule: str
    k_min: int
    k_max: int
    seed: int = 0
    n_init: int = 25
    n_boot: int = 100
    shapes: tuple[str, ...] = ("spherical", "diag", "tied", "full")
    gap_b: int = 100

    @property
    def ks(self) -> list[int]:
        return list(range(self.k_min, self.k_max + 1))

    @property
    def statement(self) -> str:
        shapes = ""
        if self.method == "gmm":
            shapes = (" and the covariance shape among "
                      + ", ".join(f"{SHAPES[s]} ({SHAPE_WORDS[s]})" for s in self.shapes))
        boot = (f"; stability from {self.n_boot} bootstrap resamples" if self.n_boot else "")
        starts = ("no random starts (k-medoids is deterministic)" if self.method == "pam_gower"
                  else f"{self.n_init} random starts")
        return (f"Subgroups by {METHOD_WORDS[self.method]}; the number of subgroups chosen "
                f"between {self.k_min} and {self.k_max}{shapes} by {RULE_WORDS[self.rule]}, "
                f"declared before the run; seed {self.seed}, {starts}{boot}.")


def declare(method: str, rule: str | None = None, *, k_range: tuple[int, int] | None = None,
            seed: int = 0, n_init: int = 25, n_boot: int = 100,
            shapes: Sequence[str] = ("spherical", "diag", "tied", "full"),
            gap_b: int = 100) -> Plan:
    """Fix the plan before the run: the method, its rule for *k* (each method's first rule when
    none is named), the range searched (the silhouette needs two subgroups or more; the gap and
    BIC may answer one: "no subgroups"), the seed and the resampling counts."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    rule = rule or RULES[method][0]
    if rule not in RULES[method]:
        raise ValueError(f"{method} chooses k by {' or '.join(RULES[method])}, not {rule!r}")
    lo, hi = k_range or ((2, 8) if rule == "silhouette" else (1, 8))
    if rule == "gap":
        lo = 1  # the gap compares each k with the next, from one subgroup up
    if rule == "silhouette" and lo < 2:
        raise ValueError("the silhouette is defined for two subgroups or more")
    if not 1 <= lo <= hi:
        raise ValueError(f"the range of k must be 1 ≤ low ≤ high, not {lo}–{hi}")
    shapes = tuple(shapes)
    if method == "gmm" and (not shapes or any(s not in SHAPES for s in shapes)):
        raise ValueError(f"the mixture's shapes are among {tuple(SHAPES)}")
    if n_init < 1 or n_boot < 0 or gap_b < 1:
        raise ValueError("starts and resamples are counts")
    return Plan(method, rule, int(lo), int(hi), int(seed), int(n_init), int(n_boot),
                shapes if method == "gmm" else (), int(gap_b))


# ── the columns, read honestly ───────────────────────────────────────────────


@dataclass(frozen=True)
class Features:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]  # unordered: a match or a mismatch
    ordinal: tuple[str, ...]  # ordered categories: their rank
    # whole-number columns of 3 to 10 values whose reading is not settled: codes or amounts? (held
    # among the numbers until answered; ``run`` refuses while any is here)
    ambiguous: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.ordinal

    @property
    def non_numeric(self) -> tuple[str, ...]:
        return self.categorical + self.ordinal


def read_features(frame: pd.DataFrame, columns: Sequence[str], *, categorical: Sequence[str] = (),
                  numeric: Sequence[str] = (),
                  outcome: str | Sequence[str] | None = None) -> Features:
    """Sort ``columns`` into numbers, categories and ordered categories, refusing what clustering
    cannot honestly read: the outcome, a blank value, a column with one value.

    A column is a category when it is text, yes/no, a pandas category, has only two values (a
    0/1 code is a category, not an amount), or is named in ``categorical`` (its settled reading);
    an ordered pandas category is ordinal. A column named in ``numeric`` is an amount. A column of
    3 to 10 whole-number values named in neither may be codes (race coded 1–5) or amounts (a
    count): it is listed as ``ambiguous``, and :func:`run` asks before reading it."""
    outcomes = {outcome} if isinstance(outcome, str) else set(outcome or ())
    columns = list(dict.fromkeys(columns))
    if not columns:
        raise SubgroupsRefused("No columns to group people by.")
    leaked = [c for c in columns if c in outcomes]
    if leaked:
        raise SubgroupsRefused(
            f"{_listed(leaked)} is the outcome: subgroups are found without it, so it cannot be "
            "one of the columns that defines them.",
            [{"label": f"Group by the other columns, without {_listed(leaked)}",
              "drop": leaked}])
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"no such columns: {missing}")
    blank = [c for c in columns if frame[c].isna().any()]
    if blank:
        raise SubgroupsRefused(
            f"{_listed(blank)} {'has' if len(blank) == 1 else 'have'} blank values; clustering "
            "needs every value.",
            [{"label": "Fill the blank values first (the missing-values step)", "step": "fill"},
             {"label": f"Group without {_listed(blank)}", "drop": blank}])
    flat = [c for c in columns if frame[c].nunique(dropna=True) < 2]
    if flat:
        one = len(flat) == 1
        raise SubgroupsRefused(
            f"{_listed(flat)} {'is' if one else 'are'} the same for everyone, so "
            f"{'it' if one else 'they'} cannot tell people apart.",
            [{"label": f"Group without {_listed(flat)}", "drop": flat}])
    named, amounts = set(categorical), set(numeric)
    both = sorted(named & amounts)
    if both:
        raise ValueError(f"{both} named both a category and a number")
    numeric, cats, ords, ambiguous = [], [], [], []
    lo, hi = CODE_VALUES
    for c in columns:
        s = frame[c]
        if isinstance(s.dtype, pd.CategoricalDtype) and s.dtype.ordered:
            ords.append(c)
        elif c in amounts:
            numeric.append(c)
        elif (c in named or isinstance(s.dtype, pd.CategoricalDtype)
              or pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s)
              or s.nunique() == 2):
            cats.append(c)
        else:
            numeric.append(c)
            v = s.to_numpy(dtype=float)
            if lo <= s.nunique() <= hi and np.all(v == np.round(v)):
                ambiguous.append(c)
    return Features(tuple(numeric), tuple(cats), tuple(ords), tuple(ambiguous))


def _listed(items: Sequence[str]) -> str:
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _weights(frame: pd.DataFrame, weights: str | Sequence[float] | np.ndarray | None
             ) -> np.ndarray | None:
    if weights is None:
        return None
    w = np.asarray(frame[weights] if isinstance(weights, str) else weights, dtype=float)
    if len(w) != len(frame) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise SubgroupsRefused("Every weight must be a positive number.",
                               [{"label": "Keep the rows with a positive weight",
                                 "rows": "positive_weight"}])
    return w / w.mean()  # a weight counts relative to the others: scale-free from here on


def standardize(X: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray]:
    """z-scores as R's ``scale()`` makes them (the standard deviation with n − 1). With survey
    weights, the weighted mean and the design-based SD ``survey::svyvar`` estimates (Lumley 2004):
    Σw(x − m)² / Σw · n/(n − 1), n the number of rows, so multiplying every weight by one constant
    changes nothing and equal weights give ``scale()``."""
    X = np.asarray(X, dtype=float)
    if w is None:
        mean = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
    else:
        w = np.asarray(w, dtype=float)
        n, W = len(w), w.sum()
        mean = (w[:, None] * X).sum(axis=0) / W
        sd = np.sqrt((w[:, None] * (X - mean) ** 2).sum(axis=0) / W * n / (n - 1))
    return (X - mean) / sd, mean, sd


# ── the Gower distance and PAM, as R's cluster package computes them ────────


@dataclass(frozen=True)
class GowerSpec:
    """What the Gower distance learned from the rows it was fitted on: each number's range, each
    ordered category's levels."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    ordinal: tuple[str, ...]
    low: Mapping[str, float]
    span: Mapping[str, float]
    levels: Mapping[str, tuple[Any, ...]]


def gower_spec(frame: pd.DataFrame, features: Features) -> GowerSpec:
    low, span, levels = {}, {}, {}
    for c in features.numeric:
        v = frame[c].to_numpy(dtype=float)
        low[c], span[c] = float(v.min()), float(v.max() - v.min())
    for c in features.ordinal:
        levels[c] = tuple(frame[c].cat.categories)
        r = frame[c].cat.codes.to_numpy(dtype=float)
        low[c], span[c] = float(r.min()), float(r.max() - r.min())
    return GowerSpec(features.numeric, features.categorical, features.ordinal, low, span, levels)


def _gower_parts(frame: pd.DataFrame, spec: GowerSpec) -> tuple[np.ndarray, np.ndarray]:
    """Each row's scaled numbers and ranks (as columns), and its categories as codes."""
    scaled = []
    for c in spec.numeric:
        v = frame[c].to_numpy(dtype=float)
        scaled.append(v / spec.span[c] if spec.span[c] > 0 else v * 0.0)
    for c in spec.ordinal:
        r = pd.Categorical(frame[c], categories=spec.levels[c], ordered=True).codes.astype(float)
        scaled.append(r / spec.span[c] if spec.span[c] > 0 else r * 0.0)
    cats = [frame[c].astype(str).to_numpy() for c in spec.categorical]
    return (np.column_stack(scaled) if scaled else np.empty((len(frame), 0)),
            np.column_stack(cats) if cats else np.empty((len(frame), 0), dtype=object))


def gower(a: pd.DataFrame, spec: GowerSpec, b: pd.DataFrame | None = None) -> np.ndarray:
    """The Gower dissimilarity (``cluster::daisy(metric = "gower")`` with no blank values): the
    mean over columns of |difference| / range for numbers and ranks, and 0 or 1 for a category
    that matches or not. ``b`` absent: every pair of ``a``'s rows."""
    na, ca = _gower_parts(a, spec)
    nb, cb = (na, ca) if b is None else _gower_parts(b, spec)
    p = na.shape[1] + ca.shape[1]
    d = np.zeros((len(na), len(nb)))
    for j in range(na.shape[1]):
        d += np.abs(na[:, j][:, None] - nb[:, j][None, :])
    for j in range(ca.shape[1]):
        d += (ca[:, j][:, None] != cb[:, j][None, :])
    return d / p


def pam(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Partitioning around medoids on the dissimilarities ``D``, exactly as ``cluster::pam`` runs
    its original algorithm (``pamonce = 0``): BUILD (each new medoid the last of the largest
    gains), then SWAP (candidates in row order, the medoid each would replace in row order, the
    first largest decrease), until no swap lowers the total. Returns the medoids' rows in row
    order, each row's medoid index into them (the first nearest), and the mean dissimilarity to
    the medoid."""
    D = np.asarray(D, dtype=float)
    n = len(D)
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and the number of rows")
    big = 1.1 * D.max() + 1.0
    nearest = np.full(n, big)
    chosen = np.zeros(n, dtype=bool)
    for _ in range(k):
        gain = np.maximum(nearest[None, :] - D, 0.0).sum(axis=1)
        gain[chosen] = -np.inf
        best = n - 1 - int(np.argmax(gain[::-1]))  # the last of the largest, as `ammax <= beter`
        chosen[best] = True
        nearest = np.minimum(nearest, D[best])
    tol = 16 * np.finfo(float).eps
    while True:
        med = np.flatnonzero(chosen)
        dm = D[med]  # medoid × row
        order = np.argsort(dm, axis=0, kind="stable")
        da = dm[order[0], np.arange(n)]
        db = dm[order[1], np.arange(n)] if len(med) > 1 else np.full(n, big)
        sky = da.sum()
        cand = np.flatnonzero(~chosen)
        if not cand.size:
            break
        # T[h, i]: the change in the total if candidate h replaces medoid i (Kaufman & Rousseeuw
        # p. 104), with R's test `dys(i, j) == dysma[j]` for "j's nearest is i"
        Dc = D[cand]  # candidate × row
        T = np.empty((len(cand), len(med)))
        for m, i in enumerate(med):
            own = D[i] == da
            T[:, m] = np.where(own[None, :], np.minimum(db[None, :], Dc) - da[None, :],
                               np.minimum(Dc - da[None, :], 0.0)).sum(axis=1)
        flat = int(np.argmin(T))  # first in (candidate, medoid) row order, as `dzsky > dz`
        h, m = divmod(flat, len(med))
        if not T[h, m] < -tol * abs(sky):
            break
        chosen[med[m]] = False
        chosen[cand[h]] = True
    med = np.flatnonzero(chosen)
    labels = np.argmin(D[med], axis=0)
    labels[med] = np.arange(len(med))
    cost = float(D[med[labels], np.arange(n)].mean())
    return med, labels, cost


# ── k-means and the mixture ──────────────────────────────────────────────────


def _kmeans(Z: np.ndarray, k: int, w: np.ndarray | None, seed: int, n_init: int
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Labels, centers and the within-subgroup sum of squares (weighted when ``w`` is): the best
    of ``n_init`` starts, each k-means++ seeded from the declared seed, run by Lloyd's algorithm
    and then by Hartigan's single-person transfers (:func:`hartigan`), which reach the lower
    optima R's default Hartigan–Wong algorithm finds."""
    ww = np.ones(len(Z)) if w is None else np.asarray(w, dtype=float)
    if k == 1:
        c = (ww[:, None] * Z).sum(axis=0) / ww.sum()
        return np.zeros(len(Z), dtype=int), c[None, :], float((ww * ((Z - c) ** 2).sum(1)).sum())
    from sklearn.cluster import KMeans

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for start in np.random.SeedSequence(seed).generate_state(n_init):
        km = KMeans(n_clusters=k, n_init=1, random_state=int(start), algorithm="lloyd",
                    max_iter=1000, tol=0.0).fit(Z, sample_weight=w)
        labels, centers = hartigan(Z, km.labels_.astype(int), k, ww)
        wss = float((ww * ((Z - centers[labels]) ** 2).sum(1)).sum())
        if best is None or wss < best[2] * (1 - 1e-12):
            best = (labels, centers, wss)
    assert best is not None
    return best


def hartigan(Z: np.ndarray, labels: np.ndarray, k: int, w: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Move one person at a time to the subgroup that lowers the within sum of squares
    (Hartigan 1975, ch. 4): from A to B when W_B/(W_B + w)·‖x − c_B‖² < W_A/(W_A − w)·‖x − c_A‖²,
    the centers updated after each move, until no move helps. Every Hartigan optimum is a Lloyd
    optimum (each person is nearest its own center), and many Lloyd optima are not."""
    labels = labels.copy()
    W = np.bincount(labels, weights=w, minlength=k).astype(float)
    centers = np.zeros((k, Z.shape[1]))
    np.add.at(centers, labels, w[:, None] * Z)
    centers /= W[:, None]
    for _ in range(1000):
        d = ((Z[:, None, :] - centers[None]) ** 2).sum(-1)
        own = W[labels]
        here = d[np.arange(len(Z)), labels]
        stay = np.where(own > w, own / np.maximum(own - w, 1e-300) * here, -np.inf)
        move = W[None, :] / (W[None, :] + w[:, None]) * d
        move[np.arange(len(Z)), labels] = np.inf
        maybe = np.flatnonzero(move.min(axis=1) < stay * (1 - 1e-12))
        moved = False
        for i in maybe:
            a = labels[i]
            if W[a] <= w[i]:
                continue
            di = ((Z[i] - centers) ** 2).sum(-1)
            gain = W / (W + w[i]) * di
            gain[a] = np.inf
            b = int(np.argmin(gain))
            if gain[b] < W[a] / (W[a] - w[i]) * di[a] * (1 - 1e-12):
                centers[a] = (W[a] * centers[a] - w[i] * Z[i]) / (W[a] - w[i])
                centers[b] = (W[b] * centers[b] + w[i] * Z[i]) / (W[b] + w[i])
                W[a] -= w[i]
                W[b] += w[i]
                labels[i] = b
                moved = True
        if not moved:
            break
    W = np.bincount(labels, weights=w, minlength=k).astype(float)
    centers = np.vstack([(w[labels == g, None] * Z[labels == g]).sum(0) / W[g]
                         for g in range(k)])
    return labels, centers


def _gmm(Z: np.ndarray, k: int, shape: str, seed: int, n_init: int):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(n_components=k, covariance_type=shape, n_init=n_init,
                           random_state=seed, max_iter=2000, tol=1e-8, reg_covar=1e-6).fit(Z)


def gmm_parameters(k: int, d: int, shape: str) -> int:
    """Free parameters, as mclust counts them: mixing proportions, means, covariances."""
    cov = {"spherical": k, "diag": k * d, "tied": d * (d + 1) // 2,
           "full": k * d * (d + 1) // 2}[shape]
    return (k - 1) + k * d + cov


# ── the rules ────────────────────────────────────────────────────────────────


def silhouette(labels: np.ndarray, *, X: np.ndarray | None = None, D: np.ndarray | None = None,
               w: np.ndarray | None = None, chunk: int = 2048) -> np.ndarray:
    """Each row's silhouette width (Rousseeuw 1987; ``cluster::silhouette``) on the Euclidean
    distance of ``X`` or the dissimilarities ``D``: s = (b − a) / max(a, b), with a the mean
    distance to the others in its own subgroup and b the smallest mean distance to another; 0 in
    a subgroup of one. With survey weights each mean is weighted: a is the weighted mean distance
    to the *others* in the subgroup, Σ_{j≠i} w_j d_ij / Σ_{j≠i} w_j, and b the weighted mean
    distance to another subgroup's members, so multiplying every weight by one constant changes
    nothing and equal weights give Rousseeuw's widths exactly. (WeightedCluster's ``ASW`` divides
    by W − 1, which reads the weights as counts of repeated rows and changes with their scale; its
    ``ASWw`` divides by W, counting the person at distance 0 from themself, which gives a lone
    person a width of 1 (Studer 2013, LIVES Working Paper 24, R WeightedCluster).)"""
    labels = np.asarray(labels)
    n = len(labels)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    groups = np.unique(labels)
    member = labels[:, None] == groups[None, :]  # row × subgroup
    W = (w[:, None] * member).sum(axis=0)
    N = member.sum(axis=0)  # people, to tell a subgroup of one
    out = np.empty(n)
    for start in range(0, n, chunk):
        rows = slice(start, min(n, start + chunk))
        if D is not None:
            d = np.asarray(D[rows], dtype=float)
        else:
            from scipy.spatial.distance import cdist

            d = cdist(X[rows], X)
        S = d @ (w[:, None] * member)  # row × subgroup: weighted distance sums
        own = member[rows]
        g = np.argmax(own, axis=1)
        alone = N[g] < 2
        others = np.where(alone, 1.0, W[g] - w[rows])  # the weight of the others in the subgroup
        a = np.where(alone, 0.0, S[own] / others)
        mean_other = np.where(own, np.inf, S / W[None, :])
        b = mean_other.min(axis=1)
        top = np.maximum(a, b)
        s = np.where(~alone & (top > 0), (b - a) / np.where(top > 0, top, 1), 0.0)
        out[rows] = s
    return out


def tibs_se_max(gap: Sequence[float], se: Sequence[float]) -> int:
    """Tibshirani et al.'s rule (2001, §3; ``cluster::maxSE(method = "Tibs2001SEmax")``): the
    smallest k with Gap(k) ≥ Gap(k+1) − s(k+1); the largest k searched when none qualifies.
    Returns k, counting from 1."""
    gap, se = np.asarray(gap, dtype=float), np.asarray(se, dtype=float)
    K = len(gap)
    ok = gap[:-1] >= (gap - se)[1:]
    return int(np.argmax(ok)) + 1 if ok.any() else K


def _within_ss(Z: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for g in np.unique(labels):
        part = Z[labels == g]
        total += float(((part - part.mean(axis=0)) ** 2).sum())
    return total


def gap_statistic(Z: np.ndarray, k_max: int, cluster: Callable[[np.ndarray, int], np.ndarray],
                  B: int, rng: np.random.Generator) -> pd.DataFrame:
    """The gap statistic (Tibshirani et al. 2001) with squared Euclidean distances: log W_k of the
    data against its mean over ``B`` reference sets drawn uniformly in the box of the data's
    principal components (the paper's reference (b); ``cluster::clusGap(spaceH0 = "scaledPCA",
    d.power = 2)``, whose W is half this one's). ``cluster(X, k)`` returns labels."""
    Z = np.asarray(Z, dtype=float)
    n = len(Z)
    logW = np.array([math.log(_within_ss(Z, cluster(Z, k))) for k in range(1, k_max + 1)])
    mean = Z.mean(axis=0)
    centered = Z - mean
    V = np.linalg.svd(centered, full_matrices=False)[2].T
    rotated = centered @ V
    lo, hi = rotated.min(axis=0), rotated.max(axis=0)
    ref = np.empty((B, k_max))
    for b in range(B):
        z = rng.uniform(lo, hi, size=(n, len(lo))) @ V.T + mean
        ref[b] = [math.log(_within_ss(z, cluster(z, k))) for k in range(1, k_max + 1)]
    e_logw = ref.mean(axis=0)
    sd = ref.std(axis=0, ddof=1) if B > 1 else np.zeros(k_max)
    return pd.DataFrame({"k": np.arange(1, k_max + 1), "log_w": logW, "e_log_w": e_logw,
                         "gap": e_logw - logW, "se": np.sqrt(1 + 1 / B) * sd})


# ── stability (Hennig 2007) ──────────────────────────────────────────────────


def resamples(n: int, B: int, rng: np.random.Generator, *, strata: np.ndarray | None = None,
              psu: np.ndarray | None = None) -> list[np.ndarray]:
    """``B`` bootstrap resamples, each the distinct rows drawn in the order first drawn (as
    ``fpc::clusterboot`` keeps them). Under a survey design whole PSUs are drawn with replacement
    within each stratum, as many as the stratum has."""
    out = []
    if psu is None and strata is None:
        for _ in range(B):
            out.append(pd.unique(rng.integers(0, n, n)))
        return out
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    psu = np.arange(n) if psu is None else np.asarray(psu)
    units: list[list[np.ndarray]] = []
    for h in pd.unique(strata):
        in_h = strata == h
        units.append([np.flatnonzero(in_h & (psu == p)) for p in pd.unique(psu[in_h])])
    for _ in range(B):
        drawn = [stratum[i] for stratum in units
                 for i in rng.integers(0, len(stratum), len(stratum))]
        out.append(pd.unique(np.concatenate(drawn)))
    return out


def replicate_weights(w: np.ndarray, B: int, rng: np.random.Generator, *,
                      strata: np.ndarray | None = None, psu: np.ndarray | None = None
                      ) -> list[tuple[np.ndarray, np.ndarray]]:
    """``B`` replicates of the rescaled bootstrap for complex surveys (Rao, Wu & Yue 1992; R's
    ``survey::as.svrepdesign(type = "subbootstrap")``): in each stratum of n_h PSUs, n_h − 1 are
    drawn with replacement, and a drawn PSU's people get the weight w · n_h / (n_h − 1) · (times
    drawn), so each replicate weight's expectation is the weight itself. A stratum of one PSU keeps
    it, at its weight. With no design, each person is a PSU in one stratum. Each replicate is the
    people drawn (in the order first drawn) and their replicate weights."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    psu = np.arange(n) if psu is None else np.asarray(psu)
    units: list[list[np.ndarray]] = []
    for h in pd.unique(strata):
        in_h = strata == h
        units.append([np.flatnonzero(in_h & (psu == p)) for p in pd.unique(psu[in_h])])
    out = []
    for _ in range(B):
        factor = np.zeros(n)
        order: list[np.ndarray] = []
        for stratum in units:
            n_h = len(stratum)
            if n_h == 1:
                factor[stratum[0]] = 1.0
                order.append(stratum[0])
                continue
            for i in rng.integers(0, n_h, n_h - 1):
                factor[stratum[i]] += n_h / (n_h - 1)
                order.append(stratum[i])
        rows = pd.unique(np.concatenate(order))
        out.append((rows, w[rows] * factor[rows]))
    return out


def bootstrap_jaccard(labels: np.ndarray, refit: Callable[..., np.ndarray],
                      draws: Sequence[np.ndarray], *, w: np.ndarray | None = None,
                      draw_weights: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Subgroup × resample: each original subgroup's Jaccard similarity with the most similar
    subgroup found again on the resample, both restricted to the resampled rows (Hennig 2007;
    ``fpc::clusterboot(bootmethod = "boot")``). ``refit(rows)`` clusters those rows again and
    returns their labels; with ``draw_weights`` (the replicate weights of
    :func:`replicate_weights`) it is called ``refit(rows, weights)``. With survey weights ``w``
    the Jaccard counts weights, not people: |A ∩ B|_w / |A ∪ B|_w."""
    labels = np.asarray(labels)
    groups = np.unique(labels)
    out = np.zeros((len(groups), len(draws)))
    for b, rows in enumerate(draws):
        again = np.asarray(refit(rows) if draw_weights is None else refit(rows, draw_weights[b]))
        wr = np.ones(len(rows)) if w is None else np.asarray(w, dtype=float)[rows]
        mine = (labels[rows][:, None] == groups[None, :]) * wr[:, None]
        theirs = (again[:, None] == np.unique(again)[None, :]) * wr[:, None]
        inter = mine.T @ (theirs > 0)
        union = mine.sum(0)[:, None] + theirs.sum(0)[None, :] - inter
        jac = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
        out[:, b] = jac.max(axis=1) if jac.size else 0.0
    return out


@dataclass(frozen=True)
class Stability:
    mean_jaccard: tuple[float, ...]  # per subgroup, numbered as the result numbers them
    dissolved: tuple[int, ...]  # resamples with a Jaccard of 0.5 or less
    recovered: tuple[int, ...]  # resamples with a Jaccard above 0.75
    n_boot: int
    by_design: bool  # PSUs resampled within strata
    rescaled: bool = False  # survey weights: the rescaled bootstrap, the Jaccard in weights

    def words(self, i: int) -> str:
        j = self.mean_jaccard[i]
        if j <= DISSOLVED:
            return "not stable: it dissolves when the people are resampled"
        if j <= 0.6:
            return "weak: found again in some resamples only"
        if j <= RECOVERED:
            return "a pattern, but its members shift between resamples"
        if j <= 0.85:
            return "stable"
        return "highly stable"


# ── the run ──────────────────────────────────────────────────────────────────


@dataclass
class Subgroups:
    plan: Plan
    features: Features
    k: int
    labels: np.ndarray  # 0 … k−1, the largest subgroup first
    sizes: tuple[int, ...]
    shares: tuple[float, ...]  # the weighted share when weights were used
    table: pd.DataFrame  # the rule's score for every k searched (and shape, for a mixture)
    profiles: pd.DataFrame  # each subgroup's means, and its most common category
    stability: Stability | None
    weighted: bool
    shape: str | None = None
    medoids: tuple[int, ...] = ()  # rows (PAM)
    centers: np.ndarray | None = field(default=None, repr=False)  # standardized (k-means)
    seeds: Mapping[str, Any] = field(default_factory=dict)
    model: Any = field(default=None, repr=False)  # the chosen mixture
    relabel: Mapping[int, int] = field(default_factory=dict)  # subgroup -> the fit's own label

    @property
    def summary(self) -> str:
        sizes = ", ".join(str(s) for s in self.sizes[:-1]) + (
            f" and {self.sizes[-1]}" if self.k > 1 else str(self.sizes[0]))
        if self.k == 1:
            return (f"No subgroups: by the declared rule ({RULE_WORDS[self.plan.rule]}), the "
                    f"{sum(self.sizes)} people are best described as one group.")
        return (f"{self.k} subgroups of {sizes} people, found by {METHOD_WORDS[self.plan.method]}"
                f" and chosen by {RULE_WORDS[self.plan.rule]}.")


def run(frame: pd.DataFrame, columns: Sequence[str], plan: Plan, *,
        categorical: Sequence[str] = (), numeric: Sequence[str] = (),
        outcome: str | Sequence[str] | None = None,
        weights: str | Sequence[float] | np.ndarray | None = None,
        strata: str | None = None, psu: str | None = None) -> Subgroups:
    """Find subgroups of similar people by the declared ``plan``, never reading the outcome."""
    if not isinstance(plan, Plan):
        raise TypeError("run takes a declared plan (subgroups.declare), so k's rule comes first")
    features = read_features(frame, columns, categorical=categorical, numeric=numeric,
                             outcome=outcome)
    w = _weights(frame, weights)
    _refuse(plan, features, w is not None)
    n = len(frame)
    if plan.k_max >= n:
        fewer = min(n - 1, max(2, n // 10))
        raise SubgroupsRefused(f"{n} people cannot be split into up to {plan.k_max} subgroups.",
                               [{"label": "Search fewer subgroups",
                                 "k_range": [plan.k_min, fewer]}] if plan.k_min <= fewer else [])
    if plan.method == "pam_gower" and n > PAM_MAX_ROWS:
        raise SubgroupsRefused(
            f"k-medoids compares every pair of people, and {n} people are more than its "
            f"{PAM_MAX_ROWS} allow here.",
            [_numbers_only(plan, features, "kmeans", "silhouette")] if features.numeric else [])
    seeds = {"starts": plan.seed, "gap_reference": [plan.seed, 1], "bootstrap": [plan.seed, 2]}
    shape = None
    medoids: tuple[int, ...] = ()
    centers = None
    model = None
    if plan.method == "pam_gower":
        spec = gower_spec(frame, features)
        D = gower(frame[list(features.columns)], spec)
        rows = []
        fits = {}
        for k in plan.ks:
            med, lab, cost = pam(D, k)
            fits[k] = (med, lab)
            rows.append({"k": k, "silhouette": float(silhouette(lab, D=D).mean()),
                         "mean_dissimilarity": cost})
        table = pd.DataFrame(rows)
        k = int(table.loc[table["silhouette"].idxmax(), "k"])
        med, labels = fits[k]

        def refit(rows: np.ndarray) -> np.ndarray:
            return pam(D[np.ix_(rows, rows)], k)[1]
    else:
        Z, _, _ = standardize(frame[list(features.numeric)].to_numpy(dtype=float), w)
        if plan.method == "kmeans":
            if plan.rule == "silhouette":
                fits = {k: _kmeans(Z, k, w, plan.seed, plan.n_init) for k in plan.ks}
                table = pd.DataFrame([{"k": k, "silhouette": float(np.average(
                    silhouette(fits[k][0], X=Z, w=w), weights=w)),
                    "within_ss": fits[k][2]} for k in plan.ks])
                k = int(table.loc[table["silhouette"].idxmax(), "k"])
            else:  # unweighted: the gap is refused under weights
                fits = {}

                def cluster(X: np.ndarray, kk: int) -> np.ndarray:
                    if X is Z:  # the data's own fits are kept for the chosen k
                        if kk not in fits:
                            fits[kk] = _kmeans(Z, kk, None, plan.seed, plan.n_init)
                        return fits[kk][0]
                    return _kmeans(X, kk, None, plan.seed, plan.n_init)[0]

                table = gap_statistic(Z, plan.k_max, cluster, plan.gap_b,
                                      np.random.default_rng([plan.seed, 1]))
                k = tibs_se_max(table["gap"], table["se"])
            labels, centers = fits[k][0], fits[k][1]

            def refit(rows: np.ndarray, rw: np.ndarray | None = None) -> np.ndarray:
                # under weights, the replicate's rescaled weights (Rao, Wu & Yue 1992)
                return _kmeans(Z[rows], k, rw, plan.seed, plan.n_init)[0]
        else:
            rows = []
            models = {}
            for s in plan.shapes:
                for k in plan.ks:
                    gm = _gmm(Z, k, s, plan.seed, plan.n_init)
                    ll = float(gm.score(Z) * n)
                    p = gmm_parameters(k, Z.shape[1], s)
                    models[(k, s)] = gm
                    rows.append({"k": k, "shape": s, "mclust": SHAPES[s], "loglik": ll,
                                 "parameters": p, "bic": -2 * ll + p * math.log(n)})
            table = pd.DataFrame(rows)
            best = table.loc[table["bic"].idxmin()]
            k, shape = int(best["k"]), str(best["shape"])
            model = models[(k, shape)]
            labels = model.predict(Z)

            def refit(rows: np.ndarray) -> np.ndarray:
                return _gmm(Z[rows], k, shape, plan.seed, plan.n_init).predict(Z[rows])
    labels, relabel = _largest_first(np.asarray(labels), w)
    if plan.method == "pam_gower":
        medoids = tuple(int(med[relabel[i]]) for i in range(k))
    if centers is not None:
        centers = centers[[relabel[i] for i in range(k)]]
    stability = None
    if plan.n_boot and k > 1:
        rng = np.random.default_rng([plan.seed, 2])
        by_design = strata is not None or psu is not None
        design = {"strata": None if strata is None else frame[strata].to_numpy(),
                  "psu": None if psu is None else frame[psu].to_numpy()}
        if w is None:
            jac = bootstrap_jaccard(labels, refit, resamples(n, plan.n_boot, rng, **design))
        else:
            reps = replicate_weights(w, plan.n_boot, rng, **design)
            jac = bootstrap_jaccard(labels, refit, [r for r, _ in reps], w=w,
                                    draw_weights=[rw for _, rw in reps])
        stability = Stability(tuple(float(x) for x in jac.mean(axis=1)),
                              tuple(int(x) for x in (jac <= DISSOLVED).sum(axis=1)),
                              tuple(int(x) for x in (jac > RECOVERED).sum(axis=1)),
                              plan.n_boot, by_design, w is not None)
    ww = np.ones(n) if w is None else w
    sizes = tuple(int((labels == g).sum()) for g in range(k))
    shares = tuple(float(ww[labels == g].sum() / ww.sum()) for g in range(k))
    return Subgroups(plan, features, k, labels, sizes, shares, table,
                     profiles(frame, features, labels, w), stability, w is not None, shape,
                     medoids, centers, seeds, model, relabel)


def _range_for(plan: Plan, rule: str) -> list[int]:
    """The declared range of k, made valid for ``rule`` (the silhouette needs two subgroups)."""
    lo, hi = plan.k_min, plan.k_max
    if rule == "silhouette":
        lo = max(2, lo)
    elif rule == "gap":
        lo = 1
    return [lo, max(lo, hi)]


def _switch(plan: Plan, method: str, rule: str) -> dict[str, Any]:
    if (method, rule) == (plan.method, plan.rule):
        return {}
    return {"method": method, "rule": rule, "k_range": _range_for(plan, rule)}


def _numbers_only(plan: Plan, features: Features, method: str, rule: str,
                  weighted: bool = False) -> dict[str, Any]:
    cats = list(features.non_numeric)
    label = ("Weighted k-means with the silhouette rule (each person counts as the people their "
             "weight stands for)" if weighted else
             "Group by the numbers only" if (method, rule) == (plan.method, plan.rule) else
             "Group by the numbers only with k-means")
    if weighted and cats:
        label += ", by the numbers only"
    if cats:
        label += f", without {_listed(cats)}"
    out: dict[str, Any] = {"label": label, **_switch(plan, method, rule)}
    if cats:
        out["drop"] = cats
    return out


def _sample_only(plan: Plan, method: str | None = None, rule: str | None = None
                 ) -> dict[str, Any]:
    label = "Describe these participants only (unweighted, sample-only)"
    if method == "pam_gower" and plan.method != "pam_gower":
        label += ", with k-medoids on the Gower distance, which reads categories"
    return {"label": label, "weights": None,
            **(_switch(plan, method, rule) if method and rule else {})}


def _refuse(plan: Plan, features: Features, weighted: bool) -> None:
    """Refuse what the plan cannot honestly run, each refusal with exits built for this plan so
    that following one (:func:`follow_exit`) never walks into another of these refusals: the
    reading of whole-number codes first, then categories, then survey weights."""
    weighted_ok = plan.method == "kmeans" and plan.rule == "silhouette"
    if features.ambiguous:
        amb = list(features.ambiguous)
        one = len(amb) == 1
        exits: list[dict[str, Any]] = [
            {"label": f"Read {_listed(amb)} as amounts (a count or a score)", "numeric": amb}]
        if plan.method == "pam_gower":
            exits.append({"label": f"Read {_listed(amb)} as codes for categories",
                          "categorical": amb})
        elif not weighted:
            exits.append({"label": f"Read {_listed(amb)} as codes for categories, with k-medoids "
                                   "on the Gower distance, which reads categories",
                          "categorical": amb, **_switch(plan, "pam_gower", "silhouette")})
        if len(features.columns) > len(amb):
            exits.append({"label": f"Group without {_listed(amb)}", "drop": amb})
        raise SubgroupsRefused(
            f"{_listed(amb)} {'holds' if one else 'hold'} only a few whole-number values: "
            f"{'it' if one else 'they'} may be codes for categories (race coded 1 to 5), which "
            "would be read as amounts, or amounts (a count). Say which.", exits)
    cats = list(features.non_numeric)
    if plan.method in ("kmeans", "gmm") and cats:
        exits = []
        if features.numeric:
            exits.append(_numbers_only(plan, features, "kmeans", "silhouette", True)
                         if weighted and not weighted_ok else
                         _numbers_only(plan, features, plan.method, plan.rule))
        if weighted:  # k-medoids is refused under weights: only without them
            exits.append(_sample_only(plan, "pam_gower", "silhouette"))
        else:
            exits.append({"label": "Use k-medoids with the Gower distance, which reads "
                                   "categories as matches and mismatches",
                          **_switch(plan, "pam_gower", "silhouette")})
        raise SubgroupsRefused(
            f"{_listed(cats)} {'is a category' if len(cats) == 1 else 'are categories'}: "
            f"{'k-means' if plan.method == 'kmeans' else 'a Gaussian mixture'} measures distance "
            "on numbers, and coding a category as 0/1 columns would make its distances "
            f"arbitrary ({HUANG}).", exits)
    if not weighted or weighted_ok:
        return
    exits = ([_numbers_only(plan, features, "kmeans", "silhouette", True)]
             if features.numeric else [])
    exits.append(_sample_only(plan))
    if plan.method == "gmm":
        raise SubgroupsRefused(
            "A Gaussian mixture's BIC is a likelihood's, and survey weights make no likelihood: "
            "there is no design-based BIC to choose the number of subgroups by.", exits)
    if plan.method == "pam_gower":
        raise SubgroupsRefused(
            "k-medoids has no design-based form here: a medoid is one participant, and weights "
            "would only change which one.", exits)
    raise SubgroupsRefused(
        "The gap statistic compares the data with structureless data drawn in their box; "
        "that reference has no design-based form, so it cannot use survey weights.", exits)


def follow_exit(exit: Mapping[str, Any], plan: Plan, columns: Sequence[str],
                **options: Any) -> tuple[Plan, list[str], dict[str, Any]]:
    """What following a refusal's exit asks: the plan re-declared with the exit's method, rule and
    range (its seed, starts and resamples kept), the columns without those it drops, and
    :func:`run`'s keyword options with its readings added (``numeric``, ``categorical``) and the
    weights and design removed when it describes the sample only. An exit naming a step or rows
    (``step``, ``rows``) is outside this run and is returned unchanged."""
    method = exit.get("method", plan.method)
    rule = exit.get("rule", plan.rule if method == plan.method else None)
    k_range = tuple(exit.get("k_range", (plan.k_min, plan.k_max)))
    new = declare(method, rule, k_range=k_range, seed=plan.seed, n_init=plan.n_init,
                  n_boot=plan.n_boot, shapes=plan.shapes or ("spherical", "diag", "tied",
                                                             "full"), gap_b=plan.gap_b)
    drop = set(exit.get("drop", ()))
    cols = [c for c in columns if c not in drop]
    opts = dict(options)
    for reading in ("numeric", "categorical"):
        if reading in exit:
            opts[reading] = list(opts.get(reading, ())) + list(exit[reading])
    if "weights" in exit and exit["weights"] is None:
        for key in ("weights", "strata", "psu"):
            opts.pop(key, None)
    return new, cols, opts


def _largest_first(labels: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, dict[int, int]]:
    """Number the subgroups by size, largest first (ties: the first to appear); ``relabel`` maps a
    new number to the old one."""
    old = list(pd.unique(labels))
    size = {g: int((labels == g).sum()) for g in old}
    order = sorted(old, key=lambda g: (-size[g], old.index(g)))
    new = {g: i for i, g in enumerate(order)}
    return np.array([new[g] for g in labels]), {i: g for g, i in new.items()}


def profiles(frame: pd.DataFrame, features: Features, labels: np.ndarray,
             w: np.ndarray | None = None) -> pd.DataFrame:
    """Each subgroup's profile: the mean of each number (weighted when weights are given) and,
    beside it, how far that is from everyone's mean in standard deviations; each category's most
    common level and its share."""
    ww = np.ones(len(frame)) if w is None else w
    rows = []
    for g in range(int(labels.max()) + 1):
        inside = labels == g
        row: dict[str, Any] = {"subgroup": g + 1, "people": int(inside.sum())}
        for c in features.numeric:
            v = frame[c].to_numpy(dtype=float)
            z = standardize(v[:, None], w)[0][:, 0]
            row[c] = float(np.average(v[inside], weights=ww[inside]))
            row[f"{c} (SD from all)"] = float(np.average(z[inside], weights=ww[inside]))
        for c in features.non_numeric:
            share = pd.Series(ww[inside]).groupby(frame[c].to_numpy()[inside]).sum()
            share = share / share.sum()
            top = share.sort_values(ascending=False, kind="stable")
            row[c] = top.index[0]
            row[f"{c} (share)"] = float(top.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


# ── the methods sentence ─────────────────────────────────────────────────────


def methods_sentence(result: Subgroups) -> str:
    """The methods sentence: the method, the declared rule and its range, the seed and starts,
    the stability resamples and what they found, and that no outcome was used."""
    p = result.plan
    what = {"kmeans": "k-means clustering on standardized variables",
            "gmm": "Gaussian mixture models",
            "pam_gower": "k-medoids (partitioning around medoids) on the Gower dissimilarity"
            }[p.method]
    cites = {"kmeans": HARTIGAN_WONG, "gmm": FRALEY_RAFTERY,
             "pam_gower": f"{KAUFMAN_ROUSSEEUW}; {GOWER}"}[p.method]
    rule = {"silhouette": f"the largest average silhouette width ({ROUSSEEUW})",
            "gap": f"the gap statistic with the one-standard-error rule ({TIBSHIRANI})",
            "bic": f"the lowest BIC across covariance structures "
                   f"{', '.join(SHAPES[s] for s in p.shapes)} ({SCHWARZ})"}[p.rule]
    weights = (" Survey weights entered the standardization (the design-based standard "
               f"deviation; {LUMLEY}), the k-means objective and the silhouette, each unchanged by "
               "the weights' scale." if result.weighted else "")
    chosen = (f"{result.k} subgroups" if result.k > 1 else "a single group (no subgroups)")
    shape = f", covariance structure {SHAPES[result.shape]}" if result.shape else ""
    starts = ("the algorithm is deterministic" if p.method == "pam_gower"
              else f"seed {p.seed}, {p.n_init} random starts")
    s = (f"Subgroups of participants were identified by {what} ({cites}) on "
         f"{len(result.features.columns)} variables. The number of subgroups was chosen between "
         f"{p.k_min} and {p.k_max} by {rule}, a rule declared before the analysis; it gave "
         f"{chosen}{shape} ({starts}).{weights}")
    if result.stability is not None:
        st = result.stability
        if st.rescaled:
            how = ("the rescaled bootstrap for complex surveys: "
                   + ("primary sampling units resampled within strata, n_h − 1 from a stratum of "
                      "n_h" if st.by_design else "n − 1 participants drawn with replacement")
                   + f", with the weights rescaled ({RAO_WU_YUE}); Jaccard similarities in "
                   "survey weights")
        else:
            how = ("primary sampling units resampled within strata" if st.by_design
                   else "participants resampled with replacement")
        js = ", ".join(f"{j:.2f}" for j in st.mean_jaccard)
        s += (f" Stability was assessed by {st.n_boot} bootstrap resamples ({how}) as each "
              f"subgroup's mean Jaccard similarity with its closest subgroup in the resample "
              f"({HENNIG}): {js}.")
    return s + " No outcome was used to form the subgroups."


def _clause(run_: Mapping[str, Any]) -> str | None:
    result = run_.get("subgroups")
    return methods_sentence(result) if isinstance(result, Subgroups) else None


# ── under Predict: membership as a feature, fitted inside each training fold ──


class SubgroupMembership:
    """Subgroup membership as features (one 0/1 column per subgroup), an estimator in
    scikit-learn's shape: ``fit`` runs the declared plan on the training rows only (the rule, the
    standardization, the centers or medoids; never the outcome, which it ignores), and
    ``transform`` places any rows, the held-out fold's included, with what the training rows
    taught it: the nearest center, the most likely mixture component, or the nearest medoid on
    the training rows' Gower ranges."""

    def __init__(self, method: str = "kmeans", rule: str | None = None,
                 k_range: tuple[int, int] | None = None, seed: int = 0, n_init: int = 25,
                 shapes: Sequence[str] = ("spherical", "diag", "tied", "full"),
                 categorical: Sequence[str] = (), numeric: Sequence[str] = ()):
        self.method = method
        self.rule = rule
        self.k_range = k_range
        self.seed = seed
        self.n_init = n_init
        self.shapes = shapes
        self.categorical = categorical
        self.numeric = numeric

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("method", "rule", "k_range", "seed", "n_init",
                                              "shapes", "categorical", "numeric")}

    def set_params(self, **params: Any) -> "SubgroupMembership":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: pd.DataFrame, y: Any = None,
            sample_weight: np.ndarray | None = None) -> "SubgroupMembership":
        """Fit on the training rows. ``sample_weight`` (the training fold's survey weights, routed
        by the pipeline) enters the standardization, the k-means objective and the silhouette;
        the methods with no design-based form refuse it, as :func:`run` does."""
        X = pd.DataFrame(X)
        plan = declare(self.method, self.rule, k_range=self.k_range, seed=self.seed,
                       n_init=self.n_init, n_boot=0, shapes=self.shapes)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        result = run(X, list(X.columns), plan, categorical=self.categorical,
                     numeric=self.numeric, weights=w)
        self.plan_, self.result_, self.k_ = plan, result, result.k
        f = result.features
        if plan.method == "pam_gower":
            self.spec_ = gower_spec(X, f)
            self.medoid_rows_ = X.iloc[list(result.medoids)][list(f.columns)].copy()
        else:
            raw = X[list(f.numeric)].to_numpy(dtype=float)
            Z, self.mean_, self.sd_ = standardize(raw, w)
            if plan.method == "kmeans":
                self.centers_ = result.centers
            else:
                self.gmm_ = result.model
                self.order_ = [result.relabel[i] for i in range(result.k)]
        self.feature_names_out_ = [f"subgroup_{i + 1}" for i in range(result.k)]
        return self

    def assign(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        f = self.result_.features
        if self.plan_.method == "pam_gower":
            return np.argmin(gower(X[list(f.columns)], self.spec_, self.medoid_rows_), axis=1)
        Z = (X[list(f.numeric)].to_numpy(dtype=float) - self.mean_) / self.sd_
        if self.plan_.method == "kmeans":
            return np.argmin(((Z[:, None, :] - self.centers_[None]) ** 2).sum(-1), axis=1)
        comp = self.gmm_.predict(Z)
        back = {old: new for new, old in enumerate(self.order_)}
        return np.array([back[c] for c in comp])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        lab = self.assign(X)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame((lab[:, None] == np.arange(self.k_)[None, :]).astype(float),
                            columns=self.feature_names_out_, index=index)

    def fit_transform(self, X: pd.DataFrame, y: Any = None,
                      sample_weight: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


# ── the method contract (BLUEPRINT §13), in the one registry ─────────────────

CONTRACT = "subgroups"
_HERE = "turbotab.core.methods.subgroups"
_SCOPE = ("Training rows: it learns the standardization, the number of subgroups and the centers "
          "(or medoids, or mixture) from the rows it groups, never from the outcome (the scope "
          "test holds it there). Under Predict it is refitted inside each training fold and "
          "places the held-out rows with what the fold taught it; described on its own (the "
          "Describe goal, D1) it reads every analyzed row.")


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if CONTRACT in CONTRACTS:
        return
    under_inference = ("Under inference the subgroups are described beside the analysis (sizes, "
                       "profiles, stability); a difference in the outcome between subgroups "
                       "found in the same data is not a confirmatory test")

    def option(key: str, label: str, customary: str, prediction: str, inference: str,
               rung: str, inference_rung: str | None = None) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"prediction": prediction, "inference": inference},
                              {"prediction": rung, "inference": inference_rung or (
                                  "available" if rung == "recommended" else rung)})  # type: ignore[dict-item]

    register_contract(MethodContract(
        key=CONTRACT, label="Subgroups of similar people (cluster analysis)",
        slot="in_fold", scope="training_fold", scope_note=_SCOPE, run_order=0.9,
        package="D4",
        needs=("two or more columns that describe people, none of them the outcome",
               "no blank values (filled first, or the column left out)",
               "a declared rule for the number of subgroups and a seed",
               "the survey weights and design, when the sample is weighted"),
        question="Look for subgroups of similar people, with the number of subgroups chosen by "
                 "a rule declared first?",
        options=(
            option("kmeans_silhouette",
                   "Groups around average profiles (k-means on standardized numbers), the "
                   "number by the average silhouette width",
                   f"The most used method in dietary-pattern cluster analyses ({NEWBY_TUCKER}); "
                   f"{HARTIGAN_WONG}; the silhouette, {ROUSSEEUW}",
                   "Sound as a feature when refitted inside each training fold; numbers only; "
                   "survey weights, passed to each fold's fit as sample weights, enter its "
                   "standardization, objective and silhouette",
                   f"{under_inference}; sound as a description", "recommended"),
            option("kmeans_gap",
                   "Groups around average profiles (k-means), the number by the gap statistic",
                   f"{TIBSHIRANI}; R cluster::clusGap",
                   "Sound inside each training fold; it can answer \"no subgroups\"; no survey "
                   "weights (its structureless reference has no design-based form)",
                   f"{under_inference}; sound as a description of an unweighted sample",
                   "available"),
            option("gmm_bic",
                   "Overlapping bell-shaped groups (a Gaussian mixture), the number and shape "
                   "by BIC",
                   f"{FRALEY_RAFTERY}; {SCRUCCA}; BIC, {SCHWARZ}",
                   "Sound inside each training fold; it can answer \"no subgroups\"; refused "
                   "under survey weights (no design-based BIC)",
                   f"{under_inference}; sound as a description of an unweighted sample",
                   "available"),
            option("pam_gower_silhouette",
                   "Groups around typical members, numbers and categories together (k-medoids "
                   "on the Gower distance), the number by the average silhouette width",
                   f"{KAUFMAN_ROUSSEEUW}; {GOWER}; R cluster::pam and cluster::daisy",
                   "Sound for mixed columns inside each training fold; up to "
                   f"{PAM_MAX_ROWS} rows; refused under survey weights",
                   f"{under_inference}; sound as a description of an unweighted sample",
                   "available"),
            option("kmeans_one_hot",
                   "k-means with categories coded as 0/1 columns",
                   f"Seen when categories are dummy-coded so that k-means will run (the "
                   f"limitation {HUANG} addresses)",
                   f"Unsound: the distances between categories become arbitrary ({HUANG}); "
                   "refused, with k-medoids on the Gower distance or the numbers alone as the "
                   "exits",
                   f"Unsound for the same reason ({HUANG})", "refused", "refused"),
            option("k_after_looking",
                   "The number of subgroups chosen after looking at the groups",
                   f"Common: k is often chosen by how interpretable the groups look, one of "
                   f"the subjective decisions {NEWBY_TUCKER} review",
                   "Unsound: the choice is fitted to the data by eye and cannot be refitted in "
                   "each fold; the rule is declared before the run",
                   "Unsound: a k chosen by eye makes the subgroups unreplicable; the rule is "
                   "declared before the run", "refused", "refused"),
            option("none", "No subgroups", "Most analyses",
                   "Sound: the model reads the columns themselves",
                   "Sound: the analysis does not need subgroups", "available"),
        ),
        storyboard=("Read the columns, sorting numbers from categories, and leave out the "
                    "outcome",
                    "Standardize the numbers (or measure Gower distances, for categories)",
                    "Fit each number of subgroups in the declared range, from the declared seed",
                    "Score each by the declared rule and keep the best",
                    "Number the subgroups largest first and describe each one",
                    "Resample the people and refit, to see which subgroups hold"),
        relations=(
            Relation("conflicts", "categories in k-means or a mixture",
                     "A category is refused for k-means and a mixture, never coded as 0/1 "
                     f"columns silently ({HUANG}).", rung="refused",
                     when=("kmeans_silhouette", "kmeans_gap", "gmm_bic"),
                     exits=("group by the numbers only", "k-medoids on the Gower distance"),
                     condition="a text, yes/no, two-valued or declared-category column",
                     enforced_by=f"{_HERE}:_refuse", id="categories_refused"),
            Relation("conflicts", "whole-number codes read as amounts",
                     "A column of a few whole-number values (race coded 1 to 5) is asked about, "
                     "never read as amounts until its reading is settled.", rung="refused",
                     exits=("read it as amounts", "read it as codes for categories",
                            "group without it"),
                     condition="3 to 10 whole-number values, read neither as codes nor amounts",
                     enforced_by=f"{_HERE}:_refuse", id="codes_asked"),
            Relation("conflicts", "the outcome as a grouping column",
                     "The outcome never defines the subgroups.", rung="refused",
                     exits=("group by the other columns",),
                     condition="the outcome among the columns",
                     enforced_by=f"{_HERE}:read_features", id="no_outcome"),
            Relation("implies", "the rule declared before the run",
                     "The rule for the number of subgroups, its range and the seed are fixed "
                     "before the run and stated in the methods.",
                     condition="any subgroup method", enforced_by=f"{_HERE}:declare",
                     id="rule_declared_first"),
            Relation("implies", "refitted inside each training fold",
                     "Under Predict, the rule, the standardization and the centers are fitted on "
                     "each training fold, and the held-out rows are placed with them.",
                     purposes=("prediction",), condition="membership used as a feature",
                     enforced_by=f"{_HERE}:SubgroupMembership", id="in_fold"),
            Relation("implies", "bootstrap stability",
                     f"Each subgroup's stability is its mean bootstrap Jaccard ({HENNIG}); a "
                     "subgroup that dissolves is said to.",
                     condition="two or more subgroups", enforced_by=f"{_HERE}:bootstrap_jaccard",
                     id="stability"),
            Relation("implies", "survey_population",
                     "With survey weights, k-means weighs each person by their weight in the "
                     "standardization, its objective and the silhouette, none changed by the "
                     "weights' scale, and the stability uses the rescaled bootstrap: PSUs "
                     f"within strata, the weights rescaled ({RAO_WU_YUE}).",
                     when=("kmeans_silhouette",), condition="the surveyed-population answer",
                     enforced_by=f"{_HERE}:run", id="weighted_kmeans"),
            Relation("conflicts", "survey weights without a design-based form",
                     "The gap statistic, the mixture's BIC and k-medoids have no design-based "
                     "form and are refused under survey weights.", rung="refused",
                     when=("kmeans_gap", "gmm_bic", "pam_gower_silhouette"),
                     exits=("weighted k-means with the silhouette rule",
                            "describe these participants only (unweighted)"),
                     condition="the surveyed-population answer",
                     enforced_by=f"{_HERE}:_refuse", id="weights_refused"),
        ),
        sources=(MACQUEEN, HARTIGAN, HARTIGAN_WONG, ROUSSEEUW, TIBSHIRANI, SCHWARZ, FRALEY_RAFTERY,
                 SCRUCCA, KAUFMAN_ROUSSEEUW, GOWER, HENNIG, HUANG, NEWBY_TUCKER, RAO_WU_YUE,
                 LUMLEY),
        place="the Describe goal (D1) and, as a feature, the prediction pipeline's in-fold steps",
        leash={"prediction": "available", "inference": "available"},
        short="subgroup membership", clause=_clause, sentence=f"{_HERE}:methods_sentence"))


_register_contract()


__all__ = ["CONTRACT", "Features", "GowerSpec", "Plan", "Stability", "SubgroupMembership",
           "Subgroups", "SubgroupsRefused", "bootstrap_jaccard", "declare", "follow_exit",
           "gap_statistic", "gmm_parameters", "gower", "gower_spec", "methods_sentence", "pam",
           "profiles", "read_features", "replicate_weights", "resamples", "run", "silhouette",
           "standardize", "tibs_se_max"]

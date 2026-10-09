"""D3 · Dietary patterns, the engine core (V2_DEFINITION_OF_DONE, amendment of 2026-10-07;
crosswalk/SIZING.md row D3).

1. **Principal components**, from the correlation matrix of the food groups in each declared form
   (standardized, per 1,000 kcal, energy residuals), with varimax rotation and unit-variance
   scores: R ``stats::prcomp`` + ``stats::varimax``, and for the residual form R's own ``lm``
   residuals, to 1e-8.
2. **Exploratory factor analysis** by principal axis factoring with iterated communalities:
   R ``psych::fa(fm = "pa")`` (communalities, varimax-rotated loadings, regression scores) to 1e-6,
   and ``psych::fa(rotate = "varimax")`` itself to 5e-3 (psych stops its rotation at
   ``stats::varimax``'s default tolerance; the varimax criterion here is never lower).
3. **The retention rules:** parallel analysis keeps the number ``psych::fa.parallel(fa = "pc",
   quant = .95)`` keeps, with reference eigenvalues within Monte Carlo error of R's own
   simulation (the 95th percentile of 2,000 random correlation matrices); the
   eigenvalue-above-1 count equals R's; the scree rule keeps the stated number and refuses without
   one.
4. **Cluster analysis:** k-means finds R ``stats::kmeans``'s partition and within-cluster sum of
   squares; the average silhouette width equals R ``cluster::silhouette``'s; the silhouette rule
   picks the planted number of groups.
5. **Reduced rank regression** (Hoffmann et al. 2004): a hand computation in R (``lm.wfit`` on the
   standardized responses, ``eigen`` of the fitted responses' covariance) to 1e-8, and the first
   factor's explained response variation is the maximum over every food-group combination
   (``scipy.optimize`` from random starts). The outcome as a response is refused, by name and by
   a column equal to the outcome.
6. **Survey weights:** weighted PCA, factor analysis and reduced rank regression equal R's
   ``cov.wt(method = "unbiased")`` route (and ``psych::fa`` on the weighted correlation matrix);
   weighted k-means equals Lloyd's algorithm by hand in R from the same start; the weighted
   silhouette equals a hand computation in R; integer weights give exactly the patterns of the
   rows repeated that many times.
7. **Under Predict** the patterns are learned from the training rows only: the held-out scores are
   the training rows' energy coefficients, centers, scales and coefficients applied by hand, and the
   lockbox scope test observes ``training_fold`` (never ``model``: the outcome moves nothing).
8. **The contracts:** four, in the one registry, each option labeled customary (with a source)
   and sound for both purposes, eigenvalues above 1 ranked lower than parallel analysis, the
   outcome-as-response conflict refused with its exits, and the methods sentence stating the rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import dietary_patterns as D
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

FOODS = tuple(f"g{i}" for i in range(8))
R_FOODS = "foods <- c(" + ", ".join(f'"{f}"' for f in FOODS) + ")\n"


def diet(n: int = 300, seed: int = 11) -> pd.DataFrame:
    """Eight food groups from two planted patterns, total energy that drives every intake, three
    intermediate responses (two follow the patterns), an outcome, and survey weights."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 2))
    L = np.array([[.8, 0], [.75, 0], [.7, .1], [0, .8], [.1, .75], [0, .7], [.35, .3], [0, .05]])
    energy = rng.normal(2000, 350, n)
    X = f @ L.T + rng.standard_normal((n, 8)) * .55 + 4 + np.outer(energy / 1000,
                                                                     rng.uniform(.5, 1.5, 8))
    df = pd.DataFrame(X, columns=list(FOODS))
    df["energy"] = energy
    df["b1"] = f[:, 0] + rng.standard_normal(n) * .8
    df["b2"] = .6 * f[:, 1] + rng.standard_normal(n)
    df["b3"] = .3 * f[:, 0] - .4 * f[:, 1] + rng.standard_normal(n)
    df["y"] = .5 * f[:, 0] + rng.standard_normal(n)
    df["w"] = rng.uniform(.4, 3.0, n)
    df["wi"] = rng.integers(1, 4, n).astype(float)
    return df


def groups(n_each: int = 60, seed: int = 5) -> pd.DataFrame:
    """Three well-separated groups of eaters in five food groups."""
    rng = np.random.default_rng(seed)
    means = np.array([[3, 0, 0, 1, 0], [0, 3, 0, -1, 1], [0, 0, 3, 0, -1]], dtype=float)
    X = np.vstack([m + rng.standard_normal((n_each, 5)) * .5 for m in means])
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(5)])
    df["w"] = rng.uniform(.5, 2.5, len(df))
    return df


def align(mine: np.ndarray, theirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The column permutation and signs that take ``theirs`` onto ``mine`` (patterns are defined
    up to order and sign)."""
    mine, theirs = np.atleast_2d(mine), np.atleast_2d(theirs)
    cross = mine.T @ theirs
    order = np.abs(cross).argmax(axis=1)
    assert len(set(order)) == len(order), "the patterns do not pair up one to one"
    signs = np.sign(cross[np.arange(len(order)), order])
    return order, signs


def same(mine: np.ndarray, theirs: np.ndarray, tol: float, *, like: np.ndarray | None = None,
         like_mine: np.ndarray | None = None) -> None:
    """``mine`` equals ``theirs`` up to column order and sign (found on ``like_mine``/``like``)."""
    order, signs = align(mine if like_mine is None else like_mine,
                         theirs if like is None else like)
    np.testing.assert_allclose(mine, np.asarray(theirs)[:, order] * signs, atol=tol, rtol=0)


def varimax_criterion(L: np.ndarray) -> float:
    """Kaiser's (1958) normalized varimax criterion: the summed variance of the squared loadings
    of each pattern, after each food group's row is scaled to unit length."""
    x = L / np.sqrt((L ** 2).sum(axis=1))[:, None]
    return float(((x ** 4).sum(axis=0) - (x ** 2).sum(axis=0) ** 2 / len(x)).sum())


# ── 1. principal components ───────────────────────────────────────────────────

_PCA_R = R_FOODS + """
d <- read.csv(data_csv)
x <- as.matrix(d[, foods])
if (form == "residual") x <- sapply(foods, function(f) resid(lm(d[[f]] ~ d$energy)))
if (form == "density") x <- x / d$energy * 1000
pc <- prcomp(x, center = TRUE, scale. = TRUE)
L <- pc$rotation[, 1:m] %*% diag(pc$sdev[1:m])
v <- GPArotation::Varimax(L, normalize = TRUE, eps = 1e-14, maxit = 100000)
S <- pc$x[, 1:m] %*% diag(1 / pc$sdev[1:m]) %*% v$Th
out(list(ev = pc$sdev^2, L = unclass(v$loadings), S = S))
"""


@needs_r
@pytest.mark.parametrize("form", ["standardized", "density", "residual"])
def test_principal_components_equal_prcomp_and_varimax(form, tmp_path):
    df = diet()
    spec = D.PatternSpec("pca", FOODS, inputs=form, energy="energy", count_rule="scree",
                         n_patterns=2)
    fit = D.fit_patterns(df, spec)
    r = run_r(f'form <- "{form}"; m <- 2\n' + _PCA_R, {"data": df}, tmp_path)
    np.testing.assert_allclose(fit.eigenvalues, r["ev"], atol=1e-10)
    L, S = np.array(r["L"]), np.array(r["S"])
    same(fit.loadings.to_numpy(), L, 1e-8)
    same(fit.scores(df).to_numpy(), S, 1e-8, like=L, like_mine=fit.loadings.to_numpy())
    scores = fit.scores(df).to_numpy()
    np.testing.assert_allclose(np.cov(scores, rowvar=False), np.eye(2), atol=1e-10)
    if form == "residual":  # energy adjusted, the planted patterns: g0–g2 on one, g3–g5 on the other
        top = fit.defining(0.6)
        assert {frozenset(v) for v in top.values()} == {frozenset(FOODS[:3]),
                                                        frozenset(FOODS[3:6])}


# ── 2. exploratory factor analysis ────────────────────────────────────────────

_FA_R = R_FOODS + """
suppressPackageStartupMessages(library(psych))
d <- read.csv(data_csv)
x <- as.matrix(d[, foods])
if (weighted) {
  r <- cov.wt(x, wt = d$w / sum(d$w), cor = TRUE, method = "unbiased")$cor
} else r <- cor(x)
f0 <- fa(r, nfactors = 2, n.obs = nrow(x), fm = "pa", rotate = "none", min.err = 1e-14,
         max.iter = 100000, warnings = FALSE)
v <- GPArotation::Varimax(unclass(f0$loadings), normalize = TRUE, eps = 1e-14,
                          maxit = 100000)
Lr <- unclass(v$loadings)
f1 <- fa(r, nfactors = 2, n.obs = nrow(x), fm = "pa", rotate = "varimax", min.err = 1e-14,
         max.iter = 100000, warnings = FALSE)
out(list(L = Lr, h = unname(f0$communality), W = solve(r, Lr), L1 = unclass(f1$loadings)))
"""


@needs_r
@pytest.mark.parametrize("weighted", [False, True])
def test_factor_analysis_equals_psych_principal_axis(weighted, tmp_path):
    df = diet()
    spec = D.PatternSpec("factor_analysis", FOODS, count_rule="scree", n_patterns=2,
                         weights="w" if weighted else None)
    fit = D.fit_patterns(df, spec)
    r = run_r(f"weighted <- {'TRUE' if weighted else 'FALSE'}\n" + _FA_R, {"data": df}, tmp_path)
    L = np.array(r["L"])
    same(fit.loadings.to_numpy(), L, 1e-6)
    np.testing.assert_allclose(fit.communalities, r["h"], atol=1e-6)
    same(fit.coefficients, np.array(r["W"]), 1e-6, like=L, like_mine=fit.loadings.to_numpy())
    # psych's own rotation stops at stats::varimax's default (eps = 1e-5), short of the maximum:
    # close to it, and never a higher varimax criterion than the rotation run to its end
    same(fit.loadings.to_numpy(), np.array(r["L1"]), 5e-3)
    assert varimax_criterion(fit.loadings.to_numpy()) >= varimax_criterion(np.array(r["L1"]))
    assert fit.sentence().startswith("Dietary patterns were derived by exploratory factor analysis")


# ── 3. how many to keep ───────────────────────────────────────────────────────

_PARALLEL_R = R_FOODS + """
suppressPackageStartupMessages(library(psych))
set.seed(3)
d <- read.csv(data_csv)
x <- as.matrix(d[, foods])
pa <- fa.parallel(x, fa = "pc", n.iter = 2000, quant = .95, plot = FALSE)
sims <- replicate(2000, eigen(cor(matrix(rnorm(nrow(x) * ncol(x)), nrow(x))), symmetric = TRUE,
                                only.values = TRUE)$values)
out(list(ncomp = pa$ncomp, q95 = apply(sims, 1, quantile, .95),
         kaiser = sum(eigen(cor(x))$values > 1)))
"""


@needs_r
def test_parallel_analysis_keeps_what_psych_keeps(tmp_path):
    df = diet()
    r = run_r(_PARALLEL_R, {"data": df}, tmp_path)
    fit = D.fit_patterns(df, D.PatternSpec("pca", FOODS, iterations=2000, seed=4))
    assert fit.count == r["ncomp"] == 2
    np.testing.assert_allclose(fit.reference, r["q95"], rtol=0.02)
    kaiser = D.fit_patterns(df, D.PatternSpec("pca", FOODS, count_rule="eigenvalue_over_one"))
    assert kaiser.count == r["kaiser"]
    with pytest.raises(D.PatternRefused, match="scree rule needs"):
        D.fit_patterns(df, D.PatternSpec("pca", FOODS, count_rule="scree"))
    assert D.fit_patterns(df, D.PatternSpec("pca", FOODS, count_rule="scree",
                                            n_patterns=3)).count == 3


def test_parallel_analysis_keeps_nothing_from_noise_and_says_so():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.standard_normal((500, 8)), columns=list(FOODS))
    with pytest.raises(D.PatternRefused, match="No pattern stands out") as refused:
        D.fit_patterns(noise, D.PatternSpec("pca", FOODS, iterations=300))
    assert refused.value.exits
    # Kaiser's rule keeps several components from the same noise (Zwick & Velicer 1986)
    assert D.fit_patterns(noise, D.PatternSpec("pca", FOODS,
                                               count_rule="eigenvalue_over_one")).count >= 2


# ── 4. cluster analysis ───────────────────────────────────────────────────────

_KMEANS_R = """
suppressPackageStartupMessages(library(cluster))
d <- read.csv(data_csv)
x <- scale(as.matrix(d[, grep("^c", names(d))]))
set.seed(9)
km <- kmeans(x, centers = 3, nstart = 100, iter.max = 1000)
sil <- silhouette(km$cluster, dist(x))
lab <- read.csv(labels_csv)$label
mine <- silhouette(lab, dist(x))
out(list(cluster = km$cluster, tot = km$tot.withinss, sil = mean(sil[, "sil_width"]),
         sil_mine = mean(mine[, "sil_width"])))
"""


@needs_r
def test_kmeans_and_silhouette_equal_r(tmp_path):
    df = groups()
    cols = tuple(c for c in df.columns if c.startswith("c"))
    fit = D.fit_patterns(df, D.PatternSpec("cluster_analysis", cols, k_range=(2, 6)))
    assert fit.count == 3  # the silhouette rule finds the planted groups
    labels = pd.DataFrame({"label": fit.labels})
    r = run_r(_KMEANS_R, {"data": df, "labels": labels}, tmp_path)
    table = pd.crosstab(fit.labels, np.array(r["cluster"]))
    assert ((table > 0).sum(axis=1) == 1).all() and ((table > 0).sum(axis=0) == 1).all()
    Z = fit.inputs.apply(df)
    within = float(((Z - fit.centers[fit.labels - 1]) ** 2).sum())
    assert within == pytest.approx(r["tot"], rel=1e-10)
    assert fit.silhouettes[3] == pytest.approx(r["sil_mine"], abs=1e-10)
    assert fit.silhouettes[3] == pytest.approx(r["sil"], abs=1e-10)
    assert fit.silhouettes[3] == max(fit.silhouettes.values())


_WEIGHTED_KMEANS_R = """
d <- read.csv(data_csv); w <- d$w
x <- as.matrix(d[, grep("^c", names(d))])
cw <- cov.wt(x, wt = w / sum(w), method = "unbiased")
Z <- sweep(sweep(x, 2, cw$center), 2, sqrt(diag(cw$cov)), "/")
C <- as.matrix(read.csv(init_csv)); k <- nrow(C)
for (it in 1:1000) {
  dd <- sapply(1:k, function(j) rowSums(sweep(Z, 2, C[j, ])^2))
  lab <- max.col(-dd, ties.method = "first")
  Cn <- t(sapply(1:k, function(j) colSums(Z[lab == j, , drop = FALSE] * w[lab == j]) /
                 sum(w[lab == j])))
  if (max(abs(Cn - C)) < 1e-15) break
  C <- Cn
}
D <- as.matrix(dist(Z)); s <- numeric(nrow(Z))
for (i in seq_len(nrow(Z))) {
  own <- lab == lab[i]; own[i] <- FALSE
  if (!any(own)) next
  a <- sum(w[own] * D[i, own]) / sum(w[own])
  b <- min(sapply(setdiff(unique(lab), lab[i]),
                  function(j) { o <- lab == j; sum(w[o] * D[i, o]) / sum(w[o]) }))
  s[i] <- (b - a) / max(a, b)
}
out(list(centers = C, label = lab, sil = sum(w * s) / sum(w)))
"""


@needs_r
def test_weighted_kmeans_and_silhouette_equal_a_hand_computation_in_r(tmp_path):
    df = groups()
    cols = tuple(c for c in df.columns if c.startswith("c"))
    spec = D.PatternSpec("cluster_analysis", cols, cluster_rule="declared", k=3, weights="w")
    w = df["w"].to_numpy()
    inputs = D.fit_inputs(df, "standardized", cols, None, w)
    Z = inputs.apply(df)
    init = Z[[0, 70, 130]]  # one row from each planted group, then a start that is not ideal
    init = init + np.array([[.4, -.2, .1, 0, .3]])
    labels, centers, _ = D.kmeans(Z, 3, w, init=init)
    r = run_r(_WEIGHTED_KMEANS_R, {"data": df, "init": pd.DataFrame(init, columns=list(cols))},
              tmp_path)
    Rc = np.array(r["centers"])
    order = [int(np.argmin(((Rc - c) ** 2).sum(axis=1))) for c in centers]
    np.testing.assert_allclose(centers, Rc[order], atol=1e-10)
    width = D.silhouette_widths(Z, [labels], w)[0]
    assert width == pytest.approx(r["sil"], abs=1e-10)
    fit = D.fit_patterns(df, spec)
    np.testing.assert_allclose(np.sort(fit.centers, axis=0), np.sort(centers, axis=0), atol=1e-8)
    assert fit.shares.sum() == pytest.approx(1.0)


# ── 5. reduced rank regression ────────────────────────────────────────────────

_RRR_R = R_FOODS + """
d <- read.csv(data_csv); w <- if (weighted) d$w else rep(1, nrow(d))
resp <- c("b1", "b2", "b3")
wz <- function(M) {
  cw <- cov.wt(M, wt = w / sum(w), method = "unbiased")
  sweep(sweep(M, 2, cw$center), 2, sqrt(diag(cw$cov)), "/")
}
X <- as.matrix(d[, foods])
if (form == "residual") X <- sapply(foods, function(f) resid(lm(d[[f]] ~ d$energy, weights = w)))
Z <- wz(X); Y <- wz(as.matrix(d[, resp]))
fit <- lm.wfit(cbind(1, Z), Y, w)
B <- fit$coefficients[-1, ]
Fh <- Z %*% B
e <- eigen(cov.wt(Fh, wt = w / sum(w), method = "unbiased")$cov, symmetric = TRUE)
A <- B %*% e$vectors %*% diag(1 / sqrt(e$values))
T <- Z %*% A
cr <- cov.wt(cbind(Z, Y, T), wt = w / sum(w), cor = TRUE, method = "unbiased")$cor
p <- length(foods)
out(list(A = A, L = cr[1:p, (p + 4):(p + 6)], Ly = cr[(p + 1):(p + 3), (p + 4):(p + 6)],
         explained = e$values / 3, T = T))
"""


@needs_r
@pytest.mark.parametrize("weighted,form", [(False, "standardized"), (True, "residual")])
def test_reduced_rank_regression_equals_a_hand_computation_in_r(weighted, form, tmp_path):
    df = diet()
    spec = D.PatternSpec("reduced_rank_regression", FOODS, responses=("b1", "b2", "b3"),
                         outcome="y", inputs=form, energy="energy",
                         weights="w" if weighted else None)
    fit = D.fit_patterns(df, spec, y=df["y"])
    r = run_r(f"weighted <- {'TRUE' if weighted else 'FALSE'}; form <- \"{form}\"\n" + _RRR_R,
              {"data": df}, tmp_path)
    L = np.array(r["L"])
    np.testing.assert_allclose(fit.response_explained, r["explained"], atol=1e-10)
    same(fit.loadings.to_numpy(), L, 1e-8)
    same(fit.response_loadings.to_numpy(), np.array(r["Ly"]), 1e-8, like=L,
         like_mine=fit.loadings.to_numpy())
    same(fit.coefficients, np.array(r["A"]), 1e-8, like=L, like_mine=fit.loadings.to_numpy())
    same(fit.scores(df).to_numpy(), np.array(r["T"]), 1e-8, like=L,
         like_mine=fit.loadings.to_numpy())
    # explained response variation is each factor's sum of squared response correlations
    np.testing.assert_allclose((fit.response_loadings ** 2).sum().to_numpy() / 3,
                               fit.response_explained, atol=1e-10)


def test_the_first_reduced_rank_factor_explains_the_most_response_variation():
    from scipy.optimize import minimize

    df = diet()
    resp = ("b1", "b2", "b3")
    fit = D.fit_patterns(df, D.PatternSpec("reduced_rank_regression", FOODS, responses=resp))
    Z = fit.inputs.apply(df)
    Y = df[list(resp)].to_numpy()

    def explained(a: np.ndarray) -> float:
        t = Z @ a
        return float(sum(np.corrcoef(t, Y[:, i])[0, 1] ** 2 for i in range(3)) / 3)

    rng = np.random.default_rng(0)
    best = max(-minimize(lambda a: -explained(a), rng.standard_normal(8), method="BFGS").fun
               for _ in range(8))
    assert best == pytest.approx(fit.response_explained[0], abs=1e-6)
    assert explained(fit.coefficients[:, 0]) == pytest.approx(fit.response_explained[0], abs=1e-10)


def test_the_outcome_is_refused_as_a_response():
    df = diet()
    with pytest.raises(D.PatternRefused, match="outcome being studied, y") as refused:
        D.fit_patterns(df, D.PatternSpec("reduced_rank_regression", FOODS,
                                         responses=("b1", "y"), outcome="y"))
    assert any("intermediate responses" in e for e in refused.value.exits)
    renamed = df.assign(marker=df["y"])  # the outcome under another name
    with pytest.raises(D.PatternRefused, match="holds the outcome"):
        D.PatternTransformer(D.PatternSpec("reduced_rank_regression", FOODS,
                                           responses=("b1", "marker"))).fit(renamed, renamed["y"])
    with pytest.raises(D.PatternRefused, match="both a food group and a response"):
        D.fit_patterns(df, D.PatternSpec("reduced_rank_regression", FOODS, responses=("g0",)))
    with pytest.raises(D.PatternRefused, match="at most 2"):
        D.fit_patterns(df, D.PatternSpec("reduced_rank_regression", FOODS,
                                         responses=("b1", "b2"), n_patterns=3))


# ── 6. survey weights ─────────────────────────────────────────────────────────

_WEIGHTED_PCA_R = R_FOODS + """
d <- read.csv(data_csv); w <- d$w
x <- sapply(foods, function(f) resid(lm(d[[f]] ~ d$energy, weights = w)))
cw <- cov.wt(x, wt = w / sum(w), cor = TRUE, method = "unbiased")
e <- eigen(cw$cor, symmetric = TRUE)
L <- e$vectors[, 1:2] %*% diag(sqrt(e$values[1:2]))
v <- GPArotation::Varimax(L, normalize = TRUE, eps = 1e-14, maxit = 100000)
Z <- sweep(sweep(x, 2, cw$center), 2, sqrt(diag(cw$cov)), "/")
S <- Z %*% e$vectors[, 1:2] %*% diag(1 / sqrt(e$values[1:2])) %*% v$Th
out(list(ev = e$values, L = unclass(v$loadings), S = S))
"""


@needs_r
def test_weighted_components_equal_r_cov_wt(tmp_path):
    df = diet()
    spec = D.PatternSpec("pca", FOODS, inputs="residual", energy="energy", count_rule="scree",
                         n_patterns=2, weights="w")
    fit = D.fit_patterns(df, spec)
    r = run_r(_WEIGHTED_PCA_R, {"data": df}, tmp_path)
    L = np.array(r["L"])
    np.testing.assert_allclose(fit.eigenvalues, r["ev"], atol=1e-10)
    same(fit.loadings.to_numpy(), L, 1e-8)
    same(fit.scores(df).to_numpy(), np.array(r["S"]), 1e-8, like=L,
         like_mine=fit.loadings.to_numpy())
    unweighted = D.fit_patterns(df, D.PatternSpec("pca", FOODS, inputs="residual",
                                                  energy="energy", count_rule="scree",
                                                  n_patterns=2))
    assert not np.allclose(unweighted.loadings.to_numpy(), fit.loadings.to_numpy(), atol=1e-3)
    assert "weighted by the survey weights" in fit.sentence()


@pytest.mark.parametrize("method", ["pca", "factor_analysis", "reduced_rank_regression",
                                    "cluster_analysis"])
def test_integer_weights_give_the_patterns_of_the_repeated_rows(method):
    df = diet() if method != "cluster_analysis" else groups().assign(wi=lambda d: np.random.default_rng(1).integers(1, 4, len(d)).astype(float))
    cols = FOODS if method != "cluster_analysis" else tuple(f"c{i}" for i in range(5))
    kw = dict(responses=("b1", "b2", "b3")) if method == "reduced_rank_regression" else {}
    kw.update(dict(cluster_rule="declared", k=3) if method == "cluster_analysis" else
              dict(count_rule="scree", n_patterns=2))
    weighted = D.fit_patterns(df, D.PatternSpec(method, cols, weights="wi", **kw))
    repeated = df.loc[df.index.repeat(df["wi"].astype(int))].reset_index(drop=True)
    plain = D.fit_patterns(repeated, D.PatternSpec(method, cols, **kw))
    if method == "cluster_analysis":
        np.testing.assert_allclose(np.sort(weighted.profile.to_numpy(), axis=1),
                                   np.sort(plain.profile.to_numpy(), axis=1), atol=1e-10)
    else:
        same(weighted.loadings.to_numpy(), plain.loadings.to_numpy(), 1e-9)
    if method == "reduced_rank_regression":
        np.testing.assert_allclose(weighted.response_explained, plain.response_explained,
                                   atol=1e-10)


def test_weights_that_are_not_weights_are_refused():
    df = diet().assign(w=lambda d: d["w"] * np.where(d.index == 3, -1, 1))
    with pytest.raises(D.PatternRefused, match="survey weights must be") as refused:
        D.fit_patterns(df, D.PatternSpec("pca", FOODS, weights="w"))
    assert refused.value.exits


# ── 7. under Predict, inside each training fold ───────────────────────────────


@pytest.mark.parametrize("method", ["pca", "factor_analysis", "reduced_rank_regression"])
def test_held_out_scores_use_only_the_training_rows(method):
    df = diet()
    train, test = df.iloc[:200], df.iloc[200:]
    kw = dict(responses=("b1", "b2", "b3"), n_patterns=2) if method == "reduced_rank_regression" \
        else dict(count_rule="scree", n_patterns=2)
    spec = D.PatternSpec(method, FOODS, inputs="residual", energy="energy", weights="w", **kw)
    step = D.PatternTransformer(spec).fit(train, train["y"])
    out = step.transform(test)
    assert not set(FOODS) & set(out.columns) and {"pattern_1", "pattern_2"} <= set(out.columns)
    # by hand: the training rows' weighted energy regression, center and scale
    w = train["w"].to_numpy()
    sw = np.sqrt(w)
    design = np.column_stack([np.ones(len(train)), train["energy"]])
    coef = np.linalg.lstsq(design * sw[:, None], train[list(FOODS)].to_numpy() * sw[:, None],
                           rcond=None)[0]
    resid_train = train[list(FOODS)].to_numpy() - design @ coef
    center = np.average(resid_train, axis=0, weights=w)
    var = (w[:, None] * (resid_train - center) ** 2).sum(0) / (w.sum() - (w ** 2).sum() / w.sum())
    held = (test[list(FOODS)].to_numpy() - np.column_stack([np.ones(len(test)), test["energy"]])
            @ coef - center) / np.sqrt(var)
    np.testing.assert_allclose(out[["pattern_1", "pattern_2"]].to_numpy(),
                               held @ step.patterns_.coefficients, atol=1e-10)


@pytest.mark.parametrize("method", ["pca", "factor_analysis", "reduced_rank_regression"])
def test_the_scope_test_observes_training_fold(method):
    from turbotab.core.contracts import observed_scope

    df = diet(n=120)
    kw = dict(responses=("b1", "b2", "b3"), n_patterns=2) if method == "reduced_rank_regression" \
        else dict(count_rule="scree", n_patterns=2)
    spec = D.PatternSpec(method, FOODS, inputs="residual", energy="energy", **kw)
    cols = [*FOODS, "energy", "b1", "b2", "b3"]

    def fit_transform(frame: pd.DataFrame, reference: pd.Series, y: np.ndarray) -> pd.DataFrame:
        step = D.PatternTransformer(spec).fit(frame[cols], y)
        return step.transform(frame[cols])[["pattern_1", "pattern_2"]]

    scope = observed_scope(fit_transform, df[cols], np.zeros(len(df), bool), df["y"].to_numpy(),
                           row=7)
    assert scope == "training_fold"


def test_held_out_people_join_the_nearest_training_cluster():
    df = groups()
    cols = tuple(c for c in df.columns if c.startswith("c"))
    train, test = df.iloc[::2], df.iloc[1::2]
    step = D.PatternTransformer(D.PatternSpec("cluster_analysis", cols, cluster_rule="declared",
                                              k=3)).fit(train)
    fitted = step.patterns_
    Z = (test[list(cols)].to_numpy() - fitted.inputs.center) / fitted.inputs.scale
    nearest = ((Z[:, None, :] - fitted.centers[None]) ** 2).sum(axis=2).argmin(axis=1) + 1
    np.testing.assert_array_equal(step.transform(test)["pattern_cluster"].to_numpy(), nearest)


# ── 8. the contracts ──────────────────────────────────────────────────────────


def test_the_four_contracts_enter_the_one_registry():
    import importlib

    from turbotab.core import contracts as C

    registry = C.contracts()
    keys = {D.CONTRACT, D.INPUTS_CONTRACT, D.COUNT_CONTRACT, D.CLUSTERS_CONTRACT}
    assert {k for k, c in registry.items() if c.package == "PATTERNS"} == keys
    for key in keys:
        c = registry[key]
        assert c.slot == "in_fold" and c.scope == "training_fold", key
        assert c.needs and c.question and c.storyboard and c.options and c.sources, key
        for o in c.options:
            assert o.label and o.customary, (key, o.key)
            for purpose in C.PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] in C.RUNGS, (key, o.key, purpose)
        for r in c.relations:
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
    count = registry[D.COUNT_CONTRACT]
    for purpose in C.PURPOSES:
        ranked = [o["key"] for o in count.options_for(purpose)]
        assert ranked.index("parallel_analysis") < ranked.index("eigenvalue_over_one")
        assert count.option("eigenvalue_over_one").rung[purpose] == "rank_lower"
        assert count.option("parallel_analysis").rung[purpose] == "recommended"
    conflict = registry[D.CONTRACT].relation("outcome_as_response")
    assert conflict.kind == "conflicts" and conflict.rung == "refused" and conflict.exits
    assert conflict.when == ("reduced_rank_regression",)
    assert "Kaiser 1960" in count.option("eigenvalue_over_one").customary
    assert "Horn 1965" in count.option("parallel_analysis").customary
    assert "Hoffmann et al. 2004" in registry[D.CONTRACT].option("reduced_rank_regression").customary
    C.run_order(list(registry))  # the run order agrees with every precedes relation


def test_the_methods_sentence_states_the_method_the_form_and_the_rule():
    df = diet()
    fit = D.fit_patterns(df, D.PatternSpec("pca", FOODS, inputs="residual", energy="energy",
                                           iterations=200))
    said = fit.sentence("prediction")
    assert said.startswith("Dietary patterns were derived by principal component analysis of the "
                           "correlation matrix of 8 food groups (each adjusted for total energy "
                           "by the residual method and standardized)")
    assert "2 components were retained by parallel analysis (Horn 1965)" in said
    assert "rotated by varimax (Kaiser 1958)" in said
    assert "95th percentile of 200 random data sets" in said
    assert said.endswith("derived within each training fold and applied to the held-out fold.")
    assert D.methods_sentence(fit.summary()) == fit.sentence()
    # per 1,000 kcal, energy still drives every food group: one component, so nothing is rotated
    one = D.fit_patterns(df, D.PatternSpec("pca", FOODS, inputs="density", energy="energy",
                                           iterations=200))
    assert one.count == 1
    assert "1 component was retained" in one.sentence() and "varimax" not in one.sentence()
    assert "computed from the components" in one.sentence()
    from turbotab.core.contracts import paragraph

    assert paragraph([D.CONTRACT], {D.CONTRACT: fit.summary()}, "inference").startswith(
        "Dietary patterns were derived by principal component analysis")
    clusters = D.fit_patterns(groups(), D.PatternSpec("cluster_analysis",
                                                      tuple(f"c{i}" for i in range(5))))
    assert "the number of clusters, 3, had the largest average silhouette width among 2 to 8" \
        in clusters.sentence()

"""D4, subgroups of similar people: every number held to an independent reference.

R is the reference (``r_reference.run_r``): ``stats::kmeans`` and ``cluster::silhouette`` for
k-means and its rule; ``survey::svymean`` and ``survey::svyvar`` for the weighted standardization,
the rows repeated by their integer weights for the weighted k-means objective, the weighted
silhouette written out in R, and ``survey::svytotal`` for the rescaled bootstrap's variance;
``cluster::clusGap`` and ``cluster::maxSE`` for the gap statistic; ``cluster::daisy`` and
``cluster::pam`` for the Gower distance and k-medoids; ``mclust::mclustBIC`` for the mixture's BIC;
``fpc::clusterboot`` for the bootstrap Jaccard, fed the very resamples it drew. Hand computations
check the gap rule and the Jaccard on small cases. The rest holds the engine's promises: the rule
declared first, categories and the outcome refused, survey weights honored or refused with an
exit, the same numbers from the same seed, and membership fitted inside each training fold.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import subgroups as S
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r


def _has(*packages: str) -> bool:
    if RSCRIPT is None:
        return False
    code = "cat(all(sapply(c(%s), requireNamespace, quietly = TRUE)))" % ", ".join(
        f'"{p}"' for p in packages)
    done = subprocess.run([RSCRIPT, "--vanilla", "-e", code], capture_output=True, text=True)
    return done.stdout.strip().endswith("TRUE")


needs_mclust = pytest.mark.skipif(not _has("mclust"), reason="R package mclust not installed")
needs_fpc = pytest.mark.skipif(not _has("fpc"), reason="R package fpc not installed")
needs_survey = pytest.mark.skipif(not _has("survey"), reason="R package survey not installed")


def blobs(n_each=(50, 40, 30), d=4, spread=2.2, seed=7) -> pd.DataFrame:
    """Overlapping groups, so the rules have something to weigh."""
    rng = np.random.default_rng(seed)
    parts = []
    for g, m in enumerate(n_each):
        center = rng.normal(0, spread, d)
        parts.append(rng.normal(center, 1.0, (m, d)) * np.linspace(1, 3, d))
    return pd.DataFrame(np.vstack(parts), columns=[f"x{i + 1}" for i in range(d)])


def mixed(n=90, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    return pd.DataFrame({
        "age": np.round(30 + 15 * g + rng.normal(0, 6, n), 1),
        "bmi": np.round(22 + 3 * g + rng.normal(0, 2.5, n), 2),
        "smoke": np.where(rng.random(n) < 0.2 + 0.3 * (g == 2), "yes", "no"),
        "region": np.array(["north", "south", "east"])[(g + (rng.random(n) < 0.25)) % 3],
        "activity": pd.Categorical(np.array(["low", "mid", "high"])[
            np.clip(g + rng.integers(-1, 2, n), 0, 2)], categories=["low", "mid", "high"],
            ordered=True),
    })


def _same_partition(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


# ── k-means and its silhouette, against R ───────────────────────────────────


@needs_r
def test_kmeans_and_silhouette_match_r_for_every_k(tmp_path):
    df = blobs()
    plan = S.declare("kmeans", "silhouette", k_range=(2, 6), n_boot=0, n_init=50)
    res = S.run(df, list(df.columns), plan)
    r = run_r("""
library(cluster)
x <- scale(as.matrix(read.csv(data_csv)))
set.seed(1)
o <- lapply(2:6, function(k) {
  km <- kmeans(x, k, nstart = 200, iter.max = 100)
  list(k = k, wss = km$tot.withinss, sil = mean(silhouette(km$cluster, dist(x))[, 3]),
       cl = km$cluster)
})
out(list(z = x[1:3, ], rows = o))
""", {"data": df}, tmp_path)
    Z, _, _ = S.standardize(df.to_numpy())
    np.testing.assert_allclose(Z[:3], np.asarray(r["z"]), rtol=1e-13)
    for row in r["rows"]:
        ours = res.table.set_index("k").loc[row["k"]]
        assert ours["within_ss"] == pytest.approx(row["wss"], rel=1e-9), row["k"]
        assert ours["silhouette"] == pytest.approx(row["sil"], abs=1e-12), row["k"]
    best = max(r["rows"], key=lambda o: o["sil"])
    assert res.k == best["k"]
    assert _same_partition(res.labels, best["cl"])


@needs_r
@needs_survey
def test_weighted_kmeans_is_design_based_against_r(tmp_path):
    """Survey weights, read as how many people each participant stands for: the standardization is
    ``survey::svymean`` and ``svyvar``'s; the weighted within sum of squares is the one R's
    ``kmeans`` minimizes on the rows repeated by their integer weights (identical rows always share
    a subgroup), over the mean weight; the silhouette is each person's weighted mean distance to
    the others in their subgroup against the nearest other subgroup, written out in R on R's own
    partitions; and both choose the same k."""
    df = blobs((30, 25, 20), d=3, seed=11)
    w = np.random.default_rng(5).integers(1, 4, len(df))
    df["w"] = w
    plan = S.declare("kmeans", "silhouette", k_range=(2, 5), n_boot=0, n_init=50)
    res = S.run(df, ["x1", "x2", "x3"], plan, weights="w")
    r = run_r("""
library(cluster); suppressPackageStartupMessages(library(survey))
d <- read.csv(data_csv)
des <- svydesign(ids = ~1, weights = ~w, data = d)
m <- coef(svymean(~x1 + x2 + x3, des)); v <- diag(as.matrix(svyvar(~x1 + x2 + x3, des)))
z <- sweep(sweep(as.matrix(d[, c("x1", "x2", "x3")]), 2, m), 2, sqrt(v), "/")
D <- as.matrix(dist(z)); w <- d$w
wsil <- function(cl) sapply(seq_along(cl), function(i) {
  others <- cl == cl[i]; others[i] <- FALSE
  if (!any(others)) return(0)
  a <- sum(w[others] * D[i, others]) / sum(w[others])
  b <- min(sapply(setdiff(unique(cl), cl[i]), function(g) {
    s <- cl == g; sum(w[s] * D[i, s]) / sum(w[s]) }))
  (b - a) / max(a, b) })
idx <- rep(seq_len(nrow(d)), w)
set.seed(2)
o <- lapply(2:5, function(k) {
  km <- kmeans(z[idx, ], k, nstart = 200, iter.max = 100)
  cl <- km$cluster[match(seq_len(nrow(d)), idx)]
  list(k = k, wss = km$tot.withinss / mean(w), sil = weighted.mean(wsil(cl), w))
})
out(list(center = unname(m), sd = unname(sqrt(v)), rows = o))
""", {"data": df}, tmp_path)
    _, mean, sd = S.standardize(df[["x1", "x2", "x3"]].to_numpy(), w.astype(float))
    np.testing.assert_allclose(mean, r["center"], rtol=1e-12)
    np.testing.assert_allclose(sd, r["sd"], rtol=1e-12)
    for row in r["rows"]:
        ours = res.table.set_index("k").loc[row["k"]]
        assert ours["within_ss"] == pytest.approx(row["wss"], rel=1e-9), row["k"]
        assert ours["silhouette"] == pytest.approx(row["sil"], abs=1e-12), row["k"]
    assert res.weighted and res.k == max(r["rows"], key=lambda o: o["sil"])["k"]
    assert sum(res.shares) == pytest.approx(1.0)


def test_the_weighted_silhouette_by_hand():
    """Three people on a line at 0, 1 and 4, weights 1, 3 and 2, subgroups {0, 1} and {4}: the
    person at 0 has a = 1 (the one other in the subgroup), b = 4, s = 3/4; the person at 1 has
    a = 1, b = 3, s = 2/3; the person at 4 is alone, s = 0. The weighted mean is
    (1·3/4 + 3·2/3 + 2·0) / 6 = 11/24, whatever the weights are multiplied by."""
    X = np.array([[0.0], [1.0], [4.0]])
    labels = np.array([0, 0, 1])
    for c in (1.0, 0.01, 1e4):
        w = c * np.array([1.0, 3.0, 2.0])
        s = S.silhouette(labels, X=X, w=w)
        np.testing.assert_allclose(s, [3 / 4, 2 / 3, 0.0], rtol=1e-14)
        assert np.average(s, weights=w) == pytest.approx(11 / 24, rel=1e-14)


def test_multiplying_every_weight_by_one_constant_changes_nothing():
    """A survey weight counts people relative to the others: the same weights times 0.3, 0.01,
    1000 or scaled to sum to 1 give the same subgroups, rule scores, shares and stability (weights
    summing to 1 once gave NaN centers and a KeyError), and equal weights of any size give the
    unweighted silhouette exactly."""
    df = blobs((40, 30, 20), d=3, seed=2)
    w = np.random.default_rng(9).integers(1, 5, len(df)).astype(float)
    plan = S.declare("kmeans", "silhouette", k_range=(2, 5), n_boot=8, n_init=5)
    cols = ["x1", "x2", "x3"]
    base = S.run(df.assign(w=w), cols, plan, weights="w")
    for c in (0.3, 0.01, 1000.0, 1 / w.sum()):
        other = S.run(df.assign(w=w * c), cols, plan, weights="w")
        assert other.k == base.k and np.array_equal(other.labels, base.labels), c
        np.testing.assert_allclose(other.table["silhouette"], base.table["silhouette"],
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(other.shares, base.shares, rtol=1e-12)
        np.testing.assert_allclose(other.stability.mean_jaccard, base.stability.mean_jaccard,
                                   rtol=1e-12)
    plain = S.run(df, cols, S.declare("kmeans", "silhouette", k_range=(2, 5), n_boot=0,
                                      n_init=5))
    flat = S.run(df.assign(w=0.01), cols, S.declare("kmeans", "silhouette", k_range=(2, 5),
                                                    n_boot=0, n_init=5), weights="w")
    np.testing.assert_allclose(flat.table["silhouette"], plain.table["silhouette"], atol=1e-13)


# ── the gap statistic, against R ────────────────────────────────────────────


@needs_r
def test_gap_statistic_matches_clusgap(tmp_path):
    """log W is exact (R's W with d.power = 2 is half the within sum of squares); the expected
    log W and the gap are Monte Carlo, held within four standard errors of the difference; the
    one-standard-error rule picks the same k."""
    df = blobs((40, 35, 30), d=3, spread=5.0, seed=21)
    B = 100
    plan = S.declare("kmeans", "gap", k_range=(1, 6), n_boot=0, n_init=10, gap_b=B)
    res = S.run(df, list(df.columns), plan)
    r = run_r(f"""
library(cluster)
x <- scale(as.matrix(read.csv(data_csv)))
set.seed(3)
g <- clusGap(x, function(x, k) kmeans(x, k, nstart = 10, iter.max = 100), K.max = 6,
             B = {B}, d.power = 2, spaceH0 = "scaledPCA", verbose = FALSE)
t <- g$Tab
out(list(logW = t[, "logW"], ElogW = t[, "E.logW"], gap = t[, "gap"], se = t[, "SE.sim"],
         k = maxSE(t[, "gap"], t[, "SE.sim"], method = "Tibs2001SEmax")))
""", {"data": df}, tmp_path)
    t = res.table
    np.testing.assert_allclose(t["log_w"] + np.log(0.5), r["logW"], rtol=1e-9)
    sd_ours = t["se"].to_numpy() / np.sqrt(1 + 1 / B)
    sd_r = np.asarray(r["se"]) / np.sqrt(1 + 1 / B)
    tol = 4 * np.sqrt((sd_ours ** 2 + sd_r ** 2) / B)
    assert np.all(np.abs(t["gap"].to_numpy() - r["gap"]) <= tol)
    assert res.k == r["k"] == 3


@needs_r
def test_the_gap_rule_is_maxse_tibs2001(tmp_path):
    cases = [([0.1, 0.5, 0.52, 0.4], [0.05, 0.05, 0.05, 0.05]),
             ([0.1, 0.3, 0.6, 0.9], [0.01, 0.01, 0.01, 0.01]),
             ([0.5, 0.45, 0.7], [0.1, 0.02, 0.02]),
             ([0.2, 0.25, 0.24, 0.3, 0.29], [0.06, 0.02, 0.02, 0.02, 0.02])]
    frame = pd.DataFrame([{"case": i, "f": f, "s": s} for i, (fs, ss) in enumerate(cases)
                          for f, s in zip(fs, ss)])
    r = run_r("""
library(cluster)
d <- read.csv(data_csv)
out(lapply(split(d, d$case), function(c) maxSE(c$f, c$s, method = "Tibs2001SEmax")))
""", {"data": frame}, tmp_path)
    assert [S.tibs_se_max(f, s) for f, s in cases] == [r[str(i)] for i in range(len(cases))]


def test_the_gap_rule_by_hand():
    # Gap(1) = 0.1 < Gap(2) − s(2) = 0.45; Gap(2) = 0.5 ≥ Gap(3) − s(3) = 0.47: k = 2
    assert S.tibs_se_max([0.1, 0.5, 0.52, 0.4], [0.05] * 4) == 2
    assert S.tibs_se_max([0.1, 0.3, 0.6, 0.9], [0.01] * 4) == 4  # none qualifies: the largest


# ── Gower and k-medoids, against R ──────────────────────────────────────────


@needs_r
def test_gower_and_pam_match_daisy_and_pam(tmp_path):
    df = mixed()
    plan = S.declare("pam_gower", k_range=(2, 6), n_boot=0)
    res = S.run(df, list(df.columns), plan)
    feats = S.read_features(df, list(df.columns))
    assert feats.numeric == ("age", "bmi") and feats.ordinal == ("activity",)
    assert set(feats.categorical) == {"smoke", "region"}
    D = S.gower(df[list(feats.columns)], S.gower_spec(df, feats))
    r = run_r("""
library(cluster)
d <- read.csv(data_csv, stringsAsFactors = TRUE)
d$activity <- factor(d$activity, levels = c("low", "mid", "high"), ordered = TRUE)
d <- d[, c("age", "bmi", "smoke", "region", "activity")]
D <- daisy(d, metric = "gower")
o <- lapply(2:6, function(k) {
  p <- pam(D, k, diss = TRUE)
  list(k = k, medoids = p$id.med, cl = p$clustering, sil = p$silinfo$avg.width,
       obj = unname(p$objective["swap"]))
})
out(list(D = as.matrix(D), rows = o))
""", {"data": df.assign(activity=df["activity"].astype(str))}, tmp_path)
    np.testing.assert_allclose(D, np.asarray(r["D"]), atol=1e-14)
    for row in r["rows"]:
        med, lab, cost = S.pam(D, row["k"])
        assert sorted((med + 1).tolist()) == sorted(np.atleast_1d(row["medoids"]).tolist())
        assert _same_partition(lab, row["cl"]), row["k"]
        assert cost == pytest.approx(row["obj"], rel=1e-12)
        assert res.table.set_index("k").loc[row["k"], "silhouette"] == pytest.approx(
            row["sil"], abs=1e-12)
    assert res.k == max(r["rows"], key=lambda o: o["sil"])["k"]


@needs_r
def test_pam_breaks_ties_as_r_does(tmp_path):
    """Categories alone make many equal distances: BUILD keeps the last of the largest gains
    and SWAP the first largest decrease, as cluster::pam's C code does."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({c: rng.choice(list("abc"), 40) for c in ("p", "q", "r")})
    feats = S.read_features(df, list(df.columns))
    D = S.gower(df, S.gower_spec(df, feats))
    r = run_r("""
library(cluster)
d <- read.csv(data_csv, stringsAsFactors = TRUE)
D <- daisy(d, metric = "gower")
out(lapply(2:5, function(k) { p <- pam(D, k, diss = TRUE); list(m = p$id.med, cl = p$clustering) }))
""", {"data": df}, tmp_path)
    for k, row in zip(range(2, 6), r):
        med, lab, _ = S.pam(D, k)
        assert (med + 1).tolist() == sorted(row["m"])
        assert _same_partition(lab, row["cl"])


# ── the mixture's BIC, against mclust ───────────────────────────────────────


@needs_r
@needs_mclust
def test_mixture_bic_matches_mclust(tmp_path):
    rng = np.random.default_rng(4)
    df = pd.DataFrame(np.vstack([
        rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 120),
        rng.multivariate_normal([4, 1], [[0.5, 0], [0, 2]], 90),
        rng.multivariate_normal([1, 5], [[1.5, -0.4], [-0.4, 0.7]], 70)]), columns=["a", "b"])
    plan = S.declare("gmm", k_range=(1, 4), n_boot=0, n_init=10)
    res = S.run(df, ["a", "b"], plan)
    r = run_r("""
suppressPackageStartupMessages(library(mclust))
x <- scale(as.matrix(read.csv(data_csv)))
# mclust's hierarchical start can stop in a lower optimum; the best of its own randomized
# starts (hcRandomPairs, as its documentation advises) is the maximum likelihood to compare with
fit <- function(init) mclustBIC(x, G = 1:4, modelNames = c("VII", "VVI", "EEE", "VVV"),
  initialization = init, control = emControl(tol = c(1e-10, sqrt(.Machine$double.eps)),
                                             itmax = 10000))
b <- fit(list())
for (s in 1:20) b <- mclustBICupdate(b, fit(list(hcPairs = hcRandomPairs(x, seed = s))))
out(list(bic = lapply(colnames(b), function(m) unname(b[, m])), models = colnames(b),
         best = names(summary(b))[1]))
""", {"data": df}, tmp_path)
    t = res.table.set_index(["mclust", "k"])
    for model, column in zip(r["models"], r["bic"]):
        for k, bic in zip(range(1, 5), column):
            # mclust's BIC is 2 log L − p log n, larger better; ours is its negative
            assert -t.loc[(model, k), "bic"] == pytest.approx(bic, abs=1e-3), (model, k)
    model, k = r["best"].split(",")
    assert (S.SHAPES[res.shape], res.k) == (model, int(k))


def unequal_blobs() -> pd.DataFrame:
    """Three unequal, stretched groups in four columns on scales 0.1 to 10 (n = 240)."""
    rng = np.random.default_rng(20261009)
    X = np.vstack([
        rng.multivariate_normal([0, 0, 0, 0], np.diag([1, 0.5, 1, 1]), 110),
        rng.multivariate_normal([3, 2, -1, 0.5], [[1, .6, 0, 0], [.6, 1, 0, 0], [0, 0, .5, 0],
                                                  [0, 0, 0, 1]], 80),
        rng.multivariate_normal([-2, 4, 2, -1], np.eye(4) * 0.4, 50)]) * [1, 10, 0.1, 3]
    return pd.DataFrame(X, columns=["a", "b", "c", "d"])


@needs_r
@needs_mclust
def test_the_mixture_choice_matches_mclust_where_em_optima_differ(tmp_path):
    """EM stops in local optima, and sklearn's and mclust's stop in different ones once k passes
    the groups the data hold: here cells with k of 4 or more sit up to a couple of dozen BIC
    units either side of mclust's best of 21 starts. What the rule reads holds: the chosen model,
    its k and its BIC are mclust's, and every cell up to the true k agrees."""
    df = unequal_blobs()
    res = S.run(df, list(df.columns), S.declare("gmm", k_range=(1, 6), seed=1, n_init=10,
                                                n_boot=0))
    r = run_r("""
suppressPackageStartupMessages(library(mclust))
x <- scale(as.matrix(read.csv(data_csv)))
fit <- function(init) mclustBIC(x, G = 1:6, modelNames = c("VII", "VVI", "EEE", "VVV"),
  initialization = init)
b <- fit(list())
for (s in 1:20) b <- mclustBICupdate(b, fit(list(hcPairs = hcRandomPairs(x, seed = s))))
out(list(bic = lapply(colnames(b), function(m) unname(b[, m])), models = colnames(b),
         best = names(summary(b))[1]))
""", {"data": df}, tmp_path)
    t = res.table.set_index(["mclust", "k"])
    model, k = r["best"].split(",")
    assert (S.SHAPES[res.shape], res.k) == (model, int(k)) == ("VVI", 3)
    bics = dict(zip(r["models"], r["bic"]))
    assert -t.loc[(model, int(k)), "bic"] == pytest.approx(bics[model][int(k) - 1], abs=1e-3)
    for m, column in bics.items():
        for kk in (1, 2, 3):
            assert -t.loc[(m, kk), "bic"] == pytest.approx(column[kk - 1], abs=0.05), (m, kk)


# ── stability, against fpc::clusterboot ─────────────────────────────────────


@needs_r
@needs_fpc
def test_bootstrap_jaccard_equals_clusterboot_on_its_own_resamples(tmp_path):
    """fpc's clusterboot draws its resamples; a recording clustering method hands them back,
    and the engine's Jaccard on the same resamples equals clusterboot's run by run. The
    clustering is Lloyd's algorithm from the resample's first k rows, the same in R and here."""
    df = blobs((35, 30, 25), d=2, spread=1.6, seed=13)
    X = S.standardize(df.to_numpy())[0]
    k, B = 3, 25
    r = run_r(f"""
suppressPackageStartupMessages(library(fpc))
x <- as.matrix(read.csv(data_csv)); rownames(x) <- seq_len(nrow(x))
seen <- list()
lloyd <- function(data, k) {{
  seen[[length(seen) + 1]] <<- as.integer(rownames(data))
  km <- kmeans(data, centers = data[1:k, , drop = FALSE], algorithm = "Lloyd", iter.max = 500)
  list(result = km, nc = k, partition = km$cluster,
       clusterlist = lapply(1:k, function(i) km$cluster == i), clustermethod = "lloyd")
}}
set.seed(5)
cb <- clusterboot(x, B = {B}, bootmethod = "boot", clustermethod = lloyd, k = {k},
                  count = FALSE, showplots = FALSE)
out(list(seen = seen, boot = cb$bootresult, mean = cb$bootmean, brd = cb$bootbrd,
         rec = cb$bootrecover, part = cb$partition))
""", {"data": pd.DataFrame(X, columns=["a", "b"])}, tmp_path)
    from sklearn.cluster import KMeans

    def lloyd(rows):
        sub = X[rows]
        return KMeans(k, init=sub[:k], n_init=1, algorithm="lloyd", max_iter=500,
                      tol=0.0).fit(sub).labels_

    original = lloyd(np.arange(len(X)))
    assert _same_partition(original, r["part"])
    draws = [np.asarray(s) - 1 for s in r["seen"][1:]]  # the first is the full data
    assert len(draws) == B
    jac = S.bootstrap_jaccard(original, lloyd, draws)
    # R numbers the clusters by Lloyd's labels; map ours to R's by the original partition
    order = [int(np.unique(original[np.asarray(r["part"]) == g + 1])[0]) for g in range(k)]
    groups = np.unique(original)
    rows = [int(np.flatnonzero(groups == o)[0]) for o in order]
    np.testing.assert_allclose(jac[rows], np.asarray(r["boot"]), atol=1e-12)
    assert jac.min() < 0.95  # not a trivial case
    np.testing.assert_allclose(jac[rows].mean(axis=1), r["mean"], atol=1e-12)
    assert ((jac[rows] <= S.DISSOLVED).sum(1)).tolist() == r["brd"]
    assert ((jac[rows] > S.RECOVERED).sum(1)).tolist() == r["rec"]


@needs_r
@needs_fpc
def test_kmeans_stability_agrees_with_clusterboot(tmp_path):
    """With each side drawing its own resamples, the mean Jaccard of each k-means subgroup agrees
    with clusterboot's kmeansCBI within Monte Carlo error."""
    df = blobs((60, 50, 40), d=2, spread=1.8, seed=17)
    B = 200
    res = S.run(df, list(df.columns), S.declare("kmeans", k_range=(3, 3), n_boot=B, n_init=10))
    Z = S.standardize(df.to_numpy())[0]
    r = run_r(f"""
suppressPackageStartupMessages(library(fpc))
x <- as.matrix(read.csv(data_csv))
set.seed(6)
cb <- clusterboot(x, B = {B}, bootmethod = "boot", clustermethod = kmeansCBI, krange = 3,
                  runs = 10, count = FALSE, showplots = FALSE)
out(list(mean = cb$bootmean, part = cb$partition, sd = apply(cb$bootresult, 1, sd)))
""", {"data": pd.DataFrame(Z, columns=["a", "b"])}, tmp_path)
    assert _same_partition(res.labels, r["part"])
    ours = {}
    for g in range(3):
        theirs = int(np.bincount(np.asarray(r["part"])[res.labels == g]).argmax()) - 1
        ours[theirs] = res.stability.mean_jaccard[g]
    for g in range(3):
        tol = 4 * r["sd"][g] * np.sqrt(2 / B) + 1e-3
        assert abs(ours[g] - r["mean"][g]) <= tol, (g, ours[g], r["mean"][g])


def test_jaccard_by_hand():
    labels = np.array([0, 0, 0, 1, 1, 1])
    draws = [np.array([0, 1, 3, 4, 5])]
    # the resample's clustering puts row 1 with rows 3–5: {0} and {1, 3, 4, 5}
    jac = S.bootstrap_jaccard(labels, lambda rows: np.array([0, 1, 1, 1, 1]), draws)
    # subgroup 0 on the resample is {0, 1}: best match {0} (1/2) or {1,3,4,5} (1/5) -> 0.5
    # subgroup 1 is {3, 4, 5}: best match {1, 3, 4, 5} -> 3/4
    np.testing.assert_allclose(jac[:, 0], [0.5, 0.75])


def test_the_weighted_jaccard_by_hand():
    """With survey weights the Jaccard counts weights: rows 0–5 weigh 1, 1, 4, 1, 1, 2; the
    resample keeps 0, 1, 3, 4, 5 and finds {0} and {1, 3, 4, 5}. Subgroup 0 on the resample is
    {0, 1} (weight 2): against {0}, 1/2; against {1, 3, 4, 5} (weight 5), 1/6; so 1/2. Subgroup 1
    is {3, 4, 5} (weight 4) inside {1, 3, 4, 5}: 4/5."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([1.0, 1.0, 4.0, 1.0, 1.0, 2.0])
    seen = []

    def refit(rows, rw):
        seen.append(rw)
        return np.array([0, 1, 1, 1, 1])

    jac = S.bootstrap_jaccard(labels, refit, [np.array([0, 1, 3, 4, 5])], w=w * 7,
                              draw_weights=[np.ones(5)])
    np.testing.assert_allclose(jac[:, 0], [0.5, 0.8])
    assert len(seen) == 1


def test_the_rescaled_bootstrap_weights_by_hand():
    """Rao, Wu & Yue (1992): a stratum of n_h PSUs draws n_h − 1 of them with replacement, and a
    drawn PSU's people weigh w · n_h / (n_h − 1) · (times drawn); a stratum of one PSU keeps it."""
    strata = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    psu = np.array([0, 0, 1, 1, 2, 2, 0, 0])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    for rows, rw in S.replicate_weights(w, 50, np.random.default_rng(3), strata=strata, psu=psu):
        factor = np.zeros(len(w))
        factor[rows] = rw / w[rows]
        assert set(rows) >= {6, 7} and np.allclose(factor[6:], 1.0)  # the lone PSU, kept
        per_psu = factor[[0, 2, 4]]
        assert np.allclose(factor[[1, 3, 5]], per_psu)  # whole PSUs
        assert np.allclose(per_psu / 1.5, np.round(per_psu / 1.5))  # 3/2 per draw
        assert per_psu.sum() / 1.5 == pytest.approx(2)  # n_h − 1 = 2 draws
        assert set(rows) == set(np.flatnonzero(factor > 0))


@needs_r
@needs_survey
def test_the_rescaled_bootstrap_variance_is_the_design_variance(tmp_path):
    """The rescaled bootstrap's variance of a weighted total is, in expectation, the usual
    with-replacement design variance (Rao, Wu & Yue 1992), which ``survey::svytotal`` reports:
    4,000 replicates land within 8% of it (the Monte Carlo error is about 2%)."""
    df = blobs()
    n = len(df)
    df["w"] = 1.0 + np.arange(n) % 3
    df["stratum"] = np.arange(n) % 4
    df["psu"] = (np.arange(n) // 4) % 5
    reps = S.replicate_weights(df["w"].to_numpy(), 4000, np.random.default_rng(1),
                               strata=df["stratum"].to_numpy(), psu=df["psu"].to_numpy())
    y = df["x1"].to_numpy()
    totals = np.array([(rw * y[rows]).sum() for rows, rw in reps])
    r = run_r("""
suppressPackageStartupMessages(library(survey))
d <- read.csv(data_csv)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = d)
t <- svytotal(~x1, des)
out(list(total = as.numeric(coef(t)), var = as.numeric(SE(t))^2))
""", {"data": df}, tmp_path)
    assert totals.mean() == pytest.approx(r["total"], rel=0.02, abs=0.05 * np.sqrt(r["var"]))
    assert totals.var(ddof=0) == pytest.approx(r["var"], rel=0.08)


# ── what the engine promises ────────────────────────────────────────────────


def test_the_rule_is_declared_before_the_run():
    plan = S.declare("kmeans", "gap", k_range=(2, 7), seed=11)
    assert plan.k_min == 1  # the gap compares from one subgroup up
    assert "gap statistic" in plan.statement and "declared before the run" in plan.statement
    assert "seed 11" in plan.statement
    with pytest.raises(ValueError):
        S.declare("gmm", "silhouette")
    with pytest.raises(ValueError):
        S.declare("kmeans", "silhouette", k_range=(1, 4))
    with pytest.raises(TypeError):
        S.run(blobs(), ["x1", "x2"], {"method": "kmeans", "k": 3})  # type: ignore[arg-type]


def test_categories_are_refused_for_kmeans_and_mixtures_with_exits():
    df = mixed()
    for method in ("kmeans", "gmm"):
        with pytest.raises(S.SubgroupsRefused) as e:
            S.run(df, ["age", "bmi", "smoke"], S.declare(method, n_boot=0))
        exits = e.value.exits
        assert exits[0]["drop"] == ["smoke"] and exits[1]["method"] == "pam_gower"
        assert "0/1" in str(e.value)
    # a two-valued number is a code, and a declared reading makes a number a category
    df["sex"] = np.where(np.arange(len(df)) % 2, 1, 2)
    assert S.read_features(df, ["age", "sex"]).categorical == ("sex",)
    assert S.read_features(df, ["age", "bmi"], categorical=["bmi"]).categorical == ("bmi",)


def test_the_outcome_blanks_and_constants_are_refused():
    df = blobs()
    df["y"] = df["x1"] > 0
    with pytest.raises(S.SubgroupsRefused, match="outcome"):
        S.run(df, ["x1", "x2", "y"], S.declare("kmeans", n_boot=0), outcome="y")
    df.loc[3, "x2"] = np.nan
    with pytest.raises(S.SubgroupsRefused, match="blank") as e:
        S.run(df, ["x1", "x2"], S.declare("kmeans", n_boot=0))
    assert e.value.exits[1]["drop"] == ["x2"]
    df["c"] = 1.0
    with pytest.raises(S.SubgroupsRefused, match="the same for everyone"):
        S.read_features(df, ["x1", "c"])


@pytest.mark.parametrize("method,rule", [("gmm", "bic"), ("kmeans", "gap"),
                                         ("pam_gower", "silhouette")])
def test_survey_weights_without_a_design_based_form_are_refused_with_exits(method, rule):
    df = blobs()
    df["w"] = 1.0 + np.arange(len(df)) % 3
    with pytest.raises(S.SubgroupsRefused) as e:
        S.run(df, ["x1", "x2"], S.declare(method, rule, n_boot=0), weights="w")
    labels = [x["label"] for x in e.value.exits]
    assert any("Weighted k-means" in x for x in labels)
    assert any("sample-only" in x for x in labels)


def _coded(n=90, seed=3) -> pd.DataFrame:
    df = mixed(n, seed)
    df["race"] = np.random.default_rng(seed).integers(1, 6, n)  # codes 1–5, read as nothing yet
    df["w"] = 0.5 + np.random.default_rng(seed + 1).random(n)
    return df


EXIT_CASES = [  # (method, rule, columns, weighted): each refused, each exit followed
    ("kmeans", "silhouette", ["age", "bmi", "smoke"], False),
    ("kmeans", "silhouette", ["age", "bmi", "smoke"], True),
    ("kmeans", "gap", ["age", "bmi", "smoke"], False),
    ("kmeans", "gap", ["age", "bmi", "smoke"], True),
    ("kmeans", "gap", ["age", "bmi"], True),
    ("gmm", "bic", ["age", "bmi", "smoke"], False),
    ("gmm", "bic", ["age", "bmi", "smoke"], True),
    ("gmm", "bic", ["age", "bmi"], True),
    ("kmeans", "silhouette", ["smoke", "region"], False),
    ("pam_gower", "silhouette", ["age", "bmi", "smoke", "region", "activity"], True),
    ("pam_gower", "silhouette", ["smoke", "region"], True),
    ("kmeans", "silhouette", ["age", "bmi", "race"], False),
    ("kmeans", "silhouette", ["age", "bmi", "race"], True),
    ("gmm", "bic", ["age", "bmi", "race"], False),
    ("pam_gower", "silhouette", ["age", "smoke", "race"], False),
    ("kmeans", "silhouette", ["race"], False),
]


@pytest.mark.parametrize("method,rule,columns,weighted", EXIT_CASES)
def test_every_exit_runs_without_another_refusal(method, rule, columns, weighted):
    """An exit is a way forward: following any exit a refusal offers (``follow_exit``) runs.
    Weights with k-medoids on columns holding categories once offered weighted k-means with the
    categories still in, which the category rule then refused."""
    df = _coded()
    plan = S.declare(method, rule, k_range=(2, 3), n_init=2, n_boot=0, shapes=("diag",))
    options = {"weights": "w"} if weighted else {}
    with pytest.raises(S.SubgroupsRefused) as e:
        S.run(df, columns, plan, **options)
    assert e.value.exits
    for exit in e.value.exits:
        new, cols, opts = S.follow_exit(exit, plan, columns, **options)
        result = S.run(df, cols, new, **opts)
        assert result.k >= 1, exit["label"]
        if weighted and "weights" not in exit:
            assert result.weighted and (new.method, new.rule) == ("kmeans", "silhouette")


def test_whole_number_codes_are_asked_about_not_read_as_amounts():
    """Race coded 1 to 5 is not an amount: a column of 3 to 10 whole-number values whose reading
    is not settled is refused, with its readings as the exits; settled (from the readings ledger)
    it runs as what it is."""
    df = _coded()
    with pytest.raises(S.SubgroupsRefused, match="whole-number") as e:
        S.run(df, ["age", "bmi", "race"], S.declare("kmeans", n_boot=0, n_init=2))
    kinds = [next(k for k in ("numeric", "categorical", "drop") if k in x) for x in e.value.exits]
    assert kinds == ["numeric", "categorical", "drop"]
    assert e.value.exits[1]["method"] == "pam_gower"
    assert S.read_features(df, ["age", "race"]).ambiguous == ("race",)
    assert S.read_features(df, ["age", "race"], numeric=["race"]).ambiguous == ()
    assert S.read_features(df, ["age", "race"], categorical=["race"]).categorical == ("race",)
    df["children"] = np.arange(len(df)) % 12  # twelve values: read as a count
    assert S.read_features(df, ["age", "children"]).ambiguous == ()
    with pytest.raises(ValueError, match="both"):
        S.read_features(df, ["race"], numeric=["race"], categorical=["race"])


def test_weighted_kmeans_resamples_psus_within_strata():
    df = blobs()
    df["w"] = 1.0 + np.arange(len(df)) % 3
    df["stratum"] = np.arange(len(df)) % 4
    df["psu"] = (np.arange(len(df)) // 4) % 5
    res = S.run(df, ["x1", "x2", "x3"], S.declare("kmeans", k_range=(2, 4), n_boot=10,
                                                  n_init=5), weights="w", strata="stratum",
                psu="psu")
    assert res.stability.by_design and res.weighted
    draws = S.resamples(len(df), 3, np.random.default_rng(0), strata=df["stratum"].to_numpy(),
                        psu=df["psu"].to_numpy())
    for rows in draws:  # whole PSUs: a drawn row brings its PSU's rows in its stratum
        for s, p in set(zip(df["stratum"].to_numpy()[rows], df["psu"].to_numpy()[rows])):
            whole = np.flatnonzero((df["stratum"] == s) & (df["psu"] == p))
            assert set(whole) <= set(rows)
    assert "primary sampling units resampled within strata" in S.methods_sentence(res)


def test_the_same_seed_gives_the_same_numbers_and_the_largest_subgroup_first():
    df = mixed()
    for plan, cols in ((S.declare("kmeans", k_range=(2, 5), n_boot=15, n_init=5),
                        ["age", "bmi"]),
                       (S.declare("gmm", k_range=(1, 3), n_boot=10, n_init=3,
                                  shapes=("diag", "full")), ["age", "bmi"]),
                       (S.declare("pam_gower", k_range=(2, 4), n_boot=10), list(df.columns))):
        a, b = S.run(df, cols, plan), S.run(df, cols, plan)
        assert np.array_equal(a.labels, b.labels)
        pd.testing.assert_frame_equal(a.table, b.table)
        assert a.stability == b.stability
        assert list(a.sizes) == sorted(a.sizes, reverse=True)
        assert a.seeds["starts"] == plan.seed
        assert "No outcome was used" in S.methods_sentence(a)


def test_membership_is_fitted_in_the_training_fold_and_never_reads_the_outcome():
    from turbotab.core.contracts import observed_scope

    df = blobs((25, 20, 15), d=3)
    y = np.random.default_rng(0).normal(size=len(df))
    for method in ("kmeans", "gmm"):
        est = S.SubgroupMembership(method, k_range=(3, 3), n_init=5, shapes=("diag",))

        def fit_transform(frame, reference, yy, est=est):
            # what row i is placed by: its subgroup's center (or component mean), learned in fit
            est.fit(frame, yy)
            lab = est.assign(frame)
            centers = est.centers_ if method == "kmeans" else est.gmm_.means_[est.order_]
            return centers[lab]

        assert observed_scope(fit_transform, df, np.zeros(len(df), bool), y, 0) == \
            "training_fold", method
    # in a pipeline, each fold fits its own subgroups: the held-out rows never move the centers
    train, test = df.iloc[:40], df.iloc[40:]
    est = S.SubgroupMembership("kmeans", k_range=(2, 4), n_init=5).fit(train)
    before = est.centers_.copy()
    out = est.transform(test * 3.0)
    assert np.array_equal(est.centers_, before)
    assert list(out.columns) == [f"subgroup_{i + 1}" for i in range(est.k_)]
    assert (out.sum(axis=1) == 1).all()


def test_membership_runs_inside_cross_validation():
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    df = mixed()
    y = df["age"].to_numpy() * 0.1 + np.random.default_rng(1).normal(size=len(df))
    est = S.SubgroupMembership("pam_gower", k_range=(2, 4))
    assert clone(est).get_params() == est.get_params()
    scores = cross_val_score(make_pipeline(est, LinearRegression()), df, y, cv=3)
    assert np.isfinite(scores).all()


def test_the_contract_is_registered_with_labels_and_sources():
    from turbotab.core.contracts import contract, contracts

    c = contracts()[S.CONTRACT] if S.CONTRACT in contracts() else contract(S.CONTRACT)
    assert c.slot == "in_fold" and c.scope == "training_fold"
    keys = {o.key for o in c.options}
    assert {"kmeans_silhouette", "kmeans_gap", "gmm_bic", "pam_gower_silhouette"} <= keys
    for refused in ("kmeans_one_hot", "k_after_looking"):
        assert set(c.option(refused).rung.values()) == {"refused"}
    for o in c.options:
        assert o.customary and all(o.sound.values())
    assert S.HENNIG in c.sources and S.TIBSHIRANI in c.sources
    import importlib

    for r in c.relations:
        module, name = r.enforced_by.split(":")
        assert hasattr(importlib.import_module(module), name), r.enforced_by


def test_membership_takes_each_folds_survey_weights():
    """Under Predict the training fold's survey weights reach the fit (``sample_weight``, routed by
    the pipeline): the fitted standardization and centers are the weighted run's, the methods with
    no design-based form refuse them, and they route through cross-validation fold by fold."""
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    df = blobs((30, 25, 20), d=3, seed=4)
    w = np.random.default_rng(2).integers(1, 5, len(df)).astype(float)
    est = S.SubgroupMembership("kmeans", k_range=(2, 4), n_init=5).fit(df, sample_weight=w)
    ran = S.run(df.assign(w=w), list(df.columns), S.declare("kmeans", k_range=(2, 4), n_init=5,
                                                             n_boot=0), weights="w")
    _, mean, sd = S.standardize(df.to_numpy(), w)
    np.testing.assert_allclose(est.mean_, mean)
    np.testing.assert_allclose(est.sd_, sd)
    np.testing.assert_allclose(est.centers_, ran.centers)
    assert est.result_.weighted
    with pytest.raises(S.SubgroupsRefused, match="BIC"):
        S.SubgroupMembership("gmm", k_range=(2, 3), n_init=2).fit(df, sample_weight=w)
    y = df["x1"].to_numpy() + np.random.default_rng(1).normal(size=len(df))
    pipe = make_pipeline(S.SubgroupMembership("kmeans", k_range=(2, 3), n_init=2),
                         LinearRegression())
    with sklearn.config_context(enable_metadata_routing=False):
        scores = cross_val_score(pipe, df, y, cv=3,
                                 params={"subgroupmembership__sample_weight": w})
    assert np.isfinite(scores).all()


def test_each_refusal_names_the_function_that_raises_it():
    """A refusing relation's ``enforced_by`` is the function that raises its refusal, and the one
    for weighted k-means is the run that hands the weights to every step."""
    import importlib
    import inspect

    from turbotab.core.contracts import contract, contracts

    c = contracts()[S.CONTRACT] if S.CONTRACT in contracts() else contract(S.CONTRACT)
    by_id = {r.name: r for r in c.relations}
    assert by_id["categories_refused"].enforced_by.endswith(":_refuse")
    assert by_id["weights_refused"].enforced_by.endswith(":_refuse")
    assert by_id["codes_asked"].enforced_by.endswith(":_refuse")
    assert by_id["weighted_kmeans"].enforced_by.endswith(":run")
    for r in c.relations:
        if r.rung == "refused":
            module, name = r.enforced_by.split(":")
            assert "SubgroupsRefused(" in inspect.getsource(
                getattr(importlib.import_module(module), name)), r.name
    words = c.option("kmeans_silhouette").sound["prediction"]
    assert "sample weights" in words

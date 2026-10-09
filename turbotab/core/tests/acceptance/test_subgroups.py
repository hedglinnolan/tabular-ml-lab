"""D4, subgroups of similar people: every number held to an independent reference.

R is the reference (``r_reference.run_r``): ``stats::kmeans`` and ``cluster::silhouette`` for
k-means and its rule, run on the rows repeated by their integer weights for the weighted form;
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
def test_weighted_kmeans_equals_the_rows_repeated_in_r(tmp_path):
    """Survey weights in k-means count each person as their weight: with integer weights the
    standardization, the within sum of squares and the silhouette equal R's on the repeated
    rows."""
    df = blobs((30, 25, 20), d=3, seed=11)
    w = np.random.default_rng(5).integers(1, 4, len(df))
    df["w"] = w
    plan = S.declare("kmeans", "silhouette", k_range=(2, 5), n_boot=0, n_init=50)
    res = S.run(df, ["x1", "x2", "x3"], plan, weights="w")
    r = run_r("""
library(cluster)
d <- read.csv(data_csv)
big <- d[rep(seq_len(nrow(d)), d$w), c("x1", "x2", "x3")]
x <- scale(as.matrix(big))
set.seed(2)
o <- lapply(2:5, function(k) {
  km <- kmeans(x, k, nstart = 200, iter.max = 100)
  list(k = k, wss = km$tot.withinss, sil = mean(silhouette(km$cluster, dist(x))[, 3]))
})
out(list(center = attr(x, "scaled:center"), sd = attr(x, "scaled:scale"), rows = o))
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

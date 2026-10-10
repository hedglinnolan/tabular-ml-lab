"""T3: leave-one-site-out (internal–external) validation as an engine method core
(`turbotab.core.methods.site_validation`).

Every number is held to an independent reference: the pooling to R's metafor (``rma`` with REML and
``test = "knha"``, or DerSimonian–Laird, and ``predict(…, predtype = "Riley")``), and by hand to
Snell et al.'s (2018) equations (5) and (6) written out; each held-out site's C statistic to pROC's
DeLong variance, its calibration-in-the-large and slope to R's ``glm`` (``offset``) and ``lm``; and
the whole run end to end, from the core's held-out predictions, to pROC and metafor in one script.
That nothing learned crosses into the held-out site is checked by a model that records the rows it
was fitted on. A package R does not find skips its test (set ``R_LIBS`` to a library that holds it).
"""
from __future__ import annotations

import math
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.methods import site_validation as V
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r


def _r_has(package: str) -> bool:
    if RSCRIPT is None:
        return False
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


def _skip_without(*packages: str) -> None:
    for package in packages:
        if not _r_has(package):
            pytest.skip(f"R's {package} is not installed (set R_LIBS to a library that has it)")


# Eight sites' estimates and standard errors with real heterogeneity, and five nearly homogeneous.
YI = [0.71, 0.78, 0.66, 0.74, 0.81, 0.69, 0.76, 0.63]
SEI = [0.021, 0.034, 0.027, 0.019, 0.041, 0.030, 0.025, 0.036]
YI_FLAT = [1.02, 0.98, 1.01, 0.99, 1.00]
SEI_FLAT = [0.05, 0.06, 0.04, 0.07, 0.05]


def _sites(seed: int = 4, n: int = 1500, levels: str = "ABCDEFG") -> tuple[pd.DataFrame, np.ndarray,
                                                                         np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    site = rng.choice(list(levels), n)
    shift = dict(zip(levels, rng.normal(0, 0.6, len(levels))))
    slope = dict(zip(levels, rng.uniform(0.6, 1.4, len(levels))))
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    s = np.array([shift[v] for v in site])
    k = np.array([slope[v] for v in site])
    lp = -0.4 + k * (0.9 * X["a"] + 0.5 * X["b"]) + s
    event = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    amount = 3 + k * (2 * X["a"] - X["b"]) + s + rng.normal(0, 1, n)
    return X, event, amount.to_numpy(), site


def _logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=1000))


# ── the pooling ──────────────────────────────────────────────────────────────


@needs_r
@pytest.mark.parametrize("scale", ["identity", "logit"])
def test_reml_with_hksj_and_the_riley_interval_match_metafor(tmp_path, scale):
    _skip_without("metafor")
    got = V.pool("c_statistic", YI, SEI, scale=scale, method="reml_hksj")
    ref = run_r(f"""
suppressPackageStartupMessages(library(metafor))
d <- read.csv(m_csv)
if ("{scale}" == "logit") {{ d$sei <- d$sei / (d$yi * (1 - d$yi)); d$yi <- qlogis(d$yi) }}
ctl <- list(threshold = 1e-12, maxiter = 1000)
k <- rma(yi, sei = sei, data = d, method = "REML", test = "knha", control = ctl)
z <- rma(yi, sei = sei, data = d, method = "REML", control = ctl)
p <- predict(z, predtype = "Riley")
out(list(b = k$b[1], lb = k$ci.lb, ub = k$ci.ub, se = z$se, se_hk = k$se, tau2 = k$tau2,
         i2 = k$I2, q = k$QE, qp = k$QEp, pi = c(p$pi.lb, p$pi.ub)))
""", {"m": pd.DataFrame({"yi": YI, "sei": SEI})}, tmp_path)
    back = (lambda x: 1 / (1 + math.exp(-x))) if scale == "logit" else (lambda x: x)
    assert got.k == 8 and got.scale == scale
    assert got.estimate == pytest.approx(back(ref["b"]), rel=1e-8)
    assert got.ci_low == pytest.approx(back(ref["lb"]), rel=1e-8)
    assert got.ci_high == pytest.approx(back(ref["ub"]), rel=1e-8)
    assert got.tau2 == pytest.approx(ref["tau2"], rel=1e-7)
    assert got.i2 * 100 == pytest.approx(ref["i2"], rel=1e-7)
    assert got.se == pytest.approx(ref["se"], rel=1e-8)
    assert got.se_hk == pytest.approx(ref["se_hk"], rel=1e-8)
    assert got.q == pytest.approx(ref["q"], rel=1e-10) and got.q_p == pytest.approx(ref["qp"], rel=1e-8)
    assert got.pi_low == pytest.approx(back(ref["pi"][0]), rel=1e-8)
    assert got.pi_high == pytest.approx(back(ref["pi"][1]), rel=1e-8)


@needs_r
def test_dersimonian_laird_with_a_wald_interval_matches_metafor(tmp_path):
    _skip_without("metafor")
    got = V.pool("slope", YI, SEI, method="dl_wald")
    ref = run_r("""
suppressPackageStartupMessages(library(metafor))
d <- read.csv(m_csv)
z <- rma(yi, sei = sei, data = d, method = "DL")
p <- predict(z, predtype = "Riley")
out(list(b = z$b[1], lb = z$ci.lb, ub = z$ci.ub, tau2 = z$tau2, i2 = z$I2,
         pi = c(p$pi.lb, p$pi.ub)))
""", {"m": pd.DataFrame({"yi": YI, "sei": SEI})}, tmp_path)
    assert got.estimate == pytest.approx(ref["b"], rel=1e-10)
    assert (got.ci_low, got.ci_high) == (pytest.approx(ref["lb"], rel=1e-10),
                                         pytest.approx(ref["ub"], rel=1e-10))
    assert got.tau2 == pytest.approx(ref["tau2"], rel=1e-10)
    assert got.i2 * 100 == pytest.approx(ref["i2"], rel=1e-10)
    assert (got.pi_low, got.pi_high) == (pytest.approx(ref["pi"][0], rel=1e-10),
                                         pytest.approx(ref["pi"][1], rel=1e-10))


def test_with_no_spread_between_sites_the_intervals_are_snells_equations_by_hand():
    """τ² = 0 (the sites differ less than chance), so ŵ = 1/S², and Snell et al. 2018 equation (5)
    and (6) are computed here from the numbers alone."""
    got = V.pool("slope", YI_FLAT, SEI_FLAT)
    y, v = np.array(YI_FLAT), np.array(SEI_FLAT) ** 2
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    se = math.sqrt(1 / w.sum())
    q = math.sqrt((w * (y - mu) ** 2).sum() / 4)
    assert got.tau2 == 0.0 and got.i2 == 0.0
    assert got.estimate == pytest.approx(mu, rel=1e-12)
    assert got.ci_low == pytest.approx(mu - stats.t.ppf(0.975, 4) * q * se, rel=1e-12)
    assert got.pi_high == pytest.approx(mu + stats.t.ppf(0.975, 3) * se, rel=1e-12)


def test_two_sites_pool_without_a_prediction_interval_and_one_does_not_pool():
    two = V.pool("slope", [1.0, 1.1], [0.05, 0.06])
    assert two.k == 2 and two.pi_low is None and "needs three" in two.note
    one = V.pool("slope", [1.0, math.nan], [0.05, 0.06])
    assert one.estimate is None and "nothing to pool" in one.note


# ── each held-out site, and the whole run, against R ─────────────────────────


@needs_r
def test_the_held_out_scores_and_their_pooling_match_proc_glm_and_metafor(tmp_path):
    _skip_without("pROC", "metafor")
    X, event, _, site = _sites()
    r = V.site_validation("binary", _logistic, X, event, site, site_column="hospital")
    # The held-out predictions, made again by hand for R: fit on the other sites, score the site.
    preds = np.empty(len(event))
    for level in sorted(set(site)):
        m = _logistic().fit(X[site != level], event[site != level])
        preds[site == level] = m.predict_proba(X[site == level])[:, 1]
    ref = run_r("""
suppressPackageStartupMessages({library(pROC); library(metafor)})
d <- read.csv(p_csv)
d$lp <- qlogis(d$p)
gc <- glm.control(epsilon = 1e-14, maxit = 100)
rows <- lapply(sort(unique(d$site)), function(s) {
  x <- d[d$site == s, ]
  r <- roc(x$y, x$p, levels = c(0, 1), direction = "<", quiet = TRUE)
  a <- glm(y ~ 1 + offset(lp), data = x, family = binomial, control = gc)
  b <- glm(y ~ lp, data = x, family = binomial, control = gc)
  c(auc = as.numeric(auc(r)), auc_se = sqrt(var(r, method = "delong")),
    citl = unname(coef(a)[1]), citl_se = unname(sqrt(vcov(a)[1, 1])),
    slope = unname(coef(b)[2]), slope_se = unname(sqrt(vcov(b)[2, 2])))
})
m <- as.data.frame(do.call(rbind, rows))
ctl <- list(threshold = 1e-12, maxiter = 1000)
pool <- function(yi, sei) {
  k <- rma(yi, sei = sei, method = "REML", test = "knha", control = ctl)
  z <- rma(yi, sei = sei, method = "REML", control = ctl)
  p <- predict(z, predtype = "Riley")
  c(k$b[1], k$ci.lb, k$ci.ub, p$pi.lb, p$pi.ub, k$tau2, k$I2)
}
out(list(sites = as.list(m),
         c = pool(qlogis(m$auc), m$auc_se / (m$auc * (1 - m$auc))),
         citl = pool(m$citl, m$citl_se), slope = pool(m$slope, m$slope_se)))
""", {"p": pd.DataFrame({"site": site, "y": event, "p": preds})}, tmp_path)
    assert [s.site for s in r.sites] == list("ABCDEFG")
    for s, (auc, auc_se, citl, citl_se, slope, slope_se) in zip(
            r.sites, zip(*(ref["sites"][k] for k in ("auc", "auc_se", "citl", "citl_se", "slope",
                                                      "slope_se")))):
        assert s.c_statistic.estimate == pytest.approx(auc, rel=1e-9)
        assert s.c_statistic.se == pytest.approx(auc_se, rel=1e-7)
        assert s.citl.estimate == pytest.approx(citl, rel=1e-6, abs=1e-9)
        assert s.citl.se == pytest.approx(citl_se, rel=1e-6)
        assert s.slope.estimate == pytest.approx(slope, rel=1e-6)
        assert s.slope.se == pytest.approx(slope_se, rel=1e-6)
    for name, scale in (("c_statistic", "logit"), ("citl", "identity"), ("slope", "identity")):
        b, lb, ub, pl, pu, tau2, i2 = ref["c" if name == "c_statistic" else name]
        back = (lambda x: 1 / (1 + math.exp(-x))) if scale == "logit" else (lambda x: x)
        p = r.pooled[name]
        assert p.k == 7 and p.scale == scale
        assert p.estimate == pytest.approx(back(b), rel=1e-6)
        assert (p.ci_low, p.ci_high) == (pytest.approx(back(lb), rel=1e-6),
                                         pytest.approx(back(ub), rel=1e-6))
        assert (p.pi_low, p.pi_high) == (pytest.approx(back(pl), rel=1e-6),
                                         pytest.approx(back(pu), rel=1e-6))
        assert p.tau2 == pytest.approx(tau2, rel=1e-5, abs=1e-10)
        assert p.i2 * 100 == pytest.approx(i2, rel=1e-5, abs=1e-8)
    assert "held out in turn (7 scored)" in r.says and "prediction interval" in r.says


@needs_r
def test_a_number_s_calibration_matches_lm_and_the_t_test(tmp_path):
    from sklearn.linear_model import LinearRegression

    X, _, amount, site = _sites(seed=9)
    r = V.site_validation("regression", LinearRegression, X, amount, site)
    preds = np.empty(len(amount))
    means = {}
    for level in sorted(set(site)):
        m = LinearRegression().fit(X[site != level], amount[site != level])
        preds[site == level] = m.predict(X[site == level])
        means[level] = amount[site != level].mean()
    ref = run_r("""
d <- read.csv(p_csv)
rows <- lapply(sort(unique(d$site)), function(s) {
  x <- d[d$site == s, ]
  t <- t.test(x$y - x$p)
  f <- lm(y ~ p, data = x)
  c(citl = unname(t$estimate), citl_se = unname(t$stderr), slope = unname(coef(f)[2]),
    slope_se = unname(sqrt(vcov(f)[2, 2])))
})
out(as.list(as.data.frame(do.call(rbind, rows))))
""", {"p": pd.DataFrame({"site": site, "y": amount, "p": preds})}, tmp_path)
    for i, s in enumerate(r.sites):
        assert s.c_statistic is None
        assert s.citl.estimate == pytest.approx(ref["citl"][i], rel=1e-9)
        assert s.citl.se == pytest.approx(ref["citl_se"][i], rel=1e-9)
        assert s.slope.estimate == pytest.approx(ref["slope"][i], rel=1e-9)
        assert s.slope.se == pytest.approx(ref["slope_se"][i], rel=1e-9)
        yy, pp = amount[site == s.site], preds[site == s.site]
        assert s.r2 == pytest.approx(1 - ((yy - pp) ** 2).sum() / ((yy - means[s.site]) ** 2).sum())
    assert set(r.pooled) == {"citl", "slope"}


# ── nothing learned crosses into the held-out site ───────────────────────────


class _Recorder:
    """A model that records the rows (their ``a`` values) it was fitted on."""

    seen: list[set[float]] = []

    def fit(self, X, y):
        type(self).seen.append(set(np.round(X["a"].to_numpy(), 12)))
        from sklearn.linear_model import LogisticRegression

        self.m = LogisticRegression().fit(X, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def test_each_site_is_scored_by_a_model_fitted_without_it():
    X, event, _, site = _sites()
    _Recorder.seen = []
    V.site_validation("binary", _Recorder, X, event, site)
    assert len(_Recorder.seen) == 7
    for level, seen in zip(sorted(set(site)), _Recorder.seen):
        held = set(np.round(X["a"].to_numpy()[site == level], 12))
        others = set(np.round(X["a"].to_numpy()[site != level], 12))
        assert seen == others and not (seen & held)


# ── refusals and the sites that cannot be scored ──────────────────────────────


def test_fewer_than_three_sites_are_refused_with_the_k_fold_exit():
    X, event, _, site = _sites(levels="AB")
    with pytest.raises(V.SiteValidationRefused, match="at least 3") as caught:
        V.site_validation("binary", _logistic, X, event, site)
    assert caught.value.exits[0]["validation"] == "kfold"


def test_a_person_in_two_sites_is_refused():
    X, event, _, site = _sites()
    person = np.arange(len(event)) // 2  # pairs of rows, often in two sites
    with pytest.raises(V.SiteValidationRefused, match="more than one site") as caught:
        V.site_validation("binary", _logistic, X, event, site, units=person, unit_name="person")
    assert len(caught.value.exits) == 2


def test_a_population_design_and_other_outcomes_and_goals_are_refused():
    X, event, _, site = _sites()
    with pytest.raises(V.SiteValidationRefused, match="design-weighted") as caught:
        V.site_validation("binary", _logistic, X, event, site, design=object())
    assert caught.value.exits[0]["decision"] == {"kind": "set_survey", "estimand": "sample"}
    with pytest.raises(V.SiteValidationRefused, match="multiclass"):
        V.site_validation("multiclass", _logistic, X, event, site)
    with pytest.raises(V.SiteValidationRefused, match="Leave-one-site-out"):
        V.site_validation("binary", _logistic, X, event, site, goal="estimate")


def test_a_site_with_one_outcome_only_is_listed_and_left_out_of_the_pooling():
    X, event, _, site = _sites()
    event = event.copy()
    event[site == "C"] = 0
    r = V.site_validation("binary", _logistic, X, event, site)
    c = next(s for s in r.sites if s.site == "C")
    assert c.c_statistic is None and "one outcome only" in c.note
    assert r.pooled["c_statistic"].k == 6


def test_rows_with_no_site_are_never_fitted_on_nor_scored():
    X, event, _, site = _sites()
    site = site.astype(object)
    site[:25] = None
    _Recorder.seen = []
    r = V.site_validation("binary", _Recorder, X, event, site)
    assert r.left_out == 25 and V.NOT_RECORDED not in [s.site for s in r.sites]
    blank = set(np.round(X["a"].to_numpy()[:25], 12))
    assert all(not (seen & blank) for seen in _Recorder.seen)


def test_repeated_rows_within_a_site_cluster_its_intervals():
    X, event, _, site = _sites()
    person = pd.Series(site).astype(str) + "-" + (pd.Series(np.arange(len(site))) % 40).astype(str)
    r = V.site_validation("binary", _logistic, X, event, site, units=person.to_numpy(),
                          unit_name="person")
    plain = V.site_validation("binary", _logistic, X, event, site)
    assert r.clustered_by == "person"
    a, b = r.sites[0], plain.sites[0]
    assert a.c_statistic.estimate == pytest.approx(b.c_statistic.estimate)
    assert a.c_statistic.se != pytest.approx(b.c_statistic.se)


def test_the_contract_is_registered_with_its_sentence_and_relations():
    from turbotab.core.contracts import contract

    c = contract("site_validation")
    assert c.slot == "evaluation" and c.scope == "training_fold"
    assert [o["key"] for o in c.options_for("prediction")] == ["reml_hksj", "dl_wald"]
    assert {o["rung"] for o in c.options_for("inference")} == {"not_offered"}
    for rid in ("in_fold", "pooling_scales", "new_site_interval", "min_sites", "unit_across_sites",
                "survey_refused", "inference_refused"):
        assert c.relation(rid).enforced_by.startswith("turbotab.core.methods.site_validation:")
    sentence = V.site_validation_sentence()
    assert "Riley et al. 2016" in sentence and "Snell et al. 2018" in sentence

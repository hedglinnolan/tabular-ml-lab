"""E2: randomized-trial analyses as an engine method core (`turbotab.core.methods.trials`).

Every number is held to an independent reference: R's ``lm`` and ``emmeans`` for analysis of
covariance; R's ``beeca`` (``method = "Ye"``) for the standardized risk difference and ratio; R's
``lme4`` with ``pbkrtest`` (``vcovAdj``, ``get_Lb_ddf``) and ``emmeans`` for the mixed model with
Kenward–Roger; R's ``gee`` for the estimating equations, ``geesmv`` (``GEE.var.md``,
``GEE.var.fg``) for the Mancl–DeRouen and Fay–Graubard sandwiches, and the published formulas
written out in R (Kauermann–Carroll's principal root of (I − H), and the delta method for the
standardized risks); ``mice::pool`` for the pooled delta-adjusted analyses, and the draws of the
Bayesian regression written out in R from the same random variates. Counts are held to hand
counts. A package R does not find skips its test (set ``R_LIBS`` to a library that holds it).
"""
from __future__ import annotations

import importlib
import math
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import trials as T
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r


def _r_has(package: str) -> bool:
    if RSCRIPT is None:
        return False
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


def _skip_without(*packages: str) -> None:
    for p in packages:
        if not _r_has(p):
            pytest.skip(f"R's {p} is not installed (set R_LIBS to a library that has it)")


# ── fixtures ─────────────────────────────────────────────────────────────────


def _parallel(seed: int = 20261009, n: int = 240, arms: tuple[str, ...] = ("ctl", "trt"),
              missing_y: int = 24, missing_age: int = 12) -> pd.DataFrame:
    """A parallel trial stratified by site, with a baseline measure, a covariate partly missing,
    adherence, the arm received, a few people screened but not randomized, and missing outcomes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"arm": rng.choice(list(arms), n), "site": rng.choice(["a", "b", "c"], n),
                       "age": rng.normal(50, 10, n).round(1), "sex": rng.choice(["f", "m"], n)})
    df["base"] = rng.normal(10, 2, n)
    effect = {a: 0.9 * i for i, a in enumerate(arms)}
    df["y"] = (0.6 * df.base + df.arm.map(effect) + 0.02 * df.age + 0.3 * (df.site == "b")
               + rng.normal(0, 1.5, n))
    lp = -0.4 + df.arm.map({a: 0.6 * i for i, a in enumerate(arms)}) + 0.15 * (df.base - 10) \
        + 0.4 * (df.site == "c")
    df["event"] = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    gone = rng.choice(n, missing_y, replace=False)
    df.loc[gone, ["y", "event"]] = np.nan
    df.loc[rng.choice(n, missing_age, replace=False), "age"] = np.nan
    df["adherent"] = np.where(rng.random(n) < 0.85, "yes", "no")
    df["received"] = np.where(rng.random(n) < 0.93, df.arm, np.where(df.arm == arms[0],
                                                                       arms[-1], arms[0]))
    screened = pd.DataFrame({"arm": [None] * 6, "site": "a", "age": 40.0, "sex": "f", "base": 9.0,
                             "y": np.nan, "event": np.nan, "adherent": None, "received": None})
    return pd.concat([df, screened], ignore_index=True)


SPEC = T.TrialSpec(arm="arm", control="ctl", outcome="y", strata=("site",), covariates=("age",),
                   baseline="base", adherent="adherent", received="received")
BSPEC = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                    covariates=("age",), baseline="base", adherent="adherent")


def _cluster(seed: int = 7, K: int = 14, sizes: tuple[int, int] = (4, 40),
             equal: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = np.full(K, equal) if equal else rng.integers(sizes[0], sizes[1], K)
    df = pd.DataFrame({"cluster": np.repeat(np.arange(1, K + 1), m)})
    df["arm"] = df.cluster.map({c: ("ctl" if c % 2 else "trt") for c in range(1, K + 1)})
    u = rng.normal(0, 0.8, K + 1)
    df["urban"] = df.cluster.map({c: float(c % 3 == 0) for c in range(1, K + 1)})  # cluster level
    df["x"] = rng.normal(0, 1, len(df))
    df["y"] = (u[df.cluster] + 0.6 * (df.arm == "trt") + 0.8 * df.x + 0.3 * df.urban
               + rng.normal(0, 2, len(df)))
    lp = 0.6 * u[df.cluster] + 0.5 * (df.arm == "trt") + 0.3 * df.x - 0.2
    df["event"] = (rng.random(len(df)) < 1 / (1 + np.exp(-lp))).astype(int)
    return df


def _filled(df: pd.DataFrame, column: str, pool: pd.Series) -> pd.DataFrame:
    """The mean fill and its indicator as White & Thompson describe it, written out by hand."""
    out = df.copy()
    mean = df.loc[pool, column].mean()
    out[f"{column}_miss"] = df[column].isna().astype(float)
    out[column] = df[column].fillna(mean)
    return out


# ── the analysis sets, the CONSORT flow and the baseline table ───────────────


def test_intention_to_treat_keeps_everyone_randomized_as_randomized_and_per_protocol_the_adherent():
    df = _parallel()
    sets = T.analysis_sets(df, SPEC)
    randomized = df.arm.notna()
    assert sets.itt.equals(randomized)
    assert sets.per_protocol.equals(randomized & (df.adherent == "yes"))
    # the arm received never moves anyone: crossovers stay in the arm they were randomized to
    crossed = randomized & (df.received != df.arm)
    assert crossed.any() and sets.itt[crossed].all()
    assert (sets.arm[crossed] == df.arm[crossed]).all()


def test_the_analysis_set_is_row_local_as_its_contract_declares():
    """Lockbox constitution §06's test: a person's membership moves with nothing but their own
    row."""
    from turbotab.core.contracts import observed_scope, scope_of

    df = _parallel().dropna(subset=["arm"]).reset_index(drop=True)
    frame = df[["base", "age"]].copy()

    def fit_transform(f: pd.DataFrame, reference: pd.Series, y: np.ndarray) -> pd.DataFrame:
        full = df.copy()
        full[["base", "age"]] = f[["base", "age"]].to_numpy()
        full["y"] = y
        return T.in_analysis_set(full, SPEC, "per_protocol").astype(float).to_frame()

    y = df.y.fillna(0).to_numpy()
    assert observed_scope(fit_transform, frame, np.zeros(len(df), bool), y, 3) == "row_local"
    assert scope_of("trial_analysis_set") == "row_local"


def test_the_consort_flow_counts_each_arm_by_stage_as_counted_by_hand():
    df = _parallel()
    flow = T.consort_flow(df, SPEC)
    assert (flow.assessed, flow.not_randomized, flow.randomized) == (len(df), 6, len(df) - 6)
    for a in flow.arms:
        rows = df[df.arm == a.arm]
        assert a.allocated == len(rows)
        assert a.received_allocated == int((rows.received == a.arm).sum())
        assert a.did_not_receive_allocated == int((rows.received != a.arm).sum())
        assert a.followed_up == a.analyzed_itt == int(rows.y.notna().sum())
        assert a.lost_to_follow_up == int(rows.y.isna().sum())
        assert a.analyzed_itt_sensitivity == len(rows)
        assert a.per_protocol == int((rows.adherent == "yes").sum())
        assert a.excluded_from_per_protocol == int((rows.adherent != "yes").sum())
        assert a.analyzed_per_protocol == int(((rows.adherent == "yes") & rows.y.notna()).sum())
        assert a.clusters_allocated is None
    c = _cluster()
    c.loc[c.index[:3], "y"] = np.nan
    cflow = T.consort_flow(c, T.TrialSpec(arm="arm", control="ctl", outcome="y", cluster="cluster",
                                         trial_design="cluster_randomized_trial"))
    assert cflow.cluster_trial
    for a in cflow.arms:
        rows = c[c.arm == a.arm]
        assert a.clusters_allocated == rows.cluster.nunique()
        assert a.clusters_analyzed == rows[rows.y.notna()].cluster.nunique()
        assert a.cluster_sizes == rows.cluster.value_counts().sort_index().tolist()


def test_the_baseline_table_describes_each_arm_and_has_no_test():
    df = _parallel()
    table = T.baseline_table(df, SPEC, ["age", "sex", "site"])
    assert table.arms == ["ctl", "trt"]
    age = next(r for r in table.rows if r.variable == "age")
    for a in table.arms:
        v = df.loc[df.arm == a, "age"]
        got = age.by_arm[a]
        assert got["mean"] == pytest.approx(v.mean(), rel=1e-12)
        assert got["sd"] == pytest.approx(v.std(ddof=1), rel=1e-12)
        assert got["median"] == pytest.approx(v.median(), rel=1e-12)
        assert got["missing"] == int(v.isna().sum())
    female = next(r for r in table.rows if r.variable == "sex" and r.level == "f")
    for a in table.arms:
        v = df.loc[df.arm == a, "sex"]
        assert female.by_arm[a]["count"] == int((v == "f").sum())
        assert female.by_arm[a]["percent"] == pytest.approx(100 * (v == "f").mean())
    for r in table.rows:
        for summaries in r.by_arm.values():
            assert not {"p", "p_value", "test", "statistic"} & set(summaries)
    with pytest.raises(T.TrialRefused) as refused:
        T.baseline_table(df, SPEC, ["age"], tests=True)
    assert refused.value.term == "significance tests of baseline balance"
    exit_ = refused.value.exits[0]
    assert T.baseline_table(df, SPEC, ["age"], tests=exit_["tests"]).rows


# ── parallel trials against R ────────────────────────────────────────────────


@needs_r
@pytest.mark.parametrize("arms", [("ctl", "trt"), ("ctl", "high", "low")])
def test_ancova_matches_lm_and_emmeans(tmp_path, arms):
    _skip_without("emmeans")
    df = _parallel(arms=arms)
    r = T.estimate(df, SPEC)
    rand = df.arm.notna()
    data = _filled(df[rand], "age", pd.Series(True, index=df[rand].index))
    data = data[data.y.notna()]
    levels = ", ".join(f'"{a}"' for a in ["ctl", *sorted(arms[1:])])
    ref = run_r(f"""
suppressPackageStartupMessages(library(emmeans))
d <- read.csv(data_csv)
d$arm <- factor(d$arm, levels = c({levels})); d$site <- factor(d$site)
f <- lm(y ~ arm + site + base + age + age_miss, data = d)
s <- summary(f)$coefficients; ci <- confint(f)
es <- summary(emmeans(f, "arm", weights = "proportional"))
k <- 1 + seq_len({len(arms) - 1})
out(list(est = unname(s[k, 1]), se = unname(s[k, 2]), p = unname(s[k, 4]),
         lo = unname(ci[k, 1]), hi = unname(ci[k, 2]), df = f$df.residual,
         means = es$emmean, mse = es$SE, mlo = es$lower.CL, mhi = es$upper.CL,
         n = nrow(d), coef = unname(coef(f))))
""", {"data": data}, tmp_path)
    for key in ("est", "se", "p", "lo", "hi"):
        ref[key] = np.atleast_1d(ref[key])
    assert sum(r.n_analyzed.values()) == ref["n"]
    diffs = [c for c in r.contrasts if c.measure == "mean_difference"]
    for i, c in enumerate(diffs):
        assert c.estimate == pytest.approx(ref["est"][i], rel=1e-10)
        assert c.se == pytest.approx(ref["se"][i], rel=1e-10)
        assert (c.lower, c.upper) == pytest.approx((ref["lo"][i], ref["hi"][i]), rel=1e-10)
        assert c.p == pytest.approx(ref["p"][i], rel=1e-8)
        assert c.df == ref["df"]
    for i, m in enumerate(r.arm_estimates):
        assert m.estimate == pytest.approx(ref["means"][i], rel=1e-10)
        assert m.se == pytest.approx(ref["mse"][i], rel=1e-10)
        assert (m.lower, m.upper) == pytest.approx((ref["mlo"][i], ref["mhi"][i]), rel=1e-10)
    assert r.coefficients == pytest.approx(ref["coef"], rel=1e-9, abs=1e-12)
    assert r.filled_baselines == {"age": int(data.age_miss.sum())}
    assert r.causal and "Assignment to" in r.says


@needs_r
@pytest.mark.parametrize("strata", [True, False])
def test_the_standardized_risks_match_beeca_with_ye_s_variance(tmp_path, strata):
    _skip_without("beeca")
    df = _parallel(seed=31, n=400)
    spec = BSPEC if strata else T.TrialSpec(arm="arm", control="ctl", outcome="event",
                                            covariates=("site", "age"), baseline="base")
    r = T.estimate(df, spec)
    rand = df.arm.notna()
    data = _filled(df[rand], "age", pd.Series(True, index=df[rand].index))
    data = data[data.event.notna()]
    ref = run_r(f"""
suppressPackageStartupMessages(library(beeca))
d <- read.csv(data_csv)
d$arm <- factor(d$arm, levels = c("ctl", "trt")); d$site <- factor(d$site)
f <- glm(event ~ arm + site + base + age + age_miss, family = binomial, data = d,
         control = glm.control(epsilon = 1e-15, maxit = 100))
strata <- {'"site"' if strata else 'NULL'}
a <- get_marginal_effect(f, trt = "arm", strata = strata, method = "Ye", contrast = "diff",
                         reference = "ctl")
b <- get_marginal_effect(f, trt = "arm", strata = strata, method = "Ye", contrast = "logrr",
                         reference = "ctl")
out(list(means = unname(a$counterfactual.means), V = unname(a$robust_varcov),
         rd = unname(a$marginal_est), rd_se = unname(a$marginal_se),
         lrr = unname(b$marginal_est), lrr_se = unname(b$marginal_se)))
""", {"data": data}, tmp_path)
    rd, rr = r.contrast(measure="risk_difference"), r.contrast(measure="risk_ratio")
    assert [m.estimate for m in r.arm_estimates] == pytest.approx(ref["means"], rel=1e-9)
    V = np.array(ref["V"]).reshape(2, 2)
    assert [m.se for m in r.arm_estimates] == pytest.approx(np.sqrt(np.diag(V)), rel=1e-8)
    assert rd.estimate == pytest.approx(ref["rd"], rel=1e-9)
    assert rd.se == pytest.approx(ref["rd_se"], rel=1e-8)
    assert math.log(rr.estimate) == pytest.approx(ref["lrr"], rel=1e-9)
    assert rr.se == pytest.approx(ref["lrr_se"], rel=1e-8)
    z = 1.959963984540054
    assert (rd.lower, rd.upper) == pytest.approx((ref["rd"] - z * ref["rd_se"],
                                                  ref["rd"] + z * ref["rd_se"]), rel=1e-8)
    assert (rr.lower, rr.upper) == pytest.approx(
        (math.exp(ref["lrr"] - z * ref["lrr_se"]), math.exp(ref["lrr"] + z * ref["lrr_se"])),
        rel=1e-8)
    assert rd.df is None and "percentage points" in r.says


def test_without_covariates_the_standardized_risk_difference_is_the_two_proportions():
    """With the arm alone, the standardized risks are each arm's proportion of events and Ye's
    variance is p(1 − p)/n per arm (by hand)."""
    df = _parallel(seed=3, n=300)
    r = T.estimate(df, T.TrialSpec(arm="arm", control="ctl", outcome="event"))
    obs = df[df.arm.notna() & df.event.notna()]
    p = obs.groupby("arm").event.mean()
    n = obs.groupby("arm").size()
    rd = r.contrast(measure="risk_difference")
    assert rd.estimate == pytest.approx(p["trt"] - p["ctl"], rel=1e-10)
    # Ye's variance with n − 1 denominators: Var_j(Y)/π_j over n is p(1 − p)/(n_j − 1)·… by hand
    var = sum(obs.loc[obs.arm == a, "event"].var(ddof=1) / n[a] for a in ("ctl", "trt"))
    assert rd.se == pytest.approx(math.sqrt(var), rel=1e-10)


# ── cluster-randomized trials against R ──────────────────────────────────────


CSPEC = T.TrialSpec(arm="arm", control="ctl", outcome="y", trial_design="cluster_randomized_trial",
                    cluster="cluster", covariates=("x", "urban"))

_LMER = """
suppressPackageStartupMessages({library(lme4); library(pbkrtest); library(emmeans)})
d <- read.csv(data_csv)
d$arm <- factor(d$arm, levels = c("ctl", "trt"))
ctl <- lmerControl(optimizer = "nloptwrap", calc.derivs = FALSE,
                   optCtrl = list(xtol_abs = 1e-14, ftol_abs = 1e-15, xtol_rel = 1e-14,
                                  ftol_rel = 1e-15, maxeval = 1e5))
f <- lmer(y ~ arm + x + urban + (1 | cluster), data = d, REML = TRUE, control = ctl)
Va <- vcovAdj(f)
L <- c(0, 1, 0, 0)
em <- summary(emmeans(f, "arm", weights = "proportional", lmer.df = "kenward-roger"))
vc <- as.data.frame(VarCorr(f))
out(list(beta = unname(fixef(f)), su = vc$vcov[1], se = vc$vcov[2],
         phi = unname(as.matrix(vcov(f))), phia = unname(as.matrix(Va)),
         df = get_Lb_ddf(f, L), means = em$emmean, mse = em$SE, mdf = em$df))
"""


@needs_r
@pytest.mark.parametrize("seed", [7, 11])
def test_the_mixed_model_and_its_kenward_roger_intervals_match_pbkrtest(tmp_path, seed):
    _skip_without("lme4", "pbkrtest", "emmeans")
    df = _cluster(seed=seed)
    r = T.estimate(df, CSPEC)
    ref = run_r(_LMER, {"data": df}, tmp_path)
    assert r.coefficients == pytest.approx(ref["beta"], rel=1e-6, abs=1e-9)
    assert r.variance_components["between_clusters"] == pytest.approx(ref["su"], rel=1e-5)
    assert r.variance_components["within_clusters"] == pytest.approx(ref["se"], rel=1e-6)
    c = r.contrast()
    phia = np.array(ref["phia"]).reshape(4, 4)
    assert c.se == pytest.approx(math.sqrt(phia[1, 1]), rel=1e-5)
    assert c.df == pytest.approx(ref["df"], rel=1e-5)
    for i, m in enumerate(r.arm_estimates):
        assert m.estimate == pytest.approx(ref["means"][i], rel=1e-6)
        assert m.se == pytest.approx(ref["mse"][i], rel=1e-5)
        assert m.df == pytest.approx(ref["mdf"][i], rel=1e-5)
    assert r.icc == pytest.approx(ref["su"] / (ref["su"] + ref["se"]), rel=1e-5)
    assert r.icc_source == "the mixed model's variance components"


@needs_r
def test_kenward_roger_at_r_s_variance_components_is_pbkrtest_to_machine_precision(tmp_path):
    """The adjustment itself, free of the two programs' optimizers: at lme4's σ²_u and σ²_e the
    adjusted covariance and the degrees of freedom are pbkrtest's."""
    _skip_without("lme4", "pbkrtest", "emmeans")
    df = _cluster(seed=5, K=10)
    ref = run_r(_LMER, {"data": df}, tmp_path)
    X = np.column_stack([np.ones(len(df)), (df.arm == "trt").astype(float), df.x, df.urban])
    codes = pd.factorize(df.cluster)[0]
    kr = T.kenward_roger(X, codes, ref["su"], ref["se"], np.array(ref["phi"]).reshape(4, 4))
    assert kr.phi_adjusted == pytest.approx(np.array(ref["phia"]).reshape(4, 4), rel=1e-9,
                                            abs=1e-14)
    assert T.kr_df(np.array([0, 1, 0, 0.0]), kr) == pytest.approx(ref["df"], rel=1e-9)


_GEE_HAND = """
suppressPackageStartupMessages({library(gee)})
gee_cov <- function(beta, alpha, X, y, id, family, type, b = 0.75) {
  eta <- drop(X %*% beta)
  mu <- if (family == "gaussian") eta else plogis(eta)
  v <- if (family == "gaussian") rep(1, length(mu)) else mu * (1 - mu)
  parts <- lapply(unique(id), function(g) {
    r <- id == g; m <- sum(r)
    R <- matrix(alpha, m, m); diag(R) <- 1
    A <- diag(sqrt(v[r]), m)
    list(D = v[r] * X[r, , drop = FALSE], Vi = solve(A %*% R %*% A), e = y[r] - mu[r])
  })
  Omega <- solve(Reduce(`+`, lapply(parts, function(p) t(p$D) %*% p$Vi %*% p$D)))
  meat <- 0
  for (p in parts) {
    m <- length(p$e)
    H <- p$D %*% Omega %*% t(p$D) %*% p$Vi
    u <- switch(type,
      lz = t(p$D) %*% p$Vi %*% p$e,
      md = t(p$D) %*% p$Vi %*% solve(diag(m) - H) %*% p$e,
      kc = {ev <- eigen(diag(m) - H); Q <- ev$vectors
            S <- Q %*% diag(ev$values^-0.5, m) %*% solve(Q)
            t(p$D) %*% p$Vi %*% Re(S) %*% p$e},
      fg = {Qm <- t(p$D) %*% p$Vi %*% p$D %*% Omega
            ((1 - pmin(b, diag(Qm)))^-0.5) * (t(p$D) %*% p$Vi %*% p$e)})
    meat <- meat + u %*% t(u)
  }
  Omega %*% meat %*% Omega
}
d <- read.csv(data_csv)
d$arm <- factor(d$arm, levels = c("ctl", "trt"))
fam <- if (family == "gaussian") gaussian else binomial
g <- gee(as.formula(paste(outcome, "~ arm + x + urban")), id = cluster, data = d, family = fam,
         corstr = "exchangeable", tol = 1e-14, maxiter = 500)
X <- model.matrix(~ arm + x + urban, d)
beta <- unname(coef(g)); alpha <- g$working.correlation[1, 2]
covs <- lapply(c("lz", "kc", "md", "fg"), function(t) unname(gee_cov(beta, alpha, X, d[[outcome]],
                                                                   d$cluster, family, t)))
names(covs) <- c("lz", "kc", "md", "fg")
"""


@needs_r
@pytest.mark.parametrize("family", ["gaussian", "binary"])
def test_the_estimating_equations_match_r_gee(tmp_path, family):
    _skip_without("gee")
    df = _cluster(seed=13)
    outcome = "y" if family == "gaussian" else "event"
    X = np.column_stack([np.ones(len(df)), (df.arm == "trt").astype(float), df.x, df.urban])
    codes = pd.factorize(df.cluster)[0]
    fit = T.fit_gee(X, df[outcome].to_numpy(float), codes, family)
    ref = run_r(f'family <- "{family if family == "gaussian" else "binomial"}"; '
                f'outcome <- "{outcome}"\n' + _GEE_HAND + """
out(list(beta = beta, alpha = alpha, scale = g$scale, robust = unname(g$robust.variance)))
""", {"data": df}, tmp_path)
    assert fit.beta == pytest.approx(ref["beta"], rel=1e-9)
    assert fit.alpha == pytest.approx(ref["alpha"], rel=1e-8)
    assert fit.phi == pytest.approx(ref["scale"], rel=1e-8)
    lz = T.gee_covariance(X, df[outcome].to_numpy(float), codes, fit.beta, fit.alpha,
                          family, "lz")
    assert lz == pytest.approx(np.array(ref["robust"]).reshape(4, 4), rel=1e-7, abs=1e-12)


@needs_r
@pytest.mark.parametrize("family", ["gaussian", "binary"])
def test_the_corrected_sandwiches_match_the_published_formulas_and_geesmv(tmp_path, family):
    """Mancl–DeRouen and Fay–Graubard against geesmv itself (its ``gee`` held to a tight
    tolerance); all four against the formulas written out in R, Kauermann–Carroll as the principal
    root of the non-symmetric I − H_g. geesmv's Gaussian Fay–Graubard reads every cluster as the
    largest, so its clusters here are of one size."""
    _skip_without("gee", "geesmv")
    df = _cluster(seed=17, equal=12 if family == "gaussian" else None)
    outcome = "y" if family == "gaussian" else "event"
    y = df[outcome].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), (df.arm == "trt").astype(float), df.x, df.urban])
    codes = pd.factorize(df.cluster)[0]
    fit = T.fit_gee(X, y, codes, family)
    ref = run_r(f'family <- "{family if family == "gaussian" else "binomial"}"; '
                f'outcome <- "{outcome}"\n' + _GEE_HAND + """
suppressPackageStartupMessages(library(geesmv))
imp <- parent.env(asNamespace("geesmv"))
tight <- function(formula, data, id, family, corstr) {
  call <- match.call(); call[[1]] <- quote(gee::gee); call$tol <- 1e-14; call$maxiter <- 500
  eval(call, parent.frame())
}
unlockBinding("gee", imp); assign("gee", tight, envir = imp); lockBinding("gee", imp)
fml <- as.formula(paste(outcome, "~ arm + x + urban"))
md <- GEE.var.md(fml, id = "cluster", family = fam, data = d, corstr = "exchangeable")
fg <- GEE.var.fg(fml, id = "cluster", family = fam, data = d, corstr = "exchangeable")
out(list(covs = covs, md = unname(md$cov.beta), fg = unname(fg$cov.beta)))
""", {"data": df}, tmp_path)
    for kind in ("lz", "kc", "md", "fg"):
        mine = T.gee_covariance(X, y, codes, fit.beta, fit.alpha, family, kind)
        assert mine == pytest.approx(np.array(ref["covs"][kind]).reshape(4, 4), rel=1e-7,
                                     abs=1e-12), kind
    assert np.diag(T.gee_covariance(X, y, codes, fit.beta, fit.alpha, family, "md")) == \
        pytest.approx(ref["md"], rel=1e-7)
    assert np.diag(T.gee_covariance(X, y, codes, fit.beta, fit.alpha, family, "fg")) == \
        pytest.approx(ref["fg"], rel=1e-7)


@needs_r
@pytest.mark.parametrize("correction", ["kc", "fg", "md"])
def test_a_cluster_trial_s_standardized_risks_are_the_delta_method_on_the_corrected_covariance(
        tmp_path, correction):
    _skip_without("gee")
    df = _cluster(seed=19, K=16)
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", covariates=("x", "urban"),
                       trial_design="cluster_randomized_trial", cluster="cluster")
    r = T.estimate(df, spec, correction=correction)
    ref = run_r('family <- "binomial"; outcome <- "event"\n' + _GEE_HAND + f"""
V <- covs[["{correction}"]]
X0 <- X; X0[, 2] <- 0; X1 <- X; X1[, 2] <- 1
m0 <- plogis(drop(X0 %*% beta)); m1 <- plogis(drop(X1 %*% beta))
g0 <- colMeans(X0 * m0 * (1 - m0)); g1 <- colMeans(X1 * m1 * (1 - m1))
t0 <- mean(m0); t1 <- mean(m1)
gd <- g1 - g0; gr <- g1 / t1 - g0 / t0
K <- length(unique(d$cluster)); dfree <- K - 3
q <- qt(0.975, dfree)
rd <- t1 - t0; se <- sqrt(drop(t(gd) %*% V %*% gd))
lr <- log(t1 / t0); sr <- sqrt(drop(t(gr) %*% V %*% gr))
out(list(rd = rd, se = se, lo = rd - q * se, hi = rd + q * se, rr = exp(lr), sr = sr,
         rlo = exp(lr - q * sr), rhi = exp(lr + q * sr), df = dfree, alpha = alpha,
         risks = c(t0, t1)))
""", {"data": df}, tmp_path)
    rd, rr = r.contrast(measure="risk_difference"), r.contrast(measure="risk_ratio")
    assert r.correction == correction and rd.df == ref["df"] == 16 - 3
    assert (rd.estimate, rd.se, rd.lower, rd.upper) == pytest.approx(
        (ref["rd"], ref["se"], ref["lo"], ref["hi"]), rel=1e-7)
    assert (rr.estimate, rr.se, rr.lower, rr.upper) == pytest.approx(
        (ref["rr"], ref["sr"], ref["rlo"], ref["rhi"]), rel=1e-7)
    assert [m.estimate for m in r.arm_estimates] == pytest.approx(ref["risks"], rel=1e-8)
    assert r.icc == pytest.approx(ref["alpha"], rel=1e-7)


def test_the_correction_follows_li_and_redden_s_rule_on_the_cluster_sizes():
    even = _cluster(seed=3, equal=20)
    assert T.size_cv(pd.factorize(even.cluster)[0]) == 0.0
    assert T.estimate(even, CSPEC, cluster_method="gee").correction == "kc"
    rng = np.random.default_rng(2)
    uneven = _cluster(seed=3, sizes=(2, 120))
    codes = pd.factorize(uneven.cluster)[0]
    sizes = np.bincount(codes)
    assert T.size_cv(codes) == pytest.approx(sizes.std(ddof=1) / sizes.mean())
    expected = "kc" if T.size_cv(codes) < 0.6 else "fg"
    assert T.estimate(uneven, CSPEC, cluster_method="gee").correction == expected
    del rng


def test_gee_intervals_are_t_on_the_clusters_less_the_cluster_level_terms():
    df = _cluster(seed=23, K=12)
    r = T.estimate(df, CSPEC, cluster_method="gee", correction="kc")
    # the intercept, the arm and the urban flag are constant within clusters; x is not
    assert r.contrast().df == 12 - 3
    assert "12 − 3 = 9" in r.interval


# ── missing outcomes: delta-adjusted multiple imputation ─────────────────────


@needs_r
def test_a_draw_is_the_bayesian_regression_written_out_in_r(tmp_path):
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = X @ [1, 2, -1] + rng.normal(size=30)
    Xm = np.column_stack([np.ones(5), rng.normal(size=(5, 2))])
    fit = T._ols(X, y)
    chi2, zb, zy = 23.4, rng.standard_normal(3), rng.standard_normal(5)
    mine = T.normal_draw(fit.beta, fit.xtx_inv, fit.rss, fit.df, Xm, chi2, zb, zy)
    frame = pd.DataFrame(np.column_stack([X, y]), columns=["i", "a", "b", "y"])
    extra = pd.DataFrame(np.column_stack([Xm, np.r_[zy], np.r_[zb, np.full(2, np.nan)]]),
                         columns=["i", "a", "b", "zy", "zb"])
    ref = run_r(f"""
d <- read.csv(obs_csv); m <- read.csv(mis_csv)
X <- as.matrix(d[, c("i", "a", "b")]); Xm <- as.matrix(m[, c("i", "a", "b")])
f <- lm.fit(X, d$y); rss <- sum(f$residuals^2)
v <- solve(crossprod(X)); s <- sqrt(rss / {chi2})
b <- f$coefficients + s * drop(t(chol(v)) %*% m$zb[1:3])
out(list(y = drop(Xm %*% b) + s * m$zy))
""", {"obs": frame, "mis": extra}, tmp_path)
    assert mine == pytest.approx(ref["y"], rel=1e-10)


def test_the_shift_moves_only_the_imputed_outcomes_of_its_arm_by_exactly_delta():
    df = _parallel(seed=8)
    _, base = T.delta_analysis(df, SPEC, arm="trt", delta=0.0, m=5)
    _, shifted = T.delta_analysis(df, SPEC, arm="trt", delta=-1.7, m=5)
    rand = df[df.arm.notna()]
    imputed_trt = rand.y.isna() & (rand.arm == "trt")
    for a, b in zip(base, shifted):
        diff = (b - a).to_numpy()
        assert diff[imputed_trt.to_numpy()] == pytest.approx(-1.7, abs=1e-12)
        assert np.all(diff[~imputed_trt.to_numpy()] == 0)
        assert a[rand.y.notna().to_numpy()].equals(rand.y[rand.y.notna()].rename(a.name))


@needs_r
@pytest.mark.parametrize("delta", [0.0, -1.25])
def test_the_pooled_delta_adjusted_analysis_is_mice_pool_on_the_completed_sets(tmp_path, delta):
    _skip_without("mice")
    df = _parallel(seed=9)
    row, completed = T.delta_analysis(df, SPEC, arm="trt", delta=delta, m=12)
    rand = df[df.arm.notna()]
    data = _filled(rand, "age", pd.Series(True, index=rand.index))
    for i, c in enumerate(completed):
        data[f"y{i}"] = c.to_numpy()
    ref = run_r("""
suppressPackageStartupMessages(library(mice))
d <- read.csv(data_csv)
d$arm <- factor(d$arm, levels = c("ctl", "trt")); d$site <- factor(d$site)
fits <- lapply(0:11, function(i) lm(as.formula(paste0("y", i, " ~ arm + site + base + age + age_miss")),
                                    data = d))
s <- summary(pool(as.mira(fits)), conf.int = TRUE)
r <- s[s$term == "armtrt", ]
out(list(est = r$estimate, se = r$std.error, df = r$df, lo = r$`2.5 %`, hi = r$`97.5 %`,
         p = r$p.value))
""", {"data": data}, tmp_path)
    assert row.estimate == pytest.approx(ref["est"], rel=1e-10)
    assert row.df == pytest.approx(ref["df"], rel=1e-9)
    assert (row.lower, row.upper) == pytest.approx((ref["lo"], ref["hi"]), rel=1e-9)
    assert row.p == pytest.approx(ref["p"], rel=1e-7)


@needs_r
def test_a_yes_no_outcome_s_pooled_risk_difference_is_beeca_per_set_and_mice_s_rules(tmp_path):
    _skip_without("mice", "beeca")
    df = _parallel(seed=10, n=360)
    row, completed = T.delta_analysis(df, BSPEC, arm="trt", delta=-0.8, m=8)
    rand = df[df.arm.notna()]
    data = _filled(rand, "age", pd.Series(True, index=rand.index))
    for i, c in enumerate(completed):
        data[f"e{i}"] = c.to_numpy()
    ref = run_r("""
suppressPackageStartupMessages({library(mice); library(beeca)})
d <- read.csv(data_csv)
d$arm <- factor(d$arm, levels = c("ctl", "trt")); d$site <- factor(d$site)
q <- u <- numeric(8)
for (i in 0:7) {
  d$ev <- d[[paste0("e", i)]]
  f <- glm(ev ~ arm + site + base + age + age_miss, family = binomial, data = d,
           control = glm.control(epsilon = 1e-15, maxit = 100))
  a <- get_marginal_effect(f, trt = "arm", strata = "site", method = "Ye", contrast = "diff",
                           reference = "ctl")
  q[i + 1] <- a$marginal_est; u[i + 1] <- a$marginal_se^2
}
p <- pool.scalar(q, u, n = Inf)
se <- sqrt(p$t); tq <- qt(0.975, p$df)
out(list(est = p$qbar, se = se, df = p$df, lo = p$qbar - tq * se, hi = p$qbar + tq * se))
""", {"data": data}, tmp_path)
    assert row.estimate == pytest.approx(ref["est"], rel=1e-8)
    assert row.df == pytest.approx(ref["df"], rel=1e-7)
    assert (row.lower, row.upper) == pytest.approx((ref["lo"], ref["hi"]), rel=1e-7)


def test_the_tipping_point_is_where_the_interval_s_bound_reaches_no_difference():
    df = _parallel(seed=12)
    tp = T.tipping_point(df, SPEC, m=20)
    c = tp.contrasts[0]
    assert c.at_mar.excludes_null and c.tipping_delta is not None and c.tipping_delta < 0
    at = T.delta_analysis(df, SPEC, arm="trt", delta=c.tipping_delta, m=20)[0]
    before = T.delta_analysis(df, SPEC, arm="trt", delta=c.tipping_delta * 0.999, m=20)[0]
    assert not at.excludes_null and before.excludes_null
    assert at.lower == pytest.approx(0.0, abs=1e-5)
    assert tp.n_missing == {a: int((df.arm == a).sum() - df.loc[df.arm == a, "y"].notna().sum())
                            for a in ("ctl", "trt")}
    # every randomized person enters the sensitivity analysis, those with no outcome included
    assert tp.n_randomized == {a: int((df.arm == a).sum()) for a in ("ctl", "trt")}
    _, completed = T.delta_analysis(df, SPEC, arm="trt", delta=0.0, m=3)
    assert all(len(c) == int(df.arm.notna().sum()) and c.notna().all() for c in completed)
    assert [r.delta for r in c.rows][0] == 0.0 and c.rows[-1].delta == pytest.approx(
        1.25 * c.tipping_delta)
    estimates = [r.estimate for r in c.rows]
    assert estimates == sorted(estimates, reverse=True)  # a lower δ, a smaller difference
    assert "shifted by" in tp.says


def test_at_no_shift_the_imputed_analysis_agrees_with_the_primary_one_within_its_noise():
    df = _parallel(seed=14, n=600, missing_y=60)
    primary = T.estimate(df, SPEC).contrast()
    row, _ = T.delta_analysis(df, SPEC, arm="trt", delta=0.0, m=200)
    assert abs(row.estimate - primary.estimate) < 0.25 * primary.se


def test_the_number_of_imputations_is_at_least_the_percentage_missing():
    df = _parallel(seed=15, n=200, missing_y=50)
    tp = T.tipping_point(df, SPEC)
    assert tp.m == 25  # 50 of 200 randomized: 25%, above the floor of 20


# ── refusals, each with exits that run ───────────────────────────────────────


def _runs(fn, refusal: T.TrialRefused, **base) -> None:
    """Every exit that names an argument runs without another refusal."""
    for e in refusal.exits:
        keys = {k: v for k, v in e.items() if k in ("tests", "analysis_set", "cluster_method",
                                                     "secondary", "covariates_prespecified",
                                                     "measure")}
        if keys:
            fn(**{**base, **keys})


def test_survey_weights_prediction_and_an_observational_design_are_refused():
    df = _parallel()
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, SPEC, survey_design=object())
    assert r.value.exits[0]["decision"] == {"kind": "set_survey", "estimand": "sample"}
    T.estimate(df, SPEC, survey_design=None)
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, SPEC, goal="prediction")
    assert r.value.exits[0]["goal"] == "inference"
    T.estimate(df, SPEC, goal=r.value.exits[0]["goal"])
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, T.TrialSpec(arm="arm", control="ctl", outcome="y",
                                   trial_design="observational"))
    assert r.value.exits[0]["decision"]["kind"] == "set_design"
    with pytest.raises(T.TrialRefused) as r:
        T.tipping_point(df, SPEC, survey_design=object())
    assert r.value.term == "survey design in a randomized trial"
    with pytest.raises(T.TrialRefused) as r:
        T.tipping_point(df, SPEC, goal="prediction")
    assert T.tipping_point(df, SPEC, m=5, goal=r.value.exits[0]["goal"]).contrasts
    with pytest.raises(T.TrialRefused):
        T.analysis_sets(df, SPEC) and T.baseline_table(df, SPEC, ["age"], goal="prediction")


def test_covariates_chosen_after_the_data_are_refused_from_the_primary_analysis():
    df = _parallel()
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, SPEC, covariates_prespecified=False)
    assert r.value.term == "post hoc covariate adjustment"
    _runs(lambda **k: T.estimate(df, SPEC, **k), r.value, covariates_prespecified=False)
    secondary = T.estimate(df, SPEC, covariates_prespecified=False, secondary=True)
    assert not secondary.causal and not secondary.covariates_prespecified
    # the arms are still the randomized arms: what is lost is a model fixed in advance, not
    # randomization
    assert "still the randomized arms" in secondary.says
    assert "chosen after the data were seen" in secondary.says
    assert "not protected by randomization" not in secondary.says
    assert "not read as the trial's effect" in secondary.says


def test_adherence_or_the_arm_received_cannot_be_an_adjustment_term():
    df = _parallel()
    for col in ("adherent", "received"):
        spec = T.TrialSpec(arm="arm", control="ctl", outcome="y", covariates=(col,),
                           adherent="adherent", received="received")
        with pytest.raises(T.TrialRefused) as r:
            T.estimate(df, spec)
        assert r.value.term == "post-randomization covariate"


def test_the_cluster_trial_refusals_and_their_exits():
    df = _cluster()
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", covariates=("x",),
                       trial_design="cluster_randomized_trial", cluster="cluster")
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, spec, cluster_method="mixed")
    assert r.value.term == "Kenward–Roger for a binary outcome"
    _runs(lambda **k: T.estimate(df, spec, **k), r.value)
    mixed = df.copy()
    mixed.loc[mixed.index[0], "arm"] = "trt" if mixed.arm.iloc[0] == "ctl" else "ctl"
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(mixed, CSPEC)
    assert r.value.term == "arm varies within cluster"
    parallel = r.value.exits[0]["decision"]
    assert parallel == {"kind": "set_design", "design": "parallel_trial"}
    T.estimate(mixed, T.TrialSpec(arm="arm", control="ctl", outcome="y", covariates=("x",),
                                  trial_design=parallel["design"]))
    few = df[df.cluster.isin([1, 2, 3])]
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(few, CSPEC)
    assert r.value.term == "too few clusters"
    with pytest.raises(T.TrialRefused) as r:
        T.tipping_point(df.assign(y=df.y.where(df.index % 9 > 0)), CSPEC)
    assert r.value.term == "multilevel imputation for a cluster-randomized trial"
    small = _cluster(K=8)
    assert any("Only 8 clusters" in c for c in T.estimate(small, CSPEC).concerns)


def test_the_per_protocol_set_is_refused_without_adherence_and_for_the_tipping_point():
    df = _parallel()
    bare = T.TrialSpec(arm="arm", control="ctl", outcome="y")
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, bare, analysis_set="per_protocol")
    _runs(lambda **k: T.estimate(df, bare, **k), r.value)
    with pytest.raises(T.TrialRefused) as r:
        T.tipping_point(df, SPEC, analysis_set="per_protocol")
    _runs(lambda **k: T.tipping_point(df, SPEC, m=5, **k), r.value)
    complete = df.dropna(subset=["y"])
    with pytest.raises(T.TrialRefused) as r:
        T.tipping_point(complete, SPEC)
    assert r.value.term == "no missing outcomes"


def test_a_measure_that_does_not_fit_the_outcome_is_refused_with_the_one_that_does():
    df = _parallel()
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, SPEC, measure="risk_ratio")
    _runs(lambda **k: T.estimate(df, SPEC, **k), r.value)
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, BSPEC, measure="mean_difference")
    _runs(lambda **k: T.estimate(df, BSPEC, **k), r.value)


def test_separation_is_refused_in_plain_words():
    """Re-pinned (claims backtest B-281): this test had everyone at site c have the event and
    held the effect refused with the plain reason; a site the data separate is refused now
    naming the site and its people (the test after this one). Here the arm separates: when
    everyone in the control arm had the event its risk is 100%, and the refusal says so."""
    df = _parallel(seed=21)
    df.loc[(df.arm == "ctl") & df.event.notna(), "event"] = 1.0
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, BSPEC)
    assert r.value.term == "separation in the logistic model"
    assert "Everyone in ctl had the event" in str(r.value) and "100%" in str(r.value)


def _ye_by_hand(y: np.ndarray, arm: np.ndarray, cf: np.ndarray, mu: np.ndarray,
                strata: np.ndarray) -> np.ndarray:
    """Ye, Shao, Yi & Zhao's (2023) covariance of the standardized risks of the arms (the columns
    of ``cf``), written out from their Theorem 1 as R's ``beeca`` ``varcov_ye`` computes it (n − 1
    denominators), less the stratified randomization's term Σ_z (n_z/n) R_z(diag π − ππᵀ)R_z, all
    over n."""
    n, K = cf.shape
    pi = np.array([np.mean(arm == j) for j in range(K)])
    V = np.zeros((K, K))
    for j in range(K):
        yj, cj = y[arm == j], cf[arm == j, j]
        V[j, j] = ((np.var(yj, ddof=1) - 2 * np.cov(yj, cj)[0, 1] + np.var(cf[:, j], ddof=1))
                   / pi[j] + 2 * np.cov(yj, cj)[0, 1] - np.var(cf[:, j], ddof=1))
        for k in set(range(K)) - {j}:
            V[j, k] = (np.cov(yj, cf[arm == j, k])[0, 1]
                       + np.cov(y[arm == k], cf[arm == k, j])[0, 1]
                       - np.cov(cf[:, j], cf[:, k])[0, 1])
    for z in np.unique(strata):
        R = np.diag([np.mean((y - mu)[(strata == z) & (arm == j)]) / pi[j] for j in range(K)])
        V -= R @ (np.diag(pi) - np.outer(pi, pi)) @ R * np.mean(strata == z)
    return V / n


def _separated_said(said: T.TrialRefused, terms: str, people: int, n: int) -> None:
    """The refusal of a trial the data separate partly (claims backtest B-281, its sixth review):
    it names the terms that separate them and counts the people, with fewer covariates and
    describing the outcomes as its exits."""
    assert said.term == "separation in the logistic model", str(said)
    assert (f"From {terms}, the data predict exactly whether {people} of the {n} people had the "
            f"event, so the logistic model has no finite fit for them") in str(said), str(said)
    assert "set by where the fit stopped, not by the data" in str(said), str(said)
    assert [e.get("covariates") for e in said.exits] == [[], None]
    assert said.exits[-1]["describe"] is True


def test_a_group_with_every_event_is_refused_naming_it_and_an_empty_arm_says_why():
    """Claims backtest B-281. Everyone at site c had the event, so the logistic model separates:
    site c's coefficient has no finite value. Known truth: site c's people are the ones
    separated, and no one else is, since statsmodels' ``Logit`` converges on the other sites'
    people, which a direction separating anyone there would not allow.

    Re-pinned (its sixth review: the fallback). At the fit's limit the standardized risks are
    finite (Lesaffre & Albert 1989, *J R Stat Soc B* 51:109), and this test had held them to
    statsmodels' fit on the other sites with site c's risks at 1; reading that fit is withdrawn,
    since ties to float rounding made it report wrong numbers elsewhere. The effect is refused,
    naming site c and counting its people, with fewer covariates and describing the outcomes as
    its exits. The effect had been refused before B-281 because "no finite estimate exists".

    When no one in an arm had the event, its risk is 0% and the risk difference is still finite
    (minus the other arm's risk), but Ye et al.'s variance, built from each arm's spread of
    outcomes around its fitted risks, counts no uncertainty in that arm: the effect is refused
    for that reason, and fewer covariates is no way out (the arm alone separates too)."""
    import statsmodels.api as sm

    df = _parallel(seed=21)
    df.loc[df.site == "c", "event"] = 1.0
    rand = df.arm.notna()
    data = _filled(df[rand], "age", pd.Series(True, index=df[rand].index))
    data = data[data.event.notna()].reset_index(drop=True)
    y = data.event.to_numpy()
    at_c = (data.site == "c").to_numpy()
    X = np.column_stack([np.ones(len(data)), data.arm == "trt", data.site == "b", data.base,
                         data.age, data.age_miss]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rest = sm.Logit(y[~at_c], X[~at_c]).fit(method="newton", tol=1e-14, maxiter=100, disp=0)
    assert rest.mle_retvals["converged"]
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, BSPEC)
    _separated_said(r.value, "site[c]", int(at_c.sum()), len(data))

    none = _parallel(seed=21)
    none.loc[(none.arm == "trt") & none.event.notna(), "event"] = 0.0
    for spec in (BSPEC, T.TrialSpec(arm="arm", control="ctl", outcome="event")):
        with pytest.raises(T.TrialRefused) as refused:
            T.estimate(none, spec)
        said = str(refused.value)
        assert "trt" in said and "no finite estimate exists" not in said, said
        assert not any("covariates" in e for e in refused.value.exits)


@pytest.mark.parametrize("column, mean, sd", [("ph", 7.4, 0.05), ("year", 2021.0, 1.5)])
def test_a_covariate_far_from_zero_leaves_a_separated_trial_as_it_is_centered(column, mean, sd):
    """Claims backtest B-281, its review. A logistic model's fitted risks do not depend on where a
    covariate's zero is: shifting it moves only the intercept. So the site-c trial above, with a
    baseline pH of 7.4 ± 0.05 (or a year of 2021 ± 1.5) beside age, gives the same standardized
    risks, difference, ratio and standard errors as the same trial with that column centered by
    hand. The limit fit had scaled its columns without centering them, so a column some 150 SDs
    from zero was nearly the intercept's twin; their near-flat direction loaded on the arm, and
    the trial was refused as if an arm had separated.

    Re-pinned (its sixth review: the fallback): a partly separated trial is refused, so the two
    trials are held to the same refusal, naming site c and counting its people, never the arm,
    where they had been held to the same standardized risks."""
    df = _parallel(seed=21)
    df.loc[df.site == "c", "event"] = 1.0
    df[column] = mean + np.random.default_rng(5).normal(0, sd, len(df))
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("age", column), baseline="base", adherent="adherent")
    said = []
    for frame in (df, df.assign(**{column: df[column] - df[column].mean()})):
        with pytest.raises(T.TrialRefused) as r:
            T.estimate(frame, spec)
        said.append(str(r.value))
        analyzed = frame[frame.arm.notna() & frame.event.notna()]
        _separated_said(r.value, "site[c]", int((analyzed.site == "c").sum()), len(analyzed))
        assert "arm[" not in str(r.value), str(r.value)
    assert said[0] == said[1]


def _small_trial(seed: int, n: int = 30) -> pd.DataFrame:
    """A small trial: 30 people, a site and three normal covariates, six terms in all (claims
    backtest B-281, its second review); about one seed in 25 is completely separated."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
                       "x3": rng.normal(0, 1, n)})
    lp = -0.2 + 0.8 * (df.arm == "trt") + 1.2 * df.x1 + 0.8 * df.x2 + 0.5 * df.x3
    df["event"] = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    return df


SMALL = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                    covariates=("x1", "x2", "x3"))


def _small_design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df.arm == "trt", df.site == "b", df.x1, df.x2,
                            df.x3]).astype(float)


def _separated_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The people some direction of the coefficients separates (Albert & Anderson 1984,
    *Biometrika* 71:1), by one linear program: maximize Σ tᵢ over 0 ≤ tᵢ ≤ sᵢxᵢᵀd with every
    sᵢxᵢᵀd ≥ 0, d free and sᵢ = ±1 by the outcome. The sum of the directions that reach each
    separable person strictly reaches them all, so scaled up it puts each such tᵢ at 1, and every
    other tᵢ is 0. Everyone: complete separation. The columns are standardized first, which moves
    no one in or out."""
    from scipy.optimize import linprog

    n, p = X.shape
    sd = X.std(axis=0)
    Z = np.where(sd > 0, (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 1.0)
    A = (2 * y - 1)[:, None] * Z
    res = linprog(np.r_[np.zeros(p), -np.ones(n)],
                  A_ub=np.block([[-A, np.eye(n)], [-A, np.zeros((n, n))]]), b_ub=np.zeros(2 * n),
                  bounds=[(None, None)] * p + [(0, 1)] * n, method="highs")
    assert res.status == 0, res.message
    return res.x[p:] > 0.5


def test_complete_separation_is_refused_with_no_fit_left():
    """Claims backtest B-281, its second review. When the arm and the covariates predict every
    outcome exactly (complete separation; Albert & Anderson 1984), every coefficient runs off at
    once and every fitted risk reaches 0 or 1 together, so no fit is left to rest the standardized
    risks on: the risk difference and its standard error are wherever Newton's method stopped. Of
    300 small trials (30 people, six terms), the linear program above finds 13 completely
    separated, seed 6 among them. Each is refused for that reason, with fewer covariates and
    describing the outcomes as its exits, and every other trial is reported. A trial whose outcome
    is whether age is over 50 is refused too, at any origin of age, and fewer covariates runs.

    On the prior code 12 of the 13 were reported, 11 with no note (seed 6: a risk difference of
    +0.46, SE 0.066, the arm's coefficient +251) and one saying site b separates all 30 people,
    and one was refused as if an arm had separated; three of the four age trials were reported,
    their risk difference near −0.01 moving when age's origin moved, and one refused as an arm.

    Re-pinned (its sixth review: the fallback): of the other trials, the one the program above
    finds partly separated (seed 9, 9 people at site b) is refused for separation, not complete
    separation, counting those people, where it had been reported."""
    complete, partly = [], []
    for seed in range(300):
        df = _small_trial(seed)
        sep = _separated_rows(_small_design(df), df.event.to_numpy())
        if not sep.any():
            T.estimate(df, SMALL)
            continue
        if not sep.all():
            partly.append(seed)
            with pytest.raises(T.TrialRefused) as r:
                T.estimate(df, SMALL)
            assert r.value.term == "separation in the logistic model", (seed, str(r.value))
            assert f"exactly whether {int(sep.sum())} of the 30 people" in str(r.value)
            continue
        complete.append(seed)
        with pytest.raises(T.TrialRefused) as r:
            T.estimate(df, SMALL)
        assert r.value.term == "complete separation in the logistic model", (seed, str(r.value))
        assert "predict every person's outcome exactly (all 30 of them)" in str(r.value)
        assert [e.get("covariates") for e in r.value.exits] == [[], None]
        assert r.value.exits[-1]["describe"] is True
    assert 6 in complete and len(complete) == 13, complete
    assert partly == [9], partly
    T.estimate(_small_trial(6), T.TrialSpec(**{**SMALL.__dict__, "covariates": ()}))

    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("age",))
    for seed in range(4):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], 200),
                           "site": rng.choice(["a", "b"], 200), "age": rng.normal(50, 10, 200)})
        df["event"] = (df.age > 50).astype(float)
        for shift in (0.0, 7000.0):
            with pytest.raises(T.TrialRefused) as r:
                T.estimate(df.assign(age=df.age + shift), spec)
            assert r.value.term == "complete separation in the logistic model", str(r.value)
            assert "(all 200 of them)" in str(r.value)
        T.estimate(df, T.TrialSpec(**{**spec.__dict__, "covariates": ()}))


def _plane_trial(seed: int, n: int = 60, left: int = 12) -> pd.DataFrame:
    """Everyone's outcome is whether x1 + x2 > 0, but for ``left`` people on that plane
    (x2 = −x1), whose outcomes alternate: the data separate all but some of them (or everyone,
    when the other terms split them too), and the people left may or may not settle the arm."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
                       "x3": rng.normal(0, 1, n)})
    df["event"] = (df.x1 + df.x2 > 0).astype(float)
    on = df.index[:left]
    df.loc[on, "x2"] = -df.loc[on, "x1"]
    df.loc[on, "event"] = (np.arange(left) % 2).astype(float)
    return df


def test_a_partly_separated_trial_is_refused_counting_the_people_the_data_separate():
    """Claims backtest B-281, its second review. Known truth: the linear program above finds the
    separated people.

    1. Sixty trials where everyone but 12 people on a plane is separated, or everyone is: each
       is refused, for complete separation when everyone is separated and otherwise for
       separation, counting the people the program finds, with fewer covariates as its first
       exit.
    2. The site-c trial above (everyone there had the event) with a weight site c recorded in
       grams, the others in kilograms (SD 2 kg), over 40 seeds: the program finds site c's people
       and only them, and each trial is refused, naming site c and counting them.

    On the code before the second review the completely separated trials of (1) were reported,
    refused as an arm, or failed on a negative variance, and 32 of the 40 trials of (2) were
    refused as if an arm had separated.

    Re-pinned (its sixth review: the fallback). This test had held each partly separated trial of
    (1) whose people left settle the arm, and each trial of (2), to statsmodels' fit on the people
    the program leaves, with the separated people at their own outcome under every arm (Lesaffre &
    Albert 1989), and the rest of (1) to a refusal. Reading the people left from a fit of their
    own is withdrawn, since ties to float rounding made it report wrong numbers elsewhere, so
    every partly separated trial is refused."""
    tally = {"complete": 0, "partly": 0}
    for seed in range(60):
        df = _plane_trial(seed)
        sep = _separated_rows(_small_design(df), df.event.to_numpy())
        assert sep.any(), seed
        tally["complete" if sep.all() else "partly"] += 1
        with pytest.raises(T.TrialRefused) as r:
            T.estimate(df, SMALL)
        assert ("complete " in r.value.term) == sep.all(), (seed, str(r.value))
        assert r.value.exits[0]["covariates"] == []
        if not sep.all():
            assert r.value.term == "separation in the logistic model", (seed, str(r.value))
            assert f"exactly whether {int(sep.sum())} of the 60 people" in str(r.value), seed
    assert min(tally.values()) >= 5, tally

    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("age", "weight"), baseline="base", adherent="adherent")
    for seed in range(40):
        df = _parallel(seed=seed)
        df.loc[df.site == "c", "event"] = 1.0
        kg = 75 + np.random.default_rng(seed + 7).normal(0, 2, len(df))
        df["weight"] = np.where(df.site == "c", 1000 * kg, kg)
        rand = df.arm.notna()
        data = _filled(df[rand], "age", pd.Series(True, index=df[rand].index))
        data = data[data.event.notna()].reset_index(drop=True)
        X = np.column_stack([np.ones(len(data)), data.arm == "trt", data.site == "b",
                             data.site == "c", data.base, data.age, data.age_miss,
                             data.weight]).astype(float)
        at_c = (data.site == "c").to_numpy()
        assert (_separated_rows(X, data.event.to_numpy()) == at_c).all(), seed
        with pytest.raises(T.TrialRefused) as r:
            T.estimate(df, spec)
        _separated_said(r.value, "site[c]", int(at_c.sum()), len(data))


def _milligrams(seed: int, scale: float, n: int = 240) -> pd.DataFrame:
    """Everyone at site c had the event, which otherwise rises with the arm and with weight, 0.1
    on the log odds a kilogram; site c recorded weight ``scale`` times larger (at 1e6, milligrams
    where the others wrote kilograms)."""
    rng = np.random.default_rng(7000 + seed)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b", "c"], n)})
    kg = 75 + rng.normal(0, 2, n)
    df["weight"] = np.where(df.site == "c", scale * kg, kg)
    lp = -0.4 + 0.6 * (df.arm == "trt") + 0.1 * (kg - 75)
    df["event"] = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    df.loc[df.site == "c", "event"] = 1.0
    return df


MG = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                 covariates=("weight",))


def test_a_weight_recorded_a_million_times_larger_where_everyone_had_the_event_is_refused(
        monkeypatch):
    """Claims backtest B-281, its fourth review. Everyone at site c had the event, and site c
    recorded weight 1e6 to 1e7 times larger than sites a and b, where it matters (0.1 a
    kilogram). Known truth: site c is separated (its indicator puts everyone there at 1) and no
    one else is, since statsmodels' ``Logit`` converges on sites a and b alone, which a direction
    separating anyone there would not allow. When the linear program that finds the separated
    people fails, the site-c trial above is refused, not read from a fit.

    On the code before the fourth review 54 of the 60 were reported wrong, the risk difference
    off by up to 0.02 (about 0.4 SE), under a note saying the rest of the model was the fit to
    everyone else: the limit fit's Newton step, read on the information, squared the weight's
    small spread among sites a and b into a direction it could not move along, and stopped short.
    Reading those people from a fit of their own but taking the separated people from fitted
    risks within 1e-9 of 0 or 1 still refused 4 of the 60: in 3 those risks missed a site-c
    person, and in 1 the limit fit did not converge.

    Re-pinned (its sixth review: the fallback). This test had held each of the 60 trials (1e6,
    3e6 and 1e7, 20 seeds each) to statsmodels' fit to sites a and b with site c's risks at 1
    (Lesaffre & Albert 1989), or to a refusal, no more than 2 refused. Reading the people left
    from a fit of their own is withdrawn, since ties to float rounding made it report wrong
    numbers elsewhere, so each of the 60 is refused, naming site c and counting its people."""
    import scipy.optimize
    import statsmodels.api as sm

    for scale in (1e6, 3e6, 1e7):
        for seed in range(20):
            df = _milligrams(seed, scale)
            y = df.event.to_numpy()
            at_c = (df.site == "c").to_numpy()
            X = np.column_stack([np.ones(len(df)), df.arm == "trt", df.site == "b",
                                 df.weight]).astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rest = sm.Logit(y[~at_c], X[~at_c]).fit(method="newton", tol=1e-12, maxiter=100,
                                                         disp=0)
            assert rest.mle_retvals["converged"], (scale, seed)
            with pytest.raises(T.TrialRefused) as r:
                T.estimate(df, MG)
            _separated_said(r.value, "site[c]", int(at_c.sum()), len(df))

    def fails(*args, **kwargs):
        return scipy.optimize.OptimizeResult(status=4, x=None, success=False,
                                             message="numerical difficulties")

    monkeypatch.setattr(scipy.optimize, "linprog", fails)
    site_c = _parallel(seed=21)
    site_c.loc[site_c.site == "c", "event"] = 1.0
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(site_c, BSPEC)
    assert r.value.term == "separation in the logistic model"
    assert "runs to 0% or 100%" in str(r.value), str(r.value)


def _exact_fit(X: np.ndarray, y: np.ndarray) -> list | None:
    """The logistic model's maximum-likelihood coefficients on the float data X, in 60-digit
    decimal arithmetic (Python's ``decimal``: exp and ln correctly rounded, the data converted
    exactly), by Newton's method with step halving from zero; None when it does not converge. The
    log likelihood is concave, so where Newton's step falls below 1e-30 of the coefficients, with
    the gradient below 1e-30, the fit is its finite maximum and no one is separated; along a
    direction that separates someone the step never shrinks so."""
    from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, localcontext

    n, p = X.shape
    with localcontext(Context(prec=60, Emax=MAX_EMAX, Emin=MIN_EMIN)):
        one, tiny = Decimal(1), Decimal("1e-30")
        rows = [[Decimal(float(v)) for v in row] for row in X]
        ys = [Decimal(int(v)) for v in y]

        def etas(b):
            return [sum(r[j] * b[j] for j in range(p)) for r in rows]

        def loglik(e):
            return sum(ys[i] * v - (v + (one + (-v).exp()).ln() if v > 0 else (one + v.exp()).ln())
                       for i, v in enumerate(e))

        b = [Decimal(0)] * p
        e = etas(b)
        cur = loglik(e)
        try:
            for _ in range(300):
                m = [one / (one + (-v).exp()) for v in e]
                g = [sum((ys[i] - m[i]) * rows[i][j] for i in range(n)) for j in range(p)]
                H = [[sum(m[i] * (one - m[i]) * rows[i][j] * rows[i][k] for i in range(n))
                      for k in range(p)] + [g[j]] for j in range(p)]
                for c in range(p):  # Gaussian elimination, partial pivoting
                    piv = max(range(c, p), key=lambda r: abs(H[r][c]))
                    H[c], H[piv] = H[piv], H[c]
                    for r in range(c + 1, p):
                        f = H[r][c] / H[c][c]
                        H[r] = [a - f * h for a, h in zip(H[r], H[c])]
                step = [Decimal(0)] * p
                for c in reversed(range(p)):
                    step[c] = (H[c][p] - sum(H[c][k] * step[k] for k in range(c + 1, p))) / H[c][c]
                if max(abs(v) for v in g) < tiny and all(
                        abs(s) < tiny * (1 + abs(v)) for s, v in zip(step, b)):
                    return b
                t = one
                while True:
                    nb = [v + t * s for v, s in zip(b, step)]
                    ne = etas(nb)
                    new = loglik(ne)
                    if new >= cur or t < Decimal("1e-20"):
                        break
                    t /= 2
                b, e, cur = nb, ne, new
        except ArithmeticError:  # a pivot of zero: the information has lost a direction
            return None
    return None


def _exact_effect(X: np.ndarray, y: np.ndarray, b: list, strata: np.ndarray
                  ) -> tuple[float, float]:
    """The standardized risk difference of arm 1 (column 1 of X) against arm 0 and Ye et al.'s
    standard error written out above, at the coefficients b of :func:`_exact_fit`, each person's
    risk under either arm computed in the same arithmetic."""
    from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, localcontext

    arm = X[:, 1].astype(int)
    cf = np.empty((len(y), 2))
    with localcontext(Context(prec=60, Emax=MAX_EMAX, Emin=MIN_EMIN)):
        for i, row in enumerate(X):
            for k in (0, 1):
                v = sum(Decimal(float(x)) * c for x, c in zip(np.r_[row[0], k, row[2:]], b))
                cf[i, k] = float(1 / (1 + (-v).exp()))
    V = _ye_by_hand(y, arm, cf, cf[np.arange(len(y)), arm], strata)
    return cf[:, 1].mean() - cf[:, 0].mean(), math.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])


def _bmi_trial(seed: int, m: int, n: int = 200) -> pd.DataFrame:
    """Everyone's diagnosis is whether their body-mass index, weight over height squared, is over
    30, with weight recorded to 0.1 kg and no one within 0.08 of 30, but for the first ``m``, whose
    weight was recorded as 30·h², so that their computed BMI is 30 give or take one unit in the
    last place, and whose diagnoses are mixed, likelier in the treated arm."""
    rng = np.random.default_rng(99500 + seed)
    h = rng.uniform(1.5, 2.0, n)
    b = rng.normal(28, 5, n)
    b = np.where(np.abs(b - 30) < 0.1, 30 + np.sign(b - 30 + 1e-9) * 0.1, b)
    w = np.round(b * h ** 2, 1)
    w[:m] = 30 * h[:m] ** 2
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "bmi": w / h ** 2})
    df["event"] = (df.bmi > 30).astype(float)
    lp = -0.2 + 1.0 * (df.arm[:m] == "trt")
    df.loc[:m - 1, "event"] = (rng.random(m) < 1 / (1 + np.exp(-lp))).astype(float)
    return df


def _x_design(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df.arm == "trt", df.site == "b",
                            df[column]]).astype(float)


def _refused_or_exact(df: pd.DataFrame, spec: T.TrialSpec, column: str) -> str:
    """The trial refused, or reported as :func:`_exact_fit` with no note that the data separate
    anyone (a finite fit exists, so no one is), the risk difference within 1e-6 of its standard
    error and the standard error to 1e-6; which it was."""
    try:
        r = T.estimate(df, spec)
    except T.TrialRefused as e:
        assert e.term == "separation in the logistic model", str(e)
        return "refused"
    X, y = _x_design(df, column), df.event.to_numpy()
    b = _exact_fit(X, y)
    assert b is not None, "reported, but no finite fit is certified in 60 digits"
    rd, se = _exact_effect(X, y, b, pd.factorize(df.site)[0])
    got = r.contrast(measure="risk_difference")
    assert abs(got.estimate - rd) <= 1e-6 * se, (got.estimate, rd, se)
    assert got.se == pytest.approx(se, rel=1e-6)
    assert not any("separate the outcome" in c for c in r.concerns), r.concerns
    return "reported"


def test_a_covariate_the_people_left_share_only_to_rounding_is_refused_or_read_exactly():
    """Claims backtest B-281, its fifth review. A covariate computed from two others, as a
    body-mass index is from weight and height, can tie people at a threshold only to float
    rounding. Here everyone's diagnosis is whether BMI is over 30, but for 12 or 24 people whose
    computed BMI is 30 give or take one unit in the last place (a spread of 7.1e-15), and whose
    diagnoses are mixed; then the same trials with those people all at site a, which leaves site
    b's coefficient free among them. Known truth: the maximum-likelihood fit on the float data in
    60-digit arithmetic (:func:`_exact_fit`), each arm's risks and Ye et al.'s variance written
    out above; where that fit is not certified (the rounding itself may separate some of them in
    exact arithmetic, or the fit lie beyond 60 digits' reach), no report can be checked, and only
    a refusal passes. Over 40 trials (10 seeds at each size, at two sites and at one) each is
    refused, or reported as that fit with no note that anyone is separated.

    On the prior code 10 of the 20 at two sites were reported wrong, by 0.05 to 1.06 standard
    errors, with no note: the linear program's tolerance accepted the tie as a separation, the
    people left were centered and scaled on themselves, so that BMI's rounding became a
    covariate of unit spread, and with no direction left free nothing refused. At 24 people, seed
    4, the risk difference was 0.000 where the fit gives 0.024 (SE 0.023), the coefficient on BMI
    −4.3e14. At one site 5 of the 20 were reported wrong, by 0.05 to 0.96 standard errors, under a
    note that site b separates the outcome, and 6 more were reported where no fit is certified:
    site b left a direction free, so only the rounding's spread tells such a trial apart."""
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("bmi",))
    for one_site in (False, True):
        for m in (12, 24):
            for seed in range(10):
                df = _bmi_trial(seed, m)
                assert 0 < np.ptp(df.bmi[:m]) < 1e-14, (m, seed)
                if one_site:
                    df = df.assign(site=np.where(df.index < m, "a", df.site))
                _refused_or_exact(df, spec, "bmi")


def _tied(sigma: float, seed: int, n: int = 160, m: int = 30) -> pd.DataFrame:
    """Everyone's outcome is whether x is above 0, with |x| at least 0.05, but for the first ``m``,
    at x = 0 up to noise of size ``sigma``, whose outcomes are mixed, likelier in the treated
    arm."""
    rng = np.random.default_rng(99000 + seed)
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "x": np.sign(x) * (0.05 + np.abs(x))})
    df["event"] = (df.x > 0).astype(float)
    df.loc[:m - 1, "x"] = sigma * rng.normal(0, 1, m)
    lp = -0.2 + 1.0 * (df.loc[:m - 1, "arm"] == "trt")
    df.loc[:m - 1, "event"] = (rng.random(m) < 1 / (1 + np.exp(-lp))).astype(float)
    return df


def test_a_tied_group_is_refused_and_one_tied_up_to_noise_read_as_noise_or_refused():
    """Claims backtest B-281, its fifth review. Everyone's outcome is whether x is above 0, with
    |x| at least 0.05, but for 30 people at x = 0 up to noise of size σ, whose outcomes are mixed.

    - σ = 0: x separates the other 130, and the 30 are left, since statsmodels' ``Logit`` on
      them converges, which a direction separating anyone there would not allow. Each of 10
      trials is refused, naming x and counting the 130.
    - σ = 1e-6: no one is separated, and the truth is the finite maximum-likelihood fit in
      60-digit arithmetic (:func:`_exact_fit`). Each of 10 is reported as it, with no note.
    - σ from 1e-12 to 1e-8, where float arithmetic hardly tells the noise from a tie: the truth
      is that fit too, and each of 30 is refused or reported as it, with no note; so is each of
      30 more with the 30 all at site a, which leaves site b's coefficient free among them.
      Where that fit is not certified, only a refusal passes.

    On the code before the fifth review 15 of the 30 at σ from 1e-12 to 1e-8 were reported
    wrong, by 0.01 to 0.06 standard errors, with no note: the linear program had named the 130
    separated, and the 30's noise was scaled up to a covariate of unit spread. At one site 17 of
    the 30 were reported wrong, by 0.001 to 0.06 standard errors, under a note that the data
    separate the outcome on site b for 130 people, and 9 more where no fit is certified: the 30
    left site b free, so only their distance from the program's separating plane tells such a
    trial apart.

    Re-pinned (its sixth review: the fallback). This test had held each trial at σ = 0 to
    statsmodels' fit on the 30 with the 130 at their own outcomes (Lesaffre & Albert 1989) and Ye
    et al.'s variance written out above. Reading the people left from a fit of their own is
    withdrawn, since ties to float rounding made it report wrong numbers, so each is refused."""
    import statsmodels.api as sm

    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("x",))
    for seed in range(10):
        df = _tied(0.0, seed)
        y, tied = df.event.to_numpy(), np.arange(len(df)) < 30
        X = _x_design(df, "x")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rest = sm.Logit(y[tied], X[tied, :3]).fit(method="newton", tol=1e-12, maxiter=100,
                                                      disp=0)
        assert rest.mle_retvals["converged"], seed
        with pytest.raises(T.TrialRefused) as r:
            T.estimate(df, spec)
        _separated_said(r.value, "x", 130, 160)
    for seed in range(10):
        assert _refused_or_exact(_tied(1e-6, seed), spec, "x") == "reported", seed
    for one_site in (False, True):
        for sigma in (1e-12, 1e-10, 1e-8):
            for seed in range(10):
                df = _tied(sigma, seed)
                if one_site:
                    df = df.assign(site=np.where(df.index < 30, "a", df.site))
                _refused_or_exact(df, spec, "x")


def test_people_the_program_names_are_refused_whatever_the_rest_leave_free(monkeypatch):
    """Claims backtest B-281, its fifth review. Here no one is separated (statsmodels' ``Logit``
    converges on everyone), but the coefficient on a pH of 7.4 ± 0.02 is near 55, past
    :func:`_logistic`'s bound, so the program is asked; as the program is, the trial is reported.
    Made to name one person who had the event, the trial is refused, not reported with that
    person's risk at 100% under either arm.

    On the code before the fifth review it was reported with no note, its risk difference 0.0795
    where it is 0.0828 as the program is.

    Re-pinned (its sixth review: the fallback). The refusal had been the plain separation reason,
    since the people left leave no direction free to separate that person, so that the program
    and the columns disagree. Now every trial the program names anyone in is refused for
    separation, counting the people it names, whatever the people left leave free; the program
    is made to name the person here in its own solution."""
    import scipy.optimize
    import statsmodels.api as sm

    rng = np.random.default_rng(91000)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], 150), "site": rng.choice(["a", "b"], 150),
                       "ph": 7.4 + rng.normal(0, 0.02, 150)})
    lp = -0.3 + 0.7 * (df.arm == "trt") + 60 * (df.ph - 7.4)
    df["event"] = (rng.random(150) < 1 / (1 + np.exp(-lp))).astype(float)
    fit = sm.Logit(df.event.to_numpy(), _x_design(df, "ph")).fit(method="newton", tol=1e-12,
                                                                 disp=0)
    assert fit.mle_retvals["converged"] and fit.params[-1] > 25
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("ph",))
    T.estimate(df, spec)
    solve, who = scipy.optimize.linprog, int(np.argmax(df.event.to_numpy()))

    def one_named(c, *args, **kwargs):
        res = solve(c, *args, **kwargs)
        x = res.x.copy()
        x[np.flatnonzero(np.asarray(c) == -1)[who]] = 1.0
        return scipy.optimize.OptimizeResult({**res, "x": x})

    monkeypatch.setattr(scipy.optimize, "linprog", one_named)
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, spec)
    assert r.value.term == "separation in the logistic model"
    assert "the data predict exactly whether 1 of the 150 people had the event" in str(r.value), \
        str(r.value)


def _plane_ulp(seed: int, ulp: bool, n: int = 200, m: int = 24) -> pd.DataFrame:
    """Everyone's outcome is whether x1 + x2 > 0, with |x1 + x2| at least 0.05, but for the first
    ``m``, on the plane x2 = −x1, whose outcomes are mixed, likelier in the treated arm; with
    ``ulp``, each of their x2 one unit in the last place off it, so that float arithmetic's
    x1 + x2 is not 0 (the fifth review's re-review: BMI's tie, spread over two columns)."""
    rng = np.random.default_rng(78000 + seed)
    x1 = rng.normal(0, 1, n)
    s = rng.normal(0, 1, n)
    s = np.sign(s) * (0.05 + np.abs(s))
    x2 = s - x1
    sign = rng.choice([-1.0, 1.0], m)
    x2[:m] = -x1[:m]
    if ulp:
        x2[:m] = np.nextafter(x2[:m], sign * np.inf)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "x1": x1, "x2": x2})
    df["event"] = (s > 0).astype(float)
    lp = -0.2 + 1.0 * (df.arm[:m] == "trt")
    df.loc[:m - 1, "event"] = (rng.random(m) < 1 / (1 + np.exp(-lp))).astype(float)
    return df


def _summed_score(seed: int, m: int, n: int = 200) -> pd.DataFrame:
    """A diagnosis made from a summed score: two subscores recorded to 0.1 and the diagnosis
    whether they sum to more than 30, but for the first ``m``, whose subscores sum to 30 in
    decimal, with mixed diagnoses (the fifth review's re-review)."""
    rng = np.random.default_rng(77000 + seed)
    x1 = np.round(rng.uniform(5, 25, n), 1)
    s = rng.normal(29, 4, n)
    s = np.where(np.abs(s - 30) < 0.15, 30 + np.sign(s - 30 + 1e-9) * 0.15, s)
    x2 = np.round(s - x1, 1)
    x2[:m] = np.round(30 - x1[:m], 1)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "x1": x1, "x2": x2})
    df["event"] = (np.round(df.x1 + df.x2, 6) > 30).astype(float)
    lp = -0.2 + 1.0 * (df.arm[:m] == "trt")
    df.loc[:m - 1, "event"] = (rng.random(m) < 1 / (1 + np.exp(-lp))).astype(float)
    return df


def test_a_trial_the_program_finds_anyone_separated_in_is_refused_never_reported(monkeypatch):
    """Claims backtest B-281, its sixth review: the fallback. When the linear program that finds
    the separated people names some of them but not everyone, the effect is refused, never
    reported: the data predict those people's outcomes exactly, so the logistic model has no
    finite fit for them, and reading the rest from a fit of their own (Lesaffre & Albert 1989) is
    withdrawn, since ties to float rounding, on one covariate and then on a combination of them,
    made it report a wrong risk difference. Every program the trial core solves is watched, its
    tᵢ read as the solver returns them, across the generators the earlier rounds built: the
    site-c trial, at pH 7.4, at year 2021 and with weight in grams at site c; the milligram
    trials; the BMI trials, at two sites and at one; the tied and noisy-tie trials; the plane
    trials; the one-ulp plane and its control on the plane; and the summed score. A refusal
    that counts the people the program names counts them as the program does. The program names
    people in every trial of six of these sets, and in most of the rest; the tied trials at
    σ = 1e-6 are reported, as no one is separated there.

    On the code before this review 156 of the 257 trials in which the program named some people
    were reported: the 60 milligram trials, the 13 site-c trials, the 10 tied trials at σ = 0, 35
    of the 43 plane trials, the 20 one-ulp planes and controls, and 18 of the 19 summed scores.
    The one-ulp plane at seed 2 was reported 0.098 standard errors from the maximum-likelihood
    fit in 60-digit arithmetic, and a summed score 0.54."""
    import scipy.optimize

    solve = scipy.optimize.linprog
    named: list[tuple[int, int]] = []

    def watched(c, *args, **kwargs):
        res = solve(c, *args, **kwargs)
        if res.status == 0:
            t = res.x[np.asarray(c) == -1]
            named.append((int(np.sum(t > 0.5)), len(t)))
        return res

    monkeypatch.setattr(scipy.optimize, "linprog", watched)
    bmi = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                      covariates=("bmi",))
    x = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",), covariates=("x",))
    x12 = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                      covariates=("x1", "x2"))
    trials: list[tuple[str, pd.DataFrame, T.TrialSpec]] = []
    site_c = _parallel(seed=21)
    site_c.loc[site_c.site == "c", "event"] = 1.0
    trials.append(("site c", site_c, BSPEC))
    for column, mean, sd in (("ph", 7.4, 0.05), ("year", 2021.0, 1.5)):
        far = site_c.assign(**{column: mean + np.random.default_rng(5).normal(0, sd, len(site_c))})
        trials.append(("site c", far, T.TrialSpec(**{**BSPEC.__dict__,
                                                     "covariates": ("age", column)})))
    for seed in range(10):
        grams = _parallel(seed=seed)
        grams.loc[grams.site == "c", "event"] = 1.0
        kg = 75 + np.random.default_rng(seed + 7).normal(0, 2, len(grams))
        grams["weight"] = np.where(grams.site == "c", 1000 * kg, kg)
        trials.append(("site c", grams, T.TrialSpec(**{**BSPEC.__dict__,
                                                       "covariates": ("age", "weight")})))
    trials += [("milligrams", _milligrams(seed, scale), MG) for scale in (1e6, 3e6, 1e7)
               for seed in range(20)]
    for m in (12, 24):
        for seed in range(10):
            df = _bmi_trial(seed, m)
            trials += [("bmi", df, bmi),
                       ("bmi", df.assign(site=np.where(df.index < m, "a", df.site)), bmi)]
    trials += [("tied", _tied(0.0, seed), x) for seed in range(10)]
    trials += [("noise 1e-6", _tied(1e-6, seed), x) for seed in range(10)]
    for sigma in (1e-12, 1e-10, 1e-8):
        for seed in range(10):
            df = _tied(sigma, seed)
            trials += [("noise", df, x),
                       ("noise", df.assign(site=np.where(df.index < 30, "a", df.site)), x)]
    trials += [("plane", _plane_trial(seed), SMALL) for seed in range(60)]
    trials += [("one-ulp plane", _plane_ulp(seed, ulp), x12) for ulp in (True, False)
               for seed in range(10)]
    trials += [("summed score", _summed_score(seed, m), x12) for m in (12, 24)
               for seed in range(10)]

    tally: dict[str, list[int]] = {}
    for kind, df, spec in trials:
        named.clear()
        try:
            T.estimate(df, spec)
            said = None
        except T.TrialRefused as e:
            said = e
        partly = [(k, n) for k, n in named if 0 < k < n]
        seen = tally.setdefault(kind, [0, 0, 0])  # trials, the program naming people, reported
        seen[0] += 1
        seen[1] += bool(partly)
        seen[2] += said is None
        if not partly:
            continue
        assert said is not None, (kind, partly, "reported, though the program named people")
        if "the data predict exactly whether" in str(said):
            k, n = partly[-1]
            assert f"exactly whether {k} of the {n} people had the event" in str(said), str(said)
    for kind in ("site c", "milligrams", "bmi", "tied", "one-ulp plane"):
        assert tally[kind][1] == tally[kind][0], (kind, tally[kind])
    for kind, least in (("plane", 30), ("noise", 30), ("summed score", 15)):
        assert tally[kind][1] >= least, (kind, tally[kind])
    assert tally["noise 1e-6"] == [10, 0, 10], tally["noise 1e-6"]


def _discordant_pair(eps: float, seed: int, n: int = 200) -> pd.DataFrame:
    """Everyone's outcome is whether x is above 0, but for two people, both treated at site a,
    at x = ε without the event and at x = −ε with it: the only two whose outcomes x does not
    predict, so that they alone keep x from separating everyone (the fifth review's discordant
    pair, graded by the sixth's re-review)."""
    rng = np.random.default_rng(96000 + seed)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], n), "site": rng.choice(["a", "b"], n),
                       "x": rng.normal(0, 1, n)})
    df["event"] = (df.x > 0).astype(float)
    df.loc[0, ["x", "event", "arm", "site"]] = [eps, 0.0, "trt", "a"]
    df.loc[1, ["x", "event", "arm", "site"]] = [-eps, 1.0, "trt", "a"]
    return df


def test_a_fit_float64_cannot_resolve_is_refused_or_read_exactly():
    """Claims backtest B-281, its sixth review's re-review. In a trial whose outcome is whether x
    is above 0 but for a discordant pair at x = ±ε, both treated at site a, no one is separated,
    since the pair keeps x from separating the rest, and the maximum-likelihood fit is finite;
    but x's coefficient is 700 to 1,200, all but 4 or 5 of the 200 have a risk within 1e-10 of 0
    or 1, and the pair, at risks near ½, fix the intercept and the arm's coefficient nearly only
    in their sum. On the standardized columns the information's smallest eigenvalue is 1.2e-15 to
    5.5e-15 of its largest at that fit, so float64 resolves the fit along that combination only
    to its rounding. Known truth: the maximum-likelihood fit on the float data in 60-digit
    arithmetic (:func:`_exact_fit`), certified for each, each arm's risks and Ye et al.'s variance
    written out above. At ε = 5e-6 seed 76, ε = 1.5e-6 seed 77 and ε = 1e-6 seed 8, each trial is
    refused for its nearly singular information, with fewer covariates and describing the
    outcomes as its exits, or reported within 1e-3 of a standard error of that fit, its standard
    error to 1e-3.

    On the code before this round (6c08230e) the three were reported, where the base (e6e58809)
    refused them: seed 76 at a risk difference of −0.0061451 (SE 0.0066068) where the fit gives
    −0.0060474 (SE 0.0065954), 1.48e-2 standard errors off, its intercept and arm coefficient
    0.11 off and opposite; seed 77 1.28e-3 off; seed 8 9.0e-4 off."""
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("x",))
    for eps, seed in ((5e-6, 76), (1.5e-6, 77), (1e-6, 8)):
        df = _discordant_pair(eps, seed)
        X, y = _x_design(df, "x"), df.event.to_numpy()
        b = _exact_fit(X, y)
        assert b is not None, (eps, seed, "no finite fit certified in 60 digits")
        try:
            r = T.estimate(df, spec)
        except T.TrialRefused as e:
            assert e.term == "nearly singular information in the logistic model", str(e)
            assert "the rounding of the numbers, not by the data" in str(e), str(e)
            assert [x.get("covariates") for x in e.exits] == [[], None]
            assert e.exits[-1]["describe"] is True
            continue
        rd, se = _exact_effect(X, y, b, pd.factorize(df.site)[0])
        got = r.contrast(measure="risk_difference")
        assert abs(got.estimate - rd) <= 1e-3 * se, (eps, seed, got.estimate, rd, se)
        assert got.se == pytest.approx(se, rel=1e-3), (eps, seed, got.se, se)


def _three_arms(seed: int = 2065, n: int = 45) -> pd.DataFrame:
    """A small three-arm trial at two sites, with three normal covariates."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "pa", "pb"], n), "site": rng.choice(["a", "b"], n),
                       "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
                       "x3": rng.normal(0, 1, n)})
    lp = (-0.2 + 0.8 * (df.arm == "pa") - 0.6 * (df.arm == "pb") + 1.6 * df.x1 + 1.0 * df.x2
          + 0.5 * df.x3)
    df["event"] = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    return df


THREE = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                    covariates=("x1", "x2", "x3"))


def test_a_variance_below_zero_in_a_small_arm_is_refused_in_plain_words():
    """Claims backtest B-281, its fourth review: a failure older than the branch. Ye et al.'s
    variance is a sum of covariances, not of squares, and in a small arm with many terms it can
    come out below zero. Here three arms of 14, 20 and 11 people and seven terms, with no one
    separated (statsmodels' ``Logit`` converges on everyone): from statsmodels' fitted risks,
    Ye et al.'s variance written out above is below zero for pb's standardized risk. The effect
    is refused, saying that the estimate breaks down in an arm this small and naming pb's 11
    people, with fewer covariates (which runs) and describing the outcomes as its exits. With 2
    of pb's outcomes missing, the delta-adjusted and tipping-point analyses refuse the same way,
    since a completed data set breaks down there too.

    On the prior code the effect failed with math.sqrt's ValueError (math domain error), and the
    sensitivity analyses pooled the completed sets as if their variances held."""
    import statsmodels.api as sm

    df = _three_arms()
    arm = df.arm.map({"ctl": 0, "pa": 1, "pb": 2}).to_numpy()
    X = np.column_stack([np.ones(len(df)), arm == 1, arm == 2, df.site == "b", df.x1, df.x2,
                         df.x3]).astype(float)
    y = df.event.to_numpy()
    fit = sm.Logit(y, X).fit(method="newton", tol=1e-12, maxiter=100, disp=0)
    assert fit.mle_retvals["converged"]
    cf = np.empty((len(df), 3))
    for k in range(3):
        at = X.copy()
        at[:, 1:3] = 0.0
        if k:
            at[:, k] = 1.0
        cf[:, k] = fit.predict(at)
    V = _ye_by_hand(y, arm, cf, cf[np.arange(len(df)), arm], pd.factorize(df.site)[0])
    assert V[2, 2] < 0 < min(V[0, 0], V[1, 1]), np.diag(V)

    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, THREE)
    said = str(r.value)
    assert r.value.term == "negative variance estimate", said
    assert "below zero for pb" in said and "11 people in pb" in said, said
    assert "breaks down in an arm this small" in said, said
    assert [e.get("covariates") for e in r.value.exits] == [[], None]
    assert r.value.exits[-1]["describe"] is True
    T.estimate(df, T.TrialSpec(**{**THREE.__dict__, "covariates": ()}))

    gone = df.copy()
    gone.loc[gone.index[gone.arm == "pb"][:2], "event"] = np.nan
    for run in (lambda: T.delta_analysis(gone, THREE, arm="pb", delta=0.0, m=10),
                lambda: T.tipping_point(gone, THREE, m=10)):
        with pytest.raises(T.TrialRefused) as r:
            run()
        assert r.value.term == "negative variance estimate", str(r.value)
        assert "completed by imputation" in str(r.value), str(r.value)


def test_the_sensitivity_analyses_of_a_completely_separated_trial_count_no_imputed_outcome():
    """Claims backtest B-281, its fourth review. The tipping-point and delta-adjusted analyses
    fit each completed data set as the primary analysis does, so a completed set that the arm
    and the covariates predicted exactly would be refused with the text that counts every
    person's outcome, imputed ones among them. No completed set can be: the imputation model is
    the analysis model fitted to the people with an outcome, whose columns have full rank, so a
    direction that separated a completed set would separate some of them too, and that model
    refuses first. Here everyone's outcome is whether age is over 50, 20 of 200 outcomes
    missing: the primary analysis is refused for complete separation, counting the 180 with an
    outcome, and both sensitivity analyses are refused for separation in the imputation model,
    with no count of outcomes predicted exactly; with fewer covariates they run. Pinned as the
    code already behaves: no wording changed."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], 200), "site": rng.choice(["a", "b"], 200),
                       "age": rng.normal(50, 10, 200)})
    df["event"] = (df.age > 50).astype(float)
    df.loc[rng.choice(200, 20, replace=False), "event"] = np.nan
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("age",))
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, spec)
    assert r.value.term == "complete separation in the logistic model"
    assert "every person's outcome exactly (all 180 of them)" in str(r.value), str(r.value)
    for run in (lambda s: T.tipping_point(df, s, m=5),
                lambda s: T.delta_analysis(df, s, arm="trt", delta=1.0, m=5)):
        with pytest.raises(T.TrialRefused) as r:
            run(spec)
        said = str(r.value)
        assert r.value.term == "separation in the logistic model", said
        assert "runs to 0% or 100%" in said and "every person" not in said, said
        assert "200" not in said and "180" not in said, said
        assert r.value.exits[0]["covariates"] == []
        run(T.TrialSpec(**{**spec.__dict__, "covariates": ()}))


# ── wording ──────────────────────────────────────────────────────────────────


def test_causal_wording_only_for_intention_to_treat_and_per_protocol_never_alone():
    df = _parallel()
    itt = T.estimate(df, SPEC)
    pp = T.estimate(df, SPEC, analysis_set="per_protocol")
    assert itt.causal and itt.says.startswith("Assignment to trt changed y by")
    assert "intention to treat" in itt.says
    assert not pp.causal and pp.says.startswith("Among those who followed the protocol")
    assert "changed" not in pp.says and pp.concerns
    with pytest.raises(T.TrialRefused) as r:
        T.trial_sentence(per_protocol=pp)
    assert r.value.exits[0]["analysis_set"] == "itt"
    text = T.trial_sentence(itt, pp, T.tipping_point(df, SPEC, m=20))
    assert "intention to treat" in text and "Per protocol" in text and "tipping point" in text
    assert "named before the data were seen" in text and "White & Thompson 2005" in text


def test_every_sentence_prints_an_interval_on_its_estimate_s_scale():
    """A risk difference is printed in percentage points, and so is its interval, in the
    intention-to-treat sentence and in the per-protocol clause alike."""
    df = _parallel()
    itt = T.estimate(df, BSPEC)
    pp = T.estimate(df, BSPEC, analysis_set="per_protocol")
    text = T.trial_sentence(itt, pp)
    for r in (itt, pp):
        c = r.contrast()
        points = f"{T._num(100 * c.estimate)} percentage points"
        interval = f"95% CI {T._num(100 * c.lower)} to {T._num(100 * c.upper)}"
        assert points in r.says and interval in r.says
        assert points in text and interval in text
        raw = f"95% CI {T._num(c.lower)} to"
        assert raw not in text, raw
    # the per-protocol clause names the arm and its control
    assert f"Per protocol, among those who followed the protocol: trt against ctl, " in text
    rr = T.estimate(df, BSPEC, measure="risk_ratio").contrast(measure="risk_ratio")
    assert f"95% CI {T._num(rr.lower)} to {T._num(rr.upper)}" in \
        T.estimate(df, BSPEC, measure="risk_ratio").says


def test_a_change_is_asserted_only_when_the_interval_excludes_no_difference_and_every_arm_is_said():
    df = _parallel(arms=("ctl", "a", "b"), n=300)
    df.loc[df.arm == "a", "y"] -= 0.9  # arm a: no effect
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="y", strata=("site",), baseline="base",
                       adherent="adherent")
    r = T.estimate(df, spec)
    a, b = r.contrast("a"), r.contrast("b")
    assert a.lower < 0 < a.upper and (b.lower > 0 or b.upper < 0)
    says = r.says
    assert "Assignment to a changed" not in says
    assert ("The estimated effect of assignment to a on y was "
            f"{T._num(a.estimate)} (95% CI {T._num(a.lower)} to {T._num(a.upper)}, which includes "
            "no difference) compared with ctl, so a change is not shown.") in says
    assert (f"Assignment to b changed y by {T._num(b.estimate)} (95% CI {T._num(b.lower)} to "
            f"{T._num(b.upper)}) compared with ctl.") in says
    pp = T.estimate(df, spec, analysis_set="per_protocol")
    text = T.trial_sentence(r, pp)
    for c in pp.contrasts:
        assert f"{c.arm} against ctl, {T._num(c.estimate)} (95% CI {T._num(c.lower)}" in text
        assert f"between {c.arm} and ctl" in pp.says
    # a ratio whose interval includes 1 says so too
    bin_ = T.estimate(df, T.TrialSpec(arm="arm", control="ctl", outcome="event"),
                      measure="risk_ratio")
    for c in bin_.contrasts:
        if c.measure == "risk_ratio" and c.lower < 1 < c.upper:
            assert f"to {T._num(c.upper)}, which includes no difference)" in bin_.says


def test_every_trial_function_refuses_a_survey_design_and_the_delta_analysis_refuses_as_the_tipping_point():
    df = _parallel()
    survey = object()
    for call in (lambda: T.consort_flow(df, SPEC, survey_design=survey),
                 lambda: T.baseline_table(df, SPEC, ["age"], survey_design=survey),
                 lambda: T.delta_analysis(df, SPEC, arm="trt", delta=1.0, m=5,
                                          survey_design=survey)):
        with pytest.raises(T.TrialRefused) as r:
            call()
        assert r.value.term == "survey design in a randomized trial"
        assert r.value.exits[0]["decision"] == {"kind": "set_survey", "estimand": "sample"}
    with pytest.raises(T.TrialRefused) as r:
        T.delta_analysis(df, SPEC, arm="trt", delta=1.0, m=5, goal="prediction")
    assert r.value.exits[0]["goal"] == "inference"
    with pytest.raises(T.TrialRefused) as r:
        T.delta_analysis(df, SPEC, arm="trt", delta=1.0, m=5, analysis_set="per_protocol")
    assert r.value.term == "tipping point outside intention to treat"
    c = _cluster()
    with pytest.raises(T.TrialRefused) as r:
        T.delta_analysis(c.assign(y=c.y.where(c.index % 9 > 0)), CSPEC, arm="trt", delta=1.0)
    assert r.value.term == "multilevel imputation for a cluster-randomized trial"
    row, done = T.delta_analysis(df, SPEC, arm="trt", delta=1.0, m=5)
    assert len(done) == 5 and math.isfinite(row.estimate)


def test_each_refusal_has_exits_that_fit_its_cause():
    df = _parallel()
    # one arm
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df[df.arm.isna() | (df.arm == "ctl")], SPEC)
    assert r.value.term == "one arm" and r.value.exits
    assert {e["decision"]["kind"] for e in r.value.exits} == {"confirm_role", "set_design"}
    # too few outcomes in an arm, under intention to treat
    thin = df.copy()
    thin.loc[(thin.arm == "trt") & thin.index.isin(thin.index[thin.arm == "trt"][1:]), "y"] = np.nan
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(thin, SPEC)
    assert r.value.term == "too few outcomes in an arm" and r.value.exits
    assert r.value.exits[-1]["describe"] is True
    # adherence that is not yes/no: per protocol refused with exits, intention to treat untouched
    odd = df.assign(adherent=df.adherent.where(df.index % 7 > 0, "sometimes"))
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(odd, SPEC, analysis_set="per_protocol")
    assert r.value.term == "adherence is not yes/no" and "'adherent'" in str(r.value)
    assert r.value.exits[0]["decision"] == {"kind": "confirm_role", "column": None,
                                            "role": "flag"}
    _runs(lambda **k: T.estimate(odd, SPEC, **k), r.value)
    assert T.estimate(odd, SPEC).contrast().estimate == T.estimate(df, SPEC).contrast().estimate
    with pytest.raises(T.TrialRefused) as r:
        T.consort_flow(odd, SPEC)
    assert r.value.exits[1]["adherent"] is None
    T.consort_flow(odd, T.TrialSpec(**{**SPEC.__dict__, "adherent": r.value.exits[1]["adherent"]}))
    # a missing cluster: the exit is a decision naming the randomized cluster
    c = _cluster()
    gap = c.astype({"cluster": float})
    gap.loc[gap.index[:2], "cluster"] = np.nan
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(gap, CSPEC)
    assert r.value.term == "missing cluster"
    assert r.value.exits[0]["decision"] == {"kind": "confirm_role", "column": None,
                                            "role": "cluster"}
    # one cluster in an arm: no covariate change helps, so none is offered
    lone = c[c.cluster.isin([1, 2, 3, 5])]  # ctl: 1, 3, 5; trt: 2
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(lone, CSPEC)
    assert r.value.term == "too few clusters" and "one cluster in trt" in str(r.value)
    assert not any("covariates" in e for e in r.value.exits)
    assert r.value.exits[0]["decision"]["role"] == "cluster"
    # cluster-level covariates that use the clusters up: leaving them out is the exit, and it runs
    four = c[c.cluster.isin([1, 2, 3, 4])].copy()
    four["urban"] = four.cluster.map({1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0})
    four["big"] = four.cluster.map({1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0})
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="y", trial_design="cluster_randomized_trial",
                       cluster="cluster", covariates=("x", "urban", "big"))
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(four, spec)
    assert r.value.term == "too few clusters"
    leave = r.value.exits[0]
    assert leave["covariates"] == ["x"] and "urban, big" in leave["label"]
    assert T.estimate(four, T.TrialSpec(**{**spec.__dict__, "covariates": tuple(leave["covariates"])})
                      ).clusters == {"ctl": 2, "trt": 2}


def test_a_few_clusters_concern_never_displaces_the_missing_outcome_notice():
    small = _cluster(K=8)
    small.loc[small.index % 6 == 0, "y"] = np.nan
    r = T.estimate(small, CSPEC)
    n_missing = int(small.y.isna().sum())
    assert r.noticing.startswith(f"{n_missing} of {len(small)} people")
    assert "have no outcome" in r.noticing
    assert any("Only 8 clusters" in c for c in r.concerns)


# ── the contracts ────────────────────────────────────────────────────────────


def test_the_contracts_are_registered_estimate_only_and_name_code_that_exists():
    from turbotab.core.contracts import contracts

    registry = contracts()
    for key in ("trial_analysis_set", "trial_effect", "trial_missing_outcomes"):
        c = registry[key]
        assert c.package == "TRIALS" and c.sources
        for o in c.options:
            assert o.rung["prediction"] == "not_offered"
            assert o.rung["inference"] in ("recommended", "rank_lower")
            assert o.customary and o.sound["inference"]
        for r in c.relations:
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
            if r.kind == "conflicts":
                assert r.exits, (key, r.name)
    effect = registry["trial_effect"]
    assert effect.relation("no_balance_tests").rung == "refused"
    assert effect.relation("no_confounder_selection").kind == "implies"
    assert registry["trial_missing_outcomes"].relation("no_cluster_mi").rung == "refused"
    assert T.eligible("prediction").offered is False
    assert T.eligible("inference", "tipping_point").offered


def test_the_per_protocol_effect_is_refused_and_the_set_is_its_exit():
    """The effect of following the protocol is a v2.x estimand (``designs.EFFECTS``); the set
    analyzed as randomized is what E2 offers."""
    from turbotab.core.designs import effect_refusal

    class Asked:
        effect = "per_protocol"

        def model_dump(self, mode: str) -> dict:
            return {"kind": "set_estimand", "effect": self.effect}

    code, reason, exits = effect_refusal(Asked())
    assert code == "per_protocol_effect_v2x" and "per-protocol analysis set" in reason
    assert exits[0]["decision"]["effect"] == "total"
    pp = T.estimate(_parallel(), SPEC, analysis_set="per_protocol")
    assert pp.analysis_set == "per_protocol" and not pp.causal


def test_the_adjustment_is_exactly_the_named_terms_and_nothing_is_selected():
    """No covariate is searched for: a strongly prognostic column nobody named stays out, a
    useless one that was named stays in, and the randomization factors are always in."""
    df = _parallel(seed=40)
    rng = np.random.default_rng(1)
    df["noise"] = rng.normal(size=len(df))
    df["prognostic"] = df.y.fillna(0) + rng.normal(0, 0.1, len(df))
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="y", strata=("site",),
                       covariates=("noise",), baseline="base")
    r = T.estimate(df, spec)
    assert r.terms == ["(Intercept)", "arm[trt]", "site[b]", "site[c]", "base", "noise"]
    assert r.adjusted_for == ["site", "base", "noise"]
    assert "prognostic" not in " ".join(r.terms)


def test_the_contracts_relations_are_each_asserted_by_a_test_here_and_run_in_order():
    from turbotab.core import contracts as C

    registry = C.contracts()
    mine = [c for c in registry.values() if c.package == "TRIALS"]
    assert {c.key for c in mine} == {"trial_analysis_set", "trial_effect",
                                     "trial_missing_outcomes"}
    for c in mine:
        assert c.slot in C.SLOTS and c.scope in C.SCOPES and c.scope_note, c.key
        assert c.needs and c.question.endswith("?") and c.storyboard and c.options, c.key
        module, _, name = str(c.sentence).partition(":")
        assert callable(getattr(importlib.import_module(module), name)), c.key
        for r in c.relations:
            assert r.condition and r.says and r.enforced_by and r.id, (c.key, r.id)
            assert bool(r.exits) == (r.kind == "conflicts"), (c.key, r.id)
    ids = {r.id for c in mine for r in c.relations}
    assert ids == set(RELATION_TESTS)
    tests = {name for name in globals() if name.startswith("test_")}
    assert set(RELATION_TESTS.values()) <= tests, sorted(set(RELATION_TESTS.values()) - tests)
    order = C.run_order(list(registry))
    assert order.index("trial_analysis_set") < order.index("trial_effect") < \
        order.index("trial_missing_outcomes")


RELATION_TESTS = {
    "itt_keeps_everyone":
        "test_intention_to_treat_keeps_everyone_randomized_as_randomized_and_per_protocol_the_adherent",
    "pp_effect_v2x": "test_the_per_protocol_effect_is_refused_and_the_set_is_its_exit",
    "pp_beside_itt": "test_causal_wording_only_for_intention_to_treat_and_per_protocol_never_alone",
    "pp_adherence":
        "test_the_per_protocol_set_is_refused_without_adherence_and_for_the_tipping_point",
    "consort_flow": "test_the_consort_flow_counts_each_arm_by_stage_as_counted_by_hand",
    "set_not_predicted": "test_survey_weights_prediction_and_an_observational_design_are_refused",
    "set_no_survey":
        "test_every_trial_function_refuses_a_survey_design_and_the_delta_analysis_refuses_as_the_tipping_point",
    "no_confounder_selection":
        "test_the_adjustment_is_exactly_the_named_terms_and_nothing_is_selected",
    "strata_adjusted": "test_the_adjustment_is_exactly_the_named_terms_and_nothing_is_selected",
    "ancova_baseline": "test_ancova_matches_lm_and_emmeans",
    "post_hoc_refused":
        "test_covariates_chosen_after_the_data_are_refused_from_the_primary_analysis",
    "post_randomization": "test_adherence_or_the_arm_received_cannot_be_an_adjustment_term",
    "baseline_mean_imputation": "test_ancova_matches_lm_and_emmeans",
    "no_balance_tests": "test_the_baseline_table_describes_each_arm_and_has_no_test",
    "causal_itt_only":
        "test_causal_wording_only_for_intention_to_treat_and_per_protocol_never_alone",
    "needs_randomization":
        "test_survey_weights_prediction_and_an_observational_design_are_refused",
    "icc": "test_the_mixed_model_and_its_kenward_roger_intervals_match_pbkrtest",
    "cluster_is_randomized": "test_the_cluster_trial_refusals_and_their_exits",
    "cluster_floor": "test_each_refusal_has_exits_that_fit_its_cause",
    "few_clusters": "test_a_few_clusters_concern_never_displaces_the_missing_outcome_notice",
    "kc_or_fg": "test_the_correction_follows_li_and_redden_s_rule_on_the_cluster_sizes",
    "kr_numeric_only": "test_the_cluster_trial_refusals_and_their_exits",
    "separation": "test_separation_is_refused_in_plain_words",
    "primary_first":
        "test_the_contracts_relations_are_each_asserted_by_a_test_here_and_run_in_order",
    "effect_no_survey": "test_survey_weights_prediction_and_an_observational_design_are_refused",
    "effect_not_predicted":
        "test_survey_weights_prediction_and_an_observational_design_are_refused",
    "all_randomized":
        "test_the_tipping_point_is_where_the_interval_s_bound_reaches_no_difference",
    "m_rule": "test_the_number_of_imputations_is_at_least_the_percentage_missing",
    "pooled": "test_the_pooled_delta_adjusted_analysis_is_mice_pool_on_the_completed_sets",
    "no_cluster_mi": "test_the_cluster_trial_refusals_and_their_exits",
    "itt_only":
        "test_the_per_protocol_set_is_refused_without_adherence_and_for_the_tipping_point",
    "tipping_no_survey": "test_survey_weights_prediction_and_an_observational_design_are_refused",
    "tipping_not_predicted":
        "test_survey_weights_prediction_and_an_observational_design_are_refused",
}

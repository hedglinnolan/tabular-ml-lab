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
    assert not secondary.causal and "not read as an effect" in secondary.says


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
    assert "Only 8 clusters" in T.estimate(small, CSPEC).noticing


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
    df = _parallel(seed=21)
    df.loc[df.site == "c", "event"] = 1.0
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, BSPEC)
    assert r.value.term == "separation in the logistic model"
    assert "0% or 100%" in str(r.value)


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
    "cluster_floor": "test_the_cluster_trial_refusals_and_their_exits",
    "few_clusters": "test_the_cluster_trial_refusals_and_their_exits",
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

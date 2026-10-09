"""D5: Bland–Altman agreement as an engine method core (`turbotab.core.methods.agreement`).

Every number is held to an independent reference: the worked example of Bland & Altman's 1986
Lancet paper (its printed numbers); R's BlandAltmanLeh for the differences and ratios; R's ``lm``
for the slopes and the regression-based limits; a one-way ANOVA in R and Zou's (2013) MOVER written
from the paper's own terms (the person means) for repeated pairs, and SimplyAgree's mixed model for
the balanced case; R survey (``svytotal`` and ``svycontrast``, ``svymean``, ``svyvar``, ``svyglm``)
for the population answer. A package R does not find skips its test (set ``R_LIBS`` to a library
that holds it).
"""
from __future__ import annotations

import math
import subprocess

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import agreement as A
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r

# Bland & Altman 1986, Table 1: peak expiratory flow rate (l/min), first reading by each meter.
WRIGHT = [494, 395, 516, 434, 476, 557, 413, 442, 650, 433, 417, 656, 267, 478, 178, 423, 427]
MINI = [512, 430, 520, 428, 500, 600, 364, 380, 658, 445, 432, 626, 260, 477, 259, 350, 451]


def _r_has(package: str) -> bool:
    if RSCRIPT is None:
        return False
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


def _skip_without(package: str) -> None:
    if not _r_has(package):
        pytest.skip(f"R's {package} is not installed (set R_LIBS to a library that has it)")


def _classic(seed: int = 20261009, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(100, 15, n)
    a, b = x + rng.normal(0, 5, n), x + 2 + rng.normal(0, 6, n)
    a[[4, 17]] = np.nan  # pairs with a missing value leave
    return pd.DataFrame({"a": a, "b": b})


def _multiplicative(seed: int = 7, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.lognormal(4, 0.6, n)
    return pd.DataFrame({"a": x * np.exp(rng.normal(0, 0.10, n)),
                         "b": 1.05 * x * np.exp(rng.normal(0, 0.12, n))})


def _repeated(counts: list[int], seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for person, m in enumerate(counts):
        level, shift = rng.normal(80, 12), rng.normal(0, 3)
        for _ in range(m):
            x = level + rng.normal(0, 5)
            rows.append({"id": person + 1, "a": x + shift + rng.normal(0, 4),
                         "b": x + 1.5 + rng.normal(0, 3)})
    return pd.DataFrame(rows)


def _close(got: A.Interval, est: float, lo: float, hi: float, rel: float = 1e-10) -> None:
    assert got.estimate == pytest.approx(est, rel=rel)
    assert got.lower == pytest.approx(lo, rel=rel)
    assert got.upper == pytest.approx(hi, rel=rel)


# ── the published worked example ─────────────────────────────────────────────


def test_the_lancet_peak_flow_example_gives_the_papers_printed_numbers():
    """Bland & Altman 1986: "the mean difference … is −2.1 l/min … the standard deviation of the
    differences … is 38.8 l/min"; with two SDs, "the limits of agreement … −79.7 to 75.5 l/min"
    (the paper's limits are its rounded mean and SD combined)."""
    r = A.agreement(WRIGHT, MINI, names=("the Wright meter", "the mini Wright meter"))
    bias, sd = round(r.bias.estimate, 1), round(r.sd, 1)
    assert (bias, sd) == (-2.1, 38.8)
    assert (round(bias - 2 * sd, 1), round(bias + 2 * sd, 1)) == (-79.7, 75.5)
    assert r.n_pairs == 17 and not r.repeated and not r.weighted
    assert "2.12 below the mini Wright meter" in r.says


# ── differences, ratios and the regression-based limits against R ────────────


@needs_r
def test_the_differences_and_their_intervals_match_blandaltmanleh_and_the_slopes_match_lm(tmp_path):
    _skip_without("BlandAltmanLeh")
    frame = _classic()
    ref = run_r("""
library(BlandAltmanLeh)
df <- read.csv(classic_csv)
s <- bland.altman.stats(df$a, df$b, two = 1.96, conf.int = 0.95)
df <- na.omit(df); d <- df$a - df$b; m <- (df$a + df$b) / 2
fit <- lm(d ~ m); r <- abs(resid(fit)); fit2 <- lm(r ~ m)
slope <- function(f) c(coef(f), confint(f)[2, ], summary(f)$coefficients[2, 4])
out(list(lines = unname(s$lines), ci = unname(s$CI.lines), n = s$based.on, sd = sd(d),
         prop = unname(slope(fit)), spread = unname(slope(fit2))))
""", {"classic": frame}, tmp_path)
    r = A.agreement(frame["a"], frame["b"])
    lo, mean, hi = ref["lines"]
    ci = ref["ci"]
    assert r.n_pairs == ref["n"] == 58
    assert r.sd == pytest.approx(ref["sd"], rel=1e-12)
    _close(r.bias, mean, ci[2], ci[3])
    _close(r.lower_limit, lo, ci[0], ci[1])
    _close(r.upper_limit, hi, ci[4], ci[5])
    for got, want in ((r.proportional, ref["prop"]), (r.spread, ref["spread"])):
        assert [got.intercept, got.slope, got.lower, got.upper, got.p] == pytest.approx(want,
                                                                                        rel=1e-9)
    assert r.df == 57 and r.proportional.df == 56


@needs_r
def test_the_ratios_are_the_differences_of_the_logs_back_transformed(tmp_path):
    _skip_without("BlandAltmanLeh")
    frame = _multiplicative()
    ref = run_r("""
library(BlandAltmanLeh)
df <- read.csv(ratio_csv)
s <- bland.altman.stats(log(df$a), log(df$b), two = 1.96, conf.int = 0.95)
out(list(lines = exp(unname(s$lines)), ci = exp(unname(s$CI.lines))))
""", {"ratio": frame}, tmp_path)
    r = A.agreement(frame["a"], frame["b"], form="ratios", names=("A", "B"))
    lo, mean, hi = ref["lines"]
    ci = ref["ci"]
    assert r.scale == "ratio"
    _close(r.bias, mean, ci[2], ci[3])
    _close(r.lower_limit, lo, ci[0], ci[1])
    _close(r.upper_limit, hi, ci[4], ci[5])
    assert "times B" in r.says


def test_a_spread_that_grows_is_noticed_and_the_ratios_offered_first():
    frame = _multiplicative()
    r = A.agreement(frame["a"], frame["b"])
    assert r.spread_grows and r.spread.slope > 0 and r.spread.p < 0.05
    assert r.noticing and "ratios form" in r.noticing
    on_logs = A.agreement(frame["a"], frame["b"], form="ratios")
    assert not on_logs.spread_grows and on_logs.noticing is None
    steady = A.agreement(*(_classic()[c] for c in ("a", "b")))
    assert not steady.spread_grows and steady.noticing is None


@needs_r
def test_the_regression_based_limits_are_bland_and_altman_1999_by_lm(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.uniform(20, 200, 90)
    frame = pd.DataFrame({"a": 5 + 1.1 * x + rng.normal(0, 0.05 * x), "b": x})
    ref = run_r("""
df <- read.csv(reg_csv)
d <- df$a - df$b; m <- (df$a + df$b) / 2
b <- coef(lm(d ~ m)); g <- coef(lm(abs(resid(lm(d ~ m))) ~ m))
at <- c(min(m), mean(m), max(m))
center <- b[1] + b[2] * at; half <- 1.96 * sqrt(pi / 2) * (g[1] + g[2] * at)
out(list(at = at, bias = unname(center), lower = unname(center - half),
         upper = unname(center + half)))
""", {"reg": frame}, tmp_path)
    r = A.agreement(frame["a"], frame["b"], form="regression", names=("A", "B"))
    for key in ("at", "bias", "lower", "upper"):
        assert r.lines[key] == pytest.approx(ref[key], rel=1e-10), key
    assert r.proportional.p < 0.001 and r.proportional.slope > 0
    assert "changes with the size" in r.says


# ── repeated pairs per person ────────────────────────────────────────────────

_ZOU = """
d <- df$a - df$b; id <- factor(df$id); m <- (df$a + df$b) / 2
tab <- summary(aov(d ~ id))[[1]]; msb <- tab[1, 3]; msw <- tab[2, 3]
mi <- as.numeric(table(id)); n <- length(d); k <- length(mi)
lam <- (n^2 - sum(mi^2)) / ((k - 1) * n)
sb <- max((msb - msw) / lam, 0); s2 <- sb + msw
bias <- mean(d); se <- sqrt((sb * sum(mi^2) + msw * n) / n^2)
tq <- qt(0.975, k - 1)
# Zou 2013 in its own terms: the variance of the person means and the harmonic mean of the m_i
s2bar <- var(tapply(d, id, mean)); mh <- k / sum(1 / mi)
zs2 <- s2bar + (1 - 1 / mh) * msw
L <- zs2 - sqrt((s2bar * (1 - (k - 1) / qchisq(0.975, k - 1)))^2 +
                ((1 - 1 / mh) * msw * (1 - (n - k) / qchisq(0.975, n - k)))^2)
U <- zs2 + sqrt((s2bar * ((k - 1) / qchisq(0.025, k - 1) - 1))^2 +
                ((1 - 1 / mh) * msw * ((n - k) / qchisq(0.025, n - k) - 1))^2)
zl <- 1.96 * sqrt(L); zu <- 1.96 * sqrt(U); zz <- 1.96 * sqrt(zs2); e <- tq * sqrt(s2bar / k)
upper <- c(bias + zz, bias + zz - sqrt(e^2 + (zz - zl)^2), bias + zz + sqrt(e^2 + (zu - zz)^2))
lower <- c(bias - zz, bias - zz - sqrt(e^2 + (zu - zz)^2), bias - zz + sqrt(e^2 + (zz - zl)^2))
"""


@needs_r
def test_balanced_repeated_pairs_match_an_anova_zous_mover_and_simplyagree(tmp_path):
    frame = _repeated([4] * 12)
    has_simplyagree = _r_has("SimplyAgree")
    ref = run_r("df <- read.csv(rep_csv)\n" + _ZOU + """
svy <- if (requireNamespace("survey", quietly = TRUE)) {
  library(survey); des <- svydesign(ids = ~id, data = data.frame(d = d, m = m, id = df$id))
  f <- svyglm(d ~ m, des); unname(c(coef(f)[2], SE(f)[2]))
} else NULL
sa <- NULL
if (%s) {
  suppressMessages(library(SimplyAgree))
  l <- agreement_limit(x = "a", y = "b", id = "id", data = df, data_type = "nest")$loa
  sa <- c(l$bias, l$SE, l$lower.CL, l$upper.CL, l$sd_delta)
}
out(list(sb = sb, sw = msw, s2 = s2, zs2 = zs2, bias = c(bias, bias - tq * se, bias + tq * se),
         se = se, upper = upper, lower = lower, svy = svy, sa = sa))
""" % ("TRUE" if has_simplyagree else "FALSE"), {"rep": frame}, tmp_path)
    r = A.agreement(frame["a"], frame["b"], units=frame["id"])
    assert r.repeated and r.n_units == 12 and r.n_pairs == 48 and r.df == 11
    assert r.components["between"] == pytest.approx(ref["sb"], rel=1e-10)
    assert r.components["within"] == pytest.approx(ref["sw"], rel=1e-10)
    assert r.sd ** 2 == pytest.approx(ref["s2"], rel=1e-10)
    assert ref["zs2"] == pytest.approx(ref["s2"], rel=1e-10)  # balanced: the two forms agree
    _close(r.bias, *ref["bias"])
    _close(r.upper_limit, *ref["upper"])
    _close(r.lower_limit, *ref["lower"])
    if ref["svy"]:  # R's NULL arrives as {}
        assert [r.proportional.slope, r.proportional.se] == pytest.approx(ref["svy"], rel=1e-9)
    assert bool(ref["sa"]) == has_simplyagree
    if ref["sa"]:  # SimplyAgree's REML mixed model equals the ANOVA when balanced
        bias, se, lo, hi, sd = ref["sa"]
        assert [r.bias.estimate, r.bias.se, r.bias.lower, r.bias.upper] == pytest.approx(
            [bias, se, lo, hi], rel=1e-6)
        assert r.sd == pytest.approx(sd, rel=1e-6)


@needs_r
def test_unbalanced_repeated_pairs_are_bland_and_altman_2007s_anova(tmp_path):
    frame = _repeated([2, 3, 5, 6, 2, 4, 3, 6, 5, 2, 4, 3, 3, 5], seed=12)
    ref = run_r("df <- read.csv(rep_csv)\n" + _ZOU + """
out(list(sb = sb, sw = msw, bias = c(bias, bias - tq * se, bias + tq * se), lam = lam))
""", {"rep": frame}, tmp_path)
    r = A.agreement(frame["a"], frame["b"], units=frame["id"])
    assert r.components["between"] == pytest.approx(ref["sb"], rel=1e-10)
    assert r.components["within"] == pytest.approx(ref["sw"], rel=1e-10)
    _close(r.bias, *ref["bias"])
    assert r.upper_limit.estimate == pytest.approx(
        ref["bias"][0] + 1.96 * math.sqrt(ref["sb"] + ref["sw"]), rel=1e-10)
    # MOVER on MS_b/λ + (1 − 1/λ)·MS_w, by hand from the components
    lam = ref["lam"]
    k, n = 14, len(frame)
    from scipy import stats
    ms_b, ms_w = ref["sb"] * lam + ref["sw"], ref["sw"]
    s2 = ref["sb"] + ref["sw"]
    up = math.hypot(ms_b / lam * ((k - 1) / stats.chi2.ppf(0.025, k - 1) - 1),
                    (1 - 1 / lam) * ms_w * ((n - k) / stats.chi2.ppf(0.025, n - k) - 1))
    above = ref["bias"][2] - ref["bias"][0]
    sd, sd_hi = math.sqrt(s2), math.sqrt(s2 + up)
    assert r.upper_limit.upper == pytest.approx(
        ref["bias"][0] + 1.96 * sd + math.hypot(above, 1.96 * (sd_hi - sd)), rel=1e-10)


def test_one_pair_per_person_takes_each_persons_first_pair():
    frame = _repeated([3, 2, 4, 2, 3])
    r = A.agreement(frame["a"], frame["b"], units=frame["id"], one_pair_per_unit=True)
    first = frame.groupby("id").head(1)
    plain = A.agreement(first["a"].to_numpy(), first["b"].to_numpy())
    assert not r.repeated and r.n_pairs == 5
    assert r.bias == plain.bias and r.upper_limit == plain.upper_limit


# ── the population answer ────────────────────────────────────────────────────


def _surveyed(seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    stratum = rng.integers(1, 11, n)
    psu = rng.integers(1, 3, n)
    x = rng.normal(60, 10, n) + stratum
    frame = pd.DataFrame({"stratum": stratum, "psu": psu, "w": rng.uniform(0.5, 4.0, n),
                          "a": x + rng.normal(0.5 + 0.1 * stratum, 3 + 0.02 * x, n),
                          "b": x + rng.normal(0, 3, n)})
    frame.loc[rng.choice(n, 15, replace=False), "a"] = np.nan
    return frame


def _design(frame: pd.DataFrame):
    from turbotab.core.models.survey import build_design

    return build_design(frame, frame["w"].to_numpy(), weight_column="w", strata_column="stratum",
                        psu_column="psu")


@needs_r
def test_the_population_answer_matches_r_survey(tmp_path):
    _skip_without("survey")
    frame = _surveyed()
    ref = run_r("""
library(survey)
df <- read.csv(svy_csv)
df$d <- df$a - df$b; df$m <- (df$a + df$b) / 2; df$d2 <- df$d^2; df$one <- 1
df$ok <- !is.na(df$d)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = df)
sub <- subset(des, ok)
n <- sum(df$ok); cc <- n / (n - 1)
tot <- svytotal(~d + d2 + one, sub)
lim <- svycontrast(tot, list(
  lo = substitute(d / one - 1.96 * sqrt(C * (d2 / one - (d / one)^2)), list(C = cc)),
  hi = substitute(d / one + 1.96 * sqrt(C * (d2 / one - (d / one)^2)), list(C = cc))))
mu <- svymean(~d, sub); v <- svyvar(~d, sub)
f <- svyglm(d ~ m, sub)
df$r <- abs(df$d - coef(f)[1] - coef(f)[2] * df$m)
sub2 <- subset(svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = df), ok)
g <- svyglm(r ~ m, sub2)
out(list(mu = c(coef(mu), SE(mu)), lim = unname(coef(lim)), lim_se = unname(SE(lim)),
         var = unname(coef(v)), degf = degf(sub),
         f = unname(c(coef(f), SE(f)[2])), g = unname(c(coef(g), SE(g)[2]))))
""", {"svy": frame}, tmp_path)
    r = A.agreement(frame["a"], frame["b"], design=_design(frame))
    assert r.weighted and r.n_pairs == 385 and r.df == ref["degf"]
    t = float(__import__("scipy").stats.t.ppf(0.975, ref["degf"]))
    mu, se = ref["mu"]
    _close(r.bias, mu, mu - t * se, mu + t * se, rel=1e-9)
    assert r.sd ** 2 == pytest.approx(ref["var"], rel=1e-10)  # svyvar's n/(n − 1)
    for got, est, s in zip((r.lower_limit, r.upper_limit), ref["lim"], ref["lim_se"]):
        _close(got, est, est - t * s, est + t * s, rel=1e-9)
    for got, want in ((r.proportional, ref["f"]), (r.spread, ref["g"])):
        assert [got.intercept, got.slope, got.se] == pytest.approx(want, rel=1e-8)
    assert "linearization" in A.agreement_sentence(r)


def test_repeated_pairs_under_the_population_answer_are_refused_with_exits():
    frame = _repeated([3, 3, 3, 3])
    frame["stratum"], frame["psu"], frame["w"] = 1, frame["id"], 2.0
    with pytest.raises(A.AgreementRefused) as refused:
        A.agreement(frame["a"], frame["b"], units=frame["id"], design=_design(frame))
    assert "no survey-weighted version" in str(refused.value)
    assert {e["label"] for e in refused.value.exits} >= {
        "Keep one pair per person (the first)",
        "Estimate for these participants instead: record the sample-only attestation"}


# ── refusals, the goals, and two models under Predict ────────────────────────


def test_each_refusal_says_why_and_offers_a_way_forward():
    with pytest.raises(A.AgreementRefused) as ratio:
        A.agreement([1.0, 0.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], form="ratios")
    assert "above zero" in str(ratio.value) and ratio.value.exits
    frame = _repeated([3, 3, 3])
    with pytest.raises(A.AgreementRefused) as regression:
        A.agreement(frame["a"], frame["b"], units=frame["id"], form="regression")
    assert {e.get("form") for e in regression.value.exits} >= {"differences", "ratios"}
    with pytest.raises(A.AgreementRefused):
        A.agreement([1.0, 2.0], [1.5, 2.5])
    with pytest.raises(A.AgreementRefused):
        A.agreement([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])  # one constant gap: nothing to spread


def test_describe_and_predict_offer_it_and_estimate_does_not():
    assert A.eligible("describe", "methods").offered
    assert A.eligible("predict", "models").offered
    for goal, comparison in (("estimate", "methods"), ("estimate", "models"),
                             ("predict", "methods"), ("describe", "models")):
        assert not A.eligible(goal, comparison).offered
    assert A.eligible("estimate", "methods").exits[0]["goal"] == "describe"


def test_two_models_are_compared_on_their_out_of_fold_predictions_only():
    from sklearn.linear_model import LinearRegression, Ridge

    from turbotab.core.models.selection import OutOfFold

    rng = np.random.default_rng(5)
    n = 120
    X = rng.normal(size=(n, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 1, n) + 10
    fold = np.arange(n) % 4
    fold[-6:] = -1  # rows no fold scores
    pairs = [(f, (fold != f) & (fold >= 0), fold == f) for f in range(4)]
    oof = OutOfFold("regression", X, y, pairs)
    by_hand = {"linear": np.full(n, np.nan), "ridge": np.full(n, np.nan)}
    for _, fit_rows, test_rows in pairs:
        for key, model in (("linear", LinearRegression()), ("ridge", Ridge(alpha=30.0))):
            fitted = model.fit(X[fit_rows], y[fit_rows])
            oof.keep(key, fitted, test_rows)
            by_hand[key][test_rows] = fitted.predict(X[test_rows])
    r = A.model_agreement(oof, "linear", "ridge")
    assert r.n_pairs == n - 6
    want = A.agreement(by_hand["linear"], by_hand["ridge"])
    assert r.bias == want.bias and r.lower_limit == want.lower_limit
    oof.task = "multiclass"
    with pytest.raises(A.AgreementRefused) as refused:
        A.model_agreement(oof, "linear", "ridge")
    assert refused.value.exits[0]["label"] == "Compare the models by their scores instead"
    assert "out-of-fold predictions" in A.agreement_sentence(r, comparison="models")


def test_the_contract_is_registered_with_its_labels_and_enforcing_code():
    import importlib

    from turbotab.core import contracts as C

    c = C.contracts()["bland_altman"]
    assert "turbotab.core.methods.agreement" in C.DECLARING_MODULES
    assert c.package == "AGREEMENT" and c.slot == "evaluation"
    assert [o.key for o in c.options] == list(A.FORMS) == list(A.DESCRIBE_LABELS)
    for o in c.options:
        assert o.customary and o.rung["inference"] == "not_offered"
        assert o.rung["prediction"] == "available"
    for r in c.relations:
        module, name = r.enforced_by.split(":")
        assert callable(getattr(importlib.import_module(module), name)), r.id
    assert any("Bland & Altman 2007" in s for s in c.sources)


# ── the verifiers' findings of wave E1p ──────────────────────────────────────


def test_paired_series_are_matched_by_their_index_not_their_position():
    frame = _classic()
    shuffled = frame["b"].sample(frac=1.0, random_state=4)
    assert not shuffled.index.equals(frame.index)
    r = A.agreement(frame["a"], shuffled)
    want = A.agreement(frame["a"], frame["b"])
    assert r.bias == want.bias and r.sd == pytest.approx(want.sd, rel=1e-12)
    rep = _repeated([3, 2, 4, 3])
    units = rep["id"].sample(frac=1.0, random_state=2)
    assert A.agreement(rep["a"], rep["b"], units=units).components == A.agreement(
        rep["a"], rep["b"], units=rep["id"]).components
    with pytest.raises(ValueError, match="labeled by different rows"):
        A.agreement(frame["a"], frame["b"].set_axis(range(100, 100 + len(frame))))


def test_rows_labeled_by_name_need_no_row_numbers_without_a_design():
    frame = _classic()
    named = frame.set_axis([f"p{i}" for i in range(len(frame))])
    r = A.agreement(named["a"], named["b"])
    assert r.bias == A.agreement(frame["a"], frame["b"]).bias
    surveyed = _surveyed()
    with pytest.raises(ValueError, match="pass row_ids"):
        A.agreement(surveyed["a"].set_axis([f"p{i}" for i in range(len(surveyed))]),
                    surveyed["b"].set_axis([f"p{i}" for i in range(len(surveyed))]),
                    design=_design(surveyed))


@needs_r
def test_the_slope_tests_use_svyglms_residual_degrees_of_freedom(tmp_path):
    _skip_without("survey")
    frame = _surveyed()
    rep = _repeated([4] * 9, seed=5)
    ref = run_r("""
library(survey)
df <- read.csv(svy_csv)
df$d <- df$a - df$b; df$m <- (df$a + df$b) / 2; df$ok <- !is.na(df$d)
sub <- subset(svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = df), ok)
f <- svyglm(d ~ m, sub)
rp <- read.csv(rep_csv); rp$d <- rp$a - rp$b; rp$m <- (rp$a + rp$b) / 2
g <- svyglm(d ~ m, svydesign(ids = ~id, data = rp))
out(list(f = c(summary(f)$coefficients[2, 4], f$df.residual),
         g = c(summary(g)$coefficients[2, 4], g$df.residual)))
""", {"svy": frame, "rep": rep}, tmp_path)
    r = A.agreement(frame["a"], frame["b"], design=_design(frame))
    assert r.proportional.df == ref["f"][1] == r.df - 1
    assert r.proportional.p == pytest.approx(ref["f"][0], rel=1e-7)
    g = A.agreement(rep["a"], rep["b"], units=rep["id"])
    assert g.proportional.df == ref["g"][1] == 9 - 2
    assert g.proportional.p == pytest.approx(ref["g"][0], rel=1e-7)


def test_a_truncated_between_person_variance_takes_zous_standard_error_for_the_bias():
    rng = np.random.default_rng(8)
    pattern = np.array([-3.0, 3.0, -2.5, 2.5])
    rows = [{"id": i, "a": 50 + x + d, "b": 50 + x}
            for i in range(6) for d, x in zip(pattern + rng.normal(0, 0.05, 4),
                                              rng.normal(0, 5, 4))]
    frame = pd.DataFrame(rows)
    r = A.agreement(frame["a"], frame["b"], units=frame["id"])
    assert r.components["between"] == 0.0
    means = (frame["a"] - frame["b"]).groupby(frame["id"]).mean()
    assert r.bias.se == pytest.approx(math.sqrt(means.var(ddof=1) / 6), rel=1e-12)
    assert any("Zou 2013" in c for c in r.concerns)


def test_the_contract_states_its_scope_and_keys_the_survey_rules_to_describe():
    from turbotab.core import contracts as C

    c = C.contracts()["bland_altman"]
    assert c.scope == "descriptive"
    for rid in ("design_based", "repeats_not_weighted", "design_df"):
        assert c.relation(rid).purposes == (A.DESCRIBE,), rid
    # under inference every option is not offered: only the refusal that says so applies there
    assert [r.name for r in c.relations if "inference" in r.purposes] == ["estimate_refused"]
    assert not C.fired({"bland_altman": "differences"}, "inference", ("population_design",))


def test_the_refusals_speak_plainly_with_the_survey_terms_as_a_label():
    frame = _surveyed().iloc[:40].copy()
    frame["stratum"] = np.arange(40) % 4
    frame["psu"] = 1  # one cluster in each layer: nothing left for an interval
    with pytest.raises(A.AgreementRefused) as refused:
        A.agreement(frame["a"], frame["b"], design=_design(frame))
    words = str(refused.value)
    plain = words.split("(Survey terms")[0]
    assert "PSU" not in plain and "strat" not in plain and "degrees of freedom" not in plain
    assert "(Survey terms:" in words
    from turbotab.core.models.selection import OutOfFold

    oof = OutOfFold("multiclass", np.zeros((6, 1)), np.zeros(6), [])
    oof.predictions.update(one=np.zeros((6, 3)), two=np.zeros((6, 3)))
    with pytest.raises(A.AgreementRefused, match="more than two categories"):
        A.model_agreement(oof, "one", "two")

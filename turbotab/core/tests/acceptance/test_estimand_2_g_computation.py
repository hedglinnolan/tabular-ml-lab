"""ESTIMAND · 2 · marginal standardization (g-computation) to a risk difference and a risk ratio
(MODELING_SEQUENCE §0 ruling 9: "Marginal standardization (g-computation) to an RD or RR is
offered, ranked first for a common binary outcome").

On a cohort whose yes/no outcome is common (about 30%), declared with the marginal risk difference,
the ``effects`` stage standardizes the logistic model over the analyzed rows:

* the point estimates agree with a NumPy computation written out here (Newton–Raphson on the
  design built by hand, the mean predicted risk under each setting) to 1e-8, and with R (``glm``
  with a binomial family, then ``mean(predict(…, type = "response"))`` on the data with the
  exposure set each way) to 1e-8: a continuous exposure one unit higher than observed, and a
  two-valued exposure set to each value;
* its 95% interval is the percentile interval of the bootstrap that refits the whole chain on each
  resample: recomputed here on the same resamples (``numpy.random.default_rng(seed)``, n rows drawn
  with replacement each time, the convention the stage documents) by statsmodels' logistic fit and
  NumPy's quantile, to 1e-8;
* the marginal measures rank first on the estimand card when the event's share is above 10% (Zhang
  & Yu 1998) and second below it.

Source check (Zhang J, Yu KF. What's the relative risk? JAMA 1998;280:1690–1691, read as its
abstract on 2026-10-05): the odds ratio "overestimates" the relative risk when the incidence of the
outcome is common, and they flag an incidence above 10%. The rule here ranks; it refuses nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d, estimand
from turbotab.core.models.effects import BOOT
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

COVARIATES = ["age", "sex", "smoking", "activity"]  # the primary set (bmi: unknown timing, beside)


def _newton(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logistic maximum likelihood, Newton–Raphson written out (no step halving needed here)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve((X.T * (p * (1 - p))) @ X, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return beta


def _risks(X0: np.ndarray, X1: np.ndarray, beta: np.ndarray) -> tuple[float, float]:
    return float(np.mean(1 / (1 + np.exp(-X0 @ beta)))), float(np.mean(1 / (1 + np.exp(-X1 @ beta))))


@pytest.fixture(scope="module")
def common(tmp_path_factory):
    frame = ef.cohort(800, seed=7, prevalence=0.3)
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference",
                  model_1=["age", "sex"])
    return frame, ef.run(frame, tmp_path_factory.mktemp("gcomp"), st, fit=False)


R_GLM = """
d <- read.csv(rows_csv)
d$sex <- factor(d$sex, levels = c("female", "male"))
d$y <- as.numeric(d$dm == "yes")
m <- glm(y ~ fiber + age + sex + smoking + activity, data = d, family = binomial,
         control = glm.control(epsilon = 1e-15, maxit = 100))
d1 <- d; d1$fiber <- d1$fiber + 1
r0 <- mean(predict(m, newdata = d, type = "response"))
r1 <- mean(predict(m, newdata = d1, type = "response"))
m2 <- glm(y ~ supplement + age + sex + smoking + activity, data = d, family = binomial,
          control = glm.control(epsilon = 1e-15, maxit = 100))
e0 <- d; e0$supplement <- 0; e1 <- d; e1$supplement <- 1
s0 <- mean(predict(m2, newdata = e0, type = "response"))
s1 <- mean(predict(m2, newdata = e1, type = "response"))
out(list(r0 = r0, r1 = r1, s0 = s0, s1 = s1))
"""


def test_2_the_point_estimates_agree_with_numpy_by_hand(common):
    frame, run = common
    [contrast] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert contrast["setting"] == "one unit higher than observed"
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES])
    beta = _newton(X.to_numpy(), y)
    X1 = X.copy()
    X1["fiber"] += 1
    r0, r1 = _risks(X.to_numpy(), X1.to_numpy(), beta)
    assert contrast["risk_low"] == pytest.approx(r0, rel=1e-8)
    assert contrast["risk_high"] == pytest.approx(r1, rel=1e-8)
    assert contrast["rd"] == pytest.approx(r1 - r0, rel=1e-8)
    assert contrast["rr"] == pytest.approx(r1 / r0, rel=1e-8)
    # the logistic model's maximum likelihood with an intercept reproduces the event's share
    assert r0 == pytest.approx(y.mean(), rel=1e-10)


@needs_r
def test_2_the_point_estimates_agree_with_r_glm(common, tmp_path):
    frame, run = common
    r = run_r(R_GLM, {"rows": frame}, tmp_path)
    [contrast] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert contrast["rd"] == pytest.approx(r["r1"] - r["r0"], rel=1e-8)
    assert contrast["rr"] == pytest.approx(r["r1"] / r["r0"], rel=1e-8)
    # a two-valued exposure is set to each of its values
    roles = {**ef.ROLES, "supplement": "exposure"}
    roles.pop("fiber")
    st = ef.state(target="dm", task="binary", event="yes", exposure="supplement",
                  measure="risk_ratio", roles=roles,
                  amounts={**ef.AMOUNTS, "code_or_count:supplement": "amount"})
    found = ef.run(frame, tmp_path / "supplement", st, fit=False)
    marginal = found["effects"]["families"][0]["marginal"]
    assert marginal["declared"] == "risk_ratio"
    [two] = marginal["contrasts"]
    assert two["setting"] == "`1` against `0`"
    assert two["risk_low"] == pytest.approx(r["s0"], rel=1e-8)
    assert two["risk_high"] == pytest.approx(r["s1"], rel=1e-8)
    assert two["rd"] == pytest.approx(r["s1"] - r["s0"], rel=1e-8)
    assert two["rr"] == pytest.approx(r["s1"] / r["s0"], rel=1e-8)


def test_2_the_interval_is_the_whole_chain_bootstrap_percentile(common):
    """The stage's 95% interval is the percentile interval over its resamples, each refitting the
    chain; recomputed here on the same resamples with statsmodels' logistic fit."""
    frame, run = common
    [contrast] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert contrast["n_boot"] == BOOT and contrast["n_failed"] == 0 and contrast["by_unit"] is None
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES]).to_numpy()
    X1 = X.copy()
    X1[:, 1] += 1
    rng = np.random.default_rng(run["state"].split.seed)
    rds, rrs = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        fit = sm.Logit(y[idx], X[idx]).fit(disp=0, method="newton", tol=1e-14, maxiter=100)
        r0, r1 = _risks(X[idx], X1[idx], np.asarray(fit.params))
        rds.append(r1 - r0)
        rrs.append(r1 / r0)
    lo, hi = np.percentile(rds, [2.5, 97.5])
    assert contrast["rd_low"] == pytest.approx(lo, rel=1e-8)
    assert contrast["rd_high"] == pytest.approx(hi, rel=1e-8)
    lo, hi = np.percentile(rrs, [2.5, 97.5])
    assert contrast["rr_low"] == pytest.approx(lo, rel=1e-8)
    assert contrast["rr_high"] == pytest.approx(hi, rel=1e-8)
    assert contrast["rd_low"] < contrast["rd"] < contrast["rd_high"]


def test_2_the_marginal_measures_rank_first_for_a_common_outcome():
    """Above 10% the marginal risk difference and ratio lead the card; below, the conditional odds
    ratio leads and the marginal ones follow, still offered (a ranking, never a refusal)."""
    common = ef.cohort(4000, seed=1, prevalence=0.3)
    rare = ef.cohort(4000, seed=1, prevalence=0.04)
    st = ef.state(target="dm", task="binary", event="yes", measure="odds_ratio")
    share_common = estimand.event_share(common["dm"], "yes")
    share_rare = estimand.event_share(rare["dm"], "yes")
    assert share_common == pytest.approx((common["dm"] == "yes").mean())
    assert share_common > 0.10 > share_rare
    first = estimand.estimand_card(st, "binary", prevalence=share_common)["measures"]
    second = estimand.estimand_card(st, "binary", prevalence=share_rare)["measures"]
    assert [m["measure"] for m in first] == ["risk_difference", "risk_ratio", "odds_ratio"]
    assert [m["measure"] for m in second] == ["odds_ratio", "risk_difference", "risk_ratio"]
    assert all(m["fitted"] for m in first + second)
    assert "Zhang & Yu 1998" in first[0]["reason"] and "Zhang" not in second[1]["reason"]
    # the event's share is the event's, not the minority class's
    assert estimand.event_share(common["dm"], "no") == pytest.approx(1 - share_common)


def test_2_rows_that_repeat_are_resampled_by_whole_unit(tmp_path):
    """When a person's rows repeat, the bootstrap draws whole people (every row of a drawn person,
    as often as the person is drawn) and refits the chain; recomputed here by drawing the people
    with the same stream and refitting with statsmodels."""
    one = ef.cohort(300, seed=13, prevalence=0.3)
    frame = pd.concat([one, one.assign(age=one["age"] + 1.0)], ignore_index=True)
    frame["dm"] = np.where(np.random.default_rng(2).random(len(frame)) < 0.33, "yes", "no")
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference",
                  grain=d.GrainSpec(grain="repeated", id_column="pid"))
    run = ef.run(frame, tmp_path, st, fit=False)
    [c] = run["effects"]["families"][0]["marginal"]["contrasts"]
    assert c["by_unit"] == "pid"
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES]).to_numpy()
    X1 = X.copy()
    X1[:, 1] += 1
    codes, uniques = pd.factorize(frame["pid"])
    people = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    rng = np.random.default_rng(st.split.seed)
    rds = []
    for _ in range(BOOT):
        idx = np.concatenate([people[g] for g in rng.integers(0, len(people), len(people))])
        fit = sm.Logit(y[idx], X[idx]).fit(disp=0, method="newton", tol=1e-14, maxiter=100)
        r0, r1 = _risks(X[idx], X1[idx], np.asarray(fit.params))
        rds.append(r1 - r0)
    lo, hi = np.percentile(rds, [2.5, 97.5])
    assert (c["rd_low"], c["rd_high"]) == pytest.approx((lo, hi), rel=1e-8)


def test_2_a_surveyed_population_blocks_the_marginal_risks_and_records_the_way_out(tmp_path):
    """No design-based standardization is built, so under the surveyed-population answer the
    marginal risks are not served as if they were the population's (MODELING_SEQUENCE §4:
    block and record, exit the sample-only attestation); the conditional odds ratio stays."""
    frame = ef.cohort(400, seed=17, prevalence=0.3)
    rng = np.random.default_rng(5)
    frame["sampling_weight"] = rng.uniform(0.5, 3.0, len(frame))
    frame["stratum"] = rng.integers(1, 5, len(frame))
    frame["psu"] = frame["stratum"] * 10 + rng.integers(1, 3, len(frame))
    roles = {**ef.ROLES, "sampling_weight": "design", "stratum": "design", "psu": "design"}
    survey = d.SurveySpec(estimand="population", weight="sampling_weight", strata="stratum",
                          psu="psu")
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference", roles=roles,
                  survey=survey, amounts={**ef.AMOUNTS, "code_or_count:stratum": "code",
                                          "code_or_count:psu": "code"})
    run = ef.run(frame, tmp_path, st, fit=False)
    marginal = run["effects"]["families"][0]["marginal"]
    assert marginal["contrasts"] == [] and marginal["refused"].startswith(
        "The marginal risks are standardized over these participants")
    sample, conditional = (e["decision"] for e in marginal["exits"])
    assert sample["kind"] == "set_survey" and sample["estimand"] == "sample"
    assert conditional["kind"] == "set_estimand" and conditional["measure"] == "odds_ratio"
    primary = ef.sequence(run["effects"])["model_2"]
    assert primary["effects"][0]["ratio"] is not None
    assert primary["inference"]["covariance"] == "design"


# ── the bootstrap's edges (REPAIR-ESTIMAND): separated resamples and too few units ────────────


def test_2_a_separated_resample_is_kept_at_the_likelihoods_limit_numpy_by_hand():
    """A resample whose outcome happens to be separated (here: the reference level drawn with no
    event) has no finite logistic estimate, but its fitted risks have a limit, the one R's ``glm``
    approaches as its coefficients run away: 0 for the separated rows. Leaving such resamples out
    would cut the interval's most extreme draws. Reference, written out: with the exposure alone
    (a saturated model) the fitted risks are each level's own event share, so on the same resamples
    (``default_rng(seed)``, n rows with replacement) the risk difference is ``ȳ_b − ȳ_a``, ``ȳ_a =
    0`` where `a` drew no event; the risk ratio is infinite there, and its upper limit with it."""
    import math

    from turbotab.core.models import effects

    rng = np.random.default_rng(41)
    diet = np.array(["a"] * 20 + ["b"] * 40 + ["c"] * 40)
    y = np.r_[np.zeros(20), rng.random(80) < 0.4].astype(float)
    y[0] = 1.0  # one event at `a`: the full data are not separated
    X = pd.DataFrame({"diet": diet})

    def fit(X_b, y_b):
        return lambda Z: np.column_stack([np.ones(len(Z)), (Z["diet"] == "b").to_numpy(float),
                                          (Z["diet"] == "c").to_numpy(float)])

    setting = effects.Setting("`b` against `a`", "a", "b")
    found = effects.g_computation(fit, X, y, "diet", setting, n_boot=BOOT, seed=5)
    assert found.risk_low == pytest.approx(y[diet == "a"].mean(), rel=1e-10)
    assert found.risk_high == pytest.approx(y[diet == "b"].mean(), rel=1e-10)
    draws = np.random.default_rng(5)
    rds, rrs, zero = [], [], 0
    for _ in range(BOOT):
        idx = draws.integers(0, len(y), len(y))
        ya, yb = y[idx][diet[idx] == "a"], y[idx][diet[idx] == "b"]
        ra, rb = ya.mean(), yb.mean()
        zero += int(ra == 0)
        rds.append(rb - ra)
        rrs.append(math.inf if ra == 0 else rb / ra)
    assert found.n_limit == zero and zero > 0.025 * BOOT and found.n_failed == 0
    lo, hi = np.percentile(rds, [2.5, 97.5])
    assert (found.rd_low, found.rd_high) == pytest.approx((lo, hi), rel=1e-10)
    assert found.rr_low == pytest.approx(np.percentile(rrs, 2.5), rel=1e-10)
    assert math.isinf(found.rr_high) and found.n_rr_infinite == zero


SEPARATED_R = """
d <- read.csv(rows_csv)
d$diet <- factor(d$diet, levels = c("a", "b", "c"))
d$sex <- factor(d$sex, levels = c("female", "male"))
d$y <- as.numeric(d$dm == "yes")
idx <- as.matrix(read.csv(idx_csv)) + 1
ctl <- glm.control(epsilon = 1e-14, maxit = 300)
f <- y ~ diet + age + sex + smoking + activity
risk <- function(m, s, v) {
  z <- s; z$diet <- factor(rep(v, nrow(z)), levels = c("a", "b", "c"))
  mean(predict(m, newdata = z, type = "response"))
}
m <- glm(f, data = d, family = binomial, control = ctl)
point <- c(risk(m, d, "a"), risk(m, d, "b"), risk(m, d, "c"))
rd_b <- rd_c <- numeric(nrow(idx))
for (b in seq_len(nrow(idx))) {
  s <- d[idx[b, ], ]
  m <- suppressWarnings(glm(f, data = s, family = binomial, control = ctl))
  ra <- risk(m, s, "a")
  rd_b[b] <- risk(m, s, "b") - ra
  rd_c[b] <- risk(m, s, "c") - ra
}
out(list(point = point, b = unname(quantile(rd_b, c(0.025, 0.975), type = 7)),
         c = unname(quantile(rd_c, c(0.025, 0.975), type = 7))))
"""


def _separating_cohort() -> pd.DataFrame:
    """A three-level text exposure whose reference level `a` holds 45 rows and 2 events: about one
    resample in eight draws neither event, and its outcome is separated there."""
    frame = ef.cohort(650, seed=7, prevalence=0.3)
    level = np.where(np.arange(650) < 45, "a", np.where(np.arange(650) % 2, "b", "c"))
    dm = frame["dm"].to_numpy().copy()
    at_a = np.flatnonzero(level == "a")
    dm[at_a] = "no"
    dm[at_a[:2]] = "yes"
    return frame.assign(diet=level, dm=dm)


@needs_r
def test_2_separated_resamples_agree_with_r_glm_on_the_same_resamples(tmp_path):
    """Through the stage: each level against `a`, the interval from all 1,000 resamples, those whose
    outcome is separated kept at their limit and counted; the percentile limits agree with R's
    ``glm`` (``epsilon = 1e-14``, run to its limit) refit on the same resamples and standardized by
    ``predict``, to 1e-8. The count is the resamples that drew no event at `a`, counted here; the
    concern and the methods sentence say it."""
    frame = _separating_cohort()
    roles = {**ef.ROLES, "diet": "exposure"}
    roles.pop("fiber")
    st = ef.state(target="dm", task="binary", event="yes", exposure="diet",
                  measure="risk_difference", roles=roles)
    run = ef.run(frame, tmp_path / "stage", st, fit=False)
    marginal = run["effects"]["families"][0]["marginal"]
    b, c = marginal["contrasts"]
    assert (b["setting"], c["setting"]) == ("`b` against `a`", "`c` against `a`")
    rng = np.random.default_rng(st.split.seed)
    idx = np.stack([rng.integers(0, len(frame), len(frame)) for _ in range(BOOT)])
    y = (frame["dm"] == "yes").to_numpy()
    at_a = (frame["diet"] == "a").to_numpy()
    separated = int(sum(not y[i][at_a[i]].any() for i in idx))
    assert 0.05 * BOOT < separated < 0.25 * BOOT
    r = run_r(SEPARATED_R, {"rows": frame, "idx": pd.DataFrame(idx)}, tmp_path / "r")
    assert b["risk_low"] == pytest.approx(r["point"][0], rel=1e-8)
    assert b["risk_high"] == pytest.approx(r["point"][1], rel=1e-8)
    assert c["risk_high"] == pytest.approx(r["point"][2], rel=1e-8)
    for mine, key in ((b, "b"), (c, "c")):
        assert mine["n_boot"] == BOOT and mine["n_failed"] == 0 and mine["n_limit"] == separated
        assert (mine["rd_low"], mine["rd_high"]) == pytest.approx(tuple(r[key]), abs=1e-8)
        # at `a`'s limit risk of 0 the ratio is infinite in more than 2.5% of resamples
        assert mine["rr_unbounded"] is True and mine["rr_high"] is None
        assert mine["n_rr_infinite"] == separated
    assert marginal["concerns"][0] == (
        f"`b` against `a`: In {separated:,} of the 1,000 bootstrap resamples the covariates "
        f"separated the outcome (a level or a combination with no event, or only events, in that "
        f"resample). Each is kept at the likelihood's limit, where the separated rows' risks are 0 "
        f"or 1, the fit R's glm approaches as its coefficients run away; leaving them out would "
        f"cut the interval's most extreme resamples.")
    assert (f"with 95% percentile intervals from 1,000 bootstrap resamples refitting the whole "
            f"chain; in {separated:,} of them the covariates separated the outcome, and the model "
            f"was taken at the likelihood's limit (the separated rows' risks 0 or 1). "
            in run["effects"]["methods"])


def test_2_below_the_unit_floor_the_marginal_risks_carry_no_interval(tmp_path):
    """Rows that repeat within 4 sites, fewer than the unit floor (``inference.min_clusters``): the
    fit's tables refuse every interval (§2: a refusal below a floor), and so does the marginal
    standardization, a bootstrap of four units being no better than the sandwich it would replace.
    The risks themselves are reported (agreeing with the logistic model written out here, to 1e-8),
    with the same reason and exits as the tables; the E-value has no limit, and says why."""
    from turbotab.core.models.inference import min_clusters

    frame = ef.cohort(600, seed=9, prevalence=0.3).assign(site=lambda f: np.arange(len(f)) % 4)
    roles = {**ef.ROLES, "site": "cluster"}
    st = ef.state(target="dm", task="binary", event="yes", measure="risk_difference",
                  roles=roles, amounts={**ef.AMOUNTS, "code_or_count:site": "code"})
    run = ef.run(frame, tmp_path, st, fit=False)
    family = run["effects"]["families"][0]
    marginal = family["marginal"]
    [c] = marginal["contrasts"]
    assert min_clusters() > 4
    y = (frame["dm"] == "yes").astype(float).to_numpy()
    X = ef.design_matrix(frame, ["fiber", *COVARIATES])
    beta = _newton(X.to_numpy(), y)
    X1 = X.copy()
    X1["fiber"] += 1
    r0, r1 = _risks(X.to_numpy(), X1.to_numpy(), beta)
    assert (c["risk_low"], c["risk_high"]) == pytest.approx((r0, r1), rel=1e-8)
    assert c["rd_low"] is None and c["rd_high"] is None and c["rr_low"] is None
    assert c["n_boot"] == 0 and c["by_unit"] is None
    primary = ef.sequence(run["effects"])["model_2"]
    assert primary["inference"]["refused"] == marginal["interval_refused"]
    assert marginal["interval_refused"].startswith(f"`site` has 4 units, fewer than the "
                                                   f"{min_clusters()} TurboTab requires")
    assert [e["label"] for e in marginal["exits"]] == [e["label"] for e in
                                                       primary["inference"]["exits"]]
    [sens] = family["sensitivity"]
    inverse = 1 / c["rr"]  # a protective ratio's E-value is that of its inverse (VanderWeele & Ding)
    assert sens["e_value"]["limit"] is None
    assert sens["e_value"]["point"] == pytest.approx(inverse + np.sqrt(inverse * (inverse - 1)),
                                                     rel=1e-10)
    assert "for the confidence limit nearer the null none, as no interval is reported" in sens["reading"]
    assert ("from the logistic model; no interval is reported (`site` has 4 units, fewer than the "
            f"{min_clusters()} TurboTab requires for cluster-robust intervals: "
            in run["effects"]["methods"])

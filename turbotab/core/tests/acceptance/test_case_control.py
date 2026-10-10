"""E4: case-control samples and matched sets as an engine method core
(`turbotab.core.methods.case_control`).

Every number is held to an independent reference: R's ``survival::clogit`` on its own worked
example (the ``infert`` matched study of Trichopoulos et al., shipped with R) and on sets holding
several cases (``method = "exact"``); R's ``glm`` for unmatched and frequency-matched samples; R's
``survey::svyglm`` for the population answer; King & Zeng's (2001) prior correction written out by
hand on R's ``glm`` fit; R's ``binom.test`` for the share signal. A package R does not find skips
its test.
"""
from __future__ import annotations

import math
import subprocess

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import case_control as CC
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r


def _r_has(package: str) -> bool:
    if RSCRIPT is None:
        return False
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


def _skip_without(package: str) -> None:
    if not _r_has(package):
        pytest.skip(f"R's {package} is not installed (set R_LIBS to a library that has it)")


def _infert(tmp_path) -> pd.DataFrame:
    ref = run_r("data(infert); out(list(d = infert))", {}, tmp_path / "infert")
    return pd.DataFrame(ref["d"])


def _multi(seed: int = 3) -> pd.DataFrame:
    """Matched sets of 3 to 6 people holding one or two cases, a categorical and a numeric
    exposure, an age matched within each set; one set with no case and one person alone."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(70):
        m = int(rng.integers(3, 7))
        d = 2 if s % 3 == 0 else 1
        age = int(rng.integers(40, 70))
        x = rng.normal(0, 1, m)
        smoke = rng.choice(["never", "former", "current"], m, p=[0.5, 0.3, 0.2])
        eta = 0.7 * x + np.where(smoke == "current", 0.9, np.where(smoke == "former", 0.3, 0.0))
        p = np.exp(eta) / np.exp(eta).sum()
        cases = rng.choice(m, d, replace=False, p=p)
        for i in range(m):
            rows.append({"set": s + 1, "case": int(i in cases), "x": x[i], "smoke": smoke[i],
                         "age": age})
    rows += [{"set": 71, "case": 0, "x": 0.3, "smoke": "never", "age": 50},
             {"set": 71, "case": 0, "x": -0.2, "smoke": "former", "age": 50},
             {"set": 72, "case": 1, "x": 1.1, "smoke": "current", "age": 61}]
    return pd.DataFrame(rows)


def _population(seed: int = 11, n: int = 40_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    band = rng.choice(["40-49", "50-59", "60-69"], n)
    eta = -4.0 + 0.8 * x - 0.5 * z + np.where(band == "60-69", 1.0, np.where(band == "50-59", 0.5, 0))
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return pd.DataFrame({"case": y, "x": x, "z": z, "band": band, "risk": 1 / (1 + np.exp(-eta))})


def _sampled(pop: pd.DataFrame, controls_per_case: int = 2, seed: int = 4) -> pd.DataFrame:
    """Every case and a random draw of controls: an unmatched case-control sample."""
    rng = np.random.default_rng(seed)
    cases = pop.index[pop["case"] == 1]
    controls = rng.choice(pop.index[pop["case"] == 0], controls_per_case * len(cases), replace=False)
    return pop.loc[np.concatenate([cases, controls])].reset_index(drop=True)


# ── matched sets against survival::clogit ────────────────────────────────────


@needs_r
def test_the_infert_matched_study_gives_clogits_own_numbers(tmp_path):
    """R's own clogit example: ``clogit(case ~ spontaneous + induced + strata(stratum), infert)``,
    83 sets of one case and two controls (one with a single control), matched on age, parity and
    education."""
    ref = run_r("""
library(survival)
data(infert)
f <- clogit(case ~ spontaneous + induced + strata(stratum), data = infert)
s <- summary(f)
out(list(coef = unname(coef(f)), se = unname(sqrt(diag(vcov(f)))),
         ci = unname(s$conf.int[, 3:4]), p = unname(s$coefficients[, 5]),
         loglik = f$loglik, lr = unname(s$logtest)))
""", {}, tmp_path)
    df = _infert(tmp_path)
    r = CC.odds_ratios(df, "case", ["spontaneous", "induced"], sampling="individually_matched",
                       sets="stratum", matching=["age", "parity", "education"])
    assert [row.log_or for row in r.rows] == pytest.approx(ref["coef"], rel=1e-9)
    assert [row.se for row in r.rows] == pytest.approx(ref["se"], rel=1e-8)
    lo, hi = np.asarray(ref["ci"]).T
    assert [row.lower for row in r.rows] == pytest.approx(list(lo), rel=1e-8)
    assert [row.upper for row in r.rows] == pytest.approx(list(hi), rel=1e-8)
    assert [row.p for row in r.rows] == pytest.approx(ref["p"], rel=1e-6)
    assert [r.loglik_null, r.loglik] == pytest.approx(ref["loglik"], rel=1e-10)
    assert r.lr_test[0] == pytest.approx(ref["lr"][0], rel=1e-9) and r.lr_test[1] == ref["lr"][1]
    assert r.n_informative_sets == 83 and r.controls_per_case == {"1:1": 1, "1:2": 82}
    assert r.n_cases == 83 and r.n_controls == 165
    assert "7.29 times as high per unit of spontaneous" in r.says
    sentence = CC.case_control_sentence(r)
    assert "82 sets of one case and two controls" in sentence
    assert "Gail" not in sentence  # no set held more than one case
    assert "Prevalence and absolute risks were not estimated" in sentence


@needs_r
def test_sets_with_several_cases_take_the_exact_conditional_likelihood(tmp_path):
    """Sets of two cases and up to four controls, a three-level category coded as R codes a
    factor (its first level, alphabetically, the reference), a set with no case and a person alone
    (both uninformative): R's clogit with ``method = "exact"``."""
    df = _multi()
    ref = run_r("""
library(survival)
df <- read.csv(multi_csv)
df$smoke <- factor(df$smoke)
f <- clogit(case ~ x + smoke + strata(set), data = df, method = "exact")
out(list(coef = unname(coef(f)), se = unname(sqrt(diag(vcov(f)))), names = names(coef(f)),
         loglik = f$loglik, n = f$n, nevent = f$nevent))
""", {"multi": df}, tmp_path)
    r = CC.odds_ratios(df, "case", ["x", "smoke"], sampling="individually_matched", sets="set",
                       matching=["age"])
    assert [row.term for row in r.rows] == ["x", "smoke=former", "smoke=never"]
    assert ref["names"] == ["x", "smokeformer", "smokenever"]
    assert [row.log_or for row in r.rows] == pytest.approx(ref["coef"], rel=1e-8)
    assert [row.se for row in r.rows] == pytest.approx(ref["se"], rel=1e-7)
    assert [r.loglik_null, r.loglik] == pytest.approx(ref["loglik"], rel=1e-10)
    assert r.n_informative_sets == 70 and r.n_sets == 72
    assert any(k.startswith("2:") for k in r.controls_per_case)
    assert any("2 sets without both a case and a control" in c for c in r.concerns)
    assert any("Gail, Lubin & Rubinstein 1981" in c for c in r.concerns)
    assert "Gail, Lubin & Rubinstein 1981" in CC.case_control_sentence(r)


def test_one_case_per_set_is_the_stratified_cox_partial_likelihood_by_hand():
    """With one case per set the conditional likelihood is Σ_sets [η_case − log Σ_members e^η]."""
    df = _multi(seed=8)
    df = df[df.groupby("set")["case"].transform("sum") == 1]
    fit = CC.conditional_logistic(df[["x"]], df["case"], df["set"])
    b = fit.beta[0]
    by_hand = sum(float(g["x"][g["case"] == 1].iloc[0] * b - np.log(np.exp(b * g["x"]).sum()))
                  for _, g in df.groupby("set"))
    assert fit.loglik == pytest.approx(by_hand, rel=1e-12)
    assert fit.loglik_null == pytest.approx(-sum(np.log(len(g)) for _, g in df.groupby("set")))


# ── unmatched and frequency-matched samples against glm ──────────────────────


@needs_r
def test_unmatched_and_frequency_matched_odds_ratios_match_glm(tmp_path):
    df = _sampled(_population())
    df.loc[[3, 9], "z"] = np.nan  # rows with a missing value leave
    ref = run_r("""
df <- read.csv(cc_csv)
df$band <- factor(df$band)
df <- df[complete.cases(df[, c("case", "x", "z", "band")]), ]
tight <- glm.control(epsilon = 1e-14, maxit = 100)
f <- glm(case ~ x + z + band, family = binomial, data = df, control = tight)
g <- glm(case ~ x + z, family = binomial, data = df, control = tight)
null <- glm(case ~ 1, family = binomial, data = df)
strata <- glm(case ~ band, family = binomial, data = df, control = tight)
lrt <- anova(strata, f, test = "LRT")
ci <- confint.default(f)
out(list(coef = unname(coef(f)), se = unname(sqrt(diag(vcov(f)))), lo = unname(ci[, 1]),
         hi = unname(ci[, 2]), p = unname(summary(f)$coefficients[, 4]),
         ll = as.numeric(logLik(f)), ll0 = as.numeric(logLik(null)),
         ll_strata = as.numeric(logLik(strata)), lr = c(lrt$Deviance[2], lrt$Df[2],
         lrt[["Pr(>Chi)"]][2]),
         g = unname(coef(g)), gse = unname(sqrt(diag(vcov(g)))), n = nrow(df),
         gll0 = as.numeric(logLik(glm(case ~ 1, family = binomial, data = df))),
         glr = c(2 * (as.numeric(logLik(g)) - as.numeric(logLik(null))), 2)))
""", {"cc": df}, tmp_path)
    freq = CC.odds_ratios(df, "case", ["x"], sampling="frequency_matched", matching=["band"],
                          adjust=["z"])
    # x and z only: the matching factor's own odds ratios are not reported
    assert [row.term for row in freq.rows] == ["x", "z"]
    assert [row.log_or for row in freq.rows] == pytest.approx(ref["coef"][1:3], rel=1e-8)
    assert [row.se for row in freq.rows] == pytest.approx(ref["se"][1:3], rel=1e-7)
    assert [row.lower for row in freq.rows] == pytest.approx(np.exp(ref["lo"][1:3]), rel=1e-7)
    assert [row.upper for row in freq.rows] == pytest.approx(np.exp(ref["hi"][1:3]), rel=1e-7)
    assert [row.p for row in freq.rows] == pytest.approx(ref["p"][1:3], rel=1e-6)
    assert freq.loglik == pytest.approx(ref["ll"], rel=1e-10)
    # the likelihood-ratio test covers the reported terms (x, z) only, against the model with
    # the matching strata alone, as anova(glm(case ~ band), glm(case ~ x + z + band)) does
    assert freq.loglik_null == pytest.approx(ref["ll_strata"], rel=1e-10)
    assert freq.lr_test[0] == pytest.approx(ref["lr"][0], rel=1e-8)
    assert freq.lr_test[1] == ref["lr"][1] == 2
    assert freq.lr_test[2] == pytest.approx(ref["lr"][2], rel=1e-6)
    assert freq.n_rows == ref["n"] and freq.matching == ("band",)
    assert any("2 rows with a missing value leave" in c for c in freq.concerns)
    assert any("one indicator per stratum" in c for c in freq.concerns)
    # a matching stratum coded as numbers (0, 1, 2) is still three strata, not one slope
    df["band_code"] = df["band"].map({"40-49": 0, "50-59": 1, "60-69": 2})
    coded = CC.odds_ratios(df, "case", ["x"], sampling="frequency_matched",
                           matching=["band_code"], adjust=["z"])
    assert [row.log_or for row in coded.rows] == pytest.approx(ref["coef"][1:3], rel=1e-8)
    assert [row.se for row in coded.rows] == pytest.approx(ref["se"][1:3], rel=1e-7)
    assert coded.loglik == pytest.approx(ref["ll"], rel=1e-10)
    assert coded.lr_test[1] == 2
    plain = CC.odds_ratios(df, "case", ["x", "z"])
    assert [row.log_or for row in plain.rows] == pytest.approx(ref["g"][1:], rel=1e-8)
    assert [row.se for row in plain.rows] == pytest.approx(ref["gse"][1:], rel=1e-7)
    assert plain.loglik_null == pytest.approx(ref["gll0"], rel=1e-10)
    assert plain.lr_test[0] == pytest.approx(ref["glr"][0], rel=1e-8) and plain.lr_test[1] == 2
    assert "intercept reflects the sampling" in CC.case_control_sentence(freq)
    assert "adjusted for the matching factors (band)" in CC.case_control_sentence(freq)


def test_a_frequency_matched_sample_with_no_matching_factor_named_is_refused():
    df = _sampled(_population())
    with pytest.raises(CC.CaseControlRefused) as refused:
        CC.odds_ratios(df, "case", ["x"], sampling="frequency_matched")
    assert refused.value.exits == [{"label": "Name the matching factors", "needs": "matching"}]
    with pytest.raises(CC.CaseControlRefused, match="matched on") as both:
        CC.odds_ratios(df, "case", ["x", "band"], sampling="frequency_matched", matching=["band"])
    assert both.value.exits[0]["drop"] == ["band"]


def test_a_matching_factor_given_as_a_raw_measurement_is_a_straight_line_with_a_concern():
    """A numeric matching factor with more distinct values than strata are drawn in is the
    measurement itself: it is adjusted for as a straight line (the glm-checked fit with it among
    the adjustment columns), and the concern says so and asks for the band column."""
    df = _sampled(_population())
    df["age"] = np.random.default_rng(1).integers(40, 70, len(df))
    fit = CC.odds_ratios(df, "case", ["x"], sampling="frequency_matched", matching=["age"])
    line = CC.odds_ratios(df, "case", ["x", "age"])
    assert fit.row("x").log_or == pytest.approx(line.row("x").log_or, rel=1e-12)
    assert fit.row("x").se == pytest.approx(line.row("x").se, rel=1e-12)
    assert [row.term for row in fit.rows] == ["x"] and fit.lr_test[1] == 1
    assert any("age has 30 distinct values" in c and "name the band column" in c
               for c in fit.concerns)
    assert any("(age as a straight line)" in c for c in fit.concerns)


# ── the population answer ────────────────────────────────────────────────────


def _surveyed(seed: int = 21) -> pd.DataFrame:
    df = _sampled(_population(seed=seed), seed=seed)
    rng = np.random.default_rng(seed)
    df["stratum"] = rng.integers(1, 6, len(df))
    df["psu"] = df["stratum"] * 10 + rng.integers(1, 4, len(df))
    df["w"] = rng.uniform(0.5, 4.0, len(df))
    return df


def _design(frame: pd.DataFrame):
    from turbotab.core.models.survey import build_design

    return build_design(frame, frame["w"].to_numpy(), weight_column="w", strata_column="stratum",
                        psu_column="psu")


@needs_r
def test_the_population_answer_matches_svyglm(tmp_path):
    _skip_without("survey")
    df = _surveyed()
    ref = run_r("""
library(survey)
df <- read.csv(svy_csv)
df$band <- factor(df$band)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = df)
f <- svyglm(case ~ x + z + band, design = des, family = quasibinomial(),
            control = glm.control(epsilon = 1e-14, maxit = 100))
out(list(coef = unname(coef(f)), se = unname(SE(f)), degf = degf(des)))
""", {"svy": df}, tmp_path)
    r = CC.odds_ratios(df, "case", ["x", "z"], sampling="frequency_matched", matching=["band"],
                       design=_design(df))
    assert r.weighted and r.df == ref["degf"]
    from scipy import stats

    t = stats.t.ppf(0.975, ref["degf"])
    for row, b, s in zip(r.rows, ref["coef"][1:3], ref["se"][1:3]):
        assert row.log_or == pytest.approx(b, rel=1e-8)
        assert row.se == pytest.approx(s, rel=1e-7)
        assert row.lower == pytest.approx(math.exp(b - t * s), rel=1e-7)
        assert row.upper == pytest.approx(math.exp(b + t * s), rel=1e-7)
        assert row.df == ref["degf"]
    assert "linearization" in CC.case_control_sentence(r)


def test_matched_sets_under_a_survey_design_are_refused_with_exits():
    df = _multi()
    df["stratum"], df["psu"], df["w"] = 1, df["set"], 1.5
    with pytest.raises(CC.CaseControlRefused) as refused:
        CC.odds_ratios(df, "case", ["x"], sampling="individually_matched", sets="set",
                       matching=["age"], design=_design(df))
    assert "no survey-weighted version" in refused.value.reason
    assert refused.value.exits[0]["decision"] == {"kind": "set_survey", "estimand": "sample"}
    assert refused.value.exits[1]["sampling"] == "frequency_matched"
    # the frequency-matched exit runs
    assert CC.odds_ratios(df, "case", ["x"], sampling="frequency_matched", matching=["age"],
                          design=_design(df)).weighted


# ── what is refused, and why ─────────────────────────────────────────────────


@pytest.mark.parametrize("sampling", CC.SAMPLINGS)
@pytest.mark.parametrize("goal", CC.GOALS)
def test_prevalence_and_absolute_risk_are_refused_under_every_goal(goal, sampling):
    prev = CC.eligible(goal, sampling, "prevalence")
    assert not prev.offered and "sampled separately" in prev.says
    assert prev.exits[0]["ask"] == "odds_ratios"
    if goal != "predict":
        risk = CC.eligible(goal, sampling, "absolute_risk")
        assert not risk.offered and risk.exits[0]["ask"] == "odds_ratios"
    with pytest.raises(CC.CaseControlRefused) as refused:
        prev.require()
    assert "(Technical term: outcome-dependent sampling" in str(refused.value)
    assert refused.value.reason == prev.says


def test_the_goals_offer_odds_ratios_under_estimate_and_risks_only_with_the_prevalence():
    for sampling in CC.SAMPLINGS:
        assert CC.eligible("estimate", sampling, "odds_ratios").offered
        assert not CC.eligible("predict", sampling, "odds_ratios").offered
        assert not CC.eligible("describe", sampling, "odds_ratios").offered
    assert CC.eligible("predict", "unmatched", "predicted_risk", prevalence=0.02).offered
    no_prev = CC.eligible("predict", "unmatched", "predicted_risk")
    assert not no_prev.offered and {e.get("needs") for e in no_prev.exits} >= {
        "population_prevalence"}
    one = CC.eligible("predict", "frequency_matched", "predicted_risk", prevalence=0.02)
    assert not one.offered and one.exits[0]["needs"] == "prevalence_by_stratum"
    assert CC.eligible("predict", "frequency_matched", "predicted_risk",
                       prevalence={"a": 0.01}).offered
    assert CC.eligible("predict", "unmatched", "ranking").offered
    assert "Janes & Pepe 2008" in CC.eligible("predict", "frequency_matched", "ranking").says
    assert not CC.eligible("predict", "individually_matched", "ranking").offered
    assert CC.eligible("predict", "individually_matched", "within_set_ranking").offered
    assert not CC.eligible("predict", "unmatched", "within_set_ranking").offered


def test_the_option_offered_first_follows_the_sampling_and_the_stated_prevalence():
    from turbotab.core.reference import defaults

    first = CC.offered_first
    assert first("case_control_effects", "estimate", "unmatched")[0] == "unconditional"
    assert first("case_control_effects", "estimate", "individually_matched")[0] == "conditional"
    assert first("case_control_risks", "predict", "unmatched", prevalence=0.01)[0] == \
        "prior_correction"
    assert first("case_control_risks", "predict", "unmatched")[0] == "ranking_only"
    assert first("case_control_risks", "predict", "frequency_matched",
                 prevalence={"a": 0.01})[0] == "prior_correction"
    assert first("case_control_risks", "predict", "individually_matched",
                 prevalence=0.01)[0] == "within_set_ranking"
    # the packets defer to these cases instead of printing the registry's static rank 1
    assert [c.key for c in defaults.cases("case_control_effects", "inference")] == [
        "unconditional", "unconditional", "conditional"]
    assert defaults.cases("case_control_risks", "inference") is None


def test_absolute_risk_for_matched_sets_is_refused_for_a_methods_reason():
    e = CC.eligible("predict", "individually_matched", "predicted_risk", prevalence=0.02)
    assert not e.offered
    assert "limit of the design, not a missing feature" in e.says
    assert "no baseline risk" in e.says
    assert [x["ask"] for x in e.exits] == ["within_set_ranking", "odds_ratios"]
    from sklearn.linear_model import LogisticRegression

    df = _multi()
    with pytest.raises(CC.CaseControlRefused) as refused:
        CC.predicted_risks(LogisticRegression(), df[["x"]], df["case"],
                           sampling="individually_matched", prevalence=0.02, sets=df["set"])
    assert refused.value.exits[0]["ask"] == "within_set_ranking"
    assert "conditional likelihood" in refused.value.term


def test_a_matching_factor_or_a_separating_exposure_is_refused_in_the_conditional_model():
    df = _multi()
    with pytest.raises(CC.CaseControlRefused, match="same for every member of each matched set") \
            as matched_on:
        CC.odds_ratios(df, "case", ["x", "age"], sampling="individually_matched", sets="set")
    assert matched_on.value.exits == [{"label": "Leave age out", "drop": ["age"]}]
    df["perfect"] = df["case"] * 2.0 + df["set"] * 0.01  # every case above its own controls
    with pytest.raises(CC.CaseControlRefused, match="infinite") as separated:
        CC.odds_ratios(df, "case", ["x", "perfect"], sampling="individually_matched", sets="set")
    # only the separating column is named: the exposure beside it stays
    assert separated.value.exits == [{"label": "Leave perfect out", "drop": ["perfect"]}]
    assert separated.value.reason.startswith("Within every set, perfect puts each case above")
    df["also"] = df["case"] * 3.0 + np.random.default_rng(5).uniform(0, 0.5, len(df))
    with pytest.raises(CC.CaseControlRefused) as two:
        CC.odds_ratios(df, "case", ["x", "perfect", "also"], sampling="individually_matched",
                       sets="set")
    assert "perfect and also each put each case above" in two.value.reason
    assert two.value.exits == [{"label": "Leave perfect and also out", "drop": ["perfect", "also"]}]
    pop = _sampled(_population())
    pop["flag"] = pop["case"]
    with pytest.raises(CC.CaseControlRefused, match="the cases from the controls completely") \
            as flagged:
        CC.odds_ratios(pop, "case", ["x", "flag"])
    assert flagged.value.exits == [{"label": "Leave flag out", "drop": ["flag"]}]
    assert "x" not in flagged.value.reason.split(" separates")[0]
    with pytest.raises(CC.CaseControlRefused, match="no column names the sets"):
        CC.odds_ratios(df, "case", ["x"], sampling="individually_matched")


def test_a_level_constant_within_every_set_is_named_with_a_merge_exit():
    """A category level that only appears in sets where every member has it has no odds ratio
    within sets, but the other levels of its column do: the level is named, not the column."""
    df = _multi()
    first = df["set"].isin([1, 2])
    df.loc[first, "smoke"] = "pipe"
    with pytest.raises(CC.CaseControlRefused) as level:
        CC.odds_ratios(df, "case", ["x", "smoke"], sampling="individually_matched", sets="set")
    assert "either everyone or no one has smoke = pipe" in level.value.reason
    assert "matched on" not in level.value.reason
    assert level.value.exits == [
        {"label": "Merge smoke = pipe into another level of smoke",
         "merge": {"column": "smoke", "levels": ["pipe"]}},
        {"label": "Leave smoke out", "drop": ["smoke"]}]
    assert "level constant within every matched set" in level.value.term
    # following the merge exit fits, with smoke's other levels estimated
    df.loc[first, "smoke"] = "current"
    fit = CC.odds_ratios(df, "case", ["x", "smoke"], sampling="individually_matched", sets="set")
    assert [row.term for row in fit.rows] == ["x", "smoke=former", "smoke=never"]


def test_unmatched_separation_names_the_separating_level_or_the_smallest_group():
    pop = _sampled(_population())
    pop["e"] = np.where(np.arange(len(pop)) % 2 == 0, "a", "b")
    pop.loc[pop.index[pop["case"] == 1][:4], "e"] = "rare"  # a level held by cases alone
    with pytest.raises(CC.CaseControlRefused) as rare:
        CC.odds_ratios(pop, "case", ["x", "e"])
    assert rare.value.reason.startswith("e = rare separates the cases from the controls")
    assert rare.value.exits[0] == {"label": "Merge e = rare into another level of e",
                                   "merge": {"column": "e", "levels": ["rare"]}}
    rng = np.random.default_rng(2)
    sign = np.where(pop["case"] == 1, 1.0, -1.0)
    pop["u"] = rng.uniform(-10, 10, len(pop))
    pop["v"] = -pop["u"] + sign * rng.uniform(0.1, 1.0, len(pop))  # u + v separates; neither alone
    with pytest.raises(CC.CaseControlRefused) as joint:
        CC.odds_ratios(pop, "case", ["x", "u", "v"])
    assert joint.value.reason.startswith("Together, u and v separate the cases")
    assert joint.value.exits == [{"label": "Leave u out", "drop": ["u"]},
                                 {"label": "Leave v out", "drop": ["v"]}]


# ── Predict: the prior correction, inside each training fold ─────────────────


def _lr():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000)


@needs_r
def test_the_prior_correction_is_king_and_zengs_equation_on_glms_fit(tmp_path):
    """King & Zeng 2001, eq. 7: the corrected intercept is β̂₀ − ln[((1 − τ)/τ)(ȳ/(1 − ȳ))]; the
    slopes are glm's. Every risk is plogis of that line."""
    df = _sampled(_population())
    tau = 0.035
    ref = run_r("""
df <- read.csv(cc_csv)
f <- glm(case ~ x + z, family = binomial, data = df,
         control = glm.control(epsilon = 1e-14, maxit = 100))
out(list(coef = unname(coef(f)), ybar = mean(df$case)))
""", {"cc": df[["case", "x", "z"]]}, tmp_path)
    b0, b1, b2 = ref["coef"]
    ybar = ref["ybar"]
    corrected = b0 - math.log(((1 - tau) / tau) * (ybar / (1 - ybar)))
    m = CC.PriorCorrected(_lr(), tau).fit(df[["x", "z"]], df["case"])
    assert m.sample_share_ == pytest.approx(ybar, rel=1e-15)
    assert b0 - m.offset_ == pytest.approx(corrected, rel=1e-14)
    risk = m.predict_proba(df[["x", "z"]])[:, 1]
    by_hand = 1 / (1 + np.exp(-(corrected + b1 * df["x"] + b2 * df["z"])))
    assert risk == pytest.approx(by_hand.to_numpy(), rel=1e-5)
    # τ equal to the sample's own share leaves the probabilities as they were
    same = CC.PriorCorrected(_lr(), ybar).fit(df[["x", "z"]], df["case"])
    assert same.offset_ == pytest.approx(0.0, abs=1e-14)


def test_recalibrated_risks_recover_the_populations_risks():
    """A statistical check of the whole correction: trained on every case and two controls each,
    recalibrated to the population's prevalence, the risks agree with the true risks and their
    average over the population with its prevalence."""
    pop = _population(seed=12, n=60_000)
    pop["band_code"] = pop["band"].map({"40-49": 0, "50-59": 1, "60-69": 2})
    df = _sampled(pop, seed=5)
    tau = float(pop["case"].mean())
    features = ["x", "z", "band_code"]
    m = CC.PriorCorrected(_lr(), tau).fit(df[features], df["case"])
    risk = m.predict_proba(pop[features])[:, 1]
    raw = m.estimator_.predict_proba(pop[features])[:, 1]
    assert abs(risk.mean() - tau) < 0.15 * tau
    assert abs(raw.mean() - tau) > 5 * tau  # the sample's probabilities are not risks
    assert np.corrcoef(risk, pop["risk"])[0, 1] > 0.95


def test_the_correction_is_learned_in_each_training_fold_and_never_from_held_out_rows():
    df = _sampled(_population())
    X, y = df[["x", "z"]], df["case"].to_numpy()
    out = CC.predicted_risks(_lr(), X, y, prevalence=0.035, folds=4, seed=2)
    for f, share, offset in zip(np.unique(out.fold), out.sample_share, out.offset):
        train = out.fold != f
        assert share == pytest.approx(y[train].mean(), rel=1e-15)
        assert offset == pytest.approx(CC.prior_correction_offset(y[train].mean(), 0.035))
    # changing the held-out rows' outcomes moves none of their own risks
    held = out.fold == 0
    flipped = y.copy()
    flipped[held] = 1 - flipped[held]
    again = CC.predicted_risks(_lr(), X, flipped, prevalence=0.035, folds=4, seed=2)
    if np.array_equal(again.fold, out.fold):
        assert again.risk[held] == pytest.approx(out.risk[held], rel=1e-12)
    else:  # the outcome stratifies the folds; refit with the first draw's folds by hand
        f0 = out.fold == 0
        m = CC.PriorCorrected(_lr(), 0.035).fit(X[~f0], flipped[~f0])
        assert m.predict_proba(X[f0])[:, 1] == pytest.approx(out.risk[f0], rel=1e-9)
    assert "recalibrated" in out.says and "King & Zeng 2001" in CC.risks_sentence(out)
    with pytest.raises(CC.CaseControlRefused) as no_prev:
        CC.predicted_risks(_lr(), X, y)
    assert no_prev.value.exits[0]["needs"] == "population_prevalence"
    with pytest.raises(CC.CaseControlRefused, match="correct the same thing twice"):
        CC.predicted_risks(_lr(), X, y, prevalence=0.03, design=object())
    # no prevalence is ever assumed: built without one, the corrected model refuses to fit
    assert CC.PriorCorrected(_lr()).prevalence is None
    with pytest.raises(CC.CaseControlRefused, match="not risks") as unstated:
        CC.PriorCorrected(_lr()).fit(X, y)
    assert unstated.value.exits[0]["needs"] == "population_prevalence"


def test_frequency_matching_takes_a_prevalence_per_stratum_with_the_stratum_among_the_features():
    pop = _population()
    df = _sampled(pop)
    df["band_code"] = df["band"].map({"40-49": 0, "50-59": 1, "60-69": 2})
    X = df[["x", "z", "band_code"]]
    prev = {0: 0.02, 1: 0.03, 2: 0.05}
    m = CC.PriorCorrected(_lr(), prev, stratum="band_code").fit(X, df["case"])
    for level, p in prev.items():
        share = df["case"][df["band_code"] == level].mean()
        assert m.offset_[level] == pytest.approx(
            math.log(share / (1 - share)) - math.log(p / (1 - p)), rel=1e-14)
    raw = m.estimator_.predict_proba(X)[:, 1]
    want = CC.recalibrate(raw, df["band_code"].map(m.offset_).to_numpy())
    assert m.predict_proba(X)[:, 1] == pytest.approx(want, rel=1e-14)
    with pytest.raises(CC.CaseControlRefused, match="one population prevalence cannot"):
        CC.predicted_risks(_lr(), X, df["case"], sampling="frequency_matched", prevalence=0.03,
                           stratum="band_code")
    with pytest.raises(CC.CaseControlRefused, match="among the model's features"):
        CC.PriorCorrected(_lr(), prev, stratum="band_code").fit(df[["x", "z"]], df["case"])
    out = CC.predicted_risks(_lr(), X, df["case"], sampling="frequency_matched", prevalence=prev,
                             stratum="band_code", folds=3)
    assert np.isfinite(out.risk).all() and "matching stratum" in out.says


def test_matched_sets_stay_whole_inside_folds_and_sets_sharing_a_person_are_joined():
    df = _multi()
    fold = CC.case_control_folds(df["case"], sets=df["set"], folds=5, seed=1)
    assert (pd.Series(fold).groupby(df["set"]).nunique() == 1).all()
    assert len(np.unique(fold)) == 5
    # incidence-density sampling: a control in set 1 is the same person as a control in set 2
    person = [f"p{i}" for i in range(len(df))]
    one, two = df.index[(df["set"] == 1) & (df["case"] == 0)][0], \
        df.index[(df["set"] == 2) & (df["case"] == 0)][0]
    person[two] = person[one]
    for seed in range(6):
        fold = CC.case_control_folds(df["case"], sets=df["set"], persons=person, seed=seed)
        assert len(set(fold[df["set"].isin([1, 2]).to_numpy()])) == 1, seed


@needs_r
def test_within_set_ranking_fits_the_conditional_model_in_each_training_fold(tmp_path):
    df = _infert(tmp_path)
    out = CC.matched_set_scores(df, "case", ["spontaneous", "induced"], "stratum", folds=4, seed=3)
    train = df[out.fold != 0]
    ref = run_r("""
library(survival)
df <- read.csv(train_csv)
out(list(coef = unname(coef(clogit(case ~ spontaneous + induced + strata(stratum), data = df)))))
""", {"train": train}, tmp_path)
    assert out.coefficients[0] == pytest.approx(ref["coef"], rel=1e-8)
    held = out.fold == 0
    assert out.score[held] == pytest.approx(
        (df.loc[held, ["spontaneous", "induced"]].to_numpy() @ np.asarray(ref["coef"])), rel=1e-8)
    # the concordance by hand: case–control pairs within sets, a tie half
    pairs = wins = 0.0
    for _, g in df.assign(s=out.score).groupby("stratum"):
        for c in g["s"][g["case"] == 1]:
            for k in g["s"][g["case"] == 0]:
                pairs += 1
                wins += 1.0 if c > k else 0.5 if c == k else 0.0
    assert out.n_pairs == pairs and out.concordance == pytest.approx(wins / pairs, rel=1e-12)
    assert "not risks" in out.says


# ── detection helpers ────────────────────────────────────────────────────────


@needs_r
def test_an_outcome_share_far_above_the_stated_rate_is_a_signal(tmp_path):
    y = np.r_[np.ones(60), np.zeros(140)]
    ref = run_r("out(list(p = binom.test(60, 200, 0.04, alternative = 'greater')$p.value))", {},
                tmp_path)
    s = CC.prevalence_signal(y, 0.04)
    assert s is not None and s.kind == "outcome_share_above_population"
    assert s.evidence["p"] == pytest.approx(ref["p"], rel=1e-8)
    assert s.evidence["sampling_odds_ratio"] == pytest.approx((0.3 / 0.7) / (0.04 / 0.96))
    assert "may have been sampled separately" in s.says
    assert CC.prevalence_signal(y, 0.25) is None  # 30% against 25%: not far above
    assert CC.prevalence_signal(np.r_[np.ones(2), np.zeros(8)], 0.04) is None  # chance allows it


def test_a_set_column_whose_sets_each_hold_one_case_is_a_signal(tmp_path):
    if RSCRIPT is None:
        pytest.skip("R (Rscript) is not installed")
    df = _infert(tmp_path)
    s = CC.matched_sets_signal(df, "case", "stratum")
    assert s is not None and s.column == "stratum"
    assert s.evidence["sets"] == 83 and s.evidence["one_case_sets"] == 83
    assert s.evidence["ratios"] == {"1:1": 1, "1:2": 82}
    assert set(s.evidence["matching_factors"]) >= {"age", "parity", "education"}
    assert "every one holds exactly one case" in s.says
    found = CC.set_column_signals(df, "case")
    assert [x.column for x in found][:1] == ["stratum"]
    assert "pooled.stratum" not in [x.column for x in found]  # its groups hold many cases
    rng = np.random.default_rng(0)
    clusters = pd.DataFrame({"case": rng.integers(0, 2, 300), "site": rng.integers(0, 30, 300)})
    assert CC.matched_sets_signal(clusters, "case", "site") is None


# ── the contracts ────────────────────────────────────────────────────────────


def test_the_contracts_are_registered_with_their_labels_and_enforcing_code():
    import importlib

    from turbotab.core import contracts as C
    from turbotab.core.export import citations
    from turbotab.core.reference import catalog

    reg = C.contracts()
    assert "turbotab.core.methods.case_control" in C.DECLARING_MODULES
    assert {k for k, c in reg.items() if c.package == "CASECONTROL"} == {
        "case_control_effects", "case_control_risks"}
    for key in ("case_control_effects", "case_control_risks"):
        c = reg[key]
        assert key in catalog.CONTRACT_LENSES
        assert c.slot == "model" and c.scope == "model" and c.needs and c.storyboard
        for o in c.options:
            assert o.label and o.customary
        for r in c.relations:
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert callable(getattr(importlib.import_module(module), name)), r.id
        for s in c.sources:
            assert citations.resolve(s), s
    risks = reg["case_control_risks"]
    assert all(o.rung["inference"] == "not_offered" for o in risks.options)
    assert risks.option("prior_correction").rung["prediction"] == "recommended"
    assert risks.relation("matched_risk_refused").rung == "refused"
    effects = reg["case_control_effects"]
    assert effects.relation("matched_design_refused").purposes == ("inference",)
    assert C.run_order(["case_control_risks", "case_control_effects"]) == [
        "case_control_effects", "case_control_risks"]
    assert "gail1981likelihood" in citations.keys_cited()
    # design-based logistic regression is Lumley 2010's chapter 6, "Categorical Data Regression"
    # (Crossref's chapter records under 10.1002/9780470580066; chapter 5 is "Ratios and Linear
    # Regression")
    assert CC.SOURCES["lumley"].endswith("ch. 6")


def test_the_refusals_speak_plainly_with_the_technical_term_as_a_label():
    e = CC.eligible("predict", "individually_matched", "predicted_risk", prevalence=0.1)
    with pytest.raises(CC.CaseControlRefused) as refused:
        e.require()
    plain = refused.value.reason
    assert "conditional" not in plain and "likelihood" not in plain and "intercept" not in plain
    assert str(refused.value).endswith(f"(Technical term: {e.term}.)")

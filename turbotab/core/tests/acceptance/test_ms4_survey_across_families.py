"""MS4 · A population estimand under a survey design binds every family and every display
(docs/turbotab-next/MODELING_SEQUENCE.md §0 ruling 6, §2 "Survey design with a population
estimand", §4 row "Population estimand without a design-based estimator", §5 MS4; the review
record ``audit/modeling-sequence-review.json``, the finding on §2's survey relation).

The package's five acceptance items, in its order:

1. Survey-weighted Cox (Binder's pseudo-likelihood, Taylor linearization with strata and PSUs):
   coefficients agree with R ``survey::svycoxph`` to 1e-6 and standard errors to 1e-4, relative,
   on an NHANES-shaped fixture with linked-mortality-style follow-up.
2. The weighted proportional-odds model agrees with R ``survey::svyolr`` (coefficients 1e-5,
   standard errors 1e-4, relative).
3. Substitution refits under a population estimand use the weights and design-based variance
   (agree with R ``svyglm`` + ``svycontrast`` to 1e-6 / 1e-4).
4. Every family and display under the population answer has a design-based estimator, or is
   blocked and recorded with the sample-only attestation as its exit; a chain test asserts that no
   table or curve under a population answer carries simple-random-sample intervals.
5. Strata with a single PSU are handled by a stated rule, R survey's
   ``options(survey.lonely.psu = "adjust")``, matching R.

**References.** R 4.6 with ``survey`` 4.5, run in a subprocess on a CSV the test writes
(``survey_r.py``); R is never imported by the app, and a machine without ``Rscript`` skips these
tests. Where R's own default would add numerical noise of its own, the test says so and runs R
tighter: ``coxph`` to ``eps = 1e-14``, and ``svyolr``'s Hessian (``optim``'s finite differences of
its gradient, ``ndeps = 1e-3`` by default) with ``ndeps = 1e-6``.

**Sources.** Binder 1983, *Int Stat Rev* 51:279 and 1992, *Int Stat Rev* 60:249 (pseudo-likelihood
for regression and Cox models under complex sampling); Lumley, *Complex Surveys* (2010) §5 and the
``survey`` package's documentation of ``svycoxph``, ``svyolr`` and ``survey.lonely.psu``
("'adjust': center the stratum at the population mean rather than the stratum mean"); Graubard &
Korn 1999, *Biometrics* 55:652 (predictive margins with survey data); Korn & Graubard 1990, *Am
Stat* 44:270 (the adjusted Wald F); NHANES Analytic Guidelines 2011–2016 §3.2.3 (domains, design
degrees of freedom). STROBE 12(d): "If applicable, describe analytical methods taking account of
sampling strategy."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.models.inference import INDEPENDENT, Outcome
from turbotab.core.models.ordinal import PROPORTIONAL_ODDS, SURVEY_ESTIMATOR as PO_ESTIMATOR
from turbotab.core.models.survey import (LONELY_PSU, LONELY_RULE, build_design, survey_table,
                                         total_variance)
from turbotab.core.models.survival import COX, SURVEY_ESTIMATOR as COX_ESTIMATOR, survival_outcome
from turbotab.core.tests.acceptance.survey_r import (RSCRIPT, needs_r, nhanes_diet,
                                                     nhanes_mortality, run_r)

R_DESIGN = ('f <- read.csv("f.csv")\n'
            "d <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~{w}, nest = TRUE, "
            "data = f)\n")


def design_of(frame: pd.DataFrame, weight: str = "WTMEC2YR"):
    """The engine's design over every row of ``frame`` (row ids 0…n−1)."""
    return build_design(frame, frame[weight].to_numpy(dtype=float), weight_column=weight,
                        strata_column="SDMVSTRA", psu_column="SDMVPSU")


def rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def column(table, key: str) -> np.ndarray:
    return np.array([r[key] for r in table.rows], dtype=float)


# ── 1 · survey-weighted Cox ──────────────────────────────────────────────────


COX_R = R_DESIGN.format(w="WTMEC2YR") + """
s <- subset(d, eligible == 1)
ctl <- coxph.control(eps = 1e-14, iter.max = 200, toler.chol = 1e-15)
a <- svycoxph(Surv(permth_exm, mortstat) ~ fiber + RIDAGEYR + female + kcal, design = s,
              control = ctl)
b <- svycoxph(Surv(entry_age, exit_age, mortstat) ~ fiber + female + kcal, design = s,
              control = ctl)
u <- coxph(Surv(permth_exm, mortstat) ~ fiber + RIDAGEYR + female + kcal, data = f,
           subset = eligible == 1, control = ctl)
out(list(follow_up = unname(coef(a)), follow_up_se = unname(sqrt(diag(vcov(a)))),
         age = unname(coef(b)), age_se = unname(sqrt(diag(vcov(b)))),
         unweighted = unname(coef(u)), degf = degf(s), n = nrow(s), events = sum(s$variables$mortstat)))
"""


@pytest.fixture(scope="module")
def mortality() -> pd.DataFrame:
    return nhanes_mortality()


@pytest.fixture(scope="module")
def cox_reference(mortality, tmp_path_factory):
    if RSCRIPT is None:
        pytest.skip("R (Rscript) is not installed")
    return run_r(COX_R, {"f": mortality}, tmp_path_factory.mktemp("cox_r"))


@needs_r
@pytest.mark.parametrize("scale", ["follow_up", "age"])
def test_1_survey_weighted_cox_agrees_with_svycoxph(mortality, cox_reference, scale):
    """Binder's pseudo-likelihood on the NHANES-shaped mortality fixture, two ways to follow up:
    ``permth_exm`` (whole months from the exam, so 604 deaths share far fewer distinct times and
    Efron's weighted tie handling is exercised), and age as the time scale (entry at the exam's age:
    left truncation, R's counting-process ``Surv(start, stop, event)``). The under-20s, with no
    mortality follow-up, stay in the design as a domain (R: ``subset(design, eligible == 1)``).

    Reference: R ``survey::svycoxph`` (``coxph`` weighted by the survey weight, its variance
    ``svyrecvar`` of the weighted dfbeta residuals). Bounds: coefficients 1e-6 and standard errors
    1e-4, relative; measured about 1e-11. The interval is ``β ± t(d)·SE`` on the design degrees of
    freedom (R's ``degf``), and the weighted estimate is not the unweighted one (R ``coxph``)."""
    f = mortality
    design = design_of(f)
    rows = f[f["eligible"] == 1]
    if scale == "follow_up":
        columns = ["fiber", "RIDAGEYR", "female", "kcal"]
        y = survival_outcome(rows["mortstat"], rows["permth_exm"])
    else:
        columns = ["fiber", "female", "kcal"]
        y = survival_outcome(rows["mortstat"], rows["exit_age"], rows["entry_age"])
    outcome = Outcome(name="mortstat", labels={1: "1", 0: "0"})
    table = COX.inference_matrix(rows[columns], y, task="time_to_event", classes=None,
                                 clusters=INDEPENDENT, outcome=outcome, rows="all", survey=design)
    r = cox_reference
    assert table.info["estimator"] == COX_ESTIMATOR
    assert table.info["covariance"] == "design" and table.info["scale"] == "hazard_ratio"
    assert rel(column(table, "estimate"), r[scale]) < 1e-6
    assert rel(column(table, "se"), r[f"{scale}_se"]) < 1e-4
    df = table.info["survey"]["df"]
    assert df == r["degf"] == 16
    assert table.info["survey"]["n_domain"] == r["n"] and int(y["event"].sum()) == r["events"]
    q = stats.t.ppf(0.975, df)
    for row in table.rows:
        assert row["df"] == df
        assert row["ci_low"] == pytest.approx(row["estimate"] - q * row["se"], rel=1e-12)
        assert row["ratio"] == pytest.approx(np.exp(row["estimate"]), rel=1e-12)
        assert row["p"] == pytest.approx(2 * stats.t.sf(abs(row["estimate"] / row["se"]), df),
                                         rel=1e-9)
    if scale == "follow_up":
        assert rel(column(table, "estimate"), r["unweighted"]) > 0.05  # the weights matter
    # A common factor on every weight changes nothing (Binder's equations are homogeneous).
    scaled = f.assign(WTMEC2YR=f["WTMEC2YR"] * 1000.0)
    again = COX.inference_matrix(rows[columns], y, task="time_to_event", classes=None,
                                 clusters=INDEPENDENT, outcome=outcome, rows="all",
                                 survey=design_of(scaled))
    assert rel(column(again, "se"), column(table, "se")) < 1e-9


# ── 2 · the weighted proportional-odds model ─────────────────────────────────


OLR_R = R_DESIGN.format(w="WTMEC2YR") + """
s <- subset(d, eligible == 1)
o <- svyolr(factor(health) ~ fiber + RIDAGEYR + female, design = s,
            control = list(reltol = 1e-15, maxit = 10000, ndeps = rep(1e-6, 6)))
o3 <- svyolr(factor(health) ~ fiber + RIDAGEYR + female, design = s)
out(list(est = unname(coef(o)), se = unname(sqrt(diag(vcov(o)))),
         default_se = unname(sqrt(diag(vcov(o3)))), degf = degf(s)))
"""


@needs_r
def test_2_weighted_proportional_odds_agrees_with_svyolr(mortality, tmp_path):
    """The weighted cumulative-logit model of self-rated health (four ordered levels) on fiber, age
    and sex, the under-20s a domain. Reference: R ``survey::svyolr`` (coefficients, then the
    cut-points ζ, with ``logit P(Y ≤ j) = ζ_j − xβ`` as here), its variance ``svyrecvar`` of the
    weighted scores times the inverse Hessian. Bounds: estimates 1e-5 and standard errors 1e-4,
    relative; measured about 5e-7 and 3e-8.

    R's Hessian is ``optim``'s finite difference of its gradient. At its default step (``ndeps =
    1e-3``) R's own standard errors sit about 4e-4 from the analytic Hessian's; with ``ndeps =
    1e-6`` the two agree to 3e-8, so the reference runs R at that step, and the default is held
    to 1e-3 to show the gap is R's finite difference, not the estimator."""
    f = mortality
    design = design_of(f)
    rows = f[f["eligible"] == 1]
    levels = ["1", "2", "3", "4"]
    codes = rows["health"].to_numpy() - 1
    table = PROPORTIONAL_ODDS.inference_matrix(
        rows[["fiber", "RIDAGEYR", "female"]], codes, task="ordinal", classes=levels,
        clusters=INDEPENDENT, outcome=Outcome(name="health"), rows="all", survey=design)
    r = run_r(OLR_R, {"f": f}, tmp_path)
    assert table.info["estimator"] == PO_ESTIMATOR and table.info["covariance"] == "design"
    assert [row["feature"] for row in table.rows][-3:] == [
        "(cut-point 1 | 2)", "(cut-point 2 | 3)", "(cut-point 3 | 4)"]
    assert rel(column(table, "estimate"), r["est"]) < 1e-5
    assert rel(column(table, "se"), r["se"]) < 1e-4
    assert rel(column(table, "se"), r["default_se"]) < 1e-3
    assert table.info["survey"]["df"] == r["degf"]
    assert table.info["scale"] == "odds_ratio"
    brant = [c for c in table.concerns if "Brant" in c]
    assert all("not a design-based test" in c or "could not be computed" in c for c in brant)


# ── 5 · lonely PSUs ──────────────────────────────────────────────────────────


LONELY_R = R_DESIGN.format(w="WTMEC2YR") + """
res <- list()
for (name in c("full", "drop_stratum", "drop_lonely", "domain_lonely", "partial")) {
  dom <- read.csv(paste0("dom_", name, ".csv"))[[1]] == 1
  s <- subset(d, dom)
  x <- as.matrix(f[dom, c("u1", "u2")])
  res[[name]] <- unclass(svyrecvar(x, s$cluster, s$strata, s$fpc))
}
s <- subset(d, eligible == 1 & SDMVSTRA != 103)
ctl <- coxph.control(eps = 1e-14, iter.max = 200, toler.chol = 1e-15)
g <- svyglm(fiber ~ RIDAGEYR + female + kcal, design = s)
l <- svyglm(mortstat ~ fiber + RIDAGEYR + female, design = s, family = quasibinomial(),
            control = glm.control(epsilon = 1e-14, maxit = 100))
x <- svycoxph(Surv(permth_exm, mortstat) ~ fiber + RIDAGEYR + female, design = s, control = ctl)
o <- svyolr(factor(health) ~ fiber + RIDAGEYR + female, design = s,
            control = list(reltol = 1e-15, maxit = 10000, ndeps = rep(1e-6, 6)))
res$glm <- list(est = unname(coef(g)), se = unname(sqrt(diag(vcov(g)))))
res$logit <- list(est = unname(coef(l)), se = unname(sqrt(diag(vcov(l)))))
res$cox <- list(est = unname(coef(x)), se = unname(sqrt(diag(vcov(x)))))
res$olr <- list(est = unname(coef(o)), se = unname(sqrt(diag(vcov(o)))))
res$degf <- degf(s)
out(res)
"""


@needs_r
def test_5_lonely_psus_are_centered_as_r_survey_adjust_does(tmp_path):
    """Two strata of the mortality fixture have a single PSU (``114``, ``115``). The stated rule
    is R's ``options(survey.lonely.psu = "adjust")`` as ``survey`` 4.5 computes it: a lonely
    stratum's PSU total is centered at the total of the scores over the number of PSUs in the
    strata that hold analysis rows, with ``n_h/(n_h − 1)`` taken as 1, and a stratum holding no
    analysis row adds nothing.

    References: R ``svyrecvar`` on the same scores for five domains (the whole design; one stratum
    out; a lonely stratum out; one PSU of a two-PSU stratum out, which R by default does not treat
    as lonely; a random 70% of rows), to 1e-12; and R ``svyglm`` (least squares and logistic),
    ``svycoxph`` and ``svyolr`` on a domain that leaves out stratum ``103``, so the centering's
    denominator changes, to 1e-6 (estimates) and 1e-4 (standard errors). Each table names its
    lonely strata and the rule."""
    f = nhanes_mortality(lonely=2)
    rng = np.random.default_rng(5)
    f["u1"], f["u2"] = rng.normal(size=len(f)), rng.normal(size=len(f))
    domains = {
        "full": np.ones(len(f), dtype=bool),
        "drop_stratum": f["SDMVSTRA"].ne(103).to_numpy(),
        "drop_lonely": f["SDMVSTRA"].ne(115).to_numpy(),
        "domain_lonely": ~((f["SDMVSTRA"] == 104) & (f["SDMVPSU"] == 2)).to_numpy(),
        "partial": rng.random(len(f)) < 0.7,
    }
    for name, dom in domains.items():
        pd.Series(dom.astype(int)).to_csv(tmp_path / f"dom_{name}.csv", index=False)
    r = run_r(LONELY_R, {"f": f}, tmp_path)
    design = design_of(f)
    assert sorted(design.lonely_strata()) == [114, 115] and LONELY_PSU == "adjust"
    for name, dom in domains.items():
        u = f[["u1", "u2"]].to_numpy() * dom[:, None]
        assert rel(total_variance(u, design, dom).meat, r[name]) < 1e-12, name

    rows = f[(f["eligible"] == 1) & (f["SDMVSTRA"] != 103)]
    X = rows[["fiber", "RIDAGEYR", "female"]]
    glm = survey_table("regression", rows[["RIDAGEYR", "female", "kcal"]], rows["fiber"], None,
                       design)
    logit = survey_table("binary", X, rows["mortstat"].to_numpy(), [0, 1], design)
    cox = COX.inference_matrix(X, survival_outcome(rows["mortstat"], rows["permth_exm"]),
                               task="time_to_event", classes=None, clusters=INDEPENDENT,
                               survey=design)
    olr = PROPORTIONAL_ODDS.inference_matrix(X, rows["health"].to_numpy() - 1, task="ordinal",
                                             classes=["1", "2", "3", "4"], clusters=INDEPENDENT,
                                             survey=design)
    for name, table in (("glm", glm), ("logit", logit), ("cox", cox), ("olr", olr)):
        assert rel(column(table, "estimate"), r[name]["est"]) < 1e-6, name
        assert rel(column(table, "se"), r[name]["se"]) < 1e-4, name
        assert table.info["survey"]["df"] == r["degf"], name
        assert table.info["survey"]["lonely_strata"] == ["114", "115"], name
        said = next(c for c in table.concerns if "single PSU" in c)
        assert LONELY_RULE in said and "`114` and `115`" in said, said

    # The methods sentence states the rule, word for word.
    from turbotab.core import voice
    from turbotab.core.decisions import ProjectState, SetSurvey

    class Store:
        def materialize(self, columns, rows):
            return f[list(columns)]

    sentence = voice.sentence_for(
        SetSurvey(estimand="population", weight="WTMEC2YR", strata="SDMVSTRA", psu="SDMVPSU"),
        ProjectState(purpose="inference"), {"datastore": Store()})
    per = f.groupby("SDMVSTRA")["SDMVPSU"].nunique()
    assert (int(per.sum()), len(per), int((per == 1).sum())) == (29, 15, 2)
    assert sentence == (
        "The estimates describe the surveyed population: rows were weighted by `WTMEC2YR`, and "
        "standard errors were estimated by Taylor series linearization over `SDMVPSU` nested "
        "within `SDMVSTRA` (`29` PSUs in `15` strata); `2` strata with a single PSU were centered "
        "at the mean PSU total of the strata holding analysis rows (R survey's lonely.psu "
        "\"adjust\"), first-stage units taken as sampled with replacement, with t intervals on "
        "the PSUs minus the strata that hold the analysis rows.")


# ── 3 · the substitution curve over the surveyed population ──────────────────


DIET_ROLES = {"SEQN": "identifier", "protein_g": "exposure", "fat_g": "exposure",
              "carb_g": "exposure", "energy_kcal": "energy", "age": "covariate",
              "female": "covariate", "WTDRD1": "design", "SDMVSTRA": "design",
              "SDMVPSU": "design"}
POPULATION = {"estimand": "population", "weight": "WTDRD1", "strata": "SDMVSTRA",
              "psu": "SDMVPSU"}


def survey_stages(frame: pd.DataFrame, folder: Path, *, target: str, task: str,
                  roles: dict[str, str], models: list[str], survey: dict | None = POPULATION,
                  substitution: dict | None = None, analyzed: np.ndarray | None = None,
                  **slots) -> dict:
    """Ingest ``frame`` (every row in the working table, so in the survey design) and run the
    design, fit and (when asked) substitution stages under inference, the analyzed rows
    ``analyzed`` (default: every row) all training rows. Returns the artifacts and the state."""
    from turbotab.core.decisions import SplitSpec, SubstitutionSpec, SurveySpec
    from turbotab.core.models.artifacts import FitArtifact, SubstitutionArtifact
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
    from turbotab.core.tests import modeling_fixtures as mf

    paths = mf.ingest_frame(frame, folder)
    rows = np.arange(len(frame)) if analyzed is None else np.asarray(analyzed)
    confirmations = {"code_or_count:female": "code", "code_or_count:SDMVSTRA": "code",
                     "code_or_count:SDMVPSU": "code", **slots.pop("shape_confirmations", {})}
    st = mf.state(roles=roles, target=target, task=task, models=models, purpose="inference",
                  split=SplitSpec(holdout=0.0, seed=0, folds=5),
                  survey=SurveySpec(**survey) if survey else None,
                  substitution=SubstitutionSpec(**substitution) if substitution else None,
                  column_units=mf.grams(*[c for c in frame.columns if c.endswith("_g")]),
                  shape_confirmations=confirmations, **slots)
    split = mf.split_bundle(rows, holdout=0.0)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    FitArtifact.model_validate(fit.data)
    out = {"design": design, "fit": fit, "state": st, "paths": paths}
    if substitution:
        sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
        SubstitutionArtifact.model_validate(sub)
        out["substitution"] = sub
    return out


@pytest.fixture(scope="module")
def diet() -> pd.DataFrame:
    return nhanes_diet()


SWAP_R = R_DESIGN.format(w="WTDRD1") + """
s <- subset(d, eligible == 1)
g <- svyglm(crp ~ protein_g + fat_g + carb_g + energy_kcal + age + female, design = s)
ks <- scan("ks.txt", quiet = TRUE)
est <- c(); se <- c()
for (k in ks) {
  v <- svycontrast(g, c(fat_g = -k / 9, carb_g = k / 4))
  est <- c(est, unname(coef(v))); se <- c(se, unname(SE(v)))
}
out(list(est = est, se = se, coef = as.list(coef(g)), degf = degf(s)))
"""


@needs_r
def test_3_the_population_substitution_agrees_with_svyglm_and_svycontrast(diet, tmp_path):
    """Moving k kcal from fat to carbohydrate under the population answer, through the real design,
    fit and substitution stages: the linear model is refit with the dietary weight ``WTDRD1``, the
    curve averages each participant's change weighted by the people they stand for, and its band
    is Taylor linearization over the design on t(d). The under-20s are outside the analysis and
    stay in the design.

    Reference: R ``svyglm`` on the same rows and ``svycontrast`` of ``(−k/9)·β_fat + (k/4)·β_carb``
    at each k: with every term linear the change is the same on every row, so the population curve
    is that contrast exactly. Bounds: the curve 1e-6 and its standard error (the band's half-width
    over t(d)) 1e-4, relative. The coefficient table is R's ``svyglm`` too. The 200 bootstrap
    refits asked for are not drawn: a row bootstrap ignores the strata and PSUs."""
    f = diet
    analyzed = np.flatnonzero(f["eligible"].to_numpy() == 1)
    out = survey_stages(f.drop(columns=["eligible", "over", "high_crp"]), tmp_path / "stages",
                        target="crp", task="regression", roles=DIET_ROLES, models=["linear"],
                        substitution={"donor": "fat_g", "recipient": "carb_g", "step_kcal": 100.0,
                                      "n_boot": 200}, analyzed=analyzed)
    sub = out["substitution"]
    ks = np.asarray(sub["ks"], dtype=float)
    (tmp_path / "r").mkdir()
    np.savetxt(tmp_path / "r" / "ks.txt", ks)
    r = run_r(SWAP_R, {"f": f}, tmp_path / "r")
    curve = sub["models"][0]
    band = sub["band"]
    assert band["method"] == "design" and band["n_boot"] == 0 and band["df"] == r["degf"]
    q = stats.t.ppf(0.975, band["df"])
    live = [i for i, v in enumerate(curve["delta"]) if v is not None and ks[i] > 0]
    assert len(live) >= 3
    for i in live:
        assert curve["delta"][i] == pytest.approx(r["est"][i], rel=1e-6)
        se = (curve["ci_high"][i] - curve["delta"][i]) / q
        assert se == pytest.approx(r["se"][i], rel=1e-4)
        assert curve["ci_low"][i] == pytest.approx(curve["delta"][i] - q * se, rel=1e-9)
    assert curve["delta"][0] == 0.0 and curve["ci_low"][0] == curve["ci_high"][0] == 0.0
    table = out["fit"].data["models"][0]
    assert table["inference"]["covariance"] == "design"
    coefficients = {row["feature"]: row["estimate"] for row in table["coefficients"]}
    # ``female`` is a code (0/1), one indicator against 0; R's numeric 0/1 term is the same column.
    named = {"(Intercept)": "(intercept)", "female": "female_1"}
    for name, value in r["coef"].items():
        assert coefficients[named.get(name, name)] == pytest.approx(value, rel=1e-6)
    assert "200 bootstrap refits asked for are not drawn" in sub["note"]
    assert "Taylor linearization over the survey design" in band["caption"]
    assert sub["basis"] == (f"Averaged over the {len(analyzed):,} analyzed rows in the survey "
                            f"design, each counted as the people its weight `WTDRD1` stands for.")
    assert "over the surveyed population" in sub["estimand"]
    from turbotab.core import voice
    from turbotab.core.decisions import SetSubstitution

    said = voice.sentence_for(SetSubstitution(donor="fat_g", recipient="carb_g", step_kcal=100.0,
                                              n_boot=200), out["state"])
    assert said == (
        "The substitution studied is `fat_g` replaced by `carb_g`, in steps of `100` kcal at the "
        "same total energy; over the surveyed population its curve is the weighted mean of each "
        "participant's change in the survey-weighted fit, and its band comes from Taylor "
        "linearization over the survey design.")


LOGIT_R = R_DESIGN.format(w="WTDRD1") + """
s <- subset(d, eligible == 1)
g <- svyglm(high_crp ~ protein_g + fat_g + carb_g + energy_kcal + age + female, design = s,
            family = quasibinomial(), influence = TRUE,
            control = glm.control(epsilon = 1e-14, maxit = 100))
inf <- attr(g, "influence")
rows <- s$variables
w <- weights(s)
ks <- scan("ks.txt", quiet = TRUE)
est <- c(); se <- c()
for (i in seq_along(ks)) {
  m <- rows[[paste0("m", i)]]
  if (sum(m) == 0) { est <- c(est, NA); se <- c(se, NA); next }
  moved <- rows
  moved$fat_g <- rows[[paste0("fat", i)]]
  moved$carb_g <- rows[[paste0("carb", i)]]
  eta0 <- predict(g, newdata = rows, type = "link")
  eta1 <- predict(g, newdata = moved, type = "link")
  X0 <- model.matrix(delete.response(terms(g)), rows)
  X1 <- model.matrix(delete.response(terms(g)), moved)
  d0 <- g$family$mu.eta(eta0); d1 <- g$family$mu.eta(eta1)
  delta <- g$family$linkinv(eta1) - g$family$linkinv(eta0)
  wm <- w * m
  theta <- sum(wm * delta) / sum(wm)
  grad <- colSums(wm * (d1 * X1 - d0 * X0)) / sum(wm)
  z <- wm * (delta - theta) / sum(wm) + inf %*% grad
  est <- c(est, theta)
  se <- c(se, sqrt(svyrecvar(z, s$cluster, s$strata, s$fpc)[1, 1]))
}
out(list(est = est, se = se, degf = degf(s)))
"""


@needs_r
def test_3_a_yes_no_outcome_s_population_curve_matches_r_s_linearization(diet, tmp_path):
    """The same swap on a yes/no outcome (CRP above 3 mg/L): the curve is the population's average
    change in the predicted probability, which now differs from row to row, so the band carries
    both the error of the weighted fit and which people were sampled (Graubard & Korn 1999):
    ``z_i = w_i m_i (Δ_i − θ)/Σ w m + ψ_iᵀ g``.

    Reference: the same linearization computed in R from R's own pieces: ``svyglm``'s fit and its
    influence functions (``influence = TRUE``, which ``svyrecvar`` turns into its ``vcov``), R's
    ``predict`` and ``mu.eta`` for Δ and its gradient, and ``svyrecvar`` for the variance. The rows
    on support at each k and their shifted fat and carbohydrate are the app's definitions
    (``methods.substitution.Shift``), handed to R as columns. Bounds: 1e-6 and 1e-4, relative."""
    from turbotab.core.methods.substitution import Shift

    f = diet
    analyzed = np.flatnonzero(f["eligible"].to_numpy() == 1)
    out = survey_stages(f.drop(columns=["eligible", "over", "crp"]), tmp_path / "stages",
                        target="high_crp", task="binary", roles=DIET_ROLES, models=["linear"],
                        substitution={"donor": "fat_g", "recipient": "carb_g",
                                      "step_kcal": 100.0}, analyzed=analyzed, event="1")
    sub = out["substitution"]
    curve = sub["models"][0]
    ks = np.asarray(sub["ks"], dtype=float)
    X = f.iloc[analyzed][["protein_g", "fat_g", "carb_g", "energy_kcal", "age", "female"]]
    shift = Shift(X, donor="fat_g", recipient="carb_g", kcal_per_unit={"fat_g": 9.0, "carb_g": 4.0},
                  total="energy_kcal")
    assert shift.valid(X).all()
    handed = f.copy()
    for i, k in enumerate(ks, start=1):
        moved, amount, composition = shift.checks(X, float(k))
        on = amount & composition & (curve["delta"][i - 1] is not None)
        handed[f"m{i}"] = 0
        handed[f"fat{i}"], handed[f"carb{i}"] = handed["fat_g"], handed["carb_g"]
        handed.loc[X.index[on], f"m{i}"] = 1
        handed.loc[X.index, f"fat{i}"] = moved["fat_g"].to_numpy()
        handed.loc[X.index, f"carb{i}"] = moved["carb_g"].to_numpy()
    (tmp_path / "r").mkdir()
    np.savetxt(tmp_path / "r" / "ks.txt", ks)
    r = run_r(LOGIT_R, {"f": handed}, tmp_path / "r")
    df = sub["band"]["df"]
    assert df == r["degf"]
    q = stats.t.ppf(0.975, df)
    live = [i for i, v in enumerate(curve["delta"]) if v is not None and ks[i] > 0]
    assert len(live) >= 3
    for i in live:
        assert curve["delta"][i] == pytest.approx(r["est"][i], rel=1e-6)
        assert (curve["ci_high"][i] - curve["delta"][i]) / q == pytest.approx(r["se"][i], rel=1e-4)


# ── 4 · the chain: no table or curve under the population answer is SRS ──────


SAMPLE_ONLY = {"kind": "set_survey", "estimand": "sample"}


def _offers_the_attestation(exits: list[dict]) -> bool:
    return any(e.get("decision") == SAMPLE_ONLY for e in exits)


def assert_design_based_or_blocked(fit: dict, substitution: dict | None = None) -> dict[str, str]:
    """The chain test's invariant (MODELING_SEQUENCE §4; the review's "no table or curve under a
    population answer carries SRS intervals"), walked over every interval-bearing display the fit
    and the substitution make: each one with an interval is design-based (Taylor linearization,
    on t with the design's degrees of freedom), and each one that is not is blocked, with the
    sample-only attestation among its exits. Returns each family's verdict."""
    verdict: dict[str, str] = {}
    for model in fit["models"]:
        info = model["inference"] or {}
        rows = model["coefficients"] or []
        with_intervals = [r for r in rows if r.get("ci_low") is not None]
        if with_intervals:
            assert info["covariance"] == "design", (model["family"], info.get("caption"))
            df = info["survey"]["df"]
            assert df is not None, model["family"]
            if (info.get("missing") or {}).get("method") == "multiple_imputation":
                # Pooled by Rubin's rules, each copy's t(d) the complete-data df: Barnard–Rubin
                # degrees of freedom never exceed the design's (MS2: ν_com = the design df).
                assert all(0 < r["df"] <= df + 1e-9 for r in with_intervals), model["family"]
            else:
                assert all(r["df"] == df for r in with_intervals), model["family"]
            assert "Taylor linearization over the survey design" in info["caption"]
            verdict[model["family"]] = "design"
        elif info.get("refused"):
            assert info["covariance"] == "none" and not with_intervals
            assert _offers_the_attestation(info["exits"]), (model["family"], info["exits"])
            assert "has no design-based estimator" in info["refused"]
            verdict[model["family"]] = "blocked"
        else:  # no coefficient table at all (a family that only predicts): nothing to estimate
            assert not rows, model["family"]
            verdict[model["family"]] = "no table"
        for test in model.get("exposure_tests") or []:
            if test.get("p") is not None:
                assert "survey design" in test["caption"], test["caption"]
        if model["cv"]:
            assert any("Cross-validated scores are unweighted" in c for c in model["concerns"])
    for curve in (substitution or {}).get("models", []):
        if any(v is not None for v in (curve.get("ci_low") or [])):
            assert substitution["band"]["method"] == "design" and substitution["band"]["df"]
        elif curve.get("refused"):
            assert all(v is None for v in curve["delta"])
            assert _offers_the_attestation(curve["exits"]), curve["exits"]
        else:
            assert all(v is None for v in curve["delta"][1:]), curve["family"]
    return verdict


def test_4_no_table_or_curve_under_the_population_answer_carries_srs_intervals(tmp_path):
    """The chain test (MODELING_SEQUENCE §2, "Survey design with a population estimand implies a
    design-based estimator for every family and display, or block-and-record with the sample-only
    exit"; §4's row; §6 chain 1's survey part). Every family on the shelf and every display is run
    under the population answer through the real design, fit and substitution stages, on the
    NHANES-shaped tables:

    * a continuous outcome: the linear family (design-based table, spline tests, substitution
      curve with its design band), and elastic net, gradient-boosted trees, feature-wise
      regression, GEE and the random-intercept mixed model, each blocked: no coefficient and no
      curve, the reason and the exits (the design-based family; the sample-only attestation);
    * a yes/no outcome with quintiles of an exposure: the logistic table and its trend test;
    * an ordered outcome: the proportional-odds model and the multinomial one, both design-based;
    * a time to event (linked mortality) with a spline of fiber: survey-weighted Cox and its
      adjusted Wald tests;
    * multiple imputation (MS2's frame, the MI package's): each completed copy is analyzed
      design-based, so the pooled table is too;
    * regression calibration: blocked, as it has no design-based variance here.

    The invariant is checked structurally (:func:`assert_design_based_or_blocked`), and each §2
    relation the population contract declares (``models.survey.CONTRACTS``) is seen to fire."""
    from turbotab.core.decisions import ExposureFormSpec, FollowUpSpec, MissingSpec
    from turbotab.core.models.survey import CONTRACTS

    diet = nhanes_diet(seed=11, n_per_psu=40)
    analyzed = np.flatnonzero(diet["eligible"].to_numpy() == 1)
    shelf = ["linear", "elastic_net", "boosted_trees", "featurewise", "gee", "mixed"]
    run = survey_stages(diet.drop(columns=["eligible", "over", "high_crp"]), tmp_path / "a",
                        target="crp", task="regression", roles=DIET_ROLES, models=shelf,
                        substitution={"donor": "fat_g", "recipient": "carb_g", "step_kcal": 100.0,
                                      "n_boot": 50},
                        analyzed=analyzed,
                        exposure_forms={"fat_g": ExposureFormSpec(form="spline", knots=4)})
    verdict = assert_design_based_or_blocked(run["fit"].data, run["substitution"])
    assert verdict == {"linear": "design", "elastic_net": "blocked", "boosted_trees": "no table",
                       "featurewise": "blocked", "gee": "blocked", "mixed": "blocked"}
    linear = next(m for m in run["fit"].data["models"] if m["family"] == "linear")
    spline = {t["test"]: t for t in linear["exposure_tests"]}
    df = linear["inference"]["survey"]["df"]
    assert spline["overall"]["df_num"] == 3 and spline["overall"]["df_den"] == df - 3 + 1
    assert spline["nonlinear"]["df_num"] == 2 and spline["nonlinear"]["df_den"] == df - 2 + 1
    exits = next(m for m in run["fit"].data["models"] if m["family"] == "mixed")["inference"]["exits"]
    assert exits[0]["decision"] == {"kind": "select_models", "models": [
        "linear", "elastic_net", "boosted_trees", "featurewise", "gee"]}
    curves = {c["family"]: c for c in run["substitution"]["models"]}
    assert curves["linear"]["refused"] is None and curves["linear"]["ci_low"][1] is not None
    for blocked in ("elastic_net", "boosted_trees", "gee", "mixed"):
        assert "has no design-based estimator" in curves[blocked]["refused"]
    # The population answer invalidates the row-resampling band (§2): the design's replaces it.
    assert run["substitution"]["band"]["method"] == "design"
    assert "The 50 bootstrap refits asked for are not drawn" in run["substitution"]["note"]
    # The record: the select_models sentence names each estimator and each block, word for word.
    from turbotab.core import voice
    from turbotab.core.decisions import SelectModels

    assert voice.sentence_for(SelectModels(models=shelf), run["state"]) == (
        "`6` model families were chosen: linear regression, elastic net, gradient-boosted trees, "
        "feature-wise least-squares tests with Benjamini–Hochberg false-discovery control, "
        "generalized estimating equations and a random-intercept mixed model. For the surveyed "
        "population, least squares was weighted by `WTDRD1`, with standard errors by Taylor "
        "linearization over the survey design; the elastic net, the gradient-boosted tree model, "
        "feature-wise regression, the GEE model and the random-intercept mixed model have no "
        "design-based estimator, so their estimates were blocked and not reported.")

    binary = survey_stages(diet.drop(columns=["eligible", "over", "crp"]), tmp_path / "b",
                           target="high_crp", task="binary", roles=DIET_ROLES,
                           models=["linear", "gee"], analyzed=analyzed, event="1",
                           exposure_forms={"protein_g": ExposureFormSpec(form="quintiles")})
    assert assert_design_based_or_blocked(binary["fit"].data) == {"linear": "design",
                                                                 "gee": "blocked"}
    logistic = next(m for m in binary["fit"].data["models"] if m["family"] == "linear")
    quintile = {t["test"]: t for t in logistic["exposure_tests"]}
    trend = quintile["trend"]
    assert "the cut points and medians are the analyzed rows' own, unweighted" in \
        trend["caption"].lower()
    # MS2's global test that every quintile indicator is zero (the MI package's) is the design's
    # too: the adjusted Wald F on the design's degrees of freedom, as the spline's tests are.
    d_bin = logistic["inference"]["survey"]["df"]
    assert quintile["global"]["df_num"] == 4 and quintile["global"]["df_den"] == d_bin - 4 + 1
    assert "taylor-linearized" in quintile["global"]["caption"].lower()

    mortality = nhanes_mortality(seed=5, n_per_psu=35)
    roles = {"SEQN": "identifier", "fiber": "exposure", "RIDAGEYR": "covariate",
             "female": "covariate", "kcal": "covariate", "WTMEC2YR": "design",
             "SDMVSTRA": "design", "SDMVPSU": "design"}
    population = {"estimand": "population", "weight": "WTMEC2YR", "strata": "SDMVSTRA",
                  "psu": "SDMVPSU"}
    eligible = np.flatnonzero(mortality["eligible"].to_numpy() == 1)
    keep = ["SEQN", "fiber", "RIDAGEYR", "female", "kcal", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
    ordinal = survey_stages(mortality[keep + ["health"]], tmp_path / "c", target="health",
                            task="ordinal", roles=roles, models=["proportional_odds", "linear"],
                            survey=population, analyzed=eligible)
    assert assert_design_based_or_blocked(ordinal["fit"].data) == {"proportional_odds": "design",
                                                                  "linear": "design"}
    cox = survey_stages(mortality[keep + ["mortstat", "permth_exm"]], tmp_path / "d",
                        target="mortstat", task="time_to_event", roles={**roles,
                                                                        "permth_exm": "time"},
                        models=["cox"], survey=population, analyzed=eligible, event="1",
                        follow_up=FollowUpSpec(time_column="permth_exm"),
                        exposure_forms={"fiber": ExposureFormSpec(form="spline", knots=4)})
    assert assert_design_based_or_blocked(cox["fit"].data) == {"cox": "design"}
    model = cox["fit"].data["models"][0]
    assert model["inference"]["estimator"] == COX_ESTIMATOR
    assert {t["test"] for t in model["exposure_tests"]} == {"overall", "nonlinear"}

    gaps = diet.drop(columns=["eligible", "over", "high_crp"]).copy()
    gaps.loc[gaps.sample(frac=0.08, random_state=1).index, "protein_g"] = np.nan
    imputed = survey_stages(gaps, tmp_path / "e", target="crp", task="regression",
                            roles=DIET_ROLES, models=["linear"], analyzed=analyzed,
                            missing=MissingSpec(strategy="multiple_imputation", m=20))
    pooled = imputed["fit"].data["models"][0]
    assert assert_design_based_or_blocked(imputed["fit"].data) == {"linear": "design"}
    assert pooled["inference"]["estimator"] == "survey-weighted least squares"
    assert pooled["inference"]["missing"]["method"] == "multiple_imputation"

    from turbotab.core.decisions import MeasurementErrorSpec
    from turbotab.core.stages.calibration import POPULATION as CALIBRATION_BLOCK, calibration_stage
    from turbotab.core.tests import modeling_fixtures as mf

    st = run["state"].model_copy(update={"measurement_error": MeasurementErrorSpec(
        method="regression_calibration")})
    calibrated = calibration_stage(mf.context(st, {"working": {}, "structure": None},
                                              run["paths"])).data
    assert calibrated["applies"] is False and calibrated["reason"] == CALIBRATION_BLOCK
    assert "these participants" in CALIBRATION_BLOCK

    # The shelf says it before the families are chosen: those with a design-based estimator
    # first, the rest after them with the reason; the shelf is never shortened.
    from turbotab.core.stages.modeling import shelf_stage
    from turbotab.core.tests import modeling_fixtures as mf

    predictors = ["protein_g", "fat_g", "carb_g", "energy_kcal", "age", "female"]
    ranked = shelf_stage(mf.context(run["state"], {
        "cohort": mf.cohort_bundle(analyzed, predictors),
        "target_info": mf.target_info("regression", "crp")}, run["paths"]))
    keys = [s["key"] for s in ranked["families"]]
    assert keys[0] == "linear" and set(shelf) <= set(keys)
    for family in ranked["families"][1:]:
        assert family["concerns"][0] == (
            "Under the surveyed population it has no design-based estimator, so its estimates "
            "are blocked and recorded; survey-weighted least squares has one.")

    # A family with no design-based estimator, reached by any other display (the sensitivity
    # analyses' refit, for one), is blocked the same way (``stages.modeling._inference_table``).
    from turbotab.core.models import get_family
    from turbotab.core.stages.modeling import _inference_table

    fitted = run["fit"].objects["fitted"]["mixed"]
    X_all = diet.drop(columns=["eligible", "over", "high_crp"]).iloc[analyzed]
    design = run["fit"].objects["survey_design"]
    table = _inference_table(get_family("mixed"), fitted, X_all[fitted.feature_names_in_],
                             X_all["crp"].to_numpy(), task="regression", clusters=INDEPENDENT,
                             outcome=None, rows="all", survey=design, models=["mixed"])
    assert table.rows == [] and _offers_the_attestation(table.info["exits"])
    assert table.info["exits"][0]["decision"] == {"kind": "select_models", "models": ["linear"]}

    # Every exit offered runs (BLUEPRINT §14.3): the sample-only attestation fits the blocked
    # family unweighted, its table carrying the attestation; the design-based family is the
    # linear one, already fit above.
    from turbotab.core.survey import ATTESTATION

    sample = survey_stages(diet.drop(columns=["eligible", "over", "high_crp"]), tmp_path / "f",
                           target="crp", task="regression", roles=DIET_ROLES,
                           models=["featurewise"], analyzed=analyzed, survey={"estimand": "sample"})
    tests = sample["fit"].data["models"][0]
    assert tests["coefficients"] and not tests["inference"]["refused"]
    assert tests["inference"]["covariance"] == "model"
    assert all(r["ci_low"] is not None for r in tests["coefficients"])
    assert any(ATTESTATION in c for c in tests["concerns"])

    # The relations the population contract declares, each seen above.
    relations = {(r.kind, r.target) for r in CONTRACTS["survey_population"].relations}
    assert relations == {
        ("implies", "every family and display"),
        ("conflicts", "mixed, GEE, feature-wise, elastic net and boosted-tree estimates"),
        ("invalidates", "the substitution band from row resampling"),
        ("implies", "multiple imputation")}


ALL_COMPONENTS_R = R_DESIGN.format(w="WTDRD1") + """
s <- subset(d, eligible == 1)
s <- update(s, kp = 4 * protein_g, kf = 9 * fat_g, kc = 4 * carb_g,
            ko = energy_kcal - 4 * protein_g - 9 * fat_g - 4 * carb_g)
g <- svyglm(crp ~ kp + kf + kc + ko + age + female, design = s)
m <- coef(svymean(~kf + kc + ko, s))
w <- m / sum(m)
v <- svycontrast(g, c(kp = 4, kf = -4 * w[["kf"]], kc = -4 * w[["kc"]], ko = -4 * w[["ko"]]))
plain <- colMeans(s$variables[, c("kf", "kc", "ko")])
out(list(est = as.numeric(coef(v)), se = as.numeric(SE(v)), shares = unname(w),
         sample_shares = unname(plain / sum(plain)), degf = degf(s)))
"""


@needs_r
def test_4_the_all_components_relative_effect_is_the_population_s(diet, tmp_path):
    """The all-components model's average relative effect (Tomova et al. 2022: the nutrient's
    coefficient less the other sources' coefficients weighted by "the proportion of the remaining
    energy intake contributed by each component") is a display of its own beside the coefficient
    table. Under the population answer its interval is design-based and its shares are the
    surveyed population's, survey-weighted means, not these participants'.

    Reference: R ``svyglm`` of the outcome on each source in kcal (the rest of energy as
    "other"), the shares from ``svymean`` over the same domain, and ``svycontrast`` of
    ``4 (β_protein − Σ w_k β_k)`` (per gram of protein). Bounds: 1e-6 and 1e-4, relative; the
    population's shares differ from the sample's here, so a row-mean share would miss."""
    from turbotab.core.decisions import EnergyAdjustment

    f = diet
    analyzed = np.flatnonzero(f["eligible"].to_numpy() == 1)
    out = survey_stages(f.drop(columns=["eligible", "over", "high_crp"]), tmp_path / "stages",
                        target="crp", task="regression", roles=DIET_ROLES, models=["linear"],
                        analyzed=analyzed,
                        energy_adjustment=EnergyAdjustment(
                            method="all_components", energy_column="energy_kcal",
                            nutrients=["protein_g", "fat_g", "carb_g"]))
    r = run_r(ALL_COMPONENTS_R, {"f": f}, tmp_path / "r")
    model = out["fit"].data["models"][0]
    assert model["inference"]["covariance"] == "design"
    row = next(x for x in model["coefficients"] if x["feature"] == "protein_g_relative")
    assert row["estimate"] == pytest.approx(r["est"], rel=1e-6)
    assert row["se"] == pytest.approx(r["se"], rel=1e-4)
    assert row["df"] == r["degf"]
    assert abs(r["shares"][0] - r["sample_shares"][0]) > 0.005  # the weights move the shares
    assert "remaining energy in the surveyed population (fat_g " in row["meaning"]
    assert f"fat_g {r['shares'][0]:.0%}" in row["meaning"]


CHAIN_R = R_DESIGN.format(w="WTMEC2YR") + """
knots <- scan("knots.txt", quiet = TRUE)
B <- Hmisc::rcspline.eval(f$fiber, knots = knots, inclx = TRUE)
f$b1 <- B[, 1]; f$b2 <- B[, 2]; f$b3 <- B[, 3]
d <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR, nest = TRUE, data = f)
s <- subset(d, !is.na(mortstat))
x <- svycoxph(Surv(permth_exm, mortstat) ~ b1 + b2 + b3 + RIDAGEYR + female + kcal, design = s,
              control = coxph.control(eps = 1e-14, iter.max = 200, toler.chol = 1e-15))
overall <- regTermTest(x, ~ b1 + b2 + b3, df = Inf)
nonlinear <- regTermTest(x, ~ b2 + b3, df = Inf)
out(list(est = unname(coef(x)), se = unname(sqrt(diag(vcov(x)))), degf = degf(s),
         overall = as.numeric(overall$chisq), nonlinear = as.numeric(nonlinear$chisq)))
"""


@needs_r
def test_4_the_nhanes_mortality_chain_runs_through_the_server(tmp_path):
    """MODELING_SEQUENCE §6 chain 1, its survey part, end to end through the real HTTP API: an
    NHANES table with linked mortality (no follow-up for the under-20s, as in the public-use
    files), an RCS of fiber, the survey question answered "the surveyed population", and
    survey-weighted Cox. The Router asks the survey question after the roles; the population answer
    makes the Cox table design-based (Binder), the spline's tests adjusted Wald F tests on the
    design's degrees of freedom, and the record says so in the methods sentences, asserted word for
    word.

    Reference: R ``svycoxph`` on Harrell's basis (``Hmisc::rcspline.eval`` at the app's knots,
    ``inclx = TRUE``), the under-20s a domain (``subset``); the Wald statistics from R's
    ``regTermTest`` (``df = Inf``: the χ² it reports is ``bᵀV⁻¹b`` on R's ``vcov``), turned into the
    adjusted F, ``(d − q + 1) W / (d q)`` on ``(q, d − q + 1)`` (Korn & Graubard 1990), by that
    formula. Bounds: coefficients 1e-6, standard errors 1e-4, F 1e-4, relative."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project
    from turbotab.core.tests.truths import Truth

    f = nhanes_mortality(seed=5, n_per_psu=35)
    f.loc[f["eligible"] == 0, ["mortstat", "permth_exm"]] = np.nan
    f = f[["SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "RIDAGEYR", "female", "fiber", "kcal",
           "mortstat", "permth_exm"]]
    path = tmp_path / "nhanes_mortality.csv"
    f.to_csv(path, index=False)
    truth = Truth({"code_or_count:female": "code", "code_or_count:SDMVSTRA": "code",
                   "code_or_count:SDMVPSU": "code", "code_or_count:mortstat": "code",
                   "code_or_count:permth_exm": "amount",
                   # WP17: fiber is the exposure; age and sex come before the diet and cause both
                   # it and death; total energy shares the diet's common causes (a confounder).
                   "exposure:mortstat": "fiber", "adjust:RIDAGEYR": "yes,yes,no",
                   "adjust:female": "yes,yes,no", "adjust:kcal": "unknown,unknown,no"},
                  fixture="nhanes_mortality (generator)")
    with local_server(tmp_path / "home") as client:
        d = open_project(client, path, truth)
        d.decide({"kind": "set_lens", "lenses": ["dietary"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "mortstat"})
        d.reach("task")
        d.decide({"kind": "set_task", "column": "mortstat", "task": "time_to_event"})
        d.answer("event", {"kind": "set_event", "column": "mortstat", "level": "1"})
        d.decide({"kind": "set_follow_up", "column": "mortstat", "time_column": "permth_exm"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        d.reach("roles")
        d.decide_roles({"SEQN": "identifier", "fiber": "exposure", "RIDAGEYR": "covariate",
                        "female": "covariate", "kcal": "covariate", "permth_exm": "time",
                        "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"})
        assert d.reach("survey")["status"] == "open"
        options = d.artifact("proposals")["survey"]["options"]
        assert [o["key"] for o in options] == ["population:WTMEC2YR", "sample"]
        d.decide(options[0]["decision"])
        d.reach("exclusions")
        d.decide({"kind": "set_exclusions", "rules": []})
        d.reach("missing")
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        d.reach("split")
        d.decide({"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        d.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        d.decide({"kind": "set_exposure_form", "column": "fiber", "form": "spline", "knots": 4})
        d.reach("models")
        d.decide({"kind": "select_models", "models": ["cox"]})
        fit = d.artifact("fit")
        said = {r["decision"]["kind"]: r["sentence"] for r in d.view()["decisions"]}
    assert assert_design_based_or_blocked(fit) == {"cox": "design"}
    model = fit["models"][0]
    tests = {t["test"]: t for t in model["exposure_tests"]}
    (tmp_path / "r").mkdir()
    np.savetxt(tmp_path / "r" / "knots.txt", tests["overall"]["knots"])
    r = run_r(CHAIN_R, {"f": f}, tmp_path / "r")
    assert [row["feature"] for row in model["coefficients"]][:3] == ["fiber", "fiber'", "fiber''"]
    assert rel([row["estimate"] for row in model["coefficients"]], r["est"]) < 1e-6
    assert rel([row["se"] for row in model["coefficients"]], r["se"]) < 1e-4
    df = r["degf"]
    assert model["inference"]["survey"]["df"] == df == 16
    for name, q in (("overall", 3), ("nonlinear", 2)):
        F = (df - q + 1) * r[name] / (df * q)
        assert tests[name]["statistic"] == pytest.approx(F, rel=1e-4)
        assert (tests[name]["df_num"], tests[name]["df_den"]) == (q, df - q + 1)
        assert tests[name]["p"] == pytest.approx(stats.f.sf(F, q, df - q + 1), rel=1e-4)
    assert said["set_survey"] == (
        "The estimates describe the surveyed population: rows were weighted by `WTMEC2YR`, and "
        "standard errors were estimated by Taylor series linearization over `SDMVPSU` nested "
        "within `SDMVSTRA` (`31` PSUs in `15` strata), first-stage units taken as sampled with "
        "replacement, with t intervals on the PSUs minus the strata that hold the analysis rows.")
    assert said["select_models"].startswith(
        "One model family was chosen: Cox proportional hazards. For the surveyed population, Cox "
        "regression (Binder's pseudo-likelihood, Efron ties) was weighted by `WTMEC2YR`, with "
        "standard errors by Taylor linearization over the survey design. ")

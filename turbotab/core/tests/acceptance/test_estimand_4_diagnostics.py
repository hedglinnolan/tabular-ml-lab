"""ESTIMAND · 4 · diagnostics are reported and never silently acted on (MODELING_SEQUENCE §1 row 11,
inference: "Diagnostics are reported (proportional hazards by Schoenfeld; residuals and
influence)"; the review record: "These are reported, and a failed check leads to a recorded change,
never a silent switch").

The references are R's own, run on the same rows (``r_reference.run_r``):

* ``survival::cox.zph(coxph(..., ties = "efron"))``, global and per term, its default
  Kaplan–Meier time scale, on right-censored rows with and without tied times and on rows with
  delayed entry, to 1e-6;
* ``survival::survSplit`` then ``coxph(Surv(tstart, time, event) ~ x + x:late + …)`` for the
  hazard ratio before and after the cut, to 1e-6 (the response the app records when the check
  fails);
* ``hatvalues`` and ``cooks.distance`` of ``lm`` and of a binomial ``glm``, to 1e-8.

R's ``coxph`` runs with ``timefix = FALSE``: by default it first merges times that are equal up to
floating-point tolerance (``aeqSurv``), which on the split follow-up merged one event time with the
cut and moved the log-likelihood by 0.006; the engine compares times exactly, as the partial
likelihood's definition (at risk when entry < t ≤ time) does, and so does R once told to.

The app's side of "a failed check leads to a recorded change" is driven through the server in
``test_estimand_chain.py``; here the stage's own reading of the check is held to the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core.models import effects
from turbotab.core.models.survival import cox_fit, survival_outcome
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r


def _cohort(n: int = 500, seed: int = 3, ties: bool = False, entry: bool = False) -> pd.DataFrame:
    """A cohort whose hazard ratio for ``x1`` changes over follow-up: log hazard ratio 1.0 before
    t = 0.5 and −0.3 after (drawn by inverting the piecewise cumulative hazard), with a binary and a
    three-level covariate that keep proportional hazards."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    grp = rng.choice(["a", "b", "c"], n)
    rest = 0.3 * x2 + 0.4 * (grp == "b")
    early, late = np.exp(1.0 * x1 + rest), np.exp(-0.3 * x1 + rest)
    tau = 0.5
    E = rng.exponential(1.0, n)
    T = np.where(E < tau * early, E / early, tau + (E - tau * early) / late)
    C = rng.exponential(2.0, n)
    time = np.minimum(T, C)
    if ties:
        time = np.round(time, 1) + 0.05
    start = rng.uniform(0, 0.3, n) * (time > 0.35) if entry else np.zeros(n)
    return pd.DataFrame({"start": start, "time": time, "event": (T <= C).astype(int), "x1": x1,
                         "x2": x2, "grp": grp})


def _design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([frame["x1"], frame["x2"], (frame["grp"] == "b").astype(float),
                         (frame["grp"] == "c").astype(float)])
    y = survival_outcome(frame["event"].to_numpy(), frame["time"].to_numpy(),
                         frame["start"].to_numpy())
    return X, y


ZPH_R = """
library(survival)
d <- read.csv(cohort_csv)
d$grp <- factor(d$grp, levels = c("a", "b", "c"))
ctl <- coxph.control(eps = 1e-14, iter.max = 100, toler.chol = 1e-15, timefix = FALSE)
f <- if (ENTRY) coxph(Surv(start, time, event) ~ x1 + x2 + grp, data = d, ties = "efron", control = ctl) else
     coxph(Surv(time, event) ~ x1 + x2 + grp, data = d, ties = "efron", control = ctl)
z <- cox.zph(f)
out(list(chisq = unname(z$table[, "chisq"]), df = unname(z$table[, "df"]),
         p = unname(z$table[, "p"]), terms = rownames(z$table), beta = unname(coef(f))))
"""


@needs_r
@pytest.mark.parametrize("ties,entry", [(False, False), (True, False), (False, True)])
def test_4_proportional_hazards_agree_with_cox_zph(tmp_path, ties, entry):
    """The score tests per term and global agree with R's ``cox.zph`` to 1e-6 (relative), the
    three-level ``grp`` tested as one term on two degrees of freedom as R groups it."""
    frame = _cohort(ties=ties, entry=entry)
    X, y = _design(frame)
    fit = cox_fit(X, y)
    assert fit.converged
    found = effects.cox_zph(X, y, fit.beta, {"x1": [0], "x2": [1], "grp": [2, 3]})
    r = run_r(ZPH_R.replace("ENTRY", "TRUE" if entry else "FALSE"), {"cohort": frame}, tmp_path)
    np.testing.assert_allclose(fit.beta, r["beta"], rtol=1e-8, atol=1e-10)
    ours = {row["term"]: row for row in [*found["terms"], found["global"]]}
    for term, chisq, df, p in zip(r["terms"], r["chisq"], r["df"], r["p"]):
        assert ours[term]["chisq"] == pytest.approx(chisq, rel=1e-6), term
        assert ours[term]["df"] == df
        assert ours[term]["p"] == pytest.approx(p, rel=1e-6, abs=1e-12), term
    # the fixture's x1 violates proportional hazards; the check sees it
    assert ours["x1"]["p"] < effects.PH_ALPHA


SPLIT_R = """
library(survival)
d <- read.csv(cohort_csv)
d$grp <- factor(d$grp, levels = c("a", "b", "c"))
d$id <- seq_len(nrow(d))
s <- survSplit(Surv(time, event) ~ ., data = d, cut = CUT, episode = "period", start = "tstart")
s$late <- as.numeric(s$period == 2)
f <- coxph(Surv(tstart, time, event) ~ x1 + x2 + grp + x1:late, data = s, ties = "efron",
           control = coxph.control(eps = 1e-14, iter.max = 100, toler.chol = 1e-15,
                                   timefix = FALSE))
b <- coef(f); V <- vcov(f)
c2 <- c(1, 0, 0, 0, 1)
out(list(before = unname(b["x1"]), before_se = sqrt(V["x1", "x1"]),
         after = sum(c2 * b), after_se = sqrt(drop(t(c2) %*% V %*% c2)), rows = nrow(s)))
"""


@needs_r
def test_4_a_failed_check_records_hazard_ratios_before_and_after_the_cut(tmp_path):
    """The response the app records for a failed proportional-hazards check of the exposure: its
    hazard ratio in each period of follow-up, split at the median event time. Reference: R's
    ``survSplit`` and ``coxph`` with the exposure's product with the later period, to 1e-6."""
    frame = _cohort(seed=11)
    X, y = _design(frame)
    cut = float(np.median(frame.loc[frame["event"] == 1, "time"]))
    found = effects.period_hazard_ratios(X, y, 0, cut)
    r = run_r(SPLIT_R.replace("CUT", repr(cut)), {"cohort": frame}, tmp_path)
    before, after = found["periods"]
    assert found["n_split_rows"] == r["rows"]
    assert before["estimate"] == pytest.approx(r["before"], rel=1e-6)
    assert before["se"] == pytest.approx(r["before_se"], rel=1e-6)
    assert after["estimate"] == pytest.approx(r["after"], rel=1e-6)
    assert after["se"] == pytest.approx(r["after_se"], rel=1e-6)
    assert before["ratio"] == pytest.approx(np.exp(r["before"]), rel=1e-6)


INFLUENCE_R = """
d <- read.csv(rows_csv)
m1 <- lm(y ~ x + z + w, data = d)
m2 <- glm(b ~ x + z + w, data = d, family = binomial,
          control = glm.control(epsilon = 1e-15, maxit = 100))
out(list(h1 = unname(hatvalues(m1)), D1 = unname(cooks.distance(m1)),
         h2 = unname(hatvalues(m2)), D2 = unname(cooks.distance(m2)),
         p1 = m1$rank, p2 = m2$rank))
"""


@needs_r
def test_4_leverage_and_cooks_distance_agree_with_r(tmp_path):
    """Each row's leverage and Cook's distance for a least-squares and a logistic model agree with
    R's ``hatvalues`` and ``cooks.distance`` to 1e-8 (relative; absolute below 1e-14)."""
    rng = np.random.default_rng(5)
    n = 300
    x, z, w = rng.normal(size=n), rng.normal(size=n), rng.binomial(1, 0.5, n).astype(float)
    x[0] = 6.0  # one row far out on x: a high-leverage, influential row
    y = 1 + 0.5 * x - 0.3 * z + rng.normal(size=n)
    y[0] += 8
    b = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 0.8 * x + 0.5 * z - 0.6 * w)))).astype(float)
    frame = pd.DataFrame({"y": y, "b": b, "x": x, "z": z, "w": w})
    X = np.column_stack([np.ones(n), x, z, w])
    ols = effects.ols_influence(X, y)
    logit = effects.logistic_influence(X, b)
    r = run_r(INFLUENCE_R, {"rows": frame}, tmp_path)
    np.testing.assert_allclose(ols["leverage"], r["h1"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(ols["cooks"], r["D1"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(logit["leverage"], r["h2"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(logit["cooks"], r["D2"], rtol=1e-8, atol=1e-14)
    assert ols["p"] == r["p1"] and logit["p"] == r["p2"]
    # the influential row is the one Cook's (1977) median-F reference flags
    threshold = effects.cook_threshold(ols["p"], n)
    assert np.flatnonzero(ols["cooks"] > threshold).tolist() == [0]


# ── the stage: the check, its exits, and the recorded response ───────────────


def _cox_state(responses: dict[str, str] | None = None) -> d.ProjectState:
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "grp": "covariate",
             "time": "time"}
    answers = {"x2": ef.CONFOUNDER, "grp": ef.CONFOUNDER}
    return ef.state(target="event", task="time_to_event", exposure="x1", measure="hazard_ratio",
                    roles=roles, answers=answers, event="1", models=["cox"],
                    amounts={"code_or_count:x2": "amount"}, responses=responses,
                    follow_up=d.FollowUpSpec(time_column="time"))


@needs_r
def test_4_the_stage_reports_the_check_and_offers_a_recorded_response(tmp_path):
    """On a cohort whose exposure's hazard ratio changes over follow-up, the effects stage reports
    the proportional-hazards tests R's ``cox.zph`` gives for the primary model (to 1e-6), calls the
    check failed for the exposure, keeps the hazard ratio, and offers two recorded responses; once
    the period response is recorded, the hazard ratio before and after the median event time is
    shown beside it, as R's ``survSplit`` and ``coxph`` give them (to 1e-6). Nothing changes until
    a response is recorded."""
    frame = _cohort(seed=11).assign(pid=lambda f: np.arange(len(f)))
    run = ef.run(frame, tmp_path / "before", _cox_state(), fit=False)
    family = run["effects"]["families"][0]
    [check] = family["diagnostics"]
    r = run_r(ZPH_R.replace("ENTRY", "FALSE"), {"cohort": frame}, tmp_path / "r")
    tests = {t["term"]: t for t in check["tests"]}
    for term, chisq, p in zip(r["terms"], r["chisq"], r["p"]):
        assert tests[term]["statistic"] == pytest.approx(chisq, rel=1e-6), term
        assert tests[term]["p"] == pytest.approx(p, rel=1e-6, abs=1e-12), term
    assert check["status"] == "failed" and check["response"] is None and check["change"] is None
    assert [e["decision"] for e in check["exits"]] == [
        {"kind": "respond_diagnostic", "exposure": "x1", "check": "proportional_hazards",
         "action": "period_hazard_ratios"},
        {"kind": "respond_diagnostic", "exposure": "x1", "check": "proportional_hazards",
         "action": "keep_labeled"}]
    primary = ef.sequence(run["effects"])["model_2"]["effects"][0]
    assert primary["feature"] == "x1" and primary["estimate"] == pytest.approx(r["beta"][0], rel=1e-8)

    after = ef.run(frame, tmp_path / "after",
                   _cox_state({"proportional_hazards": "period_hazard_ratios"}), fit=False)
    [check] = after["effects"]["families"][0]["diagnostics"]
    assert check["status"] == "failed" and check["response"] == "period_hazard_ratios"
    assert check["exits"] == []
    cut = float(np.median(frame.loc[frame["event"] == 1, "time"]))
    ref = run_r(SPLIT_R.replace("CUT", repr(cut)), {"cohort": frame}, tmp_path / "split")
    before_row, after_row = check["change"]
    assert before_row["estimate"] == pytest.approx(ref["before"], rel=1e-6)
    assert after_row["estimate"] == pytest.approx(ref["after"], rel=1e-6)
    assert after_row["se"] == pytest.approx(ref["after_se"], rel=1e-6)
    # the primary estimate itself is unchanged: the response adds, it never silently switches
    assert ef.sequence(after["effects"])["model_2"]["effects"][0]["estimate"] == primary["estimate"]
    assert ("the response recorded is the exposure's hazard ratio before and after the median "
            "event time" in after["effects"]["methods"])
    assert "no response is recorded yet" in run["effects"]["methods"]


INFLUENTIAL_R = """
d <- read.csv(rows_csv)
d$sex <- factor(d$sex, levels = c("female", "male"))
m <- lm(glucose ~ fiber + age + sex + smoking + activity, data = d)
D <- cooks.distance(m); p <- m$rank; n <- nrow(d)
keep <- D <= qf(0.5, p, n - p)
out(list(flagged = I(which(!keep) - 1), threshold = qf(0.5, p, n - p), maxD = max(D)))
"""


@needs_r
def test_4_an_influential_row_fails_the_check_and_its_response_refits_without_it(tmp_path):
    """A row far out on the exposure with a wild outcome fails the influence check; the recorded
    response refits the primary without the rows R's ``cooks.distance`` puts above the median of
    F(p, n − p), and the refit agrees with statsmodels on those rows (HC3, to 1e-8)."""
    frame = ef.cohort(300, seed=2)
    frame.loc[0, ["fiber", "glucose"]] = [70.0, 260.0]
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    base = dict(target="glucose", task="regression", measure="mean_difference", roles=roles,
                answers=answers)
    r = run_r(INFLUENTIAL_R, {"rows": frame}, tmp_path / "r")
    flagged = [int(i) for i in r["flagged"]]
    assert flagged == [0]
    first = ef.run(frame, tmp_path / "before", ef.state(**base), fit=False)
    [check] = first["effects"]["families"][0]["diagnostics"]
    assert check["status"] == "failed" and check["flagged"] == len(flagged)
    assert check["threshold"] == pytest.approx(r["threshold"], rel=1e-10)
    assert check["largest"] == pytest.approx(r["maxD"], rel=1e-8)
    assert {e["decision"]["action"] for e in check["exits"]} == {"without_influential",
                                                                "keep_labeled"}
    second = ef.run(frame, tmp_path / "after",
                    ef.state(**base, responses={"influence": "without_influential"}), fit=False)
    [check] = second["effects"]["families"][0]["diagnostics"]
    kept = frame.drop(index=flagged)
    X = ef.design_matrix(kept, ["fiber", "age", "sex", "smoking", "activity"])
    reference = sm.OLS(kept["glucose"].to_numpy(float), X).fit(cov_type="HC3", use_t=True)
    [row] = check["change"]
    assert row["estimate"] == pytest.approx(reference.params["fiber"], rel=1e-8)
    assert row["se"] == pytest.approx(reference.bse["fiber"], rel=1e-8)
    assert check["change_label"] == ("The primary model refit without the 1 influential row, on "
                                     "299 rows.")

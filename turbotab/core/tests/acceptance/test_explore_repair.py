"""EXPLORE repair · the independent verifier's open items, closed (wave 2, one round).

* **(2) Nonlinearity by inner cross-validation on a yes/no outcome.** The inner comparison fitted
  each logistic regression with a fit that called a converged coefficient above 50 a failure, so a
  spline basis on a 0–1 share (or on an N(0, 1) predictor) scored an infinite loss and was never
  chosen. Reference: R's ``glm`` on the same inner splits, its basis from ``Hmisc::rcspline.eval``
  on each inner training fold; the truth is a simulated U.
* **(3) Stepwise at p ≥ n_fold − 1.** Accepted, then the fit crashed. Refused now where the cohort
  shows it, with the selections that work at p ≫ n as exits; the boundary is computed here from
  its definition (a model with every column and an intercept needs fewer columns than rows less
  one), and the backstop names its way forward.
* **(4) Subgroups by the code-or-amount reading.** A confirmed 12-category code was cut into
  thirds of its code numbers and a 7-value count read as 7 categories (BLUEPRINT §14.3). The
  groups now follow the ledger, asked while unsettled; pandas counts each group. **Shrinkage** on
  unscaled predictors diverged and took the whole evaluation stage down with a false "separated"
  message; its intercept is R's ``glm(y ~ 1 + offset(s·lp), binomial)`` now, the recalibrated
  model's coefficients are the ones shrunk (NumPy least squares on the deployed log-odds), and a
  failure is said in the record, never the stage.
* **(5) Riley's minimum** counts the spline rule's columns (the criteria by hand from Riley et al.,
  *BMJ* 2020;368:m441). The benchmark test now asserts the served numbers against hand refits.
* **(8) Under inference no cross-validated score is read, served or said**: the explanations carry
  no floor, and the split's sentence and the export say none was estimated.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import Refusal
from turbotab.core.methods import levers as L
from turbotab.core.models import decision_curve as DC
from turbotab.core.tests.acceptance import explore_references as ref
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.graph_runner import GraphRun


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


# ═════════════════════════════════════════════════════════════════════════════
# (2) nonlinearity by inner cross-validation on a yes/no outcome
# ═════════════════════════════════════════════════════════════════════════════


def _u_table(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """A known truth: the log-odds a U in a 0–1 share and in an N(0, 1) predictor, linear in z."""
    rng = np.random.default_rng(seed)
    share = rng.uniform(0, 1, n)
    c = rng.normal(size=n)
    z = rng.normal(size=n)
    eta = -2.5 + 24 * (share - 0.5) ** 2 + 1.0 * c ** 2 + 0.5 * z
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return pd.DataFrame({"share": share, "c": c, "z": z, "y": y})


INNER_CV_R = """
suppressPackageStartupMessages(library(Hmisc))
d <- read.csv(d_csv)
cols <- c("share", "c", "z")
ctl <- glm.control(epsilon = 1e-14, maxit = 200)
fitloss <- function(Xtr, ytr, Xte, yte) {
  b <- suppressWarnings(glm.fit(cbind(1, Xtr), ytr, family = binomial(), control = ctl))$coefficients
  p <- plogis(drop(cbind(1, Xte) %*% b))
  list(loss = -mean(yte * log(p) + (1 - yte) * log(1 - p)), maxb = max(abs(b)))
}
res <- list()
for (col in cols) {
  lin <- c(); spl <- c(); w <- c(); maxb <- 0
  for (f in sort(unique(d$fold))) {
    tr <- d[d$fold != f, ]; te <- d[d$fold == f, ]
    a <- fitloss(as.matrix(tr[, cols]), tr$y, as.matrix(te[, cols]), te$y)
    kn <- rcspline.eval(tr[[col]], nk = NK, knots.only = TRUE)
    others <- setdiff(cols, col)
    Btr <- cbind(rcspline.eval(tr[[col]], knots = kn, inclx = TRUE), as.matrix(tr[, others]))
    Bte <- cbind(rcspline.eval(te[[col]], knots = kn, inclx = TRUE), as.matrix(te[, others]))
    s <- fitloss(Btr, tr$y, Bte, te$y)
    lin <- c(lin, a$loss); spl <- c(spl, s$loss); w <- c(w, nrow(te)); maxb <- max(maxb, s$maxb)
  }
  res[[col]] <- list(lin = sum(w * lin) / sum(w), spl = sum(w * spl) / sum(w), maxb = maxb)
}
out(res)
"""


@needs_r
def test_2_inner_cv_on_a_yes_no_outcome_scores_each_form_as_r_glm_does_and_bends_the_u(tmp_path):
    """The inner comparison of forms (MODELING_SEQUENCE §1 row 5, prediction: "linear · restricted
    cubic spline · chosen by an in-fold rule or inner CV") on a yes/no outcome: each candidate's
    linear and spline log loss over the inner folds, the others linear, as R's ``glm`` scores them
    on the same splits with Hmisc's basis on each inner training fold (to 10⁻⁸). The spline's
    converged coefficients run past 50 here (a 0–1 share's basis is small), which no longer reads
    as a failure: the U-shaped share and the U-shaped N(0, 1) predictor are bent, as the truth has
    them, and each choice is the one R's losses make."""
    frame = _u_table()
    n = len(frame)
    fold = np.random.default_rng(11).permutation(n) % 5
    splits = [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(5)]
    X, y = frame[["share", "c", "z"]], frame["y"].to_numpy()
    step = L.InnerCVForms(["share", "c", "z"], "binary", splits, 0).fit(X, y)
    k = ref.harrell_k(int(min(y.sum(), n - y.sum())))  # Harrell §4.4: the rarer class's count
    assert step.k_ == k == 5
    r = run_r(INNER_CV_R.replace("NK", str(k)), {"d": frame.assign(fold=fold)}, tmp_path)
    for column in ("share", "c", "z"):
        linear, spline = step.losses_[column]
        assert math.isfinite(spline), column
        assert linear == pytest.approx(r[column]["lin"], rel=1e-8, abs=0)
        assert spline == pytest.approx(r[column]["spl"], rel=1e-8, abs=0)
    chosen = {c for c in ("share", "c", "z") if r[c]["spl"] < r[c]["lin"]}
    assert set(step.knots_) == chosen
    assert {"share", "c"} <= chosen  # the truth bends both
    assert r["share"]["maxb"] > 50  # the regime the old fit called a failure


def test_2_through_the_fit_the_inner_cv_rule_bends_the_u_on_a_yes_no_outcome(tmp_path):
    """The same rule as the fit runs it (``set_levers``, forms by inner cross-validation) on a
    yes/no outcome: the final model's step bends the U-shaped share and the U-shaped N(0, 1)
    predictor, and every inner loss is finite."""
    frame = _u_table(1500, 8)
    frame["y"] = np.where(frame["y"] == 1, "yes", "no")
    frame.insert(0, "pid", np.arange(len(frame)))
    frame.to_csv(tmp_path / "u.csv", index=False)
    run = GraphRun(tmp_path / "u.csv", tmp_path / "p")
    roles = {"pid": "identifier", "share": "covariate", "c": "covariate", "z": "covariate"}
    state = d.ProjectState(
        lens=["clinical"], target="y", task="binary", event="yes", purpose="prediction",
        roles=roles, missing="complete_case",
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=2, folds=5), models=["linear"],
        levers=d.LeverSpec(forms="inner_cv"))
    out = run.run(state, upto=["fit"])
    step = out["fit"].objects["fitted"]["linear"].named_steps["lever_forms"]
    assert {"share", "c"} <= set(step.knots_)
    assert all(math.isfinite(v) for pair in step.losses_.values() for v in pair)
    run.close()


# ═════════════════════════════════════════════════════════════════════════════
# (3) stepwise selection needs more rows than columns in the smallest training fold
# ═════════════════════════════════════════════════════════════════════════════


def _wide_ctx(n: int, p: int, *, holdout: float = 0.0, folds: int = 5, levers=None,
              selection=None, distinct: int | None = None) -> dict:
    """A validation context as the server builds it: its column summaries as dicts, the cohort's
    rows and predictors."""
    cols = [f"g{i:03d}" for i in range(p)]
    state = d.ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="prediction",
        roles={c: "covariate" for c in cols}, split=d.SplitSpec(holdout=holdout, seed=0, folds=folds),
        models=["linear"], levers=levers, selection=selection)
    info = {c: {"dtype": "numeric", "n_unique": distinct or n, "n_missing": 0} for c in cols}
    cohort = {"n_final": n, "predictors": cols}
    return {"state": state, "column_info": info, "columns": [*cols, "y"],
            "artifact": lambda stage: cohort if stage == "cohort" else None}


def _fold_rows(n: int, holdout: float, folds: int) -> int:
    """The smallest training fold, by definition: the holdout drawn first, then (K − 1)/K of the
    rest."""
    return math.floor(math.floor(n * (1 - holdout)) * (folds - 1) / folds)


@pytest.mark.parametrize("n,holdout", [(60, 0.0), (60, 0.2), (200, 0.0)])
def test_3_stepwise_is_refused_exactly_where_the_smallest_fold_cannot_fit_every_column(n, holdout):
    """Backward elimination starts from every candidate column and an intercept, which needs
    p < n_fold − 1 (``MASS::stepAIC`` stops there: "AIC is -infinity for this model"). Below the
    boundary the answer is accepted; at it, refused with the selections that work at p ≫ n, each
    of which is accepted."""
    rows = _fold_rows(n, holdout, 5)
    fits = _wide_ctx(n, rows - 2, holdout=holdout)
    assert d.validate(d.SetSelection(method="stepwise", pre_selected="no"), fits)
    ctx = _wide_ctx(n, rows - 1, holdout=holdout)
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetSelection(method="stepwise", pre_selected="no"), ctx)
    r = caught.value
    assert r.code == "stepwise_needs_rows"
    assert r.message == (
        f"Backward elimination starts from all {rows - 1:,} candidate columns, and the smallest "
        f"training fold has {rows:,} rows: a model with every column and an intercept needs more "
        f"rows than that, so it cannot start. At p ≫ n a penalized or screening selection comes "
        f"first.")
    assert [e["label"] for e in r.exits] == ["Elastic net, in each training fold",
                                             "In-fold screening by correlation with the outcome",
                                             "No selection"]
    for e in r.exits:
        assert d.validate(e["decision"], ctx)
    # A labeled sensitivity analysis under inference is a different procedure: never refused here.
    inference = dict(ctx, state=ctx["state"].model_copy(update={"purpose": "inference"}))
    assert d.validate(d.SetSelection(method="stepwise", sensitivity=True), inference)
    # the contract's conflict fires for stepwise under prediction, and for nothing else (§13)
    from turbotab.core import contracts as C

    consequence = ["too_few_rows_for_every_column"]
    assert [f.relation.name for f in C.fired({"variable_selection": "stepwise"}, "prediction",
                                             consequence)] == ["stepwise_needs_rows"]
    assert C.fired({"variable_selection": "elastic_net"}, "prediction", consequence) == []
    assert C.fired({"variable_selection": "stepwise"}, "inference", consequence) == []


def test_3_splines_that_would_crowd_out_a_recorded_stepwise_selection_are_refused():
    """A form rule adds k − 2 columns per continuous predictor (k by Harrell's rule on the fold:
    48 rows give 4 knots), so set after a stepwise selection it is refused where the columns reach
    the boundary, with the exit that keeps every predictor linear; below it, accepted."""
    rows = _fold_rows(60, 0.0, 5)
    assert rows == 48 and ref.harrell_k(rows) == 4
    stepwise = d.SelectionSpec(method="stepwise", pre_selected="no")
    fits = _wide_ctx(60, 15, selection=stepwise)  # 15 × 3 = 45 < 47
    assert d.validate(d.SetLevers(forms="rule"), fits)
    ctx = _wide_ctx(60, 16, selection=stepwise)  # 16 × 3 = 48 ≥ 47
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetLevers(forms="rule"), ctx)
    assert caught.value.code == "stepwise_needs_rows"
    assert caught.value.message == (
        "With these splines backward elimination would start from all 48 candidate columns, and "
        "the smallest training fold has 48 rows: a model with every column and an intercept needs "
        "more rows than that.")
    keep = caught.value.exits[0]
    assert keep["label"] == "Keep every predictor linear" and d.validate(keep["decision"], ctx)
    # and the stepwise answer itself, given after the splines, sees the same columns
    with pytest.raises(Refusal):
        d.validate(d.SetSelection(method="stepwise", pre_selected="no"),
                   _wide_ctx(60, 16, levers=d.LeverSpec(forms="rule")))


def test_3_at_the_boundary_the_step_runs_and_past_it_the_backstop_names_the_way_forward():
    """On a fold of 48 rows: 46 columns run backward elimination to its end; 47 cannot start, and
    the step says so with the selections that work instead (the answer is refused before it gets
    here; this is for an answer a later change crowded)."""
    from turbotab.core.models.variable_selection import Selector

    rng = np.random.default_rng(3)
    y = rng.normal(size=48)
    ok = pd.DataFrame(rng.normal(size=(48, 46)), columns=[f"g{i}" for i in range(46)])
    kept = Selector("stepwise", "regression").fit(ok, y + ok["g0"].to_numpy())
    assert "g0" in kept.kept_
    crowded = pd.DataFrame(rng.normal(size=(48, 47)), columns=[f"g{i}" for i in range(47)])
    with pytest.raises(ValueError) as caught:
        Selector("stepwise", "regression").fit(crowded, y)
    assert str(caught.value) == (
        "Backward elimination starts from all 47 candidate columns, and these training rows "
        "number 48: a model with every column and an intercept needs more rows than that. Choose "
        "the elastic net or in-fold screening in the selection question.")


@pytest.mark.parametrize("p", [44, 45])
def test_3_the_answers_boundary_is_the_one_the_fits_step_meets(tmp_path, p):
    """The refusal's count is the selection step's own: 60 rows in 5 folds leave 48 training rows
    (⌊60 · 4/5⌋), and p numbers beside a three-level category put p + 2 columns before backward
    elimination (counted here by hand). With 44 numbers (46 columns, below 47) the answer is accepted
    on the cohort and the summaries as the server holds them, the fit runs, and the fitted step
    received exactly those 46 columns; with 45 (47 columns) the answer is refused, and the fit run
    without the validator (as a later answer could leave it) stops with the backstop that names the
    way forward."""
    rng = np.random.default_rng(5)
    n = 60
    X = rng.normal(size=(n, p)).round(4)
    frame = pd.DataFrame(X, columns=[f"g{i:03d}" for i in range(p)])
    frame["grp"] = rng.choice(["a", "b", "c"], n)
    frame["y"] = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)).round(4)
    frame.insert(0, "pid", np.arange(n))
    frame.to_csv(tmp_path / "w.csv", index=False)
    run = GraphRun(tmp_path / "w.csv", tmp_path / "p")
    roles = {"pid": "identifier", "grp": "covariate", **{f"g{i:03d}": "covariate" for i in range(p)}}
    state = d.ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="prediction", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=1, folds=5), models=["linear"])
    cohort = run.run(state, upto=["cohort"])["cohort"].data
    ctx = {"state": state, "column_info": {c["name"]: c for c in run.info["columns"]},
           "columns": list(frame.columns),
           "artifact": lambda stage: cohort if stage == "cohort" else None}
    columns, rows = p + (3 - 1), _fold_rows(n, 0.0, 5)
    assert rows == 48
    answer = d.SetSelection(method="stepwise", pre_selected="no")
    chosen = d.fold_onto(state, answer)
    if columns < rows - 1:
        assert d.validate(answer, ctx)
        step = run.run(chosen, upto=["fit"])["fit"].objects["fitted"]["linear"].named_steps["select"]
        assert step.n_features_in_ == columns == 46
    else:
        with pytest.raises(Refusal) as caught:
            d.validate(answer, ctx)
        assert caught.value.code == "stepwise_needs_rows"
        assert caught.value.message.startswith(
            f"Backward elimination starts from all {columns} candidate columns, and the smallest "
            f"training fold has {rows} rows")
        with pytest.raises(ValueError) as stopped:
            run.run(chosen, upto=["fit"])
        assert str(stopped.value) == (
            f"Backward elimination starts from all {columns} candidate columns, and these training "
            f"rows number {rows}: a model with every column and an intercept needs more rows than "
            f"that. Choose the elastic net or in-fold screening in the selection question.")
    run.close()


# ═════════════════════════════════════════════════════════════════════════════
# (4) subgroups by the code-or-amount reading
# ═════════════════════════════════════════════════════════════════════════════


def _groups_table(n: int = 900, seed: int = 21) -> pd.DataFrame:
    """A 12-category code (``region``), a 6-category code named like race (``race``), a count of
    7 values (``visits``), a yes/no (``sex``) and a measurement (``pir``), beside two predictors and
    a yes/no outcome; ``x2`` is blank more often in two of the race codes."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    race = rng.integers(1, 7, n)
    lp = -0.4 + 0.9 * x1 - 0.6 * x2
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-lp)), "yes", "no")
    x2 = np.where(rng.random(n) < np.where(race >= 5, 0.25, 0.05), np.nan, x2)
    return pd.DataFrame({"pid": np.arange(n), "x1": x1, "x2": x2, "region": rng.integers(1, 13, n),
                         "race": race, "visits": rng.integers(0, 7, n),
                         "sex": rng.integers(1, 3, n), "pir": rng.uniform(0, 5, n), "y": y})


def _confirmed(**values: str) -> dict[str, str]:
    return {f"code_or_count:{c}": v for c, v in values.items()}


def _sizes(values: np.ndarray, how: str) -> dict[str, int]:
    """Each group's rows by pandas: the levels as text, or the thirds by NumPy's type-7 quantiles
    (≤ the first cut, up to the second, above it)."""
    s = pd.Series(values)
    if how == "levels":
        return {str(k): int(v) for k, v in s.astype(int).astype(str).value_counts().items()}
    lo, hi = np.quantile(values, [1 / 3, 2 / 3])
    return {f"≤ {lo:.4g}": int((s <= lo).sum()),
            f"{lo:.4g}–{hi:.4g}": int(((s > lo) & (s <= hi)).sum()),
            f"> {hi:.4g}": int((s > hi).sum())}


def test_4_subgroups_are_grouped_by_the_settled_reading_never_by_the_count_of_values():
    """BLUEPRINT §14.3 ("the consumer sets the question's scope"; "every confirmation is honored"):
    a 12-category code confirmed as codes is scored in its 12 levels and a 7-value count confirmed
    as an amount in its thirds, whatever their counts of values; confirming each the other way gives
    the other grouping; declaring the code categorical is a confirmation too. Unsettled, each is
    named, not scored, with the two confirmations that settle it. A yes/no is two groups either way
    and a measurement's decimals settle an amount by their values, so neither is asked. Each group's
    rows are pandas' counts."""
    frame = _groups_table()
    y = (frame["y"] == "yes").astype(int).to_numpy()
    risk = np.random.default_rng(1).uniform(0.05, 0.95, len(y))
    P = np.column_stack([1 - risk, risk])
    columns = {c: frame[c].to_numpy() for c in ("region", "visits", "sex", "pir")}

    def scored(state):
        found = DC.subgroup_performance("binary", y, P, columns, classes=[0, 1], metrics=["auc"],
                                        state=state)
        return {c["column"]: c for c in found}

    waiting = scored(d.ProjectState())
    for column in ("region", "visits"):
        entry = waiting[column]
        assert entry["grouped_by"] is None and entry["groups"] == []
        assert entry["note"] == (f"Whether `{column}`'s numbers are codes or amounts is not "
                                 f"settled, so its groups wait on that answer: its levels as codes, "
                                 f"its thirds as an amount.")
        asked = {(e["decision"]["kind"], e["decision"]["reading"], e["decision"]["column"],
                  e["decision"]["value"]) for e in entry["ask"]}
        assert asked == {("confirm_reading", "code_or_count", column, "code"),
                         ("confirm_reading", "code_or_count", column, "amount")}
    assert waiting["sex"]["grouped_by"] == "levels" and waiting["pir"]["grouped_by"] == "thirds"
    for state, want in (
            (d.ProjectState(reading_confirmations=_confirmed(region="code", visits="amount")),
             {"region": "levels", "visits": "thirds"}),
            (d.ProjectState(reading_confirmations=_confirmed(region="amount", visits="code")),
             {"region": "thirds", "visits": "levels"}),
            (d.ProjectState(categorical=["region"],
                            reading_confirmations=_confirmed(visits="amount")),
             {"region": "levels", "visits": "thirds"})):
        got = scored(state)
        for column, how in want.items():
            entry = got[column]
            assert entry["grouped_by"] == how, (column, state)
            assert {g["group"]: g["n"] for g in entry["groups"]} == _sizes(columns[column], how)
    assert len(scored(d.ProjectState(reading_confirmations=_confirmed(region="code")))
               ["region"]["groups"]) == 12
    # the intended-use contract's relation to the ledger fires wherever subgroups are named (§13)
    from turbotab.core import contracts as C

    fired = C.fired({"intended_use": "risk_estimation"}, "prediction",
                    ["code_or_count_reading", "subgroup_performance"])
    assert {f.relation.name for f in fired} == {"subgroups_read_the_ledger", "subgroups_scored"}


def test_4_naming_an_unsettled_column_as_a_subgroup_asks_its_reading_first():
    """Subgroup performance is a number-changing consumer of the column's code-or-amount reading
    (BLUEPRINT §14.1), so ``set_intended_use`` asks it as the fit asks its predictors' (one
    confirmation per reading) and accepts the answer once it is settled, either way; a yes/no
    column is never asked."""
    info = {"region": {"dtype": "integer", "n_unique": 12, "n_missing": 0},
            "sex": {"dtype": "integer", "n_unique": 2, "n_missing": 0},
            "x1": {"dtype": "numeric", "n_unique": 900, "n_missing": 0}}
    state = d.ProjectState(lens=["clinical"], target="y", task="binary", purpose="prediction")
    ctx = {"state": state, "column_info": info, "columns": [*info, "y"]}
    use = d.SetIntendedUse(use="risk_estimation", subgroups=["region", "sex"])
    with pytest.raises(Refusal) as caught:
        d.validate(use, ctx)
    r = caught.value
    assert r.code == "reading_unsettled"
    assert r.message.startswith(
        "Subgroup performance groups a column by its levels when its numbers are codes and by its "
        "thirds when they are an amount, and this one is not settled. Tell me about this column: "
        "`region`: ")
    assert {(e["decision"]["column"], e["decision"]["value"]) for e in r.exits} == {
        ("region", "code"), ("region", "amount")}
    for e in r.exits:
        answered = d.fold_onto(state, d.parse_decision(e["decision"]))
        assert d.validate(use, dict(ctx, state=answered))
    assert d.validate(d.SetIntendedUse(use="risk_estimation", subgroups=["sex"]), ctx)


@pytest.fixture(scope="module")
def grouped(tmp_path_factory):
    """The evaluation and Explore on the groups table, first with nothing confirmed, then with the
    race code confirmed as codes and the visit count as an amount."""
    folder = tmp_path_factory.mktemp("groups")
    frame = _groups_table()
    frame.to_csv(folder / "g.csv", index=False)
    run = GraphRun(folder / "g.csv", folder / "p")
    roles = {"pid": "identifier", "x1": "covariate", "x2": "covariate", "region": "excluded",
             "race": "excluded", "visits": "excluded", "sex": "excluded", "pir": "excluded"}
    state = d.ProjectState(
        lens=["clinical"], target="y", task="binary", event="yes", purpose="prediction",
        roles=roles, missing=d.MissingSpec(strategy="impute"),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.2, seed=4, folds=5), models=["linear"],
        intended_use=d.IntendedUseSpec(use="risk_estimation", subgroups=["race", "visits"]))
    waiting = run.run(state, upto=["explore", "evaluation"])
    settled = run.run(state.model_copy(update={
        "reading_confirmations": _confirmed(race="code", visits="amount")}),
        upto=["explore", "evaluation"])
    yield frame, waiting, settled
    run.close()


def _training(out) -> np.ndarray:
    a = out["split"].frames["assignment"]
    return a.loc[a["partition"].astype(str).str.lower().isin(["train", "training"]),
                 "row_id"].to_numpy(dtype=np.int64)


def test_4_through_the_stages_the_groups_wait_on_the_reading_then_follow_it(grouped):
    """Through the evaluation and Explore stages: unsettled, the race code and the visit count are
    named, not scored, and Explore's quality finding asks the reading instead of drawing groups;
    settled, the race code is scored in its 6 levels and the visit count in its thirds, each
    group's rows pandas' count of the training rows, and Explore's missing shares by group are
    pandas' too (TRIPOD+AI 7)."""
    frame, waiting, settled = grouped
    ev = {s["column"]: s for s in waiting["evaluation"].data["subgroups"]}
    assert ev["race"]["grouped_by"] is None and ev["visits"]["grouped_by"] is None
    quality = next(f for f in waiting["explore"].data["findings"]
                   if f["id"] == "explore::quality::race")
    assert quality["groups"] == []
    assert quality["summary"] == "`race`'s groups wait on whether its numbers are codes or amounts"
    assert quality["lever"]["question"] == "Are `race`'s numbers codes or amounts?"
    assert {(o["decision"]["column"], o["decision"]["value"]) for o in quality["lever"]["options"]} \
        == {("race", "code"), ("race", "amount")}

    train = frame.set_index("pid").loc[_training(settled)]
    ev = {s["column"]: s for s in settled["evaluation"].data["subgroups"]}
    assert ev["race"]["grouped_by"] == "levels" and ev["visits"]["grouped_by"] == "thirds"
    assert {g["group"]: g["n"] for g in ev["race"]["groups"]} == _sizes(train["race"].to_numpy(),
                                                                       "levels")
    assert {g["group"]: g["n"] for g in ev["visits"]["groups"]} == _sizes(
        train["visits"].to_numpy(), "thirds")
    quality = next(f for f in settled["explore"].data["findings"]
                   if f["id"] == "explore::quality::race")
    blank = train[["x1", "x2"]].isna().any(axis=1)
    by_hand = {str(k): float(v) for k, v in blank.groupby(train["race"].astype(str)).mean().items()}
    assert {g["group"]: g["missing_share"] for g in quality["groups"]} == pytest.approx(by_hand,
                                                                                     abs=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# (4) shrinkage by the calibration slope on unscaled predictors
# ═════════════════════════════════════════════════════════════════════════════


def _energy_table(n: int = 800, seed: int = 31) -> pd.DataFrame:
    """Energy in kcal and age in years, as recorded, and fiber in grams with a bend: the linear
    predictor without its intercept sits near 8, far from 0."""
    rng = np.random.default_rng(seed)
    kcal = rng.normal(2100, 500, n)
    age = rng.uniform(20, 80, n)
    fiber = rng.gamma(4, 4, n)
    eta = -9.0 + 0.0028 * kcal + 0.04 * age + 0.12 * (fiber - 16) - 0.004 * (fiber - 16) ** 2
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-eta)), "yes", "no")
    return pd.DataFrame({"pid": np.arange(n), "kcal": kcal, "age": age, "fiber_g": fiber, "y": y})


OFFSET_R = """
d <- read.csv(s_csv)
fit <- glm(y ~ 1 + offset(SLOPE * lp), family = binomial, data = d,
           control = glm.control(epsilon = 1e-14, maxit = 100))
out(list(a = unname(coef(fit)[1])))
"""


@needs_r
def test_4_the_shrunk_intercept_on_unscaled_predictors_is_r_glms(tmp_path):
    """Steyerberg (2019, §13.2): the coefficients × s and the intercept re-estimated with the shrunk
    linear predictor as an offset, ``glm(y ~ 1 + offset(s·lp), binomial)``. With energy in kcal and
    age in years the offset sits near 8, where Newton's method from 0 without step halving ran
    away (−8.4, 57.6, −∞) and was reported as separation; the intercept is R's now (to 10⁻⁸), far
    from 0."""
    frame = _energy_table()
    X = frame[["kcal", "age"]]
    y = (frame["y"] == "yes").astype(float).to_numpy()
    beta, s = np.array([0.0028, 0.04]) * 1.05, 0.92
    got = DC.shrinkage("binary", X, y, beta, s)
    lp = X.to_numpy() @ beta
    r = run_r(OFFSET_R.replace("SLOPE", repr(s)), {"s": pd.DataFrame({"y": y, "lp": lp})},
              tmp_path)
    assert got["intercept"] == pytest.approx(r["a"], abs=1e-8)
    assert r["a"] < -5
    assert [c["shrunk"] for c in got["coefficients"]] == pytest.approx(list(s * beta), abs=1e-15)


def test_4_a_shrinkage_that_cannot_be_estimated_is_said_never_a_false_separation():
    """The intercept-only likelihood with an offset has its maximum whenever both outcomes occur
    (it is concave and its score runs from the events' count to minus the others'), so the one
    failure left is an outcome with one value, said as such."""
    X = pd.DataFrame({"kcal": np.linspace(1500, 2500, 50)})
    with pytest.raises(ValueError) as caught:
        DC.shrinkage("binary", X, np.ones(50), [0.003], 0.9)
    assert str(caught.value) == ("The intercept could not be re-estimated: every row has the same "
                                 "outcome, so its maximum likelihood is infinite.")


@pytest.fixture(scope="module")
def shrunk(tmp_path_factory):
    """The evaluation with shrinkage declared on the energy table (a spline on fiber in grams set by
    hand), without and with an imbalance correction."""
    folder = tmp_path_factory.mktemp("shrunk")
    frame = _energy_table()
    frame.to_csv(folder / "e.csv", index=False)
    run = GraphRun(folder / "e.csv", folder / "p")
    roles = {"pid": "identifier", "kcal": "covariate", "age": "covariate", "fiber_g": "covariate"}
    base = d.ProjectState(
        lens=["clinical"], target="y", task="binary", event="yes", purpose="prediction",
        roles=roles, missing="complete_case",
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=5, folds=5), models=["linear"],
        exposure_forms={"fiber_g": d.ExposureFormSpec(form="spline", knots=4)},
        intended_use=d.IntendedUseSpec(use="decision_support"),
        updating=d.UpdatingSpec(method="shrinkage"))
    plain = run.run(base, upto=["evaluation"])
    corrected = run.run(base.model_copy(update={"levers": d.LeverSpec(imbalance="weights")}),
                        upto=["evaluation"])
    yield frame, plain, corrected, base
    run.close()


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """The model matrix by hand, in the served order: energy and age as recorded, fiber's
    restricted cubic spline (Harrell's basis written out, knots at his percentiles)."""
    fiber = frame["fiber_g"].to_numpy()
    basis = ref.rcs_columns(fiber, ref.harrell_knots(fiber, 4))
    columns = {"kcal": frame["kcal"].to_numpy(), "age": frame["age"].to_numpy(),
               "fiber_g": basis[:, 0], "fiber_g'": basis[:, 1], "fiber_g''": basis[:, 2]}
    return np.column_stack([columns[f] for f in features])


@needs_r
def test_4_through_the_evaluation_the_shrunk_model_is_r_glms_and_the_rest_stands(shrunk, tmp_path):
    """Through the evaluation stage, a yes/no outcome with energy and age unscaled and a spline on
    fiber in grams: the shrinkage is served (no error, no note), its intercept R's offset ``glm`` on
    the model matrix built here (to 10⁻⁸), its coefficients the served ones × the slope, and the
    benchmark, the interpretable model's cost and the decision curve stand beside it. With an
    imbalance correction the coefficients shrunk are the deployed (recalibrated) model's: NumPy's
    least squares of the deployed log-odds on the model matrix gives them exactly, and they are not
    the class-weighted fit's."""
    frame, plain, corrected, _ = shrunk
    for tag, out in (("plain", plain), ("corrected", corrected)):
        ev = out["evaluation"].data
        assert ev["benchmark"] is not None and ev["decision_curve"] is not None
        sh = ev["shrinkage"]
        assert "note" not in sh and sh["sentence"].startswith(
            f"As model updating, the regression's coefficients were multiplied by "
            f"{sh['factor']:.3f}, ")
        features = [c["feature"] for c in sh["coefficients"]]
        estimate = np.asarray([c["estimate"] for c in sh["coefficients"]])
        assert [c["shrunk"] for c in sh["coefficients"]] == pytest.approx(
            list(sh["factor"] * estimate), rel=1e-15)
        y = (frame["y"] == "yes").astype(float).to_numpy()
        lp = _matrix(frame, features) @ estimate
        r = run_r(OFFSET_R.replace("SLOPE", repr(sh["factor"])),
                  {"s": pd.DataFrame({"y": y, "lp": lp})}, tmp_path / tag)
        assert sh["intercept"] == pytest.approx(r["a"], abs=1e-8), tag
    final = corrected["fit"].objects["fitted"]["linear"]
    inputs = corrected["design"].objects["spec"]["inputs"]
    sh = corrected["evaluation"].data["shrinkage"]
    features = [c["feature"] for c in sh["coefficients"]]
    deployed = final.predict_proba(frame[inputs])[:, 1]
    A = np.column_stack([np.ones(len(frame)), _matrix(frame, features)])
    by_hand = np.linalg.lstsq(A, _logit(deployed), rcond=None)[0][1:]
    estimate = np.asarray([c["estimate"] for c in sh["coefficients"]])
    assert estimate == pytest.approx(by_hand, rel=1e-6, abs=1e-12)
    raw = np.asarray(final.steps[-1][1].estimator_.coef_, dtype=float).ravel()
    assert not np.allclose(estimate, raw, rtol=1e-3)


def test_4_a_shrinkage_that_fails_is_said_in_the_record_with_its_exit_and_the_stage_stands(
        shrunk, monkeypatch):
    """Should the shrunk intercept still fail to be estimated (the one case left is an outcome with
    one value, which the fit refuses first), the evaluation's shrinkage says so in the record, with
    the exit that answers the updating question "none" (which validates), and returns its offer
    rather than raising, so the stage goes on to compute what follows it (MODELING_SEQUENCE §4:
    never a stage error with no exit; the old false "separated" error took the whole stage down).
    Note, exit and sentence verbatim; nothing shrunk is served."""
    from turbotab.core.stages.evaluation import _shrinkage

    frame, plain, _, base = shrunk
    said = ("The intercept could not be re-estimated: every row has the same outcome, so its "
            "maximum likelihood is infinite.")

    def fails(*args, **kwargs):
        raise ValueError(said)

    monkeypatch.setattr(DC, "shrinkage", fails)
    fit = plain["fit"]
    X = frame[plain["design"].objects["spec"]["inputs"]]
    y = (frame["y"] == "yes").astype(int).to_numpy()
    offer = _shrinkage(base, "binary", fit.data, fit, X, y)
    slope = offer["factor"]
    assert offer["note"] == f"{said} The model was not updated."
    assert offer["sentence"] == (
        f"Shrinkage by the calibration slope ({slope:.3f}) was declared as model updating but not "
        f"applied, as the intercept could not be re-estimated: every row has the same outcome, so "
        f"its maximum likelihood is infinite.")
    assert offer["exit"] == {"label": "No model updating",
                             "decision": {"kind": "set_updating", "method": "none"}}
    assert "intercept" not in offer and "coefficients" not in offer
    answered = d.parse_decision(offer["exit"]["decision"])
    assert d.validate(answered, {"state": base, "columns": list(frame.columns)})
    assert d.fold_onto(base, answered).updating.method == "none"


# ═════════════════════════════════════════════════════════════════════════════
# (5) Riley's minimum counts the spline rule's columns
# ═════════════════════════════════════════════════════════════════════════════


def _riley_continuous(p: int, mean: float, sd: float, r2: float = 0.15) -> int:
    """Riley et al. (BMJ 2020;368:m441; Stat Med 2019;38:1276) for a continuous outcome, by hand, as
    ``pmsampsize`` states them: C2 = 234 + p; C3 the smallest n ≥ p + 2 whose expected shrinkage
    1 + (p − 2)/(n ln(1 − R²_app)), R²_app = (R²(n − p − 1) + p)/(n − 1), reaches 0.9; C4 =
    1 + p(1 − R²)/0.05; C1 the smallest n from there whose intercept interval, mean + t·√(SD²(1 −
    R²)/n), is within 1.1 times the mean. R² anticipated at 0.15."""
    from scipy import stats

    c2 = 234 + p
    n = p + 2
    while True:
        apparent = (r2 * (n - p - 1) + p) / (n - 1)
        if apparent < 1 and 1 + (p - 2) / (n * math.log(1 - apparent)) >= 0.9:
            break
        n += 1
    c3 = n
    c4 = math.ceil(1 + p * (1 - r2) / 0.05)
    n = max(c2, c3, c4)
    while (mean + stats.t.ppf(0.975, n - p - 1) * math.sqrt(sd * sd * (1 - r2) / n)) / mean > 1.1:
        n += 1
    return max(n, c2, c3, c4)


def _wide_regression(n: int = 800, seed: int = 41) -> pd.DataFrame:
    """Ten continuous predictors, a three-level category and a 0/1 flag; a positive outcome."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    frame["grp"] = rng.choice(["a", "b", "c"], n)
    frame["flag"] = rng.integers(0, 2, n)
    frame["y"] = 100 + X[:, :3] @ np.array([3.0, -2.0, 1.0]) + 2 * frame["flag"] + rng.normal(0, 8, n)
    frame.insert(0, "pid", np.arange(n))
    return frame


@pytest.fixture(scope="module")
def shelves(tmp_path_factory):
    """The shelf on the ten-predictor table with no lever, the spline rule, the inner
    cross-validated choice, and the rule beside a spline set by hand on ``x0``."""
    folder = tmp_path_factory.mktemp("shelves")
    frame = _wide_regression()
    frame.to_csv(folder / "w.csv", index=False)
    run = GraphRun(folder / "w.csv", folder / "p")
    roles = {"pid": "identifier", **{c: "covariate" for c in frame.columns
                                     if c not in ("pid", "y")}}
    base = d.ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="prediction", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=1, folds=5), models=["linear", "boosted_trees"])
    out = {}
    for tag, update in (
            ("none", {}), ("rule", {"levers": d.LeverSpec(forms="rule")}),
            ("inner_cv", {"levers": d.LeverSpec(forms="inner_cv")}),
            ("rule_and_hand", {"levers": d.LeverSpec(forms="rule"), "exposure_forms": {
                "x0": d.ExposureFormSpec(form="spline", knots=3)}})):
        out[tag] = run.run(base.model_copy(update=update), upto=["shelf"])["shelf"]
    yield frame, out
    run.close()


def test_5_rileys_minimum_counts_the_spline_rules_columns_among_the_candidate_parameters(shelves):
    """Riley et al. count "candidate predictor parameters", non-linear terms included. With the
    spline rule (or the inner cross-validated choice, whose candidates are the same splines) each
    of the ten continuous predictors is k − 1 = 4 columns (5 knots by Harrell's rule above an
    effective size of 100: here the 800 training rows), the three-level category 2 and the flag 1:
    43, not 13; the 800 rows meet the minimum for 13 and fall short of the one for 43. A spline set by hand is counted once, at its own k (3 knots: 2 columns), and not
    bent again by the rule. Each minimum is the criteria by hand, and the shelf's sentence and
    basis line say so."""
    frame, out = shelves
    y = frame["y"].to_numpy()
    mean, sd = float(y.mean()), float(y.std(ddof=1))
    assert ref.harrell_k(len(frame)) == 5
    for tag, parameters, terms in (("none", 13, 12), ("rule", 43, 42), ("inner_cv", 43, 42),
                                   ("rule_and_hand", 41, 40)):
        size = out[tag]["sample_size"]
        assert size["parameters"] == parameters, tag
        assert size["minimum"] == _riley_continuous(parameters, mean, sd), tag
        assert size["below"] is (len(frame) < size["minimum"])
        assert size["sentence"].startswith(
            f"Riley et al.'s minimum sample size for {parameters} candidate predictor parameters "
            f"is {size['minimum']:,} rows (the binding criterion: ")
        clause = (f" ({terms} model terms, with the spline and quintile columns)"
                  if terms != 12 else "")
        assert out[tag]["basis"] == f"Ranked for 800 training rows and 12 predictors{clause}."
    assert out["none"]["sample_size"]["below"] is False
    assert out["rule"]["sample_size"]["below"] is True
    # each rule's relation to Riley's count fires under prediction only (§13)
    from turbotab.core import contracts as C

    for key, option, name in (("spline_rule", "rule", "spline_rule_counted_by_riley"),
                              ("inner_cv_form", "inner_cv", "inner_cv_counted_by_riley")):
        assert [f.relation.name for f in C.fired({key: option}, "prediction",
                                                 ["candidate_parameters"])] == [name]
        assert C.fired({key: option}, "inference", ["candidate_parameters"]) == []
    order = [f["key"] for f in out["rule"]["families"]]
    assert order.index("linear") < order.index("boosted_trees")  # below it, the regression first
    assert out["rule"]["sample_size"]["sentence"].endswith(
        "fewer candidate parameters (fewer knots, outcome-blind data reduction) would lower the "
        "minimum.")


def test_5_a_yes_no_outcome_reads_the_rules_k_on_the_rarer_classs_count(tmp_path):
    """Harrell's effective size for a yes/no outcome is the rarer class's count (RMS §4.4), so with
    72 events among 600 rows the rule places 4 knots (3 columns per predictor): 10 continuous
    predictors are 30 candidate parameters, and the minimum is Riley's B criteria by hand on the
    training rows' outcome proportion."""
    from turbotab.core.tests.acceptance.test_explore_5_shelf import _riley_binary

    rng = np.random.default_rng(43)
    n = 600
    X = rng.normal(size=(n, 10))
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    frame["y"] = np.where(np.arange(n) < 72, "yes", "no")
    frame = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    frame.insert(0, "pid", np.arange(n))
    frame.to_csv(tmp_path / "b.csv", index=False)
    run = GraphRun(tmp_path / "b.csv", tmp_path / "p")
    state = d.ProjectState(
        lens=["clinical"], target="y", task="binary", event="yes", purpose="prediction",
        roles={"pid": "identifier", **{f"x{i}": "covariate" for i in range(10)}},
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=1, folds=5), models=["linear"],
        levers=d.LeverSpec(forms="rule"))
    size = run.run(state, upto=["shelf"])["shelf"]["sample_size"]
    assert ref.harrell_k(72) == 4
    assert size["parameters"] == 30
    assert size["minimum"] == _riley_binary(30, 72 / 600)
    run.close()


# ═════════════════════════════════════════════════════════════════════════════
# through the server: the leash's exits post, and nothing goes to status error
# ═════════════════════════════════════════════════════════════════════════════


def _open(client, path, truth, *, target: str, task: str, roles: dict[str, str], event=None):
    """The opening sequence under prediction, in the Router's order."""
    from turbotab.core.tests.acceptance.server_drive import open_project

    drive = open_project(client, path, truth)
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": target})
    if event is not None:
        drive.answer("event", {"kind": "set_event", "column": target, "level": event})
    drive.answer("task", {"kind": "set_task", "column": target, "task": task})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "prediction"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
    drive.reach("roles")
    drive.decide_roles(roles)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    return drive


def test_3_through_the_server_stepwise_at_p_much_greater_than_n_is_refused_and_its_exit_fits(
        tmp_path):
    """The verifier's case (50 rows, 70 predictors): ``set_selection`` stepwise was accepted and the
    fit then failed. Through the API it is refused now, with the fold's rows and the columns named
    (40 = ⌊50 · 4/5⌋), and its in-fold screening exit posts and the fit is computed."""
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.core.tests.truths import Truth

    rng = np.random.default_rng(51)
    n, p = 50, 70
    X = rng.normal(size=(n, p)).round(4)
    frame = pd.DataFrame(X, columns=[f"g{i:03d}" for i in range(p)])
    frame["y"] = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)).round(4)
    frame.insert(0, "pid", np.arange(1, n + 1))
    frame.to_csv(tmp_path / "wide.csv", index=False)
    roles = {"pid": "identifier", **{f"g{i:03d}": "covariate" for i in range(p)}}
    with local_server(tmp_path / "home") as client:
        drive = _open(client, tmp_path / "wide.csv",
                      Truth({"code_or_count:pid": "amount"}, fixture="the wide table"),
                      target="y", task="regression", roles=roles)
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        drive.artifact("cohort")
        r = drive.post({"kind": "set_selection", "method": "stepwise", "pre_selected": "no"})
        assert r.status_code == 409, r.text[:400]
        error = r.json()["error"]
        assert error["code"] == "stepwise_needs_rows"
        assert error["message"].startswith(
            f"Backward elimination starts from all {p} candidate columns, and the smallest "
            f"training fold has {_fold_rows(n, 0.0, 5)} rows")
        screening = next(e for e in error["exits"] if e["decision"]["method"] == "screening")
        assert drive.post(screening["decision"]).status_code == 200
        fit = drive.artifact("fit", timeout=600)  # computed, never an error
        assert [m["family"] for m in fit["models"]] == ["linear"]


def test_4_through_the_server_subgroups_ask_their_reading_and_shrinkage_never_takes_the_stage_down(
        tmp_path):
    """Through the API on the energy table (energy and age unscaled, a spline on fiber set by hand,
    a yes/no outcome) with a 12-category region code beside it: naming the region as a subgroup is
    refused until its reading is settled, the confirmation posts, and the answer then posts; with
    shrinkage declared the evaluation stage is computed (not an error), its shrinkage carries an
    intercept and no note, and the region is scored in its 12 levels, each pandas' count."""
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.core.tests.truths import Truth

    frame = _energy_table()
    frame["region"] = np.random.default_rng(9).integers(1, 13, len(frame))
    frame.to_csv(tmp_path / "energy.csv", index=False)
    roles = {"pid": "identifier", "kcal": "covariate", "age": "covariate", "fiber_g": "covariate",
             "region": "excluded"}
    truth = Truth({"code_or_count:pid": "amount", "unit:kcal": "kcal", "day_count:kcal": "1"},
                  fixture="the energy table")
    with local_server(tmp_path / "home") as client:
        # (the ingest reads yes/no as a boolean, so the event is its `True`)
        drive = _open(client, tmp_path / "energy.csv", truth, target="y", task="binary",
                      roles=roles, event="True")
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        assert drive.post({"kind": "set_exposure_form", "column": "fiber_g", "form": "spline",
                           "knots": 4}).status_code == 200
        use = {"kind": "set_intended_use", "use": "decision_support", "subgroups": ["region"]}
        r = drive.post(use)
        assert r.status_code == 409, r.text[:400]
        error = r.json()["error"]
        assert error["code"] == "reading_unsettled"
        code = next(e for e in error["exits"] if (e["decision"] or {}).get("value") == "code")
        assert drive.post(code["decision"]).status_code == 200
        assert drive.post(use).status_code == 200
        assert drive.post({"kind": "set_updating", "method": "shrinkage"}).status_code == 200
        ev = drive.artifact("evaluation", timeout=900)
    sh = ev["shrinkage"]
    assert sh["intercept"] is not None and "note" not in sh and sh["sentence"]
    [region] = ev["subgroups"]
    assert region["grouped_by"] == "levels"
    assert {g["group"]: g["n"] for g in region["groups"]} == _sizes(frame["region"].to_numpy(),
                                                                   "levels")
    assert ev["decision_curve"] is not None and ev["benchmark"] is not None


# ═════════════════════════════════════════════════════════════════════════════
# (8) under inference no cross-validated score is read, served or said
# ═════════════════════════════════════════════════════════════════════════════

INFERENCE_SPLIT = ("No rows were held out: every analyzed row estimates the coefficients, and no "
                   "cross-validated score is reported under inference: the declared model is "
                   "reported by its estimates.")


@pytest.fixture(scope="module")
def inferred(tmp_path_factory):
    """EXPORT's synthetic diet journey under inference (a marginal risk difference of protein),
    with a linear model and boosted trees and the explanations asked for; served and exported."""
    import time

    from turbotab.core.models.selection import read_seen
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.core.tests.acceptance.test_export import (_diet, _export, _files, _open_diet,
                                                            _wait_fresh)
    from turbotab.core.tests.truths import answer_adjustment

    folder = tmp_path_factory.mktemp("inferred")
    csv = folder / "diet.csv"
    _diet().to_csv(csv, index=False)
    with local_server(folder / "home") as client:
        drive = _open_diet(client, csv, "inference")
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        drive.decide({"kind": "set_estimand", "exposure": "protein_g", "effect": "total",
                      "contrast": "substitution", "measure": "risk_difference"})
        drive.reach("adjustment")
        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], drive.truth)
        end = time.monotonic() + 240
        while drive.artifact("proposals").get("model_sequence") is None:
            assert time.monotonic() < end
            time.sleep(0.1)
        drive.decide(drive.artifact("proposals")["model_sequence"]["decision"])
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                           "energy_column": "kcal", "nutrients": ["protein_g"]})
        drive.answer("models", {"kind": "select_models", "models": ["linear", "boosted_trees"]})
        drive.decide({"kind": "set_explain", "reseeds": 0})
        _wait_fresh(drive, ("cohort", "design", "fit", "effects", "explain"))
        drive.artifact("effects")  # the first estimates shown lock the plan
        seen = {"explain": drive.artifact("explain"), "view": drive.view(),
                "fit": drive.artifact("fit")}
        exported = _export(drive)
        assert exported.status_code == 200, exported.text[:600]
        seen["files"] = _files(exported.content)
        seen["seen"] = read_seen(client.app.state.service.workspace.project_dir(drive.pid))
    return seen


def test_8_under_inference_no_cross_validated_score_is_read_served_or_said(inferred):
    """MODELING_SEQUENCE ruling 13 and §1 row 11: under inference no cross-validated score is shown.
    The served explanation carries no floor for either family, draws the declared exposure's curve
    for both, and serving it counts no score as seen (nor does the fit); the split's record sentence
    says none is reported (verbatim); and no part of the export bundle says how a cross-validated
    score was estimated, how R² was pooled over out-of-fold predictions, or that a curve waited on
    a cross-validated score."""
    from turbotab.core.models.selection import explained_in, scored_in

    explain = inferred["explain"]
    assert {f["family"] for f in explain["families"]} == {"linear", "boosted_trees"}
    assert all(f["floor"] is None for f in explain["families"])
    assert explained_in(explain) == [] and scored_in(inferred["fit"]) == []
    assert explain["curves"] and all(k["drawn"] for c in explain["curves"] for k in c["curves"])
    assert "cross-validated" not in explain["methods"]
    assert not any(inferred["seen"].values())
    split = next(r for r in inferred["view"]["decisions"] if r["decision"]["kind"] == "set_split")
    assert split["sentence"] == INFERENCE_SPLIT
    files = inferred["files"]
    for name in ("methods.md", "methods.txt", "methods.json", "analysis_plan.json",
                 "provenance.json"):
        text = files[name].decode()
        assert INFERENCE_SPLIT in text, name
        for said in ("performance was estimated", "-fold cross-validation", "out-of-fold",
                     "cross-validated score beats"):
            assert said not in text, (name, said)

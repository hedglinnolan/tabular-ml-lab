"""EXPLORE 3 and 7 · The selection menu (MODELING_SEQUENCE §1 row 8; §0 ruling 1; §4 "Selection
outside the resampling"; TRIPOD+AI 9a) and, under inference, selection as a labeled sensitivity
analysis pooled by Rubin's rules.

Expected values: R (``MASS::stepAIC``, ``lm``/``glm`` p-values, ``anova`` and ``drop1``, and
``mice::pool`` for the pooled Wald tests across imputed copies), scikit-learn's PLS (VIP by its own
NIPALS), NumPy by hand (the in-fold screen, the inclusion count), and a simulation with a known truth
(stability selection's error control).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core import contracts as C
from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import Refusal
from turbotab.core.models import variable_selection as VS
from turbotab.core.tests.acceptance import explore_references as ref
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.graph_runner import GraphRun

PREDICTION = d.ProjectState(purpose="prediction", task="regression", target="y")
INFERENCE = d.ProjectState(purpose="inference", task="regression", target="y")


def _sparse(n: int = 200, p: int = 12, seed: int = 0, binary: bool = False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [1.0, -0.7, 0.5]
    lp = X @ beta
    y = ((rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float) if binary
         else lp + rng.normal(0, 1.0, n))
    return pd.DataFrame(X, columns=[f"v{i:02d}" for i in range(p)]), y


# ── the menu and its leash ───────────────────────────────────────────────────


def test_3_the_menu_is_ordered_by_soundness_with_the_customary_options_lower_and_unstable():
    """§1 row 8: none · elastic net · stability selection (ranked after the elastic net, labeled for
    a short reproducible panel) · in-fold screening at p ≫ n · customary options (stepwise,
    univariable screens, VIP > 1) ranked lower with their instability stated (Heinze et al. 2018)."""
    c = C.contract("variable_selection")
    order = [o["key"] for o in c.options_for("prediction")]
    assert order == ["none", "elastic_net", "stability", "screening", "stepwise", "univariable",
                     "vip"]
    by = {o["key"]: o for o in c.options_for("prediction")}
    assert by["stability"]["label"] == "Stability selection, for a short reproducible panel"
    assert "controls false selections, not prediction error" in by["stability"]["sound"]
    for key in ("stepwise", "univariable", "vip"):
        assert by[key]["rung"] == "rank_lower"
        assert by[key]["sound"] == ("Customary, unsound: its selected set changes from sample to "
                                    "sample (Heinze et al. 2018); the inclusion frequencies across "
                                    "folds and resamples show by how much")
    assert c.slot == "in_fold" and c.scope == "model"


def test_3_selection_outside_the_resampling_is_refused_with_the_exit_run_it_in_fold():
    """§4: "Selection outside the resampling: refuse (a false performance number)", with the exit
    the review names, "run it in-fold"."""
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetSelection(method="univariable", where="outside"), {"state": PREDICTION})
    r = caught.value
    assert r.code == "selection_outside_resampling"
    assert str(r) == (
        "Choosing predictors on every row before the resampling gives a false performance number: "
        "the scores would not include the optimism of the choice (Ambroise & McLachlan 2002). Run "
        "it in-fold, so every training fold repeats it.")
    assert r.exits[0]["label"] == "Run it in-fold"
    assert r.exits[0]["decision"]["where"] == "in_fold"
    assert r.exits[0]["decision"]["method"] == "univariable"
    ok = d.validate(r.exits[0]["decision"], {"state": PREDICTION})
    assert ok.where == "in_fold"


def test_3_tripod_ai_9a_asks_about_pre_selection_and_refuses_one_on_these_rows():
    """TRIPOD+AI 9a: "any pre-selection of predictors before model building". A pre-selection on
    these rows' outcome is selection outside the resampling: refused, its exits "run it in-fold"
    and "the pre-selection used other data"; the answer is said in the methods sentence."""
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetSelection(method="none", pre_selected="yes"), {"state": PREDICTION})
    assert caught.value.code == "pre_selected_on_these_rows"
    assert [e["label"] for e in caught.value.exits] == ["Run it in-fold",
                                                       "The pre-selection used other data"]
    assert caught.value.exits[0]["decision"]["method"] == "screening"
    assert caught.value.exits[0]["decision"]["pre_selected"] == "no"
    said = voice.sentence_for(d.SetSelection(method="screening", keep=20, pre_selected="no"))
    assert said == ("Within each training fold, the `20` predictors most correlated with the outcome "
                    "were kept, so every fold and resample repeated the selection (Ambroise & "
                    "McLachlan 2002); each predictor's inclusion frequency is reported. No predictor "
                    "was pre-selected on these rows' outcome (TRIPOD+AI 9a).")
    assert voice.sentence_for(d.SetSelection(method="none", pre_selected="other_data")) == (
        "No predictor selection was applied: every candidate predictor entered the models. "
        "Candidate predictors were pre-selected on other data (TRIPOD+AI 9a).")


def test_3_selection_reads_a_numeric_or_yes_no_outcome():
    ordinal = d.ProjectState(purpose="prediction", task="ordinal", target="y")
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetSelection(method="stepwise"), {"state": ordinal})
    assert caught.value.code == "selection_task"


# ── each method, on one training fold ────────────────────────────────────────


def test_3_in_fold_screening_matches_a_numpy_hand_computation_and_reads_only_the_fold():
    """Fan & Lv's screen: the m predictors with the largest |Pearson r| with the outcome on the
    training fold, by hand with ``numpy.corrcoef``; changing every other row changes nothing."""
    X, y = _sparse(160, 30, seed=2)
    fold = np.arange(len(y)) % 4 != 0
    step = VS.Selector("screening", "regression", keep=5).fit(X[fold], y[fold])
    r = np.array([abs(np.corrcoef(X[fold][c], y[fold])[0, 1]) for c in X.columns])
    want = sorted(X.columns[np.argsort(-r)[:5]])
    assert sorted(step.selected_) == want
    Xb = X.copy()
    Xb.loc[~fold] = 99.0
    again = VS.Selector("screening", "regression", keep=5).fit(Xb[fold], y[fold])
    assert again.selected_ == step.selected_
    assert VS.screening_keep(200, None) == int(200 / math.log(200))


@needs_r
@pytest.mark.parametrize("binary", [False, True], ids=["lm", "glm"])
def test_3_backward_elimination_by_aic_matches_r_mass_stepaic(binary, tmp_path):
    """The customary stepwise option, in-fold: backward elimination by AIC as
    ``MASS::stepAIC(direction = "backward")`` runs it on the same rows (its kept set)."""
    X, y = _sparse(220, 8, seed=3 if binary else 4, binary=binary)
    step = VS.Selector("stepwise", "binary" if binary else "regression").fit(X, y)
    family = "binomial" if binary else "gaussian"
    r = run_r(f"""
        suppressPackageStartupMessages(library(MASS))
        d <- read.csv(s_csv)
        full <- glm(y ~ ., family = {family}, data = d)
        kept <- attr(terms(stepAIC(full, direction = "backward", trace = 0)), "term.labels")
        out(list(kept = kept))
    """, {"s": X.assign(y=y)}, tmp_path)
    assert sorted(step.selected_) == sorted(r["kept"])


@needs_r
def test_3_the_univariable_screen_matches_r_p_values_for_single_and_multi_column_terms(tmp_path):
    """Hosmer–Lemeshow's univariable screen: one column, ``lm``'s t test (a numeric outcome) or
    ``glm``'s Wald z (a yes/no one); a term of several columns (a category's indicators), ``anova``'s
    F test or its likelihood-ratio test. To 10⁻⁸."""
    X, y = _sparse(240, 4, seed=5)
    Xb, yb = _sparse(240, 4, seed=6, binary=True)
    Z, _ = VS.standardized(X)
    Zb, _ = VS.standardized(Xb)
    groups = [[0], [1], [2, 3]]
    p_lin = VS.univariable_p(Z, y, "regression", groups)
    p_log = VS.univariable_p(Zb, yb, "binary", groups)
    r = run_r("""
        d <- read.csv(a_csv); b <- read.csv(b_csv)
        pl <- c(summary(lm(y ~ v00, d))$coefficients[2, 4], summary(lm(y ~ v01, d))$coefficients[2, 4],
                anova(lm(y ~ 1, d), lm(y ~ v02 + v03, d))$`Pr(>F)`[2])
        g <- function(f) glm(f, family = binomial, data = b,
                              control = glm.control(epsilon = 1e-14, maxit = 100))
        pb <- c(summary(g(y ~ v00))$coefficients[2, 4], summary(g(y ~ v01))$coefficients[2, 4])
        lr <- anova(g(y ~ 1), g(y ~ v02 + v03), test = "LRT")$`Pr(>Chi)`[2]
        out(list(pl = pl, pb = pb, lr = lr))
    """, {"a": X.assign(y=y), "b": Xb.assign(y=yb)}, tmp_path)
    assert np.max(np.abs(p_lin - r["pl"])) < 1e-8
    assert np.max(np.abs(p_log[:2] - r["pb"])) < 1e-8
    assert abs(p_log[2] - r["lr"]) < 1e-8


def test_3_vip_matches_scikit_learns_pls():
    """VIP > 1 (Wold's rule) over a two-component PLS: the app's NIPALS against scikit-learn's
    ``PLSRegression`` on the same standardized matrix, to 10⁻⁸."""
    X, y = _sparse(150, 10, seed=7)
    Z, _ = VS.standardized(X)
    assert np.max(np.abs(VS.pls_vip(Z, y, 2) - ref.vip_from_sklearn(Z, y, 2))) < 1e-8


def test_3_stability_selection_keeps_a_short_panel_and_controls_false_selections():
    """Shah & Samworth (2013), complementary pairs; Meinshausen & Bühlmann's bound q²/((2τ − 1)p) on
    the expected number of false selections. Simulation with a known truth: 3 true predictors among
    60, n = 200, 15 datasets, B = 25 pairs: every true predictor is kept in most datasets and the
    mean number of noise predictors kept is below the bound (q chosen so the bound is 1)."""
    p, tau = 60, VS.STABILITY_TAU
    q = VS.stability_q(p, tau)
    bound = VS.stability_bound(q, tau, p)
    assert q == math.floor(math.sqrt((2 * tau - 1) * p)) and bound <= 1.0
    false, true_hits = [], 0
    for seed in range(15):
        X, y = _sparse(200, p, seed=100 + seed)
        step = VS.Selector("stability", "regression", pairs=25, seed=seed).fit(X, y)
        kept = set(step.selected_)
        false.append(len(kept - {"v00", "v01", "v02"}))
        true_hits += len(kept & {"v00", "v01"})
        assert step.q_ == q and abs(step.bound_ - bound) < 1e-12
    assert np.mean(false) <= bound
    assert true_hits >= 0.9 * 2 * 15
    # the halves are complementary: disjoint and of ⌊n/2⌋ rows each
    rng = np.random.default_rng(0)
    order = rng.permutation(201)
    a, b = order[:100], order[100:200]
    assert not set(a) & set(b) and len(a) == len(b) == 100


def test_3_a_spline_or_a_category_is_kept_or_dropped_whole():
    """A term is the columns one input became: the selection keeps a spline's linear and nonlinear
    columns together and a category's indicators together."""
    X, y = _sparse(200, 4, seed=8)
    X["v00'"] = X["v00"] ** 2
    X["grp_2"] = (X["v03"] > 0).astype(float)
    X["grp_3"] = (X["v03"] > 1).astype(float)
    inputs = ["v00", "v01", "v02", "v03", "grp"]
    step = VS.Selector("screening", "regression", keep=2, inputs=inputs).fit(X, y)
    assert step.terms_["v00"] == ["v00", "v00'"] and step.terms_["grp"] == ["grp_2", "grp_3"]
    kept = set(step.kept_)
    for term, cols in step.terms_.items():
        assert set(cols) <= kept or not set(cols) & kept, term


# ── in-fold through the fit, with inclusion frequencies ─────────────────────


def test_3_selection_runs_in_every_fold_of_the_fit_and_its_inclusion_frequencies_are_counted(tmp_path):
    """The selection step sits in every family's pipeline (after construction, before the fit), so
    every fold of the comparison substrate refits it; the evaluation stage reports each predictor's
    inclusion frequency over the substrate's first 20 training folds. Reference: the screen redone
    by hand on each of those folds (top m by |r| over the fold's raw columns)."""
    X, y = _sparse(150, 10, seed=9)
    frame = X.assign(pid=np.arange(len(y)), y=y)
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    roles = {"pid": "identifier", **{c: "covariate" for c in X.columns}}
    state = d.ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="prediction", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=1, folds=5), models=["linear"],
        selection=d.SelectionSpec(method="screening", keep=3, pre_selected="no"))
    out = run.run(state, upto=["evaluation"])
    pipe = out["design"].objects["pipelines"]["linear"]
    assert [n for n, _ in pipe.steps][-2:] == ["select", "model"]
    inclusion = out["evaluation"].data["inclusion"]
    comparison = out["fit"].objects["comparison"]
    ids = comparison["train_ids"]
    rows = frame.set_index(frame.index.astype(np.int64)).loc[ids]
    counts = {c: 0 for c in X.columns}
    for _, fit_rows, _ in comparison["pairs"][:20]:
        part = rows[fit_rows]
        r = {c: abs(np.corrcoef(part[c], part["y"])[0, 1]) for c in X.columns}
        for c in sorted(r, key=lambda c: -r[c])[:3]:
            counts[c] += 1
    assert inclusion["folds"] == 20
    assert {e["column"]: e["kept"] for e in inclusion["columns"]} == counts
    run.close()


def test_3_the_selection_step_scope_is_the_outcome_model_and_it_is_fitted_in_fold():
    """Lockbox constitution §06's test (``contracts.observed_scope``): the selection reads the
    outcome, so its scope is ``model``, which the contract declares; it is never fitted before the
    seal."""
    X, y = _sparse(120, 8, seed=10)
    frame = X.copy()

    def fit_transform(f, reference, yy):
        step = VS.Selector("screening", "regression", keep=3).fit(f, yy)
        return pd.DataFrame({c: (f[c] if c in step.kept_ else 0.0) for c in f.columns})

    scope = C.observed_scope(fit_transform, frame, np.zeros(len(frame), dtype=bool), y, 5)
    assert scope == "model" == C.contract("variable_selection").scope


# ── under inference: a labeled sensitivity analysis only ─────────────────────


def test_7_under_inference_selection_is_only_a_labeled_sensitivity_analysis():
    """Ruling 1 (a, b): choosing what to report by significance is refused; stepwise selection is
    refused as the primary model; backward elimination by pooled Wald tests runs as a labeled
    sensitivity analysis beside the declared model."""
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetSelection(method="stepwise"), {"state": INFERENCE})
    assert caught.value.code == "selection_not_for_inference"
    exit_ = caught.value.exits[0]
    assert exit_["label"] == "Run it as a labeled sensitivity analysis"
    assert exit_["decision"]["sensitivity"] is True and exit_["decision"]["method"] == "stepwise"
    with pytest.raises(Refusal) as caught:
        d.validate(d.SetSelection(method="elastic_net", sensitivity=True), {"state": INFERENCE})
    assert caught.value.code == "sensitivity_is_backward_wald"
    assert d.validate(d.SetSelection(method="stepwise", sensitivity=True), {"state": INFERENCE})
    assert voice.sentence_for(d.SetSelection(method="stepwise", sensitivity=True)) == (
        "As a labeled sensitivity analysis, covariates were removed by backward elimination at "
        "α = `0.157`, each Wald test pooled across the imputed copies by Rubin's rules (Wood et al. "
        "2008), the exposure kept; the declared model is the reported one.")
    from turbotab.core.models.pipeline import _explore_answer

    state = INFERENCE.model_copy(update={"selection": d.SelectionSpec(method="stepwise",
                                                                     sensitivity=True)})
    assert _explore_answer(state, "selection") is None  # never a step of the declared model


def _copies(seed: int = 11, m: int = 5, n: int = 300):
    """Five completed copies of a table with x2 and x3 missing at random: each filled by a
    stochastic regression draw on the observed rows (an independent imputation, not the app's)."""
    rng = np.random.default_rng(seed)
    x1, x2, x3, x4 = (rng.normal(size=n) for _ in range(4))
    x2 = x2 + 0.5 * x1
    y = 0.6 * x1 + 0.4 * x2 + 0.05 * x3 + rng.normal(0, 1, n)
    miss2 = rng.random(n) < 0.2
    miss3 = rng.random(n) < 0.15
    copies = []
    for k in range(m):
        draw = np.random.default_rng(seed * 100 + k)
        filled = {}
        for name, col, miss in (("x2", x2, miss2), ("x3", x3, miss3)):
            A = np.column_stack([np.ones(n), x1, y])
            beta = np.linalg.lstsq(A[~miss], col[~miss], rcond=None)[0]
            sd = np.std(col[~miss] - A[~miss] @ beta, ddof=3)
            filled[name] = np.where(miss, A @ beta + draw.normal(0, sd, n), col)
        copies.append(pd.DataFrame({"x1": x1, "x2": filled["x2"], "x3": filled["x3"], "x4": x4,
                                    "y": y}))
    return copies


@needs_r
def test_7_backward_wald_pools_each_test_by_rubins_rules_as_r_mice_does(tmp_path):
    """Wood, White & Royston (2008): selection after multiple imputation tests each term by Rubin's
    rules across the copies. Each step: least squares in each copy; each unforced coefficient's
    pooled estimate over its total variance with Barnard & Rubin's df (complete-data df n − k); the
    term with the largest p above α = 0.157 leaves. Reference: the same elimination in R with
    ``summary(mice::pool(as.mira(fits)))``'s p-values, step by step, to 10⁻⁸."""
    copies = _copies()
    frames = [c[["x1", "x2", "x3", "x4"]] for c in copies]
    outcomes = [c["y"].to_numpy() for c in copies]
    terms = {c: [c] for c in ("x1", "x2", "x3", "x4")}
    got = VS.backward_wald("regression", frames, outcomes, terms, forced=["x1"])
    long = pd.concat([c.assign(imp=k + 1) for k, c in enumerate(copies)], ignore_index=True)
    r = run_r("""
        suppressPackageStartupMessages(library(mice))
        d <- read.csv(m_csv)
        terms <- c("x1", "x2", "x3", "x4"); forced <- "x1"; path <- list()
        repeat {
          f <- as.formula(paste("y ~", paste(terms, collapse = " + ")))
          fits <- lapply(split(d, d$imp), function(k) lm(f, data = k))
          s <- summary(pool(as.mira(fits)))
          p <- setNames(s$p.value, as.character(s$term))[setdiff(terms, forced)]
          path[[length(path) + 1]] <- as.list(p)
          worst <- names(which.max(p))
          if (length(p) == 0 || p[[worst]] <= 0.157) break
          terms <- setdiff(terms, worst)
        }
        out(list(path = path, kept = terms))
    """, {"m": long}, tmp_path)
    assert got["kept"] == r["kept"]
    assert len(got["path"]) == len(r["path"])
    for step, want in zip(got["path"], r["path"]):
        assert set(step["tested"]) == set(want)
        for term, p in want.items():
            assert abs(step["tested"][term] - p) < 1e-8, (step["step"], term)


def test_7_terms_are_grouped_by_the_input_each_column_came_from():
    assert VS.term_groups(["fat", "fat_sat", "fat_sat_adj", "race_2", "race_3", "x", "x'", "z"],
                          ["fat", "fat_sat", "race", "x"]) == {
        "fat": ["fat"], "fat_sat": ["fat_sat", "fat_sat_adj"], "race": ["race_2", "race_3"],
        "x": ["x", "x'"], "z": ["z"]}

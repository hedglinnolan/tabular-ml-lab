"""EXPLORE 5 and 6 · Riley's minimum before the shelf, the benchmark and the baseline always fitted,
and what the interpretable model costs or gains (MODELING_SEQUENCE §1 row 9; §5; TRIPOD+AI 10).

Expected values: Riley et al.'s criteria written out from the paper (B1, B3, B4; *BMJ* 2020;368:m441,
box 1, as ``pmsampsize`` computes them), least squares with Harrell's spline basis by hand on each
training fold, an independent replay of the BBC-CV draws over the flexible set, and simulations with
a known truth (an interaction a regression with splines cannot draw).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.models.selection import NO_COST, OutOfFold, interpretable_cost, shelf_order
from turbotab.core.tests.acceptance import explore_references as ref
from turbotab.core.tests.graph_runner import GraphRun


def _riley_binary(p: int, phi: float) -> int:
    """Riley et al. (BMJ 2020) box 1, by hand: B1 = (1.96/0.05)² φ(1 − φ); R²_CS anticipated at 15%
    of its maximum 1 − (φ^φ (1 − φ)^(1 − φ))²; B3 = p / ((S − 1) ln(1 − R²/S)) at S = 0.9; B4 the same
    at S = R² / (R² + 0.05·max)."""
    mx = 1 - (phi ** phi * (1 - phi) ** (1 - phi)) ** 2
    r2 = 0.15 * mx
    b1 = math.ceil((1.96 / 0.05) ** 2 * phi * (1 - phi))
    b3 = math.ceil(p / ((0.9 - 1) * math.log(1 - r2 / 0.9)))
    s4 = r2 / (r2 + 0.05 * mx)
    b4 = math.ceil(p / ((s4 - 1) * math.log(1 - r2 / s4)))
    return max(b1, b3, b4)


def _binary_table(n: int, p: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    lp = -0.6 + 0.7 * X[:, 0] - 0.5 * X[:, 1]
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    frame["pid"] = np.arange(n)
    frame["y"] = np.where(rng.random(n) < 1 / (1 + np.exp(-lp)), "yes", "no")
    return frame


def _state(frame: pd.DataFrame, task: str, models: list[str], folds: int = 5,
           **extra) -> d.ProjectState:
    roles = {"pid": "identifier", **{c: "covariate" for c in frame.columns
                                     if c not in ("pid", "y")}}
    return d.ProjectState(
        lens=["clinical"], target="y", task=task, purpose="prediction", roles=roles,
        event="yes" if task == "binary" else None, missing="complete_case",
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=1, folds=folds), models=models, **extra)


# ── Riley's minimum before the shelf ─────────────────────────────────────────


@pytest.mark.parametrize("n,below", [(250, True), (4000, False)])
def test_5_rileys_minimum_runs_before_the_shelf_and_below_it_the_regressions_rank_first(
        tmp_path, n, below):
    """TRIPOD+AI 10 ("justify that the study size was sufficient"); the review: "Compute Riley's
    minimum n … before the shelf … Below the minimum: regression families first, flexible learners
    ranked lower with the stated reason". Reference: the criteria by hand on the training rows'
    outcome proportion and the candidate parameters."""
    frame = _binary_table(n, 12, seed=n)
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    out = run.run(_state(frame, "binary", ["boosted_trees", "linear"]), upto=["shelf"])
    shelf = out["shelf"]
    size = shelf["sample_size"]
    ids = out["split"].frames["assignment"]
    train = ids[ids["partition"].astype(str).str.lower().isin(["train", "training"])]
    events = frame.set_index("pid").loc[train["row_id"].to_numpy(), "y"].eq("yes")
    phi = float(min(events.mean(), 1 - events.mean()))
    assert size["parameters"] == 12 and size["n_rows"] == len(train)
    assert size["minimum"] == _riley_binary(12, phi)
    assert size["below"] is below
    order = [f["key"] for f in shelf["families"]]
    flexible = next(f for f in shelf["families"] if f["key"] == "boosted_trees")
    # MC-1: a quiet name beside each concern, none yet (MC-5 writes them)
    assert all(f["terms"] == [None] * len(f["concerns"]) for f in shelf["families"])
    if below:
        # the flexible learners (C6a phase 2 added the forest and XGBoost) after every regression
        trees = {"boosted_trees", "random_forest", "xgboost"}
        assert min(order.index(k) for k in trees) > max(order.index(k) for k in order
                                                        if k not in trees)
        assert flexible["concerns"][0] == (
            f"{len(train):,} rows: below Riley et al.'s minimum sample size, a flexible learner may "
            f"need over 10 times as many events per variable as a regression to reach a stable "
            f"score (van der Ploeg, Austin & Steyerberg 2014), so the regression families rank "
            f"first.")
        assert size["sentence"].startswith(
            f"Riley et al.'s minimum sample size for 12 candidate predictor parameters is "
            f"{size['minimum']:,} rows (the binding criterion: ")
        assert "fall short of it" in size["sentence"]
    else:
        assert size["sentence"].endswith(f"these {len(train):,} rows meet it.")
        assert not any("Riley et al.'s minimum sample size, a flexible" in c
                       for c in flexible["concerns"])
    run.close()


def test_5_under_inference_the_shelf_is_ranked_by_the_estimand_not_by_riley():
    from turbotab.core.models import Situation, rank

    s = Situation(task="binary", purpose="inference", n_rows=200, n_features=10, n_events=60)
    ranked = rank(s)
    again, size = shelf_order(ranked, s)
    assert size is None and [f.key for f, _ in again] == [f.key for f, _ in ranked]


# ── the benchmark and the baseline, always fitted ────────────────────────────


def test_5_the_regression_with_splines_benchmark_and_the_baseline_are_always_fitted_on_the_same_folds(
        tmp_path):
    """The review: "Always fit the no-predictor baseline and a properly specified regression benchmark
    with splines". The benchmark is least squares with every continuous predictor a restricted cubic
    spline, k by Harrell's rule on each training fold (5 knots above 100 rows) at Harrell's
    percentiles, on the families' own folds. Reference: every fold of the comparison substrate (5
    folds drawn 10 times) refit here by hand, never through the app's pipeline: the benchmark by
    ``numpy.linalg.lstsq`` on the basis written out from RMS eq. 2.25, the baseline as each training
    fold's mean, the linear family as least squares on the raw predictors. The evaluation stage's
    served numbers are held to them (to 10⁻¹⁰): the benchmark's MSE (the mean of its folds'; the 260
    rows make five equal folds of 52, so pooled and averaged agree), its gain over the baseline (each
    fold's R² against the training fold's mean, which is 1 − the benchmark's MSE over the
    baseline's, the baseline's own R² there being 0) with Nadeau & Bengio's corrected interval, and
    its paired difference from the linear family on the same folds. (EXPLORE repair: the test used
    to compare its hand folds with a re-run of the app's own pipeline, never the served score.)"""
    rng = np.random.default_rng(12)
    n = 260
    X = rng.normal(size=(n, 2))
    y = 1 + np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)
    frame = pd.DataFrame(X, columns=["p1", "p2"]).assign(pid=np.arange(n), y=y)
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    out = run.run(_state(frame, "regression", ["linear"]), upto=["evaluation"])
    comparison = out["fit"].objects["comparison"]
    rows = frame.set_index(frame.index.astype(np.int64)).loc[comparison["train_ids"]]
    bench = out["evaluation"].data["benchmark"]
    assert bench["label"] == "Regression with splines (benchmark)"
    from turbotab.core.stages.evaluation import BENCHMARK

    mse_bench, mse_base, mse_linear, shares = [], [], [], []
    for _, fit_rows, test_rows in comparison["pairs"]:
        tr, te = rows[fit_rows], rows[test_rows]
        k = ref.harrell_k(len(tr))
        knots = {c: ref.harrell_knots(tr[c], k) for c in ("p1", "p2")}
        A_tr = np.column_stack([np.ones(len(tr))] + [ref.rcs_columns(tr[c].to_numpy(), knots[c])
                                                     for c in ("p1", "p2")])
        A_te = np.column_stack([np.ones(len(te))] + [ref.rcs_columns(te[c].to_numpy(), knots[c])
                                                     for c in ("p1", "p2")])
        beta = np.linalg.lstsq(A_tr, tr["y"].to_numpy(), rcond=None)[0]
        mse_bench.append(float(np.mean((te["y"].to_numpy() - A_te @ beta) ** 2)))
        mse_base.append(float(np.mean((te["y"].to_numpy() - tr["y"].mean()) ** 2)))
        L_tr = np.column_stack([np.ones(len(tr)), tr[["p1", "p2"]].to_numpy()])
        L_te = np.column_stack([np.ones(len(te)), te[["p1", "p2"]].to_numpy()])
        b = np.linalg.lstsq(L_tr, tr["y"].to_numpy(), rcond=None)[0]
        mse_linear.append(float(np.mean((te["y"].to_numpy() - L_te @ b) ** 2)))
        shares.append(len(te) / len(tr))
    mse_bench, mse_base, mse_linear = map(np.asarray, (mse_bench, mse_base, mse_linear))
    assert len(mse_bench) == 50 and set(shares) == {52 / 208}
    assert bench["metric"] == "mse"
    assert bench["estimate"] == pytest.approx(float(mse_bench.mean()), rel=1e-10, abs=0)
    # against the baseline: each fold's R² of the benchmark against the training fold's mean
    gain, se = ref.corrected_t(1.0 - mse_bench / mse_base, float(np.mean(shares)))
    vb = bench["versus_baseline"]
    assert vb["metric"] == "r2" and vb["verdict"] == "better"
    assert vb["gain"] == pytest.approx(gain, rel=1e-10, abs=0)
    assert vb["se"] == pytest.approx(se, rel=1e-10, abs=0)
    # against the linear family, paired on the same folds (a lower MSE favors the benchmark)
    differences = {(x["a"], x["b"]): x for x in bench["differences"]}
    linear = differences[(BENCHMARK, "linear")]
    mean, se = ref.corrected_t(mse_linear - mse_bench, float(np.mean(shares)))
    assert linear["folds"] == 50 and linear["repeats"] == 10
    assert linear["difference"] == pytest.approx(mean, rel=1e-10, abs=0)
    assert linear["se"] == pytest.approx(se, rel=1e-10, abs=0)
    assert linear["difference"] > 0  # the sine bends: the splines are ahead
    assert bench["sentence"] == (
        "The regression-with-splines benchmark (every continuous predictor a restricted cubic "
        "spline, knots by Harrell's rule, on the same in-fold preprocessing) scored cross-validated "
        f"MSE {bench['estimate']:.3f} on the families' 50 paired folds, better than the no-predictor "
        "baseline.")
    run.close()


# ── what the interpretable model costs or gains ──────────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_6_the_cost_or_gain_is_bbc_corrected_over_the_flexible_set(task):
    """The review: "the signed paired difference between the best inherently interpretable family and
    the best flexible family … Remove the selection optimism of picking the 'best flexible' family
    (BBC-CV over the flexible set)". Reference: the draws replayed by hand
    (``explore_references.interpretable_bbc``), the mean and percentile interval to 10⁻¹²."""
    rng = np.random.default_rng(13)
    n = 300
    if task == "regression":
        y = rng.normal(size=n)
        preds = {k: y + rng.normal(0, s, n) for k, s in (("spl", 0.9), ("gb1", 0.95), ("gb2", 1.0))}
        P = preds
    else:
        p = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < p).astype(float)
        raw = {k: np.clip(p + rng.normal(0, s, n), 0.02, 0.98)
               for k, s in (("spl", 0.05), ("gb1", 0.08), ("gb2", 0.1))}
        preds = raw
        P = {k: np.column_stack([1 - v, v]) for k, v in raw.items()}
    oof = OutOfFold(task, None, y, [(0, np.ones(n, bool), np.ones(n, bool))])
    oof.predictions.update(P)
    metric = "mse" if task == "regression" else "log_loss"
    got = interpretable_cost(task, metric, oof, "spl", ["gb1", "gb2"], replicates=300, seed=5)
    mean, low, high = ref.interpretable_bbc(y, preds, "spl", ["gb1", "gb2"], task=task, B=300,
                                            seed=5)
    assert abs(got["score"]["difference"] - mean) < 1e-12
    assert abs(got["score"]["ci_low"] - low) < 1e-12 and abs(got["score"]["ci_high"] - high) < 1e-12
    assert got["text"].startswith("What the interpretable model costs or gains: spl against the "
                                  "best of gb1, gb2, chosen anew in each of 300 resamples")
    assert got["calibration"] is not None


def test_6_an_interval_spanning_zero_says_no_measurable_cost_on_these_rows():
    rng = np.random.default_rng(14)
    n = 400
    y = rng.normal(size=n)
    noise = rng.normal(0, 1, n)
    oof = OutOfFold("regression", None, y, [(0, np.ones(n, bool), np.ones(n, bool))])
    oof.predictions.update({"spl": y + noise, "gb": y - noise})  # the same loss on every row
    got = interpretable_cost("regression", "mse", oof, "spl", ["gb"], replicates=200, seed=1)
    assert got["score"]["verdict"] == NO_COST == "no measurable cost on these rows"
    assert "no measurable cost on these rows" in got["text"]


def test_6_through_the_stage_an_interaction_a_spline_regression_cannot_draw_costs_it(tmp_path):
    """Simulation with a known truth: y = 2·sign(a·b) + ε, an interaction no additive regression with
    splines can draw and boosted trees can. The interpretable model costs on the proper score: the
    interval lies below 0. (On an additive truth the same comparison gains or costs nothing.)"""
    rng = np.random.default_rng(15)
    n = 360
    X = rng.normal(size=(n, 2))
    y = 2 * np.sign(X[:, 0] * X[:, 1]) + rng.normal(0, 0.5, n)
    frame = pd.DataFrame(X, columns=["p1", "p2"]).assign(pid=np.arange(n), y=y)
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    out = run.run(_state(frame, "regression", ["linear", "boosted_trees"], folds=3),
                  upto=["evaluation"])
    cost = out["evaluation"].data["interpretable"]
    assert cost["flexible"] == ["boosted_trees"] and cost["interpretable"] == "spline_benchmark"
    assert cost["score"]["ci_high"] < 0
    assert cost["score"]["verdict"] == "the interpretable model costs"
    run.close()


def test_5_at_p_much_greater_than_n_no_benchmark_is_fitted_and_the_record_says_why(tmp_path):
    """More candidate predictors than rows: least squares has no unique fit, with or without
    splines, so no benchmark is fitted, nothing is weighed against it, and the record says so."""
    rng = np.random.default_rng(16)
    n, p = 60, 80
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)
    frame = pd.DataFrame(X, columns=[f"g{i:03d}" for i in range(p)]).assign(pid=np.arange(n), y=y)
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    out = run.run(_state(frame, "regression", ["elastic_net"]), upto=["evaluation"])
    ev = out["evaluation"].data
    assert ev["benchmark"] is None and ev["interpretable"] is None
    assert ev["note"] == (f"{p} candidate predictors for {n} rows: a regression with splines on "
                          f"every predictor has no unique fit, so no benchmark is fitted and "
                          f"nothing is weighed against it.")
    run.close()

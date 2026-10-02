"""Tier A: the model-side stages (M1_CONTRACT §3, §7). These numbers go into papers.

Every check is against an independent computation: scikit-learn's own ``cross_validate`` on the
same folds and pipeline, metrics recomputed from the refit pipeline, statsmodels fit directly on
a matrix built by hand, the analytic substitution delta of a linear model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    make_scorer,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import PredefinedSplit, cross_validate

from turbotab.core.consequences import Lineage
from turbotab.core.jobs import Cancelled
from turbotab.core.methods.energy import EnergyAdjuster
from turbotab.core.models import families, get_family
from turbotab.core.models.artifacts import DesignArtifact, FitArtifact, ShelfArtifact, SubstitutionArtifact
from turbotab.core.models.steps import StratifiedEnergyAdjuster
from turbotab.core.stages.modeling import design_stage, fit_stage, shelf_stage, substitution_stage
from turbotab.core.tests import modeling_fixtures as mf

REPO = Path(__file__).resolve().parents[3]
PREDICTORS = [c for c, r in mf.NHANES_ROLES.items() if r in ("exposure", "covariate", "energy")]
FAMILIES = ["linear", "elastic_net", "boosted_trees"]


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    frame = mf.nhanes_like(400, seed=3)
    frame["glucose_high"] = np.where(frame["glucose"] > frame["glucose"].median(), "high", "normal")
    frame["glucose_band"] = pd.cut(frame["glucose"], 3, labels=["low", "mid", "top"]).astype(str)
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("nhanes_like"))
    return frame, paths


def run(st, paths, split, task="regression", target="glucose"):
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return design, fit


def raw_inputs(paths, columns, row_ids) -> pd.DataFrame:
    """The inputs as the parquet holds them, read without the stage's own reader."""
    frame = pd.read_parquet(paths["data"]).set_index("__row_id").loc[row_ids, columns]
    for c in frame.columns:
        if frame[c].dtype == bool:
            frame[c] = frame[c].astype(float)
    frame.index.name = "row_id"
    return frame


def train_arrays(split):
    a = split.frames["assignment"]
    train = a[a["partition"] == "train"]
    return train["row_id"].to_numpy(), train["fold"].to_numpy(), a[a["partition"] != "train"]["row_id"].to_numpy()


# ── cross-validation and holdout ─────────────────────────────────────────────

SCORING = {
    "regression": {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
    "binary": {
        "auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "brier": make_scorer(brier_score_loss, response_method="predict_proba",
                             greater_is_better=False, pos_label="normal"),
        "log_loss": "neg_log_loss",
    },
    "multiclass": {"accuracy": "accuracy", "macro_f1": "f1_macro", "log_loss": "neg_log_loss"},
}
NEGATED = {"rmse", "mae", "brier", "log_loss"}


class _AppFitted:
    """A family's pipeline fit as the fit stage fits it (its inner splits drawn by
    ``inner_cv.fit_pipeline``, which the acceptance tests check on their own), so that
    scikit-learn's ``cross_validate`` can run the outer loop and the scoring independently."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def get_params(self, deep=True):
        return {"pipeline": self.pipeline}

    def set_params(self, **params):
        self.pipeline = params.get("pipeline", self.pipeline)
        return self

    def fit(self, X, y):
        from turbotab.core.models.inner_cv import fit_pipeline

        self.fitted_ = fit_pipeline(clone(self.pipeline), X, y)
        if hasattr(self.fitted_, "classes_"):
            self.classes_ = self.fitted_.classes_
        return self

    def predict(self, X):
        return self.fitted_.predict(X)

    def predict_proba(self, X):
        return self.fitted_.predict_proba(X)


class _AppRegressor(_AppFitted):
    def __sklearn_tags__(self):
        from sklearn.base import BaseEstimator, RegressorMixin

        return type("R", (RegressorMixin, BaseEstimator), {})().__sklearn_tags__()


class _AppClassifier(_AppFitted):
    def __sklearn_tags__(self):
        from sklearn.base import BaseEstimator, ClassifierMixin

        return type("C", (ClassifierMixin, BaseEstimator), {})().__sklearn_tags__()


@pytest.mark.parametrize("task,target", [("regression", "glucose"), ("binary", "glucose_high"),
                                         ("multiclass", "glucose_band")])
def test_cv_metrics_equal_an_independent_cross_validate(table, task, target):
    """scikit-learn's ``cross_validate`` runs the folds and scores them; R² against each fold's
    training mean, and the pooled estimate (``metrics.py``), are recomputed here in numpy from the
    fold models' own predictions."""
    frame, paths = table
    # The analysis rows are whatever the split names; classification runs on fewer to stay quick.
    split = mf.split_bundle(np.arange(len(frame) if task == "regression" else 240), seed=4)
    st = mf.state(energy_adjustment=mf.energy("residual"), target=target)
    design, fit = run(st, paths, split, task, target)
    FitArtifact.model_validate(fit.data)
    train_ids, folds, _ = train_arrays(split)
    X = raw_inputs(paths, design.objects["spec"]["inputs"], train_ids)
    y = frame.loc[train_ids, target].to_numpy()
    by_family = {m["family"]: m for m in fit.data["models"]}
    wrapper = _AppRegressor if task == "regression" else _AppClassifier
    for key in FAMILIES:
        result = cross_validate(wrapper(design.objects["pipelines"][key]), X, y,
                                cv=PredefinedSplit(folds), scoring=SCORING[task],
                                return_estimator=True, return_indices=True)
        for metric in SCORING[task]:
            expected = result[f"test_{metric}"] * (-1 if metric in NEGATED else 1)
            got = by_family[key]["cv"][metric]
            if metric == "r2":  # against the training rows' mean, fold by fold
                expected = []
                for est, tr, te in zip(result["estimator"], result["indices"]["train"],
                                       result["indices"]["test"]):
                    pred = est.predict(X.iloc[te])
                    expected.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum())
                expected = np.asarray(expected)
            np.testing.assert_allclose(got["folds"], expected, rtol=0, atol=1e-9, err_msg=f"{key} {metric}")
            assert got["mean"] == pytest.approx(expected.mean(), abs=1e-9)
            assert got["sd"] == pytest.approx(expected.std(ddof=1), abs=1e-9)
            if task != "regression":
                assert got["estimator"] == "fold_mean" and got["estimate"] == pytest.approx(expected.mean(), abs=1e-9)
        if task == "regression":  # pooled over every out-of-fold prediction
            sse = sst = sae = 0.0
            for est, tr, te in zip(result["estimator"], result["indices"]["train"], result["indices"]["test"]):
                e = y[te] - est.predict(X.iloc[te])
                sse += (e ** 2).sum()
                sst += ((y[te] - y[tr].mean()) ** 2).sum()
                sae += np.abs(e).sum()
            cv = by_family[key]["cv"]
            assert {cv[m]["estimator"] for m in ("r2", "rmse", "mae")} == {"pooled"}
            assert cv["r2"]["estimate"] == pytest.approx(1 - sse / sst, abs=1e-9)
            assert cv["rmse"]["estimate"] == pytest.approx(np.sqrt(sse / len(y)), abs=1e-9)
            assert cv["mae"]["estimate"] == pytest.approx(sae / len(y), abs=1e-9)


def opened(fit):
    """The fit's data with the seal opened: the held-out scores live only in its sealed frame."""
    from turbotab.core.seal import SEALED_SCORES, scores_by_family, serve_fit

    assert all(m["holdout"] is None for m in fit.data["models"])  # never in the public data
    return serve_fit(fit.data, opened=True,
                     scores=lambda: scores_by_family(fit.frames[SEALED_SCORES]))


def test_holdout_metrics_equal_direct_computation_from_the_refit_pipeline(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), seed=5)
    st = mf.state(energy_adjustment=mf.energy("partition"))
    design, fit = run(st, paths, split)
    train_ids, _, hold_ids = train_arrays(split)
    assert fit.data["n_train"] == len(train_ids) and fit.data["n_holdout"] == len(hold_ids)
    assert fit.data["holdout_sealed"] is True
    columns = design.objects["spec"]["inputs"]
    X_train, X_hold = raw_inputs(paths, columns, train_ids), raw_inputs(paths, columns, hold_ids)
    y_train, y_hold = frame.loc[train_ids, "glucose"], frame.loc[hold_ids, "glucose"]
    for m in opened(fit)["models"]:
        refit = fit.objects["fitted"][m["family"]]
        pred = refit.predict(X_hold)
        # R² on the held-out rows is against the training rows' mean, not the held-out rows' own.
        r2_train_mean = 1 - ((y_hold - pred) ** 2).sum() / ((y_hold - y_train.mean()) ** 2).sum()
        assert m["holdout"]["r2"] == pytest.approx(r2_train_mean, abs=1e-12)
        assert m["holdout"]["r2"] != pytest.approx(r2_score(y_hold, pred), abs=1e-6)
        assert m["holdout"]["rmse"] == pytest.approx(root_mean_squared_error(y_hold, pred), abs=1e-12)
        assert m["holdout"]["mae"] == pytest.approx(mean_absolute_error(y_hold, pred), abs=1e-12)
        # The refit pipeline is the pipeline fit once on every training row, nothing else (its
        # inner splits drawn by fit_pipeline, as in every fold).
        from turbotab.core.models.inner_cv import fit_pipeline

        again = fit_pipeline(clone(design.objects["pipelines"][m["family"]]), X_train, y_train)
        np.testing.assert_allclose(again.predict(X_hold), pred, rtol=0, atol=1e-9)


def test_holdout_zero_means_cross_validation_only(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    design, fit = run(mf.state(energy_adjustment=mf.energy("none"), models=["linear"]), paths, split)
    assert fit.data["n_holdout"] == 0
    assert fit.data["models"][0]["holdout"] is None
    assert fit.data["holdout_sealed"] is False and not fit.frames  # nothing held out, nothing sealed


def test_binary_holdout_uses_the_second_class_as_positive(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), seed=6)
    st = mf.state(energy_adjustment=mf.energy("residual"), target="glucose_high", models=["linear"])
    design, fit = run(st, paths, split, "binary", "glucose_high")
    _, _, hold_ids = train_arrays(split)
    X_hold = raw_inputs(paths, design.objects["spec"]["inputs"], hold_ids)
    y_hold = frame.loc[hold_ids, "glucose_high"].to_numpy()
    refit = fit.objects["fitted"]["linear"]
    assert list(refit.classes_) == ["high", "normal"]
    proba = refit.predict_proba(X_hold)
    got = opened(fit)["models"][0]["holdout"]
    assert got["auc"] == pytest.approx(roc_auc_score(y_hold == "normal", proba[:, 1]), abs=1e-12)
    assert got["brier"] == pytest.approx(brier_score_loss(y_hold == "normal", proba[:, 1]), abs=1e-12)
    assert got["log_loss"] == pytest.approx(log_loss(y_hold, proba), abs=1e-12)


# ── leakage ──────────────────────────────────────────────────────────────────


def test_every_fitted_step_sees_only_training_fold_rows(tmp_path, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV, LinearRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import statsmodels.api as sm

    frame = mf.nhanes_like(300, seed=7, missing=True)
    paths = mf.ingest_frame(frame, tmp_path)
    frame = frame[frame["glucose"].notna()]
    split = mf.split_bundle(frame.index.to_numpy(), seed=7)
    train_ids, folds, hold_ids = train_arrays(split)
    allowed = {frozenset(train_ids[folds != k].tolist()) for k in np.unique(folds)}
    allowed.add(frozenset(train_ids.tolist()))
    seen: dict[str, list[frozenset]] = {}

    def spy(cls, name):
        original = cls.fit

        def fit(self, X, *args, **kwargs):
            seen.setdefault(name, []).append(frozenset(pd.Index(X.index).tolist()))
            return original(self, X, *args, **kwargs)

        monkeypatch.setattr(cls, "fit", fit)

    for cls, name in [(SimpleImputer, "imputer"), (EnergyAdjuster, "energy"), (OneHotEncoder, "onehot"),
                      (StandardScaler, "scaler"), (LinearRegression, "ols"), (ElasticNetCV, "enet"),
                      (HistGradientBoostingRegressor, "trees")]:
        spy(cls, name)
    ols_rows: list[frozenset] = []
    real_ols = sm.OLS

    def ols(endog, exog, *args, **kwargs):
        ols_rows.append(frozenset(exog.index.tolist()))
        return real_ols(endog, exog, *args, **kwargs)

    monkeypatch.setattr(sm, "OLS", ols)

    st = mf.state(energy_adjustment=mf.energy("residual"), missing="impute", purpose="inference")
    seen.clear()
    run(st, paths, split)
    design_rows = seen.pop("energy")[0]  # the design's shared steps: every training row, once
    assert design_rows == frozenset(train_ids.tolist())
    seen.clear()
    ols_rows.clear()
    design = design_stage(mf.context(st, {"split": split, "target_info": mf.target_info("regression")}, paths))
    seen.clear()
    fit_stage(mf.context(st, {"design": design, "split": split,
                              "target_info": mf.target_info("regression")}, paths))
    holdout = set(hold_ids.tolist())
    n_folds = len(np.unique(folds))
    for name in ("imputer", "energy", "onehot", "scaler", "ols", "enet", "trees"):
        rows = seen[name]
        assert rows, name
        for r in rows:
            assert r in allowed, f"{name} was fit on rows that are not one training fold"
            assert not (r & holdout), f"{name} saw held-out rows"
        # Every fold and the refit, each once per family that has the step.
        assert set(rows) == allowed, name
    assert len(seen["energy"]) == 3 * (n_folds + 1)
    assert ols_rows == [frozenset(train_ids.tolist())]


def test_the_fit_stops_between_folds_when_cancelled_and_reports_progress(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)))
    st = mf.state(energy_adjustment=mf.energy("none"), models=["linear", "boosted_trees"])
    ti = mf.target_info("regression")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    progress: list = []
    fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths, progress))
    messages = [m for _, m in progress]
    assert "Linear model: fold 1 of 5" in messages and "Boosted trees: fold 5 of 5" in messages
    assert [f for f, _ in progress] == sorted(f for f, _ in progress)
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(Cancelled):
        fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths,
                             cancelled=cancelled))


# ── coefficients ─────────────────────────────────────────────────────────────


def _by_feature(rows):
    return {r["feature"]: r for r in rows}


def test_statsmodels_coefficients_equal_a_direct_fit_on_a_matrix_built_by_hand(table):
    import statsmodels.api as sm

    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), seed=8)
    nutrients = ["protein", "carb", "fat_total"]
    roles = {"age": "covariate", "gender": "covariate", "kcal": "energy",
             **{n: "exposure" for n in nutrients}}
    st = mf.state(roles=roles, energy_adjustment=mf.energy("residual", nutrients),
                  purpose="inference", models=["linear"])
    _, fit = run(st, paths, split)
    train_ids, _, _ = train_arrays(split)
    d = frame.loc[train_ids]
    matrix = pd.DataFrame({"gender_male": (d["gender"] == "male").astype(float), "age": d["age"]})
    E = d["kcal"]
    for n in nutrients:  # N_adj = N − b (E − mean E), b from OLS of N on E over the training rows
        b = np.cov(d[n], E, ddof=1)[0, 1] / np.var(E, ddof=1)
        matrix[f"{n}_adj"] = d[n] - b * (E - E.mean())
    direct = sm.OLS(d["glucose"].to_numpy(), sm.add_constant(matrix)).fit()
    got = _by_feature(fit.data["models"][0]["coefficients"])
    ci = direct.conf_int(0.05)
    for name in matrix.columns:
        assert got[name]["estimate"] == pytest.approx(direct.params[name], rel=1e-8)
        assert got[name]["ci_low"] == pytest.approx(ci.loc[name, 0], rel=1e-8)
        assert got[name]["ci_high"] == pytest.approx(ci.loc[name, 1], rel=1e-8)
        assert got[name]["p"] == pytest.approx(direct.pvalues[name], rel=1e-6, abs=1e-300)
    assert got["(intercept)"]["estimate"] == pytest.approx(direct.params["const"], rel=1e-8)


def test_repeated_rows_get_cluster_robust_intervals_by_the_identifier(tmp_path):
    import statsmodels.api as sm

    frame = mf.nhanes_like(300, seed=9, repeats=2)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), groups=frame["SEQN"].to_numpy(), grouped_by="SEQN")
    roles = {"age": "covariate", "kcal": "energy", "fat_total": "exposure", "carb": "exposure"}
    st = mf.state(roles=roles, energy_adjustment=mf.energy("standard", ["fat_total", "carb"]),
                  purpose="inference", models=["linear"])
    _, fit = run(st, paths, split)
    train_ids, _, _ = train_arrays(split)
    d = frame.loc[train_ids]
    X = sm.add_constant(d[["age", "kcal", "fat_total", "carb"]])
    direct = sm.OLS(d["glucose"].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["SEQN"])[0]})
    model = fit.data["models"][0]
    got = _by_feature(model["coefficients"])
    ci = direct.conf_int(0.05)
    for name in ("age", "kcal", "fat_total", "carb"):
        assert got[name]["estimate"] == pytest.approx(direct.params[name], rel=1e-8)
        assert got[name]["ci_low"] == pytest.approx(ci.loc[name, 0], rel=1e-8)
        assert got[name]["ci_high"] == pytest.approx(ci.loc[name, 1], rel=1e-8)
    assert any("cluster-robust by SEQN" in c for c in model["concerns"])


def test_logistic_inference_coefficients_equal_a_direct_logit(table):
    import statsmodels.api as sm

    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), seed=10)
    roles = {"age": "covariate", "kcal": "energy", "fat_total": "exposure", "carb": "exposure"}
    st = mf.state(roles=roles, target="glucose_high", energy_adjustment=mf.energy("none"),
                  purpose="inference", models=["linear"])
    _, fit = run(st, paths, split, "binary", "glucose_high")
    train_ids, _, _ = train_arrays(split)
    d = frame.loc[train_ids]
    direct = sm.Logit((d["glucose_high"] == "normal").astype(float),
                      sm.add_constant(d[["age", "kcal", "fat_total", "carb"]])).fit(disp=0)
    got = _by_feature(fit.data["models"][0]["coefficients"])
    ci = direct.conf_int(0.05)
    for name in ("age", "kcal", "fat_total", "carb"):
        assert got[name]["estimate"] == pytest.approx(direct.params[name], rel=1e-6)
        assert got[name]["ci_low"] == pytest.approx(ci.loc[name, 0], rel=1e-6)
        assert got[name]["ci_high"] == pytest.approx(ci.loc[name, 1], rel=1e-6)
    # The unpenalized sklearn fit that makes the predictions agrees with the statsmodels table.
    model = fit.objects["fitted"]["linear"][-1]
    np.testing.assert_allclose(model.coef_[0], direct.params[["age", "kcal", "fat_total", "carb"]],
                               rtol=1e-4)


def test_elastic_net_coefficients_are_per_unit_with_the_scaling_undone(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), seed=11)
    design, fit = run(mf.state(energy_adjustment=mf.energy("none"), models=["elastic_net"]), paths, split)
    pipe = fit.objects["fitted"]["elastic_net"]
    assert [name for name, _ in pipe.steps][-2:] == ["scale", "model"]
    train_ids, _, _ = train_arrays(split)
    X = raw_inputs(paths, design.objects["spec"]["inputs"], train_ids)
    unscaled = pipe[:-2].transform(X)  # the matrix before standardizing, in the columns' own units
    rows = _by_feature(fit.data["models"][0]["coefficients"])
    by_hand = rows["(intercept)"]["estimate"] + unscaled.to_numpy() @ np.array(
        [rows[c]["estimate"] for c in unscaled.columns])
    np.testing.assert_allclose(by_hand, pipe.predict(X), rtol=0, atol=1e-8)
    assert all(r["ci_low"] is None and r["p"] is None for r in rows.values())


# ── lineage ──────────────────────────────────────────────────────────────────


def _lineage(design) -> Lineage:
    lineage = Lineage.model_validate(design.data["lineage"])
    return lineage


def _lane(lineage, lane):
    return {n.column: n for n in lineage.nodes if n.lane == lane}


def _links_into(lineage, node_id):
    return {(l.source, l.operation) for l in lineage.links if l.target == node_id}


NUTRIENTS3 = ["protein", "carb", "fat_total"]


@pytest.mark.parametrize("method", ["none", "standard", "residual", "density_multivariate", "density",
                                    "partition"])
def test_lineage_is_correct_for_each_energy_method_on_the_nhanes_columns(table, method):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)))
    st = mf.state(energy_adjustment=mf.energy(method, NUTRIENTS3))
    design = design_stage(mf.context(st, {"split": split, "target_info": mf.target_info("regression")}, paths))
    DesignArtifact.model_validate(design.data)
    lineage = _lineage(design)
    raw, adjusted, matrix = _lane(lineage, "raw"), _lane(lineage, "adjusted"), _lane(lineage, "matrix")
    assert set(raw) == set(PREDICTORS)
    assert raw["kcal"].role == "energy" and raw["protein"].role == "exposure"
    # gender is one-hot encoded with its first level as the reference, for every method
    assert "gender" not in matrix and "gender_male" in matrix
    assert _links_into(lineage, "mx:gender_male") == {("adj:gender", "one-hot")}
    untouched = [c for c in PREDICTORS if c not in (*NUTRIENTS3, "kcal", "gender")]
    for c in untouched:
        assert _links_into(lineage, f"adj:{c}") == {(f"raw:{c}", "kept")}
        assert adjusted[c].formula is None
    if method in ("none", "standard"):
        for c in (*NUTRIENTS3, "kcal"):
            assert _links_into(lineage, f"adj:{c}") == {(f"raw:{c}", "kept")}
            assert c in matrix
        return
    if method == "residual":
        outputs = {n: f"{n}_adj" for n in NUTRIENTS3}
    elif method.startswith("density"):
        outputs = {n: f"{n}_per_kcal" for n in NUTRIENTS3}
    else:
        outputs = {n: f"kcal_from_{n}" for n in NUTRIENTS3}
    op = {"residual": "energy-adjusted (residual)", "density": "energy-adjusted (density)",
          "density_multivariate": "energy-adjusted (density)",
          "partition": "energy-adjusted (partition)"}[method]
    for n, out in outputs.items():
        assert n not in adjusted and n not in matrix
        assert out in adjusted and out in matrix and adjusted[out].role == "exposure"
        assert adjusted[out].formula
        expected = {(f"raw:{n}", op)} if method == "partition" else {(f"raw:{n}", op), ("raw:kcal", op)}
        assert _links_into(lineage, f"adj:{out}") == expected
    if method == "residual":
        assert "kcal" not in adjusted and "kcal" not in matrix
        assert "OLS on" in adjusted["protein_adj"].formula
    elif method == "density":
        assert "kcal" not in matrix
    elif method == "density_multivariate":
        assert _links_into(lineage, "adj:kcal") == {("raw:kcal", "kept")} and "kcal" in matrix
    else:
        assert "kcal" not in matrix and "kcal_from_other" in matrix
        assert _links_into(lineage, "adj:kcal_from_other") == {
            (f"raw:{c}", op) for c in ("kcal", *NUTRIENTS3)}
        assert adjusted["kcal_from_other"].role == "energy"
    assert design.data["matrix"]["n_cols"] == len(matrix)
    assert design.data["estimand"]


def test_imputed_columns_are_named_in_the_lineage_and_only_they(tmp_path):
    frame = mf.nhanes_like(300, seed=12, missing=True)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)))
    st = mf.state(energy_adjustment=mf.energy("residual"), missing="impute")
    design = design_stage(mf.context(st, {"split": split, "target_info": mf.target_info("regression")}, paths))
    lineage = _lineage(design)
    assert _links_into(lineage, "adj:bmi") == {("raw:bmi", "imputed")}
    assert _links_into(lineage, "adj:age") == {("raw:age", "kept")}
    assert ("raw:fat_total", "imputed, energy-adjusted (residual)") in _links_into(lineage, "adj:fat_total_adj")
    steps = [s["key"] for s in design.data["models"][0]["steps"]]
    assert steps == ["impute", "energy", "onehot", "model"]
    assert [s["key"] for s in design.data["models"][1]["steps"]] == ["impute", "energy", "onehot", "scale", "model"]


def test_missing_values_without_a_strategy_stop_the_design_with_a_reason(tmp_path):
    frame = mf.nhanes_like(200, seed=13, missing=True)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)))
    ctx = mf.context(mf.state(energy_adjustment=mf.energy("none"), missing=None),
                     {"split": split, "target_info": mf.target_info("regression")}, paths)
    with pytest.raises(ValueError, match="no missing-values strategy"):
        design_stage(ctx)
    # Boosted trees use missing values as they are, so they alone may go ahead.
    ctx.state = mf.state(energy_adjustment=mf.energy("none"), missing=None, models=["boosted_trees"])
    design_stage(ctx)


def test_a_wide_table_collapses_into_count_nodes_that_still_add_up(tmp_path):
    rng = np.random.default_rng(14)
    n, genes = 80, 120
    frame = pd.DataFrame(rng.normal(size=(n, genes)), columns=[f"gene_{i:04d}" for i in range(genes)])
    frame["age"] = rng.integers(20, 80, n).astype(float)
    frame["sex"] = rng.choice(["F", "M"], n)
    frame["y"] = frame["gene_0001"] + rng.normal(size=n)
    paths = mf.ingest_frame(frame, tmp_path)
    roles = {**{f"gene_{i:04d}": "exposure" for i in range(genes)}, "age": "covariate", "sex": "covariate"}
    st = mf.state(roles=roles, target="y", energy_adjustment=None, models=["elastic_net"])
    design = design_stage(mf.context(st, {"split": mf.split_bundle(np.arange(n)),
                                          "target_info": mf.target_info("regression", "y")}, paths))
    lineage = _lineage(design)
    assert lineage.collapsed
    for lane, total in (("raw", genes + 2), ("matrix", genes + 2)):
        assert sum(node.count for node in lineage.nodes if node.lane == lane) == total
    # sex is one-hot encoded, so it stays visible; the genes fold into one count node per lane.
    visible = {node.column for node in lineage.nodes if node.column}
    assert {"sex", "sex_M"} <= visible
    groups = [node for node in lineage.nodes if node.column is None]
    assert any(g.lane == "raw" and g.role == "exposure" and g.count == genes for g in groups)
    assert len(lineage.nodes) < 12


# ── substitution ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["none", "standard", "residual"])
def test_a_linear_model_on_kcal_scale_inputs_gives_the_analytic_substitution_delta(tmp_path, method):
    rng = np.random.default_rng(15)
    n = 500
    total = rng.normal(2100, 400, n).clip(900)
    fat, carb = total * rng.uniform(0.25, 0.4, n), total * rng.uniform(0.4, 0.55, n)
    frame = pd.DataFrame({"energy_kcal": total, "fat_kcal": fat, "carb_kcal": carb,
                          "age": rng.integers(20, 80, n).astype(float)})
    frame["glucose"] = 80 + 0.01 * carb - 0.02 * fat + 0.1 * frame["age"] + rng.normal(0, 5, n)
    paths = mf.ingest_frame(frame, tmp_path)
    roles = {"energy_kcal": "energy", "fat_kcal": "exposure", "carb_kcal": "exposure", "age": "covariate"}
    adj = None if method == "none" else mf.EnergyAdjustment(
        method=method, energy_column="energy_kcal", nutrients=["fat_kcal", "carb_kcal"])
    st = mf.state(roles=roles, energy_adjustment=adj, models=["linear"],
                  substitution=mf.substitution("fat_kcal", "carb_kcal", 50.0))
    split = mf.split_bundle(np.arange(n))
    design, fit = run(st, paths, split)
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    SubstitutionArtifact.model_validate(sub)
    model = fit.objects["fitted"]["linear"][-1]
    beta = dict(zip(model.feature_names_in_, model.coef_))
    out = "_adj" if method == "residual" else ""
    gap = beta[f"carb_kcal{out}"] - beta[f"fat_kcal{out}"]
    curve = sub["models"][0]
    for k, delta in zip(sub["ks"], curve["delta"]):
        if delta is not None:
            assert delta == pytest.approx(k * gap, abs=1e-9)
    assert sub["ks"] == [50.0 * i for i in range(11)]
    assert sub["basis"] == f"Averaged over {int((split.frames['assignment']['partition'] == 'train').sum())} training rows."


def test_substitution_names_its_assumptions_and_skips_nothing_it_can_draw(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)))
    st = mf.state(energy_adjustment=mf.energy("residual"), substitution=mf.substitution())
    design, fit = run(st, paths, split)
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    assert [m["family"] for m in sub["models"]] == FAMILIES
    assert "fat_total is read as fat in grams at 9 kcal/g" in sub["note"]
    assert "Total energy was held fixed by assumption" in sub["note"]
    # No band unless one is asked for (M1_CONTRACT §12.7); its cost is measured and offered.
    assert all(m["ci_low"] is None and m["ci_high"] is None for m in sub["models"])
    assert sub["band"] is None and sub["band_estimate"]["n_boot"] == 50
    assert sub["band_estimate"]["seconds"] > 0
    assert "fat_sat, fat_mon and fat_poly are parts of fat_total" in sub["note"]
    assert set(sub["carried"]) == {"fat_sat", "fat_mon", "fat_poly", "sugar"}
    assert {"donor": "fat_total", "recipient": "carb"} in design.data["substitution_pairs"]
    # Sugar carries energy (carbohydrate, 4 kcal/g) and is part of carb: never paired with it.
    assert not any({p["donor"], p["recipient"]} == {"sugar", "carb"}
                   for p in design.data["substitution_pairs"])


# ── stratified residuals ─────────────────────────────────────────────────────


def test_stratified_residuals_equal_one_residual_fit_per_level():
    frame = mf.nhanes_like(400, seed=16)
    X = frame[["gender", "kcal", "protein", "carb", "age"]]
    step = StratifiedEnergyAdjuster("kcal", ["protein", "carb"], strata="gender").fit(X)
    out = step.transform(X)
    for level in ("female", "male"):
        rows = X[X["gender"] == level]
        alone = EnergyAdjuster("residual", "kcal", ["protein", "carb"]).fit(rows).transform(rows)
        for n in ("protein_adj", "carb_adj"):
            np.testing.assert_allclose(out.loc[rows.index, n], alone[n], rtol=0, atol=1e-10)
            assert abs(np.corrcoef(out.loc[rows.index, n], rows["kcal"])[0, 1]) < 1e-8
    assert "within each level of gender" in next(e for e in step.lineage() if e["output"] == "protein_adj")["formula"]
    dropped = StratifiedEnergyAdjuster("kcal", ["protein"], strata="gender", drop_strata=True).fit(X)
    assert "gender" not in dropped.transform(X).columns
    assert "gender" not in dropped.get_feature_names_out()


def test_a_stratified_design_takes_the_strata_column_even_when_it_is_not_a_predictor(table):
    frame, paths = table
    roles = {**mf.NHANES_ROLES, "gender": "excluded"}
    st = mf.state(roles=roles, energy_adjustment=mf.energy("residual", strata="gender"))
    split = mf.split_bundle(np.arange(len(frame)))
    design = design_stage(mf.context(st, {"split": split, "target_info": mf.target_info("regression")}, paths))
    lineage = _lineage(design)
    assert "gender" in _lane(lineage, "raw") and "gender" not in _lane(lineage, "matrix")
    assert ("raw:gender", "energy-adjusted (residual)") in _links_into(lineage, "adj:protein_adj")
    assert "within each level of gender" in design.data["models"][0]["steps"][0]["detail"]


# ── the shelf ────────────────────────────────────────────────────────────────


def test_the_shelf_puts_regularized_families_first_when_predictors_outnumber_rows(tmp_path):
    source = REPO / "turbotab" / "sample_data" / "genomics_expression.csv"
    frame = pd.read_csv(source)
    paths = mf.ingest_frame(frame, tmp_path)
    genes = [c for c in frame.columns if c.startswith("gene_")]
    roles = {**{g: "exposure" for g in genes}, "age": "covariate", "sex": "covariate",
             "sample_id": "identifier", "batch": "design"}
    st = mf.state(roles=roles, target="condition", lens=["genomics"], purpose="prediction")
    cohort = mf.cohort_bundle(np.arange(len(frame)), [*genes, "age", "sex"])
    shelf = shelf_stage(mf.context(st, {"cohort": cohort, "target_info": mf.target_info("binary", "condition")},
                                   paths))
    ShelfArtifact.model_validate(shelf)
    keys = [f["key"] for f in shelf["families"]]
    assert keys[0] == "elastic_net"
    assert set(keys) == {f.key for f in families("binary")}  # the shelf is never shortened
    trees = next(f for f in shelf["families"] if f["key"] == "boosted_trees")
    assert trees["fit"] == "poor"
    assert any("60 rows is small for boosted trees" in c for c in trees["concerns"])
    linear = next(f for f in shelf["families"] if f["key"] == "linear")
    assert linear["fit"] == "poor" and linear["concerns"]
    assert [f["rank"] for f in shelf["families"]] == [1, 2, 3]


def test_the_shelf_leads_with_the_linear_model_for_inference_and_trees_for_large_prediction(table):
    frame, paths = table
    cohort = mf.cohort_bundle(np.arange(len(frame)), PREDICTORS)
    ti = mf.target_info("regression")
    inference = shelf_stage(mf.context(mf.state(purpose="inference"), {"cohort": cohort, "target_info": ti}, paths))
    assert inference["families"][0]["key"] == "linear"
    big = mf.cohort_bundle(np.arange(len(frame)), PREDICTORS)
    big.data["n_final"] = 17_000
    prediction = shelf_stage(mf.context(mf.state(purpose="prediction"), {"cohort": big, "target_info": ti}, paths))
    assert prediction["families"][0]["key"] == "boosted_trees"


def test_every_family_declares_what_the_shelf_and_the_pipeline_need():
    for family in families():
        assert len(family.inductive_bias.split()) <= 20
        assert family.strengths and family.cautions
        for task in family.tasks:
            assert family.build(task, "prediction", 100, 5) is not None
    assert get_family("boosted_trees").handles_missing and not get_family("boosted_trees").needs_scaling
    assert get_family("elastic_net").needs_scaling


def test_select_models_refuses_an_unknown_family_and_offers_the_known_ones():
    from turbotab.core.decisions import Refusal, validate

    with pytest.raises(Refusal) as refused:
        validate({"kind": "select_models", "models": ["linear", "random_forest"]}, {"task": "regression"})
    assert refused.value.code == "unknown_model"
    assert refused.value.exits[0]["decision"]["models"] == ["linear"]
    validate({"kind": "select_models", "models": ["boosted_trees"]}, {"task": "multiclass"})

"""W1-C · the drivers press Fit, and a light NHANES Predict walk matches independent references.

The reference journeys (``turbotab.core.reference.journeys``) and the M3 fixture capture follow the
Router question by question. Since Fit became a press (SIZING P0.8), the Router's last question
under Predict, the seal, waits for it (``interview.AFTER_FIT``), and a fit estimated over about two
minutes waits for it in the scheduler (RECIPES §4.4): a driver that never presses Fit waits until
it runs out of time. The drivers now press it once the Router reaches it (``journeys.FitPresser``).

The light walk: the NHANES export (fasting glucose) under Predict, seven predictors, every row with
glucose recorded, a random 20% held out, the linear model and ridge, the linear model declared final
and the seal opened, all through the real server in process (``server_drive.local_server``, as the
acceptance drives run). Its held-out numbers are then checked against paths that share no code with
the engine:

* the split rebuilt from the seal's held-out row ids (the split stage's sealed frame);
* scikit-learn's ``LinearRegression`` refit on the training rows and the recorded predictors (the
  cohort's), one-hot coded by pandas: the held-out RMSE, MAE and R², R² measured against the
  training rows' mean as the split's record says it is;
* the no-predictor baseline the engine serves, ``FittedModel.baseline`` (M1_CONTRACT §12.6): the
  cross-validated MSE of the training-fold mean on the split's five folds, the folds drawn by
  scikit-learn's ``KFold`` at the split's seed over the training rows and the score by numpy, with
  scikit-learn's out-of-fold least squares on the same folds against the linear family's
  cross-validated MSE (so the folds are the engine's), and the gain over the baseline on the
  comparison substrate (ten draws of the five folds; each draw's pooled out-of-fold R² against the
  training-fold mean, averaged) against numpy;
* the no-predictor baseline's held-out RMSE by numpy (the held-out y around the training mean): the
  engine serves no held-out baseline yet (the baseline above is cross-validated only), so it is
  checked against the engine's once served, and a strict expected failure says it is not;
* the held-out calibration (Van Calster et al. 2016, on the outcome's scale as
  ``models.performance`` defines it): the slope is statsmodels' OLS slope of y on ŷ, the intercept
  (calibration-in-the-large) statsmodels' OLS of y − ŷ on a constant, each with its standard error,
  and the observed and expected means by numpy. The engine's intercept is calibration-in-the-large,
  not the intercept of the y-on-ŷ regression the plan's acceptance 2 names (a ruling is asked for).

And the binding rule the drivers rely on: when the Router reaches Fit, no estimate stage serves a
score (none computed yet, or served withheld for Fit with every score removed); only the press
serves them.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from turbotab.core.reference import journeys as J

TOL = 1e-8
PREDICTORS = ("age", "gender", "bmi", "waist", "bp_sys", "hdl", "triglycerides")


# ── when the Router has reached Fit ──────────────────────────────────────────


def _view(first: dict[str, Any] | None, fit: dict[str, Any] | None = None,
          **stages: dict[str, Any]) -> dict[str, Any]:
    interview = [{"key": "models", "status": "answered", "waiting_on": []}]
    if first is not None:
        interview.append(first)
    return {"interview": interview, "decisions": [{}] * 3,
            "stages": {"fit": fit or {"status": "fresh", "key": "k"}, **stages}}


def test_the_router_reaches_fit_when_the_seal_waits_for_it_or_a_stage_is_held_for_it():
    seal = {"key": "open_seal", "status": "waiting", "waiting_on": ["fit"]}
    # The seal waits for Fit while the fit is done: Results wait for the press.
    assert J.reaches_fit(_view(seal))
    # A fit the scheduler holds for Fit (its estimate exceeds about 2 minutes), wherever the
    # Router stands.
    assert J.reaches_fit(_view(None, {"status": "idle", "held": "fit"}))
    assert J.reaches_fit(_view(None, None, effects={"status": "idle", "held": "fit"}))
    # Not yet: the fit is computing, or failed (the step opens and says why), or was stopped, or
    # the seal waits on something else, or nothing waits on the fit.
    assert not J.reaches_fit(_view(seal, {"status": "running", "key": "k"}))
    assert not J.reaches_fit(_view(seal, {"status": "queued", "key": "k"}))
    assert not J.reaches_fit(_view(seal, {"status": "error", "key": "k"}))
    assert not J.reaches_fit(_view(seal, {"status": "stale", "key": "k", "cancelled": True}))
    assert not J.reaches_fit(_view({"key": "open_seal", "status": "waiting",
                                    "waiting_on": ["shelf"]}))
    assert not J.reaches_fit(_view({"key": "models", "status": "open", "waiting_on": []}))
    assert not J.reaches_fit(_view(None, {"status": "idle", "held": "estimate"}))


def test_fit_is_pressed_once_for_each_state_of_the_log_and_the_fit():
    class Client:
        def __init__(self) -> None:
            self.posts: list[str] = []

        def post(self, url: str, json: Any = None) -> Any:  # noqa: A002
            self.posts.append(url)

            class R:
                status_code = 200

                @staticmethod
                def json() -> dict[str, Any]:
                    return {}
            return R()

    c, notes = Client(), []
    press = J.FitPresser(c, "p1", notes.append)
    seal = {"key": "open_seal", "status": "waiting", "waiting_on": ["fit"]}
    assert press(_view(seal)) and not press(_view(seal))  # once, not again for the same state
    assert c.posts == ["/api/projects/p1/fit"] and notes == ["the Router reached Fit: pressed"]
    assert press(_view(seal, {"status": "fresh", "key": "another"}))  # the fit changed
    assert not press(_view({"key": "models", "status": "open", "waiting_on": []}))
    assert len(c.posts) == 2


# ── the light NHANES Predict walk ────────────────────────────────────────────


def _spec(columns: list[str]) -> Any:
    roles = {c: ("identifier" if c == "SEQN" else "covariate" if c in PREDICTORS else "excluded")
             for c in columns if c != "glucose"}
    return J.Journey(
        "w1-light-predict", "clinical", "prediction",
        "How well do age, sex and body measures predict fasting glucose?",
        J.NHANES_FILE, J.NHANES_WHY, J.nhanes_path,
        lambda: J.truth(J.nhanes_readings(), fixture=J.NHANES_KEY),
        target="glucose", lenses=("clinical",), roles=roles,
        answers={"exclusions": {"kind": "set_exclusions", "rules": []},
                 "missing": {"kind": "set_missing", "strategy": "complete_case"},
                 "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
                 "models": {"kind": "select_models", "models": ["linear", "ridge"]},
                 "open_seal": {"kind": "open_seal", "family": "linear"}},
        # On a driver that never presses Fit the seal waits for good: it stops here instead.
        timeout=300.0)


# Every score the engine serves on a fitted model: none may be served before Fit is pressed.
SCORES = ("holdout", "holdout_detail", "versus_baseline", "calibration", "compared_on",
          "performance", "optimism", "nested_cv")


def _watching(seen: list[dict[str, Any]]) -> type[J.FitPresser]:
    """The drivers' FitPresser, reading every estimate stage as served just before each press."""
    from turbotab.core.estimand import ESTIMATE_STAGES

    class Watching(J.FitPresser):
        def __call__(self, view: dict[str, Any]) -> bool:
            if J.reaches_fit(view):
                fit = (view.get("stages") or {}).get("fit") or {}
                at = (len(view.get("decisions") or []), fit.get("key"), fit.get("status"))
                if at not in self.pressed:
                    stages = [n for n in ESTIMATE_STAGES if n in (view.get("stages") or {})]
                    seen.append({n: self.c.get(f"/api/projects/{self.pid}/stages/{n}").json()
                                 for n in stages})
            return super().__call__(view)

    return Watching


def _walk(home: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

    before_press: list[dict[str, Any]] = []
    monkeypatch.setattr(J, "FitPresser", _watching(before_press))

    path = J.nhanes_path()
    spec = _spec(list(pd.read_csv(path, nrows=1).columns))
    run = J.Run(spec)
    with local_server(home) as client:
        run.client = J.Recording(client, run)
        r = client.post("/api/projects", json={"path": str(path)})
        assert r.status_code == 200, r.text
        run.pid = r.json()["id"]
        truth = spec.truth()
        truth.guess = J._server_guess(client, run.pid)
        truth.ranked = J._server_ranked(client, run.pid)
        run.drive = Drive(run.client, run.pid, truth)
        run.drive.artifact("ingest", timeout=300)
        followed = J.follow(run)
        results = followed and J.wait_results(run, timeout=300)
        view = run.drive.view()
        stage = lambda name: client.get(f"/api/projects/{run.pid}/stages/{name}").json()  # noqa: E731
        return {"followed": followed, "results": results, "notes": list(run.notes), "view": view,
                "fit": stage("fit"), "cohort": stage("cohort"), "before_press": before_press,
                "sealed": run.drive.sealed() if followed else set()}


# About 75 s on a developer machine (the server's workers fit the families, the evaluation and the
# explanations beside them): the full tier runs it (.github/workflows/v2-full.yml).
@pytest.mark.slow
def test_a_light_nhanes_predict_walk_presses_fit_and_its_held_out_numbers_match_references(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    t0 = time.monotonic()
    walk = _walk(tmp_path / "home", monkeypatch)
    # The driver pressed Fit when the Router reached it, and walked to the Router's end.
    assert walk["followed"], walk["notes"]
    assert "the Router reached Fit: pressed" in walk["notes"], walk["notes"]
    # No estimate before Fit: when the Router reached it, the fit (computed: two light families
    # are not held) was served withheld for Fit, every model without a score.
    # No estimate before Fit: when the Router reached it, no estimate stage served a score. The
    # fit, light as it is, had not been computed (a fit that computes before the press is served
    # withheld for Fit with every score removed, and that is checked the same way).
    before = walk["before_press"][0]
    assert set(before) >= {"fit", "evaluation", "explain"}, sorted(before)
    for name, served in before.items():
        artifact = served.get("artifact")
        if artifact is None:
            continue
        assert artifact.get("withheld_for") == "fit", (name, artifact.get("withheld_for"))
        for model in artifact.get("models") or [] if name == "fit" else []:
            assert model["cv"] == {} and model["baseline"]["value"] is None, model["family"]
            assert model["coefficients"] is None, model["family"]
            for key in SCORES:
                assert model.get(key) is None, (model["family"], key)
    assert before["fit"]["artifact"] is None or before["fit"]["artifact"]["withheld_for"] == "fit"
    assert walk["results"], walk["notes"]
    view = walk["view"]
    assert [s["key"] for s in view["interview"] if s["status"] in ("open", "waiting")] == []
    assert view["state"]["seal_opened"]
    assert view["decisions"][-1]["decision"]["kind"] == "open_seal"
    fit = walk["fit"]
    assert fit["fresh"] and fit["artifact"]["withheld_for"] is None  # served: Fit was pressed
    linear = next(m for m in fit["artifact"]["models"] if m["family"] == "linear")

    # The split, rebuilt from the seal's held-out row ids (row ids are the table's row positions).
    table = pd.read_csv(J.nhanes_path())
    predictors = walk["cohort"]["artifact"]["predictors"]
    assert sorted(predictors) == sorted(PREDICTORS)
    kept = table.dropna(subset=[*predictors, "glucose"])
    held = kept.index.isin(sorted(walk["sealed"]))
    assert int(held.sum()) == fit["artifact"]["n_holdout"] == len(walk["sealed"])
    assert int((~held).sum()) == fit["artifact"]["n_train"]
    X = pd.get_dummies(kept[predictors], drop_first=True, dtype=float)
    y = kept["glucose"].to_numpy(float)

    # scikit-learn's least squares, refit on the training rows and the recorded predictors.
    ols = LinearRegression().fit(X[~held], y[~held])
    y_held, y_hat = y[held], ols.predict(X[held])
    train_mean = float(y[~held].mean())
    e = y_held - y_hat
    expected = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                "r2": float(1 - np.sum(e ** 2) / np.sum((y_held - train_mean) ** 2))}
    for metric, value in expected.items():
        assert linear["holdout"][metric] == pytest.approx(value, rel=TOL, abs=TOL), metric
    # The recorded opening quotes the same held-out scores.
    opened = view["decisions"][-1]["decision"]["scores"]["linear"]
    for metric, value in expected.items():
        assert opened[metric] == pytest.approx(value, rel=TOL, abs=TOL), metric

    # The no-predictor baseline the engine serves (cross-validated, M1_CONTRACT §12.6): the
    # training-fold mean scored on the split's folds, scikit-learn's KFold at the split's seed over
    # the training rows (in the table's order); the score by numpy.
    X_train, y_train = X[~held].to_numpy(), y[~held]

    def out_of_fold(seed: int) -> tuple[np.ndarray, np.ndarray]:
        """The training-fold mean and least squares, each scored out of fold on KFold(5, seed)."""
        mean_oof, ols_oof = np.empty_like(y_train), np.empty_like(y_train)
        for fit_rows, scored in KFold(5, shuffle=True, random_state=seed).split(X_train):
            mean_oof[scored] = y_train[fit_rows].mean()
            ols_oof[scored] = LinearRegression().fit(X_train[fit_rows], y_train[fit_rows]).predict(
                X_train[scored])
        return mean_oof, ols_oof

    mean_oof, ols_oof = out_of_fold(0)
    # The folds are the engine's: least squares on them gives the linear family's pooled CV MSE.
    assert linear["cv"]["mse"]["estimate"] == pytest.approx(
        float(np.mean((y_train - ols_oof) ** 2)), rel=TOL, abs=TOL)
    assert linear["baseline"]["metric"] == fit["artifact"]["primary_metric"] == "mse"
    assert linear["baseline"]["value"] == pytest.approx(
        float(np.mean((y_train - mean_oof) ** 2)), rel=TOL, abs=TOL)
    # The gain over it (models/baseline.py), on R², on the comparison substrate: the split's
    # folds and nine more draws of them at seeds 1 to 9 (the seed + r of models/folds.py), the
    # linear family's pooled out-of-fold R² against the training-fold mean in each, averaged (the
    # baseline's own pooled R² is 0 by construction).
    comparison = fit["artifact"]["comparison"]
    assert (comparison["folds"], comparison["repeats"], comparison["shared"]) == (5, 10, 1)
    r2s, mses = [], []
    for seed in range(comparison["repeats"]):
        mean_r, ols_r = (mean_oof, ols_oof) if seed == 0 else out_of_fold(seed)
        r2s.append(1 - np.sum((y_train - ols_r) ** 2) / np.sum((y_train - mean_r) ** 2))
        mses.append(np.mean((y_train - ols_r) ** 2))
    versus = linear["versus_baseline"]
    assert versus["metric"] == "r2" and versus["df"] == 5 * 10 - 1
    assert versus["gain"] == pytest.approx(float(np.mean(r2s)), rel=TOL, abs=TOL)
    assert linear["compared_on"]["estimate"] == pytest.approx(float(np.mean(mses)), rel=TOL, abs=TOL)

    # The no-predictor baseline on the held-out rows: their y around the training rows' mean, by
    # numpy. The engine serves none yet (the open gap the strict expected failure below names);
    # once it does, it is this.
    baseline_rmse = float(np.sqrt(np.mean((y_held - train_mean) ** 2)))
    served = _held_out_baseline(linear)
    if served is not None:
        assert served == pytest.approx(baseline_rmse, rel=TOL, abs=TOL)

    # The held-out calibration, by statsmodels: the slope of y on ŷ, and the mean of y − ŷ.
    calibration = linear["holdout_detail"]["calibration"]
    assert calibration["n"] == int(held.sum())
    assert calibration["observed"] == pytest.approx(float(y_held.mean()), rel=TOL, abs=TOL)
    assert calibration["expected"] == pytest.approx(float(y_hat.mean()), rel=TOL, abs=TOL)
    slope = sm.OLS(y_held, sm.add_constant(y_hat)).fit()
    large = sm.OLS(y_held - y_hat, np.ones_like(y_held)).fit()
    assert calibration["slope"]["estimate"] == pytest.approx(slope.params[1], rel=TOL, abs=TOL)
    assert calibration["slope"]["se"] == pytest.approx(slope.bse[1], rel=TOL, abs=TOL)
    assert calibration["intercept"]["estimate"] == pytest.approx(large.params[0], rel=TOL, abs=TOL)
    assert calibration["intercept"]["se"] == pytest.approx(large.bse[0], rel=TOL, abs=TOL)
    # The engine's intercept is calibration-in-the-large (models/performance.py, Van Calster et al.
    # 2016), not the intercept of statsmodels' OLS of y on ŷ that the plan's acceptance 2 names:
    # the two differ here, so a change of either definition is seen.
    assert abs(calibration["intercept"]["estimate"] - slope.params[0]) > 1e-3
    print(f"the light walk took {time.monotonic() - t0:.0f} s")


def _held_out_baseline(model: dict[str, Any]) -> float | None:
    """The no-predictor baseline's held-out RMSE, wherever the engine serves it on a fitted model:
    a ``baseline`` beside the held-out scores or in their detail. None while it serves none."""
    for where in (model.get("holdout_detail") or {}, model.get("holdout") or {}):
        found = where.get("baseline") if isinstance(where, dict) else None
        if isinstance(found, dict):
            if found.get("rmse") is not None:
                return float(found["rmse"])
            if found.get("mse") is not None:
                return float(np.sqrt(found["mse"]))
        elif isinstance(found, (int, float)):
            return float(found)
    return None


@pytest.mark.xfail(strict=True, reason=(
    "The engine serves no held-out no-predictor baseline: FittedModel.baseline is the "
    "cross-validated primary (models/artifacts.py, Baseline), and HoldoutDetail carries none, so "
    "W1-G has no engine number for 'held-out scores against the baseline'. The fix belongs to "
    "models/ (phase 3's files): a baseline in HoldoutDetail, its RMSE (or MSE) on the held-out rows "
    "around the training mean, sealed with the rest. When it lands this passes, the strict mark "
    "fails, and the walk above checks its value against numpy."))
def test_the_engine_serves_a_held_out_no_predictor_baseline():
    from turbotab.core.models.artifacts import HoldoutDetail

    assert "baseline" in HoldoutDetail.model_fields

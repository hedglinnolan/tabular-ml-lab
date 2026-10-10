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
* the no-predictor baseline's held-out RMSE by numpy (the held-out y around the training mean),
  against the one the engine's held-out R² is measured against (R² = 1 − MSE/MSE_baseline);
* the held-out calibration (Van Calster et al. 2016, on the outcome's scale as
  ``models.performance`` defines it): the slope is statsmodels' OLS slope of y on ŷ, the intercept
  (calibration-in-the-large) statsmodels' OLS of y − ŷ on a constant, each with its standard error.
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


def _walk(home: Path) -> dict[str, Any]:
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

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
                "fit": stage("fit"), "cohort": stage("cohort"),
                "sealed": run.drive.sealed() if followed else set()}


# About 75 s on a developer machine (the server's workers fit the families, the evaluation and the
# explanations beside them): the full tier runs it (.github/workflows/v2-full.yml).
@pytest.mark.slow
def test_a_light_nhanes_predict_walk_presses_fit_and_its_held_out_numbers_match_references(
        tmp_path: Path):
    t0 = time.monotonic()
    walk = _walk(tmp_path / "home")
    # The driver pressed Fit when the Router reached it, and walked to the Router's end.
    assert walk["followed"], walk["notes"]
    assert "the Router reached Fit: pressed" in walk["notes"], walk["notes"]
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

    # The no-predictor baseline on the held-out rows: their y around the training rows' mean
    # (numpy), against the baseline the engine's held-out R² is measured against.
    baseline = float(np.sqrt(np.mean((y_held - train_mean) ** 2)))
    engine_baseline = float(np.sqrt(linear["holdout"]["mse"] / (1 - linear["holdout"]["r2"])))
    assert engine_baseline == pytest.approx(baseline, rel=TOL, abs=TOL)

    # The held-out calibration, by statsmodels: the slope of y on ŷ, and the mean of y − ŷ.
    calibration = linear["holdout_detail"]["calibration"]
    assert calibration["n"] == int(held.sum())
    slope = sm.OLS(y_held, sm.add_constant(y_hat)).fit()
    large = sm.OLS(y_held - y_hat, np.ones_like(y_held)).fit()
    assert calibration["slope"]["estimate"] == pytest.approx(slope.params[1], rel=TOL, abs=TOL)
    assert calibration["slope"]["se"] == pytest.approx(slope.bse[1], rel=TOL, abs=TOL)
    assert calibration["intercept"]["estimate"] == pytest.approx(large.params[0], rel=TOL, abs=TOL)
    assert calibration["intercept"]["se"] == pytest.approx(large.bse[0], rel=TOL, abs=TOL)
    print(f"the light walk took {time.monotonic() - t0:.0f} s")

"""W1 part 2 ruling 2 · the held-out no-predictor baseline and the held-out verdict against it.

The baseline on the held-out rows is the cross-validated baseline's model (the outcome's training
mean, the class prior) fit on the training rows and scored as each family's held-out scores are
(``stages.modeling.held_out_baseline``), served in ``HoldoutDetail.baseline``. The held-out verdict
(``models.baseline.versus_held_out``) applies the cross-validated verdict's rule and floor to a
held-out gain and its served interval: for a numeric outcome the held-out R² against the training
rows' mean. No other task serves an interval for the gain, so none gets a verdict.

The rule's cases are built from their definition (an interval above zero, a gain above the floor);
the stages' cases from constructed tables whose truth is known (a real signal, none), with each
number checked against numpy or scikit-learn. The numpy truths and the table builders are the
reference test's (``test_w1_press_fit_references``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from turbotab.core.models import baseline as B
from turbotab.core.models.performance import Interval
from turbotab.core.tests.test_w1_press_fit_references import (TOL, _fit_stages, _regression_table,
                                                              _served, _split_rows)

Z = float(norm.ppf(0.975))


def _gain(estimate: float, se: float | None) -> Interval:
    """A served 95% interval, ``estimate ± z·se`` (z from scipy), as the held-out R²'s is."""
    if se is None:
        return Interval(estimate=estimate, method="influence function")
    return Interval(estimate=estimate, se=se, ci_low=estimate - Z * se, ci_high=estimate + Z * se,
                    method="influence function")


# ── the rule, from its definition ────────────────────────────────────────────


def test_a_held_out_gain_well_above_the_floor_whose_interval_clears_zero_is_better():
    versus = B.versus_held_out("r2", _gain(5 * B.MIN_GAIN, B.MIN_GAIN / 2), 400)
    assert versus.verdict == "better" and versus.df is None
    assert versus.gain == 5 * B.MIN_GAIN and versus.ci_low > 0
    assert versus.tolerance == pytest.approx(max(B.MIN_GAIN, Z * B.MIN_GAIN / 2), rel=1e-12)


def test_a_held_out_gain_under_the_floor_is_not_better_though_its_interval_clears_zero():
    versus = B.versus_held_out("r2", _gain(B.MIN_GAIN / 2, B.MIN_GAIN / 20), 4000)
    assert versus.ci_low > 0 and versus.verdict == "no_better"
    assert versus.tolerance == B.MIN_GAIN


def test_a_held_out_gain_whose_interval_spans_zero_is_not_better():
    versus = B.versus_held_out("r2", _gain(5 * B.MIN_GAIN, 5 * B.MIN_GAIN), 400)
    assert versus.ci_low < 0 < versus.gain and versus.verdict == "no_better"


def test_a_held_out_gain_below_zero_is_worse():
    assert B.versus_held_out("r2", _gain(-B.MIN_GAIN, B.MIN_GAIN), 400).verdict == "worse"


def test_no_interval_served_for_the_gain_gives_no_verdict():
    assert B.versus_held_out("r2", _gain(0.5, None), 1) is None
    assert B.versus_held_out("r2", None, 400) is None
    assert B.versus_held_out("r2", Interval(estimate=None), 400) is None


def test_the_held_out_floor_is_the_claims_value_read_when_the_verdict_is_made(monkeypatch):
    """The floor is ``baseline.MIN_GAIN`` (``claims.value("baseline.min_gain", ...)``) as it stands
    when the verdict is made, and the rule's sentence interpolates it: a floor raised past the gain
    turns "better" into "no_better", and the sentence names the raised floor."""
    gain = _gain(5 * B.MIN_GAIN, B.MIN_GAIN / 2)
    before = B.versus_held_out("r2", gain, 400)
    assert before.verdict == "better"
    assert f"at least {B.MIN_GAIN:g} (a convention)" in before.tolerance_basis
    raised = 10 * B.MIN_GAIN
    monkeypatch.setattr(B, "MIN_GAIN", raised)
    after = B.versus_held_out("r2", gain, 400)
    assert after.verdict == "no_better" and after.tolerance == raised
    assert f"at least {raised:g} (a convention)" in after.tolerance_basis
    assert "400 held-out rows" in after.tolerance_basis


def test_the_cross_validated_verdict_keeps_its_rule_and_words():
    """The rule now shared with the held-out verdict leaves the cross-validated one as it was: five
    fold gains of 0.05 ± 0.01 (sample SD by numpy) clear zero and the floor, with the corrected
    resampled t's interval (Nadeau & Bengio) and the sentence it always gave."""
    from scipy.stats import t

    gains = np.array([0.04, 0.05, 0.06, 0.05, 0.05])
    versus = B.versus_baseline("r2", list(gains), [0.0] * 5)
    se = np.sqrt((1 / 5 + 1 / 4) * gains.var(ddof=1))
    half = float(t.ppf(0.975, 4)) * se
    assert versus.verdict == "better" and versus.df == 4
    assert versus.ci_low == pytest.approx(0.05 - half, rel=1e-12)
    assert versus.tolerance_basis == (
        f"Better only when the gain's 95% interval lies above zero and the gain is at least "
        f"{B.MIN_GAIN:g} (a convention): a gain above {max(B.MIN_GAIN, half):.3f}. The interval is "
        f"corrected for the 5 folds sharing training rows (Nadeau & Bengio 2003).")


# ── through the stages ───────────────────────────────────────────────────────


def _state(task: str, target: str, models: list[str], **more):
    from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec

    columns = {"x1": "covariate", "x2": "covariate"}
    return ProjectState(lens=["clinical"], target=target, task=task, purpose="prediction",
                        roles={"id": "identifier", **columns}, missing="complete_case",
                        grain=GrainSpec(grain="one_row_per_unit", id_column="id"),
                        split=SplitSpec(holdout=0.2, seed=4, folds=5), models=models, **more)


def test_with_no_signal_the_held_out_verdict_does_not_call_the_model_better(tmp_path: Path):
    """y independent of both predictors (slope 0): the held-out verdict is never "better", and is
    "worse" exactly when least squares' held-out R² against the training mean (scikit-learn,
    numpy) is below zero."""
    table, frame = _regression_table(tmp_path, 600, slope=0.0, seed=12)
    split, fit = _fit_stages(table, _state("regression", "y", ["linear"]))
    train, held = _split_rows(split)
    linear = next(m for m in _served(fit, opened=True)["models"] if m["family"] == "linear")
    versus = linear["holdout_detail"]["versus_baseline"]
    X, y = frame[["x1", "x2"]].to_numpy(float), frame["y"].to_numpy(float)
    y_hat = LinearRegression().fit(X[train], y[train]).predict(X[held])
    m = float(y[train].mean())
    r2 = 1 - np.sum((y[held] - y_hat) ** 2) / np.sum((y[held] - m) ** 2)
    assert versus["gain"] == pytest.approx(r2, rel=TOL, abs=TOL)
    assert versus["verdict"] != "better"
    assert versus["verdict"] == ("worse" if r2 < 0 else "no_better")


def test_a_yes_no_outcome_gets_its_held_out_baseline_and_no_verdict(tmp_path: Path):
    """For a yes/no outcome the baseline is the training rows' class prior p, scored on the
    held-out rows by numpy: log loss −mean(y·log p + (1 − y)·log(1 − p)), Brier mean((p − y)²),
    and an AUC of ½ (one score for every row orders no pair). Its held-out intervals are each
    score's own, not the gain's, so no verdict is served."""
    from turbotab.core.tests.stage_harness import Ingested

    rng = np.random.default_rng(13)
    n = 600
    x = rng.normal(size=(n, 2))
    event = rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 1.2 * x[:, 0])))
    frame = pd.DataFrame({"id": np.arange(n), "x1": x[:, 0], "x2": x[:, 1],
                          "event": np.where(event, "yes", "no")})
    source = tmp_path / "binary.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "ingested")
    split, fit = _fit_stages(table, _state("binary", "event", ["linear"], event="yes"))
    train, held = _split_rows(split)
    linear = next(m for m in _served(fit, opened=True)["models"] if m["family"] == "linear")
    detail = linear["holdout_detail"]
    p = float(event[train].mean())
    yh = event[held].astype(float)
    want = {"log_loss": float(-np.mean(yh * np.log(p) + (1 - yh) * np.log(1 - p))),
            "brier": float(np.mean((p - yh) ** 2)), "auc": 0.5}
    assert detail["baseline"]["label"] == "the class prior"
    assert set(detail["baseline"]["scores"]) == set(linear["holdout"]) == set(want)
    for metric, value in want.items():
        assert detail["baseline"]["scores"][metric] == pytest.approx(value, rel=TOL, abs=TOL), metric
    assert detail["versus_baseline"] is None


def _class_and_survival_frame() -> pd.DataFrame:
    """400 rows, two predictors, a three-class outcome ``mc`` and a time to event (``time``,
    ``dead``) that both follow ``lp = 0.8·x1 − 0.5·x2``: exponential event times with rate
    ``exp(lp)/4`` censored by a uniform(0.5, 8) follow-up, so no two times tie."""
    rng = np.random.default_rng(29)
    n = 400
    x = rng.normal(size=(n, 2))
    lp = x @ np.array([0.8, -0.5])
    levels = np.array(["red", "green", "blue"])
    frame = pd.DataFrame({"id": np.arange(n), "x1": x[:, 0], "x2": x[:, 1],
                          "mc": levels[np.clip(np.digitize(lp + rng.normal(size=n), [-0.5, 0.5]),
                                               0, 2)]})
    t_event, t_cens = rng.exponential(np.exp(-lp) * 4), rng.uniform(0.5, 8, n)
    frame["time"], frame["dead"] = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int)
    return frame


def _run(tmp_path: Path, frame: pd.DataFrame, name: str, state: Any) -> tuple[Any, dict[str, Any]]:
    """The split and the first family as served after the opening, for ``frame`` under ``state``."""
    from turbotab.core.tests.stage_harness import Ingested

    source = tmp_path / f"{name}.csv"
    frame.to_csv(source, index=False)
    split, fit = _fit_stages(Ingested(source, tmp_path / name), state)
    return split, _served(fit, opened=True)["models"][0]


def test_a_class_outcome_gets_its_held_out_baseline_and_no_verdict(tmp_path: Path):
    """A three-class outcome: the training rows' class prior on every held-out row, by numpy: log
    loss −mean(log p[class]), accuracy the share of the training rows' most common class, macro-F1
    that class's F1 (precision its held-out share, recall 1) over the three classes. No interval
    is served for the gain, so no verdict."""
    frame = _class_and_survival_frame()
    split, model = _run(tmp_path, frame[["id", "x1", "x2", "mc"]], "mc",
                        _state("multiclass", "mc", ["linear"]))
    train, held = _split_rows(split)
    y_train, y_held = frame["mc"].to_numpy()[train], frame["mc"].to_numpy()[held]
    prior = {c: float(np.mean(y_train == c)) for c in np.unique(y_train)}
    common = max(prior, key=prior.get)
    share = float(np.mean(y_held == common))
    want = {"log_loss": float(-np.mean([np.log(prior[c]) for c in y_held])), "accuracy": share,
            "macro_f1": 2 * share / (share + 1) / 3}
    detail = model["holdout_detail"]
    assert detail["baseline"]["label"] == "the class prior"
    assert set(detail["baseline"]["scores"]) == set(model["holdout"]) == set(want)
    for metric, value in want.items():
        assert detail["baseline"]["scores"][metric] == pytest.approx(value, rel=TOL, abs=TOL), metric
    assert detail["versus_baseline"] is None


def test_a_time_to_event_gets_its_held_out_baseline_as_lifelines_scores_it_and_no_verdict(
        tmp_path: Path):
    """A time to event, scored at the declared horizon h = 2: one risk for everyone, so Harrell's
    C is ½ (it orders no pair), and the Brier score at h by lifelines (Graf et al. 1999): the risk
    1 − exp(−H(h)) from the Nelson–Aalen cumulative hazard of the training rows, and each
    held-out row's loss weighted by the Kaplan–Meier survival of censoring on the held-out rows,
    G: an event by h loses (1 − risk)²/G(T−) (the left limit at its time), a row still followed
    past h loses risk²/G(h), and a row censored before h loses 0. No interval is served for the
    gain, so no verdict."""
    from lifelines import KaplanMeierFitter, NelsonAalenFitter

    from turbotab.core.decisions import FollowUpSpec

    frame = _class_and_survival_frame()
    h = 2.0
    tte = _state("time_to_event", "dead", ["cox"], event="1",
                 follow_up=FollowUpSpec(time_column="time", prediction_horizon=h))
    split, model = _run(tmp_path, frame[["id", "x1", "x2", "time", "dead"]], "tte", tte)
    train, held = _split_rows(split)
    time, dead = frame["time"].to_numpy(), frame["dead"].to_numpy()
    hazard = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(time[train], dead[train])
    risk = float(-np.expm1(-hazard.cumulative_hazard_at_times([h]).iloc[0]))
    censoring = KaplanMeierFitter().fit(time[held], 1 - dead[held])

    def G(t: Any) -> np.ndarray:
        return censoring.survival_function_at_times(t).to_numpy()

    t_held, d_held = time[held], dead[held]
    died, alive = (d_held == 1) & (t_held <= h), t_held > h
    loss = np.zeros(len(held))
    loss[died] = (1 - risk) ** 2 / G(np.nextafter(t_held[died], -np.inf))
    loss[alive] = risk ** 2 / G([h])[0]
    assert died.sum() >= 10 and alive.sum() >= 10 and (~died & ~alive).sum() >= 1
    detail = model["holdout_detail"]
    assert detail["baseline"]["label"] == "one risk for everyone"
    assert set(detail["baseline"]["scores"]) == set(model["holdout"]) == {"c_index", "brier_t"}
    assert detail["baseline"]["scores"]["c_index"] == 0.5
    assert detail["baseline"]["scores"]["brier_t"] == pytest.approx(float(loss.mean()), rel=TOL,
                                                                    abs=TOL)
    assert detail["versus_baseline"] is None


def test_an_ordered_outcome_gets_its_held_out_baseline_and_no_verdict(tmp_path: Path):
    """An ordered outcome of four numeric levels: the training rows' level prior p on every
    held-out row, by numpy with F the cumulative prior: C ½ (one mean level for every row orders
    no pair), the ranked probability score mean over rows of Σₖ (Fₖ − 1{y ≤ k})²/(K − 1) over the
    K − 1 cut-points, log loss −mean(log p[y]) (and scikit-learn's), and the absolute error in
    levels against the predictive median, the lowest level where F reaches ½. No interval is
    served for the gain, so no verdict."""
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(31)
    n = 400
    x = rng.normal(size=(n, 2))
    grade = 1 + np.digitize(x @ np.array([0.9, -0.4]) + rng.logistic(size=n), [-1.5, 0.0, 1.5])
    frame = pd.DataFrame({"id": np.arange(n), "x1": x[:, 0], "x2": x[:, 1], "grade": grade})
    split, model = _run(tmp_path, frame, "ord", _state("ordinal", "grade", ["proportional_odds"]))
    train, held = _split_rows(split)
    K = 4
    codes_train, codes_held = grade[train] - 1, grade[held] - 1
    prior = np.bincount(codes_train, minlength=K) / len(codes_train)
    F = np.cumsum(prior)
    assert np.all(prior > 0) and np.min(np.abs(F[:-1] - 0.5)) > 1e-6  # the median is unambiguous
    observed = codes_held[:, None] <= np.arange(K - 1)[None, :]
    median = int(np.argmax(F >= 0.5))
    want = {"c_index": 0.5,
            "rps": float(np.mean(np.sum((F[None, :-1] - observed) ** 2, axis=1) / (K - 1))),
            "log_loss": float(-np.mean(np.log(prior[codes_held]))),
            "mae_levels": float(np.mean(np.abs(median - codes_held)))}
    assert want["log_loss"] == pytest.approx(
        log_loss(codes_held, np.tile(prior, (len(held), 1)), labels=list(range(K))), rel=1e-12)
    detail = model["holdout_detail"]
    assert detail["baseline"]["label"] == "the level prior"
    assert set(detail["baseline"]["scores"]) == set(model["holdout"]) == set(want)
    for metric, value in want.items():
        assert detail["baseline"]["scores"][metric] == pytest.approx(value, rel=TOL, abs=TOL), metric
    assert detail["versus_baseline"] is None

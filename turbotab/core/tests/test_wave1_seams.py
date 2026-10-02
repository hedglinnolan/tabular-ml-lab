"""The seams between the wave-1 methods packages (WP6, WP8, WP9, WP10, WP11, WP12a–c), as the
integration merge found them.

Each package was finished and tested on its own branch from the same base. Where one package's
code reads what another changed, the merged behavior is held here: every test names the two
packages it joins and the reference it is checked against, computed by a path independent of the
code under test (statsmodels on hand-built matrices, or the definition written out).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core.decisions import EnergyAdjustment, SplitSpec
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import references as ref


def _stages(frame: pd.DataFrame, folder: Path, roles: dict[str, str], adjustment: Any, *,
            task: str = "regression", target: str = "y", holdout: float = 0.2,
            purpose: str = "inference", seed: int = 3) -> tuple[dict[str, Any], np.ndarray]:
    """Ingest, design and fit with the linear family and a holdout. Returns the fit artifact's
    first model and the training mask over the frame's rows."""
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target=target, task=task, models=["linear"],
                  energy_adjustment=adjustment, purpose=purpose,
                  split=SplitSpec(holdout=holdout, seed=seed, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=holdout, seed=seed)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    train = split.frames["assignment"]["partition"].to_numpy() == "train"
    return fit.data["models"][0], train


# ── WP6 × WP8: the all-components relative effect comes from the table's own rows ────────────


def _all_components(n: int, seed: int) -> pd.DataFrame:
    """WP6's all-components fixture (audit D/exp1_energy_models.py), at a smaller n."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 15, n)
    size = rng.normal(0, 1, n) - 0.03 * (age - 50)
    energy = 2100 + 450 * size
    share_p = np.clip(rng.normal(0.16, 0.03, n), 0.05, 0.4)
    share_f = np.clip(rng.normal(0.34, 0.05, n), 0.1, 0.6)
    kp, kf = energy * share_p, energy * share_f
    kc = energy - kp - kf
    y = 0.010 * kp + 0.004 * kc + 0.2 * (age - 50) + rng.normal(0, 5, n)
    return pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9,
                         "kcal": kp + kf + kc, "age": age, "y": y})


ROLES = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "kcal": "energy",
         "age": "covariate"}
ALL = EnergyAdjustment(method="all_components", energy_column="kcal",
                       nutrients=["protein_g", "carb_g", "fat_g"])


def test_the_relative_effect_is_estimated_from_every_analyzed_row_under_inference(tmp_path):
    """WP8 estimates the inference table from every analyzed row (BLUEPRINT §12 ruling 3); WP6's
    average relative effect (Tomova et al. 2022) is a contrast of that table, so it must come from
    the same rows. Before the merge fix it was computed from the training fit with the all-rows
    clusters, which disagree in length.

    Reference: statsmodels OLS of y on the three sources in kcal and age over all rows,
    θ = 4·(β_p − w_c·β_c − w_f·β_f) with w the sources' mean shares of the remaining energy over
    all rows, and its HC3 interval written out from the definition on t(n − p).
    """
    frame = _all_components(1500, seed=11)
    model, train = _stages(frame, tmp_path, ROLES, ALL)
    assert model["coefficients_n"] == len(frame) and (~train).sum() > 0

    X = pd.DataFrame({"kp": frame["protein_g"] * 4, "kc": frame["carb_g"] * 4,
                      "kf": frame["fat_g"] * 9, "age": frame["age"]})
    fit = sm.OLS(frame["y"].to_numpy(), sm.add_constant(X)).fit()
    w_c = X["kc"].mean() / (X["kc"].mean() + X["kf"].mean())
    c = np.array([0.0, 4.0, -4.0 * w_c, -4.0 * (1 - w_c), 0.0])
    theta = float(c @ fit.params.to_numpy())
    exog = sm.add_constant(X).to_numpy(dtype=float)
    se = float(np.sqrt(c @ ref.hc3_by_definition(exog, np.asarray(fit.resid)) @ c))
    q = stats.t.ppf(0.975, len(X) - exog.shape[1])

    rows = {r["feature"]: r for r in model["coefficients"]}
    row = rows["protein_g_relative"]
    assert row["estimate"] == pytest.approx(theta, rel=1e-8)
    assert row["ci_low"] == pytest.approx(theta - q * se, rel=1e-6)
    assert row["ci_high"] == pytest.approx(theta + q * se, rel=1e-6)
    assert rows["kcal_from_protein_g"]["estimate"] == pytest.approx(fit.params["kp"], rel=1e-8)


def test_a_binary_relative_effect_is_on_the_tables_odds_ratio_scale(tmp_path):
    """WP8 puts a binary table on the odds-ratio scale (exp of the estimate and of the interval's
    ends, ME-07); WP6's relative-effect rows are part of that table, so they carry the same ratios.
    Reference: exp of the row's own per-gram estimate and interval ends."""
    frame = _all_components(1500, seed=12)
    frame["y"] = np.where(frame["y"] > frame["y"].median(), "high", "low")
    model, _ = _stages(frame, tmp_path, ROLES, ALL, task="binary")
    assert model["inference"]["scale"] == "odds_ratio"
    row = {r["feature"]: r for r in model["coefficients"]}["protein_g_relative"]
    assert row["ratio"] == pytest.approx(np.exp(row["estimate"]), rel=1e-12)
    assert row["ratio_low"] == pytest.approx(np.exp(row["ci_low"]), rel=1e-12)
    assert row["ratio_high"] == pytest.approx(np.exp(row["ci_high"]), rel=1e-12)


# ── WP8 × WP9: choosing among families under repeated k-fold ─────────────────────────────────


def test_selection_optimism_under_repeated_kfold_bootstraps_one_repeats_predictions(tmp_path):
    """WP8's bootstrap bias-corrected CV (Tsamardinos et al. 2018) resamples Π, each training row's
    out-of-fold prediction, once per row. WP9's repeated k-fold scores every row once per repeat,
    so the fit stage keeps the first repeat's predictions for Π (the repeat WP9's out-of-fold
    calibration reads too) and passes the other repeats' fits through.

    Reference: Π rebuilt here by refitting each family's pipeline on the split's first fold column
    alone (scikit-learn pipelines through ``fit_pipeline``), then the same resampling; the fit
    stage's selection must equal it. Π built from the last repeat instead gives another estimate,
    so the check discriminates.
    """
    import warnings

    from sklearn.base import clone

    from turbotab.core.decisions import GrainSpec, ProjectState
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import LABELS, cross_validate, fold_pairs
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.models.selection import OutOfFold, selection_optimism
    from turbotab.core.stages.modeling import coded_outcome, read_assignment
    from turbotab.core.stages.rows import cohort_stage, split_stage
    from turbotab.core.stages.target import target_info_stage
    from turbotab.core.tests.stage_harness import Ingested

    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])
    risk = 1 / (1 + np.exp(-(X.to_numpy() @ np.array([0.7, -0.5, 0.2, 0.0]))))
    frame = X.assign(id=np.arange(n), event=np.where(rng.random(n) < risk, "yes", "no"))
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    families = ["linear", "elastic_net"]
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="yes",
                      purpose="prediction", roles={"id": "identifier", **{c: "covariate" for c in X}},
                      missing="complete_case",
                      grain=GrainSpec(grain="one_row_per_unit", id_column="id"), models=families,
                      split=SplitSpec(holdout=0.0, seed=4, folds=5, validation="repeated_kfold",
                                      repeats=3))
    info = table.run(target_info_stage, st)
    cohort = table.run(cohort_stage, st, {"target_info": info})
    split = table.run(split_stage, st, {"cohort": cohort, "target_info": info})
    design = table.run(design_stage, st, {"split": split, "target_info": info})
    fit = table.run(fit_stage, st, {"design": design, "split": split, "target_info": info})
    assert fit.data["repeats"] == 3
    selection = fit.data["selection"]
    assert selection is not None and selection["families"] == families

    # The training rows as the fit stage reads them (from the ingested store, in the split's order:
    # elastic net's inner folds are keyed by the rows' stored contents).
    assignment = read_assignment(split)
    with table.store() as store:
        stored = modeling_frame(store, [*X.columns, "event"], assignment.index.to_numpy())
    Xs = stored[list(X.columns)]
    ys = np.asarray(coded_outcome("binary", stored["event"].to_numpy(), "yes"))
    assert set(np.unique(ys)) == {0, 1}
    pipelines = design.objects["pipelines"]

    def rebuilt(column: str) -> dict[str, Any]:
        pairs = fold_pairs(assignment[column].to_numpy().astype(int))
        oof = OutOfFold("binary", Xs, ys, pairs)
        results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for key in families:
                wrapped = oof.wrap(key, lambda m, Xf, yf, r: fit_pipeline(m, Xf, yf))
                results[key] = cross_validate("binary", lambda _k=key: clone(pipelines[_k]), Xs, ys,
                                              pairs, fit=wrapped)
        # The best family and its CV score are the fit's own (the mean over every repeat).
        cv = {m["family"]: m["cv"]["auc"]["estimate"] for m in fit.data["models"]}
        results = {k: _Fixed(cv[k]) for k in results}
        return selection_optimism("binary", "auc", results, oof,
                                  {m["family"]: m["label"] for m in fit.data["models"]}, LABELS["auc"])

    first = rebuilt("fold")
    assert selection["best"] == first["best"]
    assert selection["corrected"] == pytest.approx(first["corrected"], abs=1e-9)
    assert selection["optimism"] == pytest.approx(first["optimism"], abs=1e-9)
    assert selection["wins"] == first["wins"]
    last = rebuilt("fold_r2")
    assert abs(last["corrected"] - selection["corrected"]) > 1e-6


class _Fixed:
    """A cross-validation result whose summary is one fixed estimate (the fit's own CV AUC)."""

    def __init__(self, estimate: float):
        self.estimate = estimate

    def summary(self, task: str) -> dict[str, Any]:
        return {"auc": {"estimate": self.estimate}}

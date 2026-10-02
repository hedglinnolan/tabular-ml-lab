"""Tier A: the four fixes the live NHANES journey asked for (M1_CONTRACT §12.4–§12.7).

* Blanks that mean "not asked": leaving ``meds_hbp`` and ``meds_chol`` out keeps the rows complete
  cases dropped, and the counts equal a pandas computation over the source CSV.
* Nested nutrients: the parts of a total move with it, in proportion, on the real rows.
* The baseline: each model's baseline equals a direct DummyRegressor / DummyClassifier CV on the
  same folds, and a model that loses to it says so.
* The band: refits, with width for a linear model on NHANES, around the curve, and cancel stops it.

The NHANES checks skip where the export is not on the machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import PredefinedSplit, cross_validate

from turbotab.core.decisions import ExclusionRule, MissingSpec, ProjectState
from turbotab.core.jobs import Cancelled
from turbotab.core.methods.nesting import nested_components
from turbotab.core.methods.substitution import Shift
from turbotab.core.models.artifacts import FitArtifact, SubstitutionArtifact
from turbotab.core.stages.data import profile_stage
from turbotab.core.stages.modeling import BAND_ROWS, design_stage, fit_stage, substitution_stage
from turbotab.core.stages.proposals import proposals_stage
from turbotab.core.stages.rows import PREDICTOR_ROLES, compute_cohort, roles_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.stage_harness import NHANES, Ingested

needs_nhanes = pytest.mark.skipif(not NHANES.is_file(), reason="the NHANES export is not on this machine")
BLANK = ["meds_hbp", "meds_chol"]
PARTS = ["fat_sat", "fat_mon", "fat_poly"]
KCAL_RULE = ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intake")


@pytest.fixture(scope="module")
def nhanes(tmp_path_factory):
    if not NHANES.is_file():
        pytest.skip("the NHANES export is not on this machine")
    table = Ingested(NHANES, tmp_path_factory.mktemp("nhanes"))
    return table, pd.read_csv(NHANES)


def _paths(table: Ingested) -> dict[str, str]:
    return {"data": str(table.parquet), "source": str(table.source)}


# ── 4 · blanks that mean "not asked" ─────────────────────────────────────────


def _pandas_flow(df: pd.DataFrame, roles: dict[str, str], drop: list[str]) -> tuple[list[int], list[int]]:
    keep = df["glucose"].notna()
    n = [len(df), int(keep.sum())]
    kcal = df["kcal"]
    keep &= kcal.isna() | ((kcal >= 500) & (kcal <= 5000))
    n.append(int(keep.sum()))
    preds = [c for c, r in roles.items() if r in PREDICTOR_ROLES and c not in drop]
    keep &= df[preds].notna().all(axis=1)
    n.append(int(keep.sum()))
    return n, df.index[keep].tolist()


@needs_nhanes
@pytest.mark.parametrize("drop", [[], BLANK])
def test_leaving_the_blank_columns_out_counts_as_pandas_does(nhanes, drop):
    table, df = nhanes
    roles = dict(mf.NHANES_ROLES)
    state = ProjectState(target="glucose", roles=roles, exclusions=[KCAL_RULE],
                         missing=MissingSpec(strategy="complete_case", drop_columns=drop))
    with table.store() as store:
        steps, kept, preds = compute_cohort(store, state, table.info)
    expected, rows = _pandas_flow(df, roles, drop)
    assert [s["n"] for s in steps] == expected
    assert kept.tolist() == rows
    assert not set(drop) & set(preds) and set(roles) - set(drop) >= set(preds)
    # The live journey's numbers: complete cases kept 2,943; leaving the two columns out keeps all.
    assert expected[2] == 21_348 and expected[3] == (21_348 if drop else 2_943)


@needs_nhanes
def test_the_missing_reading_offers_to_leave_out_the_columns_blank_because_not_asked(nhanes):
    table, df = nhanes
    roles = dict(mf.NHANES_ROLES)
    state = ProjectState(lens=["dietary"], target="glucose", roles=roles)
    artifact = table.run(proposals_stage, state, {"roles": None})
    reading = artifact["missing"]
    by = {e["column"]: e for e in reading["columns"]}
    assert set(by) == set(BLANK)  # the only predictors with blanks
    for c in BLANK:
        assert by[c]["n_missing"] == int(df[c].isna().sum())
        assert by[c]["share"] == pytest.approx(df[c].isna().mean(), abs=1e-4)
        assert by[c]["likely_not_asked"] and "not asked" in by[c]["reason"]
    union = int(df[BLANK].isna().any(axis=1).sum())
    assert reading["leave_out"] == {"columns": ["meds_chol", "meds_hbp"], "n_rows": union,
                                    "share": pytest.approx(union / len(df), abs=1e-4)}
    assert union == 18_853  # of all 21,849 rows; 18,405 of the 21,348 inside the kcal rule


@needs_nhanes
def test_the_design_takes_the_left_out_columns_out_of_the_model(nhanes):
    table, df = nhanes
    state = mf.state(energy_adjustment=mf.energy("residual"), models=["linear"],
                     missing=MissingSpec(strategy="complete_case", drop_columns=BLANK))
    split = mf.split_bundle(np.arange(len(df)), seed=0)
    design = design_stage(mf.context(state, {"split": split, "target_info": mf.target_info("regression")},
                                     _paths(table)))
    assert design.data["left_out"] == BLANK
    assert not set(BLANK) & set(design.objects["spec"]["inputs"])
    raw = {n["column"] for n in design.data["lineage"]["nodes"] if n["lane"] == "raw"}
    assert not set(BLANK) & raw and "kcal" in raw


# ── 5 · nested nutrients ─────────────────────────────────────────────────────


@needs_nhanes
def test_the_roles_say_which_nutrients_are_parts_of_which_on_nhanes(nhanes):
    table, _ = nhanes
    state = ProjectState(lens=["dietary"], target="glucose")
    profile = table.run(profile_stage, state)
    roles = table.run(roles_stage, state, {"profile": profile})
    nested = {c["column"]: c["nested_in"] for c in roles["columns"] if c["nested_in"]}
    assert nested == {"sugar": "carb", **{p: "fat_total" for p in PARTS}}


@needs_nhanes
def test_moving_fat_total_on_nhanes_moves_its_parts_in_proportion_and_keeps_them_summing(nhanes):
    _, df = nhanes
    nested = nested_components(df, ["fat_total", *PARTS, "carb", "sugar"])
    shift = Shift(df, donor="fat_total", recipient="carb", kcal_per_unit={"fat_total": 9.0, "carb": 4.0},
                  nested=nested)
    rest = (df["fat_total"] - df[PARTS].sum(axis=1)).to_numpy()
    for k in (100.0, 300.0):
        new = shift.values(df, k)
        _, on = shift.apply(df, k)  # the rows the curve averages over
        assert on.mean() > 0.5
        ratio = new["fat_total"] / df["fat_total"].to_numpy()
        for part in PARTS:
            np.testing.assert_allclose(new[part][on], (df[part].to_numpy() * ratio)[on], rtol=1e-12)
            np.testing.assert_allclose((new[part] / new["fat_total"])[on],
                                       (df[part] / df["fat_total"]).to_numpy()[on], rtol=1e-12)
        np.testing.assert_allclose((sum(new[p] for p in PARTS) + rest * ratio)[on],
                                   new["fat_total"][on], rtol=0, atol=1e-9)
        assert (new["fat_total"][~on] < df["fat_total"].min()).any() or on.all()
        # carb's own part moves with it too, so sugar's share of carb holds.
        np.testing.assert_allclose((new["sugar"] / new["carb"])[on], (df["sugar"] / df["carb"]).to_numpy()[on],
                                   rtol=1e-12)


# ── 6 · the baseline ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    frame = mf.nhanes_like(400, seed=3)
    frame["glucose_high"] = np.where(frame["glucose"] > frame["glucose"].median(), "high", "normal")
    frame["glucose_band"] = pd.cut(frame["glucose"], 3, labels=["low", "mid", "top"]).astype(str)
    frame["noise"] = np.random.default_rng(9).normal(100, 10, len(frame))
    return frame, mf.ingest_frame(frame, tmp_path_factory.mktemp("baseline"))


def _run(st, paths, split, task, target):
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return design, fit


@pytest.mark.parametrize("task,target,dummy,scoring,metric", [
    ("regression", "glucose", DummyRegressor(strategy="mean"), "r2", "r2"),
    ("binary", "glucose_high", DummyClassifier(strategy="prior"), "roc_auc", "auc"),
    # Log loss, a proper scoring rule, is the multiclass primary (audit ME-10, WP9).
    ("multiclass", "glucose_band", DummyClassifier(strategy="prior"), "neg_log_loss", "log_loss"),
])
def test_the_baseline_is_a_dummy_model_cross_validated_on_the_same_folds(table, task, target, dummy,
                                                                          scoring, metric):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame) if task == "regression" else 240), seed=4)
    st = mf.state(energy_adjustment=mf.energy("none"), target=target, models=["linear"])
    _, fit = _run(st, paths, split, task, target)
    FitArtifact.model_validate(fit.data)
    a = split.frames["assignment"]
    train = a[a["partition"] == "train"]
    y = frame.loc[train["row_id"], target].to_numpy()
    X = np.zeros((len(y), 1))
    result = cross_validate(dummy, X, y, cv=PredefinedSplit(train["fold"].to_numpy()),
                            scoring=scoring, return_estimator=True, return_indices=True)
    expected = result["test_score"].mean()
    if scoring.startswith("neg_"):
        expected = -expected  # scikit-learn negates a loss so that higher is better
    if task == "regression":
        # R² is pooled over every out-of-fold prediction, against each fold's training mean: the
        # training mean's own pooled R² is exactly 0.
        sse = sst = 0.0
        for est, tr, te in zip(result["estimator"], result["indices"]["train"], result["indices"]["test"]):
            sse += ((y[te] - est.predict(X[te])) ** 2).sum()
            sst += ((y[te] - y[tr].mean()) ** 2).sum()
        expected = 1 - sse / sst
    baseline = fit.data["models"][0]["baseline"]
    assert baseline["metric"] == metric
    assert baseline["value"] == pytest.approx(expected, abs=1e-12)


def test_a_model_that_loses_to_the_baseline_says_so_first(table):
    frame, paths = table
    split = mf.split_bundle(np.arange(len(frame)), seed=4)
    st = mf.state(energy_adjustment=mf.energy("none"), target="noise", models=["linear"])
    _, fit = _run(st, paths, split, "regression", "noise")
    model = fit.data["models"][0]
    r2, base = model["cv"]["r2"]["mean"], model["baseline"]["value"]
    assert r2 < base  # twenty predictors fit to noise do worse than its average
    assert model["concerns"][0].startswith("Predicts worse than the outcome's average: CV R² −")
    assert "for the average" in model["concerns"][0]
    good = _run(mf.state(energy_adjustment=mf.energy("none"), models=["linear"]), paths, split,
                "regression", "glucose")[1].data["models"][0]
    assert not any("worse than" in c for c in good["concerns"])


# ── 7 · the band ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def nhanes_fit(nhanes):
    """Linear model on NHANES (blank columns left out, residual energy), fat_total → carb."""
    table, df = nhanes
    st = mf.state(energy_adjustment=mf.energy("residual"), models=["linear"],
                  missing=MissingSpec(strategy="complete_case", drop_columns=BLANK),
                  substitution=mf.substitution("fat_total", "carb"))
    split = mf.split_bundle(np.arange(len(df)), seed=0)
    design, fit = _run(st, _paths(table), split, "regression", "glucose")
    return table, st, design, fit


@needs_nhanes
def test_the_band_on_nhanes_comes_from_refits_has_width_and_holds_the_curve(nhanes_fit):
    table, st, design, fit = nhanes_fit
    st = st.model_copy(update={"substitution": st.substitution.model_copy(update={"n_boot": 50})})
    progress: list = []
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, _paths(table), progress))
    SubstitutionArtifact.model_validate(sub)
    curve = sub["models"][0]
    widths = []
    for k, delta, low, high in zip(sub["ks"], curve["delta"], curve["ci_low"], curve["ci_high"]):
        if delta is None:
            assert low is None and high is None
        elif k > 0:
            assert low < delta < high
            widths.append(high - low)
    assert widths and min(widths) > 0
    # Every training row is resampled; past BAND_ROWS a refit draws that many and the band is
    # rescaled to the full sample (audit MA-12: 2,000-row refits made the band about 2x too wide).
    n_train = len(design.frames["training"])
    band = sub["band"]
    assert {k: band[k] for k in ("n_boot", "n_rows", "grouped_by", "failed", "n_units",
                                 "resample_units", "interval")} == {
        "n_boot": 50, "n_rows": n_train, "grouped_by": None, "failed": 0, "n_units": n_train,
        "resample_units": min(n_train, BAND_ROWS), "interval": "normal"}
    assert band["scale"] == pytest.approx(np.sqrt(min(n_train, BAND_ROWS) / n_train))
    assert "Refits that succeeded: Linear model 50 of 50." in band["caption"]
    assert sub["band_estimate"] is None
    assert "refits of its model, on bootstrap resamples" in sub["note"]
    assert "Linear model: refit 50 of 50" in [m for _, m in progress]
    assert [f for f, _ in progress] == sorted(f for f, _ in progress)
    assert sub["carried"] == ["fat_sat", "fat_mon", "fat_poly", "sugar"]


@needs_nhanes
def test_cancel_stops_the_band_between_refits(nhanes_fit):
    table, st, design, fit = nhanes_fit
    st = st.model_copy(update={"substitution": st.substitution.model_copy(update={"n_boot": 50})})
    progress: list = []
    state = {"refits": 0}

    def cancelled() -> bool:
        return state["refits"] >= 5

    def on_progress(fraction: float, message: str) -> None:
        progress.append(message)
        if "refit" in message:
            state["refits"] += 1

    ctx = mf.context(st, {"design": design, "fit": fit}, _paths(table), cancelled=cancelled)
    ctx.on_progress = on_progress
    with pytest.raises(Cancelled):
        substitution_stage(ctx)
    refits = [m for m in progress if "refit" in m]
    assert refits[-1] == "Linear model: refit 5 of 50" and len(refits) == 5


# ── what the validators refuse ───────────────────────────────────────────────


def test_a_total_and_its_own_part_are_refused_as_a_substitution():
    from turbotab.core.decisions import Refusal, validate

    roles = {"columns": [{"column": p, "nested_in": "fat_total"} for p in PARTS]}
    ctx = {"state": mf.state(), "artifact": lambda stage: roles if stage == "roles" else None}
    with pytest.raises(Refusal) as refused:
        validate({"kind": "set_substitution", "donor": "fat_total", "recipient": "fat_sat"}, ctx)
    assert refused.value.code == "part_of_the_other"
    validate({"kind": "set_substitution", "donor": "fat_sat", "recipient": "carb"}, ctx)


def test_energy_adjustment_cannot_use_a_column_the_missing_values_left_out():
    from turbotab.core.decisions import Refusal, validate

    st = mf.state(missing=MissingSpec(strategy="complete_case", drop_columns=["protein"]))
    with pytest.raises(Refusal) as refused:
        validate({"kind": "set_energy_adjustment", "method": "residual", "energy_column": "kcal",
                  "nutrients": ["protein", "carb"]}, {"state": st})
    assert refused.value.code == "left_out"
    assert refused.value.exits[0]["decision"]["nutrients"] == ["carb"]

"""The scorer's refusals and its number for a numeric outcome, on plans rebuilt by replay from short
reference journeys on draws of the hardened Predict generator (``core/export/scorer.py``;
SWARM_PLAN §8: "a scorer that predicts on new rows"). Every expected value is read from outside
the engine: statsmodels' least squares, the logs' own lines.

* A numeric outcome under prediction, two families fitted: unnamed, the scorer refuses and names
  both; named, the straight line's predictions on new rows are statsmodels' OLS refit on the same
  training rows.
* Refused, each by what it names: new rows whose ``age`` is text where the plan read numbers; a
  plan that shrinks its model after the fit (``set_updating``, recorded by the engine on a copy of
  a rebuilt plan); a working table that combines each patient's visits into one row; and one that
  leaves a metabolite panel's pooled QC injections out (its rows numbered again), which combines
  nothing and is said so.

The binary branch's own number (statsmodels' Logit) and the refusals of a plan under inference, a
family not kept and a working table the steps do not reproduce are in ``test_replay_fixtures.py``.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import generators as G  # noqa: E402

pytestmark = pytest.mark.slow

SEED, ROWS = 101, 400
NUMERIC_OUTCOME = "albumin_g_dl"
SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}


def numeric_outcome(seed: int, n: int = ROWS) -> pd.DataFrame:
    """The readmission draw with albumin as the outcome (no yes/no column left)."""
    return G.readmission(seed, n).drop(columns=[G.READMISSION_OUTCOME])


def visits(seed: int) -> pd.DataFrame:
    """The readmission draw as three visits a patient, 30 days apart: each patient's rows are to be
    combined into one."""
    frame = G.readmission(seed, 300).drop(columns=["encounter_id"])
    i = np.arange(len(frame))
    frame.insert(0, "patient_id", [f"P{k:03d}" for k in i // 3])
    frame.insert(1, "visit_date", (pd.Timestamp("2025-01-01")
                                   + pd.to_timedelta(30 * (i % 3), unit="D")).strftime("%Y-%m-%d"))
    return frame


def small_panel(seed: int) -> pd.DataFrame:
    """The measurement fixture's draw with as many features as the pooled-QC finding reads at
    least (``reference_rows.MIN_FEATURES``), so it fires and the fit is quick."""
    from turbotab.core.reference_rows import MIN_FEATURES

    frame = G.metabolite_panel(seed)
    features = [c for c in frame.columns if c.startswith("mz_")]
    return frame.drop(columns=features[MIN_FEATURES:])


def qc_aside(run, view) -> None:
    """Before the outcome: the pooled-QC finding answered by setting its rows aside."""
    from turbotab.core.methods.qc_drift import ASIDE, FINDING
    from turbotab.core.reference import journeys as J

    findings = J._artifact(run, "findings")["findings"]
    qc = next(f for f in findings if f["id"] == FINDING)
    J.post_answer(run, {"kind": "apply_repair", "finding_id": qc["id"], "option": ASIDE})


def _service(home: Path):
    from turbotab.core.config import Settings
    from turbotab.server.service import ProjectService

    return ProjectService(dataclasses.replace(Settings.from_env(), home=home, mode="local",
                                              workers=2), replay=True)


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """Three journeys' bundles, each rebuilt by replay cut before its opening and kept: the numeric
    outcome with two families, the yes/no outcome with one, the visits combined per patient."""
    from turbotab.core.export import replay
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder = tmp_path_factory.mktemp("plans")
    truth = dict(FIXTURE_TRUTHS[G.READMISSION_FILE])
    tables = {"numeric": numeric_outcome(SEED), "binary": G.readmission(SEED, ROWS),
              "visits": visits(SEED), "panel": small_panel(G.PANEL_SEED)}
    paths = {}
    for name, frame in tables.items():
        paths[name] = folder / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    specs = [
        J.Journey("numeric", "clinical", "prediction", "?", str(paths["numeric"]), "test",
                  lambda: paths["numeric"], lambda: J.truth(truth, fixture="numeric.csv"),
                  target=NUMERIC_OUTCOME, lenses=("clinical",),
                  answers={"split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear", "elastic_net"]}},
                  before={"target": J.apply_first_repair(".")}),
        J.Journey("binary", "clinical", "prediction", "?", str(paths["binary"]), "test",
                  lambda: paths["binary"], lambda: J.truth(truth, fixture="binary.csv"),
                  target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                  answers={"split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear"]}},
                  before={"target": J.apply_first_repair(".")}),
        J.Journey("visits", "clinical", "prediction", "?", str(paths["visits"]), "test",
                  lambda: paths["visits"], lambda: J.truth(truth, fixture="visits.csv"),
                  target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                  answers={"grain": {"kind": "set_grain", "grain": "repeated",
                                     "id_column": "patient_id"},
                           "repeat_kind": {"kind": "set_repeat_kind",
                                           "repeat_kind": "time_points",
                                           "time_column": "visit_date"},
                           "unit": {"kind": "set_unit", "unit": "unit"},
                           "aggregation": {"kind": "set_aggregation", "method": "first",
                                           "outcome": "last"},
                           "split": SPLIT,
                           "models": {"kind": "select_models", "models": ["linear"]}},
                  before={"target": J.apply_first_repair(".")}),
        J.Journey("panel", "metabolomics", "prediction", "?", str(paths["panel"]), "test",
                  lambda: paths["panel"],
                  lambda: J.truth(dict(FIXTURE_TRUTHS[G.PANEL_FILE]), fixture="panel.csv"),
                  target=G.PANEL_OUTCOME, lenses=("metabolomics",), event="1",
                  answers={"split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear"]}},
                  before={"target": qc_aside}),
    ]
    out = {"folder": folder}
    for spec in specs:
        capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
        assert capture["export"]["status"] == 200, capture["notes"]
        home = folder / f"kept-{spec.name}"
        report = replay.run(folder / f"{spec.name}.zip", workers=2, cut=True, keep=True,
                            home=str(home))
        assert report.finished is True, report.messages
        out[spec.name] = (home, report.project)
    return out


def _training_rows(home: Path, pid: str) -> np.ndarray:
    from turbotab.core.export.replay import _wait
    from turbotab.core.export.scorer import _bundle
    from turbotab.core.stages.modeling import row_ids_of

    service = _service(home)
    try:
        service.engine.on_decision(pid)
        _wait(service, pid, ("design",), 600)
        design = _bundle(service, pid, "design", service.engine.status(pid))
        return row_ids_of(design.frames["training"])
    finally:
        service.close()


def test_a_numeric_outcome_is_statsmodels_least_squares_on_the_same_training_rows(plans):
    """Two families fitted: unnamed, refused with both named; the straight line's predictions on
    1,000 new rows are statsmodels' OLS on the plan's training rows of the table (sex coded as the
    repair codes it, `f` as 1), to 1e-9."""
    import statsmodels.api as sm

    from turbotab.core.export import scorer

    home, pid = plans["numeric"]
    new = numeric_outcome(SEED + 50, 1000).drop(columns=[NUMERIC_OUTCOME])
    with pytest.raises(scorer.ScoreRefused,
                       match=r"kept 2 families \(elastic_net, linear\): name the one to score"):
        scorer.predict_in(home, pid, new, workers=2)
    scored = scorer.predict_in(home, pid, new, family="linear", workers=2)
    table = numeric_outcome(SEED)
    predictors = [c for c in table.columns if c not in ("encounter_id", NUMERIC_OUTCOME)]
    assert scored.task == "regression" and scored.event is None
    assert scored.inputs == predictors

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[predictors].copy()
        X["sex"] = (X["sex"] == "F").astype(float)
        return sm.add_constant(X.astype(float), has_constant="add")

    train = _training_rows(home, pid)
    assert len(train) == scored.checked_rows == int(round(ROWS * 0.8))
    ols = sm.OLS(table.loc[train, NUMERIC_OUTCOME].to_numpy(), design(table.loc[train])).fit()
    expected = np.asarray(ols.predict(design(new)))
    assert np.max(np.abs(scored.predictions - expected)) < 1e-9


def test_new_rows_whose_age_is_text_are_refused_by_name(plans):
    from turbotab.core.export import scorer

    home, pid = plans["binary"]
    new = G.readmission(SEED + 51, 50).drop(columns=[G.READMISSION_OUTCOME])
    new["age"] = [f"{a} years" for a in new["age"]]
    with pytest.raises(scorer.ScoreRefused, match="In the new rows `age` read as text"):
        scorer.predict_in(home, pid, new, workers=2)


def test_a_plan_that_shrinks_its_model_after_the_fit_is_refused(plans, tmp_path):
    """``set_updating`` shrinkage, recorded by the engine itself on a copy of the rebuilt plan: the
    predictions would leave the shrinkage out, so the scorer refuses; the copy's log is the
    rebuilt one plus that record."""
    from turbotab.core.export import scorer

    home, pid = plans["binary"]
    copy = tmp_path / "home"
    shutil.copytree(home, copy)
    service = _service(copy)
    try:
        before = len(service.log(pid).records())
        service.decide(pid, {"kind": "set_updating", "method": "shrinkage"})
        records = service.log(pid).records()
        assert len(records) == before + 1 and records[-1].decision.kind == "set_updating"
    finally:
        service.close()
    new = G.readmission(SEED + 52, 50).drop(columns=[G.READMISSION_OUTCOME])
    with pytest.raises(scorer.ScoreRefused, match=r"updates its model \(shrinkage\) after the fit"):
        scorer.predict_in(copy, pid, new, workers=2)


def test_a_working_table_that_combines_each_patients_visits_is_refused(plans):
    from turbotab.core.export import scorer

    home, pid = plans["visits"]
    with zipfile.ZipFile(plans["folder"] / "visits.zip") as z:
        log = [json.loads(x) for x in z.read("turbotab-export/decisions.jsonl").splitlines()]
    assert {"set_grain", "set_aggregation"} <= {r["decision"]["kind"] for r in log}
    new = visits(SEED + 53).drop(columns=[G.READMISSION_OUTCOME])
    with pytest.raises(scorer.ScoreRefused, match="The working table combines rows"):
        scorer.predict_in(home, pid, new, workers=2)


def test_a_working_table_that_leaves_the_pooled_qcs_out_is_refused_as_such(plans):
    """The QC rows leave and the rest are numbered again (a row map), but nothing is combined: the
    refusal says which."""
    from turbotab.core.export import scorer
    from turbotab.core.methods.qc_drift import ASIDE, FINDING

    home, pid = plans["panel"]
    with zipfile.ZipFile(plans["folder"] / "panel.zip") as z:
        log = [json.loads(x)["decision"] for x in z.read("turbotab-export/decisions.jsonl")
               .splitlines()]
    assert {"kind": "apply_repair", "finding_id": FINDING, "option": ASIDE} in [
        {k: d.get(k) for k in ("kind", "finding_id", "option")} for d in log]
    assert "set_aggregation" not in [d["kind"] for d in log]
    new = small_panel(G.PANEL_SEED + 1).drop(columns=[G.PANEL_OUTCOME])
    with pytest.raises(scorer.ScoreRefused,
                       match="leaves reference rows out and numbers the rest again") as refused:
        scorer.predict_in(home, pid, new, workers=2)
    assert "combines" not in refused.value.message

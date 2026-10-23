"""The scorer's refusals and its number for a numeric outcome, on plans rebuilt by replay from short
reference journeys on draws of the hardened Predict generator (``core/export/scorer.py``;
SWARM_PLAN §8: "a scorer that predicts on new rows"). Every expected value is read from outside
the engine: statsmodels' least squares, the logs' own lines.

* A numeric outcome under prediction, two families fitted: unnamed, the scorer refuses and names
  both; named, the straight line's predictions on new rows are statsmodels' OLS refit on the same
  training rows.
* Refused, each by what it names: new rows whose ``age`` is text where the plan read numbers, and
  new rows with no ``age`` at all; a plan that shrinks its model after the fit (``set_updating``,
  recorded by the engine on a copy of a rebuilt plan); a working table that combines each
  patient's visits into one row; one that leaves a metabolite panel's pooled QC injections out
  (its rows numbered again), which combines nothing and is said so; an outcome of three levels
  (the readmission draw's, the readmitted split by a coin of its own), which a yes/no's event
  probability would misreport; a metabolite panel exported features-in-rows and turned; a plan
  whose outcome is analyzed on the log scale (``ln_albumin_g_dl``), whose model predicts the log;
  and, from a plan fit on complete cases, new rows with a blank in a predictor its pipeline does
  not fill, the columns named with "or", each with its rows blank (pandas), and the rows blank in
  any of them counted, on rows blank in different columns and on rows blank in both (the same
  rows with no blank are scored as statsmodels' Logit).

Each plan is built the first time a test asks for it, so a test run by name pays for its own.

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
THREE_LEVELS = "readmission"  # none, within_7d or after_7d
TURNED_FEATURES = 8
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


def three_levels(seed: int, n: int = ROWS) -> pd.DataFrame:
    """The readmission draw with its outcome in three levels: not readmitted, or readmitted within
    a week or after it, by a fair coin from a stream of its own."""
    frame = G.readmission(seed, n)
    coin = np.random.default_rng([seed, 3]).random(n) < 0.5
    readmitted = frame.pop(G.READMISSION_OUTCOME).to_numpy() == 1
    frame[THREE_LEVELS] = np.where(~readmitted, "none", np.where(coin, "within_7d", "after_7d"))
    return frame


def tenth_blank(seed: int, n: int = ROWS) -> pd.DataFrame:
    """The readmission draw with every tenth row's hemoglobin blank: a complete-case plan's
    table."""
    frame = G.readmission(seed, n)
    frame.loc[frame.index % 10 == 0, "hemoglobin_g_dl"] = np.nan
    return frame


def turned_panel(seed: int) -> pd.DataFrame:
    """The measurement fixture's participants exported features-in-rows, as an assay table often
    is: a row for age, BMI, each of the first features detected in every participant, and the
    outcome; a column per sample, named by its id."""
    frame = G.metabolite_panel(seed)
    frame = frame[frame["sample_type"] == "participant"].set_index("sample_id")
    features = [c for c in frame.columns if c.startswith("mz_") and frame[c].notna().all()]
    block = frame[["age", "bmi", *features[:TURNED_FEATURES], G.PANEL_OUTCOME]]
    turned = block.astype(float).T
    turned.index.name = "feature_id"
    return turned.reset_index()


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


class Plans:
    """Each journey's bundle, rebuilt by replay cut before its opening and kept, the first time a
    test asks for it by name (``plans["folder"]`` holds the bundles)."""

    def __init__(self, folder: Path, specs: list):
        self.folder, self.specs, self.built = folder, {s.name: s for s in specs}, {}

    def __getitem__(self, name: str):
        from turbotab.core.export import replay
        from turbotab.core.reference import journeys as J

        if name == "folder":
            return self.folder
        if name not in self.built:
            spec, folder = self.specs[name], self.folder
            capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
            assert capture["export"]["status"] == 200, capture["notes"]
            home = folder / f"kept-{spec.name}"
            report = replay.run(folder / f"{spec.name}.zip", workers=2, cut=True, keep=True,
                                home=str(home))
            assert report.finished is True, report.messages
            self.built[name] = (home, report.project)
        return self.built[name]


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """The journeys: the numeric outcome with two families, the yes/no outcome with one, the
    visits combined per patient, the panel with its pooled QCs set aside, the outcome of three
    levels, the panel turned, the numeric outcome on the log scale, and the yes/no outcome fit on
    complete cases."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder = tmp_path_factory.mktemp("plans")
    truth = dict(FIXTURE_TRUTHS[G.READMISSION_FILE])
    tables = {"numeric": numeric_outcome(SEED), "binary": G.readmission(SEED, ROWS),
              "visits": visits(SEED), "panel": small_panel(G.PANEL_SEED),
              "multiclass": three_levels(SEED), "turned": turned_panel(G.PANEL_SEED),
              "logscale": numeric_outcome(SEED), "complete": tenth_blank(SEED)}
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
        J.Journey("multiclass", "clinical", "prediction", "?", str(paths["multiclass"]), "test",
                  lambda: paths["multiclass"], lambda: J.truth(truth, fixture="multiclass.csv"),
                  target=THREE_LEVELS, lenses=("clinical",),
                  answers={"split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear"]}},
                  # sex's coding only: the outcome's "none" against the rest is a finding too
                  before={"target": J.apply_first_repair("sex")}),
        J.Journey("turned", "metabolomics", "prediction", "?", str(paths["turned"]), "test",
                  lambda: paths["turned"],
                  lambda: J.truth(dict(FIXTURE_TRUTHS[G.PANEL_FILE]), fixture="turned.csv"),
                  target=G.PANEL_OUTCOME, lenses=("metabolomics",), event="1",
                  answers={"orientation": {"kind": "set_orientation",
                                           "orientation": "feature_major"},
                           "split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear"]}}),
        J.Journey("logscale", "clinical", "prediction", "?", str(paths["logscale"]), "test",
                  lambda: paths["logscale"], lambda: J.truth(truth, fixture="logscale.csv"),
                  target=NUMERIC_OUTCOME, lenses=("clinical",),
                  answers={"split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear"]}},
                  before={"target": J.apply_first_repair("."),
                          "split": J.declare({"kind": "set_outcome_scale",
                                              "column": NUMERIC_OUTCOME, "scale": "log"})}),
        J.Journey("complete", "clinical", "prediction", "?", str(paths["complete"]), "test",
                  lambda: paths["complete"], lambda: J.truth(truth, fixture="complete.csv"),
                  target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                  answers={"missing": {"kind": "set_missing", "strategy": "complete_case"},
                           "split": SPLIT, "models": {"kind": "select_models",
                                                      "models": ["linear"]}},
                  before={"target": J.apply_first_repair(".")}),
    ]
    return Plans(folder, specs)


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


def test_new_rows_with_no_age_are_refused_by_name(plans):
    """``age`` is a column of the table and one of the plan's predictors (``spec.inputs``): rows
    without it are refused before any prediction, the column named; rows without ``sex`` too
    name both, in the plan's order."""
    from turbotab.core.export import scorer

    home, pid = plans["binary"]
    new = G.readmission(SEED + 54, 50).drop(columns=[G.READMISSION_OUTCOME])
    with pytest.raises(scorer.ScoreRefused,
                       match=r"^The new rows cannot be read as the plan's predictors: they lack "
                             r"`age`\.$"):
        scorer.predict_in(home, pid, new.drop(columns=["age"]), workers=2)
    with pytest.raises(scorer.ScoreRefused, match=r"they lack `age` and `sex`\.$"):
        scorer.predict_in(home, pid, new.drop(columns=["sex", "age"]), workers=2)


def test_an_outcome_of_three_levels_is_refused_not_read_as_an_event(plans):
    """The plan's own record says multiclass (its ``set_task``, the engine's detection): a yes/no's
    probability of the level coded 1 would be one class's probability reported as the event's."""
    from turbotab.core.export import scorer

    home, pid = plans["multiclass"]
    with zipfile.ZipFile(plans["folder"] / "multiclass.zip") as z:
        log = [json.loads(x)["decision"] for x in z.read("turbotab-export/decisions.jsonl")
               .splitlines()]
    assert {"kind": "set_task", "column": THREE_LEVELS, "task": "multiclass"} in [
        {k: d.get(k) for k in ("kind", "column", "task")} for d in log]
    new = three_levels(SEED + 55, 50).drop(columns=[THREE_LEVELS])
    with pytest.raises(scorer.ScoreRefused,
                       match="A multiclass outcome is not scored here: only a yes/no outcome or "
                             "a number"):
        scorer.predict_in(home, pid, new, workers=2)


def test_a_panel_exported_features_in_rows_is_refused_as_turned(plans):
    """The table's rows are features (``set_orientation`` feature_major, recorded): new samples
    would come as columns of such a table, which the scorer does not turn."""
    from turbotab.core.export import scorer

    home, pid = plans["turned"]
    with zipfile.ZipFile(plans["folder"] / "turned.zip") as z:
        log = [json.loads(x)["decision"] for x in z.read("turbotab-export/decisions.jsonl")
               .splitlines()]
    assert {"kind": "set_orientation", "orientation": "feature_major"} in [
        {k: d.get(k) for k in ("kind", "orientation")} for d in log]
    new = turned_panel(G.PANEL_SEED + 1)
    with pytest.raises(scorer.ScoreRefused, match="The table is feature-major: new rows cannot be "
                                                  "read the way its turned working table was"):
        scorer.predict_in(home, pid, new, workers=2)


def _log_of(name: str, folder: Path) -> list[dict]:
    with zipfile.ZipFile(folder / f"{name}.zip") as z:
        return [json.loads(x)["decision"] for x in z.read("turbotab-export/decisions.jsonl")
                .splitlines()]


def test_a_plan_that_analyzes_the_log_of_its_outcome_is_refused_naming_it(plans):
    """``set_outcome_scale`` log makes ``ln_<column>`` the outcome (the record's own answer): the
    model predicts the log, so the scorer refuses, naming the analyzed outcome and the column,
    rather than return log-scale numbers as predictions of albumin. The plan on the original
    scale is scored as statsmodels' OLS (the least-squares test above)."""
    from turbotab.core.export import scorer

    home, pid = plans["logscale"]
    assert {"kind": "set_outcome_scale", "column": NUMERIC_OUTCOME, "scale": "log"} in [
        {k: d.get(k) for k in ("kind", "column", "scale")} for d in _log_of("logscale",
                                                                             plans["folder"])]
    new = numeric_outcome(SEED + 56, 100).drop(columns=[NUMERIC_OUTCOME])
    with pytest.raises(scorer.ScoreRefused) as refused:
        scorer.predict_in(home, pid, new, workers=2)
    assert refused.value.message.startswith(
        f"The plan analyzes `ln_{NUMERIC_OUTCOME}`, the log of its outcome `{NUMERIC_OUTCOME}`")
    assert f"no prediction of `{NUMERIC_OUTCOME}` is returned" in refused.value.message


def test_a_complete_case_plan_refuses_new_rows_with_a_blank_it_does_not_fill(plans):
    """Fit on complete cases (``set_missing`` complete_case, recorded), the pipeline holds no
    imputer. New rows with no blank are scored as statsmodels' Logit refit on the plan's training
    rows, none of which has a blank; the same rows with every fifth hemoglobin blank and one age
    blank are refused before any prediction, both columns named in the plan's order and the 11
    rows with a blank counted, and so are rows whose hemoglobin is blank throughout (said as
    blanks, not as text)."""
    import statsmodels.api as sm

    from turbotab.core.export import scorer

    home, pid = plans["complete"]
    assert {"kind": "set_missing", "strategy": "complete_case"} in [
        {k: d.get(k) for k in ("kind", "strategy")} for d in _log_of("complete", plans["folder"])]
    new = G.readmission(SEED + 57, 50).drop(columns=[G.READMISSION_OUTCOME])
    assert not new.isna().any().any()
    scored = scorer.predict_in(home, pid, new, workers=2)
    table = tenth_blank(SEED)
    predictors = [c for c in table.columns if c not in ("encounter_id", G.READMISSION_OUTCOME)]
    assert scored.inputs == predictors

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[predictors].copy()
        X["sex"] = (X["sex"] == "F").astype(float)
        return sm.add_constant(X.astype(float), has_constant="add")

    train = _training_rows(home, pid)
    assert len(train) == scored.checked_rows and not table.loc[train].isna().any().any()
    refit = sm.Logit(table.loc[train, G.READMISSION_OUTCOME].to_numpy(),
                     design(table.loc[train])).fit(disp=0, tol=1e-12, maxiter=200)
    assert np.max(np.abs(scored.predictions - np.asarray(refit.predict(design(new))))) < 1e-6
    blank = new.copy()
    blank.loc[blank.index % 5 == 0, "hemoglobin_g_dl"] = np.nan
    blank.loc[3, "age"] = np.nan
    count = int(blank[predictors].isna().any(axis=1).sum())
    per = {c: int(blank[c].isna().sum()) for c in ("age", "hemoglobin_g_dl")}
    assert count == 11 and per == {"age": 1, "hemoglobin_g_dl": 10}
    assert not blank[list(per)].isna().all(axis=1).any()  # no row is blank in both
    with pytest.raises(scorer.ScoreRefused) as refused:
        scorer.predict_in(home, pid, blank, workers=2)
    assert refused.value.message.startswith(
        f"{count} of the 50 new rows hold a blank in `age` ({per['age']} row) or "
        f"`hemoglobin_g_dl` ({per['hemoglobin_g_dl']} rows), predictors the plan's fitted pipeline "
        f"does not fill")
    assert "`complete_case`" in refused.value.message
    both = blank.copy()  # two rows blank in both columns: counted once among the rows
    both.loc[[0, 5], "age"] = np.nan
    count = int(both[predictors].isna().any(axis=1).sum())
    per = {c: int(both[c].isna().sum()) for c in ("age", "hemoglobin_g_dl")}
    assert count == 11 and per == {"age": 3, "hemoglobin_g_dl": 10}
    with pytest.raises(scorer.ScoreRefused, match=(
            rf"^{count} of the 50 new rows hold a blank in `age` \({per['age']} rows\) or "
            rf"`hemoglobin_g_dl` \({per['hemoglobin_g_dl']} rows\), predictors the plan's")):
        scorer.predict_in(home, pid, both, workers=2)
    empty = new.assign(hemoglobin_g_dl=np.nan)  # no value to read: blanks, not text
    with pytest.raises(scorer.ScoreRefused, match=r"^50 of the 50 new rows hold a blank in "
                                                  r"`hemoglobin_g_dl`, a predictor the plan's"):
        scorer.predict_in(home, pid, empty, workers=2)


def test_the_columns_with_a_blank_are_said_with_or_each_with_its_rows():
    """Blanks in five columns, the i-th blank in its first i rows (counted by construction): the
    first three said with their rows, the rest counted, joined by "or"; one column alone is just
    named (its rows are the sentence's count)."""
    from turbotab.core.export import scorer

    names = ["a", "b", "c", "d", "e"]
    frame = pd.DataFrame({c: [np.nan] * (i + 1) + [1.0] * (9 - i) for i, c in enumerate(names)})
    assert [int(frame[c].isna().sum()) for c in names] == [1, 2, 3, 4, 5]
    assert scorer._blank_in(names, frame) == "`a` (1 row), `b` (2 rows), `c` (3 rows) or 2 more"
    assert scorer._blank_in(names[:2], frame) == "`a` (1 row) or `b` (2 rows)"
    assert scorer._blank_in(["c"], frame) == "`c`"

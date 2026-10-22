"""Replay's rebuilds and the scorer on the swarm's generated fixtures, through the engine
(``core/export/replay.py``, ``core/export/scorer.py``; SWARM_PLAN §6–§8). Two short reference
journeys (``turbotab.core.reference.journeys``) make the bundles: the default path on a small draw
of the hardened Predict generator with the linear family, and the oracle's answers on a small draw
of the Estimate generator. Every expected value is read from outside the engine: the logs' own
lines, hashlib, statsmodels.

* Truncation: a plain replay still reproduces the bundle (nothing is cut unless asked); cut, the
  Predict log loses its opening and the rebuild reports no held-out score but the same
  cross-validated ones; the Estimate log is cut before its lock and gives the same estimate.
* A fresh draw: the table is the generator's own bytes at the new seed, and the rebuilt crude
  estimate is statsmodels' least squares on that table; a generator that does not write the
  recorded table at the recorded seed is refused before anything is built.
* The findings rule: the sex repair is kept where its finding fires on the new draw, and left out,
  named, where the new draw's sex has a third level and the finding cannot fire; a missing-value
  code only the new draw holds raises a finding no record answers, and the report names it.
* The scorer: its predictions on new rows are statsmodels' maximum likelihood refit on the same
  training rows; it refuses a working table it cannot reproduce (one value changed), a plan under
  inference, and a family the fit did not keep.
"""
from __future__ import annotations

import hashlib
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

PREDICT_SEED, ESTIMATE_SEED = 101, 102
ROWS = 400


def small_readmission(seed: int) -> pd.DataFrame:
    return G.readmission(seed, ROWS)


def three_level_sex(seed: int) -> pd.DataFrame:
    """``small_readmission`` at the recorded seed; at any other seed every tenth row's sex is
    "U", so sex is no longer a two-level text and its binary-text finding cannot fire."""
    frame = small_readmission(seed)
    if seed != PREDICT_SEED:
        frame.loc[frame.index % 10 == 0, "sex"] = "U"
    return frame


def sentinel_codes(seed: int) -> pd.DataFrame:
    """``small_readmission`` at the recorded seed; at any other seed every 25th row's hemoglobin
    is 999, a missing-value code, so a finding fires that the record never answered."""
    frame = small_readmission(seed)
    if seed != PREDICT_SEED:
        frame.loc[frame.index % 25 == 0, "hemoglobin_g_dl"] = 999.0
    return frame


def small_bp(seed: int) -> pd.DataFrame:
    return G.blood_pressure(seed, ROWS)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder = tmp_path_factory.mktemp("bundles")
    predict = folder / G.READMISSION_FILE
    small_readmission(PREDICT_SEED).to_csv(predict, index=False)
    estimate = folder / G.ESTIMATE_FILE
    small_bp(ESTIMATE_SEED).to_csv(estimate, index=False)
    specs = [
        J.Journey("p2-predict", "clinical", "prediction", "?", str(predict), "test",
                  lambda: predict,
                  lambda: J.truth(dict(FIXTURE_TRUTHS[G.READMISSION_FILE]), fixture=predict.name),
                  target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                  answers={"split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
                           "models": {"kind": "select_models", "models": ["linear"]}},
                  before={"target": J.apply_first_repair(".")}),
        J.Journey("p2-estimate", "clinical", "inference", "?", str(estimate), "test",
                  lambda: estimate,
                  lambda: J.truth(dict(FIXTURE_TRUTHS[G.ESTIMATE_FILE]), fixture=estimate.name),
                  target="sbp_mmhg", lenses=("clinical",), exposure="walks_daily",
                  roles={"person_id": "identifier", "age": "covariate", "chronic_pain": "covariate",
                         "walks_daily": "exposure", "height_cm": "covariate",
                         "health_concern_score": "covariate"}),
    ]
    for spec in specs:
        capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
        assert capture["export"]["status"] == 200, capture["notes"]
    return {"predict": folder / "p2-predict.zip", "estimate": folder / "p2-estimate.zip",
            "folder": folder}


def _bundle_files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if "/" in n}


def _kinds(path: Path) -> list[tuple[int, str]]:
    log = _bundle_files(path)["decisions.jsonl"]
    return [(json.loads(x)["seq"], json.loads(x)["decision"]["kind"]) for x in log.splitlines()]


def _recorded_estimates(path: Path) -> dict:
    return json.loads(_bundle_files(path)["provenance.json"])["estimates"]


# ── truncation ───────────────────────────────────────────────────────────────


def test_a_plain_replay_cuts_nothing_and_still_reproduces(bundles):
    from turbotab.core.export import replay

    report = replay.run(bundles["predict"], workers=2)
    assert report.reproduced is True and report.cut is None and report.finished is None


def test_cut_the_predict_log_loses_its_opening_and_keeps_its_cross_validated_scores(bundles):
    from turbotab.core.export import replay

    kinds = _kinds(bundles["predict"])
    opening = next(seq for seq, kind in kinds if kind == "open_seal")
    report = replay.run(bundles["predict"], workers=2, cut=True)
    assert report.finished is True, report.messages
    assert report.cut == replay.Cut(kind="open_seal", seq=opening,
                                    kept=sum(1 for s, _ in kinds if s < opening),
                                    left=sum(1 for s, _ in kinds if s >= opening))
    assert report.decisions == report.cut.kept
    recorded = _recorded_estimates(bundles["predict"])
    again = report.estimates_again
    cv = [k for k in recorded if "/cv/" in k]
    assert cv and all(again[k] == recorded[k] for k in cv)
    assert any("/holdout/" in k for k in recorded)
    assert not any("/holdout/" in k for k in again)


def test_cut_the_estimate_log_before_its_lock_every_stage_computes_and_no_estimate_is_served(
        bundles):
    """Before the lock the engine serves no estimate (the plan locks when Fit is pressed, SIZING
    P0.8): the rebuild's result stages are computed, and its bundle carries none of the estimates
    the locked record's does, for the sweep's own Fit to lock after its revisit."""
    from turbotab.core.export import replay

    kinds = _kinds(bundles["estimate"])
    lock = next(seq for seq, kind in kinds if kind == "lock_plan")
    report = replay.run(bundles["estimate"], workers=2, cut=True)
    assert report.finished is True, report.messages
    assert report.cut.kind == "lock_plan" and report.cut.seq == lock
    assert report.decisions == sum(1 for s, _ in kinds if s < lock)
    key = "table2|linear/model_2/walks_daily|estimate"
    assert key in _recorded_estimates(bundles["estimate"])
    assert not any(k.startswith("table2|") for k in report.estimates_again)


# ── a fresh draw ─────────────────────────────────────────────────────────────


def test_a_fresh_draw_rebuilds_on_the_generators_table_at_the_new_seed(bundles):
    import statsmodels.api as sm

    from turbotab.core.export import replay

    seed = ESTIMATE_SEED + 1
    report = replay.run(bundles["estimate"], workers=2,
                        fresh=replay.FreshDraw(small_bp, ESTIMATE_SEED, seed))
    assert report.finished is True, report.messages
    table = small_bp(seed).to_csv(index=False).encode()
    assert report.fresh.sha256 == hashlib.sha256(table).hexdigest()
    assert report.fresh.regenerated_sha256 == report.fresh.recorded_sha256
    frame = small_bp(seed)
    crude = sm.OLS(frame["sbp_mmhg"], sm.add_constant(frame[["walks_daily"]].astype(float))).fit()
    key = "table2|linear/crude/walks_daily|estimate"
    assert report.estimates_again[key] == pytest.approx(crude.params["walks_daily"], abs=1e-9)
    assert report.estimates_again[key] != _recorded_estimates(bundles["estimate"])[key]


def test_a_fresh_draw_from_a_generator_that_did_not_write_the_table_is_refused(bundles):
    from turbotab.core.export import replay

    report = replay.run(bundles["estimate"], workers=2,
                        fresh=replay.FreshDraw(small_bp, ESTIMATE_SEED + 5, ESTIMATE_SEED + 6))
    assert report.refused and "does not write the recorded table" in report.refused
    assert report.home is None and report.project is None and report.finished is None


# ── the findings rule ────────────────────────────────────────────────────────


def test_the_sex_repair_is_kept_where_its_finding_fires_on_the_new_draw(bundles):
    from turbotab.core.export import replay

    assert "apply_repair" in [k for _, k in _kinds(bundles["predict"])]
    report = replay.run(bundles["predict"], workers=2, cut=True,
                        fresh=replay.FreshDraw(small_readmission, PREDICT_SEED, PREDICT_SEED + 3))
    assert report.finished is True, report.messages
    assert report.left_out == [] and report.decisions == report.cut.kept
    assert report.unanswered == []


def test_a_finding_only_the_new_draw_raises_stays_unanswered_and_is_named(bundles):
    from turbotab.core.export import replay

    report = replay.run(bundles["predict"], workers=2, cut=True,
                        fresh=replay.FreshDraw(sentinel_codes, PREDICT_SEED, PREDICT_SEED + 3))
    assert report.finished is True, report.messages
    assert report.left_out == []
    assert report.unanswered == ["sentinel_missing__hemoglobin_g_dl"]


def test_the_sex_repair_is_left_out_and_named_where_its_finding_cannot_fire(bundles):
    from turbotab.core.export import replay

    report = replay.run(bundles["predict"], workers=2, cut=True,
                        fresh=replay.FreshDraw(three_level_sex, PREDICT_SEED, PREDICT_SEED + 3))
    assert report.finished is True, report.messages
    [left] = report.left_out
    assert left.kind == "apply_repair" and left.finding_id == "binary_text__sex"
    assert "does not fire on this draw" in left.why
    assert report.decisions == report.cut.kept - 1


# ── the scorer ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def kept(bundles):
    from turbotab.core.export import replay

    home = bundles["folder"] / "kept-predict"
    report = replay.run(bundles["predict"], workers=2, cut=True, keep=True, home=str(home))
    assert report.finished is True, report.messages
    return home, report.project


def _training_rows(home: Path, pid: str) -> np.ndarray:
    import dataclasses

    from turbotab.core.config import Settings
    from turbotab.core.export.scorer import _bundle
    from turbotab.core.stages.modeling import row_ids_of
    from turbotab.server.service import ProjectService

    service = ProjectService(dataclasses.replace(Settings.from_env(), home=home, mode="local",
                                                 workers=2), replay=True)
    try:
        service.engine.on_decision(pid)
        from turbotab.core.export.replay import _wait

        _wait(service, pid, ("design",), 600)
        design = _bundle(service, pid, "design", service.engine.status(pid))
        return row_ids_of(design.frames["training"])
    finally:
        service.close()


def test_the_scorer_predicts_what_an_independent_refit_on_the_same_rows_predicts(kept):
    """statsmodels' Logit (Newton) on the plan's training rows of the table, its predictors as the
    table holds them with sex recoded as the repair says (`f` counts as 1), on 2,000 new rows."""
    import statsmodels.api as sm

    from turbotab.core.export import scorer

    home, pid = kept
    rows = small_readmission(PREDICT_SEED + 50)
    new = G.readmission(PREDICT_SEED + 50, 2000).drop(columns=[G.READMISSION_OUTCOME])
    scored = scorer.predict_in(home, pid, new, workers=2)
    predictors = [c for c in rows.columns if c not in ("encounter_id", G.READMISSION_OUTCOME)]
    assert scored.family == "linear" and scored.task == "binary"
    assert scored.inputs == predictors and scored.event == "1"

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[predictors].copy()
        X["sex"] = (X["sex"] == "F").astype(float)
        return sm.add_constant(X.astype(float), has_constant="add")

    table = small_readmission(PREDICT_SEED)
    train = _training_rows(home, pid)
    assert len(train) == scored.checked_rows == int(round(ROWS * 0.8))
    refit = sm.Logit(table.loc[train, G.READMISSION_OUTCOME].to_numpy(),
                     design(table.loc[train])).fit(disp=0, tol=1e-12, maxiter=200)
    expected = np.asarray(refit.predict(design(new)))
    assert np.max(np.abs(scored.predictions - expected)) < 1e-6
    assert scorer.brier_regret(expected, expected) == 0.0


def test_the_scorer_refuses_a_working_table_its_steps_do_not_reproduce(kept, tmp_path):
    """One training row's age changed in a copy of the plan's working table: refused, by name."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from turbotab.core.export import scorer

    home, pid = kept
    copy = tmp_path / "home"
    shutil.copytree(home, copy)
    tables = sorted((copy / "projects" / pid / "cache" / "working").glob("*/files/table.parquet"))
    assert tables, sorted(p.name for p in (copy / "projects" / pid).iterdir())
    for path in tables:
        t = pq.read_table(path)
        age = t.column("age").to_pylist()
        age[0] = age[0] + 1
        t = t.set_column(t.column_names.index("age"), "age", pa.array(age, t.schema.field("age").type))
        pq.write_table(t, path)
    new = G.readmission(7, 50).drop(columns=[G.READMISSION_OUTCOME])
    with pytest.raises(scorer.ScoreRefused, match="`age`"):
        scorer.predict_in(copy, pid, new, workers=2)
    with pytest.raises(scorer.ScoreRefused, match="no `boosted_trees` model"):
        scorer.predict_in(home, pid, new, family="boosted_trees", workers=2)


def test_the_scorer_refuses_a_plan_under_inference(bundles):
    from turbotab.core.export import replay, scorer

    home = bundles["folder"] / "kept-estimate"
    report = replay.run(bundles["estimate"], workers=2, cut=True, keep=True, home=str(home))
    assert report.finished is True, report.messages
    with pytest.raises(scorer.ScoreRefused, match="declared for inference"):
        scorer.predict_in(home, report.project, small_bp(9), workers=2)

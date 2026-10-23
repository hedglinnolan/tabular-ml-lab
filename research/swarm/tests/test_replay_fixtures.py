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
  recorded table at the recorded seed is refused before anything is built, and so is the Predict
  log uncut, whose opening holds the recorded draw's held-out scores.
* A spline whose k the server set by Harrell's rule (sizes: a numeric outcome's row count; the
  smaller of the events and non-events, counted by pandas): rebuilt as recorded on a new draw
  whose analyzed rows give the size the record states. Under inference a draw of another size is
  refused, even at the same k. Under prediction a draw of another size at the same k is rebuilt
  with the form record stating the new size, in its sentence and the methods text (on the pilot's
  Predict table and its next two draws too, the rebuilt record and cross-validated scores those
  of the same answers given on the draw directly); one whose size gives another k is refused, and
  so is the recorded table itself when an exclusion recorded after the spline gives the rule
  another k. A later exclusion that changes the size (on the rows aged 60 and over, refused; 50
  and over, rebuilt stating its size) is named as the record that changed it; readings confirmed
  after the spline, which change no row of these blank-free draws (pandas), are not.
* The findings rule: the sex repair is kept as its own bytes where its finding fires on the new
  draw, and left out, named, where the new draw's sex has a third level and the finding cannot
  fire; a missing-value code only the new draw holds raises a finding no record answers, and the
  report names it; where the new draw spells sex otherwise, the recorded coding is not one its
  finding offers, and the replay is refused naming the record.
* Read-numbers repairs: hemoglobin written with "." for a blank and 999 for a missing value, so the
  missing-value finding fires only once the text is read as numbers. On a new draw of the same
  kind both repairs are kept and the working table holds no 999 (its blanks counted from the
  generator's own frame); on a new draw whose hemoglobin is plain numbers neither finding fires and
  both repairs are left out.
* QC-RLSC on the measurement fixture, through the engine's findings on two draws of the panel: the
  per-batch correction recorded on the committed draw is replayed on the next with that draw's
  offered settings, so a feature only the new draw's correction drops leaves the predictors; the
  sex repair, whose settings both draws offer, is its own bytes.
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
from typing import Sequence

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


def spelled_sex(seed: int) -> pd.DataFrame:
    """``small_readmission`` at the recorded seed; at any other seed sex is spelled out ("female",
    "male"), so the recorded coding ("f" counts as 1) is not one the new draw's finding offers."""
    frame = small_readmission(seed)
    if seed != PREDICT_SEED:
        frame["sex"] = frame["sex"].map({"F": "female", "M": "male"})
    return frame


def text_hemoglobin(seed: int) -> pd.DataFrame:
    """``small_readmission`` with hemoglobin written as text: "." in every 20th row (a blank) and
    999 in every 25th otherwise (a missing-value code)."""
    frame = small_readmission(seed)
    hb = frame["hemoglobin_g_dl"].astype(object)
    hb[frame.index % 25 == 0] = 999.0
    hb[frame.index % 20 == 0] = "."
    frame["hemoglobin_g_dl"] = hb
    return frame


def numbers_at_new_seeds(seed: int) -> pd.DataFrame:
    """``text_hemoglobin`` at the recorded seed; at any other seed hemoglobin is plain numbers, with
    no blank and no code, so neither of its findings fires."""
    return text_hemoglobin(seed) if seed == PREDICT_SEED else small_readmission(seed)


def small_bp(seed: int) -> pd.DataFrame:
    return G.blood_pressure(seed, ROWS)


def repair_each(run, view) -> None:
    """Before the outcome: each finding's first repair, one at a time, reading the findings again
    after each (a finding that fires only once a repair is applied is answered too)."""
    import time

    from turbotab.core.reference import journeys as J

    done: set[str] = set()
    for _ in range(8):
        time.sleep(1.0)
        todo = [f for f in J._artifact(run, "findings")["findings"]
                if f.get("repairs") and f["id"] not in done]
        if not todo:
            return
        done.add(todo[0]["id"])
        J.post_answer(run, {"kind": "apply_repair", "finding_id": todo[0]["id"],
                            "option": todo[0]["repairs"][0]["key"]})
    raise AssertionError(f"the findings kept raising new repairs: {sorted(done)}")


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
    readings = folder / "readings" / G.READMISSION_FILE
    readings.parent.mkdir()
    text_hemoglobin(PREDICT_SEED).to_csv(readings, index=False)
    specs.append(J.Journey(
        "p2-readings", "clinical", "prediction", "?", str(readings), "test", lambda: readings,
        lambda: J.truth(dict(FIXTURE_TRUTHS[G.READMISSION_FILE]), fixture=readings.name),
        target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
        answers={"split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
                 "models": {"kind": "select_models", "models": ["linear"]}},
        before={"target": repair_each}))
    for spec in specs:
        capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
        assert capture["export"]["status"] == 200, capture["notes"]
    return {"predict": folder / "p2-predict.zip", "estimate": folder / "p2-estimate.zip",
            "readings": folder / "p2-readings.zip", "folder": folder}


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


def test_a_fresh_draw_of_the_uncut_predict_log_is_refused_naming_its_opening(bundles):
    """The Predict log opens its held-out rows; uncut, its opening's scores (the recorded draw's)
    would be the rebuilt result. Refused before any project is built, the record and its score
    named, and ``--truncate open_seal`` given; cut, the same draw is rebuilt (the findings-rule
    tests below), as is the Estimate log uncut (above)."""
    from turbotab.core.export import replay

    log = [json.loads(x) for x in _bundle_files(bundles["predict"])["decisions.jsonl"].splitlines()]
    [opening] = [r for r in log if r["decision"]["kind"] == "open_seal"]
    score = opening["decision"]["scores"]["linear"]["log_loss"]
    assert opening["decision"]["n_holdout"] == int(round(ROWS * 0.2))
    report = replay.run(bundles["predict"], workers=2,
                        fresh=replay.FreshDraw(small_readmission, PREDICT_SEED, PREDICT_SEED + 3))
    assert report.refused is not None, report.messages
    assert report.refused.startswith(
        f"Record {opening['seq']} (open_seal) holds the recorded draw's held-out scores as the "
        f"plan's reported result (`linear`'s held-out `log_loss` {score:.3g} on "
        f"{int(round(ROWS * 0.2))} rows)")
    assert "--truncate open_seal" in report.refused
    assert report.finished is None and report.project is None and report.home is None


# ── a spline whose k is Harrell's rule ───────────────────────────────────────


def fewer_bp_rows(seed: int) -> pd.DataFrame:
    """``small_bp`` at the recorded seed; at any other seed 20 rows fewer, so the analyzed rows'
    effective sample size (their number, for a numeric outcome) is another."""
    return G.blood_pressure(seed, ROWS if seed == ESTIMATE_SEED else ROWS - 20)


def harrell_k(n: float) -> int:
    """Harrell's rule as the record states it: 3 knots below an effective sample size of 30, 5
    from 100, else 4."""
    return 3 if n < 30 else 5 if n >= 100 else 4


def events_or_not(frame: pd.DataFrame) -> int:
    """The smaller of the events and the non-events: a yes/no outcome's effective sample size."""
    y = frame[G.READMISSION_OUTCOME].to_numpy()
    return int(min(y.sum(), len(y) - y.sum()))


def spline_once_the_rows_are_read(run, view) -> None:
    """Before the models: ``age`` read as an amount (as the fixture's truth reads it; a form waits
    for that reading), then, once the outcome's task, the working table, the cohort and the split
    are computed again (what the rule reads), a spline of ``age`` with no k given."""
    from turbotab.core.reference import journeys as J

    J.post_answer(run, {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
                        "value": "amount"})
    for stage in ("target_info", "working", "cohort", "split"):
        J._artifact(run, stage)
    J.post_answer(run, {"kind": "set_exposure_form", "column": "age", "form": "spline"})


@pytest.fixture(scope="module")
def splines(tmp_path_factory):
    """Two journeys whose ``age`` takes a spline with no k given, so the server records k by
    Harrell's rule and the effective sample size it read: the Estimate fixture's oracle answers
    with ``set_forms``, and the Predict fixture's default path with ``set_exposure_form``."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder = tmp_path_factory.mktemp("splines")
    predict = folder / G.READMISSION_FILE
    small_readmission(PREDICT_SEED).to_csv(predict, index=False)
    estimate = folder / G.ESTIMATE_FILE
    small_bp(ESTIMATE_SEED).to_csv(estimate, index=False)
    specs = [
        J.Journey("spline-estimate", "clinical", "inference", "?", str(estimate), "test",
                  lambda: estimate,
                  lambda: J.truth(dict(FIXTURE_TRUTHS[G.ESTIMATE_FILE]), fixture=estimate.name),
                  target="sbp_mmhg", lenses=("clinical",), exposure="walks_daily",
                  roles={"person_id": "identifier", "age": "covariate", "chronic_pain": "covariate",
                         "walks_daily": "exposure", "height_cm": "covariate",
                         "health_concern_score": "covariate"},
                  before={"models": J.declare({"kind": "set_forms",
                                               "forms": {"age": {"form": "spline"}}})}),
        J.Journey("spline-predict", "clinical", "prediction", "?", str(predict), "test",
                  lambda: predict,
                  lambda: J.truth(dict(FIXTURE_TRUTHS[G.READMISSION_FILE]), fixture=predict.name),
                  target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                  answers={"split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
                           "models": {"kind": "select_models", "models": ["linear"]}},
                  before={"target": J.apply_first_repair("."),
                          "models": spline_once_the_rows_are_read}),
    ]
    for spec in specs:
        capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
        assert capture["export"]["status"] == 200, capture["notes"]
    return {"estimate": folder / "spline-estimate.zip", "predict": folder / "spline-predict.zip"}


def _age_record(log: bytes) -> dict:
    """The last record of ``log`` that gives ``age`` its form, as its line's JSON."""
    found = None
    for line in log.splitlines():
        r = json.loads(line)
        d = r["decision"]
        if (d["kind"] == "set_forms" and "age" in d["forms"]) or (
                d["kind"] == "set_exposure_form" and d["column"] == "age"):
            found = r
    assert found is not None
    return found


def _form(record: dict) -> dict:
    d = record["decision"]
    if d["kind"] == "set_forms":
        return d["forms"]["age"]
    return {k: v for k, v in d.items() if k not in ("kind", "column")}


def _age_spline(path: Path) -> tuple[int, str, dict]:
    """The last record that gives ``age`` its form: its seq, kind and form."""
    record = _age_record(_bundle_files(path)["decisions.jsonl"])
    return record["seq"], record["decision"]["kind"], _form(record)


def _rebuilt(report) -> dict[str, bytes]:
    """The rebuilt bundle's files, in the home a rebuild kept."""
    return _bundle_files(Path(report.home) / "replay" / "bundle.zip")


def _effective_words(n: int) -> str:
    """Harrell's rule's size as the record's sentence states one for a yes/no outcome."""
    return f"here {n:,}, the smaller of the events and the non-events"


# The kinds in these journeys' logs that write what the analyzed rows are read under: the
# exclusions, and the readings, which the participant flow's complete cases read (the models
# answer does not).
ROW_KINDS = ("set_exclusions", "confirm_readings", "confirm_reading")


def _later(path: Path, seq: int) -> list[tuple[int, str]]:
    """The records after ``seq`` of ``ROW_KINDS``, as (seq, kind)."""
    return [(s, k) for s, k in _kinds(path) if s > seq and k in ROW_KINDS]


def _changed(*named: tuple[int, str]) -> str:
    """The clause naming the later records whose own step changed the size the rule reads, as a
    refusal names them; none named, no clause."""
    if not named:
        return ""
    listed = [f"{s} ({k})" for s, k in named]
    words = listed[0] if len(listed) == 1 else f"{', '.join(listed[:-1])} and {listed[-1]}"
    return (f", after record{'s' if len(listed) > 1 else ''} {words}, which changed the size it "
            f"reads")


def _readings_change_no_row(later: Sequence[tuple[int, str]], *frames: pd.DataFrame) -> None:
    """Each of ``later`` is a confirmed reading, and no cell of ``frames`` (the draws replayed) is
    blank (pandas): a reading confirmed then changes no row the cohort keeps (complete cases drop
    only rows with a blank), so it changes no size the rule reads."""
    assert all(k in ("confirm_readings", "confirm_reading") for _, k in later)
    assert all(not f.isna().any().any() for f in frames)


def test_an_estimate_spline_is_rebuilt_where_the_new_draw_reads_its_size_and_refused_elsewhere(
        splines):
    """The record's k = 5 by the rule on 400 analyzed rows (the table's, a numeric outcome's size
    is its rows). A new draw of 400 rows reads the same size: rebuilt, the estimate served. A new
    draw of 380 rows reads 380 (k = 5 still, but the record states 400, and under inference the
    engine asks the form question again on another size): refused, both sizes named, nothing
    kept."""
    from turbotab.core.export import replay

    seq, kind, form = _age_spline(splines["estimate"])
    assert (form["form"], form["knots_rule"], form["n_effective"], form["knots"]) == (
        "spline", "harrell", float(ROWS), harrell_k(ROWS))
    same = replay.run(splines["estimate"], workers=2,
                      fresh=replay.FreshDraw(small_bp, ESTIMATE_SEED, ESTIMATE_SEED + 1))
    assert same.refused is None and same.finished is True, (same.refused, same.messages)
    assert "table2|linear/model_2/walks_daily|estimate" in same.estimates_again
    assert not any("Harrell" in m for m in same.messages)
    fewer = replay.run(splines["estimate"], workers=2,
                       fresh=replay.FreshDraw(fewer_bp_rows, ESTIMATE_SEED, ESTIMATE_SEED + 1))
    assert harrell_k(ROWS - 20) == harrell_k(ROWS)
    # No later record writes what the rows are read under, so none is named.
    assert _later(splines["estimate"], seq) == [], _kinds(splines["estimate"])
    assert fewer.refused == (
        f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
        f"rows: k = {harrell_k(ROWS)} on an effective sample size of {ROWS} on the recorded draw, "
        f"and on this draw the rule reads {ROWS - 20} (k = {harrell_k(ROWS - 20)}) on the rows as "
        f"the log ends. A fresh draw replays the record's k and the size it states as recorded, "
        f"so the rebuild is refused."
    ), (fewer.refused, fewer.messages)
    assert fewer.project is None and fewer.home is None and fewer.finished is None


def test_a_predict_spline_is_rebuilt_where_the_rule_gives_its_k_again_and_refused_elsewhere(
        splines, tmp_path):
    """Under prediction the record's k reads the smaller of the events and the non-events on the
    split's rows (all 400): 117 at the recorded seed (pandas, on the generator's frame), k = 5,
    and the engine keeps that k whatever the rows. Cut before the opening: a draw with 117 again
    is rebuilt as recorded; one with 110 (k = 5 again) is rebuilt with its form record stating
    110, in its sentence and the rebuilt bundle's methods text, both sizes in the report; one
    with 95 (k = 4) is refused. The readings confirmed after the spline change no row of these
    blank-free draws, so neither sentence names them."""
    from turbotab.core.export import replay

    seq, kind, form = _age_spline(splines["predict"])
    recorded = events_or_not(small_readmission(PREDICT_SEED))
    assert (form["form"], form["knots_rule"], form["n_effective"], form["knots"]) == (
        "spline", "harrell", float(recorded), harrell_k(recorded))
    alike, nearer, other = PREDICT_SEED + 122, PREDICT_SEED + 1, PREDICT_SEED + 3
    assert events_or_not(small_readmission(alike)) == recorded
    near = events_or_not(small_readmission(nearer))
    assert near != recorded and harrell_k(near) == harrell_k(recorded)
    fewer = events_or_not(small_readmission(other))
    assert harrell_k(fewer) != harrell_k(recorded)
    _readings_change_no_row(_later(splines["predict"], seq),
                            *(small_readmission(s) for s in (alike, nearer, other)))
    same = replay.run(splines["predict"], workers=2, cut=True, keep=True, home=str(tmp_path / "a"),
                      fresh=replay.FreshDraw(small_readmission, PREDICT_SEED, alike))
    assert same.refused is None and same.finished is True, (same.refused, same.messages)
    assert any("/cv/" in k for k in same.estimates_again)
    assert not any("Harrell" in m for m in same.messages)
    assert _form(_age_record(_rebuilt(same)["decisions.jsonl"])) == form
    refilled = replay.run(splines["predict"], workers=2, cut=True, keep=True,
                          home=str(tmp_path / "b"),
                          fresh=replay.FreshDraw(small_readmission, PREDICT_SEED, nearer))
    assert refilled.refused is None and refilled.finished is True, (refilled.refused,
                                                                    refilled.messages)
    files = _rebuilt(refilled)
    record = _age_record(files["decisions.jsonl"])
    assert record["seq"] == seq and _form(record) == {**form, "n_effective": float(near)}
    assert _effective_words(near) in record["sentence"]
    methods = files["methods.md"].decode()
    assert _effective_words(near) in methods and _effective_words(recorded) not in methods
    assert [m for m in refilled.messages if "Harrell" in m] == [
        f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
        f"rows: k = {harrell_k(recorded)} on an effective sample size of {recorded} on the "
        f"recorded draw, and on this draw the rule reads {near} (k = {harrell_k(near)}) on the "
        f"rows as the log ends. Under prediction the record's k is replayed as recorded, "
        f"and it is the rule's on this draw too, so the record is replayed stating this draw's "
        f"size, {near}, in place of {recorded}."]
    refused = replay.run(splines["predict"], workers=2, cut=True,
                         fresh=replay.FreshDraw(small_readmission, PREDICT_SEED, other))
    assert refused.refused == (
        f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
        f"rows: k = {harrell_k(recorded)} on an effective sample size of {recorded} on the "
        f"recorded draw, and on this draw the rule reads {fewer} (k = {harrell_k(fewer)}) on the "
        f"rows as the log ends. Under prediction a fresh draw replays the record's k as "
        f"recorded, not the rule's on this draw, so the rebuild is refused."), (refused.refused,
                                                                                refused.messages)
    assert refused.project is None and refused.finished is None


def _predict_spline_journey(folder: Path, name: str, frame: pd.DataFrame, before) -> Path:
    """The default Predict path on ``frame`` (saved as the fixture's file), with the linear
    family and ``before`` the models question; its bundle."""
    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / G.READMISSION_FILE
    frame.to_csv(path, index=False)
    spec = J.Journey(name, "clinical", "prediction", "?", str(path), "test", lambda: path,
                     lambda: J.truth(dict(FIXTURE_TRUTHS[G.READMISSION_FILE]), fixture=path.name),
                     target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                     answers={"split": {"kind": "set_split", "holdout": 0.2, "seed": 0,
                                        "folds": 5},
                              "models": {"kind": "select_models", "models": ["linear"]}},
                     before={"target": J.apply_first_repair("."), "models": before})
    capture = J.run_journey(spec, bundles=folder, out_dir=folder / "captures")
    assert capture["export"]["status"] == 200, capture["notes"]
    return folder / f"{name}.zip"


SIXTY = 60


def spline_then_sixty_and_over(run, view) -> None:
    """The spline of ``age`` once the rows are read, then an exclusion of the rows under 60."""
    from turbotab.core.reference import journeys as J

    spline_once_the_rows_are_read(run, view)
    J.post_answer(run, {"kind": "set_exclusions", "rules": [
        {"kind": "range", "column": "age", "low": SIXTY, "reason": "sixty and over"}]})


def test_a_later_exclusion_that_gives_the_rule_another_k_is_named_where_the_draw_is_refused(
        tmp_path):
    """The spline is declared on all 400 rows (117, k = 5) and an exclusion of the rows under 60
    is recorded after it, so the rows as the log ends give 96 (pandas, on the rows aged 60 and
    over; k = 4) even on a draw whose bytes are the recorded table's: the rebuild is refused,
    the size said to be read on the rows as the log ends and the exclusion named as the record
    that changed it, not the readings confirmed after the spline (no cell of the table is
    blank)."""
    from turbotab.core.export import replay

    frame = small_readmission(PREDICT_SEED)
    bundle = _predict_spline_journey(tmp_path / "later", "spline-later", frame,
                                     spline_then_sixty_and_over)
    seq, kind, form = _age_spline(bundle)
    recorded, kept = events_or_not(frame), events_or_not(frame[frame["age"] >= SIXTY])
    assert form["n_effective"] == float(recorded) and harrell_k(kept) != harrell_k(recorded)
    [exclusion] = [s for s, k in _kinds(bundle) if k == "set_exclusions" and s > seq]
    _readings_change_no_row([x for x in _later(bundle, seq) if x[0] != exclusion], frame)
    report = replay.run(bundle, workers=2, cut=True, fresh=replay.FreshDraw(
        lambda seed: small_readmission(PREDICT_SEED), PREDICT_SEED, PREDICT_SEED + 1))
    assert report.fresh.sha256 == report.fresh.recorded_sha256
    assert report.refused == (
        f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
        f"rows: k = {harrell_k(recorded)} on an effective sample size of {recorded} on the "
        f"recorded draw, and on this draw the rule reads {kept} (k = {harrell_k(kept)}) on the "
        f"rows as the log ends{_changed((exclusion, 'set_exclusions'))}. Under prediction a "
        f"fresh draw replays the record's k as recorded, not the rule's on this draw, so the "
        f"rebuild is refused."
    ), (report.refused, report.messages)


FIFTY = 50


def spline_then_fifty_and_over(run, view) -> None:
    """The spline of ``age`` once the rows are read, then an exclusion of the rows under 50."""
    from turbotab.core.reference import journeys as J

    spline_once_the_rows_are_read(run, view)
    J.post_answer(run, {"kind": "set_exclusions", "rules": [
        {"kind": "range", "column": "age", "low": FIFTY, "reason": "fifty and over"}]})


def test_a_later_exclusion_that_changes_the_rules_size_is_named_where_the_record_is_refilled(
        tmp_path):
    """The spline is declared on all 400 rows (117, k = 5), an exclusion of the rows under 50 is
    recorded after it and readings are confirmed after that. On a draw whose bytes are the
    recorded table's, the rows as the log ends give 114 (pandas, on the rows aged 50 and over;
    k = 5 again): the record is replayed stating 114, and its message names the exclusion, whose
    step changed the size the rule reads, and not the readings confirmed after it, which change
    no row of this table (no cell is blank)."""
    from turbotab.core.export import replay

    frame = small_readmission(PREDICT_SEED)
    bundle = _predict_spline_journey(tmp_path / "fifty", "spline-fifty", frame,
                                     spline_then_fifty_and_over)
    seq, kind, form = _age_spline(bundle)
    recorded, kept = events_or_not(frame), events_or_not(frame[frame["age"] >= FIFTY])
    assert form["n_effective"] == float(recorded)
    assert kept != recorded and harrell_k(kept) == harrell_k(recorded)
    [exclusion] = [s for s, k in _kinds(bundle) if k == "set_exclusions" and s > seq]
    confirmed = [x for x in _later(bundle, seq) if x[0] > exclusion]
    assert confirmed, _kinds(bundle)
    _readings_change_no_row(confirmed, frame)
    report = replay.run(bundle, workers=2, cut=True, fresh=replay.FreshDraw(
        lambda seed: small_readmission(PREDICT_SEED), PREDICT_SEED, PREDICT_SEED + 1))
    assert report.refused is None and report.finished is True, (report.refused, report.messages)
    assert [m for m in report.messages if "Harrell" in m] == [
        f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the analyzed "
        f"rows: k = {harrell_k(recorded)} on an effective sample size of {recorded} on the "
        f"recorded draw, and on this draw the rule reads {kept} (k = {harrell_k(kept)}) on the "
        f"rows as the log ends{_changed((exclusion, 'set_exclusions'))}. Under prediction the "
        f"record's k is replayed as recorded, and it is the rule's on this draw too, so the "
        f"record is replayed stating this draw's size, {kept}, in place of {recorded}."]


def test_the_pilot_predict_fixture_is_rebuilt_on_two_fresh_draws_stating_each_draws_size(
        tmp_path):
    """The pilot's Predict table (the generator at its committed seed, 2000 rows: the committed
    file's bytes) with a spline of ``age`` by the rule: 459 events or non-events, whichever is
    fewer (pandas), k = 5. Its next two draws give 506 and 517, k = 5 again: each is rebuilt, its
    form record stating its own size, in the sentence the methods text quotes, and its message
    names no later record (the readings confirmed after the spline change no row of these
    blank-free draws). On the first, the rebuilt record and cross-validated scores are those of
    the same answers given on that draw directly (its own journey)."""
    from turbotab.core.export import replay

    seed = G.READMISSION_SEED
    pilot = G.readmission(seed)
    assert pilot.to_csv(index=False).encode() == (
        REPO / "research" / "swarm" / "fixtures" / G.READMISSION_FILE).read_bytes()
    bundle = _predict_spline_journey(tmp_path / "pilot", "spline-pilot", pilot,
                                     spline_once_the_rows_are_read)
    seq, kind, form = _age_spline(bundle)
    recorded = events_or_not(pilot)
    assert (form["knots_rule"], form["n_effective"], form["knots"]) == (
        "harrell", float(recorded), harrell_k(recorded))
    _readings_change_no_row(_later(bundle, seq), pilot, G.readmission(seed + 1),
                            G.readmission(seed + 2))
    rebuilt = {}
    for new in (seed + 1, seed + 2):
        size = events_or_not(G.readmission(new))
        assert size != recorded and harrell_k(size) == harrell_k(recorded)
        report = replay.run(bundle, workers=2, cut=True, keep=True, home=str(tmp_path / str(new)),
                            fresh=replay.FreshDraw(G.readmission, seed, new))
        assert report.refused is None and report.finished is True, (report.refused,
                                                                    report.messages)
        files = _rebuilt(report)
        record = _age_record(files["decisions.jsonl"])
        assert _form(record) == {**form, "n_effective": float(size)}
        assert _effective_words(size) in record["sentence"]
        assert _effective_words(size) in files["methods.md"].decode()
        assert _effective_words(recorded) not in files["methods.md"].decode()
        assert [m for m in report.messages if "Harrell" in m] == [
            f"Record {seq} ({kind}) gives `age` a spline whose k is Harrell's rule on the "
            f"analyzed rows: k = {harrell_k(recorded)} on an effective sample size of "
            f"{recorded:,} on the recorded draw, and on this draw the rule reads {size:,} "
            f"(k = {harrell_k(size)}) on the rows as the log ends. Under prediction the record's "
            f"k is replayed as recorded, and it is the rule's on this draw too, so the record is "
            f"replayed stating this draw's size, {size:,}, in place of {recorded:,}."]
        rebuilt[new] = (record, report.estimates_again)
    direct = _predict_spline_journey(tmp_path / "direct", "spline-direct",
                                     G.readmission(seed + 1), spline_once_the_rows_are_read)
    mine = _age_record(_bundle_files(direct)["decisions.jsonl"])
    record, again = rebuilt[seed + 1]
    assert (record["decision"], record["sentence"]) == (mine["decision"], mine["sentence"])
    cv = {k: v for k, v in _recorded_estimates(direct).items() if "/cv/" in k}
    assert cv and set(cv) <= set(again)
    assert all(again[k] == pytest.approx(v, abs=1e-12) for k, v in cv.items())


# ── the findings rule ────────────────────────────────────────────────────────


def test_the_sex_repair_is_kept_where_its_finding_fires_on_the_new_draw(bundles):
    from turbotab.core.export import replay

    assert "apply_repair" in [k for _, k in _kinds(bundles["predict"])]
    report = replay.run(bundles["predict"], workers=2, cut=True,
                        fresh=replay.FreshDraw(small_readmission, PREDICT_SEED, PREDICT_SEED + 3))
    assert report.finished is True, report.messages
    assert report.left_out == [] and report.decisions == report.cut.kept
    assert report.unanswered == [] and report.redrawn == []  # its settings are the draw's own


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


def test_a_coding_the_new_draws_finding_does_not_offer_refuses_the_replay(bundles):
    from turbotab.core.export import replay

    seq = next(seq for seq, kind in _kinds(bundles["predict"]) if kind == "apply_repair")
    report = replay.run(bundles["predict"], workers=2, cut=True,
                        fresh=replay.FreshDraw(spelled_sex, PREDICT_SEED, PREDICT_SEED + 3))
    assert report.refused is not None, report.messages
    assert f"Record {seq} (apply_repair) cannot be replayed on this draw" in report.refused
    assert "`binary_text__sex` offers `level` on this draw only with other choices" in \
        report.refused
    assert report.finished is None and report.project is None and report.home is None


# ── read-numbers repairs ─────────────────────────────────────────────────────


def _finding_records(path: Path) -> list[tuple[int, str]]:
    log = _bundle_files(path)["decisions.jsonl"]
    return [(json.loads(x)["seq"], json.loads(x)["decision"]["finding_id"])
            for x in log.splitlines()
            if json.loads(x)["decision"]["kind"] in ("apply_repair", "dismiss_finding",
                                                     "defer_finding")]


def _working_column(home: Path, pid: str, column: str) -> pd.Series:
    import dataclasses

    from turbotab.core.config import Settings
    from turbotab.core.datastore import DataStore
    from turbotab.core.export.replay import _wait
    from turbotab.core.export.scorer import _bundle
    from turbotab.core.stages.working import TABLE
    from turbotab.server.service import ProjectService

    service = ProjectService(dataclasses.replace(Settings.from_env(), home=home, mode="local",
                                                 workers=2), replay=True)
    try:
        service.engine.on_decision(pid)
        _wait(service, pid, ("working",), 600)
        working = _bundle(service, pid, "working", service.engine.status(pid))
        return DataStore(Path(working.files[TABLE]), 10 ** 9).materialize([column])[column]
    finally:
        service.close()


def test_a_finding_that_fires_once_a_kept_read_numbers_repair_reads_the_text_keeps_its_record(
        bundles):
    """The record read hemoglobin's text as numbers, then marked 999 missing (a finding the text
    alone does not raise). On a new draw of the same kind both are kept, and the working table's
    hemoglobin is blank exactly where the generator wrote "." or 999."""
    from turbotab.core.export import replay

    ids = [fid for _, fid in _finding_records(bundles["readings"])]
    assert ids[0] == "text_numbers__hemoglobin_g_dl"
    assert "sentinel_missing__hemoglobin_g_dl" in ids[1:]
    seed = PREDICT_SEED + 3
    home = bundles["folder"] / "kept-readings"
    report = replay.run(bundles["readings"], workers=2, cut=True, keep=True, home=str(home),
                        fresh=replay.FreshDraw(text_hemoglobin, PREDICT_SEED, seed))
    assert report.finished is True, report.messages
    assert report.left_out == [] and report.redrawn == [] and report.unanswered == []
    hb = _working_column(home, report.project, "hemoglobin_g_dl")
    written = text_hemoglobin(seed)["hemoglobin_g_dl"]
    blank = written.isin([".", 999.0]).to_numpy()
    assert hb.isna().to_numpy().tolist() == blank.tolist() and blank.sum() > 0
    assert not (hb == 999).any()
    assert np.array_equal(hb.to_numpy()[~blank], written[~blank].astype(float).to_numpy())


def test_where_the_read_numbers_finding_does_not_fire_both_repairs_are_left_out(bundles):
    from turbotab.core.export import replay

    records = dict((fid, seq) for seq, fid in _finding_records(bundles["readings"]))
    report = replay.run(bundles["readings"], workers=2, cut=True,
                        fresh=replay.FreshDraw(numbers_at_new_seeds, PREDICT_SEED,
                                               PREDICT_SEED + 3))
    assert report.finished is True, report.messages
    assert [(o.seq, o.finding_id) for o in report.left_out] == sorted(
        (records[f], f) for f in ("text_numbers__hemoglobin_g_dl",
                                  "sentinel_missing__hemoglobin_g_dl"))
    assert all("does not fire on this draw" in o.why for o in report.left_out)
    assert report.decisions == report.cut.kept - 2


# ── QC-RLSC on the measurement fixture ───────────────────────────────────────


def test_a_qc_rlsc_repair_is_replayed_with_the_new_draws_dropped_features(tmp_path):
    """Recorded on the panel's committed draw: the engine's own per-batch QC-RLSC offer, answered
    as offered, and the sex repair. On the next seed's draw the per-batch form drops other
    features, so the record is replayed with that draw's offered settings (the findings rule run on
    a project made from it); every feature only that draw's correction drops leaves the
    predictors, which the recorded settings kept. The sex repair is its own bytes."""
    import dataclasses

    from turbotab.core import decisions
    from turbotab.core.config import Settings
    from turbotab.core.export import replay
    from turbotab.core.methods.qc_drift import FINDING
    from turbotab.core.repairs import findings_of
    from turbotab.server.service import ProjectService

    def per_batch(service, pid) -> dict:
        replay._wait(service, pid, ("findings",), 900)
        found = {f["id"]: f for f in findings_of(service.engine.get(pid, "findings").artifact)}
        [form] = [o for o in found[FINDING]["repairs"]
                  if o["key"] == "qc_rlsc_lc" and o["decision"]["params"].get("batch_column")]
        return {"params": form["decision"]["params"], "sex": found["binary_text__sex"]}

    def dropped(params: dict) -> set[str]:
        return {c for v in (params.get("dropped") or {}).values() for c in v}

    tables = {}
    for name, seed in (("first", G.PANEL_SEED), ("second", G.PANEL_SEED + 1)):
        tables[name] = tmp_path / name / G.PANEL_FILE
        tables[name].parent.mkdir()
        G.metabolite_panel(seed).to_csv(tables[name], index=False)
    service = ProjectService(dataclasses.replace(Settings.from_env(), home=tmp_path / "home",
                                                 mode="local", workers=2), replay=True)
    try:
        first = service.workspace.create_project("first", str(tables["first"]), "path").id
        service.decide(first, {"kind": "set_lens", "lenses": ["metabolomics"]})
        service.decide(first, {"kind": "set_orientation", "orientation": "sample_major"})
        offer = per_batch(service, first)
        service.decide(first, {"kind": "apply_repair", "finding_id": FINDING,
                               "option": "qc_rlsc_lc", "params": offer["params"]})
        sex = offer["sex"]["repairs"][0]
        service.decide(first, {"kind": "apply_repair", "finding_id": "binary_text__sex",
                               "option": sex["key"], "params": sex["decision"]["params"]})
        lines = replay.log_lines(service.workspace.decisions_path(first).read_bytes())
        second = service.workspace.create_project("second", str(tables["second"]), "path").id
        written = service.workspace.decisions_path(second)
        ruled, _ = replay.fresh_findings(service, second, written, lines, 900)
        written.write_bytes(replay.join_lines(ruled.kept))
        service.engine.on_decision(second)
        again = per_batch(service, second)
    finally:
        service.close()
    qc_seq = next(r.seq for r, _ in lines if getattr(r.decision, "finding_id", None) == FINDING)
    assert ruled.left_out == [] and ruled.unmatched == []
    [redrawn] = ruled.redrawn
    assert (redrawn.seq, redrawn.finding_id, redrawn.option) == (qc_seq, FINDING, "qc_rlsc_lc")
    assert "dropped" in redrawn.settings
    by_seq = {r.seq: (r, line) for r, line in ruled.kept}
    assert json.loads(by_seq[qc_seq][1])["decision"]["params"] == again["params"]
    recorded = {r.seq: line for r, line in lines}
    assert all(line == recorded[seq] for seq, (_, line) in by_seq.items() if seq != qc_seq)
    only_new = dropped(again["params"]) - dropped(offer["params"])
    assert only_new  # the two draws' corrections drop different features (the premise)
    out_then = decisions.left_out(decisions.fold([r for r, _ in lines]))
    out_now = decisions.left_out(decisions.fold([r for r, _ in ruled.kept]))
    assert only_new <= set(out_now) and not only_new & set(out_then)


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

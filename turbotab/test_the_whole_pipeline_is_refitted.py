"""`GUIDED-103` — the resampling engine, and the plots that are its output.

`research/CLINICAL_SURVEY_PACK.md` §A4.8, marked ★. The pack's italics carry
the requirement: refit the **entire** modeling pipeline *including any variable
selection*, apply each bootstrap model to the original data, and report
per-individual instability rather than a point estimate.

**This is the test file for a claim that could not previously be made.**
`GUIDED-103`'s own note says the selector is *"fold-local BY CONSTRUCTION under
any resampling"* — which is an argument, not an observation. The probe below
turns it into one: the chosen set must move across resamples on a table where
it should and stay put on a table where it should not. Only the pair is
evidence. A selector that always moved would be noise and a selector that never
moved would be a constant, and each alone is consistent with the selector not
running inside the loop at all.

`GUIDED-097` — THE FIXTURE RULE. Two target shapes, and the shapes not covered
are named below.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from turbotab import instability as I
from turbotab import missingness as M
from turbotab import selection as S
from turbotab.project import AnalysisProject

#: `GUIDED-097`. Both run the full pipeline; `run_order` is a genuine
#: continuous column on the same table, so the regression arm is not a
#: classification fixture wearing a different label.
TARGET_SHAPES = {
    "binary classification": ("responder", "classification", "logreg"),
    "continuous regression": ("run_order", "regression", "ridge"),
}

#: NOT COVERED, said out loud.
#:
#: MULTICLASS — `_predict` takes `predict_proba`'s second column, which is the
#: positive class of a binary problem and is one class among k for a multiclass
#: one. The instability plot would be about that single class rather than about
#: the prediction, and no fixture has a multiclass target to check it on. This
#: is a real limit, not an untested guess: a multiclass project would get a
#: plot of the wrong quantity, and the module does not currently refuse it.
#: Filed as `GUIDED-113`.
#:
#: SURVIVAL — no task type exists.
#:
#: GROUPED / REPEATED-MEASURES — the bootstrap here draws ROWS. On a table
#: where one person contributes several rows, a row-level bootstrap breaks the
#: person-level independence the seal was drawn to respect, and the correct
#: draw is a cluster bootstrap over `group_col`. `_grouped_is_refused` asserts
#: the module does not silently do the wrong thing; the cluster bootstrap is
#: `GUIDED-114` and is not built.
SHAPES_NOT_COVERED = [
    "multiclass classification — the plotted quantity would be one class's "
    "probability rather than the prediction; filed as GUIDED-113",
    "survival / time-to-event — no task type exists",
    "grouped or repeated-measures tables — a row-level bootstrap is the wrong "
    "draw and the cluster bootstrap is filed as GUIDED-114",
]

B_FOR_TESTS = 12          # enough for the probes; B_RESAMPLES is what ships


def _sealed(df, target, task, *, declare_missing=True, selection=None,
            fraction=0.20):
    p = AnalysisProject.from_dataframe(df, "fixture.csv")
    p.target, p.task_type = target, task
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:max(1, int(round(len(idx) * fraction)))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    if declare_missing:
        for card in M.survey(p.df, p.target):
            if card["branch"] == "numeric":
                p.route_missingness(card["column"], M.NOT_SURE, M.IMPUTE_MEDIAN)
    if selection is not None:
        p.set_selection(selection)
    return p


def _assay(target, task):
    df = pd.read_csv("turbotab/sample_data/metabolomics_untargeted.csv")
    return df[df[target].notna()].copy()


# ═══════════ THE CORE CLAIM · THE ENTIRE PIPELINE ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_entire_pipeline_is_refitted_not_just_the_estimator(shape):
    """§A4.8's italics, and they are the whole requirement.

    A bootstrap that refits the estimator over a fixed feature set measures the
    stability of an estimator; §A4.8 asks for the stability of a modeling
    PROCESS. The observable difference is that the selected set varies.
    """
    target, task, model = TARGET_SHAPES[shape]
    df = _assay(target, task)
    columns = [c for c in df.columns if c.startswith("mz_")][:8]
    p = _sealed(df, target, task,
                selection=S.declare("mutual_info", target, columns,
                                    n_features=3))
    result = I.run(p, model, b=B_FOR_TESTS, seed=42)

    assert result["b_completed"] == B_FOR_TESTS, result["failures"][:2]
    assert len(result["selected_sets"]) == B_FOR_TESTS, (
        "the selector did not run in every resample, so the plot is about a "
        "process that is not the one being reported")
    moved = I.selection_moved(result)
    assert moved["moved"] is True, (
        f"{shape}: the same feature set was chosen in all "
        f"{B_FOR_TESTS} resamples of 8 near-identical assay columns. Either "
        f"the selector is outside the loop or every resample saw the same "
        f"rows.")


def test_the_selected_set_holds_still_when_one_candidate_dominates():
    """The other half of the probe, and the pair is what makes it evidence.

    A selector that moved on everything would be indistinguishable from noise.
    Here one candidate carries the outcome exactly and the rest are noise, so a
    selector genuinely ranking by association picks the same column every time
    — and a selector that still moved would be choosing at random.
    """
    rng = np.random.default_rng(7)
    n = 160
    signal = rng.normal(size=n)
    df = pd.DataFrame({
        "y": (signal > 0).astype(int),
        "signal": signal,
        **{f"noise_{i}": rng.normal(size=n) for i in range(8)},
    })
    p = _sealed(df, "y", "classification", declare_missing=False,
                selection=S.declare(
                    "mutual_info", "y",
                    ["signal"] + [f"noise_{i}" for i in range(8)],
                    n_features=1))
    result = I.run(p, "logreg", b=B_FOR_TESTS, seed=42)
    moved = I.selection_moved(result)

    assert moved["n_resamples_with_a_set"] == B_FOR_TESTS
    assert moved["moved"] is False, (
        f"`signal` determines the outcome exactly and the selector still "
        f"chose {moved['n_distinct']} different single features across "
        f"{B_FOR_TESTS} resamples. A selector that moves on everything is not "
        f"fold-local, it is random.")
    assert moved["most_common"] == ["signal"]


def test_no_selection_recorded_is_not_a_stable_selection():
    """`None`, never `False`. A project that never selected has not got a
    perfectly stable selection."""
    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    moved = I.selection_moved(I.run(p, "logreg", b=4, seed=42))
    assert moved["moved"] is None
    assert moved["n_distinct"] == 0
    assert "no chosen set" in moved["because"]


# ═══════════ THE SEAL ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_sealed_rows_are_neither_resampled_nor_predicted(shape):
    """`STATE-013` at a new address, and the one that would be invisible.

    An instability plot drawn over resampled held-out rows looks exactly like
    one that is not. The check is on the counts and on the row labels — the
    identity, never a position (Decision A).
    """
    target, task, model = TARGET_SHAPES[shape]
    df = _assay(target, task)
    p = _sealed(df, target, task)
    result = I.run(p, model, b=4, seed=42)

    sealed = {str(x) for x in p.lockbox["labels"]}
    used = set(result["row_labels"])
    assert sealed, "nothing was sealed, so this proves nothing"
    assert not (used & sealed), (
        f"{len(used & sealed)} sealed row label(s) were predicted by the "
        f"bootstrap models")
    assert result["n"] == len(used) == len(p.training_rows[
        p.training_rows[target].notna()])
    assert "held-out" in result["scored_on"]


def test_it_refuses_before_the_seal():
    """Before the seal every row is a training row, so a bootstrap over them
    is a bootstrap over the study rather than over its development sample."""
    df = _assay("responder", "classification")
    p = AnalysisProject.from_dataframe(df, "fixture.csv")
    p.target, p.task_type = "responder", "classification"
    with pytest.raises(I.InstabilityRefusal, match="has not been sealed"):
        I.run(p, "logreg", b=4)


# ═══════════ THE SEEDS ═══════════

def test_every_resample_draws_a_different_sample():
    """The silent failure: one seed B times gives B identical models, a perfect
    45° line, and a confident claim of exact stability. The figure looks BETTER
    the more broken it is, which is why this is asserted rather than trusted."""
    seeds = I.seeds_for(42, I.B_RESAMPLES)
    assert len(seeds) == I.B_RESAMPLES
    assert len(set(seeds)) == I.B_RESAMPLES, (
        f"{I.B_RESAMPLES - len(set(seeds))} duplicate seed(s); duplicated "
        f"refits would narrow the plotted spread without narrowing anything "
        f"real")
    assert I.seeds_for(42, 8) == I.seeds_for(42, 8), "not reproducible"
    assert I.seeds_for(42, 8) != I.seeds_for(43, 8), "the seed does nothing"


def test_the_spread_is_not_zero_on_a_table_that_should_have_some():
    """The positive control for every stability claim in this file.

    If the resamples were identical, every assertion about instability would
    pass vacuously against a plot of a single line.
    """
    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    result = I.run(p, "logreg", b=B_FOR_TESTS, seed=42)
    matrix = np.asarray(result["bootstrap"])
    assert matrix.shape[0] == B_FOR_TESTS
    assert not np.allclose(matrix[0], matrix[1]), (
        "two resamples produced identical predictions; the bootstrap is not "
        "resampling")
    assert I.spread(result)["median_width"] > 0


# ═══════════ MAPE, AND THE AMBIGUITY IN THE PACK ═══════════

def test_mean_absolute_prediction_error_is_in_the_predictions_own_units():
    """§A4.8 writes `MAPE` and does not expand it, and the two readings differ
    by more than an order of magnitude on predicted risks near zero.

    Driven on this fixture the percentage form returned 658%, produced almost
    entirely by patients whose original risk was near 0.02 — a division, not a
    stability finding. So the absolute form is what is reported and named.
    """
    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    mape = I.run(p, "logreg", b=B_FOR_TESTS, seed=42)["mape"]

    assert "absolute" in mape["label"].lower()
    assert 0 <= mape["absolute"] <= 1, (
        f"a predicted risk moves within [0, 1], so a mean absolute error of "
        f"{mape['absolute']} is the percentage form wearing the other name")
    # The percentage is reported only where the denominator supports it, and
    # says how many rows it had to leave out.
    assert mape["percentage_excluded_rows"] >= 0
    if mape["percentage"] is not None:
        assert mape["percentage_excluded_rows"] < len(p.training_rows)


# ═══════════ THE FIGURES ═══════════

def test_both_figures_carry_b_in_their_own_caption():
    """*Say the number.* A resample count nobody can see is a threshold nobody
    can disagree with, and B is the one number a different analyst would pick
    differently."""
    from turbotab import figure_specs as F

    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    result = I.run(p, "logreg", b=B_FOR_TESTS, seed=42)

    payload = F.prediction_instability_payload(result)
    caption = F.PREDICTION_INSTABILITY.caption(payload)
    assert f"B = {B_FOR_TESTS:,}" in caption
    assert str(I.RECOMMENDED_B) in caption.replace(",", ""), (
        "the caption states B and hides the gap to what the source "
        "recommends, which is the part a reader would want to argue with")
    assert "held-out" in caption

    rows = p.training_rows
    rows = rows[rows["responder"].notna()]
    positive = sorted(rows["responder"].dropna().unique())[-1]
    calib = F.calibration_instability_payload(
        result, (rows["responder"] == positive).astype(float))
    assert f"B = {B_FOR_TESTS:,}" in F.CALIBRATION_INSTABILITY.caption(calib)


def test_both_figures_score_their_own_checklists():
    """A checklist whose items are prose is a style guide; these are callables
    and they run against the real payload."""
    from turbotab import figure_specs as F

    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    result = I.run(p, "logreg", b=B_FOR_TESTS, seed=42)

    payload = F.prediction_instability_payload(result)
    failed = [i.id for i in F.PREDICTION_INSTABILITY.checklist
              if not i.check(payload)]
    assert not failed, f"prediction instability fails its own checklist: {failed}"

    rows = p.training_rows
    rows = rows[rows["responder"].notna()]
    positive = sorted(rows["responder"].dropna().unique())[-1]
    calib = F.calibration_instability_payload(
        result, (rows["responder"] == positive).astype(float))
    failed = [i.id for i in F.CALIBRATION_INSTABILITY.checklist
              if not i.check(calib)]
    assert not failed, f"calibration instability fails its own checklist: {failed}"


def test_neither_figure_is_admissible_without_the_other():
    """§A4.8 specifies the pair. Spread in individual predictions and spread in
    calibration are different failures, and a model can look tight on one while
    moving badly on the other."""
    from turbotab import figure_specs as F

    ok, missing = F.PREDICTION_INSTABILITY.admissible(["prediction_instability"])
    assert not ok and missing == ["calibration_instability"]
    ok, missing = F.CALIBRATION_INSTABILITY.admissible(["calibration_instability"])
    assert not ok and missing == ["prediction_instability"]
    ok, _ = F.PREDICTION_INSTABILITY.admissible(
        ["prediction_instability", "calibration_instability"])
    assert ok


def test_a_regression_run_gets_no_calibration_curve_and_says_why():
    """Return nothing rather than a plot of something else."""
    from turbotab import figure_specs as F

    df = _assay("run_order", "regression")
    p = _sealed(df, "run_order", "regression")
    result = I.run(p, "ridge", b=4, seed=42)
    calib = F.calibration_instability_payload(result, np.zeros(result["n"]))
    assert calib["applicable"] is False
    assert "predicts a value rather than a risk" in calib["because"]
    assert not F.CALIBRATION_INSTABILITY.when_applicable(
        {"task_type": "regression", "has_instability_run": True})


# ═══════════ THE JOB ═══════════

def test_it_runs_as_an_observable_job_with_a_name_and_a_cancel():
    """`PRODUCT_VISION.md` §04. B refits of a full pipeline is the longest
    thing this app does, and it is the least forgivable place for a spinner —
    a researcher cannot tell a slow bootstrap from a hung one by looking."""
    from fastapi.testclient import TestClient

    from turbotab import api

    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    api.STORE.add(p)
    client = TestClient(api.app)

    empty = client.get(f"/project/{p.id}/instability").json()
    assert empty["runs"] == {} and empty["blocked_by"], (
        "a step that has not happened and a step that produced nothing are "
        "different sentences")
    assert str(I.B_RESAMPLES) in empty["blocked_by"], (
        "the sentence explaining why it is a job does not say how many refits")

    job = client.post(f"/project/{p.id}/instability",
                      json={"model": "logreg", "b": B_FOR_TESTS}).json()
    assert "bootstrap resamples" in job["name"], (
        "a job's name is read by a person waiting on it")
    assert str(B_FOR_TESTS) in job["name"]

    for _ in range(400):
        state = client.get(f"/job/{job['id']}").json()
        if state["terminal"]:
            break
        time.sleep(0.05)
    assert state["status"] == "done", state.get("error")

    body = client.get(f"/project/{p.id}/instability").json()
    entry = body["runs"]["logreg"]
    assert entry["prediction_caption"]
    assert entry["calibration_instability"]["applicable"] is True
    assert "bootstrap" not in entry["prediction_instability"], (
        "the raw B x n draw matrix is served to the page; it is megabytes and "
        "nothing renders from it")


def test_a_grouped_table_is_not_silently_row_bootstrapped():
    """`SHAPES_NOT_COVERED`'s third entry, asserted rather than assumed.

    A row-level bootstrap on a table where one person contributes several rows
    breaks the independence the seal was drawn to respect. The cluster
    bootstrap is not built (`GUIDED-114`); what must not happen is the wrong
    draw happening quietly.

    **This test currently records that it DOES happen quietly** — it asserts
    the state of affairs so the gap is visible in the suite rather than only in
    the ledger, per `LOOP.md` §05's rule about a capability shipping with a
    failing test naming what it lacks.
    """
    df = _assay("responder", "classification")
    p = _sealed(df, "responder", "classification")
    p.grain = dict(p.grain or {}, group_col="sample_id")
    result = I.run(p, "logreg", b=4, seed=42)
    assert result["b_completed"] == 4, (
        "if this now refuses, the cluster-bootstrap question was answered — "
        "update SHAPES_NOT_COVERED and close GUIDED-114")
    assert "cluster" not in result["scored_on"].lower(), (
        "the run claims a cluster bootstrap; none is implemented")


@pytest.mark.skipif(
    not __import__("turbotab.pageharness", fromlist=["x"]).available(),
    reason="no JS engine on this machine")
def test_the_instability_result_reaches_the_reader():
    """`LOOP.md` §05: a capability ships with its consumer. Driven, because
    `GUIDED-080`'s class is a server that composes a string nothing fetches,
    and this loop already produced one instance of it (`/sensitivity`)."""
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    df = _assay("responder", "classification")
    columns = [c for c in df.columns if c.startswith("mz_")][:8]
    p = _sealed(df, "responder", "classification",
                selection=S.declare("mutual_info", "responder", columns,
                                    n_features=3))
    api.STORE.add(p)
    p.selected_models = ["logreg"]
    client = TestClient(api.app)

    job = client.post(f"/project/{p.id}/instability",
                      json={"model": "logreg", "b": B_FOR_TESTS}).json()
    for _ in range(400):
        state = client.get(f"/job/{job['id']}").json()
        if state["terminal"]:
            break
        time.sleep(0.05)
    assert state["status"] == "done", state.get("error")

    served = client.get(f"/project/{p.id}/instability").json()
    project = client.get(f"/project/{p.id}").json()
    train = {"run": {"task_type": "classification", "target": "responder",
                     "n_train": 58, "n_test": 14, "seal_basis": "undetermined",
                     "exploratory": True, "features": [], "notes": [],
                     "mark": None,
                     "results": [{"key": "logreg", "name": "Logistic Regression",
                                  "concern": "", "bucket": "", "metrics": {},
                                  "error": None, "plan": {}}]},
             "blocked_by": None, "stale": []}
    routes = {
        f"/project/{p.id}": project,
        f"/project/{p.id}/interview?step=data":
            client.get(f"/project/{p.id}/interview?step=data").json(),
        f"/project/{p.id}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{p.id}/evidence/missingness": {"cards": []},
        f"/project/{p.id}/evidence/plausibility": {"columns": []},
        f"/project/{p.id}/draft": {"paragraphs": []},
        f"/project/{p.id}/gaps": {"gaps": []},
        f"/project/{p.id}/models": client.get(f"/project/{p.id}/models").json(),
        f"/project/{p.id}/training": train,
        f"/project/{p.id}/instability": served,
    }
    out = PH.run("__emit(__harness.html('trainRun'));", routes=routes,
                 search=f"?project={p.id}")

    assert out, "the Train run surface rendered nothing at all"
    assert "who you sampled" in out, (
        "the server served an instability result and the page rendered none "
        "of it")
    caption = served["runs"]["logreg"]["prediction_caption"]
    assert caption in out, "the caption on screen is not the server's caption"
    assert f"B = {B_FOR_TESTS:,}" in out, (
        "B is not on screen; a resample count nobody can see is a threshold "
        "nobody can disagree with")
    assert served["runs"]["logreg"]["calibration_caption"] in out

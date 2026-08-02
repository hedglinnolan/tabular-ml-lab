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
#: GROUPED / REPEATED-MEASURES IS COVERED AS OF L39 (`GUIDED-114`) and is
#: named here so the change is visible: the draw is a cluster bootstrap over
#: the recorded `group_col`, and the scheme is disclosed either way.
#: `clinical_longitudinal.csv` is the fixture, with the repeat chain walked
#: rather than fabricated.
SHAPES_NOT_COVERED = [
    "multiclass classification — the plotted quantity would be one class's "
    "probability rather than the prediction; filed as GUIDED-113",
    "survival / time-to-event — no task type exists",
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


def _longitudinal():
    """A genuinely grouped project: `clinical_longitudinal.csv` has ~2.4 rows
    per `subject_id`, and the repeat chain is WALKED rather than fabricated —
    constitution §01 puts those questions before the seal, and a test that set
    `grain` directly would be testing a project the app cannot produce."""
    from turbotab import repeats as R

    df = pd.read_csv("turbotab/sample_data/clinical_longitudinal.csv")
    df = df[df["sbp"].notna()].copy()
    p = AnalysisProject.from_dataframe(df, "clinical_longitudinal.csv")
    p.target, p.task_type = "sbp", "regression"
    p.set_grain("people_repeat", group_col="subject_id")
    p.set_repeat_kind(R.TIME_POINTS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.20))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    return p


def test_a_grouped_table_draws_whole_groups():
    """`GUIDED-114`. The seal and the bootstrap disagreed about what a row is.

    Constitution §02 makes the grain a precondition of the seal precisely
    because whether one person appears in several rows decides how held-out
    rows are chosen — and the resampling then ignored that answer.
    """
    p = _longitudinal()
    rows = p.training_rows
    rows = rows[rows["sbp"].notna()]
    scheme = I.scheme_for(p, rows)

    assert scheme["scheme"] == I.CLUSTER_BOOTSTRAP
    assert scheme["group_col"] == "subject_id"
    assert scheme["n_groups"] == rows["subject_id"].nunique()
    assert scheme["n_groups"] < len(rows), (
        "the fixture has one row per subject, so it cannot exercise clustering")

    result = I.run(p, "ridge", b=6, seed=42)
    assert result["sampling"]["scheme"] == I.CLUSTER_BOOTSTRAP
    # THE SIZE VARIES, and that is the cluster bootstrap rather than a defect:
    # groups have unequal numbers of rows. Forcing a fixed size would mean
    # truncating whole people, which breaks the independence assumption again
    # from the other side.
    assert result["sampling"]["rows_drawn_min"] != \
        result["sampling"]["rows_drawn_max"]


def test_the_draw_takes_every_row_of_each_chosen_group():
    """Driven on the draw itself, because *drew clusters* is a claim about
    which rows were in the sample and nothing downstream can show it."""
    p = _longitudinal()
    rows = p.training_rows
    rows = rows[rows["sbp"].notna()]
    scheme = I.scheme_for(p, rows)
    draw = I._draw(np.random.default_rng(7), rows, scheme)

    taken = rows.iloc[draw]
    sizes = rows.groupby("subject_id").size()
    for subject, n in taken.groupby("subject_id").size().items():
        assert n % sizes[subject] == 0, (
            f"subject {subject} appears {n} times and has {sizes[subject]} "
            f"rows; a cluster draw takes whole groups or none")


def test_the_row_bootstrap_understates_the_spread_and_that_is_measured():
    """The finding's own claim, turned into a number.

    `GUIDED-114`: *the instability plot understates the spread — a figure that
    errs toward reassurance.* This measures the understatement rather than
    asserting it, and the direction is the assertion: a row bootstrap on a
    grouped table must not report MORE spread than a cluster draw, because the
    whole reason to prefer the cluster draw is that rows within a subject are
    not exchangeable.
    """
    p = _longitudinal()
    cluster = I.spread(I.run(p, "ridge", b=30, seed=42))["median_width"]

    recorded = p.grain
    try:
        # The pre-L39 behavior, forced rather than simulated.
        p.grain = dict(recorded, group_col=None)
        row = I.spread(I.run(p, "ridge", b=30, seed=42))["median_width"]
    finally:
        p.grain = recorded

    assert row < cluster, (
        f"the row bootstrap reported {row:.4f} median interval width and the "
        f"cluster draw {cluster:.4f}. The row draw is expected to be NARROWER "
        f"— it pulls the same subject in repeatedly, so the refits agree more "
        f"than independent samples would.")
    assert (cluster - row) / cluster > 0.05, (
        f"the two schemes differ by less than 5% ({row:.4f} vs {cluster:.4f}), "
        f"so this fixture does not exercise the difference and the test is "
        f"not evidence of anything")


def test_the_scheme_is_disclosed_whichever_one_was_drawn():
    """**The half of `GUIDED-114` that is not about sampling at all.**

    The adjudicator drove the old payload on a grouped project: 141,126
    characters across eighteen keys containing `group`, `cluster`, `subject`,
    `person`, `understate` and `repeated` ZERO times. Drawing clusters without
    saying so would fix the number and leave the silence, and `GUIDED-089`'s
    accepted precedent is the opposite — the trainer could not honor the
    recorded plan and every run said so in its own notes.
    """
    import json

    grouped = I.run(_longitudinal(), "ridge", b=4, seed=42)
    blob = json.dumps(grouped).lower()
    for word in ("group", "cluster", "subject"):
        assert word in blob, (
            f"a grouped project's instability payload never says {word!r}")
    assert grouped["sampling"]["sentence"]

    # AND ON AN UNGROUPED ONE, because a disclosure that appears only when
    # something is wrong is a warning, and the reader of an ungrouped plot is
    # entitled to know the scheme too.
    plain = I.run(_sealed_assay(), "logreg", b=4, seed=42)
    assert plain["sampling"]["scheme"] == I.ROW_BOOTSTRAP
    assert plain["sampling"]["understates"] is False
    assert "independent observation" in plain["sampling"]["sentence"]


def test_a_recorded_group_column_that_is_not_in_the_frame_says_so():
    """Return nothing rather than a wrong value — here, say the bound rather
    than imply an estimate. If the grain names a column the training rows do
    not have, the cluster draw is impossible and the row draw understates; the
    honest output is the number plus the words LOWER BOUND."""
    p = _longitudinal()
    p.grain = dict(p.grain, group_col="a_column_that_is_not_here")
    result = I.run(p, "ridge", b=4, seed=42)

    assert result["sampling"]["scheme"] == I.ROW_BOOTSTRAP
    assert result["sampling"]["understates"] is True
    assert "LOWER BOUND" in result["sampling"]["sentence"]


def _sealed_assay():
    df = _assay("responder", "classification")
    return _sealed(df, "responder", "classification")


def test_both_figures_carry_the_sampling_sentence():
    """The disclosure has to reach the FIGURE, not just the payload: the plot
    looks identical under either scheme, so the caption is where a reader meets
    the difference."""
    from turbotab import figure_specs as F

    p = _longitudinal()
    result = I.run(p, "ridge", b=4, seed=42)
    payload = F.prediction_instability_payload(result)
    caption = F.PREDICTION_INSTABILITY.caption(payload)

    assert "subject_id" in caption
    assert "whole subject_ids" in caption
    assert not [i.id for i in F.PREDICTION_INSTABILITY.checklist
                if not i.check(payload)]


@pytest.mark.skipif(
    not __import__("turbotab.pageharness", fromlist=["x"]).available(),
    reason="no JS engine on this machine")
def test_the_instability_result_reaches_the_reader():
    """`LOOP.md` §05: a capability ships with its consumer. Driven, because
    `GUIDED-080`'s class is a server that composes a string nothing fetches."""
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    p = _longitudinal()
    api.STORE.add(p)
    p.selected_models = ["ridge"]
    client = TestClient(api.app)

    job = client.post(f"/project/{p.id}/instability",
                      json={"model": "ridge", "b": 6}).json()
    for _ in range(600):
        state = client.get(f"/job/{job['id']}").json()
        if state["terminal"]:
            break
        time.sleep(0.05)
    assert state["status"] == "done", state.get("error")

    served = client.get(f"/project/{p.id}/instability").json()
    project = client.get(f"/project/{p.id}").json()
    train = {"run": {"task_type": "regression", "target": "sbp",
                     "n_train": 480, "n_test": 120, "seal_basis": "grouped",
                     "exploratory": False, "features": [], "notes": [],
                     "mark": None,
                     "results": [{"key": "ridge", "name": "Ridge",
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
    assert "who you sampled" in out
    assert served["runs"]["ridge"]["prediction_caption"] in out
    assert "B = 6" in out
    # `GUIDED-114`. THE DISCLOSURE REACHES THE READER, not just the payload.
    sentence = served["runs"]["ridge"]["prediction_instability"]["sampling"]["sentence"]
    assert sentence in out, (
        "the server said which sampling scheme it drew and the page rendered "
        "none of it — the plot looks identical under either, so this sentence "
        "is the only place a reader meets the difference")
    assert "subject_id" in out

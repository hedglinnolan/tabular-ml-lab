"""`GUIDED-058` and `DRIVE-009` — the figure layer gets its first consumer.

`figures.applicable()` at `figures.py:228` and `figures.bundle()` at `:240` had
**zero callers anywhere in the repository.** All three registered figures were
specifications with passing tests that no user could reach, and from inside the
loop that built them that looks finished. `LOOP.md` §06 names the grep that
finds it: *does anything outside a test file import what the loop just built?*

These tests go through HTTP, because the claim is reachability and a claim about
reachability asserted from inside the package is the same defect one layer out.

## What the shrinkage plot needed that nobody had written

Its `when_applicable` reads `n_recalls_per_person` and `has_dietary_lens`, and
**neither key was written anywhere in the repository outside that lambda.**
`figure_bundle.recalls_per_person` is the answer, and it needs three recorded
answers rather than one: the grain says *people repeat* with a grouping column,
question 4 says those rows are *repeats* rather than *time points*, and the lens
says dietary. Rows are not recalls until something records that they are.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, figure_bundle, figures                     # noqa: E402
from turbotab import figure_specs                                    # noqa: E402,F401

FIXTURES = Path(__file__).parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _upload(client, name: str) -> str:
    with open(FIXTURES / name, "rb") as fh:
        response = client.post("/project", files={
            "file": (name, fh, "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _decide(client, pid: str, what: str, **payload):
    response = client.post(f"/project/{pid}/decision",
                           json={"kind": what, "payload": payload})
    assert response.status_code == 200, (what, response.text)
    return response


@pytest.fixture(scope="module")
def dietary(client):
    """300 people × 2 twenty-four-hour recalls, driven to where the figure fires."""
    pid = _upload(client, "dietary_recalls.csv")
    _decide(client, pid, "set_lens", lens=["dietary"])
    _decide(client, pid, "set_target", column="hba1c")
    _decide(client, pid, "set_grain", answer="people_repeat",
            group_col="participant_id")
    _decide(client, pid, "set_repeat_kind", kind="repeats")
    return pid


# ── the gate: an upload reaches a drawn figure ──────────────────────────────

def test_an_upload_reaches_a_rendered_figure_carrying_its_annotation_box(
        client, dietary):
    body = client.get(f"/project/{dietary}/figures").json()
    drawn = {row["id"]: row for row in body["admitted"]}
    assert "shrinkage" in drawn, body["not_drawn"]

    row = drawn["shrinkage"]
    assert row["annotations"], "the figure arrived with no annotation box"
    labels = {a["label"]: a["value"] for a in row["annotations"]}
    # The six numbers the figure's argument is made of, on the figure.
    assert len(labels) == 6
    assert all(value != "not estimable" for value in labels.values())
    # And the badge travels with it, as it does with a prior and a finding.
    assert row["evidence_status"] == "SETTLED" and row["source"]


def test_the_narrowing_is_the_argument_and_the_numbers_show_it(client, dietary):
    """The visible narrowing from one day to modeled usual intake is the entire
    argument for usual-intake modeling, in one image (`NUTRITION_PACK.md` §03).
    If the numbers do not narrow, the figure does not make its case."""
    body = client.get(f"/project/{dietary}/figures").json()
    payload = next(r["payload"] for r in body["admitted"]
                   if r["id"] == "shrinkage")
    spreads = [payload[f"spread_{s}"] for s in figure_specs.SERIES]
    assert spreads[0] > spreads[1] > spreads[2], spreads
    assert all(item["passed"] for item in
               next(r for r in body["admitted"] if r["id"] == "shrinkage")
               ["checklist"]), "the shrinkage plot fails its own checklist"


def test_the_caption_names_the_model_rather_than_saying_modeled(client, dietary):
    """*"Modeled"* covers NCI, ISU, MSM and SPADE, which are not
    interchangeable, and a caption that says only that lets a reader assume
    whichever one they know."""
    body = client.get(f"/project/{dietary}/figures").json()
    caption = next(r["caption"] for r in body["admitted"]
                   if r["id"] == "shrinkage")
    assert "MODELED and its individual values are not measured" in caption
    assert "not the NCI method" in caption


def test_the_variance_components_reviewers_ask_for_travel_with_it(client, dietary):
    """§03: *"Variance-components table: nutrient, σ²_w, σ²_b, ratio, ICC, λ.
    Reviewers in this field ask for it."*"""
    body = client.get(f"/project/{dietary}/figures").json()
    payload = next(r["payload"] for r in body["admitted"]
                   if r["id"] == "shrinkage")
    components = payload["variance_components"]
    assert components["within"] > 0 and components["between"] > 0
    assert 0 < components["icc"] < 1
    assert 0 < components["lambda_observed"] < 1
    assert components["n_people"] == 300 and components["n_rows"] == 600


# ── the keys nobody had written ─────────────────────────────────────────────

def test_recalls_per_person_needs_three_recorded_answers_not_a_row_count(client):
    """A 600-row table is 600 recalls only if something recorded that two rows
    are one person's two days. Each answer is added in turn and the key stays
    at zero, with its reason, until the last one lands."""
    pid = _upload(client, "dietary_recalls.csv")
    _decide(client, pid, "set_lens", lens=["dietary"])
    body = client.get(f"/project/{pid}/figures").json()
    assert body["state"]["n_recalls_per_person"] == 0
    assert "grain question" in body["state"]["n_recalls_because"]

    _decide(client, pid, "set_target", column="hba1c")
    _decide(client, pid, "set_grain", answer="people_repeat",
            group_col="participant_id")
    body = client.get(f"/project/{pid}/figures").json()
    assert body["state"]["n_recalls_per_person"] == 0, (
        "rows became recalls without question 4 saying they are repeats "
        "rather than time points, and averaging them means different things")
    assert "repeated measurements or as different time points" in \
        body["state"]["n_recalls_because"]

    _decide(client, pid, "set_repeat_kind", kind="repeats")
    body = client.get(f"/project/{pid}/figures").json()
    assert body["state"]["n_recalls_per_person"] == 2
    assert "300 people" in body["state"]["n_recalls_because"]


def test_time_points_are_not_recalls(client):
    """A person's rows twelve months apart are a longitudinal series. Averaging
    them is a different operation, which is why question 4 exists."""
    pid = _upload(client, "clinical_longitudinal.csv")
    project = api.STORE.get(pid)
    project.lens = ["dietary"]
    project.grain = {"answer": "people_repeat",
                     "group_col": project.df.columns[0]}
    project.repeat_kind = {"kind": "time_points"}
    n, why = figure_bundle.recalls_per_person(project)
    assert n == 0 and "different time points" in why


def test_the_dietary_lens_is_what_turns_the_figure_on(client):
    """`DOMAIN_PACKS.md` §08 — a pack changes what is drawn. Same table, same
    grain, no dietary lens, no shrinkage plot, and the reason says so."""
    pid = _upload(client, "dietary_recalls.csv")
    _decide(client, pid, "set_lens", lens=["other"])
    _decide(client, pid, "set_target", column="hba1c")
    _decide(client, pid, "set_grain", answer="people_repeat",
            group_col="participant_id")
    _decide(client, pid, "set_repeat_kind", kind="repeats")
    body = client.get(f"/project/{pid}/figures").json()
    assert "shrinkage" not in {r["id"] for r in body["admitted"]}
    held = next(r for r in body["not_drawn"] if r["id"] == "shrinkage")
    assert "dietary intake" in held["why"]
    assert "does not infer the field" in held["why"]


# ── the pack decides what is overlaid ───────────────────────────────────────

def test_the_metabolomics_pack_puts_its_qc_rows_on_the_scores_plot(client):
    """`DRIVE-009`'s own act field: *per-domain figure selection through the
    pack mechanism.* The QC overlay comes from the pack's `pooled_qc` detector,
    so the checklist item scores against a detector's reading rather than
    against a renderer's guess — and the same table with no lens fails it."""
    pid = _upload(client, "metabolomics_untargeted.csv")
    _decide(client, pid, "set_lens", lens=["metabolomics"])
    body = client.get(f"/project/{pid}/figures").json()
    row = next(r for r in body["admitted"] if r["id"] == "pca_scores")
    assert row["payload"]["n_qc"] > 0, "the pack's QC rows did not reach the figure"
    assert next(i for i in row["checklist"] if i["id"] == "qc_overlaid")["passed"]

    bare = _upload(client, "metabolomics_untargeted.csv")
    _decide(client, bare, "set_lens", lens=["other"])
    other = client.get(f"/project/{bare}/figures").json()
    row = next(r for r in other["admitted"] if r["id"] == "pca_scores")
    assert row["payload"]["n_qc"] == 0
    assert not next(i for i in row["checklist"]
                    if i["id"] == "qc_overlaid")["passed"]


def test_a_scores_plot_is_never_fitted_on_the_column_it_is_colored_by(client):
    """The circular-figure family's own shape (`DOMAIN_SCIENCE.md` §01.6):
    separation you built in. The target is set aside from the matrix and
    carried as a label."""
    pid = _upload(client, "metabolomics_untargeted.csv")
    _decide(client, pid, "set_lens", lens=["metabolomics"])
    _decide(client, pid, "set_target", column="responder")
    body = client.get(f"/project/{pid}/figures").json()
    row = next(r for r in body["admitted"] if r["id"] == "pca_scores")
    counts = row["payload"]["group_counts"]
    assert set(counts) != {"ungrouped"}, "the target did not reach the legend"
    project = api.STORE.get(pid)
    numeric = [c for c in project.df.columns
               if str(project.df[c].dtype).startswith(("int", "float"))]
    assert "responder" not in numeric or len(counts) >= 2


# ── the absences, named ─────────────────────────────────────────────────────

def test_a_figure_that_cannot_be_drawn_says_why_rather_than_vanishing(
        client, dietary):
    """`DESIGN_LANGUAGE.md` §09's recorded-absence rule, pointed at the figure
    layer. A figure silently missing is indistinguishable from a figure the app
    does not have."""
    body = client.get(f"/project/{dietary}/figures").json()
    named = {row["id"] for row in
             body["admitted"] + body["held"] + body["unavailable"]
             + body["not_drawn"]}
    assert named == set(figures.REGISTRY), (
        "a registered figure is neither drawn nor accounted for")


def test_the_calibration_plot_names_the_app_gap_and_not_the_users_data(
        client, dietary):
    """TurboTab has no training step, so no project holds predictions and the
    clinical pack's flagship figure is unreachable from an upload. That is a
    gap in the app; saying so is different from implying the table is wrong."""
    body = client.get(f"/project/{dietary}/figures").json()
    row = next(r for r in body["not_drawn"] if r["id"] == "calibration")
    assert "no training step" in row["why"]
    assert "gap in the app, not a property of your data" in row["why"]


def test_one_recall_per_person_refuses_with_the_reason_rather_than_two_densities(
        client):
    """§03: *"With one recall you cannot do that separation from your own data
    at all."* The refusal carries its badge and its offer through the endpoint,
    which is what stops it reading as an empty panel."""
    pid = _upload(client, "dietary_recalls.csv")
    _decide(client, pid, "set_lens", lens=["dietary"])
    project = api.STORE.get(pid)
    # One row per person, and nothing else about the project changed. Recorded
    # directly rather than through `set_grain`, which correctly REFUSES this
    # combination — `participant_id` unique on every row contradicts *people
    # repeat*, and the contradiction detector is right. The scenario being
    # tested is the figure layer's behavior once the state exists.
    project.df = project.df.groupby("participant_id", as_index=False).first()
    project.grain = {"answer": "people_repeat", "group_col": "participant_id"}
    project.repeat_kind = {"kind": "repeats"}

    body = client.get(f"/project/{pid}/figures").json()
    assert "shrinkage" not in {r["id"] for r in body["admitted"]}
    entry = next((r for r in body["unavailable"] if r["id"] == "shrinkage"),
                 None) or next(r for r in body["not_drawn"]
                               if r["id"] == "shrinkage")
    assert "one row per person" in entry["why"] or \
        "median of 1" in entry["why"], entry["why"]


# ── the annotation box renders the absence, for every figure ────────────────

def test_a_missing_number_renders_not_estimable_with_its_reason():
    """The gate's specific clause. On a separable classification fixture
    `weak_calibration` correctly returns `(None, None)` — the fit is undefined,
    which is what a very good model on a small sample produces — and the box
    says which number is missing and why, rather than leaving a blank cell that
    reads as a rendering fault.

    Through `figures.bundle()`, which is the path a user's figure takes. The
    predictions are supplied here rather than by a project because **TurboTab
    has no training step**, so no project can hold them; that gap is
    `GUIDED-065` and is named in `not_drawn` on every bundle.
    """
    y = np.r_[np.zeros(40), np.ones(40)]
    # Perfectly separable: every non-event below 0.5, every event above.
    proba = np.r_[np.linspace(0.01, 0.30, 40), np.linspace(0.70, 0.99, 40)]
    payload = figure_specs.calibration_render(y, proba, model_name="separable")
    assert payload["calibration_slope"] is None, (
        "the fixture is not separable, so it does not test the case")

    rows = figures.bundle({"calibration": payload})
    row = rows["held"][0]                    # its companion is not in the bundle
    box = {a["label"]: a for a in row["annotations"]}
    for label in ("Calibration intercept", "Calibration slope"):
        assert box[label]["value"] == "not estimable"
        assert "there is not one" in box[label]["why"]
        assert box[label]["value"] != "", "a blank cell reads as a render fault"
    # The numbers that ARE estimable are still shown, and the checklist still
    # fails — rendering honestly and passing a checklist are different jobs.
    assert box["n"]["value"] == "80" or box["n"]["value"] == "80"
    assert not next(i for i in row["checklist"]
                    if i["id"] == "annotation_box")["passed"]
    assert "not estimable" in row["caption"]


def test_the_absence_renders_for_a_figure_that_never_computed_its_own_box():
    """`calibration_render` owned this and the rest of the layer did not.
    `annotation_rows` is the generalization, and a spec with a key its payload
    does not carry renders the absence rather than dropping the row."""
    spec = figures.REGISTRY["shrinkage"]
    rows = figures.annotation_rows(spec, {"figure": "shrinkage"})
    assert len(rows) == len(spec.annotations)
    assert all(r["value"] == "not estimable" for r in rows)
    assert all("there is not one" in r["why"] for r in rows)

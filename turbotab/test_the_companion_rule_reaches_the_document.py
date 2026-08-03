"""`GUIDED-131` — the admissibility rule had no consumer at its own boundary.

`turbotab/figures.py` states the companion rule as **admissibility**:

> A CONFIRMATORY figure whose companion is absent is not admitted — not warned
> about, not caption-caveated, not rendered greyed out. **The bundle does not
> contain it.**

And `DOMAIN_SCIENCE.md` §01.6 says *refuse to let a confirmatory figure into the
results bundle without its validation companion.*

**The results bundle that leaves the building is the manuscript.** Driven at the
L40 adjudication on `leaky_sepsis.csv` with `calibration.companions` restored to
its pre-L40 value: `/figures` put it in `held`, `promote_figure` returned 200,
and `/manuscript` carried it as `CONFIRMATORY / promoted: True` with
`passed: True` and neither `why_held` nor the word *companion* anywhere in the
payload. `api.py` built the figure list from the whole registry and never read
`figures.bundle`.

## Why this is a report and not a refusal

`PRODUCT_VISION.md`: *a marked figure is promoted as the author marked it.* A
route that declined the promotion would overrule the author in their own
document, which is the thing that ruling forbids. The other half of the same
ruling is that **the record is not laundered, and the validator is what reports
it** — `promoted_exploratory` was built to exactly that shape for the tier, and
this is its twin for the companion.

## The rule was unenforced for four figures, not one

`roc`, `decision_curve`, `calibration_instability` and
`classification_instability` each declare a companion and each is drawable on a
project that does not draw what it declares. The instance the L40 adjudication
found was one cell of that.

## What this file does NOT do

It does not touch `promote_figure`. A test that asserted a 4xx there would be
asserting the opposite of the ruling.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

# `TEST-041`. `figures.REGISTRY` is populated only as an import side
# effect of `figure_specs`, so a file that reads the registry without
# importing its populator is reading whatever an EARLIER FILE happened
# to load. Module scope, not inside a fixture: the first test in a file
# runs before any fixture that imports `api`.
from turbotab import figure_specs  # noqa: F401 — populates FIG.REGISTRY
from turbotab import figures as FIG
from turbotab import jobwait as JW
from turbotab import manuscript as MS
from turbotab.project import AnalysisProject

#: `GUIDED-097`. Two fixtures of different target shape, because the claim is
#: about the DOCUMENT and a document is produced for both.
TARGET_SHAPES = {
    "binary classification": ("clinical_risk.csv", "readmit_30d",
                              "classification", "logreg"),
    "continuous regression": ("survey_instrument.csv", "age", "regression",
                              "ridge"),
}

#: NOT COVERED, said out loud.
#:
#: A PENDING COMPANION. `figures.PENDING` holds four declared-but-unbuilt
#: figures and no registered figure names one as a companion, so the case
#: *promoted, companion is declared pending* cannot be constructed from the real
#: registry — and constructing it from a literal would be `GUIDED-134` exactly.
#: The behavior is stated instead: `PENDING` ids are not in `REGISTRY`, so a
#: pending companion can never be promoted and would report every time.
#:
#: MULTICLASS. The clinical figures decline a three-class target, so no
#: confirmatory figure is promotable there to begin with.
SHAPES_NOT_COVERED = [
    "a companion that is declared PENDING rather than registered — no "
    "registered figure names one, so the case cannot be built from the real "
    "registry and building it from a literal would be GUIDED-134 itself",
    "multiclass — the clinical figures decline a three-class target, so there "
    "is no promotable confirmatory figure to orphan",
]


def _driven(shape):
    """Upload → target → grain → eligibility → seal → **train through the
    route**, and hand back `(client, project_id)`.

    **The route, not `training.train`.** The first version of this file built
    the project in-process and set `training_run` on it, and a revert probe
    reported `GREEN — NOT LOAD-BEARING` on the assertion that the companion gap
    turns `passed` false. It was right: `api.get_manuscript` reads the run from
    `_RUNS`, which only the `/train` route populates, so `validate` was called
    with `run=None`, three *unrelated* checks failed for a missing analysis
    population, and `passed` was already `False` before this loop touched
    anything. The assertion was true and it was testing nothing.

    Driven properly the validator returns `passed: True` with zero failures,
    which is what makes the flip attributable — and it is also the exact state
    `GUIDED-131` was filed about: a clean report over a promoted CONFIRMATORY
    figure with no companion in the results.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    name, target, task, model = TARGET_SHAPES[shape]
    client = TestClient(api.app)
    with open(f"turbotab/sample_data/{name}", "rb") as handle:
        project_id = client.post(
            "/project", files={"file": (name, handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_target", {"column": target}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})):
        response = client.post(f"/project/{project_id}/decision",
                               json={"kind": kind, "payload": payload})
        assert response.status_code == 200, (kind, response.text[:400])

    job = client.post(f"/project/{project_id}/train",
                      json={"models": [model]}).json()
    # `TEST-040`: a deadline, not an iteration count. A bounded loop with no
    # wait elapses in milliseconds and reports "still running" as "wrong
    # answer" — see `turbotab/jobwait.py`.
    JW.settle_done(client, job)
    return client, project_id


def _sealed_not_fitted(shape):
    """The same journey, stopped one step early. A real state, not a stub."""
    from fastapi.testclient import TestClient

    from turbotab import api

    name, target, _task, _model = TARGET_SHAPES[shape]
    client = TestClient(api.app)
    with open(f"turbotab/sample_data/{name}", "rb") as handle:
        project_id = client.post(
            "/project", files={"file": (name, handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_target", {"column": target}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"}),
                          ("seal", {})):
        response = client.post(f"/project/{project_id}/decision",
                               json={"kind": kind, "payload": payload})
        assert response.status_code == 200, (kind, response.text[:400])
    return client, project_id


def _manuscript(client, project_id):
    return client.get(f"/project/{project_id}/manuscript").json()


def _promote(client, project_id, *figure_ids, on=True):
    for figure_id in figure_ids:
        response = client.post(f"/project/{project_id}/decision", json={
            "kind": "promote_figure", "subject": figure_id,
            "payload": {"figure_id": figure_id, "promoted": on}})
        assert response.status_code == 200, response.text
    return _manuscript(client, project_id)


# ═══════════ THE RULE IS REAL SOMEWHERE ELSE — THAT WAS THE PROBLEM ═══════════

def test_more_than_one_figure_declares_a_companion_it_can_outlive():
    """The finding was reported as one figure and it is four.

    Not a count for its own sake: a cross-section built against one instance is
    a cross-section shaped by that instance, and the reason this was invisible
    for six loops is that everyone was looking at `calibration`.
    """
    declaring = {fid: spec.companions for fid, spec in FIG.REGISTRY.items()
                 if spec.companions}
    assert len(declaring) >= 4, declaring
    for figure_id, companions in declaring.items():
        assert FIG.REGISTRY[figure_id].tier == FIG.CONFIRMATORY, figure_id
        for companion in companions:
            assert companion in FIG.REGISTRY or companion in FIG.PENDING, (
                f"{figure_id} declares `{companion}` and it resolves nowhere")


# ═══════════ THE CROSS-SECTION ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_a_clean_report_is_what_the_gap_has_to_survive(shape):
    """**The precondition, asserted rather than assumed.**

    `GUIDED-131` is that the document reported itself clean over a promoted
    figure with no companion. If the validator were failing anyway, every
    assertion below about `passed` would be true and would be measuring
    something else — which is exactly what the first draft of this file did,
    and a revert probe caught it as `GREEN — NOT LOAD-BEARING`.
    """
    client, project_id = _driven(shape)
    before = _manuscript(client, project_id)
    assert before["promoted_without_companion"] == []
    assert before["n_failed"] == 0, [
        row["Check"] for row in before["rows"] if row["Status"] == "FAIL"]
    assert before["passed"] is True, (
        "this project's report is not clean to begin with, so nothing below "
        "can attribute a failure to the companion gap")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_a_promoted_figure_alone_is_reported_not_refused(shape):
    """The whole ruling in one test: **200, and reported.**

    `roc` declares two companions and is promoted alone. The promotion is
    recorded exactly as the author made it, the prose is untouched, and the
    validation report — the separate honest list — names what is missing.
    """
    client, project_id = _driven(shape)
    assert _manuscript(client, project_id)["passed"] is True
    document = _promote(client, project_id, "roc")

    orphaned = document["promoted_without_companion"]
    assert [row["id"] for row in orphaned] == ["roc"], orphaned
    assert orphaned[0]["missing_companions"] == ["calibration", "decision_curve"]
    assert "circular-figure" in orphaned[0]["because"]
    assert "has not changed your document" in orphaned[0]["because"]

    assert document["n_failed"] == 0, (
        "an unrelated check failed, so the flip below is not attributable")
    assert document["passed"] is False, (
        "a promoted confirmatory figure with no companion left the report "
        "reading as clean, which is exactly GUIDED-131")

    # THE PROSE IS STILL THE AUTHOR'S. The ruling forbids a caption caveat as
    # firmly as it forbids a refusal.
    prose = document["rendered"]["methods"] + document["rendered"]["report"]
    assert "companion" not in prose.lower(), (
        "the companion gap was written into the author's document; it belongs "
        "in the validation report beside it")


def test_promoting_the_companion_clears_it():
    """The positive control, and it has to be the *user's* action that clears
    it — a check nothing can satisfy is a check nobody can act on."""
    client, project_id = _driven("binary classification")

    assert _promote(client, project_id, "roc")["promoted_without_companion"]
    partial = _promote(client, project_id, "calibration")
    assert [row["missing_companions"] for row in
            partial["promoted_without_companion"]] == [["decision_curve"]], (
        "promoting one of two companions did not narrow the report")
    whole = _promote(client, project_id, "decision_curve")
    assert whole["promoted_without_companion"] == [], (
        "every declared companion is in the results and the gap is still "
        "reported")
    assert whole["passed"] is True, (
        "the author satisfied every companion and the report still does not "
        "read clean, so the check cannot be discharged")


def test_an_exploratory_promotion_is_not_reported_here():
    """The negative control. `pca_scores` declares no companion and cannot —
    an EXPLORATORY figure makes no claim, so requiring validation of it would
    be the uncalibrated caution the ruling rejects, one surface over."""
    client, project_id = _driven("binary classification")
    document = _promote(client, project_id, "pca_scores")
    assert document["promoted_without_companion"] == []
    assert [row["id"] for row in document["promoted_exploratory"]] == ["pca_scores"]


def test_the_two_cross_sections_are_independent():
    """Both fire, separately, on the same document. They answer different
    questions — *is this figure's tier what the results imply* and *is its
    validation here* — and folding them into one list would lose which."""
    client, project_id = _driven("binary classification")
    document = _promote(client, project_id, "pca_scores", "roc")
    assert [row["id"] for row in document["promoted_exploratory"]] == ["pca_scores"]
    assert [row["id"] for row in document["promoted_without_companion"]] == ["roc"]


# ═══════════ WHAT THE BUNDLE READ IS FOR ═══════════

def test_a_companion_this_project_cannot_draw_gets_the_stronger_sentence():
    """**The half that needed `api.py` to read `figures.bundle`.**

    A companion the author can promote and a companion this project cannot draw
    are the same problem for a reviewer and different problems for the author.
    Telling someone to promote a figure that does not exist for their data is
    advice that cannot be followed, and the bundle is the only thing that knows.
    """
    client, project_id = _driven("binary classification")
    figures = client.get(f"/project/{project_id}/figures").json()
    drawable = {row["id"] for row in figures["admitted"] + figures["held"]}
    assert "roc" in drawable, "the fitted project cannot draw the ROC at all"

    # A project one step earlier in the journey: sealed, not yet fitted. Every
    # figure that needs held-out predictions is unavailable, which is a real
    # state the app passes through rather than a contrivance.
    unfitted_client, unfitted_id = _sealed_not_fitted("binary classification")
    unfitted = unfitted_client.get(f"/project/{unfitted_id}/figures").json()
    assert "roc" not in {row["id"] for row in
                         unfitted["admitted"] + unfitted["held"]}, (
        "the unfitted project draws the ROC after all, so this proves nothing")

    document = _promote(unfitted_client, unfitted_id, "calibration")
    orphaned = document["promoted_without_companion"]
    assert [row["id"] for row in orphaned] == ["calibration"]
    assert orphaned[0]["undrawable_companions"] == ["roc"]
    assert "cannot draw roc at all" in orphaned[0]["because"]
    assert "is not the remedy" in orphaned[0]["because"]


def test_the_manuscript_route_reads_the_bundle():
    """`GUIDED-131`'s evidence line, asserted so it cannot come back.

    Not a grep for `figure_bundle` in the file — trap 5. The claim is that the
    route's figure rows carry a per-project `drawn` verdict, which only the
    bundle can supply, and a route that stopped reading it would answer `None`.
    """
    client, project_id = _driven("binary classification")
    document = _manuscript(client, project_id)
    rows = document["document"]["figures"]
    assert rows, "the manuscript carries no figure rows at all"
    assert all("drawn" in row for row in rows)
    assert any(row["drawn"] is True for row in rows), (
        "every figure came back undrawable on a fitted project, so `drawn` is "
        "not being read from the bundle")
    assert any(row["drawn"] is False for row in rows), (
        "every figure came back drawable, which no project achieves — the "
        "survey figures need a survey table")


# ═══════════ THE COMPANION IS RESOLVED FROM THE REAL REGISTRY ═══════════

def test_a_made_up_companion_cannot_reach_this_check():
    """**`GUIDED-134`, applied to the check being written rather than found in
    an old one.**

    The defect class is a guard satisfied by a stand-in the real registry can
    never supply. So the cross-section resolves companions from
    `figures.REGISTRY` and from nowhere else: a `companions` key invented on the
    document row is not read, and cannot make this check pass or fail.
    """
    document = {"figures": [
        {"id": "roc", "title": "ROC curve", "tier": "CONFIRMATORY",
         "promoted": True,
         # Ignored. If this were read, the next line would come back empty.
         "companions": []},
    ]}
    reported = MS.promoted_without_companion(document)
    assert [row["missing_companions"] for row in reported] == [
        ["calibration", "decision_curve"]], (
        "the check read a `companions` key off the row, so a caller could "
        "silence it by supplying one — which is the class GUIDED-134 names")


def test_a_promoted_id_the_registry_does_not_know_is_reported_not_skipped():
    """`figures.bundle` skips an unregistered id with `continue`, on a line
    marked `# pragma: no cover`, and that skip is what let `GUIDED-128` hide
    for six loops.

    A promoted id nothing registers is a document referring to a figure that
    does not exist. That is worse than a missing companion, not smaller, so it
    is reported rather than passed over.
    """
    document = {"figures": [
        {"id": "discrimination", "title": "Discrimination",
         "tier": "CONFIRMATORY", "promoted": True},
    ]}
    reported = MS.promoted_without_companion(document)
    assert len(reported) == 1
    assert "no figure with that id is registered" in reported[0]["because"]


def test_the_check_survives_a_validator_that_cannot_load(monkeypatch):
    """The cross-section is the app's own and does not need `ml/` to run.

    A companion gap that disappeared when the validator failed to import would
    be the silence this whole surface exists to remove.
    """
    df = pd.read_csv("turbotab/sample_data/clinical_risk.csv")
    project = AnalysisProject.from_dataframe(df, "clinical_risk.csv")
    project.target, project.task_type = "readmit_30d", "classification"
    figures = [{"id": "roc", "title": "ROC curve", "tier": "CONFIRMATORY",
                "promoted": True}]

    import builtins
    real_import = builtins.__import__

    def refuse(name, *args, **kw):
        if name == "ml.manuscript_validator":
            raise ImportError("no validator here")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", refuse)
    out = MS.validate(project.to_dict(), figures=figures)
    assert out["available"] is False
    assert [row["id"] for row in out["promoted_without_companion"]] == ["roc"]


# ═══════════ AND IT REACHES A PERSON ═══════════

@pytest.mark.skipif(
    not __import__("turbotab.pageharness", fromlist=["x"]).available(),
    reason="no JS engine on this machine")
def test_the_page_renders_the_companion_gap():
    """**Trap 6.** The server composes a user-facing string and the interface
    never renders it — measured at six surfaces, and a seventh would have been
    this one. Driven through the page's own controller, not grepped.
    """
    from turbotab import pageharness as PH

    client, pid = _driven("binary classification")
    document = _promote(client, pid, "roc")
    assert document["promoted_without_companion"], "nothing to render"

    routes = {
        f"/project/{pid}": client.get(f"/project/{pid}").json(),
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            {"questions": [], "steps": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/draft": client.get(f"/project/{pid}/draft").json(),
        f"/project/{pid}/gaps": {"gaps": []},
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/manuscript": document,
    }
    out = PH.run("__emit({html: __harness.html('reportBox')});",
                 routes=routes, search=f"?project={pid}")
    html = out["html"]
    assert "promoted without" in html, (
        "the companion gap is composed on the server and the page never shows "
        "it, which is the defect class this door has already paid for six "
        "times")
    assert "calibration, decision_curve" in html
    assert "Every consistency check" not in html, (
        "the panel says every check is met while listing a companion gap "
        "beneath it")


def test_the_payload_says_companion_where_it_used_to_say_nothing():
    """The finding's own evidence sentence, inverted into an assertion:
    *neither `why_held` nor the word companion anywhere in the payload.*"""
    client, project_id = _driven("binary classification")
    document = _promote(client, project_id, "roc")
    assert "companion" in json.dumps(document).lower()

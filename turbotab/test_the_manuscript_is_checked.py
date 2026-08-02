"""`GUIDED-107` — the manuscript as data, checked by the machine that exists.

`AUDIT-001` was a section composed correctly in isolation, asserting what no
other section supported. `ml/manuscript_validator.py` is the machine built to
catch that, and it has been reachable only from `pages/10` while
`turbotab/draft.py` imported nothing from `ml/` at all. Report shipped at L36
without it.

The product owner's two rulings (`PRODUCT_VISION.md`) are implemented rather
than re-derived, and each has a test here:

- **the manuscript is data before it is a document** — `test_the_counts_are_numbers_not_prose`
- **a marked figure is promoted as the author marked it** — `test_a_promoted_exploratory_figure_is_not_caveated_in_the_prose`

`GUIDED-097` — THE FIXTURE RULE. Two target shapes, and the shapes not covered
are named below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import manuscript as MS
from turbotab import training as T
from turbotab.project import AnalysisProject

TARGET_SHAPES = {
    "binary classification": ("metabolomics_untargeted.csv", "responder",
                              "classification", "logreg"),
    "continuous regression": ("survey_instrument.csv", "age", "regression",
                              "ridge"),
}

#: NOT COVERED, said out loud.
#:
#: MULTICLASS — the validator's *"selection metric language matches task type"*
#: check knows two task types, because the app does. A multiclass manuscript
#: would be checked as a binary one.
#:
#: SURVIVAL — no task type.
#:
#: THE LATEX DOCUMENT — deliberately unrendered this loop (`GUIDED-115`), so
#: every validator check that reads `latex_text` runs against an empty string.
#: Those checks look for markdown artifacts leaking into a LaTeX file; passing
#: the markdown instead would manufacture failures about a document that does
#: not exist, and passing nothing means those checks are inert rather than
#: wrong. `test_the_latex_half_is_absent_and_says_so` asserts the absence is
#: declared rather than silent.
SHAPES_NOT_COVERED = [
    "multiclass classification — the task-type metric check knows two types",
    "survival / time-to-event — no task type exists",
    "the LaTeX document — deferred as GUIDED-115; its validator checks are "
    "inert rather than passing, and the deferral is served in the payload",
]


def _project(name, target, task, *, fit=True):
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    df = df[df[target].notna()].copy()
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.20))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    run = None
    if fit:
        run = T.train(p, [TARGET_SHAPES_BY_NAME[(name, target)]]).to_dict()
    return p, run


TARGET_SHAPES_BY_NAME = {(n, t): m for n, t, _, m in TARGET_SHAPES.values()}


# ═══════════ THE VALIDATOR IS REUSED, NOT REBUILT ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_draft_is_checked_by_mls_validator(shape):
    """`AUDIT-008`: the core already holds the capability and the path that
    needs it does not read it. This is that path, reading it."""
    name, target, task, _ = TARGET_SHAPES[shape]
    p, run = _project(name, target, task)
    out = MS.validate(p.to_dict(), run=run)

    assert out["available"] is True, out.get("because")
    assert out["rows"], "the validator ran and made no checks at all"
    names = {r["Check"] for r in out["rows"]}
    # Named checks rather than a count, so a validator that lost one is caught.
    assert any("Analysis population" in n for n in names)
    assert any("predictor count" in n for n in names)
    assert any("metric language matches task type" in n for n in names)


def test_it_does_not_write_a_second_validator():
    """The instruction was *reuse it; do not write a second one*, and a second
    one is the shape `STATE-034` warns about — two systems answering one
    question that can disagree."""
    import ast

    tree = ast.parse(open("turbotab/manuscript.py").read())
    imports = {n.module for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module}
    assert "ml.manuscript_validator" in imports, (
        "turbotab/manuscript.py does not import the validator, so whatever it "
        "reports is a second opinion rather than the one that exists")
    source = ast.unparse(tree)
    for invented in ("def _check_", "CHECKS = ", "def _validate_analysis"):
        assert invented not in source, (
            f"turbotab/manuscript.py defines {invented!r}; the validator's "
            f"checks live in ml/")


# ═══════════ THE FIRST RULING · DATA BEFORE DOCUMENT ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_counts_are_numbers_not_prose(shape):
    """*A checklist cannot be run against prose.*

    The counts the validator compares are carried as integers on the structured
    document, which is what makes L10's checklist engine possible at all.
    """
    name, target, task, _ = TARGET_SHAPES[shape]
    p, run = _project(name, target, task)
    doc = MS.structure(p.to_dict(), run=run)

    counts = doc["context"]["population_counts"]
    assert isinstance(counts.get("analysis_total"), int)
    assert counts["train"] + counts["test"] == counts["analysis_total"]
    assert doc["context"]["included_models"], "no model reached the document"
    assert doc["sections"], "the structured document has no sections"
    assert all("key" in s and "title" in s for s in doc["sections"])


def test_a_count_the_app_does_not_hold_is_absent_rather_than_guessed():
    """Return nothing rather than a wrong value.

    A guessed count is worse than none here: the validator would confirm it
    against the prose it was rendered into, which is a check that can only
    pass — agreement reported where nothing was compared.
    """
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, _ = _project(name, target, task, fit=False)
    doc = MS.structure(p.to_dict(), run=None)
    assert doc["context"]["feature_counts"] == {}
    assert doc["context"]["included_models"] == []
    rendered = MS.to_markdown(doc)
    assert "participants" not in rendered["report"] or \
        doc["context"]["population_counts"].get("analysis_total")


# ═══════════ THE SECOND AND FOURTH RULINGS · PROMOTION ═══════════

def test_a_promoted_exploratory_figure_is_not_caveated_in_the_prose():
    """*Promoted as the author marked it.* The manuscript is the author's
    document; the app drafts it and the researcher signs it.

    The alternative — annotating the caption with the tier — was considered and
    rejected as the second uncalibrated layer of caution: a caveat on every
    promoted figure makes a real concern and a routine one read identically.
    """
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, run = _project(name, target, task)
    figures = [{"id": "pca_scores", "title": "PCA scores",
                "tier": "EXPLORATORY", "promoted": True}]
    out = MS.validate(p.to_dict(), run=run, figures=figures)
    rendered = out["rendered"]

    prose = rendered["methods"] + rendered["report"]
    assert "EXPLORATORY" not in prose, (
        "the tier was annotated into the manuscript; the ruling is that no "
        "tier annotation is added on the way in")
    assert "exploratory" not in prose.lower()

    # AND IT IS REPORTED, separately.
    flagged = out["promoted_exploratory"]
    assert [f["id"] for f in flagged] == ["pca_scores"]
    assert "a reviewer is likely to ask" in flagged[0]["because"]
    assert out["passed"] is False, (
        "a promoted EXPLORATORY figure left the report reading as clean")


def test_a_promoted_confirmatory_figure_is_not_flagged():
    """The positive control. A check that flagged every promoted figure would
    be the uncalibrated caution the ruling rejects, moved one surface over."""
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, run = _project(name, target, task)
    out = MS.validate(p.to_dict(), run=run, figures=[
        {"id": "calibration", "title": "Calibration plot",
         "tier": "CONFIRMATORY", "promoted": True},
        {"id": "pca_scores", "title": "PCA scores",
         "tier": "EXPLORATORY", "promoted": False}])
    assert out["promoted_exploratory"] == []


# ═══════════ WHAT THE WIRING FOUND ═══════════

def test_the_sections_the_draft_cannot_source_are_named_not_silent():
    """`GUIDED-116`, and it is what wiring the validator found on its first run.

    `draft.py` folds over DECISIONS, and which model was fitted is a property
    of a RUN. So the manuscript states its analysis population, its cohort, its
    preprocessing and its predictors, and never says which model was fitted or
    how it was scored — while the abstract asserts an analysis population. That
    is `AUDIT-001`'s class in the artifact that leaves the building.

    Reported as a named gap rather than left as a bare FAIL, because a
    validator that finds no section reports nothing wrong, and a reader would
    take the failure for a formatting slip.
    """
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, run = _project(name, target, task)
    out = MS.validate(p.to_dict(), run=run)

    headings = {u["heading"] for u in out["unsourced_sections"]}
    assert headings == {"Model Development", "Model Evaluation"}
    for entry in out["unsourced_sections"]:
        assert len(entry["because"]) > 60, "a gap named with no reason"

    # And the sections that ARE rendered but nothing checks.
    assert set(out["unchecked_sections"]) == {
        "Data Sources", "Exploratory Analysis", "Missing Data", "Limitations"}, (
        out["unchecked_sections"])

    blocked = [r for r in out["rows"]
               if r["Status"] == "FAIL" and r["blocked_by_missing_section"]]
    assert blocked, (
        "no failing check was attributed to a missing section, so the "
        "separation this reports is not doing anything")
    assert out["n_failed_for_a_missing_section"] == len(blocked)
    assert out["n_failed"] > out["n_failed_for_a_missing_section"], (
        "every failure was attributed to a missing section; the remaining "
        "ones are the real findings and there should be some")


def test_the_latex_half_is_absent_and_says_so():
    """`GUIDED-115`. Deferred per the loop's scope note, and DECLARED.

    Passing the markdown as the LaTeX document would manufacture failures about
    a document that does not exist; passing nothing leaves those checks inert.
    Either way the absence must be visible, or a reader takes the report as
    covering an export that was never made.
    """
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, run = _project(name, target, task)
    out = MS.validate(p.to_dict(), run=run)
    assert "LaTeX export is not wired yet" in out["latex_deferred"]
    assert "latex_report" in out["latex_deferred"], (
        "the deferral does not name where it will render from, so it reads as "
        "an omission rather than a decision")
    assert "latex" not in out["rendered"]


def test_the_heading_map_is_the_validators_own_vocabulary():
    """A heading is an interface here, and that coupling is worth a test.

    A renderer that called Study Design 'Population' would SILENTLY DISABLE the
    check rather than fail it — the validator extracts by literal heading and
    finds nothing, and nothing found is nothing wrong.
    """
    source = open("ml/manuscript_validator.py").read()
    for heading in MS.CHECKED_HEADINGS:
        assert f'"{heading}"' in source, (
            f"'{heading}' is declared CHECKED and "
            f"ml/manuscript_validator.py never extracts it, so a section the "
            f"report implies was checked was not")
    # And the other direction: a heading rendered but not extracted must be
    # declared unchecked rather than quietly assumed covered.
    rendered = {h for _, (_, h, w) in MS._HEADINGS.items() if w == "methods"}
    for heading in rendered - set(MS.CHECKED_HEADINGS):
        assert f'"{heading}"' not in source, (
            f"'{heading}' IS extracted by the validator but is listed as "
            f"unchecked, which understates the coverage")
    assert f'"{MS.ABSTRACT_HEADING[1]}"' in source, (
        "the abstract heading the renderer emits is not the one the validator "
        "extracts, so the abstract-vs-methods cross-check silently does "
        "nothing")
    for heading in MS.REQUIRED_BUT_UNSOURCED:
        assert f'"{heading}"' in source, (
            f"'{heading}' is declared as a gap the validator cares about and "
            f"the validator never mentions it")


# ═══════════ L38-D1 · THE FIGURE THAT REFUSES ═══════════

def test_kaplan_meier_is_pending_and_names_what_is_missing():
    """`GUIDED-118`. The refusal IS the result.

    The app cannot represent a time-to-event outcome: `set_task_type` takes
    classification and regression, there is no censoring concept, and no
    control declares a time + event-indicator pair. A curve drawn from a
    column the app believes is an ordinary number would be a survival claim
    about something nobody declared to be survival data.

    Registered as `Pending` rather than left absent, because this app's own
    rule is that a refusal offering nothing is indistinguishable from a
    missing feature.
    """
    from turbotab import figures
    import turbotab.figure_specs                            # noqa: F401

    entry = figures.resolve("kaplan_meier")
    assert entry["status"] == figures.PENDING_STATUS
    assert entry["blocked_by"] == "GUIDED-118"
    assert "A4.6" in entry["specified_in"]

    needs = entry["needs"]
    for required in ("follow-up time", "event indicator", "censored",
                     "competing", "cumulative incidence"):
        assert required in needs, (
            f"the pending entry does not mention {required!r}; a reader "
            f"planning the build would ship the naive figure")
    # THE ANTI-PATTERN IS NAMED BEFORE THE FIGURE IS BUILT, not after.
    # §A4.6 is SETTLED that 1 - KM overestimates cumulative incidence under
    # competing risks and calls it a very common error, so the entry has to
    # carry it — the detection decides WHICH figure is correct, and a build
    # that read only "draw a KM curve" would get it wrong.
    assert "overestimates cumulative incidence" in needs


def test_the_app_still_cannot_represent_a_survival_target():
    """The premise of the refusal above, asserted so that the day it changes
    this test fails and the pending entry gets revisited."""
    from turbotab.project import AnalysisProject, ProjectError

    df = pd.DataFrame({"t": [1, 2, 3, 4] * 5, "event": [0, 1] * 10,
                       "x": range(20)})
    p = AnalysisProject.from_dataframe(df, "survival.csv")
    with pytest.raises(ProjectError, match="not a task type"):
        p.override_task_type("survival")


# ═══════════ PROMOTION, RECORDED ═══════════

def test_promotion_is_a_recorded_decision_and_ignores_the_tier():
    """`promotable` has sat on every figure spec since L26 with no consumer.

    The recorder does NOT consult the tier, and that is the ruling rather than
    an omission: a route that refused an `EXPLORATORY` figure would be the app
    overruling the author in their own document.
    """
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, _ = _project(name, target, task, fit=False)

    decision = p.promote_figure("pca_scores")
    assert p.promoted_figures == ["pca_scores"]
    assert decision.kind == "promote_figure"
    assert "placed in the results" in decision.text
    assert "EXPLORATORY" not in decision.text, (
        "the recorded sentence annotates the tier; the ruling is that no tier "
        "annotation is added on the way in")

    # And it comes back out, because the past is editable.
    p.promote_figure("pca_scores", promoted=False)
    assert p.promoted_figures == []


def test_the_route_records_a_promotion_and_the_report_notices():
    from fastapi.testclient import TestClient

    from turbotab import api

    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, _ = _project(name, target, task, fit=False)
    api.STORE.add(p)
    client = TestClient(api.app)

    before = client.get(f"/project/{p.id}/manuscript").json()
    assert before["promoted_exploratory"] == []

    ok = client.post(f"/project/{p.id}/decision", json={
        "kind": "promote_figure", "subject": "pca_scores",
        "payload": {"figure_id": "pca_scores", "promoted": True}})
    assert ok.status_code == 200, ok.text

    after = client.get(f"/project/{p.id}/manuscript").json()
    assert [f["id"] for f in after["promoted_exploratory"]] == ["pca_scores"]
    assert after["passed"] is False, (
        "a promoted EXPLORATORY figure left the report reading as clean")
    # The prose is still the author's document.
    assert "EXPLORATORY" not in (after["rendered"]["methods"]
                                 + after["rendered"]["report"])


@pytest.mark.xfail(strict=True, reason=(
    "GUIDED-119. The promotion DECISION and its consequence are built and "
    "driven; the page has no control that records one, so a user cannot "
    "promote a figure from the Guided door. LOOP.md §05 permits shipping a "
    "capability with a failing test naming the consumer it lacks, and this is "
    "that test — it flips to passing the day the control exists."))
def test_the_page_offers_a_way_to_promote_a_figure():
    page = open("turbotab/web/index.html").read()
    assert "promote_figure" in page, (
        "no control in the Guided door records a promotion, so `promotable` "
        "still has no user-facing consumer")

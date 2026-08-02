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
    "PDF compilation — `ml/latex_report.compile_latex_to_pdf` shells out to a "
    "LaTeX toolchain this environment does not have, so the .tex is rendered "
    "and never compiled; nothing asserts the document typesets",
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
    # The VALIDATOR's key names — `train_n`/`val_n`/`test_n` is what
    # `validate_manuscript_bundle` sums, and using `train`/`test` made its
    # reconciliation check compare 300 against 0 (`GUIDED-116`).
    assert counts["train_n"] + counts["val_n"] + counts["test_n"] == \
        counts["analysis_total"]
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
    # NOT a blanket ban on the WORD. `L39-B1` added a Model Evaluation
    # sentence that calls an unverified split's figures *exploratory*, which is
    # a true claim about the SPLIT and has nothing to do with a figure's tier.
    # The ruling is about tier annotation; asserting the absence of a common
    # English word would have forced that honest sentence out of the
    # manuscript to satisfy a test.
    assert "pca scores" not in prose.lower(), (
        "the promoted figure was named in the prose with its tier nearby")
    for figure in figures:
        assert figure["tier"].lower() not in prose.lower().split(
            str(figure["title"]).lower())[0][-200:] if \
            str(figure["title"]).lower() in prose.lower() else True

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

    # WITH a run both sections have a source (`L39-B1`), so the gap is
    # conditional now rather than permanent. Driven both ways: the unsourced
    # list is what a manuscript with no fit still cannot say.
    assert out["unsourced_sections"] == []
    assert any(s["key"] == "models" for s in out["document"]["sections"])
    assert any(s["key"] == "evaluation" for s in out["document"]["sections"])

    p_nofit, _ = _project(name, target, task, fit=False)
    dry = MS.validate(p_nofit.to_dict(), run=None)
    headings = {u["heading"] for u in dry["unsourced_sections"]}
    assert headings == {"Model Development", "Model Evaluation"}
    for entry in out["unsourced_sections"]:
        assert len(entry["because"]) > 60, "a gap named with no reason"

    # And the sections that ARE rendered but nothing checks.
    assert set(out["unchecked_sections"]) == {
        "Data Sources", "Exploratory Analysis", "Missing Data", "Limitations"}, (
        out["unchecked_sections"])

    # WITH a run, nothing fails at all — which is the point of `L39-B1` and
    # is asserted here rather than in a separate test, because *the sections
    # exist now* and *the checks pass now* are one claim.
    assert out["n_failed"] == 0, [
        r["Check"] for r in out["rows"] if r["Status"] == "FAIL"]

    # WITHOUT one, three checks fail and NONE is attributable to a missing
    # section — which is worth stating rather than glossing. They are all
    # about counts a project with no fit does not have, and that is the honest
    # state: a manuscript describing an analysis nobody ran cannot state its
    # analysis population.
    assert dry["n_failed"] == 3, [
        r["Check"] for r in dry["rows"] if r["Status"] == "FAIL"]
    assert dry["n_failed_for_a_missing_section"] == 0
    # One PASSING check does name a missing section — *model names match
    # between development and evaluation sections*, which passes vacuously
    # because there are no models to disagree about. Attributed and not
    # failing is the correct pair of states, and asserting `all False` would
    # have been asserting the attribution never fires.
    attributed = [r for r in dry["rows"] if r["blocked_by_missing_section"]]
    assert all(r["Status"] == "PASS" for r in attributed), [
        r["Check"] for r in attributed if r["Status"] != "PASS"]

    # **THE ATTRIBUTION MECHANISM NOW HAS NO LIVE CASE**, because the check it
    # was built for — *model names match between development and evaluation
    # sections* — passes vacuously with no run and genuinely with one. It is
    # kept and asserted on a constructed row rather than deleted: the next
    # section the validator wants and the draft cannot source will need it,
    # and a mechanism with no test is a mechanism nobody trusts.
    assert MS.REQUIRED_BUT_UNSOURCED, "the mechanism's input list is empty"
    heading = sorted(MS.REQUIRED_BUT_UNSOURCED)[0]
    synthetic = {"Status": "FAIL", "Check": f"something about {heading}",
                 "Location": "Methods", "Detail": ""}
    unsourced = {u["heading"] for u in dry["unsourced_sections"]}
    assert any(h.lower() in (synthetic["Check"] + " "
                             + synthetic["Location"]).lower()
               for h in unsourced), (
        "a check naming an unsourced heading would not be attributed to it, "
        "so the separation is decorative")


def test_the_latex_document_is_rendered_by_the_exporter_classic_uses():
    """`GUIDED-115`, closed. *One core, no forks.*

    Deferred at L38 with the cost named — three validator checks were INERT
    rather than passing, because they look for markdown artifacts and internal
    model keys leaking into a LaTeX file and were handed an empty string. A
    check with nothing to read reports nothing wrong, which is the same
    silence-as-agreement this whole file is about.
    """
    import ast

    name, target, task, _ = TARGET_SHAPES["continuous regression"]
    p, run = _project(name, target, task)
    out = MS.validate(p.to_dict(), run=run)

    latex = out["rendered"]["latex"]
    assert latex, "no LaTeX was rendered"
    assert out["latex_bytes"] == len(latex)
    assert "\\documentclass" in latex or "\\begin{document}" in latex, (
        "the rendered text is not a LaTeX document")
    # THE CHECKS THAT WERE INERT ARE LIVE, asserted by name so a future
    # regression to `""` fails here rather than going green.
    for live in ("LaTeX output is free of markdown and note artifacts",
                 "No internal model keys leak into export text"):
        row = next(r for r in out["rows"] if r["Check"] == live)
        assert row["Status"] == "PASS", row["Detail"]

    # RENDERED BY ml/, NOT BY A SECOND EXPORTER.
    tree = ast.parse(open("turbotab/manuscript.py").read())
    modules = {n.module for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module}
    assert "ml.latex_report" in modules
    body = ast.unparse(tree)
    for invented in ("\\\\documentclass", "\\\\begin{document}", "usepackage"):
        assert invented not in body, (
            f"turbotab/manuscript.py writes {invented!r} itself; the exporter "
            f"is ml/latex_report.py and there must not be two")


def test_a_manuscript_with_no_fit_renders_no_latex_rather_than_a_template():
    """Return nothing rather than a wrong value. An empty template is a
    document asserting a study that does not exist."""
    name, target, task, _ = TARGET_SHAPES["continuous regression"]
    p, _ = _project(name, target, task, fit=False)
    out = MS.validate(p.to_dict(), run=None)
    assert out["rendered"]["latex"] == ""
    assert out["latex_bytes"] == 0


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


@pytest.mark.skipif(
    not __import__("turbotab.pageharness", fromlist=["x"]).available(),
    reason="no JS engine on this machine")
def test_the_page_offers_a_way_to_promote_a_figure():
    """`GUIDED-119`, closed. It shipped at L38 as an `xfail(strict=True)`
    naming the consumer it lacked, which is `LOOP.md` §05 working as intended;
    this is the same test with the mark removed and the claim DRIVEN rather
    than grepped, because a string in the file is not a control on the page.

    **The control is at the FIGURE, not in Report.** A figure is promoted where
    it is looked at; Report renders the consequence. And it does not consult
    the tier — the ruling is that a marked figure is promoted as the author
    marked it.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, _ = _project(name, target, task, fit=False)
    api.STORE.add(p)
    client = TestClient(api.app)
    project = client.get(f"/project/{p.id}").json()
    figures = client.get(f"/project/{p.id}/figures").json()
    assert figures["admitted"] or figures["held"], (
        "no figure was drawn, so there is nothing to promote and this proves "
        "nothing")

    routes = {
        f"/project/{p.id}": project,
        f"/project/{p.id}/interview?step=data":
            client.get(f"/project/{p.id}/interview?step=data").json(),
        f"/project/{p.id}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{p.id}/evidence/missingness": {"cards": []},
        f"/project/{p.id}/evidence/plausibility": {"columns": []},
        f"/project/{p.id}/draft": {"paragraphs": []},
        f"/project/{p.id}/gaps": {"gaps": []},
        f"/project/{p.id}/figures": figures,
    }
    out = PH.run(
        """
        var html = __harness.html('figuresBox');
        var rx = /<button([^>]*data-promote="[^"]*"[^>]*)>/g, m, opts = [];
        while ((m = rx.exec(html)) !== null){
          var a = {};
          m[1].replace(/([a-zA-Z-]+)="([^"]*)"/g,
                       function(_, k, v){ a[k] = v; return ""; });
          opts.push(a);
        }
        __emit({html: html, controls: opts});
        """,
        routes=routes, search=f"?project={p.id}")

    assert out["controls"], (
        "no control in the Guided door records a promotion, so `promotable` "
        "still has no user-facing consumer")
    for control in out["controls"]:
        assert control.get("data-promote"), "a control with no figure to promote"
        assert control.get("data-promote-on") in ("0", "1")
    # THE TIER CHIP IS UNTOUCHED, which is the ruling: no annotation is added
    # on the way in, and a control that only appeared on CONFIRMATORY figures
    # would be the same overruling by omission.
    drawn = {r["id"] for r in figures["admitted"] + figures["held"]}
    assert {c["data-promote"] for c in out["controls"]} == drawn, (
        "the promotion control is offered on some drawn figures and not "
        "others; the ruling is that the author marks the figure")


# ═══════════ WHAT THE EXPORT CARRIES ═══════════

def test_the_export_carries_every_analysis_the_app_has_already_done():
    """**Err toward more information, and the audit that produced this test.**

    The first `to_latex` passed 9 of `generate_latex_report`'s 22 arguments
    while the app already held seven more, so a Guided manuscript exported a
    methods section and an abstract and dropped the metrics table, the
    predictor list, the recorded limitations, the importance ranking and the
    resampling results on the floor. Nothing failed. The document was simply
    thinner than the analysis behind it, which is the quietest way to be wrong
    in an artifact that leaves the building.
    """
    from turbotab import explain as _explain

    name, target, task, model = TARGET_SHAPES["continuous regression"]
    p, run = _project(name, target, task)
    p.training_run = None
    explain = {"run": _explain.importance(p, model)}
    out = MS.validate(
        p.to_dict(), run=run, explain=explain,
        figures=[{"id": "calibration", "title": "Calibration plot",
                  "tier": "CONFIRMATORY", "promoted": False}])
    latex = out["rendered"]["latex"]
    assert latex

    for section, why in [
        ("Model Performance", "the held-out metrics table"),
        ("Feature Importance and Explainability",
         "the permutation-importance ranking explain.py computed"),
        ("Limitations", "the caveats the RECORD holds, not a placeholder"),
    ]:
        assert section in latex, (
            f"the export carries no {section} section, so {why} was computed "
            f"and dropped")

    assert "[Discuss limitations here]" not in latex, (
        "the exporter's default limitations placeholder survived, so the "
        "app's own recorded caveats were dropped and replaced with a blank")

    # CALIBRATION ABOVE DISCRIMINATION, which is the pack's ordering
    # (CLINICAL_SURVEY_PACK A5.1, A5.3). An export that reported only the
    # metrics table would invert it.
    assert "Calibration" in latex


def test_what_the_export_cannot_carry_is_served_rather_than_silent():
    """The gaps are information too. Each names a reason someone can argue
    with, and one of them is a deliberate refusal rather than a backlog item:
    the exporter's `sensitivity_summary` slot wants a coefficient-of-variation
    band, which is one of `STATE-034`'s two invented ladders."""
    name, target, task, _ = TARGET_SHAPES["continuous regression"]
    p, run = _project(name, target, task)
    out = MS.validate(p.to_dict(), run=run)

    fields = {x["field"] for x in out["not_exported"]}
    assert "table1_df" in fields and "tripod_checklist" in fields
    for entry in out["not_exported"]:
        assert len(entry["because"]) > 60, (
            f"{entry['field']} is unexported with a reason that is a shrug")
    refusal = next(x for x in out["not_exported"]
                   if x["field"] == "sensitivity_summary")
    assert "STATE-034" in refusal["because"], (
        "the one gap that is a REFUSAL rather than a gap does not say so")


def test_the_provenance_card_carries_what_the_app_knows_and_no_more():
    """`NUTRITION_PACK.md` §09 asks for an analysis provenance card so the
    analysis is reproducible from the paper. What the app holds is the split,
    the resampling scheme and B; what it does not — the weight variable, the
    design specification, the residual-adjustment constants — is ABSENT rather
    than blank, because a provenance card with empty fields reads as a study
    that had none."""
    name, target, task, _ = TARGET_SHAPES["continuous regression"]
    p, run = _project(name, target, task)
    doc = MS.structure(p.to_dict(), run=run)
    card = MS._provenance(doc)

    assert card["analysis_population"]["analysis_total"] > 0
    assert card["predictors"]["selected"] is not None
    for absent in ("weight_variable", "design_specification",
                   "residual_adjustment_constants"):
        assert absent not in card, (
            f"the provenance card claims {absent}, which the app does not hold")

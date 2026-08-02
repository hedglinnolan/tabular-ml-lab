"""turbotab.manuscript — the draft as DATA, checked by the machine that exists.

`GUIDED-107`, and the product owner has ruled on both open questions
(`PRODUCT_VISION.md`, the subsection above the resolution statement). This
module implements the rulings; it does not re-derive them.

> **The manuscript is data before it is a document.** `draft.py` composes one
> structured document and two thin renderers emit Markdown and LaTeX… The
> larger scope was chosen over shipping Markdown alone, and the reason is
> downstream: **L10's checklist engine has to read the manuscript**, and a
> checklist cannot be run against prose.

> **A marked figure is promoted as the author marked it.** No tier annotation
> is added on the way in… *The manuscript is the author's document. The app
> drafts it; the researcher signs it.*

## The part with a correctness consequence

`ml/manuscript_validator.py::validate_manuscript_bundle` already extracts the
analysis *n*, the final predictor count and Table 1's overall *n* from rendered
text and **checks them against each other**; it checks that the model named in
the prose is the model selected, and flags metric terms invalid for the task
type. It has been reachable only from `pages/10`, and `turbotab/draft.py`
imports nothing from `ml/` at all.

**`AUDIT-001` was exactly this class** — a section composed correctly in
isolation, asserting what no other section supported — and Report shipped at
L36 without the machine built to catch it. So this reuses the validator rather
than writing a second one, and the adapter below is the whole cost of reuse:
the validator reads Classic's headings, so the Markdown renderer emits those
headings.

**That is a real constraint and it is stated rather than hidden.** The renderer
is not free to name its sections whatever reads best, because a heading is an
interface here. `_HEADINGS` is where that coupling lives, in one place, with
the validator's extractor named beside each entry.

## The separation the fourth ruling asks for

A promoted `EXPLORATORY` figure is **not** caveated in the prose — the author
gets the document they asked for — and **is** reported by the validator, as a
cross-section check. `promoted_exploratory` below is that check. The author
gets their document and a separate honest list of what a reviewer will notice.

## What is deliberately not here

**The LaTeX renderer.** `ml/latex_report.py` is 1,066 lines, already detainted,
imports headless, and is the exporter Classic uses — so *"one core, no forks"*
says Guided must render through it rather than growing a private one. That is a
mapping job onto its expected context dict, and the loop's scope note says to
ship the validator wiring and the structured document first and defer it. It is
deferred, and `GUIDED-115` is what it is deferred as. The structured document
is what makes it a renderer rather than a rewrite.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from turbotab import draft as _draft

#: Guided section key → the heading the VALIDATOR extracts, and which extractor
#: reads it. The coupling is real: `_extract_section(methods_text, "Study
#: Design", level=3)` is a literal in `ml/manuscript_validator.py`, so a
#: renderer that called this section "Population" would silently disable the
#: check rather than fail it — a validator that finds nothing reports nothing
#: wrong.
_HEADINGS = {
    # key: (markdown level, heading text, which document it belongs to)
    "target": (3, "Study Design", "methods"),
    "data": (3, "Data Sources", "methods"),
    "explore": (3, "Exploratory Analysis", "methods"),
    "preprocess": (3, "Missing Data", "methods"),
    "features": (3, "Predictor Variables", "methods"),
    "limitations": (3, "Limitations", "methods"),
}

#: Which of those headings the validator actually EXTRACTS, and therefore which
#: sections are checked. Two of six — and the gap was found by the test that
#: asserts this list against `ml/manuscript_validator.py`'s own source.
#:
#: The four others are rendered anyway, because a manuscript needs them and the
#: validator's coverage is not the same question as the document's. What must
#: not happen is a reader taking a clean validation report as covering the
#: whole document: `unchecked_sections` says which parts nothing looked at.
#: `GUIDED-117`.
CHECKED_HEADINGS = frozenset({"Study Design", "Predictor Variables"})

#: Sections the VALIDATOR looks for that `draft.SECTIONS` has no source for,
#: and this is the first thing wiring the validator found.
#:
#: `draft.py` folds over DECISIONS, and Guided records no decision that says
#: how a model was developed or how it was evaluated — those are properties of
#: a *run*, not of a choice, so the fold has nothing to put there. A manuscript
#: exported today therefore states its analysis population, its cohort, its
#: preprocessing and its predictors, and never says which model was fitted or
#: how it was scored.
#:
#: **That is `AUDIT-001`'s class**: a section composed correctly in isolation,
#: asserting what no other section supports. The validator's *"model names
#: match between development and evaluation sections"* check cannot pass,
#: because neither section exists — and a validator that finds nothing reports
#: nothing wrong, so this is surfaced as a named gap rather than left as a
#: silent FAIL somebody reads as a formatting problem. Filed as `GUIDED-116`.
REQUIRED_BUT_UNSOURCED = {
    "Model Development": (
        "Guided records decisions, and which model was fitted is a property of "
        "the run rather than of a decision, so the draft has no source for "
        "this section. The run holds it; nothing carries it into the "
        "manuscript."),
    "Model Evaluation": (
        "Same source gap: the held-out metrics live on the run and the draft "
        "folds over decisions, so the manuscript states the analysis "
        "population and never says how the model was scored."),
}

#: The one heading the validator reads out of the REPORT rather than the
#: methods. Written as a constant and USED by `to_markdown`, so the literal
#: appears once — the same coupling `_HEADINGS` documents, and the same failure
#: if it drifts: `_extract_section(report_text, "Abstract (Draft)", level=2)`
#: finds nothing and reports nothing wrong.
ABSTRACT_HEADING = (2, "Abstract (Draft)")


def structure(project_dict: Dict[str, Any],
              *, run: Optional[Dict[str, Any]] = None,
              figures: Optional[List[Dict[str, Any]]] = None
              ) -> Dict[str, Any]:
    """The manuscript as data: sections, counts, figures, and the gaps.

    One object, from which both renderers and the checklist engine read. The
    counts are lifted out of the prose and carried as NUMBERS — which is the
    point of the ruling: a checklist cannot be run against prose, and neither
    can a consistency check that has to compare two of them.
    """
    body = _draft.draft(project_dict)
    counts = _counts(project_dict, run)
    return {
        "sections": body["sections"],
        "standfirst": body["standfirst"],
        "gap_marker": body["gap_marker"],
        "n_gaps": body["n_gaps"],
        "n_sentences": body["n_sentences"],
        "is_empty": body["is_empty"],
        # THE MACHINE-READABLE HALF. `population_counts` and `feature_counts`
        # are the validator's own key names, so the adapter is a pass-through
        # rather than a translation that could drift.
        "context": counts,
        "figures": list(figures or []),
        # THE SECTIONS A MANUSCRIPT NEEDS AND THIS DRAFT CANNOT SOURCE, named
        # rather than left as an absence. See `REQUIRED_BUT_UNSOURCED`.
        "unsourced_sections": [{"heading": h, "because": why}
                               for h, why in REQUIRED_BUT_UNSOURCED.items()],
        # WHICH RENDERED SECTIONS NOTHING CHECKS. A validation report that
        # listed only what it found would let its silence read as coverage.
        "unchecked_sections": sorted(
            heading for _, (_, heading, where) in _HEADINGS.items()
            if where == "methods" and heading not in CHECKED_HEADINGS),
        "task_type": project_dict.get("task_type") or "",
        "target": project_dict.get("target") or "",
    }


def _counts(project_dict: Dict[str, Any],
            run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The numbers the validator compares against each other.

    Empty where the app does not hold one. A count guessed here would be a
    number the validator then confirms against the prose it came from, which is
    a check that can only pass — worse than no check, because it reports
    agreement.
    """
    lockbox = project_dict.get("lockbox") or {}
    out: Dict[str, Any] = {"population_counts": {}, "feature_counts": {},
                           "included_models": []}
    if run:
        n_train = run.get("n_train")
        n_test = run.get("n_test")
        if n_train is not None and n_test is not None:
            out["population_counts"] = {
                "analysis_total": int(n_train) + int(n_test),
                "train": int(n_train),
                "test": int(n_test),
            }
        out["included_models"] = [r.get("key") for r in (run.get("results") or [])
                                  if not r.get("error")]
        surviving = next((r.get("selected_features") for r in
                          (run.get("results") or [])
                          if r.get("selected_features")), None)
        if surviving is not None:
            out["feature_counts"] = {"final": len(surviving),
                                     "candidates": len(run.get("features") or [])}
    elif lockbox.get("n_total"):
        out["population_counts"] = {"analysis_total": int(lockbox["n_total"]),
                                    "test": int(lockbox.get("n_test") or 0)}
    return out


def to_markdown(doc: Dict[str, Any]) -> Dict[str, str]:
    """Two documents, because the validator reads two.

    `methods` carries the sections; `report` carries the abstract the validator
    cross-checks the methods against. Thin by design — every sentence is
    already composed by `draft.py`, and a renderer that reworded anything would
    put a third voice between the record and the manuscript.
    """
    by_key = {s["key"]: s for s in doc["sections"]}
    methods: List[str] = ["## Methods\n"]
    for key, (level, heading, where) in _HEADINGS.items():
        section = by_key.get(key)
        if not section or where != "methods":
            continue
        sentences = section.get("sentences") or []
        if not sentences:
            continue
        methods.append(f"{'#' * level} {heading}\n")
        for item in sentences:
            methods.append(item["text"])
        methods.append("")

    counts = doc.get("context", {}).get("population_counts") or {}
    level, heading = ABSTRACT_HEADING
    report: List[str] = [f"{'#' * level} {heading}\n"]
    if counts.get("analysis_total"):
        # THE ONE SENTENCE THIS MODULE COMPOSES, and it exists so the validator
        # has two independent statements of the analysis n to compare. It is
        # built from the COUNT rather than from the methods text, so a
        # disagreement between them is a real disagreement rather than a
        # copy of itself.
        report.append(
            f"We analyzed {counts['analysis_total']:,} participants"
            + (f", of whom {counts['test']:,} were held out for evaluation."
               if counts.get("test") else "."))
    else:
        report.append(
            f"{_draft.AUTHOR_GAP} — no model has been fitted, so the analysis "
            f"population is not yet fixed.")
    report.append("")
    return {"methods": "\n".join(methods), "report": "\n".join(report)}


def promoted_exploratory(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Figures the author promoted that carry the `EXPLORATORY` tier.

    **Not a caveat in the prose.** The ruling is that a marked figure is
    promoted as the author marked it, and that annotating the caption would be
    the second uncalibrated layer of caution this project forbids elsewhere: a
    warning printed on every promoted figure makes a real concern and a routine
    one read identically.

    So the tier stays on the figure and this is reported in the VALIDATION
    report instead — the author gets the document they asked for and a separate
    honest list of what a reviewer will notice.
    """
    out = []
    for figure in doc.get("figures") or []:
        if figure.get("promoted") and figure.get("tier") == "EXPLORATORY":
            out.append({
                "id": figure.get("id"),
                "title": figure.get("title") or figure.get("id"),
                "because": (
                    f"{figure.get('title') or figure.get('id')} is registered "
                    f"EXPLORATORY and appears in the results. That is your "
                    f"call and the app has not changed the caption; a reviewer "
                    f"is likely to ask what confirms it."),
            })
    return out


def validate(project_dict: Dict[str, Any], *,
             run: Optional[Dict[str, Any]] = None,
             figures: Optional[List[Dict[str, Any]]] = None
             ) -> Dict[str, Any]:
    """Render the manuscript and run `ml`'s validator over it.

    Reuses `validate_manuscript_bundle` rather than writing a second one:
    `AUDIT-008` is *the core already holds the capability and the path that
    needs it does not read it*, and this is that path.
    """
    doc = structure(project_dict, run=run, figures=figures)
    rendered = to_markdown(doc)
    promoted = promoted_exploratory(doc)

    try:
        from ml.manuscript_validator import validate_manuscript_bundle
    except Exception as exc:                                # pragma: no cover
        return {"available": False, "because": (
            f"The manuscript validator could not be loaded ({exc}), so this "
            f"draft has not been checked. It is not reported as passing."),
            "rows": [], "promoted_exploratory": promoted, "document": doc}

    report = validate_manuscript_bundle(
        doc["context"], rendered["methods"], rendered["report"],
        # NO LATEX YET, and passing the Markdown here would be worse than
        # passing nothing: the validator's LaTeX checks look for markdown
        # artifacts leaking into a LaTeX file, and handing them markdown would
        # manufacture failures about a document that does not exist.
        "", doc["task_type"] or "classification")

    rows = report.to_rows()
    # THE VALIDATOR'S FAILURES ARE NOT ALL THE SAME KIND, and reporting them as
    # one list would let a structural gap read as a formatting slip. A check
    # that cannot pass because its section does not exist is separated out and
    # says so, in the words of the gap rather than of the check.
    unsourced = {u["heading"] for u in doc["unsourced_sections"]}
    for row in rows:
        row["blocked_by_missing_section"] = any(
            h.lower() in (row["Check"] + " " + row["Location"]).lower()
            for h in unsourced)
    return {
        "available": True,
        "rows": rows,
        "unsourced_sections": doc["unsourced_sections"],
        "unchecked_sections": doc["unchecked_sections"],
        "n_failed": len(report.failed_checks),
        "n_failed_for_a_missing_section": sum(
            1 for r in rows
            if r["Status"] == "FAIL" and r["blocked_by_missing_section"]),
        "passed": bool(report.passed) and not promoted,
        "promoted_exploratory": promoted,
        "document": doc,
        "rendered": rendered,
        "latex_deferred": (
            "LaTeX export is not wired yet. `ml/latex_report.py` is the "
            "exporter Classic uses and is where it will render from, so this "
            "draft has one exporter waiting rather than two that can "
            "disagree."),
    }

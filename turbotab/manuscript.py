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

## The second cross-section, and the boundary it was written for (`GUIDED-131`)

`turbotab/figures.py` states the companion rule as **admissibility** — *"the
bundle does not contain it"* — and `DOMAIN_SCIENCE.md` §01.6 says *refuse to let
a confirmatory figure into the results bundle without its validation companion.*

**The results bundle that leaves the building is this document.** Until L41
nothing here read the companion declaration at all: `figures.bundle` enforced it
on the `/figures` surface, and a figure it had held promoted into the manuscript
with `passed: True` and neither `why_held` nor the word *companion* anywhere in
the payload. `api.py` built the figure list from the whole registry and never
read the bundle.

`promoted_without_companion` is the twin of `promoted_exploratory`, and it is a
**report, not a refusal**, for the same reason: `PRODUCT_VISION.md` rules that a
marked figure is promoted as the author marked it, and refusing the promotion
would be the app overruling the author in their own document. What the ruling
also says is that **the record is not laundered**, and the validator is the
surface that keeps that true.

## The LaTeX renderer, wired at L39 (`GUIDED-115`)

`ml/latex_report.generate_latex_report` is the exporter Classic uses — 1,066
lines, detainted, headless — and *"one core, no forks"* says Guided renders
through it rather than growing a private one. `to_latex` below is a MAPPING,
not a second exporter: the structured document already holds every argument it
takes, which is what the ruling meant by *the manuscript is data before it is a
document*.

**And wiring it turned three validator checks from inert to live.** They look
for markdown artifacts and internal model keys leaking into a LaTeX file; with
`latex_text=""` they had nothing to read and reported PASS on a document
nobody had made. That is the shape of every finding in this file: silence read
as agreement.
"""
from __future__ import annotations

import re
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
    # `GUIDED-116`. Sourced from the RUN rather than from the decision fold —
    # see `_with_run_sections`. Both are extracted by the validator, which is
    # why its *model names match between development and evaluation* check
    # could never pass before they existed.
    "models": (3, "Model Development", "methods"),
    "evaluation": (3, "Model Evaluation", "methods"),
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
CHECKED_HEADINGS = frozenset({"Study Design", "Predictor Variables",
                              "Model Development", "Model Evaluation"})

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
              figures: Optional[List[Dict[str, Any]]] = None,
              explain: Optional[Dict[str, Any]] = None,
              sensitivity: Optional[Dict[str, Any]] = None,
              instability: Optional[Dict[str, Any]] = None,
              strobe_nut: Optional[Dict[str, Any]] = None,
              ) -> Dict[str, Any]:
    """The manuscript as data: sections, counts, figures, and the gaps.

    One object, from which both renderers and the checklist engine read. The
    counts are lifted out of the prose and carried as NUMBERS — which is the
    point of the ruling: a checklist cannot be run against prose, and neither
    can a consistency check that has to compare two of them.
    """
    body = _draft.draft(project_dict)
    counts = _counts(project_dict, run)
    sections = _with_run_sections(body["sections"], run, counts)
    sections = _with_analysis_sections(sections, explain, sensitivity,
                                       instability)
    sections = _with_strobe_nut(sections, strobe_nut)
    return {
        "sections": sections,
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
        # WHAT IS STILL UNSOURCED, which is now conditional rather than
        # permanent: with a run, both sections have a source; without one, the
        # manuscript describes an analysis nobody has fitted and says which
        # parts are missing and why (`GUIDED-116`).
        "unsourced_sections": (
            [] if run else [{"heading": h, "because": why}
                            for h, why in REQUIRED_BUT_UNSOURCED.items()]),
        # WHICH RENDERED SECTIONS NOTHING CHECKS. A validation report that
        # listed only what it found would let its silence read as coverage.
        "unchecked_sections": sorted(
            heading for _, (_, heading, where) in _HEADINGS.items()
            if where == "methods" and heading not in CHECKED_HEADINGS),
        "task_type": project_dict.get("task_type") or "",
        "target": project_dict.get("target") or "",
        # EVERYTHING ELSE THE APP HOLDS, carried on the document rather than
        # recomposed by each renderer. See `to_latex` for why the list is this
        # long and `NOT_EXPORTED` for what is still missing.
        "explain": explain or None,
        "sensitivity": sensitivity or None,
        "instability": instability or None,
        "run": run or None,
        "strobe_nut": strobe_nut or None,
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
            # THE VALIDATOR'S OWN KEY NAMES. `train_n`/`val_n`/`test_n` is
            # what `validate_manuscript_bundle` sums, and the first version
            # used `train`/`test` — so the reconciliation check compared 300
            # against 0 and failed for a reason that had nothing to do with the
            # manuscript. A key is an interface here exactly as a heading is.
            #
            # `val_n` is 0 and stated rather than omitted: Guided has two
            # partitions, and a missing key would let the sum quietly agree for
            # the wrong reason on a project that did have three.
            out["population_counts"] = {
                "analysis_total": int(n_train) + int(n_test),
                "train_n": int(n_train),
                "val_n": 0,
                "test_n": int(n_test),
            }
        out["included_models"] = [r.get("key") for r in (run.get("results") or [])
                                  if not r.get("error")]
        surviving = next((r.get("selected_features") for r in
                          (run.get("results") or [])
                          if r.get("selected_features")), None)
        candidates = len(run.get("features") or [])
        if surviving is not None:
            out["feature_counts"] = {"selected": len(surviving),
                                     "candidates": candidates,
                                     "selection_ran": True}
        elif candidates:
            # NO SELECTION WAS RECORDED, so every candidate IS the final set.
            # Stating that is not a guess: the run's `features` list is the
            # columns the model was handed. Leaving it absent made the
            # validator fall back to `len(feature_names_for_manuscript)`, which
            # is 0 on a Guided project, so the check compared the abstract's
            # silence against zero and failed for a reason that was about
            # neither.
            out["feature_counts"] = {"selected": candidates,
                                     "candidates": candidates,
                                     "selection_ran": False}
    elif lockbox.get("n_total"):
        out["population_counts"] = {"analysis_total": int(lockbox["n_total"]),
                                    "test_n": int(lockbox.get("n_test") or 0)}
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
        # THE ABSTRACT, and it exists so the validator has two independent
        # statements of the analysis n to compare. `_extract_analysis_n`'s own
        # third pattern, because the wording is an interface.
        report.append(
            f"A dataset of {counts['analysis_total']:,} observations was "
            f"analyzed"
            + (f", of which {counts['test_n']:,} were held out for "
               f"evaluation." if counts.get("test_n") else "."))
        features = (doc.get("context", {}).get("feature_counts") or {})
        if features.get("selected") is not None:
            report.append(
                f"The final modeling set contained {features['selected']:,} "
                f"predictors.")
    else:
        report.append(
            f"{_draft.AUTHOR_GAP} — no model has been fitted, so the analysis "
            f"population is not yet fixed.")
    report.append("")
    return {"methods": "\n".join(methods), "report": "\n".join(report)}


def to_latex(doc: Dict[str, Any]) -> str:
    """The manuscript as LaTeX, rendered by the exporter Classic uses.

    `GUIDED-115`. A mapping onto `generate_latex_report`'s arguments and
    nothing more — every one of them is already on the structured document,
    which is the argument for having built that first.

    Returns `""` where the document has no analysis population, because a
    manuscript with no fit has no results section to render and an empty
    template would be a document asserting a study that does not exist.
    """
    from ml.latex_report import generate_latex_report

    context = doc.get("context") or {}
    counts = context.get("population_counts") or {}
    # GATED ON A FITTED MODEL, not merely on a population count. `_counts`
    # falls back to the lockbox's own total when there is no run, which is the
    # right input for the ABSTRACT's cohort sentence and the wrong one here: a
    # LaTeX manuscript has a results section, and rendering the template around
    # an analysis nobody ran would be a document asserting a study that does
    # not exist.
    if not counts.get("analysis_total") or not context.get("included_models"):
        return ""
    rendered = to_markdown(doc)
    run = doc.get("run") or {}
    task_type = doc.get("task_type") or "regression"

    # THE METRICS TABLE. `_metrics_to_latex_table` keys on the exporter's own
    # metric names and reads `res["metrics"]`, so this is a reshape of
    # `run.results` and not a recomputation — the numbers are the ones the run
    # produced, and a second derivation of a held-out metric is the last thing
    # this app should grow.
    model_results = {
        model_name(r): {"metrics": r.get("metrics") or {}}
        for r in (run.get("results") or []) if not r.get("error")
    }

    return generate_latex_report(
        methods_section=rendered["methods"],
        abstract=rendered["report"].split("\n", 1)[-1].strip(),
        task_type=task_type,
        target_name=doc.get("target") or "outcome",
        n_total=int(counts.get("analysis_total") or 0),
        n_train=int(counts.get("train_n") or 0),
        n_val=int(counts.get("val_n") or 0),
        n_test=int(counts.get("test_n") or 0),
        # EVERYTHING BELOW WAS DROPPED BY THE FIRST VERSION. See `NOT_EXPORTED`
        # for the arguments that are still unfilled and why; these are the ones
        # the app already held while the export left them behind.
        model_results=model_results,
        feature_names=list(run.get("features") or []),
        limitations=_limitations(doc),
        calibration_text=_calibration_text(doc),
        explainability_summary=_explainability(doc),
        data_config=_provenance(doc),
        manuscript_context=doc.get("context") or {},
    )


def _limitations(doc: Dict[str, Any]) -> str:
    """The limitations the RECORD already holds, rather than a placeholder.

    `draft.py` routes `trim_training_rows`, `acknowledge_blocker` and the
    per-model preparation caveat here, and every one of them is a thing the
    study cannot conclude. The exporter's default is
    `"[Discuss limitations here]"`, so shipping without this meant an export
    that silently dropped the app's own recorded caveats and replaced them
    with a blank.
    """
    for section in doc.get("sections") or []:
        if section["key"] == "limitations" and section.get("sentences"):
            return " ".join(str(i["text"]) for i in section["sentences"])
    return ""


def _calibration_text(doc: Dict[str, Any]) -> str:
    """What the app can say about calibration, or nothing.

    `CLINICAL_SURVEY_PACK.md` §A5.1 and §A5.3 rank calibration ABOVE
    discrimination, so an export that reported only the metrics table would
    invert the field's own ordering. The app draws a calibration figure; what
    it does not yet do is carry the intercept and slope into the manuscript,
    which is why this returns the figure's presence rather than its numbers
    and says so.
    """
    # `AUDIT-015`. THIS GATE COULD NEVER FIRE, and the reason is a shape
    # change one layer up. It used to read membership — `"calibration" in
    # {f["id"] for f in doc["figures"]}` — which was right when `figures` was
    # the list of figures this project drew. `GUIDED-131` made it the WHOLE
    # REGISTRY, carrying drawability as a per-row field, so the id is
    # unconditionally present and the guard became a tautology.
    #
    # The result was the manuscript stating *"Calibration was assessed
    # graphically"* on every project with a fitted model — including
    # regression, where the app's own `/figures` surface says the opposite
    # sentence for the same project: *"Calibration is a claim about predicted
    # PROBABILITIES, and this is a regression task."*
    #
    # `drawn is None` means the bundle could not say, and silence is what the
    # governing rule permits there. Only `True` earns the sentence.
    calibration = next((f for f in (doc.get("figures") or [])
                        if f.get("id") == "calibration"), None)
    if calibration is None or calibration.get("drawn") is not True:
        return ""
    return ("Calibration was assessed graphically; the calibration plot is "
            "reported with its intercept, slope and C-statistic. "
            f"{_draft.AUTHOR_GAP} — state whether calibration in the "
            f"clinically relevant risk range is adequate for the intended use.")


def _explainability(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The importance ranking, in the exporter's own shape.

    `turbotab/explain.py` computes permutation importance on the HELD-OUT rows
    and composes its own methods sentence, and until now none of it reached the
    export.
    """
    explain = doc.get("explain") or {}
    run = explain.get("run") or {}
    ranked = run.get("ranked") or []
    if not ranked:
        return None
    return {
        "top_features": [str(r.get("feature") or r.get("column") or "")
                         for r in ranked[:5]],
        "permutation_importance_available": True,
        # SAID, NOT IMPLIED. `explain.py` records both reasons SHAP is absent,
        # and an export that left the key off would let a reader assume it was
        # simply not run.
        "shap_available": False,
    }


def _provenance(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The analysis provenance card `NUTRITION_PACK.md` §09 asks for.

    > *analysis provenance card: package versions, seeds, weight variable,
    > design specification, and the exact residual-adjustment constants — so
    > the analysis is reproducible from the paper.*

    What the app holds is the seed, the split, the sampling scheme and the
    number of resamples. What it does not hold — the weight variable, the
    design specification, the residual-adjustment constants — belongs to the
    complex-survey and energy-adjustment work that is not built, so it is
    absent rather than blank.
    """
    context = doc.get("context") or {}
    instability = doc.get("instability") or {}
    out: Dict[str, Any] = {
        "analysis_population": context.get("population_counts") or {},
        "predictors": context.get("feature_counts") or {},
    }
    for key, entry in (instability.get("runs") or {}).items():
        sampling = ((entry.get("prediction_instability") or {})
                    .get("sampling") or {})
        if sampling:
            out["resampling_scheme"] = sampling.get("scheme")
            out["resampling_disclosure"] = sampling.get("sentence")
        b = (entry.get("prediction_instability") or {}).get("b_completed")
        if b:
            out["bootstrap_resamples"] = b
        break
    return out


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


def promoted_without_companion(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Promoted figures whose declared validation companion is not in the results.

    **`GUIDED-131`.** The companion rule is stated as admissibility and
    `figures.bundle` enforces it — on the `/figures` surface. It had no consumer
    at the boundary it was written for, which is this document: the results
    bundle that leaves the building is the manuscript, and a CONFIRMATORY figure
    the bundle had *held* promoted straight through it with `passed: True`.

    **A report, not a refusal.** The ruling is that a marked figure is promoted
    as the author marked it; a route that declined the promotion would overrule
    the author in their own document. The other half of the same ruling is that
    the record is not laundered, and this is where that is kept true.

    **Companions are resolved from `figures.REGISTRY` and from nowhere else.**
    Not from a `companions` key on the passed-in row, which a caller could
    forget, get wrong, or — worse — supply from a literal that no project can
    produce. `GUIDED-134` is exactly that failure one layer down: a test handed
    `bundle()` a bare dict key that was never a registered figure, and the rule
    looked enforced for six loops. A check whose stand-in cannot be faked is the
    fix for that class, so this reads the real registry or it reports nothing.

    `drawn` on a figure row, where the caller supplied it, says whether the
    companion could have been put in the results at all. It changes the
    sentence, never the verdict: a companion the author *can* promote and one
    this project cannot draw are different problems for the author and the same
    problem for the reviewer.
    """
    from turbotab import figures as _figures
    import turbotab.figure_specs                       # noqa: F401 — registers

    rows = list(doc.get("figures") or [])
    by_id = {str(f.get("id")): f for f in rows}
    promoted = {fid for fid, f in by_id.items() if f.get("promoted")}

    out: List[Dict[str, Any]] = []
    for figure_id in sorted(promoted):
        row = by_id[figure_id]
        title = row.get("title") or figure_id
        spec = _figures.REGISTRY.get(figure_id)
        if spec is None:
            # NOT `continue`. `figures.bundle` skips an unregistered id on a
            # line marked `# pragma: no cover`, and that skip is what let
            # `GUIDED-128` hide for six loops. A promoted id the registry does
            # not know is a document referring to a figure that does not exist,
            # which is worse than a missing companion, not smaller.
            out.append({
                "id": figure_id, "title": title, "missing_companions": [],
                "because": (
                    f"{title} is promoted into the results and no figure with "
                    f"that id is registered, so nothing can say what it shows "
                    f"or what would validate it.")})
            continue
        missing = [c for c in spec.companions if c not in promoted]
        if not missing:
            continue

        undrawable = [c for c in missing
                      if by_id.get(c, {}).get("drawn") is False]
        names = ", ".join(missing)
        plural = "" if len(missing) == 1 else "s"
        because = (
            f"{title} is registered {spec.tier} and its companion "
            f"figure{plural} {names} {'is' if len(missing) == 1 else 'are'} not "
            f"in these results. A confirmatory figure without its validation "
            f"beside it is the shape every circular-figure defect takes, and a "
            f"reviewer is likely to ask for it. The app has not changed your "
            f"document.")
        if undrawable:
            because += (
                f" This project cannot draw {', '.join(undrawable)} at all, so "
                f"promoting {'it' if len(undrawable) == 1 else 'them'} is not "
                f"the remedy — the claim is the one that needs narrowing.")
        out.append({
            "id": figure_id, "title": title, "tier": spec.tier,
            "missing_companions": missing,
            "undrawable_companions": undrawable,
            "because": because})
    return out


# ══ `GUIDED-179` · A CHECK THAT CANNOT RUN SAYS WHICH QUANTITY IS MISSING ══
#
# `AGENT_ONBOARD.md` §00 gives the app three branches: it may assert truly, it
# may be **silent**, and it may **refuse**. *"Expected analysis N=None, abstract
# N=None, study design N=None"* is a **FOURTH**, and nobody authorized it. It is
# not silence — a sentence is on the screen. It is not an assertion — `None`
# claims nothing. It is not a refusal — it does not say what is missing or why.
# It is a Python repr rendered to a researcher, and the researcher's only
# available reading is *the app is broken*.
#
# The vocabulary for the third branch already exists here and is not reinvented.
# `figures.NOT_ESTIMABLE` (`figures.py:430`) is this project's token for a
# number **not shown because there is not one**, and `figure_specs.py:171-178`
# pairs it with a `why` naming the cause. The closing sentence below is borrowed
# verbatim from `figures._ABSENT` (`figures.py:432-434`) so the checklist and the
# annotation box say the absence in one voice.

#: `label=None` inside a validator `Detail`. The label is the quantity's own
#: name in the check's words, and the char class excludes `,` so a detail
#: carrying three of them yields three labels rather than one greedy run.
_MISSING_VALUE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 _:'()-]*?)\s*=\s*"
    r"(?P<value>None|nan|NaN|NaT)(?![A-Za-z0-9_])")

#: The validator's label, lowercased → (what a researcher would call it, and
#: WHO was supposed to supply it). Three kinds, because they have three
#: different causes and an author can act on only one of them:
#:
#: - `app` — a number `_counts` builds from the run or the lockbox.
#: - `prose` — a number the validator reads back out of the drafted manuscript.
#: - `unheld` — a key of Classic's export context that Guided's `_counts` never
#:   writes at all (`ml/latex_report.py:111-113` supplies both of these). Saying
#:   *the manuscript states none* about one of those would be false: nothing
#:   asked the manuscript for it.
#:
#: Collapsing the three would tell an author their draft is silent when the
#: truth is that nothing was fitted. An unknown label keeps its own name rather
#: than being renamed by a guess.
_QUANTITY_NAMES = {
    "expected analysis n": ("the analysis population", "app"),
    "analysis_total": ("the analysis population", "app"),
    "abstract n": ("the N stated in the abstract", "prose"),
    "study design n": ("the N stated in Study Design", "prose"),
    "expected predictors": ("the final predictor count", "app"),
    "abstract": ("the predictor count stated in the abstract", "prose"),
    "predictor section": (
        "the predictor count stated in Predictor Variables", "prose"),
    "table 1 overall n": ("Table 1's overall N", "app"),
    "best_metric_name": ("the metric the model was selected on", "unheld"),
    "manuscript_primary_model": ("the model named as primary", "unheld"),
}

#: WHY the number is absent. Which one is true is read off the document rather
#: than assumed: `_counts` returns an empty `population_counts` both when there
#: is no run AND when a run recorded no split, and those are not one sentence.
_NO_RUN_CAUSE = ("no model has been fitted in this project and its sealed "
                 "lockbox carries no row total")
_RUN_CAUSE = "the fitted run did not record it"
_PROSE_CAUSE = "the drafted manuscript states no such number"
_UNHELD_CAUSE = ("nothing in this project's manuscript context records one — "
                 "it is a field the Classic export door supplies and this door "
                 "does not")

#: Borrowed verbatim from `figures._ABSENT`. One voice for one absence.
_RENDER_IS_NOT_THE_FAULT = (
    "A number is not shown because there is not one, rather than because it "
    "failed to render.")

#: THE MECHANISM, SAID TO THE AUTHOR. `validate` takes `table1` as its own
#: parameter, independent of `run` (`manuscript.py:594-602`, `:646`), and
#: `api.get_manuscript` builds it from `project.working_table` rather than from
#: a fitted model. So Table 1 knows its N on a project that has never been
#: trained while `_counts` has none — one side of the comparison exists and the
#: other does not, and a reviewer reading `N=None` beside `N=288` would conclude
#: the app lost a number it never had.
_TABLE1_ASYMMETRY = (
    "Table 1 itself was built and describes {n:,} rows: it is generated from "
    "this project's working table, not from a run, which is why one side of "
    "this comparison exists and the other does not.")


def _and_list(items: List[str]) -> str:
    """`a`, `a and b`, `a, b and c`. No serial comma — the rest of this file
    does not use one."""
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _rows_that_say_what_is_missing(rows: List[Dict[str, Any]],
                                   doc: Dict[str, Any],
                                   table1: Optional[Any]
                                   ) -> List[Dict[str, Any]]:
    """Rewrite every `Detail` carrying a `None` into a sentence.

    **The check's own words and every real number it found are kept.** Each
    `label=None` becomes `label not estimable`, so *"Expected analysis N=None,
    Table 1 overall N=288"* keeps the 288 — that number is the informative half
    and a scrub that dropped it would trade one silence for another. The WHY is
    appended, naming the quantities in a researcher's words and the cause in the
    app's.

    Applied to PASS rows too, and the lead sentence differs. A passing check
    that mentions an absent quantity **did run and did pass** — *"selection
    metric language matches task type"* passes because no invalid metric term
    appears, not because a metric exists — so it gets the absence without the
    claim. Saying *this check has nothing to compare* over a PASS would be the
    fourth branch again with better grammar.
    """
    from turbotab.figures import NOT_ESTIMABLE

    run = doc.get("run")
    table1_n = None
    if table1 is not None:
        # No `or []` — a pandas Index has no truth value, and the validator's
        # own `_extract_table1_overall_n` iterates it exactly this way.
        for column in getattr(table1, "columns", []):
            match = re.search(r"Overall\s+\(N=([\d,]+)\)", str(column))
            if match:
                table1_n = int(match.group(1).replace(",", ""))
                break

    out: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        detail = str(row.get("Detail") or "")
        found = list(_MISSING_VALUE_RE.finditer(detail))
        row["missing_quantities"] = []
        row["cannot_run"] = False
        if not found:
            out.append(row)
            continue

        names: List[str] = []
        kinds = set()
        for match in found:
            label = match.group("label").strip()
            name, kind = _QUANTITY_NAMES.get(label.lower(), (label, "app"))
            if name not in names:
                names.append(name)
            kinds.add(kind)

        rewritten = _MISSING_VALUE_RE.sub(
            lambda m: f"{m.group('label').strip()} {NOT_ESTIMABLE}", detail)

        causes: List[str] = []
        if "app" in kinds:
            # A run that recorded no split is a different failure from no run at
            # all, and the author can act on only one of them.
            causes.append(_NO_RUN_CAUSE if not run else _RUN_CAUSE)
        if "prose" in kinds:
            causes.append(_PROSE_CAUSE)
        if "unheld" in kinds:
            causes.append(_UNHELD_CAUSE)

        cannot_run = row.get("Status") == "FAIL"
        verb = "is" if len(names) == 1 else "are"
        body = (f"{_and_list(names)} {verb} {NOT_ESTIMABLE}, because "
                f"{', and because '.join(causes)}.")
        sentence = (f"This check has nothing to compare: {body}" if cannot_run
                    # A PASS row opens the sentence itself, so the quantity name
                    # is capitalized rather than arriving mid-sentence after a
                    # full stop.
                    else body[:1].upper() + body[1:])
        parts = [rewritten.rstrip(), sentence]
        if row.get("Location") == "Table 1" and table1_n is not None:
            parts.append(_TABLE1_ASYMMETRY.format(n=table1_n))
        parts.append(_RENDER_IS_NOT_THE_FAULT)

        row["Detail"] = " ".join(p for p in parts if p)
        row["missing_quantities"] = names
        row["cannot_run"] = cannot_run
        out.append(row)
    return out


def _checklist_counts(rows: List[Dict[str, Any]],
                      unsourced: List[Dict[str, Any]],
                      promoted: List[Dict[str, Any]],
                      orphaned: List[Dict[str, Any]]) -> Dict[str, Any]:
    """**The header and the list count different populations, and both are
    right.** `GUIDED-179`, second half.

    The panel's header is `rows.length + " checks, " + failed.length + " unmet"`
    (`web/index.html:2770-2773`) and its body then renders the failed checks
    **plus** the unsourced sections, the promoted EXPLORATORY figures and the
    promoted figures missing a companion (`:2782-2812`). On a project with no
    run that reads *"13 checks, 4 unmet"* above **six** items.

    Neither number is wrong. 13 and 4 are exact about the validator's checks;
    the six is exact about what a reviewer will notice. What is missing is the
    sentence saying they are different populations — and reconciling them by
    making the header count the list would erase a distinction this file already
    holds deliberately: *"a section the draft cannot source is not the same as a
    check that failed, and rendering them alike would let a structural gap read
    as a formatting slip."*

    So the payload SAYS which. It is served rather than rendered: `index.html`
    is outside this part's edit boundary, so the panel still shows the bare
    header and this sentence reaches no screen yet. That is a stated limit, not
    a claim of completion.
    """
    failed = [r for r in rows if r.get("Status") == "FAIL"]
    beyond = [
        (len(unsourced), "section(s) the draft cannot source at all"),
        (len(promoted), "promoted EXPLORATORY figure(s)"),
        (len(orphaned), "promoted figure(s) missing a validation companion"),
    ]
    n_beyond = sum(n for n, _ in beyond)
    named = [f"{n} {what}" for n, what in beyond if n]
    if n_beyond:
        because = (
            f"The header counts the validator's checks and the list shows more "
            f"than checks; both counts are right. There are {len(rows)} checks "
            f"and {len(failed)} of them are unmet. The list shows those "
            f"{len(failed)} and {n_beyond} further item(s) that are not "
            f"validator checks at all: {_and_list(named)}. A section the draft "
            f"cannot source is a structural gap the validator has no check for, "
            f"so it is listed here and not counted there, because rendering "
            f"them alike would let a structural gap read as a formatting slip.")
    else:
        because = (
            f"The header and the list count the same {len(failed)} item(s) "
            f"here: of {len(rows)} validator checks {len(failed)} are unmet, "
            f"and nothing beyond the checks was found to list beside them.")
    return {
        "n_checks": len(rows),
        "n_unmet_checks": len(failed),
        "n_items_listed": len(failed) + n_beyond,
        "n_listed_that_are_not_checks": n_beyond,
        "header_and_list_count_the_same_population": n_beyond == 0,
        "because": because,
    }


def validate(project_dict: Dict[str, Any], *,
             run: Optional[Dict[str, Any]] = None,
             figures: Optional[List[Dict[str, Any]]] = None,
             explain: Optional[Dict[str, Any]] = None,
             sensitivity: Optional[Dict[str, Any]] = None,
             instability: Optional[Dict[str, Any]] = None,
             table1: Optional[Any] = None,
             strobe_nut: Optional[Dict[str, Any]] = None,
             ) -> Dict[str, Any]:
    """Render the manuscript and run `ml`'s validator over it.

    Reuses `validate_manuscript_bundle` rather than writing a second one:
    `AUDIT-008` is *the core already holds the capability and the path that
    needs it does not read it*, and this is that path.
    """
    doc = structure(project_dict, run=run, figures=figures, explain=explain,
                    sensitivity=sensitivity, instability=instability,
                    strobe_nut=strobe_nut)
    rendered = to_markdown(doc)
    try:
        latex = to_latex(doc)
    except Exception as exc:                                # pragma: no cover
        latex = ""
        doc["latex_unavailable"] = str(exc)
    promoted = promoted_exploratory(doc)
    # `GUIDED-131`. Computed BEFORE the validator import, and served on the
    # unavailable branch too: this cross-section is the app's own, it does not
    # need `ml/` to run, and a companion gap that vanished when the validator
    # failed to load would be the silence this file exists to remove.
    orphaned = promoted_without_companion(doc)

    try:
        from ml.manuscript_validator import validate_manuscript_bundle
    except Exception as exc:                                # pragma: no cover
        return {"available": False, "because": (
            f"The manuscript validator could not be loaded ({exc}), so this "
            f"draft has not been checked. It is not reported as passing."),
            "rows": [], "promoted_exploratory": promoted,
            "promoted_without_companion": orphaned, "document": doc}

    report = validate_manuscript_bundle(
        doc["context"], rendered["methods"], rendered["report"],
        # THE REAL LATEX DOCUMENT (`GUIDED-115`). Until L39 this was `""`,
        # which left three checks inert — they look for markdown artifacts and
        # internal model keys leaking into a LaTeX file and had nothing to
        # read, so they reported PASS about a document nobody had made.
        latex, doc["task_type"] or "classification",
        # `GUIDED-122`. Until L40 this was omitted, so *Table 1 population
        # matches the analysis cohort* and *Table 1 includes all finalized
        # predictors* both short-circuited to PASS on a table that did not
        # exist. Two more checks that were not passing, merely not running —
        # the state L38 named for the LaTeX checks and L39 closed.
        table1_df=table1)

    # `GUIDED-179`. THE REPR NEVER LEAVES THIS FUNCTION. `to_rows` renders the
    # validator's details with f-strings, so an absent quantity arrives as the
    # literal `None` — see the check composed at
    # `ml/manuscript_validator.py:174-184`. This is the boundary that serves the
    # checklist to a researcher, so it is the boundary that owes them a
    # sentence; `ml/` keeps its own vocabulary for its own callers.
    rows = _rows_that_say_what_is_missing(report.to_rows(), doc, table1)
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
        # `GUIDED-179`. WHAT THE HEADER COUNTS AND WHAT THE LIST SHOWS, and the
        # sentence saying they are not the same population.
        "checklist_counts": _checklist_counts(
            rows, doc["unsourced_sections"], promoted, orphaned),
        # `GUIDED-131`. `orphaned` joins the conjunction rather than being
        # served beside a `passed: True`, which is the exact state the finding
        # was filed about: the document carried a held CONFIRMATORY figure and
        # reported itself clean.
        "passed": bool(report.passed) and not promoted and not orphaned,
        "promoted_exploratory": promoted,
        "promoted_without_companion": orphaned,
        "document": doc,
        "rendered": {**rendered, "latex": latex},
        "latex_bytes": len(latex),
        "latex_unavailable": doc.get("latex_unavailable"),
        # WHAT THE EXPORT STILL CANNOT CARRY, served rather than left to be
        # discovered. *Err toward more information* applies to the gaps too.
        "not_exported": [{"field": k, "because": v}
                         for k, v in NOT_EXPORTED.items()],
        "strobe_nut": doc.get("strobe_nut"),
        "table1_rows": 0 if table1 is None else int(len(table1)),
        "table1_columns": [] if table1 is None else [str(c) for c in table1.columns],
    }


def _with_run_sections(sections: List[Dict[str, Any]],
                       run: Optional[Dict[str, Any]],
                       counts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """**The second source. `GUIDED-116`.**

    `draft.py` is a pure fold over DECISIONS, and both missing sections
    describe a RUN — which model was fitted, on how many rows, and how it
    scored. That is why the fold had nothing to put there, and it is why the
    fix is not a new decision kind: nobody *decides* a held-out RMSE.

    So the run is a second source, folded in here rather than in `draft.py`,
    which stays the pure function it is documented as being. Without a run the
    sections are absent and `unsourced_sections` says why — a manuscript that
    described a model nobody had fitted would be worse than one that stops.

    Study Design also gains its analysis population from this source, and that
    matters more than it looks: the abstract asserts an *n* and until now the
    methods never stated one, so the validator's very first check — *analysis
    population is consistent across abstract and study design* — compared a
    number against nothing. **Both statements are rendered from
    `counts`**, which would make the check self-confirming, except that
    `counts` is built from the RUN and the seal's own sentence beside it is
    built from the RECORD. A project whose rows were trimmed after the seal has
    two numbers that disagree, and that is exactly the disagreement worth
    catching.
    """
    if not run:
        return sections

    by_key = {s["key"]: s for s in sections}
    out = [dict(s) for s in sections]

    population = counts.get("population_counts") or {}
    total = population.get("analysis_total")
    if total is not None and "target" in by_key:
        # THE VALIDATOR'S OWN PHRASING. `_extract_analysis_n` reads three fixed
        # patterns and nothing else, so a sentence that says the same thing in
        # different words leaves the check comparing a number against `None` —
        # which passes as *nothing found* rather than failing. The wording is
        # an interface, exactly as `_HEADINGS` is.
        line = (f"A total of {total:,} observations were available for "
                f"analysis, of which {population.get('train_n', 0):,} were "
                f"used for model development and "
                f"{population.get('test_n', 0):,} were held out for "
                f"evaluation.")
        for section in out:
            if section["key"] == "target":
                section["sentences"] = [
                    {"text": line, "kind": "derived", "subject": "",
                     "at": None, "has_gap": False}] + list(section["sentences"])

    features = counts.get("feature_counts") or {}
    if features.get("selected") is not None:
        # INTO `Predictor Variables`, because that is the section
        # `_extract_final_predictor_count` reads. Putting it in Model
        # Development would leave the check comparing the abstract against
        # `None`, which passes as *nothing found* rather than failing — the
        # silent-disable failure `GUIDED-117` is about.
        line = (
            f"Feature selection retained {features['selected']:,} predictors "
            f"for final modeling from {features.get('candidates', 0):,} "
            f"candidates."
            if features.get("selection_ran") else
            f"No feature selection was performed, so all "
            f"{features['selected']:,} candidate predictors were carried "
            f"forward; the final modeling set contained "
            f"{features['selected']:,} predictors.")
        for section in out:
            if section["key"] == "features":
                section["sentences"] = list(section["sentences"]) + [_line(line)]

    out.append({"key": "models", "title": "Model development",
                "sentences": _development(run, counts), "waiting_for": None})
    out.append({"key": "evaluation", "title": "Model evaluation",
                "sentences": _evaluation(run), "waiting_for": None})
    return out


def model_name(result: Dict[str, Any]) -> str:
    """The model's name IN THE VALIDATOR'S VOCABULARY.

    `ml/narrative_engine._MODEL_NAMES` and `ml/model_registry` disagree —
    `histgb_clf` is *"Histogram Gradient Boosting (Classifier)"* in the first
    and *"(Classification)"* in the second — and the validator's *model names
    match between development and evaluation sections* check reads the first.
    Writing the registry's name made that check fail on any project using a
    model whose two names differ, which is the same coupling `_HEADINGS`
    documents: a manuscript is read by a machine with its own vocabulary.

    **The disagreement is filed as `GUIDED-124`** rather than papered over
    here; this reads the validator's table where it has the key so the
    manuscript and its checker agree today, and falls back to the registry's
    name so a model neither table knows still gets a name a reader recognizes.
    And it must not fall back to the KEY — the validator's own *no internal
    model keys leak into export text* check forbids exactly that.
    """
    key = str(result.get("key") or "")
    try:
        from ml.narrative_engine import _MODEL_NAMES
        known = _MODEL_NAMES.get(key) or _MODEL_NAMES.get(key.lower())
    except Exception:                                       # pragma: no cover
        known = None
    return str(known or result.get("name") or key)


def _line(text: str) -> Dict[str, Any]:
    return {"text": text, "kind": "run", "subject": "", "at": None,
            "has_gap": _draft.AUTHOR_GAP in text}


def _development(run: Dict[str, Any], counts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Which models were fitted, on what, and where the plan could not be kept.

    The DIVERGENCES travel, because `GUIDED-095`'s discipline is that a model
    which cannot honor a recorded declaration says so per model — and a methods
    section that reported the plan without them would be describing the
    analysis the user specified rather than the one that ran.
    """
    results = [r for r in (run.get("results") or []) if not r.get("error")]
    names = [model_name(r) for r in results]
    lines: List[Dict[str, Any]] = []
    if not names:
        return [_line(f"No model completed a fit. {_draft.AUTHOR_GAP} — state "
                      f"why no model is reported.")]

    lines.append(_line(
        f"{_join(names)} {'were' if len(names) > 1 else 'was'} fitted on the "
        f"{run.get('n_train', 0):,} development observations. Every step that "
        f"estimates anything from data — imputation, scaling, encoding and any "
        f"feature selection — was fitted inside the model's own pipeline, so "
        f"no parameter was estimated from the held-out observations."))

    diverged = [(model_name(r), d)
                for r in results
                for d in ((r.get("plan") or {}).get("divergences") or [])]
    if diverged:
        for name, d in diverged:
            # `Divergence`'s OWN field names, checked against the dataclass
            # rather than guessed. The first version read `recorded`/`fitted`,
            # which do not exist, and rendered "the recorded plan could not be
            # applied as written:" followed by nothing — a methods sentence
            # that announces a caveat and does not deliver it, which is worse
            # than omitting it.
            requested = str(d.get("requested") or "").strip()
            applied = str(d.get("applied") or "").strip()
            why = str(d.get("why") or "").strip()
            subject = str(d.get("subject") or "").strip()
            if not (requested and applied):
                continue
            lines.append(_line(
                f"For {name}, the recorded handling of "
                f"`{subject}` could not be applied: {requested} was recorded "
                f"and {applied} was fitted"
                + (f", because {why.rstrip('.')}." if why else ".")))
    return lines


def _evaluation(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """How it scored, on the sealed rows, with the metric named.

    The metric names are the evaluator's own, and the validator flags terms
    invalid for the task type — so a regression section that said `accuracy`
    would be caught. That check has been running against a section that did not
    exist.
    """
    results = [r for r in (run.get("results") or []) if not r.get("error")]
    if not results:
        return [_line("No model produced a held-out score.")]
    lines = [_line(
        f"Performance was estimated on the {run.get('n_test', 0):,} held-out "
        f"observations, which were sealed before any exploration and were not "
        f"used to fit or to select anything.")]
    for result in results:
        metrics = result.get("metrics") or {}
        if not metrics:
            continue
        stated = ", ".join(f"{k} {v:.3f}" for k, v in sorted(metrics.items())
                           if isinstance(v, (int, float)))
        lines.append(_line(f"{model_name(result)}: {stated}."))
    if run.get("exploratory"):
        lines.append(_line(
            "This split is not a verified clean one, so these figures are "
            f"exploratory. {_draft.AUTHOR_GAP} — state what that means for "
            f"the conclusion."))
    return lines


def _join(names: List[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


#: **What `generate_latex_report` accepts that Guided still cannot supply**,
#: and why — because *err toward more information* only works if the gaps are
#: also information. Each entry is filed rather than left to be discovered.
#:
#: The audit that produced this list is the reason it exists: the first version
#: of `to_latex` passed **9 of the exporter's 22 arguments** while the app
#: already held seven more, so a Guided manuscript exported a methods section
#: and an abstract and dropped the metrics table, the predictor list, the
#: limitations paragraph, the importance ranking and the instability results on
#: the floor. Nothing failed; the document was simply thinner than the analysis.
NOT_EXPORTED = {
    "tripod_checklist": (
        "The checklist engine is `DOMAIN_SCIENCE` primitive 6 and is unbuilt "
        "(`GUIDED-111`). STROBE-nut is four checklists' worth of the same "
        "artifact and is the reason the manuscript had to become data first."),
    "stat_validation_summary": (
        "Guided has no hypothesis-testing step; `pages/09` is `classic-only` "
        "and blocked on figure tiering (`stats-correlation-test` in the "
        "register)."),
    "sensitivity_summary": (
        "DELIBERATELY NOT MAPPED. The exporter's slot expects Classic's shape "
        "— `seed_stability` with a `cv_percent` — and that coefficient-of-"
        "variation band is one of the two invented ladders `STATE-034` is "
        "about, which `turbotab/sensitivity.py` exists not to inherit. "
        "Guided's fork reports whether the leading model changed, so its own "
        "sentence goes into the methods section instead of being reshaped to "
        "fit a verdict system this app rejected."),
    "authors / affiliation / title": (
        "The author's, not the app's. They render as the exporter's own "
        "placeholders, which is the `[AUTHOR REQUIRED]` rule in another "
        "vocabulary."),
}


def _with_analysis_sections(sections: List[Dict[str, Any]],
                            explain: Optional[Dict[str, Any]],
                            sensitivity: Optional[Dict[str, Any]],
                            instability: Optional[Dict[str, Any]],
                            ) -> List[Dict[str, Any]]:
    """Fold the analyses that are neither decisions nor the primary run.

    Each already composes its own methods sentence — `explain.methods_sentence`,
    `sensitivity.methods_sentence`, the instability caption — and this puts
    them in the document rather than leaving them on three surfaces the export
    never visits. That is the same `AUDIT-008` shape as the validator itself:
    the app holds the sentence and the path that needs it does not read it.
    """
    out = [dict(s) for s in sections]
    extra: List[Dict[str, Any]] = []

    if explain and explain.get("run"):
        run = explain["run"]
        line = run.get("methods_sentence") or run.get("narrative")
        if line:
            extra.append(_line(str(line)))
    if sensitivity and sensitivity.get("methods_sentence"):
        extra.append(_line(str(sensitivity["methods_sentence"])))
    if instability:
        for key, entry in (instability.get("runs") or {}).items():
            caption = entry.get("prediction_caption")
            if caption:
                extra.append(_line(str(caption)))
            sampling = ((entry.get("prediction_instability") or {})
                        .get("sampling") or {})
            if sampling.get("understates"):
                extra.append(_line(str(sampling.get("sentence") or "")))

    if not extra:
        return out
    for section in out:
        if section["key"] == "evaluation":
            section["sentences"] = list(section["sentences"]) + extra
            return out
    out.append({"key": "evaluation", "title": "Model evaluation",
                "sentences": extra, "waiting_for": None})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# `GUIDED-122` — Table 1, reusing the one that exists
# ─────────────────────────────────────────────────────────────────────────────

#: `CLINICAL_SURVEY_PACK.md` §A3's own defaults, and every one of them is a
#: correctness position rather than a preference:
#:
#: * **SMDs, not p-values.** *"p-values in a baseline table answer a question
#:   nobody asked — whether the observed groups differ from a hypothetical
#:   random draw — and they systematically mislead by declaring trivial
#:   differences significant in large cohorts and important differences
#:   non-significant in small ones. The STROBE explanation-and-elaboration
#:   document states that significance tests should be avoided in descriptive
#:   tables."* SETTLED.
#: * **Missing counts in the table**, per variable. *"Reviewers ask for it
#:   every time, and burying it in the text is the most common revision
#:   request on a cohort paper."*
#: * **Medians with IQRs for skewed variables.** *"A mean CRP of 42 mg/L with
#:   SD 90 tells the reader nothing about a typical patient."*
#:
#: And the SMD is **shown, never stamped**: §A3 records that the 0.10 rule of
#: thumb behaves poorly at small n and says to *show the value and let the
#: reader judge — never stamp PASS/FAIL*. `ml/table_one` prints the number and
#: no verdict, which is why it can be reused as it is.
TABLE1_EVIDENCE = {
    "source": "research/CLINICAL_SURVEY_PACK.md#A3 · Table One / cohort description",
    "evidence_status": "SETTLED",
    "claim": ("Build Table 1 with standardized mean differences rather than "
              "p-values; significance tests should be avoided in descriptive "
              "tables. Show the SMD value and let the reader judge — never "
              "stamp PASS or FAIL."),
}


def table_one(project: Any, run: Optional[Dict[str, Any]] = None):
    """Table 1 for this project, or `None` where there is nothing to describe.

    **Reuses `ml/table_one.generate_table1`** — `FEATURE_PARITY.md` §1 lists it
    as shared and it already implements every §A3 requirement, including the
    SMD formulas and the Yang & Dalton multinomial extension. Writing a second
    one would be the fork `ROADMAP.md` forbids, and this loop has already
    closed two of those in core.

    **All candidate predictors belong in it — §A3 calls that non-negotiable**,
    so the variable list comes from the frame the model is actually fed rather
    than from the file: `training.feature_frame` applies the target, the
    grouping key and the identifier exclusion, which is exactly the set a
    reader needs described.
    """
    from ml.table_one import Table1Config, generate_table1, \
        partition_table1_variables
    from turbotab import training as _training

    frame = getattr(project, "working_table", None)
    if frame is None or frame.empty or not project.target:
        return None
    features = _training.feature_frame(project, frame)
    if features.empty:
        return None

    continuous, categorical = partition_table1_variables(
        features, list(features.columns))
    # THE STRATIFIER IS THE OUTCOME for a prediction paper, which is §A3's
    # first named case. Only where it has few enough levels to be columns —
    # a continuous outcome has no strata, and a table with 240 columns is not
    # a table.
    grouping = None
    levels = int(frame[project.target].nunique())
    if project.task_type == "classification" and 2 <= levels <= 5:
        grouping = str(project.target)

    table, metadata = generate_table1(
        frame, Table1Config(
            grouping_var=grouping,
            continuous_vars=continuous,
            categorical_vars=categorical,
            # §A3, SETTLED. Not a preference.
            show_pvalues=False,
            show_smd=bool(grouping),
            show_missing=True,
            use_median_iqr_if_skewed=True,
        ))
    # `DRIVE-040`. THE COLUMN HEADERS NAME THE LEVELS THE USER CHOSE.
    #
    # `generate_table1` strata-labels from the column's values, and by the time
    # it runs the outcome has been encoded — so run 5 read `0 (n=770)` and
    # `1 (n=5527)` where the user had answered `False` and `True`. The values
    # are right and they are not names.
    #
    # Renamed HERE rather than inside `ml/table_one.py`: that module is shared
    # with the Streamlit door (`FEATURE_PARITY.md` §1) and has no access to a
    # recorded decision, so teaching it about one would be the fork the roadmap
    # forbids. This is the boundary that holds both.
    #
    # Silent where the record cannot say — every project sealed before `L62`,
    # and any column the repair never touched. A header then reads exactly as
    # it did before, which is a number rather than a wrong word.
    table = _name_the_outcome_columns(table, project)
    return table, {**metadata, "grouping_var": grouping,
                   "n_continuous": len(continuous),
                   "n_categorical": len(categorical), **TABLE1_EVIDENCE}


def _name_the_outcome_columns(table, project):
    """Rewrite `0 (n=46)` to `ctl (n=46)` where the record names the level."""
    import re as _re

    from turbotab import training as _training

    names = _training.outcome_level_names(project)
    if not names or not hasattr(table, "rename"):
        return table
    # The header `generate_table1` composes is `<level> (n=<count>)`, so the
    # rewrite is anchored on that shape rather than on a bare substring — a
    # loose match would rename `Overall (N=300)` the moment a level is called
    # `Overall`.
    spelled = {str(value): name for value, name in names.items()}
    renamed = {}
    for column in table.columns:
        match = _re.fullmatch(r"(.+?) \(n=(\d+)\)", str(column))
        if match and match.group(1) in spelled:
            renamed[column] = f"{spelled[match.group(1)]} (n={match.group(2)})"
    return table.rename(columns=renamed) if renamed else table


def _with_strobe_nut(sections: List[Dict[str, Any]],
                     checklist: Optional[Dict[str, Any]]
                     ) -> List[Dict[str, Any]]:
    """`GUIDED-123`. The nutrition checklist reaching the manuscript.

    Into the METHODS, in the reviewer's own order, because §09's order is part
    of the specification — *a nutrition reviewer reads your methods in a fixed
    order and checks six things* — and a checklist rendered elsewhere in a
    different sequence is a different artifact.

    Unanswered items become `[AUTHOR REQUIRED]` gaps rather than being
    dropped, which is how this app has always handled a claim only the author
    can make. An item the APP owes says so, so the two kinds of gap are not
    confused: one is waiting for the researcher and one is waiting for us.
    """
    if not checklist or not checklist.get("items"):
        return sections
    from turbotab import strobe_nut as _sn

    out = [dict(s) for s in sections]
    lines = [_line(text) for text in _sn.methods_sentences_from(checklist)]
    out.append({"key": "dietary_reporting",
                "title": "Dietary assessment reporting (STROBE-nut)",
                "sentences": lines, "waiting_for": None})
    return out

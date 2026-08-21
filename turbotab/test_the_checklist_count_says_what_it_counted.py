"""`MISC-029` — the reviewer panel counted thirteen units of scrutiny where ten
were applied.

## The claim to repair is the DENOMINATOR, not the sentence

Driven on the real page, a Guided draft read **"13 checks, 0 unmet"** and, under
it, *"Every consistency check the validator makes is met by this draft."*

**That sentence is not literally false, and attacking it as one fixes the wrong
thing.** It renders only when nothing failed (`index.html`, gated on
`!failed.length` and both promoted-figure lists being empty) — and when nothing
failed, every check the validator made did pass. What was false was the number:
of the thirteen, three could not have said anything else, because the Guided
manuscript context carries no finalized predictor list, no pre-selection
predictor count, and on an unfitted project no model at all. The sentence merely
ratified it.

## The denominator is a PAIR, and both previously published values were wrong

`turbotab/manuscript.py::_counts` produces **two** contexts carrying the same
three keys, so any sentence beginning *"the Guided context"* selects neither.
Measured here rather than quoted: `test_the_pinned_checks_are_measured_per_branch`
holds the context and `task_type` fixed and varies only the manuscript bundle,
which is the only way to ask whether an author could have moved a verdict.

| branch | pinned | live |
|---|---:|---:|
| a run is held | **3** — #2, #5, #9, all at `PASS` | 10 |
| no run / lockbox | **4** — #5, #6, #9 at `PASS` and **#2 at `FAIL`** | 9 |

**A first measurement of this reported five pinned on the run branch and was
wrong**, because its fragment corpus never stated the context's own
`analysis_total` or predictor count and never contained a real internal model
key — so three checks that *can* dissent looked frozen. The corpus below is
derived from the context for that reason. A negative claim over a filtered
population is not a check until something proves the population is non-empty for
the right reason (`AGENT_ONBOARD.md` §07 trap 5c).

## Why new row fields and not a third `status` value

Measured, both ways. `ml/manuscript_validator.py`'s `to_rows` collapses any
non-`PASS` status to the literal `"FAIL"` while `failed_checks` keys on
`== "FAIL"` and excludes a third value — so a third value serves `n_failed: 0`
and `passed: True` beside thirteen rows carrying eight `FAIL`, and renders
*"13 checks, 8 unmet"* on a clean draft. That is trap #7 committed inside the
fix for it. Two new fields turn 0 committed tests red; a third status value
turns 6 red across 3 files.

## `Split counts reconcile to analysis population` is deliberately NOT declared

It is vacuous in two opposite ways — an identity that cannot `FAIL` on the run
branch, pinned at `FAIL` on the lockbox branch — and neither is repaired by
saying so. It gets a comparand instead (`MISC-028`, `MISC-031`). If a later loop
finds itself hand-declaring it to make these numbers come out, the abstraction
here is wrong and that is the finding.

## `GUIDED-097` — the fixture rule

Two target shapes, each on both branches of `_counts`. The shapes not covered
are named in `SHAPES_NOT_COVERED`.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml.manuscript_validator import validate_manuscript_bundle
from ml.narrative_engine import _MODEL_NAMES
from turbotab import api
from turbotab import eventfixture
from turbotab import manuscript as MS
from turbotab import training as T
from turbotab.project import AnalysisProject

FIXTURES = Path(__file__).resolve().parent / "sample_data"

TARGET_SHAPES = {
    "binary_classification": ("metabolomics_untargeted.csv", "responder",
                              "classification", "logreg"),
    "continuous_regression": ("survey_instrument.csv", "age", "regression",
                              "ridge"),
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = [
    "multiclass classification — the selection-metric check knows two task "
    "types because the app does, so a multiclass draft is checked as a binary "
    "one and nothing here drives that",
    "survival / time-to-event — no task type exists in this app",
    "the CLASSIC context, which writes every key whose absence declares a "
    "check here and therefore has nothing declared. It is driven for its "
    "*absence* of declarations in "
    "`test_the_classic_context_declares_nothing_because_it_writes_the_keys`, "
    "not through pages/10 itself — a Streamlit page cannot be driven from "
    "pytest and this file does not pretend otherwise",
    "a project whose promoted figures produce `promoted_exploratory` or "
    "`promoted_without_companion` rows; neither is non-empty on these "
    "fixtures",
]

#: The three checks the Guided door cannot score, by name rather than by index,
#: so a validator that reorders its checks does not silently repoint this.
DECLARABLE = {
    "Table 1 includes all finalized predictors",
    "Model names match between development and evaluation sections",
    "Abstract feature-selection language matches actual reduction",
}
NEVER_DECLARED = "Split counts reconcile to analysis population"


def _project(name, target, task, model, *, fit):
    df = pd.read_csv(FIXTURES / name)
    df = df[df[target].notna()].copy()
    project = AnalysisProject.from_dataframe(df, name)
    project.target, project.task_type = target, task
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.20))]
    project.seal_lockbox(labels, fraction=len(labels) / len(project.df))
    eventfixture.choose_event(project, required=(task == "classification"))
    run = T.train(project, [model]).to_dict() if fit else None
    return project, run


def _validated(shape, *, fit):
    name, target, task, model = TARGET_SHAPES[shape]
    project, run = _project(name, target, task, model, fit=fit)
    return MS.validate(project.to_dict(), run=run)


# ═══════════ 1 · THE MEASUREMENT, PER BRANCH, BEFORE THE FIX IS QUOTED ══════

def _corpus(context, rendered, latex):
    """Bundle fragments **derived from the context**, so a check that can
    agree is given something to agree with."""
    counts = context.get("population_counts") or {}
    total = counts.get("analysis_total")
    feats = context.get("feature_counts") or {}
    selected = feats.get("selected")
    models = context.get("included_models") or []
    names = [_MODEL_NAMES.get(k, k) for k in models] or ["Ridge",
                                                         "Logistic Regression"]
    internal = sorted(k for k in _MODEL_NAMES if "_" in k)
    return [
        "", rendered["methods"], rendered["report"], latex,
        f"## Abstract (Draft)\nA dataset of {total} observations were "
        f"available for analysis. We retained {selected} predictors for final "
        f"modeling.\n",
        f"## Methods\n### Study Design\nA dataset of {total} observations.\n",
        f"### Predictor Variables\nWe retained {selected} predictors for "
        f"final modeling.\n",
        "## Abstract (Draft)\nA dataset of 1 observations. We retained 999 "
        "predictors for final modeling.\n",
        "## Methods\n### Study Design\nA dataset of 999 observations.\n",
        "### Predictor Variables\nWe retained 0 predictors for final "
        "modeling.\n",
        "### Model Development\n" + ", ".join(names) + "\n",
        "### Model Evaluation\n" + ", ".join(names) + "\n",
        "### Model Development\nnothing here\n",
        "### Model Evaluation\nnothing here\n",
        "### Model Development\naccuracy and f1 and auc\n",
        "### Model Development\nrmse and mae and r2\n",
        "\\subsection{Model Development}\nMAE and R2 and auc and f1\n",
        "### Model Development\nRidge was selected as the primary model.\n",
        "no manuscript-primary model was explicitly selected",
        "## Abstract (Draft)\nfeature selection reduced the predictor set.\n",
        "[PLACEHOLDER] **bold** [NOTE: x]\n## heading\n",
        " ".join(internal[:8]),
        " ".join(k.upper() for k in internal[:8]),
        "no action needed, favorable to analysis, workflow-derived abstract, "
        "[applicable to",
        "Two periods.. and Table X and a dash-.",
    ]


def _table1_variants(context, columns):
    counts = context.get("population_counts") or {}
    total = counts.get("analysis_total") or 1
    names = list(context.get("feature_names_for_manuscript") or [])
    cols = list(columns)[:6] or ["a"]
    return [
        None,
        pd.DataFrame({f"Overall (N={total})": ["1"] * len(cols)}, index=cols),
        pd.DataFrame({"Overall (N=99999)": ["1"] * len(cols)}, index=cols),
        pd.DataFrame({"Overall (N=0)": []}, index=[]),
        pd.DataFrame({f"Overall (N={total})": ["1"] * 3},
                     index=["zzz_1", "zzz_2", "zzz_3"]),
        pd.DataFrame({f"Overall (N={total})": ["1"] * max(1, len(names))},
                     index=(names or ["zzz"])),
    ]


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
@pytest.mark.parametrize("fit", [True, False], ids=["run", "no_run"])
def test_the_declared_checks_are_exactly_the_ones_no_manuscript_can_move(
        shape, fit, capsys):
    """**The load-bearing measurement, and the declaration is checked against
    it rather than against a list.**

    Holds the context and `task_type` fixed and varies only the bundle. A check
    whose verdict set has one element over 600 randomized bundles could not have
    been moved by anything an author writes; the declaration must name exactly
    those, minus the one this loop deliberately repairs instead.
    """
    name, target, task, model = TARGET_SHAPES[shape]
    project, run = _project(name, target, task, model, fit=fit)
    doc = MS.structure(project.to_dict(), run=run)
    context = doc["context"]
    rendered = MS.to_markdown(doc)
    try:
        latex = MS.to_latex(doc)
    except Exception:                                       # pragma: no cover
        latex = ""
    corpus = _corpus(context, rendered, latex)
    tables = _table1_variants(context, project.df.columns)

    rng = random.Random(20260821)
    verdicts: dict = {}
    for _ in range(600):
        def blob():
            return "\n".join(rng.sample(corpus,
                                        rng.randint(0, min(6, len(corpus)))))
        report = validate_manuscript_bundle(
            context, blob(), blob(), blob(), task,
            table1_df=rng.choice(tables))
        for check in report.checks:
            verdicts.setdefault(check.name, set()).add(check.status)

    # THE CONTROL, BEFORE THE PINNED SET IS QUOTED. A corpus that moved nothing
    # would report every check pinned, in the same words as a broken validator.
    live = {n for n, v in verdicts.items() if len(v) > 1}
    assert len(live) >= 7, (
        f"only {sorted(live)} moved over 600 bundles; the corpus is not "
        f"exercising the validator and its silence about the rest means "
        f"nothing")

    pinned = {n for n, v in verdicts.items() if len(v) == 1}
    report = validate_manuscript_bundle(
        context, rendered["methods"], rendered["report"], latex, task)
    declared = {c.name for c in report.declared_checks}

    assert declared == pinned - {NEVER_DECLARED}, (
        f"declared={sorted(declared)} but the checks no manuscript can move "
        f"are {sorted(pinned)}. Every pinned check except "
        f"{NEVER_DECLARED!r} must declare itself, and nothing that can "
        f"dissent may.")
    assert NEVER_DECLARED in pinned, (
        f"{NEVER_DECLARED!r} is no longer pinned on the {'run' if fit else 'no-run'} "
        f"branch. If MISC-028/MISC-031 gave it a real comparand, remove it "
        f"from NEVER_DECLARED here — do not add it to the declared set.")

    with capsys.disabled():
        print(f"\n  {shape} {'run' if fit else 'no-run'}: "
              f"{len(pinned)} pinned of {len(verdicts)}, "
              f"{len(declared)} declared, {len(live)} live")


# ═══════════ 2 · THE COUNT THE PAYLOAD SERVES ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
@pytest.mark.parametrize("fit", [True, False], ids=["run", "no_run"])
def test_the_served_counts_separate_the_roster_from_the_scrutiny(shape, fit):
    """`n_checks` keeps its meaning; `n_scored` is the new one.

    Redefining a served field in place would be this finding one layer down —
    a number whose name stops matching what it counts.
    """
    out = _validated(shape, fit=fit)
    counts = out["checklist_counts"]
    rows = out["rows"]

    assert counts["n_checks"] == len(rows), counts
    assert counts["n_scored"] == sum(1 for r in rows if r["scored"]), counts
    assert counts["n_declared"] == sum(1 for r in rows if not r["scored"])
    assert counts["n_scored"] + counts["n_declared"] == counts["n_checks"]
    assert counts["n_scored"] < counts["n_checks"], (
        "nothing is declared on a Guided draft, so this file is asserting "
        "about a state the door does not reach")
    assert {d["Check"] for d in counts["declared"]} <= DECLARABLE, counts
    for entry in counts["declared"]:
        assert entry["because"].strip(), entry
        assert "decided before the draft was read" in entry["because"], entry


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
@pytest.mark.parametrize("fit", [True, False], ids=["run", "no_run"])
def test_a_declared_check_never_reports_a_failure(shape, fit):
    """A contradiction, guarded at zero rather than assumed away.

    A check that says it could not be scored and also reports `FAIL` would have
    the app asserting a defect it says it could not have detected — the
    governing rule failing in the direction the panel exists to prevent.
    """
    out = _validated(shape, fit=fit)
    assert out["checklist_counts"]["n_declared_that_failed"] == 0, [
        (r["Check"], r["Detail"]) for r in out["rows"]
        if not r["scored"] and r["Status"] == "FAIL"]


def test_the_gate_the_classic_download_reads_did_not_move():
    """`AGENT_ONBOARD.md` §08 check 2, asserted rather than promised.

    `passed` gates the Classic export. `scored` is additive reporting and
    `failed_checks` still keys on `status == "FAIL"` alone, so the set that
    blocks a download is the set that blocked it before.
    """
    import inspect

    from ml import manuscript_validator as MV

    source = inspect.getsource(MV.ManuscriptValidationReport.failed_checks
                               .fget)
    assert 'check.status == "FAIL"' in source, source
    assert "scored" not in source, (
        "failed_checks now reads `scored`, so a declared check can change what "
        "blocks a Classic download. That is a threshold moving in the loop "
        "that pressured it: " + source)


def test_the_classic_table_does_not_silently_grow_two_columns():
    """`to_rows` grew two keys and `pages/10` renders it with `pd.DataFrame`.

    Without an explicit column list they would land in the Classic validation
    table with no code change and nothing covering them.
    """
    page = (Path(__file__).resolve().parents[1] / "pages"
            / "10_Report_Export.py").read_text(encoding="utf-8")
    assert re.search(
        r"pd\.DataFrame\(validation_report\.to_rows\(\)\)\[\s*\n?\s*"
        r'\["Status", "Check", "Location", "Detail"\]\]', page), (
        "pages/10_Report_Export.py no longer selects the validation table's "
        "columns explicitly, so `scored` and `declared_because` are being "
        "rendered to a Classic user with nothing asserting anything about "
        "them")


def test_the_classic_context_declares_nothing_because_it_writes_the_keys():
    """**Classic is the healthier door and the mechanism must say so.**

    Not a fixture standing in for Classic: the four keys are read out of
    `_build_manuscript_context`'s own return statement, and a context carrying
    them is put through the same validator. If Classic ever stopped writing one,
    this goes red rather than the count quietly inflating one door over.
    """
    page = (Path(__file__).resolve().parents[1] / "pages"
            / "10_Report_Export.py").read_text(encoding="utf-8")
    for key in ("'feature_names_for_manuscript'", "'feature_counts'",
                "'manuscript_primary_model'", "'best_metric_name'"):
        assert f"{key}:" in page, (
            f"pages/10_Report_Export.py no longer writes {key} into the "
            f"manuscript context, so a Classic check that used to be live is "
            f"now declared and nothing here noticed")

    classic_like = {
        "population_counts": {"analysis_total": 100, "train_n": 60,
                              "val_n": 20, "test_n": 20},
        "feature_counts": {"original": 12, "candidate": 12, "selected": 8},
        "feature_names_for_manuscript": ["age", "bmi"],
        "manuscript_primary_model": "rf",
        "best_metric_name": "auc",
        "included_models": ["rf", "logreg"],
    }
    report = validate_manuscript_bundle(classic_like, "", "", "",
                                        "classification")
    assert not report.declared_checks, [c.name for c in
                                        report.declared_checks]
    assert len(report.scored_checks) == len(report.checks) == 13


# ═══════ 3 · IT REACHES A PERSON, OR IT IS L64-B ONE LOOP LATER ═══════

#: The page is driven on the SMALL fixtures. `metabolomics_untargeted.csv` has
#: 397 predictors and building its route table takes over two minutes; the
#: claim here is about what `validationHTML` renders, and it is the same
#: renderer whatever the column count. Measured: `clinical_risk.csv` end to end
#: is 1.2s.
PAGE_SHAPES = {
    "clinical_classification": ("clinical_risk.csv", "clinical",
                                "readmit_30d"),
    "survey_regression": ("survey_instrument.csv", "survey", "age"),
}

_ROUTE_TAIL = ("interview?step=data", "interview?step=explore",
               "interview?step=features", "capabilities", "features",
               "recipes", "preprocess", "figures", "draft", "manuscript",
               "checklist", "models", "training", "instability", "explain",
               "sensitivity", "evidence/plausibility", "evidence/missingness")

_RUN_NOTE = re.compile(r'class="run-note">([^<]*)</span>')
_DEC_HEAD = re.compile(r'<p class="dec-head">([^<]*)</p>')


def _drive_the_report_panel(fixture, lens, target):
    """The page's own controller, against real API responses."""
    from turbotab import pageharness as H

    client = TestClient(api.app)
    with (FIXTURES / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": [lens]}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    table = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in _ROUTE_TAIL:
        response = client.get(f"/project/{pid}/{path}")
        table[f"/project/{pid}/{path}"] = (response.json()
                                           if response.status_code == 200
                                           else {})
    box = H.run("__emit((document.getElementById('reportBox')||{}).innerHTML "
                "|| '');", routes=table, search=f"?project={pid}") or ""
    return box, table[f"/project/{pid}/manuscript"]


@pytest.mark.parametrize("shape", sorted(PAGE_SHAPES))
def test_the_header_a_person_sees_counts_only_what_was_scored(shape, capsys):
    """**The proving number, and it is not a unit test.**

    `L64-B` shipped a scored/declared badge one registry over, its unit tests
    passed, and a real bundle served **28 checklist rows with 0 declared** —
    the mechanism proven in tests and unproven in service (`GUIDED-238`). So
    this asserts the string `validationHTML` puts in the DOM, through the
    page's own controller against real API responses, rather than a
    transcription of its arithmetic into Python.
    """
    from turbotab import pageharness as H

    if not H.available():                                  # pragma: no cover
        pytest.skip("the page harness needs node, which is absent here")

    fixture, lens, target = PAGE_SHAPES[shape]
    box, manuscript = _drive_the_report_panel(fixture, lens, target)

    assert len(box) > 2000, (
        f"the report panel rendered {len(box)} characters; nothing below is "
        f"about the page a person sees")
    note = _RUN_NOTE.search(box)
    assert note, f"no validation header rendered at all:\n{box[:1200]}"
    header = note.group(1)

    counts = manuscript["checklist_counts"]
    assert counts["n_declared"] > 0, (
        "nothing is declared on this project, so the header below cannot "
        "distinguish the fix from its absence")
    assert header.startswith(f"{counts['n_scored']} checks, "), (
        f"the header reads {header!r}; it must count the {counts['n_scored']} "
        f"SCORED checks, not the {counts['n_checks']} the validator makes. "
        f"That difference is MISC-029 — thirteen units of scrutiny asserted "
        f"where {counts['n_scored']} were applied.")
    assert str(counts["n_checks"]) not in header.split(",")[0], header
    assert f"{counts['n_declared']} declared" in header, header

    # AND THE DECLARED ONES ARE ON SCREEN, not merely subtracted. A header that
    # says a number was withheld and never says which has moved the falsehood
    # rather than removed it — and a served field nothing renders is trap #6.
    rendered = _DEC_HEAD.findall(box)
    assert len(rendered) == counts["n_declared"], (
        f"header says {counts['n_declared']} declared and the panel renders "
        f"{len(rendered)}: {rendered}")
    assert set(rendered) == {d["Check"] for d in counts["declared"]}, rendered
    for entry in counts["declared"]:
        assert entry["because"][:60] in box, (
            f"{entry['Check']!r} is counted as declared and its reason never "
            f"reaches the page, so a reader cannot find out what was not "
            f"checked")

    with capsys.disabled():
        print(f"\n  {shape}: rendered header {header!r} · "
              f"{len(rendered)} declared block(s) on screen · "
              f"{len(box)} chars")


def test_the_panel_stops_computing_its_own_denominator():
    """`GUIDED-179`'s missing consumer, asserted as a consumer.

    The header used to be `MANU.rows.length` computed client-side, which is
    what let the served count and the rendered one mean different things. This
    is a claim about the file and is asserted over the file — the behavior is
    driven above.
    """
    page = (Path(__file__).resolve().parent / "web"
            / "index.html").read_text(encoding="utf-8")
    panel = page[page.index("function validationHTML()"):]
    panel = panel[:panel.index("\n  function ", 10)]
    assert "checklist_counts" in panel, (
        "validationHTML no longer reads the served counts, so it is computing "
        "a denominator the server also computes and the two can drift")
    assert 'num((MANU.rows || []).length, 0) +\n            " checks, "' \
        not in panel, "the client-side denominator is back"


def test_the_shapes_not_covered_are_named():
    """A fixture list that does not say what it omits reads as coverage."""
    assert len(SHAPES_NOT_COVERED) >= 4
    assert all(len(s) > 40 for s in SHAPES_NOT_COVERED)

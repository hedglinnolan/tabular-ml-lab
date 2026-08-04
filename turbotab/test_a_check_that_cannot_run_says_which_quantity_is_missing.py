"""`GUIDED-179` — the reviewer checklist rendered a Python repr, and miscounted.

## The fourth branch

`AGENT_ONBOARD.md` §00 gives the app three branches: it may **assert truly**, it
may be **silent**, and it may **refuse**. Driven on a project with no run, the
*"What a reviewer will notice"* panel rendered:

> Expected analysis N=None, abstract N=None, study design N=None.

That is a **fourth branch and nobody authorized it.** It is not silence — a
sentence is on the screen. It is not an assertion — `None` claims nothing. It is
not a refusal — it names no missing quantity and gives no reason. It is a Python
repr shown to a researcher, whose only available reading is *the app is broken*,
and it appeared in **six** of the thirteen rendered checklist strings on both
fixtures below (four FAIL details, two PASS details).

The vocabulary for the honest branch was already here and is not reinvented:
`figures.NOT_ESTIMABLE` (`turbotab/figures.py:430`) is this project's token for a
number **not shown because there is not one**, and `figure_specs.py:171-178`
pairs it with a `why` naming the cause. `turbotab/manuscript.py`'s
`_rows_that_say_what_is_missing` reproduces that pairing on the checklist, at the
boundary that serves it — `ml/manuscript_validator.py` keeps its own vocabulary
for its own callers.

## What was found about the count disagreement — it is (i), TWO POPULATIONS

The header read *"13 checks, 4 unmet"* above **six** items, and the instruction
was to find out which of the two was wrong before reconciling them. **Neither
is.** They count different populations and both are exact:

- the header is `rows.length + " checks, " + failed.length + " unmet"`
  (`turbotab/web/index.html:2770-2773`) and counts **validator checks** — 13
  rows, 4 with `Status == "FAIL"`;
- the body then renders those 4 **plus** `unsourced_sections`,
  `promoted_exploratory` and `promoted_without_companion` (`:2782-2812`). On
  both fixtures that is 4 + 2 = 6.

The two extra items are the *Model Development* and *Model Evaluation* sections
`GUIDED-116` records as unsourceable. They are **not validator checks** — the
validator has no check for a section that does not exist — and both
`turbotab/manuscript.py:648-651` and `index.html:2782-2784` hold that separation
deliberately: *"rendering them alike would let a structural gap read as a
formatting slip."*

So making the header count the list would have destroyed a real distinction, and
making the list show only checks would have hidden two structural gaps. The
remedy is the third one: **the payload SAYS which.** `checklist_counts.because`
names both populations, both numbers, and what the difference consists of, and
this file asserts that sentence is present and arithmetically true.

**Stated limit.** `web/index.html` is outside this part's edit boundary, so the
sentence is served and the panel does not yet render it. The header a user sees
is unchanged. That is the remaining half of `GUIDED-179` and it is one
`validationHTML` edit.

## `GUIDED-097` — the fixture rule

Two target shapes are driven through HTTP, because the claim is about what
reaches a researcher and a claim about rendering asserted from inside the module
is the same defect one layer out. The shapes **not** covered are named in
`SHAPES_NOT_COVERED` below.

## `TEST-045`

Every parametrize id here is ASCII.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                            # noqa: E402
from turbotab import figures as _figures                            # noqa: E402
from turbotab import manuscript as MS                               # noqa: E402
from turbotab import training as T                                  # noqa: E402
from turbotab.project import AnalysisProject                        # noqa: E402

FIXTURES = Path(__file__).parent / "sample_data"

#: `GUIDED-097`. Two shapes, driven end to end. The binary-classification case
#: is the one the finding was filed from; the regression case exists to prove
#: the repr is not a property of one fixture — Table 1 is stratified by the
#: outcome in the first and unstratified in the second, so `Table 1 overall N`
#: is reached by two different routes.
TARGET_SHAPES = {
    "binary_classification": ("clinical_labs.csv", ["clinical"], "readmitted"),
    "continuous_regression": ("survey_instrument.csv", ["clinical"], "age"),
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = [
    "multiclass classification (`multiclass_stage.csv`) — the validator's "
    "selection-metric check knows two task types because the app does, so a "
    "multiclass checklist is checked as a binary one and nothing here drives "
    "that path",
    "survival / time-to-event — no task type exists in this app at all",
    "a project whose promoted figures produce `promoted_exploratory` or "
    "`promoted_without_companion` rows — both are counted by "
    "`_checklist_counts` and neither is non-empty on these two fixtures, so "
    "the branch is exercised for `unsourced_sections` only",
]

#: The repr forms that must never reach a researcher. `None` is the one
#: `GUIDED-179` was filed for; `nan` and `NaT` are the same class arriving from
#: pandas, and are asserted against so the next one is caught by this file
#: rather than by a user.
_REPR_LEAK = re.compile(r"(?<![A-Za-z0-9_])(None|nan|NaN|NaT)(?![A-Za-z0-9_])")

#: A brace in a rendered sentence means a dict or an unformatted template was
#: str()'d into it.
_BRACE_LEAK = re.compile(r"[{}]")

#: WHAT THE PANEL RENDERS. Named as the four payload keys `validationHTML`
#: reads (`index.html:2782-2812`) plus the count sentence, and then walked
#: **wholesale** — every string leaf under them, at any depth — rather than by
#: naming fields, so a new field added to a row is swept without editing this.
#:
#: `rendered.latex`, `rendered.methods` and `document` are deliberately outside
#: this set and it is not a loophole: they are the manuscript itself, a LaTeX
#: document legitimately contains thousands of braces, and the panel does not
#: render them. The repr sweep below runs over the WHOLE payload anyway, because
#: `=None` is illegitimate everywhere; only the brace rule is scoped.
CHECKLIST_SURFACE = ("rows", "unsourced_sections", "promoted_exploratory",
                     "promoted_without_companion", "checklist_counts",
                     "because")


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _upload(client, name: str) -> str:
    with open(FIXTURES / name, "rb") as fh:
        response = client.post("/project", files={
            "file": (name, fh, "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _decide(client, pid: str, kind: str, **payload):
    response = client.post(f"/project/{pid}/decision",
                           json={"kind": kind, "payload": payload})
    assert response.status_code == 200, (kind, response.text)
    return response


def _manuscript(client, shape: str):
    """A lens, a target, no run — the state `GUIDED-179` was filed from."""
    name, lens, target = TARGET_SHAPES[shape]
    pid = _upload(client, name)
    _decide(client, pid, "set_lens", lens=lens)
    _decide(client, pid, "set_target", column=target)
    response = client.get(f"/project/{pid}/manuscript")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is True, body.get("because")
    return body


def _strings(node, path="", out=None):
    """Every string leaf under `node`, with the path that reached it."""
    out = [] if out is None else out
    if isinstance(node, str):
        out.append((path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            _strings(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _strings(value, f"{path}[{i}]", out)
    return out


def _surface_strings(payload):
    out = []
    for key in CHECKLIST_SURFACE:
        if key in payload:
            _strings(payload[key], key, out)
    return out


def _panel_arithmetic(payload):
    """The header and the list, recomputed exactly as `validationHTML` does.

    Deliberately re-derived from the raw payload rather than read off
    `checklist_counts`: a count that agrees with itself is the self-confirming
    check `_counts`' own docstring warns about.
    """
    rows = payload.get("rows") or []
    failed = [r for r in rows if r.get("Status") == "FAIL"]
    beyond = (len(payload.get("unsourced_sections") or [])
              + len(payload.get("promoted_exploratory") or [])
              + len(payload.get("promoted_without_companion") or []))
    return len(rows), len(failed), len(failed) + beyond, beyond


# ═════════════ 1 · NO REPR REACHES A RENDERED CHECKLIST STRING ═════════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_no_python_repr_reaches_any_rendered_checklist_string(client, shape):
    """`GUIDED-179`, the fourth branch closed.

    By regex over the whole payload, not by checking one field: the defect was
    reported for `Detail`, and it was also in two PASS details and would land in
    any new string composed the same way.
    """
    body = _manuscript(client, shape)

    whole = json.dumps(body, default=str)
    assert not _REPR_LEAK.search(whole), [
        "a Python repr is somewhere in the manuscript payload",
        sorted({m.group(0) for m in _REPR_LEAK.finditer(whole)}),
        [f"{p}: {s}" for p, s in _strings(body)
         if _REPR_LEAK.search(s)][:6]]

    for path, text in _surface_strings(body):
        assert not _REPR_LEAK.search(text), (path, text)
        assert not _BRACE_LEAK.search(text), (path, text)


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_a_check_that_cannot_run_names_the_quantity_and_the_reason(
        client, shape):
    """Absence is not enough: the sentence has to say WHICH and WHY.

    A scrub that replaced `None` with an empty string would pass the test above
    and be the *silent* branch pretending to be the *refuse* branch, which is the
    same substitution `GUIDED-129` refused for the annotation box.
    """
    body = _manuscript(client, shape)
    rows = body["rows"]

    # THE GUARD. Without it this whole file goes green on a payload that simply
    # never had a missing quantity, which is the state the fix is FOR.
    with_absence = [r for r in rows if r.get("missing_quantities")]
    assert len(with_absence) >= 4, [
        "these fixtures are supposed to reach the checklist with no run and "
        "therefore no analysis population; if nothing is missing the assertions "
        "below prove nothing",
        [(r["Check"], r["Detail"]) for r in rows]]

    for row in with_absence:
        detail = row["Detail"]
        # WHICH — every named quantity appears in the sentence a user reads.
        assert row["missing_quantities"], row
        for name in row["missing_quantities"]:
            # Case-insensitive because a PASS row opens its sentence with the
            # quantity, so "the metric..." is rendered "The metric...". The
            # claim is that the name is IN the sentence, not that it is
            # mid-sentence.
            assert name.lower() in detail.lower(), (name, detail)
        # WHY — a reason, in the register the annotation box already uses.
        assert _figures.NOT_ESTIMABLE in detail, detail
        assert "because" in detail, detail
        assert ("A number is not shown because there is not one, rather than "
                "because it failed to render.") in detail, detail

    cannot_run = [r for r in rows if r.get("cannot_run")]
    assert cannot_run, rows
    for row in cannot_run:
        assert row["Status"] == "FAIL", row
        assert "This check has nothing to compare" in row["Detail"], row
    # A PASS row that mentions an absent quantity DID run and DID pass. Claiming
    # otherwise would be the fourth branch again with better grammar.
    for row in rows:
        if row["Status"] == "PASS":
            assert "This check has nothing to compare" not in row["Detail"], row


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_table_one_asymmetry_is_named_rather_than_shown_as_a_gap(
        client, shape):
    """THE MECHANISM, said to the author.

    `validate` takes `table1` as its own parameter, independent of `run`
    (`turbotab/manuscript.py:594-602` and the `table1_df=table1` call at
    `:646`), and `api.get_manuscript` builds it from `project.working_table`.
    So Table 1 knows its N on a project nobody has trained while `_counts`
    returns `population_counts: {}` — one side of the comparison exists and the
    other does not. Rendering that as `N=None` vs `N=288` told a reviewer the
    app had lost a number it never had.
    """
    body = _manuscript(client, shape)
    row = next(r for r in body["rows"]
               if r["Check"] == "Table 1 population matches the analysis cohort")
    assert row["Status"] == "FAIL", row
    assert "one side of this comparison exists and the other does not" \
        in row["Detail"], row["Detail"]
    # The number Table 1 DOES have is kept, not scrubbed with the one it lacks.
    overall = [c for c in body["table1_columns"] if c.startswith("Overall")]
    assert overall, body["table1_columns"]
    n = int(re.search(r"N=([\d,]+)", overall[0]).group(1).replace(",", ""))
    assert f"{n:,}" in row["Detail"] or str(n) in row["Detail"], row["Detail"]


# ═══════════ 2 · THE COUNT AND THE LIST — TWO POPULATIONS, NAMED ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_header_count_and_the_displayed_list_are_reconciled_or_named(
        client, shape):
    """The finding is (i): two different populations, so the payload says which.

    Not reconciled by making the header read the list — that would hide a real
    distinction `manuscript.py:648-651` holds on purpose. The assertion is the
    branch the instruction allows for a legitimate difference: the sentence
    naming it is present, and it is arithmetically true.
    """
    body = _manuscript(client, shape)
    n_checks, n_unmet, n_listed, n_beyond = _panel_arithmetic(body)
    counts = body["checklist_counts"]

    # The served counts match the panel's own arithmetic, re-derived.
    assert counts["n_checks"] == n_checks, (counts, n_checks)
    assert counts["n_unmet_checks"] == n_unmet, (counts, n_unmet)
    assert counts["n_items_listed"] == n_listed, (counts, n_listed)
    assert counts["n_listed_that_are_not_checks"] == n_beyond, counts
    assert counts["header_and_list_count_the_same_population"] is (n_beyond == 0)

    because = counts["because"]
    if n_unmet == n_listed:
        assert counts["header_and_list_count_the_same_population"] is True
        return

    # THE DIFFERENCE IS NAMED. Both numbers, and what the extra items are.
    assert counts["header_and_list_count_the_same_population"] is False
    assert str(n_checks) in because, because
    assert str(n_unmet) in because, because
    assert str(n_beyond) in because, because
    assert "not validator checks" in because, because
    assert "the draft cannot source" in because, because
    # And the reason the two are kept apart, rather than a bare restatement.
    assert "formatting slip" in because, because


def test_the_two_extra_items_really_are_not_checks(client):
    """The premise the naming sentence rests on, asserted rather than assumed.

    If an unsourced section were ALSO a validator row, the header would be
    undercounting and the fix would be laundering a real bug.
    """
    body = _manuscript(client, "binary_classification")
    headings = {u["heading"] for u in body["unsourced_sections"]}
    assert headings == {"Model Development", "Model Evaluation"}, headings
    checks = " | ".join(r["Check"] for r in body["rows"])
    for heading in headings:
        assert f"No {heading} section" not in checks, (heading, checks)


# ═══════ 3 · THE FIX DOES NOT DAMAGE A PAYLOAD THAT HAS ITS NUMBERS ═══════

def test_a_fitted_project_still_passes_and_the_counts_agree():
    """A third shape, fitted, so the *agreeing* branch is not dead code.

    `MS.validate` directly rather than through HTTP: the training route is a
    background job and this asserts about the rewrite, not about the queue.
    """
    df = pd.read_csv(FIXTURES / "metabolomics_untargeted.csv")
    df = df[df["responder"].notna()].copy()
    project = AnalysisProject.from_dataframe(df, "metabolomics_untargeted.csv")
    project.target, project.task_type = "responder", "classification"
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.20))]
    project.seal_lockbox(labels, fraction=len(labels) / len(project.df))
    run = T.train(project, ["logreg"]).to_dict()

    out = MS.validate(project.to_dict(), run=run)
    assert out["available"] is True, out.get("because")
    assert out["n_failed"] == 0, [(r["Check"], r["Detail"])
                                  for r in out["rows"] if r["Status"] == "FAIL"]

    counts = out["checklist_counts"]
    assert counts["header_and_list_count_the_same_population"] is True, counts
    assert counts["n_items_listed"] == counts["n_unmet_checks"] == 0, counts
    assert "the same" in counts["because"], counts["because"]

    for path, text in _surface_strings(out):
        assert not _REPR_LEAK.search(text), (path, text)


def test_the_shapes_not_covered_are_named():
    """`GUIDED-097`. A fixture list that does not say what it omits reads as
    coverage, which is the silence this whole file is about."""
    assert len(TARGET_SHAPES) >= 2
    assert len(SHAPES_NOT_COVERED) >= 3
    assert all(name.isascii() for name in TARGET_SHAPES)     # `TEST-045`

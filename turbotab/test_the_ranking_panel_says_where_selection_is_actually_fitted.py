"""`AUDIT-027` · the 'Rank them for me' panel and the record say the same scope.

## The false sentence

`selection.evidence` returned, whenever any row was withheld:

    Ranked on training rows only, and not applied. **What is actually selected
    is refitted inside each training fold**, so this ordering is indicative
    rather than the answer.

Nothing in this app refits selection inside a fold. `api` records
`scope=train_rows` for this door — with a comment saying so — `training.train`
does exactly one `pipe.fit(X_train, y_train)` per model, and `pipeline_plan`
records `scope_fitted=train_rows` and raises a `Divergence` when a spec asked
for the stronger one. The panel was the last surface still asserting it, and it
bypassed `declare`'s own rewrite three functions above it — the rewrite that
exists precisely so a door that fits once can SAY `train_rows` instead of
implying the stronger claim (`GUIDED-104`).

`CLINICAL_SURVEY_PACK.md` §A5.5 is why it matters rather than being pedantry:
*internal validation must resample the entire modeling pipeline — imputation,
transformation, selection, tuning.* Telling a researcher that selection is
refitted per fold is telling them their selection sits inside a resampling
loop. It does not. What it sits inside is the single train/test split the same
paragraph calls the weakest option.

## The correction

Same subject, weaker claim, true:

    …What is actually selected is fitted **once over the training rows
    (held-out rows excluded)** — this door fits each model one time, so there
    is a single fold — so this ordering is indicative rather than the answer.

The phrase now comes from `selection._SCOPE_PHRASE[FITTED_SCOPE]`, the same
table `declare` rewrites with, so the panel and the record cannot drift again;
and `pipeline_plan` reads `FITTED_SCOPE` for `scope_fitted`, so there is one
name for *what this door fits* rather than a literal in each module.

## Driven, not described

Every assertion below drives the real route the row names — Guided door →
Features step → 'Rank them for me' → `GET /project/{id}/selection/evidence` —
through `TestClient(api.app)`. Nothing here reads source text.

## Fixture shapes — `GUIDED-097`

Two shapes of different target type: a continuous outcome and a three-level
string outcome. `SHAPES_NOT_COVERED` names what is not driven and why.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                        # noqa: E402
from turbotab import selection as _sel                          # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: The sentence the row was filed against, quoted so a revert names itself.
THE_FALSE_CLAIM = "refitted inside each training fold"

#: `GUIDED-097`. Two target shapes, deliberately of different type: the
#: evidence note is composed before any scoring branch is chosen, so a shape
#: whose preview refuses (`kind == "none"`) and one whose preview computes must
#: both be exercised — a continuous outcome scores with `f_regression`, a
#: three-level string outcome with `mutual_info_classif`.
TARGET_SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c", ["age", "bmi", "sbp"]),
    "multiclass_string": ("multiclass_stage.csv", "disease_stage",
                          ["age", "bmi", "hba1c"]),
}

#: NOT COVERED, named here rather than discovered later.
SHAPES_NOT_COVERED = {
    "binary numeric (0/1)": (
        "`leaky_sepsis.csv` has a 0/1 target. The note is composed from "
        "`n_rows_withheld` and `FITTED_SCOPE` and reads neither the target nor "
        "the task type, so the behavior is expected to be identical — which is "
        "exactly why `GUIDED-097` says to say it is undriven rather than to "
        "assume it."),
    "a project that recorded scope=train_folds explicitly": (
        "The API's `set_selection` defaults to `TRAIN_ROWS` and no surface "
        "offers the other value, so this is unreachable through the door. If "
        "it ever becomes reachable, the note stays true — it states what is "
        "FITTED — and `pipeline_plan` is the surface that must then report the "
        "divergence from what was RECORDED. Undriven here."),
    "no seal drawn": (
        "Covered by the existing "
        "`test_selection_evidence_without_a_mask_says_it_saw_everything`, "
        "which pins the other branch of this same note."),
}


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed(client, shape):
    """A project through the Guided door, sealed, with a selection recorded."""
    fixture, target, candidates = TARGET_SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])
        return r

    decide("set_target", column=target)
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    spec = decide("set_selection", method="mutual_info",
                  candidates=candidates,
                  n_features=2).json()["selection_spec"]
    return pid, spec


def _evidence(client, pid):
    r = client.get(f"/project/{pid}/selection/evidence")
    assert r.status_code == 200, r.text[:250]
    return r.json()


# ═══════════ 1 · the row's own failing assertion ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_panel_does_not_tell_the_user_selection_is_refitted_per_fold(
        client, shape):
    """`AUDIT-027`'s `failing_assertion`, verbatim, on the route it names.

    **The positive control comes first** (`GUIDED-045`): an absence assertion
    over an empty string is true and means nothing. So the note must exist,
    must be about where selection lands, and must have been produced by a
    project that really withheld rows — otherwise the branch carrying the
    false sentence was never entered.
    """
    pid, _ = _sealed(client, shape)
    body = _evidence(client, pid)

    # POSITIVE CONTROL — the sweep has something to sweep.
    assert body["n_rows_withheld"] > 0, (
        f"{shape}: nothing was withheld, so the note under test is the "
        f"no-seal branch and this assertion would pass vacuously")
    assert body["note"].strip(), f"{shape}: the panel returned no note at all"

    # THE ASSERTION THE ROW FILED.
    assert THE_FALSE_CLAIM not in body["note"], (
        f"{shape}: the ranking panel tells the user that what is actually "
        f"selected is {THE_FALSE_CLAIM} — this door selects once over the "
        f"training rows. training.train fits each model one time on the "
        f"training partition and pipeline_plan records scope_fitted="
        f"{_sel.FITTED_SCOPE!r}. AUDIT-027. Note: {body['note']!r}")

    # AND NOT BY DELETION. `AUDIT-028`'s model: the sentence says LESS and
    # stays TRUE, it does not go quiet about its subject. A note that dropped
    # the clause entirely would satisfy the assertion above and fail the row.
    assert "actually selected is fitted" in body["note"], (
        f"{shape}: the note no longer says where the selection itself lands, "
        f"which is a deletion rather than a correction. Note: "
        f"{body['note']!r}")


# ═══════════ 2 · the parity form: the panel and the record agree ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_panel_describes_the_same_scope_the_record_stores(client, shape):
    """The row's second form, as an equality rather than a substring hunt.

    `declare` returns `scope` and `fit_on`; the evidence payload now carries
    the same two facts under `selection_scope` / `selection_fit_on` (trap 7 —
    the machine-readable half must not be lossier or falser than the prose).
    Before the fix the record said `train_rows` and the panel said fold-
    refitted, and nothing on the wire let a reader see the disagreement.
    """
    pid, spec = _sealed(client, shape)
    body = _evidence(client, pid)

    # POSITIVE CONTROL — the record really recorded a scope.
    assert spec["scope"] == _sel.TRAIN_ROWS, (
        f"{shape}: the Guided door recorded scope={spec['scope']!r}; this "
        f"test's subject is the door that records train_rows")

    assert body["selection_scope"] == spec["scope"], (
        f"{shape}: the panel says selection is fitted "
        f"{body['selection_scope']!r} and the record stores "
        f"{spec['scope']!r}")
    assert body["selection_fit_on"] == spec["fit_on"], (
        f"{shape}: the panel says {body['selection_fit_on']!r} and the record "
        f"says {spec['fit_on']!r}")

    # And the prose carries the same fact as the payload beside it.
    assert spec["fit_on"] == "training rows only"
    assert "once over the training rows" in body["note"], (
        f"{shape}: the payload says {body['selection_fit_on']!r} and the "
        f"sentence a person reads does not. Note: {body['note']!r}")


# ═══════════ 3 · the vocabulary has one home ═══════════

def test_the_recorded_sentence_and_the_panel_are_composed_from_one_table():
    """`declare`'s rewrite and the panel's note draw the same clause.

    The defect was two hand-written sentences about one fact. This pins them to
    `_SCOPE_PHRASE`, so a future edit to either wording moves both — the shape
    `missingness._SCOPE_PHRASE` already has one module over.
    """
    spec = _sel.declare("mutual_info", "y", ["a", "b", "c"], n_features=2,
                        scope=_sel.TRAIN_ROWS)
    clause = _sel._SCOPE_PHRASE[_sel.FITTED_SCOPE]
    assert clause in spec["sentence"], (
        f"the recorded sentence does not carry the scope clause: "
        f"{spec['sentence']!r}")
    assert _sel.FITTED_SCOPE == _sel.TRAIN_ROWS, (
        "this door fits each model once; if that changed, `training.train` "
        "changed and every sentence composed from FITTED_SCOPE moved with it")


def test_the_plan_records_the_same_scope_the_panel_states(client):
    """Anything outside a test file must read `FITTED_SCOPE`, or the constant
    is a capability with no consumer (`AGENT_ONBOARD` §07 trap 1).

    `pipeline_plan.compose` records `scope_fitted` on the selection step — the
    production reader — and it is DRIVEN here rather than grepped: a real
    project with a real recorded spec, composed into a real plan, and the step
    the model would actually be handed is the thing read.
    """
    from turbotab import pipeline_plan as _plan
    from turbotab import training as _training

    pid, spec = _sealed(client, "continuous")
    project = api._project(pid)
    plan = _plan.compose(project, "ridge", _training.feature_frame(project))

    steps = [s for s in plan.steps if s.source == "selection"]
    # POSITIVE CONTROL — the plan really carries a selection step.
    assert steps, "the composed plan has no selection step to read"
    fitted = steps[0].params["scope_fitted"]
    assert fitted == _sel.FITTED_SCOPE == spec["scope"], (
        f"the plan fits {fitted!r}, the record stores {spec['scope']!r} and "
        f"the panel states {_sel.FITTED_SCOPE!r}; these are one fact")

"""D1 of the 2026-10 sweep over HTTP: a form previewed for a column the adjustment answers leave out
of the model is a typed preview, never a 500 (on Sullivan 2021 the plan's spline for `RIDAGEYR`,
previewed again after answers that left it out, raised KeyError in ``plan_previews``).

The journey is the NHANES one (sugar → fasting glucose, under Estimate), as
``test_confirm_sweep_api`` drives it, to the form question. The truth is the fixture's declared causal reading (``truths.py``): HDL is downstream of
the diet, so the total effect's primary model leaves it out; age comes before the diet and causes
both, so the model holds it.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
from turbotab.core.tests.stage_harness import NHANES
from turbotab.core.tests.truths import FIXTURE_TRUTHS, answer_refusal, answers, asked, fixture_truth
from turbotab.server import schemas
from turbotab.server.tests.conftest import wait_for, when_reached
from turbotab.server.tests.test_quest_api import ROLES, rule

TRUTH = FIXTURE_TRUTHS["_tt_tmp_nhanes.csv"]
SPLINE = {"form": "spline", "knots": 4}
LEFT_OUT = {"kind": "set_forms", "forms": {"hdl": SPLINE}}
MIXED = {"kind": "set_forms", "forms": {"hdl": SPLINE, "age": SPLINE}}


def preview(drive: Any, body: dict[str, Any]) -> Any:
    """``body`` previewed, again while the Router holds it behind a question a reshaped table is
    recomputing (``conftest.when_reached``)."""
    return when_reached(lambda: drive.c.post(f"/api/projects/{drive.pid}/preview", json=body))


def settle_reads(drive: Any, body: dict[str, Any], rounds: int = 4) -> None:
    """The readings ``body``'s preview asks first, answered from the fixture's truth and recorded,
    as a user who knows the table answers the ask: a refusal's (the answer's own validators), or
    the fit's ask on the preview's caution (``plan_previews.asks_first``)."""
    post = lambda d: drive.c.post(f"/api/projects/{drive.pid}/decisions", json=d)  # noqa: E731
    for _ in range(rounds):
        response = answer_refusal(post, preview(drive, body), drive.truth,
                                  lambda: preview(drive, body))
        assert response.status_code == 200, response.text[:600]
        caution = response.json().get("caution") or {}
        if not asked(caution.get("exits") or []):
            return
        for decision in answers({"exits": caution["exits"]}, drive.truth):
            r = post(decision)
            assert r.status_code == 200, (decision, r.text[:600])
    raise AssertionError(f"the preview of {body} still asks after {rounds} rounds")


def test_a_form_previewed_for_a_column_the_answers_leave_out_is_a_typed_preview(client):
    # causes sugar, causes glucose, after sugar: HDL a mediator, age a confounder
    assert TRUTH["adjust:hdl"] == "no,yes,yes" and TRUTH["adjust:age"] == "yes,yes,no"
    drive = open_project(client, NHANES, fixture_truth("_tt_tmp_nhanes.csv"))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "glucose"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(ROLES)
    drive.answer("exclusions", rule(500, 5000))
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    answer_plan(drive, "sugar")
    drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                       "energy_column": "kcal", "nutrients": ["sugar"]})
    assert drive.reach("form", timeout=240)["status"] == "open"  # the form question, as asked
    for body in (LEFT_OUT, MIXED):
        settle_reads(drive, body)
    wait_for(client, drive.pid, {"cohort": "fresh", "split": "fresh"}, timeout=240)
    recorded = len(drive.view()["decisions"])

    left_out = preview(drive, LEFT_OUT)
    assert left_out.status_code == 200, left_out.text[:600]
    said = schemas.PreviewResult.model_validate(left_out.json())
    assert said.views == [] and said.caution is None
    assert said.note == ("The adjustment answers leave `hdl` out of the primary model, so its "
                         "form changes nothing that model sees.")

    # Named beside it, the column the model holds is drawn, and the note is silent.
    mixed = preview(drive, MIXED)
    assert mixed.status_code == 200, mixed.text[:600]
    drawn = schemas.PreviewResult.model_validate(mixed.json())
    assert drawn.views and drawn.views[0].title == "How `age` enters the model"
    assert drawn.note is None
    assert len(drive.view()["decisions"]) == recorded  # a preview records nothing

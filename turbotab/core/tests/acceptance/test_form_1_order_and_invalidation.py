"""FORM · 1 and 3 · the form question's place in the Router, what a transform does to it, and why
a non-significant nonlinearity test never refits a straight line (MODELING_SEQUENCE §1 row 5, §2,
§4; the chain test of BLUEPRINT §13).

One dietary inference journey through the real server (``server_drive``), answered from its
fixture's declared truth (``form_fixtures.TRUTH``):

1. The form question sits after the domain transforms (the energy model) and before the model
   families; the families are refused "not yet" while it is open, and no estimate is served before
   it is answered (the served gate).
2. Its card asks about the declared exposure and the adjusted continuous confounder (``age``),
   states ``activity`` (eight whole values) and total energy instead, and proposes a spline with k
   by Harrell's rule on the effective sample size (900 rows: 5 knots).
3. The one-tap answer records each form on its present scale with its unit; the fit's knots are
   Harrell's percentiles of the raw values (written out by hand here).
4. After the estimates were seen, the energy model becomes the residual method: a transform of the
   exposure. The exposure's form, its knots and the estimand's unit are stale: the form question
   opens again (``followup`` "stale"), the fit is withheld, the design applies no stale spline, the
   caption's unit is the residual's; the confounder's form, whose scale did not change, stands.
5. Asked again, the form is declared on the residual scale: the knots are Harrell's percentiles of
   the residual ``N − b(E − Ē)`` computed here by least squares, and the record labels the curve as
   the nutrient's curve on the energy-adjusted scale at mean energy, offering spline(N) + E.
6. A switch of the confounder's spline to a straight line once the estimates were seen is recorded
   "After the estimates were seen", with Grambsch & O'Brien's warning, verbatim.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from turbotab.core import contracts
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests.acceptance import form_fixtures as ff
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.truths import Truth


def _fresh(drive: Any, stage: str, until: Any, timeout: float = 240.0) -> dict[str, Any]:
    end = time.monotonic() + timeout
    while True:
        found = drive.artifact(stage)
        if until(found):
            return found
        assert time.monotonic() < end, f"{stage} never reached the state the test waits for"
        time.sleep(0.1)


def _step(drive: Any, key: str) -> dict[str, Any]:
    return next(s for s in drive.view()["interview"] if s["key"] == key)


@pytest.fixture(scope="module")
def journey(tmp_path_factory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("form_chain")
    frame = ff.diet()
    csv = folder / "diet.csv"
    frame.to_csv(csv, index=False)
    truth = Truth(ff.TRUTH, fixture="the FORM chain")
    seen: dict[str, Any] = {"frame": frame}
    with local_server(folder / "home") as client:
        drive = open_project(client, csv, truth)
        ff.open_inference(drive)
        drive.answer("estimand", {"kind": "set_estimand", "exposure": "protein_g",
                                  "effect": "total", "contrast": "substitution",
                                  "measure": "mean_difference"})
        drive.reach("adjustment")
        from turbotab.core.tests.truths import answer_adjustment

        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], truth)
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                           "energy_column": "kcal", "nutrients": ["protein_g"]})
        end = time.monotonic() + 240
        while _step(drive, "form")["status"] != "open":
            assert time.monotonic() < end, _step(drive, "form")
            time.sleep(0.1)
        seen["steps_open"] = drive.view()["interview"]
        early = drive.post({"kind": "select_models", "models": ["linear"]})
        seen["models_early"] = (early.status_code, early.json())
        seen["card"] = drive.artifact("forms")
        r = drive.post(seen["card"]["answer"])
        assert r.status_code == 200, r.text[:600]
        seen["record_forms"] = drive.view()["decisions"][-1]
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        seen["fit"] = drive.artifact("fit")
        seen["view_fit"] = drive.view()
        # A transform of the exposure, after the estimates were seen.
        drive.decide({"kind": "set_energy_adjustment", "method": "residual",
                      "energy_column": "kcal", "nutrients": ["protein_g"]})
        seen["view_stale"] = drive.view()
        seen["fit_withheld"] = _fresh(drive, "fit", lambda f: f.get("withheld") is not None)
        seen["design_stale"] = drive.artifact("design")
        seen["card_residual"] = _fresh(
            drive, "forms", lambda c: any(n["receives"] == "protein_g_adj" for n in c["needs"]))
        drive.decide({"kind": "set_exposure_form", "column": "protein_g", "form": "spline"})
        seen["record_residual"] = drive.view()["decisions"][-1]
        seen["view_reanswered"] = drive.view()
        seen["fit_residual"] = _fresh(drive, "fit", lambda f: f.get("withheld") is None)
        # The confounder's spline dropped once the estimates were seen.
        drive.decide({"kind": "set_exposure_form", "column": "age", "form": "linear"})
        seen["record_switch"] = drive.view()["decisions"][-1]
        seen["methods"] = client.get(f"/api/projects/{drive.pid}/methods").json()
    return seen


# ── 1 · the order ────────────────────────────────────────────────────────────


def test_1_the_form_question_comes_after_the_domain_transforms_and_before_the_families(journey):
    from turbotab.core.interview import QUESTION_KEYS

    order = list(QUESTION_KEYS)
    assert order.index("energy_adjustment") < order.index("form") < order.index("models")
    assert order.index("form") < order.index("modification") < order.index("causal")
    steps = {s["key"]: s for s in journey["steps_open"]}
    assert steps["energy_adjustment"]["status"] == "answered"
    assert steps["form"]["status"] == "open" and steps["models"]["status"] == "waiting"
    status, body = journey["models_early"]
    assert status == 409 and body["error"]["code"] == "not_yet"
    assert body["error"]["message"].startswith("The functional-form question comes before this one")
    # the relation is declared in the one registry and fires for this chain
    fired = {f.relation.name for f in contracts.fired(
        {"exposure_transform": "residual", "functional_form": "spline"}, "inference")}
    assert {"transform-invalidates-form", "transform-precedes-form"} <= fired


def test_1_the_card_asks_the_exposure_and_the_continuous_confounder_with_k_by_the_rule(journey):
    card = journey["card"]
    needs = {n["column"]: n for n in card["needs"]}
    assert set(needs) == {"protein_g", "age"}
    assert needs["protein_g"]["role"] == "exposure" and needs["age"]["role"] == "confounder"
    stated = {s["column"]: s["why"] for s in card["stated"]}
    assert stated["activity"] == "8 distinct values: it enters as recorded"
    assert stated["kcal"] == "total energy: the energy model sets its term"
    # Harrell's rule on the effective sample size: 900 analyzed rows of a numeric outcome → 5 knots
    assert card["n_effective"] == 900 and card["rule_knots"] == 5
    assert card["rule"] == ("k = 5 by Harrell's rule (3 knots below an effective sample size of "
                            "30, 5 from 100, else 4; here 900, the number of analyzed rows)")
    for n in needs.values():
        assert n["scale"] == "raw" and n["proposal"] == {"form": "spline", "knots": 5,
                                                          "knots_rule": "harrell"}
        assert [o["value"] for o in n["options"]][0] == "spline"
    assert needs["protein_g"]["unit"] == "unit of `protein_g`"
    # a confounder in three or fewer groups and a data-derived cut point are blocked and recorded
    rungs = {o["value"]: o["rung"] for o in needs["age"]["options"]}
    assert rungs["optimal"] == "block_and_record" and rungs["quintiles"] == "rank_lower"


def test_1_each_form_is_recorded_on_its_scale_and_the_knots_are_harrells_percentiles(journey):
    record = journey["record_forms"]["decision"]
    assert record["kind"] == "set_forms"
    for column in ("protein_g", "age"):
        spec = record["forms"][column]
        assert spec["knots"] == 5 and spec["knots_rule"] == "harrell"
        assert spec["n_effective"] == 900 and spec["scale"] == "raw"
        assert spec["unit"] == f"unit of `{column}`"
    model = journey["fit"]["models"][0]
    tests = {(t["column"], t["test"]): t for t in model["exposure_tests"]}
    frame = journey["frame"]
    for column in ("protein_g", "age"):
        assert np.allclose(tests[(column, "overall")]["knots"],
                           ff.harrell_knots(frame[column], 5), rtol=0, atol=1e-10)


# ── 1 · a transform of the exposure re-asks its form, knots and unit ─────────


def test_1_a_transform_of_the_exposure_leaves_its_form_stale_and_reasks_it(journey):
    view = journey["view_stale"]
    state = view["state"]
    steps = {s["key"]: s for s in view["interview"]}
    assert steps["form"]["status"] in ("open", "waiting")
    from turbotab.core.decisions import ProjectState

    st = ProjectState(**state)
    assert set(ef.stale_forms(st)) == {"protein_g"}
    assert set(ef.current_forms(st)) == {"age"}  # the confounder's scale did not change
    # the estimates wait for the question, and the design applies no stale spline
    withheld = journey["fit_withheld"]
    assert withheld["withheld"] == ("No estimate is shown until the functional-form question is "
                                    "answered: the form of what you study and of each continuous "
                                    "covariate is declared on its final scale before any "
                                    "estimate.")
    links = journey["design_stale"]["lineage"]["links"]
    assert not any("protein_g_adj" in str(link) and "spline" in link["operation"]
                   for link in links)
    assert "restricted cubic spline, 5 knots" in {link["operation"] for link in links}  # age's
    # the estimand's unit is the residual's now, never the stale raw unit
    from turbotab.core.estimand import caption

    assert "per unit of `protein_g`'s energy-adjusted residual on `kcal`, at mean energy" in \
        caption(st)


def test_1_cut_points_declared_on_the_raw_scale_are_never_kept_after_a_transform():
    """The relation holds for cut points as for knots: quintiles declared on raw grams are stale
    under a log residual and under a density, never applied, the unit the new scale's; the
    standard model, which keeps the nutrient as recorded, leaves them standing; a scale score's
    form is stale once the scale's key changes."""
    from turbotab.core.decisions import EnergyAdjustment, ExposureFormSpec, ProjectState

    raw = ExposureFormSpec(form="quintiles", scale="raw", unit="unit of `protein_g`")
    base = ProjectState(purpose="inference", exposure_forms={"protein_g": raw})
    assert set(ef.current_forms(base)) == {"protein_g"}
    for adj in (EnergyAdjustment(method="residual", energy_column="kcal", nutrients=["protein_g"],
                                 log_transform=True),
                EnergyAdjustment(method="density", energy_column="kcal", nutrients=["protein_g"])):
        st = base.model_copy(update={"energy_adjustment": adj})
        assert set(ef.stale_forms(st)) == {"protein_g"}
        assert ef.design_forms(st, ["protein_g"], ["protein_g"]) == {}
        assert ef.estimand_unit(st, "protein_g") == ef.scale_words(st, "protein_g") != raw.unit
    # the standard model keeps the nutrient as recorded: no transform, the form stands
    std = base.model_copy(update={"energy_adjustment": EnergyAdjustment(
        method="standard", energy_column="kcal", nutrients=["protein_g"])})
    assert set(ef.current_forms(std)) == {"protein_g"} and ef.estimand_unit(std, "protein_g") == \
        "unit of `protein_g`"
    # a scale score's form belongs to its scoring: a new key re-asks it
    from turbotab.core.decisions import ScaleSpec

    scale = ScaleSpec(name="pss", items=["q1", "q2", "q3", "q4"], low=0, high=4, kind="reflective")
    scored = ProjectState(purpose="inference", scales=[scale])
    spline = ExposureFormSpec(form="spline", knots=3, scale=ef.transform_signature(scored, "pss"))
    st = scored.model_copy(update={"exposure_forms": {"pss": spline}})
    assert set(ef.current_forms(st)) == {"pss"}
    rekeyed = st.model_copy(update={"scales": [scale.model_copy(update={"reverse": ["q2"]})]})
    assert set(ef.stale_forms(rekeyed)) == {"pss"}
    assert ef.scale_words(st, "pss") == "point of the `pss` score (the sum of 4 items, each 0–4)"


def test_1_the_card_asks_again_on_the_residual_scale_and_the_new_knots_are_the_residuals(journey):
    card = journey["card_residual"]
    need = next(n for n in card["needs"] if n["column"] == "protein_g")
    assert need["receives"] == "protein_g_adj"
    assert need["scale"] == "energy:residual:kcal:linear:"
    assert need["unit"] == "unit of `protein_g`'s energy-adjusted residual on `kcal`, at mean energy"
    record = journey["record_residual"]["decision"]
    assert record["scale"] == "energy:residual:kcal:linear:" and record["knots"] == 5
    steps = {s["key"]: s for s in journey["view_reanswered"]["interview"]}
    assert steps["form"]["status"] == "answered"
    # The residual N − b(E − Ē), b by least squares on every analyzed row, written out here.
    frame = journey["frame"]
    N, E = frame["protein_g"].to_numpy(float), frame["kcal"].to_numpy(float)
    b = np.cov(N, E, ddof=1)[0, 1] / np.var(E, ddof=1)
    residual = N - b * (E - E.mean())
    tests = {(t["column"], t["test"]): t for t in journey["fit_residual"]["models"][0]
             ["exposure_tests"]}
    assert np.allclose(tests[("protein_g_adj", "overall")]["knots"],
                       ff.harrell_knots(residual, 5), rtol=0, atol=1e-8)


def test_8_a_spline_on_a_residual_is_labeled_and_offers_the_substitution_route(journey):
    need = next(n for n in journey["card_residual"]["needs"] if n["column"] == "protein_g")
    label = ("the curve of `protein_g` on the energy-adjusted scale at mean energy (its residual "
             "on `kcal`), not the substitution curve at fixed total energy: the residual model "
             "equals the standard model only for a straight line; a spline of `protein_g` with "
             "`kcal` beside it (the standard model) is the route to the substitution curve")
    assert need["label"] == label
    assert need["route"] == [
        {"kind": "set_energy_adjustment", "method": "standard", "energy_column": "kcal",
         "nutrients": ["protein_g"]},
        {"kind": "set_exposure_form", "column": "protein_g", "form": "spline", "knots": 5}]
    assert journey["record_residual"]["sentence"] == (
        "After the estimates were seen, `protein_g` entered the models as a restricted cubic "
        "spline with `5` knots, k = 5 by Harrell's rule (3 knots below an effective sample size "
        "of 30, 5 from 100, else 4; here 900, the number of analyzed rows), at the 5, 27.5, 50, "
        "72.5 and 95 percentiles of its energy-adjusted values in the rows each model was fit on "
        "(Harrell's placement); the test of association is the Wald test that every term is "
        "zero, and nonlinearity was tested by a Wald test that its nonlinear terms are zero, a "
        "non-significant result never refitting a straight line; quintiles were reported beside "
        f"it, their boundaries and reference stated, with the p for linear trend (customary) "
        f"across quintile medians; this is {label}.")


# ── 3 · a non-significant nonlinearity test never silently refits a line ─────


def test_3_the_switch_to_a_straight_line_is_recorded_after_the_estimates_were_seen(journey):
    record = journey["record_switch"]
    assert record["after_estimates"] is True
    assert record["sentence"] == (
        "After the estimates were seen, `age` entered the models as a straight line, in place of "
        "the restricted cubic spline declared before; dropping a curve after its nonlinearity "
        "test inflates the test of association's type I error (Grambsch & O'Brien 1991, Stat "
        "Med 10:697).")
    assert record["sentence"] in journey["methods"]["text"]

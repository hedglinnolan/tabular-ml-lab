"""L63-B. Three figures answer *which model is this about?* and all three
answered it wrongly, in three different ways.

- `GUIDED-243` — the ROC's bootstrap generator was created once and consumed
  inside the per-model loop, so a model's published interval depended on how
  many models were looped before it. Unreachable while the payload held one
  curve; the widening below is what makes it reachable, which is why it lands
  first.
- `GUIDED-242` — the forest plot returned on the first fitted model exposing
  `coef_`, and its payload carried no model key while its caption said *"Model
  coefficients for N predictors"*.
- `GUIDED-236` — `figure_bundle._risks_or_refuse` returned a dict of exactly
  one model to a spec that has always taken a dict, looped it, counted it in
  its caption and carried a checklist item about overlaying models.

Plus the four consequences the widening creates, which land with it:
`DRIVE-016`'s series cap, the decision curve's concatenated risk rug, the
`models_overlaid` predicate that could not fail, and the companion asymmetry.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import eventfixture                                # noqa: E402
from turbotab import figure_bundle as FB                         # noqa: E402
from turbotab import figure_specs as FS                          # noqa: E402
from turbotab import pageharness as PH                           # noqa: E402
from turbotab import training as T                               # noqa: E402
from turbotab.project import AnalysisProject                     # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def _fitted(models):
    """A real clinical project, fitted on `models`, in the order given.

    The order is the point: it is the order the user ticked them, and it is the
    only thing that differs between the two halves of every assertion below.
    """
    df = pd.read_csv(os.path.join(DATA, "clinical_risk.csv"))
    df = df[df["readmit_30d"].notna()].copy()
    project = AnalysisProject.from_dataframe(df, "clinical_risk.csv")
    # `"classification"` EXACTLY, and it is not cosmetic: `training.py:639`
    # gates `probabilities` on this string, so `"binary classification"` — the
    # phrase the shape tables use for the FIXTURE — fits every model and scores
    # none of them, and every assertion here would then be about an empty run.
    project.target, project.task_type = "readmit_30d", "classification"
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    project.seal_lockbox(labels, fraction=len(labels) / len(project.df))
    eventfixture.choose_event(project, required=True)
    project.training_run = T.train(project, models)
    return project


def _row(bundle, figure_id):
    for row in (bundle.get("admitted") or []) + (bundle.get("held") or []):
        if row.get("id") == figure_id:
            return row
    return None


def _synthetic(n=140, seed=5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    a = np.clip(y * 0.35 + rng.random(n) * 0.6, 0, 1)
    b = np.clip(y * 0.20 + rng.random(n) * 0.8, 0, 1)
    return y, a, b


# ═══════════ B1 · GUIDED-243 — the interval does not depend on tick order ════

def test_a_models_interval_is_the_same_alone_first_and_last(capsys):
    """The whole defect, in the only three positions a model can be in.

    `rng` was created once above the loop and drawn from inside it, so every
    model consumed the stream the models before it had already advanced. The
    point estimate was stable and only the interval moved, which is why nothing
    looked wrong.
    """
    y, a, b = _synthetic()
    alone = FS.roc_payload(y, {"A": a})["curves"]["A"]
    first = FS.roc_payload(y, {"A": a, "B": b})["curves"]["A"]
    last = FS.roc_payload(y, {"B": b, "A": a})["curves"]["A"]

    # The precondition, established rather than hoped for: an interval that
    # came back `None` would satisfy every equality below.
    assert alone["c_interval"], "no interval was estimable, so this checks nothing"
    assert alone["c_interval"] == first["c_interval"] == last["c_interval"], (
        f"the same model published {alone['c_interval']} alone, "
        f"{first['c_interval']} first and {last['c_interval']} last. The dict "
        f"order is the user's tick order, so this is the same figure "
        f"publishing a different confidence interval depending on what a user "
        f"clicked first")
    assert alone["c_statistic"] == first["c_statistic"] == last["c_statistic"]
    with capsys.disabled():
        print(f"\n  A alone/first/last → {alone['c_interval']} "
              f"(C = {alone['c_statistic']})")


def test_the_bootstrap_resamples_are_shared_across_models():
    """The other half of the seed decision, asserted so it is not undone.

    Independent draws per model would also be order-independent, and would put
    a difference between two models' intervals that came from the generator
    rather than from the models. Both models see the same replicates, so a
    model with identical risks to another gets an identical interval.
    """
    y, a, _ = _synthetic()
    both = FS.roc_payload(y, {"A": a, "A_copy": np.array(a)})["curves"]
    assert both["A"]["c_interval"] == both["A_copy"]["c_interval"], (
        "two models with byte-identical risks got different intervals, so the "
        "resamples are not shared and the difference is the generator's")


# ═══════════ B2 · GUIDED-242 — the forest plot names its model ═══════════════

def test_the_forest_plot_names_the_model_its_coefficients_came_from(capsys):
    """Same rows, same data, different published coefficients, no model named.

    Both `logreg` and `lda` expose `coef_` on this fixture, and
    `_coefficients_for` returned on the first one it reached.
    """
    published = {}
    for order in (["logreg", "lda"], ["lda", "logreg"]):
        row = _row(FB.render(_fitted(order)), "forest")
        assert row is not None, f"the forest plot was not drawn for {order}"
        payload = row["payload"]
        assert payload.get("model"), (
            f"ticking {order} published coefficients under no model name at "
            f"all; the caption reads {row['caption'][:120]!r}")
        assert payload["model"] in row["caption"], (
            "the payload names the model and the caption does not, which is "
            "the half a reader sees")
        assert payload["n_models_with_coefficients"] == 2, payload
        assert payload["other_models_with_coefficients"], (
            "two models expose coefficients and the payload names no other, "
            "so a reader cannot tell this was a choice")
        published[order[0]] = (payload["model"],
                               [r["estimate"] for r in payload["rows"][:3]])

    (first_model, first_rows), (second_model, second_rows) = (
        published["logreg"], published["lda"])
    assert first_model != second_model, (
        "both tick orders drew the same model, so this fixture no longer "
        "exercises the defect")
    assert first_rows != second_rows, (
        "the two orders published identical coefficients, so there is nothing "
        "for the model name to disambiguate and this test proves nothing")
    with capsys.disabled():
        print(f"\n  {first_model}: {first_rows}\n  {second_model}: {second_rows}")


def test_a_forest_payload_with_two_models_and_no_name_fails_its_checklist():
    """The revert probe for the rule, in place.

    `test_a_linear_axis_for_ratio_measures_fails_the_checklist` is the model:
    the predicate has to be shown failing on the shape it exists to catch, or
    it is `models_overlaid`'s own defect one figure over.
    """
    payload = FS.forest_payload([{"name": "age", "estimate": 1.4}],
                                model="Logistic Regression",
                                other_models=["Linear Discriminant Analysis"])
    assert not [i.id for i in FS.FOREST.checklist if not i.check(payload)]

    payload["model"] = None
    failed = [i.id for i in FS.FOREST.checklist if not i.check(payload)]
    assert "names_the_model_it_is_about" in failed, (
        "a payload saying two models have coefficients and naming none passed "
        "the item that exists to catch exactly that")


def test_one_model_with_coefficients_may_stay_silent():
    """The governing rule permits silence; it forbids a false assertion.

    With a single candidate there is nothing to disambiguate, so a payload that
    names no model is not making a claim a reader can be wrong about. A
    predicate that failed here would force a name onto every direct caller and
    would be a style rule wearing a checklist item's clothes.
    """
    payload = FS.forest_payload([{"name": "age", "estimate": 1.4}])
    assert payload["n_models_with_coefficients"] == 0
    assert not [i.id for i in FS.FOREST.checklist if not i.check(payload)]


# ═══════════ B3 · GUIDED-236 — every scored model reaches the axis ══════════

def test_the_roc_overlays_every_model_that_scored(capsys):
    """The row itself, end to end through `figure_bundle.render`."""
    project = _fitted(["logreg", "lda", "rf"])
    scored = [r for r in project.training_run.results if r.probabilities]
    assert len(scored) >= 2, (
        f"only {len(scored)} model produced held-out probabilities, so there "
        f"is nothing to overlay and this test asserts nothing")

    row = _row(FB.render(project), "roc")
    assert row is not None, "the ROC was not drawn"
    payload = row["payload"]
    curves, excluded = payload["curves"], payload["excluded_models"]
    assert len(curves) >= 2, (
        f"{len(scored)} models scored and the payload carries {len(curves)} "
        f"curve(s): {sorted(curves)}")
    # THE INVARIANT, which is stronger than `len(curves) == len(scored)`:
    # every model that scored is either on the axis or named as not being.
    # `rf` is genuinely not overlayable here — `RFWrapper` forwards no
    # `classes_`, so the run recorded no event for it (`GUIDED-245`) — and the
    # honest form of that is a named exclusion, not a silent absence.
    assert payload["n_models_scored"] == len(scored)
    assert len(curves) + len(excluded) == len(scored), (
        f"{len(scored)} scored, {len(curves)} drawn, {len(excluded)} named: "
        f"{[e['model'] for e in excluded]}")
    assert not [i["id"] for i in row["checklist"] if not i["passed"]]
    for entry in excluded:
        assert entry["model"] in row["caption"], (
            f"{entry['model']} was dropped and the caption does not say so")
    with capsys.disabled():
        print(f"\n  {len(curves)} curves: "
              + "; ".join(f"{k} C={v['c_statistic']:.3f}"
                          for k, v in curves.items())
              + (f" · excluded {[e['model'] for e in excluded]}"
                 if excluded else ""))


def test_ticking_an_uncalibratable_model_first_does_not_delete_the_figures():
    """`GUIDED-245`'s consequence, and the reason `predictions_for` picks the
    first CALIBRATABLE run rather than the first scored one.

    `RFWrapper` does not forward `classes_`, so `rf` returns 120 held-out
    probabilities and records no `positive_label`. `predictions_for` took
    `scored[0]` and refused when that one model recorded no event — so ticking
    `['rf', 'logreg']` made `has_predictions` false and the ROC, the
    calibration plot AND the decision curve all disappeared from a project that
    had fitted a perfectly calibratable logistic regression. `['logreg', 'rf']`
    drew all three. Tick order decided whether three clinical figures existed.
    """
    CLINICAL = {"roc", "calibration", "decision_curve"}
    both_orders = {}
    for order in (["rf", "logreg"], ["logreg", "rf"]):
        bundle = FB.render(_fitted(order))
        both_orders[order[0]] = {r["id"] for r in
                                 bundle["admitted"] + bundle["held"]} & CLINICAL
    assert both_orders["rf"] == both_orders["logreg"] == CLINICAL, (
        f"the clinical figures depend on tick order: "
        f"rf-first drew {sorted(both_orders['rf'])}, "
        f"logreg-first drew {sorted(both_orders['logreg'])}")


@pytest.mark.parametrize("only", ["glm", "rf", "logreg"])
def test_one_wrapper_backed_model_still_draws_the_clinical_figures(only, capsys):
    """`GUIDED-245`, and it is the whole user-visible claim.

    L63 fixed the ORDERING half — `predictions_for` takes the first
    calibratable run, so a user who ticks a good model beside a bad one is
    fine. **A user who ticks one bad model was not.** Driven at L63's HEAD on
    `clinical_risk.csv`: fitting only `glm` gave `positive_label=None`,
    `has_predictions=False`, and the ROC, the calibration plot and the decision
    curve were all absent from the served bundle. `rf` behaved identically.
    Only `logreg` — a raw sklearn estimator, not a wrapper — drew all three.

    The cause was one attribute: `training.py:648` records which class a
    model's probabilities are about by reading `classes_` off the fitted
    pipeline, and `BaseModelWrapper` forwarded `coef_` and `intercept_` and not
    that. `models/glm.py:25` holds a real `LogisticRegression`, so the answer
    was available and unexposed the whole time.

    `logreg` is parametrized beside the two wrappers as the control: it drew
    all three before this fix and must still.
    """
    project = _fitted([only])
    scored = [r for r in project.training_run.results if r.probabilities]
    assert scored, f"{only} produced no held-out probabilities at all"
    assert all(r.positive_label is not None for r in scored), (
        f"{only} scored {len(scored)} model(s) and none recorded which class "
        f"its probabilities are about, so `predictions_for` must refuse and "
        f"the three clinical figures cannot be drawn — `GUIDED-245`")

    bundle = FB.render(project)
    drawn = {row["id"] for row in bundle["admitted"] + bundle["held"]}
    missing = {"roc", "calibration", "decision_curve"} - drawn
    assert not missing, (
        f"a project fitted with only `{only}` is missing {sorted(missing)}. "
        f"not_drawn says: "
        f"{ {r['id']: r.get('why', '')[:80] for r in bundle['not_drawn']} }")
    with capsys.disabled():
        print(f"\n  {only}: event={scored[0].positive_label!r} · "
              f"{len(drawn & {'roc', 'calibration', 'decision_curve'})}/3 drawn")


def test_a_project_whose_models_all_record_no_event_still_refuses():
    """The other polarity, and it must not be lost to the fix above.

    Where NOTHING records which class its probabilities are about, refusing is
    the honest branch — guessing `1` is right on a 0/1 target and silently
    wrong on every other one (`GUIDED-093`). A fix that made the figures appear
    here would have traded a false absence for a false picture.

    **THE FIXTURE CHANGED AT `L64-A3` AND THE TEST DID NOT.** It used to fit
    `rf` and rely on `RFWrapper` not forwarding `classes_` — so the refusal it
    proves was proved by a *defect*, and its own message said so: *"this
    fixture's models now record an event, so it no longer exercises the
    refusal — `GUIDED-245` may be fixed."* It is. Every registry model that
    produces probabilities now records its event, which is the whole of Part A,
    so there is no fitted model left that reaches this branch.

    The state is therefore constructed rather than found: a real run, with the
    recorded label cleared on every scored result. That is exactly what a
    wrapper which cannot name its classes produces, and it is what the branch
    is about — the refusal is a property of the RECORD, not of any one
    estimator, so building the record directly is the honest fixture rather
    than a stand-in for one.
    """
    project = _fitted(["logreg", "rf"])
    scored = [r for r in project.training_run.results if r.probabilities]
    assert len(scored) >= 2, (
        f"only {len(scored)} model scored, so clearing the labels below does "
        f"not exercise a project where SEVERAL models cannot name their event")
    # THE CONTROL, and it is what stops this becoming trap #3 — a fixture
    # manufacturing the absence it then asserts. The labels are real before
    # they are cleared, so the clearing is the only difference between a
    # project that draws all three figures and one that refuses.
    assert all(r.positive_label is not None for r in scored), (
        "the models did not record an event to begin with, so this fixture is "
        "not showing that the refusal survives — it is showing GUIDED-245")
    assert FB.predictions_for(project) is not None
    for result in scored:
        result.positive_label = None

    assert FB.predictions_for(project) is None
    state = FB.state(project)
    assert state["has_predictions"] is False
    # AND THE REASON IT GIVES IS TRUE. This fell through to *"There are
    # predictions and the curve should be drawn"* — a sentence asserting the
    # opposite of the state it was explaining, on the surface whose only job is
    # to explain that state.
    why = state["has_predictions_because"]
    assert "should be drawn" not in why, why
    assert "Random Forest" in why and "which class" in why, why

    # AND THE TWO FIGURES THAT VANISH WITH IT NOW SAY WHY. `L64-A4`: `roc` and
    # `decision_curve` gate on exactly what `calibration` gates on and fell
    # through to *"This figure does not apply to this project"* — on a binary
    # classification project with 120 held-out predictions, where the figure
    # applies perfectly and the real reason was already written one branch over.
    bundle = FB.render(project)
    not_drawn = {row["id"]: row.get("why", "") for row in bundle["not_drawn"]}
    for figure_id in ("roc", "decision_curve", "calibration"):
        assert figure_id in not_drawn, f"{figure_id} was drawn"
        assert "does not apply" not in not_drawn[figure_id], (
            f"{figure_id} is absent for a reason the app knows and can name, "
            f"and it says {not_drawn[figure_id]!r}")
        assert "which class" in not_drawn[figure_id], not_drawn[figure_id]


def test_a_model_that_cannot_name_its_event_does_not_silence_one_that_can():
    """`L64-A4`, the other half, and the wrapper fix HIDES it rather than
    fixing it.

    `_no_predictions_because` tested *is any model unnamed* before it tested
    *did any model record a label*, so a project that ticked one good model and
    one bad one served `has_predictions: True` — all three clinical figures
    drawn — beside a sentence telling the user to *"fit a model that records
    its outcome level … and the curve is drawn"*, about a curve that was on
    screen. Driven at L63's HEAD on `['glm','logreg']`, both halves in one
    response.

    Every registry model now records its event, so the state has to be
    constructed. That is the point: the contradiction is a property of the
    sentence builder, and it would still be there if a wrapper regressed.
    """
    project = _fitted(["logreg", "rf"])
    scored = [r for r in project.training_run.results if r.probabilities]
    assert len(scored) >= 2
    unnamed = next(r for r in scored if r.name != "Logistic Regression")
    unnamed.positive_label = None

    state = FB.state(project)
    assert state["has_predictions"] is True, (
        "a model that cannot name its event silenced one that can, which is "
        "GUIDED-245's ordering half and L63 fixed it")
    why = state["has_predictions_because"]
    assert str(unnamed.name) in why, (
        f"the model that could not contribute is not named at all: {why!r}")
    assert "and the curve is drawn" in why, why
    assert "Fit a model that records its outcome level" not in why, (
        f"the served state instructs the user to fit a model that records its "
        f"outcome level, on a project that has one and has drawn the curve "
        f"from it: {why!r}")
    drawn = {row["id"] for row in FB.render(project)["admitted"]}
    assert {"roc", "calibration", "decision_curve"} <= drawn


def test_the_calibration_caption_names_the_model_it_drew():
    """It said *"Calibration of model"* for every project this app has drawn.

    `calibration_payload` takes `model_name=` defaulting to the literal string
    `"model"` and its caption reads that key; the only live caller dropped it
    and wrote the real name to a different key one line later. The figure held
    the answer in its own payload and could not say it.
    """
    project = _fitted(["logreg", "lda"])
    row = _row(FB.render(project), "calibration")
    assert row is not None, "the calibration plot was not drawn"
    name = row["payload"]["model"]
    assert name and name != "model"
    assert row["payload"]["model_name"] == name, (
        f"the payload carries the model under `model` and {name!r} is not in "
        f"`model_name`, which is the key the caption reads")
    assert f"Calibration of {name}" in row["caption"], (
        f"the caption does not name the model: {row['caption'][:160]!r}")


def test_the_decision_curve_gets_the_same_widening():
    """It reads the same helper, so it had the same defect from the same line.

    `_decision_curve_payload` and `_roc_payload` are the only two readers of
    `_risks_or_refuse`; a fix that widened one of them would have left the
    other drawing a single model under a caption that counts them.
    """
    project = _fitted(["logreg", "lda"])
    row = _row(FB.render(project), "decision_curve")
    assert row is not None, "the decision curve was not drawn"
    assert len(row["payload"]["models"]) == 2, row["payload"]["models"].keys()


def test_every_overlaid_model_is_binarized_against_the_same_event():
    """`GUIDED-093` one level out, and the reason it is a defensive assertion.

    `positive_label` is per result. Two models binarized against different
    events on one axis is a picture of two different outcomes, drawn
    confidently. No fixture in this repository produces a disagreement — a
    failed fit already has `probabilities = None` and never reaches `scored` —
    so this asserts the invariant holds rather than that the branch fires.
    """
    project = _fitted(["logreg", "lda"])
    scored = [r for r in project.training_run.results if r.probabilities]
    assert len({r.positive_label for r in scored}) == 1, (
        "this fixture's models disagree about the event, which would make the "
        "exclusion branch reachable — rewrite this test to drive it")
    y_true, risks, excluded, n_scored = FB._risks_or_refuse(project)
    assert not excluded and len(risks) == n_scored == len(scored)


def test_a_model_about_a_different_event_is_excluded_and_named():
    """The branch above, driven on a project mutated to disagree.

    Constructed rather than found: nothing in `sample_data/` fits two models
    against different events. What is asserted is that the drop is NAMED — a
    silent one would be the same defect the exclusion exists to prevent.
    """
    project = _fitted(["logreg", "lda"])
    scored = [r for r in project.training_run.results if r.probabilities]
    assert len(scored) >= 2
    scored[-1].positive_label = "a level nothing was fitted against"

    _y, risks, excluded, n_scored = FB._risks_or_refuse(project)
    assert n_scored == len(scored)
    assert len(risks) == len(scored) - 1, (
        "the model about another event was overlaid anyway")
    assert [e["model"] for e in excluded] == [str(scored[-1].name)]
    assert "two events on one axis" in excluded[0]["why"]

    payload = FS.roc_payload(_y, risks, excluded=excluded, n_scored=n_scored)
    assert not [i.id for i in FS.ROC.checklist if not i.check(payload)], (
        "a NAMED exclusion should satisfy the accounting item; only a silent "
        "drop should fail it")
    assert scored[-1].name in FS.ROC.caption(payload), (
        "the model was dropped and the caption does not say so, so the only "
        "record of it is in a payload key nobody reads")


# ═══════════ B4 · the consequences the widening creates ═════════════════════

def test_the_models_overlaid_item_can_fail(capsys):
    """`GUIDED-236`'s namesake: a predicate that could not ever be false.

    It read `p["legend"] == "inside"` against a key the producer hardcodes
    three lines away, and came back `passed: true` on a real two-model project
    over a payload holding one curve.
    """
    y, a, b = _synthetic()
    honest = FS.roc_payload(y, {"A": a, "B": b}, n_scored=2)
    assert not [i.id for i in FS.ROC.checklist if not i.check(honest)]

    # The pre-fix shape exactly: two models scored, one curve published, no
    # exclusion recorded.
    silent = FS.roc_payload(y, {"A": a}, n_scored=2)
    failed = [i.id for i in FS.ROC.checklist if not i.check(silent)]
    assert "models_overlaid" in failed, (
        "two models scored, one curve was published and nothing was named as "
        "excluded, and the item that exists to catch that passed")
    with capsys.disabled():
        print(f"\n  2 scored / 1 curve / 0 named → failed {failed}")


def test_the_risk_rug_is_keyed_per_model():
    """Widening concatenated every model's risks into one unlabeled list.

    Driven before the fix: 120 values became 480 across four overlaid
    distributions, under a guard reading `bool(p.get("risk_rug"))` that stays
    green over all four.
    """
    y, a, b = _synthetic()
    payload = FS.decision_curve_payload(y, {"A": a, "B": b}, n_scored=2)
    assert set(payload["risk_rug"]) == {"A", "B"}, payload["risk_rug"]
    assert set(payload["risk_rug"]) == set(payload["models"])
    assert not [i.id for i in FS.DECISION_CURVE.checklist
                if not i.check(payload)]

    flattened = dict(payload)
    flattened["risk_rug"] = [v for vals in payload["risk_rug"].values()
                             for v in vals]
    failed = [i.id for i in FS.DECISION_CURVE.checklist
              if not i.check(flattened)]
    assert "risk_distribution" in failed, (
        "a rug that concatenated both models into one unlabeled list passed "
        "the item about showing the distribution of predicted risks")


def test_the_decision_curve_accounts_for_the_models_it_did_not_draw(capsys):
    """`GUIDED-247`. The decision curve got the curves and not the accounting.

    `_risks_or_refuse` already excludes a model that cannot name its event and
    hands BOTH readers the same two keys — and only the ROC read them, so the
    decision curve drew the survivors and said nothing about the one it
    dropped. Driven: the DCA caption did not contain the excluded model's name
    while the ROC caption, on the same numbers, did.

    **The item insists on the CALLER-PASSED count, and that conjunct is the
    whole of it.** The decision curve's model dict has no drop path — every
    model handed in appears — so on the default path `len(models) +
    len(excluded)` equals the defaulted count by construction. An item without
    that conjunct would have been an 87th unfalsifiable item, in the same
    registry, in the loop that is removing them (`GUIDED-238`).
    """
    y, a, b = _synthetic()
    spec = FS.DECISION_CURVE

    def failed(payload):
        return [i.id for i in spec.checklist if not i.check(payload)]

    # Defaulted count: the item refuses to certify an accounting nobody stated.
    assert "models_accounted_for" in failed(
        FS.decision_curve_payload(y, {"A": a, "B": b}))
    # Two scored, two drawn.
    assert not failed(FS.decision_curve_payload(y, {"A": a, "B": b},
                                                n_scored=2))
    # THE PROVING ASSERTION, and it is the ROC's own one figure over: two
    # scored, ONE curve, nothing named → fail.
    assert "models_accounted_for" in failed(
        FS.decision_curve_payload(y, {"A": a}, n_scored=2))
    # Name it and it passes — and the name reaches the caption.
    excluded = [{"model": "B", "why": "the run recorded no event for its "
                                      "probabilities"}]
    payload = FS.decision_curve_payload(y, {"A": a}, excluded=excluded,
                                        n_scored=2)
    assert not failed(payload)
    caption = spec.caption(payload)
    assert "B is not drawn" in caption, caption[-200:]
    # Both branches of the threshold clause already end in a period, so copying
    # the ROC's join would have produced a doubled full stop.
    assert ".." not in caption.replace("...", ""), caption[-200:]
    with capsys.disabled():
        print(f"\n  DCA caption tail: …{caption[-90:]}")


def test_the_real_decision_curve_path_states_its_count():
    """And the shipped caller passes it, or the item above is unreachable.

    A capability with no consumer is `AGENT_ONBOARD.md` §07 trap #1, and an
    accounting item nothing feeds is exactly that.
    """
    project = _fitted(["logreg", "lda"])
    row = _row(FB.render(project), "decision_curve")
    assert row is not None
    assert row["payload"]["n_models_scored_stated"] is True, (
        "the bundle does not state how many models it scored, so the "
        "accounting item cannot fire on any real project")
    assert not [i["id"] for i in row["checklist"] if not i["passed"]]


def test_the_roc_caption_says_which_model_the_companion_covers():
    """The companion asymmetry, on the wire.

    The ROC declares `companions=("calibration", "decision_curve")` and the
    companion rule promotes them into the manuscript together — but calibration
    is about ONE model while the ROC now overlays N, and nothing said so.
    """
    project = _fitted(["logreg", "lda"])
    row = _row(FB.render(project), "roc")
    assert row is not None
    covered = row["payload"]["companion_covers_model"]
    assert covered, "the payload does not say which model calibration covers"

    calibration = _row(FB.render(project), "calibration")
    assert calibration is not None
    assert calibration["payload"]["model"] == covered, (
        f"the ROC says the companion covers {covered!r} and the calibration "
        f"payload says {calibration['payload']['model']!r}")
    assert len(row["payload"]["curves"]) > 1
    assert covered in row["caption"], (
        f"the caption does not name the model the companion covers, so a "
        f"reader sees two curves promoted beside a calibration plot for one "
        f"of them: {row['caption'][:400]!r}")


def test_a_single_curve_roc_does_not_mention_the_companion():
    """The other polarity. With one curve there is no asymmetry to disclose,
    and a caption that says so anyway is noise a reader has to discount."""
    project = _fitted(["logreg"])
    row = _row(FB.render(project), "roc")
    assert row is not None and len(row["payload"]["curves"]) == 1
    assert "covers" not in row["caption"], row["caption"][:300]


# ═══════════ DRIVE-016 · the fourth series is not the first one again ═══════

_FOUR = ["Alpha", "Beta", "Gamma", "Delta"]
_PID = None


def _four_series_routes():
    """A REAL project's routes, with a real four-curve ROC row swapped in.

    The page's controller needs the whole route set to paint anything, so a
    bundle handed to `/figures` alone renders an empty box — which would make
    every assertion below pass on nothing. The row swapped in is a genuine
    `figure_bundle.render` row with a genuine `figure_specs.roc_payload`; only
    the number of curves is manufactured, because no fixture in this repository
    fits four models that all record an event (`GUIDED-245`).
    """
    global _PID
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(os.path.join(DATA, "clinical_risk.csv"), "rb") as fh:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", fh, "text/csv")}).json()["id"]
    _PID = pid
    for kind, payload in (("set_target", {"column": "readmit_30d"}),
                          ("set_purpose", {"answer": "prediction"}),
                          ("set_grain", {"answer": "one_row_per_person"}),
                          ("set_eligibility", {"answer": "everyone"})):
        got = client.post(f"/project/{pid}/decision",
                          json={"kind": kind, "payload": payload})
        assert got.status_code == 200, (kind, got.text[:200])

    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "checklist",
                 "evidence/plausibility", "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = (got.json() if got.status_code == 200
                                            else {})
    routes[f"/project/{pid}/figures"] = _four_series_bundle()
    return routes


def _four_series_bundle():
    y, _a, _b = _synthetic()
    rng = np.random.default_rng(11)
    risks = {}
    for i, name in enumerate(_FOUR):
        risks[name] = np.clip(y * (0.4 - 0.05 * i) + rng.random(len(y)) * 0.6,
                              0, 1)
    payload = FS.roc_payload(y, risks, n_scored=len(_FOUR))
    assert len(payload["curves"]) == len(_FOUR)
    row = {"id": "roc", "title": "Discrimination", "tier": "CONFIRMATORY",
           "payload": payload, "caption": FS.ROC.caption(payload),
           "checklist": [{"id": i.id, "text": i.text, "passed": i.check(payload)}
                         for i in FS.ROC.checklist],
           "annotations": [], "drawable": True, "figure": "roc"}
    return {"admitted": [row], "held": [], "not_drawn": [], "unavailable": []}


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_a_fourth_series_is_told_apart_from_the_first(theme, capsys):
    """`DRIVE-016`. `WEBC` held four hues including `--c4` and indexed
    `[idx % length]`, while the ramp's own gate says three is the measured
    maximum and `--c4` is a low-chroma *Other* that never passes the CVD floor.

    The models page offers twelve models and caps nothing, so a fourth series
    is reachable. It now reuses `--c1` and is separated by a dash instead —
    the redundant channel §07 already relies on for print.

    **Asserted in DARK specifically**, because that is where `DRIVE-016` says
    the old ramp failed; `light` is parametrized beside it so a rule that only
    holds in one theme cannot pass as a rule.
    """
    if not PH.available():
        pytest.skip("no JS engine on this machine")
    bundle = _four_series_bundle()
    out = PH.run(
        "document.documentElement.setAttribute('data-theme', "
        f"{theme!r});\n"
        "for (var i = 0; i < 10; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({box: __harness.html('figuresBox') || ''});",
        routes=_four_series_routes(), search=f"?project={_PID}")

    paths = re.findall(r"<path\b[^>]*>", out["box"])
    curves = [p for p in paths if 'stroke-width="2.4"' in p]
    assert len(curves) == 4, (
        f"{len(curves)} curve path(s) drawn for four series:\n{out['box'][:900]}")

    strokes = [re.search(r'stroke="([^"]+)"', p).group(1) for p in curves]
    dashes = [(re.search(r'stroke-dasharray="([^"]+)"', p) or [None, ""])[1]
              for p in curves]

    assert strokes[3] == strokes[0], (
        f"the fourth series has its own hue ({strokes[3]}); the ramp is three "
        f"and a fourth token has to re-pass the CVD gate first")
    assert "--c4" not in " ".join(strokes), (
        f"`--c4` is being used as a categorical series: {strokes}")
    assert dashes[0] == "", f"the first series is dashed: {dashes}"
    assert dashes[3], (
        f"the fourth series reuses the first series' hue and carries no dash, "
        f"so the two are byte-identical strokes: {strokes} / {dashes}")
    assert len(set(zip(strokes, dashes))) == 4, (
        f"two of four series render identically: {list(zip(strokes, dashes))}")
    with capsys.disabled():
        print(f"\n  [{theme}] " + " · ".join(
            f"{n}={s}{'/' + d if d else ''}"
            for n, s, d in zip(_FOUR, strokes, dashes)))


def test_the_journal_face_separates_four_series_without_hue():
    """The same four series in journal view: one ink, four distinct dashes."""
    if not PH.available():
        pytest.skip("no JS engine on this machine")
    bundle = _four_series_bundle()
    out = PH.run(
        "for (var i = 0; i < 10; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__harness.dispatch('click', __harness.target("
        "{'data-journal': 'roc', 'data-journal-on': '1'}));\n"
        "for (var q = 0; q < 6; q++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({box: __harness.html('figuresBox') || ''});",
        routes=_four_series_routes(), search=f"?project={_PID}")

    curves = [p for p in re.findall(r"<path\b[^>]*>", out["box"])
              if 'stroke-width="1.3"' in p]
    assert len(curves) == 4, f"{len(curves)} journal curve(s):{out['box'][:800]}"
    inks = {re.search(r'stroke="([^"]+)"', p).group(1) for p in curves}
    assert inks == {"#111111"}, f"journal view is using more than one ink: {inks}"
    dashes = [(re.search(r'stroke-dasharray="([^"]+)"', p) or [None, ""])[1]
              for p in curves]
    assert len(set(dashes)) == 4, (
        f"four curves in one ink share {len(set(dashes))} dash pattern(s): "
        f"{dashes} — series a reader cannot tell apart are one series drawn "
        f"four times")

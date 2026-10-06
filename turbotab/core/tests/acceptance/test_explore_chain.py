"""EXPLORE · the registry and the chain test (BLUEPRINT §13: "A chain test asserts that every implied
consequence appears in the participant flow, the lineage and the methods sentence").

Two runs of the real stage graph (``graph_runner.GraphRun``, as the workers run it):

* **A · prediction.** A yes/no outcome with a curved predictor, a near-constant one, sex and age, a
  site grouping and a stratified cluster survey design answered as the surveyed population; 20% held
  out; the outcome's relationship with ``x2`` viewed, then ``x2`` bent by hand; the levers as in-fold
  rules (splines by Harrell's rule, the variance filter, balanced weights with recalibration); in-fold
  screening; decision support with sex and age as subgroups; shrinkage as model updating.
* **B · inference.** A numeric outcome with blanks, multiple imputation, a declared exposure and the
  selection sensitivity analysis.

Every relation the package's contracts declare is mapped to the test that asserts it fires
(:data:`RELATION_TESTS`); a relation no test exercises fails
:func:`test_every_relation_the_contracts_declare_is_asserted`.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from turbotab.core import contracts as C
from turbotab.core import decisions as d
from turbotab.core.tests.graph_runner import GraphRun

EXPLORE = ("explore", "spline_rule", "inner_cv_form", "variance_filter", "imbalance_correction",
           "variable_selection", "intended_use", "design_based_cv")

# relation id → the test (module:function) that asserts it fires
RELATION_TESTS = {
    "outcome_views_recorded": "test_explore_1_stack:test_1_outcome_views_are_recorded_and_disclosed_in_the_methods_text",
    "levers_in_fold_first": "test_explore_1_stack:test_2_every_lever_is_offered_first_as_an_in_fold_rule",
    "by_hand_outside_corrected": "test_explore_chain:test_chain_a_prediction",
    "holdout_covers_hand_levers": "test_explore_chain:test_chain_a_prediction",
    "quality_across_groups": "test_explore_chain:test_chain_a_prediction",
    "spline_rule_in_fold": "test_explore_chain:test_chain_a_prediction",
    "form_chosen_in_fold": "test_explore_1_stack:test_2_the_inner_cv_form_choice_matches_an_independent_loop_on_the_training_fold",
    "filter_before_selection": "test_explore_chain:test_chain_a_prediction",
    "correction_recalibrated": "test_explore_chain:test_chain_a_prediction",
    "selection_outside_refused": "test_explore_3_selection:test_3_selection_outside_the_resampling_is_refused_with_the_exit_run_it_in_fold",
    "inclusion_frequencies": "test_explore_chain:test_chain_a_prediction",
    "selection_only_sensitivity": "test_explore_3_selection:test_7_under_inference_selection_is_only_a_labeled_sensitivity_analysis",
    "selection_pooled_wald": "test_explore_chain:test_chain_b_inference",
    "decision_support_curve": "test_explore_chain:test_chain_a_prediction",
    "threshold_chosen_in_fold": "test_explore_chain:test_chain_a_prediction",
    "subgroups_scored": "test_explore_chain:test_chain_a_prediction",
    "shrinkage_offered": "test_explore_chain:test_chain_a_prediction",
    "design_based_scores": "test_explore_chain:test_chain_a_prediction",
    "no_cv_score_under_inference": "test_explore_8_design_cv:test_8_under_inference_no_cross_validated_score_is_shown",
    # EXPLORE repair
    "stepwise_needs_rows": "test_explore_repair:test_3_stepwise_is_refused_exactly_where_the_smallest_fold_cannot_fit_every_column",
    "spline_rule_counted_by_riley": "test_explore_repair:test_5_rileys_minimum_counts_the_spline_rules_columns_among_the_candidate_parameters",
    "inner_cv_counted_by_riley": "test_explore_repair:test_5_rileys_minimum_counts_the_spline_rules_columns_among_the_candidate_parameters",
    "subgroups_read_the_ledger": "test_explore_repair:test_4_subgroups_are_grouped_by_the_settled_reading_never_by_the_count_of_values",
}


def test_every_explore_method_enters_through_the_one_registry():
    """BLUEPRINT §13: slot, data scope, needs, routing (question; options labeled customary and sound
    for each purpose, a rung for each), storyboard, sentence and relations, in one registry; a
    conflict is refused or blocked and recorded, never silent; the run order agrees with every
    *precedes* relation."""
    registry = C.contracts()
    assert {k for k, c in registry.items() if c.package == "EXPLORE"} == set(EXPLORE)
    for key in EXPLORE:
        c = registry[key]
        assert c.slot in C.SLOTS and c.scope in C.SCOPES, key
        assert c.needs and c.question and c.storyboard and c.options and c.sentence, key
        for o in c.options:
            assert o.label and o.customary, (key, o.key)
            for purpose in C.PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] in C.RUNGS, (key, o.key, purpose)
        for r in c.relations:
            assert r.id, (key, r)
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
            if r.kind == "conflicts":
                assert r.rung in ("refused", "block_and_record") and r.exits, (key, r.name)
        module, name = str(c.sentence).split(":")
        assert callable(getattr(importlib.import_module(module), name)), c.sentence
    C.run_order(list(registry))
    assert C.run_order(["variable_selection", "variance_filter", "spline_rule"]) == [
        "spline_rule", "variance_filter", "variable_selection"]


def test_every_relation_the_contracts_declare_is_asserted():
    declared = {r.id for key in EXPLORE for r in C.contract(key).relations}
    assert declared == set(RELATION_TESTS)
    for where in RELATION_TESTS.values():
        module, name = where.split(":")
        found = importlib.import_module(f"turbotab.core.tests.acceptance.{module}")
        assert callable(getattr(found, name)), where


def test_the_imbalance_corrections_scope_is_the_outcome_models():
    """The correction reweights or redraws by the outcome's classes and recalibrates on them:
    ``contracts.observed_scope`` says ``model``, as its contract declares."""
    from sklearn.linear_model import LogisticRegression

    from turbotab.core.methods.levers import ImbalanceCorrected

    rng = np.random.default_rng(31)
    n = 200
    frame = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a1", "a2"])
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + frame["a1"])))).astype(int)

    def fit_transform(f, r, yy):
        m = ImbalanceCorrected(LogisticRegression(), "weights", seed=0).fit(f, yy)
        return m.predict_proba(f)[:, 1]

    assert C.observed_scope(fit_transform, frame, np.zeros(n, bool), y, 4) == "model" == \
        C.scope_of("imbalance_correction")


# ── A · prediction ───────────────────────────────────────────────────────────


def _chain_a_table(seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(8):  # strata
        for j in range(4):  # PSUs
            for _ in range(15):
                x1, x2, x3 = rng.normal(size=3)
                sex = int(rng.integers(1, 3))
                age = float(rng.uniform(20, 80))
                lp = -1.2 + 0.9 * x1 - 0.7 * x2 ** 2 + 0.3 * x3 + 0.3 * (sex == 2)
                rows.append({"x1": x1, "x2": x2, "x3": x3, "sex": sex, "age": age,
                             "flat": 0.0 if rng.random() < 0.98 else 1.0,
                             "site": f"site_{int(rng.integers(0, 4))}",
                             "stratum": h + 1, "psu": j + 1, "wt": float(rng.uniform(800, 3000)),
                             "y": "yes" if rng.random() < 1 / (1 + np.exp(-lp)) else "no"})
    frame = pd.DataFrame(rows)
    frame.insert(0, "pid", np.arange(len(frame)))
    return frame


@pytest.fixture(scope="module")
def chain_a(tmp_path_factory):
    folder = tmp_path_factory.mktemp("chain_a")
    frame = _chain_a_table()
    frame.to_csv(folder / "t.csv", index=False)
    run = GraphRun(folder / "t.csv", folder / "p")
    roles = {"pid": "identifier", "x1": "covariate", "x2": "covariate", "x3": "covariate",
             "sex": "covariate", "age": "covariate", "flat": "covariate", "site": "design",
             "stratum": "design", "psu": "design", "wt": "design"}
    view = d.OutcomeViewSpec(view="relationship", column="x2", target="y", rows="training",
                             n_rows=384, levers={"form": "linear", "role": "covariate",
                                                 "kept": "yes"})
    state = d.ProjectState(
        lens=["clinical"], target="y", task="binary", purpose="prediction", roles=roles,
        event="yes", missing="complete_case", categorical=["sex"],
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.2, seed=6, folds=5), models=["linear"],
        clusters=d.ClusterSpec(column="site"),
        survey=d.SurveySpec(estimand="population", weight="wt", strata="stratum", psu="psu"),
        outcome_views={"relationship:x2": view},
        exposure_forms={"x2": d.ExposureFormSpec(form="spline", knots=4)},
        levers=d.LeverSpec(forms="rule", variance_filter="near_zero", imbalance="weights"),
        selection=d.SelectionSpec(method="screening", keep=4, pre_selected="no"),
        intended_use=d.IntendedUseSpec(use="decision_support", subgroups=["sex", "age"]),
        updating=d.UpdatingSpec(method="shrinkage"))
    out = run.run(state, upto=["explore", "evaluation"])
    yield frame, state, out
    run.close()


def test_chain_a_prediction(chain_a):
    """Chain A's order and every relation it touches: the levers sit in each family's pipeline in
    MODELING_SEQUENCE §1.1's order (construction, filters, selection, the corrected model); Explore
    read the training rows, recorded the view and disclosed the lever set by hand after it as
    outside the corrected score, covered by the holdout; the evaluation counted inclusion
    frequencies, drew the decision curve with thresholds chosen in each fold, scored the subgroups
    with intervals, shrank the regression, ran internal–external CV by site and design-based CV."""
    from turbotab.core.methods.levers import ImbalanceCorrected

    frame, state, out = chain_a
    steps = out["design"].objects["pipelines"]["linear"].steps
    names = [n for n, _ in steps]
    assert names.index("form") < names.index("lever_forms") < names.index("lever_filter") \
        < names.index("select") < names.index("model")
    assert isinstance(steps[-1][1], ImbalanceCorrected)
    explore = out["explore"].data
    assert explore["rows"] == "training" and explore["n_holdout"] > 0
    assert next(f for f in explore["findings"] if f["id"] == "explore::relationship::x2")["viewed"]
    assert [h["column"] for h in explore["hand_levers"]] == ["x2"]
    assert explore["hand_levers"][0]["sentence"].endswith(
        "its optimism is not in the corrected score; the held-out rows, never viewed, cover it.")
    assert {"explore::quality::sex", "explore::quality::age"} <= {f["id"] for f in explore["findings"]}
    ev = out["evaluation"].data
    assert ev["inclusion"]["folds"] == 20
    curve = ev["decision_curve"]
    assert len(curve["in_fold"]["folds"]) == 5 and curve["in_fold"]["declared"] is False
    assert {s["column"] for s in ev["subgroups"]} == {"sex", "age"}
    assert all("auc" in g["scores"] for s in ev["subgroups"] for g in s["groups"] if g["scores"])
    assert ev["shrinkage"]["factor"] > 0 and ev["shrinkage"]["intercept"] is not None
    assert {x["family"] for x in ev["internal_external"]} == {"linear", "spline_benchmark"}
    # internal–external: each site held out in turn, its training rows all scored (Collins et al.
    # 2024, Box 4), the random-effects summary over the sites (MS6's ``internal_external``)
    train = frame.loc[out["fit"].objects["comparison"]["train_ids"]]
    linear = next(x for x in ev["internal_external"] if x["family"] == "linear")
    assert {c["cluster"]: c["n"] for c in linear["clusters"]} == train["site"].value_counts().to_dict()
    assert linear["cluster"] == "site" and linear["pooled"]["k"] == 4
    assert ev["design_based"]["label"] == "design-based cross-validation"
    assert ev["design_based"]["folds"] == 4
    assert any(s.startswith("The corrected score does not include the optimism of the lever set "
                            "by hand after an outcome view: ") for s in ev["sentences"])
    # every relation the chain touches fires
    choices = {"explore": "by_hand", "spline_rule": "rule", "variance_filter": "near_zero",
               "variable_selection": "screening", "imbalance_correction": "weights",
               "intended_use": "decision_support", "design_based_cv": "population"}
    consequences = ["outcome_view_recorded", "in_fold_rule_first", "outside_corrected_score",
                    "holdout_covers", "quality_by_group", "in_fold_rule", "inclusion_frequencies",
                    "recalibration", "decision_curve", "threshold_in_fold", "subgroup_performance",
                    "model_updating", "design_folds"]
    fired = {f.relation.name for f in C.fired(choices, "prediction", consequences)}
    assert fired == {"outcome_views_recorded", "levers_in_fold_first", "by_hand_outside_corrected",
                     "holdout_covers_hand_levers", "quality_across_groups", "spline_rule_in_fold",
                     "filter_before_selection", "inclusion_frequencies", "correction_recalibrated",
                     "decision_support_curve", "threshold_chosen_in_fold", "subgroups_scored",
                     "shrinkage_offered", "design_based_scores"}
    assert C.paragraph({"spline_rule": "rule", "variance_filter": "near_zero",
                        "variable_selection": "screening", "imbalance_correction": "weights"},
                       {}, "prediction") == (
        "Within each training fold, splines with knots by Harrell's rule, the variance filter, the "
        "selection and the imbalance correction with its recalibration were fitted and applied to "
        "the held-out fold.")


def test_chain_a_the_held_out_rows_are_never_read_by_explore_or_the_evaluation(chain_a):
    """The seal (M2_CONTRACT §3): everything Explore and the evaluation show comes from the training
    rows; the comparison substrate the evaluation refits on holds no held-out row."""
    frame, state, out = chain_a
    assignment = out["split"].frames["assignment"]
    held = set(assignment.loc[assignment["partition"].astype(str).str.lower() == "holdout",
                              "row_id"].astype(int))
    assert held and not held & set(out["fit"].objects["comparison"]["train_ids"].tolist())
    assert out["explore"].data["n_rows"] == len(frame) - len(held)


# ── B · inference ────────────────────────────────────────────────────────────


def test_chain_b_inference(tmp_path):
    """Chain B: under inference with multiple imputation the evaluation stage shows no
    cross-validated score and runs the declared selection sensitivity analysis: backward elimination
    by Wald tests pooled across the 20 imputed copies by Rubin's rules, the declared exposure kept,
    said as a labeled sensitivity analysis beside the declared model."""
    rng = np.random.default_rng(51)
    n = 400
    x1, x2, x3, x4 = (rng.normal(size=n) for _ in range(4))
    y = 0.6 * x1 + 0.5 * x2 + rng.normal(0, 1, n)
    x2 = np.where(rng.random(n) < 0.15, np.nan, x2)
    frame = pd.DataFrame({"pid": np.arange(n), "x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    frame.to_csv(tmp_path / "t.csv", index=False)
    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "x3": "covariate",
             "x4": "covariate"}
    state = d.ProjectState(
        lens=["clinical"], target="y", task="regression", purpose="inference", roles=roles,
        missing=d.MissingSpec(strategy="multiple_imputation", m=20),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.0, seed=2, folds=5), models=["linear"],
        estimand=d.EstimandSpec(exposure="x1", effect="total", measure="mean_difference"),
        selection=d.SelectionSpec(method="stepwise", sensitivity=True))
    out = run.run(state, upto=["evaluation"])
    ev = out["evaluation"].data
    assert ev["scores_shown"] is False and ev["benchmark"] is None
    found = ev["estimates"]
    assert found["copies"] == 20 and found["forced"] == ["x1"]
    assert "x1" in found["kept"] and "x2" in found["kept"]
    dropped = [s["dropped"] for s in found["path"] if s["dropped"]]
    assert set(dropped) <= {"x3", "x4"}
    from turbotab.core.voice import listing

    assert found["sentence"] == (
        "As a labeled sensitivity analysis (not the reported model), backward elimination at "
        "α = 0.157 with Wald tests pooled across the imputed copies by Rubin's rules (Wood, White & "
        f"Royston 2008), the exposure kept, removed {listing(dropped) if dropped else 'no covariate'}; "
        "the declared model is the reported one.")
    from turbotab.core.estimand import ESTIMATE_STAGES
    from turbotab.core.plan_lock import shows_estimates

    assert "evaluation" in ESTIMATE_STAGES and shows_estimates("evaluation", ev)
    fired = {f.relation.name for f in C.fired({"variable_selection": "stepwise"}, "inference",
                                              ["rubins_rules_wald"])}
    assert fired == {"selection_pooled_wald"}
    run.close()

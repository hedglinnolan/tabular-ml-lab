"""FORM · the method contracts (BLUEPRINT §13) and the chain: every method the package adds enters
through a contract in the one registry, declares every part §13 asks for, holds the scope the
lockbox §06 test observes, and every relation it declares is asserted to fire by a named test
(:data:`RELATION_TESTS`; MODELING_SEQUENCE §2, §6)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from turbotab.core import contracts as C
from turbotab.core.methods import exposure_form as ef

HERE = "turbotab/core/tests/acceptance/"
ONE = HERE + "test_form_1_order_and_invalidation::"
TWO = HERE + "test_form_2_rcs_rule_and_r::"
THREE = HERE + "test_form_3_5_tests_and_quintiles::"
FOUR = HERE + "test_form_4_confounders::"
SIX = HERE + "test_form_6_mass_at_zero::"
SEVEN = HERE + "test_form_7_modification::"
EIGHT = HERE + "test_form_8_energy_labels::"
REPAIR = HERE + "test_form_repair_1::"
RELATION_TESTS = {
    ("exposure_transform", "transform-invalidates-form"):
        ONE + "test_1_a_transform_of_the_exposure_leaves_its_form_stale_and_reasks_it",
    ("exposure_transform", "transform-precedes-form"):
        ONE + "test_1_the_form_question_comes_after_the_domain_transforms_and_before_the_families",
    ("functional_form", "k-by-rule"):
        TWO + "test_2_a_spline_declared_without_k_records_the_rule_and_its_sentence_states_it",
    ("functional_form", "no-silent-linear-refit"):
        THREE + "test_3_a_non_significant_nonlinearity_test_keeps_every_spline_term",
    ("functional_form", "quintiles-beside"):
        THREE + "test_5_quintiles_are_produced_beside_the_spline_with_boundaries_and_reference",
    ("functional_form", "form-mi-d1"):
        TWO + "test_2_under_multiple_imputation_the_tests_are_pooled_by_d1_as_mitml_pools_them",
    ("functional_form", "optimal-cut-blocked"):
        FOUR + "test_4_a_data_derived_cut_point_is_blocked_under_inference_and_found_as_by_hand",
    ("functional_form", "coarse-confounder-blocked"):
        FOUR + "test_4_a_confounder_in_three_or_fewer_groups_is_blocked_and_recorded",
    ("functional_form", "mass-at-zero"):
        SIX + "test_6_the_card_sees_the_mass_at_zero_and_leads_with_non_consumers_apart",
    ("functional_form", "consumers-only-domain"):
        SIX + "test_6_the_consumers_only_domain_is_an_estimand_change_in_the_flow_and_the_caption",
    ("functional_form", "residual-curve-label"):
        ONE + "test_8_a_spline_on_a_residual_is_labeled_and_offers_the_substitution_route",
    ("functional_form", "log-or-spline-substitution"):
        EIGHT + "test_8_a_log_on_an_energy_component_labels_the_curve_a_k_specific_average",
    ("functional_form", "share-reallocation-refused"):
        EIGHT + "test_8_a_share_reallocation_is_refused_with_the_ilr_reason",
    **{(key, "single-reference"):
       SEVEN + "test_7_the_reported_effects_reri_and_ratio_of_ratios_agree_with_r_glm_by_hand"
       for key in ("effect_modification", "interaction")},
    **{(key, "both-scales"):
       SEVEN + "test_7_the_reported_effects_reri_and_ratio_of_ratios_agree_with_r_glm_by_hand"
       for key in ("effect_modification", "interaction")},
    **{(key, "post-hoc-labeled"):
       SEVEN + "test_7_a_modifier_declared_after_the_estimates_is_suggested_by_data_inspection"
       for key in ("effect_modification", "interaction")},
    **{(key, "modification-mi-compatible"):
       SEVEN + "test_7_under_multiple_imputation_the_products_are_in_the_imputation_model"
       for key in ("effect_modification", "interaction")},
    **{(key, "modification-in-plan"):
       HERE + "test_form_contracts::test_a_declared_modifier_is_part_of_the_plan_the_lock_holds"
       for key in ("effect_modification", "interaction")},
    ("effect_modification", "modification-own-set"):
        SEVEN + "test_7_the_sentence_and_the_family_are_written_as_declared",
    ("interaction", "interaction-reasks-adjustment"):
        SEVEN + "test_7_an_interaction_asks_the_adjustment_set_again_for_the_second_exposure",
    # the repair round (``test_form_repair_1``)
    ("functional_form", "rows-rederive-k"):
        REPAIR + "test_r1_through_the_server_k_and_its_n_follow_the_rows_the_fit_sees",
    ("functional_form", "codes-take-no-form"):
        REPAIR + "test_r3_through_the_server_the_reading_is_asked_first_and_codes_enter_as_"
                 "indicators",
    ("functional_form", "separation-plr"):
        REPAIR + "test_r5_under_separation_the_test_of_association_is_a_penalized_likelihood_"
                 "ratio_test",
    **{(key, "stratum-estimable"):
       REPAIR + "test_r4_such_a_modifier_is_refused_at_declaration_with_its_way_forward"
       for key in ("effect_modification", "interaction")},
    **{(key, "withdrawn-still-counted"):
       REPAIR + "test_r6_a_withdrawal_after_the_estimates_keeps_the_test_in_the_family"
       for key in ("effect_modification", "interaction")},
    **{(key, "separation-profile"):
       REPAIR + "test_r5_under_separation_the_modification_reports_profile_intervals_and_a_lr_"
                "test" for key in ("effect_modification", "interaction")},
}
KEYS = ("exposure_transform", "functional_form", "effect_modification", "interaction")


def test_every_form_method_enters_through_a_contract_declaring_section_13():
    registry = C.contracts()
    assert {k for k, c in registry.items() if c.package == "FORM"} == set(KEYS)
    for key in KEYS:
        c = registry[key]
        assert c.slot in C.SLOTS and c.scope in C.SCOPES, key
        assert c.needs and c.question and c.storyboard and c.options and c.sentence, key
        for o in c.options:
            assert o.label and o.customary
            for purpose in C.PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] in C.RUNGS, (key, o.key, purpose)
        module, name = str(c.sentence).split(":")
        assert callable(getattr(importlib.import_module(module), name))
        for r in c.relations:
            if r.kind == "conflicts":
                assert r.rung in ("refused", "block_and_record") and r.exits, (key, r.name)
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert hasattr(importlib.import_module(module), name), r.enforced_by
    order = C.run_order(list(KEYS))
    assert order.index("exposure_transform") < order.index("functional_form")
    rungs = {o.key: o.rung for o in registry["functional_form"].options}
    assert rungs["optimal"] == {"inference": "block_and_record", "prediction": "rank_lower"}
    assert rungs["quintiles"]["inference"] == "rank_lower"


def test_every_relation_the_contracts_declare_is_asserted():
    """Each relation names the test that asserts it fires; a relation no test exercises fails."""
    registry = C.contracts()
    declared = {(key, r.name) for key in KEYS for r in registry[key].relations}
    assert declared == set(RELATION_TESTS), declared ^ set(RELATION_TESTS)
    for path in set(RELATION_TESTS.values()):
        module, name = path.split("::")
        found = importlib.import_module(module.replace("/", "."))
        assert callable(getattr(found, name)), path


def test_the_forms_scope_is_the_one_the_lockbox_test_observes():
    """§13: the scope is not taken on trust. Knots placed on study rows: training fold; a
    data-derived cut point also reads the outcome: model."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.gamma(3.0, 2.0, 120)})
    x = frame["x"].to_numpy()
    # the outcome steps up at x's 30th percentile, so the search finds a cut there
    y = np.where(x > np.quantile(x, 0.3), 2.0, 0.0) + rng.normal(0, 0.3, 120)
    reference = np.zeros(120, dtype=bool)

    def spline(f, ref, yy):
        return ef.ExposureForms({"x": {"form": "spline", "knots": 4}}).fit(f).transform(f)

    def optimal(f, ref, yy):
        return ef.ExposureForms({"x": {"form": "optimal"}}).fit(f, yy).transform(f)

    assert C.observed_scope(spline, frame, reference, y, 5) == C.contract(
        "functional_form").scope == "training_fold"
    # watched: the row just above the cut the search finds, which the outcome's shuffle moves
    cut = ef.ExposureForms({"x": {"form": "optimal"}}).fit(frame, y).cuts_["x"][0]
    above = np.flatnonzero(x > cut)
    row = int(above[np.argmin(x[above] - cut)])
    assert C.observed_scope(optimal, frame, reference, y, row) == C.contract(
        "functional_form").scope_of("optimal") == "model"


def test_a_declared_modifier_is_part_of_the_plan_the_lock_holds():
    from turbotab.core import plan_lock
    from turbotab.core.estimand import ESTIMATE_STAGES

    assert "modification" in ESTIMATE_STAGES
    assert {"modifications", "exposure_forms", "form_domains"} <= set(plan_lock.plan_slots())


@pytest.mark.parametrize("key", ["form", "modification"])
def test_the_new_questions_carry_their_teaching_and_their_names(key):
    from turbotab.core import teaching
    from turbotab.core.voice import question_name

    entry = teaching.entry(key)
    assert entry.question and entry.one_liner and entry.drawer and entry.drawer.sections
    assert question_name(key) in ("the functional-form question", "the effect-modifier question")

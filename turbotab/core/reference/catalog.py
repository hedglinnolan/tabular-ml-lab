"""Whose packet reviews each method in full, and the methods v2 offers without a contract (the gaps).

The method contracts (BLUEPRINT §13) and the model families (M1_CONTRACT §7) do not say which of
the five lenses they belong to: a contract declares where it runs and what it may learn from, never
whose field it is, and no contract is gated on a lens by the registry. This module says, once for
every registered contract and family, which lenses' methodologists review it in full (``shared``:
every lens's). It decides how much of a method a packet shows, never whether it is listed: every
packet lists every registered contract and family (in full, in the shared table, or among the
methods another lens reviews in full), because the app reaches a method through the data, not the
lens, and a packet that shortened the list would hide what the app offers. The reference test fails
when a contract or a family is registered without an entry here, or an entry names one that is not
registered, or a packet leaves one out.

The gaps are the methods v2 offers that no contract declares: those V2_DEFINITION_OF_DONE §2 lists,
and every decision the Router records that writes a methods sentence but belongs to no contract.
Each names where v2 offers it, the lenses it serves, the decision kinds that record it, the code
that implements it, and, where its options already carry the customary and sound labels outside
the registry (``turbotab.core.custom_sound``), which question holds them. Every decision kind the
decision log accepts is accounted for: a contract records it (its ``decision``, or
:data:`UNDECLARED_DECISIONS` where the contract does not name it), a gap records it, or it records
no method at all (:data:`NOT_METHODS`, each with why). The test fails on a kind in none of these, so
a new decision cannot ship without the reference saying what it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LENSES: tuple[str, ...] = ("dietary", "clinical", "metabolomics", "genomics", "survey")
SHARED = "shared"

LENS_TITLES: dict[str, str] = {
    "dietary": "Dietary assessment",
    "clinical": "Clinical",
    "metabolomics": "Metabolomics",
    "genomics": "Genomics",
    "survey": "Survey instruments",
}

# Every registered method contract: the lenses whose packet reviews it in full, or (SHARED,).
# "survey" is the survey-instrument lens (scales and items). A complex survey design (weights,
# strata, PSUs: NHANES) is gated on its columns and the purpose, never on a lens
# (``turbotab.core.survey``), so every packet lists it; the dietary, clinical and survey
# methodologists review it in full.
CONTRACT_LENSES: dict[str, tuple[str, ...]] = {
    # data in (V2 definition of done §1)
    "join_files": (SHARED,),
    "import_codebook": (SHARED,),
    # metabolomics: the QC chain before the seal, then the in-fold chain (MODELING_SEQUENCE §6.3)
    "qc_detection_filter": ("metabolomics",),
    "qc_rlsc": ("metabolomics",),
    "qc_rsd_filter": ("metabolomics",),
    "qc_pqn": ("metabolomics",),
    "qc_rows_leave": ("metabolomics",),
    "d_ratio_filter": ("metabolomics",),
    "detection_limit": ("metabolomics",),
    "censored_below_detection": ("metabolomics", "clinical"),
    # metabolomics and genomics: the omics chain fit in-fold
    "omics_normalization": ("metabolomics", "genomics"),
    "log_transform": ("metabolomics", "genomics"),
    "autoscaling": ("metabolomics", "genomics"),
    "screen": ("metabolomics", "genomics"),
    "batch": ("metabolomics", "genomics"),
    # survey instruments
    "scales": ("survey",),
    # dietary
    "nci_usual_intake": ("dietary",),
    "regression_calibration": ("dietary",),
    "multiclass_substitution": ("dietary",),
    "survey_substitution": ("dietary",),
    # a complex survey design (NHANES): reviewed in full by the dietary, clinical and survey lenses
    "survey_population": ("dietary", "clinical", "survey"),
    "survey_linear": ("dietary", "clinical", "survey"),
    "survey_cox": ("dietary", "clinical", "survey"),
    "survey_ordinal": ("dietary", "clinical", "survey"),
    "design_based_cv": ("dietary", "clinical", "survey"),
    "copies_not_repeats": ("dietary", "clinical"),
    "imputed_copies_pooled": ("dietary", "clinical"),
    # the inference machinery every lens shares (MODELING_SEQUENCE §1 rows 2–12)
    "effect_measure": (SHARED,),
    "g_computation": (SHARED,),
    "adjustment_guesses": (SHARED,),
    "grouping_by_structure": (SHARED,),
    "exposure_transform": (SHARED,),
    "functional_form": (SHARED,),
    "effect_modification": (SHARED,),
    "interaction": (SHARED,),
    "multiple_imputation_compatible": (SHARED,),
    "multiple_imputation_passive": (SHARED,),
    "multiple_imputation_single_level": (SHARED,),
    "model_sequence": (SHARED,),
    "diagnostics": (SHARED,),
    "unmeasured_confounding": (SHARED,),
    "evalue_sd": (SHARED,),
    "exposure_family": (SHARED,),
    "multiplicity": (SHARED,),
    "plan_export": (SHARED,),
    "fit_statistics_withheld": (SHARED,),
    # the causal lane (shortest leash)
    "dml_plr": (SHARED,),
    "dml_irm": (SHARED,),
    "tmle": (SHARED,),
    "pds_lasso": (SHARED,),
    "time_varying": (SHARED,),
    # the prediction machinery every lens shares (MODELING_SEQUENCE §1, prediction)
    "explore": (SHARED,),
    "spline_rule": (SHARED,),
    "inner_cv_form": (SHARED,),
    "variance_filter": (SHARED,),
    "imbalance_correction": (SHARED,),
    "variable_selection": (SHARED,),
    "intended_use": (SHARED,),
    "proper_primary": (SHARED,),
    "comparison_substrate": (SHARED,),
    "bbc_cv": (SHARED,),
    "bootstrap_optimism": (SHARED,),
    "horizon_calibration": (SHARED,),
    "nested_cv_interval": (SHARED,),
    "explain": (SHARED,),
    # the export ends every journey
    "manuscript_export": (SHARED,),
}

# Every registered model family: the lenses whose packet reviews it in full, or (SHARED,). No
# family is gated on a lens: the shelf ranks every family that can model the outcome by its own
# assessment (the feature-wise family scores an omics lens higher; the screened elastic net scores
# fewer features than rows lower), so every packet lists every family.
FAMILY_LENSES: dict[str, tuple[str, ...]] = {
    "linear": (SHARED,),
    "elastic_net": (SHARED,),
    "boosted_trees": (SHARED,),
    "featurewise": ("metabolomics", "genomics"),
    "proportional_odds": (SHARED,),
    "mixed": (SHARED,),
    "gee": (SHARED,),
    "cox": (SHARED,),
    # registered by the omics chain (``turbotab.core.methods.omics``): p ≫ n prediction
    "screened_elastic_net": ("metabolomics", "genomics"),
}


# Decision kinds that record a contract's method although the contract names another kind (or none)
# as its ``decision``: kind → contract.
UNDECLARED_DECISIONS: dict[str, str] = {
    "set_batch": "batch",
    "set_multiplicity": "multiplicity",
    "set_forms": "functional_form",  # the form question's one tap: a set_exposure_form per column
}

# The contracts whose method no Router question, proposals card, finding or control asks or states:
# the app records one only when its decision is posted to the API (the reference journeys declare
# each, "declared, not asked"), so without that its default is not applied at all. Each says what a
# user of the app gets instead and which V2_DEFINITION_OF_DONE §2 row owns asking it. The reference
# test checks the premise against the Router, the proposals and the frontend.
NOT_ASKED: dict[str, str] = {
    "batch": "No Router question, proposals card, finding or control posts `set_batch` (the one "
             "exit that does, on the confounding refusal, reads the column as not a batch). From "
             "the app a batch column is whatever its role says, a covariate or left out, and "
             "ComBat is never fit; the reference journeys post `set_batch` themselves. Asking it "
             "is V2_DEFINITION_OF_DONE §2's “Genomics, extended: batch correction (ComBat) fit "
             "in-fold” row.",
    "scales": "No Router question, proposals card or control posts `set_scales`: from the app "
              "each item enters the models on its own, and no score, reliability or correction "
              "is computed; the reference journeys post `set_scales` themselves. Asking it is "
              "V2_DEFINITION_OF_DONE §2's “Survey instruments: scale scoring with reliability” "
              "row.",
}

# Decision kinds that record no method: the research question, the readings of what the data hold,
# and the record's own housekeeping. Each says why it is not a method.
NOT_METHODS: dict[str, str] = {
    "set_lens": "declares the lenses the table is read through: whose field the question is",
    "set_target": "names the outcome: the research question",
    "set_event": "names the outcome's event level: the research question",
    "set_task": "states what kind of outcome it is (a number, yes/no, ordered levels, classes, a "
                "time to event): what is modeled, read from the data and confirmed",
    "set_purpose": "declares inference or prediction: the research question",
    "set_roles": "gives each column its role in the research question",
    "confirm_role": "confirms a role the app proposed below high confidence: a reading",
    "confirm_reading": "confirms what a column holds (BLUEPRINT §14): a reading",
    "confirm_readings": "confirms several readings in one block",
    "set_column_unit": "states a column's unit and, for energy, its days: a reading",
    "dismiss_finding": "sets a finding aside, recorded with its reason; nothing is transformed",
    "defer_finding": "defers a finding, recorded; nothing is transformed",
    "revert": "undoes an earlier decision; the record keeps both",
}


@dataclass(frozen=True)
class Gap:
    """A method v2 offers that has no method contract (BLUEPRINT §13)."""

    key: str
    name: str
    row: str  # where v2 offers it: the row of V2_DEFINITION_OF_DONE §2, or the Router's question
    lenses: tuple[str, ...]
    decisions: tuple[str, ...]  # the decision kinds that record it ("" for none: computed)
    modules: tuple[str, ...]  # the code that implements it
    labels: str = ""  # where its options carry the customary and sound labels, if anywhere
    contracted_parts: tuple[str, ...] = ()  # contracts that cover part of it
    note: str = ""


GAPS: tuple[Gap, ...] = (
    Gap("energy_adjustment", "Energy adjustment (the five models and all components)",
        "Dietary", ("dietary",), ("set_energy_adjustment",),
        ("turbotab.core.methods.energy", "turbotab.core.methods.percent_energy"),
        labels="custom_sound:energy_adjustment",
        contracted_parts=("exposure_transform",),
        note="Its options carry both labels (methods.energy.METHOD_TABLE, ranked by RANKING), "
             "but its slot, scope, storyboard and relations are declared nowhere as a contract; "
             "exposure_transform names it only as one of the exposure's transforms."),
    Gap("implausible_intake", "Implausible intake: fixed kcal rules and the Goldberg screen",
        "Dietary", ("dietary",), ("set_exclusions",),
        ("turbotab.core.stages.proposals", "turbotab.core.methods.misreporting"),
        labels="custom_sound:exclusions"),
    Gap("secondary_analyses", "Declared secondary analyses (the exclusions' sensitivity view)",
        "Inference (shared); Dietary", (SHARED,), ("set_sensitivity",),
        ("turbotab.core.stages.secondary", "turbotab.core.stages.sensitivity")),
    Gap("recall_averaging", "Repeated recalls combined by averaging", "Dietary", ("dietary",),
        ("set_aggregation",), ("turbotab.core.structural", "turbotab.core.stages.working"),
        contracted_parts=("regression_calibration",)),
    Gap("substitution_curves", "Substitution curves with refit bands (one outcome)", "Dietary",
        ("dietary",), ("set_substitution",), ("turbotab.core.methods.substitution",),
        contracted_parts=("multiclass_substitution", "survey_substitution"),
        note="Only the multiclass and survey-weighted variants have contracts."),
    Gap("missing_values", "Missing values: complete cases, a fill in each training fold, "
        "indicators and a missing category", "Prediction (shared); Inference (shared)",
        (SHARED,), ("set_missing",), ("turbotab.core.methods.missing",),
        labels="custom_sound:missing",
        contracted_parts=("multiple_imputation_compatible", "multiple_imputation_passive",
                          "multiple_imputation_single_level", "detection_limit")),
    Gap("split", "The seal and the split (a holdout, k-fold and repeated k-fold, internal–external)",
        "Prediction (shared)", (SHARED,), ("set_split", "open_seal"),
        ("turbotab.core.seal", "turbotab.core.models.validation"), labels="custom_sound:split",
        contracted_parts=("bootstrap_optimism", "comparison_substrate", "bbc_cv",
                          "nested_cv_interval")),
    Gap("in_fold_pipeline", "In-fold preprocessing and nested tuning (the pipeline's fill and "
        "scaling; the penalized families' inner cross-validation)", "Prediction (shared)",
        (SHARED,), ("select_models",),
        ("turbotab.core.models.pipeline", "turbotab.core.models.inner_cv")),
    Gap("delong", "DeLong's interval and test for the AUC", "Prediction (shared)", (SHARED,), (),
        ("turbotab.core.models.performance",)),
    Gap("price_of_explainability", "The price of explainability, measured", "Prediction (shared)",
        (SHARED,), (), ("turbotab.core.models.cost",)),
    Gap("robust_intervals", "Cluster-robust (CR2) and HC3 intervals", "Inference (shared)",
        (SHARED,), ("select_models", "set_clusters"), ("turbotab.core.models.inference",),
        contracted_parts=("grouping_by_structure",)),
    Gap("calibration", "Calibration of a continuous or yes/no outcome's predictions",
        "Prediction (shared); Clinical", (SHARED,), (), ("turbotab.core.models.performance",),
        contracted_parts=("horizon_calibration",),
        note="horizon_calibration covers a time to event and an ordinal or multiclass outcome."),
    Gap("plausibility_repairs", "Plausibility repairs and sentinel codes", "Clinical; Survey",
        ("clinical", "survey"), ("apply_repair",),
        ("turbotab.core.repairs", "turbotab.core.detectors.plausibility",
         "turbotab.core.detectors.codes"),
        contracted_parts=("censored_below_detection",)),
    Gap("time_points", "Time points and the temporal seal", "Clinical", ("clinical",),
        ("set_repeat_kind", "set_temporal"), ("turbotab.core.seal", "turbotab.core.structural")),
    Gap("riley_sample_size", "Riley's minimum sample size", "Clinical", ("clinical",), (),
        ("turbotab.core.models.sample_size",)),
    Gap("orientation", "Orientation (features in rows, turned)", "Metabolomics",
        ("metabolomics", "genomics"), ("set_orientation", "set_feature_table"),
        ("turbotab.core.detectors.orientation",)),
    Gap("model_families", "The model families (linear and logistic, elastic net, boosted trees, "
        "feature-wise, proportional odds, mixed, GEE, Cox, screened elastic net)",
        "Prediction (shared); Inference (shared)", (SHARED,), ("select_models",),
        ("turbotab.core.models.base",),
        note="Declared in the model-family registry (tasks, inductive bias, strengths, cautions, "
             "each family's own assessment), not as contracts: no family declares a question, "
             "options labeled customary and sound, a leash, a storyboard, relations or sources."),
    Gap("architecture_lane", "The architecture lane (the fitted equation, split structure, "
        "shrinkage)", "Explainability", (SHARED,), (), ("turbotab.core.models.explain",),
        contracted_parts=("explain",),
        note="A display on the canvas; the explain contract covers the curves, SHAP and the "
             "interaction ranking."),
    # Decisions the Router records that write a methods sentence (export.methods.SECTION_OF) and
    # belong to no contract.
    Gap("model_updating", "Model updating by uniform shrinkage of the regression's coefficients",
        "Prediction (shared): the evaluation stage's offer (TRIPOD+AI 12f)", (SHARED,),
        ("set_updating",), ("turbotab.core.stages.evaluation", "turbotab.core.models.decision_curve"),
        note="Offered beside the calibration slope: “As model updating, the regression's "
             "coefficients were multiplied by …”."),
    Gap("outcome_scale", "The outcome's scale: the log scale (a ratio of geometric means), an "
        "ordinal outcome's level order, and its unit", "The Router's task question (its follow-ups)",
        (SHARED,), ("set_outcome_scale", "set_outcome_order", "set_outcome_unit"),
        ("turbotab.core.structural", "turbotab.core.interview", "turbotab.core.models.ordinal",
         "turbotab.core.units"),
        note="The log scale changes the estimand (a ratio of geometric means for a difference in "
             "means); the order and the unit are asked, never inferred."),
    Gap("follow_up", "Time-to-event follow-up: a landmark with left truncation, an administrative "
        "horizon, the prediction horizon, and the yes/no outcome over one period",
        "The Router's follow-up question", (SHARED,), ("set_follow_up", "set_censoring"),
        ("turbotab.core.estimand", "turbotab.core.models.survival"),
        contracted_parts=("horizon_calibration",),
        note="horizon_calibration covers scoring at the prediction horizon; who is at risk from "
             "the landmark, and when follow-up ends, are declared nowhere as a contract."),
    Gap("unit_of_analysis", "The unit of analysis for repeated rows (one row per unit, or each "
        "record)", "The Router's grain and unit questions", (SHARED,), ("set_grain", "set_unit"),
        ("turbotab.core.structural", "turbotab.core.stages.working"),
        note="With set_aggregation (recall_averaging) it decides what one row of the analysis is."),
    Gap("reseal", "Re-sealing after the held-out rows were opened", "Prediction (shared): the seal",
        (SHARED,), ("reseal",), ("turbotab.core.seal",),
        note="The scores at the first opening stay the reported result; later held-out scores are "
             "not an independent test."),
    Gap("categorical_codes", "Numbers that are codes for categories, entered as indicators",
        "Inference (shared); Prediction (shared): the code-or-amount reading", (SHARED,),
        ("set_categorical",), ("turbotab.core.readings", "turbotab.core.models.pipeline"),
        note="One indicator per level after the first, never one straight line."),
    Gap("scale_as_exposure", "A declared scale as the estimand's exposure (not offered)",
        "Survey instruments: scale scoring; Inference (shared): a declared exposure", ("survey",),
        ("set_scales", "set_estimand"), ("turbotab.core.estimand", "turbotab.core.stages.scales"),
        contracted_parts=("scales", "effect_measure"),
        note="A limit, not only a missing contract: the estimand card offers the table's columns "
             "(estimand.exposure_candidates), so a scale declared as an exposure enters the model "
             "beside the plan's exposure and its coefficient is listed among the adjustment terms; "
             "its corrected and uncorrected coefficients are the scales stage's and are not in "
             "the export bundle."),
)


def lenses_of_contract(key: str) -> tuple[str, ...]:
    return CONTRACT_LENSES[key]


def lenses_of_family(key: str) -> tuple[str, ...]:
    return FAMILY_LENSES[key]


def serves(lenses: tuple[str, ...], lens: str) -> bool:
    """True when ``lens`` is one of ``lenses`` or they are shared."""
    return lens in lenses or SHARED in lenses


def own(lenses: tuple[str, ...], lens: str) -> bool:
    """True when ``lens`` is named (a lens's own method, not a shared one)."""
    return lens in lenses


def gaps_for(lens: str) -> list[Gap]:
    return [g for g in GAPS if serves(g.lenses, lens)]


def recorded_by(c: Any) -> str | None:
    """The decision kind that records contract ``c``'s method: its own ``decision``, else the one
    :data:`UNDECLARED_DECISIONS` names for it."""
    if c.decision:
        return str(c.decision)
    return next((k for k, v in UNDECLARED_DECISIONS.items() if v == c.key), None)


__all__ = ["CONTRACT_LENSES", "FAMILY_LENSES", "GAPS", "Gap", "LENSES", "LENS_TITLES", "NOT_ASKED",
           "NOT_METHODS", "SHARED", "UNDECLARED_DECISIONS", "gaps_for", "lenses_of_contract",
           "lenses_of_family", "own", "recorded_by", "serves"]

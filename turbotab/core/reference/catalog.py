"""Which lens each method serves, and the methods v2 offers without a contract entry (the gaps).

The method contracts (BLUEPRINT §13) and the model families (M1_CONTRACT §7) do not say which of
the five lenses they belong to: a contract declares where it runs and what it may learn from, never
whose field it is. A review packet is one lens's, so this module says it, once, for every
registered contract and family. ``shared`` means every lens offers it (the inference and
prediction machinery, the causal lane, the export). The reference test fails when a contract or a
family is registered without an entry here, or an entry names one that is not registered.

The gaps are the methods V2_DEFINITION_OF_DONE §2 lists that no contract declares: each names the
row of §2 it comes from, the lenses it serves, the decision kinds that record it, the code that
implements it, and, where its options already carry the customary and sound labels outside the
registry (``turbotab.core.custom_sound``), which question holds them. The test checks that every
decision kind is one the decision log accepts and every module imports, so a gap names real code.
"""
from __future__ import annotations

from dataclasses import dataclass

LENSES: tuple[str, ...] = ("dietary", "clinical", "metabolomics", "genomics", "survey")
SHARED = "shared"

LENS_TITLES: dict[str, str] = {
    "dietary": "Dietary assessment",
    "clinical": "Clinical",
    "metabolomics": "Metabolomics",
    "genomics": "Genomics",
    "survey": "Survey instruments",
}

# Every registered method contract: the lenses whose packet lists it, or (SHARED,).
# "survey" is the survey-instrument lens (scales and items); a complex survey design (weights,
# strata, PSUs: NHANES) is the dietary and clinical lenses' concern.
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
    # a complex survey design (NHANES), dietary and clinical
    "survey_population": ("dietary", "clinical"),
    "survey_linear": ("dietary", "clinical"),
    "survey_cox": ("dietary", "clinical"),
    "survey_ordinal": ("dietary", "clinical"),
    "design_based_cv": ("dietary", "clinical"),
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

# Every registered model family: the lenses whose packet lists it, or (SHARED,).
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


@dataclass(frozen=True)
class Gap:
    """A method V2_DEFINITION_OF_DONE §2 lists that has no method contract (BLUEPRINT §13)."""

    key: str
    name: str
    row: str  # the row of V2_DEFINITION_OF_DONE §2 it comes from
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


__all__ = ["CONTRACT_LENSES", "FAMILY_LENSES", "GAPS", "Gap", "LENSES", "LENS_TITLES", "SHARED",
           "gaps_for", "lenses_of_contract", "lenses_of_family", "own", "serves"]

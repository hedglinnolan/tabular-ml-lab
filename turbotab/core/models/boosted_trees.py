"""``boosted_trees``: gradient-boosted decision trees (HistGradientBoosting), missing values native."""
from __future__ import annotations

from typing import Any

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    CLASS_SCALES,
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Knob,
    Named,
    Situation,
    Source,
    register_family,
)

SMALL_N = 500
TINY_N = 200


class BoostedTrees(FamilyBase):
    key = "boosted_trees"
    label = "Boosted trees"
    inductive_bias = ("Effects are step functions that can bend and interact; many shallow trees "
                      "each correct the last.")
    strengths = (
        "Finds curves and interactions without being told.",
        "Uses rows with missing values as they are.",
    )
    cautions = (
        "No coefficients: effects are read from curves.",
        "Overfits easily and needs many rows.",
        "Validated by cross-validation: Harrell's bootstrap overstates a near-interpolating "
        "learner's performance.",
    )
    needs_scaling = False
    handles_missing = True
    # Near-interpolating: its apparent AUC approaches 1, and Harrell's bootstrap leaves about 0.2 of
    # AUC uncorrected (the repair round's replication; Coley et al. 2023), so it is not applied.
    bootstrap_optimism = False
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it).
    identity = Identity(kind="estimator", library="scikit-learn",
                        estimator="HistGradientBoostingRegressor; HistGradientBoostingClassifier",
                        seed_policy="random_state 0, fixed")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = True
    # Under inference it gives no table: its curves describe the declared exposure.
    inference_decl = InferenceDecl(table="description_only")
    invariances = ("monotone_per_column",)
    bias_terms = (Named(plain="Many shallow trees, each fit to what the trees before it missed.",
                        known_as="gradient boosting", source=Source("friedman2001")),)
    curve_shape = "piecewise_constant"
    # No sourced closed form for any of these (MODEL_FAMILY_CONTRACT C7).
    complexity = (
        Knob(setting="max_iter", more_means="more_flexible"),  # rounds
        Knob(setting="learning_rate", more_means="more_flexible"),
        Knob(setting="max_leaf_nodes", more_means="more_flexible"),
        Knob(setting="min_samples_leaf", more_means="simpler"),
        Knob(setting="l2_regularization", more_means="simpler"),
    )
    output = "margin"
    raw_scale = CLASS_SCALES
    attribution = "trees"
    architecture = ("trees",)
    review_lenses = ("shared",)
    sources = (Source("friedman2001"),)

    def methods_label(self, task: Task | None) -> str:
        return "gradient-boosted trees"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        if task == "regression":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(random_state=0)
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=0)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        return ("Histogram gradient boosting",
                "Up to 100 trees of at most 31 leaves; on more than 10,000 rows it stops early on "
                "a tenth of its training units, the latest when the folds follow time.")

    def assess(self, s: Situation) -> Assessment:
        concerns: list[str] = []
        fit = "good"
        score = 3.0 if s.n_rows >= 2000 else 2.0
        if s.n_rows < SMALL_N:
            concerns.append(f"{s.n_rows:,} rows is small for boosted trees: they overfit easily "
                            f"and their cross-validated score is noisy.")
            fit = "poor" if s.n_rows < TINY_N else "fair"
            score = 0.5 if s.n_rows < TINY_N else 1.0
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: most splits will "
                            f"fit noise.")
            fit = "poor"
            score = min(score, 1.0)
        if s.purpose == "inference":
            concerns.append("No coefficients or intervals: effects are read from curves instead.")
            fit = "fair" if fit == "good" else fit
            score = min(score, 1.0)
        return Assessment(score, fit, tuple(concerns))


BOOSTED_TREES = register_family(BoostedTrees())

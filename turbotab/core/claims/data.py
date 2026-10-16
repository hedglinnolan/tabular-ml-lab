"""The data area's claims (R0_PLAN.md R0b, package R0b-data): the six S1 claims of the claims
backtest that the data stages, the readings, the recognizers and the teaching make
(docs/turbotab-next/notes/claims-backtest/FIXES.md, S1), each as stage 1 corrected it.

* ``data.ul_share`` (BX-040): what a share of usual intake above a UL is a share of.
* ``data.constant_column`` (C-123): a column is left out without asking only when every row holds
  one value, a blank counting as a value.
* ``data.nhanes_weight`` (C-143): the NHANES weight the least-common-denominator rule names.
* ``data.omics_raw_scale`` (C-193): an assay block read as unnormalized counts or intensities.
* ``data.clusters_adjust`` (D-058): an intercept per group and clustered intervals, except a yes/no
  outcome in small groups.
* ``data.tail_share`` (D-086): the Results design's tail question, ruled and not built.

The regime predicates below are this module's own, written from each claim's evidence and never
from the code the claim guards (CLAIMS_REGISTRY.md §10: C-123's predicate, had it called
``constant_values``, would have shared its bug). Like every claims module, this one imports nothing
from the engine but the registry itself; a predicate imports pandas only when it is called.

The author grades no part above judgment (CLAIMS_REGISTRY.md §2.4); a grader other than the author
raises a part with its evidence.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

from turbotab.core.claims import Basis, Claim, Fires, Param, Part, Regime, Silent, register

HERE = "turbotab.core.claims.data"
AUTHOR = "R0b-data builder"
TESTS = "turbotab/core/tests/test_claims_data.py"
ACCEPTANCE = "turbotab/core/tests/acceptance"
# The tail question's tests, which the build of BEATS_MODELS_RESULTS.md's E23 writes (D-086).
TAIL_TESTS = "turbotab/core/tests/test_tail_split.py"


def _get(ctx: Any, key: str, default: Any = None) -> Any:
    if isinstance(ctx, Mapping):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


# ── regime predicates ────────────────────────────────────────────────────────────────────────


def a_share_above_a_ul(ctx: Any) -> bool:
    """The cut-off is a Tolerable Upper Intake Level, a positive number."""
    cutoff = _get(ctx, "cutoff")
    return (_get(ctx, "cutoff_kind") == "UL" and isinstance(cutoff, (int, float))
            and not isinstance(cutoff, bool) and math.isfinite(cutoff) and cutoff > 0)


def one_value_on_every_row(ctx: Any) -> bool:
    """Every row of the column holds the same value, a blank counting as a value of its own:
    pandas counts the distinct values with the blanks kept, so 'yes' beside blanks is two."""
    import pandas as pd

    values = pd.Series(_get(ctx, "values"), dtype=object)
    return len(values) > 0 and int(values.nunique(dropna=False)) == 1


# The weights' names as the CDC codebooks spell them: the interview, MEC examination, phlebotomy,
# dietary-recall, fasting and OGTT samples' (WTINT2YR, WTMEC2YR, WTPH2YR, WTDRD1, WTDR2D,
# WTSAF2YR, WTSOG2YR, and the pre-pandemic PRP and PP names), and the tutorial's pooled ones
# (MEC4YR).
_NHANES_WEIGHT = re.compile(r"^(WT(INT|MEC|PH|DRD1|DR2D|DR|SAF|SOG)[A-Z0-9]*|(INT|MEC|DR|SAF|SOG)"
                            r"[0-9]+YR)$")


def names_an_nhanes_weight(ctx: Any) -> bool:
    """The table carries an NHANES sample weight, named as the CDC codebooks name it."""
    return any(_NHANES_WEIGHT.match(str(c).upper()) for c in _get(ctx, "columns") or ())


def under_an_omics_lens(ctx: Any) -> bool:
    """The genomics or the metabolomics lens is on."""
    return bool({"genomics", "metabolomics"} & set(_get(ctx, "lenses") or ()))


def names_a_grouping(ctx: Any) -> bool:
    """A column groups the participants (sites, households, clinics)."""
    return bool(_get(ctx, "grouping"))


def a_least_squares_fit_with_a_cut_off(ctx: Any) -> bool:
    """A least-squares estimate on an outcome the lens gives a clinical cut-off."""
    return _get(ctx, "fit") == "least_squares" and _get(ctx, "cut_off") is not None


# ── BX-040 · the share above a UL ────────────────────────────────────────────────────────────

register(Claim(
    id="data.ul_share",
    kind="rule",
    card="The share above a UL is a share at risk of excess only where the intakes are what the UL "
         "covers: food, water and supplements for most nutrients, so a share from foods alone is a "
         "lower bound; supplements and medicines only for magnesium; supplements and fortified "
         "foods only for folic acid, niacin and vitamin E; and preformed vitamin A (retinol) from "
         "food and supplements, not carotenoids, for vitamin A.",
    methods="The share of usual intake above a Tolerable Upper Intake Level was read against the "
            "intake that UL covers, as the IOM DRI tables define it: total intake for most "
            "nutrients, so that a share from foods alone is a lower bound; the supplemental or "
            "synthetic form alone for magnesium, folic acid, niacin and vitamin E; and preformed "
            "vitamin A from every source for vitamin A.",
    regime=Regime("a share of usual intake above a UL the IOM DRI tables set, estimated from "
                  "24-hour recalls by the NCI method, the nutrient read from the component's name",
                  requires=(f"{HERE}:a_share_above_a_ul",)),
    decision="nci_usual_intake",
    verdict="noted",
    parts=(
        Part("forms", grade="judgment", basis=Basis(
            cites=("iom1997magnesium", "iom1998folate", "iom2000vitamine", "iom2001vitamina"),
            reason="We read the five ULs that cover one form from the IOM DRI summary tables' "
                   "footnotes, as the stage 1 tests quote them (read on 2026-10-15 through the "
                   "Linus Pauling Institute's and Health Canada's copies, the NCBI and ODS copies "
                   "having refused the fetch); no one but the author has checked them against the "
                   "reports.")),
        Part("lower_bound", grade="judgment", basis=Basis(
            reason="We read a share from foods alone as a lower bound wherever the UL covers food "
                   "too: each person's intake from every source is at least their intake from "
                   "foods, so supplements can only add to the share above the UL.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="We read the nutrient from the component's name, as the iron rule does; a name "
                   "the reader does not know is read as a nutrient whose UL covers every "
                   "source.")),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_nci_usual_intake.py::"
              "test_4_the_share_above_a_ul_says_what_the_ul_covers",
              truth="independent:the IOM DRI tables' UL footnotes"),
        Fires(f"{ACCEPTANCE}/test_nci_usual_intake.py::"
              "test_4_vitamin_a_s_ul_covers_retinol_from_food_and_supplements",
              truth="independent:the IOM DRI tables' UL footnotes"),
        Silent(f"{TESTS}::test_a_ul_that_covers_every_source_is_read_as_total_intake",
               truth="independent:the IOM DRI tables' UL footnotes"),
    ),
    enforced_by=("turbotab.core.methods.usual_intake:share_phrase",
                 "turbotab.core.usual_intake:ul_form"),
    author=AUTHOR,
))


# ── C-123 · a column of one value ────────────────────────────────────────────────────────────

register(Claim(
    id="data.constant_column",
    kind="rule",
    card="every row holds the same value",
    methods="A column was left out without asking only where every row held the same value, a "
            "blank counting as a value; a column of one value beside blanks was asked about, since "
            "its blank may be the other answer.",
    regime=Regime("a column of the table as the store holds it, a blank (a missing cell) counting "
                  "as a value of its own, so a checkbox exported as 'yes' or 1 where ticked and "
                  "blank where not holds two values",
                  requires=(f"{HERE}:one_value_on_every_row",)),
    decision="set_roles",
    options=("excluded", "predictor"),
    verdict="not_offered",
    parts=(
        Part("constant", grade="judgment", basis=Basis(
            reason="We leave out, without asking, a column that holds one value on every row: it "
                   "never varies, so no model can use it (a linear model with an intercept gains "
                   "nothing from it, and a tree cannot split on it).")),
        Part("blank", grade="judgment", basis=Basis(
            reason="We count a blank as a value: a checkbox exported as 'yes' or 1 where ticked "
                   "and blank where not varies through its blanks, so it is asked about and kept "
                   "until the user says otherwise.")),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_discriminate.py::"
              "test_c3_each_value_test_can_settle_its_own_reading[role:excluded]",
              truth="construction"),
        Silent(f"{ACCEPTANCE}/test_discriminate.py::"
               "test_c3_one_value_beside_blanks_settles_no_constant",
               truth="independent:pandas"),
        Silent("turbotab/core/tests/test_rows.py::"
               "test_one_value_beside_blanks_is_asked_never_left_out_as_a_constant",
               truth="independent:pandas"),
    ),
    enforced_by=("turbotab.core.readings:constant_values",
                 "turbotab.core.stages.rows:propose_roles"),
    author=AUTHOR,
))


# ── C-143 · the NHANES weight ────────────────────────────────────────────────────────────────

register(Claim(
    id="data.nhanes_weight",
    kind="rule",
    card="An NHANES analysis takes the weight of the smallest sample its variables were measured "
         "on: the OGTT, fasting, dietary-recall, phlebotomy or examination sample, or the "
         "interview sample where every variable was recorded on interviewed participants who were "
         "not examined. A variable recorded only where a subsample's weight is positive (at least "
         "{min_rows} rows, at most {slack:.0%} of them outside it, and blank on at least "
         "{min_missing:.0%} of the table) is that subsample's, whatever its name. A fasting or "
         "OGTT weight that no variable here reads as measured on, or a phlebotomy weight beside "
         "which nothing reads as measured on its sample or a smaller one, leaves the choice to the "
         "user.",
    methods="The survey weight followed NCHS's least-common-denominator rule, the weight of the "
            "smallest sample on which every analysis variable was measured, read from the "
            "variables' names and from where their values are recorded (a variable recorded only "
            "where a subsample weight is positive, at most {slack:.0%} of its rows outside, being "
            "that subsample's); the diet behavior questionnaire's items were read as interview "
            "variables, not dietary recalls.",
    regime=Regime("an NHANES table that carries its own sample weights, named as the CDC codebooks "
                  "name them, with the analysis variables in the same table",
                  requires=(f"{HERE}:names_an_nhanes_weight",)),
    decision="set_survey.weight",
    verdict="recommended",
    params={
        "slack": Param(0.01, unit="share of a variable's recorded rows",
                       constant="turbotab.core.recognizers:SUBSAMPLE_SLACK"),
        "min_missing": Param(0.05, unit="share of the table's rows",
                             constant="turbotab.core.recognizers:SUBSAMPLE_MIN_MISSING"),
        "min_rows": Param(10, unit="rows", constant="turbotab.core.recognizers:SUBSAMPLE_MIN_ROWS"),
    },
    parts=(
        Part("rule", grade="judgment", basis=Basis(
            cites=("nhanes_weighting_tutorial",),
            reason="We follow the NHANES weighting tutorial's rule, \"use the weight of the "
                   "smallest subpopulation that includes all the variables you want to include in "
                   "your analysis\", ordering the samples from the OGTT subsample up to the "
                   "interview sample as the codebooks nest them.")),
        Part("phlebotomy", grade="judgment", basis=Basis(
            cites=("nhanes_weighting_tutorial",),
            reason="We name the phlebotomy weight for a variable recorded only where it is "
                   "positive, as the TCHOL_L codebook says for August 2021-August 2023 (\"The "
                   "phlebotomy weight should be used for analyses that use variables derived from "
                   "blood analytes\").")),
        Part("interview", grade="judgment", basis=Basis(
            cites=("nhanes_weighting_tutorial",),
            reason="We name the interview weight where every variable is recorded on interviewed "
                   "participants who were not examined, as the tutorial's Example 1 does (\"You "
                   "would use the interview weights for your analysis\"); a table of examined rows "
                   "alone keeps the examination weight.")),
        Part("diet_behavior", grade="judgment", basis=Basis(
            reason="We read the diet behavior questionnaire's DBQ and DBD items as interview "
                   "variables, as DBQ_L's weight note says (\"These questions were asked as part "
                   "of the household interview\").")),
        Part("tolerance", grade="judgment", basis=Basis(
            reason="Our own tolerance for a merge's stray rows: no source sets 1%, 5% or 10 rows; "
                   "they keep a variable measured outside a subsample from being read as the "
                   "subsample's.")),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_wp13_recognizers.py::"
              "test_6c_blood_analytes_of_2021_2023_name_the_phlebotomy_weight",
              truth="construction"),
        Fires(f"{ACCEPTANCE}/test_wp13_recognizers.py::"
              "test_6d_interview_only_variables_name_the_interview_weight",
              truth="construction"),
        Fires(f"{ACCEPTANCE}/test_wp13_recognizers.py::"
              "test_6e_diet_behavior_questions_alone_take_the_interview_weight",
              truth="construction"),
        Fires(f"{ACCEPTANCE}/test_intelligence_gate_repair.py::"
              "test_wp13_6_a_fasting_analyte_is_read_by_where_it_is_recorded",
              truth="construction"),
        Silent(f"{TESTS}::test_no_variable_recorded_only_on_blood_drawn_rows_keeps_the_mec_weight",
               truth="construction"),
        Silent(f"{TESTS}::test_a_variable_measured_outside_a_subsample_is_not_that_subsample_s",
               truth="construction"),
        Silent(f"{TESTS}::test_an_examination_variable_keeps_the_mec_weight_beside_interview_ones",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.recognizers:least_common_denominator",
                 "turbotab.core.recognizers:subsample_variables",
                 "turbotab.core.recognizers:_interview_only"),
    author=AUTHOR,
))


# ── C-193 · raw omics values ─────────────────────────────────────────────────────────────────

register(Claim(
    id="data.omics_raw_scale",
    kind="rule",
    card="Under the genomics or metabolomics lens, a block of at least {min_features} numeric "
         "columns (not the outcome, an identifier, a person's characteristic or how the samples "
         "were acquired) is read as unnormalized: as raw counts under genomics when they are "
         "non-negative whole numbers reaching {count_max:,.0f}; as raw intensities under "
         "metabolomics when they are non-negative, span {raw_range:,.0f}-fold, and reach "
         "{raw_max:,.0f} or have a median skewness of at least {raw_skew}; and under genomics as "
         "FPKM or estimated counts, which the values cannot tell apart, when they are fractional "
         "with a zero and a value of {raw_max:,.0f} or more each in {estimated_columns} columns or "
         "more, and sample totals not all a million. Other values are read as already normalized "
         "or on a log scale.",
    methods="Assay values were read as unnormalized from the values alone, never the outcome: "
            "whole-number counts reaching {count_max:,.0f}, intensities spanning "
            "{raw_range:,.0f}-fold and reaching {raw_max:,.0f} or right-skewed (median skewness "
            "at least {raw_skew}), or fractional values with zeros and values of {raw_max:,.0f} or "
            "more in {estimated_columns} columns or more (FPKM or estimated counts), over a block "
            "of at least {min_features} columns.",
    regime=Regime("a table under the genomics or metabolomics lens whose assay values are as the "
                  "file holds them; the reading looks at the values alone, never at the outcome",
                  requires=(f"{HERE}:under_an_omics_lens",)),
    decision="omics_normalization",
    verdict="noted",
    params={
        "min_features": Param(20, unit="columns",
                              constant="turbotab.core.methods.omics:MIN_FEATURES"),
        "count_max": Param(100.0, unit="largest value",
                           constant="turbotab.core.methods.omics:COUNT_MAX"),
        "raw_max": Param(1_000.0, unit="largest value",
                         constant="turbotab.core.methods.omics:RAW_MAX"),
        "raw_range": Param(100.0, unit="largest over smallest positive value",
                           constant="turbotab.core.methods.omics:RAW_RANGE"),
        "raw_skew": Param(0.5, unit="median skewness over the features",
                          constant="turbotab.core.methods.omics:RAW_SKEW"),
        "estimated_columns": Param(2, unit="columns",
                                   constant="turbotab.core.methods.omics:ESTIMATED_COLUMNS"),
    },
    parts=(
        Part("block", grade="judgment", basis=Basis(
            reason="We read 20 assay-like columns or more as a feature block, so a clinical "
                   "table's handful of laboratory values is never read as one; an unsourced "
                   "convention.")),
        Part("counts", grade="judgment", basis=Basis(
            reason="We read whole numbers reaching 100 as counts: genotype dosages (0, 1 and 2) "
                   "and scores stay below it; an unsourced convention.")),
        Part("intensities", grade="judgment", basis=Basis(
            reason="We read values spanning 100-fold that reach 1,000, or that are right-skewed "
                   "(median skewness 0.5), as unlogged: log2 intensities stay below about 40, a "
                   "log-normal feature with a 30% CV has skewness (e^s2 + 2)(e^s2 - 1)^0.5, about "
                   "0.93, and a log scale is near-symmetric. The cut-offs are unsourced "
                   "conventions.")),
        Part("estimated", grade="judgment", basis=Basis(
            reason="We ask about fractional values with a zero and the thousands each in two "
                   "columns or more, as GENOMICS_PACK §02 row 2 says to ask; an internal pack "
                   "section is no source (CLAIMS_REGISTRY.md §2.4).")),
    ),
    tests=(
        Fires("turbotab/core/tests/test_omics.py::"
              "test_the_reading_names_raw_values_and_stays_quiet_on_normalized_ones",
              truth="construction"),
        Fires("turbotab/core/tests/test_omics.py::"
              "test_estimated_counts_raise_the_normalization_question",
              truth="construction"),
        Fires("turbotab/core/tests/test_omics.py::"
              "test_estimated_counts_at_balanced_depth_are_asked_about",
              truth="construction"),
        Silent("turbotab/core/tests/test_omics.py::"
               "test_one_clinical_column_does_not_make_a_block_read_as_estimated_counts",
               truth="construction"),
        Silent("turbotab/core/tests/test_omics.py::"
               "test_lc_ms_peak_areas_are_never_read_as_estimated_counts",
               truth="construction"),
        Silent("turbotab/core/tests/test_omics.py::"
               "test_genotype_dosages_and_two_level_columns_are_not_counts",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.omics:scale_reading",
                 "turbotab.core.methods.omics:_estimated_counts"),
    author=AUTHOR,
))


# ── D-058 · groups above the person ──────────────────────────────────────────────────────────
# Both sentences are the teaching entry's (``teaching.content.CLUSTERS``), card-facing plain words:
# the card is its "why", the methods register its "Adjust and cluster" option. The number is
# ``estimand.FIXED_EFFECTS_LOGIT_ROWS``'s, which the guard reads.

register(Claim(
    id="data.clusters_adjust",
    kind="rule",
    card="Under inference a site that differs in both what you study and the outcome confounds the "
         "estimate, and intervals treating its participants as independent can be too narrow: "
         "each site gets its own intercept and clustered intervals, except a yes/no outcome in "
         "groups under {rows} rows. Under prediction, validation keeps a group together and can "
         "hold out whole groups.",
    methods="Each group gets its own intercept and clustered intervals; yes/no outcomes need "
            "{rows} rows per group.",
    goal=("inference", "prediction"),
    regime=Regime("participants grouped by a column (sites, households, clinics); under inference "
                  "a yes/no outcome's group size is read as the mean rows per group over the "
                  "analyzed rows' groups of two or more, and a numeric outcome's linear model has "
                  "no incidental-parameter bias",
                  requires=(f"{HERE}:names_a_grouping",)),
    decision="set_clusters.adjust",
    options=("fixed_effects", "cluster_only", "none"),
    verdict="refused",
    params={"rows": Param(16, unit="rows per group",
                          constant="turbotab.core.estimand:FIXED_EFFECTS_LOGIT_ROWS")},
    parts=(
        Part("confounding", grade="judgment", basis=Basis(
            reason="We give each site its own intercept under inference: a site that differs in "
                   "both what you study and the outcome confounds a pooled estimate, and its own "
                   "intercept compares participants within it.")),
        Part("intervals", grade="judgment", basis=Basis(
            reason="We cluster the intervals by the grouping: participants in one group share "
                   "more than chance, so intervals treating them as independent can be too "
                   "narrow.")),
        Part("incidental_parameters", grade="judgment", basis=Basis(
            reason="We give a yes/no outcome no intercept per group below 16 rows per group on "
                   "average: Katz (2001, Political Analysis 9:379) found the unconditional "
                   "fixed-effects logit's bias negligible from 16, and pairs double the log odds "
                   "ratio (Andersen 1970). Katz 2001 has no record in the citation registry, so "
                   "this part cannot be graded sourced yet.")),
        Part("prediction", grade="judgment", basis=Basis(
            reason="We validate by whole groups under prediction, so a group's rows never sit on "
                   "both sides of a split, and a grouping there sets no intercepts.")),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_wp17_purpose_routes.py::"
              "test_4_a_yes_no_outcome_in_small_groups_gets_no_intercept_per_group",
              truth="independent:statsmodels"),
        Fires(f"{ACCEPTANCE}/test_wp17_purpose_routes.py::"
              "test_4_intercepts_per_group_are_checked_again_at_the_task_the_models_and_the_design",
              truth="construction"),
        Fires(f"{ACCEPTANCE}/test_wp17_purpose_routes.py::"
              "test_4_under_prediction_the_grouping_names_internal_external_validation",
              truth="construction"),
        Silent(f"{ACCEPTANCE}/test_wp17_purpose_routes.py::"
               "test_4_intercepts_per_group_are_judged_on_the_analyzed_rows_groups_of_two_or_more",
               truth="construction"),
        Silent(f"{TESTS}::test_a_yes_no_outcome_in_groups_of_forty_keeps_an_intercept_per_group",
               truth="independent:statsmodels"),
    ),
    enforced_by=("turbotab.core.estimand:_clusters_name_a_grouping",
                 "turbotab.core.estimand:_intercepts_still_have_rows",
                 "turbotab.core.estimand:intercepts_refusal"),
    author=AUTHOR,
))


# ── D-086 · the tail question (ruled, not built) ─────────────────────────────────────────────
# No engine code computes it yet. Its enforcer and its tests name what the build will add, so the
# claim derives as proposed and never renders: a claim that names no code is built at once
# (``Registry.status``), and one that names no code but a test not yet written fails meta-test 1.

register(Claim(
    id="data.tail_share",
    kind="rule",
    card="For an outcome with a clinical cut-off, the estimate is split exactly at the cut-off "
         "into two parts that add up to it, and the tail's share is read against the share a "
         "uniform shift would give, the share of people on the tail's side: mostly the tail only "
         "when the {level:.0%} interval of the tail part less that share times the outcome "
         "excludes zero on the estimate's side, mostly short of the cut-off when it excludes zero "
         "on the other side, and throughout otherwise.",
    methods="The primary estimate was split exactly at the clinical cut-off into the part short of "
            "it and the part past it, whose least-squares coefficients sum to the primary's; the "
            "tail's share was judged against the share of rows on the tail's side, the share a "
            "uniform shift gives, by the {level:.0%} interval of the coefficient of the tail part "
            "less that share times the outcome, and nothing was read where the primary's own "
            "interval included zero.",
    goal=("inference",),
    regime=Regime("a least-squares estimate on an outcome with a clinical cut-off the lens states "
                  "(fasting glucose's ADA cut-off, a low hemoglobin), the lens fact naming which "
                  "side is the tail; read only where the primary's interval excludes zero",
                  requires=(f"{HERE}:a_least_squares_fit_with_a_cut_off",)),
    verdict="noted",
    params={"level": Param(0.95, unit="interval coverage")},
    parts=(
        Part("identity", grade="judgment", basis=Basis(
            reason="Least squares is linear in the outcome, so the two parts' coefficients add up "
                   "to the primary's exactly; graded judgment by its author, for a grader to "
                   "raise.")),
        Part("baseline", grade="judgment", basis=Basis(
            reason="We judge the tail's share against p, the share of rows on the tail's side: "
                   "shifting every value by d moves the tail part by d times that share, a "
                   "first-order baseline. Never against one half, which reads a uniform effect "
                   "as mostly the tail wherever most people are past the cut-off (BMI 25, LDL "
                   "100 mg/dL).")),
        Part("interval", grade="judgment", basis=Basis(
            reason="We read the tail by the 95% interval of the coefficient of tail - p*y, a "
                   "convention; the design's simulation of a BMI-like outcome found 6% and 18% of "
                   "uniform shifts read otherwise against 5% nominal.")),
    ),
    tests=(
        Fires(f"{TAIL_TESTS}::test_the_two_parts_add_up_to_the_primary", truth="construction"),
        Fires(f"{TAIL_TESTS}::test_an_effect_in_the_tail_reads_mostly_the_tail",
              truth="construction"),
        Silent(f"{TAIL_TESTS}::test_a_uniform_shift_reads_throughout_wherever_the_cut_off_sits",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.tail_split:tail_question",),
    ruled="docs/turbotab-next/calm/BEATS_MODELS_RESULTS.md §6, the tail question (Round 2, E23; "
          "claims backtest D-086)",
    author=AUTHOR,
))

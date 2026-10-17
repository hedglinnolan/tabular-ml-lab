"""The data area's claims (R0_PLAN.md R0b, package R0b-data): the six S1 claims of the claims
backtest that the data stages, the readings, the recognizers and the teaching make
(docs/turbotab-next/notes/claims-backtest/FIXES.md, S1), each as stage 1 corrected it.

* ``data.ul_share`` (BX-040): what a share of usual intake above a UL is a share of, and its
  fallback ``data.ul_share_unread`` for a component whose name is read as no nutrient.
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
raises a part with its evidence, as the package's verifier raised three parts to proved (repair
rounds 1 and 2).
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from turbotab.core.claims import Basis, Claim, Fires, Param, Part, Regime, Silent, register

HERE = "turbotab.core.claims.data"
AUTHOR = "R0b-data builder"
# The package's verifier, who raised the parts its evidence proves (repair rounds 1 and 2).
VERIFIER = "R0b-data verifier, 2026-10-16"
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


# The nutrients the IOM DRI summary tables set an adult Tolerable Upper Intake Level for, as the
# author read the tables' elements and vitamins (boron to zinc; vitamin A to choline); vitamin K,
# potassium, thiamin, riboflavin, vitamin B12, pantothenic acid, biotin, chromium and the
# carotenoids have none. Folate's is set as folic acid, so either word names it.
_IOM_UL_NUTRIENTS = frozenset({
    "boron", "calcium", "chloride", "copper", "fluoride", "iodine", "iron", "magnesium",
    "manganese", "molybdenum", "nickel", "phosphorus", "selenium", "sodium", "vanadium", "zinc",
    "vitamin a", "vitamin c", "vitamin d", "vitamin e", "niacin", "vitamin b6", "folate",
    "folic acid", "choline"})


def a_nutrient_the_iom_sets_a_ul_for(ctx: Any) -> bool:
    """The component's name was read as a nutrient (``nutrient``, in words), and the IOM DRI
    tables set that nutrient a UL. A name read as no nutrient (None) is outside."""
    nutrient = _get(ctx, "nutrient")
    return isinstance(nutrient, str) and nutrient.strip().lower() in _IOM_UL_NUTRIENTS


_ABSENT = object()


def a_name_read_as_no_nutrient(ctx: Any) -> bool:
    """The component's name was read, and read as no nutrient: ``nutrient`` is given, and it is
    None. A context that does not say how the name was read is outside."""
    return _get(ctx, "nutrient", _ABSENT) is None


def one_value_on_every_row(ctx: Any) -> bool:
    """Every row of the column holds the same value, a blank counting as a value of its own:
    pandas counts the distinct values with the blanks kept, so 'yes' beside blanks is two."""
    import pandas as pd

    values = pd.Series(_get(ctx, "values"), dtype=object)
    return len(values) > 0 and int(values.nunique(dropna=False)) == 1


# The weights' names as the CDC codebooks spell them, each the sample's word and then its cycle
# or nothing, never any other ending: the interview, MEC examination and fasting samples'
# (WTINT2YR, WTMEC4YR, WTSAF2YR, the pre-pandemic WTINTPRP, WTMECPRP and WTSAFPRP), the
# phlebotomy and OGTT samples' (WTPH2YR, WTSOG2YR), the dietary recalls' (WTDRD1, WTDR2D, the
# pre-pandemic WTDRD1PP and WTDR2DPP, 1999-2002's WTDR4YR), and the other subsamples' ("WTS", the
# subsample's letters and the cycle, within the eight characters a SAS transport name holds:
# WTSA2YR, WTSB2YR, WTSC2YR, WTSVOC2Y); and the weighting tutorial's pooled ones, the sample's
# word and the years (MEC4YR, WTMEC6YR, DRD14YR, DR2D4YR).
_NHANES_WEIGHT = re.compile(r"^(WT(INT|MEC|SAF)(\d+YR|PRP)|WT(PH|SOG)\d+YR|WTDR(D1|2D)(PP)?"
                            r"|WTS(?=[A-Z0-9]{2,5}$)[A-Z]+\d+YR?"
                            r"|(WT)?(INT|MEC|PH|DRD1|DR2D|DR|SAF|SOG)\d+YR)$")


def names_an_nhanes_weight(ctx: Any) -> bool:
    """The table carries an NHANES sample weight, named as the CDC codebooks name it."""
    return any(_NHANES_WEIGHT.match(str(c).upper()) for c in _get(ctx, "columns") or ())


def under_an_omics_lens(ctx: Any) -> bool:
    """The genomics or the metabolomics lens is on."""
    return bool({"genomics", "metabolomics"} & set(_get(ctx, "lenses") or ()))


def names_a_grouping(ctx: Any) -> bool:
    """A column groups the participants (sites, households, clinics)."""
    return bool(_get(ctx, "grouping"))


def an_outcome_its_evidence_covers(ctx: Any) -> bool:
    """Under inference, a numeric outcome (``task`` "regression") or a yes/no one ("binary"), the
    two outcomes the incidental-parameter evidence speaks to; an ordinal, multiclass or
    time-to-event outcome is outside. Under prediction, which sets no intercepts, any outcome."""
    if _get(ctx, "purpose") == "prediction":
        return True
    return _get(ctx, "task") in ("regression", "binary")


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
                  "24-hour recalls by the NCI method, for a component whose name is read as a "
                  "nutrient the tables set a UL for; a name read as no nutrient (NHANES's "
                  "supplement-recall totals such as DS1TMAGN, a bare 'Mg') is outside it",
                  requires=(f"{HERE}:a_share_above_a_ul",
                            f"{HERE}:a_nutrient_the_iom_sets_a_ul_for")),
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
        Part("lower_bound", grade="proved", graded_by=VERIFIER, basis=Basis(
            theorem="Wherever the UL covers the form the food holds, a share of usual intake above "
                    "the UL from foods alone is a lower bound on the share from every source. For "
                    "each person i, the usual intake of the covered form from every source is "
                    "T_i = F_i + S_i, with intake from supplements S_i >= 0, so T_i > u wherever "
                    "F_i > u, and 1{F_i > u} <= 1{T_i > u} for every i. For any non-negative "
                    "weights w_i (survey weights included), sum w_i 1{F_i > u} / sum w_i <= "
                    "sum w_i 1{T_i > u} / sum w_i. Its hypotheses: the UL covers the food form "
                    "(every UL but magnesium's, folic acid's, niacin's and vitamin E's; vitamin "
                    "A's covers food retinol), and both shares are over the same population and "
                    "weights. It bounds the population shares the NCI method estimates, not each "
                    "estimate.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="We read the nutrient from the component's name, as the iron rule does, and "
                   "the claim covers only a name read as a nutrient the IOM tables set a UL for, "
                   "whose list the author read from the tables. A name the reader reads as no "
                   "nutrient (a magnesium total named DS1TMAGN) takes the weaker "
                   "data.ul_share_unread. A name read as a nutrient the tables set no UL for "
                   "(potassium, protein), or only as a vitamin (vitamin_d_ug), is outside both: "
                   "the code words its share as a UL covering every source, our default and no "
                   "reading of the IOM tables.")),
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
    fallback="data.ul_share_unread",
    author=AUTHOR,
))


register(Claim(
    id="data.ul_share_unread",
    kind="rule",
    card="The share above the UL is reported as a share of the distribution: what a UL covers "
         "depends on the nutrient (for magnesium, supplements and medicines alone), and the "
         "nutrient was not read from this column's name.",
    methods="Where the nutrient was not read from the component's name, the share of usual intake "
            "above the UL was reported as a share of the distribution, not as a share at risk of "
            "excess, since what a UL covers depends on the nutrient (supplements and medicines "
            "alone for magnesium).",
    regime=Regime("a share of usual intake above a UL, estimated from 24-hour recalls by the NCI "
                  "method, for a component whose name the reader reads as no nutrient (NHANES's "
                  "supplement-recall totals such as DS1TMAGN, a bare 'Mg')",
                  requires=(f"{HERE}:a_share_above_a_ul", f"{HERE}:a_name_read_as_no_nutrient"),
                  disjoint_from=("data.ul_share",)),
    decision="nci_usual_intake",
    verdict="noted",
    parts=(
        Part("distribution", grade="judgment", basis=Basis(
            reason="We report the share as a share of the distribution, never as a share at risk "
                   "of excess: a share above a UL is one only where the intakes are what the UL "
                   "covers, and without the nutrient that cannot be read.")),
        Part("coverage", grade="judgment", basis=Basis(
            cites=("iom1997magnesium",),
            reason="We say what a UL covers depends on the nutrient, as the IOM DRI tables' "
                   "footnotes do (magnesium's covers intake from a pharmacological agent only, "
                   "calcium's every source), read as data.ul_share's forms part reads them, and "
                   "checked by no one but the author.")),
        Part("unread", grade="judgment", basis=Basis(
            reason="We take a name as naming no nutrient where the name reader reads none; a "
                   "person may read the nutrient in it (DS1TMAGN is NHANES's supplement-recall "
                   "magnesium), so the sentence says the nutrient was not read, not that the name "
                   "names none.")),
    ),
    tests=(
        Fires(f"{TESTS}::test_a_ul_share_on_a_name_read_as_no_nutrient_is_a_share_of_the_"
              "distribution", truth="independent:the IOM DRI tables' UL footnotes"),
        Silent(f"{TESTS}::test_a_ul_share_on_a_name_read_as_calcium_keeps_its_sentence",
               truth="independent:the IOM DRI tables' UL footnotes"),
        # The regime's finite cut-off: the decision refuses any other, and the share's sentence
        # renders no card from one (the verifier's V2).
        Fires(f"{TESTS}::test_a_non_finite_cut_off_is_refused_at_the_decision",
              truth="construction"),
        Fires(f"{TESTS}::test_a_share_above_an_infinite_ul_never_reaches_the_card",
              truth="construction"),
        Silent(f"{TESTS}::test_a_finite_cut_off_is_accepted_at_the_decision",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.usual_intake:share_phrase",
                 "turbotab.core.methods.usual_intake:methods_sentence",
                 "turbotab.core.decisions:SetUsualIntake"),
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
        Part("constant", grade="proved", graded_by=VERIFIER, basis=Basis(
            theorem="A column x that holds one value k on every row i, a blank counting as a "
                    "value (the predicate one_value_on_every_row), distinguishes no two rows: any "
                    "model's prediction f(x_i, z_i) equals f(k, z_i), a function of the other "
                    "columns z alone, on every row, so no split on x separates two rows; and in a "
                    "design with an intercept, x is k times the intercept column, adding no "
                    "dimension to the column space (perfect collinearity), so the least-squares "
                    "fitted values are unchanged.")),
        Part("without_asking", grade="judgment", basis=Basis(
            reason="We leave such a column out without asking: it carries nothing a model can "
                   "use (the constant part), so a question about it would leave the user nothing "
                   "to decide.")),
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
         "not examined, at least {min_rows} of them in the table. A variable recorded only where "
         "a subsample's weight is positive (at least "
         "{min_rows} rows, at most {slack:.0%} of them outside it, and blank on at least "
         "{min_missing:.0%} of the table) is that subsample's, whatever its name. A fasting or "
         "OGTT weight that no variable here reads as measured on, or a phlebotomy weight beside "
         "which nothing reads as measured on its sample or a smaller one, leaves the choice to the "
         "user, as does another subsample's weight beside which nothing reads as measured on its "
         "subsample.",
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
                   "your analysis\". The codebooks nest most of the samples: the OGTT subsample "
                   "within the fasting one, the fasting within the phlebotomy sample, both "
                   "dietary days within the first, the phlebotomy and the dietary-recall samples "
                   "each within the examination sample, and that within the interview sample. "
                   "The phlebotomy and the dietary-recall samples do not nest, neither holding the "
                   "other; ordering the dietary recalls as the smaller is our call, by size "
                   "(TCHOL_L's 7,316 positive phlebotomy weights against DR1TOT_L's 6,694 "
                   "reliable day-1 recalls, read 2026-10-15), not by nesting.")),
        Part("phlebotomy", grade="judgment", basis=Basis(
            cites=("nhanes_weighting_tutorial", "nhanes_documentation"),
            reason="We name the phlebotomy weight for a blood analyte, as the weighting "
                   "tutorial's Example 5 does for serum vitamin D (\"You would use phlebotomy "
                   "weights (wtph2yr)\") and as the TCHOL_L codebook says for August 2021-August "
                   "2023 (\"The phlebotomy weight should be used for analyses that use variables "
                   "derived from blood analytes\"). Reading a blood analyte as a variable recorded "
                   "only where that weight is positive is our call.")),
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
                   "subsample's. The same 10 rows are the interview reading's floor: a table with "
                   "fewer interviewed participants who were not examined is read as the "
                   "examination sample, and keeps its weight, as the card states; no source sets "
                   "that floor either.")),
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
        Fires(f"{TESTS}::test_a_variable_recorded_only_where_the_subsample_c_weight_is_positive_"
              "names_it", truth="construction"),
        Silent(f"{TESTS}::test_no_variable_recorded_only_on_blood_drawn_rows_keeps_the_mec_weight",
               truth="construction"),
        Silent(f"{TESTS}::test_a_variable_measured_outside_a_subsample_is_not_that_subsample_s",
               truth="construction"),
        Silent(f"{TESTS}::test_an_examination_variable_keeps_the_mec_weight_beside_interview_ones",
               truth="construction"),
        Silent(f"{TESTS}::test_a_word_that_begins_like_subsample_c_s_weight_is_no_sampling_weight",
               truth="construction"),
        Silent(f"{TESTS}::test_a_variable_on_four_rows_inside_the_fasting_subsample_is_not_that_"
               "subsample_s", truth="construction"),
        Silent(f"{TESTS}::test_a_variable_on_nearly_every_row_is_no_subsample_s_where_the_weight_"
               "covers_nearly_all", truth="construction"),
        Fires(f"{TESTS}::test_a_subsample_variable_with_a_stray_row_outside_it_is_still_that_"
              "subsample_s", truth="construction"),
        Silent(f"{TESTS}::test_a_table_with_five_rows_outside_the_examination_keeps_the_mec_weight",
               truth="construction"),
        Fires(f"{TESTS}::test_the_weight_card_states_the_rows_at_which_the_interview_reading_turns",
              truth="construction"),
    ),
    enforced_by=("turbotab.core.recognizers:least_common_denominator",
                 "turbotab.core.recognizers:subsample_variables",
                 "turbotab.core.recognizers:other_subsample_weight",
                 "turbotab.core.recognizers:_nhanes_weight_name",
                 "turbotab.core.recognizers:_interview_only"),
    author=AUTHOR,
))


# ── C-193 · raw omics values ─────────────────────────────────────────────────────────────────

# The NHANES variables on omics' exclusion list, as NHANES names them: age in years and in months,
# sex, race (RIDRETH1, and RIDRETH3 from the 2011-2012 cycle on), BMI and the income-to-poverty
# ratio. turbotab.core.methods.omics reads them, and the identifiers it holds as one word, from
# here, so the card and the methods sentence name the list itself.
OMICS_NHANES_NAMES = ("RIDAGEYR", "RIDAGEMN", "RIAGENDR", "RIDRETH1", "RIDRETH3", "BMXBMI",
                      "INDFMPIR")
# An ID counts as a whole word of a name ("sample ID", "visit_id"), or as one of these whole names.
OMICS_ONE_WORD_IDS = ("SampleID", "SubjectID", "PatientID")


def _listed(names: Sequence[str], last: str = "or") -> str:
    """``names`` listed, ``last`` before the last of them: "A, B or C"."""
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {last} {names[-1]}"


register(Claim(
    id="data.omics_raw_scale",
    kind="rule",
    card="Under the genomics or metabolomics lens, assay values are read from the numeric "
         "columns that are not the outcome, a column whose name, or a whole word of it, is an "
         "identifier (SEQN, or an ID, which counts as the whole word ID, as in a sample, subject "
         f"or patient ID, or as {_listed(OMICS_ONE_WORD_IDS)}), a person's age, sex, gender or "
         f"BMI, NHANES's {_listed(OMICS_NHANES_NAMES)}, or a time (a visit, time point, date, "
         "day, hour, year or session), a yes/no column, a "
         "column named for how the samples were acquired (a batch, plate, well, run, run order, "
         "plex or polarity) or a numbering of the rows, and that are non-negative with more than "
         "{indicator_levels} distinct values each. A "
         "reading takes at least {min_features} such columns and names only those. They are read "
         "as raw counts under genomics when {min_features} or more of them hold whole numbers, "
         "the largest reaching {count_max:,.0f}; as raw intensities under metabolomics when they "
         "span {raw_range:,.0f}-fold and either reach {raw_max:,.0f} or have a median skewness of "
         "at least {raw_skew}; and under genomics as FPKM or estimated counts, which the values "
         "cannot tell apart, when {min_features} or more of them are fractional, with a zero and "
         "a value of {raw_max:,.0f} or more each in {estimated_columns} columns or more, and "
         "sample totals not all within {per_million_near:.0%} of a million. Other values are read "
         "as already normalized or on a log scale.",
    methods="Assay values were read as unnormalized from the values alone, never the outcome, over "
            "the numeric columns other than those named, wholly or by a whole word, as an "
            "identifier (SEQN, or an ID, counted as the whole word ID or as "
            f"{_listed(OMICS_ONE_WORD_IDS)}), a participant's age, sex, gender or BMI, NHANES's "
            f"{_listed(OMICS_NHANES_NAMES)}, or a time (a visit, time point, date, day, hour, "
            "year or session), yes/no columns, those named for "
            "how the samples were acquired and a numbering of the rows, that were non-negative "
            "with more than {indicator_levels} distinct values each, at least {min_features} of "
            "them: whole-number counts reaching {count_max:,.0f}, intensities spanning "
            "{raw_range:,.0f}-fold and either reaching {raw_max:,.0f} or right-skewed (median "
            "skewness at least {raw_skew}), or fractional values with zeros and values of "
            "{raw_max:,.0f} or more in {estimated_columns} columns or more whose sample totals "
            "were not all within {per_million_near:.0%} of a million (FPKM or estimated counts).",
    regime=Regime("a table under the genomics or metabolomics lens whose assay values are as the "
                  "file holds them, an identifier, a person's age, sex, gender or BMI, or a time "
                  "beside them included and named as the reading's own list names them; a "
                  "characteristic or covariate named otherwise (smoking, race, income, a weight) "
                  "is read by its values like an assay column; the reading looks at the values "
                  "alone, never at the outcome",
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
        "indicator_levels": Param(2, unit="distinct values or fewer: an indicator, no assay "
                                       "reading",
                                  constant="turbotab.core.methods.omics:INDICATOR_LEVELS"),
        "per_million_near": Param(0.25, unit="share of a million: sample totals this near it are "
                                             "CPM or TPM, normalized",
                                  constant="turbotab.core.methods.omics:PER_MILLION_NEAR"),
    },
    parts=(
        Part("block", grade="judgment", basis=Basis(
            reason="We read 20 assay-like columns or more as a feature block, so a clinical "
                   "table's handful of laboratory values is never read as one, and leave out of it "
                   "a column of two values or fewer, an indicator however large its code; "
                   "unsourced conventions.")),
        Part("names", grade="judgment", basis=Basis(
            reason="We leave out of the block a column whose whole name, or one whole word of "
                   "it, in any case, is on omics' own list: an identifier (SEQN, id, sample_id, "
                   f"subject_id, patient_id, {', '.join(n.lower() for n in OMICS_ONE_WORD_IDS)}, "
                   "or a name ending _id), a person's characteristic (age, sex, gender, bmi, and "
                   f"NHANES's {_listed(OMICS_NHANES_NAMES, 'and')}) or a time (visit, time, "
                   "timepoint, date, day, days, hour, hours, year, years, session). The list is "
                   "short on purpose and is not the roles "
                   "stage's: its covariate and identifier words, reused in round 2, also name "
                   "assay features (Met and Met-SO, methionine in Biocrates panels; the genes MET, "
                   "IDS and PIR), so a characteristic or covariate named otherwise (smoking, race, "
                   "income, a weight) is read by its values like any other column.")),
        Part("counts", grade="judgment", basis=Basis(
            reason="We read whole numbers reaching 100 as counts: genotype dosages (0, 1 and 2) "
                   "and scores stay below it; an unsourced convention.")),
        Part("intensities", grade="judgment", basis=Basis(
            reason="We read values spanning 100-fold that either reach 1,000 or are right-skewed "
                   "(median skewness 0.5) as unlogged: log2 intensities stay below about 40, a "
                   "log-normal feature with a 30% CV has skewness (e^s2 + 2)(e^s2 - 1)^0.5, about "
                   "0.93, and a log scale is near-symmetric. The cut-offs are unsourced "
                   "conventions.")),
        Part("estimated", grade="judgment", basis=Basis(
            reason="We ask about fractional values with a zero and the thousands each in two "
                   "columns or more, as GENOMICS_PACK §02 row 2 says to ask, unless every "
                   "sample's total is within 25% of a million, the pack's own tolerance for "
                   "TMM-scaled CPM (§02 row 4, measured on its fixtures); an internal pack section "
                   "is no source (CLAIMS_REGISTRY.md §2.4).")),
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
        Fires(f"{TESTS}::test_columns_of_three_values_or_more_are_read_as_counts_and_two_level_"
              "ones_are_left_out", truth="construction"),
        Silent(f"{TESTS}::test_a_block_of_two_level_columns_is_never_read_as_raw_counts",
               truth="construction"),
        Fires(f"{TESTS}::test_fractional_values_whose_totals_leave_the_packs_tolerance_of_a_"
              "million_are_asked_about", truth="construction"),
        Silent(f"{TESTS}::test_fractional_values_whose_totals_sit_within_the_packs_tolerance_of_a_"
               "million_are_normal", truth="construction"),
        Fires(f"{TESTS}::test_a_right_skewed_panel_below_the_thousands_reads_as_raw_intensities",
              truth="construction"),
        Silent(f"{TESTS}::test_a_symmetric_block_below_the_thousands_is_not_read_as_raw_"
               "intensities", truth="construction"),
        Silent(f"{TESTS}::test_a_block_in_the_thousands_that_spans_little_is_not_read_as_raw_"
               "intensities", truth="construction"),
        Silent(f"{TESTS}::test_a_clinical_handful_of_laboratory_columns_is_no_feature_block",
               truth="construction"),
        Fires(f"{TESTS}::test_an_identifier_and_an_age_beside_the_assay_columns_stay_out_of_the_"
              "block", truth="construction"),
        Silent(f"{TESTS}::test_an_assay_column_whose_name_is_no_identifier_characteristic_or_time_"
               "stays_in", truth="construction"),
        Fires(f"{TESTS}::test_each_name_the_reading_leaves_out_leaves_the_block_and_its_totals",
              truth="construction"),
        Silent(f"{TESTS}::test_methionine_and_the_genes_met_ids_and_pir_stay_in_the_block",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.omics:scale_reading",
                 "turbotab.core.methods.omics:candidate_columns",
                 "turbotab.core.methods.omics:_excluded_by_name",
                 "turbotab.core.methods.omics:_estimated_counts"),
    author=AUTHOR,
))


# ── D-058 · groups above the person ──────────────────────────────────────────────────────────
# Both sentences are the teaching entry's (``teaching.content.CLUSTERS``), card-facing plain words:
# the card is its "why", the methods register its "Adjust and cluster" option. The guard keeps its
# own copy of the number, ``estimand.FIXED_EFFECTS_LOGIT_ROWS``, until estimand.py reads it from
# here, so ``rows`` names no constant (the reference would say that constant reads it); until then
# a test holds both sentences to the rows per group at which the guard stops refusing
# (test_claims_data.py, the grouping entry's rows per group).

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
    regime=Regime("participants grouped by a column (sites, households, clinics). Under inference, "
                  "a numeric outcome's linear model, which has no incidental-parameter bias, or a "
                  "yes/no outcome, whose group size is read as the mean rows per group over the "
                  "analyzed rows' groups of two or more. An ordinal, multiclass or time-to-event "
                  "outcome is outside it: the evidence does not reach them, and the guard gives "
                  "them an intercept per group however small the groups. Under prediction, any "
                  "outcome",
                  requires=(f"{HERE}:names_a_grouping",
                            f"{HERE}:an_outcome_its_evidence_covers")),
    decision="set_clusters.adjust",
    options=("fixed_effects", "cluster_only", "none"),
    verdict="refused",
    params={"rows": Param(16, unit="rows per group")},
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
        Part("identity", grade="proved", graded_by=VERIFIER, basis=Basis(
            theorem="Least squares is linear in the outcome: for a fixed design X and weights W "
                    "(W = I for ordinary least squares), the estimate b(v) = (X'WX)^-1 X'Wv is a "
                    "linear function of v, so b(u + w) = b(u) + b(w). For a cut-off c, "
                    "min(y, c) + max(y - c, 0) = y for every y (c + (y - c) above c, y + 0 at or "
                    "below it), so on the same rows and design the coefficients of the part short "
                    "of the cut-off and of the part past it add up to the primary's exactly.")),
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

"""The inference area's claims (R0_PLAN.md R0b): the clustered intervals' few-clusters lines, the
proportional-odds table's cluster sandwich (claims backtest A-102), the trial's separation refusal
(B-281), the study design's default and a sample drawn by its outcome (B-211), a numeric exposure's
R² read as precision (B-416), the Cox fit's monotone likelihood (B-435), and the Results design's
tail share (D-086, ruled and not built).

``FEW_CLUSTERS`` is two numbers under one name: ten clusters, the fewest Li & Redden (2015)
simulated, in the trial core (``trials.few_clusters``), and thirty, inside Cameron & Miller's
(2015) few-clusters range, for the model families' cluster-robust intervals
(``inference.few_clusters``). Each module constant reads its own claim with ``claims.value``.

The regime predicates below read only the context the engine passes (a mapping, or an object
with those attributes) and recompute what they check from it; none calls the code it guards
(CLAIMS_REGISTRY.md §10, C-123). This module imports nothing from the engine.
"""
from __future__ import annotations

from typing import Any, Mapping

from turbotab.core.claims import Basis, Claim, Fires, Param, Part, Regime, Silent, register

AUTHOR = "R0b-inference builder"
HERE = "turbotab.core.claims.inference"
TESTS = "turbotab/core/tests/test_claims_inference.py"
ACCEPTANCE = "turbotab/core/tests/acceptance"


def _get(ctx: Any, key: str, default: Any = None) -> Any:
    if isinstance(ctx, Mapping):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


# ── the regime predicates ────────────────────────────────────────────────────


def rows_repeat(ctx: Any) -> bool:
    """Some unit holds more than one of the rows: fewer distinct unit codes than rows."""
    codes = _get(ctx, "cluster_codes")
    if codes is None:
        return False
    values = [int(c) for c in codes]
    return len(set(values)) < len(values)


def proportional_odds_cluster_sandwich(ctx: Any) -> bool:
    """A proportional-odds model whose intervals are the cluster sandwich over the units' summed
    scores with the G/(G − 1) correction, on t(G − 1) (CR1)."""
    return _get(ctx, "model") == "proportional_odds" and _get(ctx, "covariance") == "CR1"


def cluster_randomized_trial(ctx: Any) -> bool:
    """A cluster-randomized trial. Its analyses here each carry a small-sample correction (the
    mixed model's Kenward–Roger, the estimating equations' Kauermann–Carroll, Fay–Graubard or
    Mancl–DeRouen sandwich), so no second predicate is needed for one."""
    return _get(ctx, "trial_design") == "cluster_randomized_trial"


def randomized_trial(ctx: Any) -> bool:
    return _get(ctx, "trial_design") in ("parallel_trial", "cluster_randomized_trial")


def yes_no_outcome(ctx: Any) -> bool:
    """Exactly two outcome values are seen."""
    values = _get(ctx, "outcome_values")
    if values is None:
        return False
    return len({float(v) for v in values}) == 2


def design_unanswered(ctx: Any) -> bool:
    return _get(ctx, "design") is None


def drawn_by_outcome_analyzed_as_observational(ctx: Any) -> bool:
    """A case-control sample or matched sets were declared, and the data are analyzed as an
    observational study (the design refusal's first exit)."""
    return (_get(ctx, "design") == "observational"
            and _get(ctx, "design_declared") in ("case_control", "matched_sets"))


def numeric_exposure(ctx: Any) -> bool:
    return _get(ctx, "exposure_kind") == "continuous"


def cox_partial_likelihood(ctx: Any) -> bool:
    return _get(ctx, "model") == "cox"


# ── the parameters shared by more than one claim ─────────────────────────────

# Thirty clusters: one number, held here once, which the proportional-odds rule states too.
THIRTY_CLUSTERS = Param(30, unit="clusters", constant="turbotab.core.models.inference:FEW_CLUSTERS")


# ── the claims ───────────────────────────────────────────────────────────────

INFERENCE_FEW_CLUSTERS = register(Claim(
    id="inference.few_clusters",
    kind="parameter",
    methods=("Fewer than {few_clusters} clusters are few: an interval that rests on the units' "
             "summed residuals or scores then rests on few of them, inside the range Cameron & "
             "Miller give for few clusters, from under twenty to under fifty."),
    goal=("inference",),
    regime=Regime("rows that repeat within units, under an interval that rests on unit-level sums "
                  "(a cluster sandwich) or on the units' own variance (a random intercept)",
                  requires=(f"{HERE}:rows_repeat",)),
    params={"few_clusters": THIRTY_CLUSTERS},
    parts=(
        Part("value", grade="judgment", basis=Basis(reason=(
            "Cameron & Miller (2015, J Hum Resour 50:317, §VI) write that 'few' may range from "
            "less than 20 clusters to less than 50 clusters in the balanced case, and more with "
            "unbalanced clusters. 30 is our pick inside that range, not a number they give, and "
            "it is not raised for unequal cluster sizes. Their paper is not in the citation "
            "registry, so no part citing it can be graded sourced yet."))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "One line serves several rules: the linear family's CR2 concern, the proportional-odds "
            "family's CR1 concern, the mixed model's and the Cox model's concerns, and the ranking "
            "of the mixed model above the estimating equations; each is its own rule, and only "
            "the proportional-odds one is registered (ordinal.cluster_robust_intervals)."))),
    ),
    enforced_by=("turbotab.core.models.inference:_cluster_concerns",
                 "turbotab.core.models.ordinal:_cr1_concerns",
                 "turbotab.core.models.repeated:mixed_table",
                 "turbotab.core.models.repeated:Mixed.assess",
                 "turbotab.core.models.repeated:GEE.assess",
                 "turbotab.core.models.survival:cox_table"),
    author=AUTHOR,
))

TRIALS_FEW_CLUSTERS = register(Claim(
    id="trials.few_clusters",
    kind="rule",
    card=("the intervals rest on the small-sample correction, and the fewest clusters Li & "
          "Redden (2015) studied was {few_clusters}."),
    methods=("fewer than {few_clusters} clusters are named among the concerns (the fewest Li & "
             "Redden 2015 studied), beside the missing-outcome notice, neither displacing the "
             "other"),
    goal=("inference",),
    regime=Regime("a cluster-randomized trial whose intervals rest on a small-sample correction: "
                  "the estimating equations' corrected sandwich (what Li & Redden simulated, for a "
                  "yes/no outcome) or the mixed model's Kenward–Roger intervals",
                  requires=(f"{HERE}:cluster_randomized_trial",)),
    decision="trial_effect.few_clusters",
    options=("gee_corrected", "mixed_kr"),
    verdict="noted",
    params={"few_clusters": Param(10, unit="clusters",
                                  constant="turbotab.core.methods.trials:FEW_CLUSTERS")},
    parts=(
        Part("value", grade="judgment", basis=Basis(cites=("li2015gee",), reason=(
            "10 is the fewest clusters Li & Redden (2015) simulated, so below it their evidence "
            "for the corrected sandwich's coverage does not reach. It marks where their evidence "
            "ends, not a count at which coverage is known to fail."))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "Li & Redden simulated the estimating equations with a yes/no outcome. We name the "
            "same concern for the mixed model's Kenward–Roger intervals on a numeric outcome, "
            "which their study does not cover."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_e2_trials.py::"
              "test_a_few_clusters_concern_never_displaces_the_missing_outcome_notice",
              truth="construction"),
        Fires(f"{TESTS}::test_a_cluster_trial_of_eight_clusters_names_the_few_clusters",
              truth="construction"),
        Silent(f"{TESTS}::test_a_cluster_trial_of_twelve_clusters_names_no_few_clusters",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.trials:estimate",),
    author=AUTHOR,
))

ORDINAL_CLUSTER_ROBUST_INTERVALS = register(Claim(
    id="ordinal.cluster_robust_intervals",
    kind="rule",
    card=("With this few clusters, and more so when their sizes differ or a column varies only "
          "between them, such intervals can still be too narrow"),
    methods=("Below {few_clusters} units, the clustered proportional-odds table says that its "
             "intervals, the cluster sandwich it computes, can still be too narrow, more so when "
             "the units differ in size or a column varies only between them."),
    goal=("inference",),
    regime=Regime("a proportional-odds model whose rows repeat within units, its intervals the "
                  "cluster sandwich over the units' summed scores with the G/(G − 1) correction, "
                  "on t(G − 1) (CR1)",
                  requires=(f"{HERE}:proportional_odds_cluster_sandwich",)),
    decision="ordinal_table.intervals",
    verdict="noted",
    params={"few_clusters": THIRTY_CLUSTERS},
    parts=(
        Part("method", grade="judgment", basis=Basis(reason=(
            "The concern names the method the table computes, CR1 with G/(G − 1) on t(G − 1), "
            "where it had borrowed the linear family's CR2 with Bell–McCaffrey degrees of "
            "freedom (claims backtest A-102). "
            "test_2_cluster_robust_intervals_match_statsmodels_sandwich_on_t_g_minus_1 checks the "
            "computation against statsmodels; the Fires test checks the words."),
            tests=(f"{ACCEPTANCE}/test_wp12a_ordinal.py::test_2_clustered_intervals_say_cr1_on_t_"
                   "g_minus_1_and_that_few_units_can_make_them_narrow",))),
        Part("direction", grade="judgment", basis=Basis(reason=(
            "t(G − 1) is the least allowance Cameron & Miller (2015, §VI) ask for; it lessens "
            "but does not remove the sandwich's downward bias, the more so for units of unequal "
            "size (MacKinnon & Webb 2017, J Appl Econom 32:233). In the Fires test's known-truth "
            "simulation (12 units of lognormal size, a column varying only between them) the 95% "
            "intervals cover β in well under 95% of 1,000 data sets."),
            tests=(f"{ACCEPTANCE}/test_wp12a_ordinal.py::test_2_clustered_intervals_say_cr1_on_t_"
                   "g_minus_1_and_that_few_units_can_make_them_narrow",))),
        Part("value", grade="judgment", basis=Basis(reason=(
            "The line is inference.few_clusters' 30, chosen for cluster-robust intervals in "
            "general, not tuned for the proportional-odds model. Above it nothing is said, "
            "though very unequal units can still leave the sandwich too narrow."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_wp12a_ordinal.py::test_2_clustered_intervals_say_cr1_on_t_g_"
              "minus_1_and_that_few_units_can_make_them_narrow", truth="construction"),
        Silent(f"{TESTS}::test_an_ordinal_table_over_sixty_units_says_nothing_of_narrow_intervals",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.models.ordinal:_cr1_concerns",
                 "turbotab.core.models.ordinal:ordinal_table"),
    author=AUTHOR,
))

TRIALS_SEPARATION = register(Claim(
    id="trials.separation",
    kind="rule",
    methods=("refused when the logistic model's risk for any group of people runs to 0% or 100% "
             "(the arm and the covariates predict exactly whether they had the event): an arm, "
             "everyone (complete separation), or a group a linear program finds or cannot rule "
             "out (for the estimating equations, any fitted risk within {risk_floor} of either); "
             "a partly separated fit is not read at its limit"),
    goal=("inference",),
    regime=Regime("a randomized trial with a yes/no outcome: the standardized risks of a logistic "
                  "model with Ye et al.'s variance, or a cluster trial's estimating equations",
                  requires=(f"{HERE}:randomized_trial", f"{HERE}:yes_no_outcome")),
    decision="trial_effect.separation",
    options=("standardized_risk", "gee_corrected"),
    verdict="refused",
    params={"risk_floor": Param(1e-10, unit="risk from 0 or 1",
                                constant="turbotab.core.methods.trials:RISK_FLOOR")},
    parts=(
        Part("direction", grade="judgment", basis=Basis(reason=(
            "We refuse rather than report. At the fit's limit the standardized risks are finite "
            "(Lesaffre & Albert 1989, J R Stat Soc B 51:109), the separated people at their own "
            "outcome under every arm, but reading that fit was withdrawn in stage 1 (claims "
            "backtest B-281, its sixth review): ties to float rounding made it report a wrong "
            "risk difference, by up to 1.06 standard errors. Until a certificate robust to "
            "rounding exists, refusing is the honest fallback, with fewer covariates and "
            "describing the outcomes as its exits."))),
        Part("reason", grade="judgment", basis=Basis(reason=(
            "The refusal says the coefficients have no finite values, not that no finite estimate "
            "exists: the risk difference is finite at the limit (e.g. 0/50 against 20/50 is "
            "0.40). Where an arm alone separates, Ye et al.'s variance, built from each arm's "
            "spread of outcomes around its fitted risks, is zero there, so no interval is given."),
            tests=(f"{ACCEPTANCE}/test_e2_trials.py::test_a_group_with_every_event_is_refused_"
                   "naming_it_and_an_empty_arm_says_why",))),
        Part("detection", grade="judgment", basis=Basis(reason=(
            "The separated people are found by one linear program (Albert & Anderson 1984, "
            "Biometrika 71:1; Konis 2007, as R's detectseparation does), solved by HiGHS on an "
            "orthonormal basis of the standardized columns. Where its answer is not decided to "
            "the module's tolerances, the effect is refused with the plain separation reason. "
            "The tolerances were measured on the stage-1 reviews' generators, not derived."))),
        Part("risk_floor", grade="judgment", basis=Basis(reason=(
            "A fitted risk within 1e-10 of 0 or 1 is read as a sign of separation: for a "
            "parallel trial it sends the fit to the linear program, and for the estimating "
            "equations it refuses at once. The value is a float64 convention, not tuned."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_e2_trials.py::test_a_group_with_every_event_is_refused_naming_"
              "it_and_an_empty_arm_says_why", truth="independent:statsmodels"),
        Fires(f"{TESTS}::test_a_cluster_trial_whose_treated_clusters_all_had_the_event_is_refused",
              truth="construction"),
        Silent(f"{TESTS}::test_a_trial_no_one_is_separated_in_is_reported_as_statsmodels_fits_it",
               truth="independent:statsmodels"),
        Silent(f"{TESTS}::test_a_cluster_trial_of_twelve_clusters_names_no_few_clusters",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.trials:_fit_risks",
                 "turbotab.core.methods.trials:_separated_people",
                 "turbotab.core.methods.trials:_separated_refusal",
                 "turbotab.core.methods.trials:_logistic",
                 "turbotab.core.methods.trials:fit_gee"),
    author=AUTHOR,
))

DESIGNS_OBSERVATIONAL_DEFAULT = register(Claim(
    id="designs.observational_default",
    kind="rule",
    card=("Read as an observational study: people were observed as they were, not assigned or "
          "sampled by their outcome. Change it if they were randomized, sampled by the outcome "
          "or matched."),
    methods=("Until the study-design question is answered, the study is read as observational, "
             "stated for the person and listed in Your question's Confirm sweep."),
    goal=("prediction", "inference"),
    regime=Regime("a project whose study-design question is unanswered",
                  requires=(f"{HERE}:design_unanswered",)),
    decision="set_design",
    options=("observational",),
    verdict="recommended",
    parts=(
        Part("default", grade="judgment", basis=Basis(reason=(
            "We read an unanswered design as observational, the design of most tables brought "
            "here, and say so where the person confirms it. It is a default the person owns, "
            "not a reading of the data."))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "It is not safe under every goal: a sample drawn by its outcome that no one declares "
            "is analyzed as observational, since the prevalence and matched-sets detectors do not "
            "yet route to the design question (claims backtest B-211; INBOX). The statement "
            "names the designs it would be wrong for."))),
    ),
    tests=(
        Fires("turbotab/core/tests/test_ordering_fixes.py::"
              "test_unanswered_the_design_is_stated_observational_and_holds_nothing_back",
              truth="construction"),
        Silent("turbotab/core/tests/test_ordering_fixes.py::"
               "test_an_observational_answer_is_recorded_and_answers_the_question",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.designs:design_of", "turbotab.core.interview:_design_gate"),
    author=AUTHOR,
))

DESIGNS_SAMPLED_BY_OUTCOME = register(Claim(
    id="designs.sampled_by_outcome",
    kind="rule",
    card=("no prevalence, absolute risk or risk difference, and no risks predicted, since the "
          "cases and the controls were sampled separately"),
    methods=("A case-control sample or matched sets analyzed as an observational study, through "
             "the design refusal's first exit, keep the design declared: the record and the "
             "methods name it and say odds ratios only, the marginal risk difference and ratio "
             "are refused with the odds ratio as the exit, the estimand card reads no prevalence "
             "from the case share, and Predict is refused."),
    goal=("prediction", "inference"),
    regime=Regime("a case-control sample or matched sets, declared, and analyzed as an "
                  "observational study through the design refusal's first exit",
                  requires=(f"{HERE}:drawn_by_outcome_analyzed_as_observational",)),
    decision="set_design.declared",
    verdict="refused",
    parts=(
        Part("direction", grade="judgment", basis=Basis(cites=("prentice1979casecontrol",),
                                                        reason=(
            "Drawn by the outcome, the case share is set by the study: the odds ratio is the "
            "population's (Prentice & Pyke 1979), but the prevalence, the fitted probabilities "
            "and the standardized risk difference are not. The Fires test's known-truth "
            "population shows each."),
            tests=("turbotab/core/tests/test_ordering_fixes.py::test_a_sample_drawn_by_its_"
                   "outcome_analyzed_as_observational_keeps_odds_ratios_only",))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "Matched sets are analyzed by the observational lane's logistic model, whose odds "
            "ratios are not conditioned on the sets; we say so in the record rather than refuse "
            "them. An unmatched sample's probabilities could be recalibrated to a stated "
            "prevalence, which Predict does not do yet."))),
    ),
    tests=(
        Fires("turbotab/core/tests/test_ordering_fixes.py::test_a_sample_drawn_by_its_outcome_"
              "analyzed_as_observational_keeps_odds_ratios_only", truth="independent:statsmodels"),
        Silent(f"{TESTS}::test_a_trial_analyzed_as_observational_keeps_its_risk_difference_and_"
               "predict", truth="construction"),
    ),
    enforced_by=("turbotab.core.designs:sampled_by_outcome",
                 "turbotab.core.decisions:_purpose_reads_the_design",
                 "turbotab.core.estimand:_estimand_measure_is_fitted",
                 "turbotab.core.estimand:estimand_card",
                 "turbotab.core.voice:_set_design"),
    author=AUTHOR,
))

CAUSAL_NUMERIC_EXPOSURE_R2 = register(Claim(
    id="causal.numeric_exposure_r2",
    kind="rule",
    methods=("A numeric exposure is estimated whatever share of its variance the covariates "
             "explain. From a cross-fitted share of {r2_line}, the positivity row names how many "
             "times wider the standard error is, a matter of precision, not of positivity, and "
             "says the interval counts that wider variance but not a bias from models of the "
             "outcome and of the exposure that misfit; the exposure's positivity is a stated "
             "assumption."),
    goal=("inference",),
    regime=Regime("a numeric exposure under Estimate an effect, by the partially linear model "
                  "with cross-fitted models of the outcome and of the exposure given the "
                  "covariates", requires=(f"{HERE}:numeric_exposure",)),
    decision="set_causal.positivity",
    verdict="noted",
    params={"r2_line": Param(0.9, unit="share of the exposure's variance",
                             constant="turbotab.core.models.causal:CONTINUOUS_R2_LINE")},
    parts=(
        Part("direction", grade="judgment", basis=Basis(reason=(
            "By Frisch–Waugh the partially linear model rests on the exposure's variance left "
            "once the covariates are known, so its standard error is about √(1/(1 − R²)) times "
            "what it would be were the exposure unrelated to them, and its interval counts that. "
            "A high R² is precision, not a positivity violation: the backtest's simulation "
            "(claims backtest B-416) covered at 0.94–0.95 with R² 0.95–0.99 and correctly "
            "specified models."),
            tests=("turbotab/core/tests/test_causal_lane.py::test_a_numeric_exposure_s_"
                   "positivity_is_stated_and_its_r2_read_as_precision",))),
        Part("value", grade="judgment", basis=Basis(reason=(
            "0.9, a variance inflation factor of 10, is Kutner, Nachtsheim & Neter's (2004, "
            "§10.5) collinearity rule of thumb, used only as where the widening is worth naming. "
            "It says nothing of positivity, and nothing turns on it but a sentence."))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "No R² guards the bias a misfit in the models of the outcome and of the exposure "
            "brings, which a small remainder magnifies; with random-forest models that bias "
            "showed at R² 0.86–0.88, below the line (B-416). The sentence says the interval does "
            "not count it."))),
    ),
    tests=(
        Fires("turbotab/core/tests/test_causal_lane.py::test_a_numeric_exposure_s_positivity_is_"
              "stated_and_its_r2_read_as_precision", truth="construction"),
        Fires(f"{TESTS}::test_an_exposure_explained_to_ninety_one_percent_names_its_widening",
              truth="construction"),
        Silent(f"{TESTS}::test_an_exposure_explained_to_eighty_five_percent_names_no_"
               "widening", truth="construction"),
    ),
    enforced_by=("turbotab.core.causal:assumption_card", "turbotab.core.models.causal:variation"),
    author=AUTHOR,
))

SURVIVAL_MONOTONE_LIKELIHOOD = register(Claim(
    id="survival.monotone_likelihood",
    kind="rule",
    methods=("A Cox fit whose partial likelihood has no maximum is refused, with no interval or "
             "p-value: where a column, or a direction along which the information falls below "
             "{flat} of its largest eigenvalue, orders every risk set (the rows with the event "
             "have the lowest, or the highest, value at every event time, as when a group has no "
             "events); where a coefficient runs past {diverged} standard deviations of its "
             "column; or where the information falls below {singular} of its largest "
             "eigenvalue at the estimate."),
    goal=("inference",),
    regime=Regime("a Cox proportional-hazards model fit by Newton's method on standardized "
                  "columns, with Efron's ties, unweighted or with case weights",
                  requires=(f"{HERE}:cox_partial_likelihood",)),
    decision="cox_table.intervals",
    verdict="refused",
    params={
        "diverged": Param(20.0, unit="standard deviations of the column",
                          constant="turbotab.core.models.survival:DIVERGED"),
        "flat": Param(1e-8, unit="share of the information's largest eigenvalue",
                      constant="turbotab.core.models.survival:FLAT"),
        "singular": Param(1e-12, unit="share of the information's largest eigenvalue",
                          constant="turbotab.core.models.survival:SINGULAR"),
    },
    parts=(
        Part("direction", grade="judgment", basis=Basis(reason=(
            "Where a column orders every risk set, the partial likelihood rises without end along "
            "it and has no finite maximum (monotone likelihood: Bryson & Johnson 1981, Biometrika "
            "68:703; Heinze & Schemper 2001, Biometrics 57:114). The Fires test shows it for each "
            "fixture by a profile likelihood written out by hand."),
            tests=(f"{ACCEPTANCE}/test_wp12b_cox_mixed_gee.py::test_3_a_group_with_no_events_is_"
                   "refused_and_one_with_only_events_is_not",))),
        Part("detection", grade="judgment", basis=Basis(reason=(
            "Newton's decrement can pass its tolerance before any coefficient reaches 20 SDs, "
            "since the information decays along an ordering as fast as the score (B-435: a log "
            "hazard ratio near −36 with p = 0). So a fit whose information has an eigenvalue "
            "below 1e-8 of its largest, or that did not converge, is checked for a column or a "
            "flat direction that orders the risk sets. 20, 1e-8 and 1e-12 are numerical "
            "conventions, not tuned against a reference."))),
        Part("wording", grade="judgment", basis=Basis(reason=(
            "The refusal says the rows with the event have the lowest or highest value at every "
            "event time. A group in which every row has the event is no monotone likelihood when "
            "its events fall among others', and is estimated: the Fires test matches lifelines "
            "there."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_wp12b_cox_mixed_gee.py::test_3_a_group_with_no_events_is_refused_"
              "and_one_with_only_events_is_not", truth="construction"),
        Silent(f"{TESTS}::test_a_cox_group_with_two_events_is_estimated_as_lifelines_fits_it",
               truth="independent:lifelines"),
    ),
    enforced_by=("turbotab.core.models.survival:cox_fit", "turbotab.core.models.survival:_ordering",
                 "turbotab.core.models.survival:no_maximum",
                 "turbotab.core.models.survival:cox_table",
                 "turbotab.core.models.survival:survey_cox_table"),
    author=AUTHOR,
))

# Claims backtest D-086: a ruling in the Results design (BEATS_MODELS_RESULTS.md §6, E23), not
# built. Its code and tests are named as they will be; until they exist it is proposed and
# renders nothing.
RESULTS_TAIL_SHARE = register(Claim(
    id="results.tail_share",
    kind="rule",
    methods=("For a least-squares estimate on an outcome whose lens names a cut-off, the "
             "coefficient splits exactly into the part past the cut-off and the part short of "
             "it, which add up to the primary's. The estimate is read as mostly the tail only "
             "when the coefficient of the tail part less the share of rows on the tail's side "
             "times the outcome has the primary's sign and its interval excludes zero; never "
             "against one half."),
    goal=("inference",),
    regime=Regime("a least-squares estimate under Estimate an effect, on an outcome whose lens "
                  "names a cut-off and which side of it is the tail, the primary's own interval "
                  "excluding zero"),
    decision="results.tail_share",
    verdict="noted",
    parts=(
        Part("identity", grade="judgment", basis=Basis(reason=(
            "y = min(y, c) + max(y − c, 0), and least squares is linear in y, so the two parts' "
            "coefficients add up to the primary's exactly; the build's test holds the sum."))),
        Part("baseline", grade="judgment", basis=Basis(reason=(
            "Shifting every value by δ moves the tail part by δ times the share of values on the "
            "tail's side, so a uniform shift gives a tail share of about p; one half is no test "
            "where most people are past the cut-off (claims backtest D-086). p is a first-order "
            "baseline, so the build's test checks a uniform shift reads throughout at about its "
            "nominal rate where most rows are past the cut-off and where few are."))),
    ),
    tests=(
        Fires("turbotab/core/tests/test_tail_share.py::test_a_tail_effect_reads_mostly_the_tail",
              truth="construction"),
        Silent("turbotab/core/tests/test_tail_share.py::test_a_uniform_shift_reads_throughout",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.models.tail_share:tail_share",),
    ruled="docs/turbotab-next/calm/BEATS_MODELS_RESULTS.md §6 (BE23, E23; claims backtest D-086)",
    author=AUTHOR,
))

CLAIMS: tuple[Claim, ...] = (
    INFERENCE_FEW_CLUSTERS, TRIALS_FEW_CLUSTERS, ORDINAL_CLUSTER_ROBUST_INTERVALS,
    TRIALS_SEPARATION, DESIGNS_OBSERVATIONAL_DEFAULT, DESIGNS_SAMPLED_BY_OUTCOME,
    CAUSAL_NUMERIC_EXPOSURE_R2, SURVIVAL_MONOTONE_LIKELIHOOD, RESULTS_TAIL_SHARE,
)

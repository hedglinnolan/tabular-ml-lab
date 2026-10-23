"""The inference area's claims (R0_PLAN.md R0b): the clustered intervals' few-clusters lines, the
proportional-odds table's cluster sandwich (claims backtest A-102), the trial's separation refusal
(B-281), the study design's default and a sample drawn by its outcome (B-211), a numeric exposure's
R² read as precision (B-416), the Cox fit's monotone likelihood (B-435), and the Results design's
tail share (D-086, ruled and not built).

``FEW_CLUSTERS`` is two numbers under one name: ten clusters, the fewest Li & Redden (2015)
simulated, in the trial core (``trials.few_clusters`` for the estimating equations' corrected
sandwich on a 0/1 outcome, the estimator and outcome they studied, carried by judgment past the
two unadjusted arms they simulated; ``trials.few_clusters_extended`` for the mixed model and a
numeric outcome), and thirty, inside Cameron & Miller's (2015) few-clusters range, for the model families'
cluster-robust intervals (``inference.few_clusters``). Each module constant reads its own claim
with ``claims.value``.

The regime predicates below read only the context the engine passes (a mapping, or an object
with those attributes) and recompute what they check from it; none calls the code it guards
(CLAIMS_REGISTRY.md §10, C-123). This module imports nothing from the engine.
"""
from __future__ import annotations

from typing import Any, Mapping

from turbotab.core.claims import Basis, Claim, Fires, Param, Part, Regime, Silent, register

AUTHOR = "R0b-inference builder"
# Who graded a part above judgment, on the evidence the package's verifier gave (§2.4).
VERIFIER = "R0b-inference verifier, 2026-10-16"
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


def estimating_equations(ctx: Any) -> bool:
    """The trial is analyzed by the estimating equations. The trial core gives them a
    small-sample corrected sandwich always (Kauermann–Carroll, Fay–Graubard or Mancl–DeRouen)."""
    return _get(ctx, "method") == "gee"


def zero_one_outcome(ctx: Any) -> bool:
    """A yes/no outcome: some outcome value is seen, and every one is 0 or 1."""
    values = _get(ctx, "outcome_values")
    if values is None:
        return False
    seen = {float(v) for v in values}
    return bool(seen) and seen <= {0.0, 1.0}


def beyond_li_redden(ctx: Any) -> bool:
    """A cluster-randomized trial analyzed by the mixed model, or by the estimating equations on
    an outcome that is not 0/1: an estimator or an outcome Li & Redden (2015) did not study."""
    return cluster_randomized_trial(ctx) and not (estimating_equations(ctx)
                                                  and zero_one_outcome(ctx))


def randomized_trial(ctx: Any) -> bool:
    return _get(ctx, "trial_design") in ("parallel_trial", "cluster_randomized_trial")


def design_unanswered(ctx: Any) -> bool:
    return _get(ctx, "design") is None


def drawn_by_outcome(ctx: Any) -> bool:
    """A case-control sample or matched sets were declared: the design answered as one (and
    refused), or the data analyzed as an observational study with it kept as declared (the design
    refusal's first exit)."""
    by_outcome = ("case_control", "matched_sets")
    design = _get(ctx, "design")
    return design in by_outcome or (design == "observational"
                                    and _get(ctx, "design_declared") in by_outcome)


def numeric_exposure(ctx: Any) -> bool:
    return _get(ctx, "exposure_kind") == "continuous"


def cox_partial_likelihood(ctx: Any) -> bool:
    return _get(ctx, "model") == "cox"


def least_squares_with_a_cut_off(ctx: Any) -> bool:
    """A least-squares estimate (ordinary or weighted) on an outcome whose lens names a finite
    cut-off and which side of it is the tail: the hypotheses of the tail split's identity."""
    cut = _get(ctx, "cut_off")
    finite = (isinstance(cut, (int, float)) and not isinstance(cut, bool)
              and cut == cut and abs(cut) != float("inf"))
    return (_get(ctx, "estimator") == "least_squares" and finite
            and _get(ctx, "tail") in ("above", "below"))


def pooled_over_copies(ctx: Any) -> bool:
    """A table pooled by Rubin's rules over completed copies (``copies``: how many, one or more;
    one, the data's own single copy, pools to its own Wald intervals)."""
    copies = _get(ctx, "copies")
    return isinstance(copies, int) and not isinstance(copies, bool) and copies >= 1


def separated_in_a_copy(ctx: Any) -> bool:
    """Some coefficient separates the outcome in some completed copy (``separated``: the names of
    the coefficients a copy's linear program finds infinite, over every copy)."""
    separated = _get(ctx, "separated")
    return separated is not None and len(separated) > 0


def primary_interval_excludes_zero(ctx: Any) -> bool:
    """The primary estimate's interval, ``(low, high)``, lies wholly above or wholly below zero."""
    interval = _get(ctx, "primary_interval")
    if interval is None or len(interval) != 2:
        return False
    low, high = (float(v) for v in interval)
    return low > 0 or high < 0


# ── the parameters shared by more than one claim ─────────────────────────────

# Ten clusters: the trial core's one number, which both its few-clusters rules state.
TEN_CLUSTERS = Param(10, unit="clusters", constant="turbotab.core.methods.trials:FEW_CLUSTERS")
# Thirty clusters: one number, held here once, which the proportional-odds rule states too.
THIRTY_CLUSTERS = Param(30, unit="clusters", constant="turbotab.core.models.inference:FEW_CLUSTERS")


# ── the claims ───────────────────────────────────────────────────────────────

INFERENCE_FEW_CLUSTERS = register(Claim(
    id="inference.few_clusters",
    kind="parameter",
    methods=("Fewer than {few_clusters} clusters are few: an interval that rests on the units' "
             "summed residuals or scores then rests on few of them, inside the range Cameron & "
             "Miller give for few clusters, from under twenty to under fifty when the clusters "
             "are of equal size, and more when they are not."),
    goal=("inference",),
    regime=Regime("rows that repeat within units, under an interval that rests on unit-level sums "
                  "(a cluster sandwich: the linear family's CR2, the proportional-odds family's "
                  "CR1, the Cox model's Lin–Wei), and, read by judgment, the mixed model's "
                  "between-unit variance and its ranking above the estimating equations",
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
            "the proportional-odds one is registered (ordinal.cluster_robust_intervals). Cameron "
            "& Miller's range is for cluster-robust (sandwich) inference. The mixed model's "
            "concern, that its between-unit variance rests on few units (which cites Luke 2017), "
            "and the ranking that puts the mixed model first below the line, read the same 30 by "
            "judgment: their evidence does not cover a random intercept's variance."))),
    ),
    enforced_by=("turbotab.core.models.inference:_cluster_concerns",
                 "turbotab.core.models.ordinal:_cr1_concerns",
                 "turbotab.core.models.repeated:mixed_table",
                 "turbotab.core.models.repeated:Mixed.assess",
                 "turbotab.core.models.repeated:GEE.assess",
                 "turbotab.core.models.survival:cox_table"),
    author=AUTHOR,
    origin="engine",
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
    regime=Regime("a cluster-randomized trial analyzed by the estimating equations with a "
                  "small-sample corrected sandwich, on a 0/1 outcome",
                  requires=(f"{HERE}:cluster_randomized_trial", f"{HERE}:estimating_equations",
                            f"{HERE}:zero_one_outcome")),
    decision="trial_effect.few_clusters",
    options=("gee_corrected",),
    verdict="noted",
    params={"few_clusters": TEN_CLUSTERS},
    parts=(
        Part("value", grade="sourced", graded_by=VERIFIER, basis=Basis(
            cites=("li2015gee",),
            where=("Li & Redden (2015), §2.1, the simulation study: 'We mimic the CRTs containing "
                   "10, 20, or 30 clusters (K) with 10, 20, 50, 100, or 150 observations in "
                   "average per cluster'; 'For simplicity but without the loss of "
                   "generalizability, we assume two intervention arms containing equal clusters "
                   "and no covariates are included'; its ICCs, 'low ICC (0.001, 0.01 and 0.05)'; "
                   "its cluster sizes, 'In our simulations, cv is at the range of 0 to 1'. The "
                   "abstract: 'the Kauermann and Carroll (KC)-correction can keep the test size "
                   "to nominal levels even when the number of clusters is as low as 10'. The "
                   "discussion: 'Another limitation is that we do not consider covariates other "
                   "than the treatment, which may be included in the real data analyses.'"),
            reason=(
                "10 is the fewest clusters Li & Redden (2015) simulated, in the design they "
                "simulated (two arms of equal cluster counts, no covariates, ICCs from 0.001 to "
                "0.05, a cluster-size coefficient of variation from 0 to 1), so below it their "
                "evidence for the corrected sandwich's coverage does not reach. It marks where "
                "their evidence ends, not a count at which coverage is known to fail."))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "The regime's predicates check the design, the estimator and a 0/1 outcome, not "
            "what Li & Redden held fixed: the trial core reads the same 10 for a covariate-"
            "adjusted analysis, for more than two arms or unequal cluster counts per arm, and "
            "for ICCs and size variations past theirs. We carry it there by judgment, on the "
            "authors' 'without the loss of generalizability'. Below 10 clusters their evidence "
            "is absent there as in their design, so the concern holds; from 10 the trial core "
            "names none, though their discussion warns that 'The addition of covariates will "
            "change the variance estimation and the degrees of freedom', so their finding that "
            "the corrected sandwich keeps its size does not cover an adjusted analysis. The "
            "claim's own Fires and Silent tests are such cases: adjusted for a covariate, with "
            "a cluster effect of SD 0.48 on the log odds (a latent ICC near 0.065), and 5 and 4 "
            "clusters per arm at 9. The trial core names the "
            "same concern, in the same words, for the mixed model's Kenward–Roger intervals and "
            "for the estimating equations on a numeric outcome, which their study does not "
            "cover; that is its own claim, by judgment (trials.few_clusters_extended)."))),
    ),
    tests=(
        Fires(f"{TESTS}::test_a_cluster_trial_of_eight_clusters_names_the_few_clusters",
              truth="construction"),
        Silent(f"{TESTS}::test_a_cluster_trial_of_twelve_clusters_names_no_few_clusters",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.trials:estimate",),
    fallback="trials.few_clusters_extended",
    author=AUTHOR,
    origin="engine",
))

TRIALS_FEW_CLUSTERS_EXTENDED = register(Claim(
    id="trials.few_clusters_extended",
    kind="rule",
    card=("the intervals rest on the small-sample correction, and the fewest clusters Li & "
          "Redden (2015) studied was {few_clusters}."),
    methods=("for the mixed model's Kenward–Roger intervals, and for the estimating equations on "
             "a numeric outcome, fewer than {few_clusters} clusters are named among the concerns "
             "as well, by judgment: Li & Redden (2015) studied the corrected sandwich with a "
             "yes/no outcome"),
    goal=("inference",),
    regime=Regime("a cluster-randomized trial analyzed by the mixed model's Kenward–Roger "
                  "intervals, or by the estimating equations' corrected sandwich on a numeric "
                  "outcome: an estimator or an outcome Li & Redden did not study",
                  requires=(f"{HERE}:beyond_li_redden",),
                  disjoint_from=("trials.few_clusters",)),
    decision="trial_effect.few_clusters",
    options=("mixed_kr", "gee_corrected"),
    verdict="noted",
    params={"few_clusters": TEN_CLUSTERS},
    parts=(
        Part("value", grade="judgment", basis=Basis(reason=(
            "We carry Li & Redden's fewest clusters, 10, to analyses they did not simulate. "
            "Their study says nothing of where the mixed model's Kenward–Roger intervals, or the "
            "corrected sandwich on a numeric outcome, begin to undercover; the line marks the "
            "same edge of evidence, not a count at which coverage is known to fail."))),
        Part("wording", grade="judgment", basis=Basis(reason=(
            "The concern's words are the trial core's one sentence for both claims: they say the "
            "intervals rest on the small-sample correction (Kenward–Roger's, or the corrected "
            "sandwich's) and name the fewest clusters Li & Redden studied, which is true of "
            "their study whatever the analysis, without saying that it covers this one."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_e2_trials.py::"
              "test_a_few_clusters_concern_never_displaces_the_missing_outcome_notice",
              truth="construction"),
        Fires(f"{TESTS}::test_a_numeric_cluster_trial_of_eight_clusters_names_the_few_clusters",
              truth="construction"),
        Silent(f"{TESTS}::test_a_numeric_cluster_trial_of_twelve_clusters_names_no_few_clusters",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.methods.trials:estimate",),
    author=AUTHOR,
    origin="engine",
))

ORDINAL_STATSMODELS_CR1 = (f"{TESTS}::test_an_ordinal_table_over_twelve_units_computes_"
                           "statsmodels_cr1_and_says_it_can_be_narrow")

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
                  requires=(f"{HERE}:proportional_odds_cluster_sandwich", f"{HERE}:rows_repeat")),
    decision="ordinal_table.intervals",
    verdict="noted",
    params={"few_clusters": THIRTY_CLUSTERS},
    parts=(
        Part("method", grade="certified", graded_by=VERIFIER, basis=Basis(reason=(
            "The concern names the method the table computes, CR1 with G/(G − 1) on t(G − 1), "
            "where it had borrowed the linear family's CR2 with Bell–McCaffrey degrees of "
            "freedom (claims backtest A-102). Over 12 units the Fires test against statsmodels "
            "finds the table's standard errors statsmodels' OrderedModel cluster sandwich, "
            "rescaled to G/(G − 1) alone, its intervals on t(11), and the line it names saying "
            "CR1 on t(11), not CR2 or Bell–McCaffrey; it fails when the table computes another "
            "sandwich under the CR1 label. The stage-1 Fires test holds the labels: the table's "
            "covariance is CR1 and its caption says t(11), and it fails when the linear family's "
            "concerns are put back. test_2_cluster_robust_intervals_match_statsmodels_sandwich_"
            "on_t_g_minus_1 checks the same computation over 150 units, where nothing is said of "
            "few units."),
            tests=(ORDINAL_STATSMODELS_CR1,
                   f"{ACCEPTANCE}/test_wp12a_ordinal.py::test_2_clustered_intervals_say_cr1_on_t_"
                   "g_minus_1_and_that_few_units_can_make_them_narrow"))),
        Part("direction", grade="certified", graded_by=VERIFIER, basis=Basis(reason=(
            "With this few units such intervals can still be too narrow: t(G − 1) is the least "
            "allowance Cameron & Miller (2015, §VI) ask for, and it lessens but does not remove "
            "the sandwich's downward bias. In the Fires test's known-truth simulation (12 units "
            "of lognormal size, a column varying only between them, β = 0.5 known) the 95% "
            "intervals cover β in fewer than 95% of 1,000 data sets by more than three Monte "
            "Carlo standard errors; statsmodels' OrderedModel with its cluster sandwich, "
            "rescaled to G/(G − 1) on t(11), covered 0.882 of 400 of the same fixtures (the "
            "verifier's check). The test holds the direction in that one design, 12 units "
            "unequal in size with a column that varies only between them; the card's comparison "
            "with other designs is the comparison part's."),
            tests=(f"{ACCEPTANCE}/test_wp12a_ordinal.py::test_2_clustered_intervals_say_cr1_on_t_"
                   "g_minus_1_and_that_few_units_can_make_them_narrow",))),
        Part("comparison", grade="judgment", basis=Basis(reason=(
            "The card's 'and more so when their sizes differ or a column varies only between "
            "them' compares designs that no test here compares: the coverage test has one "
            "design, units unequal in size with a column that varies only between them, and no "
            "fixture of equal sizes or of a column that varies within units. For unequal sizes "
            "it rests on MacKinnon & Webb (2017, J Appl Econom 32:233), who find cluster-robust "
            "tests over-reject more as the clusters' sizes grow more unequal; for a column that "
            "varies only between units, on our judgment that the sandwich then rests on one "
            "value of it per unit. MacKinnon & Webb is not in the citation registry, so no part "
            "citing it can be graded sourced yet."))),
        Part("value", grade="judgment", basis=Basis(reason=(
            "The line is inference.few_clusters' 30, chosen for cluster-robust intervals in "
            "general, not tuned for the proportional-odds model. Above it nothing is said, "
            "though very unequal units can still leave the sandwich too narrow."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_wp12a_ordinal.py::test_2_clustered_intervals_say_cr1_on_t_g_"
              "minus_1_and_that_few_units_can_make_them_narrow", truth="construction"),
        Fires(ORDINAL_STATSMODELS_CR1, truth="independent:statsmodels"),
        Silent(f"{TESTS}::test_an_ordinal_table_over_sixty_units_says_nothing_of_narrow_intervals",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.models.ordinal:_cr1_concerns",
                 "turbotab.core.models.ordinal:ordinal_table"),
    author=AUTHOR,
    origin="audit",
))

TRIALS_SEPARATION = register(Claim(
    id="trials.separation",
    kind="rule",
    methods=("refused when the logistic model's risk for any group of people runs to 0% or 100% "
             "(the arm and the covariates predict exactly whether they had the event): an arm, "
             "everyone (complete separation), or a group a linear program finds or cannot rule "
             "out; a partly separated fit is not read at its limit; the estimating equations "
             "start from the logistic model fitted without the clustering, on the raw columns, "
             "and from no other start, and where that start shows a sign of separation (no "
             "convergence, a singular information matrix, a coefficient past {coefficient_bound} "
             "on its column's own scale, or a fitted risk within {risk_floor} of 0 or 1), or "
             "their own fit puts a risk within {risk_floor} of 0 or 1, the program decides only "
             "the reason: separation where it finds anyone separated, and otherwise a start or a "
             "fit that did not settle within these bounds; the start's bound can so refuse a "
             "trial no one is separated in, as one with a covariate far from its zero whose "
             "start's intercept passes the bound"),
    goal=("inference",),
    regime=Regime("a randomized trial whose outcome is yes/no, coded 1 and 0 (some outcome "
                  "seen and each one 0 or 1, though all may be the same): the standardized risks "
                  "of a logistic model with Ye et al.'s variance, or a cluster trial's estimating "
                  "equations",
                  requires=(f"{HERE}:randomized_trial", f"{HERE}:zero_one_outcome")),
    decision="trial_effect.separation",
    options=("standardized_risk", "gee_corrected"),
    verdict="refused",
    params={
        "coefficient_bound": Param(25, unit="log odds per unit of a column, on its own scale",
                                   constant="turbotab.core.methods.trials:COEFFICIENT_BOUND"),
        "risk_floor": Param(1e-10, unit="risk from 0 or 1",
                            constant="turbotab.core.methods.trials:RISK_FLOOR"),
    },
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
            "spread of outcomes around its fitted risks, is zero there, so no interval is given; "
            "for the estimating equations the refusal names their corrected sandwich, built from "
            "each cluster's residuals around its fitted risks, in its place. Where the program "
            "finds no one separated, or cannot decide, the estimating equations give one of two "
            "refusals, as the code words them. Where their start refused, the refusal says that "
            "the logistic model fitted without the clustering, which they start from, did not "
            "settle within its bounds, and that this does not show anyone's risk runs to 0% or "
            "100%. Where their own fit put a risk within the floor, it says the fit puts "
            "someone's risk within the floor of 0% or 100%, said in words (one in so many), the "
            "edge at which its risks are not read, though no one was found separated (no group "
            "whose outcomes the arm and the covariates predict exactly). Neither says that no "
            "one is separated, since a program that cannot decide does not show it."),
            tests=(f"{ACCEPTANCE}/test_e2_trials.py::test_a_group_with_every_event_is_refused_"
                   "naming_it_and_an_empty_arm_says_why",
                   f"{TESTS}::test_a_cluster_trial_whose_treated_clusters_all_had_the_event_is_"
                   "refused",
                   f"{TESTS}::test_a_cluster_trial_adjusted_for_a_calendar_year_is_refused_as_"
                   "unsettled_not_separated",
                   f"{TESTS}::test_a_cluster_trial_tied_at_zero_up_to_noise_is_refused_as_"
                   "unsettled_not_separated",
                   f"{TESTS}::test_a_cluster_trial_whose_own_fit_reaches_the_floor_with_no_one_"
                   "separated_is_refused",
                   f"{TESTS}::test_a_share_is_said_as_one_in_so_many_in_words"))),
        Part("detection", grade="judgment", basis=Basis(reason=(
            "For a parallel trial, Newton's method runs on the raw columns first. When it does "
            "not converge, or converges with a sign of separation (a coefficient past the bound, "
            "a fitted risk within the floor), the separated people are found by one linear "
            "program (Albert & Anderson 1984, Biometrika 71:1; Konis 2007, as R's "
            "detectseparation does), solved by HiGHS on an orthonormal basis of the standardized "
            "columns; no one separated, the fit is redone on those columns. Where the program's "
            "answer is not decided to the module's tolerances, the effect is refused with the "
            "plain separation reason. The tolerances were measured on the stage-1 reviews' "
            "generators, not derived. The estimating equations start from the same Newton fit on "
            "the raw columns and from no other. Where it refuses, on those signs or on an "
            "information matrix that cannot be solved at one of its steps, the same program "
            "decides only which refusal is given: separation, naming the people, where it finds "
            "anyone separated, and otherwise the refusal that the start did not settle. Each "
            "Fires test names the trigger it meets. Treated clusters that all had the event, and "
            "smokers who all did, refuse on the singular information at a Newton step (the "
            "risks of the people the arm or smoking predicts run to 1), and the program finds "
            "them separated; replaced by a program that finds no one, both tests fail. A "
            "calendar year and a body temperature refuse on the bound, read on the intercept, "
            "and the program finds no one separated. A program that cannot decide is built on "
            "this path by the parallel tests' tied trial recast as a cluster trial (30 people at "
            "x = 0 up to noise of 1e-12 to 1e-8, whose outcomes, mixed in each arm and not "
            "ordered by x, show that no one is separated): the start stops with a coefficient "
            "past the bound and risks within the floor, the program returned None for each of "
            "its 12 trials on macOS (2026-10-16), and each is refused as unsettled; given the "
            "plain separation reason there instead, the test fails. No test builds a start that "
            "fails to converge alone on this path; the code gives that case the same refusal."),
            tests=(f"{TESTS}::test_a_cluster_trial_whose_treated_clusters_all_had_the_event_is_"
                   "refused",
                   f"{TESTS}::test_a_cluster_trial_whose_smokers_all_had_the_event_is_refused_"
                   "naming_smoking",
                   f"{TESTS}::test_a_cluster_trial_tied_at_zero_up_to_noise_is_refused_as_"
                   "unsettled_not_separated"))),
        Part("coefficient_bound", grade="judgment", basis=Basis(reason=(
            "A coefficient past the bound, in log odds per unit of its column, is a sign of "
            "separation, not a proof of it. It is not scale-free, and it is read on the intercept "
            "too: a covariate far from its zero (a calendar year, a body temperature) or on a "
            "narrow scale (a pH of 7.4 ± 0.02 with a log odds ratio of 60 per unit, about 1.2 per "
            "SD) puts a coefficient past it with no one separated. A parallel trial then asks the "
            "linear program, which rules separation out, and reports the fit (the Silent test on "
            "that pH, against statsmodels); with no bound the same fit is reported straight from "
            "Newton's method, so no test holds the bound itself there. The estimating equations' "
            "start refuses on it, and the program, finding no one separated, gives the refusal "
            "that the start did not settle: the Fires tests on a calendar year of 2021 ± 1.5 and "
            "a body temperature of 37 ± 0.5 °C over 16 clusters, where statsmodels' Logit "
            "converges and its GEE fits each trial, hold it there, since with the bound removed "
            "each of those trials is reported. The value is a convention, not tuned."),
            tests=(f"{TESTS}::test_a_cluster_trial_adjusted_for_a_calendar_year_is_refused_as_"
                   "unsettled_not_separated",
                   f"{TESTS}::test_a_cluster_trial_adjusted_for_a_body_temperature_is_refused_as_"
                   "unsettled_not_separated"))),
        Part("risk_floor", grade="judgment", basis=Basis(reason=(
            "A fitted risk within the floor of 0 or 1 is read as a sign of separation. For a "
            "parallel trial it sends a converged fit to the linear program, and, no one "
            "separated, to the refit on standardized columns, whose information float64 must "
            "resolve: the Fires test spies on the program for a fit whose only extreme risk is "
            "one person's at an outlying dose, and the Silent test finds it not asked where no "
            "risk is near 0 or 1; the reported effect is statsmodels' either way. The estimating "
            "equations read the floor twice, on their start and on their own fit, and the "
            "program decides only the reason each time. One treated person's dose of 18 puts "
            "their risk in statsmodels' Logit within the floor, with no one separated: the start "
            "reaches it, and the trial is refused as unsettled. A dose of 15.7 in another trial "
            "leaves every risk of statsmodels' Logit more than 1.5e-10 from 0 and 1, so the "
            "start settles, while its GEE puts that person's risk within 7e-11 of 1: the fit "
            "reaches the floor, and the trial is refused at that edge; with the dose at 12 it is "
            "reported as statsmodels fits it, the program not asked. With no floor both Fires "
            "tests fail. The fit's check has a separated branch no test reaches: data that "
            "separate anyone leave the start no finite fit within the bounds, so the start "
            "refuses first. The value is a float64 convention, not tuned."),
            tests=(f"{TESTS}::test_a_converged_fit_with_a_risk_at_float64_s_edge_is_checked_by_"
                   "the_linear_program",
                   f"{TESTS}::test_a_fit_with_no_risk_near_0_or_1_is_reported_without_the_linear_"
                   "program",
                   f"{TESTS}::test_a_cluster_trial_whose_start_reaches_the_floor_at_one_dose_is_"
                   "refused_as_unsettled",
                   f"{TESTS}::test_a_cluster_trial_whose_own_fit_reaches_the_floor_with_no_one_"
                   "separated_is_refused",
                   f"{TESTS}::test_the_same_trial_with_a_nearer_dose_is_reported_as_statsmodels_"
                   "fits_it"))),
        Part("scope", grade="judgment", basis=Basis(reason=(
            "The estimating equations are not given the parallel trial's treatment: their start "
            "stays the Newton fit on the raw columns, and the program decides only the reason "
            "for refusing. So a cluster trial with a yes/no outcome can still be refused though "
            "no one is separated in it: for a covariate far from its zero whose start's "
            "intercept passes the bound, and for one outlying covariate value at the floor; its "
            "reason says the start did not settle within its bounds, or the fit reached the "
            "floor, not that some group's risk runs to 0% or 100%. The package's verifier "
            "(2026-10-16), 16 clusters of 10 to 29 people, 10 seeds each: a calendar year of "
            "2021 ± 1.5 with no effect was refused 10 of 10, where statsmodels' exchangeable GEE "
            "fits each with fitted risks from 0.272 to 0.710; a body temperature of 37 ± 0.5 °C, "
            "0 of 10 with no effect, 8 of 10 at 0.8 log odds per °C and 10 of 10 at 1.2; a pH of "
            "7.4 ± 0.02 at 60 per unit, 10 of 10; one dose of 18 among doses of SD 1, 10 of 10 at "
            "the floor. Reporting them by "
            "starting from the fit on standardized columns was built and withdrawn: the round-3 "
            "verifier found that with a covariate far from zero relative to its spread (its mean "
            "over its SD from about 3e5 to 2e6) the estimating equations and their sandwich, "
            "computed on the raw columns, gave a corrected standard error up to 21% off, or a "
            "ValueError, where this refusal stands. Until they are fitted, and their sandwich "
            "taken, on standardized columns, refusing is the honest fallback."))),
    ),
    tests=(
        Fires(f"{ACCEPTANCE}/test_e2_trials.py::test_a_group_with_every_event_is_refused_naming_"
              "it_and_an_empty_arm_says_why", truth="independent:statsmodels"),
        Fires(f"{TESTS}::test_a_cluster_trial_whose_treated_clusters_all_had_the_event_is_refused",
              truth="construction"),
        Fires(f"{TESTS}::test_a_cluster_trial_whose_smokers_all_had_the_event_is_refused_naming_"
              "smoking", truth="construction"),
        Fires(f"{TESTS}::test_a_cluster_trial_adjusted_for_a_calendar_year_is_refused_as_unsettled_"
              "not_separated", truth="independent:statsmodels"),
        Fires(f"{TESTS}::test_a_cluster_trial_adjusted_for_a_body_temperature_is_refused_as_"
              "unsettled_not_separated", truth="independent:statsmodels"),
        Fires(f"{TESTS}::test_a_cluster_trial_whose_start_reaches_the_floor_at_one_dose_is_refused_"
              "as_unsettled", truth="independent:statsmodels"),
        Fires(f"{TESTS}::test_a_cluster_trial_whose_own_fit_reaches_the_floor_with_no_one_"
              "separated_is_refused", truth="independent:statsmodels"),
        Fires(f"{TESTS}::test_a_cluster_trial_tied_at_zero_up_to_noise_is_refused_as_unsettled_"
              "not_separated", truth="construction"),
        Fires(f"{TESTS}::test_a_converged_fit_with_a_risk_at_float64_s_edge_is_checked_by_the_"
              "linear_program", truth="independent:statsmodels"),
        Silent(f"{TESTS}::test_a_trial_no_one_is_separated_in_is_reported_as_statsmodels_fits_it",
               truth="independent:statsmodels"),
        Silent(f"{TESTS}::test_a_fit_with_no_risk_near_0_or_1_is_reported_without_the_linear_"
               "program", truth="independent:statsmodels"),
        Silent(f"{TESTS}::test_a_cluster_trial_of_twelve_clusters_names_no_few_clusters",
               truth="construction"),
        Silent(f"{TESTS}::test_the_same_trial_with_a_nearer_dose_is_reported_as_statsmodels_fits_"
               "it", truth="independent:statsmodels"),
    ),
    enforced_by=("turbotab.core.methods.trials:_fit_risks",
                 "turbotab.core.methods.trials:_separated_people",
                 "turbotab.core.methods.trials:_separated_refusal",
                 "turbotab.core.methods.trials:_logistic",
                 "turbotab.core.methods.trials:fit_gee",
                 "turbotab.core.methods.trials:_gee_refusal"),
    author=AUTHOR,
    origin="audit",
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
            "not a reading of the data. The interview's router marks a design that has been "
            "answered as answered before it reads the design's gate, so an answer, observational "
            "included, ends the statement (the Silent test)."))),
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
    enforced_by=("turbotab.core.interview:_design_gate", "turbotab.core.interview:route"),
    author=AUTHOR,
    origin="audit",
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
    regime=Regime("a case-control sample or matched sets, declared: refused as such (the "
                  "refusal states what the observational exit keeps out), or analyzed as an "
                  "observational study through that exit",
                  requires=(f"{HERE}:drawn_by_outcome",)),
    decision="set_design.declared",
    verdict="refused",
    parts=(
        Part("direction", grade="certified", graded_by=VERIFIER, basis=Basis(
            cites=("prentice1979casecontrol",), reason=(
                "Drawn by the outcome, the case share is set by the study: the odds ratio is the "
                "population's (Prentice & Pyke 1979), but the prevalence, the fitted "
                "probabilities and the standardized risk difference are not. The Fires test's "
                "known-truth population (200,000 people, a prevalence under 10%, a log odds "
                "ratio of 0.7; 1,000 cases and 1,000 controls drawn) shows each: statsmodels' "
                "Logit recovers the log odds ratio within 0.2, while the case share and the mean "
                "fitted probability are one half and the sample-standardized risk difference is "
                "more than 2.5 times the population's."),
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
    origin="audit",
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
    regime=Regime("a numeric exposure (what you study) under Estimate an effect, by the "
                  "partially linear model with cross-fitted models of the outcome and of what you "
                  "study given the covariates", requires=(f"{HERE}:numeric_exposure",)),
    decision="set_causal.positivity",
    verdict="noted",
    params={"r2_line": Param(0.9, unit="share of the variance of what you study",
                             constant="turbotab.core.models.causal:CONTINUOUS_R2_LINE")},
    parts=(
        Part("direction", grade="judgment", basis=Basis(reason=(
            "By Frisch–Waugh the partially linear model rests on the variance of what you study "
            "left once the covariates are known, so its standard error is about √(1/(1 − R²)) "
            "times what it would be were what you study unrelated to them, and its interval "
            "counts that. "
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
            "No R² guards the bias a misfit in the models of the outcome and of what you study "
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
        # Nothing is withheld at any R²: the reading marks none violated, and the lane estimates
        # an exposure the covariates explain to 96%.
        Silent("turbotab/core/tests/test_causal_lane.py::test_the_positivity_relation_blocks_a_"
               "propensity_and_states_a_numeric_exposure", truth="construction"),
        Silent(f"{ACCEPTANCE}/test_causal_lane.py::test_9b_a_numeric_exposure_the_covariates_"
               "nearly_determine_is_estimated_not_withheld", truth="construction"),
    ),
    enforced_by=("turbotab.core.causal:assumption_card", "turbotab.core.models.causal:variation"),
    author=AUTHOR,
    origin="audit",
))

COX_TEST_3 = (f"{ACCEPTANCE}/test_wp12b_cox_mixed_gee.py::test_3_a_group_with_no_events_is_"
              "refused_and_one_with_only_events_is_not")
COX_NO_EVENTS = (f"{TESTS}::test_a_cox_group_with_no_events_is_refused_naming_it_alone_as_the_"
                 "lowest_or_highest")

SURVIVAL_MONOTONE_LIKELIHOOD = register(Claim(
    id="survival.monotone_likelihood",
    kind="rule",
    methods=("A Cox fit whose partial likelihood has no maximum is refused, with no interval or "
             "p-value, where a column, or a direction along which the information has all but "
             "vanished, orders every risk set: the rows with the event have the lowest, or the "
             "highest, value at every event time, as when a group has no events. The refusal "
             "names that column alone, or says that a fixed combination of the columns it names "
             "does it."),
    goal=("inference",),
    regime=Regime("a Cox proportional-hazards model fit by Newton's method on standardized "
                  "columns, with Efron's ties, unweighted or with case weights",
                  requires=(f"{HERE}:cox_partial_likelihood",)),
    decision="cox_table.intervals",
    verdict="refused",
    params={
        "diverged": Param(20, unit="standard deviations of the column",
                          constant="turbotab.core.models.survival:DIVERGED"),
        "flat": Param(1e-8, unit="share of the information's largest eigenvalue",
                      constant="turbotab.core.models.survival:FLAT"),
        "singular": Param(1e-12, unit="share of the information's largest eigenvalue",
                          constant="turbotab.core.models.survival:SINGULAR"),
    },
    parts=(
        Part("direction", grade="certified", graded_by=VERIFIER, basis=Basis(reason=(
            "Where a column orders every risk set, the partial likelihood rises without end along "
            "it and has no finite maximum (monotone likelihood: Bryson & Johnson 1981, Biometrika "
            "68:703; Heinze & Schemper 2001, Biometrics 57:114). The Fires test shows it for each "
            "of its 16 fixtures by a profile likelihood written out by hand, which climbs at "
            "b_g = −10, −20 and −40, and asserts each table refused with no standard error, "
            "interval or p-value; it fails when both the ordering check and the singular "
            "trigger are removed."),
            tests=(COX_TEST_3,))),
        Part("detection", grade="judgment", basis=Basis(reason=(
            "Newton's decrement can pass its tolerance before any coefficient reaches the "
            "divergence bound, since the information decays along an ordering as fast as the "
            "score (B-435: a log hazard ratio near −36 with p = 0). So a fit whose information "
            "has an eigenvalue at or below the flat share of its largest, or that did not "
            "converge, is checked for a column or a flat direction that orders the risk sets. "
            "The Fires test on a group with no events beside age and noise columns holds that "
            "check and the flat share: with either removed, the refusal names age too. An "
            "information whose smallest eigenvalue is at or below the singular share of its "
            "largest, at the estimate, refuses the fit even where no ordering is found: it is "
            "the ordering check's backstop. On the fixtures the ordering is found first, so "
            "removing the singular share alone changes no refusal, and test_3 fails when both it "
            "and the ordering check are removed. The divergence bound, which stops the "
            "iterations, is a backstop no test reaches, and a matrix whose columns nearly copy "
            "each other is refused before the fit, as nearly singular. The three numbers are "
            "numerical conventions, not tuned against a reference."),
            tests=(COX_NO_EVENTS,))),
        Part("wording", grade="certified", graded_by=VERIFIER, basis=Basis(reason=(
            "The refusal says the rows with the event have the lowest or the highest value of the "
            "column at every event time, and names that column alone: the Fires test on a group "
            "with no events, coded g and 1 − g on the same six fixtures, reads the ordering off "
            "the rows and finds `g` named as the lowest, and as the highest, with age and the "
            "noise columns not named; it fails with the ordering check removed. A group in which "
            "every row has the event is no monotone likelihood when its events fall among "
            "others', and is estimated: the stage-1 Fires test matches statsmodels' PHReg "
            "(Efron ties) there, its estimates to 1e-8 relative and its standard errors to 1e-6, "
            "and fails when every Cox fit is refused."),
            tests=(COX_NO_EVENTS, COX_TEST_3))),
    ),
    tests=(
        Fires(COX_TEST_3, truth="construction"),
        Fires(COX_NO_EVENTS, truth="construction"),
        Silent(f"{TESTS}::test_a_cox_group_with_two_events_is_estimated_as_lifelines_fits_it",
               truth="independent:lifelines"),
    ),
    enforced_by=("turbotab.core.models.survival:cox_fit", "turbotab.core.models.survival:_ordering",
                 "turbotab.core.models.survival:no_maximum",
                 "turbotab.core.models.survival:cox_table",
                 "turbotab.core.models.survival:survey_cox_table"),
    author=AUTHOR,
    origin="audit",
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
                  "excluding zero",
                  requires=(f"{HERE}:least_squares_with_a_cut_off",
                            f"{HERE}:primary_interval_excludes_zero")),
    decision="results.tail_share",
    verdict="noted",
    parts=(
        Part("identity", grade="proved", graded_by=VERIFIER, basis=Basis(
            theorem=("For a fixed design X of full column rank and fixed weights W, the "
                     "least-squares estimator β̂(y) = (XᵀX)⁻¹Xᵀy, or (XᵀWX)⁻¹XᵀWy, is linear in y "
                     "(the 'linear' of the Gauss–Markov theorem's best linear unbiased "
                     "estimator). So for any pointwise split of the outcome, yᵢ = a(yᵢ) + b(yᵢ) "
                     "for every row, β̂(y) = β̂(a(y)) + β̂(b(y)) exactly, in exact arithmetic. "
                     "Both of the lens's splits are pointwise. A tail above c: the tail part "
                     "max(y − c, 0) with the rest min(y, c), since y ≤ c gives 0 and y, and "
                     "y > c gives y − c and c. A tail below c: the tail part min(y − c, 0) with "
                     "the rest max(y, c), since y ≥ c gives 0 and y, and y < c gives y − c and "
                     "c. Each pair adds to y."),
            reason=("The three fits share the design, the weights and the rows; the build's test "
                    "holds the sum to float64 rounding."))),
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
    origin="audit",
))

# ── R1-F's first batch (R1_PLAN.md R1-F; claims backtest FIXES.md) ───────────────────────────
#
# The inference area's sentences an analyst reads on a card, in teaching, in a refusal or in a
# reading, each fixed test first: what adjustment does in each design (D-032), multiple
# imputation refused unless the purpose is inference (A-117), what holding a near dependency's
# columns fixed does (A-121), and whether the group a missingness gap is read by is in the
# imputation model (C-241). Each part is the author's judgment: no grader other than the author
# has checked one yet (CLAIMS_REGISTRY.md §2.4).

RF1_AUTHOR = "RF-1 builder"
RF1_TESTS = "turbotab/core/tests/test_rf1_sentences.py"

DESIGNS_ADJUSTMENT_BY_DESIGN = register(Claim(
    id="designs.adjustment_by_design",
    kind="teaching",
    card="The design decides which comparisons are fair. In a trial, adjustment buys precision: a "
         "standardized risk keeps its meaning; an adjusted odds ratio becomes conditional. A "
         "case-control sample needs its sampling fractions for prevalence or absolute risk; "
         "matched sets are analyzed within each set. An observational study adjusts for what "
         "else could explain the link and speaks of associations.",
    regime=Regime("the study design's card, read before the design is answered: a randomized "
                  "trial, a sample drawn by the outcome, matched sets or an observational study"),
    decision="set_design",
    parts=(
        Part("trial", grade="judgment", basis=Basis(
            reason="Randomization balances every covariate, so adjusting for one buys precision "
                   "and leaves a standardized risk's meaning; an odds ratio is non-collapsible, "
                   "so adjusting for a covariate that predicts the outcome changes it even "
                   "without imbalance (the fires test fits both on a trial built here, with "
                   "statsmodels). The trial core standardizes its risks (methods/trials.py).")),
        Part("case_control", grade="judgment", basis=Basis(
            reason="A sample drawn by the outcome sets the share of cases, so prevalence and "
                   "absolute risk need the sampling fractions; weighting each row by the inverse "
                   "of its fraction recovers the risk, as the second test builds.")),
        Part("matched", grade="judgment", basis=Basis(
            reason="Individually matched sets are compared within each set (conditional "
                   "logistic regression), as designs.py says the milestone will analyze them.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_in_a_trial_an_adjusted_odds_ratio_is_another_effect_and_a_"
              "standardized_risk_is_not", truth="independent:statsmodels"),
        Fires(f"{RF1_TESTS}::test_a_case_control_sample_gives_absolute_risk_with_its_sampling_"
              "fractions", truth="construction"),
    ),
    enforced_by=("turbotab.core.teaching.content:DESIGN",),
    author=RF1_AUTHOR,
    origin="audit",
))

MISSING_MI_UNLESS_INFERENCE = register(Claim(
    id="missing.mi_refused_unless_inference",
    kind="rule",
    card="TurboTab's multiple imputation puts the outcome in the imputation model, so a held-out "
         "row's own outcome would fill its blanks, and a new row to predict has no outcome to "
         "fill from. Unless the purpose is inference, the fill offered is fit in each training "
         "fold without the outcome: where new rows may arrive with blanks, Sisk et al. (2023) "
         "found leaving the outcome out preferred, while with complete rows at deployment they "
         "recommend using it in multiple imputation, not in a single regression fill.",
    regime=Regime("the multiple-imputation answer when the purpose is not inference: prediction, "
                  "or no purpose answered yet"),
    decision="set_missing.strategy",
    options=("multiple_imputation", "impute", "complete_case"),
    verdict="refused",
    parts=(
        Part("leak", grade="judgment", basis=Basis(
            reason="The imputation the app runs is fit on the whole table with the outcome in "
                   "its model, so it would read a held-out row's own outcome; an in-fold "
                   "imputation with the outcome is not built, so no valid alternative is held "
                   "back.")),
        Part("sisk_scope", grade="judgment", basis=Basis(
            cites=("sisk2023missing",),
            reason="We read Sisk et al. 2023 as methods/missing.py quotes it: \"the outcome should "
                   "be used to impute development data when using multiple imputation and "
                   "omitted under regression imputation. When missingness is allowed at "
                   "deployment, omitting the outcome from the imputation model at the "
                   "development was preferred\"; the refusal cites them for that regime, and "
                   "says their use of the outcome with complete rows is for multiple "
                   "imputation, not for a single fill such as the one it offers (the RF-1 "
                   "verifier's V6, who read the abstract on PubMed). The author read the "
                   "quotation, not the paper.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_multiple_imputation_is_refused_unless_the_purpose_is_inference_"
              "in_sisks_scope", truth="construction"),
        Silent(f"{RF1_TESTS}::test_multiple_imputation_under_inference_is_not_refused",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.decisions:_missing_fits_the_purpose",),
    author=RF1_AUTHOR,
    origin="audit",
))

COLLINEAR_INSIDE = register(Claim(
    id="materiality.collinear_inside",
    kind="teaching",
    card="holding them fixed leaves little of its own variation, so which of them stay in the "
         "model changes the contrast the estimate describes.",
    goal=("inference",),
    regime=Regime("under Estimate, what you study inside a near dependency among the adjustment "
                  "terms (Belsley's condition index and variance-decomposition proportions)"),
    decision="set_adjustment.collinear",
    parts=(
        Part("contrast", grade="judgment", basis=Basis(
            reason="What you study is then close to a fixed mix of the columns beside it, so the "
                   "variation left once they are held fixed is small, and its coefficient "
                   "compares rows that differ in it with those columns held where they are: which "
                   "of them stay changes that comparison.")),
        Part("part_of_a_total", grade="judgment", basis=Basis(
            reason="Only where what you study is a part of a total beside it (sugar within "
                   "carbohydrate, read by readings.nesting) is more of it at a fixed total less "
                   "of the rest of that total, which the code then says; the claims backtest's "
                   "A-121 found the substitution reading stated for every dependency.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_a_part_beside_its_total_says_more_of_it_is_less_of_the_rest_of_"
              "the_total", truth="construction"),
        Silent(f"{RF1_TESTS}::test_a_dependency_that_is_no_total_says_the_contrast_changes_not_"
               "what_it_replaces", truth="construction"),
    ),
    enforced_by=("turbotab.core.materiality:collinear_noticing",),
    author=RF1_AUTHOR,
    origin="audit",
))

EXPLORE_MISSING_BY_GROUP = register(Claim(
    id="explore.missing_by_group",
    kind="teaching",
    card="Multiple imputation fills blanks from the analysis's variables; complete cases state "
         "their assumption",
    goal=("inference",),
    regime=Regime("under Estimate, a column named like a sociodemographic group whose groups "
                  "differ in how often the candidate predictors are blank (TRIPOD+AI 7)"),
    decision="explore.quality_by_group",
    parts=(
        Part("model", grade="judgment", basis=Basis(
            reason="The imputation model holds the design's inputs (the predictors and a "
                   "residual method's strata column) and the outcome's terms "
                   "(methods.missing.imputation_frame), with no path for an auxiliary column, so "
                   "the option says, right after the variables, whether the group is one of "
                   "them, and so whether the imputation model holds it.")),
        Part("complete_cases", grade="judgment", basis=Basis(
            reason="Complete cases stay on the menu with their assumption stated "
                   "(methods.missing.COMPLETE_CASE_ASSUMPTION).")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_a_group_that_is_a_predictor_is_said_to_be_in_the_imputation_"
              "model", truth="construction"),
        Silent(f"{RF1_TESTS}::test_a_group_that_is_no_predictor_is_said_to_be_left_out_of_the_"
               "imputation_model", truth="construction"),
    ),
    enforced_by=("turbotab.core.stages.explore:_quality_finding",),
    author=RF1_AUTHOR,
    origin="audit",
))

POOLED_FIRTH_TESTS = "turbotab/core/tests/test_pooled_firth.py"
# A relative effect θ_j = β_j − Σ_k w_k β_k reads a separated source's β with weight 1 or −w (> 0).
RELATIVE_DERIVED = "derivation:relative_effect_reads_the_separated_coefficient"

POOLED_SEPARATION = register(Claim(
    id="inference.pooled_separation",
    kind="rule",
    card=("a Wald interval on a separated coefficient's Firth estimate is not certified, and the "
          "profile penalized-likelihood interval advised in its place is not one Rubin's rules "
          "pool"),
    methods=("Under multiple imputation, a coefficient whose maximum-likelihood value is infinite "
             "in any completed copy (a column that separates the outcome, the intercept beside it, "
             "or an all-components relative effect resting on it) keeps its pooled estimate and "
             "standard error, and no pooled interval or p-value is reported for it: Rubin's rules "
             "pool each copy's estimate and standard error into a Wald interval, which is not "
             "certified on a separated coefficient's Firth estimate, and the profile "
             "penalized-likelihood interval advised in its place is not one they pool."),
    goal=("inference",),
    regime=Regime("a logistic coefficient table pooled by Rubin's rules over completed copies, in "
                  "some of which a coefficient separates the outcome (its maximum-likelihood "
                  "estimate infinite, found by the linear program) and the copy is fit by Firth's "
                  "penalized likelihood",
                  requires=(f"{HERE}:pooled_over_copies", f"{HERE}:separated_in_a_copy")),
    decision="pooled_table.intervals",
    verdict="refused",
    parts=(
        Part("wald_not_certified", grade="judgment", basis=Basis(reason=(
            "The orchestrator's ruling on the slow-fit audit's F3 (2026-10-23): a number is "
            "right or refused. The unpooled table cites Heinze & Schemper (2002, Stat Med "
            "21:2409) for profile penalized-likelihood intervals and likelihood-ratio p-values "
            "under separation in place of Wald ones (models/inference.py), and reports those. "
            "Rubin's rules (Rubin 1987) pool an estimate and its variance, so the pooled interval "
            "of a separated coefficient would be the Wald interval that citation sets aside. The "
            "paper is not in the citation registry and its author did not read it for this "
            "claim, which rests on the module's own citation: graded by judgment."))),
        Part("any_copy", grade="judgment", basis=Basis(reason=(
            "A coefficient is withheld where the linear program finds it infinite in at least one "
            "copy, since its pooled interval then rests on that copy's Firth estimate: a separated "
            "column, the intercept where the program finds it infinite beside one (a reference "
            "level every row of which has the event), and an all-components relative effect "
            "whose refit on the moved energy columns finds it infinite (each one resting on a "
            "separated energy source). A coefficient no copy finds infinite keeps Rubin's "
            "interval, even beside one that is withheld. A copy no column separates is a "
            "maximum-likelihood fit, and the caption says how many such copies the pooled table "
            "holds."))),
    ),
    tests=(
        Fires(f"{POOLED_FIRTH_TESTS}::test_a_coefficient_separated_in_the_copies_keeps_its_pooled_"
              "estimate_and_withholds_its_interval", truth="construction"),
        Silent(f"{POOLED_FIRTH_TESTS}::test_a_coefficient_no_copy_separates_keeps_rubins_wald_"
               "interval_on_t", truth="independent:scipy"),
        Silent(f"{POOLED_FIRTH_TESTS}::test_a_table_no_copy_separates_withholds_nothing",
               truth="construction"),
        Fires(f"{POOLED_FIRTH_TESTS}::test_an_intercept_infinite_in_the_copies_keeps_its_estimate_"
              "and_withholds_its_interval", truth="construction"),
        Fires(f"{POOLED_FIRTH_TESTS}::test_a_relative_effect_resting_on_a_separated_source_keeps_"
              "its_estimate_and_withholds_its_interval", truth=RELATIVE_DERIVED),
        Fires(f"{POOLED_FIRTH_TESTS}::test_the_effects_relative_effect_resting_on_a_separated_"
              "source_is_withheld", truth=RELATIVE_DERIVED),
        Silent(f"{POOLED_FIRTH_TESTS}::test_relative_effects_no_copy_separates_keep_rubins_wald_"
               "interval_on_t", truth="independent:statsmodels"),
    ),
    enforced_by=("turbotab.core.models.inference:pooled_firth",),
    author="swarm-mi-firth builder",
    origin="engine",
))

CLAIMS: tuple[Claim, ...] = (
    INFERENCE_FEW_CLUSTERS, TRIALS_FEW_CLUSTERS, TRIALS_FEW_CLUSTERS_EXTENDED,
    ORDINAL_CLUSTER_ROBUST_INTERVALS,
    TRIALS_SEPARATION, DESIGNS_OBSERVATIONAL_DEFAULT, DESIGNS_SAMPLED_BY_OUTCOME,
    CAUSAL_NUMERIC_EXPOSURE_R2, SURVIVAL_MONOTONE_LIKELIHOOD, RESULTS_TAIL_SHARE,
    DESIGNS_ADJUSTMENT_BY_DESIGN, MISSING_MI_UNLESS_INFERENCE, COLLINEAR_INSIDE,
    EXPLORE_MISSING_BY_GROUP, POOLED_SEPARATION,
)

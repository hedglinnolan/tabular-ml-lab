"""Randomized-trial analyses as an engine method core (SIZING E2; amendment of 2026-10-07).

A declared randomized trial (``parallel_trial`` or ``cluster_randomized_trial``, the design keys of
:mod:`turbotab.core.designs`) is analyzed here from explicit inputs (:class:`TrialSpec`: the arm
column and its control, the outcome, the randomization factors, the prespecified baseline
covariates, the baseline measure of the outcome, the randomized cluster, adherence). Every function
returns its result with the counts a CONSORT flow needs, or raises :class:`TrialRefused` with its
reason in plain words, the technical term as a quiet label, and its exits. Wiring into the
interview and the decisions comes later (E1, D1, C1).

**The analysis sets** (:func:`analysis_sets`). Intention to treat: everyone randomized, in the arm
they were randomized to, whatever they received or did afterwards (CONSORT 2010, Moher et al. 2010,
item 16; White, Horton, Carpenter & Pocock 2011). Per protocol: the randomized people whose
adherence column says they followed the protocol, still in the arm they were randomized to. This
is the *set*; the effect of following the protocol (weighting for adherence over time) is another
estimand, refused by :data:`turbotab.core.designs.EFFECTS` until v2.x. A per-protocol result is
never read as an effect: who adheres may differ between the arms, so randomization no longer
protects the comparison.

**Parallel trials** (:func:`estimate`). The effect of the assigned arm, adjusted for the
randomization factors (Kahan & Morris 2012: an analysis that ignores the stratification gives
intervals that are too wide) and the baseline covariates named before the data were seen, the
baseline measure of the outcome among them (analysis of covariance; Vickers & Altman 2001). The
adjustment is for precision only: nothing here searches for or selects a covariate, and a
covariate chosen after the data were seen is refused from the primary analysis
(``covariates_prespecified=False``) with a labeled secondary analysis as its exit.

* A numeric outcome: least squares of the outcome on the arm and the covariates; the mean
  difference is the arm's coefficient, on t(n − p) with the model-based variance, as R's ``lm``
  and ``emmeans`` give it; each arm's adjusted mean is the mean over the analyzed rows of its
  prediction with the arm set to it (``emmeans`` with ``weights = "proportional"``).
* A yes/no outcome: a logistic regression on the same terms, each arm's risk standardized over the
  analyzed rows (every row's predicted risk with the arm set to it, averaged), the risk difference
  and the risk ratio (on the log scale) from them, with the variance of Ye, Shao, Yi & Zhao (2023),
  which stays valid when the logistic model is wrong and, with the randomization factors named,
  subtracts the stratification's term (taking the randomization to have balanced the arms within
  each stratum, as permuted blocks do); z intervals. This is R's ``beeca`` (``method = "Ye"``).

**Cluster-randomized trials** (:func:`estimate` with ``trial_design="cluster_randomized_trial"``).
The cluster is the unit of randomization, so every cluster has one arm (refused otherwise).

* ``cluster_method="mixed"`` (a numeric outcome): a linear mixed model with a random intercept per
  cluster, fitted by REML (:func:`turbotab.core.models.repeated.fit_random_intercept`), with the
  Kenward–Roger adjusted covariance and degrees of freedom (Kenward & Roger 1997), computed as R's
  ``pbkrtest`` computes them (Halekoh & Højsgaard 2014; ``vcovAdj`` and ``Lb_ddf``). The
  intraclass correlation is σ²_u/(σ²_u + σ²_e). A yes/no outcome is refused here: the
  Kenward–Roger correction is for linear mixed models, and its exit is the GEE.
* ``cluster_method="gee"``: generalized estimating equations (Liang & Zeger 1986) with an
  exchangeable working correlation, estimated as R's ``gee`` estimates it (the scale
  φ = Σ r²/(N − p) and α = Σ_g Σ_{j≠k} r_gj r_gk /(φ(Σ_g m_g(m_g − 1) − 2p)) on Pearson residuals),
  and a small-sample corrected sandwich: Kauermann & Carroll's (2001) (I − H_g)^(−½) on each
  cluster's residuals, the principal root of the non-symmetric I − H_g (clubSandwich's CR2 for a
  geeglm takes another root, equal to it only for a numeric outcome under independence, so it is
  not the reference here); Fay & Graubard's (2001) diagonal correction with the bound b = 0.75;
  or Mancl & DeRouen's (2001) (I − H_g)^(−1). ``correction="auto"`` follows Li & Redden (2015):
  Kauermann–Carroll when the coefficient of variation of the cluster sizes is below 0.6,
  Fay–Graubard otherwise. Intervals are t on the clusters less the cluster-level coefficients
  (Li & Redden's K − 2 for two arms and no cluster-level covariate). A yes/no outcome's risks are
  standardized as in a parallel trial, by the delta method on the corrected covariance (the
  covariates' own sampling variation is not added); the intraclass correlation reported is the
  GEE's exchangeable working correlation.

**Missing outcomes.** The primary analysis analyzes the outcomes observed, which is valid when they
are missing at random given the arm and the covariates (White, Horton, Carpenter & Pocock 2011).
:func:`tipping_point` is the sensitivity analysis: multiple imputation of every missing outcome
under that assumption (proper draws, the imputation model the analysis model: Bayesian linear
regression for a number, logistic for yes/no, as R's ``mice`` ``norm`` and ``logreg`` draw them),
then the imputed outcomes of one arm shifted by δ (a number: δ on the outcome's scale; yes/no: δ
on the log odds of the event), each completed set analyzed as the primary analysis and pooled by
Rubin's rules with Barnard & Rubin's degrees of freedom; the tipping point is the shift at which
the conclusion (whether the 95% interval excludes no difference) first changes along a search
outward from no shift in each direction, the nearer of the two (Ratitch,
O'Kelly & Tosiello 2013; Cro, Morris, Kenward & Carpenter 2020). The number of imputations is at
least the percentage of randomized people with a missing outcome (White, Royston & Wood 2011).
It is offered for intention to treat in parallel trials; a cluster trial's imputation would need
a multilevel imputation model, not built, and is refused with the primary analysis as its exit.

**Baseline tables** (:func:`baseline_table`) describe each arm and never test: a difference at
baseline between randomized arms is chance by construction (CONSORT 2010 item 15; Senn 1994), and
a request for tests is refused.

**Partially missing baseline covariates** are filled by their mean over the analysis set, which uses
neither the arm nor the outcome, with an indicator of the missing value beside it (a missing
category for a category), White & Thompson's (2005) conditions for mean imputation in a
randomized trial.

**Survey designs** are refused, by the effect, the tipping point, each delta-adjusted analysis,
the baseline table and the CONSORT flow alike: a trial's inference rests on its randomization, not
on sampling weights; the exit is the sample-only answer. **Under Predict** nothing here is offered: an
assigned treatment's effect is an Estimate question.

**Wording.** Causal wording ("assignment to A changed the outcome by …") is written only for the
intention-to-treat set of a declared randomized trial, with the caveat that it is the effect of
being assigned; the per-protocol set is worded as a difference among those who followed the
protocol (:func:`trial_sentence`).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

CONF = 0.95
TRIAL_DESIGNS: tuple[str, ...] = ("parallel_trial", "cluster_randomized_trial")
SETS: tuple[str, ...] = ("itt", "per_protocol")
MEASURES: tuple[str, ...] = ("mean_difference", "risk_difference", "risk_ratio")
# The trial core's own cluster analyses, fitted here (``_mixed``, ``_gee``): they share their names
# with the mixed and GEE model families but are not those families, so nothing here reads or
# switches on a family's key.
MIXED_MODEL = "mixed"
GEE_MODEL = "gee"
CLUSTER_METHODS: tuple[str, ...] = (MIXED_MODEL, GEE_MODEL)
CORRECTIONS: tuple[str, ...] = ("auto", "kc", "fg", "md")
GOALS: tuple[str, ...] = ("inference", "prediction", "describe")
FG_BOUND = 0.75  # Fay & Graubard's b, geesmv's default
CV_RULE = 0.6  # Li & Redden 2015: KC below this coefficient of variation of cluster sizes, FG above
FEW_CLUSTERS = 10  # the fewest clusters Li & Redden 2015 simulated
GEE_TOL = 1e-12
GEE_MAXITER = 200

SOURCES = {
    "moher2010": "Moher et al. 2010, BMJ 340:c869 (CONSORT 2010 explanation and elaboration, items "
                 "15 and 16)",
    "white2011": "White, Horton, Carpenter & Pocock 2011, BMJ 342:d40 (doi:10.1136/bmj.d40)",
    "kahan2012": "Kahan & Morris 2012, Stat Med 31:328 (doi:10.1002/sim.4431)",
    "vickers2001": "Vickers & Altman 2001, BMJ 323:1123 (doi:10.1136/bmj.323.7321.1123)",
    "senn1994": "Senn 1994, Stat Med 13:1715 (doi:10.1002/sim.4780131703)",
    "white2005": "White & Thompson 2005, Stat Med 24:993 (doi:10.1002/sim.1981)",
    "ye2023": "Ye, Shao, Yi & Zhao 2023, J Am Stat Assoc 118:2370 "
              "(doi:10.1080/01621459.2022.2049278)",
    "kenward1997": "Kenward & Roger 1997, Biometrics 53:983 (doi:10.2307/2533558)",
    "halekoh2014": "Halekoh & Højsgaard 2014, J Stat Softw 59(9) (doi:10.18637/jss.v059.i09)",
    "liang1986": "Liang & Zeger 1986, Biometrika 73:13 (doi:10.1093/biomet/73.1.13)",
    "kauermann2001": "Kauermann & Carroll 2001, J Am Stat Assoc 96:1387 "
                     "(doi:10.1198/016214501753382309)",
    "fay2001": "Fay & Graubard 2001, Biometrics 57:1198 (doi:10.1111/j.0006-341X.2001.01198.x)",
    "mancl2001": "Mancl & DeRouen 2001, Biometrics 57:126 (doi:10.1111/j.0006-341X.2001.00126.x)",
    "li2015": "Li & Redden 2015, Stat Med 34:281 (doi:10.1002/sim.6344)",
    "leyrat2018": "Leyrat et al. 2018, Int J Epidemiol 47:321 (doi:10.1093/ije/dyx169)",
    "campbell2012": "Campbell et al. 2012, BMJ 345:e5661 (CONSORT 2010 for cluster-randomized "
                    "trials)",
    "ratitch2013": "Ratitch, O'Kelly & Tosiello 2013, Pharm Stat 12:337 (doi:10.1002/pst.1549)",
    "cro2020": "Cro, Morris, Kenward & Carpenter 2020, Stat Med 39:2815 (doi:10.1002/sim.8569)",
    "rubin1987": "Rubin 1987, Multiple Imputation for Nonresponse in Surveys (Wiley)",
    "barnard1999": "Barnard & Rubin 1999, Biometrika 86:948",
    "white2011mi": "White, Royston & Wood 2011, Stat Med 30:377 (doi:10.1002/sim.4067)",
}

SAMPLE_ONLY_EXIT = {"label": "Analyze the trial as randomized, without the survey weights (the "
                             "sample-only answer)",
                    "decision": {"kind": "set_survey", "estimand": "sample"}}
ESTIMATE_EXIT = {"label": "Ask it under Estimate an effect", "goal": "inference"}
DECLARE_EXIT = {"label": "Declare how people were randomized",
                "decision": {"kind": "set_design"}}
ITT_EXIT = {"label": "Analyze everyone randomized (intention to treat)", "analysis_set": "itt"}
GEE_EXIT = {"label": "Use generalized estimating equations with a corrected sandwich",
            "cluster_method": "gee"}
PRIMARY_EXIT = {"label": "Keep the primary analysis, its assumption stated (the missing outcomes "
                         "are like the observed ones with the same arm and covariates)",
                "tipping_point": False}
DESCRIBE_ARMS_EXIT = {"label": "Describe each arm instead", "tests": False}
DESCRIBE_OUTCOMES_EXIT = {"label": "Describe the outcome in each arm instead, without an effect",
                          "describe": True}
CLUSTER_COLUMN_EXIT = {"label": "Choose the column that names every person's randomized cluster",
                       "decision": {"kind": "confirm_role", "column": None, "role": "cluster"}}
SECONDARY_EXIT = {"label": "Run it as a labeled secondary analysis", "secondary": True}
LEAVE_OUT_EXIT = {"label": "Leave the covariates chosen after the data were seen out of the "
                           "primary analysis", "covariates_prespecified": True}


class TrialRefused(ValueError):
    """The data, the design or the goal cannot support the trial analysis asked for. The message
    says why in plain words, ``term`` is the technical name as a quiet label, and ``exits`` are
    the ways forward."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = (), term: str = ""):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]
        self.term = term


# ── the inputs ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrialSpec:
    """The explicit inputs of a trial analysis: columns of one frame, one row per person.

    ``arm`` is the arm each person was randomized to (blank: not randomized, counted as assessed
    only); ``control`` the arm every other is compared with. ``strata`` are the factors the
    randomization was stratified or minimized on; ``covariates`` the baseline covariates named
    before the data were seen; ``baseline`` the baseline measure of the outcome. ``cluster`` names
    the randomized cluster (a cluster-randomized trial); ``adherent`` whether each person followed
    the protocol (yes/no; blank is unknown and outside the per-protocol set); ``received`` the arm
    each person actually received, for the CONSORT flow."""

    arm: str
    control: Any
    outcome: str
    trial_design: str = "parallel_trial"
    strata: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    baseline: str | None = None
    cluster: str | None = None
    adherent: str | None = None
    received: str | None = None

    def adjusted_for(self) -> list[str]:
        """The adjustment terms in the order the model takes them."""
        out = list(self.strata)
        if self.baseline:
            out.append(self.baseline)
        out.extend(c for c in self.covariates if c not in out)
        return out


# ── which goal offers what ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Eligibility:
    offered: bool
    says: str
    exits: tuple[dict[str, Any], ...] = ()


def eligible(goal: str, what: str = "effect") -> Eligibility:
    """Whether ``goal`` (inference · prediction · describe) offers ``what`` (effect · baseline_table
    · tipping_point), in plain words."""
    if goal not in GOALS or what not in ("effect", "baseline_table", "tipping_point"):
        raise ValueError(f"goal {goal!r} or {what!r} is unknown")
    if what == "baseline_table":
        if goal == "prediction":
            return Eligibility(False, "A trial's baseline table describes the randomized arms; "
                                      "Predict does not compare arms.", (ESTIMATE_EXIT,))
        return Eligibility(True, "Each arm described at baseline, without tests.")
    if goal != "inference":
        return Eligibility(False, "The effect of an assigned treatment is a question for Estimate "
                                  "an effect.", (ESTIMATE_EXIT,))
    if what == "tipping_point":
        return Eligibility(True, "How far the missing outcomes would have to differ from the "
                                 "observed ones to change the conclusion.")
    return Eligibility(True, "The effect of being assigned to each arm, compared with the "
                             "control arm.")


def _check_goal(goal: str, what: str) -> None:
    e = eligible(goal, what)
    if not e.offered:
        raise TrialRefused(e.says, e.exits, term="not offered under this goal")


def _check_survey(survey_design: Any) -> None:
    if survey_design is not None:
        raise TrialRefused(
            "A trial's comparison rests on who was randomized to which arm, not on how people were "
            "sampled, so survey weights do not enter it.", (SAMPLE_ONLY_EXIT,),
            term="survey design in a randomized trial")


def _check_design(spec: TrialSpec) -> None:
    if spec.trial_design not in TRIAL_DESIGNS:
        raise TrialRefused(
            "These analyses are for a randomized trial; the design declared is not one, so the "
            "arms were not assigned by chance and their comparison is not protected.",
            (DECLARE_EXIT,), term=f"design {spec.trial_design!r} is not a randomized trial")
    if spec.trial_design == "cluster_randomized_trial" and not spec.cluster:
        raise TrialRefused("A cluster-randomized trial needs the column that names each person's "
                           "randomized cluster.", ({"label": "Name the cluster column",
                                                    "cluster": None},),
                           term="no cluster column")


# ── reading the columns ──────────────────────────────────────────────────────


def _adherence_refusal(column: str, what: str) -> TrialRefused:
    return TrialRefused(
        f"The adherence column {column!r} must say yes or no for each person; it holds {what}, so "
        f"who followed the protocol cannot be read from it.",
        ({"label": f"Choose a yes/no column for adherence instead of {column}",
          "decision": {"kind": "confirm_role", "column": None, "role": "flag"}},
         ITT_EXIT), term="adherence is not yes/no")


def _yes_no(values: pd.Series, column: str = "adherence") -> pd.Series:
    """A yes/no column as True / False / NA (bool, 0/1, or yes/no/true/false text)."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype("boolean")
    if pd.api.types.is_numeric_dtype(values):
        v = pd.to_numeric(values, errors="coerce")
        bad = v.notna() & ~v.isin([0, 1])
        if bad.any():
            raise _adherence_refusal(column, "other numbers")
        return v.map({1: True, 0: False}).astype("boolean")
    text = values.astype("string").str.strip().str.lower()
    mapped = text.map({"yes": True, "true": True, "1": True, "y": True,
                       "no": False, "false": False, "0": False, "n": False})
    bad = text.notna() & mapped.isna()
    if bad.any():
        raise _adherence_refusal(column, "other words")
    return mapped.astype("boolean")


def _label(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _arms(frame: pd.DataFrame, spec: TrialSpec) -> list[str]:
    """The arms as text labels, the control first, the others in sorted order."""
    labels = frame[spec.arm].dropna().map(_label)
    found = sorted(set(labels))
    control = _label(spec.control)
    if control not in found:
        raise TrialRefused(f"No one was randomized to the control arm {control!r}; the arms found "
                           f"are {', '.join(found) or 'none'}.",
                           ({"label": "Choose the control arm", "control": None},),
                           term="control arm not found")
    if len(found) < 2:
        raise TrialRefused(
            f"Everyone randomized is in one arm ({control}) of {spec.arm!r}, so there is nothing "
            f"to compare.",
            ({"label": "Choose the column that holds the arm each person was randomized to",
              "decision": {"kind": "confirm_role", "column": None, "role": "exposure"}},
             {"label": "Declare the design (a single-arm study is not a randomized comparison)",
              "decision": {"kind": "set_design"}}), term="one arm")
    return [control] + [a for a in found if a != control]


def _require(frame: pd.DataFrame, spec: TrialSpec) -> None:
    named = [spec.arm, spec.outcome, *spec.strata, *spec.covariates, spec.baseline, spec.cluster,
             spec.adherent, spec.received]
    missing = [c for c in named if c and c not in frame.columns]
    if missing:
        raise ValueError(f"columns not in the data: {missing}")
    roles = {spec.arm: "the arm", spec.outcome: "the outcome"}
    for c in [*spec.strata, *spec.covariates, *([spec.baseline] if spec.baseline else [])]:
        if c in roles:
            raise TrialRefused(f"{c!r} is {roles[c]}; it cannot also be an adjustment term.",
                               ({"label": f"Leave {c} out of the adjustment", "drop": c},),
                               term="adjustment term is the arm or the outcome")
    after = {spec.adherent: "adherence", spec.received: "the treatment received",
             spec.cluster: "the cluster"}
    for c in [*spec.strata, *spec.covariates]:
        if c in after and c != spec.cluster:
            raise TrialRefused(
                f"{c!r} is {after[c]}, which is known only after randomization; adjusting for it "
                f"would undo what randomization balanced.",
                ({"label": f"Leave {c} out of the adjustment", "drop": c},),
                term="post-randomization covariate")


def _outcome_kind(y: pd.Series) -> str:
    v = y.dropna()
    if pd.api.types.is_bool_dtype(v):
        return "binary"
    if not pd.api.types.is_numeric_dtype(v):
        raise TrialRefused("The outcome must be a number or yes/no coded 1 and 0; it holds text.",
                           ({"label": "Code the outcome as 1 (yes) and 0 (no)"},),
                           term="non-numeric outcome")
    if len(v) and set(np.unique(v.astype(float))) <= {0.0, 1.0}:
        return "binary"
    return "continuous"


# ── the analysis sets and the CONSORT flow ───────────────────────────────────


@dataclass(frozen=True)
class AnalysisSets:
    """Row masks over the frame (aligned with its index)."""

    randomized: pd.Series
    itt: pd.Series
    per_protocol: pd.Series | None
    outcome_observed: pd.Series
    arm: pd.Series  # the randomized arm as text (NA when not randomized)
    # an adherence column that is not yes/no: refused for the per-protocol set only, so intention
    # to treat (which never reads adherence) still runs
    adherence_problem: TrialRefused | None = None

    def mask(self, analysis_set: str) -> pd.Series:
        if analysis_set == "itt":
            return self.itt
        if analysis_set == "per_protocol":
            if self.adherence_problem is not None:
                raise self.adherence_problem
            if self.per_protocol is None:
                raise TrialRefused("The per-protocol set needs a column that says whether each "
                                   "person followed the protocol.",
                                   (ITT_EXIT, {"label": "Name the adherence column",
                                               "adherent": None}),
                                   term="no adherence column")
            return self.per_protocol
        raise ValueError(f"analysis_set {analysis_set!r} is not one of {SETS}")


def analysis_sets(frame: pd.DataFrame, spec: TrialSpec) -> AnalysisSets:
    """Intention to treat (every randomized person, as randomized) and, with an adherence column,
    the per-protocol set (the randomized people who followed the protocol, as randomized). Neither
    reads the outcome; the outcome's presence is returned beside them."""
    _require(frame, spec)
    arm = frame[spec.arm].map(lambda v: _label(v) if pd.notna(v) else pd.NA)
    randomized = arm.notna()
    pp = problem = None
    if spec.adherent:
        try:
            adh = _yes_no(frame[spec.adherent], spec.adherent)
            pp = randomized & adh.fillna(False).astype(bool)
        except TrialRefused as refused:
            problem = refused
    observed = frame[spec.outcome].notna()
    return AnalysisSets(randomized=randomized, itt=randomized.copy(), per_protocol=pp,
                        outcome_observed=observed, arm=arm, adherence_problem=problem)


def in_analysis_set(frame: pd.DataFrame, spec: TrialSpec, analysis_set: str = "itt") -> pd.Series:
    """Whether each row is in ``analysis_set``: its own arm and adherence only (row-local)."""
    return analysis_sets(frame, spec).mask(analysis_set)


@dataclass(frozen=True)
class ArmFlow:
    """One arm's counts by CONSORT stage (Schulz et al. 2010's flow diagram; Campbell et al. 2012
    adds the clusters)."""

    arm: str
    allocated: int
    received_allocated: int | None  # received the arm they were randomized to
    did_not_receive_allocated: int | None
    followed_up: int  # outcome observed
    lost_to_follow_up: int  # outcome missing
    analyzed_itt: int  # the primary analysis: randomized, outcome observed
    analyzed_itt_sensitivity: int  # the tipping-point analysis: every randomized person
    per_protocol: int | None
    excluded_from_per_protocol: int | None  # did not follow the protocol, or unknown
    analyzed_per_protocol: int | None
    clusters_allocated: int | None = None
    clusters_analyzed: int | None = None
    cluster_sizes: list[int] | None = None


@dataclass(frozen=True)
class ConsortFlow:
    assessed: int  # every row
    not_randomized: int
    randomized: int
    arms: list[ArmFlow]
    cluster_trial: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def consort_flow(frame: pd.DataFrame, spec: TrialSpec, *, survey_design: Any = None
                 ) -> ConsortFlow:
    """The counts per arm by stage that the CONSORT flow diagram reports (E3 draws it). A survey
    design is refused as everywhere in a trial (the sample-only answer is its exit), so the flow
    never sits beside weighted results it does not describe."""
    _check_survey(survey_design)
    _check_design(spec)
    sets = analysis_sets(frame, spec)
    if sets.adherence_problem is not None:
        p = sets.adherence_problem
        raise TrialRefused(str(p), (p.exits[0], {"label": "Draw the flow without the "
                                                          "per-protocol set", "adherent": None}),
                           term=p.term)
    arms = _arms(frame, spec)
    received = None if not spec.received else frame[spec.received].map(
        lambda v: _label(v) if pd.notna(v) else pd.NA)
    cluster = frame[spec.cluster] if spec.cluster and spec.trial_design == \
        "cluster_randomized_trial" else None
    out = []
    for a in arms:
        rows = sets.randomized & (sets.arm == a)
        obs = rows & sets.outcome_observed
        got = None if received is None else int((rows & (received == a)).sum())
        pp = None if sets.per_protocol is None else rows & sets.per_protocol
        flow = ArmFlow(
            arm=a, allocated=int(rows.sum()), received_allocated=got,
            did_not_receive_allocated=None if got is None else int(rows.sum()) - got,
            followed_up=int(obs.sum()),
            lost_to_follow_up=int((rows & ~sets.outcome_observed).sum()),
            analyzed_itt=int(obs.sum()), analyzed_itt_sensitivity=int(rows.sum()),
            per_protocol=None if pp is None else int(pp.sum()),
            excluded_from_per_protocol=None if pp is None else int((rows & ~pp).sum()),
            analyzed_per_protocol=None if pp is None else int((pp & sets.outcome_observed).sum()),
            clusters_allocated=None if cluster is None else int(cluster[rows].nunique()),
            clusters_analyzed=None if cluster is None else int(cluster[obs].nunique()),
            cluster_sizes=None if cluster is None else
            [int(v) for v in cluster[rows].value_counts().sort_index().to_numpy()])
        out.append(flow)
    return ConsortFlow(assessed=len(frame), not_randomized=int((~sets.randomized).sum()),
                       randomized=int(sets.randomized.sum()), arms=out,
                       cluster_trial=cluster is not None)


# ── the baseline table ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineRow:
    variable: str
    level: str | None  # a category's level; None for a number
    kind: str  # number · category
    by_arm: dict[str, dict[str, float | int | None]]  # arm -> its summaries


@dataclass(frozen=True)
class BaselineTable:
    arms: list[str]
    n: dict[str, int]
    rows: list[BaselineRow]
    note: str = ("Each arm is described; no test compares them, because any difference between "
                 "randomized arms at baseline is due to chance (CONSORT 2010 item 15).")


def baseline_table(frame: pd.DataFrame, spec: TrialSpec, variables: Sequence[str], *,
                   analysis_set: str = "itt", tests: bool = False,
                   goal: str = "inference", survey_design: Any = None) -> BaselineTable:
    """Each arm described at baseline: a number by its mean, SD, median and quartiles; a category
    by its count and percentage of those with a value; each with its missing count. ``tests``
    is refused: a baseline difference between randomized arms is chance by construction. A survey
    design is refused (the table describes the randomized people, unweighted; the sample-only
    answer is its exit)."""
    _check_goal(goal, "baseline_table")
    _check_survey(survey_design)
    if tests:
        raise TrialRefused(
            "Baseline differences between randomized arms are described, not tested: chance "
            "decided who went to which arm, so a test only asks whether chance happened, and a "
            "covariate worth adjusting for is named before the data are seen, not after a test "
            "flags it.", (DESCRIBE_ARMS_EXIT,), term="significance tests of baseline balance")
    _check_design(spec)
    sets = analysis_sets(frame, spec)
    arms = _arms(frame, spec)
    keep = sets.mask(analysis_set)
    rows: list[BaselineRow] = []
    n = {a: int((keep & (sets.arm == a)).sum()) for a in arms}
    for var in variables:
        if var not in frame.columns:
            raise ValueError(f"column not in the data: {var!r}")
        col = frame[var]
        numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col) \
            and col.dropna().nunique() > 2
        if numeric:
            by: dict[str, dict[str, Any]] = {}
            for a in arms:
                v = col[keep & (sets.arm == a)].astype(float)
                x = v.dropna().to_numpy()
                q = np.quantile(x, [0.25, 0.5, 0.75]) if len(x) else [None] * 3
                by[a] = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None,
                         "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                         "median": None if not len(x) else float(q[1]),
                         "q1": None if not len(x) else float(q[0]),
                         "q3": None if not len(x) else float(q[2]),
                         "missing": int(v.isna().sum())}
            rows.append(BaselineRow(var, None, "number", by))
            continue
        labels = col.map(lambda v: _label(v) if pd.notna(v) else pd.NA)
        for level in sorted(set(labels.dropna())):
            by = {}
            for a in arms:
                v = labels[keep & (sets.arm == a)]
                have = int(v.notna().sum())
                count = int((v == level).sum())
                by[a] = {"count": count, "percent": 100.0 * count / have if have else None,
                         "missing": int(v.isna().sum())}
            rows.append(BaselineRow(var, level, "category", by))
    return BaselineTable(arms=arms, n=n, rows=rows)


# ── the model matrix ─────────────────────────────────────────────────────────


@dataclass
class _Design:
    X: np.ndarray  # rows: the analyzed rows
    names: list[str]
    arm_codes: np.ndarray  # 0 = control
    arm_columns: list[int]  # the column of each non-control arm
    filled: dict[str, int]  # covariate -> missing values filled
    index: pd.Index


def _numeric_covariate(col: pd.Series) -> bool:
    """A number (or yes/no as 1 and 0) enters as itself; text enters as a category."""
    return pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)


def _design(frame: pd.DataFrame, spec: TrialSpec, arms: list[str], rows: pd.Series,
            pool: pd.Series, arm_text: pd.Series) -> _Design:
    """The model matrix on ``rows``: the intercept, one indicator per non-control arm, the
    randomization factors as categories, then the baseline outcome and the covariates. A numeric
    covariate's missing values take its mean over ``pool`` (the analysis set, never the arm or the
    outcome) with a missing-value indicator beside it; a category's missing values are a level of
    their own (White & Thompson 2005). Columns constant over ``rows`` (a level only outside them)
    are left out."""
    idx = frame.index[rows.to_numpy()]
    sub = frame.loc[idx]
    arm_codes = np.array([arms.index(a) for a in arm_text.loc[idx]], dtype=int)
    cols: list[np.ndarray] = [np.ones(len(idx))]
    names = ["(Intercept)"]
    arm_columns = []
    for k, a in enumerate(arms[1:], start=1):
        arm_columns.append(len(cols))
        cols.append((arm_codes == k).astype(float))
        names.append(f"arm[{a}]")
    filled: dict[str, int] = {}

    def categorical(c: str) -> None:
        lab = sub[c].map(lambda v: _label(v) if pd.notna(v) else "(missing)")
        if lab.eq("(missing)").any():
            filled[c] = int(lab.eq("(missing)").sum())
        levels = sorted(set(lab))
        for level in levels[1:]:
            cols.append((lab == level).to_numpy(dtype=float))
            names.append(f"{c}[{level}]")

    for c in spec.strata:
        categorical(c)
    for c in ([spec.baseline] if spec.baseline else []) + [c for c in spec.covariates
                                                          if c != spec.baseline]:
        col = frame[c]
        if _numeric_covariate(col):
            v = sub[c].astype(float)
            miss = v.isna()
            if miss.any():
                mean = float(frame.loc[pool.to_numpy(), c].astype(float).mean())
                filled[c] = int(miss.sum())
                cols.append(v.fillna(mean).to_numpy())
                names.append(c)
                cols.append(miss.to_numpy(dtype=float))
                names.append(f"{c} missing")
            else:
                cols.append(v.to_numpy())
                names.append(c)
        else:
            categorical(c)
    X = np.column_stack(cols)
    keep = [j for j in range(X.shape[1]) if j == 0 or np.ptp(X[:, j]) > 0]
    if any(j in arm_columns for j in range(X.shape[1]) if j not in keep):
        empty = [arms[arm_columns.index(j) + 1] for j in arm_columns if j not in keep]
        raise TrialRefused(f"No one in the analysis is in arm {', '.join(empty)}, so it cannot be "
                           f"compared.", (ITT_EXIT,), term="empty arm")
    X = X[:, keep]
    names = [names[j] for j in keep]
    arm_columns = [keep.index(j) for j in arm_columns]
    _full_rank(X, names, spec)
    return _Design(X=X, names=names, arm_codes=arm_codes, arm_columns=arm_columns, filled=filled,
                   index=idx)


def _full_rank(X: np.ndarray, names: list[str], spec: TrialSpec) -> None:
    rank = 0
    for j in range(X.shape[1]):
        r = np.linalg.matrix_rank(X[:, :j + 1])
        if r == rank:
            term = names[j].split("[")[0].removesuffix(" missing")
            raise TrialRefused(
                f"{term!r} adds nothing the other terms do not already say about these people (it "
                f"is determined by them), so the model cannot separate its part.",
                ({"label": f"Leave {term} out of the adjustment", "drop": term},),
                term="collinear adjustment term")
        rank = r
    if X.shape[0] <= X.shape[1]:
        raise TrialRefused(f"{X.shape[0]} people are analyzed for {X.shape[1]} model terms; the "
                           f"model needs more people than terms.",
                           ({"label": "Adjust for fewer covariates", "covariates": []},),
                           term="more terms than people")


# ── fits ─────────────────────────────────────────────────────────────────────


@dataclass
class _OLS:
    beta: np.ndarray
    cov: np.ndarray
    df: int
    sigma2: float
    xtx_inv: np.ndarray
    rss: float


def _ols(X: np.ndarray, y: np.ndarray) -> _OLS:
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - X @ beta
    df = X.shape[0] - X.shape[1]
    rss = float(resid @ resid)
    Rinv = np.linalg.inv(R)
    xtx_inv = Rinv @ Rinv.T
    s2 = rss / df
    return _OLS(beta=beta, cov=s2 * xtx_inv, df=df, sigma2=s2, xtx_inv=xtx_inv, rss=rss)


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class _Logit:
    beta: np.ndarray
    cov: np.ndarray  # the inverse information
    mu: np.ndarray


def _logistic(X: np.ndarray, y: np.ndarray, *, offset: np.ndarray | None = None) -> _Logit:
    """Maximum likelihood by Newton's method to |step| ≤ 1e-12 (R's ``glm`` with a tight
    ``epsilon``); a fitted risk of 0 or 1 or no convergence (separation) is refused."""
    off = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(X.shape[1])
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta[0] = math.log(p0 / (1 - p0))
    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for _ in range(100):
                mu = _expit(X @ beta + off)
                w = mu * (1 - mu)
                info = X.T @ (X * w[:, None])
                step = np.linalg.solve(info, X.T @ (y - mu))
                beta = beta + step
                if not np.all(np.isfinite(beta)):
                    raise _separation()
                if np.max(np.abs(step)) <= 1e-12 * (1 + np.max(np.abs(beta))):
                    break
            else:
                raise _separation()
            mu = _expit(X @ beta + off)
            if np.max(np.abs(beta)) > 25 or np.min(mu) < 1e-10 or np.max(mu) > 1 - 1e-10:
                raise _separation()
            w = mu * (1 - mu)
            cov = np.linalg.inv(X.T @ (X * w[:, None]))
    except np.linalg.LinAlgError:
        raise _separation() from None
    return _Logit(beta=beta, cov=(cov + cov.T) / 2, mu=mu)


def _separation() -> TrialRefused:
    return TrialRefused(
        "Some group of people here all had the event, or none did, so the model's risk for them "
        "runs to 0% or 100% and no finite estimate exists.",
        ({"label": "Adjust for fewer covariates", "covariates": []}, DESCRIBE_OUTCOMES_EXIT),
        term="separation in the logistic model")


# ── results ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Contrast:
    """One arm compared with the control arm."""

    arm: str
    control: str
    measure: str  # mean_difference · risk_difference · risk_ratio
    estimate: float  # a difference, or a ratio (risk_ratio)
    lower: float
    upper: float
    se: float  # on the scale the interval is built on (the log scale for a ratio)
    df: float | None  # None: a z interval
    p: float


@dataclass(frozen=True)
class ArmEstimate:
    """An arm's adjusted mean, or its standardized risk."""

    arm: str
    estimate: float
    se: float
    lower: float
    upper: float
    df: float | None
    n: int


@dataclass(frozen=True)
class TrialEffect:
    analysis_set: str
    trial_design: str
    outcome: str
    outcome_type: str  # continuous · binary
    measure: str  # the primary measure
    estimator: str
    contrasts: list[Contrast]
    arm_estimates: list[ArmEstimate]
    n_analyzed: dict[str, int]
    n_in_set: dict[str, int]
    n_missing_outcome: dict[str, int]
    adjusted_for: list[str]
    filled_baselines: dict[str, int]
    terms: list[str]
    coefficients: list[float]
    interval: str  # how the intervals were built
    causal: bool  # causal wording allowed: intention to treat under a declared randomization
    secondary: bool = False
    covariates_prespecified: bool = True
    clusters: dict[str, int] | None = None  # arm -> clusters analyzed
    icc: float | None = None
    icc_source: str | None = None
    cluster_size_cv: float | None = None
    correction: str | None = None
    variance_components: dict[str, float] | None = None
    concerns: list[str] = field(default_factory=list)
    noticing: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contrast(self, arm: str | None = None, measure: str | None = None) -> Contrast:
        for c in self.contrasts:
            if (arm is None or c.arm == arm) and c.measure == (measure or self.measure):
                return c
        raise KeyError((arm, measure))

    @property
    def says(self) -> str:
        return _says(self)


def _t_interval(est: float, se: float, df: float | None) -> tuple[float, float, float]:
    q = stats.norm.ppf(0.5 + CONF / 2) if df is None or not math.isfinite(df) else \
        stats.t.ppf(0.5 + CONF / 2, df)
    if se <= 0 or not math.isfinite(se):
        return est, est, float("nan")
    z = abs(est) / se
    p = float(2 * (stats.norm.sf(z) if df is None or not math.isfinite(df) else stats.t.sf(z, df)))
    return est - q * se, est + q * se, p


def _difference(arm: str, control: str, measure: str, est: float, se: float,
                df: float | None) -> Contrast:
    lo, hi, p = _t_interval(est, se, df)
    return Contrast(arm, control, measure, float(est), float(lo), float(hi), float(se),
                    None if df is None else float(df), p)


def _ratio(arm: str, control: str, log_est: float, se: float, df: float | None) -> Contrast:
    lo, hi, p = _t_interval(log_est, se, df)
    return Contrast(arm, control, "risk_ratio", float(math.exp(log_est)), float(math.exp(lo)),
                    float(math.exp(hi)), float(se), None if df is None else float(df), p)


# ── parallel trials ──────────────────────────────────────────────────────────


def _ancova(d: _Design, y: np.ndarray, arms: list[str]) -> tuple[list[Contrast], list[ArmEstimate],
                                                                 _OLS]:
    fit = _ols(d.X, y)
    contrasts = [_difference(arms[k + 1], arms[0], "mean_difference", fit.beta[j],
                             math.sqrt(fit.cov[j, j]), fit.df)
                 for k, j in enumerate(d.arm_columns)]
    means = []
    xbar = d.X.mean(axis=0)
    for k, a in enumerate(arms):
        L = xbar.copy()
        L[d.arm_columns] = 0.0
        if k:
            L[d.arm_columns[k - 1]] = 1.0
        est = float(L @ fit.beta)
        se = math.sqrt(float(L @ fit.cov @ L))
        lo, hi, _ = _t_interval(est, se, fit.df)
        means.append(ArmEstimate(a, est, se, lo, hi, float(fit.df), int((d.arm_codes == k).sum())))
    return contrasts, means, fit


def _counterfactual(d: _Design, beta: np.ndarray, arms: list[str]) -> np.ndarray:
    """Every analyzed row's predicted risk with its arm set to each arm in turn (n × arms)."""
    out = np.empty((d.X.shape[0], len(arms)))
    for k in range(len(arms)):
        Xk = d.X.copy()
        Xk[:, d.arm_columns] = 0.0
        if k:
            Xk[:, d.arm_columns[k - 1]] = 1.0
        out[:, k] = _expit(Xk @ beta)
    return out


def ye_variance(y: np.ndarray, arm_codes: np.ndarray, cf: np.ndarray, fitted: np.ndarray,
                strata: np.ndarray | None = None) -> np.ndarray:
    """The covariance of the standardized arm means (Ye, Shao, Yi & Zhao), as R's ``beeca``
    computes it (``varcov_ye``, ``mod = FALSE``): for arm j with share π_j,
    V_jj = [Var_j(Y) − 2 Cov_j(Y, μ̂_j) + Var(μ̂_j)]/π_j + 2 Cov_j(Y, μ̂_j) − Var(μ̂_j) and
    V_jk = Cov_j(Y, μ̂_k) + Cov_k(Y, μ̂_j) − Cov(μ̂_j, μ̂_k), where Var_j and Cov_j are over arm j's
    rows and the rest over every row (n − 1 denominators); under stratified randomization the
    term Σ_z (n_z/n) R_z (diag π − ππᵀ) R_z is subtracted, R_z = diag over arms of the mean
    residual in arm j and stratum z over π_j; all over n."""
    n, K = cf.shape
    pi = np.bincount(arm_codes, minlength=K) / n

    def cov(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.cov(a, b, ddof=1)[0, 1])

    V = np.zeros((K, K))
    for j in range(K):
        m = arm_codes == j
        yj, pj = y[m], cf[m, j]
        cop = cov(yj, pj)
        vpred = float(np.var(cf[:, j], ddof=1))
        vop = float(np.var(yj, ddof=1)) - 2 * cop + vpred
        V[j, j] = vop / pi[j] + 2 * cop - vpred
        for k in range(K):
            if k != j:
                mk = arm_codes == k
                V[j, k] = cov(yj, cf[m, k]) + cov(y[mk], cf[mk, j]) - cov(cf[:, j], cf[:, k])
    if strata is not None:
        resid = y - fitted
        omega = np.diag(pi) - np.outer(pi, pi)
        for z in np.unique(strata):
            in_z = strata == z
            rb = np.zeros(K)
            for j in range(K):
                cell = in_z & (arm_codes == j)
                rb[j] = resid[cell].mean() / pi[j] if cell.any() else 0.0
            R = np.diag(rb)
            V -= R @ omega @ R * (in_z.sum() / n)
    return V / n


def _strata_codes(frame: pd.DataFrame, spec: TrialSpec, idx: pd.Index) -> np.ndarray | None:
    if not spec.strata:
        return None
    lab = frame.loc[idx, list(spec.strata)].apply(
        lambda r: "-".join(_label(v) if pd.notna(v) else "(missing)" for v in r), axis=1)
    return pd.factorize(lab)[0]


def _standardized(d: _Design, y: np.ndarray, arms: list[str], strata: np.ndarray | None
                  ) -> tuple[list[Contrast], list[ArmEstimate], _Logit, np.ndarray, np.ndarray]:
    fit = _logistic(d.X, y)
    cf = _counterfactual(d, fit.beta, arms)
    theta = cf.mean(axis=0)
    V = ye_variance(y, d.arm_codes, cf, fit.mu, strata)
    contrasts: list[Contrast] = []
    for k in range(1, len(arms)):
        se = math.sqrt(V[k, k] + V[0, 0] - 2 * V[0, k])
        contrasts.append(_difference(arms[k], arms[0], "risk_difference", theta[k] - theta[0],
                                     se, None))
        g = np.zeros(len(arms))
        g[k], g[0] = 1 / theta[k], -1 / theta[0]
        contrasts.append(_ratio(arms[k], arms[0], math.log(theta[k] / theta[0]),
                                math.sqrt(float(g @ V @ g)), None))
    risks = []
    for k, a in enumerate(arms):
        se = math.sqrt(V[k, k])
        lo, hi, _ = _t_interval(theta[k], se, None)
        risks.append(ArmEstimate(a, float(theta[k]), se, lo, hi, None,
                                 int((d.arm_codes == k).sum())))
    return contrasts, risks, fit, theta, V


# ── cluster-randomized trials: the mixed model with Kenward–Roger ────────────


@dataclass(frozen=True)
class KenwardRoger:
    """pbkrtest's pieces: the adjusted covariance Φ_A, the P matrices and W (2 × the inverse of
    the expected information of the variance parameters σ²_u, σ²_e)."""

    phi: np.ndarray
    phi_adjusted: np.ndarray
    P: tuple[np.ndarray, np.ndarray]
    W: np.ndarray


def kenward_roger(X: np.ndarray, codes: np.ndarray, sigma2_u: float, sigma2_e: float,
                  phi: np.ndarray) -> KenwardRoger:
    """Kenward & Roger's adjusted covariance for the random-intercept model, as ``pbkrtest``'s
    ``vcovAdj`` computes it (no second derivatives: Σ = σ²_u ZZᵀ + σ²_e I is linear in its
    parameters). Every N × N product is a sum over clusters of closed forms in each cluster's
    size m, column sums s = X_gᵀ1 and cross-products X_gᵀX_g, with Σ_g⁻¹ = aI + b_gJ,
    a = 1/σ²_e, d_g = 1/(σ²_e + m σ²_u) = a + b_g m."""
    G = int(codes.max()) + 1
    P_ = X.shape[1]
    m = np.bincount(codes, minlength=G).astype(float)
    S = np.zeros((G, P_))
    np.add.at(S, codes, X)
    a = 1.0 / sigma2_e
    d = 1.0 / (sigma2_e + m * sigma2_u)
    b = (d - a) / m
    SS = np.einsum("gi,gj->gij", S, S)  # s sᵀ per cluster
    XtX = X.T @ X

    def ssum(w: np.ndarray) -> np.ndarray:
        return np.einsum("g,gij->ij", w, SS)

    c2 = 2 * a * b + b * b * m  # Σ⁻² = a²I + c2 J
    c3 = 3 * a * a * b + 3 * a * b * b * m + b ** 3 * m * m  # Σ⁻³ = a³I + c3 J
    P_u = -ssum(d ** 2)
    P_e = -(a * a * XtX + ssum(c2))
    Q_uu = ssum(d ** 3 * m)
    Q_ue = ssum(d ** 3)
    Q_ee = a ** 3 * XtX + ssum(c3)
    K_uu = float(np.sum(d ** 2 * m ** 2))
    K_ue = float(np.sum(d ** 2 * m))
    K_ee = float(np.sum(m * a * a + c2 * m))
    P = (P_u, P_e)
    Q = {(0, 0): Q_uu, (0, 1): Q_ue, (1, 1): Q_ee}
    Kt = np.array([[K_uu, K_ue], [K_ue, K_ee]])
    IE2 = np.zeros((2, 2))
    for i in range(2):
        for j in range(i, 2):
            IE2[i, j] = IE2[j, i] = (Kt[i, j] - 2 * float(np.sum(phi * Q[(i, j)]))
                                     + float(np.sum((phi @ P[i]) * (P[j] @ phi))))
    W = 2 * np.linalg.inv(IE2) if np.min(np.abs(np.linalg.eigvalsh(IE2))) > 1e-10 else \
        2 * np.linalg.pinv(IE2)
    W = (W + W.T) / 2
    U = W[0, 1] * (Q[(0, 1)] - P[0] @ phi @ P[1])
    U = U + U.T
    for i in range(2):
        U = U + W[i, i] * (Q[(i, i)] - P[i] @ phi @ P[i])
    phi_a = phi + 2 * phi @ U @ phi
    return KenwardRoger(phi=phi, phi_adjusted=(phi_a + phi_a.T) / 2, P=P, W=W)


def kr_df(L: np.ndarray, kr: KenwardRoger) -> float:
    """The Kenward–Roger denominator degrees of freedom for the contrasts ``L`` (rows), as
    ``pbkrtest::Lb_ddf`` computes them."""
    L = np.atleast_2d(np.asarray(L, dtype=float))
    V0 = kr.phi
    Theta = L.T @ np.linalg.solve(L @ V0 @ L.T, L)
    A1 = A2 = 0.0
    TV = Theta @ V0
    for i in range(2):
        for j in range(i, 2):
            e = 1.0 if i == j else 2.0
            ui = TV @ kr.P[i] @ V0
            uj = TV @ kr.P[j] @ V0
            A1 += e * kr.W[i, j] * float(np.trace(ui)) * float(np.trace(uj))
            A2 += e * kr.W[i, j] * float(np.sum(ui * uj.T))
    q = L.shape[0]
    B = (A1 + 6 * A2) / (2 * q)
    g = ((q + 1) * A1 - (q + 4) * A2) / ((q + 2) * A2)
    c1 = g / (3 * q + 2 * (1 - g))
    c2 = (q - g) / (3 * q + 2 * (1 - g))
    c3 = (q + 2 - g) / (3 * q + 2 * (1 - g))
    V0_ = 1 + c1 * B
    V1 = 1 - c2 * B
    V2 = 1 - c3 * B
    V0_ = 0.0 if abs(V0_) < 1e-10 else V0_
    rho = (1 / q) * ((1 - A2 / q) / V1 if V1 != 0 else 0.0) ** 2 * V0_ / V2
    return float(4 + (q + 2) / (q * rho - 1))


def _cluster_level(frame: pd.DataFrame, spec: TrialSpec, column: str, idx: pd.Index) -> bool:
    """Whether ``column`` takes one value within every cluster of the analyzed rows."""
    sub = frame.loc[idx, [spec.cluster, column]].astype({column: "string"}).fillna("(missing)")
    return bool((sub.groupby(spec.cluster)[column].nunique() <= 1).all())


def _cluster_codes(frame: pd.DataFrame, spec: TrialSpec, idx: pd.Index) -> np.ndarray:
    return pd.factorize(frame.loc[idx, spec.cluster].map(_label))[0].astype(np.int64)


def _mixed(d: _Design, y: np.ndarray, codes: np.ndarray, arms: list[str]
           ) -> tuple[list[Contrast], list[ArmEstimate], Any, KenwardRoger]:
    from turbotab.core.models.repeated import fit_random_intercept

    fit = fit_random_intercept(d.X, y, codes, satterthwaite=False)
    kr = kenward_roger(d.X, codes, fit.sigma2_u, fit.sigma2_e, fit.cov)
    contrasts = []
    for k, j in enumerate(d.arm_columns):
        L = np.zeros(d.X.shape[1])
        L[j] = 1.0
        df = kr_df(L, kr)
        contrasts.append(_difference(arms[k + 1], arms[0], "mean_difference", fit.beta[j],
                                     math.sqrt(kr.phi_adjusted[j, j]), df))
    means = []
    xbar = d.X.mean(axis=0)
    for k, a in enumerate(arms):
        L = xbar.copy()
        L[d.arm_columns] = 0.0
        if k:
            L[d.arm_columns[k - 1]] = 1.0
        est = float(L @ fit.beta)
        se = math.sqrt(float(L @ kr.phi_adjusted @ L))
        df = kr_df(L, kr)
        lo, hi, _ = _t_interval(est, se, df)
        means.append(ArmEstimate(a, est, se, lo, hi, df, int((d.arm_codes == k).sum())))
    return contrasts, means, fit, kr


# ── cluster-randomized trials: GEE with a corrected sandwich ─────────────────


@dataclass(frozen=True)
class GEEResult:
    beta: np.ndarray
    alpha: float  # the exchangeable working correlation
    phi: float  # the scale
    mu: np.ndarray
    iterations: int


def fit_gee(X: np.ndarray, y: np.ndarray, codes: np.ndarray, family: str) -> GEEResult:
    """GEE with an exchangeable working correlation as R's ``gee`` (4.13) fits it: start from the
    independence fit (``glm``); then, until β stops moving, compute the Pearson residuals r at the
    current β, the scale φ = Σ r²/(N − p) and
    α = Σ_g Σ_{j≠k} r_gj r_gk / (φ(Σ_g m_g(m_g − 1) − 2p)), and take one Fisher-scoring step
    with R_g = (1 − α)I + αJ."""
    from turbotab.core.models.repeated import gee_working

    task = "regression" if family == "gaussian" else "binary"
    beta = _ols(X, y).beta if family == "gaussian" else _logistic(X, y).beta
    G = int(codes.max()) + 1
    m = np.bincount(codes, minlength=G).astype(float)
    N, p = X.shape
    pairs = float(np.sum(m * (m - 1)))
    alpha = phi = 0.0
    for it in range(1, GEE_MAXITER + 1):
        mu = X @ beta if family == "gaussian" else _expit(X @ beta)
        v = np.ones(N) if family == "gaussian" else mu * (1 - mu)
        r = (y - mu) / np.sqrt(v)
        phi = float(r @ r) / (N - p)
        s1 = np.bincount(codes, weights=r, minlength=G)
        s2 = np.bincount(codes, weights=r * r, minlength=G)
        alpha = float(np.sum(s1 ** 2 - s2)) / (phi * (pairs - 2 * p))
        W, rw = gee_working(X, y, mu, codes, alpha, task)
        step = np.linalg.solve(W.T @ W, W.T @ rw)
        beta = beta + step
        if np.max(np.abs(step)) <= GEE_TOL * max(1.0, float(np.max(np.abs(beta)))):
            break
    else:
        raise TrialRefused("The estimating equations did not settle on an answer.",
                           (GEE_EXIT,), term="GEE did not converge")
    mu = X @ beta if family == "gaussian" else _expit(X @ beta)
    if family == "binary" and (np.min(mu) < 1e-10 or np.max(mu) > 1 - 1e-10):
        raise _separation()
    return GEEResult(beta=beta, alpha=alpha, phi=phi, mu=mu, iterations=it)


def gee_covariance(X: np.ndarray, y: np.ndarray, codes: np.ndarray, beta: np.ndarray,
                   alpha: float, family: str, correction: str) -> np.ndarray:
    """The GEE sandwich Ω M Ω with Ω = (Σ_g D_gᵀV_g⁻¹D_g)⁻¹ (the scale cancels), on the whitened
    working model W_g = V_g^(−½)D_g, r̃_g = V_g^(−½)(y_g − μ_g), H̃_g = W_g Ω W_gᵀ:

    * ``lz`` (Liang & Zeger): M = Σ W_gᵀ r̃_g r̃_gᵀ W_g;
    * ``kc`` (Kauermann & Carroll): r̃_g replaced by (I − H̃_g)^(−½) r̃_g, the symmetric root (with
      V_g^(½) this is the principal root (I − H_g)^(−½) for H_g = D_g Ω D_gᵀ V_g⁻¹);
    * ``md`` (Mancl & DeRouen): r̃_g replaced by (I − H̃_g)^(−1) r̃_g;
    * ``fg`` (Fay & Graubard): W_gᵀ r̃_g multiplied by diag((1 − min(b, [Q_g]_jj))^(−½)),
      Q_g = W_gᵀW_g Ω, b = 0.75, as ``geesmv::GEE.var.fg`` computes it.

    (I − H̃_g)^s is applied through the eigenvectors of H̃_g's p-dimensional part, so no m × m
    power is formed."""
    from turbotab.core.models.repeated import gee_working

    task = "regression" if family == "gaussian" else "binary"
    mu = X @ beta if family == "gaussian" else _expit(X @ beta)
    W, r = gee_working(X, y, mu, codes, alpha, task)
    Omega = np.linalg.inv(W.T @ W)
    p = X.shape[1]
    meat = np.zeros((p, p))
    power = {"kc": -0.5, "md": -1.0}.get(correction)
    for g in range(int(codes.max()) + 1):
        rows = codes == g
        Wg, rg = W[rows], r[rows]
        if correction == "fg":
            Q = (Wg.T @ Wg) @ Omega
            c = (1 - np.minimum(FG_BOUND, np.diag(Q))) ** -0.5
            u = c * (Wg.T @ rg)
        elif power is not None:
            Qg, Rg = np.linalg.qr(Wg)
            B = Rg @ Omega @ Rg.T
            lam, Z = np.linalg.eigh((B + B.T) / 2)
            U = Qg @ Z
            f = np.clip(1 - lam, 1e-12, None) ** power - 1
            u = Wg.T @ (rg + U @ (f * (U.T @ rg)))
        elif correction == "lz":
            u = Wg.T @ rg
        else:
            raise ValueError(f"correction {correction!r} is not one of lz, kc, md, fg")
        meat += np.outer(u, u)
    V = Omega @ meat @ Omega
    return (V + V.T) / 2


def cluster_level_columns(X: np.ndarray, codes: np.ndarray) -> list[int]:
    """The columns constant within every cluster (the intercept, the arm, cluster-level terms)."""
    out = []
    for j in range(X.shape[1]):
        lo = np.full(int(codes.max()) + 1, np.inf)
        hi = np.full_like(lo, -np.inf)
        np.minimum.at(lo, codes, X[:, j])
        np.maximum.at(hi, codes, X[:, j])
        if np.all(hi - lo <= 1e-12 * max(1.0, float(np.max(np.abs(X[:, j]))))):
            out.append(j)
    return out


def size_cv(codes: np.ndarray) -> float:
    """The coefficient of variation of the cluster sizes: their SD (n − 1) over their mean."""
    m = np.bincount(codes).astype(float)
    return float(m.std(ddof=1) / m.mean()) if len(m) > 1 else 0.0


def _gee(d: _Design, y: np.ndarray, codes: np.ndarray, arms: list[str], family: str,
         correction: str, df: int) -> tuple[list[Contrast], list[ArmEstimate], GEEResult,
                                            np.ndarray]:
    fit = fit_gee(d.X, y, codes, family)
    V = gee_covariance(d.X, y, codes, fit.beta, fit.alpha, family, correction)
    contrasts: list[Contrast] = []
    estimates: list[ArmEstimate] = []
    if family == "gaussian":
        for k, j in enumerate(d.arm_columns):
            contrasts.append(_difference(arms[k + 1], arms[0], "mean_difference", fit.beta[j],
                                         math.sqrt(V[j, j]), df))
        xbar = d.X.mean(axis=0)
        for k, a in enumerate(arms):
            L = xbar.copy()
            L[d.arm_columns] = 0.0
            if k:
                L[d.arm_columns[k - 1]] = 1.0
            est, se = float(L @ fit.beta), math.sqrt(float(L @ V @ L))
            lo, hi, _ = _t_interval(est, se, df)
            estimates.append(ArmEstimate(a, est, se, lo, hi, float(df),
                                         int((d.arm_codes == k).sum())))
        return contrasts, estimates, fit, V
    theta, grads = [], []
    for k in range(len(arms)):
        Xk = d.X.copy()
        Xk[:, d.arm_columns] = 0.0
        if k:
            Xk[:, d.arm_columns[k - 1]] = 1.0
        mu = _expit(Xk @ fit.beta)
        theta.append(float(mu.mean()))
        grads.append((Xk * (mu * (1 - mu))[:, None]).mean(axis=0))
    for k in range(1, len(arms)):
        g = grads[k] - grads[0]
        contrasts.append(_difference(arms[k], arms[0], "risk_difference", theta[k] - theta[0],
                                     math.sqrt(float(g @ V @ g)), df))
        g = grads[k] / theta[k] - grads[0] / theta[0]
        contrasts.append(_ratio(arms[k], arms[0], math.log(theta[k] / theta[0]),
                                math.sqrt(float(g @ V @ g)), df))
    for k, a in enumerate(arms):
        se = math.sqrt(float(grads[k] @ V @ grads[k]))
        lo, hi, _ = _t_interval(theta[k], se, df)
        estimates.append(ArmEstimate(a, theta[k], se, lo, hi, float(df),
                                     int((d.arm_codes == k).sum())))
    return contrasts, estimates, fit, V


# ── the effect ───────────────────────────────────────────────────────────────


@dataclass
class _Prepared:
    arms: list[str]
    sets: AnalysisSets
    in_set: pd.Series
    analyzed: pd.Series
    kind: str
    measure: str


def _prepare(frame: pd.DataFrame, spec: TrialSpec, analysis_set: str, measure: str | None,
             goal: str, survey_design: Any, covariates_prespecified: bool, secondary: bool,
             what: str = "effect") -> _Prepared:
    _check_goal(goal, what)
    _check_survey(survey_design)
    _check_design(spec)
    if analysis_set not in SETS:
        raise ValueError(f"analysis_set {analysis_set!r} is not one of {SETS}")
    if not covariates_prespecified and not secondary:
        raise TrialRefused(
            "Covariates chosen after the data were seen (for instance because the arms looked "
            "different on them) do not enter the primary analysis: the adjustment is decided "
            "before the outcome is analyzed, so it cannot be steered by the result.",
            (LEAVE_OUT_EXIT, SECONDARY_EXIT), term="post hoc covariate adjustment")
    sets = analysis_sets(frame, spec)
    arms = _arms(frame, spec)
    in_set = sets.mask(analysis_set)
    analyzed = in_set & sets.outcome_observed
    kind = _outcome_kind(frame.loc[in_set.to_numpy(), spec.outcome])
    if measure is None:
        measure = "mean_difference" if kind == "continuous" else "risk_difference"
    if measure not in MEASURES:
        raise ValueError(f"measure {measure!r} is not one of {MEASURES}")
    if (kind == "continuous") != (measure == "mean_difference"):
        raise TrialRefused(
            "A risk difference or ratio needs a yes/no outcome, and a mean difference a "
            "numeric one." if kind == "continuous" else
            "For a yes/no outcome the effect is a difference or a ratio of risks.",
            ({"label": "Use the mean difference" if kind == "continuous" else
              "Use the risk difference", "measure": "mean_difference" if kind == "continuous"
              else "risk_difference"},), term="measure does not fit the outcome")
    for a in arms:
        n_a = int((analyzed & (sets.arm == a)).sum())
        if n_a < 2:
            exits = ([ITT_EXIT] if analysis_set != "itt" else []) + [DESCRIBE_OUTCOMES_EXIT]
            raise TrialRefused(f"Arm {a} has {n_a} {'person' if n_a == 1 else 'people'} with an "
                               f"outcome in this analysis; each arm needs at least two.",
                               exits, term="too few outcomes in an arm")
    if spec.trial_design == "cluster_randomized_trial":
        c = frame.loc[in_set.to_numpy(), spec.cluster]
        if c.isna().any():
            raise TrialRefused(
                f"{int(c.isna().sum())} randomized "
                f"{'person has' if int(c.isna().sum()) == 1 else 'people have'} no cluster named "
                f"in {spec.cluster!r}; in a cluster-randomized trial everyone belongs to the "
                f"cluster that was randomized, and leaving them out would break intention to "
                f"treat.", (CLUSTER_COLUMN_EXIT, DESCRIBE_OUTCOMES_EXIT), term="missing cluster")
        arm_of = pd.DataFrame({"c": c.map(_label), "a": sets.arm[in_set]})
        mixed = arm_of.groupby("c")["a"].nunique()
        if (mixed > 1).any():
            raise TrialRefused(
                "People in one cluster are in different arms "
                f"({', '.join(mixed[mixed > 1].index[:3])}"
                f"), so the clusters were not what was randomized.",
                ({"label": "Declare a parallel trial (people were randomized)",
                  "decision": {"kind": "set_design", "design": "parallel_trial"}},
                 {"label": "Name the column of the clusters that were randomized",
                  "cluster": None}),
                term="arm varies within cluster")
    return _Prepared(arms=arms, sets=sets, in_set=in_set, analyzed=analyzed, kind=kind,
                     measure=measure)


def estimate(frame: pd.DataFrame, spec: TrialSpec, *, analysis_set: str = "itt",
             measure: str | None = None, cluster_method: str | None = None,
             correction: str = "auto", goal: str = "inference", survey_design: Any = None,
             covariates_prespecified: bool = True, secondary: bool = False) -> TrialEffect:
    """The effect of the assigned arm (module docstring). ``cluster_method`` (a cluster trial):
    ``mixed`` for a numeric outcome by default, ``gee`` for yes/no (and by choice for a number).
    Raises :class:`TrialRefused` with its exits."""
    pr = _prepare(frame, spec, analysis_set, measure, goal, survey_design,
                  covariates_prespecified, secondary)
    arms, sets = pr.arms, pr.sets
    d = _design(frame, spec, arms, pr.analyzed, pr.in_set, sets.arm)
    y = frame.loc[d.index, spec.outcome].astype(float).to_numpy()
    concerns: list[str] = []
    noticing = None
    clusters = icc = icc_source = cv = used = components = None
    if spec.trial_design == "parallel_trial":
        if pr.kind == "continuous":
            contrasts, estimates, fit = _ancova(d, y, arms)
            estimator = "analysis of covariance (least squares)"
            interval = f"t on the residual degrees of freedom ({fit.df})"
            coefs = fit.beta
        else:
            strata = _strata_codes(frame, spec, d.index)
            contrasts, estimates, lfit, _, _ = _standardized(d, y, arms, strata)
            estimator = ("logistic regression, risks standardized over the analyzed people, "
                         "Ye et al.'s variance" + (" with the stratification term" if
                                                   strata is not None else ""))
            interval = "normal (z), the ratio on the log scale"
            coefs = lfit.beta
    else:
        codes = _cluster_codes(frame, spec, d.index)
        K = int(codes.max()) + 1
        clusters = {a: int(pd.Series(codes[d.arm_codes == k]).nunique())
                    for k, a in enumerate(arms)}
        cv = size_cv(codes)
        method = cluster_method or ("mixed" if pr.kind == "continuous" else "gee")
        if method not in CLUSTER_METHODS:
            raise ValueError(f"cluster_method {method!r} is not one of {CLUSTER_METHODS}")
        level = cluster_level_columns(d.X, codes)
        df = K - len(level)
        per_arm = ", ".join(f"{v} in {a}" for a, v in clusters.items())
        if min(clusters.values()) < 2:
            lone = [a for a, v in clusters.items() if v < 2]
            raise TrialRefused(
                f"{K} clusters are analyzed ({per_arm}). With one cluster in "
                f"{' and '.join(lone)}, that arm's effect cannot be told apart from what is "
                f"particular to its cluster, so no comparison of the arms can be made; each arm "
                f"needs at least two clusters.",
                (CLUSTER_COLUMN_EXIT, DESCRIBE_OUTCOMES_EXIT), term="too few clusters")
        if df < 1:
            level_covs = [c for c in spec.covariates if _cluster_level(frame, spec, c, d.index)]
            exits = ([{"label": f"Leave the cluster-level covariates out "
                                f"({', '.join(level_covs)})",
                       "covariates": [c for c in spec.covariates if c not in level_covs]}]
                     if level_covs else []) + [DESCRIBE_OUTCOMES_EXIT]
            raise TrialRefused(
                f"{K} clusters are analyzed ({per_arm}) for {len(level)} terms that are the "
                f"same for everyone in a cluster; the comparison needs more clusters than those "
                f"terms.", exits, term="too few clusters")
        if K < FEW_CLUSTERS:
            concerns.append(f"Only {K} clusters were randomized; the intervals rest on the "
                            f"small-sample correction, and the fewest clusters Li & Redden (2015) "
                            f"studied was {FEW_CLUSTERS}.")
        if method == "mixed":
            if pr.kind == "binary":
                raise TrialRefused(
                    "The mixed model's small-sample correction (Kenward–Roger) is for a numeric "
                    "outcome; a yes/no outcome is analyzed by the estimating equations with a "
                    "corrected sandwich.", (GEE_EXIT,),
                    term="Kenward–Roger for a binary outcome")
            contrasts, estimates, mfit, _kr = _mixed(d, y, codes, arms)
            estimator = "linear mixed model, a random intercept per cluster (REML)"
            interval = "t on the Kenward–Roger degrees of freedom, with its adjusted covariance"
            icc, icc_source = mfit.icc, "the mixed model's variance components"
            components = {"between_clusters": mfit.sigma2_u, "within_clusters": mfit.sigma2_e}
            if mfit.boundary:
                concerns.append("The between-cluster variance was estimated as zero; the mixed "
                                "model is then least squares.")
            coefs = mfit.beta
        else:
            used = correction if correction != "auto" else ("kc" if cv < CV_RULE else "fg")
            if used not in ("kc", "fg", "md"):
                raise ValueError(f"correction {correction!r} is not one of {CORRECTIONS}")
            family = "gaussian" if pr.kind == "continuous" else "binary"
            contrasts, estimates, gfit, _ = _gee(d, y, codes, arms, family, used, df)
            name = {"kc": "Kauermann–Carroll", "fg": "Fay–Graubard", "md": "Mancl–DeRouen"}[used]
            estimator = (f"generalized estimating equations, exchangeable working correlation, "
                         f"{name} corrected sandwich" + ("" if family == "gaussian" else
                                                         ", risks standardized"))
            interval = f"t on the clusters less the cluster-level terms ({K} − {len(level)} = {df})"
            icc, icc_source = gfit.alpha, "the GEE's exchangeable working correlation"
            coefs = gfit.beta
    if analysis_set == "per_protocol":
        concerns.append("The per-protocol set keeps only those who followed the protocol; who "
                        "adheres may differ between the arms, so this comparison is not "
                        "protected by randomization and is reported beside intention to treat.")
    missing = {a: int((pr.in_set & ~sets.outcome_observed & (sets.arm == a)).sum()) for a in arms}
    if sum(missing.values()):
        share = sum(missing.values()) / int(pr.in_set.sum())
        noticing = (f"{sum(missing.values())} of {int(pr.in_set.sum())} people "
                    f"({100 * share:.0f}%) have no outcome; the analysis assumes they are like "
                    f"those observed with the same arm and covariates, and the tipping-point "
                    f"analysis tests how far that could be wrong.")
    return TrialEffect(
        analysis_set=analysis_set, trial_design=spec.trial_design, outcome=spec.outcome,
        outcome_type=pr.kind, measure=pr.measure, estimator=estimator, contrasts=contrasts,
        arm_estimates=estimates,
        n_analyzed={a: int((pr.analyzed & (sets.arm == a)).sum()) for a in arms},
        n_in_set={a: int((pr.in_set & (sets.arm == a)).sum()) for a in arms},
        n_missing_outcome=missing, adjusted_for=spec.adjusted_for(), filled_baselines=d.filled,
        terms=list(d.names), coefficients=[float(b) for b in coefs], interval=interval,
        causal=analysis_set == "itt" and not secondary, secondary=secondary,
        covariates_prespecified=covariates_prespecified, clusters=clusters,
        icc=None if icc is None else float(icc), icc_source=icc_source,
        cluster_size_cv=cv, correction=used, variance_components=components,
        concerns=concerns, noticing=noticing)


# ── missing outcomes: the delta-adjusted tipping point ───────────────────────


def normal_draw(beta_hat: np.ndarray, xtx_inv: np.ndarray, rss: float, df: int,
                X_missing: np.ndarray, chi2: float, z_beta: np.ndarray,
                z_y: np.ndarray) -> np.ndarray:
    """One proper draw of the missing outcomes from the Bayesian linear regression (Rubin 1987;
    ``mice``'s ``norm``): σ* = √(RSS/χ²), β* = β̂ + σ* L z_β with LLᵀ = (XᵀX)⁻¹ (lower Cholesky),
    y* = X β* + σ* z_y."""
    sigma = math.sqrt(rss / chi2)
    beta = beta_hat + sigma * (np.linalg.cholesky(xtx_inv) @ z_beta)
    return X_missing @ beta + sigma * z_y


def logistic_draw(beta_hat: np.ndarray, cov: np.ndarray, X_missing: np.ndarray, z_beta: np.ndarray,
                  u: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """One draw of missing yes/no outcomes (``mice``'s ``logreg``): β* = β̂ + L z_β with LLᵀ the
    inverse information, p = expit(X β* + δ), y* = 1 when u ≤ p."""
    beta = beta_hat + np.linalg.cholesky(cov) @ z_beta
    return (u <= _expit(X_missing @ beta + shift)).astype(float)


@dataclass
class _Imputer:
    """Every random number the imputations use, drawn once, so a shift δ changes nothing but δ."""

    kind: str
    X: np.ndarray  # every randomized person in the set
    y: np.ndarray  # NaN where missing
    missing: np.ndarray
    arm_codes: np.ndarray
    arm_columns: list[int]
    draws: list[tuple[Any, ...]]
    fit: Any
    strata: np.ndarray | None

    def completed(self, delta: float, arm: int) -> list[np.ndarray]:
        """The m completed outcome vectors with the imputed outcomes of ``arm`` shifted by δ."""
        Xm = self.X[self.missing]
        shift = np.where(self.arm_codes[self.missing] == arm, delta, 0.0)
        out = []
        for dr in self.draws:
            y = self.y.copy()
            if self.kind == "continuous":
                y[self.missing] = normal_draw(self.fit.beta, self.fit.xtx_inv, self.fit.rss,
                                              self.fit.df, Xm, *dr) + shift
            else:
                y[self.missing] = logistic_draw(self.fit.beta, self.fit.cov, Xm, dr[0], dr[1],
                                                shift)
            out.append(y)
        return out


def _imputer(d: _Design, y: np.ndarray, kind: str, m: int, seed: int,
             strata: np.ndarray | None) -> _Imputer:
    miss = np.isnan(y)
    Xo, yo = d.X[~miss], y[~miss]
    if np.linalg.matrix_rank(Xo) < Xo.shape[1]:
        raise TrialRefused("Among the people with an outcome, some adjustment term never varies, "
                           "so the imputation model cannot use it.",
                           ({"label": "Adjust for fewer covariates", "covariates": []},
                            PRIMARY_EXIT), term="imputation model not identified")
    rng = np.random.default_rng(seed)
    n_mis, p = int(miss.sum()), d.X.shape[1]
    if kind == "continuous":
        fit = _ols(Xo, yo)
        draws = [(float(rng.chisquare(fit.df)), rng.standard_normal(p),
                  rng.standard_normal(n_mis)) for _ in range(m)]
    else:
        fit = _logistic(Xo, yo)
        draws = [(rng.standard_normal(p), rng.random(n_mis)) for _ in range(m)]
    return _Imputer(kind=kind, X=d.X, y=y, missing=miss, arm_codes=d.arm_codes,
                    arm_columns=d.arm_columns, draws=draws, fit=fit, strata=strata)


@dataclass(frozen=True)
class TippingRow:
    delta: float
    estimate: float
    lower: float
    upper: float
    p: float
    df: float | None
    excludes_null: bool


@dataclass(frozen=True)
class Tipping:
    arm: str
    control: str
    measure: str
    at_mar: TippingRow  # δ = 0: missing at random
    tipping_delta: float | None  # the nearer shift found to change the conclusion
    searched_to: float  # |δ| searched in each direction
    rows: list[TippingRow]
    scale: str  # the outcome's units · log odds


@dataclass(frozen=True)
class TippingPoint:
    m: int
    seed: int
    n_missing: dict[str, int]
    n_randomized: dict[str, int]
    contrasts: list[Tipping]
    outcome_type: str
    concerns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def says(self) -> str:
        return _tipping_says(self)


def _pool(imp: _Imputer, delta: float, arm: int, measure: str, arms: list[str]) -> TippingRow:
    from turbotab.core.methods.imputation import pool_scalar

    ests, variances = [], []
    df_com: float | None = None
    j = imp.arm_columns[arm - 1]
    for y in imp.completed(delta, arm):
        if imp.kind == "continuous":
            fit = _ols(imp.X, y)
            ests.append(fit.beta[j])
            variances.append(fit.cov[j, j])
            df_com = fit.df
        else:
            d = _Design(X=imp.X, names=[], arm_codes=imp.arm_codes, arm_columns=imp.arm_columns,
                        filled={}, index=pd.RangeIndex(len(y)))
            lfit = _logistic(imp.X, y)
            cf = _counterfactual(d, lfit.beta, arms)
            theta = cf.mean(axis=0)
            V = ye_variance(y, imp.arm_codes, cf, lfit.mu, imp.strata)
            if measure == "risk_ratio":
                g = np.zeros(len(arms))
                g[arm], g[0] = 1 / theta[arm], -1 / theta[0]
                ests.append(math.log(theta[arm] / theta[0]))
                variances.append(float(g @ V @ g))
            else:
                ests.append(theta[arm] - theta[0])
                variances.append(V[arm, arm] + V[0, 0] - 2 * V[0, arm])
    pooled = pool_scalar(ests, variances, df_com)
    est, lo, hi = pooled.estimate, pooled.ci_low, pooled.ci_high
    if measure == "risk_ratio":
        est, lo, hi = math.exp(est), math.exp(lo), math.exp(hi)
    null = 1.0 if measure == "risk_ratio" else 0.0
    return TippingRow(delta=float(delta), estimate=float(est), lower=float(lo), upper=float(hi),
                      p=float(pooled.p), df=pooled.df, excludes_null=bool(lo > null or hi < null))


def delta_analysis(frame: pd.DataFrame, spec: TrialSpec, *, arm: Any, delta: float,
                   m: int | None = None, seed: int = 20261009, measure: str | None = None,
                   goal: str = "inference", survey_design: Any = None,
                   analysis_set: str = "itt") -> tuple[TippingRow, list[pd.Series]]:
    """One delta-adjusted analysis: the pooled result with ``arm``'s imputed outcomes shifted by
    ``delta``, and the m completed outcomes (indexed as the frame), for checking the pooling. It
    refuses what :func:`tipping_point` refuses (a survey design, Predict, the per-protocol set, a
    cluster trial)."""
    pr, d, y, imp = _tipping_setup(frame, spec, m, seed, measure, goal, survey_design,
                                   analysis_set)
    if _label(arm) not in pr.arms[1:]:
        raise ValueError(f"arm {arm!r} is not one of the arms compared with the control: "
                         f"{pr.arms[1:]}")
    k = pr.arms.index(_label(arm))
    row = _pool(imp, delta, k, pr.measure, pr.arms)
    return row, [pd.Series(c, index=d.index, name=spec.outcome) for c in imp.completed(delta, k)]


def _tipping_setup(frame: pd.DataFrame, spec: TrialSpec, m: int | None, seed: int,
                   measure: str | None, goal: str = "inference", survey_design: Any = None,
                   analysis_set: str = "itt") -> tuple[_Prepared, _Design, np.ndarray, _Imputer]:
    from turbotab.core.methods.imputation import m_rule

    if analysis_set != "itt":
        raise TrialRefused("The sensitivity analysis for missing outcomes counts every randomized "
                           "person, which is the intention-to-treat set; the per-protocol set "
                           "has already left people out.", (ITT_EXIT,),
                           term="tipping point outside intention to treat")
    pr = _prepare(frame, spec, "itt", measure, goal, survey_design, True, False, "tipping_point")
    if spec.trial_design == "cluster_randomized_trial":
        raise TrialRefused(
            "Imputing a cluster trial's missing outcomes needs an imputation model that keeps each "
            "cluster's people together (a multilevel imputation), which is not built.",
            (PRIMARY_EXIT,), term="multilevel imputation for a cluster-randomized trial")
    n_missing = int((pr.in_set & ~pr.sets.outcome_observed).sum())
    if not n_missing:
        raise TrialRefused("Every randomized person has an outcome, so there is nothing missing to "
                           "test.", (PRIMARY_EXIT,), term="no missing outcomes")
    d = _design(frame, spec, pr.arms, pr.in_set, pr.in_set, pr.sets.arm)
    y = frame.loc[d.index, spec.outcome].astype(float).to_numpy()
    m_used = int(m) if m else m_rule(n_missing, int(pr.in_set.sum()))
    strata = _strata_codes(frame, spec, d.index) if pr.kind == "binary" else None
    return pr, d, y, _imputer(d, y, pr.kind, m_used, seed, strata)


def tipping_point(frame: pd.DataFrame, spec: TrialSpec, *, measure: str | None = None,
                  m: int | None = None, seed: int = 20261009, goal: str = "inference",
                  survey_design: Any = None, analysis_set: str = "itt",
                  tol: float = 1e-6) -> TippingPoint:
    """The delta-adjusted tipping-point analysis (module docstring), one contrast at a time: the
    shift δ of that arm's imputed outcomes is searched in both directions (doubling from one SD
    of the observed outcomes, or one unit of log odds, up to 16), the change of conclusion
    bracketed and found by bisection to ``tol`` × that unit; the nearer change is the tipping
    point. ``rows`` lists the pooled result at nine shifts from 0 to 1.25 × the tipping point (or
    to the end of the search)."""
    pr, d, y, imp = _tipping_setup(frame, spec, m, seed, measure, goal, survey_design,
                                   analysis_set)
    unit = float(np.nanstd(y, ddof=1)) if pr.kind == "continuous" else 1.0
    limit = 16.0 * unit
    out = []
    for k in range(1, len(pr.arms)):
        cache: dict[float, TippingRow] = {}

        def at(delta: float) -> TippingRow:
            if delta not in cache:
                cache[delta] = _pool(imp, delta, k, pr.measure, pr.arms)
            return cache[delta]

        base = at(0.0)
        found: list[float] = []
        for sign in (-1.0, 1.0):
            lo, step = 0.0, unit
            hit = None
            while abs(lo) < limit:
                hi = sign * min(abs(lo) + step, limit)
                if at(hi).excludes_null != base.excludes_null:
                    hit = hi
                    break
                lo, step = hi, step * 2
            if hit is None:
                continue
            a, b = lo, hit
            while abs(b - a) > tol * unit:
                mid = (a + b) / 2
                if at(mid).excludes_null != base.excludes_null:
                    b = mid
                else:
                    a = mid
            found.append(b)
        tip = min(found, key=abs) if found else None
        favors = base.estimate > (1.0 if pr.measure == "risk_ratio" else 0.0)
        edge = 1.25 * tip if tip is not None else (-limit if favors else limit)
        rows = [at(float(v)) for v in np.linspace(0.0, edge, 9)]
        out.append(Tipping(arm=pr.arms[k], control=pr.arms[0], measure=pr.measure, at_mar=base,
                           tipping_delta=None if tip is None else float(tip),
                           searched_to=float(limit), rows=rows,
                           scale="the outcome's units" if pr.kind == "continuous" else
                           "log odds of the event"))
    sets = pr.sets
    return TippingPoint(
        m=len(imp.draws), seed=seed,
        n_missing={a: int((pr.in_set & ~sets.outcome_observed & (sets.arm == a)).sum())
                   for a in pr.arms},
        n_randomized={a: int((pr.in_set & (sets.arm == a)).sum()) for a in pr.arms},
        contrasts=out, outcome_type=pr.kind)


# ── words ────────────────────────────────────────────────────────────────────


def _num(x: float) -> str:
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"


def _measure_text(c: Contrast) -> str:
    if c.measure == "risk_ratio":
        return f"a risk ratio of {_num(c.estimate)}"
    if c.measure == "risk_difference":
        return f"{_num(100 * c.estimate)} percentage points"
    return _num(c.estimate)


def _excludes_null(c: Contrast) -> bool:
    null = 1.0 if c.measure == "risk_ratio" else 0.0
    return c.lower > null or c.upper < null


def _ci_text(c: Contrast) -> str:
    """The 95% interval on the scale its estimate is printed on (percentage points for a risk
    difference), saying so when it includes no difference."""
    k = 100.0 if c.measure == "risk_difference" else 1.0
    return (f"95% CI {_num(k * c.lower)} to {_num(k * c.upper)}"
            + ("" if _excludes_null(c) else ", which includes no difference"))


def _causal_clause(r: TrialEffect, c: Contrast) -> str:
    """One arm against the control, worded as the effect of assignment, as a sentence. A change
    is asserted only when the interval excludes no difference."""
    if c.measure == "risk_ratio":
        return (f"Assignment to {c.arm} gave {_measure_text(c)} for {r.outcome} ({_ci_text(c)}) "
                f"compared with {c.control}.")
    target = f"the risk of {r.outcome}" if r.outcome_type == "binary" else r.outcome
    what = f"{_measure_text(c)} ({_ci_text(c)}) compared with {c.control}"
    if _excludes_null(c):
        return f"Assignment to {c.arm} changed {target} by {what}."
    return (f"The estimated effect of assignment to {c.arm} on {target} was {what}, so a change "
            f"is not shown.")


def _difference_clause(r: TrialEffect, c: Contrast) -> str:
    """One arm against the control, worded as a difference, never as an effect."""
    if c.measure == "risk_ratio":
        return (f"the risk ratio of {r.outcome} for {c.arm} against {c.control} was "
                f"{_num(c.estimate)} ({_ci_text(c)})")
    return (f"{r.outcome} differed by {_measure_text(c)} ({_ci_text(c)}) between {c.arm} and "
            f"{c.control}")


def _primary_contrasts(r: TrialEffect) -> list[Contrast]:
    """Every arm against the control on the primary measure, in the arms' order."""
    return [c for c in r.contrasts if c.measure == r.measure]


def _says(r: TrialEffect) -> str:
    cs = _primary_contrasts(r)
    if r.causal:
        return (" ".join(_causal_clause(r, c) for c in cs) + " Read as intention to treat: the "
                "effect of being assigned, whether or not the treatment was taken.")
    body = "; ".join(_difference_clause(r, c) for c in cs)
    if r.analysis_set == "per_protocol":
        return (f"Among those who followed the protocol, {body}. Who adheres may differ between "
                f"the arms, so this comparison is not protected by randomization and is not read "
                f"as an effect.")
    why = ("its covariates were chosen after the data were seen, so the model was not fixed in "
           "advance and could have been steered by the result" if not r.covariates_prespecified
           else "it is not the analysis fixed in advance")
    return (f"In this secondary analysis, {body}. The arms are still the randomized arms, but "
            f"{why}; it is reported beside the primary analysis, not read as the trial's effect.")


def _tipping_says(t: TippingPoint) -> str:
    parts = []
    for c in t.contrasts:
        if c.tipping_delta is None:
            parts.append(f"For {c.arm}, no shift of its missing outcomes up to "
                         f"{_num(c.searched_to)} "
                         f"({c.scale}) changed the conclusion.")
        else:
            change = ("no longer excludes" if c.at_mar.excludes_null else "excludes")
            parts.append(f"For {c.arm}, the 95% interval {change} no difference once its missing "
                         f"outcomes are shifted by {_num(c.tipping_delta)} ({c.scale}) from what "
                         f"those observed with the same covariates would predict.")
    return " ".join(parts)


def trial_sentence(itt: TrialEffect | None = None, per_protocol: TrialEffect | None = None,
                   tipping: TippingPoint | None = None) -> str:
    """The methods and results sentences of a trial analysis. A per-protocol result is reported
    beside intention to treat, never alone (refused)."""
    if per_protocol is not None and itt is None:
        raise TrialRefused("A per-protocol result is reported beside the intention-to-treat "
                           "result, never instead of it.", (ITT_EXIT,),
                           term="per-protocol result without intention to treat")
    r = itt
    if r is None:
        design = "randomized trial"
        return (f"The {design} was analyzed by intention to treat (everyone randomized, in the arm "
                "they were randomized to), adjusting for the randomization factors and the "
                "baseline covariates named in advance, for precision only.")
    cluster = r.trial_design == "cluster_randomized_trial"
    sentence = ("The trial was analyzed by intention to treat: everyone randomized, in the arm "
                "they were randomized to.")
    if r.adjusted_for:
        sentence += (f" The effect of the assigned arm was adjusted for {', '.join(r.adjusted_for)}"
                     f", named before the data were seen, for precision only.")
    sentence += f" It was estimated by {r.estimator}"
    if cluster:
        sentence += (" with the cluster as the unit of randomization; the intraclass correlation "
                     f"was {_num(r.icc or 0.0)}.")
    else:
        sentence += "."
    if r.filled_baselines:
        sentence += (" Missing baseline values were filled by their mean (a missing category for "
                     "a category) with an indicator of the missing value (White & Thompson 2005).")
    sentence += " " + r.says
    if per_protocol is not None:
        pp = "; ".join(f"{c.arm} against {c.control}, {_measure_text(c)} ({_ci_text(c)})"
                       for c in _primary_contrasts(per_protocol))
        sentence += (f" Per protocol, among those who followed the protocol: {pp}. This set is "
                     f"not protected by randomization.")
    if tipping is not None:
        sentence += (f" Missing outcomes were multiply imputed ({tipping.m} imputations) under "
                     f"missing at random and shifted by δ to find the tipping point. "
                     + tipping.says)
    return sentence


# ── the contracts ────────────────────────────────────────────────────────────


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "trial_effect" in CONTRACTS:
        return
    here = "turbotab.core.methods.trials"
    est_only = ("inference",)
    not_predict = ("Not offered under Predict: the effect of an assigned treatment is an Estimate "
                   "question")

    def option(key: str, label: str, customary: str, sound: str, rung: str, order: int
               ) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"inference": sound, "prediction": not_predict},
                              {"inference": rung, "prediction": "not_offered"},
                              {"inference": order, "prediction": order})

    def refuse_prediction(id_: str, enforced: str) -> Relation:
        return Relation("conflicts", "prediction_not_offered",
                        "not offered under Predict: an assigned treatment's effect is estimated, "
                        "not predicted", purposes=("prediction",), rung="refused",
                        exits=("ask it under Estimate an effect",),
                        condition="the Predict goal", enforced_by=f"{here}:{enforced}", id=id_)

    def refuse_survey(id_: str, enforced: str) -> Relation:
        return Relation("conflicts", "survey_design_refused",
                        "survey weights are refused: a trial's inference rests on its "
                        "randomization", purposes=est_only, rung="refused",
                        exits=("the sample-only answer",), condition="a survey design",
                        enforced_by=f"{here}:{enforced}", id=id_)

    register_contract(MethodContract(
        key="trial_analysis_set", label="Who a randomized trial analyzes (the analysis set)",
        slot="eligibility", scope="row_local", package="TRIALS", run_order=4.0,
        scope_note=("Whether a person is in the set reads only that person's randomized arm and "
                    "adherence; never the outcome or another person's row."),
        needs=("the arm each person was randomized to, and the control arm",
               "for the per-protocol set: whether each person followed the protocol (yes/no)",
               "optional: the arm each person received (for the CONSORT flow)"),
        question="Who is analyzed?",
        options=(
            option("itt", "Intention to treat: everyone randomized, in the arm they were "
                          "randomized to",
                   "The primary analysis of a randomized trial (Moher et al. 2010, item 16; "
                   "White, Horton, Carpenter & Pocock 2011)",
                   "Sound: randomization protects the comparison, and the effect is that of "
                   "being assigned, whether or not the treatment was taken", "recommended", 0),
            option("per_protocol", "Per protocol: those who followed the protocol, in the arm "
                                   "they were randomized to",
                   "Reported beside intention to treat (Moher et al. 2010, item 16)",
                   "Not protected by randomization: who adheres may differ between the arms; a "
                   "set, not the effect of following the protocol, and reported beside "
                   "intention to treat, never instead", "rank_lower", 1),
        ),
        storyboard=("read each person's randomized arm", "keep everyone randomized (intention "
                    "to treat), or those who followed the protocol (per protocol)",
                    "count each arm by stage for the CONSORT flow"),
        relations=(
            Relation("implies", "no_exclusion_after_randomization",
                     "the primary analysis keeps every randomized person in the arm they were "
                     "randomized to; no one leaves for what happened after randomization",
                     purposes=est_only, when=("itt",), condition="always",
                     enforced_by=f"{here}:analysis_sets", id="itt_keeps_everyone"),
            Relation("conflicts", "per_protocol_effect",
                     "the effect of following the protocol (weighting for adherence over time) "
                     "is not available yet; the per-protocol set analyzed as randomized is",
                     purposes=est_only, rung="refused", when=("per_protocol",),
                     exits=("the per-protocol set, analyzed as randomized",
                            "the intention-to-treat effect"),
                     condition="the per-protocol effect asked for",
                     enforced_by="turbotab.core.designs:effect_refusal", id="pp_effect_v2x"),
            Relation("conflicts", "per_protocol_alone",
                     "a per-protocol result is reported beside intention to treat, never alone",
                     purposes=est_only, rung="refused", when=("per_protocol",),
                     exits=("report intention to treat beside it",),
                     condition="a per-protocol result without the intention-to-treat one",
                     enforced_by=f"{here}:trial_sentence", id="pp_beside_itt"),
            Relation("conflicts", "per_protocol_needs_adherence",
                     "the per-protocol set is refused without a yes/no adherence column",
                     purposes=est_only, rung="refused", when=("per_protocol",),
                     exits=("intention to treat", "name the adherence column"),
                     condition="no adherence column",
                     enforced_by=f"{here}:analysis_sets", id="pp_adherence"),
            Relation("implies", "consort_counts",
                     "each arm's counts by stage (allocated, received, followed up, lost, "
                     "analyzed; clusters in a cluster trial) are returned for the CONSORT flow",
                     purposes=est_only, condition="always", enforced_by=f"{here}:consort_flow",
                     id="consort_flow"),
            refuse_prediction("set_not_predicted", "eligible"),
            refuse_survey("set_no_survey", "consort_flow"),
        ),
        sources=(SOURCES["moher2010"], SOURCES["white2011"],
                 "Schulz et al. 2010, BMJ 340:c332 (CONSORT 2010 statement)",
                 SOURCES["campbell2012"]),
        sentence=f"{here}:trial_sentence"))

    register_contract(MethodContract(
        key="trial_effect", label="The effect of the assigned treatment in a randomized trial",
        slot="model", scope="model", package="TRIALS", run_order=1.5,
        scope_note=("It is fitted to the analysis set's outcomes. A missing baseline value takes "
                    "the covariate's mean over the analysis set (never the arm or the outcome)."),
        needs=("a declared randomized trial (parallel or cluster-randomized)",
               "the arm, the control arm and the outcome (a number, or yes/no)",
               "the randomization factors and the baseline covariates named before the data "
               "were seen (the baseline measure of the outcome among them)",
               "a cluster-randomized trial: the randomized cluster of each person"),
        question="How is the effect of the assigned treatment estimated?",
        options=(
            option("ancova", "Regression on the arm, the randomization factors and the baseline "
                             "covariates (analysis of covariance)",
                   "The standard analysis of a numeric outcome with a baseline measure (Vickers & "
                   "Altman 2001); the randomization factors adjusted for (Kahan & Morris 2012)",
                   "Sound for a numeric outcome in a parallel trial: adjustment for precision "
                   "only, the covariates named before the data were seen", "recommended", 0),
            option("standardized_risk", "Risk difference and ratio standardized from a logistic "
                                        "regression on the same terms",
                   "Ye, Shao, Yi & Zhao 2023 (their variance); R's beeca",
                   "Sound for a yes/no outcome in a parallel trial: the marginal risk difference "
                   "and ratio, valid when the logistic model is wrong", "recommended", 1),
            option("mixed_kr", "Linear mixed model with a random intercept per cluster, "
                               "Kenward–Roger intervals",
                   "Kenward & Roger 1997; Leyrat et al. 2018 (cluster trials with few clusters); "
                   "pbkrtest in R (Halekoh & Højsgaard 2014)",
                   "Sound for a numeric outcome in a cluster trial, also with few clusters; "
                   "refused for a yes/no outcome", "recommended", 2),
            option("gee_corrected", "Estimating equations with a small-sample corrected sandwich "
                                    "(Kauermann–Carroll, Fay–Graubard or Mancl–DeRouen)",
                   "Li & Redden 2015 (Kauermann–Carroll below a 0.6 coefficient of variation of "
                   "cluster sizes, Fay–Graubard above); Mancl & DeRouen 2001; Fay & Graubard "
                   "2001; Kauermann & Carroll 2001",
                   "Sound for a yes/no or numeric outcome in a cluster trial; intervals on the "
                   "clusters less the cluster-level terms", "recommended", 3),
        ),
        storyboard=("build the model from the arm, the randomization factors and the named "
                    "baseline covariates", "fill a missing baseline value by its mean, with an "
                    "indicator", "fit the outcome model (with the cluster in a cluster trial)",
                    "read the arm's effect against the control, with its interval",
                    "word it: causal only for intention to treat"),
        relations=(
            Relation("implies", "precision_adjustment_only",
                     "the adjustment is the randomization factors and the baseline covariates "
                     "named before the data were seen, for precision only; nothing is searched "
                     "for or selected", purposes=est_only, condition="always",
                     enforced_by=f"{here}:estimate", id="no_confounder_selection"),
            Relation("implies", "randomization_factors_adjusted",
                     "the factors the randomization was stratified or minimized on are adjusted "
                     "for (Kahan & Morris 2012)", purposes=est_only,
                     condition="randomization factors declared", enforced_by=f"{here}:estimate",
                     id="strata_adjusted"),
            Relation("implies", "baseline_outcome_adjusted",
                     "the baseline measure of the outcome is a covariate (analysis of covariance; "
                     "Vickers & Altman 2001)", purposes=est_only, when=("ancova", "mixed_kr"),
                     condition="a baseline measure named", enforced_by=f"{here}:estimate",
                     id="ancova_baseline"),
            Relation("conflicts", "post_hoc_covariates",
                     "covariates chosen after the data were seen are refused from the primary "
                     "analysis", purposes=est_only, rung="refused",
                     exits=("leave them out", "a labeled secondary analysis"),
                     condition="covariates not named in advance", enforced_by=f"{here}:estimate",
                     id="post_hoc_refused"),
            Relation("conflicts", "post_randomization_covariate",
                     "adherence or the treatment received is refused as a covariate: it is known "
                     "only after randomization", purposes=est_only, rung="refused",
                     exits=("leave it out of the adjustment",),
                     condition="an adjustment term that is adherence or the treatment received",
                     enforced_by=f"{here}:analysis_sets", id="post_randomization"),
            Relation("implies", "missing_baselines_filled",
                     "a missing baseline value takes its mean over the analysis set with a "
                     "missing-value indicator (White & Thompson 2005)", purposes=est_only,
                     condition="a baseline covariate with missing values",
                     enforced_by=f"{here}:estimate", id="baseline_mean_imputation"),
            Relation("conflicts", "baseline_balance_tests",
                     "baseline differences between the arms are described, never tested "
                     "(Moher et al. 2010, item 15; Senn 1994)", purposes=est_only, rung="refused",
                     exits=("describe each arm",), condition="tests of baseline balance asked for",
                     enforced_by=f"{here}:baseline_table", id="no_balance_tests"),
            Relation("implies", "causal_wording",
                     "causal wording only for intention to treat under a declared randomization, "
                     "as the effect of being assigned; per protocol is worded as a difference "
                     "among those who followed the protocol", purposes=est_only,
                     condition="always", enforced_by=f"{here}:trial_sentence",
                     id="causal_itt_only"),
            Relation("conflicts", "not_randomized",
                     "refused unless the design is a declared randomized trial", purposes=est_only,
                     rung="refused", exits=("declare how people were randomized",),
                     condition="an observational or undeclared design",
                     enforced_by=f"{here}:estimate", id="needs_randomization"),
            Relation("implies", "icc_reported",
                     "the intraclass correlation is reported (Campbell et al. 2012)",
                     purposes=est_only, when=("mixed_kr", "gee_corrected"),
                     condition="a cluster-randomized trial", enforced_by=f"{here}:estimate",
                     id="icc"),
            Relation("conflicts", "arm_varies_within_cluster",
                     "refused when people in one cluster are in different arms",
                     purposes=est_only, rung="refused", when=("mixed_kr", "gee_corrected"),
                     exits=("declare a parallel trial", "name the randomized cluster"),
                     condition="a cluster with more than one arm",
                     enforced_by=f"{here}:estimate", id="cluster_is_randomized"),
            Relation("conflicts", "too_few_clusters",
                     "refused with fewer than two clusters per arm or no clusters left over the "
                     "cluster-level terms", purposes=est_only, rung="refused",
                     when=("mixed_kr", "gee_corrected"),
                     exits=("choose the column of the randomized cluster",
                            "leave the cluster-level covariates out (when they are what uses the "
                            "clusters up)", "describe the outcome in each arm"),
                     condition="one cluster in an arm, or clusters ≤ cluster-level terms",
                     enforced_by=f"{here}:estimate",
                     id="cluster_floor"),
            Relation("implies", "few_clusters_noticed",
                     "fewer than 10 clusters are named among the concerns (the fewest Li & Redden "
                     "2015 studied), beside the missing-outcome notice, neither displacing the "
                     "other",
                     purposes=est_only, when=("mixed_kr", "gee_corrected"),
                     condition="fewer than 10 clusters", enforced_by=f"{here}:estimate",
                     id="few_clusters"),
            Relation("implies", "correction_by_cluster_sizes",
                     "Kauermann–Carroll when the cluster sizes' coefficient of variation is below "
                     "0.6, Fay–Graubard otherwise (Li & Redden 2015)", purposes=est_only,
                     when=("gee_corrected",), condition="the correction left to the rule",
                     enforced_by=f"{here}:estimate", id="kc_or_fg"),
            Relation("conflicts", "kenward_roger_binary",
                     "the mixed model with Kenward–Roger is refused for a yes/no outcome",
                     purposes=est_only, rung="refused", when=("mixed_kr",),
                     exits=("estimating equations with a corrected sandwich",),
                     condition="a yes/no outcome", enforced_by=f"{here}:estimate",
                     id="kr_numeric_only"),
            Relation("conflicts", "separation",
                     "refused when the logistic model's risks run to 0% or 100%",
                     purposes=est_only, rung="refused",
                     when=("standardized_risk", "gee_corrected"),
                     exits=("adjust for fewer covariates", "describe the events in each arm"),
                     condition="separation", enforced_by=f"{here}:estimate", id="separation"),
            Relation("precedes", "trial_missing_outcomes",
                     "the primary analysis is fitted before its sensitivity to missing outcomes",
                     purposes=est_only, condition="always", enforced_by=f"{here}:tipping_point",
                     id="primary_first"),
            refuse_survey("effect_no_survey", "estimate"),
            refuse_prediction("effect_not_predicted", "eligible"),
        ),
        sources=(SOURCES["vickers2001"], SOURCES["kahan2012"], SOURCES["moher2010"],
                 SOURCES["senn1994"], SOURCES["white2005"], SOURCES["ye2023"],
                 SOURCES["kenward1997"], SOURCES["halekoh2014"], SOURCES["leyrat2018"],
                 SOURCES["liang1986"], SOURCES["kauermann2001"], SOURCES["fay2001"],
                 SOURCES["mancl2001"], SOURCES["li2015"], SOURCES["campbell2012"]),
        sentence=f"{here}:trial_sentence"))

    register_contract(MethodContract(
        key="trial_missing_outcomes",
        label="How sensitive a trial's result is to its missing outcomes (the tipping point)",
        slot="evaluation", scope="model", package="TRIALS", run_order=9.0,
        scope_note=("Refits the outcome model on every randomized person's completed outcomes: "
                    "the imputation model is fitted to the observed outcomes, so it reads the "
                    "outcome."),
        needs=("a parallel trial analyzed by intention to treat",
               "at least one randomized person with a missing outcome"),
        question="How sensitive is the result to the outcomes that are missing?",
        options=(
            option("delta_tipping_point",
                   "Impute the missing outcomes, shift them by δ, and find the shift that changes "
                   "the conclusion (the tipping point)",
                   "Ratitch, O'Kelly & Tosiello 2013; Cro, Morris, Kenward & Carpenter 2020",
                   "Sound: says how different the missing would have to be, in the outcome's own "
                   "units, before the conclusion changes", "recommended", 0),
            option("missing_at_random_only",
                   "Only the primary analysis: the missing outcomes like the observed ones with "
                   "the same arm and covariates",
                   "The primary analysis's assumption (White, Horton, Carpenter & Pocock 2011)",
                   "Untested: the assumption cannot be checked from the data, so it is stated "
                   "without a sensitivity analysis", "rank_lower", 1),
        ),
        storyboard=("fit the imputation model to the observed outcomes",
                    "draw m completed data sets under missing at random",
                    "shift one arm's imputed outcomes by δ", "analyze each and pool by Rubin's "
                    "rules", "search δ until the 95% interval's conclusion changes"),
        relations=(
            Relation("implies", "every_randomized_person",
                     "the sensitivity analysis counts every randomized person (White, Horton, "
                     "Carpenter & Pocock 2011)", purposes=est_only, when=("delta_tipping_point",),
                     condition="always", enforced_by=f"{here}:tipping_point", id="all_randomized"),
            Relation("implies", "imputations_by_rule",
                     "at least as many imputations as the percentage of people with a missing "
                     "outcome (White, Royston & Wood 2011)", purposes=est_only,
                     when=("delta_tipping_point",), condition="always",
                     enforced_by=f"{here}:tipping_point", id="m_rule"),
            Relation("implies", "rubin_pooling",
                     "each completed set is analyzed as the primary analysis and pooled by "
                     "Rubin's rules (Rubin 1987) with Barnard & Rubin 1999's degrees of freedom",
                     purposes=est_only, when=("delta_tipping_point",), condition="always",
                     enforced_by=f"{here}:tipping_point", id="pooled"),
            Relation("conflicts", "cluster_trial_imputation",
                     "refused for a cluster trial: a multilevel imputation model is not built",
                     purposes=est_only, rung="refused", when=("delta_tipping_point",),
                     exits=("the primary analysis, its assumption stated",),
                     condition="a cluster-randomized trial",
                     enforced_by=f"{here}:tipping_point", id="no_cluster_mi"),
            Relation("conflicts", "per_protocol_imputation",
                     "refused for the per-protocol set: the sensitivity analysis counts every "
                     "randomized person", purposes=est_only, rung="refused",
                     when=("delta_tipping_point",), exits=("intention to treat",),
                     condition="the per-protocol set", enforced_by=f"{here}:tipping_point",
                     id="itt_only"),
            refuse_survey("tipping_no_survey", "tipping_point"),
            refuse_prediction("tipping_not_predicted", "eligible"),
        ),
        sources=(SOURCES["ratitch2013"], SOURCES["cro2020"], SOURCES["white2011"],
                 SOURCES["white2011mi"], SOURCES["rubin1987"], SOURCES["barnard1999"]),
        sentence=f"{here}:trial_sentence"))


_register_contracts()

__all__ = ["AnalysisSets", "ArmEstimate", "ArmFlow", "BaselineRow", "BaselineTable", "Contrast",
           "ConsortFlow", "Eligibility", "GEEResult", "KenwardRoger", "TippingPoint", "TippingRow",
           "Tipping", "TrialEffect", "TrialRefused", "TrialSpec", "analysis_sets", "baseline_table",
           "cluster_level_columns", "consort_flow", "delta_analysis", "eligible", "estimate",
           "fit_gee", "gee_covariance", "in_analysis_set", "kenward_roger", "kr_df",
           "logistic_draw", "normal_draw", "size_cv", "tipping_point", "trial_sentence",
           "ye_variance"]

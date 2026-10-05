"""The declared purpose routes the questions (AUDIT_REPORT §5 WP17; RO-03, RO-04, RO-08).

Four questions the Router (``turbotab/core/interview.py``) asks because of what the purpose and the
table say, each with the refusals that keep its answer honest. This module holds their gates, the
roles the adjustment answers derive, and what the server withholds while one is unanswered.

**The follow-up** (RO-03). A yes/no outcome beside a column that reads as follow-up time is asked:
*did everyone have the same follow-up, or could some leave, or the study end, before the event?*
PROBAST's explanation, item 4.6: "For prognostic models to predict long-term outcomes in which
censoring occurs, a time-to-event analysis, such as a Cox regression, should be used… Use of
logistic regression models that simply exclude censored participants with incomplete follow-up is
inappropriate." Follow-up that varies is a time to event (``set_task``, then ``set_follow_up``: the
Cox family). The same follow-up for everyone is ``set_censoring``; said over a follow-up column
that varies, it is refused until attested (block and record), and the fit carries the attestation.

**The grouping above the person** (RO-08). A column read as a site, centre, household or batch is
asked about after the roles: does it group the participants? Under inference the recommended answer
gives each group its own intercept (fixed effects) and clusters the intervals by it (CR2 with
Bell–McCaffrey df, ``models/inference.py``); clustering alone is offered with its concern; "no
grouping" over a column that reads as one is recorded only with an attestation. Under prediction
the answer names the cluster the split question's internal–external validation folds by.

**The exposure and its effect** (MODELING_SEQUENCE §1 step 2; RO-04). Under inference: which column
is the exposure, a total or a direct effect, for an energy-bearing exposure a substitution (total
energy held fixed) or an addition (its calories added), and the effect measure. Only the measures
the engine fits are offered: a difference in means, or the conditional odds, hazard, cumulative odds
or relative-risk ratio of the task's family; the marginal risk difference and risk ratio
(g-computation) are named and refused, never answered with an odds ratio.

**The adjustment set** (MODELING_SEQUENCE §1 step 3). Each covariate is asked the modified
disjunctive cause criterion as questions, and its role is derived from the answers. VanderWeele
(2019, *Eur J Epidemiol* 34:211–219, "Principles of confounder selection"): "control for each
covariate that is a cause of the exposure, or of the outcome, or of both; exclude from this set any
variable known to be an instrumental variable; and include as a covariate any proxy for an
unmeasured variable that is a common cause of both the exposure and the outcome", and "Statistical
analyses cannot in general distinguish between confounders, which ought to be controlled for in the
estimation of the total effect, versus mediators, which ought not be controlled for." So:

* a pre-exposure cause (or possible cause) of the exposure or the outcome is adjusted, as is a
  proxy for an unmeasured common cause; a cause of neither is left out;
* a known instrument is left out;
* a covariate the exposure could have changed (or measured after it) that causes the outcome is a
  mediator, and one that does not is a collider or another consequence of the exposure: both leave a
  total-effect set. Kept in it, a mediator is blocked and recorded (``keep`` with
  ``acknowledged``); "further adjusted for" it is the labeled secondary model;
* a covariate whose timing is unknown (a cross-sectional BMI) is a declared with-and-without pair:
  the primary without it, the secondary with it;
* other dietary components default to confounders through the diet's common causes, never to
  "not relevant" (the guess the card leads with, NUTRITION_PACK §08's Model 4).

Asking this of thirty covariates stays light (BLUEPRINT §14.2): covariates the pack guesses alike
form one group, each group is confirmed with one tap (one ``set_adjustment`` naming its columns),
and only the pack's guesses lead; a covariate the pack says nothing about is asked without one.

**No estimate before the plan.** Under inference no coefficient, curve or contrast is served while
the exposure, the effect or a covariate's answers are missing (``served_gate``, read by the server
like the seal's withholding of held-out scores); the caption is worded from the estimand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

VANDERWEELE = "VanderWeele 2019, Eur J Epidemiol 34:211–219"
PROBAST = "PROBAST explanation and elaboration (Moons et al. 2019), item 4.6"

Gate = tuple[str, str | None] | None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tick(value: Any) -> str:
    return f"`{value}`"


def _listing(items: Sequence[Any], limit: int = 4) -> str:
    from turbotab.core.voice import listing

    return listing(list(items), limit=limit)


# ── the follow-up (audit RO-03) ──────────────────────────────────────────────

# Words that say a column is how long a row was followed: the recognizer's time words, and the
# survival vocabulary (``futime``, ``tte``, person-years). A date (``visit_date``) says when, not how
# long, so a column whose values are dates or that names a calendar date is not one.
FOLLOW_UP_WORDS = {"followup", "fu", "futime", "fup", "tte", "survival", "surv", "persontime",
                   "pyears", "py", "censor", "censored", "censoring", "exit", "observed"}
_DURATION_WORDS = {"years", "yrs", "year", "months", "month", "days", "day", "weeks", "week",
                   "time", "duration"}
_NOT_A_DURATION = {"date", "datetime", "timestamp", "cycle", "visit", "wave", "round", "recall",
                   "session", "calendar", "baseline", "hour", "hours", "dob", "birth"}
NUMERIC = ("numeric", "integer")


def reads_as_follow_up(name: Any) -> bool:
    """The name says how long the row was observed: ``followup_years``, ``fu_days``,
    ``time_to_event``, ``survival_months``, ``futime``; not a date, a cycle or a visit index."""
    from turbotab.core.recognizers import is_rate, reads_as_time, tokens

    words = set(tokens(name))
    if not words or words & _NOT_A_DURATION or is_rate(name):
        return False
    if words & FOLLOW_UP_WORDS:
        return True
    return bool(words & _DURATION_WORDS) and reads_as_time(name) and (
        bool(words & {"to", "event", "since", "elapsed", "follow", "up", "observation"})
        or words <= _DURATION_WORDS)


def follow_up_candidates(frame: Any, columns: Mapping[str, Mapping[str, Any]],
                         target: str | None) -> list[dict[str, Any]]:
    """Numeric columns that read as a follow-up time, with whether their values vary: a follow-up
    that runs from 0.3 to 14.9 years ended at different times for different people."""
    import pandas as pd

    out: list[dict[str, Any]] = []
    for name, info in columns.items():
        if name == target or str(info.get("dtype")) not in NUMERIC or not reads_as_follow_up(name):
            continue
        if frame is None or name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        if values.empty or float(values.min()) < 0:
            continue
        low, high = float(values.min()), float(values.max())
        varies = bool(values.nunique() > 1 and high > 0 and (high - low) > 0.05 * high)
        out.append({"column": name, "min": round(low, 4), "max": round(high, 4), "varies": varies})
    return out[:5]


def effective_task(state: Any, target_info: Any = None) -> str | None:
    """The task as answered, else as the fresh ``target_info`` detected it for this outcome."""
    task = _get(state, "task")
    if task is not None:
        return str(task)
    if target_info is not None and _get(target_info, "column") == _get(state, "target"):
        found = _get(target_info, "task")
        return str(found) if found else None
    return None


def follow_up_gate(state: Any, target_info: Any = None) -> Gate:
    """Whether the follow-up question is asked: a yes/no outcome beside a column that reads as a
    follow-up time, or a time to event, whose follow-up must be named."""
    if _get(state, "target") is None:
        return None
    task = effective_task(state, target_info)
    if task is None:
        return None
    if task not in ("binary", "time_to_event"):
        return ("not_applicable", f"The outcome is read as {task.replace('_', ' ')}, so no event "
                                  f"is followed over time.")
    if task == "time_to_event":
        return None
    if target_info is None or _get(target_info, "column") != _get(state, "target"):
        return None
    candidates = _get(target_info, "follow_up")
    if candidates is None:
        return None
    if not candidates:
        return ("skipped", "no numeric column reads as a follow-up time, so the yes/no outcome is "
                           "read as counted over one period for everyone.")
    return None


def follow_up_answer(state: Any, target_info: Any = None) -> Any:
    """What answers the follow-up question now: for a time to event, its follow-up; for a yes/no
    outcome, the recorded "same for everyone"."""
    if effective_task(state, target_info) == "time_to_event":
        return _get(state, "follow_up")
    return _get(state, "censoring")


# ── the grouping above the person (audit RO-08) ──────────────────────────────


def cluster_candidates(state: Any, roles: Any = None) -> list[str]:
    """Columns that read as a group of participants (a site, a centre, a household, a batch): the
    cluster role, recorded or proposed by the roles stage (WP13), never the outcome or the unit the
    grain answer names."""
    target = _get(state, "target")
    grain = _get(state, "grain")
    unit = _get(grain, "id_column") if grain is not None else None
    recorded = _get(state, "roles") or {}
    found = [c for c, r in recorded.items() if r == "cluster"]
    data = getattr(roles, "data", roles)
    for entry in (_get(data, "columns") or []) if data is not None else []:
        column, proposed = _get(entry, "column"), _get(entry, "proposed")
        if proposed == "cluster" and column and recorded.get(column, "cluster") == "cluster":
            found.append(str(column))
    return [c for c in dict.fromkeys(found) if c not in (target, unit)]


def clusters_gate(state: Any, roles: Any = None) -> Gate:
    if _get(state, "roles") is None:
        return None
    if not cluster_candidates(state, roles):
        return ("skipped", "no column reads as a site, centre, household or batch that groups the "
                           "participants.")
    return None


def cluster_answer(state: Any) -> str | None:
    """The column the user said groups the participants, whatever the purpose."""
    spec = _get(state, "clusters")
    return _get(spec, "column") if spec is not None else None


def fixed_effects_column(state: Any) -> str | None:
    """Under inference, the grouping the model gives its own intercept per level."""
    spec = _get(state, "clusters")
    if _get(state, "purpose") != "inference" or spec is None:
        return None
    return _get(spec, "column") if _get(spec, "adjust") == "fixed_effects" else None


# ── the exposure and its effect (MODELING_SEQUENCE §1 step 2) ────────────────

# The measure each task's family estimates, conditional on the adjustment set (ruling 9: ORs and
# HRs are labeled conditional and non-collapsible).
MEASURE_OF_TASK = {"regression": "mean_difference", "binary": "odds_ratio",
                   "time_to_event": "hazard_ratio", "ordinal": "cumulative_odds_ratio",
                   "multiclass": "relative_risk_ratio"}
MEASURE_WORDS = {
    "mean_difference": "difference in the mean outcome",
    "odds_ratio": "conditional odds ratio",
    "hazard_ratio": "conditional hazard ratio",
    "cumulative_odds_ratio": "conditional cumulative odds ratio",
    "relative_risk_ratio": "conditional relative-risk ratio against the reference class",
    "risk_difference": "marginal risk difference",
    "risk_ratio": "marginal risk ratio",
    "exposure_mean_difference": ("difference in each exposure's mean between the event and the "
                                 "other level (the limma design)"),
}
NON_COLLAPSIBLE = {"odds_ratio", "hazard_ratio", "cumulative_odds_ratio", "relative_risk_ratio"}
NOT_FITTED = {
    "risk_difference": "a marginal risk difference needs standardization over the covariates "
                       "(g-computation), which TurboTab does not fit yet",
    "risk_ratio": "a marginal risk ratio needs standardization over the covariates "
                  "(g-computation), which TurboTab does not fit yet",
}


# What the feature-wise family (``models/featurewise.py``) estimates for an exposure family, each
# exposure in turn adjusted for the covariates: a numeric outcome's difference in its mean per unit
# of the exposure; a yes/no outcome's difference in the exposure's mean between the event and the
# other level (the limma design). It fits no other task, so no family is offered for one.
FAMILY_MEASURE_OF_TASK = {"regression": "mean_difference", "binary": "exposure_mean_difference"}


def fitted_measures(task: str | None, family: bool = False) -> list[str]:
    """The measures the engine fits for ``task``: one exposure's, the task's model family's; an
    exposure family's, the feature-wise family's."""
    fitted = (FAMILY_MEASURE_OF_TASK if family else MEASURE_OF_TASK).get(str(task))
    return [fitted] if fitted else []


def measures_offered(task: str | None, family: bool = False) -> list[dict[str, Any]]:
    """The effect measures the estimand question offers for ``task``: the ones the engine fits,
    then (for a yes/no outcome) the marginal ones it does not, each with why."""
    out = []
    for fitted in fitted_measures(task, family):
        out.append({"measure": fitted, "label": MEASURE_WORDS[fitted], "fitted": True,
                    "reason": ("non-collapsible: adding a covariate that predicts the outcome "
                               "changes it even without confounding" if fitted in NON_COLLAPSIBLE
                               else "the feature-wise family's: each exposure modeled on the "
                                    "outcome and the covariates" if fitted == "exposure_mean_difference"
                               else "collapsible: the conditional and the marginal difference "
                                    "agree in a linear model")})
    if task == "binary" and not family:
        out += [{"measure": m, "label": MEASURE_WORDS[m], "fitted": False, "reason": why}
                for m, why in NOT_FITTED.items()]
    return out


def _base_left_out(state: Any) -> list[str]:
    """``decisions.left_out`` before the adjustment answers: the missing-values answer's columns and
    the repairs' unusable ones."""
    spec = _get(state, "missing")
    out = list(_get(spec, "drop_columns") or []) if spec is not None else []
    from turbotab.core.repairs import unusable_columns

    return out + [c for c in unusable_columns(state) if c not in out]


PREDICTOR_ROLES = ("exposure", "covariate", "energy")


def predictor_roles(state: Any) -> dict[str, str]:
    """The settled predictors and their roles, before the adjustment answers (BLUEPRINT §14.1: a
    role that rode along unconfirmed is in no fit)."""
    from turbotab.core.readings import settled_roles

    target = _get(state, "target")
    gone = set(_base_left_out(state))
    return {c: r for c, r in settled_roles(state).items()
            if r in PREDICTOR_ROLES and c != target and c not in gone}


def exposure_candidates(state: Any) -> list[str]:
    """The columns the estimand question offers as the exposure: exposures first, then the other
    predictors (a covariate may be the exposure of interest)."""
    roles = predictor_roles(state)
    return ([c for c, r in roles.items() if r == "exposure"]
            + [c for c, r in roles.items() if r == "covariate"])


def _purpose_gate(state: Any) -> Gate:
    purpose = _get(state, "purpose")
    if purpose == "prediction":
        return ("not_applicable", "Under prediction no coefficient is read as an effect, so no "
                                  "exposure, effect or adjustment set is declared.")
    return None


def estimand_gate(state: Any) -> Gate:
    found = _purpose_gate(state)
    if found is not None or _get(state, "purpose") is None or _get(state, "roles") is None:
        return found
    if not exposure_candidates(state):
        return ("not_applicable", "No column is in the model as an exposure or covariate, so there "
                                  "is no exposure to declare.")
    return None


def family_exposures(state: Any) -> list[str]:
    """The exposure family: every settled exposure-role predictor."""
    return [c for c, r in predictor_roles(state).items() if r == "exposure"]


def exposure_key(spec: Any) -> str:
    """Whose effect the adjustment answers are for: the exposure, or the family's sentinel."""
    from turbotab.core.decisions import EXPOSURE_FAMILY

    return EXPOSURE_FAMILY if _get(spec, "family") else str(_get(spec, "exposure"))


def exposures_of(state: Any, spec: Any) -> list[str]:
    """The columns whose effects the estimand reports: the exposure, or each of the family."""
    return family_exposures(state) if _get(spec, "family") else [str(_get(spec, "exposure"))]


def current_estimand(state: Any) -> Any:
    """The recorded estimand while its exposure is still a settled predictor (a family: while two
    or more exposures are); None otherwise (a change of roles that takes the exposure out of the
    model re-asks it, never keeps it)."""
    spec = _get(state, "estimand")
    if spec is None:
        return None
    if _get(spec, "family"):
        return spec if len(family_exposures(state)) >= 2 else None
    return spec if _get(spec, "exposure") in predictor_roles(state) else None


def energy_bearing(column: str) -> bool:
    from turbotab.core.stages.rows import energy_bearing as bearing

    try:
        return bool(bearing(column))
    except Exception:  # noqa: BLE001 - a name nothing reads carries no energy we know of
        return False


def energy_contrast_applies(state: Any, exposure: str) -> bool:
    """Substitution or addition is asked of an energy-bearing exposure with total energy among the
    settled predictors (the dietary lens's energy question then decides the energy model)."""
    roles = predictor_roles(state)
    return energy_bearing(exposure) and any(r == "energy" for r in roles.values())


def family_contrast_applies(state: Any) -> bool:
    """An exposure family asks substitution or addition when any of its exposures carries energy."""
    return any(energy_contrast_applies(state, c) for c in family_exposures(state))


# ── the adjustment set (MODELING_SEQUENCE §1 step 3) ─────────────────────────

ROLE_WORDS = {
    "confounder": "confounder",
    "exposure_cause": "cause of the exposure",
    "precision": "cause of the outcome only (precision)",
    "proxy": "proxy for an unmeasured common cause",
    "mediator": "mediator",
    "collider": "consequence of the exposure (a possible collider)",
    "instrument": "instrument",
    "timing_unknown": "timing unknown",
    "not_a_cause": "cause of neither",
}
# The same, in a sentence: one covariate, then several.
ROLE_SINGULAR = {
    "confounder": "a confounder", "exposure_cause": "a cause of the exposure",
    "precision": "a cause of the outcome only", "proxy": "a proxy for an unmeasured common cause",
    "mediator": "a mediator", "collider": "a consequence of the exposure (a possible collider)",
    "instrument": "an instrument", "timing_unknown": "of unknown timing",
    "not_a_cause": "a cause of neither",
}
ROLE_PLURAL = {
    "confounder": "confounders", "exposure_cause": "causes of the exposure",
    "precision": "causes of the outcome only", "proxy": "proxies for an unmeasured common cause",
    "mediator": "mediators", "collider": "consequences of the exposure (possible colliders)",
    "instrument": "instruments", "timing_unknown": "of unknown timing",
    "not_a_cause": "causes of neither",
}


@dataclass(frozen=True)
class Derived:
    """A covariate's role, derived from its answers, and where it goes."""

    role: str
    adjusted: bool  # in the primary model
    secondary: bool  # in the "further adjusted for" model beside it
    why: str


def derive(answers: Any, effect: str = "total") -> Derived:
    """The role the modified disjunctive cause criterion gives a covariate (VanderWeele 2019).

    The criterion's own rule decides a pre-exposure covariate: adjusted when it is (or may be) a
    cause of the exposure or of the outcome, or a proxy for an unmeasured common cause; left out
    when it causes neither, or is a known instrument. A covariate the exposure could have changed
    leaves a total-effect set (a mediator when it causes the outcome; otherwise a consequence of the
    exposure, a collider when the outcome changes it too); a direct effect holds the mediators
    fixed. Unknown timing is the declared with-and-without pair."""
    a = answers
    kept = bool(_get(a, "keep") and _get(a, "acknowledged"))
    further = bool(_get(a, "further"))
    after = _get(a, "after_exposure")
    if _get(a, "instrument"):
        return Derived("instrument", False, further,
                       "a known instrument: it moves the outcome only through the exposure, and "
                       "adjusting for it amplifies any confounding left")
    if after == "yes":
        if _get(a, "causes_outcome") == "yes":
            if effect == "direct":
                return Derived("mediator", True, False,
                               "on the path from the exposure to the outcome; a direct effect "
                               "holds it fixed")
            return Derived("mediator", kept, further and not kept,
                           "on the path from the exposure to the outcome: adjusting for it removes "
                           "part of the total effect")
        return Derived("collider", kept, further and not kept,
                       "changed by the exposure without causing the outcome: adjusting for it can "
                       "open a path that is not causal")
    if after == "unknown":
        return Derived("timing_unknown", kept, not kept,
                       "the exposure may have changed it, so the estimate is declared without it "
                       "and, beside, with it")
    if _get(a, "proxy"):
        return Derived("proxy", True, False, "a proxy for an unmeasured cause of both")
    ce, co = _get(a, "causes_exposure"), _get(a, "causes_outcome")
    if ce == "no" and co == "no":
        return Derived("not_a_cause", False, further,
                       "a cause of neither the exposure nor the outcome: the criterion leaves it out")
    if co == "no":
        return Derived("exposure_cause", True, False,
                       "a cause of the exposure: the criterion adjusts for it")
    if ce == "no":
        return Derived("precision", True, False,
                       "a cause of the outcome only: adjusting for it sharpens the estimate")
    possible = "unknown" in (ce, co)
    return Derived("confounder", True, False,
                   ("a possible cause of the exposure and of the outcome: the criterion adjusts for "
                    "it" if possible else "a cause of the exposure and of the outcome"))


def _covariates_among(state: Any, roles: Mapping[str, str]) -> list[str]:
    spec = _get(state, "estimand")
    family = bool(_get(spec, "family")) if spec is not None else False
    exposure = _get(spec, "exposure") if spec is not None else None
    dietary = "dietary" in (_get(state, "lens") or [])
    fe = fixed_effects_column(state)
    # An exposure family: each exposure is reported in turn, so none is another's covariate.
    return [c for c, r in roles.items()
            if c != exposure and c != fe and not (dietary and r == "energy")
            and not (family and r == "exposure")]


def covariates(state: Any) -> list[str]:
    """The covariates in the model whose answers derive their place: every settled predictor but
    the exposure, total energy (the energy question decides it under the dietary lens) and the
    grouping the cluster answer gives fixed effects."""
    return _covariates_among(state, predictor_roles(state))


def asked_covariates(state: Any) -> list[str]:
    """The covariates the adjustment question asks about: as :func:`covariates`, and also each
    recorded predictor role still riding along unconfirmed (BLUEPRINT §14.1). Its answers are
    asked with the rest, so confirming its role later finds them given instead of reopening the
    question; until it is confirmed it is in no model whatever they say."""
    target = _get(state, "target")
    gone = set(_base_left_out(state))
    recorded = {c: r for c, r in (_get(state, "roles") or {}).items()
                if r in PREDICTOR_ROLES and c != target and c not in gone}
    settled = predictor_roles(state)
    return _covariates_among(state, {**recorded, **settled})


def current_answers(state: Any) -> dict[str, Any]:
    """Each covariate's answers given for the current exposure (an answer given for another exposure
    is re-asked, never kept: MODELING_SEQUENCE §2, "invalidates")."""
    spec = _get(state, "estimand")
    if spec is None:
        return {}
    exposure = exposure_key(spec)
    return {c: a for c, a in (_get(state, "adjustment") or {}).items()
            if _get(a, "exposure") == exposure}


def unanswered(state: Any) -> list[str]:
    answers = current_answers(state)
    return [c for c in asked_covariates(state) if c not in answers]


def derived_roles(state: Any) -> dict[str, Derived]:
    spec = current_estimand(state)
    if spec is None:
        return {}
    answers = current_answers(state)
    effect = str(_get(spec, "effect") or "total")
    return {c: derive(answers[c], effect) for c in covariates(state) if c in answers}


def adjustment_left_out(state: Any) -> list[str]:
    """Under inference, the answered covariates the primary model leaves out (``decisions.left_out``
    reads it). Nothing is left out under prediction or before the exposure is declared."""
    if _get(state, "purpose") != "inference":
        return []
    return [c for c, d in derived_roles(state).items() if not d.adjusted]


def secondary_columns(state: Any) -> list[str]:
    """The columns of the declared "further adjusted for" model: each covariate of unknown timing,
    and each the user asked to be further adjusted for."""
    if _get(state, "purpose") != "inference":
        return []
    return [c for c, d in derived_roles(state).items() if d.secondary and not d.adjusted]


def adjustment_gate(state: Any) -> Gate:
    found = estimand_gate(state)  # no exposure to declare: no adjustment set to answer either
    if found is not None or _get(state, "purpose") is None:
        return found
    spec = current_estimand(state)
    if spec is None:
        return None
    if not asked_covariates(state):
        besides = ("the exposures" if _get(spec, "family")
                   else _tick(_get(spec, "exposure")))
        return ("not_applicable", f"No column besides {besides} is in the model, so there is no "
                                  f"adjustment set to answer.")
    return None


def adjustment_answer(state: Any) -> Any:
    """The adjustment set, once every covariate has answers for the current exposure; else None."""
    if current_estimand(state) is None or unanswered(state):
        return None
    return current_answers(state)


# ── the pack's guesses (BLUEPRINT §14.2: lead with a guess only where the pack has one) ─────────

_DEMOGRAPHIC = {"age", "sex", "gender", "race", "ethnicity", "ethnic", "education", "educ",
                "income", "poverty", "pir", "smoking", "smoker", "smoke", "cigarettes",
                "occupation", "marital", "ses"}
_NHANES_DEMOGRAPHIC = {"RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
                       "DMDMARTL", "SMQ020", "SMQ040"}
_BODY = {"bmi", "weight", "height", "waist", "hip", "whr", "adiposity", "bodyfat", "ffm", "lean",
         "skinfold", "dxa", "bia"}
_NHANES_BODY = {"BMXBMI", "BMXWT", "BMXHT", "BMXWAIST", "BMXHIP", "DXDTOFAT", "DXDTOPF"}
_PACK08 = "NUTRITION_PACK §08 (the nested model table: Model 1 age, sex, energy; 2 demographics " \
          "and lifestyle; 3 body composition; 4 mutually adjusted nutrients)"

GUESSES: dict[str, dict[str, Any]] = {
    "demographic": {
        "label": "Demographics and lifestyle",
        "answers": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"},
        "reason": "set before the diet was measured, and the field's Models 1 and 2 adjust for "
                  "them as confounders",
    },
    "body": {
        "label": "Body size and composition",
        "answers": {"causes_exposure": "unknown", "causes_outcome": "yes",
                    "after_exposure": "unknown"},
        "reason": "the diet may have changed them, so the field's Model 3 adds them beside the "
                  "primary; adjusting for a mediator without saying so is the pack's anti-pattern",
    },
    "dietary": {
        "label": "Other dietary components",
        "answers": {"causes_exposure": "unknown", "causes_outcome": "unknown",
                    "after_exposure": "no"},
        "reason": "they share the diet's common causes with the exposure, so they default to "
                  "confounders, never to not relevant (the field's Model 4)",
    },
}


def guess_of(column: str) -> str | None:
    """Which of the pack's guesses a covariate takes, by its name (a proposal the user confirms,
    never a settled reading: BLUEPRINT §14)."""
    from turbotab.core.recognizers import is_nutrient, tokens

    name = str(column)
    words = set(tokens(name))
    if name.upper() in _NHANES_DEMOGRAPHIC or words & _DEMOGRAPHIC:
        return "demographic"
    if name.upper() in _NHANES_BODY or words & _BODY or {"fat", "mass"} <= words \
            or {"body", "fat"} <= words:
        return "body"
    if energy_bearing(name) or is_nutrient(name):
        return "dietary"
    return None


def adjustment_card(state: Any) -> dict[str, Any] | None:
    """What the adjustment question shows: the covariates in groups that share the pack's guess,
    each confirmed with one tap (its ``decision``), the unguessed ones asked plainly, and what the
    answers so far derive. None when the question does not apply yet."""
    spec = current_estimand(state)
    if spec is None or _get(state, "purpose") != "inference":
        return None
    exposure = exposure_key(spec)
    effect = str(_get(spec, "effect") or "total")
    answers = current_answers(state)
    waiting = [c for c in asked_covariates(state) if c not in answers]
    by_guess: dict[str | None, list[str]] = {}
    for c in waiting:
        by_guess.setdefault(guess_of(c), []).append(c)
    groups = []
    for key in ("demographic", "dietary", "body"):
        columns = by_guess.get(key)
        if not columns:
            continue
        g = GUESSES[key]
        derived = derive(g["answers"], effect)
        groups.append({
            "key": key, "label": g["label"], "columns": columns, "guess": dict(g["answers"]),
            "reason": f"{g['reason']} ({_PACK08}).", "derived": derived.role,
            "derived_words": ROLE_WORDS[derived.role],
            "decision": {"kind": "set_adjustment", "exposure": exposure,
                         "answers": {c: dict(g["answers"]) for c in columns}}})
    if by_guess.get(None):
        groups.append({"key": "unguessed", "label": "No guess", "columns": by_guess[None],
                       "guess": None, "reason": "The pack says nothing about these; each is asked.",
                       "derived": None, "derived_words": None, "decision": None})
    derived = derived_roles(state)
    return {
        "exposure": exposure, "effect": effect, "family": bool(_get(spec, "family")),
        "questions": QUESTIONS,
        "groups": groups,
        "answered": {c: {"role": d.role, "words": ROLE_WORDS[d.role], "adjusted": d.adjusted,
                         "secondary": d.secondary, "why": d.why} for c, d in derived.items()},
        "adjusted": [c for c, d in derived.items() if d.adjusted],
        "left_out": [c for c, d in derived.items() if not d.adjusted],
        "secondary": secondary_columns(state),
        "source": VANDERWEELE,
    }


# The five questions, as the card asks them (each one line).
QUESTIONS = {
    "causes_exposure": "Is it a cause of the exposure?",
    "causes_outcome": "Is it a cause of the outcome?",
    "after_exposure": "Could the exposure have changed it, or was it measured after?",
    "instrument": "Does it affect the outcome only through the exposure?",
    "proxy": "Does it stand in for an unmeasured cause of both?",
}


def estimand_card(state: Any, task: str | None) -> dict[str, Any] | None:
    """What the estimand question offers: the exposure candidates, the effects, the contrast for an
    energy-bearing exposure, and only the measures the engine fits (the others named, refused)."""
    if _get(state, "purpose") != "inference" or _get(state, "roles") is None:
        return None
    candidates = exposure_candidates(state)
    family = family_exposures(state)
    return {
        "exposures": [{"column": c, "energy_contrast": energy_contrast_applies(state, c)}
                      for c in candidates],
        # Every exposure reported in turn, with its multiplicity method (MODELING_SEQUENCE §1 step
        # 2): offered with two or more exposures, as an omics table's features are.
        "family": ({"n": len(family), "energy_contrast": family_contrast_applies(state),
                    "measures": measures_offered(task, family=True),
                    "consequence": (f"Each of the {len(family):,} exposures is reported in turn, "
                                    f"adjusted for the covariates but not for the other exposures, "
                                    f"with a false-discovery statement (the feature-wise family).")}
                   if len(family) >= 2 and fitted_measures(task, family=True) else None),
        "effects": [
            {"effect": "total", "label": "Total effect",
             "consequence": "Everything the exposure changes downstream counts; mediators stay out."},
            {"effect": "direct", "label": "Direct effect",
             "consequence": "Holds the mediators fixed; their confounders must be adjusted too."}],
        "contrasts": [
            {"contrast": "substitution", "label": "Substitution",
             "consequence": "More of it in place of other calories, total energy held fixed."},
            {"contrast": "addition", "label": "Addition",
             "consequence": "Its calories added on top, every other source held fixed."}],
        "measures": measures_offered(task),
    }


# ── the caption, worded from the estimand ────────────────────────────────────


def caption(state: Any, task: str | None = None) -> str | None:
    """The sentence the inference table is captioned with, worded from the declared estimand: which
    effect of which exposure, on what scale, conditional on what, and what was left out and why."""
    spec = current_estimand(state)
    if spec is None or _get(state, "purpose") != "inference":
        return None
    target = _get(state, "target")
    effect = "direct" if _get(spec, "effect") == "direct" else "total"
    measure = str(_get(spec, "measure"))
    contrast = _get(spec, "contrast")
    what = {"substitution": "in place of other energy sources at fixed total energy",
            "addition": "its calories added, every other energy source fixed"}.get(contrast)
    if _get(spec, "family"):
        family = family_exposures(state)
        scale = (f"as the {MEASURE_WORDS[measure]}" if measure == "exposure_mean_difference"
                 else f"as a {MEASURE_WORDS.get(measure, measure)} per unit of each")
        text = (f"The {effect} effect of each of the {len(family):,} exposures "
                f"({_listing(family, limit=3)}) on {_tick(target)}, one at a time"
                + (f" ({what})" if what else "")
                + f", {scale}, with the false-discovery rate stated")
    else:
        exposure = _get(spec, "exposure")
        text = (f"The {effect} effect of {_tick(exposure)} on {_tick(target)}"
                + (f" ({what})" if what else "")
                + f", as a {MEASURE_WORDS.get(measure, measure)} per unit of {_tick(exposure)}")
    derived = derived_roles(state)
    adjusted = [c for c, d in derived.items() if d.adjusted]
    fe = fixed_effects_column(state)
    parts = ([_listing(adjusted, limit=6)] if adjusted else []) + (
        [f"an intercept for each {_tick(fe)} (fixed effects)"] if fe else [])
    text += (f", conditional on {' and '.join(parts)}" if parts
             else ", with no covariate adjusted for")
    out = [c for c, d in derived.items() if not d.adjusted and d.role in ("mediator", "collider")]
    if out:
        text += f"; {_listing(out)} left out as {'a consequence' if len(out) == 1 else 'consequences'} of the exposure"
    if measure in NON_COLLAPSIBLE:
        text += "; a conditional ratio, which changes with the covariates even without confounding"
    second = secondary_columns(state)
    if second:
        text += f". Declared beside it: further adjusted for {_listing(second)}"
    return text + f" (the adjustment set by the disjunctive cause criterion, {VANDERWEELE})."


def primary_features(fitted_features: Iterable[str], exposure: str) -> list[str]:
    """The model-matrix columns that carry the exposure's effect (its own column, or the spline or
    quintile terms the form made of it)."""
    out = []
    for f in fitted_features:
        name = str(f)
        if name == exposure or name.startswith(f"{exposure}_") or name.startswith(f"{exposure}["):
            out.append(name)
    return out


# ── what the server withholds while the plan is unanswered ───────────────────

# The questions whose answer an estimate rests on, and the purposes they hold estimates under.
HOLDS: dict[str, tuple[str, ...]] = {
    "follow_up": ("prediction", "inference"),
    "clusters": ("inference",),
    "estimand": ("inference",),
    "adjustment": ("inference",),
}
ESTIMATE_STAGES = ("fit", "substitution", "sensitivity", "calibration", "secondary")


def served_gate(state: Any, steps: Sequence[Any]) -> dict[str, Any] | None:
    """The first unanswered question an estimate rests on, as the server says it (``withhold``);
    None when every one is answered or does not apply."""
    from turbotab.core.voice import question_name

    purpose = _get(state, "purpose")
    for step in steps:
        key = _get(step, "key")
        if key not in HOLDS or _get(step, "status") not in ("open", "waiting"):
            continue
        if purpose not in HOLDS[key] and not (key == "follow_up" and purpose is None):
            continue
        name = question_name(str(key))
        why = {
            "follow_up": "whether everyone was followed for the same time decides between a "
                         "yes/no model and a time-to-event model",
            "clusters": "whether the participants are grouped decides the model's intercepts and "
                        "how its intervals are clustered",
            "estimand": "the exposure and its effect decide which estimate is reported and what it "
                        "means",
            "adjustment": "each covariate's answers decide whether it is adjusted for, left out or "
                          "set beside the primary",
        }[str(key)]
        return {"question": key,
                "reason": f"No estimate is shown until {name} is answered: {why}.",
                "exits": [{"label": f"Answer {name}", "decision": None}]}
    return None


def withhold(stage: str, artifact: Any, gate: Mapping[str, Any]) -> Any:
    """``artifact`` with every estimate removed and the reason first: the coefficient tables, their
    tests and intervals; curves; refits. Scores stay (they describe fit, not an effect)."""
    if not isinstance(artifact, dict):
        return artifact
    out = dict(artifact)
    reason = str(gate["reason"])
    out["withheld"] = reason
    if stage == "fit":
        models = []
        for m in out.get("models") or []:
            m = dict(m)
            info = m.get("inference") or {}
            if info.get("refused") and not m.get("coefficients"):
                # Already refused with its own reason and exits (an unanswered survey question, a
                # held missing-values answer): nothing is estimated, so its refusal speaks first.
                models.append(m)
                continue
            m["coefficients"] = None
            m["inference"] = None
            m["exposure_tests"] = []
            m["concerns"] = [reason, *(m.get("concerns") or [])]
            models.append(m)
        out["models"] = models
        return out
    if stage == "sensitivity":
        out["families"] = []
        out["changes"] = {}
        return out
    for key in ("curves", "families", "models", "estimates", "rows", "fits"):
        if key in out:
            out[key] = [] if isinstance(out[key], list) else None
    return out


def annotate_fit(artifact: Any, state: Any) -> Any:
    """The served fit under a declared estimand: the caption worded from it, and which rows of each
    table are the exposure's effect (the rest are adjustment terms, not effect estimates)."""
    if not isinstance(artifact, dict) or _get(state, "purpose") != "inference":
        return artifact
    spec = current_estimand(state)
    if spec is None:
        return artifact
    exposures = exposures_of(state, spec)
    out = dict(artifact)
    out["estimand"] = {
        "exposure": exposure_key(spec), "effect": _get(spec, "effect"),
        "measure": _get(spec, "measure"),
        "contrast": _get(spec, "contrast"), "caption": caption(state, out.get("task")),
        "adjusted": [c for c, d in derived_roles(state).items() if d.adjusted],
        "left_out": {c: d.role for c, d in derived_roles(state).items() if not d.adjusted},
        "secondary": secondary_columns(state),
        "features": sorted({f for m in out.get("models") or [] for e in exposures
                            for f in primary_features([r.get("feature") for r in
                                                       (m.get("coefficients") or [])], e)}),
    }
    return out


# ── refusals ─────────────────────────────────────────────────────────────────


def _refusal(code: str, message: str, exits: Sequence[Mapping[str, Any]]) -> Exception:
    from turbotab.core.decisions import Refusal

    return Refusal(code, message, exits=exits)


def _state(ctx: Any) -> Any:
    from turbotab.core.decisions import _state as state_of

    return state_of(ctx)


def _artifact(ctx: Any, stage: str) -> Mapping[str, Any] | None:
    from turbotab.core.sequence import artifact

    return artifact(ctx, stage)


def _not_inference(what: str) -> Exception:
    from turbotab.core.decisions import SetPurpose

    return _refusal(
        "not_inference",
        f"Under prediction no coefficient is read as an effect, so {what} does not arise; it is "
        f"asked under inference.",
        [{"label": "Make the purpose inference", "decision": SetPurpose(purpose="inference")}])


# set_censoring


def _censoring_names_the_outcome(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import _UNKNOWN, SetTask, _ctx, _target_of

    target = _target_of(ctx)
    if target is _UNKNOWN:
        return
    if target is None:
        raise _refusal("no_target", "Choose the outcome first; the follow-up belongs to it.",
                       [{"label": "Choose the outcome", "decision": None}])
    if decision.column != target:
        raise _refusal("not_the_target",
                       f"The outcome is {_tick(target)}, not {_tick(decision.column)}; the follow-up "
                       f"question answers for the outcome.",
                       [{"label": f"Answer the follow-up question for {_tick(target)}",
                         "decision": None}])
    state = _state(ctx)
    task = _get(state, "task") or _ctx(ctx, "task")
    if task == "time_to_event":
        raise _refusal(
            "time_to_event_declared",
            f"{_tick(target)} is declared a time to event, whose follow-up ends at different times "
            f"by definition; name its follow-up time instead.",
            [{"label": f"Analyze {_tick(target)} as yes/no", "decision": SetTask(
                column=target, task="binary")}])
    if task is not None and task != "binary":
        raise _refusal("not_binary",
                       f"{_tick(target)} is read as {str(task).replace('_', ' ')}; the follow-up "
                       f"question is about a yes/no outcome.",
                       [{"label": "Keep the outcome as it is", "decision": None}])


def _same_follow_up_against_the_data(decision: Any, ctx: Any) -> None:
    """"The same for everyone" over a column whose values say the follow-up varies: block and record
    (PROBAST 4.6). Exits: the time to event, or the attestation."""
    from turbotab.core.decisions import SetTask

    if decision.acknowledged:
        return
    info = _artifact(ctx, "target_info") or {}
    if info.get("column") != decision.column:
        return
    varying = [c for c in info.get("follow_up") or [] if c.get("varies")]
    if not varying:
        return
    c = varying[0]
    column = str(c["column"])
    raise _refusal(
        "follow_up_varies",
        f"{_tick(column)} reads as a follow-up time and runs from {c['min']:g} to {c['max']:g}: if "
        f"it is how long each person was followed, some left or the study ended before their event "
        f"could be seen, and a yes/no model counts them as having none. A time-to-event model "
        f"keeps each person for the time they were followed ({PROBAST}).",
        [{"label": f"Analyze {_tick(decision.column)} as a time to event",
          "decision": SetTask(column=decision.column, task="time_to_event")},
         {"label": f"Everyone was followed equally long; {_tick(column)} is something else",
          "decision": decision.model_copy(update={"acknowledged": True})}])


# set_clusters


def _clusters_name_a_grouping(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import ROW_ID, SetClusters, _columns_of, _info, _target_of, _UNKNOWN

    state = _state(ctx)
    purpose = _get(state, "purpose")
    column = decision.column
    if column is not None:
        columns = _columns_of(ctx)
        if column == ROW_ID or (columns is not None and column not in columns):
            raise _refusal("unknown_column", f"This dataset has no column named {_tick(column)}.",
                           [{"label": "Choose one of the dataset's columns", "decision": None}])
        target = _target_of(ctx)
        if target is not _UNKNOWN and column == target:
            raise _refusal("cluster_is_outcome",
                           f"{_tick(column)} is the outcome, so it cannot group the participants.",
                           [{"label": "Name another column", "decision": None}])
        info = _info(ctx, column)
        if info is not None:
            n_unique = int(info.get("n_unique") or 0)
            present = None
            n_rows = _get(ctx, "n_rows")
            if n_rows is not None:
                present = int(n_rows) - int(info.get("n_missing") or 0)
            if n_unique < 2 or (present is not None and n_unique >= present):
                raise _refusal(
                    "not_a_grouping",
                    f"{_tick(column)} has {n_unique:,} distinct values"
                    + (f" over {present:,} rows" if present is not None else "")
                    + ", so it groups no participants together.",
                    [{"label": "Nothing groups the participants",
                      "decision": SetClusters(column=None, acknowledged=True)},
                     {"label": "Name another column", "decision": None}])
    if purpose == "prediction" and decision.adjust is not None:
        raise _refusal(
            "not_inference",
            "Under prediction a grouping decides how the rows are validated (by whole groups, and "
            "across groups), not the model's intercepts; it is recorded without them.",
            [{"label": f"Group by {_tick(column)}" if column else "Record the grouping",
              "decision": decision.model_copy(update={"adjust": None})}])
    if purpose == "inference" and column is not None and decision.adjust is None:
        raise _refusal(
            "how_adjusted",
            f"Under inference participants in one {_tick(column)} share more than chance: say "
            f"whether each {_tick(column)} gets its own intercept as well as clustered intervals.",
            [{"label": f"Adjust for {_tick(column)} and cluster by it",
              "decision": decision.model_copy(update={"adjust": "fixed_effects"})},
             {"label": f"Cluster the intervals by {_tick(column)} only",
              "decision": decision.model_copy(update={"adjust": "cluster_only"})}])


def _no_grouping_is_recorded(decision: Any, ctx: Any) -> None:
    """Under inference, "nothing groups them" over a column that reads as a grouping is block and
    record (audit RO-08: "adjust and cluster, or record why not")."""
    from turbotab.core.decisions import SetClusters

    state = _state(ctx)
    if decision.column is not None or decision.acknowledged or _get(state, "purpose") != "inference":
        return
    candidates = cluster_candidates(state, _artifact(ctx, "roles"))
    if not candidates:
        return
    shown = _listing(candidates, limit=3)
    raise _refusal(
        "grouping_reads",
        f"{shown} {'reads' if len(candidates) == 1 else 'read'} as a group of participants. People "
        f"in one site or household are more alike than people across them, so intervals that "
        f"treat them as independent are too narrow, and between-group differences can confound "
        f"the exposure.",
        [*({"label": f"Adjust for {_tick(c)} and cluster by it",
            "decision": SetClusters(column=c, adjust="fixed_effects")} for c in candidates[:2]),
         {"label": "They group nothing; record that",
          "decision": decision.model_copy(update={"acknowledged": True})}])


# set_estimand


def _estimand_is_for_inference(decision: Any, ctx: Any) -> None:
    if _get(_state(ctx), "purpose") == "prediction":
        raise _not_inference("which exposure and which effect")


def _estimand_names_a_predictor(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import _columns_of
    from turbotab.core.readings import confirm_exits, unsettled

    state = _state(ctx)
    if state is None or _get(state, "roles") is None:
        return
    if decision.family:
        family = family_exposures(state)
        if len(family) < 2:
            offered = exposure_candidates(state)
            raise _refusal(
                "no_family",
                f"An exposure family reports each of two or more exposures in turn; "
                f"{'only ' + _tick(family[0]) + ' is' if family else 'no column is'} in the model "
                f"as an exposure.",
                [{"label": f"The exposure is {_tick(c)}",
                  "decision": decision.model_copy(update={"family": False, "exposure": c, "contrast": (
                      decision.contrast or "substitution") if energy_contrast_applies(state, c)
                      else None})} for c in offered[:4]])
        return
    columns = _columns_of(ctx)
    exposure = decision.exposure
    if columns is not None and exposure not in columns:
        raise _refusal("unknown_column", f"This dataset has no column named {_tick(exposure)}.",
                       [{"label": "Choose one of the dataset's columns", "decision": None}])
    waiting = unsettled(state, [exposure])
    if waiting:
        raise _refusal(
            "role_unconfirmed",
            f"{_tick(exposure)}'s role was proposed below high confidence and not confirmed on its "
            f"own; confirm it before it is the exposure.",
            confirm_exits(state, waiting))
    if exposure not in predictor_roles(state):
        offered = exposure_candidates(state)
        raise _refusal(
            "not_in_model",
            f"{_tick(exposure)} is not in the model (its role is "
            f"{(_get(state, 'roles') or {}).get(exposure, 'none')}, or the missing-values answer "
            f"left it out); the exposure is one of the model's columns.",
            [{"label": f"The exposure is {_tick(c)}",
              "decision": decision.model_copy(update={"exposure": c, "contrast": (
                  decision.contrast or "substitution") if energy_contrast_applies(state, c)
                  else None})} for c in offered[:4]])


def _estimand_measure_is_fitted(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import _ctx

    state = _state(ctx)
    task = _get(state, "task") or _ctx(ctx, "task")
    measures = (FAMILY_MEASURE_OF_TASK if decision.family else MEASURE_OF_TASK)
    fitted = measures.get(str(task)) if task else None
    if decision.family and task and fitted is None:
        raise _refusal(
            "family_not_fitted",
            f"An exposure family is reported by the feature-wise family, which fits a numeric or "
            f"yes/no outcome, not a {str(task).replace('_', ' ')} one; name one exposure.",
            [{"label": "Name one exposure", "decision": None}])
    if decision.measure in NOT_FITTED:
        exits = ([{"label": f"Report the {MEASURE_WORDS[fitted]}",
                   "decision": decision.model_copy(update={"measure": fitted})}] if fitted else [])
        raise _refusal("measure_not_fitted",
                       f"The {MEASURE_WORDS[decision.measure]} is not fitted here: "
                       f"{NOT_FITTED[decision.measure]}. The {MEASURE_WORDS.get(fitted, 'model')} "
                       f"is what the model estimates.", exits)
    if fitted is not None and decision.measure != fitted:
        raise _refusal(
            "measure_mismatch",
            f"A {str(task).replace('_', ' ')} outcome's model estimates a {MEASURE_WORDS[fitted]}, "
            f"not a {MEASURE_WORDS.get(decision.measure, decision.measure)}.",
            [{"label": f"Report the {MEASURE_WORDS[fitted]}",
              "decision": decision.model_copy(update={"measure": fitted})}])


def _estimand_contrast_fits_the_exposure(decision: Any, ctx: Any) -> None:
    state = _state(ctx)
    if state is None or _get(state, "roles") is None:
        return
    applies = (family_contrast_applies(state) if decision.family
               else energy_contrast_applies(state, decision.exposure))
    named = "An exposure of the family" if decision.family else _tick(decision.exposure)
    if applies and decision.contrast is None:
        raise _refusal(
            "which_contrast",
            f"{named} carries energy and total energy is in the model: say "
            f"whether its effect is a substitution (more of it in place of other calories, total "
            f"energy fixed) or an addition (its calories added on top). The two are different "
            f"estimands (Tomova et al. 2022).",
            [{"label": "Substitution", "decision": decision.model_copy(update={"contrast": "substitution"})},
             {"label": "Addition", "decision": decision.model_copy(update={"contrast": "addition"})}])
    if not applies and decision.contrast is not None:
        raise _refusal(
            "no_energy_contrast",
            f"{'No exposure of the family' if decision.family else _tick(decision.exposure)} "
            f"carries {'' if decision.family else 'no '}energy against a total in the model, so it is "
            f"neither a substitution nor an addition of calories.",
            [{"label": "Leave the contrast out",
              "decision": decision.model_copy(update={"contrast": None})}])


# set_adjustment


def _adjustment_follows_the_estimand(decision: Any, ctx: Any) -> None:
    from turbotab.core.voice import question_name

    state = _state(ctx)
    if _get(state, "purpose") == "prediction":
        raise _not_inference("an adjustment set")
    if state is None:
        return
    spec = current_estimand(state)
    if spec is None:
        raise _refusal("no_estimand",
                       f"Declare the exposure first ({question_name('estimand')}); each covariate is "
                       f"asked about against it.",
                       [{"label": f"Answer {question_name('estimand')} first", "decision": None}])
    exposure = exposure_key(spec)
    if decision.exposure != exposure:
        said = "the exposure family" if _get(spec, "family") else _tick(exposure)
        raise _refusal("other_exposure",
                       f"The exposure is {said}, not {_tick(decision.exposure)}; "
                       f"covariates are asked about against the exposure declared.",
                       [{"label": f"Answer for {_tick(exposure)}",
                         "decision": decision.model_copy(update={"exposure": exposure})}])
    known = set(asked_covariates(state))
    strangers = [c for c in decision.answers if c not in known]
    if strangers:
        what = ("the exposure itself" if strangers == [exposure] else
                "not a covariate in the model (a covariate is a recorded exposure or covariate "
                "role; total energy is decided by the energy question)")
        raise _refusal("not_a_covariate", f"{_listing(strangers)}: {what}.",
                       [{"label": "Answer for the model's covariates only",
                         "decision": decision.model_copy(update={"answers": {
                             c: a for c, a in decision.answers.items() if c in known}})
                         if any(c in known for c in decision.answers) else None}])


def _answers_hold_together(decision: Any, ctx: Any) -> None:
    for column, a in decision.answers.items():
        if a.instrument and a.causes_outcome == "yes":
            raise _refusal(
                "instrument_causes_outcome",
                f"{_tick(column)} is answered a cause of the outcome and an instrument; an "
                f"instrument affects the outcome only through the exposure.",
                [{"label": f"{_tick(column)} is not an instrument", "decision": decision.model_copy(
                    update={"answers": {**decision.answers,
                                        column: a.model_copy(update={"instrument": False})}})}])
        if a.instrument and a.causes_exposure == "no":
            raise _refusal(
                "instrument_not_a_cause",
                f"{_tick(column)} is answered an instrument but not a cause of the exposure; an "
                f"instrument is one.",
                [{"label": f"{_tick(column)} is not an instrument", "decision": decision.model_copy(
                    update={"answers": {**decision.answers,
                                        column: a.model_copy(update={"instrument": False})}})}])


def _mediators_stay_out_of_a_total_effect(decision: Any, ctx: Any) -> None:
    """Block and record (MODELING_SEQUENCE §4): a mediator or a consequence of the exposure kept in
    a total-effect set is refused until attested; the exits leave it out and offer "further
    adjusted for" it as the labeled secondary."""
    state = _state(ctx)
    spec = current_estimand(state) if state is not None else None
    effect = str(_get(spec, "effect") or "total")
    for column, a in decision.answers.items():
        d = derive(a.model_copy(update={"acknowledged": True}), effect)
        if not a.keep:
            continue
        if d.role not in ("mediator", "collider", "timing_unknown") or (
                d.role == "mediator" and effect == "direct"):
            continue
        if a.acknowledged:
            continue
        what = {"mediator": "a mediator", "collider": "a consequence of the exposure",
                "timing_unknown": "possibly a consequence of the exposure"}[d.role]
        raise _refusal(
            "mediator_in_total_effect",
            f"{_tick(column)} is {what} by your answers: in a total-effect set it removes part of "
            f"the effect, or opens a path that is not causal ({VANDERWEELE}: confounders \"ought to "
            f"be controlled for in the estimation of the total effect\", mediators \"ought not\").",
            [{"label": f"Leave {_tick(column)} out; further adjusted for it, beside",
              "decision": decision.model_copy(update={"answers": {**decision.answers, column:
                                                                  a.model_copy(update={"keep": False, "further": True})}})},
             {"label": f"Keep {_tick(column)}; record that the estimate is not a total effect",
              "decision": decision.model_copy(update={"answers": {**decision.answers, column:
                                                                  a.model_copy(update={"acknowledged": True})}})}])


# set_energy_adjustment: the contrast the estimand declares (MODELING_SEQUENCE §2)

SUBSTITUTION_METHODS = ("standard", "residual", "residual_energy_dropped", "all_components",
                        "density", "density_multivariate")
ADDITION_METHODS = ("partition", "all_components")


def _energy_model_fits_the_contrast(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import SetEnergyAdjustment, SetEstimand

    state = _state(ctx)
    if _get(state, "purpose") != "inference":
        return
    spec = current_estimand(state)
    contrast = _get(spec, "contrast") if spec is not None else None
    if contrast is None:
        return
    allowed = SUBSTITUTION_METHODS if contrast == "substitution" else ADDITION_METHODS
    if decision.method in allowed:
        return
    from turbotab.core.methods.energy import METHOD_TABLE

    label = METHOD_TABLE.get(decision.method, {}).get("label", decision.method)
    other = "addition" if contrast == "substitution" else "substitution"
    ranked = (("all_components", "standard", "residual") if contrast == "substitution"
              else ("all_components", "partition"))
    whose = "the exposures'" if _get(spec, "family") else f"{_tick(_get(spec, 'exposure'))}'s"
    raise _refusal(
        "contrast_mismatch",
        f"The estimand is {'a substitution' if contrast == 'substitution' else 'an addition'} of "
        f"{whose} calories, and the {label[0].lower() + label[1:]} "
        f"estimates {'no substitution' if contrast == 'substitution' else 'a substitution, not an addition'} "
        f"(Tomova et al. 2022).",
        [*({"label": METHOD_TABLE[m]["label"],
            "decision": SetEnergyAdjustment(**{**decision.model_dump(exclude={"kind"}), "method": m})}
           for m in ranked),
         {"label": f"Make the estimand an {other}" if other == "addition" else f"Make the estimand a {other}",
          "decision": SetEstimand(**{**spec.model_dump(), "contrast": other})}])


# select_models: an exposure family is reported by the feature-wise family


def _models_fit_the_family(decision: Any, ctx: Any) -> None:
    """Under inference, a declared exposure family (each exposure in turn, adjusted for the
    covariates, with a false-discovery statement) is estimated by the feature-wise family alone; a
    joint model adjusts each exposure for the others, which is another estimand."""
    from turbotab.core.decisions import SelectModels

    state = _state(ctx)
    if _get(state, "purpose") != "inference":
        return
    spec = current_estimand(state)
    if spec is None or not _get(spec, "family"):
        return
    others = [m for m in decision.models if m != "featurewise"]
    if not others:
        return
    raise _refusal(
        "family_needs_featurewise",
        f"The estimand is the exposure family, each exposure in turn adjusted for the covariates; "
        f"{_listing(others)} {'adjusts' if len(others) == 1 else 'adjust'} each exposure for the "
        f"others, which is another estimand. The feature-wise family estimates this one.",
        [{"label": "Fit the feature-wise family", "decision": SelectModels(models=["featurewise"])},
         {"label": "Name one exposure instead", "decision": None}])


def _register() -> None:
    from turbotab.core.decisions import register_validator

    register_validator("set_censoring", _censoring_names_the_outcome)
    register_validator("set_censoring", _same_follow_up_against_the_data)
    register_validator("set_clusters", _clusters_name_a_grouping)
    register_validator("set_clusters", _no_grouping_is_recorded)
    register_validator("set_estimand", _estimand_is_for_inference)
    register_validator("set_estimand", _estimand_names_a_predictor)
    register_validator("set_estimand", _estimand_measure_is_fitted)
    register_validator("set_estimand", _estimand_contrast_fits_the_exposure)
    register_validator("set_adjustment", _adjustment_follows_the_estimand)
    register_validator("set_adjustment", _answers_hold_together)
    register_validator("set_adjustment", _mediators_stay_out_of_a_total_effect)
    register_validator("set_energy_adjustment", _energy_model_fits_the_contrast)
    register_validator("select_models", _models_fit_the_family)


_register()

__all__ = [
    "Derived", "ESTIMATE_STAGES", "GUESSES", "HOLDS", "MEASURE_OF_TASK", "MEASURE_WORDS",
    "NOT_FITTED", "QUESTIONS", "ROLE_PLURAL", "ROLE_SINGULAR", "ROLE_WORDS", "adjustment_answer",
    "adjustment_card", "asked_covariates",
    "adjustment_gate", "adjustment_left_out", "annotate_fit", "caption", "cluster_answer",
    "cluster_candidates", "clusters_gate", "covariates", "current_answers", "current_estimand",
    "derive", "derived_roles", "effective_task", "estimand_card", "estimand_gate",
    "exposure_candidates", "fixed_effects_column", "follow_up_answer", "follow_up_candidates",
    "follow_up_gate", "guess_of", "measures_offered", "primary_features", "reads_as_follow_up",
    "secondary_columns", "served_gate", "unanswered", "withhold",
]

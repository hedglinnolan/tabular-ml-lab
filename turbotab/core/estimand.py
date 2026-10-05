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
asked about after the roles: does it group the participants? Under inference so is any column that
can structurally group rows, whatever its name, with its guess shown (the routing gate's leash note;
``turbotab/core/groupings.py``); a sex, an age or any category of ten values or fewer never is.
Under inference the recommended answer
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

Asking this of thirty covariates stays light (BLUEPRINT §14.2): the packs guess each covariate's
answers from its name under the exposure–outcome pairing (``turbotab/core/covariate_guesses.py``;
NUTRITION_PACK §08, "The adjustment card's guesses"): demographics and lifestyle are confounders,
other nutrients possible confounders, body size, clinical measurements and medications measured
with the exposure possible mediators (the declared with-and-without pair), a measurement of the
outcome's own kind another measure of it. Covariates with the same guess form one block, confirmed
with one tap (one ``set_adjustment`` naming exactly its columns), every member's guess shown with
its reason and source; a multi-select answer settles exactly the covariates it lists; a covariate
the packs say nothing about is asked without a guess.

**No estimate before the plan.** Under inference no coefficient, curve or contrast is served while
the exposure, the effect or a covariate's answers are missing (``served_gate``, read by the server
like the seal's withholding of held-out scores); the caption is worded from the estimand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

VANDERWEELE = "VanderWeele 2019, Eur J Epidemiol 34:211–219"
VALERI = "Valeri & VanderWeele 2013, Psychol Methods 18:137–150"
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
                   "pyears", "py", "pyrs", "personyears", "censor", "censored", "censoring",
                   "exit", "observed", "followed", "tstop",
                   # NHANES linked mortality: person-months of follow-up from the interview
                   # (``PERMTH_INT``) or from the examination (``PERMTH_EXM``)
                   "permth"}
_DURATION_WORDS = {"years", "yrs", "year", "months", "month", "days", "day", "weeks", "week",
                   "time", "duration"}
# Words that, beside a duration, say it is how long a row was observed: ``years_to_cvd``,
# ``time_in_study``, ``time_at_risk``, ``person_years``, ``obs_years``, ``observation_time``.
_FOLLOW_CONTEXT = {"to", "event", "since", "elapsed", "follow", "up", "observation", "obs",
                   "study", "risk", "person", "until", "end", "stop"}
_NOT_A_DURATION = {"date", "datetime", "timestamp", "cycle", "visit", "wave", "round", "recall",
                   "session", "calendar", "baseline", "hour", "hours", "dob", "birth"}
NUMERIC = ("numeric", "integer")


def reads_as_follow_up(name: Any) -> bool:
    """The name says how long the row was observed: ``followup_years``, ``fu_days``,
    ``time_to_event``, ``survival_months``, ``futime``, ``PERMTH_INT``, ``time_in_study``,
    ``years_to_cvd``, ``length_of_follow_up``; not a date, a cycle or a visit index. A name is a
    guess the follow-up question leads with, never what decides whether it is asked (the routing
    gate: NHANES linked mortality's ``PERMTH_INT`` read as no follow-up, and the question skipped)."""
    from turbotab.core.recognizers import is_rate, tokens

    words = set(tokens(name))
    if not words or words & _NOT_A_DURATION or is_rate(name):
        return False
    if words & FOLLOW_UP_WORDS or {"follow", "up"} <= words:
        return True
    return bool(words & _DURATION_WORDS) and (bool(words & _FOLLOW_CONTEXT)
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


# The lenses whose yes/no outcomes are often events followed over time (a death, an incident
# disease in a cohort), and "something else, or not sure" (no lens): their yes/no outcome is always
# asked about its follow-up, whatever the columns are named (the routing gate: NHANES linked
# mortality's ``MORTSTAT`` beside ``PERMTH_INT`` was skipped by a name test and fit as logistic).
# Under an assay or survey lens alone a yes/no outcome is a status at sampling (a case or a control,
# an answer), so it is asked only beside a column that reads as a follow-up time.
FOLLOWED_LENSES = ("clinical", "dietary")


def follow_up_asked_always(state: Any) -> bool:
    lens = list(_get(state, "lens") or [])
    return not lens or any(k in FOLLOWED_LENSES for k in lens)


def follow_up_gate(state: Any, target_info: Any = None) -> Gate:
    """Whether the follow-up question is asked: a time to event, whose follow-up must be named;
    a yes/no outcome under the clinical or dietary lens or none (one tap when everyone was followed
    for the same time, :data:`FOLLOWED_LENSES`); under another lens, a yes/no outcome beside a
    column that reads as a follow-up time."""
    if _get(state, "target") is None:
        return None
    task = effective_task(state, target_info)
    if task is None:
        return None
    if task not in ("binary", "time_to_event"):
        return ("not_applicable", f"The outcome is read as {task.replace('_', ' ')}, so no event "
                                  f"is followed over time.")
    if task == "time_to_event" or follow_up_asked_always(state):
        return None
    if target_info is None or _get(target_info, "column") != _get(state, "target"):
        return None
    candidates = _get(target_info, "follow_up")
    if candidates is None:
        return None
    if not candidates:
        return ("skipped", "no numeric column reads as a follow-up time, and under this lens a "
                           "yes/no outcome is a status at sampling, so it is read as counted over "
                           "one period for everyone.")
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


def grouping_candidates(state: Any, roles: Any = None) -> list[dict[str, Any]]:
    """Every column the grouping question asks about, each with the guess it shows: the named
    groupings (:func:`cluster_candidates`, guessed to group the participants), then, under
    inference, every column that can structurally group rows (the routing gate's leash note;
    ``turbotab.core.groupings``), guessed from its values."""
    from turbotab.core.groupings import candidates

    named = cluster_candidates(state, roles)
    out = [{"column": c, "guess": "yes", "why": "named like a group of participants (a site, a "
                                                "centre, a household or a batch)",
            "structural": False} for c in named]
    out += [c for c in candidates(state, roles) if c["column"] not in named]
    return out


def grouping_card(state: Any, roles: Any = None) -> dict[str, Any] | None:
    """What the grouping question shows: each column it asks about with its guess and the evidence
    the guess rests on (never a settled reading: the user says whether rows sharing a value belong
    together), and the answers it takes. None when no column can group the rows."""
    found = grouping_candidates(state, roles)
    if not found:
        return None
    inference = _get(state, "purpose") == "inference"
    return {"columns": found, "inference": inference,
            "options": (["fixed_effects", "cluster_only", "none"] if inference
                        else ["group", "none"]),
            "none_recorded": any(c["guess"] == "yes" for c in found) and inference}


def clusters_gate(state: Any, roles: Any = None) -> Gate:
    if _get(state, "roles") is None:
        return None
    if not grouping_candidates(state, roles):
        return ("skipped", "no column reads as a site, centre, household or batch, or can group "
                           "the rows by its values.")
    return None


def _withdrawn(state: Any, column: str) -> bool:
    """Whether the user said, after naming ``column`` as the grouping, that it groups nothing: its
    ``cluster`` reading confirmed "no" (the answer itself confirmed "yes", so only a later word
    can say "no"; BLUEPRINT §14.3, every confirmation is honored)."""
    found = (_get(state, "reading_confirmations") or {}).get(f"cluster:{column}")
    return found == "no"


def cluster_answer(state: Any) -> str | None:
    """The column the user said groups the participants, whatever the purpose; None once a later
    confirmation said that column groups nothing."""
    spec = _get(state, "clusters")
    column = _get(spec, "column") if spec is not None else None
    return None if column is None or _withdrawn(state, str(column)) else column


def fixed_effects_column(state: Any) -> str | None:
    """Under inference, the grouping the model gives its own intercept per level (none once a later
    confirmation said it groups nothing)."""
    spec = _get(state, "clusters")
    if _get(state, "purpose") != "inference" or spec is None:
        return None
    column = cluster_answer(state)
    return column if column and _get(spec, "adjust") == "fixed_effects" else None


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
# Ruling 9: the effect measure is part of the estimand, "difference or ratio; conditional or
# marginal". A linear model's difference is both (collapsible: the conditional and the marginal
# difference agree); every ratio a family fits is conditional; the marginal risk difference and
# ratio are standardized over the analyzed rows from a yes/no outcome's logistic model.
MEASURE_SCALE = {"mean_difference": "difference", "risk_difference": "difference",
                 "exposure_mean_difference": "difference"}
MARGINAL = ("risk_difference", "risk_ratio")
MARGINAL_TASKS = ("binary",)
G_COMPUTATION = ("standardization over the analyzed rows (g-computation) from the logistic model, "
                 "with a bootstrap of the whole chain for its interval")
NOT_FITTED = {
    "risk_difference": "a marginal risk difference is standardized here from a yes/no outcome's "
                       "logistic model only",
    "risk_ratio": "a marginal risk ratio is standardized here from a yes/no outcome's logistic "
                  "model only",
}


def measure_facts(measure: str) -> dict[str, Any]:
    """Ruling 9's two labels of a measure, and whether it is collapsible."""
    scale = MEASURE_SCALE.get(measure, "ratio")
    marginal = measure in MARGINAL
    both = measure in ("mean_difference",)
    return {"scale": scale,
            "conditioning": "conditional and marginal" if both else
            ("marginal" if marginal else "conditional"),
            "collapsible": measure not in NON_COLLAPSIBLE}


# What the feature-wise family (``models/featurewise.py``) estimates for an exposure family, each
# exposure in turn adjusted for the covariates: a numeric outcome's difference in its mean per unit
# of the exposure; a yes/no outcome's difference in the exposure's mean between the event and the
# other level (the limma design). It fits no other task, so no family is offered for one.
FAMILY_MEASURE_OF_TASK = {"regression": "mean_difference", "binary": "exposure_mean_difference"}


def fitted_measures(task: str | None, family: bool = False) -> list[str]:
    """The measures the engine fits for ``task``: one exposure's, the task's model family's and,
    for a yes/no outcome, the marginal risk difference and ratio standardized from it; an exposure
    family's, the feature-wise family's."""
    fitted = (FAMILY_MEASURE_OF_TASK if family else MEASURE_OF_TASK).get(str(task))
    out = [fitted] if fitted else []
    if not family and task in MARGINAL_TASKS:
        out += list(MARGINAL)
    return out


def event_share(values: Any, event: Any) -> float | None:
    """The share of recorded rows at the event level the user named (matched as the fit codes it,
    ``stages.rows._level_key``); None while no event is named or it is no level here."""
    import pandas as pd

    from turbotab.core.stages.rows import _level_key

    present = pd.Series(values).dropna()
    if event is None or present.empty:
        return None
    hit = present.astype(object).map(_level_key) == _level_key(event)
    return float(hit.mean()) if hit.any() else None


def marginal_first(task: str | None, prevalence: float | None) -> bool:
    """Ruling 9: the marginal measures rank first for a yes/no outcome whose event is common
    (above 10%, where the odds ratio no longer approximates the risk ratio; Zhang & Yu 1998)."""
    from turbotab.core.models.effects import COMMON_OUTCOME

    return task in MARGINAL_TASKS and prevalence is not None and prevalence > COMMON_OUTCOME


def _measure_reason(measure: str, prevalence: float | None) -> str:
    from turbotab.core.models.effects import COMMON_OUTCOME, ZHANG_YU

    if measure in MARGINAL:
        common = (f"; the event is common here ({prevalence:.0%} of rows), so the odds ratio "
                  f"overstates the risk ratio ({ZHANG_YU})" if prevalence is not None
                  and prevalence > COMMON_OUTCOME else "")
        return (f"marginal: each row's predicted risk averaged over the analyzed rows "
                f"(g-computation from the logistic model), so adding a cause of the outcome only "
                f"does not change it{common}")
    if measure in NON_COLLAPSIBLE:
        return ("conditional and non-collapsible: adding a covariate that predicts the outcome "
                "changes it even without confounding")
    if measure == "exposure_mean_difference":
        return "the feature-wise family's: each exposure modeled on the outcome and the covariates"
    return "collapsible: the conditional and the marginal difference agree in a linear model"


def measures_offered(task: str | None, family: bool = False,
                     prevalence: float | None = None) -> list[dict[str, Any]]:
    """The effect measures the estimand question offers for ``task``, each with ruling 9's labels
    (a difference or a ratio; conditional or marginal; collapsible or not) and why, ranked: for a
    yes/no outcome whose event is common (``prevalence`` above 10%) the marginal risk difference and
    ratio first, else the model's conditional measure first. Another task's request for a marginal
    measure is named and refused (:data:`NOT_FITTED`)."""
    fitted = fitted_measures(task, family)
    if marginal_first(task, prevalence) and not family:
        fitted = [m for m in fitted if m in MARGINAL] + [m for m in fitted if m not in MARGINAL]
    out = [{"measure": m, "label": MEASURE_WORDS[m], "fitted": True,
            "reason": _measure_reason(m, prevalence), **measure_facts(m), "rank": i + 1}
           for i, m in enumerate(fitted)]
    if not family and task is not None and task not in MARGINAL_TASKS:
        out += [{"measure": m, "label": MEASURE_WORDS[m], "fitted": False, "reason": why,
                 **measure_facts(m), "rank": None} for m, why in NOT_FITTED.items()]
    return out


def precision_note(measure: str | None, columns: Sequence[str]) -> str | None:
    """MODELING_SEQUENCE §2 (the effect measure): "adding a precision covariate under an OR or HR
    *changes* the conditional estimand, and the app says so". None for a collapsible measure or
    when no column is a cause of the outcome only."""
    if measure not in NON_COLLAPSIBLE or not columns:
        return None
    one = len(columns) == 1
    return (f"Under a {MEASURE_WORDS[measure]}, adjusting for {_listing(list(columns))}, "
            f"{'a cause' if one else 'causes'} of the outcome only, changes the conditional "
            f"estimand, not only its precision: the ratio is non-collapsible ({DANIEL}).")


DANIEL = "Daniel, Zhang & Farewell 2021, Biom J 63:528"


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
    "mediator_confounder": "common cause of a mediator and the outcome",
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
    "mediator_confounder": "a common cause of a mediator and the outcome",
    "confounder": "a confounder", "exposure_cause": "a cause of the exposure",
    "precision": "a cause of the outcome only", "proxy": "a proxy for an unmeasured common cause",
    "mediator": "a mediator", "collider": "a consequence of the exposure (a possible collider)",
    "instrument": "an instrument", "timing_unknown": "of unknown timing",
    "not_a_cause": "a cause of neither",
}
ROLE_PLURAL = {
    "mediator_confounder": "common causes of a mediator and the outcome",
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
    direct = effect == "direct"
    # A direct effect (MODELING_SEQUENCE §1 step 3) also asks whether a covariate the criterion
    # would leave out is a common cause of a mediator and the outcome.
    confounds = _get(a, "confounds_mediator") in ("yes", "unknown") if direct else False
    if _get(a, "instrument"):
        return Derived("instrument", False, further,
                       "a known instrument: it moves the outcome only through the exposure, and "
                       "adjusting for it amplifies any confounding left")
    if after == "yes":
        if _get(a, "causes_outcome") == "yes":
            if direct:
                return Derived("mediator", True, False,
                               "on the path from the exposure to the outcome; a direct effect "
                               "holds it fixed")
            return Derived("mediator", kept, further and not kept,
                           "on the path from the exposure to the outcome: adjusting for it removes "
                           "part of the total effect")
        if confounds:
            # Changed by the exposure and a common cause of a mediator and the outcome: a regression
            # can hold it fixed only as one of the mediators (the controlled direct effect fixing
            # both), never adjust it as a confounder.
            return Derived("mediator", True, False,
                           "changed by the exposure and a common cause of a mediator and the "
                           "outcome: a direct effect holds it fixed with the mediators")
        return Derived("collider", kept, further and not kept,
                       "changed by the exposure without causing the outcome: adjusting for it can "
                       "open a path that is not causal")
    if after == "unknown":
        if direct and _get(a, "causes_outcome") != "no":
            return Derived("timing_unknown", True, False,
                           "the exposure may have changed it: a mediator a direct effect holds "
                           "fixed, or a confounder; either way it is adjusted")
        return Derived("timing_unknown", kept, not kept,
                       "the exposure may have changed it, so the estimate is declared without it "
                       "and, beside, with it")
    if _get(a, "proxy"):
        return Derived("proxy", True, False, "a proxy for an unmeasured cause of both")
    ce, co = _get(a, "causes_exposure"), _get(a, "causes_outcome")
    if ce == "no" and co == "no":
        if confounds:
            return Derived("mediator_confounder", True, False,
                           "a possible common cause of a mediator and the outcome: a direct effect "
                           "adjusts for it")
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


def direct_questions(answers: Any, effect: str) -> list[str]:
    """What a direct effect still asks of one covariate's answers (MODELING_SEQUENCE §1 step 3 and
    §2): of a covariate the direct effect holds fixed (a mediator, or one of unknown timing that is
    adjusted either way), whether the exposure's effect could differ with its level
    (``interacts``); of one the criterion leaves out (a cause of neither, a consequence of the
    exposure, an instrument), whether it is a common cause of a mediator and the outcome
    (``confounds_mediator``). Nothing under a total effect, and nothing whose answer would change
    nothing (a confounder is adjusted either way)."""
    if effect != "direct":
        return []
    found = derive(answers, effect)
    if found.adjusted and found.role in ("mediator", "timing_unknown"):
        return [] if _get(answers, "interacts") is not None else ["interacts"]
    if found.role in ("not_a_cause", "collider", "instrument"):
        return [] if _get(answers, "confounds_mediator") is not None else ["confounds_mediator"]
    return []


def unanswered(state: Any) -> list[str]:
    answers = current_answers(state)
    spec = current_estimand(state)
    effect = str(_get(spec, "effect") or "total") if spec is not None else "total"
    return [c for c in asked_covariates(state)
            if c not in answers or direct_questions(answers[c], effect)]


def derived_roles(state: Any) -> dict[str, Derived]:
    spec = current_estimand(state)
    if spec is None:
        return {}
    answers = current_answers(state)
    effect = str(_get(spec, "effect") or "total")
    return {c: derive(answers[c], effect) for c in covariates(state) if c in answers}


def mediators(state: Any) -> list[str]:
    """Under a direct effect, the covariates it holds fixed: the mediators by the answers."""
    spec = current_estimand(state)
    if spec is None or _get(spec, "effect") != "direct":
        return []
    return [c for c, d in derived_roles(state).items() if d.role == "mediator" and d.adjusted]


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

# The guesses are the packs' (``turbotab/core/covariate_guesses.py``; NUTRITION_PACK §08, "The
# adjustment card's guesses"): one per class, read under the declared exposure–outcome pairing.
from turbotab.core.covariate_guesses import CLASS_ORDER, GUESSES  # noqa: E402

# What a block's guess says, by the role its answers derive (a total effect's wording).
GUESS_WORDS = {
    "confounder": "a confounder: adjusted for in the primary",
    "exposure_cause": "a cause of the exposure: adjusted for in the primary",
    "precision": "a cause of the outcome only: adjusted for in the primary",
    "timing_unknown": "possible mediator, or measured after the exposure: the estimate is declared "
                      "without it and, beside, with it",
    "mediator": "a mediator: left out of a total effect",
    "collider": "another measure of the outcome, or a consequence of the exposure: left out",
    "not_a_cause": "a cause of neither: left out",
    "instrument": "an instrument: left out",
    "proxy": "a proxy for an unmeasured common cause: adjusted for",
    "mediator_confounder": "a common cause of a mediator and the outcome: adjusted for",
}


def guess_of(column: str, state: Any = None) -> str | None:
    """Which of the packs' guesses a covariate takes (a ``GUESSES`` key), read from its name under
    the state's exposure–outcome pairing (a proposal the user confirms, never a settled reading:
    BLUEPRINT §14); None where the packs have no guess."""
    from turbotab.core.covariate_guesses import guess

    found = guess(column, state)
    return found.key if found is not None else None


def _listed_labels(keys: Sequence[str]) -> str:
    labels = [GUESSES[k]["label"] for k in keys]
    words = [labels[0], *(x[:1].lower() + x[1:] for x in labels[1:])]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def guess_blocks(state: Any, columns: Sequence[str], exposure: str,
                 effect: str = "total") -> list[dict[str, Any]]:
    """The waiting covariates the packs have a guess for, in blocks of the same guess (the same
    answers), each answered with one tap (its ``decision``, which lists exactly its columns), every
    member with its guess's reason and source; then the unguessed, asked plainly. BLUEPRINT §14.2:
    "Lead with the guess, confirm in one tap, group by consequence"."""
    from turbotab.core.covariate_guesses import guess, pairing_of

    pairing = pairing_of(state)
    found = {c: guess(c, state, pairing) for c in columns}
    blocks: dict[tuple[tuple[str, str], ...], list[Any]] = {}
    for c in columns:
        g = found[c]
        if g is not None:
            blocks.setdefault(tuple(sorted(g.answers.items())), []).append(g)
    rank = {k: i for i, k in enumerate(CLASS_ORDER)}
    ordered = sorted(blocks.values(), key=lambda gs: min(rank.get(g.key, 99) for g in gs))
    groups = []
    for members in ordered:
        members = sorted(members, key=lambda g: (rank.get(g.key, 99), list(columns).index(g.column)))
        keys = list(dict.fromkeys(g.key for g in members))
        answers = dict(members[0].answers)
        derived = derive(answers, effect)
        cols = [g.column for g in members]
        groups.append({
            "key": keys[0], "label": _listed_labels(keys), "classes": keys, "columns": cols,
            "guess": answers, "guess_words": GUESS_WORDS.get(derived.role),
            "reason": " ".join(f"{GUESSES[k]['label']}: {GUESSES[k]['reason']} "
                               f"({GUESSES[k]['source']})." for k in keys),
            "sources": list(dict.fromkeys(g.source for g in members)),
            "members": [g.as_dict() for g in members],
            "derived": derived.role, "derived_words": ROLE_WORDS[derived.role],
            "decision": {"kind": "set_adjustment", "exposure": exposure,
                         "answers": {c: dict(answers) for c in cols}}})
    unguessed = [c for c in columns if found[c] is None]
    if unguessed:
        groups.append({"key": "unguessed", "label": "No guess", "classes": [],
                       "columns": unguessed, "guess": None, "guess_words": None,
                       "reason": "The packs say nothing about these; each is asked.",
                       "sources": [], "members": [], "derived": None, "derived_words": None,
                       "decision": None})
    return groups


def adjustment_card(state: Any) -> dict[str, Any] | None:
    """What the adjustment question shows: the covariates in blocks that share the packs' guess,
    each confirmed with one tap (its ``decision``), each member's guess with its reason and source,
    the unguessed ones asked plainly, a multi-select answer over any of them (``bulk``: one
    ``set_adjustment`` settles exactly the columns it lists), and what the answers so far derive.
    None when the question does not apply yet."""
    spec = current_estimand(state)
    if spec is None or _get(state, "purpose") != "inference":
        return None
    exposure = exposure_key(spec)
    effect = str(_get(spec, "effect") or "total")
    answers = current_answers(state)
    waiting = [c for c in asked_covariates(state) if c not in answers]
    groups = guess_blocks(state, waiting, exposure, effect)
    derived = derived_roles(state)
    # A direct effect's own questions (MODELING_SEQUENCE §1 step 3 and §2), for each covariate
    # whose answers so far leave one of them open: one line each, answered with the rest of its
    # answers (``set_adjustment``).
    pending = {c: direct_questions(answers[c], effect) for c in asked_covariates(state)
               if c in answers and direct_questions(answers[c], effect)}
    measure = _get(spec, "measure")
    for g in groups:  # the relation the measure carries (MODELING_SEQUENCE §2), said where it bites
        g["estimand_note"] = (precision_note(measure, g["columns"])
                              if g.get("derived") == "precision" else None)
    precision = [c for c, d in derived.items() if d.role == "precision" and d.adjusted]
    return {
        "exposure": exposure, "effect": effect, "family": bool(_get(spec, "family")),
        "questions": {**QUESTIONS, **(DIRECT_QUESTIONS if effect == "direct" else {})},
        "groups": groups,
        "direct_questions": [{"column": c, "fields": fields,
                              "answers": answers[c].model_dump(exclude={"exposure"})
                              if hasattr(answers[c], "model_dump") else dict(answers[c])}
                             for c, fields in pending.items()],
        "mediators": mediators(state),
        "answered": {c: {"role": d.role, "words": ROLE_WORDS[d.role], "adjusted": d.adjusted,
                         "secondary": d.secondary, "why": d.why,
                         "estimand_note": (precision_note(measure, [c])
                                           if d.role == "precision" and d.adjusted else None)}
                     for c, d in derived.items()},
        "estimand_note": precision_note(measure, precision),
        "adjusted": [c for c, d in derived.items() if d.adjusted],
        "left_out": [c for c, d in derived.items() if not d.adjusted],
        "secondary": secondary_columns(state),
        # The multi-select answer: any of the waiting covariates, chosen together and answered with
        # one set of answers, in one ``set_adjustment`` that settles exactly the columns it lists.
        "bulk": {"columns": waiting, "fields": list(QUESTIONS),
                 "decision": {"kind": "set_adjustment", "exposure": exposure, "answers": {}}},
        "source": VANDERWEELE,
    }


# A direct effect's two further questions (MODELING_SEQUENCE §1 step 3: "A direct effect also asks
# for mediator–outcome confounders"; §2: "Exposure–mediator interaction routes to counterfactual
# mediation methods"), each asked only where its answer changes the model (``direct_questions``).
DIRECT_QUESTIONS = {
    "confounds_mediator": "Is it a common cause of a mediator and the outcome?",
    "interacts": "Could the exposure's effect differ with its level?",
}

# The five questions, as the card asks them (each one line).
QUESTIONS = {
    "causes_exposure": "Is it a cause of the exposure?",
    "causes_outcome": "Is it a cause of the outcome?",
    "after_exposure": "Could the exposure have changed it, or was it measured after?",
    "instrument": "Does it affect the outcome only through the exposure?",
    "proxy": "Does it stand in for an unmeasured cause of both?",
}


# ── an exposure family's multiplicity (MODELING_SEQUENCE §2: "An exposure family implies
# multiplicity control: BH q-values for feature-wise analyses; for a few … nutrient hypotheses, the
# number of tests stated. Every member is shown. This is not selection.") ────────────────────────

BENJAMINI_HOCHBERG = "Benjamini & Hochberg 1995, J R Stat Soc B 57:289"
ROTHMAN = "Rothman 1990, Epidemiology 1:43"
METABOLOMICS_PACK = "METABOLOMICS_PACK §08"
OMICS_LENSES = ("metabolomics", "genomics")
MULTIPLICITY_WORDS = {
    "fdr_bh": "Benjamini–Hochberg q-values across the family",
    "count_stated": "unadjusted p-values, with the number of tests stated",
    "none": "no multiplicity control, recorded as a limitation",
}
# MS7's names for the methods (``set_multiplicity``), in the estimand card's words.
_FROM_MS7 = {"bh": "fdr_bh", "stated_count": "count_stated", "none": "none"}


def omics_family(state: Any) -> bool:
    return bool(set(_get(state, "lens") or []) & set(OMICS_LENSES))


def family_multiplicity(state: Any) -> tuple[str, bool]:
    """The family's one multiplicity method and whether it was kept as recorded: the latest answer,
    from the estimand card or ``set_multiplicity`` (MS7), both of which write the one slot
    (``turbotab.core.methods.omics.multiplicity_policy``); Benjamini–Hochberg when unanswered."""
    from turbotab.core.methods.omics import multiplicity_policy

    policy = multiplicity_policy(state)
    return _FROM_MS7[policy["method"]], bool(policy["acknowledged"])


def unadjusted_blocked(state: Any, n: int) -> bool:
    """Whether a family's unadjusted p-values are blocked and recorded (MS7's rung): for an omics
    family at any size, or beyond a few declared hypotheses."""
    from turbotab.core.methods.omics import multiplicity_rung

    return multiplicity_rung("stated_count", n, omics_family(state)) == "block_and_record"


def multiplicity_question(state: Any, n: int) -> dict[str, Any]:
    """The family's multiplicity method, each option labeled customary and sound (BLUEPRINT north
    star 5), soundest first: Benjamini–Hochberg always; unadjusted p-values with the count stated
    are customary for a few declared nutrient hypotheses (Rothman 1990) and unsound for an omics
    family (``METABOLOMICS_PACK §08``: "per-feature testing with multiple-testing correction is
    expected, and its absence is a fatal flaw in review"), where they are blocked and recorded."""
    omics = omics_family(state)
    blocked = unadjusted_blocked(state, n)
    options = [
        {"key": "fdr_bh", "label": MULTIPLICITY_WORDS["fdr_bh"].capitalize(),
         "customary": {"field": "metabolomics and genomics",
                       "text": "per-feature tests with false-discovery control (q < 0.05) are the "
                               "field's standard",
                       "source": METABOLOMICS_PACK},
         "sound": {"purpose": "inference", "verdict": "sound",
                   "reason": f"controls the expected share of false discoveries among the {n:,} "
                             f"tests ({BENJAMINI_HOCHBERG}), and every member is still shown"}},
        {"key": "count_stated", "label": MULTIPLICITY_WORDS["count_stated"].capitalize(),
         "customary": {"field": "nutritional epidemiology",
                       "text": "a few declared nutrient hypotheses are reported without adjustment",
                       "source": ROTHMAN},
         "sound": ({"purpose": "inference", "verdict": "unsound",
                    "reason": f"{n:,} tests at p < 0.05 give about {0.05 * n:,.0f} false positives "
                              f"by chance alone; the omics field treats an unadjusted table as a "
                              f"fatal flaw ({METABOLOMICS_PACK})"} if omics else
                   {"purpose": "inference", "verdict": "unsound",
                    "reason": f"{n:,} tests at p < 0.05 give about {0.05 * n:,.0f} false positives "
                              f"by chance alone; the count stated is customary for a few declared "
                              f"hypotheses, not {n:,}"} if blocked else
                   {"purpose": "inference", "verdict": "conditional",
                    "reason": f"sound for a few hypotheses each declared on its own, read with all "
                              f"{n:,} tests in view (STROBE item 20: \"multiplicity of analyses\")"})},
    ]
    customary_first = "fdr_bh" if omics or blocked else "count_stated"
    tension = (None if omics or blocked else
               f"For a few declared nutrient hypotheses the field reports unadjusted p-values with "
               f"the count stated ({ROTHMAN}); Benjamini–Hochberg controls the false-discovery rate "
               f"across all {n:,} of them.")
    return {"question": "multiplicity", "options": options, "customary_first": customary_first,
            "tension": tension, "n_tests": n}


def multiplicity_statement(state: Any, spec: Any, n: int) -> str:
    """The sentence a family's table and methods carry: its method and the number of tests."""
    method, _ = family_multiplicity(state)
    if method == "none":
        return (f"{n:,} exposures were tested, each in turn, with no multiplicity control, recorded "
                f"as a limitation: at p < 0.05 about {0.05 * n:,.0f} would pass by chance alone; "
                f"every member is shown.")
    if method == "fdr_bh":
        return (f"{n:,} exposures were tested, each in turn; Benjamini–Hochberg q-values control "
                f"the false-discovery rate across all {n:,} ({BENJAMINI_HOCHBERG}), and every "
                f"member is shown, significant or not.")
    kept = (" It was kept for an omics family as recorded: the field expects false-discovery "
            "control." if omics_family(state) else
            " It was kept for this many tests as recorded." if unadjusted_blocked(state, n) else "")
    return (f"{n:,} exposures were tested, each in turn; p-values are not adjusted for "
            f"multiplicity, and all {n:,} tests are stated (customary for a few declared nutrient "
            f"hypotheses, {ROTHMAN}), with every member shown.{kept}")


def estimand_card(state: Any, task: str | None, prevalence: float | None = None) -> dict[str, Any] | None:
    """What the estimand question offers: the exposure candidates, the effects, the contrast for an
    energy-bearing exposure, and the measures ruling 9 labels (difference or ratio; conditional or
    marginal), ranked by the outcome's ``prevalence`` (the event's share) for a yes/no outcome."""
    if _get(state, "purpose") != "inference" or _get(state, "roles") is None:
        return None
    candidates = exposure_candidates(state)
    family = family_exposures(state)
    # The routing gate (p16): a predictor whose role rode along unconfirmed is not offered as the
    # exposure until it is confirmed (the step's ask card asks it), but it is named here, so a
    # client choosing among the offered exposures sees that one is waiting.
    from turbotab.core.readings import confirm_exits, unsettled

    target = _get(state, "target")
    recorded = _get(state, "roles") or {}
    waiting = [c for c in unsettled(state) if recorded.get(c) in PREDICTOR_ROLES and c != target]
    return {
        "exposures": [{"column": c, "energy_contrast": energy_contrast_applies(state, c)}
                      for c in candidates],
        "unconfirmed": [{"column": c, "role": recorded.get(c), "exits": confirm_exits(state, [c])}
                        for c in waiting],
        # Every exposure reported in turn, with its multiplicity method (MODELING_SEQUENCE §1 step
        # 2): offered with two or more exposures, as an omics table's features are.
        "family": ({"n": len(family), "energy_contrast": family_contrast_applies(state),
                    "measures": measures_offered(task, family=True),
                    "multiplicity": multiplicity_question(state, len(family)),
                    "consequence": (f"Each of the {len(family):,} exposures is reported in turn, "
                                    f"adjusted for the covariates but not for the other exposures, "
                                    f"with its multiplicity method; every member is shown (the "
                                    f"feature-wise family).")}
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
        "measures": measures_offered(task, prevalence=prevalence),
        "prevalence": prevalence,
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
        method = MULTIPLICITY_WORDS[family_multiplicity(state)[0]]
        text = (f"The {effect} effect of each of the {len(family):,} exposures "
                f"({_listing(family, limit=3)}) on {_tick(target)}, one at a time"
                + (f" ({what})" if what else "")
                + f", {scale}, every member shown, with {method} ({len(family):,} tests)")
    else:
        exposure = _get(spec, "exposure")
        # FORM: the unit on the exposure's final scale (``exposure_form.estimand_unit``), a stale
        # form's never kept; a consumers-only domain names its population.
        from turbotab.core.methods.exposure_form import domain_columns, estimand_unit

        among = (f" among consumers of {_tick(exposure)}"
                 if exposure in domain_columns(state) else "")
        text = (f"The {effect} effect of {_tick(exposure)} on {_tick(target)}{among}"
                + (f" ({what})" if what else "")
                + f", as a {MEASURE_WORDS.get(measure, measure)} per "
                  f"{estimand_unit(state, str(exposure))}")
    derived = derived_roles(state)
    held = mediators(state)
    if effect == "direct" and held:
        # The controlled direct effect (Valeri & VanderWeele 2013): the mediators held fixed,
        # named as such, never listed as covariates "conditioned on".
        text = text.replace("The direct effect of", "The controlled direct effect of", 1)
        text += (f", with {'the mediator' if len(held) == 1 else 'the mediators'} "
                 f"{_listing(held, limit=6)} held fixed")
    adjusted = [c for c, d in derived.items() if d.adjusted and c not in held]
    fe = fixed_effects_column(state)
    parts = ([_listing(adjusted, limit=6)] if adjusted else []) + (
        [f"an intercept for each {_tick(fe)} (fixed effects)"] if fe else [])
    if measure in MARGINAL and not _get(spec, "family"):
        text += (f", standardized over the analyzed rows' {' and '.join(parts)}" if parts
                 else ", with no covariate to standardize over")
        text += (" by g-computation from the logistic model, whose conditional odds ratio is "
                 "shown beside it")
    else:
        text += (f", conditional on {' and '.join(parts)}" if parts
                 else ", with no covariate adjusted for")
    if effect == "direct" and held:
        answers = current_answers(state)
        open_interaction = [c for c in held if _get(answers.get(c), "interacts") in ("yes", "unknown")]
        if open_interaction:
            text += (f"; an exposure–mediator interaction with {_listing(open_interaction)} was "
                     f"not ruled out, so this is the direct effect at "
                     f"{'its' if len(open_interaction) == 1 else 'their'} reference level only "
                     f"(recorded)")
        else:
            text += "; it assumes no exposure–mediator interaction, as answered"
        text += f", and no unmeasured common cause of a mediator and the outcome ({VALERI})"
    out = [c for c, d in derived.items() if not d.adjusted and d.role in ("mediator", "collider")]
    if out:
        text += f"; {_listing(out)} left out as {'a consequence' if len(out) == 1 else 'consequences'} of the exposure"
    if measure in NON_COLLAPSIBLE:
        text += "; a conditional ratio, which changes with the covariates even without confounding"
        precision = [c for c, d in derived.items() if d.role == "precision" and d.adjusted]
        if precision:
            text += (f": adjusting for {_listing(precision)}, "
                     f"{'a cause' if len(precision) == 1 else 'causes'} of the outcome only, "
                     f"changes the conditional estimand, not only its precision")
    second = secondary_columns(state)
    if second:
        text += f". Declared beside it: further adjusted for {_listing(second)}"
    return text + f" (the adjustment set by the disjunctive cause criterion, {VANDERWEELE})."


def primary_features(fitted_features: Iterable[str], exposure: str,
                     predictors: Iterable[str] = ()) -> list[str]:
    """The model-matrix columns that carry the exposure's effect: its own column, the column an
    energy model made of it (``<exposure>_adj``, ``<exposure>_per_<E>``, ``kcal_from_<exposure>``),
    and the spline or quintile terms the form made of either (a spline's ``x'``, ``x''`` as ``rms``
    labels them, ``methods.exposure_form.spline_names``; a quintile's ``x_Q2``). A name that begins
    with the exposure's but belongs to a longer predictor of ``predictors`` (``fat_sat`` beside the
    exposure ``fat``) is that predictor's, not the exposure's."""
    exposure = str(exposure)
    longer = [str(p) for p in predictors if str(p) != exposure and str(p).startswith(f"{exposure}_")]
    out = []
    for f in fitted_features:
        name = str(f)
        base = name[len("kcal_from_"):] if name.startswith("kcal_from_") else name
        if not (base.rstrip("'") == exposure or base.startswith(f"{exposure}_")
                or base.startswith(f"{exposure}[")):
            continue
        if any(base == p or base.startswith(f"{p}_") for p in longer):
            continue
        out.append(name)
    return out


# ── the declared model sequence (MODELING_SEQUENCE §1 row 11) ────────────────
# "The declared object is a crude model, a declared adjustment sequence (Model 1: age, sex and
# energy; Model 2: plus confounders; optional Model 3: plus possible mediators, labeled) and the
# primary model, shown for the exposure only." The crude model is always shown (STROBE item 16a:
# "Give unadjusted estimates and, if applicable, confounder-adjusted estimates"); Model 2 is the
# primary; Model 3 adds the columns the adjustment answers set beside it. Model 1's columns are the
# user's: a name never decides which column is age or sex (BLUEPRINT §14), so the pack's guess
# leads and one tap declares it.

_MODEL_ONE = {"age", "sex", "gender"}
_NHANES_MODEL_ONE = {"RIDAGEYR", "RIAGENDR"}


def energy_terms_columns(state: Any) -> list[str]:
    """The total-energy columns among the settled predictors (the energy role)."""
    return [c for c, r in predictor_roles(state).items() if r == "energy"]


def model_one_allowed(state: Any) -> list[str]:
    """The columns Model 1 may hold: the primary's adjusted covariates and total energy."""
    derived = derived_roles(state)
    return [c for c, d in derived.items() if d.adjusted] + [
        c for c in energy_terms_columns(state) if c not in derived]


def model_one_guess(state: Any) -> list[str]:
    """The pack's guess at Model 1 (NUTRITION_PACK §08: "Model 1 age, sex, energy"): the allowed
    columns whose names read as age or sex, and total energy. A proposal the user confirms."""
    from turbotab.core.recognizers import tokens

    allowed = model_one_allowed(state)
    energy = set(energy_terms_columns(state))
    return [c for c in allowed if c in energy or str(c).upper() in _NHANES_MODEL_ONE
            or set(tokens(c)) & _MODEL_ONE]


def current_model_sequence(state: Any) -> Any:
    """The declared sequence while it is for the current exposure and every Model 1 column is still
    in the primary set (a change of exposure or adjustment set re-asks it: MODELING_SEQUENCE §2,
    "invalidates"); None otherwise."""
    spec = _get(state, "model_sequence")
    current = current_estimand(state)
    if spec is None or current is None or _get(spec, "exposure") != exposure_key(current):
        return None
    allowed = set(model_one_allowed(state))
    return spec if all(c in allowed for c in _get(spec, "model_1") or []) else None


def model_sequence_card(state: Any) -> dict[str, Any] | None:
    """Model 1's declaration as the effects stage offers it: the declared columns, or the pack's
    guess with the one-tap decision that declares it."""
    spec = current_estimand(state)
    if spec is None or _get(state, "purpose") != "inference":
        return None
    declared = current_model_sequence(state)
    guess = model_one_guess(state)
    return {"declared": list(_get(declared, "model_1") or []) if declared is not None else None,
            "guess": guess, "allowed": model_one_allowed(state),
            "decision": {"kind": "set_model_sequence", "exposure": exposure_key(spec),
                         "model_1": guess},
            "reason": ("The field's Model 1 adjusts for age, sex and energy (NUTRITION_PACK §08); "
                       "say which columns those are. Model 2 is the primary, Model 3 adds the "
                       "possible mediators your answers set beside it.")}


# ── a failed diagnostic's recorded response (MODELING_SEQUENCE §1 row 11) ────

DIAGNOSTIC_ACTIONS = {
    "proportional_hazards": ("period_hazard_ratios", "keep_labeled"),
    "influence": ("without_influential", "keep_labeled"),
}
ACTION_WORDS = {
    "period_hazard_ratios": "the exposure's hazard ratio before and after the median event time, "
                            "beside the average over follow-up",
    "without_influential": "the primary model refit without the influential rows, beside it",
    "keep_labeled": "the estimate kept, labeled with the failed check",
}


def current_responses(state: Any) -> dict[str, str]:
    """Each check's recorded action, while it is for the current exposure."""
    spec = current_estimand(state)
    if spec is None:
        return {}
    key = exposure_key(spec)
    return {check: str(_get(r, "action"))
            for check, r in (_get(state, "diagnostic_responses") or {}).items()
            if _get(r, "exposure") == key}


# ── what the server withholds while the plan is unanswered ───────────────────

# The questions whose answer an estimate rests on, and the purposes they hold estimates under.
HOLDS: dict[str, tuple[str, ...]] = {
    "follow_up": ("prediction", "inference"),
    "clusters": ("inference",),
    "estimand": ("inference",),
    "adjustment": ("inference",),
    # V2 causal row (turbotab/core/time_varying.py): an exposure that changes over time.
    "time_varying": ("inference",),
    # FORM (MODELING_SEQUENCE §1 row 5): the forms are declared before estimates; one a transform
    # left stale is asked again, and nothing is shown on a form that was not declared.
    "form": ("inference",),
}
# ``scales`` (MS8): a declared scale's corrected coefficient and the uncorrected one beside it;
# ``effects`` (ESTIMAND): Table 2, the marginal risks, the diagnostics and the sensitivity;
# ``causal``: the causal lane's estimate (``turbotab/core/stages/causal.py``);
# ``time_varying`` (V2 causal row): the g-methods' estimates (their diagnostics lock nothing);
# ``explain`` (EXPLAIN): explanations of the outcome models are not estimates, but they show what
# each model learned from the outcome, so under inference they wait and lock as estimates do.
# ``evaluation`` (EXPLORE): under inference only the declared selection sensitivity analysis (its
# pooled Wald tests are estimates); no cross-validated score is shown.
ESTIMATE_STAGES = ("fit", "substitution", "sensitivity", "calibration", "secondary", "scales",
                   "effects", "causal", "time_varying", "explain",
                   # FORM: each declared modifier's effects, RERI and ratio of ratios
                   "modification", "evaluation")


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
            "time_varying": "an exposure that changes over time needs its estimation lane, and "
                            "inverse-probability weights their truncation (the g-formula its "
                            "simulation's size), declared after the diagnostics are read, before "
                            "any estimate",
            "form": "the form of the exposure and of each continuous confounder is declared on "
                    "its final scale before any estimate",
        }[str(key)]
        return {"question": key,
                "reason": f"No estimate is shown until {name} is answered: {why}.",
                "exits": [{"label": f"Answer {name}", "decision": None}],
                "purpose": purpose}
    return None


# Under inference a cross-validated score describes how well the outcome model fits, not the
# reported estimate (MODELING_SEQUENCE §1 row 11: under inference the declared object is "not a
# comparison"; §0 ruling 13). While the plan is open they wait with the estimates: an R² read
# before the adjustment set is answered is one more fork in the path (Gelman & Loken 2013).
SCORES_WITHHELD = ("Its cross-validated scores (R², RMSE, MAE and the rest) wait too: under "
                   "inference they describe the outcome model's fit, not the reported estimate.")
# The model-level fields that hold an outcome-model fit statistic, and the fit-level ones.
MODEL_SCORES = ("holdout", "versus_baseline", "calibration", "holdout_detail", "optimism",
                "internal_external", "compared_on", "performance", "nested_cv",
                "calibration_levels", "calibration_horizon", "calibration_note")
FIT_SCORES = ("selection", "precision", "comparison", "comparisons_note", "result",
              "nested_offer", "at_opening")


def without_scores(model: dict[str, Any]) -> dict[str, Any]:
    """One served model with every outcome-model fit statistic removed (the cross-validated
    scores, the baseline's, the held-out ones, calibration, optimism and the comparison
    substrate); what the model is and why it waits stay."""
    m = dict(model)
    m["cv"] = {}
    for key in MODEL_SCORES:
        if key in m:
            m[key] = None
    if isinstance(m.get("baseline"), dict):
        m["baseline"] = {**m["baseline"], "value": None}
    return m


def withhold(stage: str, artifact: Any, gate: Mapping[str, Any]) -> Any:
    """``artifact`` with every estimate removed and the reason first: the coefficient tables, their
    tests and intervals; curves; refits. Under inference the outcome model's fit statistics wait
    too (:data:`SCORES_WITHHELD`); under prediction (a follow-up question unanswered) they stay,
    since there they are the result."""
    if not isinstance(artifact, dict):
        return artifact
    out = dict(artifact)
    reason = str(gate["reason"])
    out["withheld"] = reason
    inference = gate.get("purpose") == "inference"
    if stage == "fit":
        models = []
        for m in out.get("models") or []:
            m = without_scores(m) if inference else dict(m)
            info = m.get("inference") or {}
            if info.get("refused") and not m.get("coefficients"):
                # Already refused with its own reason and exits (an unanswered survey question, a
                # held missing-values answer): nothing is estimated, so its refusal speaks first.
                models.append(m)
                continue
            m["coefficients"] = None
            m["inference"] = None
            m["exposure_tests"] = []
            m["concerns"] = [reason, *([SCORES_WITHHELD] if inference else []),
                             *(m.get("concerns") or [])]
            models.append(m)
        out["models"] = models
        if inference:
            for key in FIT_SCORES:
                if key in out:
                    out[key] = None
            out["comparisons"] = []
        return out
    if stage == "sensitivity":
        out["families"] = []
        out["changes"] = {}
        return out
    if stage == "scales":
        # Each scale's reliability stays (it describes the items, not an effect); its corrected
        # coefficient and the uncorrected one beside it wait for the plan.
        scales = []
        for sc in out.get("scales") or []:
            sc = dict(sc)
            if sc.get("correction") is not None:
                sc["correction"] = None
                sc["not_corrected"] = reason
            scales.append(sc)
        out["scales"] = scales
        return out
    for key in ("curves", "families", "models", "estimates", "rows", "fits",
                "modifications"):  # FORM: the declared modifiers' estimates
        if key in out:
            out[key] = [] if isinstance(out[key], list) else None
    return out


def annotate_fit(artifact: Any, state: Any) -> Any:
    """The served fit under a declared estimand: the caption worded from it, and the Table 2
    display (Westreich & Greenland 2013): each model's ``coefficients`` hold the exposure's rows
    only (every member of an exposure family), and every other row moves to ``adjustment_terms``,
    titled "adjustment terms, not effect estimates" (``models/effects.py``)."""
    from turbotab.core.models.effects import APPENDIX_TITLE, split_rows

    if not isinstance(artifact, dict) or _get(state, "purpose") != "inference":
        return artifact
    spec = current_estimand(state)
    if spec is None:
        return artifact
    exposures = exposures_of(state, spec)
    predictors = list(predictor_roles(state))
    out = dict(artifact)
    features: set[str] = set()
    models = []
    for m in out.get("models") or []:
        m = dict(m)
        rows = m.get("coefficients")
        if rows:
            mine = {f for e in exposures
                    for f in primary_features([r.get("feature") for r in rows], e, predictors)}
            features |= mine
            shown, appendix = split_rows(rows, mine)
            m["coefficients"], m["adjustment_terms"] = shown, appendix
        models.append(m)
    out["models"] = models
    family = bool(_get(spec, "family"))
    out["estimand"] = {
        "exposure": exposure_key(spec), "effect": _get(spec, "effect"),
        "measure": _get(spec, "measure"),
        "contrast": _get(spec, "contrast"), "caption": caption(state, out.get("task")),
        "adjusted": [c for c, d in derived_roles(state).items() if d.adjusted],
        "left_out": {c: d.role for c, d in derived_roles(state).items() if not d.adjusted},
        "secondary": secondary_columns(state),
        "features": sorted(features),
        "appendix": APPENDIX_TITLE,
        "multiplicity": multiplicity_statement(state, spec, len(exposures)) if family else None,
    }
    from turbotab.core.time_varying import fit_note  # V2 causal row

    out["estimand"]["time_varying"] = fit_note(state)
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


def _follow_up_is_no_predictor(decision: Any, ctx: Any) -> None:
    """A column the follow-up answer names is the time half of a time-to-event outcome: a roles
    answer that makes it a predictor is refused, with the same answer giving it the role time (the
    routing gate: ``PERMTH_INT`` proposed an exposure beside its own outcome)."""
    from turbotab.core.decisions import SetRoles

    state = _state(ctx)
    spec = _get(state, "follow_up")
    if spec is None or _get(state, "task") != "time_to_event":
        return
    named = [c for c in (_get(spec, "time_column"), _get(spec, "entry_column")) if c]
    inside = [c for c in named if decision.roles.get(c) in PREDICTOR_ROLES]
    if not inside:
        return
    fixed = {**decision.roles, **{c: "time" for c in inside}}
    raise _refusal(
        "follow_up_as_predictor",
        f"{_listing(inside)} {'is' if len(inside) == 1 else 'are'} named as "
        f"{_tick(_get(state, 'target'))}'s follow-up (the follow-up question), part of the outcome "
        f"itself, so {'it' if len(inside) == 1 else 'they'} cannot also predict it.",
        [{"label": f"Give {_listing(inside)} the role time",
          "decision": SetRoles(**{**decision.model_dump(exclude={"kind"}), "roles": fixed})}])


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


def _no_grouping_names_what_it_denies(decision: Any, ctx: Any) -> Any:
    """"Nothing groups them" is said over the columns that read as a grouping now: each is denied
    by it, as its own ``cluster`` confirmation "no" would (``none_of``, the server's, never the
    client's); a named grouping denies nothing."""
    if decision.column is not None:
        return decision.model_copy(update={"none_of": []})
    state = _state(ctx)
    found = ([c["column"] for c in grouping_candidates(state, _artifact(ctx, "roles"))]
             if state is not None else [])
    return decision.model_copy(update={"none_of": list(found)})


def _no_grouping_is_recorded(decision: Any, ctx: Any) -> None:
    """Under inference, "nothing groups them" over a column that reads as a grouping is block and
    record (audit RO-08: "adjust and cluster, or record why not")."""
    from turbotab.core.decisions import SetClusters

    state = _state(ctx)
    if decision.column is not None or decision.acknowledged or _get(state, "purpose") != "inference":
        return
    # The named groupings, then the columns whose values read as a grouping (the guess "yes"); a
    # category with many labels (the guess "no") is asked, and "nothing" over it needs no record.
    candidates = [c["column"] for c in grouping_candidates(state, _artifact(ctx, "roles"))
                  if c["guess"] == "yes"]
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
    if task and decision.measure in fitted_measures(task, decision.family):
        return
    if decision.measure in NOT_FITTED and not (task in MARGINAL_TASKS and not decision.family):
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


def _family_declares_its_multiplicity(decision: Any, ctx: Any) -> Any:
    """An exposure family declared without a multiplicity method is declared with
    Benjamini–Hochberg (the sound default; MODELING_SEQUENCE §2), so the record says which."""
    if decision.family and decision.multiplicity is None:
        return decision.model_copy(update={"multiplicity": "fdr_bh"})
    return decision


def _omics_family_without_fdr_is_recorded(decision: Any, ctx: Any) -> None:
    """Block and record (MODELING_SEQUENCE review: "A per-feature table without it is
    block-and-record"): an omics family, or a family beyond a few declared hypotheses (MS7's rung),
    reported by unadjusted p-values waits for its attestation; the exit is Benjamini–Hochberg."""
    state = _state(ctx)
    n = len(family_exposures(state)) if state is not None else 0
    if (not decision.family or decision.multiplicity != "count_stated"
            or decision.multiplicity_acknowledged or state is None
            or not unadjusted_blocked(state, n)):
        return
    why = (f"per-feature testing without multiple-testing correction is \"a fatal flaw in review\" "
           f"({METABOLOMICS_PACK})" if omics_family(state) else
           f"the number of tests stated is customary for a few declared hypotheses ({ROTHMAN}), "
           f"not {n:,}")
    raise _refusal(
        "family_without_fdr",
        f"{'An omics' if omics_family(state) else 'A'} family of {n:,} exposures reported by "
        f"unadjusted p-values gives about {0.05 * n:,.0f} false positives at p < 0.05 by chance "
        f"alone; {why}.",
        [{"label": "Benjamini–Hochberg q-values across the family",
          "decision": decision.model_copy(update={"multiplicity": "fdr_bh"})},
         {"label": "Keep unadjusted p-values, with the number of tests stated; record that",
          "decision": decision.model_copy(update={"multiplicity_acknowledged": True})}])


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


def _a_direct_effect_asks_its_questions(decision: Any, ctx: Any) -> None:
    """A direct effect (MODELING_SEQUENCE §1 step 3 and §2). Valeri & VanderWeele (2013, *Psychol
    Methods* 18:137): "In order to ensure identifiability of controlled direct effect, two
    assumptions are needed: namely those of (i) no unmeasured confounding of the treatment-outcome
    relationship and (ii) no unmeasured confounding of mediator-outcome relationship"; and with an
    exposure–mediator interaction the controlled direct effect is (θ₁ + θ₃m)(a − a*), one number
    per level m of the mediator.

    * A known instrument is no common cause of a mediator and the outcome (it reaches the outcome
      only through the exposure): the two answers are refused together.
    * A possible exposure–mediator interaction is block and record: the coefficient is then the
      direct effect at the mediator's reference level only, and the other effects need
      counterfactual mediation methods TurboTab does not fit. The exits: the total effect, "no
      interaction", or the direct effect at the reference level, recorded.
    * A direct effect needs a mediator to hold fixed: once every covariate is answered and none is
      one, the answers are refused with the total effect as the exit."""
    from turbotab.core.decisions import SetEstimand

    state = _state(ctx)
    spec = current_estimand(state) if state is not None else None
    if spec is None or _get(spec, "effect") != "direct":
        return
    for column, a in decision.answers.items():
        if a.instrument and a.confounds_mediator == "yes":
            raise _refusal(
                "instrument_confounds_mediator",
                f"{_tick(column)} is answered an instrument and a common cause of a mediator and the "
                f"outcome; an instrument reaches the outcome only through the exposure.",
                [{"label": f"{_tick(column)} is not an instrument", "decision": decision.model_copy(
                    update={"answers": {**decision.answers,
                                        column: a.model_copy(update={"instrument": False})}})},
                 {"label": f"{_tick(column)} is no common cause of a mediator and the outcome",
                  "decision": decision.model_copy(update={"answers": {
                      **decision.answers, column: a.model_copy(update={"confounds_mediator": "no"})}})}])
    for column, a in decision.answers.items():
        d = derive(a, "direct")
        if not (d.adjusted and d.role in ("mediator", "timing_unknown")):
            continue
        if a.interacts not in ("yes", "unknown") or a.interaction_attested:
            continue
        maybe = "could" if a.interacts == "unknown" else "does"
        raise _refusal(
            "exposure_mediator_interaction",
            f"The exposure's effect {maybe} differ with {_tick(column)}'s level by your answer: the "
            f"exposure's coefficient is then the direct effect at {_tick(column)}'s reference level "
            f"only, and the direct effect at other levels, or a natural direct effect, needs "
            f"counterfactual mediation methods, which TurboTab does not fit ({VALERI}: the "
            f"controlled direct effect is (θ₁ + θ₃m)(a − a*), so it changes with the mediator's "
            f"level m).",
            [{"label": "Report the total effect instead",
              "decision": SetEstimand(**{**spec.model_dump(), "effect": "total"})},
             {"label": f"No exposure–mediator interaction with {_tick(column)}",
              "decision": decision.model_copy(update={"answers": {
                  **decision.answers, column: a.model_copy(update={"interacts": "no"})}})},
             {"label": f"Keep the direct effect at {_tick(column)}'s reference level; record the "
                       f"interaction as a limitation",
              "decision": decision.model_copy(update={"answers": {
                  **decision.answers, column: a.model_copy(update={"interaction_attested": True})}})}])
    merged = {**current_answers(state), **decision.answers}
    asked = asked_covariates(state)
    if any(c not in merged or direct_questions(merged[c], "direct") for c in asked):
        return  # not every covariate is answered yet: a mediator may still come
    held = [c for c in covariates(state) if c in merged
            and derive(merged[c], "direct").role == "mediator"]
    if held:
        return
    raise _refusal(
        "no_mediator",
        "A direct effect holds the mediators fixed, and by your answers no covariate is one (a "
        "covariate the exposure could change that causes the outcome): with nothing held fixed it "
        "is the total effect.",
        [{"label": "Report the total effect", "decision": SetEstimand(**{**spec.model_dump(),
                                                                          "effect": "total"})},
         {"label": "Answer again: name the mediator among the covariates", "decision": None}])


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
    # Each exit is a whole answer the user can take: an answer that named no energy column or no
    # nutrients ("none", say) takes the settled ones the energy question pre-fills.
    base = decision.model_dump(exclude={"kind"})
    roles = predictor_roles(state)
    if not base.get("energy_column"):
        base["energy_column"] = next((c for c, r in roles.items() if r == "energy"), None)
    if not base.get("nutrients"):
        base["nutrients"] = [c for c, r in roles.items() if r == "exposure" and energy_bearing(c)]
    raise _refusal(
        "contrast_mismatch",
        f"The estimand is {'a substitution' if contrast == 'substitution' else 'an addition'} of "
        f"{whose} calories, and the {label[0].lower() + label[1:]} "
        f"estimates {'no substitution' if contrast == 'substitution' else 'a substitution, not an addition'} "
        f"(Tomova et al. 2022).",
        [*({"label": METHOD_TABLE[m]["label"],
            "decision": SetEnergyAdjustment(**{**base, "method": m})}
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


# set_model_sequence and respond_diagnostic (MODELING_SEQUENCE §1 row 11)


def _sequence_follows_the_plan(decision: Any, ctx: Any) -> None:
    from turbotab.core.voice import question_name

    state = _state(ctx)
    if _get(state, "purpose") == "prediction":
        raise _not_inference("a model sequence")
    if state is None:
        return
    spec = current_estimand(state)
    if spec is None or adjustment_answer(state) is None and asked_covariates(state):
        raise _refusal("no_plan",
                       f"Declare the exposure and answer {question_name('adjustment')} first: "
                       f"Model 1 is a part of the primary model's adjustment set.",
                       [{"label": f"Answer {question_name('estimand')} first", "decision": None}])
    exposure = exposure_key(spec)
    if decision.exposure != exposure:
        raise _refusal("other_exposure",
                       f"The model sequence is declared for the exposure, {_tick(exposure)}, not "
                       f"{_tick(decision.exposure)}.",
                       [{"label": f"Declare it for {_tick(exposure)}",
                         "decision": decision.model_copy(update={"exposure": exposure})}])
    allowed = model_one_allowed(state)
    strangers = [c for c in decision.model_1 if c not in allowed]
    if strangers:
        raise _refusal(
            "not_in_the_primary",
            f"{_listing(strangers)} {'is' if len(strangers) == 1 else 'are'} not adjusted for in "
            f"the primary model; Model 1 holds some of its adjustment set (the field's age, sex "
            f"and energy), so each model adds to the one before.",
            [{"label": "Model 1: " + (_listing([c for c in decision.model_1 if c in allowed])
                                      or "none"),
              "decision": decision.model_copy(update={"model_1": [
                  c for c in decision.model_1 if c in allowed]})}])


def _response_fits_the_check(decision: Any, ctx: Any) -> None:
    state = _state(ctx)
    if _get(state, "purpose") == "prediction":
        raise _not_inference("a diagnostic's response")
    if decision.action not in DIAGNOSTIC_ACTIONS[decision.check]:
        raise _refusal(
            "action_not_for_check",
            f"{ACTION_WORDS[decision.action].capitalize()} is not a response to the "
            f"{decision.check.replace('_', ' ')} check.",
            [{"label": ACTION_WORDS[a].capitalize(),
              "decision": decision.model_copy(update={"action": a})}
             for a in DIAGNOSTIC_ACTIONS[decision.check]])
    if state is None:
        return
    spec = current_estimand(state)
    if spec is None:
        raise _refusal("no_estimand", "The checks are of the primary model, which waits for the "
                                      "exposure and its effect to be declared.",
                       [{"label": "Declare the exposure and its effect first", "decision": None}])
    if decision.exposure != exposure_key(spec):
        raise _refusal("other_exposure",
                       f"The checks are of the primary model for {_tick(exposure_key(spec))}.",
                       [{"label": f"Respond for {_tick(exposure_key(spec))}",
                         "decision": decision.model_copy(update={"exposure": exposure_key(spec)})}])
    # The record says the check failed, so it must have: read as the effects stage reported it.
    from turbotab.core.decisions import _ctx

    shown = _artifact(ctx, "effects")
    if shown is None:
        if callable(_ctx(ctx, "artifact")):
            raise _refusal("check_not_shown",
                           "The checks are reported with the effects, which are not computed for "
                           "this plan yet; respond once they are shown.",
                           [{"label": "Wait for the effects", "decision": None}])
        return  # no reader in this context (a direct validation): nothing to read
    found = [d for f in shown.get("families") or [] for d in f.get("diagnostics") or []
             if d.get("check") == decision.check]
    if not any(d.get("status") == "failed" for d in found):
        status = found[0].get("status") if found else "not reported"
        raise _refusal("check_not_failed",
                       f"The {decision.check.replace('_', ' ')} check of the primary model is "
                       f"{status.replace('_', ' ')}, so there is no failure to respond to.",
                       [{"label": "Keep the analysis as it is", "decision": None}])


def _register() -> None:
    from turbotab.core.decisions import register_completion, register_validator

    register_validator("set_roles", _follow_up_is_no_predictor)
    register_validator("set_censoring", _censoring_names_the_outcome)
    register_validator("set_censoring", _same_follow_up_against_the_data)
    register_validator("set_clusters", _clusters_name_a_grouping)
    register_validator("set_clusters", _no_grouping_is_recorded)
    register_completion("set_clusters", _no_grouping_names_what_it_denies)
    register_validator("set_estimand", _estimand_is_for_inference)
    register_validator("set_estimand", _estimand_names_a_predictor)
    register_validator("set_estimand", _estimand_measure_is_fitted)
    register_validator("set_estimand", _estimand_contrast_fits_the_exposure)
    register_validator("set_estimand", _omics_family_without_fdr_is_recorded)
    register_completion("set_estimand", _family_declares_its_multiplicity)
    register_validator("set_model_sequence", _sequence_follows_the_plan)
    register_validator("respond_diagnostic", _response_fits_the_check)
    register_validator("set_adjustment", _adjustment_follows_the_estimand)
    register_validator("set_adjustment", _answers_hold_together)
    register_validator("set_adjustment", _mediators_stay_out_of_a_total_effect)
    register_validator("set_adjustment", _a_direct_effect_asks_its_questions)
    register_validator("set_energy_adjustment", _energy_model_fits_the_contrast)
    register_validator("select_models", _models_fit_the_family)


_register()

__all__ = [
    "DIRECT_QUESTIONS", "Derived", "ESTIMATE_STAGES", "FIT_SCORES", "GUESSES", "GUESS_WORDS",
    "HOLDS", "MEASURE_OF_TASK", "MODEL_SCORES", "MEASURE_WORDS", "NOT_FITTED", "QUESTIONS", "ROLE_PLURAL", "ROLE_SINGULAR",
    "ROLE_WORDS", "adjustment_answer", "adjustment_card", "adjustment_gate", "adjustment_left_out",
    "annotate_fit", "asked_covariates", "caption", "cluster_answer", "cluster_candidates",
    "clusters_gate", "covariates", "current_answers", "current_estimand", "derive",
    "derived_roles", "direct_questions", "effective_task", "estimand_card", "estimand_gate",
    "exposure_candidates", "fixed_effects_column", "follow_up_answer", "follow_up_candidates",
    "follow_up_gate", "grouping_candidates", "grouping_card", "guess_blocks", "guess_of",
    "measures_offered", "mediators", "primary_features", "reads_as_follow_up", "secondary_columns",
    "served_gate", "unanswered", "withhold", "without_scores",
]

"""A time-varying exposure, routed to g-methods (V2_DEFINITION_OF_DONE §2, the causal-inference row).

The computation is ``turbotab/core/models/time_varying.py`` and the stage is
``stages/time_varying.py``. This module holds the routing: when the question is asked, what its
answer must declare, the refusals that keep it honest, and the method's §13 contract (BLUEPRINT
§13) with its relations.

**When it is asked.** Under inference, when the rows are a unit's time points kept as rows (the
grain is repeated, the repeats are time points, and the records are not combined), after one
exposure and the adjustment set are declared, and once the values show that the exposure changes
within units (the stage's ``setting``). An exposure that takes one value per unit is fixed in time,
and the standard model estimates it. The question's answer (``set_time_varying``) names the lane:

* the marginal structural model with stabilized inverse-probability weights (``msm_iptw``);
* the parametric g-formula (``gformula``);
* standard regression (``standard``).

**What the lane needs.** These are refused until present, each refusal with its way forward:

* repeated measures, one row per unit per time point;
* a settled time column (``stages.working.time_column``: the user named it with the repeats or
  temporal answer, or confirmed it on its own; a proposed one is asked, BLUEPRINT §14.1);
* a declared time ordering. For a g-method the exposure at each time point must precede the outcome
  it is paired with: "same time" or "unknown" is refused, because an exposure measured with its
  outcome cannot be told from a consequence of it. Standard regression can still describe how a
  concurrently measured exposure and outcome go together, so there it is **block and record**: kept
  only with its own attestation (``ordering_acknowledged``), and the fit's row is labeled as open to
  reverse causation.

**The leash** (BLUEPRINT §11.3; the shortest, the causal lane's). A covariate the adjustment
answers call a cause of the exposure that the exposure could have changed (``causes_exposure`` not
"no", ``after_exposure`` "yes"), and a cause of the outcome or a proxy for one, is a time-varying
confounder affected by prior exposure (:func:`affected_confounders`). Robins, Hernán & Brumback
(2000, *Epidemiology* 11:550–560, abstract): "standard approaches for adjustment of confounding are
biased when there exist time-dependent confounders that are also affected by previous treatment."
Adjusting for it in a regression removes the part of an earlier exposure's effect that runs through
it. Leaving it out leaves the later exposure confounded. So:

* standard regression with such a confounder is **block and record**: refused until attested
  (``acknowledged``), with the g-methods as its exits;
* each g-method must name every such confounder among its time-varying confounders;
* standard regression with none is offered first, and the g-methods stay available.

**Diagnostics before estimates.** Cole & Hernán (2008, *Am J Epidemiol* 168:656–664): "Estimated
weights with the mean far from one or very extreme values are indicative of nonpositivity or
misspecification of the weight model". Each g-method is answered in two steps. The lane is declared
first, and its diagnostics are shown: for the weights, their distribution, each truncation option's
effect on it and positivity at each time point; for the g-formula, positivity at each time point and
what the simulation will take. These describe the exposure (and loss to follow-up, whose rows at risk
end with an event); no estimate's association with the outcome is in them. Then the truncation, or
the simulation's size, is declared. That second declaration is refused until the fresh artifact
holds this lane's diagnostics on the data as they are (:func:`_diagnostics_come_first`), and the
server stamps it with those diagnostics' key (``diagnostics_seen``). The stage computes an estimate
only when the key of the diagnostics it computes now is that one, so a change to the data or to an
answer the weights read after the declaration withholds the estimate and re-asks the question
(:func:`lane_answer`). "Read before any estimate" is true by construction, not by the order of
requests.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from turbotab.core.contracts import ContractOption, MethodContract, Relation, register_contract

ROBINS_2000 = "Robins, Hernán & Brumback 2000, Epidemiology 11:550–560"
HERNAN_2000 = "Hernán, Brumback & Robins 2000, Epidemiology 11:561–570"
COLE_HERNAN = "Cole & Hernán 2008, Am J Epidemiol 168:656–664"
ROBINS_1986 = "Robins 1986, Math Model 7:1393–1512"
MCGRATH = "McGrath et al. 2020, Patterns 1:100008"
WHAT_IF = "Hernán & Robins, Causal Inference: What If, ch. 17 and 19–21"
VANDERWEELE_DING = "VanderWeele & Ding 2017, Ann Intern Med 167:268–274"
BELL_MCCAFFREY = "Bell & McCaffrey 2002, Surv Methodol 28:169–181"
# The same, as a methods sentence cites them.
CITE = {"robins_2000": "Robins, Hernán & Brumback 2000",
        "hernan_2000": "Hernán, Brumback & Robins 2000", "cole_hernan": "Cole & Hernán 2008",
        "robins_1986": "Robins 1986", "mcgrath": "McGrath et al. 2020",
        "bell_mccaffrey": "Bell & McCaffrey 2002"}
KEY = "time_varying"
METHOD_WORDS = {"msm_iptw": "a marginal structural model with stabilized inverse-probability "
                            "weights",
                "gformula": "the parametric g-formula", "standard": "standard regression"}
TRUNCATION_WORDS = {"none": "not truncated", "p1_p99": "truncated at the 1st and 99th percentiles",
                    "p5_p95": "truncated at the 5th and 95th percentiles"}
# The g-formula's simulation size when one count is declared without the other, and the size whose
# time the diagnostics measure: ``gfoRmula``'s own default bootstrap count is 0, so these are the
# app's convention (10,000 simulated units bound the Monte Carlo error near 0.005 for a risk of 0.3).
SIMULATIONS = 10_000
BOOTSTRAP = 500
PRECEDES = "exposure_precedes_outcome"
ORDERING_WORDS = {"same_time": "measured at the same time as",
                  "unknown": "not known to come before"}

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


# ── the §13 contract ─────────────────────────────────────────────────────────


def _clause(run: Mapping[str, Any]) -> str | None:
    """The lane's methods clause, as the stage wrote it from what it ran
    (``stages.time_varying.methods_paragraph``)."""
    return run.get("clause")


CONTRACT = register_contract(MethodContract(
    key=KEY,
    label="A time-varying exposure by g-methods",
    slot="model",
    scope="model",
    run_order=0.0,
    needs=("inference with one declared exposure that changes within units",
           "repeated measures kept as rows: one row per unit per time point",
           "a settled time column",
           "a declared time ordering: for a g-method, the exposure precedes the outcome it is "
           "paired with",
           "an exposure of 0 or 1 at each time point"),
    question="How is an exposure that changes over time estimated?",
    options=(
        ContractOption(
            "msm_iptw", "Marginal structural model (weights)",
            customary=f"customary in epidemiology for time-varying exposures ({ROBINS_2000}; "
                      f"{COLE_HERNAN})",
            sound={"inference": "sound with a confounder affected by prior exposure: the weights "
                                "adjust for it without blocking the earlier exposure's effect; it "
                                "needs a correct exposure model and positivity",
                   "prediction": "not offered: under prediction no coefficient is read as an "
                                 "effect"},
            rung={"inference": "recommended", "prediction": "refused"},
            order={"inference": 0, "prediction": 0}),
        ContractOption(
            "gformula", "Parametric g-formula",
            customary=f"less common; the g-methods literature's other estimator ({ROBINS_1986}; "
                      f"{MCGRATH})",
            sound={"inference": "sound with a confounder affected by prior exposure: the "
                                "confounders are simulated forward under each strategy; it needs a "
                                "correct model for every time-varying covariate and the outcome",
                   "prediction": "not offered: under prediction no coefficient is read as an "
                                 "effect"},
            rung={"inference": "available", "prediction": "refused"},
            order={"inference": 1, "prediction": 1}),
        ContractOption(
            "standard", "Standard regression",
            customary="customary in cohort analyses: the time-varying covariates in a pooled, "
                      "mixed or GEE model",
            sound={"inference": "sound only when no time-varying confounder is affected by prior "
                                f"exposure; otherwise biased ({ROBINS_2000})",
                   "prediction": "not offered: under prediction no coefficient is read as an "
                                 "effect"},
            rung={"inference": "block_and_record", "prediction": "refused"},
            order={"inference": 2, "prediction": 2}),
    ),
    storyboard=(
        "Model the exposure at each time point from its history: the baseline covariates and its "
        "previous value (numerator), and also the time-varying confounders (denominator)",
        "Multiply each unit's probabilities over time into stabilized weights; censoring weights "
        "the same way",
        "Read the weights' distribution and positivity at each time point, then declare the "
        "truncation",
        "Fit the outcome on the exposure history in the weighted pseudo-population, with a CR2 "
        "variance by unit",
        "For the g-formula: read positivity and what the simulation will take, declare its size, "
        "then simulate every unit always and never exposed",
    ),
    relations=(
        Relation("conflicts", "standard_adjustment_for_affected_confounders",
                 "Standard regression cannot adjust for a confounder that earlier exposure changed: "
                 "adjusting for it removes part of the effect, and leaving it out leaves later "
                 "exposure confounded. It is recorded only with an attestation; the g-methods are "
                 "the exits.",
                 purposes=("inference",), rung="block_and_record", when=("standard",),
                 exits=("Estimate by a marginal structural model (weights)",
                        "Estimate by the parametric g-formula",
                        "Keep standard regression; record that the estimate is biased by it"),
                 enforced_by="turbotab.core.time_varying:_affected_confounders_need_g_methods",
                 condition="a covariate answered a cause of the exposure that earlier exposure "
                           "could have changed, and a cause of the outcome"),
        Relation("conflicts", "concurrent_exposure_and_outcome",
                 "An exposure measured with its outcome cannot be told from a consequence of it. "
                 "Standard regression is recorded only with that attestation, and its row is "
                 "labeled as open to reverse causation; a g-method is refused.",
                 purposes=("inference",), rung="block_and_record", when=("standard",),
                 exits=("The exposure precedes the outcome it is paired with",
                        "Keep standard regression; record that the exposure is measured with its "
                        "outcome"),
                 enforced_by="turbotab.core.time_varying:_lane_declares_the_time_ordering",
                 condition="the ordering answered same time, or unknown"),
        Relation("implies", "weight_diagnostics_before_estimates",
                 "The weights' distribution, each truncation option's effect on it and positivity "
                 "are shown first; the truncation is declared after them, and no estimate is "
                 "computed until it is.",
                 purposes=("inference",), when=("msm_iptw",),
                 enforced_by="turbotab.core.time_varying:_diagnostics_come_first"),
        Relation("implies", "positivity_and_time_before_estimates",
                 "Positivity at each time point and what the simulation will take are shown first; "
                 "its size is declared after them, and no risk is simulated until it is.",
                 purposes=("inference",), when=("gformula",),
                 enforced_by="turbotab.core.time_varying:_diagnostics_come_first"),
        Relation("implies", "positivity_by_time_point",
                 "Positivity is read at each time point: how many rows were exposed and unexposed, "
                 "and the range of the fitted probability of exposure.",
                 purposes=("inference",), when=("msm_iptw", "gformula")),
        Relation("implies", "intervals_by_unit",
                 "A unit's rows are not independent: the weighted model's variance is clustered by "
                 "unit (CR2, with Bell–McCaffrey degrees of freedom), the g-formula's interval "
                 "resamples whole units, and with fewer units than the floor no interval is "
                 "reported.",
                 purposes=("inference",), when=("msm_iptw", "gformula")),
        Relation("implies", "censoring_weighted",
                 "A unit lost to follow-up contributes the time points it was seen, and the units "
                 "like it that stayed are weighted up by the inverse of their probability of having "
                 "stayed.",
                 purposes=("inference",), when=("msm_iptw",)),
        Relation("implies", "loss_assumed_independent",
                 "Units whose rows end before the last time point without the event were lost to "
                 "follow-up; with no indicator declared, the estimate assumes their loss is "
                 "unrelated to the outcome, and the sentence says so.",
                 purposes=("inference",), when=("msm_iptw", "gformula")),
        Relation("implies", "unmeasured_confounding_sensitivity",
                 "The E-value states how strong an unmeasured confounder would have to be to "
                 "explain the estimate away (MODELING_SEQUENCE §0 ruling 10).",
                 purposes=("inference",), when=("msm_iptw", "gformula")),
        Relation("invalidates", "estimand",
                 "Another exposure re-asks the lane: its confounders, weights and strategies "
                 "belong to the exposure they were declared for.",
                 purposes=("inference",), enforced_by="turbotab.core.time_varying:lane_answer"),
        Relation("disables", "standard_estimate_as_the_effect",
                 "The standard model's row for the exposure is labeled as not its effect: it "
                 "cannot adjust for a confounder affected by prior exposure.",
                 purposes=("inference",), when=("msm_iptw", "gformula"),
                 enforced_by="turbotab.core.time_varying:fit_note"),
        Relation("enables", "risks_under_always_and_never",
                 "The g-formula compares the risks had every unit always, and never, been exposed, "
                 "as a difference and a ratio.",
                 purposes=("inference",), when=("gformula",)),
    ),
    sources=(ROBINS_2000, HERNAN_2000, COLE_HERNAN, ROBINS_1986, MCGRATH, WHAT_IF,
             VANDERWEELE_DING, BELL_MCCAFFREY),
    clause=_clause,
    decision="set_time_varying",
    stage="time_varying",
    package="TIMEVARY",
    place="MODELING_SEQUENCE §1, after step 3 (the adjustment set): the time-varying exposure "
          "question, asked under inference when a unit's time points are kept as rows",
    scope_note="The weights and the outcome model read every analyzed row, and the outcome model "
               "reads the outcome, so the lane is the model itself. Its diagnostics are shown "
               "first: they describe the exposure, and under an event outcome the rows still at "
               "risk of loss to follow-up, which the event ends; none of them relates the exposure "
               "to the outcome.",
    leash={"inference": "available", "prediction": "not_offered"},
    sentence="turbotab.core.stages.time_varying:methods_paragraph",
))


# ── the setting: long records, the exposure, the confounders ─────────────────


def _structure_data(structure: Any) -> Mapping[str, Any] | None:
    data = getattr(structure, "data", structure)
    return data if isinstance(data, Mapping) else None


def records_gate(state: Any, structure: Any = None) -> Gate:
    """Whether the rows are a unit's time points kept as rows: None when they are (or while that
    is unanswered), else ``("not_applicable", reason)``."""
    from turbotab.core.stages.working import effective_grain, effective_repeat_kind

    data = _structure_data(structure)
    grain = effective_grain(state, data)
    if grain is None:
        return None
    if _get(grain, "grain") != "repeated":
        return ("not_applicable", "Each unit appears once, so no exposure changes over time.")
    kind = effective_repeat_kind(state, data)
    if kind is None:
        return None
    if kind != "time_points":
        return ("not_applicable", "The rows are repeats of one measurement, not time points, so "
                                  "no exposure changes over time.")
    unit = _get(state, "unit")
    if unit == "unit":
        return ("not_applicable", "Each unit's rows are combined, so no time points stay as rows.")
    return None


def exposure_of(state: Any) -> str | None:
    """The one exposure the estimand declares, else None (none yet, or an exposure family)."""
    from turbotab.core.estimand import current_estimand

    spec = current_estimand(state)
    if spec is None or _get(spec, "family"):
        return None
    return _get(spec, "exposure")


def affected_confounders(state: Any) -> list[str]:
    """Covariates the adjustment answers (for the current exposure) call a cause, or possible cause,
    of the exposure that the exposure could have changed, and a cause of the outcome or a proxy for
    an unmeasured one: in a long table, confounders of later exposure affected by earlier exposure."""
    from turbotab.core.estimand import current_answers

    out = []
    for column, a in current_answers(state).items():
        if (_get(a, "after_exposure") == "yes" and _get(a, "causes_exposure") != "no"
                and (_get(a, "causes_outcome") != "no" or _get(a, "proxy"))
                and not _get(a, "instrument")):
            out.append(column)
    return out


def lane_gate(state: Any, structure: Any = None, artifact: Any = None) -> Gate:
    """The Router's gate for the question (``interview.route``)."""
    from turbotab.core.estimand import current_estimand, estimand_gate

    purpose = _get(state, "purpose")
    if purpose is None:
        return None
    if purpose != "inference":
        return ("not_applicable", "Under prediction no coefficient is read as an effect, so no "
                                  "exposure is followed through time.")
    found = records_gate(state, structure)
    if found is not None:
        return found
    found = estimand_gate(state)
    if found is not None:
        return found
    spec = current_estimand(state)
    if spec is None:
        return None
    if _get(spec, "family"):
        return ("not_applicable", "An exposure family is estimated one exposure at a time by the "
                                  "feature-wise family, not followed through time.")
    data = getattr(artifact, "data", artifact)
    setting = _get(data, "setting") if isinstance(data, Mapping) else None
    exposure = _get(spec, "exposure")
    if (isinstance(setting, Mapping) and setting.get("exposure") == exposure
            and setting.get("exposure_varies") is False):
        return ("not_applicable", f"{_tick(exposure)} takes one value within every unit, so it "
                                  f"does not change over time; the standard model estimates it.")
    return None


def exposure_varies(state: Any, artifact: Any) -> bool:
    """Whether the ``time_varying`` stage read the declared exposure as changing within units (its
    setting's ``exposure_varies``, for this exposure): the g-methods' question, not a point
    exposure's."""
    data = getattr(artifact, "data", artifact)
    setting = _get(data, "setting") if isinstance(data, Mapping) else None
    exposure = exposure_of(state)
    return (exposure is not None and isinstance(setting, Mapping)
            and setting.get("exposure") == exposure and setting.get("exposure_varies") is True)


def declares_after_diagnostics(spec: Any) -> bool:
    """Whether the lane declares what comes after its diagnostics: the weights' truncation, or
    the g-formula's simulation size."""
    method = _get(spec, "method")
    if method == "msm_iptw":
        return _get(spec, "truncation") is not None
    if method == "gformula":
        return _get(spec, "simulations") is not None or _get(spec, "bootstrap") is not None
    return False


def _diagnosed(artifact: Any) -> Mapping[str, Any] | None:
    data = getattr(artifact, "data", artifact)
    found = data.get("diagnosed") if isinstance(data, Mapping) else None
    return found if isinstance(found, Mapping) else None


def lane_answer(state: Any, artifact: Any = None) -> Any:
    """The lane, once it is complete for the current exposure: a lane declared for another
    exposure is re-asked (MODELING_SEQUENCE §2, "invalidates"); a g-method waits for what is
    declared after its diagnostics (the truncation, or the simulation's size); and when the stage's
    fresh ``artifact`` shows diagnostics other than the ones that declaration came after (the data,
    or an answer the weights read, changed since), the question is asked again."""
    spec = current_lane(state)
    if spec is None:
        return None
    method = _get(spec, "method")
    if method == "msm_iptw" and _get(spec, "truncation") is None:
        return None
    if method == "gformula" and (_get(spec, "simulations") is None
                                 or _get(spec, "bootstrap") is None):
        return None
    seen = _diagnosed(artifact)
    if method != "standard" and seen is not None and seen.get("key") != _get(spec, "diagnostics_seen"):
        return None
    return spec


def current_lane(state: Any) -> Any:
    """The lane declared for the current exposure, complete or not."""
    spec = _get(state, "time_varying")
    exposure = exposure_of(state)
    if spec is None or exposure is None or _get(spec, "exposure") != exposure:
        return None
    return spec


def _concurrent(lane: Any) -> str | None:
    """How the lane's standard regression was recorded with an exposure not declared to precede its
    outcome, in words; None when the order is declared."""
    ordering = _get(lane, "ordering")
    return ORDERING_WORDS.get(ordering) if ordering != PRECEDES else None


def fit_note(state: Any) -> str | None:
    """What the standard fit's row for the exposure is, served beside a time-varying lane
    (``estimand.annotate_fit``): not the effect, when a confounder affected by prior exposure is
    adjusted for there or left out; open to reverse causation, when standard regression was
    recorded with the exposure measured with its outcome."""
    lane = lane_answer(state)
    if lane is None:
        return None
    affected = affected_confounders(state)
    x = _tick(lane.exposure)
    one = len(affected) == 1
    what = f"{_listing(affected)}, {'a confounder' if one else 'confounders'} affected by prior {x}"
    if lane.method == "standard":
        notes = []
        if affected:
            notes.append(f"Recorded: standard regression over {what}; this row for {x} is biased "
                         f"by {'it' if one else 'them'}.")
        said = _concurrent(lane)
        if said:
            notes.append(f"Recorded: {x} is {said} the outcome it is paired with, so this row for "
                         f"{x} cannot be told from a consequence of the outcome (reverse "
                         f"causation).")
        return " ".join(notes) or None
    if not affected:
        return None
    return (f"The estimate of {x} is the time-varying lane's ({METHOD_WORDS[lane.method]}). This "
            f"model cannot adjust for {what}, so its row for {x} is not the effect.")


# ── the record's sentence ────────────────────────────────────────────────────


def _order_clause(d: Any, exposure: str) -> str:
    said = _concurrent(d)
    if said is None:
        return f"; {exposure} at each time point precedes the outcome it is paired with, as declared"
    return (f"; {exposure} is {said} the outcome it is paired with, so its estimate cannot be told "
            f"from a consequence of the outcome (reverse causation), as recorded")


def record_sentence(d: Any, state: Any) -> str:
    """The sentence ``set_time_varying`` records (``voice``): the lane, its columns and the declared
    ordering; the stage's methods sentence adds what was computed."""
    target = _get(state, "target")
    on = f" on {_tick(target)}" if target else ""
    exposure = _tick(d.exposure)
    order = _order_clause(d, exposure)
    confounders = _listing(d.confounders) if d.confounders else None
    baseline = _listing(d.baseline) if d.baseline else None
    if d.method == "standard":
        affected = affected_confounders(state)
        if affected and d.acknowledged:
            return (f"{exposure} changes over time and its effect{on} is estimated by standard "
                    f"regression, as recorded, although {_listing(affected)} "
                    f"{'is a confounder' if len(affected) == 1 else 'are confounders'} that earlier "
                    f"{exposure} changed: adjusting for "
                    f"{'it' if len(affected) == 1 else 'them'} removes part of the effect, and "
                    f"leaving {'it' if len(affected) == 1 else 'them'} out leaves later exposure "
                    f"confounded ({CITE['robins_2000']}){order}")
        return (f"{exposure} changes over time; no time-varying confounder is affected by earlier "
                f"{exposure}, so standard regression estimates its effect{on}{order}")
    models = []
    if confounders:
        models.append(f"{confounders} (time-varying)")
    if baseline:
        models.append(f"{baseline} (baseline)")
    reads = f" from {', '.join(models)}" if models else ""
    if d.method == "msm_iptw":
        pattern = ("once started, it stays" if d.pattern == "initiation"
                   else "it can stop and restart")
        text = (f"{exposure} changes over time ({pattern}), so its effect{on} is estimated by "
                f"{METHOD_WORDS['msm_iptw']}: the weights model {exposure} at each time point"
                f"{reads} and its history")
        if d.censoring:
            text += f", loss to follow-up ({_tick(d.censoring)}) is weighted the same way"
        text += (f", and the weights are {TRUNCATION_WORDS[d.truncation]}, as declared after "
                 f"their diagnostics were read" if d.truncation else
                 ", and the truncation is declared after the weights' diagnostics are read")
        return text + order
    simulated = (f"{confounders} {'is' if len(d.confounders) == 1 else 'are'} simulated forward "
                 f"from each time point's history" if confounders else
                 "no time-varying confounder is simulated")
    text = (f"{exposure} changes over time, so its effect{on} is estimated by "
            f"{METHOD_WORDS['gformula']}: {simulated}, and the risks had every unit always and never "
            f"been exposed are compared")
    if d.simulations is not None and d.bootstrap is not None:
        text += (f", over {d.simulations:,} simulated units with {d.bootstrap:,} bootstrap "
                 f"resamples, a size declared after positivity and the simulation's time were read")
    else:
        text += ("; the simulation's size is declared after positivity and the time the "
                 "simulation will take are read")
    if d.censoring:
        text += f", had no unit been lost to follow-up ({_tick(d.censoring)})"
    return text + order


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


def _setting(ctx: Any, exposure: str | None = None) -> Mapping[str, Any] | None:
    """The stage's reading of the data (fresh, and for this exposure), else None."""
    data = _artifact(ctx, KEY) or {}
    setting = data.get("setting") if isinstance(data, Mapping) else None
    if not isinstance(setting, Mapping):
        return None
    if exposure is not None and setting.get("exposure") != exposure:
        return None
    return setting


def _lane_is_for_inference(decision: Any, ctx: Any) -> None:
    from turbotab.core.estimand import _not_inference

    if _get(_state(ctx), "purpose") == "prediction":
        raise _not_inference("an exposure's path through time")


def _lane_needs_time_points_as_rows(decision: Any, ctx: Any) -> None:
    """Repeated measures, one row per unit per time point (the structural answers)."""
    from turbotab.core.decisions import SetRepeatKind, SetUnit
    from turbotab.core.stages.working import (effective_grain, effective_repeat_kind,
                                              proposed_time_column)

    state = _state(ctx)
    if state is None:
        return
    structure = _artifact(ctx, "structure")
    grain = effective_grain(state, structure)
    why = ("A time-varying exposure needs repeated measures: one row per unit per time point, "
           "so the exposure's history and the confounders' can be read in order.")
    if grain is None or _get(grain, "grain") != "repeated":
        raise _refusal("no_repeated_measures",
                       f"{why} The grain answer says each unit appears once.",
                       [{"label": "Each unit appears in several rows (the grain question)",
                         "decision": None}])
    kind = effective_repeat_kind(state, structure)
    if kind != "time_points":
        column = proposed_time_column(state, structure)
        raise _refusal(
            "not_time_points",
            f"{why} The rows are not answered as time points"
            f"{' (they are answered as repeats of one measurement)' if kind == 'repeats' else ''}.",
            [{"label": "The rows are time points",
              "decision": SetRepeatKind(repeat_kind="time_points", time_column=column)}])
    if _get(state, "unit") != "row":
        raise _refusal("records_combined",
                       f"{why} Each unit's rows are combined into one, so no time points remain.",
                       [{"label": "Keep each record as a row", "decision": SetUnit(unit="row")}])


def _lane_needs_a_settled_time_column(decision: Any, ctx: Any) -> None:
    """The column that orders a unit's rows, settled (BLUEPRINT §14.1): its history is read by it."""
    from turbotab.core.readings import confirm_exit
    from turbotab.core.stages.working import proposed_time_column, time_column

    state = _state(ctx)
    if state is None:
        return
    structure = _artifact(ctx, "structure")
    if time_column(state, structure) is not None:
        return
    proposed = proposed_time_column(state, structure)
    exits = ([confirm_exit("time_column", proposed, "orders",
                           f"`{proposed}` orders each unit's rows")] if proposed else [])
    raise _refusal(
        "time_column_unsettled",
        "The weights and the g-formula read each unit's history in time order, and no column that "
        "orders the rows is settled"
        + (f": `{proposed}` is only proposed, by the repeats reading." if proposed else "."),
        [*exits, {"label": "Name the column that orders the rows (the repeats question)",
                  "decision": None}])


def _lane_follows_the_estimand(decision: Any, ctx: Any) -> None:
    from turbotab.core.estimand import current_estimand
    from turbotab.core.voice import question_name

    state = _state(ctx)
    if state is None:
        return
    spec = current_estimand(state)
    if spec is None:
        raise _refusal("no_estimand",
                       f"Declare the exposure first ({question_name('estimand')}); its path through "
                       f"time is estimated for the exposure declared.",
                       [{"label": f"Answer {question_name('estimand')} first", "decision": None}])
    if _get(spec, "family"):
        raise _refusal("family_not_followed",
                       "An exposure family is estimated one exposure at a time by the feature-wise "
                       "family; a time-varying lane follows one declared exposure.",
                       [{"label": "Name one exposure (the exposure question)", "decision": None}])
    exposure = _get(spec, "exposure")
    if decision.exposure != exposure:
        raise _refusal("other_exposure",
                       f"The exposure is {_tick(exposure)}, not {_tick(decision.exposure)}; the lane "
                       f"is declared for the exposure the estimand names.",
                       [{"label": f"Declare the lane for {_tick(exposure)}",
                         "decision": decision.model_copy(update={"exposure": exposure})}])


def _lane_declares_the_time_ordering(decision: Any, ctx: Any) -> None:
    """The exposure at each time point precedes the outcome it is paired with (an assumption only
    the user can declare; WHAT_IF ch. 19). A g-method needs it. Standard regression of a
    concurrently measured exposure is block and record: its own attestation, and a label on the
    fit's row (:func:`fit_note`)."""
    if decision.ordering == PRECEDES:
        return
    x = _tick(decision.exposure)
    said = ORDERING_WORDS[decision.ordering]
    recorded = ("measured with its outcome" if decision.ordering == "same_time"
                else "not known to precede its outcome")
    precedes = {"label": "The exposure precedes the outcome it is paired with",
                "decision": decision.model_copy(update={"ordering": PRECEDES,
                                                        "ordering_acknowledged": False})}
    if decision.method == "standard":
        if decision.ordering_acknowledged:
            return
        raise _refusal(
            "time_ordering",
            f"{x} is {said} the outcome it is paired with, so its estimate cannot be told from a "
            f"consequence of the outcome (reverse causation). Standard regression can describe how "
            f"they go together; it is recorded as that, and its row for {x} is labeled so.",
            [precedes,
             {"label": f"Keep standard regression; record that the exposure is {recorded}",
              "decision": decision.model_copy(update={"ordering_acknowledged": True})}])
    raise _refusal(
        "time_ordering",
        f"{x} is {said} the outcome it is paired with, so its estimate cannot be told from a "
        f"consequence of the outcome. g-methods need each time point's exposure to precede the "
        f"outcome it is paired with, and its confounders to precede the exposure.",
        [precedes,
         {"label": f"Estimate by standard regression, recorded as {recorded}",
          "decision": decision.model_copy(update={
              "method": "standard", "ordering_acknowledged": True, "truncation": None,
              "simulations": None, "bootstrap": None, "diagnostics_seen": None})}])


def _lane_names_model_columns(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import NUMERIC_DTYPES, _columns_of, _info, _target_of, _UNKNOWN
    from turbotab.core.estimand import asked_covariates

    state = _state(ctx)
    columns = _columns_of(ctx)
    named = [*decision.confounders, *decision.baseline,
             *([decision.censoring] if decision.censoring else [])]
    if columns is not None:
        unknown = [c for c in named if c not in columns]
        if unknown:
            raise _refusal("unknown_column",
                           f"This dataset has no column named {_listing(unknown)}.",
                           [{"label": "Choose the dataset's columns", "decision": None}])
    info = _info(ctx, decision.censoring) if decision.censoring else None
    if info is not None and (str(info.get("dtype")) not in NUMERIC_DTYPES
                             or int(info.get("n_unique") or 0) > 2):
        raise _refusal(
            "censoring_not_an_indicator",
            f"{_tick(decision.censoring)} is not 0 or 1 in every row; a loss-to-follow-up indicator "
            f"is 1 on a unit's last time point before it was lost, else 0.",
            [{"label": "No loss-to-follow-up indicator",
              "decision": decision.model_copy(update={"censoring": None})}])
    target = _target_of(ctx)
    if target is not _UNKNOWN and target in named:
        raise _refusal("outcome_as_covariate",
                       f"{_tick(target)} is the outcome; it is not a covariate of its own model.",
                       [{"label": "Leave the outcome out", "decision": decision.model_copy(update={
                           "confounders": [c for c in decision.confounders if c != target],
                           "baseline": [c for c in decision.baseline if c != target]})}])
    if state is None or _get(state, "roles") is None:
        return
    from turbotab.core.stages.time_varying import candidate_columns

    known = set(candidate_columns(state)) | set(asked_covariates(state))
    strangers = [c for c in [*decision.confounders, *decision.baseline] if c not in known]
    if strangers:
        raise _refusal(
            "not_a_covariate",
            f"{_listing(strangers)} {'is' if len(strangers) == 1 else 'are'} not among the "
            f"covariates the adjustment question answered for {_tick(decision.exposure)}; the lane "
            f"adjusts for the covariates the adjustment set declares.",
            [{"label": "Use the adjustment set's covariates only", "decision": decision.model_copy(
                update={"confounders": [c for c in decision.confounders if c in known],
                        "baseline": [c for c in decision.baseline if c in known]})}])


def _affected_confounders_need_g_methods(decision: Any, ctx: Any) -> None:
    """The leash (MODELING_SEQUENCE §4 row, V2 causal row): standard regression with a confounder
    affected by prior exposure is block and record, with the g-methods as exits; a g-method adjusts
    for every such confounder. The g-method exits carry the loss-to-follow-up indicator the decision
    names, else the one the stage's proposal reads from the values, said in the exit's label; an
    order not declared is declared only by an exit that says so."""
    state = _state(ctx)
    if state is None:
        return
    affected = affected_confounders(state)
    if not affected:
        return
    one = len(affected) == 1
    who = _listing(affected)
    named, base = list(decision.confounders), list(decision.baseline)
    proposal = (_setting(ctx, decision.exposure) or {}).get("proposal") or {}
    if not named and not base:
        # The completion fills a lane that names no covariate with the stage's proposal.
        named = list(proposal.get("confounders") or [])
        base = list(proposal.get("baseline") or [])
    if decision.method == "standard":
        if decision.acknowledged:
            return
        censoring = decision.censoring or proposal.get("censoring")
        lost = (f", weighting loss to follow-up by {_tick(censoring)}"
                if censoring and censoring != decision.censoring else "")
        concurrent = decision.ordering != PRECEDES
        g = {"confounders": list(dict.fromkeys([*named, *affected])),
             "baseline": [c for c in base if c not in affected], "acknowledged": False,
             "censoring": censoring, "ordering": PRECEDES, "ordering_acknowledged": False,
             "diagnostics_seen": None}
        declare = "Declare that the exposure precedes its outcome; e" if concurrent else "E"
        setting = _setting(ctx, decision.exposure) or {}
        binary = setting.get("exposure_binary", True)
        message = (
            f"{who} {'is a cause' if one else 'are causes'} of {_tick(decision.exposure)} that "
            f"earlier {_tick(decision.exposure)} could have changed, by your answers: a time-varying "
            f"confounder affected by prior exposure. Standard regression is biased either way: "
            f"adjusting for {'it' if one else 'them'} removes the part of an earlier exposure's "
            f"effect that runs through {'it' if one else 'them'}, and leaving "
            f"{'it' if one else 'them'} out leaves later exposure confounded ({ROBINS_2000}).")
        keep = {"label": "Keep standard regression; record that the estimate is biased by it",
                "decision": decision.model_copy(update={"acknowledged": True})}
        if not binary:
            # The g-methods here take an exposure of 0 or 1 (V2 scope), so offering them would
            # offer answers that are refused in turn.
            raise _refusal(
                "affected_confounder",
                f"{message} The g-methods here need an exposure of 0 or 1 at each time point, and "
                f"{_tick(decision.exposure)} takes {setting.get('exposure_levels')} values.",
                [keep, {"label": "Declare a yes/no exposure (the exposure question)",
                        "decision": None}])
        raise _refusal(
            "affected_confounder", message,
            [{"label": f"{declare}stimate by a marginal structural model (weights){lost}",
              "decision": decision.model_copy(update={**g, "method": "msm_iptw", "truncation": None,
                                                      "simulations": None, "bootstrap": None})},
             {"label": f"{declare}stimate by the parametric g-formula{lost}",
              "decision": decision.model_copy(update={**g, "method": "gformula", "truncation": None,
                                                      "simulations": None, "bootstrap": None})},
             keep])
    missing = [c for c in affected if c not in named]
    if missing:
        raise _refusal(
            "affected_confounder_left_out",
            f"{_listing(missing)} {'is a confounder' if len(missing) == 1 else 'are confounders'} "
            f"affected by prior {_tick(decision.exposure)}, by your answers: the g-methods exist to "
            f"adjust for {'it' if len(missing) == 1 else 'them'} at each time point.",
            [{"label": f"Adjust for {_listing(missing)} at each time point",
              "decision": decision.model_copy(update={
                  "confounders": [*named, *missing],
                  "baseline": [c for c in base if c not in missing]})}])


def _lane_fits_the_outcome(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import _ctx

    state = _state(ctx)
    task = _get(state, "task") or _ctx(ctx, "task")
    if task is None or decision.method == "standard":
        return
    if decision.method == "gformula" and task != "binary":
        raise _refusal(
            "gformula_needs_an_event",
            f"The parametric g-formula here estimates the risk of an event by each time point, "
            f"from a yes/no outcome at each one; the outcome is read as "
            f"{str(task).replace('_', ' ')}.",
            [{"label": "Estimate by a marginal structural model (weights)",
              "decision": decision.model_copy(update={"method": "msm_iptw", "simulations": None,
                                                      "bootstrap": None})}])
    if task not in ("binary", "regression"):
        raise _refusal(
            "msm_outcome",
            f"The marginal structural model here is a pooled logistic model of an event at each time "
            f"point or a linear model of a repeated measure; the outcome is read as "
            f"{str(task).replace('_', ' ')}.",
            [{"label": "Change the outcome's task (the task question)", "decision": None}])


def _each_declaration_is_its_method_s(decision: Any, ctx: Any) -> None:
    """A truncation trims inverse-probability weights; a simulation's size is the g-formula's."""
    if decision.truncation is not None and decision.method != "msm_iptw":
        raise _refusal(
            "truncation_without_weights",
            f"Truncation trims inverse-probability weights; {METHOD_WORDS[decision.method]} uses "
            f"none.",
            [{"label": "Leave the truncation out",
              "decision": decision.model_copy(update={"truncation": None})}])
    if (decision.simulations is not None or decision.bootstrap is not None) \
            and decision.method != "gformula":
        raise _refusal(
            "simulation_without_g_formula",
            f"The simulation's size is the parametric g-formula's; {METHOD_WORDS[decision.method]} "
            f"simulates nothing.",
            [{"label": "Leave the simulation's size out",
              "decision": decision.model_copy(update={"simulations": None, "bootstrap": None})}])


def _lane_reads_settled_readings(decision: Any, ctx: Any) -> None:
    """BLUEPRINT §14.1: the weight, covariate and outcome models are number-changing consumers of
    each covariate's role and each whole-number covariate's code-or-amount reading, as the fit is
    (``decisions._models_read_settled_readings``): while one waits, the lane asks."""
    from turbotab.core.decisions import _ctx, _store_of, left_out
    from turbotab.core.readings import Unsettled, predictors_or_ask

    state = _state(ctx)
    if state is None or not _get(state, "roles"):
        return
    keep = set(decision.confounders) | set(decision.baseline)
    try:
        predictors_or_ask(state, _ctx(ctx, "column_info"),
                          drop=[c for c in left_out(state) if c not in keep],
                          store=_store_of(ctx))
    except Unsettled as waiting:
        raise _refusal("reading_unsettled", str(waiting), [
            *waiting.exits, {"label": "Change the roles (the roles question)", "decision": None}]
        ) from None


def _lane_fits_the_data(decision: Any, ctx: Any) -> None:
    """What the values say to a g-method (the stage's ``setting``, when fresh for this exposure): an
    exposure of 0 and 1; a once-started exposure that never stops; baseline covariates fixed
    within units. Standard regression reads none of these."""
    setting = _setting(ctx, decision.exposure)
    if setting is None or decision.method == "standard":
        return
    if not setting.get("exposure_binary", True):
        raise _refusal(
            "exposure_not_binary",
            f"The weights and the always-or-never strategies here are for an exposure of 0 or 1 at "
            f"each time point; {_tick(decision.exposure)} takes "
            f"{setting.get('exposure_levels')} values.",
            [{"label": "Declare a yes/no exposure (the exposure question)", "decision": None}])
    stops = int(setting.get("exposure_stops") or 0)
    if decision.pattern == "initiation" and stops:
        raise _refusal(
            "exposure_stops",
            f"{_tick(decision.exposure)} stops after starting in {stops:,} "
            f"{'unit' if stops == 1 else 'units'}, so it is not an exposure that, once started, "
            f"stays.",
            [{"label": "It can stop and restart",
              "decision": decision.model_copy(update={"pattern": "switches"})}])
    varies = setting.get("varies") or {}
    moving = [c for c in decision.baseline if varies.get(c)]
    if moving:
        raise _refusal(
            "baseline_varies",
            f"{_listing(moving)} {'changes' if len(moving) == 1 else 'change'} within units, so "
            f"{'it is' if len(moving) == 1 else 'they are'} time-varying, not baseline.",
            [{"label": f"Adjust for {_listing(moving)} at each time point",
              "decision": decision.model_copy(update={
                  "baseline": [c for c in decision.baseline if c not in moving],
                  "confounders": [*decision.confounders, *moving]})}])


def lane_parts(spec: Any) -> dict[str, Any]:
    """What a lane's diagnostics depend on, beside the data: everything the weight (or the
    g-formula's) models read. The outcome model's exposure summary and the truncation do not."""
    return {"exposure": _get(spec, "exposure"), "method": _get(spec, "method"),
            "confounders": list(_get(spec, "confounders") or []),
            "baseline": list(_get(spec, "baseline") or []),
            "censoring": _get(spec, "censoring"), "pattern": _get(spec, "pattern")}


def _undeclared(decision: Any) -> Any:
    return decision.model_copy(update={"truncation": None, "simulations": None, "bootstrap": None,
                                       "diagnostics_seen": None})


def _diagnostics_come_first(decision: Any, ctx: Any) -> None:
    """The truncation, or the simulation's size, is declared after the lane's diagnostics were shown
    (V2 causal row: "diagnostics … shown before any estimate"): the stage's fresh artifact must hold
    the diagnostics of this very lane (its exposure, method, columns, censoring and pattern) on the
    data as they are now. Else refused, the exit the lane without that declaration, which shows
    them."""
    if not declares_after_diagnostics(decision):
        return
    filled = _lane_fills_its_columns(decision, ctx)
    seen = _diagnosed(_artifact(ctx, KEY))
    if seen is not None and lane_parts(seen) == lane_parts(filled):
        return
    if decision.method == "msm_iptw":
        message = (f"The truncation is declared after the weights' distribution and positivity at "
                   f"each time point are read, and they have not been shown for this lane on the "
                   f"data as they are now. No estimate is computed before them ({COLE_HERNAN}).")
        label = "Show the weights and positivity first (the lane without its truncation)"
    else:
        message = ("The simulation's size is declared after positivity at each time point and what "
                   "the simulation will take are read, and they have not been shown for this lane "
                   "on the data as they are now. No risk is simulated before them.")
        label = "Show positivity and the simulation's time first (the lane without its size)"
    raise _refusal("diagnostics_first", message, [{"label": label,
                                                   "decision": _undeclared(decision)}])


def _lane_fills_its_columns(decision: Any, ctx: Any) -> Any:
    """A lane that names no covariate takes the stage's proposal (the adjustment set's covariates,
    split by whether they change within units, and every affected confounder), as shown on the card."""
    if decision.confounders or decision.baseline:
        return decision
    setting = _setting(ctx, decision.exposure)
    if setting is None:
        return decision
    proposed = setting.get("proposal") or {}
    return decision.model_copy(update={"confounders": list(proposed.get("confounders") or []),
                                       "baseline": list(proposed.get("baseline") or [])})


def _lane_records_what_it_read(decision: Any, ctx: Any) -> Any:
    """The server's, never the client's (as ``set_roles``'s ``unconfirmed``): a declaration made
    after the diagnostics carries their key, read from the fresh artifact the validator checked;
    any other carries none. A g-formula size declared in part takes the other count's default."""
    if not declares_after_diagnostics(decision):
        return decision.model_copy(update={"diagnostics_seen": None})
    seen = _diagnosed(_artifact(ctx, KEY)) or {}
    update: dict[str, Any] = {"diagnostics_seen": seen.get("key")}
    if decision.method == "gformula":
        update["simulations"] = decision.simulations or SIMULATIONS
        update["bootstrap"] = decision.bootstrap or BOOTSTRAP
    return decision.model_copy(update=update)


def _register() -> None:
    from turbotab.core.decisions import register_completion, register_validator

    for check in (_lane_is_for_inference, _lane_needs_time_points_as_rows,
                  _lane_needs_a_settled_time_column, _lane_follows_the_estimand,
                  _lane_declares_the_time_ordering, _lane_names_model_columns,
                  _affected_confounders_need_g_methods, _lane_fits_the_outcome,
                  _each_declaration_is_its_method_s, _lane_reads_settled_readings,
                  _lane_fits_the_data, _diagnostics_come_first):
        register_validator("set_time_varying", check)
    register_completion("set_time_varying", _lane_fills_its_columns)
    register_completion("set_time_varying", _lane_records_what_it_read)


_register()

__all__ = [
    "BOOTSTRAP", "CITE", "CONTRACT", "KEY", "METHOD_WORDS", "PRECEDES", "SIMULATIONS",
    "TRUNCATION_WORDS", "affected_confounders", "current_lane", "declares_after_diagnostics",
    "exposure_of", "fit_note", "lane_answer", "lane_gate", "lane_parts", "record_sentence",
    "records_gate",
]

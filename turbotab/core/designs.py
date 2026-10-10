"""The study designs and the effects that wait for a later version: one registry (crosswalk
disagreement 10, the design slot; V2X_SEAMS seam guard 6 and recommendation 6).

A design is declared in Your question, before the goal (``q:study-design``). Until it is answered
the study is read as observational, stated for the person and listed in Your question's Confirm
sweep (``default:design_observational``). The designs the designed-experiments milestone runs
(parallel and cluster-randomized trials, case-control samples and matched sets; SIZING E1) and the
ones deferred to v2.x (crossover trials, repeated-measures trial models; V2_DEFINITION_OF_DONE §5)
are named values from the first commit, each refused with its code, its reason and an exit that
keeps the work: never deleted, so a log that recorded one still loads and a later version flips a
declaration instead of threading a new value through the Router (the ``share_reallocation``
precedent, ``core/methods/exposure_form.py``).

The effects the estimand card offers follow the same rule: "only the direct part" (a mediation
estimand), the complier effect and the per-protocol effect are named values of ``EffectKind``,
each refused with a ``*_v2x`` code and the whole effect as its exit. The direct effect's own
machinery (``estimand.direct_questions`` and its refusals) is kept for the mediation milestone.

This module is light (no import of the decisions at load), so ``decisions`` imports its
vocabulary; the refusals are registered by ``decisions`` and ``estimand``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DesignKey = Literal["observational", "parallel_trial", "cluster_randomized_trial", "case_control",
                    "matched_sets", "crossover", "repeated_measures_trial"]
OBSERVATIONAL: DesignKey = "observational"
# The code a design the designed-experiments milestone runs is refused with until it is built
# (the crosswalk's ``refusal:not-available-yet``).
NOT_YET = "not_available_yet"
NOT_YET_LEAD = "Not available yet"


@dataclass(frozen=True)
class Design:
    """One study design: how the app names it, whether this version analyzes it, and, where it
    does not, the refusal's code and reason. The milestone that runs a design adds what it routes
    (its estimands, adjustment rule, analysis sets and checklist) here."""

    key: str
    label: str
    available: bool
    code: str | None = None
    reason: str | None = None


_TRIALS_LATER = ("randomized trials are analyzed in the designed-experiments milestone (adjustment "
                 "for precision only, the analysis sets, causal wording), which is not built yet. "
                 "Until then the data can be analyzed as an observational study, and the methods "
                 "say so.")

DESIGNS: dict[str, Design] = {
    "observational": Design("observational", "Observed as they were", True),
    "parallel_trial": Design(
        "parallel_trial", "Randomized to parallel groups", False, NOT_YET,
        f"{NOT_YET_LEAD}: {_TRIALS_LATER}"),
    "cluster_randomized_trial": Design(
        "cluster_randomized_trial", "Randomized by group (clusters)", False, NOT_YET,
        f"{NOT_YET_LEAD}: cluster-randomized trials are analyzed in the designed-experiments "
        f"milestone, with the clusters as the unit of randomization, which is not built yet. Until "
        f"then the data can be analyzed as an observational study, and the methods say so."),
    "case_control": Design(
        "case_control", "Sampled by the outcome (case-control)", False, NOT_YET,
        f"{NOT_YET_LEAD}: a case-control sample is analyzed in the designed-experiments milestone "
        f"(no prevalence or absolute risk, conditional models for matched sets), which is not "
        f"built yet. Until then the data can be analyzed as an observational study, and the "
        f"methods say so."),
    "matched_sets": Design(
        "matched_sets", "Matched sets", False, NOT_YET,
        f"{NOT_YET_LEAD}: matched sets are analyzed in the designed-experiments milestone "
        f"(conditional logistic regression within each set), which is not built yet. Until then "
        f"the data can be analyzed as an observational study, and the methods say so."),
    "crossover": Design(
        "crossover", "Crossover trial", False, "crossover_v2x",
        f"{NOT_YET_LEAD}: a crossover trial (each person in several periods and sequences) needs "
        f"period and carry-over terms TurboTab does not fit in this version; it goes to v2.x "
        f"(V2_DEFINITION_OF_DONE §5). The data stay as they are: analyze them as an observational "
        f"study, or keep them for a later version."),
    "repeated_measures_trial": Design(
        "repeated_measures_trial", "Trial with repeated outcome measures", False,
        "repeated_measures_trial_v2x",
        f"{NOT_YET_LEAD}: a trial's repeated outcome measures need a repeated-measures model of "
        f"the trial (a mixed model for repeated measures) that TurboTab does not fit in this "
        f"version; it goes to v2.x (V2_DEFINITION_OF_DONE §5). The data stay as they are: analyze "
        f"them as an observational study, or keep them for a later version."),
}

OBSERVATIONAL_STATED = ("Read as an observational study: people were observed as they were, not "
                        "assigned or sampled by their outcome. Change it if they were randomized, "
                        "sampled by the outcome or matched.")


@dataclass(frozen=True)
class Effect:
    """One effect the estimand card can name: whether this version estimates it, and, where it
    does not, the refusal's code and reason (the exit is the whole effect)."""

    key: str
    label: str
    available: bool
    code: str | None = None
    reason: str | None = None


EFFECTS: dict[str, Effect] = {
    "total": Effect("total", "Total effect", True),
    "direct": Effect(
        "direct", "Only the direct part", False, "direct_effect_v2x",
        f"{NOT_YET_LEAD}: only the direct part of an effect is a mediation estimand, which needs "
        f"the common causes of each mediator and the outcome and the exposure–mediator "
        f"interaction, and mediation analysis goes to v2.x (V2_DEFINITION_OF_DONE §5). The whole "
        f"effect keeps every answer, and mediators stay out of its adjustment set."),
    "complier": Effect(
        "complier", "Among those who comply", False, "complier_effect_v2x",
        f"{NOT_YET_LEAD}: the effect among people who would take what they were assigned (a "
        f"complier effect) needs an instrumental-variable estimator TurboTab does not fit in this "
        f"version; it goes to v2.x. The whole effect keeps every answer."),
    "per_protocol": Effect(
        "per_protocol", "Of following the protocol", False, "per_protocol_effect_v2x",
        f"{NOT_YET_LEAD}: the effect of following the protocol needs weighting for adherence over "
        f"time, which goes to v2.x. A per-protocol analysis set (the adherent people analyzed as "
        f"randomized) comes with the designed-experiments milestone; the whole effect keeps every "
        f"answer."),
}


def design_refusal(decision: Any) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Why a recorded design is refused (code, reason, exits), or None when this version analyzes
    it. The first exit keeps the work: the data analyzed as an observational study."""
    found = DESIGNS.get(str(getattr(decision, "design", None)))
    if found is None or found.available:
        return None
    exits = [{"label": "Analyze it as an observational study",
              "decision": {"kind": "set_design", "design": OBSERVATIONAL}},
             {"label": "Keep the data as they are for a later version", "decision": None}]
    return str(found.code), str(found.reason), exits


def effect_refusal(decision: Any) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Why an effect is refused (code, reason, exits), or None when this version estimates it. The
    exit records the whole effect with every other part of the answer kept."""
    found = EFFECTS.get(str(getattr(decision, "effect", None)))
    if found is None or found.available:
        return None
    whole = {**decision.model_dump(mode="json"), "effect": "total"}
    return str(found.code), str(found.reason), [{"label": "The whole effect", "decision": whole}]


def design_of(state: Any) -> str:
    """The design in force: the answer, else observational (stated)."""
    return str(getattr(state, "design", None) or OBSERVATIONAL)


__all__ = ["DESIGNS", "Design", "DesignKey", "EFFECTS", "Effect", "NOT_YET", "OBSERVATIONAL",
           "OBSERVATIONAL_STATED", "design_of", "design_refusal", "effect_refusal"]

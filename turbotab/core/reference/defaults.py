"""What the app offers first where it ranks by the data, computed by the app's own ranking code.

A method contract ranks its options once per purpose (``MethodContract.options_for``), and so do
the questions labeled outside the registry (``turbotab.core.custom_sound``). Several are not
offered in that static order: the app ranks them again on the data in front of it, and a packet
that printed the static rank 1 as "the default" would state a default the app never offers (the
genomics lens's count normalization, the split at 21,849 rows, a common event's effect measure).

Each entry here names the condition the app ranks by and computes the first option for it by
calling the function the app itself ranks with, never a copy of its rule:

* ``effect_measure``: ``estimand.measures_offered`` (the marginal measures first for a common
  event; a numeric outcome's difference; an exposure family's feature-wise measure);
* ``split``: ``seal.split_offer`` with ``custom_sound.split`` (resampling first below
  ``validation.RESAMPLE_BELOW`` units, a holdout above it, time-ordered folds under the temporal
  seal, no holdout first under inference);
* ``omics_normalization``: the normalization finding's own offer (``methods.omics._offer``), by the
  data's kind;
* ``functional_form``: the form card's proposal (``methods.exposure_form.proposal``);
* ``exposure_transform``: the question that sets the transform (the energy model's ranking; a
  scale's scoring; the omics normalization);
* ``energy_adjustment`` and ``exclusions``: ``custom_sound`` with what can run on the table;
* ``diagnostics``: the failed check's own responses (``estimand.DIAGNOSTIC_ACTIONS``);
* ``evalue_sd``: ``models.effects.estimand_sd`` with and without the survey weights;
* ``unmeasured_confounding``: ``models.effects.unmeasured_confounding`` on a least-squares
  difference and on a ratio;
* ``nci_usual_intake``: ``usual_intake.suggested_model`` by the share of zero recalls;
* ``causal`` (the causal lane's question, labeled in ``causal.options``, not in the registry):
  stated, never asked by default (``causal.causal_gate``), its estimators ranked by the exposure's
  kind, the outcome, how many candidates there are for n and the survey answer. The four causal
  contracts (:data:`THROUGH`) are reached only through it, so their cases are its.

Each case states its premise and checks it against what the app computed, so a changed rule fails
here rather than printing a stale condition. ``turbotab/core/tests/test_reference.py`` holds the
cases to the reference journeys: where a journey took the app's first-ranked option, the case for
its condition must name the option it took.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

PURPOSES = ("prediction", "inference")


@dataclass(frozen=True)
class Case:
    """One condition the app ranks by, and what it offers first under it (Markdown)."""

    when: str
    first: str
    key: str  # the first option's key, for the tests' comparison with the journeys


class PremiseFailed(AssertionError):
    """The app no longer ranks as a case's words say: the words are stale."""


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise PremiseFailed(f"the app no longer ranks as the methods reference says: {what}")


# ── the effect measure ───────────────────────────────────────────────────────

TASK_WORDS = {"regression": "numeric", "binary": "yes/no", "time_to_event": "time-to-event",
              "ordinal": "ordered", "multiclass": "multiclass"}


def _first_measure(measures: list[dict[str, Any]]) -> dict[str, Any]:
    """The first fitted measure, with its reason as the card gives it for no particular table
    (the reason a card gives on a table also quotes that table's event share)."""
    from turbotab.core.estimand import _measure_reason

    first = next(m for m in measures if m.get("fitted"))
    return {**first, "reason": _measure_reason(first["measure"], None)}


def _a(words: str) -> str:
    return ("an " if words[:1].lower() in "aeiou" else "a ") + words


def effect_measure(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None
    from turbotab.core.estimand import (FAMILY_MEASURE_OF_TASK, MARGINAL, MARGINAL_TASKS,
                                        MEASURE_OF_TASK, measures_offered)
    from turbotab.core.models.effects import COMMON_OUTCOME, ZHANG_YU

    out: list[Case] = []
    for task in MEASURE_OF_TASK:
        words = TASK_WORDS.get(task, task.replace("_", " "))
        if task in MARGINAL_TASKS:
            common = _first_measure(measures_offered(task, prevalence=min(0.5, COMMON_OUTCOME * 4)))
            rare = _first_measure(measures_offered(task, prevalence=COMMON_OUTCOME / 2))
            _check(common["measure"] in MARGINAL and rare["measure"] not in MARGINAL,
                   "a common event's marginal measures rank first, a rare one's conditional")
            out.append(Case(f"{_a(words)} outcome whose event share is above "
                            f"{COMMON_OUTCOME:.0%}", f"`{common['measure']}`: {common['reason']}; "
                            f"ranked first because a common event's odds ratio overstates the "
                            f"risk ratio ({ZHANG_YU})", common["measure"]))
            out.append(Case(f"{_a(words)} outcome whose event share is {COMMON_OUTCOME:.0%} or less",
                            f"`{rare['measure']}`: {rare['reason']}", rare["measure"]))
        else:
            first = _first_measure(measures_offered(task))
            out.append(Case(f"{_a(words)} outcome", f"`{first['measure']}`: {first['reason']}",
                            first["measure"]))
    for task in FAMILY_MEASURE_OF_TASK:
        first = _first_measure(measures_offered(task, family=True))
        out.append(Case(f"an exposure family (each exposure in turn), "
                        f"{_a(TASK_WORDS.get(task, task))} outcome",
                        f"`{first['measure']}`: {first['reason']}", first["measure"]))
    return out


# ── the split ────────────────────────────────────────────────────────────────


def _split_case(purpose: str, when: str, task: str, n: int, counts: list[int] | None = None,
                time_ordered: bool = False, holdout_first: bool | None = None) -> Case:
    from turbotab.core import custom_sound
    from turbotab.core.seal import USUAL, split_offer

    offered = split_offer(purpose, task, n, counts, time_ordered=time_ordered)
    if holdout_first is not None:
        _check(offered.order[0] == "holdout" if holdout_first else offered.order[0] != "holdout",
               f"the split under {purpose}, {when}")
    first = custom_sound.split(purpose, offered.order).options[0]
    lead = offered.validation.options[0]
    if first.key == "holdout":
        text = (f"`holdout` ({USUAL:.0%}), validated by `{lead.validation}` ({lead.label}): "
                f"{first.sound.reason}")
    else:
        text = f"`{first.key}` ({lead.label if lead.validation == first.key else first.label}): " \
               f"{first.sound.reason}"
    return Case(when, text, first.key)


def split(purpose: str) -> list[Case]:
    from turbotab.core.models.validation import RESAMPLE_BELOW
    from turbotab.core.seal import FLOOR, USUAL

    if purpose == "inference":
        return [_split_case("inference", "at any size", "regression", 5_000, holdout_first=False)]
    small = RESAMPLE_BELOW // 4
    large = RESAMPLE_BELOW + RESAMPLE_BELOW // 4
    few = FLOOR * 3  # a 20% holdout of these leaves fewer than the floor's rows
    _split_case("prediction", "rows in time order, many units", "regression", large,
                time_ordered=True, holdout_first=True)  # time order: the holdout leads at any size
    return [
        _split_case("prediction", f"no time order, fewer than {RESAMPLE_BELOW:,} units", "regression",
                    small,
                    holdout_first=False),
        _split_case("prediction", f"no time order, {RESAMPLE_BELOW:,} units or more, the usual "
                                  f"{USUAL:.0%} holdout at or above its floor ({FLOOR} rows, or {FLOOR} in "
                                  f"each class)", "regression", large, holdout_first=True),
        _split_case("prediction", f"no time order, {RESAMPLE_BELOW:,} units or more, the usual "
                                  f"{USUAL:.0%} holdout below its floor (a rare event)", "binary", large,
                    [large - FLOOR, FLOOR], holdout_first=False),
        _split_case("prediction", f"rows in time order (the temporal seal), the usual {USUAL:.0%} "
                                  f"holdout below its floor", "regression", few, time_ordered=True,
                    holdout_first=False),
        _split_case("prediction", f"rows in time order (the temporal seal), the usual {USUAL:.0%} "
                                  f"holdout at or above its floor, whatever the number of units "
                                  f"(the latest rows held out)", "regression", small,
                    time_ordered=True, holdout_first=True),
    ]


# ── the omics normalization ──────────────────────────────────────────────────


def _normalizations(kind: str, pooled_qcs: bool = False) -> list[str]:
    """The keys the normalization finding offers, in its order, on a table of ``kind`` (with
    pooled-QC injections: a minority level whose rows barely vary across the features)."""
    import numpy as np
    import pandas as pd

    from turbotab.core.methods import omics

    columns = [f"f{i}" for i in range(max(omics.MIN_FEATURES, 30) + 1)]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.lognormal(8.0, 1.0, size=(24, len(columns))), columns=columns)
    if pooled_qcs:
        frame.iloc[:5] = frame.iloc[5:].mean().to_numpy() * rng.normal(1.0, 0.02, (5, len(columns)))
        frame.insert(0, "sample_type", ["QC"] * 5 + ["sample"] * 19)
    oc = SimpleNamespace(has=lambda c: True, frame=frame)
    options = omics._offer({"id": omics.FINDING}, {"kind": kind, "columns": columns,
                                                   "n_zero": 0}, oc)
    return [o.key for o in options]


def omics_normalization(purpose: str) -> list[Case]:
    counts, intensities = _normalizations("counts"), _normalizations("intensities")
    with_qcs = _normalizations("intensities", pooled_qcs=True)
    _check(bool(counts) and bool(intensities), "both kinds offer a normalization")
    _check(set(with_qcs) > set(intensities), "pooled QCs add a normalization to their reference")
    return [
        Case("raw counts (sequencing): the count finding offers "
             + ", ".join(f"`{k}`" for k in counts), f"`{counts[0]}`", counts[0]),
        Case("raw intensities (mass spectrometry): the intensity finding offers "
             + ", ".join(f"`{k}`" for k in intensities), f"`{intensities[0]}`", intensities[0]),
        Case("raw intensities with pooled-QC injections: "
             + ", ".join(f"`{k}`" for k in with_qcs), f"`{with_qcs[0]}`", with_qcs[0]),
    ]


# ── the form and the exposure's transform ────────────────────────────────────


def functional_form(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None  # under prediction the form is the in-fold rule's (spline_rule, inner_cv_form)
    from turbotab.core.methods.exposure_form import MASS_AT_ZERO, knots_by_rule, proposal

    k = knots_by_rule(None)
    plain = proposal("exposure", purpose, k, mass_at_zero=False)
    zero = proposal("exposure", purpose, k, mass_at_zero=True)
    confounder = proposal("confounder", purpose, k, mass_at_zero=True)
    _check(plain["form"] != zero["form"], "an exposure with a mass at zero is proposed apart")
    rule = "knots by Harrell's rule on the effective sample size (`exposure_form.knots_by_rule`)"
    return [
        Case("a continuous exposure or confounder (the form card's proposal, one tap for all)",
             f"`{plain['form']}`, {rule}", plain["form"]),
        Case(f"an exposure with at least {MASS_AT_ZERO:.0%} of its values exactly zero and none "
             f"negative", f"`{zero['form']}`: non-consumers apart, a spline among consumers",
             zero["form"]),
        Case("a confounder with a mass at zero", f"`{confounder['form']}`, {rule}",
             confounder["form"]),
    ]


def _energy_cases(purpose: str) -> list[Case]:
    from turbotab.core import custom_sound
    from turbotab.core.methods.energy import RANKING

    first = custom_sound.energy(purpose).options[0]
    cases = []
    if first.key == "all_components":
        rest = custom_sound.energy(purpose, applicable=[m for m in RANKING[purpose]
                                                        if m != "all_components"]).options[0]
        cases.append(Case("a table with every source of energy as a column (all components "
                          "needs each)", f"`{first.key}`: {first.sound.reason}", first.key))
        cases.append(Case("a table without every source, or whose sources nest (a total beside "
                          "its parts, such as sugar inside carbohydrate: the partition check "
                          "refuses it, `proposals.partition_check`)",
                          f"`{rest.key}`: {rest.sound.reason}", rest.key))
    else:
        cases.append(Case("a table with total energy", f"`{first.key}`: {first.sound.reason}",
                          first.key))
    return cases


def energy_adjustment(purpose: str) -> list[Case]:
    return _energy_cases(purpose)


def exposure_transform(purpose: str) -> list[Case]:
    energy = _energy_cases(purpose)
    counts, intensities = _normalizations("counts")[0], _normalizations("intensities")[0]
    return [
        *[Case(f"an intake on {c.when}", f"{c.first.split(':', 1)[0]}, the energy model's first "
                                         f"(the energy-adjustment question)", c.key)
          for c in energy],
        Case("a multi-item scale", "`score`: declared with its items and key (`set_scales`), "
                                   "never ranked; no question asks it, so from the app the items "
                                   "enter on their own (the scales contract)", "score"),
        Case("raw counts or intensities", f"the omics normalization by the data's kind "
                                          f"(`{counts}` for counts, `{intensities}` for "
                                          f"intensities)", counts),
    ]


def exclusions(purpose: str) -> list[Case]:
    from turbotab.core import custom_sound

    screens = [r[0] for r in custom_sound.EXCLUSION_ROWS if r[0] != "keep_every_row"]
    every = custom_sound.exclusions(purpose, offered=screens).options[0]
    none = custom_sound.exclusions(purpose, offered=[]).options[0]
    if every.key == none.key:
        return [Case("whatever screens can run", f"`{every.key}`: {every.sound.reason}",
                     every.key)]
    return [Case("the screens can run on the table (the Goldberg screen needs energy, age, sex, "
                 "weight and height)", f"`{every.key}`: {every.sound.reason}", every.key),
            Case("no screen can run on the table", f"`{none.key}`: {none.sound.reason}",
                 none.key)]


# ── the stated responses ─────────────────────────────────────────────────────

CHECK_WORDS = {"proportional_hazards": "a failed proportional-hazards check (a Cox fit)",
               "influence": "a failed influence check (Cook's distance)"}


def diagnostics(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None
    from turbotab.core.estimand import ACTION_WORDS, DIAGNOSTIC_ACTIONS

    out = [Case(CHECK_WORDS[check], f"`{actions[0]}`: {ACTION_WORDS[actions[0]]}", actions[0])
           for check, actions in DIAGNOSTIC_ACTIONS.items()]
    out.append(Case("no check failed", "nothing is asked: the checks are reported beside the "
                                       "estimate", ""))
    return out


def evalue_sd(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None
    from turbotab.core.models.effects import estimand_sd

    _, weighted = estimand_sd([1.0, 2.0, 4.0], [1.0, 2.0, 1.0])
    _, sample = estimand_sd([1.0, 2.0, 4.0])
    return [Case("the surveyed-population answer (the analysis weights)",
                 f"`{weighted}`: the surveyed population's SD", weighted),
            Case("the sample answer (no weights)", f"`{sample}`: the analyzed rows' SD", sample)]


def unmeasured_confounding(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None
    import numpy as np
    import pandas as pd

    from turbotab.core.models import effects

    rng = np.random.default_rng(0)
    matrix = pd.DataFrame({"x": rng.normal(size=40), "z": rng.normal(size=40)})
    y = matrix["x"].to_numpy() + rng.normal(size=40)
    linear = effects.unmeasured_confounding(
        measure="mean_difference", estimate=1.0, ci_low=0.5, ci_high=1.5, se=0.25, outcome_sd=1.5,
        matrix=matrix, y=y, exposure_column="x", benchmarks={"z": ["z"]})["methods"]
    ratio = effects.unmeasured_confounding(measure="odds_ratio", estimate=1.5, ci_low=1.2,
                                           ci_high=1.9, outcome_share=0.05)["methods"]
    no_fit = effects.unmeasured_confounding(measure="mean_difference", estimate=1.0, ci_low=0.5,
                                            ci_high=1.5, se=0.25, outcome_sd=1.5)["methods"]
    _check(linear[0] != ratio[0] and no_fit[:1] == ratio[:1],
           "a least-squares difference leads with its robustness value, anything else the E-value")
    return [Case("a difference from a least-squares fit (a linear outcome's coefficient)",
                 " then ".join(f"`{m}`" for m in linear), linear[0]),
            Case("a ratio (an odds, hazard or risk ratio), or a difference with no least-squares "
                 "fit", " then ".join(f"`{m}`" for m in ratio), ratio[0])]


# ── the usual-intake model ───────────────────────────────────────────────────


def nci_usual_intake(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None  # not offered under prediction
    from turbotab.core.usual_intake import EPISODIC_SHARE, suggested_model

    rare, why_rare = suggested_model(EPISODIC_SHARE / 2)
    common, why_common = suggested_model(min(0.5, EPISODIC_SHARE * 4))
    _check(rare != common, "the share of zero recalls decides the model ranked first")
    tail = lambda why: why.split(": ", 1)[-1]  # noqa: E731 - the reason without the share's figure
    return [Case(f"a component with {EPISODIC_SHARE:.0%} of its recalls at zero or fewer",
                 f"`{rare}`: {tail(why_rare)}", rare),
            Case(f"a component with more than {EPISODIC_SHARE:.0%} of its recalls at zero",
                 f"`{common}`: {tail(why_common)}", common)]


# ── the causal lane ──────────────────────────────────────────────────────────

# The conditions the causal lane ranks its estimators by (``causal.options``): the exposure's kind,
# then the outcome (the lane takes a numeric or a yes/no one), with, for a numeric outcome, whether
# the candidates are many for n and whether the estimate is the surveyed population's.
CAUSAL_EXPOSURES = (("binary", "a yes/no exposure"), ("continuous", "a continuous exposure"))
CAUSAL_OUTCOMES = (("regression", True, False, "a numeric outcome with many candidates for n"),
                   ("regression", False, False, "a numeric outcome with few candidates for n"),
                   ("regression", False, True, "a numeric outcome, the surveyed population"),
                   ("binary", False, False, "a yes/no outcome"))


def causal_rankings() -> list[tuple[str, list[Any]]]:
    """(the condition in words, the lane's labeled options in the order it offers them), for every
    condition it ranks by: what the methods reference and the packets print."""
    from turbotab.core.causal import options

    out = []
    for kind, exposure in CAUSAL_EXPOSURES:
        for task, many, population, outcome in CAUSAL_OUTCOMES:
            out.append((f"{exposure}, {outcome}", options(kind, task, many, population)))
    return out


def causal(purpose: str) -> list[Case] | None:
    if purpose != "inference":
        return None  # the lane is not offered under prediction (``causal.plan_reason``)
    out = []
    for when, ranked in causal_rankings():
        _check(ranked[-1].key == "none", "the primary model alone is the lane's last option")
        lane = [f"`{o.key}`" + (f" ({o.sound.verdict})" if o.sound.verdict != "sound" else "")
                for o in ranked if o.key != "none"]
        out.append(Case(when, "`none` (the primary model only), stated and not asked; “Ask me "
                              "anyway” opens the lane, which offers " + ", then ".join(lane),
                        "none"))
    return out


# The contracts the app reaches only through another question (their key → its key): the four
# causal estimators are the causal lane's options, so what the app offers first is the lane's.
THROUGH: dict[str, str] = {"dml_irm": "causal", "dml_plr": "causal", "pds_lasso": "causal",
                           "tmle": "causal"}


# Each method key (a contract's, or a question labeled outside the registry) the app ranks by the
# data, with the function computing its cases for a purpose (None: the static rank applies).
BY_CONDITION: dict[str, Callable[[str], list[Case] | None]] = {
    "effect_measure": effect_measure,
    "split": split,
    "omics_normalization": omics_normalization,
    "functional_form": functional_form,
    "exposure_transform": exposure_transform,
    "energy_adjustment": energy_adjustment,
    "exclusions": exclusions,
    "diagnostics": diagnostics,
    "evalue_sd": evalue_sd,
    "unmeasured_confounding": unmeasured_confounding,
    "nci_usual_intake": nci_usual_intake,
    "causal": causal,
}


def cases(key: str, purpose: str) -> list[Case] | None:
    """The cases the app ranks ``key`` by under ``purpose`` (a contract reached only through
    another question: that question's), or None where the static rank holds."""
    fn = BY_CONDITION.get(THROUGH.get(key, key))
    return fn(purpose) if fn is not None else None


__all__ = ["BY_CONDITION", "CAUSAL_EXPOSURES", "CAUSAL_OUTCOMES", "Case", "PremiseFailed", "THROUGH",
           "causal_rankings", "cases"]

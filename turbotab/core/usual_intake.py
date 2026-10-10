"""Routing for the NCI usual-intake method: when it is offered, its refusals, and its contract.

**The estimand.** The distribution of usual intake (percentiles, and the share below a cut-off such
as the EAR) in the population the recalls describe. It is its own estimand, distinct from an
association between intake and an outcome: MODELING_SEQUENCE §2 routes association with an
error-prone intake to regression calibration (``set_measurement_error``), and the NCI method's
person-level predictions are not measured usual intakes (NUTRITION_PACK §03: "NCI explicitly warns
INDIVINT output does not represent individual usual intake"), so this estimand never feeds a
coefficient and no person's own usual intake is reported.

**When it is offered** (the ``usual_intake`` stage's ``offer``): under the dietary lens, for an
inference purpose, when at least two people have two or more recalls (MIXTRAN v2.1: "there must be
at least two subjects with at least two positive recalls"). The recalls are either a long table's
rows repeated per person (the grain answered "repeated" and the rows read, settled, as repeats of one
measurement) or a wide table's recall-day columns, which the user names (``DR1TPROT`` beside
``DR2TPROT`` is proposed, never assumed). Under prediction it is not offered: a usual-intake
distribution describes a population, and the prediction reads each person's recalls as they were
measured. Rows that repeat as time points are visits, not replicates of one usual intake.

**Which model ranks first.** A dietary component reported on nearly every recall goes to the
amount-only model (Tooze et al. 2010); one with many zero days to the two-part model for episodically
consumed foods (Tooze et al. 2006; MODELING_SEQUENCE §2: "usual intake for an episodically consumed
food goes to the NCI two-part model"). TurboTab ranks the two-part model first above
:data:`EPISODIC_SHARE` of recalls at zero; that threshold is its own convention, stated as such.
The amount-only model on such a food is block-and-record: it runs, and the record says that its zero
days became half the smallest amount. The two-part model with no zero day at all is refused: its
probability part has nothing to estimate.

**Whole population or consumers only** (STROBE-nut nut-14: "Specify if food consumption of total
population or consumers only were used to obtain results"). The whole population is the default; the
two-part model then takes everyone to consume the food on some days (Kipnis et al. 2009: "Our two-part
model assumes that each food is ultimately consumed by all individuals"). Consumers only is an
estimand change and needs a column that says who ever consumes it (a food-frequency answer); people
with no consumption on their two recall days are not non-consumers, so "consumers on the recall
days" is refused.

**Cut-offs.** The share below the EAR is the EAR cut-point estimate of the prevalence of inadequacy
(Institute of Medicine 2000), valid only for usual intake, which is why it lives here. An AI cannot
yield a prevalence of inadequacy (NUTRITION_PACK §07: "the app must refuse to compute one and say
why"); total energy has no EAR. The UL's share is the share above it.

The EAR cut-point's own conditions (Institute of Medicine 2000, *DRI: Applications in Dietary
Assessment*, ch. 4, read 2026-10-05: it "works best ... when: 1. intakes and requirements are
independent 2. the requirement distribution is symmetrical around the EAR 3. the variance in intakes
is larger than the variance of requirements 4. true prevalence of inadequacy in the population is no
smaller than 8 to 10 percent or no larger than 90 to 92 percent") are enforced where the app can
and asked where it cannot:

* *One EAR for everyone analyzed.* An EAR is "the median requirement of a nutrient for a given life
  stage and gender group" (ch. 3), and NUTRITION_PACK §07 [SETTLED]: "Reference intakes join on age
  band, sex, pregnancy and lactation status." The recalls cannot say which group each participant
  belongs to, so the answer says it (``ear_for_all``). Unanswered, the share is computed and reported
  as a plain share, and the prevalence label is blocked and recorded with its exits (MODELING_SEQUENCE
  §4's rung). Each group's own distribution is the INBOX's subgroup item.
* *A symmetric requirement.* "when the distribution of requirements is known to be asymmetrical, as
  for iron in menstruating women, the probability approach, not the EAR cut-point method, is
  recommended" (ch. 4); NUTRITION_PACK §07 lists it among the "Exceptions that must be hard-coded".
  A component whose name reads as iron (:func:`turbotab.core.recognizers.read_nutrient`; a name is a
  proposal, so the refusal names the reading and its exits keep every number) is refused an EAR until
  the answer says its requirement is symmetric in these participants (``ear_symmetric``: no
  menstruating women among them). The probability approach is not implemented here.
* *Energy*: refused (no EAR). *The prevalence's range*: an estimate below 10% or above 90% carries
  the concern that the cut-point is least accurate there.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

EPISODIC_SHARE = 0.05  # zero recalls above this share rank the two-part model first (a convention)
MIN_REPEATERS = 2  # MIXTRAN v2.1: "at least two subjects with at least two positive recalls"

ESTIMAND = ("The distribution of usual intake (its percentiles, and the share below a cut-off such "
            "as the EAR) in the population the recalls describe, with day-to-day variation removed "
            "by the NCI method (Tooze et al. 2006; Tooze et al. 2010).")
ASSOCIATION = ("This is a comparison of its own. An association between an intake and an outcome is a "
               "different question, answered by regression calibration of what you study (Freedman "
               "et al. 2011; set_measurement_error), never by these distributions, and no person's "
               "own usual intake is estimated here.")
POPULATION_QUESTION = ("Whole population or consumers only (STROBE-nut nut-14: \"Specify if food "
                       "consumption of total population or consumers only were used to obtain "
                       "results\"). Consumers only needs a column that says who ever consumes the "
                       "food, such as a food-frequency answer.")

NOT_DIETARY = "The dietary lens is off, so usual intake is not offered."
LENS_UNANSWERED = "The lens is not answered yet."
PURPOSE_UNANSWERED = "The purpose is not answered yet."
PREDICTION = ("A usual-intake distribution describes a population: it is an inference question about a population. "
              "Under prediction the model reads each person's recalls as they were measured, so "
              "nothing here would change a prediction.")
TIME_POINTS = ("The rows repeat as time points, not as recalls of one usual intake, so their spread "
               "is change over time rather than day-to-day variation.")
REPEATS_UNSETTLED = ("Whether the repeated rows are recalls of one usual intake or time points is "
                     "not settled yet; it is asked with the repeats.")
NO_RECALL_DAYS = ("Each person has one row and no columns read as recall days. Name the columns "
                  "holding each recall day (for example `DR1TPROT` and `DR2TPROT`) to estimate a "
                  "usual-intake distribution.")
ENERGY_NO_EAR = ("Total energy has no EAR: its reference is the estimated energy requirement, and "
                 "energy intake and requirement are correlated, so no cut-point prevalence applies "
                 "(NUTRITION_PACK §07: \"Energy has no EAR-style cut-point (use EER)\").")
TOO_FEW_REPEATS = ("Fewer than two people have two or more recalls, so day-to-day variation cannot "
                   "be told apart from differences between people (MIXTRAN v2.1: \"there must be at "
                   "least two subjects with at least two positive recalls\").")
SURVEY_UNANSWERED = ("Survey design columns are in this table, and whether the usual-intake "
                     "distribution describes the surveyed population or these participants is not "
                     "answered, so no distribution is estimated.")
LONELY_PSU = ("The standard errors for the surveyed population replicate the design, which needs "
              "two or more PSUs in every stratum, so no distribution is estimated for it "
              "(MODELING_SEQUENCE §4: block and record).")
IRON_SKEWED = ("Iron's requirement distribution is skewed in menstruating women, so the share below "
               "its EAR is no prevalence of inadequacy for them: Institute of Medicine 2000, \"when "
               "the distribution of requirements is known to be asymmetrical, as for iron in "
               "menstruating women, the probability approach, not the EAR cut-point method, is "
               "recommended\" (NUTRITION_PACK §07: an exception that \"must be hard-coded\"). The "
               "probability approach is not implemented here.")
EAR_UNANSWERED = ("The share is reported as a plain share, not a prevalence of inadequacy: an EAR is "
                  "\"the median requirement of a nutrient for a given life stage and gender group\" "
                  "(Institute of Medicine 2000), and whether this cut-off is the EAR of every "
                  "participant's DRI life-stage group (age band, sex, pregnancy and lactation "
                  "status) is not answered. Across groups with different EARs one cut-off misplaces "
                  "the requirement of every group but one.")
EAR_CONDITIONS = ("The EAR cut-point estimate assumes what the recalls cannot show (Institute of "
                  "Medicine 2000): \"intakes and requirements are independent\", \"the requirement "
                  "distribution is symmetrical around the EAR\", and \"the variance in intakes is "
                  "larger than the variance of requirements\".")
# Institute of Medicine 2000: the cut-point "works best" when the "true prevalence of inadequacy in
# the population is no smaller than 8 to 10 percent or no larger than 90 to 92 percent".
EAR_RANGE = (0.10, 0.90)

ASSUMPTIONS = (
    "Each recall measures usual intake without bias on the transformed scale: the model removes "
    "day-to-day variation, not a person's own reporting bias. Keogh et al. 2020 (STRATOS): \"many "
    "investigators, in the absence of anything better, have adopted the assumption that 24-hour "
    "recalls provide unbiased measurements\".",
    "Person effects are normal on the Box-Cox scale (jointly normal for the two-part model's two "
    "parts), and the day-to-day errors are independent of them and of each other.",
    "The distribution describes the population; no person's own usual intake is estimated "
    "(NUTRITION_PACK §03: NCI \"warns INDIVINT output does not represent individual usual intake\").",
)
TWO_PART_ASSUMPTION = ("Everyone is taken to consume the food on some days: Kipnis et al. 2009, "
                       "\"Our two-part model assumes that each food is ultimately consumed by all "
                       "individuals, so that T_i > 0\"; never-consumers are not modeled.")


# ── a wide table's recall-day columns, proposed from their names ─────────────

_DAY_PATTERNS = (
    re.compile(r"^(DR)([1-9])(T.+)$", re.I),  # NHANES total nutrient intakes: DR1TPROT, DR2TPROT
    re.compile(r"^(.+?)[_.\s-]?(?:day|d|recall)[_.\s-]?([1-9])$", re.I),  # protein_day1, kcal_d2
    re.compile(r"^(?:day|d|recall)[_.\s-]?([1-9])[_.\s-](.+)$", re.I),  # day1_protein
)


def day_columns(columns: Sequence[str]) -> list[dict[str, Any]]:
    """Groups of columns whose names read as one quantity's recall days, two or more days each,
    ordered by day: ``[{"label": "TPROT", "days": ["DR1TPROT", "DR2TPROT"]}, …]``. A name is no
    corroboration (BLUEPRINT §14.3): these are proposals the user confirms by naming the days."""
    groups: dict[str, dict[int, str]] = {}
    for column in columns:
        name = str(column)
        for pattern in _DAY_PATTERNS:
            m = pattern.match(name)
            if not m:
                continue
            parts = m.groups()
            if len(parts) == 3:  # NHANES: prefix, day, rest
                label, day = parts[0] + "·" + parts[2], int(parts[1])
            elif pattern is _DAY_PATTERNS[1]:
                label, day = parts[0], int(parts[1])
            else:
                label, day = parts[1], int(parts[0])
            groups.setdefault(label.lower(), {})[day] = name
            break
    out = []
    for label, days in groups.items():
        if len(days) >= 2 and min(days) == 1:
            ordered = [days[d] for d in sorted(days)]
            out.append({"label": ordered[0], "days": ordered})
    return out


def suggested_model(zero_share: float) -> tuple[str, str]:
    """The model ranked first for a component with this share of zero recalls, and why."""
    if zero_share > EPISODIC_SHARE:
        return "two_part", (f"{zero_share:.0%} of recalls report none of it: an episodically consumed "
                            f"food, so the two-part model ranks first (Tooze et al. 2006).")
    return "amount_only", (f"{zero_share:.0%} of recalls report none of it: consumed nearly every day, "
                           f"so the amount-only model ranks first (Tooze et al. 2010).")


# ── the refusals (``set_usual_intake``) ──────────────────────────────────────


def _state(ctx: Any) -> Any:
    from turbotab.core.decisions import _state as state_of

    return state_of(ctx)


def _none(decision: Any) -> dict[str, Any]:
    return {"kind": "set_usual_intake", "nutrient": decision.nutrient, "model": "none"}


# ── exits: each a decision the client can post (or None: a question to answer) ──


def leave_out(nutrient: str) -> dict[str, Any]:
    return {"label": "Leave usual intake out",
            "decision": {"kind": "set_usual_intake", "nutrient": nutrient, "model": "none"}}


def answer(nutrient: str, spec: Any, **update: Any) -> dict[str, Any]:
    """The recorded answer for ``nutrient`` with ``update`` applied, as a postable decision."""
    body = spec.model_dump(mode="json", exclude={"kind", "nutrient"})
    return {"kind": "set_usual_intake", "nutrient": nutrient, **body, **update}


def cutoff_exits(nutrient: str, spec: Any) -> list[dict[str, Any]]:
    """No cut-off, or the share below it as a plain share (no prevalence of inadequacy)."""
    clear = {"ear_for_all": False, "ear_symmetric": False}
    return [{"label": "Show the distribution without a cut-off",
             "decision": answer(nutrient, spec, cutoff=None, cutoff_kind=None, **clear)},
            {"label": "Report the share below it as a plain share",
             "decision": answer(nutrient, spec, cutoff_kind="other", **clear)}]


def reads_as_iron(nutrient: str, days: Sequence[str] = ()) -> bool:
    """Whether the component's name (or a recall day's) reads as iron: a proposal from the name
    (BLUEPRINT §14.3), used only to ask before a prevalence is labeled, never to change a number."""
    from turbotab.core.recognizers import AmbiguousNutrient, read_nutrient

    for name in [nutrient, *days]:
        try:
            reading = read_nutrient(name)
        except AmbiguousNutrient:
            continue
        if reading is not None and reading.nutrient == "iron":
            return True
    return False


def iron_exits(nutrient: str, spec: Any) -> list[dict[str, Any]]:
    return [*cutoff_exits(nutrient, spec),
            {"label": "No participant is a menstruating woman: keep the EAR cut-point",
             "decision": answer(nutrient, spec, ear_symmetric=True)}]


def ear_exits(nutrient: str, spec: Any) -> list[dict[str, Any]]:
    return [{"label": "It is the EAR of every participant's DRI life-stage group (age band, sex, "
                      "pregnancy and lactation status)",
             "decision": answer(nutrient, spec, ear_for_all=True)},
            *cutoff_exits(nutrient, spec)]


def _columns_exist(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import ROW_ID, Refusal, _and, _columns_of

    columns = _columns_of(ctx)
    if columns is None:
        return
    named = [*(decision.days or ([] if decision.model == "none" else [decision.nutrient])),
             *([decision.order_column] if decision.order_column else []), *decision.weekend,
             *([decision.consumer_column] if decision.consumer_column else [])]
    unknown = [c for c in named if c not in columns or c == ROW_ID]
    if unknown:
        raise Refusal("unknown_column", f"This dataset has no column named {_and(unknown)}.",
                      exits=[{"label": "Name the columns again", "decision": None},
                             {"label": "Leave usual intake out", "decision": _none(decision)}])


def _for_inference(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import Refusal

    state = _state(ctx)
    if decision.model == "none" or state is None:
        return
    if state.lens is not None and "dietary" not in state.lens:
        raise Refusal("not_dietary", NOT_DIETARY,
                      exits=[{"label": "Add the dietary lens", "decision": None},
                             {"label": "Leave usual intake out", "decision": _none(decision)}])
    if state.purpose == "prediction":
        raise Refusal("not_for_prediction", PREDICTION,
                      exits=[{"label": "Change the purpose to inference", "decision": None},
                             {"label": "Leave usual intake out", "decision": _none(decision)}])


def _consumers_are_named(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import Refusal

    if decision.model == "none" or decision.population != "consumers" or decision.consumer_column:
        return
    whole = decision.model_copy(update={"population": "whole"})
    raise Refusal(
        "consumers_unnamed",
        "Consumers only is a comparison of its own (STROBE-nut nut-14), and a person with no "
        "consumption on their recall days may still consume the food on others, so the recalls "
        "cannot say who the consumers are. Name a column that says who ever consumes it (a "
        "food-frequency answer), or estimate for the whole population.",
        exits=[{"label": "Estimate for the whole population", "decision": whole},
               {"label": "Name the column that marks consumers", "decision": None}])


def _cutoff_fits_the_reference(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import Refusal
    from turbotab.core.readings import settled_roles

    if decision.model == "none" or decision.cutoff is None:
        return
    clear = {"ear_for_all": False, "ear_symmetric": False}
    plain = decision.model_copy(update={"cutoff_kind": "other", **clear})
    dropped = decision.model_copy(update={"cutoff": None, "cutoff_kind": None, **clear})
    if decision.cutoff_kind == "AI":
        raise Refusal(
            "ai_no_prevalence",
            "An Adequate Intake is not a requirement's median, so the share of usual intakes below it "
            "is no prevalence of inadequacy (Institute of Medicine 2000); NUTRITION_PACK §07: \"I can "
            "show the distribution against the AI, but I cannot compute a prevalence of inadequacy "
            "from an AI, and neither can anyone else.\"",
            exits=[{"label": "Show the distribution without a cut-off", "decision": dropped},
                   {"label": "Report the share below it as a plain share", "decision": plain}])
    state = _state(ctx)
    if decision.cutoff_kind == "EAR" and state is not None:
        energy = [c for c, r in settled_roles(state).items() if r == "energy"]
        named = decision.days or [decision.nutrient]
        if any(c in energy for c in named):
            raise Refusal(
                "energy_no_ear", ENERGY_NO_EAR,
                exits=[{"label": "Show the distribution without a cut-off", "decision": dropped},
                       {"label": "Report the share below it as a plain share", "decision": plain}])
    if (decision.cutoff_kind == "EAR" and not decision.ear_symmetric
            and reads_as_iron(decision.nutrient, decision.days)):
        named = decision.days[0] if decision.days else decision.nutrient
        raise Refusal(
            "iron_skewed_requirement",
            f"`{named}` reads as iron by its name. {IRON_SKEWED}",
            exits=iron_exits(decision.nutrient, decision))


def _register() -> None:
    from turbotab.core.decisions import register_validator

    register_validator("set_usual_intake", _columns_exist)
    register_validator("set_usual_intake", _for_inference)
    register_validator("set_usual_intake", _consumers_are_named)
    register_validator("set_usual_intake", _cutoff_fits_the_reference)


_register()


# ── the decision's sentence ──────────────────────────────────────────────────


def _sentence(d: Any, state: Any, ctx: Any) -> str:
    from turbotab.core.voice import listing, tick

    if d.model == "none":
        return f"No usual-intake distribution was estimated for {tick(d.nutrient)}"
    model = ("an amount-only model" if d.model == "amount_only"
             else "a two-part model for an episodically consumed food")
    source = (f" from the recall days {listing(d.days, limit=6)}" if d.days else "")
    whom = (f"consumers only (those {tick(d.consumer_column)} marks)"
            if d.population == "consumers" and d.consumer_column else "the whole population")
    cut = ""
    if d.cutoff is not None:
        kind = {"EAR": "the EAR", "UL": "the UL", "AI": "the AI"}.get(d.cutoff_kind or "", "a cut-off")
        cut = f", with the share {'above' if d.cutoff_kind == 'UL' else 'below'} {kind} ({d.cutoff:g})"
        if d.cutoff_kind == "EAR" and d.ear_for_all:
            cut += ", answered as the EAR of every participant's DRI life-stage group"
        if d.cutoff_kind == "EAR" and d.ear_symmetric:
            cut += (" and" if d.ear_for_all else ",") + (" with a requirement answered as symmetric "
                                                        "in them (no menstruating women)")
    # The replication (and so the number of replicates) follows the survey answer; the analysis'
    # own methods sentence states it.
    return (f"The usual-intake distribution of {tick(d.nutrient)}{source} was estimated by the NCI "
            f"method with {model}, for {whom}{cut}")


def _register_sentence() -> None:
    from turbotab.core.voice import register_sentence

    register_sentence("set_usual_intake")(_sentence)


_register_sentence()


# ── the method contract (BLUEPRINT §13) ──────────────────────────────────────


def _methods_sentence(**kwargs: Any) -> str:
    from turbotab.core.methods.usual_intake import methods_sentence

    return methods_sentence(**kwargs)


def _contract() -> Any:
    from turbotab.core.contracts import MethodContract, Option, Relation, register_contract

    return register_contract(MethodContract(
        key="nci_usual_intake",
        label="Usual-intake distribution (NCI method)",
        slot="model",
        scope="training_fold",
        scope_note=("It learns from study rows (every eligible person's recalls), so it is fit on "
                    "the analysis rows; under inference those are every eligible row (BLUEPRINT §12 "
                    "ruling 3). It reads no outcome and feeds no other model."),
        needs=("the dietary lens", "two or more recalls for at least two people",
               "a dietary component's recalls (a long table's repeated rows or a wide table's day "
               "columns)", "optional: an order column, a weekend indicator, a column marking "
               "consumers, the survey design",
               "for a prevalence of inadequacy below an EAR: the answer that it is every "
               "participant's DRI life-stage group's EAR (and, for iron, that no participant is a "
               "menstruating woman)"),
        question=("Estimate the usual-intake distribution of a dietary component (percentiles, "
                  "and the share below a cut-off)?"),
        place=("MODELING_SEQUENCE §1 step 2, exposure and estimand: its own estimand, beside any "
               "association analysis"),
        decision="set_usual_intake",
        stage="usual_intake",
        options=(
            Option("amount_only", "Amount-only model",
                   "the NCI method for nutrients consumed nearly every day (Tooze et al. 2010)",
                   {"inference": "sound when zero days are rare; above 5% of recalls at zero it "
                                 "turns those days into half the smallest amount (block and "
                                 "record)",
                    "prediction": "not offered: no prediction reads a usual-intake distribution"},
                   {"inference": "recommended", "prediction": "not_offered"}),
            Option("two_part", "Two-part model (episodically consumed foods)",
                   "the NCI method for episodically consumed foods (Tooze et al. 2006; Kipnis et "
                   "al. 2009)",
                   {"inference": "sound for a food with many zero days; refused with no zero day",
                    "prediction": "not offered"},
                   {"inference": "recommended", "prediction": "not_offered"}),
            Option("mean_of_days", "The distribution of each person's mean of recalls",
                   "customary in older surveys (\"We averaged the two 24-hour recalls to obtain "
                   "usual intake\", NUTRITION_PACK §03)",
                   {"inference": "unsound for percentiles and prevalence: the tails are too fat; "
                                 "shown only beside the usual-intake distribution, labeled",
                    "prediction": "not offered"},
                   {"inference": "rank_lower", "prediction": "not_offered"}),
        ),
        leash={"inference": "recommended", "prediction": "not_offered"},
        storyboard=(
            "Box-Cox-transform each recall (λ estimated with the model)",
            "Fit the mixed model: between-person and within-person variance (and, for an episodic "
            "food, the probability of a consumption day, correlated with the amount)",
            "Set the nuisance covariates to a first recall and the 4/7 weekday, 3/7 weekend mix",
            "Remove the within-person variance and back-transform by quadrature",
            "Read the percentiles and the share below the cut-off; replicate for standard errors",
        ),
        sentence=_methods_sentence,
        relations=(
            Relation("enables", "repeats_offer",
                     "the usual-intake distribution is offered as its own estimand",
                     condition="the dietary lens, an inference purpose and two or more recalls "
                               "for at least two people",
                     id="repeats_offer"),
            Relation("conflicts", "association_is_calibration",
                     "is not answered by the usual-intake distribution or any person's predicted "
                     "usual intake",
                     condition="an association between the intake and an outcome",
                     id="association_is_calibration",
                     rung="refused",
                     exits=("regression calibration of what you study (set_measurement_error)",)),
            Relation("implies", "zeros_two_part",
                     "the two-part model ranks first; the amount-only model is recorded with the "
                     "concern that its zero days became half the smallest amount",
                     condition="zero days above 5% of a component's recalls",
                     id="zeros_two_part"),
            Relation("conflicts", "no_zero_no_two_part",
                     "the two-part model's probability part has nothing to estimate (refused)",
                     condition="a component reported on every recall",
                     id="no_zero_no_two_part",
                     rung="refused",
                     exits=("the amount-only model",)),
            Relation("implies", "population_design",
                     "the fit and the distribution are weighted, and the standard errors come from "
                     "Fay's balanced repeated replication (two PSUs per stratum) or a bootstrap of "
                     "PSUs within strata, stated in the sentence",
                     condition="a survey design answered as the surveyed population",
                     id="population_design"),
            Relation("implies", "sample_attestation",
                     "unweighted estimates, a bootstrap over participants, and the attestation in "
                     "the sentence",
                     condition="a survey design answered as these participants",
                     id="sample_attestation"),
            Relation("conflicts", "consumers_need_a_column",
                     "refused: the recall days cannot say who the consumers are",
                     condition="consumers only with no column that says who ever consumes the food",
                     id="consumers_need_a_column",
                     rung="refused",
                     exits=("the whole population, or a column marking consumers (nut-14 stated)",)),
            Relation("conflicts", "prediction_not_offered",
                     "usual intake is not offered and its answer is refused",
                     condition="a prediction purpose",
                     id="prediction_not_offered",
                     rung="refused",
                     exits=("change the purpose to inference",)),
            Relation("conflicts", "time_points_not_recalls",
                     "they are not recalls of one usual intake: not offered, and a recorded answer "
                     "is not applied",
                     condition="rows that repeat as time points",
                     id="time_points_not_recalls",
                     rung="refused",
                     exits=("answer the repeats as repeated measurements",)),
            Relation("conflicts", "ai_no_prevalence",
                     "no prevalence of inadequacy is computed (refused)",
                     condition="a cut-off that is an Adequate Intake",
                     id="ai_no_prevalence",
                     rung="refused",
                     exits=("no cut-off, or a plain share below it",)),
            Relation("conflicts", "energy_no_ear",
                     "refused: energy has no EAR",
                     condition="an EAR for total energy",
                     id="energy_no_ear",
                     rung="refused",
                     exits=("no cut-off, or a plain share below it",)),
            Relation("conflicts", "ear_for_every_group",
                     "the share below it is reported as a plain share and the prevalence of "
                     "inadequacy is blocked and recorded until the cut-off is answered as the EAR "
                     "of every participant's DRI life-stage group",
                     condition="an EAR cut-off not answered as every participant's group's EAR",
                     id="ear_for_every_group",
                     rung="block_and_record",
                     exits=("answer that it is every participant's group's EAR, a plain share, or "
                            "no cut-off",)),
            Relation("conflicts", "iron_skewed_requirement",
                     "refused: iron's requirement is skewed in menstruating women, where the "
                     "probability approach applies, not the cut-point",
                     condition="an EAR for a component whose name reads as iron, its requirement "
                               "not answered as symmetric in these participants",
                     id="iron_skewed_requirement",
                     rung="refused",
                     exits=("no cut-off, a plain share, or the answer that no participant is a "
                            "menstruating woman",)),
            Relation("conflicts", "survey_unanswered",
                     "nothing is estimated until the survey question is answered",
                     condition="survey design columns in the table and the survey question "
                               "unanswered",
                     id="survey_unanswered",
                     rung="block_and_record",
                     exits=("answer the survey question, or the sample-only attestation",)),
            Relation("conflicts", "lonely_psu",
                     "no design-based variance: the distribution is blocked and recorded",
                     condition="the surveyed population with a stratum of one PSU",
                     id="lonely_psu",
                     rung="block_and_record",
                     exits=("the sample-only attestation",)),
            Relation("invalidates", "structure_invalidates",
                     "the recorded answer is re-read against the new structure and not applied "
                     "where the rows are no longer recalls; never silently kept",
                     condition="the repeats answer changing after the usual-intake answer",
                     id="structure_invalidates"),
        ),
        sources=("Tooze et al. 2006, J Am Diet Assoc 106:1575 (PMC2517157)",
                 "Tooze et al. 2010, Stat Med 29:2857 (PMC3865776)",
                 "Kipnis et al. 2009, Biometrics 65:1003 (PMC2881223)",
                 "NCI MIXTRAN and DISTRIB macros v2.1 (epi.grants.cancer.gov/diet/usualintakes)",
                 "Lachat et al. 2016, STROBE-nut, PLoS Med 13:e1002036 (nut-14)",
                 "Institute of Medicine 2000, Dietary Reference Intakes: Applications in Dietary "
                 "Assessment"),
    ))


CONTRACT = _contract()

__all__ = ["ASSOCIATION", "ASSUMPTIONS", "CONTRACT", "EAR_CONDITIONS", "EAR_RANGE", "EAR_UNANSWERED",
           "ENERGY_NO_EAR", "EPISODIC_SHARE", "ESTIMAND", "IRON_SKEWED", "LONELY_PSU",
           "MIN_REPEATERS", "NOT_DIETARY", "NO_RECALL_DAYS", "POPULATION_QUESTION", "PREDICTION",
           "REPEATS_UNSETTLED", "SURVEY_UNANSWERED", "TIME_POINTS", "TOO_FEW_REPEATS",
           "TWO_PART_ASSUMPTION", "answer", "cutoff_exits", "day_columns", "ear_exits",
           "iron_exits", "leave_out", "reads_as_iron", "suggested_model"]

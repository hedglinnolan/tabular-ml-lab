"""The scales question (MS8): its method contract, its leash and its sentences.

``set_scales`` declares the multi-item scales scored as predictors. The arithmetic (scoring, ω, α,
the test–retest ICC, the conditional regression calibration) and its sources are in
:mod:`turbotab.core.methods.scales`; the score is a row-local pipeline step
(:class:`~turbotab.core.methods.scales.ScaleScorer`, run after the in-fold fill and before the energy
step), and the ``scales`` stage (:mod:`turbotab.core.stages.scales`) estimates each scale's
reliability and, under inference, its corrected coefficient as a declared secondary analysis.

**The leash** (MODELING_SEQUENCE §4, BLUEPRINT §11.3):

* disattenuation by α or ω of a formative index is **refused** under inference, with two exits: a
  test–retest ICC from a repeat administration, or a calibration substudy against a reference
  measure (ruling 8; Reedy et al. 2018); the uncorrected estimate is the third way forward;
* any correction is **refused** under prediction: the deployed model sees the same error-prone score
  (the review's correction (4), quoting STRATOS §1: "In studies in which the aim is instead to derive
  a prediction model, the considerations surrounding error-prone variables can be quite different");
* an item holding a value outside the instrument's response scale is **refused** until the code is
  recoded to missing (the sentinel repair) or the response scale is corrected: a code counted as an
  answer moves every score;
* a functional form recorded on an item is **re-asked** (§2, "a domain transform … invalidates the
  functional-form answer"): scoring replaces the item, so its form no longer applies.

The question is not yet one of the Router's (``interview.QUESTION_KEYS``): the modeling sequence's
questions after the seal are the M3.5 build. It is answered through the decisions API, as the
measurement-error answer is, and its routing is declared in :data:`CONTRACT`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from turbotab.core.contracts import MethodContract, Option, Relation, register_contract
from turbotab.core.decisions import (
    ROW_ID,
    Refusal,
    SetExposureForm,
    SetScales,
    _and,
    _columns_of,
    _state,
    _store_of,
    _target_of,
    _UNKNOWN,
    left_out,
    register_validator,
)

PREDICTOR_ROLES = ("exposure", "covariate")  # the roles a scale's items may take (never energy)
RETEST_SUFFIXES = ("_t2", "_2", "_retest", "_rt", "_r", ".2", "_time2", "_followup")

# ── the method contract (BLUEPRINT §13) ───────────────────────────────────────

TRANSIENT = ("A reliability from internal consistency omits transient error, so the correction "
             "under-corrects (Schmidt, Le & Ilies 2003).")
APPROXIMATE = ("For a logistic or proportional-odds outcome, substituting the calibrated score is an "
               "approximation (Carroll et al. 2006, §4.2).")
SECONDARY = ("A declared secondary analysis beside the uncorrected estimate; the test of no "
             "association stays the uncorrected model's (Freedman et al. 2011).")
PREDICTION = ("Under prediction the deployed model sees a new row's score with the same error, so "
              "its predictions need no correction; correcting a coefficient is an inference "
              "question.")
FORMATIVE = ("A formative index (a diet-quality score, an FFQ-derived score) is defined by its "
             "components, not caused by one construct: its internal consistency is not the error "
             "that dilutes the regression, so dividing by α or ω over-corrects (Reedy et al. 2018: "
             "the HEI-2015's components showed “at least four dimensions”). Its "
             "reliability comes from a test–retest ICC of a repeat administration or from a "
             "calibration substudy.")

CONTRACT = register_contract(MethodContract(
    key="scales",
    label="Scale scores, their reliability and the correction for measurement error",
    decision="set_scales",
    slot="in-fold steps",
    scope={
        "scoring by the instrument's key (reverse coding; the sum or the mean)": "row-local",
        "item-level imputation before scoring": "training fold",
        "reliability (ω; α labeled customary)": "training fold",
        "regression calibration of the score's coefficient": "training fold",
    },
    needs=("three or more numeric items on one response scale, each a confirmed exposure or "
           "covariate",
           "the instrument's key: its reverse-coded items and its response scale",
           "for a test–retest reliability: the repeat administration's columns",
           "for a calibration substudy: a reference measure, blank outside the substudy"),
    question=("Which items form a scale, how is it scored, and is its coefficient corrected for "
              "measurement error?"),
    place=("Step 4, the domain transforms (MODELING_SEQUENCE §1): after the roles and the "
           "missing-data answer, before the functional form and the shelf. The correction is planned "
           "here and run as a declared secondary analysis (ruling 7)."),
    options=(
        Option("omega", "Reliability as ω (ω-total; ω-hierarchical when multidimensional)",
               "Reported in psychometric papers (R's psych::omega)",
               {"inference": "Sound: ω does not assume equal loadings (McNeish 2018).",
                "prediction": "Sound, and descriptive: under prediction the reliability changes "
                              "no modeling choice."},
               {"inference": "recommended", "prediction": "recommended"}),
        Option("alpha", "Reliability as Cronbach's α",
               "Customary: the reliability most papers report (Cronbach 1951)",
               {"inference": "Shown labeled customary only: its assumptions usually fail and it "
                             "usually understates reliability (McNeish 2018).",
                "prediction": "Shown labeled customary only (McNeish 2018)."},
               {"inference": "rank lower", "prediction": "rank lower"}),
        Option("none", "No correction: the uncorrected estimate",
               "Customary: most papers report the score's coefficient as estimated",
               {"inference": "Sound as the primary analysis: for one error-prone exposure its test "
                             "of no association stays valid, though the estimate is attenuated "
                             "(Freedman et al. 2011).",
                "prediction": "Sound: a new row's score carries the same error, so the model "
                              "needs no correction."},
               {"inference": "available", "prediction": "recommended"}),
        Option("rc_test_retest", "Regression calibration from a test–retest ICC",
               "Customary where a repeat administration exists",
               {"inference": "Sound as a declared secondary analysis: a repeat administration "
                             "includes transient error, conditional on the covariates.",
                "prediction": PREDICTION},
               {"inference": "recommended", "prediction": "refuse"}),
        Option("rc_calibration_substudy", "Regression calibration against a calibration substudy",
               "Customary in nutritional cohorts with a validation substudy (Freedman et al. 2011)",
               {"inference": "Sound as a declared secondary analysis, under the reference "
                             "measure's own assumptions, conditional on the covariates.",
                "prediction": PREDICTION},
               {"inference": "recommended", "prediction": "refuse"}),
        Option("rc_internal_consistency", "Regression calibration from ω (a reflective scale)",
               "Psychometric papers disattenuate by the Spearman correction (β/α)",
               {"inference": "Sound as a declared secondary analysis when conditional on the "
                             "covariates (Keogh, Shaw & Gustafson 2020). " + TRANSIENT,
                "prediction": PREDICTION},
               {"inference": "available", "prediction": "refuse"}),
        Option("spearman", "Dividing the coefficient by α or ω (the Spearman correction)",
               "Customary in psychometric papers",
               {"inference": "Unsound: with covariates the slope is attenuated by the reliability "
                             "conditional on them, so β/ω under-corrects whenever the score "
                             "correlates with the covariates (Keogh, Shaw & Gustafson 2020); "
                             "regression calibration replaces it.",
                "prediction": PREDICTION},
               {"inference": "refuse", "prediction": "refuse"}),
        Option("rc_internal_consistency_formative",
               "Regression calibration of a formative index from α or ω",
               "Seen in nutrition papers that disattenuate a diet score by its α",
               {"inference": "Refused: " + FORMATIVE, "prediction": PREDICTION},
               {"inference": "refuse", "prediction": "refuse"}),
    ),
    storyboard=(
        "Turn each reverse-coded item over on the response scale",
        "Sum (or average) the items into the score, in each completed copy",
        "Estimate ω from the items' factor structure (α beside it, customary)",
        "Regress the score on the model's other covariates",
        "Shrink each score toward that prediction by λ, the reliability given the covariates",
        "Refit the model with the calibrated score, beside the uncorrected estimate",
    ),
    sentence="turbotab.core.scales:methods_sentence",
    relations=(
        Relation("implies", "item-level multiple imputation before scoring",
                 "Under inference with missing items, the items are multiply imputed and the score "
                 "is formed in each completed copy (Eekhout et al. 2014); the reliability and the "
                 "correction are estimated in each copy and pooled by Rubin's rules.",
                 "turbotab.core.models.pipeline:shared_steps"),
        Relation("implies", "the items read as answers, not codes",
                 "The scale's answer settles each item's code-or-amount reading for the fit: its "
                 "items are answers on the response scale, summed into one score.",
                 "turbotab.core.readings:predictors_or_ask"),
        Relation("implies", "every outcome-model covariate in the calibration model",
                 "The calibration regresses the score on every other column of the model's matrix "
                 "(Boe et al. 2023).",
                 "turbotab.core.stages.scales:scales_stage"),
        Relation("implies", "a bootstrap that re-estimates the reliability",
                 "Each bootstrap replicate re-estimates the reliability (the whole factor analysis, "
                 "the test–retest mean square, or the calibration regression) before it "
                 "recalibrates.",
                 "turbotab.core.methods.scales:correct"),
        Relation("implies", "the repeat administration and the reference stay out of the model",
                 "A repeat administration or a reference measure is read by the reliability only, "
                 "never as a predictor.",
                 "turbotab.core.scales:_scale_items_are_settled_predictors"),
        Relation("conflicts", "disattenuation of a formative index by α or ω",
                 "Refused under inference: " + FORMATIVE,
                 "turbotab.core.scales:_correction_fits_purpose_and_construct",
                 exits=("a test–retest ICC from a repeat administration",
                        "a calibration substudy", "the uncorrected estimate")),
        Relation("conflicts", "a correction under prediction", "Refused: " + PREDICTION,
                 "turbotab.core.scales:_correction_fits_purpose_and_construct",
                 exits=("the uncorrected score",)),
        Relation("conflicts", "codes counted as answers",
                 "Refused until each value outside the response scale is recoded to missing.",
                 "turbotab.core.scales:_answers_fit_the_response_scale",
                 exits=("recode the codes to missing (the sentinel finding)",
                        "a wider response scale")),
        Relation("invalidates", "a functional form recorded on an item",
                 "Scoring replaces the item with the score, so a form recorded on the item is "
                 "re-asked, never silently kept.",
                 "turbotab.core.scales:_items_carry_no_form"),
        Relation("invalidates", "the calibration when the adjustment set changes",
                 "The calibration reads the model's covariates, so a change to the roles recomputes "
                 "it.",
                 "turbotab.core.stages:build_graph"),
    ),
))


# ── the leash: validators of set_scales ───────────────────────────────────────


def _named(decision: SetScales, ctx: Any) -> None:
    """Every column a scale reads exists; a score's name is not a column; the outcome is in no
    scale; a repeat administration or a reference is not also an item."""
    columns = _columns_of(ctx)
    target = _target_of(ctx)
    for s in decision.scales:
        if columns is not None:
            if s.name in columns or s.name == ROW_ID:
                free = next(f"{s.name}_{i}" for i in range(2, 1000) if f"{s.name}_{i}" not in columns)
                rest = [x.model_copy(update={"name": free}) if x.name == s.name else x
                        for x in decision.scales]
                raise Refusal(
                    "name_taken",
                    f"`{s.name}` is already a column of the table; the score needs a name of its "
                    f"own.", exits=[{"label": f"Name the score `{free}`",
                                     "decision": SetScales(scales=rest)}])
            unknown = [c for c in s.columns() if c not in columns or c == ROW_ID]
            if unknown:
                raise Refusal("unknown_column",
                              f"This dataset has no column named {_and(unknown)}.",
                              exits=[{"label": "Choose the scale's columns from the table",
                                      "decision": None}])
        if target is not _UNKNOWN and target is not None and target in s.columns():
            raise Refusal("outcome_in_scale",
                          f"`{target}` is the outcome; a scale's score is a predictor of it.",
                          exits=[{"label": "Leave the outcome out of the scale", "decision": None}])
        both = [c for c in [*s.retest, *([s.reference] if s.reference else [])] if c in s.items]
        if both:
            raise Refusal("retest_is_an_item",
                          f"{_and(both)} {'is' if len(both) == 1 else 'are'} an item of `{s.name}`; "
                          f"a repeat administration or a reference measure is a separate column.",
                          exits=[{"label": "Name the repeat administration's own columns",
                                  "decision": None}])


def _scale_items_are_settled_predictors(decision: SetScales, ctx: Any) -> None:
    """The score replaces its items in the models, so each item is a settled exposure or
    covariate (BLUEPRINT §14.1: a role nobody confirmed changes no number); a repeat administration
    or a reference measure is read by the reliability only, never as a predictor."""
    from turbotab.core.decisions import SetRoles
    from turbotab.core.readings import confirm_exits, settled_roles, unsettled

    state = _state(ctx)
    if state is None or not decision.scales:
        return
    if state.roles is None:
        raise Refusal("roles_first", "Answer the roles question first: a scale's score replaces its "
                      "items among the predictors.",
                      exits=[{"label": "Answer the roles question", "decision": None}])
    waiting = set(unsettled(state))
    items = [c for s in decision.scales for c in s.items]
    pending = [c for c in items if c in waiting]
    if pending:
        raise Refusal("role_unconfirmed",
                      f"{_and(pending)} {'has' if len(pending) == 1 else 'have'} a role nobody "
                      f"confirmed; a scale's items are read as predictors only once each is.",
                      exits=confirm_exits(state, pending))
    roles = settled_roles(state)
    gone = set(left_out(state))
    outside = [c for c in items if roles.get(c) not in PREDICTOR_ROLES or c in gone]
    if outside:
        fixed = {**state.roles, **{c: "covariate" for c in outside if c not in gone}}
        exits = ([{"label": f"Make {_and(outside)} covariates, to be scored into the scale",
                   "decision": SetRoles(roles=fixed)}] if not gone & set(outside) else [])
        raise Refusal(
            "items_not_predictors",
            f"{_and(outside)} {'is' if len(outside) == 1 else 'are'} not among the predictors "
            f"(an exposure or a covariate), so {'it' if len(outside) == 1 else 'they'} cannot be "
            f"scored into the scale that replaces {'it' if len(outside) == 1 else 'them'} in the "
            f"models.", exits=exits + [{"label": "Choose the items among the predictors",
                                        "decision": None}])
    extra = [c for s in decision.scales for c in [*s.retest, *([s.reference] if s.reference else [])]
             if state.roles.get(c) in ("exposure", "covariate", "energy")]
    if extra:
        fixed = {**state.roles, **{c: "excluded" for c in extra}}
        raise Refusal(
            "retest_in_the_model",
            f"{_and(extra)} {'is' if len(extra) == 1 else 'are'} a predictor; a repeat "
            f"administration or a reference measure is read by the reliability only, and in the "
            f"model it would adjust the score for itself.",
            exits=[{"label": f"Leave {_and(extra)} out of the models",
                    "decision": SetRoles(roles=fixed)}])


def _sentinel_exit(ctx: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    """The sentinel finding's repair, when the findings stage read codes in these items."""
    from turbotab.core.decisions import ApplyRepair
    from turbotab.core.sequence import artifact

    found = artifact(ctx, "findings") or {}
    for f in found.get("findings") or []:
        named = set(f.get("affected_columns") or f.get("columns") or [])
        if str(f.get("id", "")).endswith("sentinel_codes") and named & set(columns):
            return [{"label": "Recode the codes to missing (the sentinel finding)",
                     "decision": ApplyRepair(finding_id=str(f["id"]), option="set_missing")}]
    return []


def _answers_fit_the_response_scale(decision: SetScales, ctx: Any) -> None:
    """Every recorded answer of every item is a whole number on the instrument's response scale.
    A value outside it is a code ("don't know", "refused"), never recoded on a guess; counted as
    an answer it moves every score. An item the user recorded as codes for categories cannot be
    summed either."""
    from turbotab.core.readings import confirm_exit, confirmed_codes

    state = _state(ctx)
    codes = set(confirmed_codes(state)) if state is not None else set()
    for s in decision.scales:
        coded = [c for c in s.items if c in codes]
        if coded:
            raise Refusal(
                "items_are_codes",
                f"{_and(coded)} {'is' if len(coded) == 1 else 'are'} recorded as codes for "
                f"categories, but a scale sums its items as answers. Say which holds.",
                exits=[confirm_exit("code_or_count", c, "amount",
                                    f"`{c}` holds answers on the {s.low}–{s.high} response scale")
                       for c in coded])
    store = _store_of(ctx)
    if store is None:
        return
    for s in decision.scales:
        answers = [*s.items, *(s.retest if len(s.retest) == len(s.items) else [])]
        try:
            facts = store.whole_numbers(answers)
        except Exception:  # noqa: BLE001 - no values to read: the stage that reads them says so
            return
        text = [c for c in answers if c not in facts]
        if text:
            raise Refusal("items_not_numeric",
                          f"{_and(text)} {'holds' if len(text) == 1 else 'hold'} text, not answers "
                          f"on a numeric response scale; recode {'it' if len(text) == 1 else 'them'} "
                          f"to numbers first.",
                          exits=[{"label": "Choose numeric items", "decision": None}])
        fractional = [c for c in answers if not facts[c]["whole"]]
        if fractional:
            raise Refusal("items_not_whole",
                          f"{_and(fractional)} {'holds' if len(fractional) == 1 else 'hold'} values "
                          f"with decimals, which are not answers on the {s.low}–{s.high} response "
                          f"scale.", exits=[{"label": "Choose the scale's items", "decision": None}])
        outside = [c for c in answers if facts[c]["n_values"] and (
            facts[c]["min"] < s.low or facts[c]["max"] > s.high)]
        if outside:
            lo = min(min(facts[c]["min"] for c in outside), s.low)
            hi = max(max(facts[c]["max"] for c in outside), s.high)
            said = "; ".join(f"`{c}` {facts[c]['min']:g}–{facts[c]['max']:g}" for c in outside[:4])
            wider = [x.model_copy(update={"low": int(lo), "high": int(hi)}) if x.name == s.name
                     else x for x in decision.scales]
            raise Refusal(
                "answers_outside_the_scale",
                f"{_and(outside)} {'holds' if len(outside) == 1 else 'hold'} values outside the "
                f"{s.low}–{s.high} response scale ({said}). A value outside it is usually a code "
                f"('don't know', 'refused'); counted as an answer it moves every score. Recode the "
                f"codes to missing first, or record the instrument's wider response scale.",
                exits=[*_sentinel_exit(ctx, outside),
                       {"label": f"The response scale runs {int(lo)}–{int(hi)}",
                        "decision": SetScales(scales=wider)}])


def retest_columns(columns: Sequence[str], items: Sequence[str]) -> list[str] | None:
    """A repeat administration's items in the table, one per item by a shared suffix
    (``pss_1_t2`` for ``pss_1``), or None: offered as the exit's guess, which the user confirms."""
    have = set(columns)
    for suffix in RETEST_SUFFIXES:
        found = [f"{c}{suffix}" for c in items]
        if all(f in have for f in found):
            return found
    return None


def _correction_fits_purpose_and_construct(decision: SetScales, ctx: Any) -> None:
    """MODELING_SEQUENCE §4: a correction is refused under prediction, and disattenuation of a
    formative index by its internal consistency is refused under inference, with its exits."""
    state = _state(ctx)
    purpose = getattr(state, "purpose", None)
    corrected = [s for s in decision.scales if s.correction != "none"]
    if not corrected:
        return
    if purpose == "prediction":
        uncorrected = [s.model_copy(update={"correction": "none", "reliability":
                                            "internal_consistency", "retest": [],
                                            "reference": None}) for s in decision.scales]
        raise Refusal("not_for_prediction", PREDICTION,
                      exits=[{"label": "Keep the scores uncorrected",
                              "decision": SetScales(scales=uncorrected)},
                             {"label": "Change the purpose to inference", "decision": None}])
    for s in corrected:
        if s.kind != "formative" or s.reliability != "internal_consistency":
            continue

        def swap(**update: Any) -> SetScales:
            return SetScales(scales=[x.model_copy(update=update) if x.name == s.name else x
                                     for x in decision.scales])

        columns = _columns_of(ctx) or []
        found = retest_columns(columns, s.items)
        retest = ({"label": f"Use the test–retest ICC of the repeat administration "
                            f"({_and(found[:2])} …)",
                   "decision": swap(reliability="test_retest", retest=found)} if found else
                  {"label": "Name the repeat administration's columns, for a test–retest ICC",
                   "decision": None})
        raise Refusal(
            "formative_disattenuation",
            f"`{s.name}` is a formative index, and disattenuating it by α or ω is refused. "
            f"{FORMATIVE}",
            exits=[retest,
                   {"label": "Name the reference measure of a calibration substudy",
                    "decision": None},
                   {"label": f"Report `{s.name}`'s uncorrected estimate only",
                    "decision": swap(correction="none")}])


def _structure_fits_the_items(decision: SetScales, ctx: Any) -> None:
    from turbotab.core.methods.scales import ITEMS_PER_FACTOR

    for s in decision.scales:
        k = s.group_factors()
        if k > 1 and len(s.items) < ITEMS_PER_FACTOR * k:
            fewer = max(2, len(s.items) // ITEMS_PER_FACTOR)
            raise Refusal(
                "too_few_items",
                f"`{s.name}` has {len(s.items)} items; {k} group factors need at least "
                f"{ITEMS_PER_FACTOR * k}, three a factor, to be identified.",
                exits=[{"label": f"{fewer} group factors" if len(s.items) >= 6 else
                        "Read it as unidimensional",
                        "decision": SetScales(scales=[
                            x.model_copy(update={"factors": fewer} if len(s.items) >= 6 else
                                         {"structure": "unidimensional", "factors": None})
                            if x.name == s.name else x for x in decision.scales])}])


def _items_carry_no_form(decision: SetScales, ctx: Any) -> None:
    """§2: scoring is a domain transform of its items, so a form recorded on an item is re-asked."""
    state = _state(ctx)
    forms = (getattr(state, "exposure_forms", None) or {}) if state is not None else {}
    formed = [c for s in decision.scales for c in s.items
              if c in forms and getattr(forms[c], "form", "linear") != "linear"]
    if formed:
        c = formed[0]
        raise Refusal(
            "item_has_a_form",
            f"`{c}` has a {forms[c].form} form recorded. Scoring the scale replaces it with the "
            f"score, so its form no longer applies and is asked again: set it to a straight line "
            f"first.", exits=[{"label": f"Keep `{c}` a straight line",
                               "decision": SetExposureForm(column=c, form="linear")}])


def _form_is_not_of_a_scored_item(decision: SetExposureForm, ctx: Any) -> None:
    if decision.form == "linear":
        return
    state = _state(ctx)
    for s in (getattr(state, "scales", None) or []) if state is not None else []:
        if decision.column in s.items:
            raise Refusal(
                "scored_item",
                f"`{decision.column}` is an item of `{s.name}`, which enters the models as its "
                f"score; the item itself takes no form.",
                exits=[{"label": f"Keep `{decision.column}` a straight line",
                        "decision": SetExposureForm(column=decision.column, form="linear")}])


register_validator("set_scales", _named)
register_validator("set_scales", _structure_fits_the_items)
register_validator("set_scales", _correction_fits_purpose_and_construct)
register_validator("set_scales", _scale_items_are_settled_predictors)
register_validator("set_scales", _items_carry_no_form)
register_validator("set_scales", _answers_fit_the_response_scale)
register_validator("set_exposure_form", _form_is_not_of_a_scored_item)


# ── sentences ────────────────────────────────────────────────────────────────


def _scale_words(s: Any) -> str:
    n = len(s.items)
    turned = list(s.reverse)
    reverse = (f", {_and(turned)} reverse-coded on the {s.low}–{s.high} response scale"
               if turned else "")
    parts = "components" if s.kind == "formative" else "items"
    of = f"the {s.instrument}'s {n} {parts}" if s.instrument else f"its {n} {parts}"
    return f"`{s.name}` is the {s.scoring} of {of}{reverse}"


SOURCE_WORDS = {"internal_consistency": "its internal consistency (ω)",
                "test_retest": "the test–retest ICC of its repeat administration",
                "calibration_substudy": "a calibration substudy"}


def decision_sentence(d: Any, state: Any = None, ctx: Any = None) -> str:
    """The record's sentence for ``set_scales``."""
    if not d.scales:
        return "No multi-item scale is scored; each item enters the models on its own"
    parts = []
    for s in d.scales:
        role = "an exposure" if s.role == "exposure" else "a covariate"
        corrected = (f"its coefficient is corrected by regression calibration from "
                     f"{SOURCE_WORDS[s.reliability]}, as a secondary analysis beside the "
                     f"uncorrected one" if s.correction != "none" else
                     "its coefficient is not corrected for measurement error")
        parts.append(f"{_scale_words(s)}, a {s.kind} "
                     f"{'index' if s.kind == 'formative' else 'scale'} entering the models as "
                     f"{role}; {corrected}")
    return "; ".join(parts)


def _register_sentence() -> None:
    from turbotab.core.voice import register_sentence

    register_sentence("set_scales")(decision_sentence)


_register_sentence()

COEFFICIENT_WORDS = {"omega_total": "ω-total", "omega_hierarchical": "ω-hierarchical"}


def methods_sentence(spec: Any, result: Mapping[str, Any]) -> str:
    """One scale's methods sentence, from its stage result (``turbotab.core.stages.scales``)."""
    s = spec
    head = _scale_words(s).replace(f"`{s.name}` is", f"`{s.name}` was", 1) + "."
    rel = result.get("reliability") or {}
    mi = result.get("imputation") or {}
    imputed = (f" Missing items were multiply imputed before scoring (item level, m = "
               f"{int(mi['m'])}), and the score, its reliability and the correction were estimated "
               f"in each completed copy and pooled." if mi.get("m") else "")
    corr = result.get("correction") or {}
    boots = int(corr.get("n_boot_ok") or 0)
    each = " in each completed copy" if int(corr.get("copies") or 1) > 1 else ""
    tail = (f"the corrected coefficient was obtained by regression calibration including all model "
            f"covariates, with bootstrap CIs re-estimating the reliability ({boots:,} "
            f"replicates{each}); uncorrected and corrected estimates are both reported.")
    model = {"proportional_odds": "proportional-odds", "linear": "logistic"}.get(
        str(corr.get("family")), "")
    approximate = (f" In a {model} model the calibrated score is an approximation (Carroll et al. "
                   f"2006)." if corr.get("scale") == "odds_ratio" else "")
    source = rel.get("source")
    if source == "test_retest" and corr.get("estimate") is not None:
        return (f"{head} Reliability was estimated as the test–retest ICC from a repeat "
                f"administration (ICC(3,1) = {rel['value']:.2f}, {int(rel['n']):,} "
                f"participants with both); {tail}{approximate}{imputed}")
    if source == "calibration_substudy" and corr.get("estimate") is not None:
        return (f"{head} Reliability was estimated from a calibration substudy against "
                f"`{s.reference}` ({int(rel['n']):,} participants; attenuation factor λ = "
                f"{corr['attenuation']:.2f} given the covariates); {tail}{approximate}{imputed}")
    if source == "internal_consistency" and rel.get("value") is not None:
        word = COEFFICIENT_WORDS[rel["coefficient"]]
        factors = ("one factor" if int(rel.get("factors") or 1) == 1 else
                   f"{int(rel['factors'])} group factors under a general factor, Schmid–Leiman")
        said = (f" Its reliability was {word} = {rel['value']:.2f} for the score (McDonald's ω; "
                f"minimum-residual factor analysis of the item correlations, {factors}; "
                f"Cronbach's α = {rel['alpha']:.2f}, reported as customary).")
        if corr.get("estimate") is not None:
            return (f"{head}{said} {tail[0].upper()}{tail[1:]} A reliability from internal "
                    f"consistency omits transient error, so the correction under-corrects."
                    f"{approximate}{imputed}")
        return f"{head}{said} {_uncorrected(result)}{imputed}"
    if source == "test_retest" and rel.get("value") is not None:
        return (f"{head} Its test–retest reliability was ICC(3,1) = {rel['value']:.2f} "
                f"({int(rel['n']):,} participants with both administrations). "
                f"{_uncorrected(result)}{imputed}")
    return f"{head} {_uncorrected(result)}{imputed}"


def _uncorrected(result: Mapping[str, Any]) -> str:
    why = result.get("not_corrected")
    return ("Its coefficient was not corrected for measurement error." + (f" {why}" if why else ""))


__all__ = ["CONTRACT", "FORMATIVE", "PREDICTION", "SECONDARY", "TRANSIENT", "decision_sentence",
           "methods_sentence", "retest_columns"]

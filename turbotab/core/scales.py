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
  answer moves every score. The same holds for every other column the answer reads as values: a
  repeat administration recorded as one score must lie in the range that score can take, and a
  calibration substudy's reference measure must have a settled amount reading (BLUEPRINT §14.3)
  and hold no missing-value code;
* a reflective scale's answers are whole numbers on its response scale; a formative index's
  components may be continuous scores (the HEI-2015's are prorated, 0–5 or 0–10);
* a functional form recorded on an item is **re-asked** (§2, "a domain transform … invalidates the
  functional-form answer"): scoring replaces the item, so its form no longer applies;
* an answer that takes an item out of the predictors (another role, an exclusion, the outcome, a
  left-out column) or puts a repeat administration into the model is **refused** until the scales
  answer changes (§2, *invalidates*): the scale is asked again, never kept silently.

Several scores corrected in one model are calibrated jointly (ruling 7, Rosner), and on grouped rows
both intervals keep each group's rows together (§2, repeated units or clusters): the uncorrected one
is the coefficient table's CR2 interval, the corrected one a bootstrap of whole clusters.

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
    slot="in_fold",
    scope="training_fold",
    # After the fill (and the batch step) and before the energy step (models/pipeline.py).
    run_order=5.5,
    parts={
        "scoring by the instrument's key (reverse coding; the sum or the mean)": "row_local",
        "item-level imputation before scoring": "training_fold",
        "reliability (ω; α labeled customary)": "training_fold",
        "regression calibration of the score's coefficient": "training_fold",
    },
    needs=("three or more numeric items on one response scale, each a confirmed exposure or "
           "covariate (a reflective scale's answers whole numbers; a formative index's components "
           "may be continuous scores)",
           "the instrument's key: its reverse-coded items and its response scale",
           "for a test–retest reliability: the repeat administration's columns, or its score",
           "for a calibration substudy: a reference measure read as an amount, blank outside the "
           "substudy"),
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
               {"inference": "rank_lower", "prediction": "rank_lower"}),
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
               {"inference": "recommended", "prediction": "refused"}),
        Option("rc_calibration_substudy", "Regression calibration against a calibration substudy",
               "Customary in nutritional cohorts with a validation substudy (Freedman et al. 2011)",
               {"inference": "Sound as a declared secondary analysis, under the reference "
                             "measure's own assumptions, conditional on the covariates.",
                "prediction": PREDICTION},
               {"inference": "recommended", "prediction": "refused"}),
        Option("rc_internal_consistency", "Regression calibration from ω (a reflective scale)",
               "Psychometric papers disattenuate by the Spearman correction (β/α)",
               {"inference": "Sound as a declared secondary analysis when conditional on the "
                             "covariates (Keogh, Shaw & Gustafson 2020). " + TRANSIENT,
                "prediction": PREDICTION},
               {"inference": "available", "prediction": "refused"}),
        Option("spearman", "Dividing the coefficient by α or ω (the Spearman correction)",
               "Customary in psychometric papers",
               {"inference": "Unsound: with covariates the slope is attenuated by the reliability "
                             "conditional on them, so β/ω under-corrects whenever the score "
                             "correlates with the covariates (Keogh, Shaw & Gustafson 2020); "
                             "regression calibration replaces it.",
                "prediction": PREDICTION},
               {"inference": "refused", "prediction": "refused"}),
        Option("rc_internal_consistency_formative",
               "Regression calibration of a formative index from α or ω",
               "Seen in nutrition papers that disattenuate a diet score by its α",
               {"inference": "Refused: " + FORMATIVE, "prediction": PREDICTION},
               {"inference": "refused", "prediction": "refused"}),
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
                 enforced_by="turbotab.core.models.pipeline:shared_steps"),
        Relation("implies", "the items read as answers, not codes",
                 "The scale's answer settles each item's code-or-amount reading for the fit: its "
                 "items are answers on the response scale, summed into one score.",
                 enforced_by="turbotab.core.readings:predictors_or_ask"),
        Relation("implies", "every outcome-model covariate in the calibration model",
                 "The calibration regresses the score on every other column of the model's matrix "
                 "(Boe et al. 2023).",
                 enforced_by="turbotab.core.stages.scales:scales_stage"),
        Relation("implies", "a bootstrap that re-estimates the reliability",
                 "Each bootstrap replicate re-estimates the reliability (the whole factor analysis, "
                 "the test–retest mean square, or the calibration regression) before it "
                 "recalibrates.",
                 enforced_by="turbotab.core.methods.scales:correct"),
        Relation("implies", "the repeat administration and the reference stay out of the model",
                 "A repeat administration or a reference measure is read by the reliability only, "
                 "never as a predictor.",
                 enforced_by="turbotab.core.scales:_scale_items_are_settled_predictors"),
        Relation("implies", "the reference measure read as an amount",
                 "The calibration regresses a substudy's reference measure on the score, so its "
                 "code-or-amount reading is settled (by its values or the user) and it holds no "
                 "missing-value code before it is read (BLUEPRINT §14.3).",
                 enforced_by="turbotab.core.scales:_reference_reads_as_an_amount"),
        Relation("implies", "multivariate calibration when several scores are corrected",
                 "Two or more corrected scores in one model are calibrated jointly (Rosner, "
                 "Spiegelman & Willett 1990; MODELING_SEQUENCE §0 ruling 7): each one's calibration "
                 "conditions on the others.",
                 enforced_by="turbotab.core.methods.scales:calibrate_jointly"),
        Relation("implies", "cluster-aware intervals on grouped rows",
                 "Rows grouped by a unit or cluster keep it in both intervals: the uncorrected one "
                 "is the coefficient table's CR2 interval, and the correction's bootstrap resamples "
                 "whole clusters (ruling 7); too few clusters block the correction with the table's "
                 "own exits.",
                 enforced_by="turbotab.core.methods.scales:bootstrap_draws"),
        Relation("conflicts", "disattenuation of a formative index by α or ω",
                 "Refused under inference: " + FORMATIVE,
                 enforced_by="turbotab.core.scales:_correction_fits_purpose_and_construct",
                 rung="refused", exits=("a test–retest ICC from a repeat administration",
                        "a calibration substudy", "the uncorrected estimate")),
        Relation("conflicts", "a correction under prediction", "Refused: " + PREDICTION,
                 enforced_by="turbotab.core.scales:_correction_fits_purpose_and_construct",
                 rung="refused", exits=("the uncorrected score",)),
        Relation("conflicts", "codes counted as answers",
                 "Refused until each value outside the response scale (or, for a repeat "
                 "administration recorded as one score, outside the range that score can take; for "
                 "a reference measure, a missing-value code beyond its values) is recoded to "
                 "missing.",
                 enforced_by="turbotab.core.scales:_answers_fit_the_response_scale",
                 rung="refused", exits=("recode the codes to missing (the findings' code repair)",
                        "a wider response scale", "the internal consistency instead",
                        "the uncorrected estimate")),
        Relation("invalidates", "a functional form recorded on an item",
                 "Scoring replaces the item with the score, so a form recorded on the item is "
                 "re-asked, never silently kept.",
                 enforced_by="turbotab.core.scales:_items_carry_no_form"),
        Relation("invalidates", "the scales answer when an item leaves the predictors",
                 "An answer that takes an item out of the predictors (another role, an exclusion, "
                 "the outcome, a left-out column) or puts a repeat administration in the model is "
                 "refused until the scales answer changes; the scale is asked again, never kept.",
                 enforced_by="turbotab.core.scales:_answers_keep_the_scales_whole"),
        Relation("invalidates", "the calibration when the adjustment set changes",
                 "The calibration reads the model's covariates, so a change to the roles recomputes "
                 "it.",
                 enforced_by="turbotab.core.stages:build_graph"),
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
    """The findings' code repair (the survey pack's sentinel codes in a Likert block, or a column's
    numeric missing-value codes), when the findings stage read codes in these columns and the
    repair is not applied yet."""
    from turbotab.core.decisions import ApplyRepair
    from turbotab.core.sequence import artifact

    found = artifact(ctx, "findings") or {}
    state = _state(ctx)
    done = {fid for fid, d in ((getattr(state, "findings", None) or {}) if state is not None
                               else {}).items() if getattr(d, "action", None) == "applied"}
    for f in found.get("findings") or []:
        fid = str(f.get("id", ""))
        named = set(f.get("affected_columns") or f.get("columns") or [])
        coded = fid.endswith("sentinel_codes") or fid.startswith("sentinel_missing__")
        if coded and fid not in done and named & set(columns):
            return [{"label": "Recode the codes to missing (the findings' code repair)",
                     "decision": ApplyRepair(finding_id=fid, option="set_missing")}]
    return []


def score_range(s: Any) -> tuple[float, float]:
    """The values a scale's score can take: k answers on the ``low``–``high`` response scale,
    summed (k·low to k·high) or averaged (low to high)."""
    k = len(s.items)
    if (getattr(s, "scoring", None) or "sum") == "mean":
        return float(s.low), float(s.high)
    return float(k * s.low), float(k * s.high)


def codes_beyond(values: Any) -> list[float]:
    """Conventional missing-value codes (999, 77, -9 …; Classic's list) lying beyond every other
    value of a measurement, with the findings' "far" gap between (at least ten typical spacings
    and half the rest's range; a negative code in a column of non-negative values, two spacings).
    The findings' own reading (``detectors.codes``) also asks that a code recur, so that one
    extreme value is never blanked on a guess; a calibration substudy's reference measure has no
    response scale to bound it, and one code moves the calibration's slope, so here one row is
    enough to refuse (nothing is recoded: the refusal asks)."""
    from ml.import_doctor import NUMERIC_SENTINELS  # Classic's list, read and not copied

    from turbotab.core.detectors.codes import FAR_RANGE, FAR_SPACINGS

    import numpy as np

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return []
    known = {float(v) for v in NUMERIC_SENTINELS}
    distinct = np.unique(x)

    def spacing(v: Any) -> float:
        gaps = np.diff(np.sort(v))
        gaps = gaps[gaps > 0]
        return float(np.median(gaps)) if len(gaps) else 1.0

    flagged: list[float] = []
    top = []
    for v in distinct[::-1]:
        if v > 0 and v in known:
            top.append(float(v))
            continue
        break
    for j, v in enumerate(sorted(top)):
        rest = distinct[distinct < v]
        if len(rest) and v - rest.max() >= max(FAR_SPACINGS * spacing(rest),
                                                FAR_RANGE * (rest.max() - rest.min())):
            flagged += sorted(top)[j:]
            break
    bottom = []
    for v in distinct:
        if v < 0 and v in known:
            bottom.append(float(v))
            continue
        break
    rest = distinct[distinct > max(bottom)] if bottom else distinct
    if bottom and len(rest):
        for v in bottom:
            gap = rest.min() - v
            if (rest.min() >= 0 and gap >= 2 * spacing(rest)) or gap >= max(
                    FAR_SPACINGS * spacing(rest), FAR_RANGE * (rest.max() - rest.min())):
                flagged.append(v)
    return sorted(flagged)


def _blank(store: Any, columns: Sequence[str]) -> list[str]:
    """The columns with no value on any row."""
    try:
        info = store.info()
    except Exception:  # noqa: BLE001 - no summary: the stage that reads the values says so
        return []
    empty = {c.name for c in info.columns if int(c.n_missing) >= int(info.n_rows)}
    return [c for c in columns if c in empty]


def _swap(decision: SetScales, name: str, **update: Any) -> SetScales:
    return SetScales(scales=[x.model_copy(update=update) if x.name == name else x
                             for x in decision.scales])


def _without_the_repeat(decision: SetScales, s: Any) -> list[dict[str, Any]]:
    """The ways forward that stop reading a repeat administration or a reference measure: ω for a
    reflective scale, the uncorrected estimate for any."""
    out = []
    if s.kind == "reflective":
        out.append({"label": f"Correct `{s.name}` from its internal consistency (ω) instead",
                    "decision": _swap(decision, s.name, reliability="internal_consistency",
                                      retest=[], reference=None)})
    out.append({"label": f"Report `{s.name}`'s uncorrected estimate only",
                "decision": _swap(decision, s.name, correction="none",
                                  reliability="internal_consistency", retest=[], reference=None)})
    return out


def _answers_fit_the_response_scale(decision: SetScales, ctx: Any) -> None:
    """Every recorded answer of every item (and of a repeat administration's items) lies on the
    instrument's response scale; a reflective scale's answers are whole numbers there, while a
    formative index's components may be continuous scores (the HEI-2015's are prorated, 0–5 or
    0–10). A repeat administration given as one column holds the score itself, so its values lie
    in the range that score can take. A value outside is a code ("don't know", "refused"), never
    recoded on a guess; counted as an answer it moves every score, the test–retest ICC and the
    correction. An item the user recorded as codes for categories cannot be summed either. A
    calibration substudy's reference measure is read as an amount, so its reading is settled and
    it holds no codes (:func:`_reference_reads_as_an_amount`)."""
    from turbotab.core.readings import confirm_exit, confirmed_codes

    state = _state(ctx)
    codes = set(confirmed_codes(state)) if state is not None else set()
    for s in decision.scales:
        coded = [c for c in [*s.items, *s.retest] if c in codes]
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
        _items_answer_the_scale(decision, s, ctx, store)
        if len(s.retest) == 1:
            _repeat_score_fits_the_score(decision, s, ctx, store)
        if s.reference:
            _reference_reads_as_an_amount(decision, s, ctx, store)


def _items_answer_the_scale(decision: SetScales, s: Any, ctx: Any, store: Any) -> None:
    import math

    answers = [*s.items, *(s.retest if len(s.retest) == len(s.items) else [])]
    blank = _blank(store, answers)
    if blank:
        one = len(blank) == 1
        kept = [c for c in s.items if c not in blank]
        # Offered where the shorter scale is still one the answer can record: three items a
        # factor, and no repeat administration read item by item.
        fits = len(kept) >= 3 * s.group_factors() and not s.retest
        exits = ([{"label": f"Score `{s.name}` from its other {len(kept)} items",
                   "decision": _swap(decision, s.name, items=kept,
                                     reverse=[c for c in s.reverse if c in kept])}]
                 if fits and not set(blank) - set(s.items) else [])
        raise Refusal("items_blank",
                      f"{_and(blank)} {'has' if one else 'have'} no answer on any row, so "
                      f"{'it' if one else 'they'} cannot be scored into `{s.name}`.",
                      exits=exits + [{"label": "Choose the scale's items", "decision": None}])
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
    if fractional and s.kind == "reflective":
        formative = {"kind": "formative", "structure": "unidimensional", "factors": None}
        if s.reliability == "internal_consistency":
            formative["correction"] = "none"  # an index's ω corrects nothing (ruling 8)
        raise Refusal(
            "items_not_whole",
            f"{_and(fractional)} {'holds' if len(fractional) == 1 else 'hold'} values with "
            f"decimals, which are not answers on the {s.low}–{s.high} response scale of a "
            f"reflective scale. A formative index's components may be continuous scores (the "
            f"HEI-2015's are prorated); if `{s.name}` is one, say so.",
            exits=[{"label": f"`{s.name}` is a formative index of continuous component scores",
                    "decision": _swap(decision, s.name, **formative)},
                   {"label": "Choose the scale's items", "decision": None}])
    outside = [c for c in answers if facts[c]["n_values"] and (
        facts[c]["min"] < s.low or facts[c]["max"] > s.high)]
    if outside:
        lo = math.floor(min(min(facts[c]["min"] for c in outside), s.low))
        hi = math.ceil(max(max(facts[c]["max"] for c in outside), s.high))
        said = "; ".join(f"`{c}` {facts[c]['min']:g}–{facts[c]['max']:g}" for c in outside[:4])
        word = "answer" if s.kind == "reflective" else "component score"
        raise Refusal(
            "answers_outside_the_scale",
            f"{_and(outside)} {'holds' if len(outside) == 1 else 'hold'} values outside the "
            f"{s.low}–{s.high} response scale ({said}). A value outside it is usually a code "
            f"('don't know', 'refused'); counted as an {word} it moves every score. Recode the "
            f"codes to missing first, or record the instrument's wider response scale.",
            exits=[*_sentinel_exit(ctx, outside),
                   {"label": f"The response scale runs {lo}–{hi}",
                    "decision": _swap(decision, s.name, low=lo, high=hi)}])


def _repeat_score_fits_the_score(decision: SetScales, s: Any, ctx: Any, store: Any) -> None:
    """A repeat administration given as one column is the score itself: its values lie where the
    score can, ``score_range``; outside, they are codes that move the ICC and the correction."""
    column = s.retest[0]
    if _blank(store, [column]):
        raise Refusal("retest_blank",
                      f"`{column}`, the repeat administration of `{s.name}`, has no value on any "
                      f"row.", exits=[*_without_the_repeat(decision, s),
                                      {"label": "Name the repeat administration's column",
                                       "decision": None}])
    try:
        facts = store.whole_numbers([column])
    except Exception:  # noqa: BLE001 - no values to read: the stage that reads them says so
        return
    if column not in facts:
        raise Refusal("retest_not_numeric",
                      f"`{column}`, the repeat administration of `{s.name}`, holds text, not "
                      f"scores.", exits=[*_without_the_repeat(decision, s),
                                         {"label": "Name the repeat administration's column",
                                          "decision": None}])
    f = facts[column]
    lo, hi = score_range(s)
    if f["n_values"] and (f["min"] < lo or f["max"] > hi):
        found = retest_columns(_columns_of(ctx) or [], s.items)
        items = ([{"label": f"Score the repeat administration from its items "
                            f"({_and(found[:2])} …)",
                   "decision": _swap(decision, s.name, retest=found)}] if found else [])
        how = "summed" if (s.scoring or "sum") == "sum" else "averaged"
        raise Refusal(
            "retest_outside_the_score",
            f"`{column}`, the repeat administration of `{s.name}`, holds values from "
            f"{f['min']:g} to {f['max']:g}, outside the {lo:g}–{hi:g} its score can take "
            f"({len(s.items)} answers on the {s.low}–{s.high} response scale, {how}). A value "
            f"outside it is a code ('don't know', 'refused', not administered); counted as a "
            f"score it moves the test–retest ICC and the correction. Recode the codes to missing "
            f"first.",
            exits=[*_sentinel_exit(ctx, [column]), *items, *_without_the_repeat(decision, s)])


def _reference_reads_as_an_amount(decision: SetScales, s: Any, ctx: Any, store: Any) -> None:
    """BLUEPRINT §14.3: the correction reads a calibration substudy's reference measure as an amount
    (its values regressed on the score), so its code-or-amount reading must be settled (by its
    values or the user's confirmation; the readings ledger decides which, and its confirmation
    exit asks), and it may hold no missing-value code (:func:`codes_beyond`)."""
    from turbotab.core.readings import confirm_exit, confirmation, code_or_count_reading, whole_facts

    column = s.reference
    state = _state(ctx)
    other = {"label": "Name another reference measure", "decision": None}
    if _blank(store, [column]):
        raise Refusal("reference_blank",
                      f"`{column}`, the reference measure of `{s.name}`'s calibration substudy, "
                      f"has no value on any row.",
                      exits=[other, *_without_the_repeat(decision, s)])
    if confirmation(state, "code_or_count", column) == "code":
        raise Refusal(
            "reference_is_codes",
            f"`{column}` is recorded as codes for categories, but a calibration substudy "
            f"regresses its reference measure on the score as an amount. Say which holds.",
            exits=[confirm_exit("code_or_count", column, "amount",
                                f"`{column}` is a measurement, the reference of `{s.name}`"),
                   other, *_without_the_repeat(decision, s)])
    facts = whole_facts([column], None, store).get(column)
    if facts is None:
        raise Refusal("reference_not_numeric",
                      f"`{column}`, the reference measure of `{s.name}`, holds text, not "
                      f"measurements.", exits=[other, *_without_the_repeat(decision, s)])
    reading = code_or_count_reading(state, column, facts, scope="fit")
    if reading is not None and not reading.settled:
        raise Refusal(
            "reference_unsettled",
            f"`{column}`, the reference measure of `{s.name}`, holds {reading.evidence}: whether "
            f"they are amounts or codes for categories is not settled by the values, and the "
            f"calibration regresses them on the score as amounts. Confirm it first.",
            exits=[confirm_exit("code_or_count", column, "amount",
                                f"`{column}` is a measurement, the reference of `{s.name}`"),
                   other, *_without_the_repeat(decision, s)])
    try:
        values = store.materialize([column], None)[column]
    except Exception:  # noqa: BLE001 - no values to read: the stage that reads them says so
        return
    found = codes_beyond(values)
    if found:
        said = ", ".join(f"{v:g}" for v in found)
        raise Refusal(
            "reference_codes",
            f"`{column}`, the reference measure of `{s.name}`, holds {said}, beyond every other "
            f"value with a wide gap between: that is how a missing-value code ('don't know', not "
            f"measured) looks, and one such value moves the calibration's slope. Recode it to "
            f"missing first.",
            exits=[*_sentinel_exit(ctx, [column]), other, *_without_the_repeat(decision, s)])


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


# ── §2 "invalidates": an answer that takes a scale's item out of the predictors ────────────────

# The answers that move a column into or out of the predictors: its role (recorded, confirmed or
# reverted), the outcome, the missing-values answer's and a repair's left-out columns, and WP17's
# adjustment answers (read for the current exposure).
ROLE_CHANGING = ("set_roles", "confirm_role", "confirm_reading", "confirm_readings", "revert",
                 "set_target", "set_missing", "apply_repair", "set_adjustment", "set_estimand",
                 "import_codebook")


def scale_breaks(state: Any) -> dict[str, tuple[list[str], list[str]]]:
    """Each recorded scale an answer has broken, by name: ``(lost, entered)``. ``lost``: its items
    that are no longer a settled exposure or covariate (another role, a role riding along
    unconfirmed, left out by the missing-values, repair or adjustment answers, or the outcome);
    ``entered``: its repeat administration's or reference measure's columns made predictors or
    the outcome."""
    from turbotab.core.readings import settled_roles

    scales = getattr(state, "scales", None) or []
    if not scales:
        return {}
    roles = settled_roles(state)
    recorded = getattr(state, "roles", None) or {}
    gone = set(left_out(state))
    target = getattr(state, "target", None)
    out: dict[str, tuple[list[str], list[str]]] = {}
    for s in scales:
        lost = [c for c in s.items
                if roles.get(c) not in PREDICTOR_ROLES or c in gone or c == target]
        entered = [c for c in [*s.retest, *([s.reference] if s.reference else [])]
                   if recorded.get(c) in (*PREDICTOR_ROLES, "energy") or c == target]
        if lost or entered:
            out[s.name] = (lost, entered)
    return out


def _scales_exits(scales: Sequence[Any], s: Any, lost: Sequence[str],
                  entered: Sequence[str]) -> list[dict[str, Any]]:
    """The scales answers that leave the change acceptable: the scale without the lost items (where
    three a factor remain), or without its repeat administration or reference; no scale at all."""
    def replaced(**update: Any) -> SetScales:
        return SetScales(scales=[x.model_copy(update=update) if x.name == s.name else x
                                 for x in scales])

    exits: list[dict[str, Any]] = []
    kept = [c for c in s.items if c not in lost]
    if lost and len(kept) >= 3 * s.group_factors() and not set(entered):
        update: dict[str, Any] = {"items": kept, "reverse": [c for c in s.reverse if c in kept]}
        if len(s.retest) == len(s.items) and len(s.retest) > 1:
            update["retest"] = [r for c, r in zip(s.items, s.retest) if c not in lost]
        elif s.retest:
            # A repeat administration recorded as one score sums the item that leaves.
            update.update(retest=[], reliability="internal_consistency",
                          correction=s.correction if s.kind == "reflective" else "none")
        exits.append({"label": f"Take {_and(list(lost))} out of `{s.name}` "
                               f"({len(kept)} items remain), then record this answer",
                      "decision": replaced(**update)})
    if entered and not lost:
        update = {"retest": [], "reference": None, "reliability": "internal_consistency"}
        if s.kind == "formative":
            update["correction"] = "none"
        exits.append({"label": f"Stop reading {_and(list(entered))} as `{s.name}`'s reliability, "
                               f"then record this answer", "decision": replaced(**update)})
    exits.append({"label": f"Stop scoring `{s.name}` (its items enter the models on their own), "
                           f"then record this answer",
                  "decision": SetScales(scales=[x for x in scales if x.name != s.name])})
    exits.append({"label": "Keep the answers as they are", "decision": None})
    return exits


def _answers_keep_the_scales_whole(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §2 (*invalidates*; BLUEPRINT §13): the scales answer rests on its items
    being settled predictors, scored into one column, and on its repeat administration and
    reference staying out of the model. An answer that would change that (an item excluded or given
    another role, its role reverted or left riding along, the item made the outcome or left out by
    another answer; a repeat administration made a predictor) is refused until the scales answer
    changes: the scale is asked again, never kept silently or left to fail in the pipeline."""
    from turbotab.core.decisions import _roles_record_what_rode_along, state_after

    now = _state(ctx)
    recorded = list(getattr(now, "scales", None) or []) if now is not None else []
    if not recorded:
        return
    if decision.kind == "set_roles":
        # As it would be recorded: the roles that ride along unconfirmed are named on it.
        decision = _roles_record_what_rode_along(decision, ctx)
    after = state_after(decision, ctx)
    if after is None:
        return
    before = scale_breaks(now)
    for name, (lost, entered) in scale_breaks(after).items():
        was_lost, was_entered = before.get(name, ([], []))
        new_lost = [c for c in lost if c not in was_lost]
        new_entered = [c for c in entered if c not in was_entered]
        if not new_lost and not new_entered:
            continue
        s = next((x for x in recorded if x.name == name),
                 next(x for x in (getattr(after, "scales", None) or []) if x.name == name))
        if new_lost:
            one = len(new_lost) == 1
            said = (f"{_and(new_lost)} {'is an item' if one else 'are items'} of `{name}`, whose "
                    f"score replaces its items in the models. This answer would take "
                    f"{'it' if one else 'them'} out of the predictors (no longer a confirmed "
                    f"exposure or covariate, left out, or the outcome), so the score could not be "
                    f"formed as declared.")
        else:
            one = len(new_entered) == 1
            said = (f"{_and(new_entered)} {'is' if one else 'are'} read by `{name}`'s reliability "
                    f"only (its repeat administration or reference measure). This answer would put "
                    f"{'it' if one else 'them'} in the model, where {'it' if one else 'they'} would "
                    f"adjust the score for itself.")
        raise Refusal(
            "scale_invalidated",
            f"{said} The scales answer rests on it, so it changes first and is asked again, never "
            f"kept silently: take the column out of the scale or stop scoring the scale, then "
            f"record this answer.",
            exits=_scales_exits(recorded, s, new_lost, new_entered))


for _kind in ROLE_CHANGING:
    register_validator(_kind, _answers_keep_the_scales_whole)

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
    from turbotab.core.models.survey import population_answer

    # MS4: under the surveyed population the scales stage blocks the correction (no design-based
    # estimator), so the record says so; the methods text restates this sentence whole when the
    # survey answer changes (``voice.restate``).
    population = population_answer(state)
    parts = []
    for s in d.scales:
        role = "an exposure" if s.role == "exposure" else "a covariate"
        if s.correction == "none":
            corrected = "its coefficient is not corrected for measurement error"
        elif population:
            corrected = (f"its correction by regression calibration from "
                         f"{SOURCE_WORDS[s.reliability]} was asked for, but under the surveyed "
                         f"population it has no design-based estimator, so it was blocked and "
                         f"recorded and the coefficient is not corrected")
        else:
            corrected = (f"its coefficient is corrected by regression calibration from "
                         f"{SOURCE_WORDS[s.reliability]}, as a secondary analysis beside the "
                         f"uncorrected one")
        parts.append(f"{_scale_words(s)}, a {s.kind} "
                     f"{'index' if s.kind == 'formative' else 'scale'} entering the models as "
                     f"{role}; {corrected}")
    return "; ".join(parts)


def _register_sentence() -> None:
    from turbotab.core.voice import register_sentence, restated_whole

    register_sentence("set_scales")(decision_sentence)
    restated_whole("set_scales")


_register_sentence()

COEFFICIENT_WORDS = {"omega_total": "ω-total", "omega_hierarchical": "ω-hierarchical"}


def joint_label(others: Sequence[str]) -> str:
    """The label of a score calibrated together with ``others`` (MODELING_SEQUENCE §2)."""
    return (f"Calibrated jointly with {_and(list(others))} (multivariate regression calibration; "
            f"Rosner, Spiegelman & Willett 1990): each corrected score's calibration conditions on "
            f"the others, their errors assumed independent of one another.")


def clustered_label(column: str, n_clusters: int | None = None) -> str:
    """The label of a correction on grouped rows (MODELING_SEQUENCE §2, ruling 7)."""
    many = f" ({int(n_clusters):,} of them)" if n_clusters else ""
    return (f"Both intervals keep each `{column}`'s rows together: the uncorrected one is the "
            f"coefficient table's cluster-robust interval (CR2), and the corrected one's bootstrap "
            f"resamples whole `{column}` clusters{many}.")


def uncorrected_alongside(others: Sequence[str]) -> str:
    """The concern on a correction beside another scale's score left uncorrected in the model."""
    one = len(others) == 1
    return (f"{_and(list(others))} {'is' if one else 'are'} also a scale score in the model, "
            f"entered uncorrected, so this correction reads {'it' if one else 'them'} as measured "
            f"without error; with two or more error-prone exposures, estimates “may become "
            f"attenuated, inflated, or can even change direction” (Freedman et al. 2011).")


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
    clustered = corr.get("clustered_by")
    resampling = f", resampling whole `{clustered}` clusters" if clustered else ""
    tail = (f"the corrected coefficient was obtained by regression calibration including all model "
            f"covariates, with bootstrap CIs re-estimating the reliability ({boots:,} "
            f"replicates{each}{resampling}); uncorrected and corrected estimates are both reported.")
    if clustered:
        tail += f" Both intervals are clustered by `{clustered}`."
    jointly = list(corr.get("jointly") or [])
    if jointly:
        tail += (f" It was calibrated jointly with {_and(jointly)} (multivariate regression "
                 f"calibration; Rosner, Spiegelman & Willett 1990), their errors assumed "
                 f"independent.")
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


__all__ = ["CONTRACT", "FORMATIVE", "PREDICTION", "ROLE_CHANGING", "SECONDARY", "TRANSIENT",
           "clustered_label", "codes_beyond", "decision_sentence", "joint_label",
           "methods_sentence", "retest_columns", "scale_breaks", "score_range",
           "uncorrected_alongside"]

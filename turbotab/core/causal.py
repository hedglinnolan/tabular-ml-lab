"""The causal lane: the shortest leash (V2 definition of done §2; MODELING_SEQUENCE §0 ruling 1 rung
(d), ruling 6, ruling 10; BLUEPRINT §11.3 and §13).

Double/debiased machine learning, targeted maximum likelihood and the post-double-selection lasso
(``turbotab/core/models/causal.py``) estimate the declared exposure's effect with flexible learners
over the adjustment set the disjunctive cause criterion's answers chose (``turbotab/core/
estimand.py``). Data-adaptive adjustment with valid inference over a candidate set chosen from
subject knowledge is rung (d) of ruling 1: allowed, and ranked high when the candidates are many
relative to n. This module holds the lane's leash; the stage that runs it is
``turbotab/core/stages/causal.py``.

**The leash** (BLUEPRINT §11.3: "a causal question gets a shorter leash than exploratory
prediction"):

* **Never under prediction** (refuse): no coefficient is read as an effect there.
* **Only after the plan** (refuse until answered): the exposure, its effect and every covariate's
  answers come first, so the candidate set is the analyst's, never the data's; a direct effect and
  an exposure family are outside the lane (refused with the reason). The Router states the lane
  rather than asking it: the primary model is the declared analysis, the top-ranked estimator is
  named one step away, and when the candidates are many relative to n the reason says why it
  ranks first.
* **The whole plan before any estimate**: the causal question comes before the models question,
  and the causal estimate is computed only once the model families are chosen too, so the primary
  model and the lane are declared together before either estimate is shown (the plan lock records
  both; MODELING_SEQUENCE §1 row 12).
* **The assumptions before any estimate** (refuse until declared): no unmeasured confounding,
  positivity, consistency and time ordering (Hernán & Robins, *Causal Inference: What If*, 2020,
  §3.1–3.5) are each shown with the diagnostic the data allow, and the decision records each one
  as declared. The estimate is computed only after.
* **Positivity** (block and record): 1% or more of the rows with a propensity outside
  ``[b, 1 − b]`` (b, tmle's truncation level ``5/(√n ln n)``; the 1% is this app's stated line;
  for the effect among the exposed, a propensity above ``1 − b``, its own reading), or
  a continuous exposure the covariates explain to R² ≥ 0.9 (a variance inflation factor of 10),
  are refused until the analyst trims to the overlap population
  (propensities in [0.1, 0.9]; Crump, Hotz, Imbens & Mitnik 2009, *Biometrika* 96:187) or records
  that every row is kept. The methods sentence then says which, with the violation's numbers.
* **A survey design** (MODELING_SEQUENCE ruling 6): the weights enter every nuisance fit and the
  estimating equation of the DML and TMLE estimators; the post-double-selection lasso's plug-in
  penalty is derived for independent, unweighted rows, so under the surveyed-population answer it
  is block and record, with the weighted partially linear model and the sample-only attestation as
  its exits.
* **Sensitivity to unmeasured confounding is required** (ruling 10). The E-value and the
  Cinelli–Hazlett robustness value are ESTIMAND's (``turbotab/core/models/effects.py``); the one
  call site is :func:`sensitivity_for`, and the methods text says what it reported.

**The method contracts** (BLUEPRINT §13) are :data:`CONTRACTS`: each estimator's slot, data scope,
needs, routing, storyboard, sentence and relations. The relations it touches in MODELING_SEQUENCE
§2 are :data:`RELATIONS`, each asserted by the chain test
(``turbotab/core/tests/acceptance/test_causal_lane.py``).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from turbotab.core.custom_sound import Customary, LabeledOption, Sound

CHERNOZHUKOV = "Chernozhukov et al. 2018, Econom J 21:C1"
DOUBLEML = "as R's DoubleML 1.0.2 computes it"
VAN_DER_LAAN = "van der Laan & Rubin 2006, Int J Biostat 2(1)"
TMLE_R = "as R's tmle 2.1.1 computes it"
BCH = "Belloni, Chernozhukov & Hansen 2014, Rev Econ Stud 81:608"
HERNAN = "Hernán & Robins 2020, Causal Inference: What If, §3"
CRUMP = "Crump et al. 2009, Biometrika 96:187"
GRUBER = "Gruber et al. 2022, Am J Epidemiol 191:1640"
SCHULER = "Schuler & Rose 2017, Am J Epidemiol 185:65"
WALTER = "Walter & Tiemeier 2009, Eur J Epidemiol 24:733"
HARRELL = "Harrell, Regression Modeling Strategies (2015), §4.4"

METHODS: tuple[str, ...] = ("dml_plr", "dml_irm", "tmle", "pds_lasso")
METHOD_LABELS: dict[str, str] = {
    "none": "The primary model only",
    "dml_plr": "Double ML, partially linear",
    "dml_irm": "Double ML, interactive",
    "tmle": "Targeted maximum likelihood",
    "pds_lasso": "Post-double-selection lasso",
}
METHOD_PHRASES: dict[str, str] = {
    "dml_plr": "double/debiased machine learning in the partially linear model",
    "dml_irm": "double/debiased machine learning in the interactive model",
    "tmle": "targeted maximum likelihood",
    "pds_lasso": "post-double-selection lasso",
}
TRIM_AT = 0.1  # Crump et al. 2009's rule of thumb: keep propensities in [0.1, 0.9]
ASSUMPTIONS: tuple[str, ...] = ("no_unmeasured_confounding", "positivity", "consistency",
                                "time_ordering")
ASSUMPTION_WORDS: dict[str, str] = {
    "no_unmeasured_confounding": "no unmeasured confounding",
    "positivity": "positivity",
    "consistency": "consistency",
    "time_ordering": "time ordering",
}
# Harrell's limiting sample size: a model fits about one parameter per 10 of n (a numeric
# outcome) or of the rarer outcome level (a yes/no one); more candidates than that is "many".
CANDIDATES_PER = 10


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


# ── the method contracts (BLUEPRINT §13), in the one registry (``turbotab.core.contracts``) ──

PACKAGE = "CAUSAL"


def _relation(id: str, kind: str, condition: str, target: str, says: str,
              exit: str | None = None, rung: str | None = None,
              purposes: tuple[str, ...] = ("inference",)) -> Any:
    from turbotab.core.contracts import Relation

    return Relation(kind, target, says, purposes=purposes, rung=rung,  # type: ignore[arg-type]
                    exits=(exit,) if exit else (), condition=condition, id=id)


# How one decision leads to another (MODELING_SEQUENCE §2), by id; each method's contract names
# the ones it touches, and the chain test asserts that each fires.
RELATIONS: dict[str, Any] = {r.name: r for r in (
    _relation("prediction_refuses", "conflicts", "purpose = prediction", "set_causal",
              "Under prediction no coefficient is read as an effect, so the causal lane is refused.",
              "Make the purpose inference", rung="refused", purposes=("prediction",)),
    _relation("plan_enables", "enables", "set_estimand + set_adjustment", "set_causal",
              "Because the exposure, its effect and every covariate's answers are declared, the "
              "causal lane is offered over that adjustment set."),
    _relation("exposure_invalidates", "invalidates", "set_estimand", "set_causal",
              "A new exposure re-asks the causal lane: its answer was given for another exposure."),
    _relation("rung_d", "conflicts", "set_adjustment", "data-driven selection",
              "Selection by the data is refused under inference except rung (d): the lasso and the "
              "learners choose only among the covariates the answers adjust for, never among those "
              "they leave out.",
              "Adjust for a covariate by its answers, not by the data", rung="refused"),
    _relation("assumptions_first", "precedes", "set_causal", "the estimate",
              "The four assumptions are declared, each beside its diagnostic, before any estimate is "
              "computed."),
    _relation("positivity_blocks", "conflicts", "a positivity violation", "the estimate",
              "Propensities outside the truncation bound (or a continuous exposure the covariates "
              "nearly determine) block the estimate until trimmed or recorded.",
              "Trim to the overlap population, or keep every row and record it",
              rung="block_and_record"),
    _relation("trim_changes_estimand", "implies", "trimming", "the estimand",
              "Trimming changes the estimand to the effect among the rows where both exposure levels "
              "are plausible; the sentence says so."),
    _relation("survey_weights", "implies", "set_survey = population",
              "every nuisance fit and the estimating equation",
              "Under the surveyed-population answer the weights enter every nuisance fit and the "
              "estimating equation, the folds keep each PSU together, and the variance is linearized "
              "over the design."),
    _relation("survey_blocks_pds", "conflicts", "set_survey = population", "pds_lasso",
              "The plug-in lasso's penalty assumes unweighted rows, so post-double selection under "
              "the surveyed-population answer is blocked and recorded.",
              "The weighted partially linear model, or a sample-only estimate recorded as such",
              rung="block_and_record"),
    _relation("clusters_group", "implies", "set_clusters / repeated units", "the folds and the "
              "variance",
              "Rows that belong together stay in one fold, and the variance is cluster-robust by the "
              "group, refused below the unit floor."),
    _relation("mi_blocks", "conflicts", "set_missing = multiple_imputation", "the causal lane",
              "Multiple imputation implies pooling every estimate; the causal lane is not pooled over "
              "imputations in v2, so incomplete rows are blocked and recorded.",
              "Estimate on the complete rows, its assumption stated", rung="block_and_record"),
    _relation("measure_marginal", "conflicts", "a conditional ratio measure", "the causal lane",
              "DML and TMLE estimate a marginal effect; a conditional odds or hazard ratio is "
              "another estimand.",
              "Declare the marginal risk difference", rung="refused"),
    _relation("form_single", "conflicts", "a spline or quintile exposure form", "dml_plr / pds_lasso",
              "A declared spline or quintiles of the exposure is several terms, not the one "
              "coefficient the lane estimates.",
              "Declare the exposure's form linear", rung="refused"),
    _relation("sensitivity_required", "implies", "set_causal", "sensitivity to unmeasured confounding",
              "The causal lane always carries a sensitivity analysis to unmeasured confounding "
              "(E-value; robustness value), computed by ESTIMAND's "
              "``models.effects.unmeasured_confounding``."),
    _relation("plan_lock", "implies", "the first causal estimate shown", "lock_plan",
              "The first causal estimate displayed locks the analysis plan, as any estimate does."),
    # Wave 2a: an exposure that changes within units is the time-varying lane's (g-methods); these
    # estimators adjust for a point exposure's confounders only, as standard regression does.
    _relation("time_varying_exposure", "conflicts", "the exposure changes over time within units",
              "the causal lane",
              "An exposure that changes over time is estimated by the g-methods of the time-varying "
              "lane: a confounder that earlier exposure changed biases a point-exposure estimate, "
              "by these learners as by standard regression.",
              "Answer the time-varying question (a marginal structural model or the g-formula)",
              rung="refused"),
)}

_COMMON = ("prediction_refuses", "plan_enables", "exposure_invalidates", "rung_d",
           "assumptions_first", "positivity_blocks", "clusters_group", "mi_blocks",
           "sensitivity_required", "plan_lock", "time_varying_exposure")
# The leash per purpose (BLUEPRINT §11.3): refused under prediction; under inference rung (d),
# allowed after the plan, the assumptions declared first, a positivity violation blocked and recorded.
_LEASH = {"prediction": "refused", "inference": "available"}
_CROSS_FIT = ("every nuisance prediction comes from learners fit on the other folds of the analyzed "
              "rows (cross-fitting), never on the row itself; the outcome's own learners make it "
              "the outcome model's scope")


def _contract(key: str, needs: tuple[str, ...], storyboard: tuple[str, ...], relations: tuple[str, ...],
              scope_note: str, customary: str, sound: str) -> Any:
    from turbotab.core.contracts import ContractOption, MethodContract, register_contract

    return register_contract(MethodContract(
        key=key, label=METHOD_LABELS[key], slot="model", scope="model", scope_note=scope_note,
        needs=needs, question="causal", place="MODELING_SEQUENCE §0 ruling 1, rung (d)",
        decision="set_causal", stage="causal", leash=_LEASH, storyboard=storyboard,
        sentence="turbotab.core.causal:methods_sentence",
        options=(ContractOption(
            key, METHOD_LABELS[key], customary,
            sound={"inference": sound,
                   "prediction": "Refused: under prediction no coefficient is read as an effect."},
            rung={"inference": "available", "prediction": "refused"}),),
        relations=tuple(RELATIONS[r] for r in relations), package=PACKAGE,
        sources=(HERNAN, CHERNOZHUKOV if key.startswith("dml") else
                 VAN_DER_LAAN if key == "tmle" else BCH)))


CONTRACTS: dict[str, Any] = {c.key: c for c in (
    _contract(
        "dml_plr",
        ("a declared exposure, numeric", "the adjustment set's answers", "a numeric or yes/no "
         "outcome"),
        ("Predict the outcome from the adjustment set on the other folds",
         "Predict the exposure from the adjustment set on the other folds",
         "Keep what each prediction misses",
         "Regress the outcome's residual on the exposure's",
         "Repeat over sample splits; take the median"),
        (*_COMMON, "survey_weights", "measure_marginal", "form_single"), _CROSS_FIT,
        f"Standard in econometrics, rare in nutrition so far ({CHERNOZHUKOV})",
        "Sound: an orthogonal score with cross-fitting gives valid intervals with flexible "
        "learners; for a yes/no exposure it is a variance-weighted average, not the average effect"),
    _contract(
        "dml_irm",
        ("a declared exposure, yes/no", "the adjustment set's answers",
         "a numeric or yes/no outcome"),
        ("Predict the propensity on the other folds",
         "Predict the outcome at each exposure level on the other folds",
         "Combine them in the doubly robust score",
         "Average the score; repeat over sample splits; take the median"),
        (*_COMMON, "survey_weights", "trim_changes_estimand", "measure_marginal"), _CROSS_FIT,
        f"Standard in econometrics, rare in nutrition so far ({CHERNOZHUKOV})",
        "Sound: a doubly robust score with cross-fitting gives valid intervals with flexible "
        "learners"),
    _contract(
        "tmle",
        ("a declared exposure, yes/no", "the adjustment set's answers",
         "a numeric or yes/no outcome"),
        ("Fit the outcome model and the propensity",
         "Fluctuate the outcome model along the clever covariates",
         "Average the targeted predictions at each exposure level",
         "Read the interval off the influence curve"),
        (*_COMMON, "survey_weights", "trim_changes_estimand", "measure_marginal"),
        "every analyzed row for main-terms models; " + _CROSS_FIT + " for flexible learners",
        f"Growing in epidemiology, rare in nutrition so far ({SCHULER})",
        "Sound: doubly robust and efficient, and keeps each estimate inside the outcome's range"),
    _contract(
        "pds_lasso",
        ("a declared exposure", "the adjustment set's answers", "a numeric outcome"),
        ("Lasso the outcome on the candidates",
         "Lasso the exposure on the candidates",
         "Keep the union of both selections",
         "Regress the outcome on the exposure and the union"),
        (*_COMMON, "survey_blocks_pds", "form_single"),
        "every analyzed row: the lasso selects among the declared candidates only, and the outcome "
        "lasso reads the outcome",
        f"Standard in applied economics ({BCH})",
        "Sound when the candidates are many for n: selecting by both lassos keeps the intervals "
        "valid; with few there is little to select"),
)}


# ── what the lane needs from the plan ────────────────────────────────────────


def current_causal(state: Any) -> Any:
    """The recorded causal answer while it was given for the declared exposure under inference;
    None otherwise (a new exposure re-asks it, never keeps it: MODELING_SEQUENCE §2)."""
    from turbotab.core.estimand import current_estimand

    spec = _get(state, "causal")
    if spec is None or _get(state, "purpose") != "inference":
        return None
    estimand = current_estimand(state)
    if estimand is None or _get(estimand, "family"):
        return None
    return spec if _get(spec, "exposure") == _get(estimand, "exposure") else None


def plan_reason(state: Any) -> str | None:
    """Why the causal lane is not offered on this plan, or None when it is (or the plan is still
    being answered: the Router holds the question then)."""
    from turbotab.core.estimand import current_estimand

    purpose = _get(state, "purpose")
    if purpose == "prediction":
        return ("Under prediction no coefficient is read as an effect, so no causal estimate is "
                "offered.")
    spec = current_estimand(state)
    if spec is None:
        return None
    if _get(spec, "family"):
        return ("DML and TMLE estimate one exposure's effect; an exposure family is reported "
                "feature-wise, each in turn.")
    if _get(spec, "effect") == "direct":
        return ("A direct effect needs mediation methods, which the causal lane does not fit; it "
                "estimates the total effect.")
    task = _get(state, "task")
    if task is not None and task not in ("regression", "binary"):
        return (f"The causal lane estimates the effect on a numeric or yes/no outcome; this one is "
                f"{str(task).replace('_', ' ')}.")
    return None


STATED = ("the primary model estimates the declared effect; double/debiased machine learning, "
          "targeted maximum likelihood and post-double selection are one step away.")


def stated_reason(first: str | None, many: bool, candidates: int, n_limit: int) -> str:
    """The Router's stated skip: the primary model alone, the top-ranked estimator one step away,
    and when the candidates are many relative to n, why it ranks first (rung (d) ranks high)."""
    label = METHOD_LABELS.get(str(first), "a causal estimator")
    if many and first == "pds_lasso":
        return (f"the primary model estimates the declared effect; with {candidates:,} candidate "
                f"terms for a limiting sample size of {n_limit:,} (more than one per 10), "
                f"{label.lower()} ranks first and is one step away.")
    if many:
        return (f"the primary model estimates the declared effect on more than one candidate term "
                f"per 10 of its limiting sample size ({candidates:,} for {n_limit:,}); "
                f"{label.lower()} and the other causal estimators are one step away.")
    return (f"the primary model estimates the declared effect; {label.lower()} and the other "
            f"causal estimators are one step away.")


TIME_VARYING_REASON = ("{exposure} changes over time within units, so its effect is estimated by "
                       "the time-varying lane's g-methods; the causal lane estimates a point "
                       "exposure's effect.")


def causal_gate(state: Any, design: Any,
                time_varying: Any = None) -> tuple[str, str | None] | None:
    """The Router's gate for the causal question: not applicable (with the reason) under
    prediction or outside the lane; otherwise stated, never asked by default: the primary model is
    the declared analysis and the lane is one step away ("Ask me anyway"), its top-ranked
    estimator named, and when the candidates are many relative to n, why it ranks first (rung (d):
    "allow, ranked high"). Stated while its card computes too, so the lane never holds the models
    question back."""
    reason = plan_reason(state)
    if reason is not None:
        return ("not_applicable", reason)
    from turbotab.core.estimand import adjustment_answer, current_estimand, estimand_gate
    from turbotab.core.time_varying import exposure_varies

    if exposure_varies(state, time_varying):  # relation ``time_varying_exposure``
        spec = current_estimand(state)
        return ("not_applicable",
                TIME_VARYING_REASON.format(exposure=_tick(_get(spec, "exposure"))))

    found = estimand_gate(state)
    if found is not None and found[0] == "not_applicable":
        return ("not_applicable", found[1] or "No exposure is declared, so there is no effect to "
                                              "estimate.")
    spec = current_estimand(state)
    if spec is None or adjustment_answer(state) is None:
        return None  # it waits behind the exposure and adjustment questions, as every later one
    data = getattr(design, "data", design)
    if not isinstance(data, Mapping) or _get(data, "exposure") != _get(spec, "exposure"):
        return ("skipped", STATED)
    if not data.get("offered"):
        return ("not_applicable", str(data.get("reason") or "The causal lane does not apply."))
    return ("skipped", str(data.get("stated") or STATED))


def many_candidates(n_limit: int, candidates: int) -> bool:
    """Whether the candidates are many relative to n: more than one per 10 of the limiting sample
    size (Harrell's rule; :data:`CANDIDATES_PER`)."""
    return candidates * CANDIDATES_PER > max(0, int(n_limit))


# ── the options, labeled (north star 5) ──────────────────────────────────────


def options(exposure_kind: str | None, task: str | None, many: bool,
            survey_population: bool) -> list[LabeledOption]:
    """The causal question's options for this plan, soundest first, each labeled customary in
    its field and sound for inference (BLUEPRINT north star 5)."""
    binary = exposure_kind == "binary"
    linear_outcome = task == "regression"
    rows: list[tuple[int, LabeledOption]] = []

    def add(rank: int, key: str, field: str, text: str, source: str, verdict: str, reason: str) -> None:
        rows.append((rank, LabeledOption(
            key=key, label=METHOD_LABELS[key],
            customary=Customary(field=field, text=text, source=source),
            sound=Sound(purpose="inference", verdict=verdict, reason=reason))))

    if binary:
        add(1, "tmle", "epidemiology", "growing in epidemiology; rare in nutrition so far", SCHULER,
            "sound", "doubly robust, efficient, and keeps each estimate inside the outcome's range")
        add(2, "dml_irm", "econometrics", "standard in econometrics; rare in nutrition so far",
            CHERNOZHUKOV, "sound",
            "doubly robust score with cross-fitting: valid intervals with flexible learners")
        add(4, "dml_plr", "econometrics", "standard in econometrics; rare in nutrition so far",
            CHERNOZHUKOV, "conditional",
            "for a yes/no exposure it estimates a variance-weighted average, not the average effect")
    else:
        add(1, "dml_plr", "econometrics", "standard in econometrics; rare in nutrition so far",
            CHERNOZHUKOV, "sound",
            "orthogonal score with cross-fitting: valid intervals with flexible learners")
    if linear_outcome:
        if survey_population:
            add(6, "pds_lasso", "economics", "standard in applied economics", BCH, "unsound",
                "its plug-in penalty assumes unweighted rows, so a population estimate is blocked")
        elif many:
            add(0, "pds_lasso", "economics", "standard in applied economics", BCH, "sound",
                "many candidates for n: selecting by both lassos keeps the intervals valid")
        else:
            add(5, "pds_lasso", "economics", "standard in applied economics", BCH, "conditional",
                "with few candidates for n there is little to select; valid all the same")
    add(9, "none", "nutritional epidemiology",
        "a regression adjusted for a set chosen from subject knowledge is the field's analysis",
        WALTER,
        "conditional", "valid when the outcome model's form is right; the lane checks that")
    rows.sort(key=lambda r: r[0])
    return [o for _, o in rows]


# ── the assumptions, each with its diagnostic ────────────────────────────────


def assumption_card(*, exposure: str, exposure_kind: str, adjusted: Sequence[str],
                    timing_unknown: Sequence[str], overlap: Mapping[str, Any] | None,
                    variation: Mapping[str, Any] | None, contrast: str | None,
                    level: str | None) -> list[dict[str, Any]]:
    """The four assumptions (Hernán & Robins 2020, §3), each stated in one line beside what the
    data say about it. Shown before any estimate; the decision declares each one."""
    x = _tick(exposure)
    out: list[dict[str, Any]] = []
    shown = _listing(adjusted, limit=6) if adjusted else "no covariate"
    out.append({
        "key": "no_unmeasured_confounding", "label": "No unmeasured confounding",
        "statement": f"Given the adjustment set, {x} is as good as randomly assigned.",
        "diagnostic": (f"Adjusted for {shown}, as the disjunctive cause criterion's answers "
                       f"decided. No data can confirm this: a sensitivity analysis to an "
                       f"unmeasured confounder is required beside the estimate."),
        "status": "untestable", "source": HERNAN})
    if exposure_kind == "binary" and overlap is not None:
        b = float(overlap["bound"])
        diag = (f"Propensities run from {overlap['min_exposed']:.3f} among the exposed and up to "
                f"{overlap['max_unexposed']:.3f} among the unexposed; "
                f"{overlap['n_outside']:,} rows lie outside [{b:.4f}, {1 - b:.4f}]; effective "
                f"sample sizes {overlap['ess_exposed']:,.0f} of {overlap['n_exposed']:,} exposed "
                f"and {overlap['ess_unexposed']:,.0f} of {overlap['n_unexposed']:,} unexposed.")
        statement = f"Every kind of participant could have had either level of {x}."
        status = "violated" if overlap.get("violated") else "checked"
    elif variation is not None:
        diag = (f"The covariates explain {variation['r2']:.0%} of {x}'s variance; what is left has "
                f"a standard deviation of {variation['residual_sd']:.4g}.")
        statement = f"{x} still varies at every level of the covariates."
        status = "violated" if variation.get("violated") else "checked"
    else:
        diag, status = "Computed with the estimate.", "pending"
        statement = f"{x} varies at every level of the covariates."
    out.append({"key": "positivity", "label": "Positivity", "statement": statement,
                "diagnostic": diag, "status": status, "source": HERNAN})
    what = (f"as {_tick(level)} against the other level" if exposure_kind == "binary" and level
            else "per unit")
    energy = {"substitution": "; as a substitution at fixed total energy, which names the "
                              "intervention (Tomova et al. 2022)",
              "addition": "; as an addition of its calories, every other source fixed"}.get(
        contrast or "", "")
    out.append({
        "key": "consistency", "label": "Consistency",
        "statement": f"{x} is one well-defined intervention, so the outcome observed under it is "
                     f"the outcome it would cause.",
        "diagnostic": f"{x} is compared {what}{energy}.", "status": "stated", "source": HERNAN})
    unknown = list(timing_unknown)
    out.append({
        "key": "time_ordering", "label": "Time ordering",
        "statement": f"The covariates precede {x}, and {x} precedes the outcome.",
        "diagnostic": ((f"{_listing(unknown)} {'is' if len(unknown) == 1 else 'are'} of unknown "
                        f"timing and stay out of the primary set (the declared with-and-without "
                        f"pair); the rest were answered as set before the exposure.") if unknown
                       else "Every adjusted covariate was answered as set before the exposure."),
        "status": "stated", "source": HERNAN})
    return out


# ── sensitivity to unmeasured confounding (ruling 10) ────────────────────────


CHERNOZHUKOV_OVB = "Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, NBER w30302"
# Why an estimator's effect is not one least-squares coefficient (no Cinelli–Hazlett robustness
# value), in words true for every learner it takes: TMLE with main-terms models fits them on every
# row, not by cross-fitting, so its reason names what it averages instead.
NOT_ONE_COEFFICIENT: dict[str, str] = {
    "dml_irm": "double/debiased machine learning in the interactive model fits its nuisance "
               "models by cross-fitted learners",
    "tmle": "targeted maximum likelihood averages its targeted outcome model's predictions, not "
            "one coefficient",
}
# The robustness value's name in the methods text, per estimator whose effect is one coefficient.
ROBUSTNESS_NAMES: dict[str, str] = {
    "pds_lasso": "the Cinelli–Hazlett robustness value (each selected covariate a named benchmark)",
    "dml_plr": ("the Cinelli–Hazlett robustness value of the final least-squares step, the "
                "outcome's residual on the exposure's (the form the omitted-variable bound of "
                f"{CHERNOZHUKOV_OVB} takes in the partially linear model), the median over the "
                "sample splits"),
}


def final_stage_robustness(final_stage: Sequence[Mapping[str, float]], *, exposure: str,
                           estimate: float, alpha: float = 0.05) -> dict[str, Any]:
    """The partially linear model's robustness value: the Cinelli–Hazlett robustness value of each
    split's final least-squares step (the outcome's residual regressed on the exposure's, no
    intercept: its classical t and ``n − 1`` degrees of freedom, as ``lm(u ~ 0 + v)`` reports them),
    by ESTIMAND's :func:`turbotab.core.models.effects.robustness_value`, the median over splits as
    the estimate is. In the partially linear model the omitted-variable bias bound of Chernozhukov
    et al. 2022 (``|bias| ≤ S·C_Y·C_D``, ``S² = E[ε²]/E[(D − m)²]``) is this regression's, so with
    both strengths equal its point robustness value is Cinelli & Hazlett's ``½(√(f⁴ + 4f²) − f²)``,
    ``f² = θ²/S²``; the value at α is Cinelli & Hazlett's for the same fit."""
    import numpy as np

    from turbotab.core.models import effects

    ts = [float(s["t"]) for s in final_stage]
    dofs = [float(s["dof"]) for s in final_stage]
    return {"exposure": exposure, "estimate": float(estimate),
            "t": float(np.median(ts)), "dof": float(np.median(dofs)),
            "partial_r2": float(np.median([effects.partial_r2(t, f) for t, f in zip(ts, dofs)])),
            "rv": float(np.median([effects.robustness_value(t, f) for t, f in zip(ts, dofs)])),
            "rv_alpha": float(np.median([effects.robustness_value(t, f, alpha=alpha)
                                         for t, f in zip(ts, dofs)])),
            "benchmarks": [], "alpha": alpha, "splits": len(ts),
            "step": "the final least-squares step (the outcome's residual on the exposure's)"}


def sensitivity_for(estimates: Sequence[Mapping[str, Any]], *, method: str, exposure: str,
                    outcome: str, outcome_sd: float | None, matrix: Any = None, y: Any = None,
                    exposure_column: str | None = None,
                    benchmarks: Mapping[str, Sequence[str]] | None = None,
                    final_stage: Sequence[Mapping[str, float]] | None = None,
                    ratio_refused: str | None = None, population: str = "all") -> dict[str, Any]:
    """Sensitivity to unmeasured confounding, REQUIRED in the causal lane (MODELING_SEQUENCE §0
    ruling 10), by ESTIMAND's :func:`turbotab.core.models.effects.unmeasured_confounding`, the one
    function every inference result calls (and its ``robustness_value``):

    * a numeric outcome's difference: the E-value of the standardized difference (VanderWeele &
      Ding 2017's approximation, from the outcome's standard deviation and the estimate's standard
      error);
    * a yes/no outcome: the E-value of the risk ratio the estimator reports beside its risk
      difference (targeted maximum likelihood's marginal one; the interactive model's from its
      doubly robust means, marginal or among the exposed);
    * post-double selection, whose estimate is a least-squares coefficient on the exposure and the
      selected covariates (``matrix``, ``y``): the Cinelli–Hazlett robustness value too, ranked
      first, each selected covariate a named benchmark (``benchmarks``);
    * the partially linear model, whose estimate is one least-squares coefficient of the outcome's
      residual on the exposure's (``final_stage``): that fit's robustness value, ranked first
      (:func:`final_stage_robustness`), for a numeric or a yes/no outcome.

    The interactive model's and TMLE's effects average predictions, so they have no robustness
    value; a risk difference with no ratio beside it (the partially linear model on a yes/no
    outcome, or a ratio refused because a level has no events: ``ratio_refused``) has no E-value.
    Each says so, and a lane estimate whose required analysis could not be computed says that too.
    ``estimates`` are the artifact's (``CausalEstimate`` dumps), the reported one first."""
    from turbotab.core.models import effects
    from turbotab.core.stages.effects import sensitivity_reading

    first = dict(estimates[0])
    out: dict[str, Any] = {"required": True, "computed": False, "method": method,
                           "measure": first["measure"],
                           "estimate": first.get("estimate"), "methods": [], "e_value": None,
                           "robustness": None, "reading": "", "not_computed": None}
    least_squares = matrix is not None and y is not None and exposure_column is not None
    what = "the estimate"
    if first["measure"] == "mean_difference":
        found = effects.unmeasured_confounding(
            measure="mean_difference", estimate=float(first["estimate"]), ci_low=first.get("ci_low"),
            ci_high=first.get("ci_high"), se=first.get("se"), outcome_sd=outcome_sd,
            matrix=matrix if least_squares else None, y=y if least_squares else None,
            exposure_column=exposure_column if least_squares else None, benchmarks=benchmarks)
    else:
        ratio = next((e for e in estimates if e.get("measure") == "risk_ratio"), None)
        found = (effects.unmeasured_confounding(
            measure="risk_ratio", estimate=float(ratio["estimate"]), ci_low=ratio.get("ci_low"),
            ci_high=ratio.get("ci_high")) if ratio is not None else
            {"methods": [], "e_value": None, "robustness": None})
        what = ("the risk ratio among the exposed" if population == "exposed"
                else "the marginal risk ratio")
        if least_squares:  # a linear-probability least-squares coefficient: its robustness value
            found["robustness"] = effects.linear_sensitivity(
                matrix, y, exposure_column, benchmarks).as_dict()
            found["methods"] = ["robustness_value", *found["methods"]]
    if final_stage and found.get("robustness") is None:
        found["robustness"] = final_stage_robustness(final_stage, exposure=exposure,
                                                     estimate=float(first["estimate"]))
        found["methods"] = ["robustness_value", *found["methods"]]
    out.update(methods=list(found["methods"]), e_value=found.get("e_value"),
               robustness=found.get("robustness"), computed=bool(found["methods"]))
    missing = []
    if found.get("robustness") is None:
        missing.append("no robustness value: it is defined for one least-squares coefficient "
                       f"({effects.CINELLI_HAZLETT}), and "
                       + NOT_ONE_COEFFICIENT.get(method, f"{METHOD_PHRASES.get(method, method)} "
                                                         f"gave no least-squares fit for it"))
    if found.get("e_value") is None:
        missing.append(f"no E-value: {ratio_refused}" if ratio_refused else
                       "no E-value: a risk difference with no risk ratio beside it carries no "
                       "risks to form one from")
    said = "; ".join(missing)
    out["not_computed"] = (said[:1].upper() + said[1:] + ".") if missing else None
    out["reading"] = sensitivity_reading(found, exposure, outcome, what)
    return out


def sensitivity_sentence(sensitivity: Mapping[str, Any] | None) -> str:
    """The methods text's sentence for the lane's sensitivity analysis (ESTIMAND's wording), or,
    where the required analysis could not be computed, why."""
    from turbotab.core.stages.effects import sensitivity_clause

    if not sensitivity:
        return ""
    if sensitivity.get("computed"):
        method = str(sensitivity.get("method") or "pds_lasso")
        return " " + sensitivity_clause(sensitivity["methods"], {
            "robustness_value": ROBUSTNESS_NAMES.get(method, ROBUSTNESS_NAMES["pds_lasso"])})
    return (" Sensitivity to unmeasured confounding, required in the causal lane, could not be "
            f"computed for this estimate: {str(sensitivity.get('not_computed') or '')[:1].lower()}"
            f"{str(sensitivity.get('not_computed') or '')[1:]}")


# ── the methods sentence ─────────────────────────────────────────────────────


def _rows(n: int) -> str:
    return f"{n:,} complete {'row' if n == 1 else 'rows'}"


def methods_sentence(*, method: str, exposure: str, outcome: str, effect: str,
                     adjusted: Sequence[str], learner_words: str, folds: int, repetitions: int,
                     n: int, population: str = "all", level: str | None = None,
                     bound: float | None = None, trimmed: int = 0, n_trim_kept: int | None = None,
                     weight: str | None = None, cluster: str | None = None,
                     selected: Mapping[str, Sequence[str]] | None = None,
                     candidates: int | None = None, trim_at: float = TRIM_AT,
                     measure_words: str = "difference in the mean outcome",
                     violation: str | None = None, sample_only: bool = False,
                     ratios_refused: Mapping[str, str] | None = None) -> str:
    """The sentence the methods section carries for the causal estimate (asserted verbatim by the
    acceptance test). Every number in it is the estimate's own.

    The block-and-record choices it rests on are said in it, never left to the Record alone:
    ``violation`` (what the estimator's own positivity reading found, when every row was kept on
    the record) replaces positivity among the declared assumptions with the violation and its
    consequence; ``sample_only`` (an estimate recorded as unweighted under the surveyed-population
    answer, post-double selection's exit) says the estimate is unweighted and for these
    participants only;
    ``ratios_refused`` (a yes/no outcome whose level has no events) says which ratios are not
    reported and why."""
    x, y = _tick(exposure), _tick(outcome)
    covariates = _listing(adjusted, limit=6) if adjusted else "no covariate"
    head = f"The {effect} effect of {x} on {y} was also estimated"
    # A yes/no exposure is compared between its levels; in the partially linear model that
    # contrast is averaged with weights by the exposure's residual variance, not over everyone.
    per = (f"between {x} = {_tick(level)} and the other level" if level is not None
           else f"per unit of {x}")
    if method == "dml_plr":
        what = (f"the {measure_words} {per}, averaged with weights by the exposure's residual "
                f"variance rather than over everyone" if level is not None
                else f"the {measure_words} {per}")
        text = (f"{head} by double/debiased machine learning in the partially linear model "
                f"({CHERNOZHUKOV}; {DOUBLEML}), as {what}: "
                f"{learner_words} predicted the outcome and the exposure from {covariates}, each "
                f"fit on the other {folds - 1} of {folds} folds, and the outcome's residual was "
                f"regressed on the exposure's; the estimate is the median over {repetitions} "
                f"random sample {'split' if repetitions == 1 else 'splits'}")
    elif method == "dml_irm":
        exposed = population == "exposed"
        whom = (f"the effect among the exposed ({x} = {_tick(level)})" if exposed
                else _everyone(trimmed))
        # The effect among the exposed needs the outcome model of the unexposed only (DoubleML's
        # ATTE score fits g₀ alone).
        predicted = ("the outcome among the unexposed" if exposed
                     else "the outcome at each exposure level")
        text = (f"{head} by double/debiased machine learning in the interactive model "
                f"({CHERNOZHUKOV}; {DOUBLEML}), as {whom}: the {measure_words} between {x} = "
                f"{_tick(level)} and the other level; {learner_words} predicted {predicted} "
                f"and the propensity from {covariates}, each fit on the other {folds - 1} of "
                f"{folds} folds, with propensities bounded to [{bound:.4f}, {1 - bound:.4f}]; the "
                f"estimate is the median over {repetitions} random sample "
                f"{'split' if repetitions == 1 else 'splits'}")
    elif method == "tmle":
        fitted = ("each fit on every analyzed row" if learner_words.startswith("main-terms")
                  else f"each fit on the other {folds - 1} of {folds} folds")
        text = (f"{head} by targeted maximum likelihood ({VAN_DER_LAAN}; {TMLE_R}), as "
                f"{_everyone(trimmed)}: the {measure_words} between {x} = {_tick(level)} "
                f"and the other level; {learner_words} modeled the outcome and the propensity given "
                f"{covariates}, {fitted}, with the propensity truncated below at {bound:.4f} for "
                f"each level ({GRUBER}), and the interval is read off the influence curve")
    elif method == "pds_lasso":
        sel = selected or {}
        union = list(sel.get("union") or [])
        p = len(adjusted) if candidates is None else int(candidates)
        text = (f"{head} by post-double-selection lasso ({BCH}), as the {measure_words} "
                f"{per}: plug-in lassos over the {p} declared model "
                f"{'term' if p == 1 else 'terms'} of {covariates} selected "
                f"{_listing(sel.get('outcome') or [], limit=5) or 'none'} for the outcome and "
                f"{_listing(sel.get('exposure') or [], limit=5) or 'none'} for the exposure, and "
                f"the outcome was regressed on {x} and "
                f"{_listing(union, limit=6) if union else 'no covariate'}, with HC3 standard "
                f"errors")
    else:
        raise ValueError(f"No methods sentence for {method!r}.")
    text += f", on {_rows(n)}."
    if trimmed:
        text += (f" The {trimmed:,} {'row' if trimmed == 1 else 'rows'} with a propensity outside "
                 f"[{trim_at:g}, {1 - trim_at:g}] {'was' if trimmed == 1 else 'were'} trimmed "
                 f"({CRUMP}), so the estimate is the effect among the {n_trim_kept:,} rows where "
                 f"both exposure levels are plausible.")
    if weight:
        text += (f" The survey weight {_tick(weight)} entered every nuisance fit and the estimating "
                 f"equation, the folds kept each PSU's rows together, and the variance is "
                 f"linearized over the design's strata and PSUs.")
    elif cluster:
        text += (f" The folds kept each {_tick(cluster)}'s rows together, and the variance is "
                 f"cluster-robust by {_tick(cluster)}.")
    if sample_only:
        why = (": the plug-in lasso's penalty assumes unweighted rows, so" if method == "pds_lasso"
               else ", so")
        text += (f" As recorded, the survey weights were not used{why} the estimate is unweighted "
                 f"and describes these participants only, not the surveyed population.")
    if ratios_refused:
        text += _ratios_refused_sentence(ratios_refused)
    if violation:
        where = (f"one level of {x} is all but impossible given the covariates"
                 if level is not None else f"{x} barely varies given the covariates")
        text += (" It rests on the declared assumptions of no unmeasured confounding given the "
                 f"adjustment set, consistency and time ordering. Positivity is practically "
                 f"violated: {violation}; every row was kept, as recorded, so the estimate "
                 f"extrapolates where {where}, a stated limitation.")
    else:
        text += (" It rests on the declared assumptions of no unmeasured confounding given the "
                 "adjustment set, positivity, consistency and time ordering.")
    return text


def _ratios_refused_sentence(refused: Mapping[str, str]) -> str:
    """Which ratios of risks are not reported (``{"marginal risk ratio": why, …}``), and why (a
    level with no events, or only events)."""
    reasons = list(dict.fromkeys(refused.values()))
    if len(reasons) == 1:
        return f" No {' or '.join(refused)} is reported: {reasons[0]}."
    return "".join(f" No {name} is reported: {why}." for name, why in refused.items())


def _everyone(trimmed: int) -> str:
    return ("the average effect in the overlap population" if trimmed
            else "the average effect over everyone")


# ── the answer's sentence (the Record) ───────────────────────────────────────


def _record_sentence(d: Any, state: Any, ctx: Any) -> str:
    if d.method == "none":
        return (f"No causal machine-learning estimate was set beside the primary model for the "
                f"effect of {_tick(d.exposure)}")
    from turbotab.core.models.causal import LEARNER_WORDS

    words = LEARNER_WORDS.get(str(d.learner), "the default learner for the table's size")
    text = (f"The effect of {_tick(d.exposure)} is also estimated by {METHOD_PHRASES[d.method]}"
            + ("" if d.method == "pds_lasso" else f", its nuisance models fit by {words}")
            + (f", over {d.folds} folds and {d.repetitions} sample "
               f"{'split' if d.repetitions == 1 else 'splits'}"
               if d.method in ("dml_plr", "dml_irm")
               or (d.method == "tmle" and d.learner not in (None, "linear"))
               else "")
            + (", as the effect among the exposed" if d.population == "exposed" else ""))
    declared = [ASSUMPTION_WORDS[a] for a in ASSUMPTIONS if a in d.assumptions]
    text += f"; declared before any estimate: {', '.join(declared[:-1])} and {declared[-1]}" \
        if len(declared) > 1 else ""
    if d.trim is not None:
        text += (f"; rows with a propensity outside [{d.trim:g}, {1 - d.trim:g}] are trimmed, so it "
                 f"estimates the effect in the overlap population")
    # Kept on the record: said as a violation only where the card read one for this estimand (the
    # card absent, the record stands as the analyst wrote it).
    if d.acknowledged and card_violation(ctx, d) is not False:
        text += ("; every row is kept although positivity is practically violated, a stated "
                 "limitation")
    if d.sample_only:
        text += "; unweighted, for these participants only, as recorded"
    return text


# ── refusals (the leash) ─────────────────────────────────────────────────────


def _refusal(code: str, message: str, exits: Sequence[Mapping[str, Any]]) -> Exception:
    from turbotab.core.decisions import Refusal

    return Refusal(code, message, exits=exits)


def _state(ctx: Any) -> Any:
    from turbotab.core.decisions import _state as state_of

    return state_of(ctx)


def _design_artifact(ctx: Any) -> Mapping[str, Any] | None:
    from turbotab.core.sequence import artifact

    found = artifact(ctx, "causal_design")
    return found if found and found.get("purpose") == "inference" else None


def _with(decision: Any, **change: Any) -> Any:
    return decision.model_copy(update=change)


def _causal_is_for_inference(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import SetPurpose

    if _get(_state(ctx), "purpose") == "prediction":
        raise _refusal(
            "not_inference",
            "Under prediction no coefficient is read as an effect, so no causal estimate is "
            "offered; the causal lane is asked under inference.",
            [{"label": "Make the purpose inference", "decision": SetPurpose(purpose="inference")}])


def _point_exposure_only(decision: Any, ctx: Any) -> None:
    """Relation ``time_varying_exposure`` (refused): the ``time_varying`` stage read the declared
    exposure as changing within units, so its effect is the g-methods' (the time-varying lane)."""
    from turbotab.core.decisions import SetCausal
    from turbotab.core.sequence import artifact
    from turbotab.core.time_varying import exposure_varies

    if decision.method == "none" or not exposure_varies(_state(ctx), artifact(ctx, "time_varying")):
        return
    raise _refusal(
        "time_varying_exposure",
        TIME_VARYING_REASON.format(exposure=_tick(decision.exposure)) + " A confounder that "
        "earlier exposure changed would bias these learners' estimate as it biases standard "
        "regression's.",
        [{"label": "Answer the time-varying question (a marginal structural model or the "
                   "g-formula)", "decision": None},
         {"label": "The primary model only",
          "decision": SetCausal(exposure=decision.exposure, method="none")}])


def _causal_follows_the_plan(decision: Any, ctx: Any) -> None:
    from turbotab.core.estimand import adjustment_answer, current_estimand
    from turbotab.core.voice import question_name

    state = _state(ctx)
    if state is None:
        return
    spec = current_estimand(state)
    if spec is None:
        raise _refusal("no_estimand",
                       f"Declare the exposure and its effect first ({question_name('estimand')}); "
                       f"the causal lane estimates that effect, over the adjustment set.",
                       [{"label": f"Answer {question_name('estimand')} first", "decision": None}])
    reason = plan_reason(state)
    if reason is not None:
        raise _refusal("outside_the_lane", reason, [{"label": "Keep the primary model only",
                                                     "decision": _with(decision, method="none")}])
    if decision.exposure != _get(spec, "exposure"):
        raise _refusal("other_exposure",
                       f"The exposure is {_tick(_get(spec, 'exposure'))}, not "
                       f"{_tick(decision.exposure)}; the causal lane answers for the exposure "
                       f"declared.",
                       [{"label": f"Answer for {_tick(_get(spec, 'exposure'))}",
                         "decision": _with(decision, exposure=_get(spec, "exposure"))}])
    if decision.method != "none" and adjustment_answer(state) is None:
        raise _refusal("no_adjustment",
                       f"Answer {question_name('adjustment')} first: the causal lane adjusts for the "
                       f"covariates the answers keep, and selects among them only (rung (d)).",
                       [{"label": f"Answer {question_name('adjustment')} first", "decision": None}])


def _causal_fits_the_estimand(decision: Any, ctx: Any) -> None:
    """The method must estimate the declared measure on this exposure and outcome."""
    from turbotab.core.decisions import _ctx
    from turbotab.core.estimand import MEASURE_WORDS, current_estimand

    if decision.method == "none":
        return
    state = _state(ctx)
    spec = current_estimand(state) if state is not None else None
    task = _get(state, "task") or _ctx(ctx, "task")
    measure = _get(spec, "measure")
    if task == "binary" and measure in ("odds_ratio",):
        raise _refusal(
            "measure_conditional",
            f"DML and TMLE estimate a marginal effect, and the declared measure is the "
            f"{MEASURE_WORDS['odds_ratio']} (non-collapsible): another estimand. The marginal risk "
            f"difference is the causal lane's measure for a yes/no outcome.",
            [{"label": "Declare the marginal risk difference",
              "decision": _estimand_with(spec, measure="risk_difference")},
             {"label": "Keep the primary model only", "decision": _with(decision, method="none")}])
    if decision.method == "pds_lasso" and task == "binary":
        raise _refusal(
            "pds_needs_numeric_outcome",
            "The post-double-selection lasso is a linear model of the outcome; for a yes/no "
            "outcome TMLE or the interactive DML model estimate the risk difference.",
            [{"label": METHOD_LABELS["tmle"], "decision": _with(decision, method="tmle")},
             {"label": METHOD_LABELS["dml_irm"], "decision": _with(decision, method="dml_irm")}])
    design = _design_artifact(ctx)
    kind = _get(design, "exposure_kind") if design is not None else None
    if decision.method in ("dml_irm", "tmle") and kind == "continuous":
        raise _refusal(
            "needs_binary_exposure",
            f"{METHOD_LABELS[decision.method]} compares two levels of the exposure, and "
            f"{_tick(decision.exposure)} is numeric; the partially linear model estimates its "
            f"effect per unit.",
            [{"label": METHOD_LABELS["dml_plr"], "decision": _with(decision, method="dml_plr",
                                                                   population="all", trim=None)}])
    if decision.population == "exposed" and decision.method != "dml_irm":
        raise _refusal(
            "att_needs_irm",
            "The effect among the exposed is the interactive DML model's second score; the other "
            "methods here estimate the average effect over everyone.",
            [{"label": f"{METHOD_LABELS['dml_irm']}, among the exposed",
              "decision": _with(decision, method="dml_irm")},
             {"label": "The average effect over everyone",
              "decision": _with(decision, population="all")}])
    if decision.trim is not None and (kind == "continuous" or decision.method in ("dml_plr",
                                                                                  "pds_lasso")):
        raise _refusal(
            "trim_needs_propensity",
            "Trimming keeps the rows whose propensity to be exposed is neither near 0 nor near 1; "
            "it applies to a yes/no exposure's interactive model or TMLE.",
            [{"label": "Keep every row", "decision": _with(decision, trim=None)}])
    forms = _get(state, "exposure_forms") or {}
    form = _get(forms.get(decision.exposure), "form")
    if form in ("spline", "quintiles") and kind != "binary":
        from turbotab.core.decisions import SetExposureForm

        raise _refusal(
            "form_is_several_terms",
            f"{_tick(decision.exposure)} is declared as "
            f"{'a spline' if form == 'spline' else 'quintiles'}, "
            f"several terms, and the causal lane estimates one effect per unit of it.",
            [{"label": f"Declare {_tick(decision.exposure)}'s form linear",
              "decision": SetExposureForm(column=decision.exposure, form="linear")},
             {"label": "Keep the primary model only", "decision": _with(decision, method="none")}])


def _estimand_with(spec: Any, **change: Any) -> Any:
    from turbotab.core.decisions import SetEstimand

    if spec is None:
        return None
    return SetEstimand(**{**spec.model_dump(), **change})


def _assumptions_are_declared(decision: Any, ctx: Any) -> None:
    """The four assumptions are declared, each beside its diagnostic, before any estimate."""
    if decision.method == "none":
        return
    missing = [a for a in ASSUMPTIONS if a not in decision.assumptions]
    if not missing:
        return
    design = _design_artifact(ctx)
    card = {a["key"]: a for a in (design or {}).get("assumptions") or []}
    lines = []
    for a in missing:
        entry = card.get(a)
        if entry:
            lines.append(f"{entry['label']}: {entry['statement']} {entry['diagnostic']}")
        else:
            lines.append(f"{ASSUMPTION_WORDS[a][0].upper()}{ASSUMPTION_WORDS[a][1:]}.")
    raise _refusal(
        "assumptions_first",
        "The causal estimate rests on assumptions the data cannot check, so each is declared "
        "before any estimate is shown. " + " ".join(lines),
        [{"label": "Declare these assumptions and estimate",
          "decision": _with(decision, assumptions=list(ASSUMPTIONS))},
         {"label": "Keep the primary model only", "decision": _with(decision, method="none")}])


def card_positivity(design: Mapping[str, Any] | None, population: str) -> Mapping[str, Any] | None:
    """The causal card's positivity reading for the estimand asked: the effect among the exposed
    reads its own (a propensity above ``1 − b``: an exposed-like row with no comparable unexposed
    one), every other estimand the average effect's (a propensity outside ``[b, 1 − b]``)."""
    if design is None:
        return None
    return design.get("positivity_att" if population == "exposed" else "positivity")


def card_violation(ctx: Any, decision: Any) -> bool | None:
    """Whether the causal card reads positivity as practically violated for this decision's
    estimand; None when there is no card (or no reading) to say."""
    found = card_positivity(_design_artifact(ctx), str(decision.population))
    return None if found is None else bool(found.get("violated"))


def _positivity_holds(decision: Any, ctx: Any) -> None:
    """Block and record: a practical positivity violation is refused until the rows are trimmed to
    the overlap population or every row is kept on the record (MODELING_SEQUENCE §4). The reading
    is the estimand's own: the effect among the exposed is not refused for unexposed rows with a
    propensity near 0, which it never extrapolates to."""
    if decision.method == "none" or decision.acknowledged or decision.trim is not None:
        return
    design = _design_artifact(ctx)
    positivity = card_positivity(design, str(decision.population)) or {}
    if not positivity.get("violated"):
        return
    raise _refusal("positivity", str(positivity.get("reason")), positivity_exits(
        decision, binary=(design or {}).get("exposure_kind") == "binary"))


def positivity_exits(decision: Any, *, binary: bool) -> list[dict[str, Any]]:
    """The ways past a positivity violation: trimming (a yes/no exposure), or the record. Trimming
    estimates the average effect in the overlap population; asked for the effect among the
    exposed, its label says it changes that estimand too."""
    exits: list[dict[str, Any]] = []
    if binary:
        method = decision.method if decision.method in ("dml_irm", "tmle") else "tmle"
        label = f"Trim to propensities in [{TRIM_AT:g}, {1 - TRIM_AT:g}] (the overlap population)"
        if decision.population == "exposed":
            label = (f"Trim to propensities in [{TRIM_AT:g}, {1 - TRIM_AT:g}] and estimate the "
                     f"average effect there (the overlap population), not the effect among the "
                     f"exposed")
        exits.append({"label": label,
                      "decision": _with(decision, method=method, trim=TRIM_AT, population="all")})
    exits.append({"label": "Keep every row; record that positivity is practically violated",
                  "decision": _with(decision, acknowledged=True)})
    return exits


def _survey_weights_enter(decision: Any, ctx: Any) -> None:
    """Under the surveyed-population answer the weights enter every nuisance fit and the
    estimating equation; the post-double-selection lasso cannot take them, so it is blocked and
    recorded with the weighted partially linear model and the sample-only attestation as exits."""
    state = _state(ctx)
    survey = _get(state, "survey")
    if decision.method != "pds_lasso" or decision.sample_only:
        return
    if survey is None or _get(survey, "estimand") != "population":
        return
    raise _refusal(
        "survey_pds",
        "The estimate is for the surveyed population, and the post-double-selection lasso's "
        "plug-in penalty is derived for independent, unweighted rows: its selection cannot carry "
        "the survey weights. The partially linear DML model takes them in every fit.",
        [{"label": f"{METHOD_LABELS['dml_plr']}, survey-weighted",
          "decision": _with(decision, method="dml_plr")},
         {"label": "Post-double selection for these participants only, recorded as unweighted",
          "decision": _with(decision, sample_only=True)}])


def _complete_rows(decision: Any, ctx: Any) -> None:
    """Block and record: under a fill or multiple imputation the causal lane would need every
    estimate pooled over imputations, which it is not in v2; it runs on the complete rows only
    once that is recorded."""
    if decision.method == "none" or decision.complete_rows:
        return
    design = _design_artifact(ctx)
    missing = (design or {}).get("missing") or {}
    if not missing.get("blocked"):
        return
    raise _refusal("incomplete_rows", str(missing.get("reason")),
                   [{"label": "Estimate on the complete rows, its assumption stated",
                     "decision": _with(decision, complete_rows=True)},
                    {"label": "Keep the primary model only",
                     "decision": _with(decision, method="none")}])


def _learner_is_named(decision: Any, ctx: Any) -> Any:
    """The record names the learner the estimate will use: an unnamed one is the default for the
    table's size (``stages.causal.default_learner``), read from the causal card's row count."""
    if decision.method in ("none", "pds_lasso") or decision.learner is not None:
        return decision
    design = _design_artifact(ctx)
    n = _get(design, "n") if design is not None else None
    if not n:
        return decision
    from turbotab.core.stages.causal import default_learner

    return decision.model_copy(update={"learner": default_learner(int(n))})


def _register() -> None:
    from turbotab.core.decisions import register_completion, register_validator
    from turbotab.core.voice import register_sentence

    register_validator("set_causal", _causal_is_for_inference)
    register_validator("set_causal", _causal_follows_the_plan)
    register_validator("set_causal", _point_exposure_only)
    register_validator("set_causal", _causal_fits_the_estimand)
    register_validator("set_causal", _survey_weights_enter)
    register_validator("set_causal", _assumptions_are_declared)
    register_validator("set_causal", _complete_rows)
    register_validator("set_causal", _positivity_holds)
    register_completion("set_causal", _learner_is_named)
    register_sentence("set_causal")(_record_sentence)


_register()

__all__ = [
    "ASSUMPTIONS", "ASSUMPTION_WORDS", "CANDIDATES_PER", "CONTRACTS", "METHODS", "METHOD_LABELS",
    "RELATIONS", "ROBUSTNESS_NAMES", "STATED", "TRIM_AT", "assumption_card",
    "card_positivity", "card_violation", "causal_gate", "final_stage_robustness", "stated_reason",
    "current_causal", "many_candidates", "methods_sentence", "options", "plan_reason",
    "positivity_exits", "sensitivity_for", "sensitivity_sentence",
]

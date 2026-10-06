"""The domain packs' guesses for the adjustment card (MODELING_SEQUENCE §1 step 3; BLUEPRINT §14.2).

The adjustment set is asked through the modified disjunctive cause criterion, covariate by
covariate (``turbotab/core/estimand.py``). Asked plainly of thirty covariates that is thirty
questions, so the card leads with a guess wherever the packs have one, and covariates with the same
guess are answered as one block, one tap (BLUEPRINT §14.2: "Lead with the guess, confirm in one tap,
group by consequence"). A guess is a proposal the user confirms or changes, never a settled reading:
it is read from a column's name (BLUEPRINT §14: a name is never corroboration), so it only leads the
card, with its reason and its source visible beside it.

**The classes.** A covariate's name is read into one of six classes, most specific first:

* **medications** that treat a clinical measurement: lipid-lowering (statins), antihypertensives,
  glucose-lowering (metformin, insulin as a treatment, the NHANES ``BPQ``/``DIQ`` items);
* **clinical measurements**, in six groups: glycemic (glucose, HbA1c, insulin), lipids (HDL, LDL,
  triglycerides, total cholesterol), inflammation (CRP), liver (ALT, AST, GGT), kidney (creatinine,
  eGFR, urine albumin) and blood pressure;
* **body size and composition** (BMI, weight, waist, body fat);
* **demographics** (age, sex, race and ethnicity, education, income);
* **lifestyle** (smoking, alcohol, physical activity, coffee, tea and caffeine);
* **other dietary components** (any nutrient or energy source the recognizer reads).

**The guess depends on the exposure–outcome pairing** (NUTRITION_PACK §08, "The adjustment card's
guesses"). The exposure's and the outcome's own classes decide it:

* demographics and lifestyle: a pre-exposure common cause, adjusted as a confounder
  (``causes_exposure`` yes, ``causes_outcome`` yes, ``after_exposure`` no), the field's Models 1–2;
* other dietary components: a possible common cause through the diet's shared causes, adjusted
  (unknown, unknown, no), the field's Model 4, never "not relevant";
* body size, clinical measurements and medications, measured at the same visit as the exposure
  (cross-sectional) or at baseline with it: a possible mediator, or measured after the exposure
  (unknown, yes, unknown), so the estimate is declared without it and, beside, with it (MODELING_
  SEQUENCE §1 step 3: "Unknown timing (cross-sectional BMI) → a declared with-and-without pair").
  Schisterman, Cole & Platt (2009, *Epidemiology* 20:488): "We define overadjustment bias as
  control for an intermediate variable (or a descending proxy for an intermediate variable) on a
  causal path from exposure to outcome." A medication that treats the outcome carries Tobin et al.
  (2005, *Stat Med* 24:2911–2935), who found that "fitting a conventional regression model with
  treatment as a binary covariate" is "fundamentally flawed" for a treated quantitative trait, as
  is ignoring the treatment; their remedies (a constant added to treated values, censored normal
  regression) are not built, so the card says so;
* a measurement of the outcome's own group taken at the same visit as the outcome (HbA1c beside
  fasting glucose, HDL beside LDL, weight beside BMI): another measure of the state the outcome
  measures, so the exposure could have changed it as it could the outcome, and adjusting for it
  would remove part of the effect. It is a descending proxy for an intermediate in Schisterman's
  sense, not a cause of the exposure, so the criterion leaves it out (no, no, yes). It is not
  called a consequence of the outcome: HDL is no consequence of LDL;
* the same measurement at baseline beside an outcome measured over follow-up (HbA1c beside
  incident diabetes, blood pressure beside incident hypertension, LDL beside LDL at twelve
  months): measured before the outcome, it is no consequence of it and can predict it, and it may
  have changed the diet or been changed by it, so it takes its class's own guess, the declared
  with-and-without pair (unknown, yes, unknown). Beside a change since baseline, adjusting for the
  baseline the exposure may have changed can itself bias the analysis (Glymour et al. 2005, *Am J
  Epidemiol* 162:267–278), so neither model alone is the default;
* a measurement of the exposure's own group (total cholesterol beside an LDL exposure), or a
  habit of the exposure's own (caffeine beside a coffee exposure): another measure of the exposure;
  the packs have no guess, and it is asked.

**An outcome over follow-up** is a time-to-event outcome, a yes/no outcome whose name says it is
incident (``incident_diabetes``, ``t2d_onset``, ``new_htn``), or an outcome whose name says it was
measured at follow-up, as a change since baseline, or at a time since it (``ldl_followup``,
``hba1c_change``, ``delta_bmi``, ``weight_12m``). The name only leads a guess the user confirms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

VANDERWEELE = "VanderWeele 2019, Eur J Epidemiol 34:211–219"
SCHISTERMAN = "Schisterman, Cole & Platt 2009, Epidemiology 20:488"
TOBIN = "Tobin et al. 2005, Stat Med 24:2911–2935"
GLYMOUR = "Glymour et al. 2005, Am J Epidemiol 162:267–278"
PACK08 = "NUTRITION_PACK §08"
SCHISTERMAN_QUOTE = ("We define overadjustment bias as control for an intermediate variable (or "
                     "a descending proxy for an intermediate variable) on a causal path from "
                     "exposure to outcome.")
TOBIN_QUOTE = ("(ii) fitting a conventional regression model with treatment as a binary covariate")

# ── the answers each guess gives (the modified disjunctive cause criterion's three questions) ──

PRE_EXPOSURE = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
DIETARY = {"causes_exposure": "unknown", "causes_outcome": "unknown", "after_exposure": "no"}
TIMING_UNKNOWN = {"causes_exposure": "unknown", "causes_outcome": "yes",
                  "after_exposure": "unknown"}
OUTCOME_MEASURE = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "yes"}

# One guess per class (the answers, the class's label and the reason the card leads with). The
# adjustment card reads these at call time, so a test may set one class's answers.
GUESSES: dict[str, dict[str, Any]] = {
    "demographic": {
        "label": "Demographics",
        "answers": dict(PRE_EXPOSURE),
        "reason": "set before the diet was measured, and the field's Models 1 and 2 adjust for "
                  "them as confounders",
        "source": f"{PACK08} (Models 1 and 2); {VANDERWEELE}",
    },
    "lifestyle": {
        "label": "Lifestyle",
        "answers": dict(PRE_EXPOSURE),
        "reason": "habits that shape both what people eat and the outcome; the field's Model 2 "
                  "adjusts for them as confounders",
        "source": f"{PACK08} (Model 2: demographics and lifestyle); {VANDERWEELE}",
    },
    "dietary": {
        "label": "Other dietary components",
        "answers": dict(DIETARY),
        "reason": "they share the diet's common causes with the exposure, so they default to "
                  "confounders, never to not relevant (the field's Model 4)",
        "source": f"{PACK08} (Model 4: mutually adjusted nutrients)",
    },
    "body": {
        "label": "Body size and composition",
        "answers": dict(TIMING_UNKNOWN),
        "reason": "the diet may have changed them, so the field's Model 3 adds them beside the "
                  "primary; adjusting for a mediator without saying so is the pack's anti-pattern",
        "source": f"{PACK08} (Model 3; the anti-pattern \"adjusting for BMI when BMI is a "
                  f"mediator without saying so\")",
    },
    "clinical": {
        "label": "Clinical measurements",
        "answers": dict(TIMING_UNKNOWN),
        "reason": "possible mediators, or measured after the exposure: the exposure may have "
                  "changed them, so the estimate is declared without them and, beside, with them",
        "source": f"{SCHISTERMAN}; MODELING_SEQUENCE §1 step 3 (unknown timing: a declared "
                  f"with-and-without pair)",
    },
    "medication": {
        "label": "Medications",
        "answers": dict(TIMING_UNKNOWN),
        "reason": "a treatment the exposure may have led to, measured at the same time: a possible "
                  "mediator, so the estimate is declared without it and, beside, with it",
        "source": f"{SCHISTERMAN}; MODELING_SEQUENCE §1 step 3 (unknown timing: a declared "
                  f"with-and-without pair)",
    },
    "outcome_measure": {
        "label": "Other measures of the outcome",
        "answers": dict(OUTCOME_MEASURE),
        "reason": "another measure of the state the outcome measures, taken at the same visit: the "
                  "exposure could have changed it as it could the outcome, and adjusting for it "
                  "would remove part of the effect, so the criterion leaves it out",
        "source": f"{SCHISTERMAN} (\"a descending proxy for an intermediate variable\"); "
                  f"{VANDERWEELE}",
    },
}
# The order the card lists the classes in, within a block and between blocks.
CLASS_ORDER = ("demographic", "lifestyle", "dietary", "body", "clinical", "medication",
               "outcome_measure")

# ── reading a name into its class ─────────────────────────────────────────────

_DEMOGRAPHIC = {"age", "sex", "gender", "race", "ethnicity", "ethnic", "education", "educ",
                "income", "poverty", "pir", "occupation", "marital", "ses"}
_NHANES_DEMOGRAPHIC = {"RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "DMDEDUC3",
                       "INDFMPIR", "INDHHIN2", "DMDMARTL"}
_BODY = {"bmi", "weight", "height", "waist", "hip", "whr", "adiposity", "bodyfat", "ffm", "lean",
         "skinfold", "dxa", "bia"}
_NHANES_BODY = {"BMXBMI", "BMXWT", "BMXHT", "BMXWAIST", "BMXHIP", "DXDTOFAT", "DXDTOPF"}

# Lifestyle: (words any one of which reads it, word pairs that read it, NHANES variable names).
_LIFESTYLE: dict[str, tuple[set[str], tuple[frozenset[str], ...], set[str]]] = {
    "smoking": ({"smoking", "smoker", "smokers", "smoke", "smokes", "cigarettes", "cigarette",
                 "cigs", "packyears", "tobacco", "cotinine"},
                (frozenset({"pack", "years"}),),
                {"SMQ020", "SMQ040", "SMD650", "LBXCOT"}),
    "alcohol": ({"alcohol", "drinks", "drinking", "drinker", "etoh", "beer", "wine", "liquor"},
                (), {"ALQ101", "ALQ110", "ALQ120Q", "ALQ121", "ALQ130", "DR1TALCO", "DR2TALCO"}),
    "physical activity": ({"mvpa", "exercise", "exercising", "steps", "sedentary", "pa", "mets"},
                          (frozenset({"physical", "activity"}), frozenset({"met", "min"}),
                           frozenset({"met", "minutes"}), frozenset({"met", "hours"})),
                          {"PAD615", "PAD630", "PAD660", "PAD675", "PAD680", "PAQ605", "PAQ620",
                           "PAQ635", "PAQ650", "PAQ665"}),
    "coffee, tea or caffeine": ({"coffee", "tea", "caffeine", "caff", "espresso"}, (),
                                {"DR1TCAFF", "DR2TCAFF"}),
}

# Clinical measurements by group: (words, word pairs, NHANES variable names).
_CLINICAL: dict[str, tuple[set[str], tuple[frozenset[str], ...], set[str]]] = {
    "glycemic": ({"glucose", "glu", "fpg", "fbg", "hba1c", "a1c", "ghb", "glycohemoglobin",
                  "insulin", "homa", "homair", "fructosamine"}, (),
                 {"LBXGLU", "LBDGLUSI", "LBXGH", "LBXIN", "LBDINSI", "LBXSGL", "LBDSGLSI"}),
    "lipids": ({"hdl", "ldl", "vldl", "triglycerides", "triglyceride", "trig", "trigs", "tg",
                "nonhdl", "apob", "apoa1", "lpa"},
               (frozenset({"total", "chol"}), frozenset({"total", "cholesterol"}),
                frozenset({"serum", "chol"}), frozenset({"serum", "cholesterol"}),
                frozenset({"plasma", "cholesterol"})),
               {"LBDHDD", "LBDHDDSI", "LBXTR", "LBDTRSI", "LBDLDL", "LBDLDLSI", "LBXTC",
                "LBDTCSI", "LBXAPB"}),
    "inflammation": ({"crp", "hscrp", "il6", "fibrinogen"}, (), {"LBXCRP", "LBXHSCRP"}),
    "liver": ({"alt", "ast", "ggt", "alp", "bilirubin", "sgpt", "sgot"}, (),
              {"LBXSATSI", "LBXSASSI", "LBXSGTSI", "LBXSAPSI", "LBXSTB", "LBDSTBSI"}),
    "kidney": ({"creatinine", "creat", "egfr", "gfr", "bun", "urea", "uacr", "acr", "albuminuria",
                "cystatin", "microalbumin"}, (frozenset({"uric", "acid"}),),
               {"LBXSCR", "LBDSCRSI", "LBXSBU", "URDACT", "URXUMA", "URXUCR", "LBXSUA"}),
    "blood pressure": ({"sbp", "dbp", "systolic", "diastolic"},
                       (frozenset({"bp", "sys"}), frozenset({"bp", "di"}), frozenset({"bp", "dia"}),
                        frozenset({"blood", "pressure"})),
                       {"BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXOSY1",
                        "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"}),
}
# An outcome named as a disease belongs to the group that defines it.
_DISEASE_GROUP = {
    "glycemic": {"diabetes", "diabetic", "dm", "t2d", "t2dm", "prediabetes"},
    "blood pressure": {"hypertension", "htn", "hypertensive"},
    "lipids": {"dyslipidemia", "hyperlipidemia", "hypercholesterolemia"},
    "kidney": {"ckd"},
    "liver": {"nafld", "masld", "steatosis"},
}
_OBESITY = {"obesity", "obese", "overweight"}
# A yes/no outcome named as new disease over follow-up (``incident_diabetes``, ``t2d_onset``).
_INCIDENT = {"incident", "incidence", "onset", "new", "developed", "conversion", "converted"}
# An outcome of any kind named as measured over follow-up, or as a change since baseline
# (``ldl_followup``, ``weight_fu``, ``hba1c_change``, ``delta_bmi``, ``sbp_endline``), or with a
# time since baseline (``hba1c_12m``, ``weight_5y``, ``ldl_24wk``). ``v1``, ``t0`` and ``w2`` name
# an occasion that may be the baseline itself, so they read as nothing.
_FOLLOW_UP = {"followup", "fu", "change", "changes", "delta", "endline", "final"}
_SINCE = re.compile(r"^\d+(?:m|mo|mos|mth|mths|month|months|y|yr|yrs|year|years|w|wk|wks|week|"
                    r"weeks)$")

_MED_WORDS = {"med", "meds", "medication", "medications", "medicine", "medicines", "drug", "drugs",
              "pill", "pills", "rx", "treated", "treatment", "therapy", "use", "user", "users",
              "taking", "tx", "lowering", "agent", "agents"}
# Medications by the group of the measurement they treat: (drug words, the words that name what
# a medication word treats, NHANES variable names).
_MEDICATION: dict[str, tuple[set[str], set[str], set[str]]] = {
    "lipids": ({"statin", "statins", "atorvastatin", "simvastatin", "rosuvastatin",
                "pravastatin", "lovastatin", "fluvastatin", "pitavastatin", "ezetimibe",
                "fibrate", "fibrates"},
               {"chol", "cholesterol", "lipid", "lipids", "hyperlipidemia", "dyslipidemia", "ldl"},
               {"BPQ090D", "BPQ100D"}),
    "blood pressure": ({"antihypertensive", "antihypertensives", "acei", "arb", "arbs", "diuretic",
                        "diuretics", "betablocker", "betablockers", "amlodipine", "lisinopril",
                        "losartan", "hydrochlorothiazide"},
                       {"hbp", "bp", "hypertension", "htn", "pressure"},
                       {"BPQ040A", "BPQ050A"}),
    "glycemic": ({"metformin", "antidiabetic", "antidiabetics", "hypoglycemic", "hypoglycemics",
                  "sulfonylurea", "sulfonylureas", "glp1", "sglt2", "dpp4"},
                 {"diabetes", "diabetic", "dm", "glucose", "insulin", "t2d", "glycemic"},
                 {"DIQ050", "DIQ070"}),
}
_TREATS = {"lipids": "lipid-lowering medication (a statin)",
           "blood pressure": "antihypertensive medication",
           "glycemic": "glucose-lowering medication"}


@dataclass(frozen=True)
class Reading:
    """A name read into a covariate class (and, for a measurement or a medication, its group)."""

    cls: str
    group: str | None = None
    what: str | None = None  # the lifestyle habit, or the medication, in words


def _matches(words: set[str], upper: str,
             spec: tuple[set[str], tuple[frozenset[str], ...], set[str]]) -> bool:
    single, pairs, nhanes = spec
    return bool(words & single) or any(p <= words for p in pairs) or upper in nhanes


def read_name(name: Any) -> Reading | None:
    """The class a covariate's name reads as (most specific first), or None."""
    from turbotab.core.estimand import energy_bearing
    from turbotab.core.recognizers import is_nutrient, tokens

    text = str(name)
    upper = text.upper()
    words = set(tokens(text))
    for group, (drugs, treats, nhanes) in _MEDICATION.items():
        if words & drugs or upper in nhanes or (words & _MED_WORDS and words & treats):
            return Reading("medication", group, _TREATS[group])
    for group, spec in _CLINICAL.items():
        if _matches(words, upper, spec):
            return Reading("clinical", group)
    if upper in _NHANES_BODY or words & _BODY or {"fat", "mass"} <= words \
            or {"body", "fat"} <= words:
        return Reading("body", "body")
    if upper in _NHANES_DEMOGRAPHIC or words & _DEMOGRAPHIC:
        return Reading("demographic")
    for habit, spec in _LIFESTYLE.items():
        if _matches(words, upper, spec):
            return Reading("lifestyle", None, habit)
    if energy_bearing(text) or is_nutrient(text):
        return Reading("dietary")
    return None


def _measured_group(name: Any) -> tuple[str | None, str | None]:
    """The class and group of an exposure or an outcome: a clinical measurement (or a disease
    named by its group), body size, or another class."""
    from turbotab.core.recognizers import tokens

    found = read_name(name)
    if found is not None and found.cls in ("clinical", "body"):
        return found.cls, found.group
    words = set(tokens(str(name)))
    for group, names in _DISEASE_GROUP.items():
        if words & names:
            return "clinical", group
    if words & _OBESITY:
        return "body", "body"
    return (found.cls, found.group) if found is not None else (None, None)


@dataclass(frozen=True)
class Pairing:
    """The exposure–outcome pairing a guess is made under."""

    exposure: str | None
    outcome: str | None
    exposure_cls: str | None
    exposure_group: str | None
    outcome_cls: str | None
    outcome_group: str | None
    followed: bool  # an outcome over follow-up: the covariates were measured at baseline
    exposure_habit: str | None = None  # a lifestyle exposure's habit (smoking, alcohol, …)
    task: str | None = None

    @property
    def when(self) -> str:
        return ("measured at baseline with the exposure" if self.followed else
                "measured at the same visit as the exposure")


def pairing_of(state: Any) -> Pairing:
    """The declared exposure and the outcome, each read into its class."""
    from turbotab.core.estimand import _get, current_estimand

    spec = current_estimand(state)
    exposure = _get(spec, "exposure") if spec is not None else None
    outcome = _get(state, "target")
    e_cls, e_group = _measured_group(exposure) if exposure else (None, None)
    if spec is not None and _get(spec, "family"):
        e_cls, e_group = "dietary", None  # an exposure family: nutrients or features in turn
    o_cls, o_group = _measured_group(outcome) if outcome else (None, None)
    habit = read_name(exposure) if exposure and e_cls == "lifestyle" else None
    task = _get(state, "task")
    return Pairing(exposure, outcome, e_cls, e_group, o_cls, o_group, followed(outcome, task),
                   habit.what if habit is not None else None, task)


def followed(outcome: Any, task: Any) -> bool:
    """The outcome was measured over follow-up, so the covariates were measured at baseline, before
    it: a time to event, a yes/no outcome whose name says it is incident, or an outcome whose name
    says it was measured at follow-up, as a change since baseline, or at a time since it. A name
    only leads the guess the user confirms (BLUEPRINT §14)."""
    from turbotab.core.recognizers import tokens

    if task == "time_to_event":
        return True
    if outcome is None:
        return False
    words = tokens(str(outcome))
    if task == "binary" and set(words) & _INCIDENT:
        return True
    return (bool(set(words) & _FOLLOW_UP) or {"follow", "up"} <= set(words)
            or any(_SINCE.match(w) for w in words))


@dataclass(frozen=True)
class Guess:
    """One covariate's guess: the class (``GUESSES`` key) whose answers it takes, what it reads as,
    the reason under this pairing, and the source."""

    column: str
    key: str
    answers: Mapping[str, str]
    what: str
    reason: str
    source: str
    group: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"column": self.column, "class": self.key, "label": GUESSES[self.key]["label"],
                "what": self.what, "group": self.group, "answers": dict(self.answers),
                "reason": self.reason, "source": self.source}


def _tick(value: Any) -> str:
    return f"`{value}`"


def guess(column: str, state: Any, pairing: Pairing | None = None) -> Guess | None:
    """The packs' guess for one covariate under the declared exposure–outcome pairing, or None
    where they have none (the card then asks it plainly)."""
    found = read_name(column)
    if found is None:
        return None
    p = pairing if pairing is not None else pairing_of(state)

    def made(key: str, what: str, reason: str | None = None, source: str | None = None) -> Guess:
        g = GUESSES[key]
        return Guess(str(column), key, dict(g["answers"]), what, reason or g["reason"],
                     source or g["source"], found.group)

    if found.cls == "demographic":
        return made("demographic", "a demographic characteristic")
    if found.cls == "lifestyle":
        if p.exposure_habit == found.what:
            return None  # another measure of the exposure's own habit: asked
        return made("lifestyle", f"a lifestyle habit ({found.what})")
    if found.cls == "dietary":
        return made("dietary", "another dietary component")
    same_outcome = p.outcome_cls == found.cls and p.outcome_group == found.group
    same_exposure = p.exposure_cls == found.cls and p.exposure_group == found.group
    if found.cls in ("body", "clinical"):
        what = ("body size or composition" if found.cls == "body"
                else f"a clinical measurement ({found.group})")
        kind = "body size" if found.cls == "body" else found.group
        if same_outcome and p.followed:
            # A baseline level of the outcome's own kind beside an outcome measured over follow-up
            # (HbA1c beside incident diabetes, LDL beside LDL at twelve months): before the
            # outcome, so no consequence of it, and it can predict it. The pack's row for a
            # measurement taken with the exposure applies.
            event = p.task in ("time_to_event", "binary")
            later = "the event" if event else "the outcome's measurement"
            change = (f"{GLYMOUR} (adjusting for a baseline the exposure may have changed can bias "
                      f"an analysis of change); ")
            return made(found.cls, f"{what}, a baseline level of the outcome's own kind",
                        f"a level of the outcome {_tick(p.outcome)}'s own kind ({kind}) {p.when}, "
                        f"so before {later}: no consequence of the "
                        f"{'event' if event else 'outcome'}, and it can predict it; it may have "
                        f"changed the diet (a confounder) or been changed by it (a mediator), so "
                        f"the estimate is declared without it and, beside, with it",
                        f"{PACK08} (a measurement taken with the exposure: unknown, yes, unknown); "
                        f"{SCHISTERMAN}; {'' if event else change}MODELING_SEQUENCE §1 step 3 "
                        f"(unknown timing: a declared with-and-without pair)")
        if same_outcome:
            return made("outcome_measure", f"{what}, the outcome's own kind",
                        f"another measure of the outcome {_tick(p.outcome)}'s own kind ({kind}), "
                        f"{p.when}: it measures the state the outcome measures, so the exposure "
                        f"could have changed it as it could the outcome, and adjusting for it "
                        f"would remove part of the effect; the criterion leaves it out")
        if same_exposure:
            return None  # another measure of the exposure: no guess, it is asked
        if found.cls == "body":
            return made("body", what)
        return made("clinical", what,
                    f"{p.when}, the exposure may have changed it: a possible mediator, or measured "
                    f"after the exposure, so the estimate is declared without it and, beside, with "
                    f"it")
    # a medication
    what = f"{found.what}"
    if p.outcome_cls == "clinical" and p.outcome_group == found.group:
        return made("medication", what,
                    f"it treats the outcome {_tick(p.outcome)}: those treated have lowered values, "
                    f"and adjusting for the treatment as a covariate does not undo that (Tobin et "
                    f"al. 2005 call it fundamentally flawed, as is ignoring it); the estimate is "
                    f"declared without it and, beside, with it, and the correction they recommend "
                    f"is not built",
                    f"{TOBIN}; {SCHISTERMAN}; MODELING_SEQUENCE §1 step 3")
    if p.exposure_cls == "clinical" and p.exposure_group == found.group:
        return made("medication", what,
                    f"it lowers the exposure {_tick(p.exposure)} and follows its earlier level: a "
                    f"possible mediator, {p.when}, so the estimate is declared without it and, "
                    f"beside, with it")
    return made("medication", what,
                f"a treatment the exposure may have led to, {p.when}: a possible mediator, so the "
                f"estimate is declared without it and, beside, with it")


def outcome_measures(answers: Mapping[str, Any], state: Any) -> list[str]:
    """The covariates whose answers take the packs' guess "another measure of the outcome's own
    kind" under the state's pairing (the three answers equal to it, the guess the card showed), so
    the record says what the card said rather than the criterion's generic "consequence of the
    exposure". Each is the user's answer; the name only says which guess it agreed with."""
    pairing = pairing_of(state)
    out = []
    for column, a in answers.items():
        given = {k: (a.get(k) if isinstance(a, Mapping) else getattr(a, k, None))
                 for k in OUTCOME_MEASURE}
        if given != OUTCOME_MEASURE:
            continue
        found = guess(column, state, pairing)
        if found is not None and found.key == "outcome_measure":
            out.append(str(column))
    return out


__all__ = ["CLASS_ORDER", "DIETARY", "GLYMOUR", "GUESSES", "Guess", "OUTCOME_MEASURE",
           "PRE_EXPOSURE", "Pairing", "Reading", "SCHISTERMAN", "SCHISTERMAN_QUOTE", "TIMING_UNKNOWN", "TOBIN",
           "TOBIN_QUOTE", "followed", "guess", "outcome_measures", "pairing_of", "read_name"]

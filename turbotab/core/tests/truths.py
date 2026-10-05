"""A fixture's declared truth, and how a test driver answers the server's questions from it.

BLUEPRINT §14.3: "Every confirmation is honored … Test drivers answer from the fixture's truth,
never with a constant." A driver that answered every code-or-amount question "amount" hid that the
fit ignored the user's "codes" (the fourth gate). So each fixture declares what its readings really
are (:class:`Truth`), and a driver answers a refusal that only asks for readings
(:func:`answer_refusal`) from that declaration alone: every reading listed, in one block
confirmation, or the total-energy unit and days recorded as declared. A reading the fixture declares
nothing for fails the test, named, rather than being answered by a guess.
"""
from __future__ import annotations

from typing import Any, Callable

ASKING = ("reading_unsettled", "role_unconfirmed", "energy_unit_unconfirmed")


class Truth(dict):
    """``{"<kind>:<column>": value}`` for each reading the fixture's author knows:
    ``code_or_count:smoking`` → ``code``; ``cluster:PID`` → ``no``; ``unit:weight`` → ``lb``;
    ``day_count:kcal`` → ``1``; ``sex_coding:sex`` → ``female=2,male=1``; ``role:hhid`` →
    ``cluster``; ``detection_limit:crp`` → ``half_limit`` (the below-detection repair's option)."""

    def __init__(self, readings: dict[str, Any] | None = None, *, fixture: str = "this fixture"):
        super().__init__({k: str(v) for k, v in (readings or {}).items()})
        self.fixture = fixture

    def answer(self, reading: str, column: str) -> str:
        key = f"{reading}:{column}"
        if key not in self:
            raise AssertionError(f"{self.fixture} declares no truth for the {key} reading the "
                                 f"server asked about; declare it (Truth) rather than guess it")
        return str(self[key])


def asked(exits: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The readings a refusal's exits ask about, once each, in the ask's order."""
    out: list[tuple[str, str]] = []
    for item in exits:
        d = item.get("decision") or {}
        pairs = ([(i["reading"], i["column"]) for i in d.get("items") or []]
                 if d.get("kind") == "confirm_readings" else
                 [(d["reading"], d["column"])] if d.get("kind") == "confirm_reading" else [])
        out += [p for p in pairs if p not in out]
    return out


def answers(error: dict[str, Any], truth: Truth) -> list[dict[str, Any]]:
    """The decisions that answer a refusal's questions from ``truth``: one ``confirm_readings``
    listing every reading asked, each with its declared value; for an energy column whose unit or
    days are asked (``set_column_unit`` exits), the unit and days declared; for a column whose values
    below a detection limit are asked (``apply_repair`` exits of a ``below_detection__<column>``
    finding), the offered repair the truth names. Empty when the refusal asks for no reading."""
    exits = error.get("exits") or []
    out: list[dict[str, Any]] = []
    readings = asked(exits)
    if readings:
        out.append({"kind": "confirm_readings", "items": [
            {"reading": k, "column": c, "value": truth.answer(k, c)} for k, c in readings]})
    units = []
    for item in exits:
        d = item.get("decision") or {}
        if d.get("kind") == "set_column_unit" and d["column"] not in units:
            units.append(d["column"])
    for column in units:
        out.append({"kind": "set_column_unit", "column": column,
                    "unit": truth.answer("unit", column),
                    "days": int(truth.answer("day_count", column))})
    limits: dict[str, list[dict[str, Any]]] = {}
    for item in exits:
        d = item.get("decision") or {}
        if d.get("kind") == "apply_repair" and str(d.get("finding_id", "")).startswith(
                "below_detection__"):
            limits.setdefault(str(d["finding_id"]).split("__", 1)[1], []).append(d)
    for column, offered in limits.items():
        chosen = truth.answer("detection_limit", column)
        match = [d for d in offered if d.get("option") == chosen]
        if not match:
            raise AssertionError(f"{truth.fixture} declares detection_limit:{column} = {chosen}, "
                                 f"which the refusal does not offer")
        out.append(match[0])
    return out


def answer_refusal(post: Callable[[dict[str, Any]], Any], response: Any, truth: Truth,
                   retry: Callable[[], Any], rounds: int = 4) -> Any:
    """While ``response`` is a refusal that only asks for readings, post the answers ``truth``
    declares (:func:`answers`) and ``retry``; returns the last response."""
    for _ in range(rounds):
        if response.status_code != 409:
            return response
        error = response.json().get("error") or {}
        if error.get("code") not in ASKING:
            return response
        decisions = answers(error, truth)
        if not decisions:
            return response
        for decision in decisions:
            r = post(decision)
            assert r.status_code == 200, (decision, r.text[:600])
        response = retry()
    return response


__all__ = ["ASKING", "FIXTURE_TRUTHS", "Truth", "answer_refusal", "answers", "asked",
           "fixture_truth"]


# The sample fixtures' truths, each from the fixture's own data card (``turbotab/sample_data/*.md``)
# and generator (``make_fixtures.py``): what a user who knows the table answers when asked.
FIXTURE_TRUTHS: dict[str, dict[str, str]] = {
    # 300 people × 2 twenty-four-hour recalls (dietary_recalls.md): age in whole years, energy in
    # kcal for the one day each recall covers, sodium in whole mg; the recall's number names which
    # recall it was.
    "dietary_recalls.csv": {
        "code_or_count:age": "amount", "code_or_count:energy_kcal": "amount",
        "code_or_count:sodium_mg": "amount", "code_or_count:recall_number": "code",
        "unit:energy_kcal": "kcal", "day_count:energy_kcal": "1",
        "sex_coding:sex": "female=F,male=M",
    },
    # 200 people × 3 scheduled visits (clinical_longitudinal.md): the visit's number names the
    # visit; age, blood pressures, heart rate and glucose are whole-number measurements.
    "clinical_longitudinal.csv": {
        "code_or_count:visit": "code", "code_or_count:age": "amount", "code_or_count:sbp": "amount",
        "code_or_count:dbp": "amount", "code_or_count:heart_rate": "amount",
        "code_or_count:glucose": "amount", "code_or_count:progressed": "code",
        "unit:weight_kg": "kg", "unit:height_cm": "cm", "unit:age": "years",
    },
    "clinical_labs.csv": {
        "code_or_count:age": "amount", "code_or_count:sbp": "amount", "code_or_count:dbp": "amount",
        "code_or_count:bnp": "amount",
    },
    "longitudinal_visits.csv": {
        "code_or_count:visit": "code", "code_or_count:age": "amount",
        "code_or_count:bp_sys": "amount",
    },
    "clinic_visits.csv": {
        "code_or_count:age": "amount", "code_or_count:glucose": "amount",
        "code_or_count:bp_1": "amount", "code_or_count:bp_2": "amount", "code_or_count:bp_3": "amount",
    },
    # NHANES-shaped exports (nhanes_dietary.md, nhanes_kilojoules.md): DR1TKCAL is the first day's
    # recall, in kcal (in kJ in the kilojoule export); SDMVSTRA and SDMVPSU are design codes.
    "nhanes_dietary.csv": {
        "unit:DR1TKCAL": "kcal", "day_count:DR1TKCAL": "1",
        "code_or_count:SDMVSTRA": "code", "code_or_count:SDMVPSU": "code",
    },
    "nhanes_kilojoules.csv": {
        "unit:DR1TKCAL": "kj", "day_count:DR1TKCAL": "1",
        "code_or_count:SDMVSTRA": "code", "code_or_count:SDMVPSU": "code",
    },
    "nhanes_partial_design.csv": {"unit:DR1TKCAL": "kcal", "day_count:DR1TKCAL": "1"},
    # Hospital encounters (clinical_risk.md): the Charlson index and prior admissions are counts,
    # length of stay whole days, sodium whole mmol/L.
    "clinical_risk.csv": {
        "code_or_count:age": "amount", "code_or_count:charlson_index": "amount",
        "code_or_count:prior_admissions_12mo": "amount", "code_or_count:sodium_mmol_l": "amount",
        "code_or_count:length_of_stay_days": "amount", "sex_coding:sex": "female=F,male=M",
    },
    # The real NHANES export (untracked, ``stage_harness.NHANES``): energy the first day's recall in
    # kcal; age in whole years and HDL and triglycerides in whole mg/dL, amounts; the cycle's year a
    # code for the survey cycle.
    "_tt_tmp_nhanes.csv": {
        "unit:kcal": "kcal", "day_count:kcal": "1", "code_or_count:age": "amount",
        "code_or_count:hdl": "amount", "code_or_count:triglycerides": "amount",
        "code_or_count:kcal": "amount", "code_or_count:cycle_begin_year": "code", "unit:weight": "kg", "unit:height": "cm",
        "unit:age": "years",
        # WP17: each covariate's causal place for sugar → fasting glucose, as the author reads it
        # (causes sugar, causes glucose, changed by sugar or measured after it). Age, gender and
        # the survey cycle come before the diet and cause both; the other nutrients share the
        # diet's common causes; body size may itself follow the diet (a cross-sectional measure of
        # unknown timing); blood pressure, HDL, triglycerides and the medications are downstream of
        # the diet and cause glucose's level or its treatment: mediators.
        "exposure:glucose": "sugar", "contrast:sugar": "substitution",
        **{f"adjust:{c}": "yes,yes,no" for c in ("age", "gender", "cycle_begin_year")},
        **{f"adjust:{c}": "unknown,unknown,no" for c in (
            "protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly")},
        **{f"adjust:{c}": "unknown,yes,unknown" for c in ("weight", "height", "bmi", "waist")},
        **{f"adjust:{c}": "no,yes,yes" for c in (
            "bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp", "meds_chol")},
    },
    "binary_shapes.csv": {"code_or_count:age": "amount", "code_or_count:sbp": "amount"},
    "multiclass_stage.csv": {"code_or_count:age": "amount"},
    # One 40-item five-point Likert instrument (survey_instrument.md): each item's 1–5 is a response
    # code; age in whole years.
    **{name: {"code_or_count:age": "amount",
              **{f"code_or_count:item_{i:02d}": "code" for i in range(1, 41)}}
       for name in ("survey_instrument.csv", "survey_sentinels.csv")},
}


def fixture_truth(name: str) -> Truth:
    """The declared truth of a sample fixture, by file name (empty when none is declared)."""
    return Truth(FIXTURE_TRUTHS.get(name, {}), fixture=name)


# ── WP17: the adjustment set, answered from the fixture's causal truth ───────

ADJUSTMENT_FIELDS = ("causes_exposure", "causes_outcome", "after_exposure")


# A direct effect's own answers (MODELING_SEQUENCE §1 step 3 and §2), declared as ``field=value``.
DIRECT_FIELDS = ("confounds_mediator", "interacts")


def adjustment_truth(truth: Truth, column: str) -> dict[str, Any]:
    """A covariate's answers to the disjunctive cause criterion, as the fixture's author knows its
    causal place: ``adjust:<column>`` → ``"yes,yes,no"`` (causes the exposure, causes the outcome,
    changed by the exposure), optionally followed by ``,instrument`` or ``,proxy``, and, for a
    direct effect, ``,confounds_mediator=yes`` (a common cause of a mediator and the outcome) or
    ``,interacts=no`` (the exposure's effect does not differ with this mediator's level)."""
    parts = [p.strip() for p in truth.answer("adjust", column).split(",")]
    out: dict[str, Any] = dict(zip(ADJUSTMENT_FIELDS, parts[:3]))
    out["instrument"] = "instrument" in parts[3:]
    out["proxy"] = "proxy" in parts[3:]
    for part in parts[3:]:
        name, _, value = part.partition("=")
        if name in DIRECT_FIELDS and value:
            out[name] = value
    return out


def answer_adjustment(post: Callable[[dict[str, Any]], Any], card: dict[str, Any],
                      truth: Truth) -> list[dict[str, Any]]:
    """Answer the adjustment card as the fixture's author would (BLUEPRINT §14.2): a group whose
    every column's truth is the card's guess is confirmed with its one tap (the group's own
    decision); every other column is answered from its truth, columns with the same answers in one
    ``set_adjustment``. Returns the decisions posted, in order."""
    posted: list[dict[str, Any]] = []
    exposure = card["exposure"]
    for group in card["groups"]:
        guess = group.get("guess")
        truths = {c: adjustment_truth(truth, c) for c in group["columns"]}
        if guess is not None and all({k: t[k] for k in ADJUSTMENT_FIELDS} == {
                k: guess[k] for k in ADJUSTMENT_FIELDS} and not t["instrument"] and not t["proxy"]
                and not any(k in t for k in DIRECT_FIELDS) for t in truths.values()):
            decisions = [group["decision"]]
        else:
            by: dict[tuple[Any, ...], list[str]] = {}
            for c, t in truths.items():
                by.setdefault(tuple(sorted(t.items())), []).append(c)
            decisions = [{"kind": "set_adjustment", "exposure": exposure,
                          "answers": {c: dict(key) for c in columns}}
                         for key, columns in by.items()]
        for d in decisions:
            r = post(d)
            assert r.status_code == 200, (d, r.text[:600])
            posted.append(d)
    return posted

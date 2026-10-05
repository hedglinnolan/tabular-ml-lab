"""Capture the /lab/methods-map prototype's fixture from the real server (not part of the bundle).

    TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 TURBOTAB_HOME=$(mktemp -d) \
        venv/bin/python -m turbotab.server --port 8873          # repo root
    venv/bin/python turbotab/frontend/src/explore/methods-map/capture.py \
        --base http://127.0.0.1:8873 --raw /tmp/methods-map-raw    # repo root

The NHANES export is the acceptance tests' (``$TURBOTAB_NHANES_CSV``, else ``_tt_tmp_nhanes.csv``
in this checkout or the main one). Every reading the server asks about is answered from the
fixture's declared truth (``turbotab/core/tests/truths.py``), never a constant.

Four drives, each through the HTTP API:

* **main** (inference, sugar → fasting glucose): the opening answers, the readings (three single
  confirmations, then one block), the exclusions, the seal, the exposure and its estimand, the
  adjustment set from the truth, the energy model, the form, Model 1; every option of each slot is
  recorded once to take its sentence, then the canonical answer is recorded again (superseded
  answers before anything was seen fold out of the methods text). Previews are taken last, on the
  answered state, so their row counts are true of it. Then the plan lock and the methods text.
* **branches**: a second project on the same answers, then every combination the prototype can
  record (exclusions × form × energy model), each one's Table 2, secondary and sensitivity fits.
  Their energy sentences are the after-the-estimates ones the record shows for a post-lock edit.
* **prediction**: the same table under prediction, up to the seal question, with its previews.
* **in process**: sentence variants that are pure functions of the decision (the engine's own
  ``voice.sentence_for``, checked against the server on the captured ones), the criterion's
  derivations (``estimand.derive``) and each plan's SHA-256 (``plan_lock``).

Then ``trim`` writes ``fixture.json`` beside this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

from turbotab.core.tests.truths import adjustment_truth, fixture_truth  # noqa: E402

NUTRIENTS = ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
COVARIATES = ["age", "gender", "cycle_begin_year", *NUTRIENTS, "weight", "height", "bmi", "waist",
              "bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp", "meds_chol"]
FLAGS = ["imputed_weight", "imputed_height", "imputed_bmi", "imputed_waist", "imputed_bp_sys",
         "imputed_bp_di"]
# The author's roles (test_wp17_purpose_routes.NHANES_ROLES): what a user who knows the table answers.
ROLES = {"SEQN": "identifier", "sugar": "exposure", "kcal": "energy",
         **{c: "exposure" for c in NUTRIENTS},
         **{c: "covariate" for c in COVARIATES if c not in NUTRIENTS},
         **{c: "flag" for c in FLAGS}}
ENERGY_NUTRIENTS = ["sugar", *NUTRIENTS]
ENERGY_OK = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density"]
ENERGY_REFUSED = ["none", "all_components", "partition"]
EXCLUSION_KEYS = ["willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000",
                  "sex_neutral_500_3500"]
SENSITIVITY_KEYS = ["willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000"]
FORMS = {"spline": {"form": "spline", "knots": 4}, "linear": {"form": "linear"},
         "quintiles": {"form": "quintiles"}}
SINGLES = 3  # single confirmations before the block confirm unlocks (BLUEPRINT §11.4)


def nhanes() -> Path:
    from turbotab.core.tests.stage_harness import NHANES
    return Path(os.environ.get("TURBOTAB_NHANES_CSV") or NHANES)


class Api:
    def __init__(self, base: str):
        self.c = httpx.Client(base_url=base, timeout=600)

    def create(self, path: Path) -> str:
        r = self.c.post("/api/projects", json={"path": str(path)})
        r.raise_for_status()
        pid = r.json()["id"]
        self.stage(pid, "ingest")
        return pid

    def view(self, pid: str) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{pid}").json()

    def stage(self, pid: str, name: str, timeout: float = 600) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            st = self.view(pid)["stages"][name]
            if st["status"] == "fresh":
                return self.c.get(f"/api/projects/{pid}/stages/{name}").json()["artifact"]
            if st["status"] == "error":
                raise RuntimeError(f"{name}: {st}")
            if time.monotonic() > end:
                raise TimeoutError(f"{name} never fresh: {st}")
            time.sleep(0.1)

    def post(self, pid: str, body: dict[str, Any], timeout: float = 300) -> tuple[int, Any]:
        """Post; while the Router holds it behind a recomputing question ("not_yet"), post again."""
        end = time.monotonic() + timeout
        while True:
            r = self.c.post(f"/api/projects/{pid}/decisions", json=body)
            j = r.json()
            if r.status_code == 409 and j["error"]["code"] == "not_yet" and time.monotonic() < end:
                time.sleep(0.2)
                continue
            return r.status_code, j

    def record(self, pid: str, body: dict[str, Any]) -> dict[str, Any]:
        """Post and return the record it wrote (its sentence and flags); refuse loudly."""
        code, j = self.post(pid, body)
        if code != 200:
            raise RuntimeError(f"{body['kind']} refused: {json.dumps(j)[:900]}")
        rec = max(j["decisions"], key=lambda d: d["seq"])
        return {"sentence": rec["sentence"], "after_estimates": rec["after_estimates"],
                "seq": rec["seq"], "kind": rec["decision"]["kind"]}

    def refusal(self, pid: str, body: dict[str, Any]) -> dict[str, Any]:
        code, j = self.post(pid, body)
        if code != 409:
            raise RuntimeError(f"expected a refusal for {body}: {code}")
        return j["error"]

    def preview(self, pid: str, body: dict[str, Any]) -> dict[str, Any]:
        r = self.c.post(f"/api/projects/{pid}/preview", json=body)
        return {"ok": r.status_code == 200, "result": r.json() if r.status_code == 200 else None,
                "refusal": r.json()["error"] if r.status_code == 409 else None}

    def wait_open(self, pid: str, key: str, timeout: float = 300) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            steps = self.view(pid)["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            if step["status"] not in ("open", "waiting"):
                return step
            if first is not None and first["key"] == key and step["status"] == "open":
                return step
            if time.monotonic() > end:
                raise TimeoutError(f"{key} never open; first is {first}")
            time.sleep(0.15)


def steps_of(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": s["key"], "status": s["status"], "reason": s.get("reason")}
            for s in view["interview"]]


def rule_of(proposals: dict[str, Any], key: str) -> dict[str, Any]:
    return next(o["rule"] for o in proposals["exclusions"] if o["key"] == key)


def label_of(proposals: dict[str, Any], key: str) -> str:
    return next(o["label"] for o in proposals["exclusions"] if o["key"] == key)


def sensitivity_decision(proposals: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    labels = {o["key"]: o["label"] for o in proposals["labels"]["exclusions"]["options"]}
    return {"kind": "set_sensitivity", "analyses": [
        {"label": labels[k], "rules": [rule_of(proposals, k)]} for k in keys]}


def energy(method: str) -> dict[str, Any]:
    return {"kind": "set_energy_adjustment", "method": method, "energy_column": "kcal",
            "nutrients": ENERGY_NUTRIENTS}


def opening(api: Api, purpose: str) -> tuple[str, dict[str, Any]]:
    """The scenario's starting point: the table read, the lens, the outcome and the purpose
    answered, and the author's roles recorded (the medium and low ones wait for confirmation)."""
    pid = api.create(nhanes())
    out: dict[str, Any] = {"sentences": {}}
    out["sentences"]["lens"] = api.record(pid, {"kind": "set_lens", "lenses": ["dietary"]})
    api.wait_open(pid, "target")
    out["sentences"]["target"] = api.record(pid, {"kind": "set_target", "column": "glucose"})
    api.wait_open(pid, "purpose")
    out["sentences"]["purpose"] = api.record(pid, {"kind": "set_purpose", "purpose": purpose})
    api.wait_open(pid, "roles")
    out["roles_stage"] = api.stage(pid, "roles")
    out["sentences"]["roles"] = api.record(pid, {"kind": "set_roles", "roles": ROLES})
    view = api.view(pid)
    out["unconfirmed"] = view["state"]["roles_unconfirmed"]
    out["draft_view"] = {"summary": view["summary"], "steps": steps_of(view), "state": view["state"]}
    return pid, out


def drive_main(api: Api) -> dict[str, Any]:
    truth = fixture_truth("_tt_tmp_nhanes.csv")
    pid, out = opening(api, "inference")
    S = out["sentences"]
    unconfirmed: list[str] = out["unconfirmed"]

    # ── the readings: three on their own, then one block of exactly the rest ──
    singles = unconfirmed[:SINGLES]
    S["reading_single"] = {}
    for col in singles:
        S["reading_single"][col] = api.record(pid, {"kind": "confirm_reading", "reading": "role",
                                                    "column": col, "value": ROLES[col]})
    rest = [c for c in unconfirmed if c not in singles]
    block = {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": c, "value": ROLES[c]} for c in rest]}
    S["reading_block"] = {"items": block["items"], **api.record(pid, block)}

    # ── the screens ask what one `kcal` value spans before any reads it ──
    step = api.wait_open(pid, "exclusions")
    ask = step["ask"]
    assert ask and ask["groups"][0]["kind"] == "unit", step
    out["ask_unit"] = ask
    unit = next(e["decision"] for e in ask["exits"] if e["decision"]["kind"] == "set_column_unit")
    assert unit["unit"] == truth.answer("unit", "kcal") and unit["days"] == int(
        truth.answer("day_count", "kcal")), unit
    S["unit_kcal"] = api.record(pid, unit)

    # ── exclusions: every screen's sentence, then the fixture's answer (keep every row) ──
    api.wait_open(pid, "exclusions")
    proposals = api.stage(pid, "proposals")
    S["exclusions"] = {}
    for key in EXCLUSION_KEYS:
        S["exclusions"][key] = api.record(pid, {"kind": "set_exclusions",
                                                "rules": [rule_of(proposals, key)]})
    S["exclusions"]["goldberg_schofield"] = None
    S["exclusions"]["none"] = api.record(pid, {"kind": "set_exclusions", "rules": []})
    # the screens reported beside the primary: every subset of three (each its own sentence)
    S["sensitivity"] = {}
    for n in range(len(SENSITIVITY_KEYS), -1, -1):
        for subset in itertools.combinations(SENSITIVITY_KEYS, n):
            S["sensitivity"]["+".join(subset) or "none"] = api.record(
                pid, sensitivity_decision(proposals, list(subset)))
    # the record keeps all three (the branches capture every screen's fit)
    S["sensitivity_final"] = api.record(pid, sensitivity_decision(proposals, SENSITIVITY_KEYS))

    api.wait_open(pid, "missing")
    S["missing_before_adjustment"] = api.record(pid, {"kind": "set_missing",
                                                      "strategy": "complete_case"})
    api.wait_open(pid, "split")
    out["seal_plan"] = api.stage(pid, "seal_plan")
    S["split"] = api.record(pid, {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})

    # ── the exposure and its estimand ──
    api.wait_open(pid, "estimand")
    proposals = api.stage(pid, "proposals")
    out["estimand_card"] = proposals["estimand"]
    base = {"kind": "set_estimand", "exposure": "sugar", "effect": "total",
            "measure": "mean_difference"}
    out["refusals"] = {"which_contrast": api.refusal(pid, base)}
    S["estimand"] = api.record(pid, {**base, "contrast": "substitution"})

    # ── the adjustment set, from the fixture's causal truth ──
    api.wait_open(pid, "adjustment")
    end = time.monotonic() + 120
    while True:
        card = api.stage(pid, "proposals").get("adjustment")
        if card and card.get("exposure") == "sugar":
            break
        assert time.monotonic() < end
        time.sleep(0.2)
    out["adjustment_card"] = card
    med = adjustment_truth(truth, "hdl")
    med = {k: med[k] for k in ("causes_exposure", "causes_outcome", "after_exposure")}
    out["refusals"]["mediator_kept"] = api.refusal(pid, {
        "kind": "set_adjustment", "exposure": "sugar", "answers": {"hdl": {**med, "keep": True}}})
    S["adjustment"] = {"groups": {}, "unguessed": []}
    for group in card["groups"]:
        if group["decision"] is not None:
            S["adjustment"]["groups"][group["key"]] = api.record(pid, group["decision"])
            continue
        by: dict[tuple[Any, ...], list[str]] = {}
        for c in group["columns"]:
            t = adjustment_truth(truth, c)
            by.setdefault(tuple(t[k] for k in ("causes_exposure", "causes_outcome",
                                               "after_exposure")), []).append(c)
        for answers, columns in by.items():
            body = {"kind": "set_adjustment", "exposure": "sugar", "answers": {
                c: dict(zip(("causes_exposure", "causes_outcome", "after_exposure"), answers))
                for c in columns}}
            S["adjustment"]["unguessed"].append({"columns": columns, "answers": list(answers),
                                                  "decision": body, **api.record(pid, body)})
    out["adjustment_after"] = api.stage(pid, "proposals")["adjustment"]

    # ── the energy model: the refusals, every runnable method's sentence, then the standard ──
    api.wait_open(pid, "energy_adjustment")
    for m in ENERGY_REFUSED:
        out["refusals"][f"energy_{m}"] = api.refusal(pid, energy(m))
    S["energy"] = {m: api.record(pid, energy(m)) for m in ENERGY_OK if m != "standard"}
    S["energy"]["standard"] = api.record(pid, energy("standard"))

    # ── missing data, asked again now the model's columns are known (its sentence is then true) ──
    S["missing"] = {
        "multiple_imputation": api.record(pid, {"kind": "set_missing",
                                                "strategy": "multiple_imputation", "m": 20}),
        "complete_case": api.record(pid, {"kind": "set_missing", "strategy": "complete_case"})}

    # ── the exposure's form and Model 1 ──
    # the straight line last: the engine fits one when no form is recorded, and the map writes it in
    S["form"] = {k: api.record(pid, {"kind": "set_exposure_form", "column": "sugar", **v})
                 for k, v in FORMS.items() if k != "linear"}
    S["form"]["linear"] = api.record(pid, {"kind": "set_exposure_form", "column": "sugar",
                                           **FORMS["linear"]})
    proposals = api.stage(pid, "proposals")
    out["model_sequence"] = proposals["model_sequence"]
    S["model_sequence"] = {"guess": api.record(pid, proposals["model_sequence"]["decision"])}

    # ── the family: the fit asks first what two whole-number columns hold ──
    step = api.wait_open(pid, "models")
    out["models_step"] = step
    ask = step["ask"]
    assert ask and ask["groups"], step
    out["ask_codes"] = ask
    codes = next(e["decision"] for e in ask["exits"] if e["decision"]["kind"] == "confirm_readings")
    for item in codes["items"]:  # the guesses are the fixture's truth
        assert truth.answer("code_or_count", item["column"]) == item["value"], item
    S["reading_codes"] = {"items": codes["items"], **api.record(pid, codes)}
    out["shelf"] = api.stage(pid, "shelf")
    S["models"] = api.record(pid, {"kind": "select_models", "models": ["linear"]})

    # ── what the record reads before the lock: the cards, previews, readings, the methods ──
    proposals = api.stage(pid, "proposals")
    out["proposals"] = proposals
    out["readings"] = api.c.get(f"/api/projects/{pid}/readings").json()
    out["previews"] = previews(api, pid, proposals, out["seal_plan"])
    # A preview is true of the state it was taken on: the same previews on the other recordable
    # states (the curve; Willett's screen), so each says what it would do from there.
    excl_rule = {"willett_2013_by_sex": [rule_of(proposals, "willett_2013_by_sex")], "none": []}
    out["previews_var"] = {}
    for excl, form in [("none", "spline"), ("willett_2013_by_sex", "spline"),
                       ("willett_2013_by_sex", "linear")]:
        api.record(pid, {"kind": "set_exclusions", "rules": excl_rule[excl]})
        api.record(pid, {"kind": "set_exposure_form", "column": "sugar", **FORMS[form]})
        out["previews_var"][f"{excl}|{form}"] = previews(api, pid, proposals, out["seal_plan"])
    api.record(pid, {"kind": "set_exclusions", "rules": []})
    api.record(pid, {"kind": "set_exposure_form", "column": "sugar", **FORMS["linear"]})
    view = api.view(pid)
    out["pre_lock_view"] = {"steps": steps_of(view), "state": view["state"],
                            "summary": view["summary"]}
    out["methods_pre_lock"] = api.c.get(f"/api/projects/{pid}/methods").json()

    # ── the lock: the first estimate served ──
    out["fit"] = api.stage(pid, "fit")
    view = api.view(pid)
    lock = max(view["decisions"], key=lambda d: d["seq"])
    assert lock["decision"]["kind"] == "lock_plan", lock["decision"]["kind"]
    S["lock"] = {"sentence": lock["sentence"], "digest": lock["decision"]["digest"],
                 "state": out["pre_lock_view"]["state"]}
    out["effects"] = api.stage(pid, "effects")
    out["secondary"] = api.stage(pid, "secondary")
    out["sensitivity_stage"] = api.stage(pid, "sensitivity")
    out["methods"] = api.c.get(f"/api/projects/{pid}/methods").json()
    out["plan"] = json.loads(api.c.get(f"/api/projects/{pid}/plan").content)
    out["pid"] = pid
    return out


def previews(api: Api, pid: str, proposals: dict[str, Any], seal_plan: dict[str, Any]) -> dict:
    p: dict[str, Any] = {"exclusions": {}, "energy": {}, "split": {}, "missing": {}, "form": {},
                         "estimand": None, "adjustment": None, "unit": None, "models": None}
    p["exclusions"]["none"] = api.preview(pid, {"kind": "set_exclusions", "rules": []})
    for key in [*EXCLUSION_KEYS, "goldberg_schofield"]:
        if any(o["key"] == key for o in proposals["exclusions"]):
            p["exclusions"][key] = api.preview(pid, {"kind": "set_exclusions",
                                                     "rules": [rule_of(proposals, key)]})
    for m in [*ENERGY_OK, *ENERGY_REFUSED]:
        p["energy"][m] = api.preview(pid, energy(m))
    for opt in seal_plan["options"]:
        p["split"][str(opt["holdout"])] = api.preview(pid, {"kind": "set_split",
                                                            "holdout": opt["holdout"], "seed": 0,
                                                            "folds": 5})
    p["missing"]["complete_case"] = api.preview(pid, {"kind": "set_missing",
                                                      "strategy": "complete_case"})
    p["missing"]["multiple_imputation"] = api.preview(pid, {"kind": "set_missing",
                                                            "strategy": "multiple_imputation",
                                                            "m": 20})
    for k, v in FORMS.items():
        p["form"][k] = api.preview(pid, {"kind": "set_exposure_form", "column": "sugar", **v})
    p["estimand"] = api.preview(pid, {"kind": "set_estimand", "exposure": "sugar",
                                      "effect": "total", "measure": "mean_difference",
                                      "contrast": "substitution"})
    p["adjustment"] = api.preview(pid, {"kind": "set_adjustment", "exposure": "sugar",
                                        "answers": {"age": {"causes_exposure": "yes",
                                                            "causes_outcome": "yes",
                                                            "after_exposure": "no"}}})
    p["unit"] = api.preview(pid, {"kind": "set_column_unit", "column": "kcal", "unit": "kcal",
                                  "days": 1})
    p["unit_2days"] = api.preview(pid, {"kind": "set_column_unit", "column": "kcal",
                                        "unit": "kcal", "days": 2})
    p["models"] = api.preview(pid, {"kind": "select_models", "models": ["linear"]})
    return p


def fit_bundle(api: Api, pid: str) -> dict[str, Any]:
    try:
        return {"fit": api.stage(pid, "fit"), "effects": api.stage(pid, "effects"),
                "secondary": api.stage(pid, "secondary"),
                "sensitivity": api.stage(pid, "sensitivity")}
    except RuntimeError:
        # A combination the engine cannot fit: keep the stage's own error, as the app shows it.
        stages = api.view(pid)["stages"]
        failed = {k: v["error"] for k, v in stages.items() if v["status"] == "error"}
        return {"error": failed}


def drive_branches(api: Api) -> dict[str, Any]:
    """A second project on the canonical answers; then every recordable combination's fits."""
    truth = fixture_truth("_tt_tmp_nhanes.csv")
    pid, _ = opening(api, "inference")
    view = api.view(pid)
    rest = view["state"]["roles_unconfirmed"]
    api.record(pid, {"kind": "confirm_readings", "items": [
        *({"reading": "role", "column": c, "value": ROLES[c]} for c in rest),
        {"reading": "code_or_count", "column": "age", "value": "amount"},
        {"reading": "code_or_count", "column": "cycle_begin_year", "value": "code"}]})
    api.record(pid, {"kind": "set_column_unit", "column": "kcal", "unit": "kcal", "days": 1})
    api.wait_open(pid, "exclusions")
    proposals = api.stage(pid, "proposals")
    api.record(pid, {"kind": "set_exclusions", "rules": []})
    api.record(pid, sensitivity_decision(proposals, SENSITIVITY_KEYS))
    api.wait_open(pid, "missing")
    api.record(pid, {"kind": "set_missing", "strategy": "complete_case"})
    api.wait_open(pid, "split")
    api.record(pid, {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    api.wait_open(pid, "estimand")
    api.record(pid, {"kind": "set_estimand", "exposure": "sugar", "effect": "total",
                     "measure": "mean_difference", "contrast": "substitution"})
    api.wait_open(pid, "adjustment")
    while True:
        card = api.stage(pid, "proposals").get("adjustment")
        if card and card.get("exposure") == "sugar":
            break
        time.sleep(0.2)
    from turbotab.core.tests.truths import answer_adjustment
    answer_adjustment(lambda d: api.c.post(f"/api/projects/{pid}/decisions", json=d), card, truth)
    api.wait_open(pid, "energy_adjustment")
    api.record(pid, energy("standard"))
    api.record(pid, {"kind": "set_missing", "strategy": "complete_case"})
    api.record(pid, {"kind": "set_exposure_form", "column": "sugar", **FORMS["linear"]})
    api.record(pid, api.stage(pid, "proposals")["model_sequence"]["decision"])
    api.wait_open(pid, "models")
    api.record(pid, {"kind": "select_models", "models": ["linear"]})

    slots: dict[str, Any] = {"exclusions": {}, "exposure_forms": {}, "energy_adjustment": {},
                             "sensitivity": {}}
    fits: dict[str, Any] = {}
    after: dict[str, Any] = {"energy": {}}
    base_state = api.view(pid)["state"]
    api.stage(pid, "fit")  # the lock: every answer below is one made after the estimates were seen
    for excl in ["none", "willett_2013_by_sex"]:
        rules = [] if excl == "none" else [rule_of(proposals, excl)]
        api.record(pid, {"kind": "set_exclusions", "rules": rules})
        slots["exclusions"][excl] = api.view(pid)["state"]["exclusions"]
        for form in ["linear", "spline"]:
            rec = api.record(pid, {"kind": "set_exposure_form", "column": "sugar", **FORMS[form]})
            if excl == "none":
                after.setdefault("form", {})[form] = rec
            slots["exposure_forms"][form] = api.view(pid)["state"]["exposure_forms"]
            for method in ENERGY_OK:
                rec = api.record(pid, energy(method))
                slots["energy_adjustment"][method] = api.view(pid)["state"]["energy_adjustment"]
                if excl == "none" and form == "linear":
                    after["energy"][method] = rec
                t = time.time()
                fits[f"{excl}|{form}|{method}"] = fit_bundle(api, pid)
                print(f"  fit {excl}|{form}|{method} {time.time() - t:.1f}s", flush=True)
    for n in range(len(SENSITIVITY_KEYS) + 1):
        for subset in itertools.combinations(SENSITIVITY_KEYS, n):
            api.record(pid, sensitivity_decision(proposals, list(subset)))
            slots["sensitivity"]["+".join(subset) or "none"] = api.view(pid)["state"]["sensitivity"]
    return {"pid": pid, "fits": fits, "slots": slots, "after": after, "base_state": base_state}


def drive_prediction(api: Api) -> dict[str, Any]:
    pid, out = opening(api, "prediction")
    rest = out["unconfirmed"]
    api.record(pid, {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": c, "value": ROLES[c]} for c in rest]})
    api.record(pid, {"kind": "set_column_unit", "column": "kcal", "unit": "kcal", "days": 1})
    api.wait_open(pid, "exclusions")
    proposals = api.stage(pid, "proposals")
    out["previews"] = {"exclusions": {"none": api.preview(pid, {"kind": "set_exclusions",
                                                                "rules": []})}}
    out["previews"]["exclusions"]["willett_2013_by_sex"] = api.preview(
        pid, {"kind": "set_exclusions", "rules": [rule_of(proposals, "willett_2013_by_sex")]})
    out["sentences"]["exclusions_none"] = api.record(pid, {"kind": "set_exclusions", "rules": []})
    api.wait_open(pid, "missing")
    proposals = api.stage(pid, "proposals")
    out["proposals"] = proposals
    out["previews"]["missing"] = {
        k: api.preview(pid, {"kind": "set_missing", "strategy": k})
        for k in ("impute", "complete_case")}
    out["missing_step"] = steps_of(api.view(pid))
    out["sentences"]["missing"] = api.record(pid, {"kind": "set_missing", "strategy": "impute"})
    api.wait_open(pid, "split")
    out["seal_plan"] = api.stage(pid, "seal_plan")
    out["previews"]["split"] = {
        str(o["holdout"]): api.preview(pid, {"kind": "set_split", "holdout": o["holdout"],
                                             "seed": 0, "folds": 5})
        for o in out["seal_plan"]["options"]}
    view = api.view(pid)
    out["view"] = {"steps": steps_of(view), "state": view["state"], "summary": view["summary"]}
    # the seal answered both ways (its sentences), the guess last; then what the seal unblocks
    out["sentences"]["split"] = {
        "0.0": api.record(pid, {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5}),
        "0.2": api.record(pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})}
    out["shelf"] = api.stage(pid, "shelf")
    out["proposals_after_seal"] = api.stage(pid, "proposals")
    view = api.view(pid)
    out["view_after_seal"] = {"steps": steps_of(view), "summary": view["summary"]}
    out["readings"] = api.c.get(f"/api/projects/{pid}/readings").json()
    out["pid"] = pid
    return out


def in_process(main: dict[str, Any], branches: dict[str, Any]) -> dict[str, Any]:
    """Sentence variants that are pure functions of the decision (the engine's ``sentence_for``),
    the criterion's derivations and each plan's SHA-256 — checked against the server first."""
    from turbotab.core import estimand, plan_lock, voice
    from turbotab.core.decisions import ProjectState

    state = ProjectState(**main["draft_view"]["state"])
    S = main["sentences"]
    # every reading on its own, and the block of whatever remains after any three
    roles_cols = main["unconfirmed"]
    codes = [("age", "amount"), ("cycle_begin_year", "code")]
    items = [*({"reading": "role", "column": c, "value": ROLES[c]} for c in roles_cols),
             *({"reading": "code_or_count", "column": c, "value": v} for c, v in codes)]
    single = {}
    for it in items:
        single[f"{it['reading']}:{it['column']}"] = voice.sentence_for(
            {"kind": "confirm_reading", **it}, state)
    for col, rec in S["reading_single"].items():
        assert single[f"role:{col}"] == rec["sentence"], (single[f"role:{col}"], rec["sentence"])
    # The block lists exactly what is still unconfirmed, in the card's order. It unlocks after
    # three single confirmations; its sentence is taken for whatever three or four were done
    # singly (a bit mask of the readings it lists → one of the distinct sentences).
    keys = [f"{it['reading']}:{it['column']}" for it in items]
    sentences: list[str] = []
    index: dict[str, int] = {}
    by_mask: dict[str, int] = {}
    full = (1 << len(items)) - 1
    for k in (SINGLES, SINGLES + 1):
        for done in itertools.combinations(range(len(items)), k):
            mask = full
            for i in done:
                mask &= ~(1 << i)
            left = [items[i] for i in range(len(items)) if mask >> i & 1]
            text = voice.sentence_for({"kind": "confirm_readings", "items": left}, state)
            if text not in index:
                index[text] = len(sentences)
                sentences.append(text)
            by_mask[str(mask)] = index[text]
    for rec in (S["reading_block"], S["reading_codes"]):  # the server's own blocks
        said = voice.sentence_for({"kind": "confirm_readings", "items": rec["items"]}, state)
        assert said == rec["sentence"], (said, rec["sentence"])
    blocks = {"items": keys, "sentences": sentences, "by_mask": by_mask}
    # the criterion: every answer to its three questions, and what it derives (total effect)
    values = ("yes", "no", "unknown")
    derive = {}
    for a in itertools.product(values, repeat=3):
        ans = dict(zip(("causes_exposure", "causes_outcome", "after_exposure"), a))
        d = estimand.derive(ans, "total")
        derive[",".join(a)] = {"role": d.role, "words": estimand.ROLE_WORDS[d.role],
                               "adjusted": d.adjusted, "secondary": d.secondary, "why": d.why}
    # one column's answer recorded on its own (an answer other than the fixture's)
    est_state = ProjectState(**main["pre_lock_view"]["state"])
    texts: list[str] = []
    at: dict[str, int] = {}
    per_column: dict[str, dict[str, int]] = {}
    for group in main["adjustment_card"]["groups"]:
        for col in group["columns"]:
            per_column[col] = {}
            for a in itertools.product(values, repeat=3):
                body = {"kind": "set_adjustment", "exposure": "sugar", "answers": {
                    col: dict(zip(("causes_exposure", "causes_outcome", "after_exposure"), a))}}
                text = voice.sentence_for(body, est_state)
                if text not in at:
                    at[text] = len(texts)
                    texts.append(text)
                per_column[col][",".join(a)] = at[text]
    per_column = {"sentences": texts, "by_column": per_column}
    for rec in S["adjustment"]["unguessed"]:
        assert voice.sentence_for(rec["decision"], est_state) == rec["sentence"]
    # the plan's SHA-256 for each combination the prototype can lock
    pre = main["pre_lock_view"]["state"]
    assert plan_lock.digest(plan_lock.plan_of(ProjectState(**pre))) == S["lock"]["digest"]
    slots = branches["slots"]
    guess = main["model_sequence"]["decision"]
    model_1 = {"guess": {"exposure": "sugar", "model_1": guess["model_1"]},
               "empty": {"exposure": "sugar", "model_1": []}, "unset": None}
    model_1_empty = voice.sentence_for({**guess, "model_1": []}, est_state)
    assert voice.sentence_for(guess, est_state) == S["model_sequence"]["guess"]["sentence"]
    lock = {}
    for excl, form, method, sens, m1 in itertools.product(
            slots["exclusions"], slots["exposure_forms"], slots["energy_adjustment"],
            slots["sensitivity"], model_1):
        st = dict(pre)
        st["exclusions"] = slots["exclusions"][excl]
        st["exposure_forms"] = slots["exposure_forms"][form]
        st["energy_adjustment"] = slots["energy_adjustment"][method]
        st["sensitivity"] = slots["sensitivity"][sens]
        st["model_sequence"] = model_1[m1]
        dg = plan_lock.digest(plan_lock.plan_of(ProjectState(**st)))
        lock[f"{excl}|{form}|{method}|{sens}|{m1}"] = dg[:12]
    canon = f"none|linear|standard|{'+'.join(SENSITIVITY_KEYS)}|guess"
    assert lock[canon] == S["lock"]["digest"][:12], (lock[canon], S["lock"]["digest"])
    # The lock's sentence differs only by its SHA-256: one template per digest, from the engine.
    sample = voice.sentence_for({"kind": "lock_plan", "digest": S["lock"]["digest"]},
                                ProjectState())
    assert sample == S["lock"]["sentence"]
    # Stored once with the digest's place marked; checked to give the engine's sentence exactly for
    # every digest the prototype can show.
    template = sample.replace(S["lock"]["digest"][:12], "{digest}")
    for dg in set(lock.values()):
        said = voice.sentence_for({"kind": "lock_plan", "digest": dg + "0" * 52}, ProjectState())
        assert template.replace("{digest}", dg) == said
    return {"single": single, "blocks": blocks, "derive": derive, "per_column": per_column,
            "lock": lock, "lock_template": template, "model_1_empty": model_1_empty}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8873")
    ap.add_argument("--raw", default=str(Path(tempfile.gettempdir()) / "methods-map-raw"))
    ap.add_argument("--only", choices=["main", "branches", "prediction", "inproc", "trim"])
    args = ap.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    api = Api(args.base)

    def phase(name: str, fn):
        path = raw / f"{name}.json"
        if args.only in (None, name) and not (args.only is None and path.exists()):
            t = time.time()
            print(f"{name} …", flush=True)
            data = fn()
            path.write_text(json.dumps(data))
            print(f"{name} done in {time.time() - t:.0f}s", flush=True)
        return json.loads(path.read_text()) if path.exists() else None

    teaching_path = raw / "teaching.json"
    if not teaching_path.exists():
        teaching_path.write_text(json.dumps(api.c.get("/api/teaching").json()))
    m = phase("main", lambda: drive_main(api))
    b = phase("branches", lambda: drive_branches(api))
    p = phase("prediction", lambda: drive_prediction(api))
    ip = phase("inproc", lambda: in_process(m, b))
    if args.only in (None, "trim"):
        from trim import trim  # noqa: PLC0415 - beside this file
        fixture = trim(json.loads(teaching_path.read_text()), m, b, p, ip)
        out = HERE / "fixture.json"
        out.write_text(json.dumps(fixture, separators=(",", ":"), ensure_ascii=False))
        print(f"fixture.json {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

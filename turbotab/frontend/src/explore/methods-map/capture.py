"""Capture the /lab/methods-map prototype's fixture from the real server (not part of the bundle).

    TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 TURBOTAB_HOME=$(mktemp -d) \
        venv/bin/python -m turbotab.server --port 8973                    # repo root
    venv/bin/python turbotab/frontend/src/explore/methods-map/capture.py \
        --base http://127.0.0.1:8973 --raw /tmp/methods-map-raw            # repo root

The analysis is the three prototypes' one scenario (``../methods-shared/scenario.py``,
``SCENARIO.md``): every answer, in its order, is the scenario's, and every reading is answered from
the fixture's declared truth (``turbotab/core/tests/truths.py``). Four drives, each through the
HTTP API:

* **main**: ``scenario.run_inference``, watched. At each moment its hook takes what the map shows
  there (the cards, the asks, previews — a preview records nothing — the readings, the methods
  text before and after the lock, the fit) and never records anything: the main project holds the
  scenario's answers only. A refusal the map shows (the contrast the estimand asks for, a mediator
  kept, an energy model the estimand rules out) is the server's preview of that answer. The
  previews are taken once the plan is whole (at its own question the engine can draw neither the
  estimand nor the model sequence yet: "Nothing about this choice can be shown on your data yet").
* **branches**: the same scenario on a second project, whose hooks also record what the map offers
  besides the scenario's answers, to take the engine's sentence for each (every screen, every
  subset of the two screens reported beside, every runnable energy model, every form), and the
  previews on the other recordable states. At the adjustment card it answers the groups one at a
  time, as the map does, each previewed on the state before it (the scenario then answers them
  again, the same answers). Then, after its lock, every combination the map can record (exclusions
  × form × energy model) with its Table 2, secondary and sensitivity fits; their sentences are the
  after-the-estimates ones the record shows for a post-lock edit.
* **prediction**: ``scenario.run_prediction``, watched, to the fit (the map shows it up to the seal):
  its own readings (under prediction one block settles the role readings and four code-or-amount
  ones), exclusions, missing data and seal.
* **in process**: sentence variants that are pure functions of the decision (the engine's own
  ``voice.sentence_for``, checked against the server on the scenario's), the criterion's
  derivations (``estimand.derive``) and each plan's SHA-256 (``plan_lock``).

Then ``trim`` writes ``fixture.json`` beside this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "methods-shared"))

import scenario as S  # noqa: E402

ENERGY_NUTRIENTS = list(S.NUTRIENTS)
ENERGY_OK = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density"]
ENERGY_REFUSED = ["none", "all_components", "partition"]
EXCLUSION_KEYS = ["willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000",
                  "sex_neutral_500_3500"]
SENSITIVITY_KEYS = list(S.SENSITIVITY_KEYS)
FORMS = {"spline": {"form": "spline", "knots": 4}, "linear": {"form": "linear"},
         "quintiles": {"form": "quintiles"}}
SINGLES = 3  # single confirmations before the block confirm unlocks (BLUEPRINT §11.4)


class Api:
    """Raw polling over one project (the branch fits, whose failures are kept as the app shows)."""

    def __init__(self, base: str):
        self.c = httpx.Client(base_url=base, timeout=900)

    def view(self, pid: str) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{pid}").json()

    def stage(self, pid: str, name: str, timeout: float = 900) -> dict[str, Any]:
        from turbotab.core.estimand import ESTIMATE_STAGES

        if name in ESTIMATE_STAGES:  # served only after Fit (P0.8): pressed, as Results opens
            self.c.post(f"/api/projects/{pid}/fit")
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


def steps_of(view: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": s["key"], "status": s["status"], "reason": s.get("reason")}
            for s in view["interview"]]


def rule_of(proposals: dict[str, Any], key: str) -> dict[str, Any]:
    return S.screen_rule(proposals, key)


def energy(method: str) -> dict[str, Any]:
    return S.energy(method)


def pv(j: S.Journey, body: dict[str, Any]) -> dict[str, Any]:
    """A preview, as the map keeps it: its result, or the engine's refusal of that answer."""
    r = j.preview(body)
    return {"ok": r["status"] == 200, "result": r["body"] if r["status"] == 200 else None,
            "refusal": r["body"]["error"] if r["status"] == 409 else None}


def refusal(j: S.Journey, body: dict[str, Any]) -> dict[str, Any]:
    p = pv(j, body)
    assert p["refusal"], f"expected the engine to refuse {body}: {p}"
    return p["refusal"]


def rec(j: S.Journey, body: dict[str, Any]) -> dict[str, Any]:
    """Record on a side project (never the main one) and keep the sentence it wrote."""
    d = j.record(body)
    j.wait_quiet()
    return {"sentence": d["sentence"], "after_estimates": d["after_estimates"], "seq": d["seq"],
            "kind": d["decision"]["kind"]}


def previews(j: S.Journey, proposals: dict[str, Any], seal_plan: dict[str, Any]) -> dict:
    p: dict[str, Any] = {"exclusions": {}, "energy": {}, "split": {}, "missing": {}, "form": {},
                         "estimand": None, "adjustment": None, "unit": None, "models": None}
    p["exclusions"]["none"] = pv(j, {"kind": "set_exclusions", "rules": []})
    for key in [*EXCLUSION_KEYS, "goldberg_schofield"]:
        if any(o["key"] == key for o in proposals["exclusions"]):
            p["exclusions"][key] = pv(j, {"kind": "set_exclusions",
                                          "rules": [rule_of(proposals, key)]})
    for m in [*ENERGY_OK, *ENERGY_REFUSED]:
        p["energy"][m] = pv(j, energy(m))
    for opt in seal_plan["options"]:
        p["split"][str(opt["holdout"])] = pv(j, {"kind": "set_split", "holdout": opt["holdout"],
                                                 "seed": 0, "folds": 5})
    p["missing"]["complete_case"] = pv(j, {"kind": "set_missing", "strategy": "complete_case"})
    p["missing"]["multiple_imputation"] = pv(j, {"kind": "set_missing",
                                                 "strategy": "multiple_imputation", "m": 20})
    for k, v in FORMS.items():
        p["form"][k] = pv(j, {"kind": "set_exposure_form", "column": S.EXPOSURE, **v})
    p["estimand"] = pv(j, S.estimand())
    p["adjustment"] = pv(j, {"kind": "set_adjustment", "exposure": S.EXPOSURE,
                             "answers": {"age": {"causes_exposure": "yes", "causes_outcome": "yes",
                                                 "after_exposure": "no"}}})
    p["unit"] = pv(j, {"kind": "set_column_unit", "column": "kcal", "unit": "kcal", "days": 1})
    p["unit_2days"] = pv(j, {"kind": "set_column_unit", "column": "kcal", "unit": "kcal",
                             "days": 2})
    p["models"] = pv(j, {"kind": "select_models", "models": list(S.INFERENCE_MODELS)})
    return p


class _Collected:
    """What :func:`group_decisions` hands ``answer_adjustment`` for a post it only collects."""

    status_code = 200
    text = ""


def group_decisions(card: dict[str, Any], truth: Any, group: dict[str, Any]) -> list[dict[str, Any]]:
    """The decisions the scenario records for one group of the adjustment card (truths.py's
    ``answer_adjustment`` on that group alone), not posted."""
    from turbotab.core.tests.truths import answer_adjustment

    got: list[dict[str, Any]] = []

    def collect(d: dict[str, Any]) -> _Collected:
        got.append(d)
        return _Collected()

    answer_adjustment(collect, {**card, "groups": [group]}, truth)
    return got


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


def sentences_of(decisions: list[dict[str, Any]], adjustment_card: dict[str, Any]) -> dict[str, Any]:
    """The scenario's own sentences, read off the main project's record."""
    out: dict[str, Any] = {"reading_single": {}, "adjustment": {"groups": {}, "unguessed": []}}
    by_decision = {json.dumps(g["decision"], sort_keys=True): g["key"]
                   for g in adjustment_card["groups"] if g.get("decision")}

    def keep(d: dict[str, Any]) -> dict[str, Any]:
        return {"sentence": d["sentence"], "after_estimates": d["after_estimates"], "seq": d["seq"],
                "kind": d["decision"]["kind"]}

    for d in decisions:
        dec, kind = d["decision"], d["decision"]["kind"]
        if kind in ("set_lens", "set_target", "set_purpose", "set_roles"):
            out[kind[4:]] = keep(d)
        elif kind == "set_column_unit":
            out["unit_kcal"] = keep(d)
        elif kind == "set_exclusions":
            out["exclusions_none"] = keep(d)
        elif kind == "set_sensitivity":
            out["sensitivity_both"] = keep(d)
        elif kind == "set_missing":
            out["missing"] = keep(d)
        elif kind == "set_split":
            out["split"] = keep(d)
        elif kind == "confirm_reading":
            out["reading_single"][f"{dec['reading']}:{dec['column']}"] = keep(d)
        elif kind == "confirm_readings":
            name = "reading_block" if dec["items"][0]["reading"] == "role" else "reading_codes"
            out[name] = {"items": dec["items"], **keep(d)}
        elif kind == "set_estimand":
            out["estimand"] = keep(d)
        elif kind == "set_adjustment":
            key = by_decision.get(json.dumps(dec, sort_keys=True))
            if key is not None:
                out["adjustment"]["groups"][key] = keep(d)
            else:
                cols = list(dec["answers"])
                a = dec["answers"][cols[0]]
                out["adjustment"]["unguessed"].append({
                    "columns": cols,
                    "answers": [a["causes_exposure"], a["causes_outcome"], a["after_exposure"]],
                    "decision": dec, **keep(d)})
        elif kind == "set_energy_adjustment":
            out["energy_standard"] = keep(d)
        elif kind == "set_model_sequence":
            out["model_sequence"] = {"guess": keep(d)}
        elif kind == "select_models":
            out["models"] = keep(d)
        elif kind == "lock_plan":
            out["lock"] = {"sentence": d["sentence"], "digest": dec["digest"]}
    return out


# ── main: the scenario, watched ──────────────────────────────────────────────


def drive_main(base: str) -> dict[str, Any]:
    client = S.Client(base)
    out: dict[str, Any] = {"refusals": {}}

    def at(moment: str, j: S.Journey) -> None:
        print(f"  main · {moment}", flush=True)
        if moment == "roles":
            view = j.view()
            out["roles_stage"] = j.artifact("roles")
            out["unconfirmed"] = view["state"]["roles_unconfirmed"]
            out["draft_view"] = {"summary": view["summary"], "steps": steps_of(view),
                                 "state": view["state"]}
        elif moment == "unit_ask":
            out["ask_unit"] = j.ask("exclusions")
        elif moment == "split":
            out["seal_plan"] = j.artifact("seal_plan")
        elif moment == "readings":
            out["ask_roles"] = j.ask("estimand")
            out["readings_state"] = j.view()["state"]
        elif moment == "estimand":
            out["estimand_card"] = j.artifact("proposals")["estimand"]
            base_estimand = {k: v for k, v in S.estimand().items() if k != "contrast"}
            out["refusals"]["which_contrast"] = refusal(j, base_estimand)
        elif moment == "adjustment":
            card = j.artifact("proposals")["adjustment"]
            out["adjustment_card"] = card
            from turbotab.core.tests.truths import adjustment_truth

            med = adjustment_truth(j.truth, "hdl")
            med = {k: med[k] for k in ("causes_exposure", "causes_outcome", "after_exposure")}
            out["refusals"]["mediator_kept"] = refusal(j, {
                "kind": "set_adjustment", "exposure": S.EXPOSURE,
                "answers": {"hdl": {**med, "keep": True}}})
        elif moment == "energy":
            out["adjustment_after"] = j.artifact("proposals")["adjustment"]
            for m in ENERGY_REFUSED:
                out["refusals"][f"energy_{m}"] = refusal(j, energy(m))
        elif moment == "model_sequence":
            ms = j.artifact("proposals")["model_sequence"]
            assert ms["decision"] == S.model_sequence(), (ms["decision"], S.model_sequence())
            out["model_sequence"] = ms
        elif moment == "models":
            out["ask_codes"] = j.ask("models")
            out["models_state"] = j.view()["state"]
            out["shelf"] = j.artifact("shelf")
        elif moment == "ready":
            proposals = j.artifact("proposals")
            out["proposals"] = proposals
            out["readings"] = j.readings()
            out["previews"] = previews(j, proposals, out["seal_plan"])
            out["preview_model_sequence"] = {
                "guess": pv(j, S.model_sequence()),
                "empty": pv(j, {**S.model_sequence(), "model_1": []})}
            view = j.view()
            out["pre_lock_view"] = {"steps": steps_of(view), "state": view["state"],
                                    "summary": view["summary"]}
            out["methods_pre_lock"] = j.methods()
        elif moment == "locked":
            out["fit"] = j.artifact("fit")
            out["effects"] = j.artifact("effects")
            out["secondary"] = j.artifact("secondary")
            out["sensitivity_stage"] = j.artifact("sensitivity")
            out["methods"] = j.methods()
            out["plan"] = json.loads(client.get(f"/api/projects/{j.pid}/plan").content)
            out["decisions"] = j.view()["decisions"]

    j = S.run_inference(client, at)
    out["pid"] = j.pid
    out["sentences"] = sentences_of(out["decisions"], out["adjustment_card"])
    lock = out["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan", lock["decision"]["kind"]
    return out


# ── branches: a second project on the scenario, its sentences and its fits ──


def sensitivity_decision(proposals: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return S.sensitivity(proposals, tuple(keys))


def drive_branches(base: str, main: dict[str, Any]) -> dict[str, Any]:
    client = S.Client(base)
    api = Api(base)
    out: dict[str, Any] = {"sentences": {"exclusions": {}, "sensitivity": {}, "energy": {},
                                         "form": {}},
                           "slots": {"exclusions": {}, "exposure_forms": {}, "energy_adjustment": {},
                                     "sensitivity": {}},
                           "after": {"energy": {}, "form": {}}, "fits": {}}
    Sx = out["sentences"]
    seal_plan = main["seal_plan"]

    def at(moment: str, j: S.Journey) -> None:
        print(f"  branches · {moment}", flush=True)
        if moment == "exclusions":
            proposals = j.artifact("proposals")
            for key in EXCLUSION_KEYS:
                Sx["exclusions"][key] = rec(j, {"kind": "set_exclusions",
                                                "rules": [rule_of(proposals, key)]})
        elif moment == "adjustment":
            # the groups one at a time, each previewed on the state the ones before it leave
            card = j.artifact("proposals")["adjustment"]
            out["adjustment_previews"] = {}
            for group in card["groups"]:
                decisions = group_decisions(card, j.truth, group)
                answers: dict[str, Any] = {}
                for d in decisions:
                    answers.update(d["answers"])
                out["adjustment_previews"][group["key"]] = pv(j, {
                    "kind": "set_adjustment", "exposure": S.EXPOSURE, "answers": answers})
                for d in decisions:
                    rec(j, d)
        elif moment == "missing":
            # the scenario recorded every row and both screens; every other subset's sentence
            proposals = j.artifact("proposals")
            for n in range(len(SENSITIVITY_KEYS) - 1, -1, -1):
                for subset in itertools.combinations(SENSITIVITY_KEYS, n):
                    Sx["sensitivity"]["+".join(subset) or "none"] = rec(
                        j, sensitivity_decision(proposals, list(subset)))
            rec(j, sensitivity_decision(proposals, SENSITIVITY_KEYS))  # the scenario's again
        elif moment == "model_sequence":
            # the scenario recorded the standard model; each other runnable model's sentence
            for m in ENERGY_OK:
                if m != "standard":
                    Sx["energy"][m] = rec(j, energy(m))
            rec(j, energy("standard"))
        elif moment == "ready":
            for k in ("spline", "quintiles", "linear"):
                Sx["form"][k] = rec(j, {"kind": "set_exposure_form", "column": S.EXPOSURE,
                                        **FORMS[k]})
            # A preview is true of the state it was taken on: the same previews on the other
            # recordable states (a curve; Willett's screen as the primary).
            proposals = j.artifact("proposals")
            excl_rule = {"willett_2013_by_sex": [rule_of(proposals, "willett_2013_by_sex")],
                         "none": []}
            out["previews_var"] = {}
            for excl, form in [("none", "spline"), ("willett_2013_by_sex", "spline"),
                               ("willett_2013_by_sex", "linear")]:
                rec(j, {"kind": "set_exclusions", "rules": excl_rule[excl]})
                rec(j, {"kind": "set_exposure_form", "column": S.EXPOSURE, **FORMS[form]})
                out["previews_var"][f"{excl}|{form}"] = previews(j, proposals, seal_plan)
            rec(j, {"kind": "set_exclusions", "rules": []})
            rec(j, {"kind": "set_exposure_form", "column": S.EXPOSURE, **FORMS["linear"]})
        elif moment == "locked":
            pid = j.pid
            proposals = j.artifact("proposals")
            for excl in ["none", "willett_2013_by_sex"]:
                rules = [] if excl == "none" else [rule_of(proposals, excl)]
                rec(j, {"kind": "set_exclusions", "rules": rules})
                out["slots"]["exclusions"][excl] = j.view()["state"]["exclusions"]
                for form in ["linear", "spline"]:
                    r = rec(j, {"kind": "set_exposure_form", "column": S.EXPOSURE, **FORMS[form]})
                    if excl == "none":
                        out["after"]["form"][form] = r
                    out["slots"]["exposure_forms"][form] = j.view()["state"]["exposure_forms"]
                    for method in ENERGY_OK:
                        r = rec(j, energy(method))
                        out["slots"]["energy_adjustment"][method] = \
                            j.view()["state"]["energy_adjustment"]
                        if excl == "none" and form == "linear":
                            out["after"]["energy"][method] = r
                        t = time.time()
                        out["fits"][f"{excl}|{form}|{method}"] = fit_bundle(api, pid)
                        print(f"    fit {excl}|{form}|{method} {time.time() - t:.1f}s", flush=True)
            for n in range(len(SENSITIVITY_KEYS) + 1):
                for subset in itertools.combinations(SENSITIVITY_KEYS, n):
                    rec(j, sensitivity_decision(proposals, list(subset)))
                    out["slots"]["sensitivity"]["+".join(subset) or "none"] = \
                        j.view()["state"]["sensitivity"]

    j = S.run_inference(client, at)
    out["pid"] = j.pid
    return out


# ── prediction: the scenario's variant, watched ─────────────────────────────


def drive_prediction(base: str) -> dict[str, Any]:
    client = S.Client(base)
    out: dict[str, Any] = {"previews": {}}

    def at(moment: str, j: S.Journey) -> None:
        print(f"  prediction · {moment}", flush=True)
        if moment == "roles":
            out["unconfirmed"] = j.view()["state"]["roles_unconfirmed"]
        elif moment == "single:cycle_begin_year":
            # the block the three singles unlock (asked by the models' question): what it lists,
            # and the state it reads
            out["ask_block"] = next(s["ask"] for s in j.view()["interview"] if s.get("ask"))
            out["readings_state"] = j.view()["state"]
        elif moment == "exclusions":
            proposals = j.artifact("proposals")
            out["previews"]["exclusions"] = {
                "none": pv(j, {"kind": "set_exclusions", "rules": []}),
                "willett_2013_by_sex": pv(j, {"kind": "set_exclusions", "rules": [
                    rule_of(proposals, "willett_2013_by_sex")]})}
        elif moment == "missing":
            proposals = j.artifact("proposals")
            out["proposals"] = proposals
            out["missing_first"] = proposals["missing"]["methods"][0]["decision"]
            out["previews"]["missing"] = {
                k: pv(j, {"kind": "set_missing", "strategy": k})
                for k in ("impute", "complete_case")}
        elif moment == "split":
            out["seal_plan"] = j.artifact("seal_plan")
            out["previews"]["split"] = {
                str(o["holdout"]): pv(j, {"kind": "set_split", "holdout": o["holdout"],
                                          "seed": 0, "folds": 5})
                for o in out["seal_plan"]["options"]}
            view = j.view()
            out["view"] = {"steps": steps_of(view), "state": view["state"],
                           "summary": view["summary"]}
        elif moment == "energy":
            view = j.view()
            out["view_after_seal"] = {"steps": steps_of(view), "summary": view["summary"]}
            out["proposals_after_seal"] = j.artifact("proposals")
        elif moment == "models":
            out["shelf"] = j.artifact("shelf")
        elif moment == "fitted":
            out["fit"] = j.artifact("fit")
            out["readings"] = j.readings()
            out["decisions"] = j.view()["decisions"]
            out["methods"] = j.methods()

    j = S.run_prediction(client, at)
    out["pid"] = j.pid
    sentences: dict[str, Any] = {}
    for d in out["decisions"]:
        kind = d["decision"]["kind"]
        if kind in ("set_lens", "set_target", "set_purpose", "set_roles"):
            sentences[kind[4:]] = {"sentence": d["sentence"]}
        elif kind == "set_exclusions":
            sentences["exclusions_none"] = {"sentence": d["sentence"]}
        elif kind == "set_missing":
            sentences["missing"] = {"sentence": d["sentence"], "decision": d["decision"]}
        elif kind == "set_split":
            sentences["split_recorded"] = {"sentence": d["sentence"], "decision": d["decision"]}
        elif kind == "set_column_unit":
            sentences["unit_kcal"] = {"sentence": d["sentence"]}
        elif kind == "confirm_reading":
            sentences.setdefault("reading_single", {})[
                f"{d['decision']['reading']}:{d['decision']['column']}"] = {"sentence": d["sentence"]}
        elif kind == "confirm_readings":
            sentences["reading_block"] = {"items": d["decision"]["items"], "sentence": d["sentence"]}
    out["sentences"] = sentences
    return out


# ── in process: the engine's pure functions, checked against the server ─────


def block_sentences(items: list[dict[str, Any]], singles_from: int, state: Any) -> dict[str, Any]:
    """The block's sentence for whatever three or four of the first ``singles_from`` readings were
    confirmed singly: a bit mask of the readings it lists (over ``items``) → one of the distinct
    sentences, each the engine's own (``voice.sentence_for``) on ``state``."""
    from turbotab.core import voice

    sentences: list[str] = []
    index: dict[str, int] = {}
    by_mask: dict[str, int] = {}
    full = (1 << len(items)) - 1
    for k in (SINGLES, SINGLES + 1):
        for done in itertools.combinations(range(singles_from), k):
            mask = full
            for i in done:
                mask &= ~(1 << i)
            left = [items[i] for i in range(len(items)) if mask >> i & 1]
            text = voice.sentence_for({"kind": "confirm_readings", "items": left}, state)
            if text not in index:
                index[text] = len(sentences)
                sentences.append(text)
            by_mask[str(mask)] = index[text]
    return {"items": [f"{it['reading']}:{it['column']}" for it in items], "sentences": sentences,
            "by_mask": by_mask}


def in_process_prediction(main: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    """Prediction's readings, as its own drive asked them: the role readings (the inference card's
    order) then the four code-or-amount readings its one block also settles."""
    from turbotab.core import voice
    from turbotab.core.decisions import ProjectState

    state = ProjectState(**pred["readings_state"])
    block = pred["sentences"]["reading_block"]
    roles = role_reading_items(main)
    codes = [{k: it[k] for k in ("reading", "column", "value")} for it in block["items"]
             if it["reading"] == "code_or_count"]
    items = [*roles, *codes]
    single = {f"{it['reading']}:{it['column']}": voice.sentence_for(
        {"kind": "confirm_reading", **it}, state) for it in items}
    for key, r in pred["sentences"]["reading_single"].items():
        assert single[key] == r["sentence"], (single[key], r["sentence"])
    out = block_sentences(items, len(roles), state)
    # the scenario's block: the three singles' complement, the engine's recorded sentence
    said = voice.sentence_for({"kind": "confirm_readings", "items": block["items"]}, state)
    assert said == block["sentence"], (said, block["sentence"])
    mask = sum(1 << out["items"].index(f"{it['reading']}:{it['column']}") for it in block["items"])
    assert out["sentences"][out["by_mask"][str(mask)]] == block["sentence"]
    return {"single": single, "block": out, "codes": codes}


def in_process(main: dict[str, Any], branches: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    from turbotab.core import estimand, plan_lock, voice
    from turbotab.core.decisions import ProjectState

    Sm = main["sentences"]
    state = ProjectState(**main["readings_state"])
    role_items = role_reading_items(main)
    codes = [(g["columns"][0], g["guess"]) for g in main["ask_codes"]["groups"]]
    code_items = [{"reading": "code_or_count", "column": c, "value": v} for c, v in codes]
    codes_state = ProjectState(**main["models_state"])
    single: dict[str, str] = {}
    for it in role_items:
        single[f"role:{it['column']}"] = voice.sentence_for({"kind": "confirm_reading", **it}, state)
    for it in code_items:
        single[f"code_or_count:{it['column']}"] = voice.sentence_for(
            {"kind": "confirm_reading", **it}, codes_state)
    for key, r in Sm["reading_single"].items():
        assert single[key] == r["sentence"], (single[key], r["sentence"])
    # The role block lists exactly what is still unconfirmed, in the card's order. It unlocks after
    # three single confirmations; its sentence is taken for whatever three or four were done singly
    # (a bit mask of the role readings it lists → one of the distinct sentences).
    keys = [f"role:{it['column']}" for it in role_items]
    sentences: list[str] = []
    index: dict[str, int] = {}
    by_mask: dict[str, int] = {}
    full = (1 << len(role_items)) - 1
    for k in (SINGLES, SINGLES + 1):
        for done in itertools.combinations(range(len(role_items)), k):
            mask = full
            for i in done:
                mask &= ~(1 << i)
            left = [role_items[i] for i in range(len(role_items)) if mask >> i & 1]
            text = voice.sentence_for({"kind": "confirm_readings", "items": left}, state)
            if text not in index:
                index[text] = len(sentences)
                sentences.append(text)
            by_mask[str(mask)] = index[text]
    said = voice.sentence_for({"kind": "confirm_readings", "items": Sm["reading_block"]["items"]},
                              state)
    assert said == Sm["reading_block"]["sentence"], (said, Sm["reading_block"]["sentence"])
    said = voice.sentence_for({"kind": "confirm_readings", "items": Sm["reading_codes"]["items"]},
                              codes_state)
    assert said == Sm["reading_codes"]["sentence"], (said, Sm["reading_codes"]["sentence"])
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
                body = {"kind": "set_adjustment", "exposure": S.EXPOSURE, "answers": {
                    col: dict(zip(("causes_exposure", "causes_outcome", "after_exposure"), a))}}
                text = voice.sentence_for(body, est_state)
                if text not in at:
                    at[text] = len(texts)
                    texts.append(text)
                per_column[col][",".join(a)] = at[text]
    for u in Sm["adjustment"]["unguessed"]:
        assert voice.sentence_for(u["decision"], est_state) == u["sentence"]
    per_column_out = {"sentences": texts, "by_column": per_column}
    # the plan's SHA-256 for each combination the prototype can lock
    pre = main["pre_lock_view"]["state"]
    assert plan_lock.digest(plan_lock.plan_of(ProjectState(**pre))) == Sm["lock"]["digest"]
    slots = branches["slots"]
    forms = {"unset": pre["exposure_forms"], **slots["exposure_forms"]}
    guess = pre["model_sequence"]
    model_1 = {"guess": guess, "empty": {**guess, "model_1": []}, "unset": None}
    est_seq = S.model_sequence()
    model_1_empty = voice.sentence_for({**est_seq, "model_1": []}, est_state)
    lock = {}
    for excl, form, method, sens, m1 in itertools.product(
            slots["exclusions"], forms, slots["energy_adjustment"], slots["sensitivity"],
            model_1):
        st = dict(pre)
        st["exclusions"] = slots["exclusions"][excl]
        st["exposure_forms"] = forms[form]
        st["energy_adjustment"] = slots["energy_adjustment"][method]
        st["sensitivity"] = slots["sensitivity"][sens]
        st["model_sequence"] = model_1[m1]
        dg = plan_lock.digest(plan_lock.plan_of(ProjectState(**st)))
        lock[f"{excl}|{form}|{method}|{sens}|{m1}"] = dg[:12]
    canon = f"none|unset|standard|{'+'.join(SENSITIVITY_KEYS)}|guess"
    assert lock[canon] == Sm["lock"]["digest"][:12], (lock[canon], Sm["lock"]["digest"])
    # The lock's sentence differs only by its SHA-256: one template per digest, from the engine.
    sample = voice.sentence_for({"kind": "lock_plan", "digest": Sm["lock"]["digest"]},
                                ProjectState())
    assert sample == Sm["lock"]["sentence"]
    template = sample.replace(Sm["lock"]["digest"][:12], "{digest}")
    for dg in set(lock.values()):
        said = voice.sentence_for({"kind": "lock_plan", "digest": dg + "0" * 52}, ProjectState())
        assert template.replace("{digest}", dg) == said
    # (The prediction seal's other answer is previewed only: its sentence reads the recorder's
    # context, so it is not a pure function of the decision.)
    return {"single": single, "blocks": blocks, "derive": derive, "per_column": per_column_out,
            "lock": lock, "lock_template": template, "model_1_empty": model_1_empty,
            "prediction": in_process_prediction(main, pred)}


def role_reading_items(main: dict[str, Any]) -> list[dict[str, Any]]:
    """The role readings the card asks about, in its order (a family's columns in turn)."""
    out = []
    for g in main["ask_roles"]["groups"]:
        if g["kind"] != "role":
            continue
        for col in g["columns"]:
            out.append({"reading": "role", "column": col, "value": g["guess"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8973")
    ap.add_argument("--raw", default=str(Path(tempfile.gettempdir()) / "methods-map-raw"))
    ap.add_argument("--only", choices=["main", "branches", "prediction", "inproc", "trim"])
    args = ap.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

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
        teaching_path.write_text(json.dumps(httpx.get(f"{args.base}/api/teaching").json()))
    m = phase("main", lambda: drive_main(args.base))
    b = phase("branches", lambda: drive_branches(args.base, m))
    p = phase("prediction", lambda: drive_prediction(args.base))
    ip = phase("inproc", lambda: in_process(m, b, p))
    if args.only in (None, "trim"):
        from trim import trim  # noqa: PLC0415 - beside this file
        fixture = trim(json.loads(teaching_path.read_text()), m, b, p, ip)
        out = HERE / "fixture.json"
        out.write_text(json.dumps(fixture, separators=(",", ":"), ensure_ascii=False))
        print(f"fixture.json {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

"""Drive the real TurboTab server through the NHANES reference journey and keep what the
methods-questlog prototype shows (not part of the app; a review harness).

    TURBOTAB_HOME=<fresh> TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 venv/bin/python -m turbotab.server --port 8947
    venv/bin/python turbotab/frontend/src/explore/methods-questlog/capture_drive.py \
        --base http://127.0.0.1:8947 --out <raw dir> [--purpose inference|prediction]

Every answer comes from the fixture's declared truth (``truths.FIXTURE_TRUTHS``) through the same
helpers the acceptance drive uses (``server_drive``), never a constant. At each question it saves the
ProjectView, the methods text and the stage artifacts the prototype reads; at the energy question it
asks the server to preview every method; every refusal that only asks for readings is kept (the ask
card's real content). ``trim.py`` then cuts the raw dumps down to the prototype's fixture.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from turbotab.core.interview import QUESTION_KEYS  # noqa: E402
from turbotab.core.tests.acceptance.server_drive import (  # noqa: E402
    WP17_QUESTIONS, Drive, _post_when_reached, answer_wp17, settle_post)
from turbotab.core.tests.truths import ASKING, answers as truth_answers, fixture_truth  # noqa: E402
from turbotab.core.tests.truths import answer_adjustment  # noqa: E402

NHANES = Path("/Users/nhedglin/tabular-ml-lab/_tt_tmp_nhanes.csv")
ORDER = ["lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind",
         "unit", "aggregation", "temporal", "roles", "survey", "exclusions", "missing", "split",
         "energy_adjustment", "models"]
NUTRIENTS = ["sugar", "protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]


class Client:
    """The TestClient surface ``server_drive`` uses, over HTTP to a running server."""

    def __init__(self, base: str):
        self.h = httpx.Client(base_url=base, timeout=600)

    def get(self, url: str) -> httpx.Response:
        return self.h.get(url)

    def post(self, url: str, json: Any = None) -> httpx.Response:  # noqa: A002
        return self.h.post(url, json=json)


def plan_for(purpose: str) -> dict[str, dict[str, Any]]:
    out = {"lens": {"kind": "set_lens", "lenses": ["dietary"]},
           "target": {"kind": "set_target", "column": "glucose"},
           "task": {"kind": "set_task", "column": "glucose", "task": "regression"},
           "purpose": {"kind": "set_purpose", "purpose": purpose},
           "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
           "temporal": {"kind": "set_temporal", "temporal": False},
           "survey": {"kind": "set_survey", "estimand": "sample"},
           "exclusions": {"kind": "set_exclusions", "rules": []},
           "missing": {"kind": "set_missing", "strategy": "complete_case"},
           "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
           "energy_adjustment": {"kind": "set_energy_adjustment", "method": "none"},
           "models": {"kind": "select_models", "models": ["linear"]},
           "unit": {"kind": "set_unit", "unit": "unit"},
           "aggregation": {"kind": "set_aggregation", "method": "mean"}}
    if purpose == "inference":
        out["energy_adjustment"] = {"kind": "set_energy_adjustment", "method": "standard",
                                    "energy_column": "kcal", "nutrients": NUTRIENTS}
    else:
        out["models"] = {"kind": "select_models", "models": ["linear", "boosted_trees"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8947")
    ap.add_argument("--out", required=True)
    ap.add_argument("--purpose", default="inference")
    args = ap.parse_args()
    out = Path(args.out) / args.purpose
    out.mkdir(parents=True, exist_ok=True)
    c = Client(args.base)
    truth = fixture_truth(NHANES.name)
    plan = plan_for(args.purpose)
    log: list[dict[str, Any]] = []

    def save(name: str, obj: Any) -> None:
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))

    r = c.post("/api/projects", json={"path": str(NHANES)})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    drive = Drive(c, pid, truth)
    drive.artifact("ingest")
    save("teaching", c.get("/api/teaching").json())
    url = f"/api/projects/{pid}/decisions"

    def snap(label: str, stages: tuple[str, ...] = ()) -> None:
        view = drive.view()
        got: dict[str, Any] = {"view": view,
                               "methods": c.get(f"/api/projects/{pid}/methods").json(),
                               "readings": c.get(f"/api/projects/{pid}/readings").json()}
        for st in ("ingest", "oriented", "working", *stages):
            status = view["stages"].get(st, {}).get("status")
            if status == "fresh":
                got[st] = c.get(f"/api/projects/{pid}/stages/{st}").json()["artifact"]
        save(f"snap_{len(log):02d}_{label}", got)
        log.append({"snap": label, "at": time.time()})

    def settled(timeout: float = 600.0) -> None:
        """Wait until no stage a preview reads is computing (a preview answers on fresh artifacts
        only; mid-recompute it says nothing can be shown yet, which is true but not the picture)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            stages = drive.view()["stages"]
            busy = [s for s in ("working", "cohort", "split", "design", "roles", "proposals")
                    if stages.get(s, {}).get("status") in ("queued", "running")]
            if not busy:
                return
            time.sleep(0.1)

    def preview(name: str, decision: dict[str, Any]) -> None:
        settled()
        r = c.post(f"/api/projects/{pid}/preview", json=decision)
        save(f"preview_{name}", {"status": r.status_code, "decision": decision, "body": r.json()})

    for key in [k for k in QUESTION_KEYS if k in ORDER or k in WP17_QUESTIONS]:
        if key in ("event", "task"):
            drive.artifact("target_info", timeout=600)
        step = drive.reach(key, timeout=600)
        if step["status"] not in ("open", "waiting"):
            continue
        snap(f"before_{key}", ("roles", "proposals", "target_info", "findings", "cohort", "structure"))
        if key == "adjustment":
            # The card as the user sees it, then the author's answers (one tap per group the pack
            # guesses alike).
            card = drive.artifact("proposals").get("adjustment")
            save("adjustment_card", card)
            for g in card["groups"]:
                if g.get("decision"):
                    preview(f"adjust_{g['key']}", g["decision"])
            posted = answer_adjustment(lambda d: c.post(url, json=d), card, truth)
            save("adjustment_answers", posted)
            continue
        if key == "estimand":
            save("estimand_card", drive.artifact("proposals").get("estimand"))
            base = {"kind": "set_estimand", "effect": "total", "measure": "mean_difference"}
            preview("estimand_sugar_substitution", base | {"exposure": "sugar", "contrast": "substitution"})
            preview("estimand_sugar_addition", base | {"exposure": "sugar", "contrast": "addition"})
            preview("estimand_protein_substitution", base | {"exposure": "protein", "contrast": "substitution"})
        if key in WP17_QUESTIONS:
            answer_wp17(drive, key)
            continue
        body = plan.get(key)
        if key == "models" and args.purpose == "inference":
            # The declared secondaries (MODELING_SEQUENCE §1 row 11): the field's Model 1, and the
            # primary on the plausible energy reporters only, both declared before any estimate.
            declared = (
                ("model_sequence", {"kind": "set_model_sequence", "exposure": "sugar",
                                    "model_1": ["age", "gender", "kcal"]}),
                ("sensitivity", {"kind": "set_sensitivity", "analyses": [{
                    "label": "Plausible energy reporters only",
                    "rules": [{"kind": "range", "column": "kcal", "by": {
                        "column": "gender", "ranges": {"female": [500, 3500], "male": [800, 4000]}},
                        "reason": "Willett's plausible range: 500 to 3,500 kcal for women, "
                                  "800 to 4,000 kcal for men"}]}]}))
            for name, d in declared:
                rr = c.post(url, json=d)
                save(f"declared_{name}", {"status": rr.status_code, "decision": d,
                                          "body": rr.json() if rr.status_code != 200 else None})
                if rr.status_code == 409 and rr.json()["error"]["code"] in ASKING:
                    # The screen's bounds read the energy column's unit and days: asked, and
                    # answered from the fixture's truth (one day's intake, in kcal).
                    save(f"ask_{name}_0", {"body": d, "error": rr.json()["error"]})
                    rr = settle_post(c, pid, d, truth)
                    assert rr.status_code == 200, (name, rr.text[:600])
        if key == "roles":
            proposals = drive.artifact("roles")["columns"]
            save("roles_proposals", drive.artifact("roles"))
            body = {"kind": "set_roles", "roles": {p["column"]: p["proposed"] for p in proposals}}
            for column, role in body["roles"].items():
                truth.setdefault(f"role:{column}", role)
        if key == "energy_adjustment":
            for m in ("standard", "residual", "residual_energy_dropped", "density",
                      "density_multivariate", "partition", "all_components", "none"):
                d = {"kind": "set_energy_adjustment", "method": m}
                if m != "none":
                    d |= {"energy_column": "kcal", "nutrients": NUTRIENTS}
                preview(f"energy_{m}", d)

        def reopened(body: dict = body) -> bool:
            return drive.answer_wp17_before(body)

        if key == "roles":
            preview("roles", body)
        r = _post_when_reached(c, url, body, unblock=reopened)
        n_ask = 0
        while r.status_code == 409 and r.json()["error"]["code"] in ASKING:
            save(f"ask_{key}_{n_ask}", {"body": body, "error": r.json()["error"]})
            if n_ask == 0 and key == "models":
                # BLUEPRINT §11.4 rule 4, as a user would play it: three readings confirmed one at
                # a time (each from the fixture's truth), then the ask again, whose block confirm
                # lists exactly what is left.
                exits = r.json()["error"]["exits"]
                singles = [e["decision"] for e in exits
                           if (e["decision"] or {}).get("kind") == "confirm_reading"
                           and e["decision"]["reading"] == "role"
                           and e["decision"]["value"] == truth.answer("role", e["decision"]["column"])][:3]
                for i, d in enumerate(singles):
                    preview(f"single_{i}", d)
                    rr = c.post(url, json=d)
                    assert rr.status_code == 200, (d, rr.text[:600])
                    save(f"single_{i}", {"decision": d, "record": rr.json()["decisions"][-1]})
                n_ask += 1
                r = _post_when_reached(c, url, body, unblock=reopened)
                continue
            n_ask += 1
            for decision in truth_answers(r.json()["error"], truth):
                rr = c.post(url, json=decision)
                assert rr.status_code == 200, (decision, rr.text[:600])
            r = _post_when_reached(c, url, body, unblock=reopened)
        if r.status_code != 200:
            save(f"refused_{key}", {"body": body, "status": r.status_code, "error": r.json()})
        assert r.status_code == 200, (key, r.text[:800])
        if key == "purpose":
            snap("after_purpose", ("roles", "proposals", "target_info", "findings"))

    fit = drive.artifact("fit", timeout=1200)
    save("fit", fit)
    if args.purpose == "inference":
        for st in ("effects", "secondary", "sensitivity"):
            try:
                save(f"stage_{st}", drive.artifact(st, timeout=900))
            except AssertionError as e:
                save(f"stage_{st}", {"error": str(e)})
    snap("after_fit", ("proposals", "cohort", "split", "design", "shelf", "findings", "substitution"))
    if args.purpose == "inference":
        r = c.get(f"/api/projects/{pid}/plan")
        save("plan", {"status": r.status_code, "body": r.json() if r.status_code == 200 else r.text})
    save("log", log)
    print("done", pid)


if __name__ == "__main__":
    main()

"""Drive the real TurboTab server through the shared scenario and keep what the methods-questlog
prototype shows at each moment (not part of the app; a review harness).

    TURBOTAB_HOME=<fresh> TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 \
        venv/bin/python -m turbotab.server --port 8972                          # repo root
    venv/bin/python turbotab/frontend/src/explore/methods-questlog/capture_drive.py \
        --base http://127.0.0.1:8972 --out <raw dir>
    venv/bin/python turbotab/frontend/src/explore/methods-questlog/trim.py <raw dir>

The path is the one all three prototypes share (``methods-shared/scenario.py``, SCENARIO.md): the
scenario records every answer; this script only watches. At each moment it saves the ProjectView,
the methods text, the readings and the stage artifacts the banner and the canvas read (no estimate
stage before the plan is locked: fetching one is what records the lock), and the cards, previews
and finding evidence of the slot that moment opens. A preview records nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "methods-shared"))

import scenario as S  # noqa: E402

BANNER = ("ingest", "oriented", "working", "cohort", "split", "design", "fit", "shelf")
CARDS = ("roles", "proposals", "findings", "seal_plan")
ENERGY_METHODS = ("standard", "residual", "residual_energy_dropped", "density_multivariate",
                  "density", "partition", "all_components", "none")
EXCLUSION_KEYS = ("willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000",
                  "sex_neutral_500_3500", "goldberg_schofield")


class Watch:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.order: list[str] = []

    def save(self, name: str, obj: Any) -> None:
        (self.out / f"{name}.json").write_text(json.dumps(obj, default=str))

    def snap(self, moment: str, j: S.Journey, **extra: Any) -> dict[str, Any]:
        j.wait_quiet()
        view = j.view()
        stages: dict[str, Any] = {}
        for st in (*BANNER, *CARDS):
            status = (view["stages"].get(st) or {}).get("status")
            if status in (None, "idle", "blocked"):
                continue
            got = j.stage(st)  # None for an estimate stage before the lock
            if got and got.get("artifact") is not None:
                stages[st] = got
        snap = {"moment": moment, "view": view, "methods": j.methods(), "readings": j.readings(),
                "stages": stages, **extra}
        name = moment.replace(":", "_")
        self.save(f"snap_{name}", snap)
        self.order.append(name)
        first = next((s for s in view["interview"] if s["status"] in ("open", "waiting")), None)
        print(f"  {moment:24s} {len(view['decisions']):3d} decisions · open "
              f"{first and first['key']}{' (ask)' if first and first.get('ask') else ''}",
              flush=True)
        return snap

    def preview(self, name: str, j: S.Journey, decision: dict[str, Any]) -> None:
        self.save(f"preview_{name}", j.preview(decision))


def evidence_of(j: S.Journey, columns: set[str]) -> dict[str, Any]:
    """The findings the readings canvas shows: each finding about a column the card asks of."""
    got = j.stage("findings")
    findings = (got or {}).get("artifact") or {}
    out: dict[str, Any] = {}
    for f in findings.get("findings", []):
        fid = f.get("id") or ""
        cols = set(f.get("affected_columns") or [])
        # A finding about the asked column itself (a flag names its base column beside it), not
        # one that sweeps many columns at once.
        if not (cols & columns) or len(cols) > 2:
            continue
        r = j.c.get(f"/api/projects/{j.pid}/findings/{fid}/evidence")
        if r.status_code == 200:
            out[fid] = {"finding": f, "evidence": r.json()}
    return out


def inference(base: str, out: Path) -> None:
    w = Watch(out)
    client = S.Client(base)
    w.save("teaching", client.get("/api/teaching").json())

    def at(moment: str, j: S.Journey) -> None:
        if moment == "draft":
            snap = w.snap(moment, j)
            roles = snap["stages"]["roles"]["artifact"]
            w.save("roles_proposals", roles)
            w.preview("roles", j, {"kind": "set_roles",
                                   "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
        elif moment == "roles":
            w.snap(moment, j)
        elif moment == "unit_ask":
            ask = j.ask("exclusions")
            for e in (ask or {}).get("exits", []):
                d = e.get("decision") or {}
                if d.get("kind") == "set_column_unit":
                    w.preview(f"unit_{d.get('unit')}_{d.get('days')}", j, d)
        elif moment == "exclusions":
            snap = w.snap(moment, j)
            proposals = snap["stages"]["proposals"]["artifact"]
            w.preview("exclusions_none", j, {"kind": "set_exclusions", "rules": []})
            for key in EXCLUSION_KEYS:
                if any(o["key"] == key for o in proposals["exclusions"]):
                    w.preview(f"exclusions_{key}", j, {"kind": "set_exclusions",
                                                       "rules": [S.screen_rule(proposals, key)]})
        elif moment == "missing":
            snap = w.snap(moment, j)
            card = (snap["stages"]["proposals"]["artifact"].get("missing") or {})
            for m in card.get("methods", []):
                d = m.get("decision") or {}
                if d:
                    w.preview(f"missing_{d.get('strategy')}", j, {"kind": "set_missing", **d})
        elif moment == "split":
            snap = w.snap(moment, j)
            plan = (snap["stages"].get("seal_plan") or {}).get("artifact") or {}
            for o in plan.get("options", []):
                w.preview(f"split_{o['holdout']}", j, {"kind": "set_split", "holdout": o["holdout"],
                                                       "seed": 0, "folds": 5})
        elif moment == "readings":
            w.snap(moment, j)
            ask = j.ask("estimand") or {}
            cols = {c for g in ask.get("groups", []) for c in g["columns"]}
            w.save("evidence_readings", evidence_of(j, cols))
        elif moment.startswith("single:"):
            w.snap(moment, j)
        elif moment == "estimand":
            w.snap(moment, j)
            for exp, contrast in (("sugar", "substitution"), ("sugar", "addition"),
                                  ("protein", "substitution")):
                w.preview(f"estimand_{exp}_{contrast}", j, S.estimand(exp, contrast))
        elif moment == "adjustment":
            snap = w.snap(moment, j)
            card = snap["stages"]["proposals"]["artifact"]["adjustment"]
            # Each answer the scenario will record for this card (the group's one tap where the
            # pack guesses, else the truth's per-column answers), previewed, never recorded.
            n = [0]

            def preview_only(d: dict[str, Any]) -> Any:
                key = next((g["key"] for g in card["groups"] if g.get("decision") == d),
                           f"answers_{n[0]}")
                n[0] += 1
                got = j.preview(d)
                w.save(f"preview_adjust_{key}", got)
                return type("R", (), {"status_code": got["status"], "text": json.dumps(got)})()

            from turbotab.core.tests.truths import answer_adjustment

            w.save("adjustment_answers", answer_adjustment(preview_only, card, j.truth))
        elif moment == "energy":
            w.snap(moment, j)
            for m in ENERGY_METHODS:
                w.preview(f"energy_{m}", j, S.energy(m))
        elif moment == "model_sequence":
            w.snap(moment, j)
            w.preview("model_sequence", j, S.model_sequence())
        elif moment == "models":
            snap = w.snap(moment, j)
            ask = j.ask("models") or {}
            cols = {c for g in ask.get("groups", []) for c in g["columns"]}
            w.save("evidence_codes", evidence_of(j, cols))
            w.preview("models_linear", j, {"kind": "select_models",
                                           "models": list(S.INFERENCE_MODELS)})
            if "shelf" not in snap["stages"]:
                got = j.stage("shelf")
                if got:
                    w.save("shelf", got)
        elif moment == "ready":
            w.snap(moment, j)
        elif moment == "locked":
            w.snap(moment, j)
            for st in ("fit", "effects", "secondary", "sensitivity"):
                w.save(f"stage_{st}", j.artifact(st))
            w.save("plan", j.get("/plan"))

    print("inference", flush=True)
    j = S.run_inference(client, at)
    w.save("order", w.order)
    w.save("pid", j.pid)


def prediction(base: str, out: Path) -> None:
    w = Watch(out)
    client = S.Client(base)

    def at(moment: str, j: S.Journey) -> None:
        if moment == "fitted":
            w.snap(moment, j)
            w.save("fit", j.artifact("fit"))

    print("prediction", flush=True)
    j = S.run_prediction(client, at)
    w.save("order", w.order)
    w.save("pid", j.pid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8972")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["inference", "prediction"])
    args = ap.parse_args()
    out = Path(args.out)
    if args.only in (None, "inference"):
        inference(args.base, out / "inference")
    if args.only in (None, "prediction"):
        prediction(args.base, out / "prediction")
    print("done", flush=True)


if __name__ == "__main__":
    main()

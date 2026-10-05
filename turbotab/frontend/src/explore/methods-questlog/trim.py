"""Cut the raw dumps of ``capture_drive.py`` down to the methods-questlog prototype's fixture.

    venv/bin/python turbotab/frontend/src/explore/methods-questlog/trim.py <raw dir>

Nothing is written by hand: every sentence, guess, piece of evidence and number in fixture.json is
the server's own, from the two drives (inference and prediction) on the NHANES export. Floats are
rounded to six significant digits and the preview scatters keep the server's own sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def rnd(o: Any) -> Any:
    if isinstance(o, float):
        return float(f"{o:.6g}")
    if isinstance(o, list):
        return [rnd(x) for x in o]
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    return o


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def view_of(snap: dict[str, Any]) -> dict[str, Any]:
    v = snap["view"]
    return {
        "summary": v["summary"],
        "state": v["state"],
        "stages": v["stages"],
        "interview": v["interview"],
        "decisions": [{k: d[k] for k in ("id", "seq", "at", "sentence", "post_seal", "after_estimates",
                                         "decision", "note")} for d in v["decisions"]],
    }


BANNER = ("ingest", "oriented", "working", "cohort", "split", "design", "fit", "shelf")


def stages_of(snap: dict[str, Any], fit: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    statuses = snap["view"]["stages"]
    for st in BANNER:
        art = fit if st == "fit" and fit is not None else snap.get(st)
        if art is None:
            continue
        s = statuses.get(st, {})
        out[st] = {"stage": st, "key": s.get("key"), "fresh": True, "status": s.get("status", "fresh"),
                   "artifact": art}
    return out


def moment(snap: dict[str, Any], fit: Any = None) -> dict[str, Any]:
    return {"view": view_of(snap), "methods": snap["methods"], "stages": stages_of(snap, fit)}


def main(raw: Path) -> None:
    inf, pred = raw / "inference", raw / "prediction"
    snaps = {p.stem.split("_", 2)[2]: load(p) for p in sorted(inf.glob("snap_*.json"))}
    psnaps = {p.stem.split("_", 2)[2]: load(p) for p in sorted(pred.glob("snap_*.json"))}
    fit = load(inf / "fit.json")
    pfit = load(pred / "fit.json")

    previews = {}
    for p in sorted(inf.glob("preview_*.json")):
        name = p.stem[len("preview_"):]
        got = load(p)
        previews[name] = {"status": got["status"], "decision": got["decision"], "body": got["body"]}
    evidence = {p.stem[len("evidence_"):]: load(p) for p in sorted(inf.glob("evidence_*.json"))}

    teaching = [{k: e.get(k) for k in ("key", "title", "question", "one_liner", "why", "consumer", "options",
                                       "terms", "drawer")} for e in load(inf / "teaching.json")]

    sensitivity = load(inf / "stage_sensitivity.json")
    sens_fits = []
    for fam in sensitivity["families"]:
        for f in fam["fits"]:
            sugar = next(c for c in f["coefficients"] if c["feature"] == "sugar")
            sens_fits.append({"family": fam["family"], "label": f["label"], "n_rows": f["n_rows"],
                              "coefficient": sugar})
    effects = load(inf / "stage_effects.json")
    for fam in effects["families"]:
        for s in fam["sequence"]:
            s.pop("inference", None)
    plan = load(inf / "plan.json")["body"]

    fixture = {
        "meta": {
            "source": "_tt_tmp_nhanes.csv (the real NHANES export), driven through the real server "
                      "(turbotab.server, 2 workers, fresh TURBOTAB_HOME) by capture_drive.py",
            "generated": "2026-10-05",
            "answers": "every answer from the fixture's declared truth (truths.FIXTURE_TRUTHS)",
        },
        "teaching": teaching,
        "inference": {
            "moments": {
                "m1": moment(snaps["before_roles"]),
                "m2": moment(snaps["before_exclusions"]),
                "m3": moment(snaps["before_estimand"]),
                "m4": moment(snaps["before_adjustment"]),
                "m5": moment(snaps["before_models"]),
                "m6": moment(snaps["after_fit"], fit),
            },
            "roles": load(inf / "roles_proposals.json"),
            "readings": snaps["before_exclusions"]["readings"],
            "asks": {
                "models_0": load(inf / "ask_models_0.json")["error"],
                "models_1": load(inf / "ask_models_1.json")["error"],
                "sensitivity_0": load(inf / "ask_sensitivity_0.json")["error"],
            },
            "singles": [load(inf / f"single_{i}.json") for i in range(3)],
            "estimand_card": load(inf / "estimand_card.json"),
            "adjustment_card": load(inf / "adjustment_card.json"),
            "previews": previews,
            "evidence": evidence,
            "fit": fit,
            "effects": effects,
            "secondary_methods": load(inf / "stage_secondary.json")["methods"],
            "sensitivity": {"analyses": sensitivity["analyses"], "fits": sens_fits,
                            "methods": sensitivity["methods"]},
            "plan": {k: plan[k] for k in ("declared_at", "plan_sha256", "sha256", "status", "text",
                                          "through_record")},
        },
        "prediction": {
            "moment": moment(psnaps["after_fit"], pfit),
            "fit": pfit,
        },
    }
    out = HERE / "fixture.json"
    out.write_text(json.dumps(rnd(fixture), separators=(",", ":")))
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

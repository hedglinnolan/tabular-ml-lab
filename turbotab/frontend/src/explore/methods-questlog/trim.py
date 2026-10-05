"""Cut the raw dumps of ``capture_drive.py`` down to the methods-questlog prototype's fixture.

    venv/bin/python turbotab/frontend/src/explore/methods-questlog/trim.py <raw dir>

Nothing is written by hand: every sentence, guess, piece of evidence and number in fixture.json is
the server's own, from the shared scenario's two drives (inference and prediction) on the NHANES
export. Fields the prototype never reads are dropped, artifacts that repeat across moments are
stored once, and floats are rounded to six significant digits.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BANNER = ("ingest", "oriented", "working", "cohort", "split", "design", "fit", "shelf")
STATUS_KEEP = (*BANNER, "substitution")
TEACHING = ("purpose", "lens", "target", "roles", "exclusions", "missing", "split", "estimand",
            "adjustment", "energy_adjustment", "models", "substitution", "survey")


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


def pick(d: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    return None if d is None else {k: d.get(k) for k in keys if k in d}


def slim_ask(ask: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if ask is None else {k: v for k, v in ask.items() if k != "read_from_data"}


def slim_view(v: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": v["summary"],
        "state": v["state"],
        "stages": {k: pick(s, ("stage", "status", "key", "fresh", "error", "cancelled"))
                   for k, s in v["stages"].items() if k in STATUS_KEEP},
        "interview": [{**{k: s.get(k) for k in ("key", "status", "decision_id", "reason",
                                                 "waiting_on")}, "ask": slim_ask(s.get("ask"))}
                      for s in v["interview"]],
    }


def slim_artifact(stage: str, a: dict[str, Any]) -> dict[str, Any]:
    """What the banner and the canvas read of each stage artifact."""
    if stage == "ingest":
        return pick(a, ("n_rows", "n_cols"))
    if stage == "oriented":
        return pick(a, ("n_rows", "n_cols", "transposed"))
    if stage == "working":
        return pick(a, ("n_rows", "n_source_rows"))
    if stage == "cohort":
        return pick(a, ("steps", "n_final", "predictors", "n_base"))
    if stage == "design":
        lineage = a.get("lineage") or {}
        return {"lineage": {"nodes": [pick(n, ("id", "lane", "count", "column"))
                                      for n in lineage.get("nodes", [])], "links": []},
                "matrix": pick(a.get("matrix") or {}, ("n_cols",))}
    return a


class Store:
    def __init__(self) -> None:
        self.artifacts: dict[str, Any] = {}

    def put(self, stage: str, result: dict[str, Any]) -> str:
        slim = {**pick(result, ("stage", "key", "fresh", "status")),
                "artifact": slim_artifact(stage, result["artifact"])}
        blob = json.dumps(slim, sort_keys=True)
        ref = hashlib.sha1(blob.encode()).hexdigest()[:10]
        self.artifacts.setdefault(ref, slim)
        return ref


def moment(snap: dict[str, Any], store: Store) -> dict[str, Any]:
    stages = {st: store.put(st, snap["stages"][st]) for st in BANNER if st in snap["stages"]}
    lines = [pick(l, ("record_id", "seq", "kind", "sentence", "in_force", "post_seal",
                      "after_estimates")) for l in snap["methods"]["lines"]]
    return {"view": slim_view(snap["view"]), "methods": {"lines": lines}, "stages": stages}


def sugar_only(coefs: list[dict[str, Any]], exposure: str) -> list[dict[str, Any]]:
    return [c for c in coefs if c["feature"] == exposure]


def main(raw: Path) -> None:
    inf, pred = raw / "inference", raw / "prediction"
    order = load(inf / "order.json")
    snaps = {name: load(inf / f"snap_{name}.json") for name in order}
    store = Store()
    moments = {name: moment(s, store) for name, s in snaps.items()}

    def proposals(name: str) -> dict[str, Any]:
        return snaps[name]["stages"]["proposals"]["artifact"]

    previews = {}
    for p in sorted(inf.glob("preview_*.json")):
        got = load(p)
        previews[p.stem[len("preview_"):]] = {"status": got["status"], "decision": got["decision"],
                                              "body": got["body"]}
    evidence = {}
    for name in ("evidence_readings", "evidence_codes"):
        for fid, e in load(inf / f"{name}.json").items():
            evidence[fid] = {"columns": e["finding"]["affected_columns"],
                             "summary": e["finding"]["summary"], "evidence": e["evidence"]}

    teaching = [{k: e.get(k) for k in ("key", "title", "question", "one_liner", "why", "consumer",
                                       "options", "terms", "drawer")}
                for e in load(inf / "teaching.json") if e["key"] in TEACHING]

    exclusions = proposals("exclusions")
    split_plan = snaps["split"]["stages"]["seal_plan"]["artifact"]
    energy = proposals("energy")
    shelf = (snaps["models"]["stages"].get("shelf") or {}).get("artifact")

    fit = load(inf / "stage_fit.json")
    effects = load(inf / "stage_effects.json")
    exposure = effects["exposure"]
    for fam in effects["families"]:
        for s in fam["sequence"]:
            s.pop("inference", None)
            s["effects"] = sugar_only(s["effects"], exposure)
    sensitivity = load(inf / "stage_sensitivity.json")
    for fam in sensitivity["families"]:
        for f in fam["fits"]:
            f["coefficients"] = sugar_only(f["coefficients"], exposure)
            f.pop("inference", None)
    secondary = load(inf / "stage_secondary.json")
    plan = load(inf / "plan.json")
    m0 = fit["models"][0]

    pstore = Store()
    psnap = load(pred / "snap_fitted.json")
    pfit = load(pred / "fit.json")

    fixture = {
        "meta": {
            "source": "_tt_tmp_nhanes.csv (the real NHANES export), driven through the real server "
                      "(turbotab.server, 2 workers, fresh TURBOTAB_HOME) along the shared scenario "
                      "(methods-shared/scenario.py) by capture_drive.py",
            "answers": "the scenario's; every reading from the fixture's declared truth "
                       "(truths.FIXTURE_TRUTHS)",
        },
        "teaching": teaching,
        "inference": {
            "order": order,
            "moments": moments,
            "artifacts": store.artifacts,
            "roles": load(inf / "roles_proposals.json"),
            "cards": {
                "exclusions": {"offered": [pick(o, ("key", "label", "affected", "evidence",
                                                    "refused")) for o in exclusions["exclusions"]],
                               "labels": exclusions["labels"]["exclusions"]},
                "missing": {"card": proposals("missing")["missing"],
                            "labels": proposals("missing")["labels"].get("missing")},
                "seal_plan": pick(split_plan, ("options", "reason", "basis", "cv_first", "floor")),
                "estimand": proposals("estimand")["estimand"],
                "adjustment": proposals("adjustment")["adjustment"],
                # The scenario's answers to that card, in the order it records them.
                "adjustment_answers": load(inf / "adjustment_answers.json"),
                "energy": {"card": pick(energy["energy"], ("energy_column", "nutrients",
                                                           "applicability", "usual", "ranking")),
                           "labels": energy["labels"]["energy_adjustment"]},
                "model_sequence": proposals("model_sequence")["model_sequence"],
                "shelf": shelf,
            },
            "previews": previews,
            "evidence": evidence,
            "fit": {"n_train": fit["n_train"], "models": [{
                "family": m0["family"], "label": m0["label"],
                "adjustment_terms": m0.get("adjustment_terms") or [],
                "inference": pick(m0.get("inference") or {}, ("caption",)),
                "concerns": m0.get("concerns") or []}]},
            "effects": {k: effects[k] for k in ("exposure", "appendix_title", "rows",
                                                "measure_label", "families", "model_1")},
            "secondary_methods": secondary.get("methods"),
            "sensitivity": {k: sensitivity[k] for k in ("analyses", "families", "methods")
                            if k in sensitivity},
            "plan": pick(plan, ("declared_at", "plan_sha256", "sha256", "status",
                                "through_record")),
        },
        "prediction": {
            "moment": moment(psnap, pstore),
            "artifacts": pstore.artifacts,
            "fit": pfit,
        },
    }
    out = HERE / "fixture.json"
    out.write_text(json.dumps(rnd(fixture), separators=(",", ":"), ensure_ascii=False))
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

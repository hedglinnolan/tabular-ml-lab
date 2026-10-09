"""Trim the raw captures (drive.py) into the prototype's fixture.json (not part of the app).

    venv/bin/python turbotab/frontend/src/explore/methods-document/capture/trim.py <raw-out-dir>

Keeps, per moment of the scenario's path, the ProjectView, the readings card, the previews, and
the stage artifacts the banner, the production stage and the document read. Anything the same
across moments (an artifact, a decision record, a view's parts, a preview) is stored once and
referenced. Nothing is edited: every value is the server's own, with two compositions recorded in
the fixture rather than hidden:

* ``codes`` sits between the scenario's moments "models" and "ready": the scenario records the fit's
  code-or-amount block and the model family back to back. It is the "models" moment with that
  block's record (the server's, from "ready") appended and the fit's ask card, which the block
  answers, gone. Its stages are the "models" moment's.
* the moments after the energy model is recorded show the energy alternatives' previews the server
  computed at the energy question (moment "energy"), because the server draws a method's storyboard
  from the data as loaded and the standard model leaves every nutrient's values as loaded.

A record keeps the sentence it was recorded with; the methods text (GET /methods) re-renders a
record's sentence when a later answer changes what it says (complete cases: "all `21,849` rows
remain" until the readings make more columns predictors, then "`2,996` of `21,849` rows remain").
Each moment keeps, as ``methods``, the methods text's sentence for every record it says differently
from the record, and the prototype prints that one (fixture.ts), as the engine's methods text does.

The scenario's moment "unit_ask" is not kept twice: it is the "roles" moment's view (the screens'
unit question already open on it, the same records). Preview payloads (scatters and storyboards)
are rounded to six significant digits; the estimate stages are kept exactly as served.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
KEEP_STAGES = ("ingest", "oriented", "working", "cohort", "split", "design", "fit", "shelf",
               "roles", "proposals", "effects", "secondary", "sensitivity", "substitution")
# The production stage and banner fetch these when the view says they exist; each one the view
# reports computed must be in the fixture, or the prototype would ask a server for it.
READ_BY_STAGE_OR_BANNER = ("ingest", "oriented", "working", "cohort", "split", "design", "shelf",
                           "substitution")
# The walk, in the scenario's order: (moment id, raw capture).
PATH = [
    ("draft", "inf-draft"),
    ("roles", "inf-roles"),
    ("exclusions", "inf-exclusions"),
    ("missing", "inf-missing"),
    ("split", "inf-split"),
    ("readings", "inf-readings"),
    ("single-bp_di", "inf-single-bp_di"),
    ("single-bp_sys", "inf-single-bp_sys"),
    ("single-cycle_begin_year", "inf-single-cycle_begin_year"),
    ("estimand", "inf-estimand"),
    ("adjustment", "inf-adjustment"),
    ("energy", "inf-energy"),
    ("model_sequence", "inf-model_sequence"),
    ("models", "inf-models"),
    ("codes", None),
    ("ready", "inf-ready"),
    ("locked", "inf-locked"),
]
ENERGY_AFTER = ("model_sequence", "models", "codes", "ready", "locked")
TEACHING = ("target", "purpose", "roles", "exclusions", "missing", "split", "estimand",
            "adjustment", "energy_adjustment", "models", "task", "grain", "substitution",
            "open_seal", "clusters", "survey", "causal")


def rnd(o: Any) -> Any:
    if isinstance(o, float):
        return float(f"{o:.6g}")
    if isinstance(o, list):
        return [rnd(x) for x in o]
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    return o


class Pool:
    def __init__(self) -> None:
        self.items: dict[str, Any] = {}

    def put(self, obj: Any) -> str:
        blob = json.dumps(obj, sort_keys=True)
        ref = hashlib.sha1(blob.encode()).hexdigest()[:12]
        self.items.setdefault(ref, obj)
        return ref


def composed_codes(models: dict[str, Any], ready: dict[str, Any]) -> dict[str, Any]:
    snap = json.loads(json.dumps(models))
    block = next(d for d in ready["view"]["decisions"]
                 if d["decision"]["kind"] == "confirm_readings"
                 and all(i["reading"] == "code_or_count" for i in d["decision"]["items"]))
    snap["view"]["decisions"].append(block)
    for step in snap["view"]["interview"]:
        if step["key"] == "models":
            step["ask"] = None
    snap["moment"] = "codes"
    return snap


def main() -> None:
    raw = Path(sys.argv[1])
    load = lambda name: json.loads((raw / f"{name}.json").read_text())  # noqa: E731
    pool, decisions = Pool(), {}
    snaps: dict[str, dict[str, Any]] = {}
    for mid, source in PATH:
        snaps[mid] = composed_codes(snaps["models"], load("inf-ready")) if source is None else load(source)
    snaps["prediction"] = load("pred-fitted")
    energy_previews = {k: pool.put(rnd(v)) for k, v in snaps["energy"]["previews"].items()}
    energy_seq = max(d["seq"] for d in snaps["energy"]["view"]["decisions"])

    moments: dict[str, Any] = {}
    for mid, snap in snaps.items():
        stages: dict[str, str] = {}
        for stage, result in snap["stages"].items():
            if stage in KEEP_STAGES:
                stages[stage] = pool.put(result)
        for stage, status in snap["view"]["stages"].items():
            if stage in READ_BY_STAGE_OR_BANNER and status["status"] not in ("idle", "blocked"):
                assert stage in stages, (mid, stage, status["status"])
                assert pool.items[stages[stage]]["key"] == status["key"], (mid, stage)
        view = snap["view"]
        for d in view["decisions"]:
            decisions.setdefault(d["id"], d)
            assert decisions[d["id"]] == d, (mid, d["id"], "a record changed between moments")
        recorded = {d["id"]: d.get("sentence") for d in view["decisions"]}
        rewritten = {line["record_id"]: line["sentence"] for line in snap["methods"]["lines"]
                     if line["record_id"] in recorded
                     and line["sentence"] != recorded[line["record_id"]]}
        previews = {k: pool.put(rnd(v)) for k, v in (snap.get("previews") or {}).items()}
        note = None
        if mid in ENERGY_AFTER:
            previews.update(energy_previews)
            note = {"source": "energy", "seq": energy_seq,
                    "why": "the server draws a method's storyboard from the data as loaded; the "
                           "standard model leaves every nutrient's values as loaded"}
        moments[mid] = {
            "source": "composed: models + the code-or-amount block from ready" if mid == "codes"
            else f"{'pred' if mid == 'prediction' else 'inf'}-{snap['moment'].replace(':', '-')}",
            "view": {
                "summary": pool.put(view["summary"]),
                "state": pool.put(view["state"]),
                "stages": pool.put(view["stages"]),
                "interview": pool.put(view["interview"]),
                "decisions": [d["id"] for d in view["decisions"]],
            },
            "readings": pool.put(snap["readings"]),
            "methods": rewritten,
            "stages": stages,
            "previews": previews,
            "previewNote": note,
        }

    # The scenario's answers for the covariates the pack does not guess (the slot checks a
    # person's answers against them: the prototype captured this one path).
    card = snaps["adjustment"]["stages"]["proposals"]["artifact"]["adjustment"]
    unguessed = [c for g in card["groups"] if not g.get("guess") for c in g["columns"]]
    answers: dict[str, list[str]] = {}
    for d in snaps["energy"]["view"]["decisions"]:
        if d["decision"]["kind"] != "set_adjustment":
            continue
        for col, a in d["decision"]["answers"].items():
            if col in unguessed:
                answers[col] = [a["causes_exposure"], a["causes_outcome"], a["after_exposure"]]
    assert sorted(answers) == sorted(unguessed), (answers, unguessed)

    teaching = {t["key"]: t for t in load("teaching") if t["key"] in TEACHING}
    out = {
        "captured": "the shared scenario (methods-shared/SCENARIO.md) through the real server; "
                    "readings answered from turbotab/core/tests/truths.py",
        "path": [mid for mid, _ in PATH],
        "moments": moments,
        "decisions": decisions,
        "pool": pool.items,
        "teaching": teaching,
        "columns": load("inf-columns"),
        "derive": load("derive"),
        "adjustmentAnswers": answers,
    }
    text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    (HERE.parent / "fixture.json").write_text(text)
    print(f"fixture.json: {len(text.encode()) // 1000} KB, {len(pool.items)} pooled, "
          f"{len(decisions)} decisions, {len(moments)} moments")


if __name__ == "__main__":
    main()

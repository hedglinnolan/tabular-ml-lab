"""Trim the raw captures (drive.py) into the prototype's fixture.json (not part of the app).

    venv/bin/python turbotab/frontend/src/explore/methods-document/capture/trim.py <raw-out-dir>

Keeps, per moment, the ProjectView, the methods text, the readings card, the previews, and the
stage artifacts the banner, the production stage and the document read; artifacts that are the
same across moments are stored once. Nothing is edited: every value is the server's own. One
mapping is recorded in the fixture rather than hidden: moment m5's previews are the server's
previews of the energy methods computed one decision earlier (``inf-energy-open``, before the
standard model was recorded), because the server draws a method's storyboard from the data as
loaded and the standard model leaves every nutrient's values as loaded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP_STAGES = ("ingest", "oriented", "working", "cohort", "split", "design", "fit", "shelf",
               "roles", "proposals", "effects", "secondary", "sensitivity")
# The stage hooks fetch a stage whose status is neither idle nor blocked; every such stage the
# banner or the stage reads must be in the fixture (the rest are read by neither).
READ_BY_STAGE_OR_BANNER = ("ingest", "oriented", "working", "cohort", "split", "design", "fit",
                           "shelf", "substitution")
MOMENTS = {
    "m1": "inf-m1", "m2": "inf-m2", "m9": "inf-m9", "m3": "inf-m3", "m4": "inf-m4",
    "m5": "inf-m5", "m6": "inf-m6-after", "m7": "pred-m7",
}
TEACHING = ("target", "purpose", "roles", "exclusions", "missing", "split", "estimand",
            "adjustment", "energy_adjustment", "models", "task", "grain", "substitution",
            "open_seal", "clusters", "survey", "causal")


def main() -> None:
    raw = Path(sys.argv[1])
    artifacts: dict[str, dict] = {}
    moments: dict[str, dict] = {}
    for name, source in MOMENTS.items():
        snap = json.loads((raw / f"{source}.json").read_text())
        stages: dict[str, str] = {}
        for stage, result in snap["stages"].items():
            if stage not in KEEP_STAGES:
                continue
            blob = json.dumps(result, sort_keys=True)
            ref = hashlib.sha1(blob.encode()).hexdigest()[:12]
            artifacts.setdefault(ref, result)
            stages[stage] = ref
        for stage, status in snap["view"]["stages"].items():
            if stage in READ_BY_STAGE_OR_BANNER and status["status"] not in ("idle", "blocked"):
                assert stage in stages, (name, stage, status["status"])
                assert artifacts[stages[stage]]["key"] == status["key"], (name, stage)
        previews = snap.get("previews") or {}
        preview_note = None
        if name == "m5":
            earlier = json.loads((raw / "inf-energy-open.json").read_text())
            previews = earlier["previews"]
            preview_note = {
                "source": "inf-energy-open",
                "seq": max(d["seq"] for d in earlier["view"]["decisions"]),
                "why": "the server draws a method's storyboard from the data as loaded; the "
                       "standard model leaves every nutrient's values as loaded",
            }
        moments[name] = {
            "source": source,
            "view": snap["view"],
            "methods": snap["methods"],
            "readings": snap["readings"],
            "stages": stages,
            "previews": previews,
            "previewNote": preview_note,
            "plan": snap.get("plan"),
        }
    teaching = {t["key"]: t for t in json.loads((raw / "teaching.json").read_text())
                if t["key"] in TEACHING}
    columns = json.loads((raw / "inf-columns.json").read_text())
    out = {
        "captured": "NHANES reference journey (_tt_tmp_nhanes.csv) through the real server; "
                    "readings answered from turbotab/core/tests/truths.py",
        "moments": moments,
        "artifacts": artifacts,
        "teaching": teaching,
        "columns": columns,
    }
    text = json.dumps(out, separators=(",", ":"))
    (HERE.parent / "fixture.json").write_text(text)
    print(f"fixture.json: {len(text) // 1000} KB, {len(artifacts)} artifacts, {len(moments)} moments")


if __name__ == "__main__":
    main()

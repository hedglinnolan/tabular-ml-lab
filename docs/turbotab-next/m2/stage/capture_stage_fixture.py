"""Capture the real-data fixture behind the M2 stage's mock API (M2_CONTRACT §11).

Every number in ``turbotab/frontend/src/mocks/m2-stage-fixture.json`` comes from the real server
running M2's journeys on the sample fixtures, through its own HTTP routes (FastAPI TestClient): the
project views, the stage artifacts the stage reads, every preview the lab offers (with the coach's
notes), finding evidence, and the views the stage moves to when an answer is recorded.

Projects (each is one scenario of the stage lab, /lab/stage/m2):

* ``m2-opening``     dietary_recalls.csv: the lens, target and purpose previews
* ``m2-reshape``     dietary_recalls.csv: combining each person's two recalls (mean, first, last,
                     change), the reshape storyboard
* ``m2-orientation`` the feature-major copy of metabolomics_untargeted.csv: the turn
* ``m2-seal-*``      the split question's preview in each basis: grouped (dietary), chronological
                     (clinical_longitudinal), abandoned (dietary, "one row per unit" kept over the
                     data) and undetermined (dietary, the grain not answered); plus eligibility and
                     missing-value previews with the coach's notes
* ``m2-results``     dietary combined by mean, fitted: sealed, opened once, then changed after the
                     opening (energy adjustment residual → density)
* ``m2-repairs``     survey_sentinels.csv: the repairs' previews and the findings' evidence

Run from the repository root (about a minute; two workers):

    TURBOTAB_WORKERS=2 venv/bin/python docs/turbotab-next/m2/stage/capture_stage_fixture.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from turbotab.core.config import Settings  # noqa: E402
from turbotab.server.app import create_app  # noqa: E402

SAMPLES = ROOT / "turbotab" / "sample_data"
OUT = ROOT / "turbotab/frontend/src/mocks/m2-stage-fixture.json"
# The artifacts the stage reads (LiveScenes, Results, evidence labels).
STAGE_ARTIFACTS = ("cohort", "split", "shelf", "design", "fit", "substitution", "findings")


def sig(v: float, digits: int = 5) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return float(f"{v:.{digits}g}")


def rounded(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return sig(obj)
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return obj


def key(decision: dict) -> str:
    return json.dumps(decision, sort_keys=True, separators=(",", ":"))


class Project:
    def __init__(self, client: TestClient, pid: str, label: str, source: str):
        self.c = client
        self.pid = pid
        self.out: dict[str, Any] = {"label": label, "source": source, "snapshots": {}, "start": None,
                                    "previews": [], "records": [], "evidence": {}}

    # ── the real server ──
    def view(self) -> dict:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def wait(self, want: dict[str, str], timeout: float = 240.0) -> dict:
        end = time.monotonic() + timeout
        while True:
            view = self.view()
            stages = view["stages"]
            if all(stages[n]["status"] == s for n, s in want.items()):
                return view
            for n in want:
                if stages[n]["status"] == "error":
                    raise RuntimeError(f"{n} failed: {stages[n].get('error')}")
            if time.monotonic() > end:
                raise TimeoutError(f"{want}: { {n: stages[n]['status'] for n in want} }")
            time.sleep(0.05)

    def settle(self, timeout: float = 240.0) -> dict:
        """Wait until no stage is queued or running."""
        end = time.monotonic() + timeout
        while True:
            view = self.view()
            busy = [n for n, s in view["stages"].items() if s["status"] in ("queued", "running")]
            if not busy:
                return view
            if time.monotonic() > end:
                raise TimeoutError(f"still busy: {busy}")
            time.sleep(0.05)

    def decide(self, decision: dict) -> dict:
        r = self.c.post(f"/api/projects/{self.pid}/decisions", json=decision)
        if r.status_code != 200:
            raise RuntimeError(f"{decision}: {r.status_code} {r.text}")
        return r.json()

    def stage(self, name: str) -> dict:
        return self.c.get(f"/api/projects/{self.pid}/stages/{name}").json()

    # ── the fixture ──
    def snapshot(self, name: str) -> None:
        view = self.settle()
        artifacts = {}
        for stage in STAGE_ARTIFACTS:
            status = view["stages"].get(stage, {}).get("status")
            if status in ("fresh", "stale"):
                artifacts[stage] = self.stage(stage)["artifact"]
        self.out["snapshots"][name] = {"view": view, "artifacts": artifacts}
        if self.out["start"] is None:
            self.out["start"] = name

    def preview(self, group: str, label: str, decision: dict) -> dict:
        t0 = time.perf_counter()
        r = self.c.post(f"/api/projects/{self.pid}/preview", json=decision)
        ms = (time.perf_counter() - t0) * 1000
        body = r.json()
        self.out["previews"].append({"group": group, "label": label, "decision": decision,
                                     "key": key(decision), "status": r.status_code,
                                     "ms": round(ms, 1), "body": body})
        return body

    def record(self, decision: dict, to: str, label: str) -> None:
        """Record for real, then snapshot what the stage moves to."""
        self.decide(decision)
        self.snapshot(to)
        self.out["records"].append({"key": key(decision), "decision": decision, "to": to, "label": label})


def open_project(client: TestClient, path: Path) -> str:
    r = client.post("/api/projects", json={"path": str(path)})
    r.raise_for_status()
    return r.json()["id"]


def proposed_roles(p: Project, drop: set[str]) -> dict[str, str]:
    p.wait({"roles": "fresh"})
    roles = {c["column"]: c["proposed"] for c in p.stage("roles")["artifact"]["columns"]}
    for c in drop:
        roles.pop(c, None)
    return roles


def dietary_opening(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "dietary_recalls.csv")
    p = Project(client, pid, "The opening: lens, outcome, purpose", "dietary_recalls.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.snapshot("start")
    for lens in ("dietary", "clinical", "metabolomics"):
        p.preview("lens", f"Lens: {lens}", {"kind": "set_lens", "lenses": [lens]})
    p.decide({"kind": "set_lens", "lenses": ["dietary"]})
    p.wait({"findings": "fresh"})
    for col in ("hba1c", "energy_kcal"):
        p.preview("target", f"Outcome: {col}", {"kind": "set_target", "column": col})
    p.decide({"kind": "set_target", "column": "hba1c"})
    p.wait({"target_info": "fresh"})
    for purpose in ("prediction", "inference"):
        p.preview("purpose", f"Purpose: {purpose}", {"kind": "set_purpose", "purpose": purpose})
    return p.out


def dietary_reshape(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "dietary_recalls.csv")
    p = Project(client, pid, "Combining each person's recalls", "dietary_recalls.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["dietary"]})
    p.decide({"kind": "set_target", "column": "hba1c"})
    p.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    p.wait({"structure": "fresh"})
    p.decide({"kind": "set_unit", "unit": "unit"})
    p.snapshot("start")
    labels = {"mean": "Average each person's recalls", "first": "Keep each person's first recall",
              "last": "Keep each person's last recall", "change": "Last minus first"}
    for method, label in labels.items():
        p.preview("aggregation", label, {"kind": "set_aggregation", "method": method})
    p.record({"kind": "set_aggregation", "method": "mean"}, "combined", labels["mean"])
    return p.out


def metabolomics_orientation(client: TestClient, scratch: Path) -> dict:
    m = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv").set_index("sample_id")
    turned = m.select_dtypes("number").T
    turned.index.name = "feature_id"
    source = scratch / "metabolomics_feature_major.csv"
    turned.to_csv(source)
    pid = open_project(client, source)
    p = Project(client, pid, "A feature-major table, turned", "metabolomics_untargeted.csv, feature-major copy")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["metabolomics"]})
    p.wait({"oriented": "fresh"})
    p.snapshot("start")
    p.preview("orientation", "Rows are features: turn the table", {"kind": "set_orientation", "orientation": "feature_major"})
    p.preview("orientation", "Rows are samples: keep it", {"kind": "set_orientation", "orientation": "sample_major"})
    p.record({"kind": "set_orientation", "orientation": "feature_major"}, "turned", "Rows are features: turn the table")
    return p.out


def split_previews(p: Project) -> None:
    for h, label in ((0.1, "Hold out 10%"), (0.2, "Hold out 20%"), (0.3, "Hold out 30%"), (0.0, "Cross-validation only")):
        p.preview("split", label, {"kind": "set_split", "holdout": h, "seed": 0, "folds": 5})


def seal_grouped(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "dietary_recalls.csv")
    p = Project(client, pid, "The seal, grouped by participant", "dietary_recalls.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["dietary"]})
    p.decide({"kind": "set_target", "column": "hba1c"})
    p.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    p.decide({"kind": "set_unit", "unit": "row"})
    p.decide({"kind": "set_roles", "roles": proposed_roles(p, {"hba1c"})})
    p.wait({"proposals": "fresh", "cohort": "fresh"})
    proposals = p.stage("proposals")["artifact"]
    for prop in proposals["exclusions"][:2]:
        p.preview("exclusions", prop["label"], {"kind": "set_exclusions", "rules": [prop["rule"]]})
    p.preview("exclusions", "Keep every row", {"kind": "set_exclusions", "rules": []})
    p.decide({"kind": "set_exclusions", "rules": []})
    for strategy, label in (("complete_case", "Complete cases"), ("impute", "Fill the blanks")):
        p.preview("missing", label, {"kind": "set_missing", "strategy": strategy})
    p.decide({"kind": "set_missing", "strategy": "complete_case"})
    p.wait({"cohort": "fresh"})
    p.snapshot("start")
    split_previews(p)
    p.record({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}, "sealed", "Hold out 20%")
    return p.out


def seal_chronological(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "clinical_longitudinal.csv")
    p = Project(client, pid, "The seal, chronological", "clinical_longitudinal.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["clinical"]})
    p.decide({"kind": "set_target", "column": "progressed"})
    p.decide({"kind": "set_event", "column": "progressed", "level": "1"})
    p.decide({"kind": "set_grain", "grain": "repeated", "id_column": "subject_id"})
    p.wait({"structure": "fresh"})
    p.decide({"kind": "set_unit", "unit": "row"})
    p.decide({"kind": "set_temporal", "temporal": True, "time_column": "visit_date"})
    p.decide({"kind": "set_roles", "roles": proposed_roles(p, {"progressed"})})
    p.decide({"kind": "set_exclusions", "rules": []})
    p.decide({"kind": "set_missing", "strategy": "complete_case"})
    p.wait({"cohort": "fresh"})
    p.snapshot("start")
    split_previews(p)
    p.record({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}, "sealed", "Hold out 20%")
    return p.out


def seal_abandoned(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "dietary_recalls.csv")
    p = Project(client, pid, "The seal, grouping abandoned", "dietary_recalls.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["dietary"]})
    p.decide({"kind": "set_target", "column": "hba1c"})
    # "Each row is a different unit", kept over the data's contradiction (the attest exit).
    p.decide({"kind": "set_grain", "grain": "one_row_per_unit", "acknowledged": True})
    roles = proposed_roles(p, {"hba1c"})
    roles["participant_id"] = "identifier"
    p.decide({"kind": "set_roles", "roles": roles})
    p.decide({"kind": "set_exclusions", "rules": []})
    p.decide({"kind": "set_missing", "strategy": "complete_case"})
    p.wait({"cohort": "fresh"})
    p.snapshot("start")
    split_previews(p)
    p.record({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}, "sealed", "Hold out 20%")
    return p.out


def seal_undetermined(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "dietary_recalls.csv")
    p = Project(client, pid, "The seal, basis undetermined", "dietary_recalls.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["dietary"]})
    p.decide({"kind": "set_target", "column": "hba1c"})
    # The grain is not answered and no identifier is named: whether a person repeats is unknown.
    roles = proposed_roles(p, {"hba1c"})
    roles["participant_id"] = "excluded"
    p.decide({"kind": "set_roles", "roles": roles})
    p.decide({"kind": "set_exclusions", "rules": []})
    p.decide({"kind": "set_missing", "strategy": "complete_case"})
    p.wait({"cohort": "fresh"})
    p.snapshot("start")
    split_previews(p)
    p.record({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}, "sealed", "Hold out 20%")
    return p.out


def results(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "dietary_recalls.csv")
    p = Project(client, pid, "The Results: sealed, opened once, changed after", "dietary_recalls.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["dietary"]})
    p.decide({"kind": "set_target", "column": "hba1c"})
    p.decide({"kind": "set_purpose", "purpose": "prediction"})
    p.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    p.wait({"structure": "fresh"})
    p.decide({"kind": "set_unit", "unit": "unit"})
    p.decide({"kind": "set_aggregation", "method": "mean"})
    p.wait({"working": "fresh"})
    p.decide({"kind": "set_roles", "roles": proposed_roles(p, {"hba1c"})})
    p.decide({"kind": "set_exclusions", "rules": []})
    p.decide({"kind": "set_missing", "strategy": "complete_case"})
    p.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    p.wait({"proposals": "fresh", "split": "fresh", "shelf": "fresh"})
    energy = p.stage("proposals")["artifact"]["energy"]
    base = {"kind": "set_energy_adjustment", "energy_column": energy["energy_column"],
            "nutrients": energy["nutrients"]}
    for method, label in (("residual", "Willett residual model"), ("density", "Nutrient density"),
                          ("none", "No energy adjustment")):
        p.preview("energy_adjustment", label, base | {"method": method})
    p.decide(base | {"method": "residual"})
    fams = [f["key"] for f in p.stage("shelf")["artifact"]["families"]]
    p.decide({"kind": "select_models", "models": fams})
    p.wait({"fit": "fresh"}, timeout=600)
    pairs = p.stage("design")["artifact"].get("substitution_pairs") or []
    if pairs:
        first = pairs[0]
        p.decide({"kind": "set_substitution", "donor": first["donor"], "recipient": first["recipient"],
                  "step_kcal": 100.0, "n_boot": 0})
        p.wait({"substitution": "fresh"}, timeout=600)
    p.snapshot("sealed")
    p.preview("open_seal", "Open the seal", {"kind": "open_seal"})
    p.record({"kind": "open_seal"}, "opened", "Open the seal")
    p.decide(base | {"method": "density"})
    p.wait({"fit": "fresh"}, timeout=600)
    if pairs:
        p.wait({"substitution": "fresh"}, timeout=600)
    p.snapshot("post")
    p.out["records"].append({"key": key(base | {"method": "density"}), "decision": base | {"method": "density"},
                             "to": "post", "label": "Energy: nutrient density (after the opening)"})
    return p.out


def survey_repairs(client: TestClient) -> dict:
    pid = open_project(client, SAMPLES / "survey_sentinels.csv")
    p = Project(client, pid, "Repairs, previewed before they apply", "survey_sentinels.csv")
    p.wait({"ingest": "fresh", "profile": "fresh"})
    p.decide({"kind": "set_lens", "lenses": ["survey"]})
    p.wait({"findings": "fresh"})
    p.snapshot("start")
    findings = p.stage("findings")["artifact"]["findings"]
    shown = 0
    for f in findings:
        repairs = f.get("repairs") or []
        if not repairs or shown >= 3:
            continue
        shown += 1
        for opt in repairs[:3]:
            p.preview("repairs", f"{opt['label']} · {f['id']}", opt["decision"])
    for f in findings[:6]:
        r = client.get(f"/api/projects/{pid}/findings/{f['id']}/evidence")
        if r.status_code == 200:
            p.out["evidence"][f["id"]] = r.json()
    return p.out


def main() -> None:
    home = Path(tempfile.mkdtemp(prefix="tt-m2-stage-capture-"))
    scratch = Path(tempfile.mkdtemp(prefix="tt-m2-stage-sources-"))
    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=4 << 30)
    app = create_app(settings, frontend_dist=home / "no-frontend")
    t0 = time.monotonic()
    fixture: dict[str, Any] = {"meta": {
        "script": "docs/turbotab-next/m2/stage/capture_stage_fixture.py",
        "captured": time.strftime("%Y-%m-%d"),
    }, "projects": {}}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        builders = {
            "m2-opening": lambda: dietary_opening(client),
            "m2-reshape": lambda: dietary_reshape(client),
            "m2-orientation": lambda: metabolomics_orientation(client, scratch),
            "m2-seal-grouped": lambda: seal_grouped(client),
            "m2-seal-chronological": lambda: seal_chronological(client),
            "m2-seal-abandoned": lambda: seal_abandoned(client),
            "m2-seal-undetermined": lambda: seal_undetermined(client),
            "m2-results": lambda: results(client),
            "m2-repairs": lambda: survey_repairs(client),
        }
        for name, build in builders.items():
            t = time.monotonic()
            fixture["projects"][name] = build()
            n = len(fixture["projects"][name]["previews"])
            print(f"{name}: {n} previews, {time.monotonic() - t:.1f} s")
    OUT.write_text(json.dumps(rounded(fixture), separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()

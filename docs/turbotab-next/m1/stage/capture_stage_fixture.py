"""Capture the real-data fixture behind the stage's mock API (M1_CONTRACT §14 "stage").

Every number in ``m1-stage-fixture.json`` comes from the real server running the §9 journey on
the NHANES export, through its own HTTP routes (FastAPI TestClient): the project view, the
cohort / split / shelf / design / fit / substitution artifacts, and the previews the stage shows
for every option of the energy-adjustment, exclusions, missing-values, split and models questions.

Two pieces the M1 part 2 backend adds (§12) are computed here the way §12 specifies, so the mock
can show them before that backend lands:

* the refit uncertainty band (§12.7): each family refit on bootstrap resamples of at most 2,000
  training rows, the curve recomputed through each refit, and the 2.5th/97.5th percentiles taken.
  Its wall time is measured and recorded, so the mock's "about N s" is a measurement;
* the outcome's baseline (§12.6): the training mean, scored by the same CV folds (R² is 1 − SSE/SST
  against each fold's own mean, so the mean predicts at about 0).

Run from the repository root (about a minute; two workers):

    TURBOTAB_WORKERS=2 venv/bin/python docs/turbotab-next/m1/stage/capture_stage_fixture.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from turbotab.core.config import Settings  # noqa: E402
from turbotab.core.tests.stage_harness import NHANES  # noqa: E402  (the tracked fixture)
from turbotab.server.app import create_app  # noqa: E402

OUT = ROOT / "turbotab/frontend/src/mocks/m1-stage-fixture.json"
NUTRIENTS = ["protein", "carb", "sugar", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
BOOT = 40
BOOT_ROWS = 2_000


def sig(v: float, digits: int = 4) -> float | None:
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


class Journey:
    def __init__(self, client: TestClient):
        self.c = client
        self.pid = ""

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

    def decide(self, decision: dict) -> dict:
        r = self.c.post(f"/api/projects/{self.pid}/decisions", json=decision)
        if r.status_code != 200:
            raise RuntimeError(f"{decision}: {r.status_code} {r.text}")
        return r.json()

    def preview(self, decision: dict) -> dict:
        t0 = time.perf_counter()
        r = self.c.post(f"/api/projects/{self.pid}/preview", json=decision)
        ms = (time.perf_counter() - t0) * 1000
        body = r.json()
        return {"decision": decision, "status": r.status_code, "ms": round(ms, 1), "body": body}

    def stage(self, name: str) -> dict:
        return self.c.get(f"/api/projects/{self.pid}/stages/{name}").json()


def refit_band(home: Path, pid: str, view: dict, donor: str, recipient: str,
               step_kcal: float) -> dict[str, Any]:
    """§12.7: refit each family on bootstrap resamples of ≤ 2,000 training rows."""
    from sklearn.base import clone

    from turbotab.core.graph import read_artifact
    from turbotab.core.methods.energy import energy_factor
    from turbotab.core.methods.substitution import substitution_curve
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.modeling import row_ids_of, SUBSTITUTION_ROWS, SUBSTITUTION_STEPS
    from turbotab.core.workspace import Workspace

    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=4 << 30)
    ws = Workspace(settings)
    cache = ws.cache_dir(pid)
    design = read_artifact(cache, "design", view["stages"]["design"]["key"])
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    target = view["state"]["target"]
    train_ids = row_ids_of(design.frames["training"])
    store = ws.datastore(pid)
    frame = modeling_frame(store, [*spec.inputs, target], train_ids)
    X_all, y_all = frame[spec.inputs], frame[target].to_numpy()
    curve_ids = train_ids
    if len(curve_ids) > SUBSTITUTION_ROWS:
        curve_ids = np.sort(np.random.default_rng(0).choice(curve_ids, SUBSTITUTION_ROWS, replace=False))
    X_curve = modeling_frame(store, spec.inputs, curve_ids)
    kcal = {c: float(energy_factor(c).factor) for c in (donor, recipient)}
    ks = [step_kcal * i for i in range(SUBSTITUTION_STEPS + 1)]
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    for key in view["state"]["models"]:
        curves = []
        for _ in range(BOOT):
            idx = rng.choice(len(X_all), min(BOOT_ROWS, len(X_all)), replace=True)
            model = clone(pipelines[key]).fit(X_all.iloc[idx], y_all[idx])
            c = substitution_curve(model.predict, X_curve, donor=donor, recipient=recipient,
                                   kcal_per_unit=kcal, ks=ks, total_kind="variable")
            curves.append([np.nan if d is None else d for d in c["delta"]])
        arr = np.array(curves, dtype=float)
        lo = np.nanpercentile(arr, 2.5, axis=0)
        hi = np.nanpercentile(arr, 97.5, axis=0)
        out[key] = {"ci_low": [sig(float(v)) for v in lo], "ci_high": [sig(float(v)) for v in hi]}
    return {"n_boot": BOOT, "rows": BOOT_ROWS, "seconds": round(time.perf_counter() - t0, 1),
            "families": out}


def baseline(home: Path, pid: str, view: dict) -> dict[str, Any]:
    """§12.6: the outcome's mean, scored on the same CV folds as each model."""
    from turbotab.core.graph import read_artifact
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.stages.modeling import read_assignment
    from turbotab.core.workspace import Workspace

    ws = Workspace(Settings(home=home, mode="local", workers=2, memory_budget_bytes=4 << 30))
    split = read_artifact(ws.cache_dir(pid), "split", view["stages"]["split"]["key"])
    assignment = read_assignment(split)
    target = view["state"]["target"]
    frame = modeling_frame(ws.datastore(pid), [target], assignment.index.to_numpy())
    train = assignment["train"].to_numpy()
    y = frame[target].to_numpy()[train]
    folds = assignment.loc[train, "fold"].to_numpy().astype(int)
    r2 = []
    for k in sorted(set(folds.tolist())):
        fit, test = y[folds != k], y[folds == k]
        pred = np.full(len(test), fit.mean())
        r2.append(1 - float(((test - pred) ** 2).sum()) / float(((test - test.mean()) ** 2).sum()))
    return {"metric": "r2", "value": sig(float(np.mean(r2)))}


def main() -> None:
    home = Path(tempfile.mkdtemp(prefix="tt-stage-capture-"))
    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=4 << 30)
    app = create_app(settings, frontend_dist=home / "no-frontend")
    fixture: dict[str, Any] = {"meta": {
        "source": str(NHANES.name), "script": "docs/turbotab-next/m1/stage/capture_stage_fixture.py",
        "captured": time.strftime("%Y-%m-%d"),
    }, "previews": {}}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        j = Journey(client)
        r = client.post("/api/projects", json={"path": str(NHANES)})
        j.pid = r.json()["id"]
        j.wait({"ingest": "fresh", "profile": "fresh"})
        j.decide({"kind": "set_lens", "lenses": ["dietary"]})
        j.decide({"kind": "set_target", "column": "glucose"})
        j.wait({"roles": "fresh", "target_info": "fresh", "cohort": "fresh", "findings": "fresh",
                "proposals": "fresh"})
        j.decide({"kind": "set_purpose", "purpose": "prediction"})
        roles = {c["column"]: c["proposed"] for c in j.stage("roles")["artifact"]["columns"]}
        roles.pop("glucose", None)
        j.decide({"kind": "set_roles", "roles": roles})
        fixture["roles"] = j.stage("roles")["artifact"]
        fixture["findings"] = j.stage("findings")["artifact"]
        # What a finding's evidence route reads (§12.3): its columns' rows and first histogram.
        evidence: dict[str, Any] = {}
        numeric = {c["name"] for c in client.get(f"/api/projects/{j.pid}/columns").json()
                   if c["dtype"] in ("numeric", "integer")}
        for f in fixture["findings"]["findings"]:
            cols = f["affected_columns"][:6]
            entry: dict[str, Any] = {"table": None, "histogram": None, "column": None}
            if cols:
                entry["table"] = client.get(f"/api/projects/{j.pid}/table",
                                            params={"offset": 0, "limit": 8,
                                                    "columns": ",".join(["SEQN", *cols])}).json()
                first = next((c for c in cols if c in numeric), None)
                if first:
                    entry["column"] = first
                    entry["histogram"] = client.get(
                        f"/api/projects/{j.pid}/columns/{first}/histogram", params={"bins": 30}).json()
            evidence[f["id"]] = entry
        fixture["evidence_inputs"] = evidence
        j.wait({"proposals": "fresh", "cohort": "fresh"})
        proposals = j.stage("proposals")["artifact"]
        fixture["proposals"] = proposals

        # Exclusions: every pack screen, and keeping every row.
        ex = []
        for p in proposals["exclusions"]:
            ex.append(j.preview({"kind": "set_exclusions", "rules": [p["rule"]]}) | {"label": p["label"]})
        ex.append(j.preview({"kind": "set_exclusions", "rules": []}) | {"label": "Keep every row"})
        fixture["previews"]["exclusions"] = ex
        usual = next(p for p in proposals["exclusions"] if "sex" in p["label"].lower()) \
            if any("sex" in p["label"].lower() for p in proposals["exclusions"]) else proposals["exclusions"][0]
        j.decide({"kind": "set_exclusions", "rules": [usual["rule"]]})
        j.wait({"cohort": "fresh"})

        fixture["previews"]["missing"] = [
            j.preview({"kind": "set_missing", "strategy": s}) for s in ("complete_case", "impute")]
        j.decide({"kind": "set_missing", "strategy": "complete_case"})
        j.wait({"cohort": "fresh"})

        fixture["previews"]["split"] = [
            j.preview({"kind": "set_split", "holdout": h, "seed": 0, "folds": 5}) for h in (0.2, 0.0, 0.3)]
        j.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        j.wait({"cohort": "fresh", "split": "fresh", "shelf": "fresh"})

        energy = proposals["energy"]
        base = {"kind": "set_energy_adjustment", "energy_column": energy["energy_column"],
                "nutrients": energy["nutrients"]}
        fixture["previews"]["energy_adjustment"] = [
            j.preview(base | {"method": m}) for m in
            ("residual", "density", "density_multivariate", "standard", "partition", "none")]
        part = [n for n in energy["nutrients"] if n in ("protein", "carb", "fat_total")]
        fixture["previews"]["energy_adjustment"].append(
            j.preview(base | {"method": "partition", "nutrients": part}))
        j.decide(base | {"method": "residual"})

        fams = [f["key"] for f in j.stage("shelf")["artifact"]["families"]]
        fixture["previews"]["models"] = [j.preview({"kind": "select_models", "models": fams})] + [
            j.preview({"kind": "select_models", "models": [f]}) for f in fams]
        j.decide({"kind": "select_models", "models": fams})
        view = j.wait({"design": "fresh", "fit": "fresh"})
        fixture["fit_prediction"] = j.stage("fit")["artifact"]
        for name in ("cohort", "split", "shelf", "design"):
            fixture[name] = j.stage(name)["artifact"]
        fixture["baseline"] = baseline(home, j.pid, view)

        # Substitution: several pairs, each a real recompute.
        subs = []
        for pair in fixture["design"]["substitution_pairs"]:
            donor, recipient = pair["donor"], pair["recipient"]
            j.decide({"kind": "set_substitution", "donor": donor, "recipient": recipient,
                      "step_kcal": 100})
            t0 = time.perf_counter()
            j.wait({"substitution": "fresh"})
            art = j.stage("substitution")["artifact"]
            art["_seconds"] = round(time.perf_counter() - t0, 2)
            subs.append(art)
        fixture["substitution"] = subs
        j.decide({"kind": "set_substitution", "donor": "fat_total", "recipient": "carb",
                  "step_kcal": 100})
        view = j.wait({"substitution": "fresh"})
        fixture["band"] = refit_band(home, j.pid, view, "fat_total", "carb", 100)

        # Density, for the re-flow after a changed answer.
        j.decide(base | {"method": "density"})
        view = j.wait({"design": "fresh", "fit": "fresh", "substitution": "fresh"})
        fixture["density"] = {"design": j.stage("design")["artifact"],
                              "fit": j.stage("fit")["artifact"],
                              "substitution": j.stage("substitution")["artifact"]}
        j.decide(base | {"method": "residual"})
        j.wait({"design": "fresh", "fit": "fresh", "substitution": "fresh"})

        # Inference: the coefficient forest with its intervals.
        j.decide({"kind": "set_purpose", "purpose": "inference"})
        j.wait({"design": "fresh", "fit": "fresh"})
        fixture["fit_inference"] = j.stage("fit")["artifact"]
        j.decide({"kind": "set_purpose", "purpose": "prediction"})
        view = j.wait({"design": "fresh", "fit": "fresh", "substitution": "fresh"})
        fixture["view"] = view

    OUT.write_text(json.dumps(rounded(fixture), separators=(",", ":")) + "\n", "utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    ms = [p["ms"] for group in fixture["previews"].values() for p in group]
    print(f"previews: {len(ms)}, p95 {sorted(ms)[int(0.95 * (len(ms) - 1))]:.0f} ms")
    print(f"band: {fixture['band']['seconds']} s for {BOOT} refits per family")


if __name__ == "__main__":
    main()

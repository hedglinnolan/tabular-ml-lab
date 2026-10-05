"""Capture the methods-document prototype's moments from the real server (not part of the app).

    TURBOTAB_HOME=$(mktemp -d) TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 \
        venv/bin/python -m turbotab.server --port 8917
    venv/bin/python turbotab/frontend/src/explore/methods-document/capture/drive.py \
        http://127.0.0.1:8917 <raw-out-dir>
    venv/bin/python turbotab/frontend/src/explore/methods-document/capture/trim.py <raw-out-dir>

Drives the NHANES reference journey (``_tt_tmp_nhanes.csv``) through the HTTP API, answering every
reading the server asks from the fixture's declared truth (``turbotab/core/tests/truths.py``), never
a constant: once under inference to the locked plan, once under prediction to the fit. At each
moment it saves the ProjectView, the methods text, the readings card, every computed stage's
artifact and the previews of the options that moment's slot offers. Before the plan is locked no
estimate stage is fetched: fetching one is what records the lock (``turbotab/core/plan_lock.py``).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from turbotab.core.estimand import ESTIMATE_STAGES  # noqa: E402
from turbotab.core.tests.acceptance import server_drive as sd  # noqa: E402
from turbotab.core.tests.truths import answer_adjustment, fixture_truth  # noqa: E402


def _nhanes() -> Path:
    from turbotab.core.tests.stage_harness import NHANES

    return NHANES


class Client:
    """The acceptance drive's client shape (``get``/``post``) over a running server."""

    def __init__(self, base: str):
        self.h = httpx.Client(base_url=base, timeout=900)

    def get(self, url: str, **kw: Any) -> Any:
        return self.h.get(url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> Any:
        return self.h.post(url, json=json, **kw)


class Capture:
    def __init__(self, drive: sd.Drive, out: Path, prefix: str):
        self.d, self.out, self.prefix = drive, out, prefix
        self.locked = False

    def preview(self, decision: dict[str, Any]) -> dict[str, Any]:
        r = self.d.c.post(f"/api/projects/{self.d.pid}/preview", json=decision)
        return {"status": r.status_code, "decision": decision, "body": r.json()}

    def snap(self, moment: str, **extra: Any) -> dict[str, Any]:
        pid = self.d.pid
        view = self.d.view()
        stages: dict[str, Any] = {}
        for name, status in view["stages"].items():
            if status["status"] in ("idle", "blocked"):
                continue
            if name in ESTIMATE_STAGES and not self.locked:
                continue  # fetching an estimate records the plan lock
            r = self.d.c.get(f"/api/projects/{pid}/stages/{name}")
            if r.status_code == 200:
                stages[name] = r.json()
        out = {
            "moment": moment,
            "view": self.d.view() if self.locked else view,
            "methods": self.d.c.get(f"/api/projects/{pid}/methods").json(),
            "readings": self.d.c.get(f"/api/projects/{pid}/readings").json(),
            "stages": stages,
            **extra,
        }
        (self.out / f"{self.prefix}-{moment}.json").write_text(json.dumps(out, default=str))
        print(f"{self.prefix}-{moment}: {len(view['decisions'])} decisions, {len(stages)} stages")
        return out

    def wait_quiet(self, timeout: float = 600) -> None:
        """Until no stage is queued or running (what the client shows is then settled)."""
        end = time.monotonic() + timeout
        while True:
            busy = [k for k, s in self.d.view()["stages"].items()
                    if s["status"] in ("queued", "running")]
            if not busy:
                return
            assert time.monotonic() < end, busy
            time.sleep(0.3)


NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
ENERGY_METHODS = ["standard", "residual", "density_multivariate", "none", "residual_energy_dropped",
                  "density", "all_components", "partition"]


def energy(method: str) -> dict[str, Any]:
    return {"kind": "set_energy_adjustment", "method": method, "energy_column": "kcal",
            "nutrients": NUTRIENTS}


def roles_truth(drive: sd.Drive) -> dict[str, str]:
    """The author's roles are the proposals (the acceptance drive's rule): the truth for each."""
    props = drive.artifact("roles")["columns"]
    roles = {c["column"]: c["proposed"] for c in props}
    for column, role in roles.items():
        drive.truth.setdefault(f"role:{column}", role)
    return roles


def opening(drive: sd.Drive, purpose: str) -> None:
    drive.answer("lens", {"kind": "set_lens", "lenses": ["dietary"]})
    drive.answer("target", {"kind": "set_target", "column": "glucose"})
    drive.artifact("target_info", timeout=600)
    if drive.reach("task")["status"] in ("open", "waiting"):
        drive.answer("task", {"kind": "set_task", "column": "glucose", "task": "regression"})
    drive.answer("purpose", {"kind": "set_purpose", "purpose": purpose})


def ask_of(view: dict[str, Any], key: str) -> dict[str, Any] | None:
    return next((s.get("ask") for s in view["interview"] if s["key"] == key), None)


def inference(client: Client, out: Path) -> None:
    drive = sd.open_project(client, _nhanes(), fixture_truth(_nhanes().name))
    cap = Capture(drive, out, "inf")
    opening(drive, "inference")
    cap.wait_quiet()
    cap.snap("m1")

    roles = roles_truth(drive)
    r = drive.post({"kind": "set_roles", "roles": roles})
    assert r.status_code == 200, r.text[:600]
    drive.reach("exclusions", timeout=300)
    drive.artifact("proposals")
    cap.wait_quiet()
    unit = {"kind": "set_column_unit", "column": "kcal", "unit": drive.truth.answer("unit", "kcal"),
            "days": int(drive.truth.answer("day_count", "kcal"))}
    rules = {p["key"]: p["rule"] for p in drive.artifact("proposals")["exclusions"]}
    cap.snap("exclusions-ask", previews={
        "unit_1day": cap.preview(unit),
        "unit_2days": cap.preview({**unit, "days": 2}),
        "unit_kj": cap.preview({**unit, "unit": "kj"}),
    })

    # Single confirmation 1: the screens' card asks kcal's unit and days.
    assert drive.post(unit).status_code == 200
    cap.snap("exclusions-open", previews={
        **{k: cap.preview({"kind": "set_exclusions", "rules": [v]}) for k, v in rules.items()},
        "none": cap.preview({"kind": "set_exclusions", "rules": []}),
    })
    drive.decide({"kind": "set_exclusions", "rules": [rules["willett_2013_by_sex"]]})
    # A declared sensitivity analysis: the same model on every row (Banna et al. 2017).
    drive.decide({"kind": "set_sensitivity", "analyses": [{"label": "Every row kept", "rules": []}]})
    drive.reach("missing", timeout=300)
    drive.decide({"kind": "set_missing", "strategy": "complete_case"})
    drive.reach("split", timeout=300)
    drive.decide({"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    drive.reach("estimand", timeout=600)
    drive.artifact("proposals")
    cap.wait_quiet()
    ask = ask_of(drive.view(), "estimand")
    family = next(g for g in ask["groups"] if len(g["columns"]) > 1)
    cap.snap("m2", previews={
        "family_flag": cap.preview({"kind": "confirm_readings", "items": [
            {"reading": "role", "column": c, "value": "flag"} for c in family["columns"]]}),
        "imputed_bmi_covariate": cap.preview({"kind": "confirm_reading", "reading": "role",
                                              "column": family["columns"][0], "value": "covariate"}),
        "bp_sys_covariate": cap.preview({"kind": "confirm_reading", "reading": "role",
                                         "column": "bp_sys", "value": "covariate"}),
    })

    # The column summaries a reading's evidence is drawn from (no outcome relationship is read).
    (out / "inf-columns.json").write_text(client.get(f"/api/projects/{drive.pid}/columns").text)

    # Singles 2 and 3: two role readings, each confirmed on its own, from the truth.
    for column in ("cycle_begin_year", "hdl"):
        assert drive.post({"kind": "confirm_reading", "reading": "role", "column": column,
                           "value": drive.truth[f"role:{column}"]}).status_code == 200
    cap.wait_quiet()
    cap.snap("m9")

    block = ask_of(drive.view(), "estimand")["exits"][0]["decision"]
    for item in block["items"]:  # every value it settles is the fixture's truth
        assert drive.truth[f"role:{item['column']}"] == item["value"], item
    assert drive.post(block).status_code == 200
    cap.wait_quiet()
    base = {"kind": "set_estimand", "effect": "total", "measure": "mean_difference"}
    cap.snap("m3", previews={
        "sugar_substitution": cap.preview({**base, "exposure": "sugar", "contrast": "substitution"}),
        "sugar_addition": cap.preview({**base, "exposure": "sugar", "contrast": "addition"}),
    })

    sd.answer_estimand(drive, "sugar", effect="total", contrast="substitution")
    drive.reach("adjustment", timeout=600)
    end = time.monotonic() + 300
    while (drive.artifact("proposals").get("adjustment") or {}).get("exposure") != "sugar":
        assert time.monotonic() < end
        time.sleep(0.3)
    cap.wait_quiet()
    cap.snap("m4")
    card = drive.artifact("proposals")["adjustment"]
    answer_adjustment(lambda d: drive.post(d), card, drive.truth)
    drive.reach("energy_adjustment", timeout=600)
    cap.wait_quiet()
    cap.snap("energy-open", previews={m: cap.preview(energy(m)) for m in ENERGY_METHODS})
    drive.decide(energy("standard"))
    cap.wait_quiet()
    cap.snap("m5", previews={m: cap.preview(energy(m)) for m in ENERGY_METHODS})

    drive.reach("models", timeout=600)
    for column in ("age", "cycle_begin_year"):  # the fit's card: codes or amounts, one at a time
        assert drive.post({"kind": "confirm_reading", "reading": "code_or_count", "column": column,
                           "value": drive.truth.answer("code_or_count", column)}).status_code == 200
    drive.decide({"kind": "select_models", "models": ["linear"]})
    end = time.monotonic() + 900
    while drive.view()["stages"]["fit"]["status"] != "fresh":
        assert time.monotonic() < end
        time.sleep(0.5)
    cap.wait_quiet()
    cap.snap("fitted")
    cap.locked = True
    plan = drive.c.get(f"/api/projects/{drive.pid}/plan")
    cap.snap("m6", plan=plan.json())
    cap.wait_quiet()
    cap.snap("m6-after", plan=plan.json())


def prediction(client: Client, out: Path) -> None:
    drive = sd.open_project(client, _nhanes(), fixture_truth(_nhanes().name))
    cap = Capture(drive, out, "pred")
    cap.locked = True  # no plan lock under prediction
    opening(drive, "prediction")
    drive.decide_roles(roles_truth(drive))
    drive.reach("exclusions", timeout=300)
    rules = {p["key"]: p["rule"] for p in drive.artifact("proposals")["exclusions"]}
    drive.decide({"kind": "set_exclusions", "rules": [rules["willett_2013_by_sex"]]})
    drive.reach("missing", timeout=300)
    first = drive.artifact("proposals")["missing"]["methods"][0]["decision"]
    drive.decide({"kind": "set_missing", **first})
    drive.reach("split", timeout=300)
    drive.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    drive.reach("energy_adjustment", timeout=600)
    drive.decide(energy(drive.artifact("proposals")["energy"]["ranking"]["order"][0]))
    drive.reach("models", timeout=600)
    cap.wait_quiet()
    cap.snap("models-open")
    drive.decide({"kind": "select_models", "models": ["linear"]})
    drive.artifact("fit", timeout=900)
    cap.wait_quiet()
    cap.snap("m7")


def main() -> None:
    base, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    client = Client(base)
    inference(client, out)
    prediction(client, out)
    (out / "teaching.json").write_text(client.get("/api/teaching").text)


if __name__ == "__main__":
    main()

"""The one scenario all three living-methods prototypes show (not part of the app; a review harness).

    TURBOTAB_HOME=$(mktemp -d) TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 \
        venv/bin/python -m turbotab.server --port 8961                  # repo root
    venv/bin/python turbotab/frontend/src/explore/methods-shared/scenario.py \
        --base http://127.0.0.1:8961                                     # prints the path and Table 2

Every prototype's capture script drives the real server through :func:`run_inference` and
:func:`run_prediction`, so all three record the same answers in the same order and show the same
sentences, guesses and numbers. A capture only watches: at each named moment its hook takes what
its prototype shows (the view, the methods, previews, cards). A hook may take previews and read
anything, but it never records a decision on the scenario's project; a prototype that needs other
branches (a hovered alternative's sentence, an edit after the lock) records them on a project of
its own (:func:`open_project` with the same opening, then any answers).

The scenario (SCENARIO.md says it in words):

* NHANES dietary export (``$TURBOTAB_NHANES_CSV``, else ``_tt_tmp_nhanes.csv``), the dietary
  lens, the outcome ``glucose`` (a regression), the purpose inference;
* the roles as proposed; every reading answered from the fixture's declared truth
  (``turbotab/core/tests/truths.py``): ``kcal``'s unit and days where the screens ask them, three
  role readings confirmed one at a time (:data:`SINGLES`), then one block of the rest, then the
  fit's code-or-amount readings in one block;
* every row kept as the primary; Willett 2013's and NHS/HPFS's sex-specific screens declared as
  secondary analyses (:data:`SENSITIVITY_KEYS`); complete cases; no holdout under inference;
* the exposure ``sugar``, its total effect, as a substitution (in place of other energy sources at
  fixed total energy), on the mean-difference scale;
* the adjustment set: the pack's guess for each group that has one, the seven unguessed covariates
  from the truth (the blood pressures, HDL, triglycerides and the two medications mediators, left
  out of a total effect; ``cycle_begin_year`` a confounder);
* the energy model the engine ranks first (its default; :func:`energy_default`);
* the declared model sequence with Model 1 adjusted for ``age``, ``gender`` and ``kcal``;
* the fit (the linear family), which locks the plan; Table 2, the secondary analyses.

The prediction variant: the same table, opening and readings under prediction, every row kept,
the engine's first missing-data method, a 20% holdout, the engine's first energy model, the linear
model and boosted trees.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from turbotab.core.tests.acceptance import server_drive as sd  # noqa: E402
from turbotab.core.tests.truths import answer_adjustment, fixture_truth  # noqa: E402

# ── the answers ──────────────────────────────────────────────────────────────

LENS = "dietary"
OUTCOME = "glucose"
TASK = "regression"
EXPOSURE = "sugar"
EFFECT = "total"
CONTRAST = "substitution"
MEASURE = "mean_difference"
ENERGY_COLUMN = "kcal"
# The energy-bearing exposures every energy model names (the roles' exposures, sugar first).
NUTRIENTS = ["sugar", "protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
# The screens reported beside the every-row primary (the two sex-specific ones the card offers).
SENSITIVITY_KEYS = ("willett_2013_by_sex", "nhs_hpfs_by_sex")
# Role readings confirmed one at a time before the block confirm unlocks (BLUEPRINT §11.4 rule 4):
# the first three the readings card lists.
SINGLES = ("bp_di", "bp_sys", "cycle_begin_year")
MODEL_1 = ["age", "gender", "kcal"]
INFERENCE_SPLIT = {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5}
PREDICTION_SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5}
INFERENCE_MODELS = ["linear"]
PREDICTION_MODELS = ["linear", "boosted_trees"]


def nhanes() -> Path:
    from turbotab.core.tests.stage_harness import NHANES

    return Path(os.environ.get("TURBOTAB_NHANES_CSV") or NHANES)


def energy(method: str) -> dict[str, Any]:
    if method == "none":
        return {"kind": "set_energy_adjustment", "method": "none"}
    return {"kind": "set_energy_adjustment", "method": method, "energy_column": ENERGY_COLUMN,
            "nutrients": list(NUTRIENTS)}


def estimand(exposure: str = EXPOSURE, contrast: str = CONTRAST) -> dict[str, Any]:
    return {"kind": "set_estimand", "exposure": exposure, "effect": EFFECT, "measure": MEASURE,
            "contrast": contrast}


def model_sequence() -> dict[str, Any]:
    return {"kind": "set_model_sequence", "exposure": EXPOSURE, "model_1": list(MODEL_1)}


def screen_rule(proposals: dict[str, Any], key: str) -> dict[str, Any]:
    return next(o["rule"] for o in proposals["exclusions"] if o["key"] == key)


def sensitivity(proposals: dict[str, Any], keys: tuple[str, ...] = SENSITIVITY_KEYS) -> dict[str, Any]:
    labels = {o["key"]: o["label"] for o in proposals["labels"]["exclusions"]["options"]}
    return {"kind": "set_sensitivity", "analyses": [
        {"label": labels[k], "rules": [screen_rule(proposals, k)]} for k in keys]}


def energy_default(proposals: dict[str, Any]) -> str:
    """The energy model the engine ranks first among those that can run on these columns."""
    return proposals["energy"]["ranking"]["order"][0]


# ── the drive ────────────────────────────────────────────────────────────────


class Client:
    """The acceptance drive's client surface (``get``/``post``) over a running server."""

    def __init__(self, base: str, timeout: float = 900):
        self.h = httpx.Client(base_url=base, timeout=timeout)

    def get(self, url: str, **kw: Any) -> httpx.Response:
        return self.h.get(url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> httpx.Response:  # noqa: A002
        return self.h.post(url, json=json, **kw)


Hook = Callable[[str, "Journey"], None]


class Journey:
    """One project on the scenario: the drive, and what a hook may read or preview."""

    def __init__(self, client: Client, purpose: str):
        self.c = client
        self.purpose = purpose
        self.truth = fixture_truth("_tt_tmp_nhanes.csv")
        self.drive = sd.open_project(client, nhanes(), self.truth)
        self.drive.exposure = EXPOSURE
        self.pid = self.drive.pid
        self.locked = False  # under inference the first estimate fetched presses Fit: the lock

    # reading (never records anything)
    def view(self) -> dict[str, Any]:
        return self.drive.view()

    def get(self, path: str) -> Any:
        r = self.c.get(f"/api/projects/{self.pid}{path}")
        r.raise_for_status()
        return r.json()

    def methods(self) -> dict[str, Any]:
        return self.get("/methods")

    def readings(self) -> dict[str, Any]:
        return self.get("/readings")

    def artifact(self, stage: str, timeout: float = 600) -> dict[str, Any]:
        from turbotab.core.estimand import ESTIMATE_STAGES

        assert self.locked or stage not in ESTIMATE_STAGES or self.purpose != "inference", (
            f"fetching {stage} presses Fit, which records the plan lock; only the scenario does")
        return self.drive.artifact(stage, timeout=timeout)

    def stage(self, stage: str) -> dict[str, Any] | None:
        """A stage's current answer (``{stage, key, fresh, status, artifact}``) or None."""
        from turbotab.core.estimand import ESTIMATE_STAGES

        if stage in ESTIMATE_STAGES and not self.locked and self.purpose == "inference":
            return None
        r = self.c.get(f"/api/projects/{self.pid}/stages/{stage}")
        return r.json() if r.status_code == 200 else None

    def preview(self, decision: dict[str, Any]) -> dict[str, Any]:
        self.wait_quiet()
        r = self.c.post(f"/api/projects/{self.pid}/preview", json=decision)
        return {"status": r.status_code, "decision": decision, "body": r.json()}

    def ask(self, key: str) -> dict[str, Any] | None:
        return next((s.get("ask") for s in self.view()["interview"] if s["key"] == key), None)

    def wait_quiet(self, timeout: float = 900) -> None:
        """Until no stage is queued or running (what a client shows is then settled)."""
        end = time.monotonic() + timeout
        while True:
            busy = [k for k, s in self.view()["stages"].items()
                    if s["status"] in ("queued", "running")]
            if not busy:
                return
            assert time.monotonic() < end, busy
            time.sleep(0.3)

    # recording (the scenario's answers only)
    def record(self, body: dict[str, Any]) -> dict[str, Any]:
        r = sd._post_when_reached(self.c, f"/api/projects/{self.pid}/decisions", body)
        assert r.status_code == 200, (body["kind"], r.text[:900])
        return max(r.json()["decisions"], key=lambda d: d["seq"])


def _opening(j: Journey, purpose: str) -> None:
    d = j.drive
    d.answer("lens", {"kind": "set_lens", "lenses": [LENS]})
    d.answer("target", {"kind": "set_target", "column": OUTCOME})
    d.artifact("target_info", timeout=600)
    if d.reach("task")["status"] in ("open", "waiting"):
        d.answer("task", {"kind": "set_task", "column": OUTCOME, "task": TASK})
    d.answer("purpose", {"kind": "set_purpose", "purpose": purpose})


def _roles(j: Journey) -> dict[str, str]:
    props = j.drive.artifact("roles")["columns"]
    roles = {c["column"]: c["proposed"] for c in props}
    for column, role in roles.items():
        j.truth.setdefault(f"role:{column}", role)
    return roles


def _unit(j: Journey) -> dict[str, Any]:
    return {"kind": "set_column_unit", "column": ENERGY_COLUMN,
            "unit": j.truth.answer("unit", ENERGY_COLUMN),
            "days": int(j.truth.answer("day_count", ENERGY_COLUMN))}


def _reach(j: Journey, key: str, timeout: float = 600) -> dict[str, Any]:
    step = j.drive.reach(key, timeout=timeout)
    j.wait_quiet()
    return step


def _settle(j: Journey, key: str, at: Hook) -> None:
    """Answer the readings the open question ``key`` asks about, from the truth, as the scenario
    does: role readings three at a time singly (:data:`SINGLES`, the first three the card lists),
    then the card's block confirm, which lists exactly the readings still open; any other reading
    (the fit's codes or amounts) in the card's one block."""
    ask = j.ask(key)
    if not ask:
        return
    roles = [g for g in ask["groups"] if g["kind"] == "role"]
    if roles:
        listed = [g["columns"][0] for g in roles if len(g["columns"]) == 1][:len(SINGLES)]
        assert tuple(listed) == SINGLES, (listed, SINGLES)
        for column in SINGLES:
            j.record({"kind": "confirm_reading", "reading": "role", "column": column,
                      "value": j.truth[f"role:{column}"]})
            j.wait_quiet()
            at(f"single:{column}", j)
        ask = j.ask(key)
        if not ask:
            return
    for _ in range(3):  # each block settles what its card lists; a later card may ask the rest
        block = ask["exits"][0]["decision"]
        assert block["kind"] == "confirm_readings", ask["exits"][0]
        for item in block["items"]:
            assert j.truth.answer(item["reading"], item["column"]) == item["value"], item
        j.record(block)
        j.wait_quiet()
        ask = j.ask(key)
        if not ask:
            return
    raise AssertionError(f"{key} still asks: {ask}")


def run_inference(client: Client, at: Hook = lambda moment, j: None) -> Journey:
    """The scenario under inference, to the locked plan and its results. ``at(moment, journey)``
    is called at each moment, with the question open (or the results fresh):

    draft · roles · unit_ask · exclusions · missing · split · readings · single:<column> ×3 ·
    estimand · adjustment · energy · model_sequence · models · ready · locked
    """
    j = Journey(client, "inference")
    _opening(j, "inference")
    _reach(j, "roles")
    at("draft", j)

    j.record({"kind": "set_roles", "roles": _roles(j)})
    j.wait_quiet()
    at("roles", j)

    step = _reach(j, "exclusions")
    if step.get("ask"):
        at("unit_ask", j)
        j.record(_unit(j))
    _reach(j, "exclusions")
    proposals = j.artifact("proposals")
    at("exclusions", j)
    j.record({"kind": "set_exclusions", "rules": []})
    j.record(sensitivity(j.artifact("proposals")))

    _reach(j, "missing")
    at("missing", j)
    j.record({"kind": "set_missing", "strategy": "complete_case"})

    _reach(j, "split")
    at("split", j)
    j.record(dict(INFERENCE_SPLIT))

    _reach(j, "estimand")
    at("readings", j)
    _settle(j, "estimand", at)
    _reach(j, "estimand")
    at("estimand", j)
    j.record(estimand())

    _reach(j, "adjustment")
    end = time.monotonic() + 300
    while (j.artifact("proposals").get("adjustment") or {}).get("exposure") != EXPOSURE:
        assert time.monotonic() < end
        time.sleep(0.3)
    j.wait_quiet()
    at("adjustment", j)
    card = j.artifact("proposals")["adjustment"]
    answer_adjustment(lambda d: client.post(f"/api/projects/{j.pid}/decisions", json=d), card,
                      j.truth)

    _reach(j, "energy_adjustment")
    proposals = j.artifact("proposals")
    at("energy", j)
    j.record(energy(energy_default(proposals)))
    j.wait_quiet()

    at("model_sequence", j)
    j.record(model_sequence())
    j.wait_quiet()

    _reach(j, "models")
    at("models", j)
    _settle(j, "models", at)
    j.record({"kind": "select_models", "models": list(INFERENCE_MODELS)})
    j.wait_quiet()
    at("ready", j)

    j.locked = True
    j.artifact("fit", timeout=1200)
    for st in ("effects", "secondary", "sensitivity"):
        j.artifact(st, timeout=1200)
    j.wait_quiet()
    at("locked", j)
    return j


def run_prediction(client: Client, at: Hook = lambda moment, j: None) -> Journey:
    """The prediction variant, to the fit. Moments: draft · roles · exclusions · missing · split ·
    energy · models · fitted."""
    j = Journey(client, "prediction")
    j.locked = True  # no plan lock under prediction
    _opening(j, "prediction")
    _reach(j, "roles")
    at("draft", j)
    j.record({"kind": "set_roles", "roles": _roles(j)})
    j.wait_quiet()
    at("roles", j)
    step = _reach(j, "exclusions")
    if step.get("ask"):
        j.record(_unit(j))
    _reach(j, "exclusions")
    at("exclusions", j)
    j.record({"kind": "set_exclusions", "rules": []})
    _reach(j, "missing")
    at("missing", j)
    first = j.artifact("proposals")["missing"]["methods"][0]["decision"]
    j.record({"kind": "set_missing", **first})
    _reach(j, "split")
    at("split", j)
    j.record(dict(PREDICTION_SPLIT))
    _reach(j, "energy_adjustment")
    _settle(j, "energy_adjustment", at)
    at("energy", j)
    j.record(energy(energy_default(j.artifact("proposals"))))
    _reach(j, "models")
    at("models", j)
    _settle(j, "models", at)
    j.record({"kind": "select_models", "models": list(PREDICTION_MODELS)})
    j.artifact("fit", timeout=1800)
    j.wait_quiet()
    at("fitted", j)
    return j


# ── what every prototype must show the same ──────────────────────────────────


def table2(effects: dict[str, Any]) -> list[dict[str, Any]]:
    """Table 2's rows for the exposure: each model in the declared sequence, its estimate and
    interval, as the effects stage serves them."""
    rows = []
    fam = effects["families"][0]
    for s in fam["sequence"]:
        e = next(x for x in s["effects"] if x.get("feature", EXPOSURE) == EXPOSURE) \
            if s.get("effects") else None
        rows.append({"model": s["key"], "label": s.get("label"),
                     "estimate": e and e.get("estimate"), "ci": e and [e.get("ci_low"), e.get("ci_high")],
                     "n": s.get("n_rows")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8961")
    ap.add_argument("--out", default=None, help="write the path's moments and Table 2 here")
    args = ap.parse_args()
    client = Client(args.base)
    log: list[dict[str, Any]] = []

    def at(moment: str, j: Journey) -> None:
        v = j.view()
        first = next((s for s in v["interview"] if s["status"] in ("open", "waiting")), None)
        log.append({"moment": moment, "decisions": len(v["decisions"]),
                    "open": first and first["key"], "ask": bool(first and first.get("ask"))})
        print(f"{moment:24s} {len(v['decisions']):3d} decisions · open {first and first['key']}"
              f"{' (ask)' if first and first.get('ask') else ''}", flush=True)

    t = time.time()
    j = run_inference(client, at)
    effects = j.artifact("effects")
    rows = table2(effects)
    print(json.dumps(rows, indent=1))
    print(f"inference done in {time.time() - t:.0f}s")
    if args.out:
        Path(args.out).write_text(json.dumps({"log": log, "table2": rows,
                                              "effects": effects}, indent=1, default=str))


if __name__ == "__main__":
    main()

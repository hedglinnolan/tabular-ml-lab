"""Capture the real-data fixtures behind the M3 mock API (``npm run dev:mock`` only).

Every shape in ``turbotab/frontend/src/mocks/fixtures/m3-*.json`` is what a running TurboTab server
answered on a reference journey, through its own HTTP routes. The driver follows the Router: it
answers whatever question it opens, in its order, the way the journey's author would, and every
reading the server asks about is answered from the fixture's declared truth
(``turbotab/core/tests/truths.py``; BLUEPRINT §14.3: never a constant). Refusals that ask for
something other than readings take their first exit, as a person pressing it would.

For each journey the file holds:

* ``snapshots``: the ProjectView after each recorded decision, once no stage is queued or running
  (its ``decisions`` are ids into ``log``; stage statuses lose ``updated_at``, which the mock stamps);
* ``artifacts``: every fresh stage artifact the journey produced, once each, by ``stage@key``;
* ``records``: every decision posted, the snapshot it was posted at, and what the server answered:
  the snapshot it moved to (200) or the refusal with its exits (409);
* ``endpoints``: at the end, ``GET`` readings, methods, plan, columns, a table window and the
  files list; the ``assembly`` journey adds the join preview and the codebook previews.

**Trimming rule** (also written to each file's ``meta.trim``): a list longer than 60 items keeps its
first 60; an object with more than 120 keys keeps its first 120 (insertion order); floats keep six
significant digits. Interview steps and the decision log are never trimmed, except a ``set_roles``
or ``confirm_readings`` payload over the same limits. Each trimmed path is listed with its original
length under ``meta.trimmed``.

Run from the repository root against a server of your own (about 15 minutes on two workers):

    TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 TURBOTAB_HOME=$(mktemp -d) \\
        venv/bin/python -m turbotab.server --port 8873
    venv/bin/python docs/turbotab-next/m3/capture_fixtures.py --base http://127.0.0.1:8873 [journey …]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from turbotab.core.tests.acceptance.server_drive import (  # noqa: E402
    Drive,
    answer_wp17,
    settle_post,
)
from turbotab.core.tests.truths import Truth, fixture_truth  # noqa: E402

SAMPLES = ROOT / "turbotab" / "sample_data"
OUT = ROOT / "turbotab" / "frontend" / "src" / "mocks" / "fixtures"
NHANES = Path("/Users/nhedglin/tabular-ml-lab/_tt_tmp_nhanes.csv")
if not NHANES.is_file():
    NHANES = ROOT / "_tt_tmp_nhanes.csv"

MAX_LIST = 60
MAX_KEYS = 120
TRIM_RULE = (f"lists over {MAX_LIST} items keep their first {MAX_LIST}; objects over {MAX_KEYS} "
             f"keys keep their first {MAX_KEYS}; floats keep 6 significant digits")


# ── trimming ──────────────────────────────────────────────────────────────────


def _pattern(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def trim(obj: Any, path: str, trimmed: dict[str, int]) -> Any:
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if not math.isfinite(obj) else float(f"{obj:.6g}")
    if isinstance(obj, dict):
        items = list(obj.items())
        if len(items) > MAX_KEYS:
            p = _pattern(path)
            trimmed[p] = max(trimmed.get(p, 0), len(items))
            items = items[:MAX_KEYS]
        return {k: trim(v, f"{path}.{k}", trimmed) for k, v in items}
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if len(items) > MAX_LIST:
            p = _pattern(path)
            trimmed[p] = max(trimmed.get(p, 0), len(items))
            items = items[:MAX_LIST]
        return [trim(v, f"{path}[{i}]", trimmed) for i, v in enumerate(items)]
    return obj


def canon(decision: Any) -> str:
    return json.dumps(decision, sort_keys=True, separators=(",", ":"))


# ── packing: each version as a patch on the one before ─────────────────────────

DELETE = {"$del": 1}  # a key the newer version no longer has (null stays a value)
PACKING = ("artifacts[stage] = {keys, base, patches}: version i is base with patches[0..i-1] "
           "applied in turn; snapshots after the first carry `patch` instead of `view`. A patch "
           "merges objects key by key, replaces anything else whole, and {\"$del\": 1} removes a "
           "key (src/mocks/m3.ts applyPatch).")


def diff(a: Any, b: Any) -> Any:
    """The patch that turns ``a`` into ``b``: only the keys that changed, objects recursively."""
    out: dict[str, Any] = {}
    for k in a:
        if k not in b:
            out[k] = DELETE
    for k, v in b.items():
        if k not in a:
            out[k] = v
        elif a[k] != v:
            out[k] = diff(a[k], v) if isinstance(a[k], dict) and isinstance(v, dict) else v
    return out


def pack(out: dict[str, Any]) -> dict[str, Any]:
    by_stage: dict[str, list[tuple[str, Any]]] = {}
    for sk, art in out["artifacts"].items():
        stage, key = sk.split("@", 1)
        by_stage.setdefault(stage, []).append((key, art))
    packed: dict[str, Any] = {}
    for stage, versions in by_stage.items():
        patches = [diff(versions[i - 1][1], versions[i][1]) for i in range(1, len(versions))]
        packed[stage] = {"keys": [k for k, _ in versions], "base": versions[0][1],
                         "patches": patches}
    views = [s["view"] for s in out["snapshots"]]
    snapshots = []
    for i, s in enumerate(out["snapshots"]):
        rest = {k: v for k, v in s.items() if k != "view"}
        snapshots.append({**rest, "view": views[0]} if i == 0
                         else {**rest, "patch": diff(views[i - 1], views[i])})
    return {**out, "meta": {**out["meta"], "packing": PACKING}, "artifacts": packed,
            "snapshots": snapshots}


# ── the recording client ──────────────────────────────────────────────────────


class Recorder:
    """An httpx client the acceptance drivers use as their ``client``: every decision posted
    through it is recorded by the journey being captured."""

    def __init__(self, base: str):
        self.h = httpx.Client(base_url=base, timeout=1800)
        self.journey: "Journey | None" = None

    def get(self, url: str, **kw: Any) -> httpx.Response:
        return self.h.get(url, **kw)

    def post(self, url: str, json: Any = None, **kw: Any) -> httpx.Response:  # noqa: A002
        r = self.h.post(url, json=json, **kw)
        if self.journey is not None and url.endswith("/decisions"):
            self.journey.recorded(json, r)
        return r


class Journey:
    def __init__(self, name: str, label: str, client: Recorder, source: str):
        self.name, self.label, self.c, self.source = name, label, client, source
        self.pid = ""
        self.snapshots: list[dict[str, Any]] = []
        self.artifacts: dict[str, Any] = {}
        self.records: list[dict[str, Any]] = []
        self.log: dict[str, Any] = {}
        self.endpoints: dict[str, Any] = {}
        self.trimmed: dict[str, int] = {}
        self.notes: list[str] = []
        self.started = time.monotonic()

    # the view once nothing is queued or running
    def quiet(self, timeout: float = 1800.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        calm = 0
        while True:
            view = self.c.get(f"/api/projects/{self.pid}").json()
            busy = [n for n, s in view["stages"].items() if s["status"] in ("queued", "running")]
            calm = calm + 1 if not busy else 0
            if calm >= 2 or time.monotonic() > end:
                if busy:
                    self.notes.append(f"snapshot {len(self.snapshots)} taken with {busy} busy")
                return view
            time.sleep(0.25)

    def snapshot(self, after: Any = None) -> int:
        view = self.quiet()
        for name, s in view["stages"].items():
            key = s.get("key")
            if s["status"] != "fresh" or not key or f"{name}@{key}" in self.artifacts:
                continue
            res = self.c.get(f"/api/projects/{self.pid}/stages/{name}").json()
            if res.get("fresh") and res.get("key") == key and res.get("artifact") is not None:
                self.artifacts[f"{name}@{key}"] = trim(res["artifact"], f"artifact:{name}",
                                                       self.trimmed)
        for rec in view["decisions"]:
            if rec["id"] not in self.log:
                self.log[rec["id"]] = trim(rec, "log", self.trimmed)
        stages = {n: {k: v for k, v in s.items() if k != "updated_at"}
                  for n, s in view["stages"].items()}
        compact = {"summary": view["summary"], "state": trim(view["state"], "state", self.trimmed),
                   "decisions": [r["id"] for r in view["decisions"]], "stages": stages,
                   "interview": [trim(st, f"interview.{st['key']}", self.trimmed)
                                 for st in view["interview"]]}
        open_step = next((st["key"] for st in view["interview"] if st["status"] == "open"), None)
        first = next((st for st in view["interview"] if st["status"] in ("open", "waiting")), None)
        self.snapshots.append({"open": open_step, "first": first["key"] if first else None,
                               "after": after["kind"] if isinstance(after, dict) else None,
                               "view": compact})
        print(f"  [{self.name}] snapshot {len(self.snapshots) - 1}: open={open_step} "
              f"after={after.get('kind') if isinstance(after, dict) else None}", flush=True)
        return len(self.snapshots) - 1

    def recorded(self, decision: Any, response: httpx.Response) -> None:
        here = len(self.snapshots) - 1
        if response.status_code == 200:
            to = self.snapshot(after=decision)
            self.records.append({"from": here, "decision": trim(decision, "decision", self.trimmed),
                                 "status": 200, "to": to})
            return
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"code": "unparsed", "message": response.text[:400], "exits": []}}
        last = self.records[-1] if self.records else None
        if (last and last["status"] == response.status_code and last["from"] == here
                and canon(last["decision"]) == canon(decision)):
            return  # the same refusal again (a driver waiting behind "not yet")
        self.records.append({"from": here, "decision": trim(decision, "decision", self.trimmed),
                             "status": response.status_code,
                             "body": trim(body, "refusal", self.trimmed)})

    def capture_endpoints(self) -> None:
        base = f"/api/projects/{self.pid}"
        for name, url in (("readings", f"{base}/readings"), ("methods", f"{base}/methods"),
                          ("plan", f"{base}/plan"), ("columns", f"{base}/columns"),
                          ("table", f"{base}/table?offset=0&limit=40"),
                          ("files", f"{base}/files")):
            r = self.c.get(url)
            body: Any
            try:
                body = r.json()
            except ValueError:
                body = r.text[:2000]
            self.endpoints[name] = {"status": r.status_code,
                                    "body": trim(body, f"endpoint:{name}", self.trimmed)}

    def write(self) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        out = {
            "meta": {
                "journey": self.name, "label": self.label, "source": self.source,
                "captured": time.strftime("%Y-%m-%d"), "seconds": round(time.monotonic() - self.started),
                "script": "docs/turbotab-next/m3/capture_fixtures.py", "trim": TRIM_RULE,
                "trimmed": dict(sorted(self.trimmed.items())), "notes": self.notes,
            },
            "start": 0,
            "snapshots": self.snapshots,
            "artifacts": self.artifacts,
            "records": self.records,
            "log": self.log,
            "endpoints": self.endpoints,
        }
        path = OUT / f"m3-{self.name}.json"
        path.write_text(json.dumps(pack(out), separators=(",", ":"), ensure_ascii=False) + "\n")
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)", flush=True)
        return path


# ── following the Router ──────────────────────────────────────────────────────


def post_answer(j: Journey, d: Drive, body: dict[str, Any], prefer: Callable[[list], Any] | None = None,
                depth: int = 0) -> dict[str, Any]:
    """Post ``body``; readings are answered from the truth (``settle_post``); any other refusal
    takes its preferred exit (the first with a decision unless ``prefer`` picks one)."""
    r = settle_post(d.c, d.pid, body, d.truth, unblock=lambda: d.answer_wp17_before(body))
    if r.status_code == 200:
        return r.json()
    error = (r.json() or {}).get("error") or {}
    exits = [e for e in error.get("exits") or [] if e.get("decision")]
    if r.status_code != 409 or not exits or depth >= 4:
        raise RuntimeError(f"{body.get('kind')} refused: {r.status_code} {r.text[:800]}")
    chosen = (prefer(exits) if prefer else None) or exits[0]
    j.notes.append(f"{body.get('kind')} refused ({error.get('code')}); took the exit "
                   f"'{chosen['label']}'")
    return post_answer(j, d, chosen["decision"], prefer, depth + 1)


def artifact(d: Drive, stage: str) -> dict[str, Any]:
    return d.artifact(stage, timeout=1800)


def default_answer(j: Journey, d: Drive, spec: dict[str, Any], key: str, step: dict[str, Any],
                   view: dict[str, Any]) -> dict[str, Any] | None:
    state = view["state"]
    target = state.get("target")
    given = spec.get("answers", {}).get(key)
    if callable(given):
        return given(j, d, step, view)
    if given is not None:
        return given
    if key == "lens":
        return {"kind": "set_lens", "lenses": spec["lens"]}
    if key == "orientation":
        return {"kind": "set_orientation", "orientation": "sample_major"}
    if key == "target":
        return {"kind": "set_target", "column": spec["target"]}
    if key == "event":
        return {"kind": "set_event", "column": target, "level": str(spec["event"])}
    if key == "task":
        if step.get("followup"):
            d.task_followups()
            return None
        ti = artifact(d, "target_info")
        return {"kind": "set_task", "column": target, "task": ti["task"]}
    if key == "follow_up":
        task = state.get("task") or artifact(d, "target_info").get("task")
        if task == "time_to_event":
            return {"kind": "set_follow_up", "column": target, "time_column": spec["follow_up_time"]}
        return {"kind": "set_censoring", "column": target}
    if key == "purpose":
        return {"kind": "set_purpose", "purpose": spec["purpose"]}
    if key == "repeat_kind":
        structure = artifact(d, "structure")
        reading = (structure.get("repeats") or {}).get("reading") or "repeats"
        return {"kind": "set_repeat_kind", "repeat_kind": reading,
                "time_column": structure.get("time_column") if reading == "time_points" else None}
    if key == "unit":
        return {"kind": "set_unit", "unit": "unit"}
    if key == "aggregation":
        menu = artifact(d, "structure").get("aggregation") or {}
        return {"kind": "set_aggregation", "method": menu.get("recommended") or "mean"}
    if key == "temporal":
        return {"kind": "set_temporal", "temporal": False}
    if key == "roles":
        roles = spec.get("roles")
        if roles is None:
            proposals = artifact(d, "roles")
            roles = {c["column"]: c["proposed"] for c in proposals["columns"]}
        d.decide_roles(roles)
        return None
    if key in ("clusters", "estimand", "adjustment"):
        answer_wp17(d, key, exposure=spec.get("exposure"))
        return None
    if key == "survey":
        survey = artifact(d, "proposals").get("survey") or {}
        return (survey.get("options") or [{}])[0].get("decision") or {
            "kind": "set_survey", "estimand": "sample"}
    if key == "exclusions":
        return {"kind": "set_exclusions", "rules": []}
    if key == "missing":
        return {"kind": "set_missing", "strategy": "complete_case"}
    if key == "split":
        holdout = 0.2 if state.get("purpose") == "prediction" else 0.0
        return {"kind": "set_split", "holdout": holdout, "seed": 0, "folds": 5}
    if key == "time_varying":
        answer_wp17(d, key, exposure=spec.get("exposure"))
        return None
    if key == "energy_adjustment":
        reading = artifact(d, "proposals").get("energy") or {}
        return {"kind": "set_energy_adjustment", "method": "residual",
                "energy_column": reading.get("energy_column"), "nutrients": reading.get("nutrients")}
    if key == "causal":
        return {"kind": "set_causal", "exposure": (state.get("estimand") or {}).get("exposure"),
                "method": "none"}
    if key == "models":
        shelf = artifact(d, "shelf")
        return {"kind": "select_models", "models": [shelf["families"][0]["key"]]}
    if key == "substitution":
        pair = artifact(d, "design")["substitution_pairs"][0]
        return {"kind": "set_substitution", "donor": pair["donor"], "recipient": pair["recipient"],
                "step_kcal": 100}
    if key == "open_seal":
        return {"kind": "open_seal"}
    raise RuntimeError(f"no answer for {key}")


def follow(j: Journey, d: Drive, spec: dict[str, Any]) -> None:
    end = time.monotonic() + spec.get("timeout", 3600)
    until: Callable[[dict[str, Any]], bool] = spec.get("until") or (lambda v: False)
    before: dict[str, Callable[..., None]] = spec.get("before", {})
    done_before: set[str] = set()
    stalled = 0
    while time.monotonic() < end:
        view = d.view()
        if until(view):
            print(f"  [{j.name}] done: until holds", flush=True)
            return
        steps = view["interview"]
        first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
        if first is None:
            j.quiet()
            if until(d.view()) or not spec.get("until"):
                return
            time.sleep(1.0)
            stalled += 1
            if stalled > 600:
                j.notes.append("stopped: nothing open and the end condition never held")
                return
            continue
        key = first["key"]
        if first["status"] == "waiting":
            failed = [w for w in first.get("waiting_on") or []
                      if view["stages"].get(w, {}).get("status") == "error"]
            if failed:
                j.notes.append(f"stopped: {key} waits on failed {failed}")
                return
            time.sleep(0.3)
            continue
        stalled = 0
        if key in before and key not in done_before:
            done_before.add(key)
            before[key](j, d, view)
            continue
        body = default_answer(j, d, spec, key, first, view)
        if body is not None:
            print(f"  [{j.name}] answering {key}: {body.get('kind')}", flush=True)
            post_answer(j, d, body, spec.get("prefer", {}).get(key))
    j.notes.append("stopped: the journey ran out of time")


def run_journey(client: Recorder, spec: dict[str, Any]) -> Path:
    name = spec["name"]
    print(f"── {name}: {spec['label']}", flush=True)
    path = spec["path"]() if callable(spec["path"]) else spec["path"]
    j = Journey(name, spec["label"], client, Path(path).name if not spec.get("source") else spec["source"])
    client.journey = None
    r = client.post("/api/projects", json={"path": str(path)})
    r.raise_for_status()
    j.pid = r.json()["id"]
    truth = spec["truth"]() if callable(spec.get("truth")) else (spec.get("truth") or Truth())
    d = Drive(client, j.pid, truth)
    d.exposure = spec.get("exposure")
    d.artifact("ingest", timeout=600)
    j.snapshot()
    client.journey = j
    try:
        follow(j, d, spec)
        if spec.get("after"):
            spec["after"](j, d)
    except Exception as exc:  # noqa: BLE001 - a journey that stops is written with its reason
        j.notes.append(f"stopped by {type(exc).__name__}: {str(exc)[:600]}")
        traceback.print_exc()
    finally:
        client.journey = None
    j.quiet()
    j.capture_endpoints()
    for extra in spec.get("extra_endpoints", []):
        extra(j, d)
    return j.write()


# ── the journeys ──────────────────────────────────────────────────────────────

NUTRIENTS = ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
NHANES_COVARIATES = ["age", "gender", "cycle_begin_year", *NUTRIENTS, "weight", "height", "bmi",
                     "waist", "bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp", "meds_chol"]
NHANES_ROLES = {"SEQN": "identifier", "sugar": "exposure", "kcal": "energy",
                **{c: "exposure" for c in NUTRIENTS},
                **{c: "covariate" for c in NHANES_COVARIATES if c not in NUTRIENTS},
                **{c: "flag" for c in ("imputed_weight", "imputed_height", "imputed_bmi",
                                       "imputed_waist", "imputed_bp_sys", "imputed_bp_di")}}


def nhanes_truth() -> Truth:
    """The export's declared truth (``truths.FIXTURE_TRUTHS``), and the nutrients' units as the
    NHANES documentation states them: every ``DR1T*`` macronutrient and sugar total is in grams."""
    return Truth({**fixture_truth("_tt_tmp_nhanes.csv"),
                  **{f"unit:{c}": "g" for c in ("sugar", *NUTRIENTS)},
                  # sugars are part of total carbohydrate, the fatty acids part of total fat
                  "nested_in:sugar": "carb",
                  **{f"nested_in:{c}": "fat_total" for c in ("fat_sat", "fat_mon", "fat_poly")}},
                 fixture="_tt_tmp_nhanes.csv")


def survey_design_table() -> Path:
    """The audit's informative-weight table (``test_wp10_survey_design.informative_tables``, I4):
    a 50/50 sample of a 10/90 population, the weighted fiber slope positive, the unweighted one
    negative."""
    from turbotab.core.tests.acceptance.test_wp10_survey_design import informative_tables

    folder = Path(tempfile.mkdtemp(prefix="m3-survey-design-"))
    return informative_tables(folder)["I4"]


def survey_design_truth() -> Truth:
    from turbotab.core.tests.acceptance.test_wp10_survey_design import TABLE_TRUTH

    return Truth(TABLE_TRUTH, fixture="survey.csv (WP10 I4)")


def stage_fresh(*stages: str) -> Callable[[dict[str, Any]], bool]:
    def holds(view: dict[str, Any]) -> bool:
        first = next((s for s in view["interview"] if s["status"] in ("open", "waiting")), None)
        return first is None and all(view["stages"].get(s, {}).get("status")
                                     not in ("queued", "running", "stale") for s in stages)
    return holds


def apply_first_repair(pattern: str) -> Callable[[Journey, Drive, dict[str, Any]], None]:
    """Before the outcome question: apply the first repair option of the findings whose id
    matches ``pattern`` (the repairs come before the outcome, OPENING_SEQUENCE §01)."""

    def go(j: Journey, d: Drive, view: dict[str, Any]) -> None:
        findings = artifact(d, "findings")["findings"]
        j.notes.append("findings with repairs: " + ", ".join(
            f["id"] for f in findings if f.get("repairs")))
        for f in findings:
            if not re.search(pattern, f["id"], re.I) or not f.get("repairs"):
                continue
            option = f["repairs"][0]
            post_answer(j, d, {"kind": "apply_repair", "finding_id": f["id"],
                               "option": option["key"]})
    return go


def causal_cohort() -> Path:
    from turbotab.core.tests.acceptance.test_causal_lane import cohort

    path = Path(tempfile.mkdtemp(prefix="m3-causal-")) / "causal_cohort.csv"
    cohort(path)
    return path


def causal_truth() -> Truth:
    from turbotab.core.tests.acceptance.test_causal_lane import cohort_truth

    return cohort_truth()


def time_varying_cohort() -> Path:
    from turbotab.core.tests.acceptance.timevary_fixtures import feedback_cohort

    path = Path(tempfile.mkdtemp(prefix="m3-timevary-")) / "dash_cohort.csv"
    feedback_cohort(n=600).to_csv(path, index=False)
    return path


def time_varying_roles() -> dict[str, str]:
    from turbotab.core.tests.acceptance.test_timevary import feedback_state

    return dict(feedback_state().roles)


def declare_causal(j: Journey, d: Drive, view: dict[str, Any]) -> None:
    """The causal lane, asked for ("Ask me anyway"), before the models: TMLE for the yes/no
    exposure, its assumptions refused until declared, positivity's trimming exit taken."""
    exposure = (view["state"].get("estimand") or {}).get("exposure")
    d.artifact("causal_design", timeout=1800)
    post_answer(j, d, {"kind": "set_causal", "exposure": exposure, "method": "tmle",
                       "learner": "linear"})


def time_varying_answer(j: Journey, d: Drive, step: dict[str, Any], view: dict[str, Any]) -> None:
    """The weights' lane: the card read first, the lane without its truncation (diagnostics
    first), then the truncation declared after them."""
    d.artifact("time_varying", timeout=1800)
    exposure = (view["state"].get("estimand") or {}).get("exposure")
    body = {"kind": "set_time_varying", "exposure": exposure, "method": "msm_iptw",
            "ordering": "exposure_precedes_outcome"}
    post_answer(j, d, body)
    j.quiet()
    d.artifact("time_varying", timeout=1800)
    if d.reach("time_varying", timeout=600)["status"] in ("open", "waiting"):
        post_answer(j, d, {**body, "truncation": "p1_p99"})
    return None


def declare(*bodies: dict[str, Any]) -> Callable[[Journey, Drive, dict[str, Any]], None]:
    """A hook that declares what no Router question asks (an explanation, a sensitivity analysis,
    a scale, a measurement-error correction) so its stage runs; a refusal it cannot take an exit
    from is noted, and the journey goes on."""

    def go(j: Journey, d: Drive, view: dict[str, Any]) -> None:
        for body in bodies:
            try:
                post_answer(j, d, body)
            except Exception as exc:  # noqa: BLE001 - the stage is extra; the journey continues
                j.notes.append(f"{body['kind']} not recorded: {str(exc)[:300]}")
    return go


def recall_table_path() -> Path:
    """The recall table the calibration tests generate (``test_wp12c_calibration.recall_table``):
    repeated 24-hour recalls of protein and energy, LDL the outcome."""
    from turbotab.core.tests.acceptance.test_wp12c_calibration import recall_table

    path = Path(tempfile.mkdtemp(prefix="m3-recalls-")) / "recalls.csv"
    recall_table().to_csv(path, index=False)
    return path


def recall_truth() -> Truth:
    from turbotab.core.tests.acceptance.test_wp12c_calibration import recall_truth as truth

    return truth()


SURVEY_ITEMS = [f"item_{i:02d}" for i in range(1, 11)]


def capture_assembly(client: Recorder) -> Path:
    """The DATAIN surfaces: an XPT table, a second XPT file added and its join previewed, and the
    codebooks (an NHANES codebook page, the table's own XPT labels, an uploaded codebook)."""
    data = ROOT / "turbotab/core/tests/acceptance/datain_data"
    tmp = Path(tempfile.mkdtemp(prefix="m3-assembly-"))
    paths = {}
    for name in ("DEMO_J.XPT", "DR1TOT_J.XPT", "DEMO_J.htm"):
        out = tmp / name
        with gzip.open(data / f"{name}.gz", "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        paths[name] = out
    spec = {"name": "assembly", "label": "NHANES XPT: a file joined, codebooks read",
            "path": paths["DEMO_J.XPT"], "truth": Truth(fixture="DEMO_J.XPT"),
            "lens": ["dietary"], "until": lambda v: True}
    j = Journey("assembly", spec["label"], client, "DEMO_J.XPT")
    r = client.post("/api/projects", json={"path": str(paths["DEMO_J.XPT"])})
    r.raise_for_status()
    j.pid = r.json()["id"]
    d = Drive(client, j.pid, Truth(fixture="DEMO_J.XPT"))
    d.artifact("ingest", timeout=600)
    j.snapshot()
    base = f"/api/projects/{j.pid}"

    def keep(name: str, r: httpx.Response) -> Any:
        try:
            body = r.json()
        except ValueError:
            body = r.text[:2000]
        j.endpoints[name] = {"status": r.status_code, "body": trim(body, f"endpoint:{name}", j.trimmed)}
        return body

    added = keep("files_add", client.post(f"{base}/files", json={"path": str(paths["DR1TOT_J.XPT"])}))
    keep("files_list", client.get(f"{base}/files"))
    if isinstance(added, dict) and added.get("id"):
        keep("join_preview", client.post(f"{base}/join-preview",
                                         json={"file": added["id"], "on": "SEQN", "how": "left"}))
    keep("codebook_nhanes", client.post(f"{base}/codebooks", json={"path": str(paths["DEMO_J.htm"])}))
    keep("codebook_labels", client.post(f"{base}/codebooks", json={"labels": True}))
    with open(paths["DEMO_J.htm"], "rb") as fh:
        keep("codebook_upload", client.post(f"{base}/codebooks/upload",
                                            files={"file": ("DEMO_J.htm", fh, "text/html")}))
    with open(paths["DR1TOT_J.XPT"], "rb") as fh:
        keep("files_upload", client.post(f"{base}/files/upload",
                                         files={"file": ("DR1TOT_J.XPT", fh, "application/octet-stream")}))
    client.journey = j
    try:
        post_answer(j, d, {"kind": "set_lens", "lenses": ["dietary"]})
        preview = j.endpoints.get("join_preview", {}).get("body") or {}
        if isinstance(preview, dict) and preview.get("sentence") and not preview.get("refusal"):
            post_answer(j, d, {"kind": "join_files", "file": added["id"], "on": "SEQN", "how": "left"})
        cb = j.endpoints.get("codebook_nhanes", {}).get("body") or {}
        if isinstance(cb, dict) and cb.get("id"):
            post_answer(j, d, {"kind": "import_codebook", "codebook": cb["id"]})
    except Exception as exc:  # noqa: BLE001
        j.notes.append(f"stopped by {type(exc).__name__}: {str(exc)[:600]}")
        traceback.print_exc()
    finally:
        client.journey = None
    j.quiet()
    j.capture_endpoints()
    return j.write()


JOURNEYS: dict[str, dict[str, Any]] = {
    "nhanes-inference": {
        "label": "NHANES dietary under inference: sugar and fasting glucose, to the effects table",
        "path": NHANES, "source": "_tt_tmp_nhanes.csv",
        "truth": nhanes_truth,
        "lens": ["dietary"], "target": "glucose", "purpose": "inference", "exposure": "sugar",
        "roles": NHANES_ROLES,
        "answers": {
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
            "energy_adjustment": {"kind": "set_energy_adjustment", "method": "standard",
                                  "energy_column": "kcal", "nutrients": ["sugar", *NUTRIENTS]},
            "models": {"kind": "select_models", "models": ["linear"]},
        },
        "until": stage_fresh("fit", "effects"),
    },
    "nhanes-prediction": {
        "label": "NHANES dietary under prediction: SAS zeros repaired, to the seal opened once",
        "path": NHANES, "source": "_tt_tmp_nhanes.csv",
        "truth": nhanes_truth,
        "lens": ["dietary"], "target": "glucose", "purpose": "prediction",
        "before": {"target": apply_first_repair(r"^sas_zeros")},
        "answers": {
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
            "models": {"kind": "select_models", "models": ["linear", "boosted_trees"]},
        },
        "until": stage_fresh("fit"),
    },
    "metabolomics": {
        "label": "Untargeted metabolomics with pooled QCs under prediction",
        "path": SAMPLES / "metabolomics_untargeted.csv",
        "truth": lambda: Truth({"code_or_count:age": "amount", "code_or_count:run_order": "amount",
                                "code_or_count:batch": "code", "code_or_count:responder": "code"},
                               fixture="metabolomics_untargeted.csv"),
        "lens": ["metabolomics"], "target": "responder", "event": "1", "purpose": "prediction",
        "before": {"target": apply_first_repair(r"qc")},
        "answers": {
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
            "missing": {"kind": "set_missing", "strategy": "impute"},
        },
        "until": stage_fresh("fit"),
    },
    "genomics": {
        "label": "A genomics count matrix under prediction: the wide path",
        "path": SAMPLES / "genomics_expression.csv",
        "truth": lambda: Truth({"code_or_count:age": "amount", "code_or_count:batch": "code"},
                               fixture="genomics_expression.csv"),
        "lens": ["genomics"], "target": "condition", "event": "case", "purpose": "prediction",
        "answers": {"grain": {"kind": "set_grain", "grain": "one_row_per_unit"}},
        "until": stage_fresh("fit"),
    },
    "survey": {
        "label": "A 40-item survey instrument under prediction",
        "path": SAMPLES / "survey_instrument.csv",
        "truth": lambda: Truth({**fixture_truth("survey_instrument.csv"),
                                "code_or_count:education": "code"},
                               fixture="survey_instrument.csv"),
        "lens": ["survey"], "target": "sought_support", "event": "1", "purpose": "prediction",
        "answers": {"grain": {"kind": "set_grain", "grain": "one_row_per_unit"}},
        # The first ten items scored as one scale (the instrument's key: item_05 reverse-coded),
        # and the fitted model explained before the seal opens.
        "before": {
            "models": declare({"kind": "set_scales", "scales": [{
                "name": "support_scale", "items": SURVEY_ITEMS, "reverse": ["item_05"],
                "low": 1, "high": 5, "kind": "reflective", "role": "covariate"}]}),
            "open_seal": declare({"kind": "set_explain", "curves": "ale", "reseeds": 0}),
        },
        "until": stage_fresh("fit"),
    },
    "clinical": {
        "label": "A clinical longitudinal table: visits kept as rows, a yes/no outcome followed",
        "path": SAMPLES / "clinical_longitudinal.csv",
        "truth": lambda: fixture_truth("clinical_longitudinal.csv"),
        "lens": ["clinical"], "target": "progressed", "event": "1", "purpose": "prediction",
        "answers": {
            "grain": {"kind": "set_grain", "grain": "repeated", "id_column": "subject_id"},
            "unit": {"kind": "set_unit", "unit": "row"},
            "temporal": {"kind": "set_temporal", "temporal": True, "time_column": "visit_date"},
        },
        "until": stage_fresh("fit"),
    },
    "causal": {
        "label": "The causal lane: a yes/no exposure by TMLE, its assumptions and positivity first",
        "path": causal_cohort, "source": "causal_cohort.csv (test_causal_lane.cohort)",
        "truth": causal_truth,
        "lens": ["clinical"], "target": "sbp", "purpose": "inference", "exposure": "heavy_user",
        "roles": {"person_id": "identifier", "age": "covariate", "smoker": "covariate",
                  "income": "covariate", "heavy_user": "exposure", "fiber": "exposure",
                  "ldl": "covariate"},
        "answers": {
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
            "models": {"kind": "select_models", "models": ["linear"]},
        },
        "before": {"models": declare_causal},
        "until": stage_fresh("fit", "causal"),
    },
    "time-varying": {
        "label": "A time-varying exposure: the DASH cohort's weights read before the estimate",
        "path": time_varying_cohort, "source": "dash_cohort.csv (timevary_fixtures.feedback_cohort)",
        "truth": lambda: Truth({"adjust:sbp": "yes,yes,yes", "adjust:female": "yes,yes,no",
                                "adjust:age": "yes,yes,no", "code_or_count:female": "code",
                                "code_or_count:age": "amount", "cluster:pid": "yes"},
                               fixture="the DASH cohort"),
        "lens": ["clinical"], "target": "cvd", "event": "1", "purpose": "inference",
        "exposure": "dash", "roles": None,
        "answers": {
            "grain": {"kind": "set_grain", "grain": "repeated", "id_column": "pid"},
            "repeat_kind": {"kind": "set_repeat_kind", "repeat_kind": "time_points",
                            "time_column": "visit"},
            "unit": {"kind": "set_unit", "unit": "row"},
            "temporal": {"kind": "set_temporal", "temporal": False},
            "time_varying": time_varying_answer,
            "models": {"kind": "select_models", "models": ["linear"]},
        },
        "until": stage_fresh("time_varying", "fit"),
    },
    "survey-design": {
        "label": "A survey design under inference: whose estimate it is, then fiber and CRP",
        "path": survey_design_table, "source": "survey.csv (test_wp10_survey_design I4)",
        "truth": survey_design_truth,
        "lens": ["dietary"], "target": "LBXCRP", "purpose": "inference", "exposure": "DR1TFIBE",
        "roles": {"SEQN": "identifier", "DR1TKCAL": "covariate", "DR1TFIBE": "exposure",
                  "WTDRD1": "design", "SDMVSTRA": "design", "SDMVPSU": "design"},
        "answers": {
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
            "models": {"kind": "select_models", "models": ["linear"]},
        },
        # An analysis beside the primary: the rows with energy between 500 and 4,000 kcal.
        "before": {"models": declare({"kind": "set_sensitivity", "analyses": [{
            "label": "Energy within 500 to 4,000 kcal",
            "rules": [{"kind": "range", "column": "DR1TKCAL", "low": 500, "high": 4000,
                       "reason": "an implausible energy report"}]}]})},
        "until": stage_fresh("fit", "effects"),
    },
    "dietary-recalls": {
        "label": "Repeated 24-hour recalls under inference: each person's recalls combined, then "
                 "regression calibration",
        "path": recall_table_path, "source": "recalls.csv (test_wp12c_calibration.recall_table)",
        "truth": recall_truth,
        "lens": ["dietary"], "target": "ldl", "purpose": "inference", "exposure": "protein_g",
        "roles": {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
                  "energy_kcal": "energy", "protein_g": "exposure"},
        "answers": {
            "task": {"kind": "set_task", "column": "ldl", "task": "regression"},
            "grain": {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"},
            "repeat_kind": {"kind": "set_repeat_kind", "repeat_kind": "repeats"},
            "unit": {"kind": "set_unit", "unit": "unit"},
            "aggregation": {"kind": "set_aggregation", "method": "mean"},
            "energy_adjustment": {"kind": "set_energy_adjustment", "method": "residual",
                                  "energy_column": "energy_kcal", "nutrients": ["protein_g"]},
            "models": {"kind": "select_models", "models": ["linear"]},
        },
        "before": {"models": declare({"kind": "set_measurement_error",
                                      "method": "regression_calibration", "n_boot": 50})},
        "until": stage_fresh("fit", "calibration"),
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8873")
    parser.add_argument("--repack", action="store_true",
                        help="pack fixtures written before packing existed, and stop")
    parser.add_argument("journeys", nargs="*")
    args = parser.parse_args(argv)
    if args.repack:
        for path in sorted(OUT.iterdir()):
            out = json.loads(path.read_text())
            if "snapshots" in out and "view" in out["snapshots"][-1]:
                path.write_text(json.dumps(pack(out), separators=(",", ":"), ensure_ascii=False)
                                + "\n")
                print(f"packed {path.name} ({path.stat().st_size // 1024} KB)")
        return 0
    client = Recorder(args.base)
    health = client.get("/api/health").json()
    print("server", health, flush=True)
    names = args.journeys or [*JOURNEYS, "assembly"]
    for name in names:
        if name == "assembly":
            capture_assembly(client)
            continue
        spec = {"name": name, **JOURNEYS[name]}
        if spec["name"] == "time-varying" and spec.get("roles") is None:
            spec["roles"] = time_varying_roles()
        run_journey(client, spec)
    teaching = client.get("/api/teaching").json()
    models = client.get("/api/models").json()
    trimmed: dict[str, int] = {}
    shared = {"meta": {"captured": time.strftime("%Y-%m-%d"), "trim": TRIM_RULE},
              "health": health, "teaching": teaching, "models": trim(models, "models", trimmed)}
    (OUT / "m3-shared.json").write_text(json.dumps(shared, separators=(",", ":"), ensure_ascii=False)
                                        + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the Sullivan 2021 replication through TurboTab's Estimate lane, one project per fit.

Each run is a journey (``turbotab.core.reference.journeys``: its loop, its recording client and its
capture) over a local-mode server with two workers (``server_drive.local_server``). Its answers
are the paper's stated decisions, read from the plan file ``sullivan2021.json`` (the paper as
data: the table and its joins, the readings, the exclusions, the survey design, the models, the
variants), never written into this code. A question the plan file does not answer takes the
app's own answer, as the reference journeys do, and the capture says so; a reading the server
asks about is answered from the plan file's readings (the CDC codebooks' facts).

**The template.** The plan's nine files are joined to the table once (``--template``: nine
``join_files`` answers, each after its preview), in a home of their own; every run starts from a
copy of that home, so each project's log holds the same joins. The engine's join is a nested-loop
join on ``SEQN`` and every ingest re-applies every join so far, so the nine take about nine
minutes, more than a run's budget (the report's engine gaps).

**Refusals.** Every plan answer is posted by :func:`post_kept`: a refusal that asks for a reading
is answered from the plan file's readings; one that blocks and records the answer is kept by its
own "Keep it, recorded" exit, which records that same answer with its attestation, and the
capture notes it; any other refusal stops the run with its message (no exit is taken that does
something other than what the run measures).

What each run keeps (``research/data/derived/engine/``, ignored by git):

* ``runs/<name>.json``: the capture (every answer posted and where it came from, every refusal
  and exit taken, the methods section verbatim), the exposure's rows of the export's Table 2 (β,
  SE, df, p, the design-based n), the fit as served, the participant flow, the seconds;
* ``<name>.csv``: the analytic rows as the engine fit them, read back from the engine's own
  artifacts (the design stage's model matrix, every model column, beside the working table's
  ``SEQN``, outcome and design columns at those rows), for the verifier;
* ``bundles/<name>.zip``: the export bundle.

    scripts/py research/replication/sullivan2021/engine/run.py --template   (until "template ready")
    scripts/py research/replication/sullivan2021/engine/run.py --list
    scripts/py research/replication/sullivan2021/engine/run.py t4-glucose m0-insulin ... --budget 450
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("TURBOTAB_WORKERS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from turbotab.core.reference import journeys as J  # noqa: E402

PLAN = json.loads((HERE / "sullivan2021.json").read_text())
OUT = REPO / "research" / "data" / "derived" / "engine"
RUNS = OUT / "runs"
BUNDLES = OUT / "bundles"
LADDER = ("glucose", "insulin", "hba1c")


# ── the runs the plan file makes ─────────────────────────────────────────────


def runs() -> dict[str, dict[str, Any]]:
    """Every run: Table 4's four fits, the ladder's fifteen, M4 as its printed numbers suggest, the
    unstated decisions' refits (SULLIVAN_PLAN §3) and the alternatives, each named."""
    out: dict[str, dict[str, Any]] = {}
    for outcome in PLAN["outcomes"]:
        out[f"t4-{outcome}"] = {"outcome": outcome, "model": "table4", "group": "table4"}
    for outcome in LADDER:
        for m in ("M0", "M1", "M2", "M3", "M4"):
            out[f"{m.lower()}-{outcome}"] = {"outcome": outcome, "model": m, "group": "ladder"}
        out[f"m4t4-{outcome}"] = {"outcome": outcome, "model": "M4_table4", "group": "m4_reading"}
    for key, v in PLAN["variants"].items():
        group = "alternative" if v.get("alternative") else "unstated"
        for outcome in v.get("outcomes") or list(PLAN["outcomes"]):
            out[f"{key}-{outcome}"] = {"outcome": outcome, "model": v.get("model", "table4"),
                                       "variant": key, "group": group}
    return out


def covariates(run: dict[str, Any]) -> list[str]:
    """The run's covariate keys: its model's, BMI left out of the BMI outcome's own model."""
    keys = list(PLAN["models"][run["model"]])
    v = PLAN["variants"].get(run.get("variant") or "", {})
    keys = [k for k in keys if k not in (v.get("drop_covariates") or [])]
    keys += [k for k in v.get("add_covariates") or [] if k not in keys]
    outcome = {PLAN["outcomes"][run["outcome"]]["column"], outcome_column(run)}
    return [k for k in keys if PLAN["covariates"][k]["column"] not in outcome]


def outcome_column(run: dict[str, Any]) -> str:
    v = PLAN["variants"].get(run.get("variant") or "", {})
    return v.get("outcome_column") or PLAN["outcomes"][run["outcome"]]["column"]


def settings(run: dict[str, Any]) -> dict[str, Any]:
    """The run's answers, as data: the plan file's, with its variant's overrides applied."""
    v = PLAN["variants"].get(run.get("variant") or "", {})
    o = PLAN["outcomes"][run["outcome"]]
    survey = {k: val for k, val in PLAN["survey"].items() if k != "stated"}
    survey["weight"] = o["weight"]
    survey.update(v.get("survey") or {})
    rules = [e for e in PLAN["exclusions"] if e["key"] not in (v.get("drop_exclusions") or [])]
    rules += list(v.get("add_exclusions") or [])
    keys = covariates(run)
    forms = {PLAN["exposure"]["column"]: {"form": PLAN["exposure"]["form"]}}
    for k in keys:
        c = PLAN["covariates"][k]
        if not c.get("categorical"):
            forms[c["column"]] = {"form": c.get("form", "linear")}
    for column, f in (v.get("forms") or {}).items():
        if column in forms:
            forms[column] = dict(f)
    missing = {k: val for k, val in PLAN["missing"].items() if k in ("strategy",)}
    missing.update(v.get("missing") or {})
    columns = [PLAN["covariates"][k]["column"] for k in keys]
    return {"target": outcome_column(run), "covariates": columns,
            "categorical": [PLAN["covariates"][k]["column"] for k in keys
                            if PLAN["covariates"][k].get("categorical")],
            "survey": survey, "rules": [e["rule"] for e in rules],
            "rule_keys": [e["key"] for e in rules], "forms": forms, "missing": missing,
            "outcome_scale": v.get("outcome_scale", "original")}


# ── the journey ──────────────────────────────────────────────────────────────


def table_columns() -> list[str]:
    import pyarrow.parquet as pq

    cols = list(pq.read_schema(REPO / PLAN["table"]["main"]).names)
    for name in PLAN["table"]["joins"]:
        for c in pq.read_schema(REPO / PLAN["table"]["join_folder"] / f"{name}.parquet").names:
            if c not in cols:
                cols.append(c)
    return cols


def roles_for(s: dict[str, Any]) -> dict[str, str]:
    design = {s["survey"]["strata"], s["survey"]["psu"], PLAN["survey"]["cycle"],
              *(o["weight"] for o in PLAN["outcomes"].values())}
    roles: dict[str, str] = {}
    for c in table_columns():
        if c == s["target"]:
            continue
        if c == PLAN["table"]["on"]:
            roles[c] = "identifier"
        elif c == PLAN["exposure"]["column"]:
            roles[c] = "exposure"
        elif c in s["covariates"]:
            roles[c] = "covariate"
        elif c in design:
            roles[c] = "design"
        else:
            roles[c] = "excluded"
    return roles


TEMPLATE = OUT / "template"


def build_template(budget: float = 480.0) -> bool:
    """The table with the plan's files joined, once: a project in its own home whose log holds
    the nine ``join_files`` answers, each after its preview, as a researcher adds NHANES's
    components. Every run starts from a copy of this home, so its log holds the same joins.

    The engine's join is a nested-loop join (its ON clause guards each side's identifier, which
    keeps DuckDB from hashing it), and every ingest re-applies every join so far: the nine take
    about ten minutes. This is resumable: it stops after ``budget`` seconds and the next call
    goes on. True when every file is joined and the table is read."""
    from turbotab.core.tests.acceptance.server_drive import local_server

    started = time.monotonic()
    home = TEMPLATE / "home"
    state_file = TEMPLATE / "template.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"joined": [],
                                                                             "seconds": {}}
    t = PLAN["table"]
    with local_server(home) as client:
        if not state.get("pid"):
            r = client.post("/api/projects", json={"path": str(REPO / t["main"])})
            r.raise_for_status()
            state["pid"] = r.json()["id"]
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(state, indent=1))
        pid = state["pid"]

        def ingest_fresh(timeout: float = 1800.0) -> bool:
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                status = client.get(f"/api/projects/{pid}").json()["stages"]["ingest"]
                if status["status"] == "fresh":
                    return True
                if status["status"] == "error":
                    raise RuntimeError(f"the ingest failed: {status}")
                time.sleep(0.5)
            return False

        for name in t["joins"]:
            if name in state["joined"]:
                continue
            if time.monotonic() - started > budget:
                break
            t0 = time.monotonic()
            ingest_fresh()
            path = REPO / t["join_folder"] / f"{name}.parquet"
            added = client.post(f"/api/projects/{pid}/files", json={"path": str(path)})
            added.raise_for_status()
            fid = added.json()["id"]
            preview = client.post(f"/api/projects/{pid}/join-preview",
                                  json={"file": fid, "on": t["on"], "how": t["how"]}).json()
            if preview.get("refusal"):
                raise RuntimeError(f"the join of {name} was refused: {preview['refusal']}")
            r = client.post(f"/api/projects/{pid}/decisions",
                            json={"kind": "join_files", "file": fid, "on": t["on"],
                                  "how": t["how"]})
            if r.status_code != 200:
                raise RuntimeError(f"join_files {name}: {r.text[:600]}")
            ingest_fresh()
            state["joined"].append(name)
            state["seconds"][name] = round(time.monotonic() - t0, 1)
            state["preview"] = state.get("preview", {})
            state["preview"][name] = {k: preview.get(k) for k in ("relation", "rows", "sentence")}
            state_file.write_text(json.dumps(state, indent=1))
            print(f"joined {name} in {state['seconds'][name]} s", flush=True)
        done = len(state["joined"]) == len(t["joins"]) and ingest_fresh()
    print("template ready" if done else f"template: {len(state['joined'])} of {len(t['joins'])} "
          "files joined; run --template again", flush=True)
    return done


PLAN_SOURCE = ("the plan file (sullivan2021.json: the paper's stated decision, or the option a "
               "run measures)")
# The kinds the plan file answers itself, which ``journeys.Run.source_of`` would otherwise credit
# to the acceptance drivers (``WP17_SOURCES``).
PLAN_KINDS = ("set_estimand", "set_forms", "set_clusters")


class PlanRun(J.Run):
    """A journey's run whose capture credits the plan file for the answers it gives."""

    def source_of(self, kind: str | None) -> str:
        if kind in PLAN_KINDS and not self.source.startswith(J.OWN_EXITS):
            return PLAN_SOURCE
        return super().source_of(kind)


KEEP = ("Keep it", "Keep the single fill")


def post_kept(run: J.Run, body: dict[str, Any]) -> None:
    """Post the plan file's answer; readings it waits on are answered from the plan file's
    readings (``settle_post``). An answer the engine blocks and records is kept by its own "Keep
    it, recorded" exit, which records exactly this answer with its attestation, and the capture
    says so; any other refusal stops the run with its message and exits (no exit is taken that
    would do something other than what the run measures)."""
    run.source = PLAN_SOURCE
    r = J._settle(run, body)
    at = round(time.monotonic() - run.started, 1)
    print(f"  [{run.spec.name}] {body.get('kind')}: {J.summarize(body, 110)} ({r.status_code} at "
          f"{at} s)", flush=True)
    if r.status_code == 200:
        if run.log:
            run.log[-1].setdefault("at_s", at)
        return
    error, exits = J._exits(r)
    kept = [e for e in exits if str(e.get("label", "")).startswith(KEEP)]
    if not kept:
        raise RuntimeError(f"{body.get('kind')} refused ({error.get('code')}): "
                           f"{error.get('message')} exits: {[e.get('label') for e in exits]}")
    run.source = f"the refusal's way forward: {kept[0]['label']}"
    taken = J._settle(run, kept[0]["decision"])
    if taken.status_code != 200:
        raise RuntimeError(f"{body.get('kind')}: its keep exit was refused: {taken.text[:600]}")
    run.note(f"{body.get('kind')} was blocked and recorded ({error.get('code')}: "
             f"{error.get('message')}); kept by its exit '{kept[0]['label']}', which records this "
             "answer with its attestation")


def answer_forms(s: dict[str, Any]):
    """The form question: first the readings its ask card asks (whether a column's whole numbers
    are codes or amounts), answered from the plan file's readings as ``server_drive.answer_forms``
    answers them; then each column the card asks about takes the plan file's form."""
    calls = {"n": 0}

    def go(run: J.Run, step: dict[str, Any], view: dict[str, Any]) -> dict[str, Any] | None:
        from turbotab.core.tests.truths import answers, asked

        calls["n"] += 1
        if calls["n"] > 6:
            raise RuntimeError("the form question stayed open after six answers")
        for _ in range(4):
            step = run.drive.reach("form", timeout=300)
            ask = step.get("ask") or {}
            if step["status"] != "open" or not asked(ask.get("exits") or []):
                break
            run.source = J.READINGS_SOURCE
            for decision in answers({"exits": ask["exits"]}, run.drive.truth):
                r = run.drive.c.post(f"/api/projects/{run.pid}/decisions", json=decision)
                if r.status_code != 200:
                    raise RuntimeError(f"a reading the form question asked: {r.text[:600]}")
        if run.drive.reach("form", timeout=300)["status"] not in ("open", "waiting"):
            return None
        card = J._artifact(run, "forms")
        needs = [n["column"] for n in card.get("needs") or []]
        forms = {}
        for column in needs:
            if column in s["forms"]:
                forms[column] = s["forms"][column]
            else:
                run.note(f"the form card asked about {column}, which the plan file gives no form; "
                         "a straight line")
                forms[column] = {"form": "linear"}
        post_kept(run, {"kind": "set_forms", "forms": forms})
        return None
    return go


def answer_adjustment(run: J.Run, step: dict[str, Any], view: dict[str, Any]) -> None:
    """The adjustment question: every covariate the paper adjusts for answered as a confounder
    (the plan file's ``causal``), group by group as the card groups them
    (``truths.answer_adjustment``); the form question is left to its own answer."""
    from turbotab.core.tests.truths import answer_adjustment as answer

    exposure = PLAN["exposure"]["column"]
    end = time.monotonic() + 600
    while True:
        card = J._artifact(run, "proposals").get("adjustment")
        if card and card.get("exposure") == exposure:
            break
        if time.monotonic() > end:
            raise RuntimeError(f"the adjustment card never named {exposure}: {card!r}")
        time.sleep(0.2)
    run.source = J.WP17_SOURCES["set_adjustment"]
    answer(lambda d: run.drive.c.post(f"/api/projects/{run.pid}/decisions", json=d), card,
           run.drive.truth)
    return None


def journey(name: str, run: dict[str, Any]) -> J.Journey:
    s = settings(run)
    readings = dict(PLAN["readings"])
    for column in s["covariates"]:
        readings[f"adjust:{column}"] = "yes,yes,no"
    readings[f"exposure:{s['target']}"] = PLAN["exposure"]["column"]
    readings[f"outcome_scale:{s['target']}"] = s["outcome_scale"]
    exposure = PLAN["exposure"]["column"]
    def kept(body: dict[str, Any]):
        """An answer posted by :func:`post_kept`: a refusal is never answered by another exit."""
        return lambda run, step, view: post_kept(run, body)

    answers: dict[str, Any] = {
        "lens": kept({"kind": "set_lens", "lenses": PLAN["lens"]}),
        "purpose": kept({"kind": "set_purpose", "purpose": "inference"}),
        "clusters": kept({"kind": "set_clusters", "column": None, "acknowledged": True}),
        "survey": kept({"kind": "set_survey", **s["survey"]}),
        "exclusions": kept({"kind": "set_exclusions", "rules": s["rules"]}),
        "missing": kept({"kind": "set_missing", **s["missing"]}),
        "estimand": kept({"kind": "set_estimand", "exposure": exposure, "effect": "total",
                          "measure": PLAN["exposure"]["measure"]}),
        "adjustment": answer_adjustment,
        "form": answer_forms(s),
        "causal": kept({"kind": "set_causal", "exposure": exposure, "method": "none"}),
        "models": kept({"kind": "select_models", "models": [PLAN["model_family"]]}),
    }
    before: dict[str, Any] = {}
    return J.Journey(
        name=name, lens="sullivan2021", purpose="inference",
        question=f"Is a better self-rated diet associated with {run['outcome']} ({run['model']})?",
        fixture="the NHANES 2011-2018 files of Sullivan et al. 2021, stacked and derived by prep.py",
        why="SULLIVAN_PLAN: the first end-to-end check of the Estimate lane on a published analysis",
        path=lambda: REPO / PLAN["table"]["main"],
        truth=lambda: J.truth(readings, fixture="sullivan2021.json"),
        target=s["target"], lenses=tuple(PLAN["lens"]), exposure=exposure,
        roles=roles_for(s), answers=answers, before=before, timeout=1500.0,
        fixture_key="sullivan2021.json")


# ── what a run keeps ─────────────────────────────────────────────────────────


def served(run: J.Run, stage: str) -> dict[str, Any]:
    return (run.drive.c.get(f"/api/projects/{run.pid}/stages/{stage}").json() or {}).get(
        "artifact") or {}


def exposure_row(run: J.Run) -> dict[str, Any]:
    """The exposure's row of each fitted family, as the fit stage serves it: β, SE, df, p, its
    interval, the rows it was estimated from, how the intervals were made, and the adjustment
    terms (the appendix the server sets beside it)."""
    out: dict[str, Any] = {}
    for model in served(run, "fit").get("models") or []:
        rows = model.get("coefficients") or []
        mine = [r for r in rows if r.get("feature") == PLAN["exposure"]["column"]]
        out[model.get("family") or "model"] = {
            "exposure": mine[0] if len(mine) == 1 else mine, "n": model.get("coefficients_n"),
            "inference": model.get("inference"), "concerns": model.get("concerns"),
            "adjustment_terms": [{k: t.get(k) for k in ("feature", "estimate", "se", "df", "p")}
                                 for t in model.get("adjustment_terms") or []],
            "fit_seconds": model.get("fit_seconds")}
    return out


def cohort_flow(run: J.Run) -> Any:
    """The participant flow as the cohort stage counts it."""
    art = served(run, "cohort")
    return art.get("flow") or art.get("steps") or {k: art.get(k) for k in list(art)[:12]}


def save_rows(run: J.Run, name: str, s: dict[str, Any]) -> dict[str, Any]:
    """The analytic rows as the engine fit them, read back from its own artifacts: the design
    stage's model matrix (every model column, as the fit's shared steps made it, by row id) beside
    the working table's ``SEQN``, outcome and design columns at those rows. Written as
    ``<name>.csv``; returns its SHA-256, its rows and its columns."""
    import pandas as pd

    from turbotab.core.export.matrix import FILE
    from turbotab.core.graph import artifact_dir

    view = run.drive.view()
    service = run.drive.c.app.state.service
    cache = service.workspace.cache_dir(run.pid)
    design_key = view["stages"]["design"]["key"]
    working_key = view["stages"]["working"]["key"]
    matrix = pd.read_parquet(artifact_dir(cache, "design", design_key) / "files" / FILE)
    if "row_id" in matrix.columns:
        matrix = matrix.set_index("row_id")
    working = artifact_dir(cache, "working", working_key) / "files" / "table.parquet"
    table = pd.read_parquet(working.resolve())
    table = table.set_index("__row_id")
    # The outcome as the fit read it: the state's target (``ln_LBXIN`` on the log scale).
    outcome = view["state"]["target"]
    keep = [PLAN["table"]["on"], outcome, s["survey"]["weight"], s["survey"]["strata"],
            s["survey"]["psu"], PLAN["survey"]["cycle"]]
    keep = [c for c in dict.fromkeys(keep) if c in table.columns]
    out = table.loc[matrix.index, keep].join(matrix.add_prefix("x:"))
    out.index.name = "row_id"
    path = OUT / f"{name}.csv"
    out.to_csv(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": str(path.relative_to(REPO)), "sha256": digest, "rows": int(len(out)),
            "outcome": outcome, "columns": list(out.columns), "design_key": design_key,
            "working_key": working_key}


def table2_rows(name: str) -> list[dict[str, Any]]:
    """The exposure's rows of the export's Table 2 (``results/table2.csv``): each declared model
    of the sequence (the unadjusted one, the primary), with the design-based n (the analysis rows
    with a positive weight), β, SE, df and p, as the bundle reports them."""
    import csv
    import io
    import zipfile

    path = BUNDLES / f"{name}.zip"
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as z:
        entry = next((n for n in z.namelist() if n.endswith("results/table2.csv")), None)
        if entry is None:
            return []
        text = z.read(entry).decode("utf-8")
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("term") != PLAN["exposure"]["column"]:
            continue
        num = {k: (float(r[k]) if r.get(k) not in (None, "") else None)
               for k in ("n", "estimate", "se", "df", "p", "ci_low", "ci_high")}
        out.append({"model": r.get("model"), "key": r.get("key"),
                    "adjusted_for": r.get("adjusted_for"), **num,
                    "primary": "(primary)" in (r.get("model") or "")})
    return out


def run_one(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """One run: a copy of the template's home (the table with its nine joins), the Router followed
    with the plan file's answers to the fit and the export, and what it keeps (module docstring)."""
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

    template = json.loads((TEMPLATE / "template.json").read_text())
    if len(template.get("joined") or []) != len(PLAN["table"]["joins"]):
        raise SystemExit("the template is not built: run --template first")
    j = journey(name, spec)
    s = settings(spec)
    run = PlanRun(j)
    print(f"── {name}: {j.question}", flush=True)
    response = None
    extra: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"tt-su-{name}-") as folder:
        home = Path(folder) / "home"
        shutil.copytree(TEMPLATE / "home", home)
        try:
            with local_server(home) as client:
                run.client = J.Recording(client, run)
                run.pid = template["pid"]
                t = j.truth()
                t.guess = J._server_guess(client, run.pid)
                t.ranked = J._server_ranked(client, run.pid)
                run.drive = Drive(run.client, run.pid, t)
                run.drive.exposure = j.exposure
                run.drive.artifact("ingest", timeout=900)
                run.note(f"opened from the template: {len(template['joined'])} files joined on "
                         f"{PLAN['table']['on']} ({', '.join(template['joined'])})")
                if J.follow(run) and J.wait_results(run):
                    run.source = "the export"
                    response = J.export(run)
                else:
                    response = run.client.get(f"/api/projects/{run.pid}/export")
                extra["fit"] = exposure_row(run)
                extra["flow"] = cohort_flow(run)
                extra["rows"] = save_rows(run, name, s)
        except Exception as exc:  # noqa: BLE001 - a run that stops is kept with its reason
            run.note(f"stopped by {type(exc).__name__}: {str(exc)[:900]}")
            traceback.print_exc()
        BUNDLES.mkdir(parents=True, exist_ok=True)
        out = J.capture_of(run, response, Path(folder), BUNDLES)
    out.update(extra)
    out["table2"] = table2_rows(name)
    out["run"] = {"name": name, **spec, "settings": s}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    row = next((r for r in out["table2"] if r.get("primary")), {})
    print(f"wrote {name} (export {out['export'].get('status')}, {out['captured']['seconds']} s): "
          f"beta {row.get('estimate')} se {row.get('se')} df {row.get('df')} n {row.get('n')}",
          flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--template", action="store_true",
                        help="build (or go on building) the joined table every run starts from")
    parser.add_argument("--batch", default=None, help="every run of one group")
    parser.add_argument("--budget", type=float, default=540.0,
                        help="start no new run after this many seconds")
    args = parser.parse_args(argv)
    every = runs()
    if args.list:
        for n, r in every.items():
            print(f"{n:28} {r['group']:12} {r['model']:10} {r.get('variant') or ''}")
        return 0
    if args.template:
        return 0 if build_template() else 3
    names = list(args.names)
    if args.batch:
        names += [n for n, r in every.items() if r["group"] == args.batch]
    unknown = [n for n in names if n not in every]
    if unknown:
        parser.error(f"unknown runs: {unknown}")
    started = time.monotonic()
    for n in names:
        if time.monotonic() - started > args.budget:
            print(f"budget spent: not started {names[names.index(n):]}", flush=True)
            break
        run_one(n, every[n])
    return 0


if __name__ == "__main__":
    sys.exit(main())

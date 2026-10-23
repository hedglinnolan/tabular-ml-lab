"""Run the Bang 2009 replication sweep through the swarm's driver, one run per variant and lane.

Each run is a scripted analyst's (``research/swarm/dry_run.py``'s :class:`Analyst`) on the driver's
own server in analyst mode with two workers (``python -m research.swarm.driver start --scripted``),
on the prep's derived table (``prep.py``). Every answer is ``prereg.json``'s, read from it here and
never written into this code: the outcome and its event, the purpose by the lane, the roles (the
six predictors the variant's options name, the design columns, the rest left out), each predictor
read as a code, the eligibility, the survey answer, complete cases, the linear family (logistic
regression for a yes/no outcome), and under prediction no holdout. A question the plan does not
answer takes the app's first-ranked option, as the dry run's analyst does.

**The eligibility.** The paper's population is six rules (``prereg.json``'s ``eligibility``): an
age range, the pregnancy level rule, the three known-diabetes level rules each naming its refused
(7) and don't-know (9) codes, and the fasting range. The driver's ``answer`` records one rule per
answer of the exclusions question, as the app's own card does (``research/swarm/options.py``'s
``_exclusions``; ``ChoiceQuestions.tsx``), and a level rule of its own always leaves out a blank,
refused or don't-know row (``options.CUSTOM_LEVELS``), so it cannot record this population. So the
run first previews, through the driver, the DIQ010 rule as the driver composes it (``preview levels
--set level_column=DIQ010 --set exclude=1 --set refused=7 --set dont_know=9``), keeps that
preview, and then posts the one ``set_exclusions`` holding every rule through the driver's own
client and ``post_decision``, the request ``answer`` makes. The report names this as a gap.

What each run keeps (``research/data/derived/bang2009/engine/``, ignored by git):

* ``runs/<lane>-<variant>.json``: the driver commands and their exits, the decisions as the server
  recorded them (each sentence), the eligibility preview, the participant flow, the fitted family
  as the results serve it (its coefficients, its scores, its concerns), the analyzed rows' SEQN and
  outcome and the model matrix's columns at them (read from the run's own cache: the design stage's
  matrix and the working table), and the seconds;
* ``bundles/<lane>-<variant>.zip``: the export bundle.

    scripts/py research/replication/bang2009/run.py --list
    scripts/py research/replication/bang2009/run.py estimate:default predict:default ...
    scripts/py research/replication/bang2009/run.py --walk estimate:default   (to Fit, unpressed)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "research" / "data" / "derived" / "bang2009" / "engine"
RUNS_HOME = OUT / "homes"  # the driver's runs' home ($TURBOTAB_SWARM_RUNS, set by main)

from research.swarm import dry_run as D  # noqa: E402

PREREG = json.loads((HERE / "prereg.json").read_text("utf-8"))
DERIVED = REPO / "research" / "data" / "derived" / "bang2009" / "derived.parquet"
BUDGET_FITS, BUDGET_SECONDS = 3, 900.0
RULES_ID = "__prereg_rules__"  # the exclusions answer this run posts itself (the module docstring)
DESIGN_ROLE = "design"


# ── what each run answers, from the plan ─────────────────────────────────────


def variants() -> list[str]:
    """Every run, ``<lane>:<variant>``: every variant under the estimate lane, and under the
    predict lane every variant whose decision applies to it (``prereg.json``'s ``sweep.lanes``)."""
    out = []
    lanes = {d["id"]: d.get("lanes") for d in PREREG["decisions"]}
    for lane in ("estimate", "predict"):
        for name, changed in PREREG["sweep"]["variants"].items():
            if all(lanes.get(d) is None or lane in lanes[d] for d in changed):
                out.append(f"{lane}:{name}")
    return out


def settings(lane: str, variant: str) -> dict[str, Any]:
    """The run's answers: every decision at its default but the variant's, each option's
    ``apply`` laid over the stated plan."""
    changed = PREREG["sweep"]["variants"][variant]
    predictors = {k: p["column"] for k, p in PREREG["model"]["predictors"].items()}
    rules = copy.deepcopy(PREREG["eligibility"]["rules"])
    survey = "population"
    chosen = {}
    for d in PREREG["decisions"]:
        option = changed.get(d["id"], d["default"])
        if option not in d["options"]:
            raise SystemExit(f"{variant}: {d['id']} has no option {option!r}")
        if d.get("lanes") and lane not in d["lanes"] and d["id"] in changed:
            raise SystemExit(f"{variant}: {d['id']} is not a {lane} decision")
        chosen[d["id"]] = option
        apply = d["options"][option]["apply"]
        predictors.update(apply.get("predictors") or {})
        for name in apply.get("drop_rules") or []:
            rules.pop(name)
        for name, fields in (apply.get("rules") or {}).items():
            rules[name].update(fields)
        survey = apply.get("survey", survey)
    lane_plan = PREREG["lanes"][lane]
    exposure = predictors[lane_plan["exposure"]] if lane == "estimate" else None
    return {"lane": lane, "variant": variant, "chosen": chosen, "purpose": lane_plan["purpose"],
            "predictors": predictors, "exposure": exposure, "rules": rules,
            "survey": dict(PREREG["survey"][survey]), "survey_answer": survey,
            "holdout": lane_plan.get("holdout")}


def table_columns() -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(DERIVED).names)


def roles_for(s: dict[str, Any]) -> dict[str, str]:
    """Each column's role: the predictors (under inference the exposure and its covariates), the
    survey design's columns, the identifier, and every other column left out (the eligibility
    rules read any column)."""
    model = set(s["predictors"].values())
    design = {PREREG["survey"]["population"][k]
              for k in ("weight", "four_year_weight", "cycle", "strata", "psu")}
    roles = {}
    for c in table_columns():
        if c == PREREG["outcome"]["column"]:
            continue
        if c == "SEQN":
            roles[c] = "identifier"
        elif c in model:
            roles[c] = ("exposure" if s["purpose"] == "prediction" or c == s["exposure"]
                        else "covariate")
        elif c in design:
            roles[c] = DESIGN_ROLE
        else:
            roles[c] = "excluded"
    return roles


def spec_for(s: dict[str, Any]) -> D.Spec:
    readings = {f"code_or_count:{c}": "code" for c in s["predictors"].values()}
    readings.update({"code_or_count:RIDAGEYR": "amount", "unit:RIDAGEYR": "years"})
    if s["purpose"] == "inference":
        readings.update({f"adjust:{c}": "yes,yes,no" for c in s["predictors"].values()
                         if c != s["exposure"]})
    return D.Spec(name=f"{s['lane']}-{s['variant']}", dataset="bang2009",
                  target=PREREG["outcome"]["column"], event=PREREG["outcome"]["event"],
                  purpose=s["purpose"], lenses=["clinical"], readings=readings,
                  roles=roles_for(s), exposure=s["exposure"],
                  answers={"models": ([PREREG["model"]["family"]], [])})


def the_rules(s: dict[str, Any]) -> dict[str, Any]:
    """The one exclusions answer holding every rule, each in the engine's own rule shape."""
    return {"kind": "set_exclusions", "rules": [dict(r) for r in s["rules"].values()]}


# ── the scripted analyst ─────────────────────────────────────────────────────


class Replicator(D.Analyst):
    """The dry run's analyst with the plan's answers to the exclusions, the survey, the missing
    values and the split, and the eligibility posted as the module docstring says."""

    settings: dict[str, Any]
    kept: dict[str, Any]

    def choose(self, key: str, q: dict[str, Any]) -> tuple[list[str], list[str], str]:
        s = self.settings
        offered = {str(o.get("id")): o for o in q.get("options") or [] if o.get("id") is not None}
        if key == "exclusions":
            return [RULES_ID], [], "the paper's population, every rule of the plan"
        if key == "survey":
            want = s["survey"]
            for oid, o in offered.items():
                d = o.get("decision") or {}
                if d.get("estimand") == want["estimand"] and all(
                        d.get(k) == want[k] for k in want if k not in ("estimand", "about")):
                    return [oid], [], f"the plan's survey answer ({s['survey_answer']})"
            raise RuntimeError(f"no survey option records the plan's {want}: "
                               f"{[o.get('decision') for o in offered.values()]}")
        if key == "missing":
            strategy = PREREG["missing"]["strategy"]
            pick = next((oid for oid, o in offered.items()
                         if (o.get("decision") or {}).get("strategy") == strategy
                         and not (o.get("decision") or {}).get("indicators")
                         and not (o.get("decision") or {}).get("categorical")), None)
            if pick is None:
                raise RuntimeError(f"no missing-values option records {strategy}: "
                                   f"{[o.get('decision') for o in offered.values()]}")
            return [pick], [], "complete cases (the paper: all non-missing observations)"
        if key == "split" and s["holdout"] is not None:
            pick = next((oid for oid, o in offered.items()
                         if float((o.get("decision") or {}).get("holdout", -1)) == float(s["holdout"])),
                        None)
            if pick is None:
                raise RuntimeError(f"no split option holds out {s['holdout']}: {list(offered)}")
            return [pick], [], "cross-validation alone: no holdout, the final fit on every row"
        if key == "target":
            return [self.spec.target], [], "the card's outcome"
        if key == "event":
            # the event level as the server spells it (an integer column's 1)
            pick = next((oid for oid in offered if float(oid) == float(self.spec.event)), None)
            if pick is None:
                raise RuntimeError(f"no event level {self.spec.event}: {list(offered)}")
            return [pick], [], "the card's outcome level"
        return super().choose(key, q)

    def answer(self, ids: list[str], sets: list[str], why: str,
               marker: str = D.DEFAULT_MARKER) -> dict[str, Any]:
        self.kept.setdefault("answers", []).append({"ids": ids, "sets": sets, "why": why})
        if ids != [RULES_ID]:
            try:
                return super().answer(ids, sets, why, marker)
            except D.Refused as exc:
                # A refused answer stops the run: the dry run's analyst would take the refusal's
                # first way forward, which answers something other than the plan.
                raise RuntimeError(f"an answer was refused ({ids}, {sets}): {exc}"[:3000]) from None
        from research.swarm import driver

        told = self.settings["rules"].get("told_diabetes") or {}
        code, preview = self.cmd(
            "preview", "levels", "--set", "level_column=DIQ010",
            "--set", f"exclude={','.join(told.get('exclude') or ['1'])}",
            "--set", f"refused={','.join(told.get('refused') or [])}",
            "--set", f"dont_know={','.join(told.get('dont_know') or [])}",
            "--reason", self.reason("what the driver's own level rule records for DIQ010"))
        self.kept["driver_level_preview"] = {"exit": code, "previewed": preview.get("previewed"),
                                             "refused": preview.get("refused")}
        decision = the_rules(self.settings)
        run = driver.Run.load(self.run)
        r = driver.post_decision(run, run.client(), decision, self.reason(why))
        self.log.append({"args": ["post_decision", "set_exclusions", str(len(decision["rules"]))],
                         "exit": 0 if r.status == 200 else 1,
                         "refused": None if r.status == 200 else r.error().get("code")})
        if r.status != 200:
            raise D.Refused(json.dumps({"refused": r.error(), "posted": decision})[:3000])
        self.kept["eligibility_posted"] = decision
        return r.json()


# ── what a run keeps ─────────────────────────────────────────────────────────


def home_of(a: Replicator) -> tuple[Path, str]:
    meta = json.loads((Path(os.environ["TURBOTAB_SWARM_RUNS"]) / a.run / "driver.json").read_text())
    return Path(meta["home"]), meta["project"]


def view(a: Replicator) -> dict[str, Any]:
    from research.swarm import driver

    run = driver.Run.load(a.run)
    return driver.view_of(run, run.client())


def analyzed_rows(a: Replicator, v: dict[str, Any]) -> dict[str, Any]:
    """The analyzed rows as the engine fit them, from the run's own cache: the design stage's model
    matrix (by row id) beside the working table's SEQN, outcome and design columns there."""
    import pandas as pd

    from turbotab.core.export.matrix import FILE
    from turbotab.core.graph import artifact_dir

    home, pid = home_of(a)
    cache = home / "projects" / pid / "cache"
    stages = v["stages"]
    matrix = pd.read_parquet(artifact_dir(cache, "design", stages["design"]["key"]) / "files" / FILE)
    if "row_id" in matrix.columns:
        matrix = matrix.set_index("row_id")
    working = pd.read_parquet(artifact_dir(cache, "working", stages["working"]["key"])
                              / "files" / "table.parquet").set_index("__row_id")
    outcome = PREREG["outcome"]["column"]
    keep = ["SEQN", outcome, "WTSAF2YR", "WTSAF4YR", "SDDSRVYR", "SDMVSTRA", "SDMVPSU"]
    rows = working.loc[matrix.index, [c for c in keep if c in working.columns]]
    return {"n": int(len(rows)), "seqn": [int(x) for x in rows["SEQN"]],
            "outcome": [int(x) for x in rows[outcome]], "matrix_columns": list(matrix.columns),
            "matrix": {c: [float(x) for x in matrix[c]] for c in matrix.columns}}


def decisions_of(v: dict[str, Any]) -> list[dict[str, Any]]:
    """The decision log as the server recorded it: each record's decision and its sentence."""
    return [{"kind": (d.get("decision") or {}).get("kind"), "decision": d.get("decision"),
             "sentence": d.get("sentence"), "recorded_by": d.get("recorded_by")}
            for d in v.get("decisions") or []]


def the_family(results: dict[str, Any]) -> dict[str, Any]:
    families = (results.get("results") or {}).get("families") or \
        (results.get("results") or {}).get("models") or []
    family = PREREG["model"]["family"]
    for f in families:
        if f.get("family") == family or f.get("key") == family:
            return f
    raise RuntimeError(f"no {family} family in the results: "
                       f"{[f.get('family') or f.get('key') for f in families]}")


def run_one(name: str, walk: bool = False) -> dict[str, Any]:
    lane, variant = name.split(":", 1)
    s = settings(lane, variant)
    runs = Path(os.environ["TURBOTAB_SWARM_RUNS"])
    runs.mkdir(parents=True, exist_ok=True)
    a = Replicator(spec_for(s), runs)
    a.settings, a.kept = s, {}
    started = time.monotonic()
    code, out = a.drv("start", "--data", str(DERIVED), "--budget-fits", str(BUDGET_FITS),
                      "--budget-seconds", str(BUDGET_SECONDS), "--scripted")
    if code != 0:
        raise RuntimeError(f"start refused: {json.dumps(out)[:1000]}")
    a.run = out["started"]["run"]
    capture: dict[str, Any] = {"name": name, "lane": lane, "variant": variant, "run": a.run,
                               "settings": s, "error": None}
    try:
        try:
            q = a.until(lambda q: False)
            capture["at_fit"] = {"fit_reached": q.get("fit_reached"),
                                 "question": q.get("question")}
            if walk:
                capture["decisions"] = decisions_of(view(a))
            else:
                a.press("the Router reached Fit")
                results = a.results()
                v = view(a)
                capture["family"] = the_family(results)
                capture["decisions"] = decisions_of(v)
                capture["state"] = {k: (v.get("state") or {}).get(k)
                                    for k in ("purpose", "target", "event", "survey", "missing")}
                cohort = a.artifact("cohort")
                capture["flow"] = cohort.get("steps") or cohort.get("flow")
                capture["rows"] = analyzed_rows(a, v)
                code, exported = a.cmd("export", "--wait", "240")
                capture["export"] = {"exit": code, "summary": {k: exported.get(k) for k in
                                                               ("bundle", "path", "refused")}}
                bundle = runs / a.run / "export.zip"
                if bundle.is_file():
                    (OUT / "bundles").mkdir(parents=True, exist_ok=True)
                    shutil.copy(bundle, OUT / "bundles" / f"{lane}-{variant}.zip")
        except Exception as exc:  # noqa: BLE001 - the capture says how the run stopped
            capture["error"] = f"{type(exc).__name__}: {exc}"[:3000]
    finally:
        a.drv("stop", "--run", a.run)
    capture.update(kept=a.kept, commands=a.log, seconds=round(time.monotonic() - started, 1))
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    path = OUT / "runs" / f"{lane}-{variant}{'.walk' if walk else ''}.json"
    path.write_text(json.dumps(capture, indent=1, default=str) + "\n", "utf-8")
    return capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--walk", action="store_true", help="answer to Fit without pressing it")
    args = parser.parse_args(argv)
    os.environ.setdefault("TURBOTAB_SWARM_RUNS", str(RUNS_HOME))
    os.environ.setdefault("TURBOTAB_WORKERS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if args.list or not args.names:
        print("\n".join(variants()))
        return 0
    known = set(variants())
    bad = [n for n in args.names if n not in known]
    if bad:
        raise SystemExit(f"not a run of the plan: {bad}")
    if not DERIVED.is_file():
        raise SystemExit(f"{DERIVED} is missing: run prep.py")
    failed = 0
    for name in args.names:
        c = run_one(name, walk=args.walk)
        failed += c["error"] is not None
        print(json.dumps({"run": name, "error": c["error"], "seconds": c["seconds"]}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

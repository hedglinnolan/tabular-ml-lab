"""Capture what the calm kit needs that the earlier prototypes' fixtures do not hold (not bundled).

    TURBOTAB_HOME=$(mktemp -d) TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 \
        TURBOTAB_NHANES_CSV=_tt_tmp_nhanes.csv \
        venv/bin/python -m turbotab.server --port 8977                         # repo root
    venv/bin/python turbotab/frontend/src/explore/calm-kit/capture/capture.py \
        --base http://127.0.0.1:8977 --raw /tmp/calm-raw.json                  # repo root

The analysis is the one scenario (``../../methods-shared/scenario.py``, SCENARIO.md); every answer
is the scenario's, in its order. Two drives through the HTTP API, both stopped before the lock (the
fits are the methods map's capture, ``../../methods-map/fixture.json``; nothing here fits a model,
so nothing here can show an estimate):

* **main**: ``scenario.run_inference``, watched. At each moment the hook takes the previews of the
  question open there, on the state the walk meets it in, and records nothing:
  the unit (its guess and the two alternatives the card names), the secondary screens (each set
  the card can report beside the every-row analysis), the three role readings asked one at a time
  (the guess, and leaving the column out of the models), the block they unlock, the estimand's
  card, the fit's code-or-amount block. At ``ready`` (the whole plan, before the lock; the engine
  cannot draw the estimand at its own question) it takes the estimand's previews: every exposure
  the card offers, the direct effect and the addition contrast. Then the scenario's run is stopped.
  On that stopped project (never locked, never fitted), the addition contrast is recorded to
  preview the energy models an addition admits, and the direct effect to preview the rows it
  keeps.
* **branch**: the scenario on a second project, to the adjustment card, which it answers one group
  at a time as the walk does: before each group's answer it previews every role the walk offers for
  that group, on the state the groups before it leave.

In process: the engine's sentence for each alternative (``voice.sentence_for`` on the state the
question is asked in, checked against the server's sentence for the scenario's answer), and the
Strip's per-column change for each energy model (``models.steps.energy_step`` on the preview's own
sample, scored by ``consequences._shifts``, the measure the engine ranks its views by; the
fixture's ``derived`` says how each number was made).

``build.py`` then writes ``../fixture.json`` from this capture and the methods map's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parents[1] / "methods-shared"))

import scenario as S  # noqa: E402

ROLE_ALTERNATIVE = "excluded"  # the one alternative a role reading offers: leave it out of the models
UNITS = {"kcal_1": ("kcal", 1), "kj_1": ("kj", 1), "kcal_2": ("kcal", 2)}  # the card's own spellings
SENSITIVITY_SETS = {"both": ["willett_2013_by_sex", "nhs_hpfs_by_sex"],
                    "willett": ["willett_2013_by_sex"], "nhs": ["nhs_hpfs_by_sex"], "none": []}
EXCLUSION_KEYS = ["willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000",
                  "sex_neutral_500_3500", "goldberg_schofield"]
# One answer to the criterion's three questions per role the adjustment card offers, as
# (causes_exposure, causes_outcome, after_exposure). A group's own guess replaces its role's row.
ROLE_ANSWERS = {
    "confounder": ("yes", "yes", "no"),
    "timing_unknown": ("unknown", "yes", "unknown"),
    "mediator": ("no", "yes", "yes"),
    "not_a_cause": ("no", "no", "no"),
}
FIELDS = ("causes_exposure", "causes_outcome", "after_exposure")


class Stop(Exception):
    """Ends a drive at the moment its captures are taken (no lock, no fit)."""


def pv(j: S.Journey, body: dict[str, Any]) -> dict[str, Any]:
    r = j.preview(body)
    return {"decision": body, "status": r["status"],
            "result": r["body"] if r["status"] == 200 else None,
            "refusal": r["body"].get("error") if r["status"] != 200 else None}


def sensitivity(proposals: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    if not keys:
        return {"kind": "set_sensitivity", "analyses": []}
    return S.sensitivity(proposals, tuple(keys))


def drive(base: str) -> dict[str, Any]:
    """The main drive, with the single readings' previews taken before each is recorded."""
    client = S.Client(base)
    out: dict[str, Any] = {"previews": {}, "asks": {}, "states": {}, "cards": {}}
    P = out["previews"]
    holder: dict[str, Any] = {}

    def state(j: S.Journey) -> dict[str, Any]:
        return j.view()["state"]

    def take_single(j: S.Journey, column: str) -> None:
        guess = j.truth[f"role:{column}"]
        P["single"][column] = {
            "guess": pv(j, {"kind": "confirm_reading", "reading": "role", "column": column,
                            "value": guess}),
            "leave_out": pv(j, {"kind": "confirm_reading", "reading": "role", "column": column,
                                "value": ROLE_ALTERNATIVE})}

    def at(moment: str, j: S.Journey) -> None:
        holder["j"] = j
        print(f"  main · {moment}", flush=True)
        if moment == "unit_ask":
            out["asks"]["unit"] = j.ask("exclusions")
            out["states"]["unit"] = state(j)
            P["unit"] = {k: pv(j, {"kind": "set_column_unit", "column": S.ENERGY_COLUMN,
                                   "unit": u, "days": d}) for k, (u, d) in UNITS.items()}
        elif moment == "exclusions":
            proposals = j.artifact("proposals")
            out["cards"]["proposals_exclusions"] = {"exclusions": proposals["exclusions"],
                                                    "labels": proposals["labels"]}
            out["states"]["exclusions"] = state(j)
            P["exclusions"] = {"none": pv(j, {"kind": "set_exclusions", "rules": []})}
            for key in EXCLUSION_KEYS:
                if any(o["key"] == key for o in proposals["exclusions"]):
                    P["exclusions"][key] = pv(j, {"kind": "set_exclusions",
                                                  "rules": [S.screen_rule(proposals, key)]})
            P["sensitivity"] = {k: pv(j, sensitivity(proposals, keys))
                                for k, keys in SENSITIVITY_SETS.items()}
            out["sensitivity_decisions"] = {k: sensitivity(proposals, keys)
                                            for k, keys in SENSITIVITY_SETS.items()}
        elif moment == "missing":
            out["states"]["missing"] = state(j)
            P["missing"] = {
                "complete_case": pv(j, {"kind": "set_missing", "strategy": "complete_case"}),
                "multiple_imputation": pv(j, {"kind": "set_missing",
                                              "strategy": "multiple_imputation", "m": 20})}
            out["cards"]["missing"] = j.artifact("proposals").get("missing")
        elif moment == "split":
            out["states"]["split"] = state(j)
        elif moment == "readings":
            out["asks"]["roles"] = j.ask("estimand")
            out["states"]["readings"] = state(j)
            P["single"] = {}
            take_single(j, S.SINGLES[0])
        elif moment.startswith("single:"):
            col = moment.split(":", 1)[1]
            i = S.SINGLES.index(col)
            out["states"][f"after_single:{col}"] = state(j)
            if i + 1 < len(S.SINGLES):
                take_single(j, S.SINGLES[i + 1])
            else:
                ask = j.ask("estimand")
                out["asks"]["block"] = ask
                P["block"] = pv(j, ask["exits"][0]["decision"])
        elif moment == "estimand":
            out["cards"]["estimand"] = j.artifact("proposals")["estimand"]
            out["states"]["estimand"] = state(j)
            P["estimand_at_question"] = pv(j, S.estimand())
        elif moment == "adjustment":
            out["cards"]["adjustment"] = j.artifact("proposals")["adjustment"]
            out["states"]["adjustment"] = state(j)
        elif moment == "energy":
            out["cards"]["energy"] = j.artifact("proposals").get("energy")
            out["states"]["energy"] = state(j)
        elif moment == "model_sequence":
            out["cards"]["model_sequence"] = j.artifact("proposals")["model_sequence"]
            out["states"]["model_sequence"] = state(j)
        elif moment == "models":
            ask = j.ask("models")
            out["asks"]["codes"] = ask
            out["states"]["codes"] = state(j)
            P["codes"] = pv(j, ask["exits"][0]["decision"])
        elif moment == "ready":
            out["states"]["ready"] = state(j)
            card = out["cards"]["estimand"]
            exposures = [e["column"] for e in card["exposures"] if e.get("energy_contrast")]
            out["exposures"] = exposures
            P["estimand"] = {f"exposure:{c}": pv(j, S.estimand(exposure=c)) for c in exposures}
            P["estimand"]["effect:direct"] = pv(j, {**S.estimand(), "effect": "direct"})
            P["estimand"]["contrast:addition"] = pv(j, S.estimand(contrast="addition"))
            out["methods_ready"] = j.methods()
            out["decisions_ready"] = j.view()["decisions"]
            raise Stop()

    try:
        S.run_inference(client, at)
    except Stop:
        pass
    j = holder["j"]
    out["pid"] = j.pid

    # The stopped project (never locked, never fitted): what an addition admits, what a direct
    # effect keeps. Each is recorded here only to preview the next question on its state.
    j.record(S.estimand(contrast="addition"))
    j.wait_quiet()
    out["states"]["addition"] = state(j)
    energy_card = j.artifact("proposals").get("energy")
    out["cards"]["energy_under_addition"] = energy_card
    P["addition_energy"] = {}
    for m in ("all_components", "partition", "standard"):
        P["addition_energy"][m] = pv(j, S.energy(m))
    # the exits the engine offers when it refuses an addition model on these columns
    for key in ("all_components", "partition"):
        ref = P["addition_energy"][key]["refusal"]
        for k, ex in enumerate((ref or {}).get("exits") or []):
            if ex.get("decision") and ex["decision"].get("kind") == "set_energy_adjustment":
                P["addition_energy"][f"{key}:exit{k}"] = pv(j, ex["decision"])
    j.record({**S.estimand(), "effect": "direct"})
    j.wait_quiet()
    out["states"]["direct"] = state(j)
    P["direct_missing"] = pv(j, {"kind": "set_missing", "strategy": "complete_case"})
    return out


def drive_branch(base: str) -> dict[str, Any]:
    """The adjustment card answered one group at a time, each group's roles previewed first."""
    from turbotab.core.tests.truths import answer_adjustment

    client = S.Client(base)
    out: dict[str, Any] = {"groups": [], "previews": {}}

    class Collected:
        status_code = 200
        text = ""

    def at(moment: str, j: S.Journey) -> None:
        print(f"  branch · {moment}", flush=True)
        if moment == "unit_ask":
            # the card's kJ alternative (its own spelling, "kj"), on the state the unit is asked in
            out["unit_kj"] = pv(j, {"kind": "set_column_unit", "column": S.ENERGY_COLUMN,
                                    "unit": "kj", "days": 1})
            return
        if moment == "estimand":
            # the rows complete cases keep once the block has put its columns in the models
            out["rows_after_block"] = pv(j, {"kind": "set_missing", "strategy": "complete_case"})
            return
        if moment != "adjustment":
            return
        card = j.artifact("proposals")["adjustment"]
        out["card"] = card
        for group in card["groups"]:
            got: list[dict[str, Any]] = []
            answer_adjustment(lambda d: (got.append(d), Collected())[1],
                              {**card, "groups": [group]}, j.truth)
            # the scenario may split a group into several answers (the unguessed one: a confounder
            # and six mediators); each answer set is one card of the walk
            for n, decision in enumerate(got):
                cols = list(decision["answers"]) if "answers" in decision else group["columns"]
                key = group["key"] if len(got) == 1 else f"{group['key']}:{n}"
                ans = (decision.get("answers") or {}).get(cols[0])
                scenario = tuple(ans[f] for f in FIELDS) if ans else tuple(
                    group["guess"][f] for f in FIELDS)
                opts: dict[str, Any] = {}
                for role, triple in ROLE_ANSWERS.items():
                    body = {"kind": "set_adjustment", "exposure": card["exposure"],
                            "answers": {c: dict(zip(FIELDS, triple)) for c in cols}}
                    opts[role] = pv(j, body)
                scen_body = decision
                out["groups"].append({"key": key, "group": group["key"], "label": group["label"],
                                      "columns": cols, "scenario": list(scenario),
                                      "scenario_decision": scen_body,
                                      "state": j.view()["state"]})
                out["previews"][key] = {"options": opts, "scenario": pv(j, scen_body)}
                j.record(scen_body)
                j.wait_quiet()
        out["after_state"] = j.view()["state"]
        out["rows_after_adjustment"] = pv(j, {"kind": "set_missing", "strategy": "complete_case"})
        raise Stop()

    try:
        S.run_inference(client, at)
    except Stop:
        pass
    return out


def drive_direct(base: str, pid: str, branch: dict[str, Any]) -> dict[str, Any]:
    """On the main drive's stopped project (whole plan, the direct effect recorded last): the
    adjustment card's mediators answered as the scenario answers them, previewed under a direct
    effect, where the criterion holds them fixed."""
    import httpx

    c = httpx.Client(base_url=base, timeout=900)
    med = next(g for g in branch["groups"] if g["key"].startswith("unguessed")
               and g["scenario"] == list(ROLE_ANSWERS["mediator"]))
    body = {"kind": "set_adjustment", "exposure": S.EXPOSURE,
            "answers": {col: dict(zip(FIELDS, ROLE_ANSWERS["mediator"])) for col in med["columns"]}}
    r = c.post(f"/api/projects/{pid}/preview", json=body)
    state = c.get(f"/api/projects/{pid}").json()["state"]
    return {"mediators": {"decision": body, "status": r.status_code,
                          "result": r.json() if r.status_code == 200 else None,
                          "refusal": r.json().get("error") if r.status_code != 200 else None},
            "state": state}


# ── in process ───────────────────────────────────────────────────────────────


def in_process(main: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    from turbotab.core import voice
    from turbotab.core.decisions import ProjectState

    st = {k: ProjectState(**v) for k, v in main["states"].items()}
    sentences: dict[str, Any] = {"unit": {}, "single": {}, "estimand": {}, "adjustment": {},
                                 "sensitivity": {}}
    for k, (u, d) in UNITS.items():
        sentences["unit"][k] = voice.sentence_for(
            {"kind": "set_column_unit", "column": S.ENERGY_COLUMN, "unit": u, "days": d}, st["unit"])
    before = {S.SINGLES[0]: st["readings"], S.SINGLES[1]: st[f"after_single:{S.SINGLES[0]}"],
              S.SINGLES[2]: st[f"after_single:{S.SINGLES[1]}"]}
    for col, s in before.items():
        sentences["single"][col] = {
            "guess": voice.sentence_for({"kind": "confirm_reading", "reading": "role",
                                         "column": col, "value": "covariate"}, s),
            "leave_out": voice.sentence_for({"kind": "confirm_reading", "reading": "role",
                                             "column": col, "value": ROLE_ALTERNATIVE}, s)}
    block = main["asks"]["block"]["exits"][0]["decision"]
    sentences["block"] = voice.sentence_for(block, st[f"after_single:{S.SINGLES[2]}"])
    codes = main["asks"]["codes"]["exits"][0]["decision"]
    sentences["codes"] = voice.sentence_for(codes, st["codes"])
    for k, d in main["sensitivity_decisions"].items():
        sentences["sensitivity"][k] = voice.sentence_for(d, st["exclusions"])
    for c in main["exposures"]:
        sentences["estimand"][f"exposure:{c}"] = voice.sentence_for(S.estimand(exposure=c),
                                                                    st["estimand"])
    sentences["estimand"]["effect:direct"] = voice.sentence_for(
        {**S.estimand(), "effect": "direct"}, st["estimand"])
    sentences["estimand"]["contrast:addition"] = voice.sentence_for(
        S.estimand(contrast="addition"), st["estimand"])
    # every combination the three estimand cards can record (exposure × effect × contrast)
    sentences["estimand_combo"] = {}
    for c in main["exposures"]:
        for effect in ("total", "direct"):
            for contrast in ("substitution", "addition"):
                sentences["estimand_combo"][f"{c}|{effect}|{contrast}"] = voice.sentence_for(
                    {**S.estimand(exposure=c, contrast=contrast), "effect": effect}, st["estimand"])
    for g in branch["groups"]:
        s = ProjectState(**g["state"])
        per: dict[str, str] = {}
        for role, triple in ROLE_ANSWERS.items():
            body = {"kind": "set_adjustment", "exposure": S.EXPOSURE,
                    "answers": {c: dict(zip(FIELDS, triple)) for c in g["columns"]}}
            per[role] = voice.sentence_for(body, s)
        per["scenario"] = voice.sentence_for(g["scenario_decision"], s)
        sentences["adjustment"][g["key"]] = per
    # the server's sentences for the scenario's own answers, to check the in-process ones against
    said = {d["decision"]["kind"] + ":" + str(i): d["sentence"]
            for i, d in enumerate(main["decisions_ready"])}
    return {"sentences": sentences, "server_said": said}


def strip_numbers(main: dict[str, Any]) -> dict[str, Any]:
    """Each energy model's change to every nutrient, as the engine's own preview computes the one
    nutrient it draws (``models.previews.energy_adjustment_preview``): the preview's sample (5,000
    of the analyzed rows, ``PreviewContext.sample_row_ids``, seed 0), each method's fitted
    ``energy_step``, and per nutrient the engine's change score (``consequences._shifts``: the
    standardized Wasserstein distance), r with total energy and the SD, before and after."""
    import numpy as np
    import pandas as pd

    from turbotab.core.consequences import _shifts
    from turbotab.core.decisions import EnergyAdjustment
    from turbotab.core.models.pipeline import normalize_frame
    from turbotab.core.models.steps import energy_step

    raw = pd.read_csv(S.nhanes())
    pool = np.arange(len(raw))
    ids = np.sort(np.random.default_rng(0).choice(pool, size=5000, replace=False))
    nutrients = list(S.NUTRIENTS)
    cols = [S.ENERGY_COLUMN, *nutrients]
    frame = normalize_frame(raw.iloc[ids][cols].copy())
    frame.index = ids
    e = frame[S.ENERGY_COLUMN].to_numpy(dtype=float)

    def corr(x: Any) -> float | None:
        ok = np.isfinite(x) & np.isfinite(e)
        if ok.sum() < 3 or np.std(x[ok]) == 0:
            return None
        return float(np.corrcoef(x[ok], e[ok])[0, 1])

    def hist(values: Any, edges: Any) -> list[int]:
        return [int(c) for c in np.histogram(values[np.isfinite(values)], bins=edges)[0]]

    out: dict[str, Any] = {"rows": int(len(frame)), "pool": int(len(pool)), "methods": {}}
    for method in ("standard", "residual", "residual_energy_dropped", "density_multivariate",
                   "density"):
        adj = EnergyAdjustment(method=method, energy_column=S.ENERGY_COLUMN, nutrients=nutrients)
        step = energy_step(adj, cols)
        rows = []
        if step is None:
            out["methods"][method] = {"columns": []}
            continue
        fitted = step.fit(frame[cols])
        after = fitted.transform(frame[cols])
        lineage = {x["inputs"][0]: str(x["output"]) for x in fitted.lineage()
                   if x["inputs"] and x["operation"] != "partition-other"}
        for n in nutrients:
            name = lineage.get(n, n)
            b = frame[n].to_numpy(dtype=float)
            a = after[name].to_numpy(dtype=float) if name in after.columns else b
            both = pd.DataFrame({"v": b}, index=frame.index), pd.DataFrame({"v": a}, index=frame.index)
            changed = not np.allclose(b, a, equal_nan=True)
            shift = float(_shifts(both[0], both[1], ["v"]).get("v", 0.0)) if changed else 0.0
            lo = float(np.nanmin(np.concatenate([b, a]))) if changed else float(np.nanmin(b))
            hi = float(np.nanmax(np.concatenate([b, a]))) if changed else float(np.nanmax(b))
            edges_b = np.linspace(float(np.nanmin(b)), float(np.nanmax(b)), 31)
            edges_a = np.linspace(float(np.nanmin(a)), float(np.nanmax(a)), 31)
            rows.append({"column": n, "output": name, "changed": changed, "shift": shift,
                         "r_before": corr(b), "r_after": corr(a),
                         "mean_before": float(np.nanmean(b)), "mean_after": float(np.nanmean(a)),
                         "sd_before": float(np.nanstd(b)), "sd_after": float(np.nanstd(a)),
                         "hist_before": {"edges": [float(x) for x in edges_b],
                                         "counts": hist(b, edges_b)},
                         "hist_after": {"edges": [float(x) for x in edges_a],
                                        "counts": hist(a, edges_a)},
                         "range": [lo, hi]})
        out["methods"][method] = {"columns": rows,
                                  "energy_kept": S.ENERGY_COLUMN in after.columns}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8977")
    ap.add_argument("--raw", required=True, help="write the raw capture here (JSON)")
    ap.add_argument("--only", choices=["main", "branch", "direct", "inprocess"], default=None)
    args = ap.parse_args()
    path = Path(args.raw)
    raw: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    t = time.time()
    if args.only in (None, "main"):
        raw["main"] = drive(args.base)
        path.write_text(json.dumps(raw, default=str))
        print(f"main done in {time.time() - t:.0f}s", flush=True)
    if args.only in (None, "branch"):
        raw["branch"] = drive_branch(args.base)
        path.write_text(json.dumps(raw, default=str))
        print(f"branch done in {time.time() - t:.0f}s", flush=True)
    if args.only in (None, "direct"):
        raw["direct"] = drive_direct(args.base, raw["main"]["pid"], raw["branch"])
        path.write_text(json.dumps(raw, default=str))
    if args.only in (None, "inprocess"):
        raw["in_process"] = in_process(raw["main"], raw["branch"])
        raw["strip"] = strip_numbers(raw["main"])
        path.write_text(json.dumps(raw, default=str))
    raw["captured"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(raw, default=str))
    print(f"wrote {path} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

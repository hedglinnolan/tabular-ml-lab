"""The light Predict capture behind ``BEATS_PREDICT.md``: NHANES fasting glucose under Predict,
driven through the real server in process, with every number computed by the engine's own
functions.

Run it from the repository root (two workers, two threads, a fresh home)::

    venv/bin/python docs/turbotab-next/calm/predict-capture/capture.py --home <empty folder>

It writes ``capture.json`` beside this file (``--out`` to put it elsewhere).

**The journey.** The NHANES export (``turbotab/core/tests/fixtures/nhanes.csv.gz``, 21,849 adults,
nine survey cycles), the dietary lens, the outcome ``glucose``, the goal Predict. Its answers are
the reference journey's (``reference/journeys.py``, ``dietary-prediction``) except where the beats
rule otherwise, each said here:

* **Roles** (BEATS_PREDICT M1, requirement P3): every column known at a visit with a fasting draw
  is a predictor. The nutrients take the covariate role, as P3 compiles them under Predict, so the
  energy-model and substitution questions do not fire (the beats' ruling 5). The survey cycle is not a
  predictor (a later visit is in no cycle the model has seen); it is read only by the validation.
  The six ``imputed_*`` columns are processing flags, made after the visit.
* **The medicine answers** keep their blanks as a level of their own (``missing_category``):
  NHANES asks them only after a diagnosis, so a blank means "not asked" (BEATS_PREDICT M2, P1).
* **The intended use** (Your question): estimating the value, its performance reported by gender,
  age, and each medicine answer (W1-F's card; ``set_intended_use``).
* **Validation:** 20% of the rows held out at random (seed 0); internal–external validation by
  survey cycle on the rest (the beats' ruling 3); the families compared on 10 × 5-fold
  cross-validation, as the engine always compares them.
* **Families:** least squares, ridge and the elastic net only. No tuned trees: a tuned shelf on
  these rows takes hours. The shelf's own ranking and cost for every family are read, outcome-blind.
* **Each continuous measure may bend** (``set_levers``, forms by Harrell's rule; the beats' ruling
  6), declared in Models before Fit. ``--forms none`` keeps the engine's straight lines, the
  engine's default (the beats' first run); ``extras.py`` measures what the curves added on the
  final family directly.
* **The final model** is named before the held-out rows open, by the beats' ruling 4:
  among the families whose paired difference from the best includes zero, the simplest whose fit
  raised no concern its equation would carry into the paper.
* **The curves on shared axes** are declared before Fit for one measure of each group that carries
  the estimate (waist, triglycerides, HDL, age, total calories): a display, not a choice the
  scores read.

**What reads the outcome.** Everything after Fit reads glucose on the training rows (the
cross-validated scores, calibration, explanations), and everything after the opening reads the
held-out rows. In ``capture.json`` all of it sits under ``after_seal`` and is used only by the
Results beats; the Models beats use ``before_fit`` alone, which reads no glucose value.
"""
from __future__ import annotations

import os

# Two threads for every numerical library, before any of them is imported: a light run.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("TURBOTAB_WORKERS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

FAMILIES = ["linear", "ridge", "elastic_net"]
COMPLEXITY = {"linear": 0, "ridge": 1, "elastic_net": 2}  # tuned penalties: none, one, two
SUBGROUPS = ["gender", "age", "meds_hbp", "meds_chol"]
NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
FLAGS = ["imputed_weight", "imputed_height", "imputed_bmi", "imputed_waist", "imputed_bp_sys",
         "imputed_bp_di"]
ROLES = {
    "SEQN": "identifier", "cycle_begin_year": "excluded", "kcal": "energy",
    **{c: "covariate" for c in ("age", "gender", "bp_sys", "bp_di", "weight", "height", "bmi",
                                "waist", "hdl", "triglycerides", "meds_hbp", "meds_chol",
                                *NUTRIENTS)},
    **{c: "flag" for c in FLAGS},
}
SPLIT = {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5,
         "validation": "internal_external", "cluster": "cycle_begin_year"}
MISSING = {"kind": "set_missing", "strategy": "impute", "categorical": "missing_category"}
INTENDED_USE = {"kind": "set_intended_use", "use": "risk_estimation", "subgroups": SUBGROUPS,
                "fairness": "subgroup_performance"}
# The curves on shared axes (beat R3): one measure from each group that carries the estimate, declared
# before Fit. They are a display, not a choice the scores read (FOUNDATION §10, disagreement 14);
# P6 would pick them from the families' pooled importance.
CURVES = ["waist", "triglycerides", "hdl", "age", "kcal"]
EXPLAIN = {"kind": "set_explain", "curves": "ale", "exposures": CURVES, "reseeds": 5}
# M4's in-fold form (the beats' ruling 6): each continuous measure may bend, a restricted cubic spline
# with knots by Harrell's rule (``methods.levers.RuleSplines``); ``--forms none`` keeps straight
# lines, the engine's default, which is the run that measures what the curves added.
FORMS = {"kind": "set_levers", "forms": "rule"}
# Fasting plasma glucose bands (American Diabetes Association, Standards of Care, section 2):
# below 100 mg/dL; 100 to 125 (impaired fasting glucose); 126 or more (the diabetes range).
BANDS = ((None, 100.0, "under 100"), (100.0, 126.0, "100 to 125"), (126.0, None, "126 or more"))
CUT = 126.0
# The plain groups the cross-family importance table sums SHAP over (P6).
GROUPS = {
    "Blood lipids (same draw)": ["hdl", "triglycerides"],
    "Body size": ["weight", "height", "bmi", "waist"],
    "Blood pressure": ["bp_sys", "bp_di"],
    "Age": ["age"],
    "Gender": ["gender"],
    "Medicine answers": ["meds_hbp", "meds_chol"],
    "Yesterday's diet (one recall)": ["kcal", *NUTRIENTS],
}


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def finite(value: Any) -> Any:
    """JSON-safe: NaN and infinities become null; numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    return value


def get(client: Any, url: str) -> Any:
    r = client.get(url)
    try:
        body = r.json()
    except ValueError:
        body = {"text": r.text[:2000]}
    return {"status": r.status_code, "body": body}


# ── outcome-free summaries for the Models beats ──────────────────────────────


def predictors_summary(store: Any, ids: Any) -> dict[str, Any]:
    """Counts the Models beats quote, none of which reads glucose: the medicine answers by level,
    the rows by survey cycle, and the two together."""
    from turbotab.core.models.pipeline import modeling_frame

    frame = modeling_frame(store, ["meds_hbp", "meds_chol", "cycle_begin_year"], ids)
    out: dict[str, Any] = {"rows": int(len(frame))}
    for c in ("meds_hbp", "meds_chol"):
        s = frame[c]
        out[c] = {"yes": int((s == 1).sum()), "no": int((s == 0).sum()), "not_asked": int(s.isna().sum())}
    either = frame["meds_hbp"].notna() | frame["meds_chol"].notna()
    out["either_asked"] = int(either.sum())
    out["neither_asked"] = int((~either).sum())
    out["both_asked"] = int((frame["meds_hbp"].notna() & frame["meds_chol"].notna()).sum())
    # The mode of each answer among those asked: what a most-frequent fill would write into every
    # "not asked" (requirement P1's hazard).
    out["mode_fill_would_say"] = {c: ("yes" if (frame[c] == 1).sum() >= (frame[c] == 0).sum() else "no")
                                  for c in ("meds_hbp", "meds_chol")}
    cycles = frame["cycle_begin_year"].value_counts().sort_index()
    out["cycles"] = {str(int(k)): int(v) for k, v in cycles.items()}
    asked = frame.assign(asked=either).groupby("cycle_begin_year")["asked"].mean()
    out["asked_share_by_cycle"] = {str(int(k)): round(float(v), 4) for k, v in asked.items()}
    return out


# ── after the seal: what the Results beats read ──────────────────────────────


def first_repeat(cv: Any) -> dict[str, np.ndarray]:
    """The first repeat's out-of-fold predictions, outcome and training-fold mean, by row position."""
    reps = cv._repeat()
    rows, y, pred, ref = [], [], [], []
    for f, r in zip(cv.predictions, reps):
        if r != 0:
            continue
        rows.append(np.asarray(f.rows))
        y.append(np.asarray(f.y, dtype=float))
        pred.append(np.asarray(f.prediction, dtype=float))
        ref.append(np.full(len(f.rows), float(f.reference)))
    order = np.argsort(np.concatenate(rows))
    return {"rows": np.concatenate(rows)[order], "y": np.concatenate(y)[order],
            "pred": np.concatenate(pred)[order], "ref": np.concatenate(ref)[order]}


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    e = pred - y
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "mean_observed": float(np.mean(y)) if len(y) else None,
            "mean_predicted": float(np.mean(pred)) if len(y) else None}


def by_band(y: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for low, high, name in BANDS:
        keep = np.ones(len(y), dtype=bool)
        if low is not None:
            keep &= y >= low
        if high is not None:
            keep &= y < high
        entry = {"band": name, "share": float(keep.mean())}
        entry.update(scores(y[keep], pred[keep]))
        out.append(entry)
    return out


def at_cut(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    high, flagged = y >= CUT, pred >= CUT
    return {"cut": CUT, "n_at_or_above": int(high.sum()), "share_at_or_above": float(high.mean()),
            "predicted_at_or_above": int(flagged.sum()),
            "sensitivity": float((flagged & high).sum() / max(1, high.sum())),
            "ppv": float((flagged & high).sum() / max(1, flagged.sum())) if flagged.any() else None,
            "false_alarms_among_below": float((flagged & ~high).sum() / max(1, (~high).sum())),
            "max_prediction": float(np.max(pred)),
            "prediction_p99": float(np.quantile(pred, 0.99)),
            "observed_p99": float(np.quantile(y, 0.99))}


def by_tenth(y: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    """Mean observed against mean predicted in each tenth of the predictions (calibration's own
    picture, by the predictions, never by the outcome), with the share at or above the cut."""
    edges = np.quantile(pred, np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, 9)
    out = []
    for k in range(10):
        keep = which == k
        out.append({"tenth": k + 1, "n": int(keep.sum()),
                    "mean_predicted": float(pred[keep].mean()), "mean_observed": float(y[keep].mean()),
                    "share_at_or_above_cut": float((y[keep] >= CUT).mean())})
    return out


def outcome_shape(y: np.ndarray) -> dict[str, Any]:
    from scipy.stats import skew

    q = np.quantile(y, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"n": int(len(y)), "mean": float(y.mean()), "sd": float(y.std(ddof=1)),
            "skew": float(skew(y)), "log_skew": float(skew(np.log(y))) if (y > 0).all() else None,
            "quantiles": {"p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
                          "p75": float(q[3]), "p95": float(q[4]), "p99": float(q[5])},
            "share_100_to_125": float(((y >= 100) & (y < 126)).mean()),
            "share_at_or_above_126": float((y >= CUT).mean()),
            "share_above_200": float((y > 200).mean()),
            "squared_error_share_from_at_or_above_126_baseline": None}


def importance_table(explain: dict[str, Any]) -> dict[str, Any]:
    """P6's prototype: each family's mean |SHAP| by input and by plain group, on one scale (mg/dL),
    with each family's rank of the groups."""
    families = [f for f in explain.get("families") or [] if f.get("explained")]
    inputs: dict[str, dict[str, float]] = {}
    for fam in families:
        for row in fam.get("importance") or []:
            inputs.setdefault(row["input"], {})[fam["family"]] = float(row["mean_abs"])

    def raw_of(name: str) -> str:
        # one-hot inputs and missing levels keep their source column first in the name
        for raw in ROLES:
            if name == raw or name.startswith(f"{raw}_") or name.startswith(f"{raw}="):
                return raw
        return name

    groups: dict[str, dict[str, float]] = {}
    for name, by in inputs.items():
        raw = raw_of(name)
        group = next((g for g, cols in GROUPS.items() if raw in cols), f"other: {raw}")
        for fam, v in by.items():
            groups.setdefault(group, {}).setdefault(fam, 0.0)
            groups[group][fam] += v
    ranks = {}
    for fam in families:
        key = fam["family"]
        ordered = sorted(groups, key=lambda g: -groups[g].get(key, 0.0))
        ranks[key] = ordered
    return {"by_input": inputs, "by_group": groups, "group_rank": ranks,
            "note": "Sums of mean |SHAP| over each group's inputs: a sum over correlated inputs, "
                    "read as the group's share, never as one input's."}


def after_seal(client: Any, pid: str, final: str) -> dict[str, Any]:
    """What the Results beats read, computed after the held-out rows were opened."""
    from turbotab.core.config import default_memory_budget
    from turbotab.core.datastore import DataStore
    from turbotab.core.graph import read_artifact
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.working import working_paths

    service = client.app.state.service
    cache = service.workspace.cache_dir(pid)
    view = client.get(f"/api/projects/{pid}").json()
    stages = view["stages"]
    fit = read_artifact(cache, "fit", stages["fit"]["key"])
    design = read_artifact(cache, "design", stages["design"]["key"])
    working = read_artifact(cache, "working", stages["working"]["key"])
    split = read_artifact(cache, "split", stages["split"]["key"])
    spec = DesignSpec.from_dict(design.objects["spec"])
    comparison = fit.objects["comparison"]
    substrates = comparison["results"]
    out: dict[str, Any] = {"inputs": list(spec.inputs), "final": final}

    # Out of fold, the first repeat of the 10 × 5-fold comparison (training rows).
    oof = {k: first_repeat(cv) for k, cv in substrates.items()}
    any_key = next(iter(oof))
    y = oof[any_key]["y"]
    out["training_outcome"] = outcome_shape(y)
    base = oof[any_key]["ref"]
    out["oof"] = {"baseline": {"overall": scores(y, base), "by_band": by_band(y, base)}}
    sq = (base - y) ** 2
    out["training_outcome"]["squared_error_share_from_at_or_above_126_baseline"] = float(
        sq[y >= CUT].sum() / sq.sum())
    for k, o in oof.items():
        assert np.allclose(o["y"], y), "the families' first repeats score the same rows"
        e2 = (o["pred"] - y) ** 2
        out["oof"][k] = {"overall": scores(y, o["pred"]), "by_band": by_band(y, o["pred"]),
                         "at_cut": at_cut(y, o["pred"]), "by_tenth": by_tenth(y, o["pred"]),
                         "squared_error_share_from_at_or_above_126": float(e2[y >= CUT].sum() / e2.sum()),
                         "prediction_sd": float(np.std(o["pred"], ddof=1)),
                         "outcome_sd": float(np.std(y, ddof=1))}

    # The held-out rows, opened once: the final family's predictions from its fit on every
    # training row (the fit's own ``fitted``).
    sealed = split.frames["sealed"]["row_id"].to_numpy(dtype=np.int64)
    table = working_paths(working)["table"]
    store = DataStore(table, int(default_memory_budget()))
    held = modeling_frame(store, [*spec.inputs, "glucose"], sealed, outcome="glucose")
    # (the store is closed at the end of this function)
    yh = held["glucose"].to_numpy(dtype=float)
    keep = np.isfinite(yh)
    yh = yh[keep]
    Xh = held[list(spec.inputs)].iloc[np.flatnonzero(keep)]
    train_mean = float(np.mean(y))  # the training rows' mean, the no-predictor model on new rows
    out["held_out"] = {"n": int(len(yh)), "baseline": scores(yh, np.full(len(yh), train_mean)),
                       "baseline_by_band": by_band(yh, np.full(len(yh), train_mean))}
    for k in FAMILIES:
        model = fit.objects["fitted"].get(k)
        if model is None:
            continue
        ph = np.asarray(model.predict(Xh), dtype=float)
        out["held_out"][k] = {"overall": scores(yh, ph), "by_band": by_band(yh, ph),
                              "at_cut": at_cut(yh, ph), "by_tenth": by_tenth(yh, ph)}
    out["held_out"]["outcome"] = outcome_shape(yh)
    # Who is in the held-out rows, outcome-free: the medicine answers and the cycles.
    out["held_out"]["predictors"] = predictors_summary(store, sealed)
    out["training_predictors"] = predictors_summary(store, comparison["train_ids"])
    store.close()
    return out


# ── what is kept ─────────────────────────────────────────────────────────────


def trim(cap: dict[str, Any]) -> dict[str, Any]:
    """The capture as committed: every number the beats cite, without the per-row arrays the
    explanations and the stage cards carry (the beeswarm, the per-row attributions, each refit's
    importances, the cards' option lists and samples). ``--full`` keeps everything."""
    import copy

    out = copy.deepcopy(cap)
    before, after = out.get("before_fit") or {}, out.get("after_seal") or {}
    for name in ("whos_in", "models_entry"):
        snap = before.get(name) or {}
        prop = snap.get("proposals") or {}
        keep = {k: prop.get(k) for k in ("labels",) if k in prop}
        if "labels" in keep:
            keep["labels"] = {q: v for q, v in (keep["labels"] or {}).items() if q in ("missing",)}
        snap["proposals"] = keep
        if "cohort" in snap:
            snap["cohort"] = {k: (snap["cohort"] or {}).get(k) for k in ("n_measured", "n_analyzed",
                                                                         "steps", "flow")}
    for key in ("quest_at_plan", "methods_at_plan"):
        before.pop(key, None)
    for name in ("whos_in", "at_the_draw"):  # the quest log is kept once, entering Models
        (before.get(name) or {}).pop("quest", None)
    design = before.get("design") or {}
    before["design"] = {k: design.get(k) for k in ("matrix", "warnings", "left_out")}
    explain = after.get("explain_before_opening") or {}
    explain.pop("row_ids", None)
    for fam in explain.get("families") or []:
        fam.pop("beeswarm", None)
        fam.pop("observations", None)
        st = fam.get("stability") or {}
        for k in ("importance", "pairwise", "versus_fit"):
            st.pop(k, None)
        arch = fam.get("architecture") or {}
        eq = arch.get("equation") or {}
        if isinstance(eq, dict) and eq.get("terms"):
            eq["terms"] = eq["terms"][:12]
    for key in ("record_before_opening", "methods_before_opening", "checklist_before_opening",
                "quest_opened", "record_opened", "methods_opened", "checklist_opened",
                "triage_opened"):
        after.pop(key, None)
    for key in ("fit_before_opening", "fit_opened"):
        fit = after.get(key) or {}
        for m in fit.get("models") or []:
            m.pop("coefficients", None)
            for name in ("calibration",):
                if isinstance(m.get(name), dict):
                    m[name].pop("curve", None) if key == "fit_opened" else None
    return out


# ── the drive ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", required=True, type=Path, help="an empty folder: the fresh home")
    parser.add_argument("--out", type=Path, default=HERE / "capture.json")
    parser.add_argument("--forms", choices=("rule", "none"), default="rule",
                        help="each continuous measure bends (rule, the beats' default) or not")
    parser.add_argument("--full", action="store_true",
                        help="keep the per-row arrays the committed capture leaves out")
    args = parser.parse_args(argv)
    forms = {**FORMS, "forms": args.forms}
    home = args.home.resolve()
    if home.exists() and any(home.iterdir()):
        parser.error(f"{home} is not empty: the capture needs a fresh home")
    home.mkdir(parents=True, exist_ok=True)
    os.environ["TURBOTAB_HOME"] = str(home)

    from turbotab.core.reference.journeys import (NHANES_FILE, NHANES_KEY, Journey, Recording, Run,
                                                  _server_guess, _server_ranked, apply_first_repair,
                                                  bundle_parts, declare, export, follow,
                                                  nhanes_dietary_truth, nhanes_path, post_answer,
                                                  wait_results)
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

    cap: dict[str, Any] = {"before_fit": {}, "after_seal": {}, "notes": []}
    snaps = cap["before_fit"]

    def snapshot(name: str, *stage_names: str) -> Any:
        def go(run: Run, view: dict[str, Any]) -> None:
            c, pid = run.client, run.pid
            snap: dict[str, Any] = {"quest": get(c, f"/api/projects/{pid}/quest")["body"]}
            for stage in stage_names:
                try:
                    snap[stage] = run.drive.artifact(stage, timeout=600)
                except Exception as exc:  # noqa: BLE001 - a stage that fails is kept with its reason
                    snap[stage] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
            snaps[name] = snap
            log(f"snapshot {name}: {', '.join(stage_names) or 'quest'}")
        return go

    def choose_models_and_press(run: Run, step: dict[str, Any], view: dict[str, Any]) -> None:
        # The shelf, outcome-blind, as the families card reads it.
        snaps["shelf"] = run.drive.artifact("shelf", timeout=1800)
        run.source = "the beats' families (linear, ridge, elastic net; no tuned trees)"
        post_answer(run, {"kind": "select_models", "models": FAMILIES})
        try:  # what each model is given: the lineage and the model matrix (the flowchart's data)
            snaps["design"] = run.drive.artifact("design", timeout=600)
        except Exception as exc:  # noqa: BLE001 - kept with its reason
            snaps["design"] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
        snaps["plan"] = get(run.client, f"/api/projects/{run.pid}/plan")["body"]
        snaps["quest_at_plan"] = get(run.client, f"/api/projects/{run.pid}/quest")["body"]
        snaps["methods_at_plan"] = get(run.client, f"/api/projects/{run.pid}/methods")["body"]
        log("models chosen; pressing Fit")
        cap.setdefault("timing", {})["fit_pressed"] = time.strftime("%H:%M:%S")
        cap["fit_pressed"] = run.drive.press_fit()
        return None

    def read_before_opening(run: Run, view: dict[str, Any]) -> None:
        d, c, pid = run.drive, run.client, run.pid
        after = cap["after_seal"]
        for stage in ("fit", "evaluation", "explain"):
            log(f"waiting for {stage}")
            after[f"{stage}_before_opening"] = d.artifact(stage, timeout=5400)
            cap.setdefault("timing", {})[f"{stage}_fresh"] = time.strftime("%H:%M:%S")
        after["importance"] = importance_table(after["explain_before_opening"])
        for name in ("quest", "triage", "record", "methods", "checklist", "materiality"):
            after[f"{name}_before_opening"] = get(c, f"/api/projects/{pid}/{name}")["body"]
        log("read everything before the opening")

    def choose_final(run: Run, step: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
        """Ruling 4: among the families whose paired difference from the best includes zero (the
        families compared on 10 × 5-fold cross-validation, corrected t), the simplest whose fit
        raised no concern its equation would carry into the paper (TRIPOD+AI 22: the full model);
        the simplest of them all when every one raised one."""
        fit = cap["after_seal"]["fit_before_opening"]
        metric = fit["primary_metric"]
        scored = {m["family"]: (m.get("compared_on") or {}).get("estimate") for m in fit["models"]}
        concerns = {m["family"]: list(m.get("concerns") or []) for m in fit["models"]}
        best = min((k for k, v in scored.items() if v is not None), key=lambda k: scored[k])
        not_worse = {best}
        for d in fit.get("comparisons") or []:
            pair = {d.get("a"), d.get("b")}
            if best in pair and d.get("ci_low") is not None and d["ci_low"] <= 0 <= d["ci_high"]:
                not_worse |= pair
        clean = [k for k in not_worse if not concerns.get(k)]
        chosen = min(clean or not_worse, key=lambda k: COMPLEXITY.get(k, 9))
        cap["final_rule"] = {"metric": metric, "compared_on": scored, "best": best,
                             "not_measurably_worse": sorted(not_worse), "concerns": concerns,
                             "without_concern": sorted(clean), "chosen": chosen}
        log(f"final model by the rule: {chosen} (best on the comparison: {best})")
        return {"kind": "open_seal", "family": chosen}

    spec = Journey(
        "predict-beats", "dietary", "prediction",
        "How well does what is known at a visit with a fasting draw predict fasting glucose?",
        NHANES_FILE, "BEATS_PREDICT's light capture (docs/turbotab-next/calm/predict-capture).",
        nhanes_path, nhanes_dietary_truth, target="glucose", lenses=("dietary",), roles=ROLES,
        answers={"missing": MISSING, "split": SPLIT, "models": choose_models_and_press,
                 "open_seal": choose_final},
        before={"target": apply_first_repair(r"^sas_zeros"),
                "roles": declare(INTENDED_USE),
                "exclusions": snapshot("whos_in", "proposals", "cohort"),
                "split": snapshot("at_the_draw", "seal_plan"),
                "models": lambda run, view: (snapshot("models_entry", "split", "proposals")(run, view),
                                             declare(EXPLAIN, *([forms] if forms["forms"] != "none"
                                                                else []))(run, run.drive.view())),
                "open_seal": read_before_opening},
        needs=(str(nhanes_path()),), fixture_key=NHANES_KEY, timeout=7200.0)

    run = Run(spec)
    started = time.monotonic()
    log(f"home {home}")
    try:
        with local_server(home) as client:
            run.client = Recording(client, run)
            r = client.post("/api/projects", json={"path": str(spec.path())})
            r.raise_for_status()
            run.pid = r.json()["id"]
            t = spec.truth()
            t.guess = _server_guess(client, run.pid)
            t.ranked = _server_ranked(client, run.pid)
            run.drive = Drive(run.client, run.pid, t)
            run.drive.artifact("ingest", timeout=900)
            log("following the Router")
            ok = follow(run)
            cap["followed"] = ok
            if ok and wait_results(run, timeout=5400):
                after = cap["after_seal"]
                after["fit_opened"] = run.drive.artifact("fit", timeout=1800)
                for name in ("quest", "triage", "record", "methods", "checklist"):
                    after[f"{name}_opened"] = get(client, f"/api/projects/{run.pid}/{name}")["body"]
                response = export(run)
                cap["export_status"] = getattr(response, "status_code", None)
                if getattr(response, "status_code", None) == 200:
                    after["bundle"] = bundle_parts(response.content, home)
                final = (cap.get("final_rule") or {}).get("chosen") or FAMILIES[0]
                log("computing the Results beats' numbers with the engine's functions")
                after["computed"] = after_seal(client, run.pid, final)
            view = client.get(f"/api/projects/{run.pid}").json()
            snaps["state_at_end"] = {k: view["state"].get(k) for k in (
                "purpose", "target", "task", "split", "missing", "models", "intended_use",
                "explain", "energy_adjustment", "substitution", "survey", "exclusions")}
            snaps["interview_at_end"] = [{"key": s["key"], "status": s["status"],
                                          "reason": s.get("reason")} for s in view["interview"]]
            # Outcome-free counts for the Models beats, from the working table.
            from turbotab.core.config import default_memory_budget
            from turbotab.core.datastore import DataStore
            from turbotab.core.graph import read_artifact
            from turbotab.core.stages.working import working_paths

            cache = client.app.state.service.workspace.cache_dir(run.pid)
            working = read_artifact(cache, "working", view["stages"]["working"]["key"])
            with DataStore(working_paths(working)["table"], int(default_memory_budget())) as store:
                snaps["predictors_all_rows"] = predictors_summary(store, None)
    except Exception as exc:  # noqa: BLE001 - kept with its reason
        cap["notes"].append(f"stopped by {type(exc).__name__}: {str(exc)[:1500]}")
        traceback.print_exc()
    cap["answers"] = run.log
    cap["notes"] += run.notes
    cap["forms"] = args.forms
    cap["captured"] = {"seconds": round(time.monotonic() - started), "date": time.strftime("%Y-%m-%d"),
                       "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=10",
                                                 "HEAD"], capture_output=True, text=True).stdout.strip(),
                       "workers": 2, "threads": os.environ.get("OMP_NUM_THREADS")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(finite(cap if args.full else trim(cap)), ensure_ascii=False, indent=1,
                      default=str)
    args.out.write_text(text.replace(str(home), "<home>") + "\n", encoding="utf-8")
    log(f"wrote {args.out} ({round(time.monotonic() - started)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

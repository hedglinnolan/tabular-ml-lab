"""The light Predict capture behind ``BEATS_PREDICT.md`` (Round 1): NHANES fasting glucose under
Predict, driven through the real server in process, with every number computed by the engine's
own functions.

Run it from the repository root (two workers, two threads, a fresh home)::

    venv/bin/python docs/turbotab-next/calm/predict-capture/capture.py --home <empty folder>

It writes ``capture.json`` beside this file (``--out`` to put it elsewhere). ``--shelf-only``
stops at the families question, before Fit, and keeps the shelf (the beats read the tree families'
costs from a straight-line shelf: ``--forms none --shelf-only``); the shelf itself reads glucose's
mean and spread on the development rows, for one sample-size criterion (the beats' C17).

**The table.** The NHANES export (``turbotab/core/tests/fixtures/nhanes.csv.gz``, 21,849 adults,
nine survey cycles), prepared by the answers beat D ("What your data raised") gives before the
draw, which the engine does not serve yet (requirements P22–P24); :func:`prepare_table` applies
them and says what it changed:

* **Values filled in before the table was made** (the six ``imputed_*`` flags, 2,444 people) are
  read as missing: their method is unknown, and some are impossible (waists to 323 cm). The
  in-fold fill then fills them in each training fold, without glucose.
* **Diastolic pressures stored as a SAS zero** (119, ``5.4e-79``) are read as missing: a resting
  diastolic pressure of 0 mmHg is not a measurement. The nutrients' 33 SAS zeros stay, and the
  engine's own repair reads them as 0 g.
* **Age**: 80 and over is one age in every cycle (NHANES codes 85 for 85 and over in 2001–2006,
  and 80 for 80 and over from 2007).
* **Fasting glucose**: NHANES's published equations put each cycle's value on the instrument that
  followed it, where an equation exists (the laboratory documentation of GLU_D, GLU_E and GLU_I):
  2001–2004 (Cobas Mira) to the Hitachi 911 and on to the Modular P; 2005–2006 (Hitachi 911) to the
  Modular P; 2013–2014 (Cobas C501) to the Cobas C311 used from 2015. NHANES published none for the
  move from the University of Minnesota (2005–2012) to the University of Missouri (2013 on).

**The journey.** The dietary lens, the outcome ``glucose``, the goal Predict. Its answers are the
reference journey's (``reference/journeys.py``, ``dietary-prediction``) except where the beats rule
otherwise, each said here:

* **Roles** (beat M1, P3): every column known at a visit with a fasting draw is a predictor; the
  nutrients take the covariate role, as P3 compiles them under Predict (ruling 5). The survey cycle
  is not a predictor; it is read only by the draw and the validation. The ``imputed_*`` flags
  mark the values read as missing and are no predictor.
* **The medicine answers** keep their blanks as a level of their own (``missing_category``): a
  blank means "not asked" (beat M2, P1). The numbers' blanks are filled in each training fold.
* **The intended use** (beat Q): estimating the value, its performance reported by gender, age and
  each medicine answer (``set_intended_use``).
* **The held-out rows** (beat W, P21): the model is used after the data's years, so the latest
  cycle, 2017–2018, is sealed whole. The engine cannot draw that seal yet (it refuses
  ``set_temporal`` when each person appears once), so the run emulates it: a rule keeps the latest
  cycle out of the analysis, no rows are drawn at random, and ``extras.py --open`` opens the
  latest cycle once, after the final model and its level are declared. Internal–external
  validation by survey cycle on the other eight (the beats' ruling 3); the families compared on
  10 × 5-fold cross-validation.
* **Families:** least squares, ridge and the elastic net only. No tuned trees: a tuned shelf on
  these rows takes hours. The shelf's own ranking and cost for every family are read.
* **Each continuous measure may bend** (``set_levers``, forms by Harrell's rule; ruling 6).
  ``--forms none`` keeps straight lines.
* **The final model** is named before the latest cycle opens (``declared_before_opening``), by
  ruling 4: among the families whose paired difference from the best lies within a stated margin
  (a twentieth of the best family's gain over the no-predictor model), the simplest whose fit
  raised no concern its equation would carry into the paper.
* **The level for a later visit** (beat M5, ruling 10) is declared in the plan before Fit; the
  engine does not serve it yet (P25), so ``extras.py`` applies the declared rule and scores the
  opened cycle with it.

**What reads the outcome.** Everything after Fit reads glucose on the development rows (the
cross-validated scores, calibration, explanations), and everything after the opening reads the
held-out rows. In ``capture.json`` all of it sits under ``after_seal`` and is used only by the
Results beats; the Models beats use ``before_fit`` alone, whose numbers read no glucose value except
the shelf's own reading of its mean and spread on the development rows, for one sample-size
criterion (said in beat M3).
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
MARGIN_SHARE = 0.05  # ruling 4: equivalent within a twentieth of the best family's gain
SUBGROUPS = ["gender", "age", "meds_hbp", "meds_chol"]
NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
FLAGS = {"imputed_weight": "weight", "imputed_height": "height", "imputed_bmi": "bmi",
         "imputed_waist": "waist", "imputed_bp_sys": "bp_sys", "imputed_bp_di": "bp_di"}
ROLES = {
    "SEQN": "identifier", "cycle_begin_year": "excluded", "kcal": "energy",
    **{c: "covariate" for c in ("age", "gender", "bp_sys", "bp_di", "weight", "height", "bmi",
                                "waist", "hdl", "triglycerides", "meds_hbp", "meds_chol",
                                *NUTRIENTS)},
    **{c: "flag" for c in FLAGS},
}
LATEST = 2017  # the latest survey cycle's first year: held out whole (beat W)
# P21's seal, emulated: the engine has no holdout of whole levels of a period column, and refuses
# ``set_temporal`` for a table where each person appears once ("temporal prediction does not
# arise"). So the latest cycle is kept out of the analysis by a rule (the server reads its rows
# into the working table but no stage analyzes them), no rows are drawn at random
# (``holdout: 0``), and ``extras.py --open`` opens the latest cycle once, with the final model and
# the level the run declared, as P21's opening would.
SEALED = {"kind": "range", "column": "cycle_begin_year", "low": 2001.0, "high": float(LATEST - 2),
          "reason": "The latest survey cycle, 2017–2018, is sealed whole until the final model is "
                    "named (P21, emulated).", "missing": "exclude"}
EXCLUSIONS = {"kind": "set_exclusions", "rules": [SEALED]}
SPLIT = {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5,
         "validation": "internal_external", "cluster": "cycle_begin_year"}
MISSING = {"kind": "set_missing", "strategy": "impute", "categorical": "missing_category"}
INTENDED_USE = {"kind": "set_intended_use", "use": "risk_estimation", "subgroups": SUBGROUPS,
                "fairness": "subgroup_performance"}
# The curves on shared axes (beat R3): one measure from each group that carries the estimate, declared
# before Fit; a display, not a choice the scores read (P6 would pick them from the pooled importance).
CURVES = ["waist", "triglycerides", "hdl", "age", "kcal"]
EXPLAIN = {"kind": "set_explain", "curves": "ale", "exposures": CURVES, "reseeds": 5}
FORMS = {"kind": "set_levers", "forms": "rule"}
# Beat M5's one decision, declared with the plan before Fit (ruling 10). The engine has no such
# updating rule yet (``set_updating`` knows "none" and "shrinkage"; requirement P25), so the capture
# records the declaration here, at the press, and ``extras.py`` applies it before the opening.
LEVEL_RULE = {
    "answer": "set its level on the latest cycle, if the cycles shift",
    "trigger": "the latest development cycle's out-of-cycle mean miss has a 95% interval that "
               "excludes 0",
    "update": "the final model's intercept re-estimated on the latest development cycle "
              "(2015–2016), its coefficients kept (temporal recalibration, Booth et al. 2020)",
}
AGE_TOP = 80.0  # NHANES's top code from 2007; 2001–2006 coded 85 for 85 and over
# NHANES's published equations for fasting plasma glucose, each to the instrument that followed
# (GLU_D 2005–2006: Hitachi 911 = 0.9815 × Cobas Mira + 3.5707; GLU_E 2007–2008: Modular P =
# Hitachi 911 + 1.148; GLU_I 2015–2016: C311 = 1.023 × C501 − 0.5108). Each cycle's chain, as
# (slope, intercept) applied in order.
LAB_EQUATIONS = {
    2001: [(0.9815, 3.5707), (1.0, 1.148)],
    2003: [(0.9815, 3.5707), (1.0, 1.148)],
    2005: [(1.0, 1.148)],
    2013: [(1.023, -0.5108)],
}
LAB = [  # (cycles, laboratory, instrument, the scale its values end on)
    ("2001–2004", "University of Missouri", "Roche Cobas Mira", "Modular P (Minnesota)"),
    ("2005–2006", "University of Minnesota", "Roche/Hitachi 911", "Modular P (Minnesota)"),
    ("2007–2012", "University of Minnesota", "Roche Modular P", "Modular P (Minnesota)"),
    ("2013–2014", "University of Missouri", "Roche Cobas C501", "Cobas C311 (Missouri)"),
    ("2015–2018", "University of Missouri", "Roche Cobas C311", "Cobas C311 (Missouri)"),
]


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


# ── beat D's answers, applied before the upload ──────────────────────────────


def prepare_table(dest: Path) -> dict[str, Any]:
    """The NHANES export with beat D's answers applied (module docstring), written to ``dest``.
    Returns what changed, by column and cycle; none of it reads a glucose value (the equations
    rewrite glucose without looking at it)."""
    import pandas as pd

    from turbotab.core.repairs import is_sas_zero
    from turbotab.core.reference.journeys import nhanes_path

    df = pd.read_csv(nhanes_path())
    out: dict[str, Any] = {"rows": int(len(df))}
    # 1. Values filled in before the table was made: read as missing.
    filled: dict[str, Any] = {}
    any_flag = np.zeros(len(df), dtype=bool)
    for flag, column in FLAGS.items():
        mask = df[flag].astype(bool).to_numpy()
        any_flag |= mask
        values = df.loc[mask, column]
        measured = df.loc[~mask, column]
        filled[column] = {"n": int(mask.sum()), "filled_max": float(values.max()) if len(values) else None,
                          "filled_min": float(values.min()) if len(values) else None,
                          "measured_max": float(measured.max()), "measured_min": float(measured.min())}
        df.loc[mask, column] = np.nan
    out["filled"] = filled
    out["filled_people"] = int(any_flag.sum())
    out["filled_share"] = float(any_flag.mean())
    raw = pd.read_csv(nhanes_path())
    bmi_calc = raw["weight"] / (raw["height"] / 100.0) ** 2
    body_flag = raw[["imputed_weight", "imputed_height", "imputed_bmi"]].astype(bool).any(axis=1)
    out["bmi_identity_gap"] = {"filled_rows_max": float((raw["bmi"] - bmi_calc).abs()[body_flag].max()),
                               "measured_rows_max": float((raw["bmi"] - bmi_calc).abs()[~body_flag].max())}
    out["waist_filled_above_measured_max"] = int(
        (raw.loc[raw["imputed_waist"].astype(bool), "waist"] > raw.loc[~raw["imputed_waist"].astype(bool), "waist"].max()).sum())
    out["filled_by_cycle"] = {str(int(k)): int(v) for k, v in
                              pd.Series(any_flag).groupby(df["cycle_begin_year"]).sum().items()}
    # 2. Diastolic SAS zeros: read as missing (the nutrients' stay for the engine's repair).
    zeros = is_sas_zero(df["bp_di"].to_numpy(dtype=float))
    out["diastolic_zeros"] = int(zeros.sum())
    df.loc[zeros, "bp_di"] = np.nan
    numeric = df.select_dtypes("number")
    others = {c: int(is_sas_zero(numeric[c].to_numpy(dtype=float)).sum()) for c in numeric.columns}
    out["other_sas_zeros"] = {c: n for c, n in others.items() if n}
    # 3. Age: 80 and over as one age.
    out["age_top_by_cycle"] = {str(int(k)): float(v) for k, v in
                               df.groupby("cycle_begin_year")["age"].max().items()}
    out["age_at_85"] = int((df["age"] >= 85).sum())
    out["age_80_to_84_before_2007"] = int(((df["age"] >= 80) & (df["age"] < 85)
                                           & (df["cycle_begin_year"] < 2007)).sum())
    out["age_capped"] = int((df["age"] > AGE_TOP).sum())
    df["age"] = df["age"].clip(upper=AGE_TOP)
    # 4. Fasting glucose: NHANES's equations, forward, where one exists.
    adjusted = {}
    for cycle, chain in LAB_EQUATIONS.items():
        mask = (df["cycle_begin_year"] == cycle).to_numpy()
        values = df.loc[mask, "glucose"].to_numpy(dtype=float)
        for slope, intercept in chain:
            values = slope * values + intercept
        df.loc[mask, "glucose"] = values
        adjusted[str(cycle)] = {"rows": int(mask.sum()), "equations": chain}
    out["glucose_adjusted"] = adjusted
    out["laboratory"] = [dict(zip(("cycles", "laboratory", "instrument", "scale"), row)) for row in LAB]
    out["cycles"] = {str(int(k)): int(v) for k, v in df["cycle_begin_year"].value_counts().sort_index().items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    out["file"] = dest.name
    return out


# ── outcome-free summaries for the Models beats ──────────────────────────────


def predictors_summary(store: Any, ids: Any) -> dict[str, Any]:
    """Counts the Models beats quote, none of which reads glucose: the medicine answers by level,
    the rows by survey cycle, and the two together."""
    from turbotab.core.models.pipeline import modeling_frame

    frame = modeling_frame(store, ["meds_hbp", "meds_chol", "cycle_begin_year", "gender"], ids)
    out: dict[str, Any] = {"rows": int(len(frame))}
    for c in ("meds_hbp", "meds_chol"):
        s = frame[c]
        out[c] = {"yes": int((s == 1).sum()), "no": int((s == 0).sum()), "not_asked": int(s.isna().sum())}
    either = frame["meds_hbp"].notna() | frame["meds_chol"].notna()
    out["either_asked"] = int(either.sum())
    out["neither_asked"] = int((~either).sum())
    out["both_asked"] = int((frame["meds_hbp"].notna() & frame["meds_chol"].notna()).sum())
    out["mode_fill_would_say"] = {c: ("yes" if (frame[c] == 1).sum() >= (frame[c] == 0).sum() else "no")
                                  for c in ("meds_hbp", "meds_chol")}
    cycles = frame["cycle_begin_year"].value_counts().sort_index()
    out["cycles"] = {str(int(k)): int(v) for k, v in cycles.items()}
    asked = frame.assign(asked=either).groupby("cycle_begin_year")["asked"].mean()
    out["asked_share_by_cycle"] = {str(int(k)): round(float(v), 4) for k, v in asked.items()}
    out["gender"] = {str(k): int(v) for k, v in frame["gender"].value_counts().items()}
    return out


def measures_summary(store: Any, ids: Any) -> dict[str, Any]:
    """The outcome-free pictures of Models' Set for you (beat M4), on the development rows: waist's
    spread with Harrell's five knots and two equal steps, and how much of each body measure the
    other three carry; the missing values by measure after beat D."""
    import pandas as pd

    from turbotab.core.models.pipeline import modeling_frame

    cols = ["weight", "height", "bmi", "waist", "bp_sys", "bp_di", "age"]
    frame = modeling_frame(store, cols, ids)
    out: dict[str, Any] = {"rows": int(len(frame))}
    waist = frame["waist"].dropna().to_numpy(dtype=float)
    q = np.quantile(waist, [0.05, 0.275, 0.5, 0.725, 0.95])
    out["waist"] = {"n": int(len(waist)), "knots": [float(v) for v in q],
                    "p01": float(np.quantile(waist, 0.01)), "p99": float(np.quantile(waist, 0.99)),
                    "share_80_90": float(((waist >= 80) & (waist < 90)).mean()),
                    "share_110_120": float(((waist >= 110) & (waist < 120)).mean())}
    body = frame[["weight", "height", "bmi", "waist"]].dropna()
    shared = {}
    for c in body.columns:
        others = body.drop(columns=c).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(body)), others, others ** 2])
        y = body[c].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        shared[c] = float(1 - resid.var() / y.var())
    out["body_shared_r2"] = shared
    out["body_corr"] = {f"{a}~{b}": float(body[a].corr(body[b]))
                        for i, a in enumerate(body.columns) for b in body.columns[i + 1:]}
    log_identity = np.log(body["bmi"]) - (np.log(body["weight"]) - 2 * np.log(body["height"] / 100))
    out["bmi_identity_max_gap_log"] = float(np.abs(log_identity).max())
    out["missing_after_d"] = {c: int(frame[c].isna().sum()) for c in cols}
    out["age_max"] = float(frame["age"].max())
    return out


def riley(parameters: int) -> dict[str, Any]:
    """Riley et al.'s outcome-free criteria (C2–C4) at this many parameters (C1, the intercept,
    needs the outcome's mean and spread and is not computed here)."""
    from turbotab.core.models.sample_size import continuous_minimum

    m = continuous_minimum(parameters)
    return {"parameters": parameters, "minimum": m.n, "binding": m.binding.key,
            "criteria": [{"key": c.key, "what": c.what, "n": c.n} for c in m.criteria]}


# ── the drive ────────────────────────────────────────────────────────────────


class Stop(Exception):
    """``--shelf-only``: the shelf is read; nothing is fitted."""


def trim(cap: dict[str, Any]) -> dict[str, Any]:
    """The capture as committed: every number the beats cite, without the per-row arrays the
    explanations and the stage cards carry. ``--full`` keeps everything."""
    import copy

    out = copy.deepcopy(cap)
    before, after = out.get("before_fit") or {}, out.get("after_seal") or {}
    for name in ("whos_in", "models_entry", "at_the_draw"):
        snap = before.get(name) or {}
        snap.pop("quest", None)
        prop = snap.get("proposals") or {}
        if prop:
            snap["proposals"] = {"labels": {q: v for q, v in ((prop.get("labels") or {}).items())
                                            if q in ("missing", "split")}}
        if "cohort" in snap:
            snap["cohort"] = {k: (snap["cohort"] or {}).get(k) for k in ("n_measured", "n_analyzed",
                                                                         "steps", "flow")}
    for key in ("quest_at_plan", "methods_at_plan"):
        before.pop(key, None)
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
            if isinstance(m.get("calibration"), dict) and key == "fit_opened":
                m["calibration"].pop("curve", None)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", required=True, type=Path, help="an empty folder: the fresh home")
    parser.add_argument("--out", type=Path, default=HERE / "capture.json")
    parser.add_argument("--forms", choices=("rule", "none"), default="rule",
                        help="each continuous measure bends (rule, the beats' default) or not")
    parser.add_argument("--shelf-only", action="store_true",
                        help="stop at the families question: the shelf, nothing fitted")
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
                                                  nhanes_dietary_truth, post_answer, wait_results)
    from turbotab.core.tests.acceptance.server_drive import Drive, local_server

    cap: dict[str, Any] = {"before_fit": {}, "after_seal": {}, "notes": []}
    snaps = cap["before_fit"]
    table = home.parent / f"{home.name}-table" / "nhanes_prepared.csv"
    log("preparing the table with beat D's answers")
    snaps["prepared"] = prepare_table(table)

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
        # The shelf, as the families card reads it.
        snaps["shelf"] = run.drive.artifact("shelf", timeout=1800)
        if args.shelf_only:
            raise Stop()
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
        cap.setdefault("timing", {})["fit_pressed_monotonic"] = time.monotonic()
        cap["declared_before_fit"] = {"level_rule": LEVEL_RULE, "at": time.strftime("%H:%M:%S")}
        cap["fit_pressed"] = run.drive.press_fit()
        return None

    def read_results(run: Run) -> None:
        d, c, pid = run.drive, run.client, run.pid
        after = cap["after_seal"]
        for stage in ("fit", "evaluation", "explain"):
            log(f"waiting for {stage}")
            after[f"{stage}_before_opening"] = d.artifact(stage, timeout=5400)
            timing = cap.setdefault("timing", {})
            timing[f"{stage}_fresh"] = time.strftime("%H:%M:%S")
            if "fit_pressed_monotonic" in timing:
                timing[f"{stage}_seconds_after_press"] = round(time.monotonic()
                                                               - timing["fit_pressed_monotonic"], 1)
        for name in ("quest", "triage", "record", "methods", "checklist", "materiality"):
            after[f"{name}_before_opening"] = get(c, f"/api/projects/{pid}/{name}")["body"]
        log("read everything before the opening")

    def choose_final() -> str:
        """Ruling 4: among the families whose paired difference from the best on the comparison
        folds lies within the margin (a twentieth of the best family's gain over the no-predictor
        model on the headline's folds), the simplest whose fit raised no concern its equation would
        carry into the paper (TRIPOD+AI 22); the simplest of them all when every one raised one."""
        fit = cap["after_seal"]["fit_before_opening"]
        metric = fit["primary_metric"]
        scored = {m["family"]: (m.get("compared_on") or {}).get("estimate") for m in fit["models"]}
        headline = {m["family"]: (m.get("cv") or {}).get(metric, {}).get("estimate") for m in fit["models"]}
        baseline = next((m.get("baseline") or {}).get("value") for m in fit["models"]
                        if (m.get("baseline") or {}).get("metric") == metric)
        concerns = {m["family"]: list(m.get("concerns") or []) for m in fit["models"]}
        best = min((k for k, v in scored.items() if v is not None), key=lambda k: scored[k])
        gain = float(baseline) - float(headline[best])
        margin = MARGIN_SHARE * gain
        equivalent = {best}
        pairs = []
        for d in fit.get("comparisons") or []:
            pair = {d.get("a"), d.get("b")}
            if best in pair and d.get("ci_low") is not None:
                inside = -margin <= d["ci_low"] and d["ci_high"] <= margin
                pairs.append({"a": d["a"], "b": d["b"], "difference": d["difference"],
                              "ci_low": d["ci_low"], "ci_high": d["ci_high"], "inside": inside})
                if inside:
                    equivalent |= pair
        clean = [k for k in equivalent if not concerns.get(k)]
        chosen = min(clean or equivalent, key=lambda k: COMPLEXITY.get(k, 9))
        cap["final_rule"] = {"metric": metric, "compared_on": scored, "headline": headline,
                             "baseline": baseline, "best": best, "gain": gain, "margin": margin,
                             "pairs": pairs, "equivalent": sorted(equivalent), "concerns": concerns,
                             "without_concern": sorted(clean), "chosen": chosen}
        log(f"final model by the rule: {chosen} (best on the comparison: {best}; margin ±{margin:.1f})")
        return chosen

    spec = Journey(
        "predict-beats", "dietary", "prediction",
        "How well does what is known at a visit with a fasting draw predict fasting glucose?",
        NHANES_FILE, "BEATS_PREDICT's light capture (docs/turbotab-next/calm/predict-capture).",
        lambda: table, nhanes_dietary_truth, target="glucose", lenses=("dietary",), roles=ROLES,
        answers={"missing": MISSING, "split": SPLIT, "exclusions": EXCLUSIONS,
                 "models": choose_models_and_press},
        before={"target": apply_first_repair(r"^sas_zeros"),
                "roles": declare(INTENDED_USE),
                "exclusions": snapshot("whos_in", "proposals", "cohort"),
                "split": snapshot("at_the_draw", "seal_plan"),
                "models": lambda run, view: (snapshot("models_entry", "split", "proposals")(run, view),
                                             declare(EXPLAIN, *([forms] if forms["forms"] != "none"
                                                                else []))(run, run.drive.view()))},
        needs=(str(table),), fixture_key=NHANES_KEY, timeout=7200.0)

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
            try:
                ok = follow(run)
            except Stop:
                ok = False
                cap["notes"].append("stopped at the families question (--shelf-only): nothing fitted")
            cap["followed"] = ok
            if ok and wait_results(run, timeout=5400):
                after = cap["after_seal"]
                read_results(run)
                final = choose_final()
                cap["declared_before_opening"] = {"final": final, "at": time.strftime("%H:%M:%S")}
                response = export(run)
                cap["export_status"] = getattr(response, "status_code", None)
                if getattr(response, "status_code", None) == 200:
                    after["bundle"] = bundle_parts(response.content, home)
            view = client.get(f"/api/projects/{run.pid}").json()
            snaps["state_at_end"] = {k: view["state"].get(k) for k in (
                "purpose", "target", "task", "split", "missing", "models", "intended_use",
                "explain", "energy_adjustment", "substitution", "survey", "exclusions", "temporal")}
            snaps["interview_at_end"] = [{"key": s["key"], "status": s["status"],
                                          "reason": s.get("reason")} for s in view["interview"]]
            # Outcome-free counts for the Models beats, from the working table.
            from turbotab.core.config import default_memory_budget
            from turbotab.core.datastore import DataStore
            from turbotab.core.graph import read_artifact
            from turbotab.core.stages.working import working_paths

            cache = client.app.state.service.workspace.cache_dir(run.pid)
            working = read_artifact(cache, "working", view["stages"]["working"]["key"])
            dev_ids = None
            if view["stages"].get("split", {}).get("key"):
                split_art = read_artifact(cache, "split", view["stages"]["split"]["key"])
                assignment = split_art.frames["assignment"]
                dev_ids = assignment.loc[assignment["partition"] == "train", "row_id"].to_numpy(dtype=np.int64)
                snaps["split_facts"] = {k: split_art.data.get(k) for k in (
                    "n_train", "n_holdout", "holdout", "seed", "folds", "fold_labels", "note",
                    "fold_scheme", "cluster", "validation")}
            with DataStore(working_paths(working)["table"], int(default_memory_budget())) as store:
                snaps["predictors_all_rows"] = predictors_summary(store, None)
                if dev_ids is not None:
                    snaps["predictors_development"] = predictors_summary(store, dev_ids)
                    snaps["measures_development"] = measures_summary(store, dev_ids)
            snaps["riley_73"] = riley(73)
    except Exception as exc:  # noqa: BLE001 - kept with its reason
        cap["notes"].append(f"stopped by {type(exc).__name__}: {str(exc)[:1500]}")
        traceback.print_exc()
    cap["answers"] = run.log
    cap["notes"] += run.notes
    cap["forms"] = args.forms
    timing = cap.get("timing") or {}
    timing.pop("fit_pressed_monotonic", None)
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

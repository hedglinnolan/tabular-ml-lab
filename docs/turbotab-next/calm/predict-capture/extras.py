"""The Results beats' numbers the engine does not serve yet, computed on a finished capture with the
engine's own functions (``BEATS_PREDICT.md`` Round 1; requirements P4, P6, P10, P18, P22 and P25).

    venv/bin/python docs/turbotab-next/calm/predict-capture/extras.py --home <the capture's home> \\
        --capture <its capture.json>
    venv/bin/python docs/turbotab-next/calm/predict-capture/extras.py --home <…> --capture <…> --open

It reads the finished project's cache (the split, the design, the fit, the explanations, the working
table), never the server. Without ``--open`` it reads glucose on the development rows only (the
eight cycles 2001–2016) and writes ``extras.json``; it may be run again. ``--open`` opens the
sealed latest cycle (2017–2018) **once**: it refuses when its output exists, reads the final model
the capture declared and the level rule ``extras.json`` computed before it, and writes
``opened.json``. Both are used only by the Results beats. ``--open --dry-run`` exercises every
step of the opening on a made-up outcome without asking the store for the sealed cycle's glucose
(write it elsewhere with ``--opened-out``).

**One yardstick** (ruling 2): every verdict here is on the squared miss, said as the share of the
differences between people explained (R²) and the root mean squared error in mg/dL, each with its
95% interval; differences are paired row by row. The mean absolute error is kept as description.

* **The headline's own folds.** The engine validated by survey cycle (internal–external: each
  development cycle scored by models fit on the other seven). It serves each cycle's scores and
  their summary, but keeps the out-of-cycle predictions inside the fit; so each family's pipeline,
  exactly as the design stage built it, is refit here on the same eight folds
  (``metrics.cross_validate`` with ``inner_cv.fit_pipeline`` at the split's seed), with the
  no-predictor model beside it (each fold's training mean). Its pooled scores match the engine's
  ``cv`` (``matches_engine``).
* **Calibration** (``performance.calibration``) and by tenth of the estimates, with each tenth's
  interval; the ends are ruled on (E90, the outer tenths).
* **The diabetic range** (P4): how well the estimate ranks who is at 126 mg/dL or more (the c
  statistic, ``performance.auc_interval``); of those it estimates at 126 or more, how many are
  there and what they average; and, for teaching only, the signed miss among people picked by a
  high measured value beside a perfectly calibrated version's (regression to the mean).
* **The groups** the intended use names (gender, age in thirds, told to take a blood-pressure or
  cholesterol medicine): the paired difference in squared miss from the no-predictor model, the
  share explained against it, and the share explained within the group.
* **The cycles**: each cycle's miss, signed level and rank correlation, and the summary of the
  cycles' squared miss on the log scale with the Hartung–Knapp–Sidik–Jonkman interval and a 95%
  prediction interval for a new cycle (t on k − 2), beside the engine's identity-scale summary.
* **What each choice added** (P10): the final family refit on the same folds without a group of
  inputs, or with straight lines; paired.
* **The level for a later visit** (ruling 10, P25): the trigger (the latest development cycle's
  out-of-cycle level), the update (the final model's intercept re-estimated on that cycle), and a
  forward check on development cycles alone (a level set on 2013–2014 for a model built on
  2001–2012, scored on 2015–2016).
* **Each plain group's net SHAP** (P6), from the explanations' own rows.
* ``--open``: the latest cycle, scored once with the final model and its declared level, the other
  families secondary; the tests the opening resolves; the groups; Table 1 (development against the
  opened cycle).
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

sys.path.insert(0, str(HERE))
from capture import LATEST, finite  # noqa: E402

CUT = 126.0  # American Diabetes Association, Standards of Care §2: the diabetic fasting range
LIPIDS = ["hdl", "triglycerides"]
MEDICINES = ["meds_hbp", "meds_chol"]
GROUPS = {"lipids": LIPIDS, "medicines": MEDICINES,
          "diet": ["kcal", "protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"],
          "body_size": ["weight", "height", "bmi", "waist"], "age": ["age"],
          "blood_pressure": ["bp_sys", "bp_di"], "gender": ["gender"]}


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def interval(i: Any) -> dict[str, Any]:
    d = i.model_dump(mode="json") if hasattr(i, "model_dump") else dict(i)
    return {k: d.get(k) for k in ("estimate", "ci_low", "ci_high", "se")}


def mean_ci(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    m, se = float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))
    return {"estimate": m, "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se, "se": se}


def paired_sq(e_new: np.ndarray, e_ref: np.ndarray) -> dict[str, float]:
    """The variant's squared miss less the reference's, row by row (positive: the variant misses
    more), with a normal 95% interval over people."""
    return mean_ci(np.square(e_new) - np.square(e_ref))


def scores(y: np.ndarray, pred: np.ndarray, ref: Any) -> dict[str, Any]:
    from turbotab.core.models.performance import score_intervals

    out = {m: interval(i) for m, i in score_intervals("regression", y, pred, reference=ref).items()}
    out["bias"] = mean_ci(pred - y)
    out["n"] = int(len(y))
    return out


def by_tenth(y: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    """Mean measured against mean estimated in each tenth of the estimates (calibration's own
    picture, grouped by the estimates, never by the outcome), each with its interval."""
    edges = np.quantile(pred, np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, 9)
    out = []
    for k in range(10):
        keep = which == k
        diff = y[keep] - pred[keep]
        out.append({"tenth": k + 1, "n": int(keep.sum()), "mean_estimated": float(pred[keep].mean()),
                    "mean_measured": float(y[keep].mean()), "measured_less_estimated": mean_ci(diff),
                    "share_at_or_above_cut": float((y[keep] >= CUT).mean())})
    return out


def calibration_of(y: np.ndarray, pred: np.ndarray, where: str) -> dict[str, Any] | None:
    from turbotab.core.models.performance import calibration

    cal = calibration("regression", y, pred, where=where)
    if cal is None:
        return None
    return {"intercept": interval(cal.intercept), "slope": interval(cal.slope), "eavg": cal.eavg,
            "e90": cal.e90, "emax": cal.emax}


def tail(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    from turbotab.core.models.performance import auc_interval

    high, flagged = y >= CUT, pred >= CUT
    top = pred >= np.quantile(pred, 0.9)
    out = {"n_at_or_above": int(high.sum()), "share_at_or_above": float(high.mean()),
           "c": interval(auc_interval(high, pred)),
           "estimated_at_or_above": int(flagged.sum()),
           "there_among_estimated_at_or_above": float(high[flagged].mean()) if flagged.any() else None,
           "measured_mean_among_estimated_at_or_above": float(y[flagged].mean()) if flagged.any() else None,
           "estimated_mean_among_estimated_at_or_above": float(pred[flagged].mean()) if flagged.any() else None,
           "top_tenth": {"measured": float(y[top].mean()), "estimated": float(pred[top].mean())},
           "squared_miss_share_from_at_or_above": float(np.square(pred - y)[high].sum()
                                                        / np.square(pred - y).sum())}
    # Teaching only (beat R2's Why?): people picked by a high measured value, against their
    # estimates; and the same for a perfectly calibrated version of these estimates (isotonic, in
    # sample), which still misses them: regression to the mean, not a bias.
    iso = IsotonicRegression(out_of_bounds="clip").fit(pred, y)
    calibrated = iso.predict(pred)
    out["picked_by_measured"] = {"measured_mean": float(y[high].mean()),
                                 "estimated_mean": float(pred[high].mean()),
                                 "signed_miss": float((pred[high] - y[high]).mean()),
                                 "perfectly_calibrated_signed_miss": float((calibrated[high] - y[high]).mean())}
    return out


def random_effects_log(est: np.ndarray, se: np.ndarray) -> dict[str, Any]:
    """The cycles' MSE summarized on the log scale (delta-method SE se/θ): DerSimonian–Laird τ²,
    the Hartung–Knapp–Sidik–Jonkman interval on t(k − 1), and the prediction interval for a new
    cycle on t(k − 2) (Higgins, Thompson & Spiegelhalter 2009); back-transformed, and as RMSE."""
    from scipy import stats

    th, s = np.log(est), se / est
    k = len(th)
    w = 1 / s ** 2
    fixed = float(np.sum(w * th) / np.sum(w))
    q = float(np.sum(w * (th - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c)
    wr = 1 / (s ** 2 + tau2)
    mu = float(np.sum(wr * th) / np.sum(wr))
    se_mu = math.sqrt(1 / np.sum(wr))
    hk = math.sqrt(float(np.sum(wr * (th - mu) ** 2)) / ((k - 1) * float(np.sum(wr))))
    ci = (mu - stats.t.ppf(0.975, k - 1) * hk, mu + stats.t.ppf(0.975, k - 1) * hk)
    half = stats.t.ppf(0.975, k - 2) * math.sqrt(tau2 + se_mu ** 2)
    pi = (mu - half, mu + half)
    return {"k": k, "mse": math.exp(mu), "rmse": math.sqrt(math.exp(mu)),
            "ci_rmse": [math.sqrt(math.exp(v)) for v in ci],
            "pi_rmse": [math.sqrt(math.exp(v)) for v in pi], "tau2_log": tau2,
            "i2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0}


def load(home: Path) -> dict[str, Any]:
    from turbotab.core.config import default_memory_budget
    from turbotab.core.datastore import DataStore
    from turbotab.core.graph import read_artifact
    from turbotab.core.models.pipeline import DesignSpec
    from turbotab.core.stages.working import working_paths

    projects = sorted((home / "projects").iterdir())
    cache = projects[0] / "cache"

    def latest(stage: str) -> Any:
        return read_artifact(cache, stage, (cache / stage / "latest").read_text().strip())

    split, design, fit, working = latest("split"), latest("design"), latest("fit"), latest("working")
    explain = latest("explain") if (cache / "explain" / "latest").is_file() else None
    spec = DesignSpec.from_dict(design.objects["spec"])
    store = DataStore(working_paths(working)["table"], int(default_memory_budget()))
    return {"split": split, "design": design, "fit": fit, "explain": explain, "spec": spec,
            "store": store, "pipelines": design.objects["pipelines"],
            "seed": int(fit.objects.get("split_seed") or 0)}


def development(home: Path, capture: dict[str, Any]) -> dict[str, Any]:
    from scipy import stats
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.decision_curve import subgroup_labels
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import cross_validate
    from turbotab.core.models.performance import auc_interval
    from turbotab.core.models.pipeline import build_pipeline, modeling_frame

    L = load(home)
    split, fit, spec, store, pipelines, seed = (L["split"], L["fit"], L["spec"], L["store"],
                                                L["pipelines"], L["seed"])
    final = capture["declared_before_opening"]["final"]
    assignment = split.frames["assignment"]
    train = (assignment["partition"] == "train").to_numpy()
    ids = assignment.loc[train, "row_id"].to_numpy(dtype=np.int64)
    fold = assignment.loc[train, "fold"].to_numpy(dtype=np.int64)
    labels = list(split.data.get("fold_labels") or [])
    extra = ["cycle_begin_year", *[c for c in ("gender", "age", *MEDICINES) if c not in spec.inputs]]
    frame = modeling_frame(store, [*spec.inputs, *extra, "glucose"], ids, outcome="glucose")
    X = frame[list(spec.inputs)]
    y = frame["glucose"].to_numpy(dtype=float)
    pairs = [(k, fold != k, fold == k) for k in range(int(fold.max()) + 1)]
    log(f"{len(y):,} development rows, {len(pairs)} folds by cycle ({labels}), final family {final}")

    def fit_rows(model: Any, X_fit: Any, y_fit: Any, rows: Any = None) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_pipeline(model, X_fit, y_fit, seed=seed)

    def oof(make: Any, Xv: Any = None) -> dict[str, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate("regression", make, X if Xv is None else Xv, y, pairs, fit=fit_rows,
                                keep_predictions=True)
        pred, ref = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for f in cv.predictions:
            pred[f.rows] = np.asarray(f.prediction, dtype=float)
            ref[f.rows] = float(f.reference)
        return {"pred": pred, "ref": ref}

    out: dict[str, Any] = {"final": final, "folds": labels, "n": int(len(y)), "seed": seed, "cut": CUT}
    runs: dict[str, dict[str, np.ndarray]] = {}
    for key, pipeline in pipelines.items():
        log(f"{key}: refitting on the {len(pairs)} cycle folds")
        runs[key] = oof(lambda _p=pipeline: clone(_p))
    ref = runs[final]["ref"]
    out["outcome"] = {"mean": float(y.mean()), "sd": float(y.std(ddof=1)), "median": float(np.median(y)),
                      "skew": float(stats.skew(y)), "log_skew": float(stats.skew(np.log(y))),
                      "share_at_or_above_cut": float((y >= CUT).mean()),
                      "p25": float(np.quantile(y, 0.25)), "p75": float(np.quantile(y, 0.75))}
    served = {m["family"]: (m.get("cv") or {}).get("mse", {}).get("estimate") for m in fit.data["models"]}
    families: dict[str, Any] = {}
    for key, r in runs.items():
        pred = r["pred"]
        e = pred - y
        families[key] = {"scores": scores(y, pred, r["ref"]),
                         "calibration": calibration_of(y, pred, "out of cycle"),
                         "by_tenth": by_tenth(y, pred), "tail": tail(y, pred),
                         "prediction_sd": float(np.std(pred, ddof=1)),
                         "spearman": float(stats.spearmanr(pred, y).correlation),
                         "mse_engine": served.get(key), "mse_here": float(np.mean(e ** 2))}
    families["baseline"] = {"scores": scores(y, ref, ref)}
    out["families"] = families
    out["matches_engine"] = {k: abs(families[k]["mse_here"] - (families[k]["mse_engine"] or np.nan)) < 1e-6
                             for k in pipelines}
    out["versus_baseline"] = {k: paired_sq(runs[k]["pred"] - y, ref - y) for k in pipelines}
    out["outcome_sd"] = float(np.std(y, ddof=1))

    # What each choice added (P10): the final family without a group of inputs, or straight lines.
    family = get_family(final)

    def variant(drop: list[str] | None = None, levers: Any = "keep") -> Any:
        keep = [c for c in spec.inputs if c not in (drop or [])]
        v = replace(spec, inputs=keep, predictors=[c for c in spec.predictors if c in keep],
                    numeric=[c for c in spec.numeric if c in keep],
                    categorical=[c for c in spec.categorical if c in keep],
                    levels=[c for c in spec.levels if c in keep],
                    two_valued=[c for c in spec.two_valued if c in keep],
                    levers=spec.levers if levers == "keep" else levers)
        return v, build_pipeline(v, family, "regression", "prediction", len(y), len(keep))

    variants = {f"without_{name}": variant(drop=cols) for name, cols in GROUPS.items()}
    if spec.levers and (spec.levers or {}).get("forms") not in (None, "none"):
        variants["straight_lines"] = variant(levers=None)
    e_final = runs[final]["pred"] - y
    added: dict[str, Any] = {}
    for name, (vspec, pipeline) in variants.items():
        log(f"{final} {name}: refitting on the {len(pairs)} cycle folds")
        r = oof(lambda _p=pipeline: clone(_p), X[list(vspec.inputs)])
        pred = r["pred"]
        sc = scores(y, pred, ref)
        added[name] = {"d_mse": paired_sq(pred - y, e_final), "r2": sc["r2"], "rmse": sc["rmse"],
                       "c": tail(y, pred)["c"], "prediction_sd": float(np.std(pred, ddof=1)),
                       "d_mae": mean_ci(np.abs(pred - y) - np.abs(e_final))}
    out["added"] = added
    # The model's columns (Riley's parameters): the final pipeline's transformed width.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit.objects["fitted"][final]
    try:
        width = fitted[:-1].transform(X.iloc[:50]).shape[1]
    except Exception:  # noqa: BLE001 - the width is a check, not a result
        width = None
    out["model_columns"] = width

    # The groups the intended use names, on the final family's out-of-cycle estimates.
    asked = (frame["meds_hbp"].notna() | frame["meds_chol"].notna()).to_numpy()
    columns = {"gender": (frame["gender"].astype(str).to_numpy(), "levels"),
               "age": (frame["age"].to_numpy(dtype=float), "thirds"),
               "told_to_take_medicine": (np.where(asked, "told to take a blood-pressure or cholesterol "
                                                         "medicine", "neither"), "levels")}
    groups: dict[str, Any] = {}
    pred = runs[final]["pred"]
    for name, (values, how) in columns.items():
        lab = subgroup_labels(values, how) if how == "thirds" else np.asarray(values, dtype=object)
        rows = []
        for level in sorted(set(lab.tolist()), key=str):
            keep = lab == level
            yk, pk, rk = y[keep], pred[keep], ref[keep]
            own = float(np.mean((yk - yk.mean()) ** 2))
            rows.append({"group": str(level), "n": int(keep.sum()),
                         "share_at_or_above_cut": float((yk >= CUT).mean()),
                         "model": scores(yk, pk, rk), "baseline": scores(yk, rk, rk),
                         "d_mse_vs_baseline": paired_sq(pk - yk, rk - yk),
                         "r2_within_group": 1 - float(np.mean((pk - yk) ** 2)) / own,
                         "c": interval(auc_interval(yk >= CUT, pk)) if (yk >= CUT).any() else None,
                         "mean": float(yk.mean()), "median": float(np.median(yk))})
        groups[name] = rows
    out["groups"] = groups
    if "age" in groups:
        out["age_thirds_cuts"] = [float(v) for v in np.quantile(frame["age"].to_numpy(dtype=float),
                                                                [1 / 3, 2 / 3])]

    # The cycles: each one's miss, level and ranking; the summary on the log scale.
    cycles = []
    for k, label in enumerate(labels):
        keep = fold == k
        e = pred[keep] - y[keep]
        cycles.append({"cycle": label, "n": int(keep.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))),
                       "mse": float(np.mean(e ** 2)), "level": mean_ci(e),
                       "spearman": float(stats.spearmanr(pred[keep], y[keep]).correlation),
                       "slope": float(np.polyfit(pred[keep], y[keep], 1)[0]),
                       "median_measured": float(np.median(y[keep])), "mean_measured": float(np.mean(y[keep])),
                       "share_at_or_above_cut": float((y[keep] >= CUT).mean())})
    out["cycles"] = cycles
    model = next(m for m in fit.data["models"] if m["family"] == final)
    ie = model.get("internal_external") or {}
    est = np.array([c["primary"]["estimate"] for c in ie.get("clusters") or []], dtype=float)
    se = np.array([c["primary"]["se"] for c in ie.get("clusters") or []], dtype=float)
    out["summary_engine_identity"] = ie.get("pooled")
    out["summary_log"] = random_effects_log(est, se) if len(est) >= 3 else None
    out["corr_mse_se"] = float(np.corrcoef(est, se)[0, 1]) if len(est) >= 3 else None

    # The level for a later visit (ruling 10): the trigger, the update, and a forward check.
    latest_dev = labels[-1]
    k_latest = labels.index(latest_dev)
    keep_latest = fold == k_latest
    trigger = mean_ci(y[keep_latest] - pred[keep_latest])  # measured less estimated, out of cycle
    fitted_pred = np.asarray(fitted.predict(X.iloc[np.flatnonzero(keep_latest)]), dtype=float)
    delta = float(np.mean(y[keep_latest] - fitted_pred))
    level = {"latest_development_cycle": latest_dev, "trigger_measured_less_estimated": trigger,
             "triggered": bool(trigger["ci_low"] > 0 or trigger["ci_high"] < 0),
             "update_mg_dl": delta, "update_rows": int(keep_latest.sum())}
    # Forward check on development cycles alone: a model built on the cycles before the last two,
    # its level set on the second-to-last, scored on the last, against no update.
    if len(labels) >= 3:
        k_prev = k_latest - 1
        early = fold < k_prev
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit_pipeline(clone(pipelines[final]), X.iloc[np.flatnonzero(early)], y[early], seed=seed)
        prev, last = fold == k_prev, keep_latest
        p_prev = np.asarray(m.predict(X.iloc[np.flatnonzero(prev)]), dtype=float)
        p_last = np.asarray(m.predict(X.iloc[np.flatnonzero(last)]), dtype=float)
        shift = float(np.mean(y[prev] - p_prev))
        e_raw, e_upd = p_last - y[last], p_last + shift - y[last]
        level["forward_check"] = {
            "built_on": labels[:k_prev], "level_set_on": labels[k_prev], "scored_on": latest_dev,
            "shift": shift, "raw": {"level": mean_ci(e_raw), "rmse": float(np.sqrt(np.mean(e_raw ** 2)))},
            "updated": {"level": mean_ci(e_upd), "rmse": float(np.sqrt(np.mean(e_upd ** 2)))},
            "d_mse_updated_less_raw": paired_sq(e_upd, e_raw)}
    out["level"] = level

    # Each plain group's net SHAP (P6): the mean of |the sum of a group's SHAP values| over the
    # explanations' rows, against the summed shares (which count cancelling inputs twice).
    explain = L["explain"].data if L["explain"] is not None else None
    if explain is not None:
        net: dict[str, Any] = {}
        for fam in explain.get("families") or []:
            if not fam.get("explained"):
                continue
            phi = {b["input"]: np.asarray(b["phi"], dtype=float) for b in fam.get("beeswarm") or []}
            net[fam["family"]] = {
                g: {"net": float(np.mean(np.abs(sum(phi[c] for c in cols if c in phi)))),
                    "summed": float(sum(np.mean(np.abs(phi[c])) for c in cols if c in phi))}
                for g, cols in GROUPS.items()}
            net[fam["family"]]["_rows"] = len(next(iter(phi.values()))) if phi else 0
        out["group_shap"] = net
    store.close()
    return out


def opening(home: Path, capture: dict[str, Any], dev: dict[str, Any], dry: bool = False) -> dict[str, Any]:
    """The latest cycle, opened once (P21's opening, emulated): the declared final model with its
    declared level, the other families secondary, the tests, the groups and Table 1. ``dry``
    exercises every step without reading the sealed cycle's glucose: it never asks the store for
    it and scores a made-up outcome instead."""
    from scipy import stats

    from turbotab.core.models.decision_curve import subgroup_labels  # noqa: F401 - same thirds
    from turbotab.core.models.performance import auc_interval
    from turbotab.core.models.pipeline import modeling_frame

    L = load(home)
    split, fit, spec, store = L["split"], L["fit"], L["spec"], L["store"]
    final = capture["declared_before_opening"]["final"]
    level = dev["level"]
    delta = level["update_mg_dl"] if level["triggered"] else 0.0
    assignment = split.frames["assignment"]
    dev_ids = assignment.loc[assignment["partition"] == "train", "row_id"].to_numpy(dtype=np.int64)
    every = modeling_frame(store, ["cycle_begin_year"], None)
    sealed = every.index[every["cycle_begin_year"].to_numpy(dtype=float) == LATEST].to_numpy(dtype=np.int64)
    extra = ["cycle_begin_year", *[c for c in ("gender", "age", *MEDICINES) if c not in spec.inputs]]
    if dry:
        held = modeling_frame(store, [*spec.inputs, *extra], sealed)
        held["glucose"] = np.random.default_rng(0).gamma(9.0, 12.0, len(held))  # made up
    else:
        held = modeling_frame(store, [*spec.inputs, *extra, "glucose"], sealed, outcome="glucose")
    held = held[np.isfinite(held["glucose"].to_numpy(dtype=float))]
    yh = held["glucose"].to_numpy(dtype=float)
    Xh = held[list(spec.inputs)]
    devf = modeling_frame(store, [*spec.inputs, *extra, "glucose"], dev_ids, outcome="glucose")
    ydev = devf["glucose"].to_numpy(dtype=float)
    dev_mean = float(ydev.mean())
    out: dict[str, Any] = {"opened_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n": int(len(yh)),
                           "final": final, "level_applied": delta, "development_mean": dev_mean}
    base = np.full(len(yh), dev_mean)
    preds = {}
    for key, model in fit.objects["fitted"].items():
        preds[key] = np.asarray(model.predict(Xh), dtype=float)
    p_final = preds[final] + delta
    out["final_as_declared"] = {"scores": scores(yh, p_final, dev_mean),
                                "calibration": calibration_of(yh, p_final, "on the opened cycle"),
                                "by_tenth": by_tenth(yh, p_final), "tail": tail(yh, p_final),
                                "spearman": float(stats.spearmanr(p_final, yh).correlation)}
    out["final_without_level"] = {"scores": scores(yh, preds[final], dev_mean),
                                  "calibration": calibration_of(yh, preds[final], "on the opened cycle")}
    out["level_effect"] = paired_sq(p_final - yh, preds[final] - yh)
    out["secondary"] = {k: scores(yh, p, dev_mean) for k, p in preds.items() if k != final}
    out["baseline"] = scores(yh, base, dev_mean)
    out["baseline_latest_level"] = scores(yh, np.full(len(yh), dev_mean + delta), dev_mean)
    # The tests the opening resolves.
    summ = dev.get("summary_log") or {}
    rmse = out["final_as_declared"]["scores"]["rmse"]["estimate"]
    sel = (fit.data.get("selection") or {})
    expected = {k: math.sqrt(sel[k]) for k in ("corrected", "corrected_low", "corrected_high")
                if sel.get(k) is not None}
    out["tests"] = {
        "rmse_found": rmse,
        "cycles_prediction_interval": summ.get("pi_rmse"),
        "inside_prediction_interval": (summ.get("pi_rmse") is not None
                                       and summ["pi_rmse"][0] <= rmse <= summ["pi_rmse"][1]),
        "bbc_expected_rmse": expected,
        "level_with_update": out["final_as_declared"]["scores"]["bias"],
        "level_without_update": out["final_without_level"]["scores"]["bias"]}
    # The groups, on the opened cycle (age at the development rows' thirds).
    cuts = np.quantile(devf["age"].to_numpy(dtype=float), [1 / 3, 2 / 3])
    age = held["age"].to_numpy(dtype=float)
    age_lab = np.where(age <= cuts[0], f"≤ {cuts[0]:g}", np.where(age <= cuts[1], f"{cuts[0]:g}–{cuts[1]:g}",
                                                                  f"> {cuts[1]:g}"))
    asked = (held["meds_hbp"].notna() | held["meds_chol"].notna()).to_numpy()
    groups: dict[str, Any] = {}
    for name, lab in (("gender", held["gender"].astype(str).to_numpy()), ("age", age_lab),
                      ("told_to_take_medicine", np.where(asked, "told to take a blood-pressure or "
                                                                "cholesterol medicine", "neither"))):
        rows = []
        for lv in sorted(set(lab.tolist()), key=str):
            keep = lab == lv
            yk, pk = yh[keep], p_final[keep]
            own = float(np.mean((yk - yk.mean()) ** 2))
            rows.append({"group": str(lv), "n": int(keep.sum()),
                         "share_at_or_above_cut": float((yk >= CUT).mean()),
                         "model": scores(yk, pk, dev_mean), "baseline": scores(yk, np.full(len(yk), dev_mean), dev_mean),
                         "d_mse_vs_baseline": paired_sq(pk - yk, dev_mean - yk),
                         "r2_within_group": 1 - float(np.mean((pk - yk) ** 2)) / own,
                         "c": interval(auc_interval(yk >= CUT, pk)) if (yk >= CUT).any() and (yk < CUT).any() else None})
        groups[name] = rows
    out["groups"] = groups
    # Table 1 (TRIPOD+AI 20b, 20c): development rows against the opened cycle.
    raw_cols = ["age", "bmi", "waist", "bp_sys", "bp_di", "hdl", "triglycerides", "kcal"]
    table1: dict[str, Any] = {}
    for name, fr, yy in (("development", devf, ydev), ("opened", held, yh)):
        col = {"n": int(len(fr)), "female": float((fr["gender"].astype(str).isin(["1", "1.0", "True", "female", "F"])).mean())}
        for c in raw_cols:
            v = fr[c].to_numpy(dtype=float)
            ok = np.isfinite(v)
            col[c] = {"median": float(np.median(v[ok])), "p25": float(np.quantile(v[ok], 0.25)),
                      "p75": float(np.quantile(v[ok], 0.75)), "missing": int((~ok).sum())}
        for c in MEDICINES:
            s = fr[c]
            col[c] = {"taking": float((s == 1).mean()), "told_not_taking": float((s == 0).mean()),
                      "not_asked": float(s.isna().mean())}
        col["glucose"] = {"median": float(np.median(yy)), "p25": float(np.quantile(yy, 0.25)),
                          "p75": float(np.quantile(yy, 0.75)), "share_at_or_above_cut": float((yy >= CUT).mean()),
                          "mean": float(yy.mean())}
        table1[name] = col
    out["table1"] = table1
    store.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path, help="the run's capture.json")
    parser.add_argument("--out", type=Path, default=HERE / "extras.json")
    parser.add_argument("--open", action="store_true", help="open the latest cycle, once")
    parser.add_argument("--opened-out", type=Path, default=HERE / "opened.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --open: every step on a made-up outcome, reading no sealed value")
    args = parser.parse_args(argv)
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    if not args.open:
        dev = development(args.home, capture)
        args.out.write_text(json.dumps(finite(dev), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log(f"wrote {args.out}")
        return 0
    if args.opened_out.exists():
        parser.error(f"{args.opened_out} exists: the latest cycle is opened once")
    dev = json.loads(args.out.read_text(encoding="utf-8"))
    log("a dry run on a made-up outcome" if args.dry_run else "opening the latest cycle, once")
    out = opening(args.home, capture, dev, dry=args.dry_run)
    args.opened_out.write_text(json.dumps(finite(out), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {args.opened_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

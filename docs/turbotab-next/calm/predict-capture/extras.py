"""The Results beats' numbers the engine does not serve yet, computed on a finished capture with the
engine's own functions (``BEATS_PREDICT.md``, requirements P4, P6 and P10).

    venv/bin/python docs/turbotab-next/calm/predict-capture/extras.py --home <the capture's home>

It reads the finished project's cache (the split, the design, the fit, the working table), never
the server, and writes ``extras.json`` beside this file (``--out`` to put it elsewhere).

Everything here reads glucose: on the training rows for the cross-validated parts, and on the
held-out rows for ``held_out``, which the capture had already opened. It is used only by the
Results beats.

* **The headline's own folds.** The capture validated by survey cycle (internal–external: each
  cycle scored by models fit on the other eight). The engine serves each cycle's scores and their
  random-effects summary, but keeps the out-of-cycle predictions only inside the fit; so each
  family's pipeline, exactly as the design stage built it, is refit here on the same nine folds
  (``metrics.cross_validate`` with ``inner_cv.fit_pipeline`` at the split's seed), and the
  no-predictor model with it (each fold's training mean). Its pooled scores match the engine's
  ``cv`` (checked below), so the error by range, the tail, calibration and the groups all read the
  predictions the headline reads (P4).
* **Error by range** (P4): under 100, 100 to 125 and 126 mg/dL or more (the American Diabetes
  Association's fasting bands); at the 126 cut, how many at or above it the model places there.
* **Calibration** (``performance.calibration``) on the out-of-cycle predictions, and by tenth of
  the predictions.
* **The groups** the intended use names, each scored with ``performance.score_intervals``:
  gender, age in thirds (``decision_curve.subgroup_labels``), and a diagnosis on record (either
  medicine question asked).
* **What each choice added** (P10, the Predict side of "which choices mattered"): the final
  family refit on the same folds without the lipids from the draw (the moment of use "before any
  blood is drawn"), without the medicine answers, and with straight lines in place of curves; each
  against the final family, paired row by row over the same out-of-cycle predictions. The same
  refit without each other plain group (diet, body size, age, blood pressure, gender) gives what
  each group adds beyond the others (leave one covariate out, Lei et al. 2018).
* **The held-out rows** (already opened by the capture): the final family's error by range, at
  the cut, in the groups (age cut at the training rows' thirds) and by cycle, so the paper's
  second paragraph reports its checks on the rows the reported score comes from.
* **Each group's net SHAP** (P6): the mean of |the sum of a group's SHAP values| over the
  explanations' own 300 beeswarm rows, so inputs that move together and cancel are not counted
  twice.
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
from capture import BANDS, CUT, at_cut, by_band, by_tenth, finite, scores  # noqa: E402

LIPIDS = ["hdl", "triglycerides"]
MEDICINES = ["meds_hbp", "meds_chol"]
# The plain groups (P6): what each adds beyond the others is the final family refit without it.
GROUPS = {"diet": ["kcal", "protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"],
          "body_size": ["weight", "height", "bmi", "waist"], "age": ["age"],
          "blood_pressure": ["bp_sys", "bp_di"], "gender": ["gender"]}


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def interval(i: Any) -> dict[str, Any]:
    d = i.model_dump(mode="json") if hasattr(i, "model_dump") else dict(i)
    return {k: d.get(k) for k in ("estimate", "ci_low", "ci_high", "se")}


def paired(e_new: np.ndarray, e_ref: np.ndarray) -> dict[str, Any]:
    """The variant's loss less the reference's, row by row (positive: the variant misses more),
    for the absolute and the squared miss, each with a normal 95% interval (one row per person)."""
    out = {}
    for name, f in (("abs", np.abs), ("sq", np.square)):
        d = f(e_new) - f(e_ref)
        se = float(np.std(d, ddof=1) / math.sqrt(len(d)))
        out[name] = {"difference": float(d.mean()), "ci_low": float(d.mean() - 1.96 * se),
                     "ci_high": float(d.mean() + 1.96 * se), "se": se}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "extras.json")
    parser.add_argument("--final", default=None, help="the declared final family (default: read "
                                                      "from the fit's opening)")
    args = parser.parse_args(argv)

    from turbotab.core.config import default_memory_budget
    from turbotab.core.datastore import DataStore
    from turbotab.core.graph import read_artifact
    from turbotab.core.models import get_family
    from turbotab.core.models.decision_curve import subgroup_labels
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import cross_validate
    from turbotab.core.models.performance import calibration, score_intervals
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline, modeling_frame
    from turbotab.core.stages.working import working_paths
    from sklearn.base import clone

    projects = sorted((args.home / "projects").iterdir())
    cache = projects[0] / "cache"

    def latest(stage: str) -> Any:
        return read_artifact(cache, stage, (cache / stage / "latest").read_text().strip())

    split, design, fit, working = latest("split"), latest("design"), latest("fit"), latest("working")
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    seed = int(fit.objects.get("split_seed") or 0)
    # The declared final model is the opening's record (the cache holds the fit as computed).
    opened = [json.loads(line)["decision"] for line in
              (projects[0] / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip() and json.loads(line)["decision"].get("kind") == "open_seal"]
    final = args.final or (opened[-1].get("family") if opened else None) or "linear"
    assignment = split.frames["assignment"]
    train = (assignment["partition"] == "train").to_numpy()
    ids = assignment.loc[train, "row_id"].to_numpy(dtype=np.int64)
    fold = assignment.loc[train, "fold"].to_numpy(dtype=np.int64)
    labels = list(split.data.get("fold_labels") or [])
    store = DataStore(working_paths(working)["table"], int(default_memory_budget()))
    extra = ["cycle_begin_year", *[c for c in ("gender", "age", *MEDICINES) if c not in spec.inputs]]
    frame = modeling_frame(store, [*spec.inputs, *extra, "glucose"], ids, outcome="glucose")
    X = frame[list(spec.inputs)]
    y = frame["glucose"].to_numpy(dtype=float)
    pairs = [(k, fold != k, fold == k) for k in range(int(fold.max()) + 1)]
    log(f"{len(y):,} training rows, {len(pairs)} folds by cycle, final family {final}")

    def fit_rows(model: Any, X_fit: Any, y_fit: Any, rows: Any = None) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_pipeline(model, X_fit, y_fit, seed=seed)

    def oof(make: Any) -> dict[str, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate("regression", make, X, y, pairs, fit=fit_rows, keep_predictions=True)
        pred, ref = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for f in cv.predictions:
            pred[f.rows] = np.asarray(f.prediction, dtype=float)
            ref[f.rows] = float(f.reference)
        return {"pred": pred, "ref": ref}

    out: dict[str, Any] = {"final": final, "folds": labels, "n": int(len(y)), "seed": seed,
                           "bands": [b[2] for b in BANDS], "cut": CUT}
    runs: dict[str, dict[str, np.ndarray]] = {}
    for key, pipeline in pipelines.items():
        log(f"{key}: refitting on the nine cycle folds")
        runs[key] = oof(lambda _p=pipeline: clone(_p))
    ref = runs[final]["ref"]
    runs["baseline"] = {"pred": ref.copy(), "ref": ref}

    # What each choice added (P10): the final family without a group of inputs, or without curves.
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

    variants = {"without_lipids": variant(drop=LIPIDS), "without_medicines": variant(drop=MEDICINES),
                **{f"without_{name}": variant(drop=cols) for name, cols in GROUPS.items()}}
    if spec.levers and (spec.levers or {}).get("forms") not in (None, "none"):
        variants["straight_lines"] = variant(levers=None)
    for name, (vspec, pipeline) in variants.items():
        log(f"{final} {name}: refitting on the nine cycle folds")
        Xv = X[list(vspec.inputs)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate("regression", lambda _p=pipeline: clone(_p), Xv, y, pairs,
                                fit=fit_rows, keep_predictions=True)
        pred = np.full(len(y), np.nan)
        for f in cv.predictions:
            pred[f.rows] = np.asarray(f.prediction, dtype=float)
        runs[f"{final}:{name}"] = {"pred": pred, "ref": ref}

    families: dict[str, Any] = {}
    for key, r in runs.items():
        pred = r["pred"]
        entry: dict[str, Any] = {
            "scores": {m: interval(i) for m, i in
                       score_intervals("regression", y, pred, reference=r["ref"]).items()},
            "overall": scores(y, pred), "by_band": by_band(y, pred)}
        if key != "baseline":
            entry["at_cut"] = at_cut(y, pred)
            entry["by_tenth"] = by_tenth(y, pred)
            cal = calibration("regression", y, pred, where="out of fold")
            entry["calibration"] = None if cal is None else {
                "intercept": interval(cal.intercept), "slope": interval(cal.slope),
                "eavg": cal.eavg, "e90": cal.e90, "emax": cal.emax,
                "observed": cal.observed, "expected": cal.expected}
            e2 = (pred - y) ** 2
            entry["squared_miss_share_at_or_above_cut"] = float(e2[y >= CUT].sum() / e2.sum())
            entry["prediction_sd"] = float(np.std(pred, ddof=1))
        families[key] = entry
    out["families"] = families
    e_final = runs[final]["pred"] - y
    out["added"] = {name.split(":", 1)[1]: paired(runs[name]["pred"] - y, e_final)
                    for name in runs if name.startswith(f"{final}:")}
    out["added_note"] = ("Each variant's miss less the final family's, row by row over the same "
                         "out-of-cycle predictions: positive means the variant misses more, so the "
                         "choice it undoes was worth that much.")
    out["outcome_sd"] = float(np.std(y, ddof=1))
    e2b = (ref - y) ** 2
    out["baseline_squared_miss_share_at_or_above_cut"] = float(e2b[y >= CUT].sum() / e2b.sum())

    # Each group's net SHAP (P6): the mean over rows of |the sum of its inputs' SHAP values|, from
    # the explanations' own 300 beeswarm rows (the same rows for every input), so inputs that move
    # together and cancel (calories and the nutrients they are made of) are not counted twice.
    explain = latest("explain").data if (cache / "explain" / "latest").is_file() else None
    if explain is not None:
        members = {"diet": GROUPS["diet"], "body_size": GROUPS["body_size"], "lipids": LIPIDS,
                   "medicines": MEDICINES, "age": ["age"], "blood_pressure": GROUPS["blood_pressure"],
                   "gender": ["gender"]}
        net: dict[str, Any] = {}
        for fam in explain.get("families") or []:
            if not fam.get("explained"):
                continue
            phi = {b["input"]: np.asarray(b["phi"], dtype=float) for b in fam.get("beeswarm") or []}
            rows = len(next(iter(phi.values()))) if phi else 0
            net[fam["family"]] = {
                g: {"net": float(np.mean(np.abs(sum(phi[c] for c in cols if c in phi)))),
                    "summed": float(sum(np.mean(np.abs(phi[c])) for c in cols if c in phi)),
                    "inputs": [c for c in cols if c in phi]}
                for g, cols in members.items()}
            net[fam["family"]]["_rows"] = rows
        out["group_shap"] = net

    # The engine's own pooled scores for the same folds, to check the refit reproduces them.
    served = {m["family"]: m["cv"] for m in fit.data["models"]}
    out["matches_engine"] = {k: {"mse_here": families[k]["scores"]["mse"]["estimate"],
                                 "mse_engine": (served.get(k) or {}).get("mse", {}).get("estimate")}
                             for k in pipelines}

    # The groups the intended use names, on the final family's out-of-cycle predictions.
    groups: dict[str, Any] = {}
    asked = frame["meds_hbp"].notna() | frame["meds_chol"].notna()
    columns = {"gender": (frame["gender"].astype(str).to_numpy(), "levels"),
               "age": (frame["age"].to_numpy(dtype=float), "thirds"),
               "diagnosis_on_record": (np.where(asked, "asked either medicine question",
                                                "asked neither"), "levels")}
    for name, (values, how) in columns.items():
        lab = subgroup_labels(values, how) if how == "thirds" else np.asarray(values, dtype=object)
        rows = []
        for level in sorted(set(lab.tolist()), key=str):
            keep = lab == level
            entry = {"group": str(level), "n": int(keep.sum())}
            for key in (final, "baseline"):
                s = score_intervals("regression", y[keep], runs[key]["pred"][keep],
                                    reference=runs[key]["ref"][keep])
                entry[key] = {m: interval(i) for m, i in s.items()}
            entry["share_at_or_above_cut"] = float((y[keep] >= CUT).mean())
            entry["by_band"] = by_band(y[keep], runs[final]["pred"][keep])
            rows.append(entry)
        groups[name] = rows
    out["groups"] = groups

    # Each cycle: its miss and its signed error, the final family against the no-predictor model.
    cycles = []
    for k, label in enumerate(labels):
        keep = fold == k
        e = runs[final]["pred"][keep] - y[keep]
        b = runs["baseline"]["pred"][keep] - y[keep]
        cycles.append({"cycle": label, "n": int(keep.sum()), "mae": float(np.mean(np.abs(e))),
                       "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
                       "baseline_mae": float(np.mean(np.abs(b))), "baseline_bias": float(np.mean(b)),
                       "mean_observed": float(np.mean(y[keep])),
                       "share_at_or_above_cut": float((y[keep] >= CUT).mean())})
    out["cycles"] = cycles

    # The held-out rows, already opened by the capture: the final family's fit on every training row
    # (the fit's own ``fitted``), its groups at the training rows' age thirds, and each cycle.
    held_ids = split.frames["sealed"]["row_id"].to_numpy(dtype=np.int64)
    held = modeling_frame(store, [*spec.inputs, *extra, "glucose"], held_ids, outcome="glucose")
    held = held[np.isfinite(held["glucose"].to_numpy(dtype=float))]
    yh = held["glucose"].to_numpy(dtype=float)
    ph = np.asarray(fit.objects["fitted"][final].predict(held[list(spec.inputs)]), dtype=float)
    train_mean = float(np.mean(y))
    cuts = np.quantile(frame["age"].to_numpy(dtype=float), [1 / 3, 2 / 3])
    age_h = held["age"].to_numpy(dtype=float)
    age_lab = np.where(age_h <= cuts[0], f"≤ {cuts[0]:g}",
                       np.where(age_h <= cuts[1], f"{cuts[0]:g}–{cuts[1]:g}", f"> {cuts[1]:g}"))
    asked_h = held["meds_hbp"].notna() | held["meds_chol"].notna()
    held_groups: dict[str, Any] = {}
    for name, lab in (("gender", held["gender"].astype(str).to_numpy()), ("age", age_lab),
                      ("diagnosis_on_record", np.where(asked_h, "asked either medicine question",
                                                       "asked neither"))):
        rows = []
        for level in sorted(set(lab.tolist()), key=str):
            keep = lab == level
            s_final = score_intervals("regression", yh[keep], ph[keep], reference=train_mean)
            s_base = score_intervals("regression", yh[keep], np.full(int(keep.sum()), train_mean),
                                     reference=train_mean)
            rows.append({"group": str(level), "n": int(keep.sum()),
                         final: {m: interval(i) for m, i in s_final.items()},
                         "baseline": {m: interval(i) for m, i in s_base.items()},
                         "share_at_or_above_cut": float((yh[keep] >= CUT).mean()),
                         "at_cut": at_cut(yh[keep], ph[keep]) if (yh[keep] >= CUT).any() else None})
        held_groups[name] = rows
    held_cycles = []
    for label in sorted(set(held["cycle_begin_year"].astype(int).astype(str).tolist())):
        keep = held["cycle_begin_year"].astype(int).astype(str).to_numpy() == label
        e = ph[keep] - yh[keep]
        held_cycles.append({"cycle": label, "n": int(keep.sum()), "mae": float(np.mean(np.abs(e))),
                            "bias": float(np.mean(e)), "bias_se": float(np.std(e, ddof=1) / math.sqrt(keep.sum())),
                            "mean_observed": float(np.mean(yh[keep]))})
    out["held_out"] = {"n": int(len(yh)), "final": final, "train_mean": train_mean,
                       "scores": {m: interval(i) for m, i in
                                  score_intervals("regression", yh, ph, reference=train_mean).items()},
                       "baseline": {m: interval(i) for m, i in score_intervals(
                           "regression", yh, np.full(len(yh), train_mean), reference=train_mean).items()},
                       "by_band": by_band(yh, ph), "at_cut": at_cut(yh, ph),
                       "groups": held_groups, "cycles": held_cycles}
    store.close()
    args.out.write_text(json.dumps(finite(out), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M1 model-side stages: the shelf, the design (pipelines + column lineage), the fit, and
substitution curves (M1_CONTRACT §3).

Owner: the M1 "modeling" agent.

Row discipline: everything learned from data is learned on training rows. The design fits its
shared steps on training rows only to name the matrix's columns; the fit cross-validates on
training rows with the split's own folds (every step refit per fold, inside the pipeline) and
scores the held-out rows once, with the pipeline refit on all training rows; substitution curves
average over training rows.
"""
from __future__ import annotations

import time
import warnings
from collections import Counter
from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import turbotab.core.models  # noqa: F401 - registers the families and their previews
from turbotab.core.graph import Bundle, StageContext
from turbotab.core.jobs import Cancelled
from turbotab.core.stages.data import open_store

SUBSTITUTION_ROWS = 5_000
SUBSTITUTION_STEPS = 10  # ks = 0, step, …, 10 × step
SUBSTITUTION_BOOT = 200
MAX_PAIRS = 60
TRAIN = ("train", "training")


# ── reading upstream artifacts ───────────────────────────────────────────────


def row_ids_of(frame: pd.DataFrame) -> np.ndarray:
    """Row ids from a frame that carries them as a ``row_id`` column or as its index."""
    for name in ("row_id", "__row_id"):
        if name in frame.columns:
            return frame[name].to_numpy(dtype=np.int64)
    return frame.index.to_numpy(dtype=np.int64)


def read_assignment(split: Any) -> pd.DataFrame:
    """The split's assignment, indexed by row id: ``train`` (bool) and ``fold`` (training rows)."""
    frame = split.frames["assignment"] if isinstance(split, Bundle) else split
    ids = row_ids_of(frame)
    partition = frame["partition"].astype(str).str.lower().to_numpy()
    train = np.isin(partition, TRAIN)
    fold = pd.to_numeric(frame["fold"], errors="coerce").to_numpy()
    out = pd.DataFrame({"train": train, "fold": fold}, index=pd.Index(ids, name="row_id"))
    if out.loc[out["train"], "fold"].isna().any():
        raise ValueError("Some training rows have no cross-validation fold in the split.")
    return out


def _task(ctx: StageContext) -> str:
    return str(ctx.state.task or ctx.inputs["target_info"]["task"])


def _families(ctx: StageContext, task: str) -> list[Any]:
    from turbotab.core.models import get_family

    chosen = []
    for key in ctx.state.models or []:
        family = get_family(key)
        if task not in family.tasks:
            raise ValueError(f"{family.label} cannot model a {task} outcome.")
        chosen.append(family)
    return chosen


# ── shelf ─────────────────────────────────────────────────────────────────────


def shelf_stage(ctx: StageContext) -> dict[str, Any]:
    """Every family that can model the task, ranked for this table, concerns stated."""
    from turbotab.core.models import Situation, rank
    from turbotab.core.models.artifacts import ShelfArtifact, ShelfFamily
    from turbotab.core.models.pipeline import predictors_from_roles

    cohort = ctx.inputs["cohort"]
    data = cohort.data if isinstance(cohort, Bundle) else cohort
    task = _task(ctx)
    predictors = list(data.get("predictors") or predictors_from_roles(ctx.state.roles, ctx.state.target))
    n = int(data["n_final"])
    n_events = n_classes = None
    if task != "regression" and isinstance(cohort, Bundle) and "rows" in cohort.frames:
        with open_store(ctx) as store:
            y = store.materialize([ctx.state.target], row_ids_of(cohort.frames["rows"]))
        counts = y[ctx.state.target].value_counts(dropna=True)
        n_classes = int(len(counts))
        if task == "binary" and n_classes:
            n_events = int(counts.min())
    situation = Situation(task=task, purpose=ctx.state.purpose, n_rows=n,
                          n_features=len(predictors), n_events=n_events, n_classes=n_classes)
    ranked = rank(situation)
    events = f", {n_events:,} in the rarer class" if n_events is not None else ""
    artifact = ShelfArtifact(
        families=[
            ShelfFamily(key=f.key, label=f.label, rank=i + 1, fit=a.fit, concerns=list(a.concerns),
                        inductive_bias=f.inductive_bias)
            for i, (f, a) in enumerate(ranked)
        ],
        basis=f"Ranked for {n:,} rows and {len(predictors):,} predictors{events}.",
    )
    return artifact.model_dump(mode="json")


# ── design ────────────────────────────────────────────────────────────────────


def substitution_pairs(predictors: Sequence[str], energy_column: str | None) -> list[dict[str, str]]:
    """Ordered (donor, recipient) pairs of predictors that carry energy in a known unit."""
    from turbotab.core.methods.energy import energy_factor

    bearing = [c for c in predictors if c != energy_column and energy_factor(c).factor is not None]
    return [{"donor": d, "recipient": r} for d, r in permutations(bearing, 2)][:MAX_PAIRS]


def design_stage(ctx: StageContext) -> Bundle:
    """Each chosen family's pipeline, and the lineage from raw columns to the model matrix."""
    from turbotab.core.methods.energy import METHOD_TABLE
    from turbotab.core.models.artifacts import DesignArtifact
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import (
        build_pipeline,
        describe_steps,
        design_spec,
        input_columns,
        modeling_frame,
        predictors_from_roles,
        shared_steps,
        transformer,
        warnings_for,
    )

    state = ctx.state
    task = _task(ctx)
    families = _families(ctx, task)
    predictors = predictors_from_roles(state.roles, state.target)
    if not predictors:
        raise ValueError("No column has the role exposure, covariate or energy, so there is "
                         "nothing to model the outcome with.")
    assignment = read_assignment(ctx.inputs["split"])
    train_ids = assignment.index[assignment["train"]].to_numpy()
    adj = state.energy_adjustment
    ctx.progress(0.05, "Reading the training rows")
    with open_store(ctx) as store:
        X = modeling_frame(store, input_columns(predictors, adj), train_ids)
    spec = design_spec(state, X, predictors)
    warnings_list = warnings_for(spec, X, [f.key for f in families], {f.key: f for f in families})

    ctx.progress(0.3, "Fitting the shared steps on training rows")
    shared = transformer(shared_steps(spec))
    matrix = shared.fit_transform(X[spec.inputs])
    lineage = trace(shared.steps, spec.inputs, spec.roles, missing_counts(X[spec.inputs]))
    if "energy" in shared.named_steps:
        warnings_list.extend(_energy_warnings(shared.named_steps["energy"]))

    ctx.progress(0.8, "Building each model's pipeline")
    n_rows, n_cols = int(matrix.shape[0]), int(matrix.shape[1])
    pipelines = {f.key: build_pipeline(spec, f, task, state.purpose, n_rows, n_cols) for f in families}
    models = [{"family": f.key, "label": f.label,
               "steps": describe_steps(spec, f, task, state.purpose, n_cols)} for f in families]
    energy_column = adj.energy_column if adj is not None else None
    artifact = DesignArtifact(
        lineage=lineage,
        matrix={"n_rows": n_rows, "n_cols": n_cols},
        models=models,
        estimand=METHOD_TABLE[adj.method]["estimand"] if adj is not None else None,
        substitution_pairs=substitution_pairs(spec.predictors, energy_column),
        warnings=warnings_list,
    )
    return Bundle(
        data=artifact.model_dump(mode="json"),
        frames={"training": pd.DataFrame({"row_id": train_ids.astype(np.int64)})},
        objects={"pipelines": pipelines, "spec": spec.to_dict()},
    )


def _energy_warnings(step: Any) -> list[str]:
    out = []
    adjuster = getattr(step, "pooled_", step)
    other = getattr(adjuster, "params_", {}).get("__other__")
    if other and other.get("rows_below_zero"):
        n = other["rows_below_zero"]
        out.append(f"{n:,} training row{'s' if n != 1 else ''} get a negative kcal_from_other: the "
                   f"chosen nutrients carry more energy than the recorded total.")
    for n, p in getattr(adjuster, "params_", {}).items():
        if n != "__other__" and isinstance(p, Mapping) and p.get("r2") is not None and p["r2"] < 0.05:
            out.append(f"Energy explains {p['r2']:.0%} of {n}'s variation, so adjusting it barely "
                       f"changes it.")
    return out


# ── fit ───────────────────────────────────────────────────────────────────────


def _concerns(caught: Sequence[warnings.WarningMessage], n_fits: int) -> list[str]:
    from sklearn.exceptions import ConvergenceWarning

    counts: Counter[str] = Counter()
    for w in caught:
        if issubclass(w.category, ConvergenceWarning) or "converge" in str(w.message).lower():
            counts["converge"] += 1
        elif "unknown categor" in str(w.message).lower():
            counts["unknown"] += 1
        elif "ill-conditioned" in str(w.message).lower() or "singular" in str(w.message).lower():
            counts["singular"] += 1
    out = []
    if counts["converge"]:
        out.append(f"The optimizer stopped before converging {counts['converge']} time"
                   f"{'s' if counts['converge'] != 1 else ''} over {n_fits} fits; treat these "
                   f"numbers with care.")
    if counts["unknown"]:
        out.append("A category seen only in held-out rows was encoded as the reference level.")
    if counts["singular"]:
        out.append("The model matrix is close to singular: some columns are nearly redundant.")
    return out


def fit_stage(ctx: StageContext) -> Bundle:
    """Cross-validate each pipeline on the split's folds, then refit on all training rows."""
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import FitArtifact
    from turbotab.core.models.metrics import PRIMARY, metric_labels, score, summarize
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame

    state = ctx.state
    task = _task(ctx)
    design = ctx.inputs["design"]
    split = ctx.inputs["split"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    assignment = read_assignment(split)
    target = state.target
    grouped_by = (split.data or {}).get("grouped_by") if isinstance(split, Bundle) else None

    ctx.progress(0.01, "Reading the analysis rows")
    columns = [*spec.inputs, target] + ([grouped_by] if grouped_by and grouped_by not in spec.inputs
                                        and grouped_by != target else [])
    with open_store(ctx) as store:
        frame = modeling_frame(store, columns, assignment.index.to_numpy())
    train = assignment["train"].to_numpy()
    y_all = frame[target]
    if y_all.isna().any():
        raise ValueError(f"{int(y_all.isna().sum()):,} analysis rows have no {target}; the cohort "
                         f"should have left them out.")
    X, y = frame.loc[train, spec.inputs], y_all[train].to_numpy()
    X_hold, y_hold = frame.loc[~train, spec.inputs], y_all[~train].to_numpy()
    folds = assignment.loc[train, "fold"].to_numpy().astype(int)
    fold_keys = sorted(set(folds.tolist()))
    groups = frame.loc[train, grouped_by].to_numpy() if grouped_by else None

    keys = [k for k in (state.models or []) if k in pipelines]
    units = max(1, len(keys) * (len(fold_keys) + 2))
    done = 0

    def share(n: int) -> float:
        return 0.02 + 0.97 * n / units

    models: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for key in keys:
        family = get_family(key)
        started = time.perf_counter()
        per_fold: list[dict[str, float]] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i, k in enumerate(fold_keys):
                if ctx.cancelled():
                    raise Cancelled()
                ctx.progress(share(done), f"{family.label}: fold {i + 1} of {len(fold_keys)}")
                fit_rows, test_rows = folds != k, folds == k
                model = clone(pipelines[key]).fit(X[fit_rows], y[fit_rows])
                per_fold.append(score(task, model, X[test_rows], y[test_rows]))
                done += 1
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{family.label}: refitting on all training rows")
            final = clone(pipelines[key]).fit(X, y)
            done += 1
            holdout = score(task, final, X_hold, y_hold) if len(y_hold) else None
            ctx.progress(share(done), f"{family.label}: coefficients")
            concerns: list[str] = []
            try:
                coefficients = family.coefficients(final, X, y, task=task, purpose=state.purpose,
                                                   groups=groups)
            except Exception as exc:  # noqa: BLE001 - a table that cannot be computed is a concern
                coefficients = None
                concerns.append(f"The coefficient table could not be computed: {exc}")
            done += 1
        concerns = _concerns(caught, len(fold_keys) + 1) + concerns
        if coefficients is not None and groups is not None and state.purpose == "inference" \
                and family.key == "linear":
            concerns.append(f"Confidence intervals are cluster-robust by {grouped_by}, because its "
                            f"rows repeat.")
        fitted[key] = final
        models.append({
            "family": key,
            "label": family.label,
            "cv": summarize(task, per_fold),
            "holdout": holdout,
            "coefficients": coefficients,
            "fit_seconds": round(time.perf_counter() - started, 3),
            "concerns": concerns,
        })
    ctx.progress(1.0, "Done")
    artifact = FitArtifact(task=task, primary_metric=PRIMARY[task], metric_labels=metric_labels(task),
                           n_train=int(train.sum()), n_holdout=int((~train).sum()), models=models)
    return Bundle(data=artifact.model_dump(mode="json"), objects={"fitted": fitted})


# ── substitution ─────────────────────────────────────────────────────────────


def _predictor(task: str, pipeline: Any) -> Any:
    """What a curve follows: the prediction, or for a binary outcome the second class's probability."""
    if task == "regression":
        return pipeline.predict
    return lambda frame: pipeline.predict_proba(frame)[:, 1]


def substitution_stage(ctx: StageContext) -> dict[str, Any]:
    """Move k kcal from the donor to the recipient and follow each fitted model's prediction."""
    from turbotab.core.methods.energy import energy_factor
    from turbotab.core.methods.substitution import substitution_curve
    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import SubstitutionArtifact
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame

    sub = ctx.state.substitution
    fit, design = ctx.inputs["fit"], ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    task = fit.data["task"]
    for column in (sub.donor, sub.recipient):
        if column not in spec.inputs:
            raise ValueError(f"{column} is not one of the model's predictors, so energy cannot be "
                             f"moved through it.")
    readings = {c: energy_factor(c) for c in (sub.donor, sub.recipient)}
    for c, reading in readings.items():
        if reading.factor is None:
            raise ValueError(f"{c} carries no energy in a known unit: {reading.reason}.")
    kcal_per_unit = {c: float(r.factor) for c, r in readings.items()}

    train_ids = row_ids_of(design.frames["training"])
    if len(train_ids) > SUBSTITUTION_ROWS:
        train_ids = np.sort(np.random.default_rng(0).choice(train_ids, SUBSTITUTION_ROWS, replace=False))
    ctx.progress(0.05, "Reading training rows")
    with open_store(ctx) as store:
        X = modeling_frame(store, spec.inputs, train_ids)
    ks = [sub.step_kcal * i for i in range(SUBSTITUTION_STEPS + 1)]
    target = ctx.state.target
    models = []
    note = None
    skipped = []
    fitted = fit.objects["fitted"]
    keys = [m["family"] for m in fit.data["models"] if m["family"] in fitted]
    for i, key in enumerate(keys):
        if ctx.cancelled():
            raise Cancelled()
        family = get_family(key)
        ctx.progress(0.1 + 0.85 * i / max(1, len(keys)), f"{family.label}: moving energy")
        if task == "multiclass":
            skipped.append(family.label)
            continue
        curve = substitution_curve(_predictor(task, fitted[key]), X, donor=sub.donor,
                                   recipient=sub.recipient, kcal_per_unit=kcal_per_unit, ks=ks,
                                   total_kind="variable", n_boot=SUBSTITUTION_BOOT, random_state=0)
        note = note or curve["note"]
        models.append({
            "family": key, "label": family.label, "delta": curve["delta"],
            "ci_low": curve["ci_low"], "ci_high": curve["ci_high"],
            "on_support_fraction": curve["on_support_fraction"], "stopped_at": curve["stopped_at"],
            "effect_label": curve["effect_label"],
        })
    notes = [note] if note else []
    for c, reading in readings.items():
        if not reading.declared:
            notes.append(f"{c} is read as {reading.role} in grams at {reading.factor:g} kcal/g; its "
                         f"name does not state the unit.")
    if skipped:
        notes.append("A multiclass outcome has one curve per class, which is not drawn yet.")
    if task == "binary" and keys:
        positive = fitted[keys[0]].classes_[1]
        outcome = f"the predicted probability that {target} is {positive}"
    else:
        outcome = f"predicted {target}"
    estimand = (f"The average change in {outcome} when k kcal move from {sub.donor} to "
                f"{sub.recipient}, with every other input, total energy included, left as it was.")
    artifact = SubstitutionArtifact(
        donor=sub.donor, recipient=sub.recipient, step_kcal=float(sub.step_kcal),
        ks=[float(k) for k in ks], total_kind="variable", estimand=estimand, note=" ".join(notes),
        basis=f"Averaged over {len(X):,} training rows.", models=models,
    )
    ctx.progress(1.0, "Done")
    return artifact.model_dump(mode="json")


__all__ = ["design_stage", "fit_stage", "read_assignment", "row_ids_of", "shelf_stage",
           "substitution_pairs", "substitution_stage"]

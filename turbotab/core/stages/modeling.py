"""M1 model-side stages: the shelf, the design (pipelines + column lineage), the fit, and
substitution curves (M1_CONTRACT §3).

Owner: the M1 "modeling" agent.

Row discipline: everything learned for prediction is learned on training rows. The design fits
its shared steps on training rows only to name the matrix's columns; the fit cross-validates on
training rows with the split's own folds (every step refit per fold, inside the pipeline) and
scores the held-out rows once, with the pipeline refit on all training rows; substitution curves
average over training rows.

The seal is purpose-scoped (BLUEPRINT §12 ruling 3; AUDIT_REPORT §5 WP8, ME-12): under inference
the coefficient table is estimated from every analyzed row, held-out ones included, with the
pipeline refit on all of them and the number of rows stated. A holdout is a prediction concept: on
the NHANES export, twenty random 80% seals moved the sugar coefficient from −0.097 to −0.021 and
made it "significant" in 9 of 20, where all 2,996 rows give one answer.
"""
from __future__ import annotations

import math
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
# Each refit of the band resamples every training row (the ordinary bootstrap) up to this many
# rows; on more, it draws this many and the band is rescaled to the full sample (m-out-of-n,
# ``refit_band``). A bound on the cost of a refit, not on what the band describes.
BAND_ROWS = 10_000
# The band the Results offer, and the one band_estimate times: the curve ± 1.96 standard errors
# of 200 refits (a percentile band needs 1,000; ``methods.substitution.PERCENTILE_MIN_REFITS``).
BAND_BOOT = 200
MAX_PAIRS = 60
TRAIN = ("train", "training")


# ── reading upstream artifacts ───────────────────────────────────────────────


def row_ids_of(frame: pd.DataFrame) -> np.ndarray:
    """Row ids from a frame that carries them as a ``row_id`` column or as its index."""
    for name in ("row_id", "__row_id"):
        if name in frame.columns:
            return frame[name].to_numpy(dtype=np.int64)
    return frame.index.to_numpy(dtype=np.int64)


def repeat_columns(frame: pd.DataFrame) -> list[str]:
    """``fold_r1``, ``fold_r2``, … in order: the repeats after the first (``fold``)."""
    names = [c for c in frame.columns if str(c).startswith("fold_r") and str(c)[6:].isdigit()]
    return sorted(names, key=lambda c: int(str(c)[6:]))


def read_assignment(split: Any) -> pd.DataFrame:
    """The split's assignment, indexed by row id: ``train`` (bool) and ``fold`` (training rows)."""
    frame = split.frames["assignment"] if isinstance(split, Bundle) else split
    ids = row_ids_of(frame)
    partition = frame["partition"].astype(str).str.lower().to_numpy()
    train = np.isin(partition, TRAIN)
    fold = pd.to_numeric(frame["fold"], errors="coerce").to_numpy()
    out = pd.DataFrame({"train": train, "fold": fold}, index=pd.Index(ids, name="row_id"))
    if "order" in frame.columns:  # time-ordered folds: each row's unit rank in time
        out["order"] = pd.to_numeric(frame["order"], errors="coerce").to_numpy()
    for name in repeat_columns(frame):  # repeated k-fold: the second and later draws of the folds
        out[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy()
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
    rows = row_ids_of(cohort.frames["rows"]) if isinstance(cohort, Bundle) and "rows" in cohort.frames else None
    # The shelf informs a modeling choice made after the seal, so it reads the training rows only
    # (M2_CONTRACT §3, the held-out discipline audit): never the held-out rows' outcomes.
    split = ctx.inputs.get("split")
    trained = rows is not None and split is not None
    if trained:
        assignment = read_assignment(split)
        rows = np.intersect1d(rows, assignment.index[assignment["train"]].to_numpy())
        n = int(len(rows))
    n_events = n_classes = n_units = None
    outcome_mean = outcome_sd = None
    class_counts = None
    with open_store(ctx) as store:
        column_info = {c.name: c for c in store.info().columns}
        if rows is not None:
            y = store.materialize([ctx.state.target], rows)[ctx.state.target]
            if task == "regression":
                values = pd.to_numeric(y, errors="coerce").dropna()
                if len(values) > 1:
                    outcome_mean, outcome_sd = float(values.mean()), float(values.std(ddof=1))
            else:
                counts = y.value_counts(dropna=True)
                n_classes = int(len(counts))
                class_counts = tuple(int(c) for c in counts.to_numpy())
                if task == "binary" and n_classes:
                    n_events = int(counts.min())
                elif task == "time_to_event":
                    coded = coded_outcome(task, y.to_numpy(), ctx.state.event)
                    n_events = int((pd.to_numeric(pd.Series(coded), errors="coerce") == 1).sum())
            n_units = _units_in(ctx, store, rows)
    from turbotab.core.methods.exposure_form import model_terms

    # A spline or quintile exposure puts several columns in the model (WP12a); a sample-size
    # criterion counts them as parameters too (WP8).
    terms = model_terms(predictors, ctx.state.exposure_forms)
    situation = Situation(task=task, purpose=ctx.state.purpose, n_rows=n,
                          n_features=terms, n_events=n_events, n_classes=n_classes,
                          n_parameters=predictor_parameters(predictors, column_info,
                                                            ctx.state.categorical)
                          + (terms - len(predictors)),
                          outcome_mean=outcome_mean, outcome_sd=outcome_sd,
                          lenses=tuple(ctx.state.lens or ()), class_counts=class_counts,
                          n_units=n_units)
    ranked = rank(situation)
    # WP11: raw counts or intensities whose totals track the outcome, on the training rows only
    assay = _assay_concern(ctx, task, rows if trained else None)
    events = ("" if n_events is None else f", {n_events:,} with the event" if task == "time_to_event"
              else f", {n_events:,} in the rarer class")
    estimates = _estimates(ctx, task, rows if trained else None, [f for f, _ in ranked])
    artifact = ShelfArtifact(
        families=[
            ShelfFamily(key=f.key, label=f.label, rank=i + 1, fit=a.fit,
                        concerns=([assay] if assay else []) + list(a.concerns),
                        inductive_bias=f.inductive_bias,
                        estimate_seconds=estimates[f.key].seconds if estimates.get(f.key) else None,
                        estimate=estimates[f.key].text if estimates.get(f.key) else None)
            for i, (f, a) in enumerate(ranked)
        ],
        basis=f"Ranked for {n:,} {'training ' if trained else ''}rows and {len(predictors):,} "
              f"predictors{_terms_clause(terms, len(predictors))}{events}.",
    )
    return artifact.model_dump(mode="json")


CATEGORY_DTYPES = ("categorical", "text")


def predictor_parameters(predictors: Sequence[str], column_info: Mapping[str, Any],
                         declared: Sequence[str] | None = None) -> int:
    """The candidate predictor parameters a sample-size criterion counts: one per number, and
    k − 1 per category of k levels (text, or an integer column declared categorical), as the
    one-hot step encodes it. A column the summaries do not know counts once."""
    codes = set(declared or [])
    total = 0
    for column in predictors:
        info = column_info.get(column)
        dtype = getattr(info, "dtype", None)
        levels = int(getattr(info, "n_unique", 0) or 0)
        total += max(1, levels - 1) if (dtype in CATEGORY_DTYPES or column in codes) else 1
    return total


def _assay_concern(ctx: StageContext, task: str, train_ids: Any) -> str | None:
    """The library-size check (``methods.omics.check_on``) on the training rows: a sentence when the
    exposures are raw counts or intensities, no normalization is recorded, and their per-sample
    totals track the outcome. Never reads a held-out row."""
    from turbotab.core.methods.omics import OMICS_LENSES, check_on

    state = ctx.state
    if train_ids is None or not len(train_ids) or not state.target:
        return None
    if not any(k in OMICS_LENSES for k in state.lens or []):
        return None
    exposures = [c for c, r in (state.roles or {}).items() if r == "exposure" and c != state.target]
    if not exposures:
        return None
    with open_store(ctx) as store:
        present = [c for c in exposures if c in set(store.columns)]
        frame = store.materialize([*present, state.target], train_ids)
    frame = frame.loc[frame[state.target].notna()]
    y = coded_outcome(task, frame[state.target].to_numpy(), state.event)
    check = check_on(frame[present], state, y, task)
    return check["sentence"] if check and check["flagged"] else None


def _terms_clause(terms: int, predictors: int) -> str:
    """`` (12 model terms, with the spline and quintile columns)`` when forms add columns."""
    return f" ({terms:,} model terms, with the spline and quintile columns)" if terms != predictors else ""


def _units_in(ctx: StageContext, store: Any, rows: Any) -> int | None:
    """How many units these rows repeat within, read as the inference table reads them
    (``inference.resolve_clusters``); None when no identifier repeats in them."""
    from turbotab.core.models.inference import cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import modeling_frame

    split = ctx.inputs.get("split")
    grouped_by = ((split.data or {}).get("grouped_by") if isinstance(split, Bundle) else None)
    columns = cluster_columns(ctx.state, store.columns, [grouped_by])
    if not columns or rows is None or not len(rows):
        return None
    clusters = resolve_clusters(ctx.state, modeling_frame(store, columns, rows), [grouped_by])
    return clusters.n_clusters if clusters.clustered else None


def _estimates(ctx: StageContext, task: str, train_ids: Any, families: Sequence[Any]) -> dict[str, Any]:
    """Each family's measured fit time on these training rows (``models.cost``); {} without them.

    A timing that fails is left out (the family says nothing about its cost) and never fails the
    shelf: the ranking stands without it.
    """
    from turbotab.core.models.cost import estimate_fits

    if train_ids is None or not len(train_ids):
        return {}
    split = ctx.inputs.get("split")
    data = (split.data or {}) if isinstance(split, Bundle) else {}
    # Every fold of every repeat, and each bootstrap refit, is about one fit of the whole table.
    folds = int(data.get("folds") or 5) * int(data.get("repeats") or 1) + int(data.get("n_boot") or 0)
    scheme = str((split.data or {}).get("fold_scheme") or "random") if isinstance(split, Bundle) else "random"
    ctx.progress(0.5, "Timing one fit of each family on a sample of the training rows")
    try:
        with open_store(ctx) as store:
            return estimate_fits(store, ctx.state, task, train_ids, families, folds,
                                 cancelled=ctx.cancelled, scheme=scheme)
    except Cancelled:
        raise
    except Exception:  # noqa: BLE001 - the shelf stands without its estimates
        import logging

        logging.getLogger(__name__).exception("timing the families failed")
        return {}


# ── design ────────────────────────────────────────────────────────────────────


def substitution_pairs(predictors: Sequence[str], energy_column: str | None,
                       nested: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    """Ordered (donor, recipient) pairs of predictors that carry energy in a known unit.

    A total is never paired with its own part (``nested``: child -> parent): kcal moved between
    them go nowhere.
    """
    from turbotab.core.methods.energy import energy_factor

    nested = nested or {}
    bearing = [c for c in predictors if c != energy_column and energy_factor(c).factor is not None]
    return [{"donor": d, "recipient": r} for d, r in permutations(bearing, 2)
            if nested.get(d) != r and nested.get(r) != d][:MAX_PAIRS]


def design_stage(ctx: StageContext) -> Bundle:
    """Each chosen family's pipeline, and the lineage from raw columns to the model matrix."""
    from turbotab.core.decisions import left_out
    from turbotab.core.methods.nesting import nested_components
    from turbotab.core.models.artifacts import DesignArtifact
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import (
        build_pipeline,
        describe_steps,
        design_spec,
        input_columns,
        model_predictors,
        modeling_frame,
        shared_steps,
        transformer,
        warnings_for,
    )

    state = ctx.state
    task = _task(ctx)
    families = _families(ctx, task)
    predictors = model_predictors(state)
    if not predictors:
        raise ValueError("No column has the role exposure, covariate or energy, so there is "
                         "nothing to model the outcome with.")
    if task == "time_to_event":
        from turbotab.core.models.survival import follow_up_columns

        inside = [c for c in follow_up_columns(state) if c in predictors]
        if inside:
            raise ValueError(f"`{inside[0]}` is the outcome's follow-up time, so it cannot also be "
                             f"a predictor: give it the role time.")
    assignment = read_assignment(ctx.inputs["split"])
    train_ids = assignment.index[assignment["train"]].to_numpy()
    adj = state.energy_adjustment
    ctx.progress(0.05, "Reading the training rows")
    y_train = None
    with open_store(ctx) as store:
        X = modeling_frame(store, input_columns(predictors, adj), train_ids)
        info = {c.name: c for c in store.info().columns}  # every row's summary: as the cohort reads it
        if adj is not None and adj.method == "residual_energy_dropped" and state.target in store.columns:
            # The gap the energy-dropped residual opens on these training rows (audit ME-03).
            y_train = store.materialize([state.target], train_ids)[state.target]
    spec = design_spec(state, X, predictors, column_info=info)
    from turbotab.core.methods.omics import design_refusal

    refused = design_refusal(state, X, families)  # WP11: raw omics values into a linear family
    if refused:
        raise ValueError(refused)
    numeric_set = set(spec.numeric)  # a set: 20,000 predictors made the list test quadratic
    numeric = [c for c in spec.predictors if c in numeric_set]
    nested = nested_components(X, numeric)  # on training rows: what the substitution will move
    warnings_list = warnings_for(spec, X, [f.key for f in families], {f.key: f for f in families},
                                 nested)

    ctx.progress(0.3, "Fitting the shared steps on training rows")
    shared = transformer(shared_steps(spec))
    matrix = shared.fit_transform(X[spec.inputs])
    lineage = trace(shared.steps, spec.inputs, spec.roles, missing_counts(X[spec.inputs]))
    if "energy" in shared.named_steps:
        warnings_list.extend(_energy_warnings(shared.named_steps["energy"]))
    # The estimand and each coefficient's meaning, read off the matrix the models will see: the
    # label equals the model fitted (audit WP6; methods.energy.describe_model).
    from turbotab.core.methods.energy import describe_model
    from turbotab.core.methods.exposure_form import form_columns, form_step, formed_meanings

    # A formed exposure (WP12a) is read as the term it was before its spline or quintiles, and
    # its meaning then carried to the columns the form made of it.
    form = form_step(shared)
    formed = form_columns(form) if form is not None else {}
    seen = list(matrix.columns) + [c for c in formed if c not in set(matrix.columns)]
    described = describe_model(adj, spec.predictors, spec.roles, seen, nested=nested,
                               step=shared.named_steps.get("energy"))
    described.terms = formed_meanings(described.terms, form)
    if y_train is not None:
        warnings_list.extend(_residual_gap(state, task, spec, X, matrix, y_train, info))

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
        estimand=described.text,
        substitution_pairs=substitution_pairs(spec.predictors, energy_column, nested),
        warnings=warnings_list,
        nested=[{"column": c, "parent": p} for c, p in nested.items()],
        left_out=[c for c in left_out(state) if c in (state.roles or {})],
        terms=described.terms,
        energy_form=described.form if described.text else None,
    )
    return Bundle(
        data=artifact.model_dump(mode="json"),
        frames={"training": pd.DataFrame({"row_id": train_ids.astype(np.int64)})},
        objects={"pipelines": pipelines, "spec": spec.to_dict(), "nested": nested},
    )


def _coef(value: float) -> str:
    """A coefficient as the gap states it: three significant digits, signed, a true minus."""
    return f"{value:+.3g}".replace("-", "−")


def _residual_gap(state: Any, task: str, spec: Any, X: pd.DataFrame, matrix: pd.DataFrame,
                  y: Any, info: Mapping[str, Any]) -> list[str]:
    """The energy-dropped residual against the standard model, on the training rows (ME-03).

    The model kept with total energy is the residual with energy (identical to the standard
    model's coefficient); both are fit with the same family of least squares or logistic
    regression the linear model uses. One sentence per nutrient, at most three.
    """
    from turbotab.core.methods.energy import coefficient_gap
    from turbotab.core.models.pipeline import design_spec, shared_steps, transformer

    adj = state.energy_adjustment
    kept_adj = adj.model_copy(update={"method": "residual"})
    kept_spec = design_spec(state, X, spec.predictors, energy=kept_adj, column_info=info)
    kept = transformer(shared_steps(kept_spec)).fit_transform(X[kept_spec.inputs])
    coded = coded_outcome(task, pd.Series(y).reindex(matrix.index).to_numpy(), state.event)
    if task == "binary" and not set(pd.unique(pd.Series(coded).dropna())) <= {0, 1}:
        return []
    # A nutrient in a spline or quintiles (WP12a) has no one coefficient to compare.
    formed = {c for c, f in (getattr(state, "exposure_forms", None) or {}).items()
              if getattr(f, "form", "linear") != "linear"}
    pairs = [(f"{n}_adj", f"{n}_adj") for n in adj.nutrients
             if n not in formed and f"{n}_adj" in matrix.columns and f"{n}_adj" in kept.columns]
    gaps = coefficient_gap(task, coded, matrix, kept.loc[matrix.index], pairs)
    E = adj.energy_column
    out = []
    for g in sorted(gaps, key=lambda g: -abs(g["dropped"] - g["standard"]))[:3]:
        out.append(f"{E} left the outcome model: on {g['n_rows']:,} training rows "
                   f"{g['nutrient']}'s coefficient is {_coef(g['dropped'])}, against "
                   f"{_coef(g['standard'])} with {E} kept (the standard model's). The two agree "
                   f"only when no covariate correlates with {E}.")
    return out


def _energy_warnings(step: Any) -> list[str]:
    out = []
    adjuster = getattr(step, "pooled_", step)
    other = getattr(adjuster, "params_", {}).get("__other__")
    if other and other.get("rows_below_zero"):
        n, of = other["rows_below_zero"], other.get("n_fit") or 0
        well = other.get("rows_well_below_zero")
        median = other.get("median_share_of_energy")
        from turbotab.core.methods.energy import PARTITION_NEGATIVE_SHARE, PARTITION_SLACK

        if (well is not None and of and well / of <= PARTITION_NEGATIVE_SHARE
                and median is not None and median > -PARTITION_SLACK):
            # Slightly negative only: the general factors' own error, said as such.
            typical = f"{median:+.1%}".replace("-", "−")
            out.append(f"kcal_from_other is slightly negative on {n:,} of {of:,} training rows "
                       f"(median {typical} of total energy): the general Atwater factors (4, 4, 9 "
                       f"kcal/g) run a little above this table's own, and everything else absorbs "
                       f"the difference.")
        else:
            out.append(f"{n:,} training row{'s' if n != 1 else ''} get a negative kcal_from_other: "
                       f"the chosen nutrients carry more energy than the recorded total.")
    for n, p in getattr(adjuster, "params_", {}).items():
        if n != "__other__" and isinstance(p, Mapping) and p.get("r2") is not None and p["r2"] < 0.05:
            out.append(f"Energy explains {p['r2']:.0%} of {n}'s variation, so adjusting it barely "
                       f"changes it.")
    # Within strata: the levels too small for a slope of their own (StratifiedEnergyAdjuster).
    out.extend(step.notes() if hasattr(step, "notes") else [])
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


BASELINE_LABEL = {"regression": "the outcome's average", "binary": "the class prior",
                  "multiclass": "the class prior", "ordinal": "the level prior",
                  "time_to_event": "one risk for everyone"}


def baseline_model(task: str) -> Any:
    """What predicts without the predictors: the training mean, the training class prior, or for a
    time-to-event outcome one risk score for every row (which orders no pair: C = ½)."""
    from sklearn.dummy import DummyClassifier, DummyRegressor

    if task == "time_to_event":
        from turbotab.core.models.survival import ConstantRisk

        return ConstantRisk()
    return DummyRegressor(strategy="mean") if task == "regression" else DummyClassifier(strategy="prior")


def _baseline_cv(task: str, X: Any, y: Any, pairs: Sequence[Any],
                 repeat_of: Sequence[int] | None = None) -> Any:
    """The baseline cross-validated on the same fold pairs, scored as the models are."""
    from turbotab.core.models.metrics import cross_validate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_validate(task, lambda: baseline_model(task), X, y, pairs, repeat_of=repeat_of)


def _baseline_scores(task: str, X: Any, y: Any, folds: Any, fold_keys: Sequence[int]) -> list[dict[str, float]]:
    """Per-fold baseline scores on random folds ``fold_keys`` (kept for scripts that call it)."""
    folds = np.asarray(folds)
    pairs = [(k, folds != k, folds == k) for k in fold_keys]
    return _baseline_cv(task, X, y, pairs).per_fold


def _two(a: float, b: float) -> tuple[str, str]:
    """Two numbers to as few decimals (two at least) as tell them apart and keep each off zero,
    with true minus signs: ``−0.04`` against ``−0.002``, never against ``−0.00``."""
    def places(x: float) -> int:
        return next((p for p in (2, 3, 4) if x == 0 or abs(x) >= 0.5 * 10 ** -p), 4)

    pa, pb = places(a), places(b)
    while f"{a:.{pa}f}" == f"{b:.{pb}f}" and max(pa, pb) < 4:
        pa, pb = pa + 1, pb + 1
    return f"{a:.{pa}f}".replace("-", "−"), f"{b:.{pb}f}".replace("-", "−")


def baseline_concern(task: str, metric_label: str, model: float | None, base: float | None,
                     lower: bool = False) -> str | None:
    """A plain sentence when a family scores worse than its baseline on the primary metric
    (``lower``: the metric is better when lower, as log loss is)."""
    if model is None or base is None or not np.isfinite(model) or not np.isfinite(base):
        return None
    if (model <= base) if lower else (model >= base):
        return None
    m, b = _two(model, base)
    if task == "regression":
        return f"Predicts worse than the outcome's average: CV {metric_label} {m}, against {b} for the average."
    if task == "binary":
        return f"Separates the classes worse than the class prior: CV {metric_label} {m}, against {b} for the prior."
    if task == "ordinal":
        return (f"Orders the outcome worse than the level prior: CV {metric_label} {m}, against {b} "
                f"for the prior.")
    if task == "time_to_event":
        return (f"Orders the events worse than one risk for everyone: CV {metric_label} {m}, against "
                f"{b}.")
    return (f"Predicts the classes worse than the class prior: CV {metric_label} {m}, against {b} "
            f"for the prior.")


RANKING = {"r2": "highest R²", "auc": "highest AUC", "log_loss": "lowest log loss"}


def ranking_phrase(metric: str) -> str:
    """How families are ranked, in words the banner can use (audit ME-10): AUC ranks risks and is
    named as such, never "best"."""
    from turbotab.core.models.metrics import LABELS, higher_is_better

    return RANKING.get(metric) or (
        f"{'highest' if higher_is_better(metric) else 'lowest'} {LABELS.get(metric, metric)}")


def imbalance_sentence(task: str, y: Any, event: str | None) -> str | None:
    """A binary outcome's event share, and why nothing was resampled (audit E15; BLUEPRINT north
    star 5: "SMOTE versus calibration-preserving weighting"). Van den Goorbergh et al. (JAMIA
    2022;29:1525): "The use of random undersampling, random oversampling, or SMOTE yielded poorly
    calibrated models: the probability to belong to the minority class was strongly
    overestimated. These methods did not result in higher areas under the ROC curve"; "similar
    results were obtained by shifting the probability threshold instead"."""
    y = np.asarray(y)
    if task != "binary" or not len(y):
        return None
    if event is not None and set(pd.unique(y).tolist()) <= {0, 1}:  # coded: 1 is the event
        name, share = f"The event, `{event}`,", float(np.mean(y == 1))
    else:  # no event named: the rarer class
        values, counts = np.unique(y.astype(str), return_counts=True)
        name, share = f"The rarer class, `{values[counts.argmin()]}`,", float(counts.min() / len(y))
    return (f"{name} is {share:.0%} of the training rows. No resampling (SMOTE, over- or "
            f"undersampling) or class weighting was applied: customary in machine-learning "
            f"papers, resampling overestimates the rarer class's probability without raising the "
            f"AUC (van den Goorbergh et al. 2022); a decision moves the threshold instead.")


def precision_sentence(metric: str, summaries: Mapping[str, Any], labels: Mapping[str, str],
                       n_train: int) -> str | None:
    """Each family's standard error on these rows, in one sentence: what a "small" difference is
    here, computed on the user's data instead of a rule of thumb about sample size."""
    from turbotab.core.models.metrics import LABELS

    parts = []
    for key, summary in summaries.items():
        se = (summary.get(metric) or {}).get("se")
        if se is not None:
            parts.append(f"{se:.3f} for {labels.get(key, key)}")
    if not parts:
        return None
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return (f"On these {n_train:,} training rows the cross-validated {LABELS.get(metric, metric)} "
            f"has a standard error of {listed}; whether two families differ is read from their "
            f"paired interval, not from either one alone.")


def coded_outcome(task: str | None, y: Any, event: str | None,
                  order: Sequence[Any] | None = None) -> Any:
    """A binary or time-to-event outcome coded 1 for the level the user named as the event
    (``set_event``), else 0.

    The event is never guessed (M2_CONTRACT §1), and its methods sentence says that level was
    coded 1; the models, their metrics and coefficients must then be about that level, not about
    whichever level sorts last. Unchanged when there is no event or it is not a level here.

    An ordinal outcome becomes the codes 0…K − 1 of its order (``order``, the declared one; numbers
    by value), so every family and metric reads the levels in that order (WP12a).
    """
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        return ordinal_outcome(y, order)[0]
    if task not in ("binary", "time_to_event") or event is None:
        return y
    from turbotab.core.stages.rows import _level_key

    hit = pd.Series(np.asarray(y, dtype=object)).map(_level_key).to_numpy() == _level_key(event)
    return hit.astype(int) if hit.any() else y


def outcome_levels(task: str | None, y: Any, event: str | None) -> dict[Any, Any] | None:
    """Each class as the models hold it, mapped to its level as the data spell it.

    A binary outcome whose event was named is coded 1 for that level and 0 for the other
    (:func:`coded_outcome`); otherwise the models hold the levels themselves. None for regression.
    """
    if task not in ("binary", "multiclass"):
        return None
    values = [v for v in pd.unique(pd.Series(np.asarray(y, dtype=object))) if not pd.isna(v)]
    coded = coded_outcome(task, y, event)
    if coded is y:
        return {v: v for v in values}
    from turbotab.core.stages.rows import _level_key

    key = _level_key(event)
    hit = [v for v in values if _level_key(v) == key]
    others = [v for v in values if _level_key(v) != key]
    reference = others[0] if len(others) == 1 else (" or ".join(str(o) for o in others) or None)
    return {1: hit[0], 0: reference}


def _accepts(fn: Any, name: str) -> bool:
    import inspect

    return name in inspect.signature(fn).parameters


def _inference_table(family: Any, pipeline: Any, X: Any, y: Any, *, task: str, clusters: Any,
                     outcome: Any, rows: str, survey: Any = None) -> Any:
    """``family.inference``, handing it the outcome's names, the rows it is estimated from and the
    survey design when it takes them (a family registered before WP8 or WP10 may not). A family
    that cannot weight by the design under the population answer says so in a concern."""
    extra = {name: value for name, value in (("outcome", outcome), ("rows", rows), ("survey", survey))
             if value is not None and _accepts(family.inference, name)}
    table = family.inference(pipeline, X, y, task=task, clusters=clusters, **extra)
    if table.info.get("n_rows") is None and "rows" not in extra:
        from turbotab.core.models.inference import _on_rows

        table = _on_rows(table, len(X), rows)
    if survey is not None and "survey" not in extra:
        table.concerns.insert(0, "Fit without the survey weights: its scores and coefficients "
                                 "describe these participants, not the surveyed population.")
    return table
UNWEIGHTED_SCORES = ("Scores are unweighted: they describe these rows, not the population the survey "
                     "weights stand for.")


def _survey(ctx: StageContext, clusters: Any) -> tuple[Any, str | None]:
    """The survey answer as this fit applies it (``turbotab.core.methods.survey.for_fit``), and
    the note every model carries under prediction when the table has survey weights.

    Under inference: None without a design; else the design ("surveyed population"), the
    "these participants" answer, or a refusal while the question is unanswered. A design without a
    PSU column takes the unit the intervals cluster by as its sampling unit.
    """
    from turbotab.core.survey import reading_of

    state = ctx.state
    present = reading_of(state).present
    if state.purpose != "inference":
        return None, (UNWEIGHTED_SCORES if present else None)
    if state.survey is None and not present:
        return None, None
    from turbotab.core.methods.survey import for_fit

    unit = clusters.column if clusters is not None and clusters.clustered else None
    with open_store(ctx) as store:
        return for_fit(state, store, unit), None


# ── missing data under inference (audit §5 WP7) ──────────────────────────────


class TableMissing:
    """How the inference table handles missing predictor values: blocked (``refusal`` with its
    ``exits``), pooled over ``imputations``, or a record (``info``) and ``concerns`` its table
    carries (complete cases with their assumption and cost; a recorded single fill)."""

    def __init__(self, refusal: str | None = None, exits: Sequence[dict[str, Any]] = (),
                 imputations: Any = None, info: dict[str, Any] | None = None,
                 concerns: Sequence[str] = ()):
        self.refusal = refusal
        self.exits = list(exits)
        self.imputations = imputations
        self.info = info
        self.concerns = [c for c in concerns if c]

    def record(self, table: Any) -> Any:
        """Put the record and concerns on a table that is not blocked (a pooled table has its own)."""
        if table.info.get("refused") and table.info.get("covariance") == "none" and not table.rows:
            return table
        if self.info is not None and not table.info.get("missing"):
            table.info["missing"] = dict(self.info)
        for c in self.concerns:
            if c not in table.concerns:
                table.concerns.append(c)
        return table


def _missing_for_table(ctx: StageContext, spec: Any, X: pd.DataFrame, y: Any, task: str,
                       keys: Sequence[str], loss: Mapping[str, Any] | None = None) -> TableMissing | None:
    """Under inference, the missing-values answer as the coefficient table applies it.

    ``X`` and ``y`` are the rows the table is estimated from; ``loss`` the complete-case comparison
    the cohort made (default: the cohort artifact's, when the stage has it)."""
    from turbotab.core.methods.imputation import ImputationRefused, M_DEFAULT
    from turbotab.core.methods.missing import (COMPLETE_CASE_ASSUMPTION, INDICATOR_CAUTION,
                                               MI_ASSUMPTION, SINGLE_FILL_CAUTION,
                                               impute_for_inference, missing_block, row_loss_concern)
    from turbotab.core.models import get_family

    state = ctx.state
    answer = spec.missing or (state.missing.model_dump(mode="json") if state.missing else None)
    if not answer:
        return None
    levels = [c for c in spec.levels if c in X.columns and X[c].isna().any()]
    held = missing_block(answer, "inference", levels)
    if held is not None:
        return TableMissing(refusal=held[0], exits=held[1])
    strategy = answer.get("strategy")
    m = int(answer.get("m") or M_DEFAULT)
    if strategy == "multiple_imputation":
        gaps = [c for c in spec.inputs if c not in levels and c in X.columns and X[c].isna().any()]
        base = {"method": "multiple_imputation", "assumption": MI_ASSUMPTION, "m": m,
                "n_rows": int(len(X)), "outcome_in_model": True}
        if not gaps:
            return TableMissing(info={**base, "n_incomplete_rows": 0, "note": (
                "No predictor value is missing among the analyzed rows, so nothing was imputed.")})
        pools = [k for k in keys if hasattr(get_family(k), "inference")]
        if not pools:
            return TableMissing(info={**base, "note": "No chosen family has a table to pool."})
        seed = int(getattr(state.split, "seed", 0) or 0) if state.split is not None else 0
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if done == total or done - last[0] >= max(1, total // 50):
                last[0] = done
                ctx.progress(0.01 + 0.01 * done / max(total, 1),
                             f"Imputing missing values with the outcome: model {done:,} of {total:,}")

        try:
            imputations = impute_for_inference(spec, X, y, task, seed=seed, progress=progress,
                                               cancelled=ctx.cancelled)
        except ImputationRefused as exc:
            return TableMissing(
                refusal=f"Multiple imputation cannot run on these data: {exc}",
                exits=[{"label": "Complete cases, with their assumption stated",
                        "decision": {**answer, "kind": "set_missing", "strategy": "complete_case"}}])
        return TableMissing(imputations=imputations)
    if strategy == "complete_case":
        if loss is None:
            cohort = ctx.inputs.get("cohort")
            data = getattr(cohort, "data", cohort) if cohort is not None else None
            loss = (data or {}).get("complete_case_loss") if isinstance(data, Mapping) else None
        concern = row_loss_concern(loss)
        info = {"method": "complete_case",
                "assumption": COMPLETE_CASE_ASSUMPTION[0].upper() + COMPLETE_CASE_ASSUMPTION[1:] + ".",
                "n_dropped": (int(loss["n_dropped"]) if loss and loss.get("n_dropped") is not None
                              else (0 if loss is None else None)),
                "n_rows": int(len(X))}
        return TableMissing(info=info, concerns=[concern] if concern else [])
    if answer.get("acknowledged"):
        indicator = bool(answer.get("indicators")) or (answer.get("categorical") == "missing_category"
                                                       and bool(levels))
        caution = INDICATOR_CAUTION if indicator else SINGLE_FILL_CAUTION
        said = f" The recorded reason: {answer['reason']}" if answer.get("reason") else ""
        return TableMissing(
            info={"method": "single_fill", "assumption": caution[0].upper() + caution[1:] + ".",
                  "recorded": True, "n_rows": int(len(X)), "note": answer.get("reason")},
            concerns=[f"Kept under inference with its recorded limitation: {caution}.{said}"])
    return None


def _with_levels(fitted: Any, levels: Sequence[str] | None) -> Any:
    """An ordinal fit's cut-points named by the declared levels (as the fit stage names them)."""
    if levels is not None:
        fitted[-1].level_names_ = list(levels)
    return fitted


def pooled_table(family: Any, template: Any, imputations: Any, y: Any, *, task: str, clusters: Any,
                 outcome: Any, survey: Any, design: Any, spec: Any, rows: str,
                 fit: Any, cancelled: Any = None, energy_rows: bool = True
                 ) -> tuple[Any, list[dict[str, Any]] | None, list[dict[str, Any]], list[str]]:
    """The family's inference table pooled over the completed copies (Rubin's rules), with its
    energy rows and exposure-form tests pooled beside it: (table, pooled rows, tests, concerns).

    Each copy is analyzed exactly as an unimputed table is (``fit(model, X_k)`` refits the whole
    pipeline on it; the table, the all-components relative effects and the form tests follow). A
    table refused in one copy (too few clusters) is refused: the reason does not depend on the
    imputed values. ``rows`` = None in the result marks such a refusal."""
    from sklearn.base import clone

    from turbotab.core.methods.exposure_form import exposure_tests
    from turbotab.core.methods.imputation import pool_rows, pooled_cov
    from turbotab.core.methods.missing import mi_concerns, multiple_imputation_info
    from turbotab.core.models.inference import InferenceTable

    m = int(imputations.m)
    tables, row_sets, fits, tests = [], [], [], []
    form_concerns: list[str] = []
    for k, X_k in enumerate(imputations.frames):
        if cancelled is not None and cancelled():
            raise Cancelled()
        fitted = fit(clone(template), X_k)
        table = _inference_table(family, fitted, X_k, y, task=task, clusters=clusters,
                                 outcome=outcome, rows=rows, survey=survey)
        if table.info.get("refused"):
            table.info["missing"] = multiple_imputation_info(imputations, spec, len(y))
            return table, None, [], []
        row_sets.append(_energy_rows(table.rows, design, spec, family, fitted, X_k, y, task,
                                     clusters, outcome, survey) if energy_rows else table.rows)
        found, worries = exposure_tests(family, fitted, X_k, y, task=task, clusters=clusters,
                                        table=table, outcome=outcome, survey=survey)
        if k == 0:
            form_concerns = list(worries)
        tables.append(table)
        fits.append(fitted)
        tests.append(found)
    pooled = pool_rows(row_sets)
    info = dict(tables[0].info)
    info["caption"] = (f"Multiple imputation, m = {m}: each completed table analyzed as follows, "
                       f"then pooled by Rubin's rules, each interval on t with Barnard–Rubin degrees "
                       f"of freedom. {tables[0].info.get('caption', '')}").strip()
    info["missing"] = multiple_imputation_info(imputations, spec, len(y))
    concerns = list(tables[0].concerns) + mi_concerns(imputations, pooled, len(y))
    covs = [t.cov for t in tables]
    cov = None
    if all(c is not None for c in covs) and len({np.shape(c) for c in covs}) == 1:
        names = [str(r["feature"]) for r in tables[0].rows]
        Q = np.asarray([[r["estimate"] if r["estimate"] is not None else np.nan for r in t.rows]
                        for t in tables], dtype=float)
        if Q.shape[1] == len(names) and np.all(np.isfinite(Q)):
            cov = pooled_cov(Q, np.asarray(covs, dtype=float))
    table = InferenceTable(pooled, info, concerns, cov=cov)
    return table, pooled, _pool_form_tests(tests, tables, fits, m), form_concerns


def _pool_form_tests(tests: Sequence[Sequence[dict[str, Any]]], tables: Sequence[Any],
                     fits: Sequence[Any], m: int) -> list[dict[str, Any]]:
    """Each exposure-form test pooled over the imputations: a spline's Wald tests by Li,
    Raghunathan & Rubin's D1 (on each copy's estimates and covariance), the quintile trend's
    coefficient by Rubin's rules."""
    from turbotab.core.methods.exposure_form import form_step
    from turbotab.core.methods.imputation import pool_scalar, pooled_wald
    from turbotab.core.models.inference import format_p

    if not tests or not tests[0]:
        return []
    out: list[dict[str, Any]] = []
    step = form_step(fits[0])
    for test in tests[0]:
        same = [next((t for t in ts if t["column"] == test["column"] and t["test"] == test["test"]),
                     None) for ts in tests]
        if any(t is None for t in same):
            continue
        if test["form"] == "spline":
            if step is None:
                continue
            outputs = step._outputs(test["column"])
            names = outputs if test["test"] == "overall" else outputs[1:]
            Q, U = [], []
            for table in tables:
                where = {str(r["feature"]): j for j, r in enumerate(table.rows)}
                if table.cov is None or any(n not in where for n in names):
                    Q = []
                    break
                ids = [where[n] for n in names]
                Q.append([table.rows[j]["estimate"] for j in ids])
                U.append(np.asarray(table.cov, dtype=float)[np.ix_(ids, ids)])
            result = pooled_wald(np.asarray(Q, dtype=float), np.asarray(U, dtype=float)) if Q else None
            if result is None:
                continue
            f_ref = result["df_den"] is not None
            stat = (f"F({result['df_num']}, {result['df_den']:,.0f}) = {result['statistic']:.2f}"
                    if f_ref else f"χ²({result['df_num']}) = {result['statistic'] * result['df_num']:.2f}")
            out.append({**test, "statistic": result["statistic"] if f_ref else result["statistic"] * result["df_num"],
                        "df_num": result["df_num"], "df_den": result["df_den"],
                        "distribution": "F" if f_ref else "chi2", "p": result["p"],
                        "caption": (f"Pooled over {m} imputations by Li, Raghunathan & Rubin's D1: "
                                    f"{stat}, p = {format_p(result['p'])}. Within each imputation: "
                                    f"{test['caption']}")})
            continue
        ests = [t.get("estimate") for t in same]
        stats_ = [t.get("statistic") for t in same]
        if any(e is None for e in ests) or any(st in (None, 0) for st in stats_):
            continue
        variances = [(float(e) / float(st)) ** 2 for e, st in zip(ests, stats_)]
        dfs = [t.get("df_den") for t in same]
        df_com = None if any(d is None for d in dfs) else float(np.mean(dfs))
        pooled = pool_scalar(ests, variances, df_com)
        if pooled.p is None:
            continue
        out.append({**test, "estimate": pooled.estimate, "ci_low": pooled.ci_low,
                    "ci_high": pooled.ci_high, "p": pooled.p,
                    "statistic": pooled.estimate / math.sqrt(pooled.total), "df_den": pooled.df,
                    "distribution": "t" if pooled.df is not None else "z",
                    "caption": (f"Pooled over {m} imputations by Rubin's rules: the trend coefficient "
                                f"{pooled.estimate:+.4g} per unit, p = {format_p(pooled.p)}. Within "
                                f"each imputation: {test['caption']}")})
    return out


def fit_stage(ctx: StageContext) -> Bundle:
    """Cross-validate each pipeline on the split's folds, then refit on all training rows.

    Each model is set beside its baseline on the same folds (the outcome's training-fold mean, or
    the class prior), and a model that scores worse than it, or is not shown to beat it, says so in
    its first concern. Scores are estimated as ``turbotab.core.models.metrics`` defines: R² against
    the training rows' mean, pooled over every out-of-fold prediction. Time-ordered folds
    (``fold_scheme``) score each fold by a model fit on the folds before it; every inner split a
    model makes (elastic net's penalty, boosted trees' early stopping) is drawn the same way
    (``turbotab.core.models.inner_cv.fit_pipeline``).

    Under inference the coefficients come from the pipeline refit on every analyzed row (the
    training fit when nothing is held out), clustered by whatever unit repeats among them, and on
    the outcome's own scale: odds ratios or relative-risk ratios with the event and the reference
    level named (``models/inference.py``). With two or more families, ``selection`` states how much
    picking the best of them by cross-validation flatters it (``models/selection.py``).
    Audit WP9: every cross-validated score carries a standard error; the out-of-fold predictions
    are checked for calibration (a flagged calibration is a concern); the held-out rows get an
    interval on every score and their own calibration, sealed with the scores. The split's
    validation answer adds repeated k-fold (every repeat's folds), bootstrap optimism correction
    (``turbotab.core.models.validation``) or internal–external validation (one fold per cluster).
    """
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import FitArtifact, HoldoutDetail
    from turbotab.core.models.base import reports_coefficients
    from turbotab.core.models.baseline import compare, no_better_concern
    from turbotab.core.models.inference import Outcome, cluster_columns, resolve_clusters
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import (CV_DEFINITION, LABELS, PRIMARY, SE_DEFINITION,
                                              classes_of, cross_validate, higher_is_better,
                                              metric_labels, predict, repeated_pairs, score)
    from turbotab.core.models.performance import calibration, score_intervals
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.models.selection import OutOfFold, selection_optimism
    from turbotab.core.models.survival import follow_up_columns, time_to_event_outcome
    from turbotab.core.models.validation import (family_differences, internal_external,
                                                 optimism_bootstrap)
    from turbotab.core.seal import SEALED_DETAIL, SEALED_SCORES, sealed_detail_frame, sealed_scores_frame

    state = ctx.state
    task = _task(ctx)
    design = ctx.inputs["design"]
    split = ctx.inputs["split"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    assignment = read_assignment(split)
    target = state.target
    split_data = (split.data or {}) if isinstance(split, Bundle) else {}
    grouped_by = split_data.get("grouped_by")
    scheme = ("time_ordered" if split_data.get("fold_scheme") == "time_ordered"
              and "order" in assignment.columns else "random")

    ctx.progress(0.01, "Reading the analysis rows")
    columns = [*spec.inputs, target] + ([grouped_by] if grouped_by and grouped_by not in spec.inputs
                                        and grouped_by != target else [])
    follow_up = follow_up_columns(state) if task == "time_to_event" else []
    columns += [c for c in follow_up if c not in columns]
    inference = state.purpose == "inference"
    keys = [k for k in (state.models or []) if k in pipelines]
    # A family that models the unit itself (a random intercept, a working correlation) is told
    # each row's unit through its model step's ``units`` parameter, under either purpose.
    unit_models = [k for k in keys if "units" in pipelines[k].steps[-1][1].get_params(deep=False)]
    with open_store(ctx) as store:
        # Under inference the intervals cluster by whatever identifier repeats, however the seal
        # was drawn (AUDIT_REPORT MA-01): read every column that may name the unit.
        unit_columns = (cluster_columns(state, store.columns, [grouped_by])
                        if inference or unit_models else [])
        columns += [c for c in unit_columns if c not in columns]
        frame = modeling_frame(store, columns, assignment.index.to_numpy())
    train = assignment["train"].to_numpy()
    y_all = frame[target]
    if y_all.isna().any():
        raise ValueError(f"{int(y_all.isna().sum()):,} analysis rows have no {target}; the cohort "
                         f"should have left them out.")
    outcome = Outcome(name=target, labels=outcome_levels(task, y_all.to_numpy(), state.event))
    levels = None
    if task == "ordinal":  # codes in the declared order; the names label the cut-points
        from turbotab.core.models.ordinal import ordinal_outcome

        coded, levels = ordinal_outcome(y_all.to_numpy(), state.outcome_order, column=target)
    else:
        coded = coded_outcome(task, y_all.to_numpy(), state.event)
    # A time-to-event outcome is the event with its follow-up: one structured value per row.
    y_values = (time_to_event_outcome(state, frame, coded) if task == "time_to_event"
                else np.asarray(coded))
    X, y = frame.loc[train, spec.inputs], y_values[train]
    X_hold, y_hold = frame.loc[~train, spec.inputs], y_values[~train]
    order_all = assignment["order"].to_numpy(dtype=float) if scheme == "time_ordered" else None
    order = order_all[train] if order_all is not None else None
    validation = str(split_data.get("validation") or "kfold")
    # Every repeat's folds (repeated k-fold: ``fold``, ``fold_r1``, …); one column otherwise.
    fold_columns = [assignment.loc[train, c].to_numpy().astype(int)
                    for c in ["fold", *repeat_columns(assignment)]]
    pairs, repeat_of = repeated_pairs(fold_columns, scheme)
    unit_all = frame[grouped_by].to_numpy() if grouped_by else None
    groups = unit_all[train] if unit_all is not None else None
    hold_groups = unit_all[~train] if unit_all is not None else None
    # The rows the coefficient table is estimated from: every analyzed row under inference
    # (BLUEPRINT §12 ruling 3: a holdout is a prediction concept), the training rows otherwise.
    table_rows = np.ones(len(train), dtype=bool) if inference else train
    same_rows = bool(table_rows.sum() == train.sum())
    X_tab, y_tab = frame.loc[table_rows, spec.inputs], y_values[table_rows]
    clusters = (resolve_clusters(state, frame.loc[table_rows, unit_columns], [grouped_by])
                if inference else None)
    # A family that models the unit itself (WP12b: a random intercept, a working correlation) is
    # told the unit of every row the table may read, indexed by row id; each fit takes its own.
    unit_clusters = clusters if clusters is not None or not unit_models else resolve_clusters(
        state, frame.loc[table_rows, unit_columns], [grouped_by])
    unit_of = (pd.Series(unit_clusters.codes, index=frame.index[table_rows])
               if unit_clusters is not None and unit_clusters.clustered else None)
    if groups is not None and len(pd.unique(groups)) == len(groups):
        # One row per unit (the rows were combined per unit, M2_CONTRACT §2): the split is keyed by
        # the unit, but no unit repeats, so clustering by it changes nothing and "its rows repeat"
        # would be false.
        groups, grouped_by, unit_all, hold_groups = None, None, None, None
    # Under a survey design (audit §5 WP10, ME-06): whose estimate the table is. The design is read
    # over every row of the working table, so rows outside the analysis stay in its variance.
    survey, survey_note = _survey(ctx, clusters)
    # Missing data by purpose (BLUEPRINT §12 ruling 4; audit §5 WP7): under inference, multiple
    # imputation with the outcome and energy for the coefficient table; a single fill or the
    # missing-indicator method held until recorded; complete cases with their assumption and cost.
    missing = _missing_for_table(ctx, spec, X_tab, y_tab, task, keys) if inference else None

    def fit(model: Any, X_fit: Any, y_fit: Any, rows: Any = None) -> Any:
        """Fit a pipeline on these training rows, its inner splits drawn as the folds are."""
        take = slice(None) if rows is None else rows
        return fit_pipeline(model, X_fit, y_fit, groups=None if groups is None else groups[take],
                            order=None if order is None else order[take])

    def fit_table(model: Any, X_fit: Any = None) -> Any:
        """Fit a pipeline on the coefficient table's rows (``X_fit``: a completed copy of them,
        under multiple imputation), its inner splits drawn as the folds are."""
        return fit_pipeline(model, X_tab if X_fit is None else X_fit, y_tab,
                            groups=None if unit_all is None else unit_all[table_rows],
                            order=None if order_all is None else order_all[table_rows])

    def refit_resample(model: Any, X_b: Any, y_b: Any, units_b: Any) -> Any:
        """A bootstrap resample's fit: every copy of a unit keeps to one side of an inner split,
        and a family that models the unit counts each copy as a unit of its own (WP12b)."""
        if unit_of is not None:
            with_units(model, resampled_units(unit_of, X_b.index))
        return fit_pipeline(model, X_b, y_b, groups=units_b)

    primary = PRIMARY[task]
    base = _baseline_cv(task, X, y, pairs, repeat_of)
    base_cv = base.summary(task)
    baseline = {"metric": primary, "value": base_cv[primary]["estimate"], "label": BASELINE_LABEL[task]}
    reference = float(np.mean(y.astype(float))) if task == "regression" and len(y) else None
    n_boot = int(split_data.get("n_boot") or 0) if validation == "bootstrap" else 0
    units = max(1, len(keys) * (len(pairs) + 2 + n_boot))
    done = 0
    results: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    sealed_detail: dict[str, Any] = {}

    def share(n: int) -> float:
        return 0.02 + 0.97 * n / units

    models: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    sealed: dict[str, Any] = {}  # held-out scores: kept out of the public data (M2_CONTRACT §3)
    # Each family's out-of-fold predictions, for the selection's optimism (models/selection.py):
    # one repeat's, so each scored row is predicted once (repeated k-fold scores every row again).
    oof = OutOfFold(task, X, y, [p for p, r in zip(pairs, repeat_of) if r == 0])
    for key in keys:
        family = get_family(key)
        started = time.perf_counter()
        if not getattr(family, "predicts", True):
            # WP11: a family that only tests has no cross-validated or held-out score. Under
            # inference its table, like every other, is estimated from every analyzed row with the
            # pipeline refit on them (WP8; BLUEPRINT §12 ruling 3), clustered as they repeat.
            ctx.progress(share(done), f"{family.label}: tests")
            all_rows = inference and reports_coefficients(family)
            tested = (fit_table(with_units(clone(pipelines[key]), unit_of)) if all_rows and not same_rows
                      else fit(clone(pipelines[key]), X, y))
            X_c, y_c = (X_tab, y_tab) if all_rows else (X, y)
            unit_c = (None if unit_all is None else unit_all[table_rows]) if all_rows else groups
            models.append(_tests_only(family, tested, X_c, y_c, task, state, clusters, unit_c,
                                      baseline, started, rows="all" if all_rows else "training",
                                      survey=survey, survey_note=survey_note,
                                      missing=missing if all_rows else None))
            done += len(pairs) + 2
            continue

        def before_fold(i: int, n: int, _label: str = family.label) -> None:
            nonlocal done
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{_label}: fold {i + 1} of {n}")
            done += 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = cross_validate(task, lambda _k=key: with_units(clone(pipelines[_k]), unit_of),
                                    X, y, pairs,
                                    fit=oof.wrap(key, fit) if len(keys) > 1 else fit,
                                    before_fold=before_fold, repeat_of=repeat_of,
                                    keep_predictions=True)
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{family.label}: refitting on all training rows")
            final = fit(with_units(clone(pipelines[key]), unit_of), X, y)
            if levels is not None:
                final[-1].level_names_ = list(levels)
            done += 1
            # R² on the held-out rows is measured against the training rows' mean.
            holdout = score(task, final, X_hold, y_hold, reference=reference) if len(y_hold) else None
            if len(y_hold):  # sealed with the scores: an interval on each, and calibration
                classes = classes_of(task, final)
                held_pred = predict(task, final, X_hold)
                sealed_detail[key] = HoldoutDetail(
                    intervals=score_intervals(task, y_hold, held_pred, classes=classes,
                                              reference=reference, groups=hold_groups,
                                              unit=grouped_by),
                    calibration=calibration(task, y_hold, held_pred, classes=classes,
                                            groups=hold_groups, where="on the held-out rows"),
                ).model_dump(mode="json")
            optimism = None
            if n_boot:
                def boot_progress(b: int, total: int, _label: str = family.label) -> None:
                    nonlocal done
                    done += 1
                    ctx.progress(share(done), f"{_label}: bootstrap refit {b} of {total}")

                optimism = optimism_bootstrap(
                    task, lambda _k=key: clone(pipelines[_k]), refit_resample, X, y, n_boot=n_boot,
                    seed=int(split_data.get("seed") or 0), groups=groups, unit=grouped_by,
                    final=final, progress=boot_progress, cancelled=ctx.cancelled)
            ctx.progress(share(done), f"{family.label}: coefficients")
            concerns: list[str] = []
            interval_info = None
            n_coefficients = None
            on_all = inference and reports_coefficients(family)
            form_tests: list[dict[str, Any]] = []
            pooled = None  # the coefficient rows pooled over the imputations (multiple imputation)
            try:
                if on_all and not same_rows:
                    ctx.progress(share(done), f"{family.label}: coefficients on every analyzed row")
                table_fit = ((final if same_rows else fit_table(with_units(clone(pipelines[key]), unit_of)))
                             if on_all else final)
                X_c, y_c = (X_tab, y_tab) if on_all else (X, y)
                if clusters is not None and hasattr(family, "inference"):
                    from turbotab.core.models.survey import blocked

                    design_of = survey.design if survey is not None else None
                    if survey is not None and survey.refusal:
                        table = blocked(survey.refusal, survey.exits)
                    elif missing is not None and missing.refusal:
                        table = blocked(missing.refusal, missing.exits,
                                        estimator="not fitted: the missing-values answer decides it")
                    elif missing is not None and missing.imputations is not None and on_all:
                        ctx.progress(share(done), f"{family.label}: each of "
                                                  f"{missing.imputations.m} imputations")
                        table, pooled, form_tests, form_concerns = pooled_table(
                            family, pipelines[key], missing.imputations, y_c, task=task,
                            clusters=clusters, outcome=outcome, survey=design_of, design=design,
                            spec=spec, rows="all",
                            fit=lambda model, X_k: _with_levels(
                                fit_table(with_units(model, unit_of), X_k), levels),
                            cancelled=ctx.cancelled)
                        concerns.extend(form_concerns)
                    else:
                        # Under the population answer the table is design-based, its domain every
                        # analysis row: a holdout is a prediction concept (BLUEPRINT §12 ruling 3),
                        # and NCHS estimates over every eligible row.
                        table = _inference_table(
                            family, table_fit, X_c, y_c, task=task, clusters=clusters,
                            outcome=outcome, rows="all" if on_all else "training",
                            survey=design_of)
                    if missing is not None:
                        missing.record(table)
                    coefficients, interval_info = table.rows, table.info
                    concerns.extend(table.concerns)
                    if survey is not None and survey.concern():
                        concerns.insert(0, survey.concern())
                    if pooled is None:
                        # A spline's tests of association and of nonlinearity; quintiles' trend:
                        # on the fit, rows and design the table itself was estimated from.
                        from turbotab.core.methods.exposure_form import exposure_tests

                        form_tests, form_concerns = exposure_tests(
                            family, table_fit, X_c, y_c, task=task, clusters=clusters, table=table,
                            outcome=outcome, survey=design_of)
                        concerns.extend(form_concerns)
                else:
                    unit_c = (None if unit_all is None else unit_all[table_rows]) if on_all else groups
                    coefficients = family.coefficients(table_fit, X_c, y_c, task=task,
                                                       purpose=state.purpose, groups=unit_c)
                    if survey is not None and survey.answer == "population":
                        concerns.append("Fit without the survey weights: its scores and "
                                        "coefficients describe these participants, not the "
                                        "surveyed population.")
                    if missing is not None and missing.imputations is not None and coefficients:
                        concerns.append("These coefficients come from a single fill in each "
                                        "fold, not the multiple imputations: the family has no "
                                        "table to pool.")
                n_coefficients = len(y_c) if coefficients is not None else None
                if on_all and not same_rows and coefficients is not None:
                    concerns.append(
                        f"Under inference the coefficients are estimated from all {len(y_c):,} "
                        f"analyzed rows, the {int((~train).sum()):,} held-out ones included: a "
                        f"holdout scores prediction only, and every eligible row makes the estimates "
                        f"more precise and independent of the seal's seed.")
            except Exception as exc:  # noqa: BLE001 - a table that cannot be computed is a concern
                coefficients = None
                concerns.append(f"The coefficient table could not be computed: {exc}")
            if coefficients and pooled is None:
                # The relative effects come from the fit the table came from: under inference
                # every analyzed row, with the table's clusters and outcome scale. (Pooled tables
                # carry each imputation's, pooled with the coefficients.)
                coefficients = _energy_rows(coefficients, design, spec, family, table_fit, X_c, y_c,
                                            task, clusters, outcome,
                                            survey.design if survey is not None else None)
            done += 1
        concerns = _concerns(caught, len(pairs) + 1) + concerns
        if family.key == "linear":
            from turbotab.core.models.linear import collinearity_concern, model_matrix

            try:
                singular = collinearity_concern(model_matrix(final, X))
            except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
                singular = None
            if singular and not any("singular" in c for c in concerns):
                concerns.insert(0, singular)
        summary = result.summary(task, groups=groups, unit=grouped_by)
        cv, versus = compare(task, result, base, summary=summary)
        estimate = cv[primary]["estimate"]
        worse = baseline_concern(task, LABELS[primary], estimate, baseline["value"],
                                 lower=not higher_is_better(primary))
        tie = None if worse else no_better_concern(task, LABELS[primary], versus, estimate,
                                                   baseline["value"], _two)
        if worse or tie:
            concerns.insert(0, worse or tie)
        rows_oof, y_oof, pred_oof = result.out_of_fold(0)
        oof_calibration = calibration(
            task, y_oof, pred_oof, classes=classes_of(task, final),
            groups=None if groups is None else groups[rows_oof]) if len(y_oof) else None
        if oof_calibration is not None and oof_calibration.concern:
            concerns.append(oof_calibration.concern)
        if optimism is not None and optimism.refused:
            concerns.append(optimism.refused)
        iecv = None
        if validation == "internal_external" and split_data.get("cluster"):
            iecv = internal_external(task, result, list(split_data.get("fold_labels") or []),
                                     str(split_data["cluster"]), groups=groups, unit=grouped_by)
        if survey_note:
            concerns.append(survey_note)
        results[key], summaries[key] = result, cv
        fitted[key] = final
        sealed[key] = holdout
        models.append({
            "family": key,
            "label": family.label,
            "cv": cv,
            "holdout": None,  # sealed: the server serves it once the seal is opened
            "coefficients": coefficients,
            "fit_seconds": round(time.perf_counter() - started, 3),
            "concerns": concerns,
            "baseline": baseline,
            "versus_baseline": versus.model_dump(mode="json"),
            "inference": interval_info,
            "coefficients_n": n_coefficients,
            "calibration": oof_calibration.model_dump(mode="json") if oof_calibration else None,
            "holdout_detail": None,  # sealed: the server serves it once the seal is opened
            "optimism": optimism.model_dump(mode="json") if optimism is not None else None,
            "internal_external": iecv.model_dump(mode="json") if iecv is not None else None,
            "exposure_tests": form_tests,
        })
    # Picking the best of several families by cross-validation flatters it (audit ME-13).
    if len(results) > 1:
        ctx.progress(0.995, "Choosing among the families: bootstrapping the out-of-fold predictions")
    labels = {m["family"]: m["label"] for m in models}
    selection = selection_optimism(task, primary, results, oof, labels, LABELS[primary])
    ctx.progress(1.0, "Done")
    n_holdout = int((~train).sum())
    comparisons = family_differences(task, results, summaries, labels) if len(results) > 1 else []
    artifact = FitArtifact(task=task, primary_metric=PRIMARY[task], metric_labels=metric_labels(task),
                           n_train=int(train.sum()), n_holdout=n_holdout, models=models,
                           holdout_sealed=n_holdout > 0, fold_scheme=scheme,
                           cv_definition=CV_DEFINITION[task], validation=validation,
                           repeats=len(fold_columns), ranking=ranking_phrase(primary),
                           se_definition=SE_DEFINITION, comparisons=comparisons,
                           precision=precision_sentence(primary, summaries, labels, int(train.sum())),
                           imbalance=imbalance_sentence(task, y, state.event), selection=selection,
                           levels=levels)
    frames = {SEALED_SCORES: sealed_scores_frame(models, sealed)} if n_holdout else {}
    if n_holdout and sealed_detail:
        frames[SEALED_DETAIL] = sealed_detail_frame(sealed_detail)
    return Bundle(data=artifact.model_dump(mode="json"), frames=frames,
                  objects={"fitted": fitted, "grouped_by": grouped_by})


def _energy_rows(coefficients: list[dict[str, Any]], design: Any, spec: Any, family: Any,
                 final: Any, X: Any, y: Any, task: str, clusters: Any,
                 outcome: Any = None, survey: Any = None) -> list[dict[str, Any]]:
    """The coefficient rows with what each means (the design's ``terms``), and under the
    all-components model each nutrient's average relative effect beside them (audit ME-14):
    with the family's own intervals under inference, as a point estimate otherwise.

    ``final``, ``X`` and ``y`` are the fit and rows the coefficient table was estimated from
    (every analyzed row under inference, WP8), so the relative effects share its rows, clusters
    and scale (odds or relative-risk ratios, named by ``outcome``), and under a survey design its
    weights and Taylor-linearized intervals (WP10)."""
    terms = ((design.data or {}).get("terms") or {}) if isinstance(design, Bundle) else {}
    rows = [{**row, "meaning": terms[row["feature"]]} if row.get("feature") in terms else row
            for row in coefficients]
    adj = spec.energy_adjustment()
    if (adj is None or adj.method != "all_components" or task == "multiclass"
            or not hasattr(family, "inference")):
        return rows
    from turbotab.core.methods.energy import relative_effect_rows
    from turbotab.core.models.linear import model_matrix

    matrix = model_matrix(final, X)
    step = getattr(final, "named_steps", {}).get("energy")
    factors = dict(getattr(getattr(step, "pooled_", step), "factors_", {}) or {})
    table = None
    refit = getattr(family, "inference_matrix", None)
    if clusters is not None and refit is not None:
        classes = list(getattr(final[-1], "classes_", [])) or None
        extra = {name: value for name, value in (("outcome", outcome), ("survey", survey))
                 if value is not None and _accepts(refit, name)}

        def table(m: Any) -> Any:
            return refit(m, y, task=task, classes=classes, clusters=clusters, **extra).rows
    try:
        extra = relative_effect_rows(matrix, list(adj.nutrients), factors, table=table,
                                     coefficients=rows)
    except Exception:  # noqa: BLE001 - the coefficients stand without their contrasts
        import logging

        logging.getLogger(__name__).exception("the average relative effects failed")
        extra = []
    return rows + extra


def _tests_only(family: Any, final: Any, X: Any, y: Any, task: str, state: Any, clusters: Any,
                groups: Any, baseline: dict[str, Any], started: float,
                rows: str = "training", survey: Any = None,
                survey_note: str | None = None, missing: Any = None) -> dict[str, Any]:
    """The fit artifact's entry for a family that tests and makes no predictions: its table, its
    concerns, and no score (``cv`` empty, ``holdout`` and ``versus_baseline`` null). ``X`` and
    ``y`` are the rows the table is estimated from (``rows``: every analyzed row under inference).

    Under a survey design (WP10) the table waits for the survey answer as the linear family's does,
    and a population answer is stated as not applied: these tests are not design-weighted."""
    from turbotab.core.models.inference import _on_rows
    from turbotab.core.models.linear import as_clusters

    concerns = [f"{family.label} tests each exposure and makes no predictions, so it has no "
                f"cross-validated or held-out score."]
    coefficients = interval_info = None
    try:
        if survey is not None and survey.refusal:
            from turbotab.core.models.survey import blocked

            table = blocked(survey.refusal, survey.exits)
        elif missing is not None and (missing.refusal or missing.imputations is not None):
            from turbotab.core.models.survey import blocked

            reason = missing.refusal or (
                f"{family.label} is not pooled over multiple imputations here: its tests run "
                f"feature by feature over more columns than an imputation model holds. Choose "
                f"complete cases, or a below-detection fill for values below a limit.")
            exits = missing.exits if missing.refusal else [
                {"label": "Complete cases", "decision": {"kind": "set_missing",
                                                         "strategy": "complete_case"}}]
            table = blocked(reason, exits, estimator="not fitted: the missing-values answer decides it")
        else:
            table = family.inference(final, X, y, task=task,
                                     clusters=clusters if clusters is not None else as_clusters(groups),
                                     event=state.event)
            if table.info.get("n_rows") is None:
                table = _on_rows(table, len(X), rows)
            if missing is not None:
                missing.record(table)
            if survey is not None and survey.answer == "population":
                concerns.append("Fit without the survey weights: its tests describe these "
                                "participants, not the surveyed population.")
            elif survey is not None and survey.concern():
                concerns.append(survey.concern())
        coefficients, interval_info = table.rows, table.info
        concerns.extend(table.concerns)
    except Exception as exc:  # noqa: BLE001 - a table that cannot be computed is a concern
        concerns.append(f"The coefficient table could not be computed: {exc}")
    if survey_note:
        concerns.append(survey_note)
    return {"family": family.key, "label": family.label, "cv": {}, "holdout": None,
            "coefficients": coefficients, "fit_seconds": round(time.perf_counter() - started, 3),
            "concerns": concerns, "baseline": baseline, "versus_baseline": None,
            "inference": interval_info,
            "coefficients_n": len(X) if coefficients is not None else None}


def with_units(pipeline: Any, units: pd.Series | None) -> Any:
    """``pipeline`` with its model step told each row's unit (a Series of unit labels indexed by
    row id), when the step takes ``units``; unchanged otherwise. Returns ``pipeline``."""
    name, model = pipeline.steps[-1]
    if units is not None and "units" in model.get_params(deep=False):
        pipeline.set_params(**{f"{name}__units": units})
    return pipeline


def resampled_units(units: pd.Series, index: Any) -> pd.Series:
    """The units of a resample by whole unit (rows indexed by row id, a unit's rows repeated
    together): the k-th copy of a row is in the k-th copy of its unit, a unit of its own, as a
    cluster bootstrap counts it."""
    index = pd.Index(index)
    labels = units.reindex(index).astype(object).to_numpy()
    copy = pd.Series(np.arange(len(index))).groupby(np.asarray(index)).cumcount().to_numpy()
    return pd.Series([f"{u}#{k}" for u, k in zip(labels, copy)], index=index, dtype=object)


# ── substitution ─────────────────────────────────────────────────────────────


def _predictor(task: str, pipeline: Any) -> Any:
    """What a curve follows: the prediction, or for a binary outcome the second class's probability."""
    if task == "regression":
        return pipeline.predict
    return lambda frame: pipeline.predict_proba(frame)[:, 1]


def substitution_stage(ctx: StageContext) -> dict[str, Any]:
    """Move k kcal from the donor to the recipient and follow each fitted model's prediction.

    The curve averages over at most 5,000 training rows; its support checks amounts and, when the
    total-energy column is a model input, each moved nutrient's share of energy. With
    ``n_boot > 0`` each family is also refit on that many bootstrap resamples of the rows it was
    fit on (whole units when rows repeat; resamples of every training row up to ``BAND_ROWS``,
    rescaled beyond), and the band is drawn from the refits' curves
    (:func:`~turbotab.core.methods.substitution.refit_band`). Without one, a single refit per
    family is timed, so the offer of a band can say what it costs.
    """
    from sklearn.base import clone

    from turbotab.core.methods.energy import energy_factor
    from turbotab.core.methods.substitution import (
        MIN_REFIT_SHARE,
        PERCENTILE_MIN_REFITS,
        Shift,
        refit_band,
        substitution_curve,
    )
    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import SubstitutionArtifact
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame

    from turbotab.core.methods.percent_energy import PercentEnergyShift, is_percent_of_energy

    sub = ctx.state.substitution
    n_boot = int(getattr(sub, "n_boot", 0) or 0)
    scale = getattr(sub, "scale", "kcal") or "kcal"
    percent = [c for c in (sub.donor, sub.recipient)
               if scale == "percent_energy" and is_percent_of_energy(c)]
    fit, design = ctx.inputs["fit"], ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    nested = dict(design.objects.get("nested") or {})
    task = fit.data["task"]
    for column in (sub.donor, sub.recipient):
        if column not in spec.inputs:
            raise ValueError(f"{column} is not one of the model's predictors, so energy cannot be "
                             f"moved through it.")
    readings = {c: energy_factor(c) for c in (sub.donor, sub.recipient) if c not in percent}
    for c, reading in readings.items():
        if reading.factor is None:
            raise ValueError(f"{c} carries no energy in a known unit: {reading.reason}.")
    kcal_per_unit = {c: float(r.factor) for c, r in readings.items()}

    all_train = row_ids_of(design.frames["training"])
    train_ids = all_train
    if len(train_ids) > SUBSTITUTION_ROWS:
        train_ids = np.sort(np.random.default_rng(0).choice(train_ids, SUBSTITUTION_ROWS, replace=False))
    target = ctx.state.target
    grouped_by = (fit.objects or {}).get("grouped_by")
    ctx.progress(0.02, "Reading training rows")
    extra = [target] + ([grouped_by] if grouped_by and grouped_by not in spec.inputs
                        and grouped_by != target else [])
    with open_store(ctx) as store:
        # Every row the models were fit on: the band's refits resample them all.
        fit_frame = modeling_frame(store, [*spec.inputs, *extra], all_train)
    X_fit = fit_frame[spec.inputs]
    curve_rows = X_fit.index.get_indexer(pd.Index(train_ids))
    X = X_fit.iloc[curve_rows]
    step = float(sub.step_percent) if scale == "percent_energy" else float(sub.step_kcal)
    ks = [step * i for i in range(SUBSTITUTION_STEPS + 1)]
    total_energy = _total_energy_column(ctx.state, X, (sub.donor, sub.recipient))
    if scale == "percent_energy":
        # k percent of each row's own total energy (audit B24, D19): isocaloric on every row.
        shift = PercentEnergyShift(X, donor=sub.donor, recipient=sub.recipient,
                                   kcal_per_unit=kcal_per_unit, percent=percent, nested=nested,
                                   total=total_energy)
    else:
        shift = Shift(X, donor=sub.donor, recipient=sub.recipient, kcal_per_unit=kcal_per_unit,
                      nested=nested, total=total_energy)
    from turbotab.core.units import outcome_unit as _unit_of_outcome

    outcome_unit = _unit_of_outcome(target, fit_frame[target])[0] if task == "regression" else None
    models = []
    note = None
    support = None
    skipped = []
    fitted = fit.objects["fitted"]
    pipelines = design.objects["pipelines"]
    keys = [m["family"] for m in fit.data["models"] if m["family"] in fitted]
    undrawn = ("multiclass", "ordinal", "time_to_event")  # a curve per class or level; a hazard
    drawable = [k for k in keys if task not in undrawn]
    slot = 0.93 / max(1, len(keys))  # each family's share of the progress bar, in order
    y_fit = (coded_outcome(task, fit_frame[target].to_numpy(), ctx.state.event,
                           order=ctx.state.outcome_order) if drawable else None)
    groups = fit_frame[grouped_by].to_numpy() if grouped_by else None
    # A resample repeats rows. In a refit's inner cross-validation every copy of a row (every row
    # of a unit, when rows repeat) keeps to one fold, or a penalty is tuned on rows scored by
    # their own copies (the row-id index travels with the resample).
    group_of = pd.Series(groups, index=X_fit.index) if groups is not None else None
    interval = "percentile" if n_boot >= PERCENTILE_MIN_REFITS else "normal"
    bands: list[tuple[str, dict[str, Any]]] = []
    band_seconds = 0.0
    band_failed = 0
    estimate = 0.0
    for i, key in enumerate(keys):
        if ctx.cancelled():
            raise Cancelled()
        family = get_family(key)
        start = 0.05 + slot * i
        ctx.progress(start, f"{family.label}: moving energy")
        if task in undrawn:
            skipped.append(family.label)
            continue
        curve = substitution_curve(_predictor(task, fitted[key]), X, donor=sub.donor,
                                   recipient=sub.recipient, kcal_per_unit=kcal_per_unit, ks=ks,
                                   total_kind="variable", nested=nested, total=total_energy,
                                   scale=scale, shift=shift)
        note = note or curve["note"]
        fixed = curve["fixed_population"]
        if support is None:
            support = {"total": curve["total"], "n_rows": curve["n_rows"],
                       "not_recorded": curve["n_not_recorded"], "off_amount": curve["n_off_amount"],
                       "off_share": curve["n_off_share"], "fixed_rows": fixed["n_rows"],
                       "fixed_through": fixed["through"]}
        entry = {
            "family": key, "label": family.label, "delta": curve["delta"],
            "ci_low": None, "ci_high": None,
            "on_support_fraction": curve["on_support_fraction"], "stopped_at": curve["stopped_at"],
            "effect_label": _in_outcome_unit(curve["effect_label"], outcome_unit),
            "fixed_delta": fixed["delta"], "fixed_ci_low": None, "fixed_ci_high": None,
            "band_ok": None,
        }
        if pipelines.get(key) is not None:
            def refit(Xb: pd.DataFrame, yb: Any, _pipe: Any = pipelines[key],
                      _full: Any = fitted[key]) -> Any:
                from turbotab.core.models.inner_cv import fit_pipeline

                # Every copy of a resampled row (every row of a unit) is one inner unit, both in
                # elastic net's inner folds and in boosted trees' early-stopping rows, which stop
                # early exactly when the full fit did.
                inner = (group_of.loc[Xb.index].to_numpy() if group_of is not None
                         else Xb.index.to_numpy())
                pipe = pinned_to_full_fit(clone(_pipe), _full)
                full_units = _full.steps[-1][1].get_params(deep=False).get("units")
                if isinstance(full_units, pd.Series):  # a random intercept: one per resampled unit
                    with_units(pipe, resampled_units(full_units, Xb.index))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return _predictor(task, fit_pipeline(pipe, Xb, yb, groups=inner))

            common = dict(shift=shift, ks=ks, live=curve["live"], groups=groups, random_state=0,
                          center=curve["delta"], fixed_center=fixed["delta"],
                          curve_rows=curve_rows, max_rows=BAND_ROWS, interval=interval)
            if n_boot:
                def progress(done: int, total: int, _lo: float = start + 0.1 * slot,
                             _w: float = 0.9 * slot, _label: str = family.label) -> None:
                    ctx.progress(_lo + _w * done / total, f"{_label}: refit {done} of {total}")

                band = refit_band(refit, X_fit, y_fit, n_boot=n_boot, progress=progress, **common)
                entry.update(ci_low=band["ci_low"], ci_high=band["ci_high"],
                             fixed_ci_low=band["fixed_ci_low"], fixed_ci_high=band["fixed_ci_high"],
                             band_ok=band["n_ok"])
                bands.append((family.label, band))
                band_seconds += band["seconds"]
                band_failed += band["failed"]
            else:
                ctx.progress(start + 0.5 * slot, f"{family.label}: timing one refit for the band")
                timed = refit_band(refit, X_fit, y_fit, n_boot=1, **common)
                estimate += timed["seconds"] * BAND_BOOT
        models.append(entry)
    notes = [note] if note else []
    # The energy sources the model leaves to total energy's composite (audit ME-05), on every row
    # the models were fit on: named whenever a main source is missing, or the rest is large.
    from turbotab.core.methods.energy import MAX_OMITTED_SHARE, omitted_energy, omitted_sentence

    exposures = [c for c in spec.predictors if spec.roles.get(c) == "exposure"]
    omitted = omitted_energy(X_fit, total_energy, exposures, nested=nested) if total_energy else None
    if omitted is not None and (len(omitted["omitted"]) > 1
                                or (omitted["mean_share"] or 0.0) > MAX_OMITTED_SHARE):
        said = omitted_sentence(omitted)
        if said:
            notes.append(said)
    for c, reading in readings.items():
        if not reading.declared:
            notes.append(f"{c} is read as {reading.role} in grams at {reading.factor:g} kcal/g; its "
                         f"name does not state the unit.")
    if spec.multiple_imputation() and X_fit.isna().any().any():
        # WP7: the coefficient table pools the multiple imputations; the curve does not.
        notes.append("The curve follows the training fit, whose blanks are filled once in each "
                     "fold without the outcome; it is not pooled over the multiple imputations the "
                     "coefficient table uses, so its band leaves out their uncertainty.")
    if skipped and task in ("multiclass", "ordinal"):
        kind = "An ordinal" if task == "ordinal" else "A multiclass"
        notes.append(f"{kind} outcome has one curve per level, which is not drawn yet.")
    elif skipped:
        notes.append("A time-to-event outcome's substitution is a hazard ratio, which is not drawn "
                     "yet; the Cox coefficients are log hazard ratios per unit of each column.")
    band = None
    if n_boot and bands:
        first = bands[0][1]
        caption = _band_caption(n_boot, interval, first, len(X_fit), grouped_by, bands)
        notes.append(caption.replace("Shaded bands: ", "The shaded bands are ", 1))
        band = {"n_boot": n_boot, "n_rows": int(len(X_fit)), "grouped_by": grouped_by,
                "seconds": round(band_seconds, 3), "failed": band_failed,
                "n_units": first["n_units"], "resample_units": first["resample_size"],
                "scale": first["scale"], "interval": interval, "level": first["level"],
                "min_ok_share": MIN_REFIT_SHARE, "caption": caption}
    if task == "binary" and keys:
        # the event the user named was coded 1 (``coded_outcome``); else the level sorting last
        positive = ctx.state.event if ctx.state.event is not None else fitted[keys[0]].classes_[1]
        outcome = f"the predicted probability that {target} is {positive}"
    else:
        outcome = f"predicted {target}" + (f" (in {outcome_unit})" if outcome_unit else "")
    # Whether total energy is an input the model sees is read off the design (audit WP6: the
    # label equals the model fitted); without it, only the swap's own arithmetic holds it fixed.
    energy_out = (design.data or {}).get("energy_form") in ("none", "residual_energy_dropped",
                                                            "density")
    others = ("every other input left as it was (total energy is not in this model; only the swap "
              "itself keeps it fixed)" if energy_out else
              "every other input, total energy included, left as it was")
    moved = ("k percent of each row's own total energy moves" if scale == "percent_energy"
             else "k kcal move")
    estimand = (f"The average change in {outcome} when {moved} from {sub.donor} to "
                f"{sub.recipient}, with {others}.")
    if shift.carried:
        estimand = (f"The average change in {outcome} when {moved} from {sub.donor} to "
                    f"{sub.recipient}, their parts or totals moving with them and {others}.")
    artifact = SubstitutionArtifact(
        donor=sub.donor, recipient=sub.recipient, step_kcal=float(sub.step_kcal),
        ks=[float(k) for k in ks], total_kind="variable", estimand=estimand, note=" ".join(notes),
        basis=f"Averaged over {len(X):,} training rows.", models=models,
        carried=list(shift.carried), band=band,
        band_estimate=({"n_boot": BAND_BOOT, "seconds": round(estimate, 1)}
                       if not n_boot and drawable and estimate else None),
        support=support,
        omitted_energy=omitted,
        scale=scale,
        step_percent=float(sub.step_percent) if scale == "percent_energy" else None,
    )
    ctx.progress(1.0, "Done")
    return artifact.model_dump(mode="json")


def pinned_to_full_fit(pipeline: Any, full: Any) -> Any:
    """``pipeline`` (unfitted) with its model step's size-dependent choices pinned to the full fit's.

    A band's refit must be the estimator the curve came from. scikit-learn's histogram gradient
    boosting stops early by itself only above 10,000 rows (``early_stopping="auto"``), so a refit
    on a resample of 10,000 rows of a model fit on more would boost to the end while the model it
    stands for stopped early (on noise, 10,001 rows stopped at 23 trees; 10,000 rows ran all 100).
    The full fit's resolved choice (``do_early_stopping_``) is pinned. Returns ``pipeline``.
    """
    steps = getattr(full, "steps", None)
    if not steps:
        return pipeline
    name, model = pipeline.steps[-1]
    resolved = getattr(steps[-1][1], "do_early_stopping_", None)
    if resolved is not None and model.get_params(deep=False).get("early_stopping") == "auto":
        pipeline.set_params(**{f"{name}__early_stopping": bool(resolved)})
    return pipeline


def _total_energy_column(state: Any, X: pd.DataFrame, moved: Sequence[str]) -> str | None:
    """The total-energy column the support takes shares of: the energy adjustment's, else the one
    column with the energy role; None when no numeric one is among the model's inputs."""
    adj = state.energy_adjustment
    named = [adj.energy_column] if adj is not None and adj.energy_column else []
    if not named:
        named = [c for c, role in (state.roles or {}).items() if role == "energy"]
    named = [c for c in named if c in X.columns and c not in moved
             and pd.api.types.is_numeric_dtype(X[c])]
    if len(named) != 1:
        return None
    energy = X[named[0]].to_numpy(dtype=float, na_value=np.nan)
    return named[0] if bool((np.isfinite(energy) & (energy > 0)).any()) else None


def _band_caption(n_boot: int, interval: str, first: Mapping[str, Any], n_rows: int,
                  grouped_by: str | None, bands: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    """The saved figure's caption for the band: how it was drawn, on how many rows, from how many
    refits, and how many of each family's succeeded."""
    from statistics import NormalDist

    level = f"{first['level']:.0%}"
    if interval == "normal":
        z = NormalDist().inv_cdf(0.5 + first["level"] / 2.0)
        how = (f"{level} intervals, each curve ± {z:.2f} bootstrap standard errors from {n_boot:,} "
               f"refits of its model")
    else:
        how = (f"{level} percentile intervals of {n_boot:,} refits of each model, placed around "
               f"its curve")
    n_units, m, scale = first["n_units"], first["resample_size"], first["scale"]
    if grouped_by:
        drawn = (f"whole {grouped_by} units ({n_units:,} units, {n_rows:,} training rows)"
                 if m == n_units else f"{m:,} of the {n_units:,} {grouped_by} units "
                 f"({n_rows:,} training rows)")
    else:
        drawn = (f"all {n_rows:,} training rows" if m == n_units
                 else f"{m:,} of the {n_rows:,} training rows")
    where = f"on bootstrap resamples of {drawn}"
    if m < n_units:
        where += (f", the spread rescaled by √({m:,}/{n_units:,}) = {scale:.3f} to the full sample "
                  f"(which assumes the curve's error shrinks with the square root of its sample)")
    ok = ", ".join(f"{label} {b['n_ok']:,} of {n_boot:,}" for label, b in bands)
    text = f"Shaded bands: {how}, {where}. Refits that succeeded: {ok}."
    refused = [label for label, b in bands if b["refused"]]
    if refused:
        names = refused[0] if len(refused) == 1 else ", ".join(refused[:-1]) + " and " + refused[-1]
        text += (f" No band for {names}: a band needs at least {first['min_ok_share']:.0%} of its "
                 f"refits to succeed.")
    return text


def _in_outcome_unit(label: str | None, unit: str | None) -> str | None:
    """``+1.23 per 100 kcal at k = 100`` → ``+1.23 mg/dL per 100 kcal at k = 100``."""
    if not label or not unit or " per " not in label:
        return label
    head, tail = label.split(" per ", 1)
    return f"{head}{unit} per {tail}" if unit == "%" else f"{head} {unit} per {tail}"


__all__ = ["design_stage", "fit_stage", "pinned_to_full_fit", "read_assignment", "row_ids_of",
           "shelf_stage", "substitution_pairs", "substitution_stage"]

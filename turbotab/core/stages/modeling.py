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
    n_events = n_classes = None
    outcome_mean = outcome_sd = None
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
                if task == "binary" and n_classes:
                    n_events = int(counts.min())
    situation = Situation(task=task, purpose=ctx.state.purpose, n_rows=n,
                          n_features=len(predictors), n_events=n_events, n_classes=n_classes,
                          n_parameters=predictor_parameters(predictors, column_info,
                                                            ctx.state.categorical),
                          outcome_mean=outcome_mean, outcome_sd=outcome_sd)
    ranked = rank(situation)
    events = f", {n_events:,} in the rarer class" if n_events is not None else ""
    estimates = _estimates(ctx, task, rows if trained else None, [f for f, _ in ranked])
    artifact = ShelfArtifact(
        families=[
            ShelfFamily(key=f.key, label=f.label, rank=i + 1, fit=a.fit, concerns=list(a.concerns),
                        inductive_bias=f.inductive_bias,
                        estimate_seconds=estimates[f.key].seconds if estimates.get(f.key) else None,
                        estimate=estimates[f.key].text if estimates.get(f.key) else None)
            for i, (f, a) in enumerate(ranked)
        ],
        basis=f"Ranked for {n:,} {'training ' if trained else ''}rows and {len(predictors):,} "
              f"predictors{events}.",
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


def _estimates(ctx: StageContext, task: str, train_ids: Any, families: Sequence[Any]) -> dict[str, Any]:
    """Each family's measured fit time on these training rows (``models.cost``); {} without them.

    A timing that fails is left out (the family says nothing about its cost) and never fails the
    shelf: the ranking stands without it.
    """
    from turbotab.core.models.cost import estimate_fits

    if train_ids is None or not len(train_ids):
        return {}
    split = ctx.inputs.get("split")
    folds = int((split.data or {}).get("folds") or 5) if isinstance(split, Bundle) else 5
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

    described = describe_model(adj, spec.predictors, spec.roles, list(matrix.columns), nested=nested,
                               step=shared.named_steps.get("energy"))
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
    pairs = [(f"{n}_adj", f"{n}_adj") for n in adj.nutrients]
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
                  "multiclass": "the class prior"}


def baseline_model(task: str) -> Any:
    """What predicts without the predictors: the training mean, or the training class prior."""
    from sklearn.dummy import DummyClassifier, DummyRegressor

    return DummyRegressor(strategy="mean") if task == "regression" else DummyClassifier(strategy="prior")


def _baseline_cv(task: str, X: Any, y: Any, pairs: Sequence[Any]) -> Any:
    """The baseline cross-validated on the same fold pairs, scored as the models are."""
    from turbotab.core.models.metrics import cross_validate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_validate(task, lambda: baseline_model(task), X, y, pairs)


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


def baseline_concern(task: str, metric_label: str, model: float | None, base: float | None) -> str | None:
    """A plain sentence when a family scores worse than its baseline on the primary metric."""
    if model is None or base is None or not np.isfinite(model) or not np.isfinite(base) or model >= base:
        return None
    m, b = _two(model, base)
    if task == "regression":
        return f"Predicts worse than the outcome's average: CV {metric_label} {m}, against {b} for the average."
    if task == "binary":
        return f"Separates the classes worse than the class prior: CV {metric_label} {m}, against {b} for the prior."
    return (f"Classifies worse than always guessing the most common class: CV {metric_label} {m}, "
            f"against {b}.")


def coded_outcome(task: str | None, y: Any, event: str | None) -> Any:
    """A binary outcome coded 1 for the level the user named as the event (``set_event``), else 0.

    The event is never guessed (M2_CONTRACT §1), and its methods sentence says that level was
    coded 1; the models, their metrics and coefficients must then be about that level, not about
    whichever level sorts last. Unchanged when there is no event or it is not a level here.
    """
    if task != "binary" or event is None:
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


def _inference_table(family: Any, pipeline: Any, X: Any, y: Any, *, task: str, clusters: Any,
                     outcome: Any, rows: str) -> Any:
    """``family.inference``, handing it the outcome's names and the rows it is estimated from when
    it takes them (a family registered before WP8 may not)."""
    import inspect

    accepts = inspect.signature(family.inference).parameters
    extra = {name: value for name, value in (("outcome", outcome), ("rows", rows)) if name in accepts}
    return family.inference(pipeline, X, y, task=task, clusters=clusters, **extra)


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
    """
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import FitArtifact
    from turbotab.core.models.base import reports_coefficients
    from turbotab.core.models.baseline import compare, no_better_concern
    from turbotab.core.models.inference import Outcome, cluster_columns, resolve_clusters
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import (CV_DEFINITION, LABELS, PRIMARY, cross_validate,
                                              fold_pairs, metric_labels, score)
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.models.selection import OutOfFold, selection_optimism
    from turbotab.core.seal import SEALED_SCORES, sealed_scores_frame

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
    inference = state.purpose == "inference"
    with open_store(ctx) as store:
        # Under inference the intervals cluster by whatever identifier repeats, however the seal
        # was drawn (AUDIT_REPORT MA-01): read every column that may name the unit.
        unit_columns = cluster_columns(state, store.columns, [grouped_by]) if inference else []
        columns += [c for c in unit_columns if c not in columns]
        frame = modeling_frame(store, columns, assignment.index.to_numpy())
    train = assignment["train"].to_numpy()
    y_all = frame[target]
    if y_all.isna().any():
        raise ValueError(f"{int(y_all.isna().sum()):,} analysis rows have no {target}; the cohort "
                         f"should have left them out.")
    outcome = Outcome(name=target, labels=outcome_levels(task, y_all.to_numpy(), state.event))
    y_all = pd.Series(coded_outcome(task, y_all.to_numpy(), state.event), index=y_all.index)
    X, y = frame.loc[train, spec.inputs], y_all[train].to_numpy()
    X_hold, y_hold = frame.loc[~train, spec.inputs], y_all[~train].to_numpy()
    folds = assignment.loc[train, "fold"].to_numpy().astype(int)
    order_all = assignment["order"].to_numpy(dtype=float) if scheme == "time_ordered" else None
    order = order_all[train] if order_all is not None else None
    pairs = fold_pairs(folds, scheme)
    unit_all = frame[grouped_by].to_numpy() if grouped_by else None
    groups = unit_all[train] if unit_all is not None else None
    # The rows the coefficient table is estimated from: every analyzed row under inference
    # (BLUEPRINT §12 ruling 3: a holdout is a prediction concept), the training rows otherwise.
    table_rows = np.ones(len(train), dtype=bool) if inference else train
    same_rows = bool(table_rows.sum() == train.sum())
    X_tab, y_tab = frame.loc[table_rows, spec.inputs], y_all[table_rows].to_numpy()
    clusters = (resolve_clusters(state, frame.loc[table_rows, unit_columns], [grouped_by])
                if inference else None)
    if groups is not None and len(pd.unique(groups)) == len(groups):
        # One row per unit (the rows were combined per unit, M2_CONTRACT §2): the split is keyed by
        # the unit, but no unit repeats, so clustering by it changes nothing and "its rows repeat"
        # would be false.
        groups, grouped_by, unit_all = None, None, None

    def fit(model: Any, X_fit: Any, y_fit: Any, rows: Any = None) -> Any:
        """Fit a pipeline on these training rows, its inner splits drawn as the folds are."""
        take = slice(None) if rows is None else rows
        return fit_pipeline(model, X_fit, y_fit, groups=None if groups is None else groups[take],
                            order=None if order is None else order[take])

    def fit_table(model: Any) -> Any:
        """Fit a pipeline on the coefficient table's rows, its inner splits drawn as the folds are."""
        return fit_pipeline(model, X_tab, y_tab,
                            groups=None if unit_all is None else unit_all[table_rows],
                            order=None if order_all is None else order_all[table_rows])

    keys = [k for k in (state.models or []) if k in pipelines]
    primary = PRIMARY[task]
    base = _baseline_cv(task, X, y, pairs)
    base_cv = base.summary(task)
    baseline = {"metric": primary, "value": base_cv[primary]["estimate"], "label": BASELINE_LABEL[task]}
    reference = float(np.mean(y.astype(float))) if task == "regression" and len(y) else None
    units = max(1, len(keys) * (len(pairs) + 2))
    done = 0

    def share(n: int) -> float:
        return 0.02 + 0.97 * n / units

    models: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    sealed: dict[str, Any] = {}  # held-out scores: kept out of the public data (M2_CONTRACT §3)
    results: dict[str, Any] = {}  # each family's cross-validation, for the selection's optimism
    oof = OutOfFold(task, X, y, pairs)  # …and its out-of-fold predictions (models/selection.py)
    for key in keys:
        family = get_family(key)
        started = time.perf_counter()

        def before_fold(i: int, n: int, _label: str = family.label) -> None:
            nonlocal done
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{_label}: fold {i + 1} of {n}")
            done += 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = cross_validate(task, lambda _k=key: clone(pipelines[_k]), X, y, pairs,
                                    fit=oof.wrap(key, fit) if len(keys) > 1 else fit,
                                    before_fold=before_fold)
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{family.label}: refitting on all training rows")
            final = fit(clone(pipelines[key]), X, y)
            done += 1
            # R² on the held-out rows is measured against the training rows' mean.
            holdout = score(task, final, X_hold, y_hold, reference=reference) if len(y_hold) else None
            ctx.progress(share(done), f"{family.label}: coefficients")
            concerns: list[str] = []
            interval_info = None
            n_coefficients = None
            on_all = inference and reports_coefficients(family)
            try:
                if on_all and not same_rows:
                    ctx.progress(share(done), f"{family.label}: coefficients on every analyzed row")
                table_fit = (final if same_rows else fit_table(clone(pipelines[key]))) if on_all else final
                X_c, y_c = (X_tab, y_tab) if on_all else (X, y)
                if clusters is not None and hasattr(family, "inference"):
                    table = _inference_table(family, table_fit, X_c, y_c, task=task,
                                             clusters=clusters, outcome=outcome,
                                             rows="all" if on_all else "training")
                    coefficients, interval_info = table.rows, table.info
                    concerns.extend(table.concerns)
                else:
                    unit_c = (None if unit_all is None else unit_all[table_rows]) if on_all else groups
                    coefficients = family.coefficients(table_fit, X_c, y_c, task=task,
                                                       purpose=state.purpose, groups=unit_c)
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
            if coefficients:
                # The relative effects come from the fit the table came from: under inference
                # every analyzed row, with the table's clusters and outcome scale.
                coefficients = _energy_rows(coefficients, design, spec, family, table_fit, X_c, y_c,
                                            task, clusters, outcome)
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
        cv, versus = compare(task, result, base)
        estimate = cv[primary]["estimate"]
        worse = baseline_concern(task, LABELS[primary], estimate, baseline["value"])
        tie = None if worse else no_better_concern(task, LABELS[primary], versus, estimate,
                                                   baseline["value"], _two)
        if worse or tie:
            concerns.insert(0, worse or tie)
        fitted[key] = final
        sealed[key] = holdout
        results[key] = result
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
        })
    # Picking the best of several families by cross-validation flatters it (audit ME-13).
    if len(results) > 1:
        ctx.progress(0.995, "Choosing among the families: bootstrapping the out-of-fold predictions")
    selection = selection_optimism(task, primary, results, oof,
                                   {m["family"]: m["label"] for m in models}, LABELS[primary])
    ctx.progress(1.0, "Done")
    n_holdout = int((~train).sum())
    artifact = FitArtifact(task=task, primary_metric=PRIMARY[task], metric_labels=metric_labels(task),
                           n_train=int(train.sum()), n_holdout=n_holdout, models=models,
                           holdout_sealed=n_holdout > 0, fold_scheme=scheme,
                           cv_definition=CV_DEFINITION[task], selection=selection)
    frames = {SEALED_SCORES: sealed_scores_frame(models, sealed)} if n_holdout else {}
    return Bundle(data=artifact.model_dump(mode="json"), frames=frames,
                  objects={"fitted": fitted, "grouped_by": grouped_by})


def _energy_rows(coefficients: list[dict[str, Any]], design: Any, spec: Any, family: Any,
                 final: Any, X: Any, y: Any, task: str, clusters: Any,
                 outcome: Any = None) -> list[dict[str, Any]]:
    """The coefficient rows with what each means (the design's ``terms``), and under the
    all-components model each nutrient's average relative effect beside them (audit ME-14):
    with the family's own intervals under inference, as a point estimate otherwise.

    ``final``, ``X`` and ``y`` are the fit and rows the coefficient table was estimated from
    (every analyzed row under inference, WP8), so the relative effects share its rows, clusters
    and scale (odds or relative-risk ratios, named by ``outcome``)."""
    terms = ((design.data or {}).get("terms") or {}) if isinstance(design, Bundle) else {}
    rows = [{**row, "meaning": terms[row["feature"]]} if row.get("feature") in terms else row
            for row in coefficients]
    adj = spec.energy_adjustment()
    if (adj is None or adj.method != "all_components" or task == "multiclass"
            or not hasattr(family, "inference")):
        return rows
    from turbotab.core.methods.energy import relative_effect_rows
    from turbotab.core.models.inference import inference_table
    from turbotab.core.models.linear import model_matrix

    matrix = model_matrix(final, X)
    step = getattr(final, "named_steps", {}).get("energy")
    factors = dict(getattr(getattr(step, "pooled_", step), "factors_", {}) or {})
    table = None
    if clusters is not None:
        classes = list(getattr(final[-1], "classes_", [])) or None

        def table(m: Any) -> Any:
            return inference_table(task, m, y, classes, clusters, outcome=outcome).rows
    try:
        extra = relative_effect_rows(matrix, list(adj.nutrients), factors, table=table,
                                     coefficients=rows)
    except Exception:  # noqa: BLE001 - the coefficients stand without their contrasts
        import logging

        logging.getLogger(__name__).exception("the average relative effects failed")
        extra = []
    return rows + extra


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

    sub = ctx.state.substitution
    n_boot = int(getattr(sub, "n_boot", 0) or 0)
    fit, design = ctx.inputs["fit"], ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    nested = dict(design.objects.get("nested") or {})
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
    ks = [sub.step_kcal * i for i in range(SUBSTITUTION_STEPS + 1)]
    total_energy = _total_energy_column(ctx.state, X, (sub.donor, sub.recipient))
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
    drawable = [k for k in keys if task != "multiclass"]
    slot = 0.93 / max(1, len(keys))  # each family's share of the progress bar, in order
    y_fit = coded_outcome(task, fit_frame[target].to_numpy(), ctx.state.event)
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
        if task == "multiclass":
            skipped.append(family.label)
            continue
        curve = substitution_curve(_predictor(task, fitted[key]), X, donor=sub.donor,
                                   recipient=sub.recipient, kcal_per_unit=kcal_per_unit, ks=ks,
                                   total_kind="variable", nested=nested, total=total_energy)
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
    if skipped:
        notes.append("A multiclass outcome has one curve per class, which is not drawn yet.")
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
    estimand = (f"The average change in {outcome} when k kcal move from {sub.donor} to "
                f"{sub.recipient}, with {others}.")
    if shift.carried:
        estimand = (f"The average change in {outcome} when k kcal move from {sub.donor} to "
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

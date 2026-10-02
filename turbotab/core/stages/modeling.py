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
    if task != "regression" and rows is not None:
        with open_store(ctx) as store:
            y = store.materialize([ctx.state.target], rows)
        counts = y[ctx.state.target].value_counts(dropna=True)
        n_classes = int(len(counts))
        if task == "binary" and n_classes:
            n_events = int(counts.min())
    situation = Situation(task=task, purpose=ctx.state.purpose, n_rows=n,
                          n_features=len(predictors), n_events=n_events, n_classes=n_classes)
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
    ctx.progress(0.5, "Timing one fit of each family on a sample of the training rows")
    try:
        with open_store(ctx) as store:
            return estimate_fits(store, ctx.state, task, train_ids, families, folds,
                                 cancelled=ctx.cancelled)
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
    from turbotab.core.methods.energy import METHOD_TABLE
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
    with open_store(ctx) as store:
        X = modeling_frame(store, input_columns(predictors, adj), train_ids)
        info = {c.name: c for c in store.info().columns}  # every row's summary: as the cohort reads it
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
        substitution_pairs=substitution_pairs(spec.predictors, energy_column, nested),
        warnings=warnings_list,
        nested=[{"column": c, "parent": p} for c, p in nested.items()],
        left_out=[c for c in left_out(state) if c in (state.roles or {})],
    )
    return Bundle(
        data=artifact.model_dump(mode="json"),
        frames={"training": pd.DataFrame({"row_id": train_ids.astype(np.int64)})},
        objects={"pipelines": pipelines, "spec": spec.to_dict(), "nested": nested},
    )


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


def _baseline_scores(task: str, X: Any, y: Any, folds: Any, fold_keys: Sequence[int]) -> list[dict[str, float]]:
    from turbotab.core.models.metrics import score

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in fold_keys:
            fit_rows, test_rows = folds != k, folds == k
            dummy = baseline_model(task).fit(X[fit_rows], y[fit_rows])
            out.append(score(task, dummy, X[test_rows], y[test_rows]))
    return out


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


def fit_stage(ctx: StageContext) -> Bundle:
    """Cross-validate each pipeline on the split's folds, then refit on all training rows.

    Each model is set beside its baseline on the same folds (the outcome's training-fold mean, or
    the class prior), and a model that scores worse than it says so in its first concern.
    """
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import FitArtifact
    from turbotab.core.models.baseline import no_better_concern, versus_baseline
    from turbotab.core.models.inner_cv import with_grouped_inner_cv
    from turbotab.core.models.metrics import LABELS, PRIMARY, metric_labels, score, summarize
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.seal import SEALED_SCORES, sealed_scores_frame

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
    y_all = pd.Series(coded_outcome(task, y_all.to_numpy(), state.event), index=y_all.index)
    X, y = frame.loc[train, spec.inputs], y_all[train].to_numpy()
    X_hold, y_hold = frame.loc[~train, spec.inputs], y_all[~train].to_numpy()
    folds = assignment.loc[train, "fold"].to_numpy().astype(int)
    fold_keys = sorted(set(folds.tolist()))
    groups = frame.loc[train, grouped_by].to_numpy() if grouped_by else None
    if groups is not None and len(pd.unique(groups)) == len(groups):
        # One row per unit (the rows were combined per unit, M2_CONTRACT §2): the split is keyed by
        # the unit, but no unit repeats, so clustering by it changes nothing and "its rows repeat"
        # would be false.
        groups, grouped_by = None, None

    keys = [k for k in (state.models or []) if k in pipelines]
    primary = PRIMARY[task]
    base_per_fold = _baseline_scores(task, X, y, folds, fold_keys)
    base_cv = summarize(task, base_per_fold)
    baseline = {"metric": primary, "value": base_cv[primary]["mean"], "label": BASELINE_LABEL[task]}
    units = max(1, len(keys) * (len(fold_keys) + 2))
    done = 0

    def share(n: int) -> float:
        return 0.02 + 0.97 * n / units

    models: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    sealed: dict[str, Any] = {}  # held-out scores: kept out of the public data (M2_CONTRACT §3)
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
                # An inner cross-validation (elastic net's penalty) is grouped as the split is.
                model = with_grouped_inner_cv(clone(pipelines[key]),
                                              None if groups is None else groups[fit_rows])
                model = model.fit(X[fit_rows], y[fit_rows])
                per_fold.append(score(task, model, X[test_rows], y[test_rows]))
                done += 1
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{family.label}: refitting on all training rows")
            final = with_grouped_inner_cv(clone(pipelines[key]), groups).fit(X, y)
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
        if family.key == "linear":
            from turbotab.core.models.linear import collinearity_concern, model_matrix

            try:
                singular = collinearity_concern(model_matrix(final, X))
            except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
                singular = None
            if singular and not any("singular" in c for c in concerns):
                concerns.insert(0, singular)
        cv = summarize(task, per_fold)
        versus = versus_baseline(primary, [f[primary] for f in per_fold],
                                 [b[primary] for b in base_per_fold])
        worse = baseline_concern(task, LABELS[primary], cv[primary]["mean"], baseline["value"])
        tie = None if worse else no_better_concern(task, LABELS[primary], versus,
                                                   cv[primary]["mean"], baseline["value"], _two)
        if worse or tie:
            concerns.insert(0, worse or tie)
        if coefficients is not None and groups is not None and state.purpose == "inference" \
                and family.key == "linear":
            concerns.append(f"Confidence intervals are cluster-robust by {grouped_by}, because its "
                            f"rows repeat.")
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
        })
    ctx.progress(1.0, "Done")
    n_holdout = int((~train).sum())
    artifact = FitArtifact(task=task, primary_metric=PRIMARY[task], metric_labels=metric_labels(task),
                           n_train=int(train.sum()), n_holdout=n_holdout, models=models,
                           holdout_sealed=n_holdout > 0)
    frames = {SEALED_SCORES: sealed_scores_frame(models, sealed)} if n_holdout else {}
    return Bundle(data=artifact.model_dump(mode="json"), frames=frames,
                  objects={"fitted": fitted, "grouped_by": grouped_by})


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
                from turbotab.core.models.inner_cv import with_grouped_inner_cv

                inner = (group_of.loc[Xb.index].to_numpy() if group_of is not None
                         else Xb.index.to_numpy())
                pipe = with_grouped_inner_cv(pinned_to_full_fit(clone(_pipe), _full), inner)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return _predictor(task, pipe.fit(Xb, yb))

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
    estimand = (f"The average change in {outcome} when k kcal move from {sub.donor} to "
                f"{sub.recipient}, with every other input, total energy included, left as it was.")
    if shift.carried:
        estimand = (f"The average change in {outcome} when k kcal move from {sub.donor} to "
                    f"{sub.recipient}, their parts or totals moving with them and every other "
                    f"input, total energy included, left as it was.")
    artifact = SubstitutionArtifact(
        donor=sub.donor, recipient=sub.recipient, step_kcal=float(sub.step_kcal),
        ks=[float(k) for k in ks], total_kind="variable", estimand=estimand, note=" ".join(notes),
        basis=f"Averaged over {len(X):,} training rows.", models=models,
        carried=list(shift.carried), band=band,
        band_estimate=({"n_boot": BAND_BOOT, "seconds": round(estimate, 1)}
                       if not n_boot and drawable and estimate else None),
        support=support,
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

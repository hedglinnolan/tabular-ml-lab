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
    from turbotab.core.readings import settled_roles

    cohort = ctx.inputs["cohort"]
    data = cohort.data if isinstance(cohort, Bundle) else cohort
    task = _task(ctx)
    # The settled roles only (BLUEPRINT §14.1), as the cohort counts them.
    predictors = list(data.get("predictors")
                      or predictors_from_roles(settled_roles(ctx.state), ctx.state.target))
    n = int(data["n_final"])
    rows = row_ids_of(cohort.frames["rows"]) if isinstance(cohort, Bundle) and "rows" in cohort.frames else None
    # Under prediction the shelf informs a modeling choice made after the seal, so it reads the
    # training rows only (M2_CONTRACT §3, the held-out discipline audit): never the held-out rows'
    # outcomes. Under inference the seal is a prediction concept (BLUEPRINT §12 ruling 3): the
    # coefficient table is estimated from every analyzed row, so its sample-size criteria and the
    # basis line count those rows (the methods gate: "Ranked for 2,400 training rows … 526 with the
    # event" beside a table from 3,000).
    split = ctx.inputs.get("split")
    inference = ctx.state.purpose == "inference"
    trained = rows is not None and split is not None
    train_rows = rows
    if trained:
        assignment = read_assignment(split)
        train_rows = np.intersect1d(rows, assignment.index[assignment["train"]].to_numpy())
        if inference:
            rows = np.intersect1d(rows, assignment.index.to_numpy())
        else:
            rows = train_rows
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
    from turbotab.core.readings import confirmed_codes

    # A spline or quintile exposure puts several columns in the model (WP12a); a sample-size
    # criterion counts them as parameters too (WP8).
    terms = model_terms(predictors, ctx.state.exposure_forms)
    situation = Situation(task=task, purpose=ctx.state.purpose, n_rows=n,
                          n_features=terms, n_events=n_events, n_classes=n_classes,
                          n_parameters=predictor_parameters(predictors, column_info,
                                                            confirmed_codes(ctx.state))
                          + (terms - len(predictors)),
                          outcome_mean=outcome_mean, outcome_sd=outcome_sd,
                          lenses=tuple(ctx.state.lens or ()), class_counts=class_counts,
                          n_units=n_units)
    ranked = rank(situation)
    survey = getattr(ctx.state, "survey", None)
    if inference and survey is not None and survey.estimand == "population":
        # MS4: under the population answer a family with no design-based estimator ranks after
        # every family that has one, and says why before it is chosen (MODELING_SEQUENCE §4).
        from turbotab.core.models.survey import population_shelf

        ranked = population_shelf(ranked, task)
    # WP11: raw counts or intensities whose totals track the outcome, on the same rows (the
    # training rows under prediction, every analyzed row under inference)
    assay = _assay_concern(ctx, task, rows if trained else None)
    events = ("" if n_events is None else f", {n_events:,} with the event" if task == "time_to_event"
              else f", {n_events:,} in the rarer class")
    # Each family's fit time is measured on the training rows, which cross-validation fits on.
    estimates = _estimates(ctx, task, train_rows if trained else None, [f for f, _ in ranked])
    rows_word = ("analyzed " if inference else "training ") if trained else ""
    artifact = ShelfArtifact(
        families=[
            ShelfFamily(key=f.key, label=f.label, rank=i + 1, fit=a.fit,
                        concerns=([assay] if assay else []) + list(a.concerns),
                        inductive_bias=f.inductive_bias,
                        estimate_seconds=estimates[f.key].seconds if estimates.get(f.key) else None,
                        estimate=estimates[f.key].text if estimates.get(f.key) else None)
            for i, (f, a) in enumerate(ranked)
        ],
        basis=f"Ranked for {n:,} {rows_word}rows and {len(predictors):,} "
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


def _assay_concern(ctx: StageContext, task: str, row_ids: Any) -> str | None:
    """The library-size check (``methods.omics.check_on``) on the shelf's rows: a sentence when the
    exposures are raw counts or intensities, no normalization is recorded, and their per-sample
    totals track the outcome. Under prediction those are the training rows and it never reads a
    held-out row; under inference every analyzed row, as the coefficient table (ruling 3)."""
    from turbotab.core.methods.omics import OMICS_LENSES, check_on

    state = ctx.state
    if row_ids is None or not len(row_ids) or not state.target:
        return None
    if not any(k in OMICS_LENSES for k in state.lens or []):
        return None
    exposures = [c for c, r in (state.roles or {}).items() if r == "exposure" and c != state.target]
    if not exposures:
        return None
    with open_store(ctx) as store:
        present = [c for c in exposures if c in set(store.columns)]
        frame = store.materialize([*present, state.target], row_ids)
    frame = frame.loc[frame[state.target].notna()]
    y = coded_outcome(task, frame[state.target].to_numpy(), state.event)
    rows = "analyzed rows" if state.purpose == "inference" else "training rows"
    check = check_on(frame[present], state, y, task, rows)
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
    # MS6: under prediction the comparisons' repeated k-fold (≥ 10 × K) is counted too, before the
    # fit runs, with the split's own repeats as its first ones.
    from turbotab.core.models.folds import COMPARISON_REPEATS

    k = int(getattr(ctx.state.split, "folds", None) or data.get("folds") or 5)
    repeats = int(data.get("repeats") or 1)
    if ctx.state.purpose != "inference" and data.get("fold_scheme") != "time_ordered":
        own = int(data.get("folds") or k) * repeats if data.get("validation") == "internal_external" else 0
        folds = k * max(repeats, COMPARISON_REPEATS) + own + int(data.get("n_boot") or 0)
    else:
        folds = int(data.get("folds") or 5) * repeats + int(data.get("n_boot") or 0)
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


def shift_for(state: Any, X: pd.DataFrame, *, donor: str, recipient: str,
              kcal_per_unit: Mapping[str, float], design_nested: Mapping[str, str] | None,
              total: str | None, scale: str = "kcal", percent: Sequence[str] = ()) -> Any:
    """What moving energy from ``donor`` to ``recipient`` does to each row of ``X``
    (``methods.substitution.Shift``; k percent of each row's own energy on the share-of-energy
    scale), the parts of totals moving with them as the ledger holds the nesting
    (``readings.nesting``: the design's guess, each column's own confirmation standing over it,
    whichever column of the table it names, or none)."""
    from turbotab.core.methods.percent_energy import PercentEnergyShift
    from turbotab.core.methods.substitution import Shift
    from turbotab.core.readings import nesting

    nested = nesting(state, dict(design_nested or {}), columns=list(X.columns))
    if scale == "percent_energy":
        # k percent of each row's own total energy (audit B24, D19): isocaloric on every row.
        return PercentEnergyShift(X, donor=donor, recipient=recipient,
                                  kcal_per_unit=kcal_per_unit, percent=list(percent),
                                  nested=nested, total=total)
    return Shift(X, donor=donor, recipient=recipient, kcal_per_unit=kcal_per_unit, nested=nested,
                 total=total)


def design_nesting(state: Any, X: pd.DataFrame, numeric: Sequence[str]) -> dict[str, str]:
    """The parts of totals among the design's numeric predictors, on its rows: the names and the
    values (a part never above its total) propose them, and each column's own confirmation stands
    over the guess, naming any column of the table or none (``readings.nesting``)."""
    from turbotab.core.readings import nesting

    return nesting(state, frame=X, columns=numeric)


def substitution_pairs(predictors: Sequence[str], energy_column: str | Sequence[str] | None,
                       nested: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    """Ordered (donor, recipient) pairs of predictors that carry energy in a known unit, or as a
    share of energy (``fat_pct_kcal``: audit B24). ``energy_column``: the total-energy column or
    columns, never a donor or a recipient.

    A total is never paired with its own part (``nested``: child -> parent): kcal moved between
    them go nowhere.
    """
    from turbotab.core.methods.energy import energy_factor
    from turbotab.core.methods.percent_energy import is_percent_of_energy

    nested = nested or {}
    totals = {energy_column} if isinstance(energy_column, str) else set(energy_column or ())
    bearing = [c for c in predictors if c not in totals
               and (energy_factor(c).factor is not None or is_percent_of_energy(c))]
    return [{"donor": d, "recipient": r} for d, r in permutations(bearing, 2)
            if nested.get(d) != r and nested.get(r) != d][:MAX_PAIRS]


def design_stage(ctx: StageContext) -> Bundle:
    """Each chosen family's pipeline, and the lineage from raw columns to the model matrix."""
    from turbotab.core.decisions import left_out
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
    # BLUEPRINT §14.1: the fit is a number-changing consumer of every role and of each
    # whole-number predictor's code-or-amount reading; while one is unsettled it asks (the
    # ``select_models`` refusal carries one exit per reading) and fits nothing.
    from turbotab.core.decisions import left_out
    from turbotab.core.readings import predictors_or_ask

    with open_store(ctx) as store:
        summaries = {c.name: {"dtype": c.dtype, "n_unique": c.n_unique}
                     for c in store.info().columns}
        predictors_or_ask(state, summaries, drop=left_out(state), store=store)
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
    # The rows the design describes: under inference every analyzed row, held-out ones included,
    # because the coefficient table beside it is estimated from all of them (BLUEPRINT §12 ruling
    # 3; the methods gate: the log residual's printed elasticity, the residual gap and the energy
    # step's warnings were the training rows' while the table was every row's). Under prediction
    # the training rows, where everything the held-out score depends on is learned. The pipelines
    # themselves are unfitted templates either way; the fit stage fits them on the rows each use
    # needs (cross-validation on training folds, the table on every analyzed row).
    inference = state.purpose == "inference"
    design_ids = assignment.index.to_numpy() if inference else train_ids
    rows_word = "analyzed rows" if inference else "training rows"
    adj = state.energy_adjustment
    ctx.progress(0.05, f"Reading the {rows_word}")
    y_rows = None
    from turbotab.core.methods.batch import batch_inputs
    from turbotab.core.methods.batch import design_refusal as batch_refusal

    with open_store(ctx) as store:
        # MS7: the batch column a reference ComBat step reads, whether or not a model sees it.
        X = modeling_frame(store, list(dict.fromkeys([*input_columns(predictors, adj),
                                                      *batch_inputs(state)])), design_ids)
        info = {c.name: c for c in store.info().columns}  # every row's summary: as the cohort reads it
        if adj is not None and adj.method == "residual_energy_dropped" and state.target in store.columns:
            # The gap the energy-dropped residual opens on these rows (audit ME-03).
            y_rows = store.materialize([state.target], design_ids)[state.target]
        # MS7: a batch perfectly confounded with the outcome is refused under both purposes.
        confounded = batch_refusal(state, store, design_ids, task)
        # The partition methods convert each energy source by its kcal per unit as the readings
        # ledger settled it over every row (a recorded unit, or grams by the Atwater test), never
        # by its name (BLUEPRINT §14.3; ``readings.predictors_or_ask`` asked any still unsettled).
        factors = None
        from turbotab.core.methods.energy import PARTITION_METHODS

        if adj is not None and adj.method in PARTITION_METHODS:
            from turbotab.core.readings import energy_source_factors

            factors = energy_source_factors(state, list(adj.nutrients), store=store)
    if confounded:
        raise ValueError(confounded)
    from turbotab.core.methods.qc_drift import working_qc_sd

    spec = design_spec(state, X, predictors, column_info=info,
                       qc=working_qc_sd(ctx.inputs.get("working")), energy_factors=factors)
    from turbotab.core.methods.omics import design_refusal

    refused = design_refusal(state, X, families)  # WP11: raw omics values into a linear family
    if refused:
        raise ValueError(refused)
    numeric_set = set(spec.numeric)  # a set: 20,000 predictors made the list test quadratic
    numeric = [c for c in spec.predictors if c in numeric_set]
    nested = design_nesting(state, X, numeric)  # on the design's rows: what the substitution moves
    warnings_list = warnings_for(spec, X, [f.key for f in families], {f.key: f for f in families},
                                 nested, rows_word=rows_word)

    ctx.progress(0.3, f"Fitting the shared steps on the {rows_word}")
    shared = transformer(shared_steps(spec))
    matrix = shared.fit_transform(X[spec.inputs])
    lineage = trace(shared.steps, spec.inputs, spec.roles, missing_counts(X[spec.inputs]))
    if "energy" in shared.named_steps:
        warnings_list.extend(_energy_warnings(shared.named_steps["energy"], rows_word))
    # The estimand and each coefficient's meaning, read off the matrix the models will see: the
    # label equals the model fitted (audit WP6; methods.energy.describe_model).
    from turbotab.core.methods.energy import describe_model, total_energy_columns
    from turbotab.core.methods.exposure_form import form_columns, form_step, formed_meanings

    # A formed exposure (WP12a) is read as the term it was before its spline or quintiles, and
    # its meaning then carried to the columns the form made of it.
    form = form_step(shared)
    formed = form_columns(form) if form is not None else {}
    seen = list(matrix.columns) + [c for c in formed if c not in set(matrix.columns)]
    described = describe_model(adj, spec.predictors, spec.roles, seen, nested=nested,
                               step=shared.named_steps.get("energy"))
    described.terms = formed_meanings(described.terms, form)
    seen_set = set(seen)
    for c in total_energy_columns(spec.predictors, spec.roles):
        role = spec.roles.get(c)
        if role != "energy" and c in seen_set and described.text:
            # Audit ME-02 (repair round): total energy kept as a covariate is the standard model,
            # whatever the energy question says; the estimand above is read off this matrix.
            warnings_list.append(
                f"{c} reads as total energy and is in the model as {'an' if role == 'exposure' else 'a'} "
                f"{role}, so each nutrient's coefficient is at fixed total energy (the standard "
                f"model), not an absolute intake. Give {c} the energy role to choose the energy "
                f"model, or leave it out for the unadjusted one.")
    if y_rows is not None:
        warnings_list.extend(_residual_gap(state, task, spec, X, matrix, y_rows, info, rows_word))

    ctx.progress(0.8, "Building each model's pipeline")
    n_rows, n_cols = int(matrix.shape[0]), int(matrix.shape[1])
    # A family sizes its own choices (elastic net's inner folds) for the rows cross-validation
    # trains on, under either purpose.
    n_train = int(len(train_ids))
    pipelines = {f.key: build_pipeline(spec, f, task, state.purpose, n_train, n_cols) for f in families}
    models = [{"family": f.key, "label": f.label,
               "steps": describe_steps(spec, f, task, state.purpose, n_cols)} for f in families]
    totals = [*([adj.energy_column] if adj is not None and adj.energy_column else []),
              *total_energy_columns(spec.predictors, spec.roles)]
    artifact = DesignArtifact(
        lineage=lineage,
        matrix={"n_rows": n_rows, "n_cols": n_cols},
        models=models,
        estimand=described.text,
        substitution_pairs=substitution_pairs(spec.predictors, totals, nested),
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
                  y: Any, info: Mapping[str, Any], rows_word: str = "training rows") -> list[str]:
    """The energy-dropped residual against the standard model, on the design's rows (ME-03):
    every analyzed row under inference, as the coefficient table beside it, else the training rows.

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
        out.append(f"{E} left the outcome model: on {g['n_rows']:,} {rows_word} "
                   f"{g['nutrient']}'s coefficient is {_coef(g['dropped'])}, against "
                   f"{_coef(g['standard'])} with {E} kept (the standard model's). The two agree "
                   f"only when no covariate correlates with {E}.")
    return out


def _energy_warnings(step: Any, rows_word: str = "training rows") -> list[str]:
    """The energy step's plain statements, on the rows it was fit on (``rows_word``)."""
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
            out.append(f"kcal_from_other is slightly negative on {n:,} of {of:,} {rows_word} "
                       f"(median {typical} of total energy): the general Atwater factors (4, 4, 9 "
                       f"kcal/g) run a little above this table's own, and everything else absorbs "
                       f"the difference.")
        else:
            row = rows_word if n != 1 else rows_word[:-1]
            out.append(f"{n:,} {row} get{'s' if n == 1 else ''} a negative kcal_from_other: "
                       f"the chosen nutrients carry more energy than the recorded total.")
    for n, p in getattr(adjuster, "params_", {}).items():
        if n != "__other__" and isinstance(p, Mapping) and p.get("r2") is not None and p["r2"] < 0.05:
            out.append(f"Energy explains {p['r2']:.0%} of {n}'s variation, so adjusting it barely "
                       f"changes it.")
    # Within strata: the levels too small for a slope of their own (StratifiedEnergyAdjuster).
    out.extend(step.notes(rows_word) if hasattr(step, "notes") else [])
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
                 repeat_of: Sequence[int] | None = None, horizon: float | None = None) -> Any:
    """The baseline cross-validated on the same fold pairs, scored as the models are."""
    from turbotab.core.models.metrics import cross_validate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_validate(task, lambda: baseline_model(task), X, y, pairs, repeat_of=repeat_of,
                              horizon=horizon)


def follow_up_horizon(state: Any, y: Any) -> tuple[float | None, str | None]:
    """A time to event's prediction horizon (MS6): the one declared with the follow-up, else the
    median follow-up time of these rows (the rows the models learn from, after any end of
    follow-up the follow-up answer set), stated as such. It changes no row: the follow-up's own
    ``horizon`` is where follow-up ends (the routing gate), this one where predictions are judged."""
    spec = getattr(state, "follow_up", None)
    declared = getattr(spec, "prediction_horizon", None) if spec is not None else None
    column = getattr(spec, "time_column", None) if spec is not None else None
    named = f"`{column}`" if column else "follow-up"
    if declared is not None:
        return float(declared), (f"Scored and calibrated by {named} = `{float(declared):g}`, the "
                                 f"declared prediction horizon.")
    times = np.asarray(np.asarray(y)["time"], dtype=float) if len(y) else np.zeros(0)
    if not len(times):
        return None, None
    h = float(np.median(times))
    return h, (f"Scored and calibrated by {named} = `{h:g}`, the median follow-up time of the "
               f"training rows, as no prediction horizon was declared with the follow-up.")


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
        return f"Predicts worse than the class prior: CV {metric_label} {m}, against {b} for the prior."
    if task == "ordinal":
        return (f"Orders the outcome worse than the level prior: CV {metric_label} {m}, against {b} "
                f"for the prior.")
    if task == "time_to_event":
        return (f"Orders the events worse than one risk for everyone: CV {metric_label} {m}, against "
                f"{b}.")
    return (f"Predicts the classes worse than the class prior: CV {metric_label} {m}, against {b} "
            f"for the prior.")


RANKING = {"r2": "highest R²", "auc": "highest AUC", "log_loss": "lowest log loss",
           "mse": "lowest MSE", "rps": "lowest ranked probability score",
           "brier_t": "lowest Brier score at the horizon", "c_index": "highest C-index"}


def ranking_phrase(metric: str) -> str:
    """How families are ranked, in words the banner can use (audit ME-10): AUC ranks risks and is
    named as such, never "best"."""
    from turbotab.core.models.metrics import LABELS, higher_is_better

    return RANKING.get(metric) or (
        f"{'highest' if higher_is_better(metric) else 'lowest'} {LABELS.get(metric, metric)}")


def imbalance_sentence(task: str, y: Any, event: str | None,
                       rows_word: str = "training rows") -> str | None:
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
    return (f"{name} is {share:.0%} of the {rows_word}. No resampling (SMOTE, over- or "
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
    from turbotab.core.models.validation import score_words

    return (f"On these {n_train:,} training rows the cross-validated {score_words(metric)} "
            f"has a standard error of {listed}; whether two families differ is read from their "
            f"paired interval, not from either one alone.")


def coded_outcome(task: str | None, y: Any, event: str | None,
                  order: Sequence[Any] | None = None) -> Any:
    """A binary or time-to-event outcome coded 1 for the level the user named as the event
    (``set_event``), else 0.

    The event is never guessed (M2_CONTRACT §1), and its methods sentence says that level was
    coded 1; the models, their metrics and coefficients must then be about that level, not about
    whichever level sorts last. Unchanged when there is no event or it is not a level here, except
    that a True/False outcome is then coded True = 1 (an estimator types a 0/1 array, not one of
    Python booleans; ``outcome_levels`` names the codes ``True`` and ``False``).

    An ordinal outcome becomes the codes 0…K − 1 of its order (``order``, the declared one; numbers
    by value), so every family and metric reads the levels in that order (WP12a).
    """
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        return ordinal_outcome(y, order)[0]
    if task not in ("binary", "time_to_event"):
        return y
    if event is not None:
        from turbotab.core.stages.rows import _level_key

        hit = pd.Series(np.asarray(y, dtype=object)).map(_level_key).to_numpy() == _level_key(event)
        if hit.any():
            return hit.astype(int)
    values = np.asarray(y)
    if values.dtype == bool:
        # A True/False outcome with no event named among its levels: True is 1, as every family
        # and metric reads a 0/1 outcome (``modeling_frame`` keeps the outcome's own booleans).
        return values.astype(int)
    return y


def outcome_levels(task: str | None, y: Any, event: str | None) -> dict[Any, Any] | None:
    """Each class as the models hold it, mapped to its level as the data spell it.

    A binary or time-to-event outcome whose event was named is coded 1 for that level and 0 for
    the other (:func:`coded_outcome`), each named by its one spelling (``_level_key``: the event as
    ``set_event`` names it and its methods sentence prints it; ``True``, not ``1.0``), so every
    caption names the declared level. A True/False outcome with no event named among its levels is
    coded True = 1. Otherwise the models hold the levels themselves. None for regression.
    """
    if task not in ("binary", "multiclass", "time_to_event"):
        return None
    values = [v for v in pd.unique(pd.Series(np.asarray(y, dtype=object))) if not pd.isna(v)]
    if task == "multiclass":
        return {v: v for v in values}
    from turbotab.core.stages.rows import _level_key

    key = _level_key(event) if event is not None else None
    if key is not None and any(_level_key(v) == key for v in values):
        others = [_level_key(v) for v in values if _level_key(v) != key]
        reference = others[0] if len(others) == 1 else (" or ".join(others) or None)
        return {1: key, 0: reference}
    if values and all(isinstance(v, (bool, np.bool_)) for v in values):
        return {1: "True", 0: "False"}
    return {v: v for v in values}


def _accepts(fn: Any, name: str) -> bool:
    import inspect

    return name in inspect.signature(fn).parameters


def _inference_table(family: Any, pipeline: Any, X: Any, y: Any, *, task: str, clusters: Any,
                     outcome: Any, rows: str, survey: Any = None,
                     models: Sequence[str] | None = None) -> Any:
    """``family.inference``, handing it the outcome's names, the rows it is estimated from and the
    survey design when it takes them (a family registered before WP8 or WP10 may not). Under the
    population answer a family with no design-based estimator is blocked and recorded
    (MODELING_SEQUENCE §4; ``models.survey.no_design_estimator``), its exits the family that has
    one (``models``: the chosen families) and the sample-only attestation."""
    if survey is not None and not _accepts(family.inference, "survey"):
        from turbotab.core.models.survey import no_design_estimator

        return no_design_estimator(family, task, models)
    extra = {name: value for name, value in (("outcome", outcome), ("rows", rows), ("survey", survey))
             if value is not None and _accepts(family.inference, name)}
    table = family.inference(pipeline, X, y, task=task, clusters=clusters, **extra)
    if table.info.get("n_rows") is None and "rows" not in extra:
        from turbotab.core.models.inference import _on_rows

        table = _on_rows(table, len(X), rows)
    return table
UNWEIGHTED_SCORES = ("Scores are unweighted: they describe these rows, not the population the survey "
                     "weights stand for.")
# Under the population answer (MS4): the scores are about the fitting procedure on these rows, never
# a population estimate, and are labeled so beside every design-based table.
POPULATION_SCORES = ("Cross-validated scores are unweighted: they describe how the model predicts "
                     "these participants, not the surveyed population; the design-based estimates "
                     "are the coefficient table's.")


def _measurement_error_line(ctx: StageContext, spec: Any) -> str | None:
    """The inference table's measurement-error limitation (audit IN-22), or None.

    The self-reported intakes are the model's energy-bearing nutrient exposures; total energy
    counts as one more error-prone intake while it stays in the outcome model. ``reports`` is how
    many rows each analysis row averages (the working table's combining, ``recall_days``).
    """
    from turbotab.core.readings import unsettled
    from turbotab.core.methods.dietary_caveats import measurement_error_line
    from turbotab.core.methods.energy import total_energy_columns
    from turbotab.core.stages.proposals import energy_bearing, recall_days

    # BLUEPRINT §14: "self-reported intake" is a claim about what a column is, made only of a
    # settled role (an InBody body ``Protein`` carried by a bulk confirm is no intake).
    waiting = set(unsettled(ctx.state))
    exposures = [c for c in spec.predictors
                 if spec.roles.get(c) == "exposure" and energy_bearing(c) and c not in waiting]
    if not exposures:
        return None
    adj = spec.energy_adjustment()
    energy = [c for c in total_energy_columns(spec.predictors, spec.roles) if c not in waiting]
    if adj is not None and adj.method in ("none", "density", "residual_energy_dropped"):
        energy = [c for c in energy if spec.roles.get(c) != "energy"]  # it left the outcome model
    reports, _ = recall_days(ctx.inputs.get("working"))
    return measurement_error_line(exposures, energy, reports=reports)


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
                       keys: Sequence[str], loss: Mapping[str, Any] | None = None, *,
                       survey: Any = None, clusters: Any = None,
                       nested: Mapping[str, str] | None = None) -> TableMissing | None:
    """Under inference, the missing-values answer as the coefficient table applies it.

    ``X`` and ``y`` are the rows the table is estimated from; ``loss`` the complete-case comparison
    the cohort made (default: the cohort artifact's, when the stage has it). Under multiple
    imputation the imputation model is compatible with the analysis model (MS1) and holds the
    survey design's variables (``survey``: the ``FitSurvey`` of a population answer) and the
    clusters (``clusters``: by unit, ruling 12); ``nested``: parts of totals, which the energy
    identity leaves to their own models. Passive imputation with a declared nonlinear term, and
    single-level imputation on clustered rows, are held until recorded (MODELING_SEQUENCE §4)."""
    from turbotab.core.methods.imputation import ImputationRefused, M_DEFAULT
    from turbotab.core.methods.missing import (COMPLETE_CASE_ASSUMPTION, INDICATOR_CAUTION,
                                               MI_ASSUMPTION, SINGLE_FILL_CAUTION,
                                               impute_for_inference, missing_block, nonlinear_terms,
                                               row_loss_concern)
    from turbotab.core.models import get_family

    state = ctx.state
    answer = spec.missing or (state.missing.model_dump(mode="json") if state.missing else None)
    if not answer:
        return None
    levels = [c for c in spec.levels if c in X.columns and X[c].isna().any()]
    gaps = [c for c in spec.inputs if c not in levels and c in X.columns and X[c].isna().any()]
    imputing = answer.get("strategy") == "multiple_imputation" and bool(gaps)
    clustered = clusters is not None and getattr(clusters, "clustered", False)
    held = missing_block(answer, "inference", levels,
                         nonlinear=nonlinear_terms(spec, task) if imputing else (),
                         passive=imputing and task in ("ordinal", "multiclass"),
                         clustered=imputing and clustered)
    if held is not None:
        return TableMissing(refusal=held[0], exits=held[1])
    strategy = answer.get("strategy")
    m = int(answer.get("m") or M_DEFAULT)
    if strategy == "multiple_imputation":
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

        design = survey.design if survey is not None and getattr(survey, "answer", None) == "population" \
            else None
        try:
            imputations = impute_for_inference(spec, X, y, task, seed=seed, progress=progress,
                                               cancelled=ctx.cancelled, survey=design,
                                               clusters=clusters if clustered else None,
                                               nested=nested, factors=_settled_factors(ctx, spec))
        except ImputationRefused as exc:
            zeros = list(getattr(exc, "zeros", None) or [])
            exits: list[dict[str, Any]] = []
            if zeros:
                # A recorded zero the analysis logs: no fill or row choice resolves it, only how the
                # zeros are read, or an energy model without the log (MODELING_SEQUENCE §2, zeros).
                adj = spec.energy_adjustment()
                if adj is not None and adj.log_transform and zeros[0] in (*adj.nutrients,
                                                                           adj.energy_column):
                    exits.append({"label": "Adjust for energy without the log",
                                  "decision": {"kind": "set_energy_adjustment",
                                               **adj.model_dump(), "log_transform": False}})
                exits.append({"label": f"Say how the zeros of `{zeros[0]}` are read: values below a "
                                       f"detection limit, or true zeros", "decision": None})
            else:
                exits.append({"label": "Complete cases, with their assumption stated",
                              "decision": {**answer, "kind": "set_missing",
                                           "strategy": "complete_case"}})
            return TableMissing(refusal=f"Multiple imputation cannot run on these data: {exc}",
                                exits=exits)
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


def _settled_factors(ctx: StageContext, spec: Any) -> dict[str, float]:
    """Each energy-adjusted nutrient's kcal per unit, where the readings ledger holds it settled
    (BLUEPRINT §14.1: a recorded unit, or grams the registry's Atwater test reads; never a name):
    what the imputation's energy identity may compute with."""
    from turbotab.core.readings import factor_verdicts, kcal_per_unit

    adj = spec.energy_adjustment()
    nutrients = list(adj.nutrients) if adj is not None else []
    if not nutrients:
        return {}
    try:
        with open_store(ctx) as store:
            verdicts = factor_verdicts(ctx.state, store, nutrients)
    except Exception:  # noqa: BLE001 - no values to read: only a recorded unit settles
        verdicts = {}
    out: dict[str, float] = {}
    for c in nutrients:
        reading = kcal_per_unit(ctx.state, c, verdict=verdicts.get(c))
        if reading.settled and reading.factor is not None:
            out[c] = float(reading.factor)
    return out


def imputed_copies_column(state: Any) -> str | None:
    """Under inference, the column numbering the imputed copies the data carries (NHANES DXA,
    ``set_repeat_kind``'s ``implicate_column``), when each copy is kept as a record
    (``set_unit`` "row"): the fit then analyzes each copy and pools them by Rubin's rules (audit
    I18; MODELING_SEQUENCE §1.1). None otherwise."""
    rk = getattr(state, "repeat_kind", None)
    if (getattr(state, "purpose", None) != "inference" or rk is None
            or getattr(rk, "repeat_kind", None) != "imputed_copies"
            or not getattr(rk, "implicate_column", None) or getattr(state, "unit", None) != "row"):
        return None
    return str(rk.implicate_column)


def _copies_for_table(ctx: StageContext, spec: Any, frame: pd.DataFrame, implicate: str, y: Any,
                      unit_columns: Sequence[str]) -> tuple[TableMissing, Any]:
    """The data's own imputed copies as the table's completed copies (each with its own outcome),
    and the clustering within a copy (none: each unit appears once in a copy). Blanks left inside
    the copies are not imputed again here (nested imputation is not built): the table is blocked,
    complete cases its exit."""
    from turbotab.core.methods.missing import supplied_copies
    from turbotab.core.models.inference import INDEPENDENT

    state = ctx.state
    grain = getattr(state, "grain", None)
    unit = getattr(grain, "id_column", None) if grain is not None else None
    unit = unit if unit in frame.columns else next((c for c in unit_columns if c in frame.columns), None)
    answer = spec.missing or (state.missing.model_dump(mode="json") if state.missing else None)
    blanks = [c for c in spec.inputs if c in frame.columns and frame[c].isna().any()]
    if blanks and (answer or {}).get("strategy") != "complete_case":
        return TableMissing(
            refusal=(f"The rows are the data's imputed copies, and {', '.join(f'`{c}`' for c in blanks[:3])}"
                     f" still {'has' if len(blanks) == 1 else 'have'} blanks inside them; imputing "
                     f"within each copy (nested imputation) is not built here."),
            exits=[{"label": "Complete cases within each copy",
                    "decision": {**(answer or {}), "kind": "set_missing",
                                 "strategy": "complete_case"}}]), INDEPENDENT
    copies = supplied_copies(frame, implicate, unit, list(spec.inputs), y)
    return TableMissing(imputations=copies), INDEPENDENT


def _with_levels(fitted: Any, levels: Sequence[str] | None) -> Any:
    """An ordinal fit's cut-points named by the declared levels (as the fit stage names them)."""
    if levels is not None:
        fitted[-1].level_names_ = list(levels)
    return fitted


def pooled_table(family: Any, template: Any, imputations: Any, y: Any, *, task: str, clusters: Any,
                 outcome: Any, survey: Any, design: Any, spec: Any, rows: str,
                 fit: Any, cancelled: Any = None, energy_rows: bool = True,
                 collect: dict[str, Any] | None = None
                 ) -> tuple[Any, list[dict[str, Any]] | None, list[dict[str, Any]], list[str]]:
    """The family's inference table pooled over the completed copies (Rubin's rules), with its
    energy rows and exposure-form tests pooled beside it: (table, pooled rows, tests, concerns).

    Each copy is analyzed exactly as an unimputed table is (``fit(model, X_k)`` refits the whole
    pipeline on it; the table, the all-components relative effects and the form tests follow),
    except that its impute step refuses a blank made inside the copy and its knots and cut points
    are the ones placed once on the observed values (``missing.copy_template``; MS1). Copies the
    data carried with their own outcomes (``Imputations.outcomes``, NHANES DXA) are fit as
    ``fit(model, X_k, y_k)``. A table refused in one copy (too few clusters) is refused: the reason
    does not depend on the imputed values. ``rows`` = None in the result marks such a refusal.
    ``collect``, when given, receives each copy's fit (``fits``) and table (``tables``)."""
    from turbotab.core.methods.exposure_form import exposure_tests
    from turbotab.core.methods.imputation import pool_rows, pooled_cov
    from turbotab.core.methods.missing import copy_template, mi_concerns, multiple_imputation_info
    from turbotab.core.models.inference import InferenceTable

    m = int(imputations.m)
    plan = dict(getattr(imputations, "plan", None) or {})
    outcomes = getattr(imputations, "outcomes", None)
    tables, row_sets, fits, tests = [], [], [], []
    form_concerns: list[str] = []
    for k, X_k in enumerate(imputations.frames):
        if cancelled is not None and cancelled():
            raise Cancelled()
        y_k = y if outcomes is None else outcomes[k]
        model = copy_template(template, plan)
        fitted = fit(model, X_k) if outcomes is None else fit(model, X_k, y_k)
        table = _inference_table(family, fitted, X_k, y_k, task=task, clusters=clusters,
                                 outcome=outcome, rows=rows, survey=survey)
        if table.info.get("refused"):
            table.info["missing"] = multiple_imputation_info(imputations, spec, len(y_k))
            return table, None, [], []
        row_sets.append(_energy_rows(table.rows, design, spec, family, fitted, X_k, y_k, task,
                                     clusters, outcome, survey) if energy_rows else table.rows)
        found, worries = exposure_tests(family, fitted, X_k, y_k, task=task, clusters=clusters,
                                        table=table, outcome=outcome, survey=survey)
        if k == 0:
            form_concerns = list(worries)
        tables.append(table)
        fits.append(fitted)
        tests.append(found)
    pooled = pool_rows(row_sets)
    info = dict(tables[0].info)
    copies = getattr(imputations, "method", "") == "supplied"
    how = (f"The data's {m} imputed copies, each analyzed as follows with its own outcome" if copies
           else f"Multiple imputation, m = {m}: each completed table analyzed as follows")
    info["caption"] = (f"{how}, then pooled by Rubin's rules, each interval on t with Barnard–Rubin "
                       f"degrees of freedom. {tables[0].info.get('caption', '')}").strip()
    form_tests = _pool_form_tests(tests, tables, fits, m)
    df_com = _design_df(tables[0])
    info["missing"] = multiple_imputation_info(
        imputations, spec, len(y if outcomes is None else outcomes[0]), rows=pooled, df_com=df_com,
        tests=sum(1 for t in form_tests if t.get("distribution") in ("F", "chi2")
                  and int(t.get("df_num") or 0) > 1))
    concerns = list(tables[0].concerns) + mi_concerns(imputations, pooled, len(y if outcomes is None
                                                                                else outcomes[0]))
    covs = [t.cov for t in tables]
    cov = None
    if all(c is not None for c in covs) and len({np.shape(c) for c in covs}) == 1:
        names = [str(r["feature"]) for r in tables[0].rows]
        Q = np.asarray([[r["estimate"] if r["estimate"] is not None else np.nan for r in t.rows]
                        for t in tables], dtype=float)
        if Q.shape[1] == len(names) and np.all(np.isfinite(Q)):
            cov = pooled_cov(Q, np.asarray(covs, dtype=float))
    table = InferenceTable(pooled, info, concerns, cov=cov)
    if collect is not None:
        collect["fits"] = fits
        collect["tables"] = tables
    return table, pooled, form_tests, form_concerns


def _design_df(table: Any) -> float | None:
    """A design-based table's complete-data degrees of freedom (its design's), for Rubin's rules
    and D1 (MODELING_SEQUENCE §2: "ν_com = the design df"); None for any other table."""
    info = getattr(table, "info", None) or {}
    if info.get("covariance") != "design":
        return None
    df = (info.get("survey") or {}).get("df")
    if df is None:
        dfs = [r.get("df") for r in table.rows if r.get("df") is not None]
        df = dfs[0] if dfs else None
    return float(df) if df is not None else None


def _pool_form_tests(tests: Sequence[Sequence[dict[str, Any]]], tables: Sequence[Any],
                     fits: Sequence[Any], m: int) -> list[dict[str, Any]]:
    """Each exposure-form test pooled over the imputations: a spline's Wald tests and quintiles'
    global test by Li, Raghunathan & Rubin's D1 (on each copy's estimates and covariance; under a
    survey design on Reiter's denominator df from the design's), the quintile trend's coefficient
    by Rubin's rules."""
    from turbotab.core.methods.exposure_form import form_step
    from turbotab.core.methods.imputation import pool_scalar, pooled_wald
    from turbotab.core.models.inference import format_p

    if not tests or not tests[0]:
        return []
    out: list[dict[str, Any]] = []
    step = form_step(fits[0])
    df_com = _design_df(tables[0])
    for test in tests[0]:
        same = [next((t for t in ts if t["column"] == test["column"] and t["test"] == test["test"]),
                     None) for ts in tests]
        if any(t is None for t in same):
            continue
        if test["test"] in ("overall", "nonlinear", "global"):
            if step is None:
                continue
            outputs = step._outputs(test["column"])
            names = outputs[1:] if test["test"] == "nonlinear" else outputs
            Q, U = [], []
            for table in tables:
                where = {str(r["feature"]): j for j, r in enumerate(table.rows)}
                if table.cov is None or any(n not in where for n in names):
                    Q = []
                    break
                ids = [where[n] for n in names]
                Q.append([table.rows[j]["estimate"] for j in ids])
                U.append(np.asarray(table.cov, dtype=float)[np.ix_(ids, ids)])
            result = (pooled_wald(np.asarray(Q, dtype=float), np.asarray(U, dtype=float), df_com)
                      if Q else None)
            if result is None:
                continue
            f_ref = result["df_den"] is not None
            stat = (f"F({result['df_num']}, {result['df_den']:,.0f}) = {result['statistic']:.2f}"
                    if f_ref else f"χ²({result['df_num']}) = {result['statistic'] * result['df_num']:.2f}")
            design = (f", on Reiter's denominator df from the design's {df_com:g}"
                      if df_com is not None and f_ref else "")
            out.append({**test, "statistic": result["statistic"] if f_ref else result["statistic"] * result["df_num"],
                        "df_num": result["df_num"], "df_den": result["df_den"],
                        "distribution": "F" if f_ref else "chi2", "p": result["p"],
                        "caption": (f"Pooled over {m} imputations by Li, Raghunathan & Rubin's D1"
                                    f"{design}: {stat}, p = {format_p(result['p'])}. Within each "
                                    f"imputation: {test['caption']}")})
            continue
        ests = [t.get("estimate") for t in same]
        stats_ = [t.get("statistic") for t in same]
        if any(e is None for e in ests) or any(st in (None, 0) for st in stats_):
            continue
        variances = [(float(e) / float(st)) ** 2 for e, st in zip(ests, stats_)]
        dfs = [t.get("df_den") for t in same]
        df_com_t = None if any(d is None for d in dfs) else float(np.mean(dfs))
        pooled = pool_scalar(ests, variances, df_com_t)
        if pooled.p is None:
            continue
        medians = [t.get("medians") for t in same]
        out.append({**test, "estimate": pooled.estimate, "ci_low": pooled.ci_low,
                    "ci_high": pooled.ci_high, "p": pooled.p,
                    "statistic": pooled.estimate / math.sqrt(pooled.total), "df_den": pooled.df,
                    "distribution": "t" if pooled.df is not None else "z",
                    "medians": ([float(np.mean([mm[g] for mm in medians])) for g in range(len(medians[0]))]
                                if all(mm for mm in medians) else test.get("medians")),
                    "caption": (f"Pooled over {m} imputations by Rubin's rules: the trend coefficient "
                                f"{pooled.estimate:+.4g} per unit (each copy scored by its own "
                                f"quintile medians), p = {format_p(pooled.p)}. Within each "
                                f"imputation: {test['caption']}")})
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

    MS6 (MODELING_SEQUENCE ruling 4): the primary is strictly proper (``metrics.PRIMARY``), with AUC
    or C the customary headline beside it; under prediction every family and the baseline are also
    fit on the comparison substrate, repeated k-fold of at least 10 × K whose first repeats are the
    split's own (``folds.comparison_folds``), and the comparisons, the baseline verdict and BBC-CV
    (by whole unit) read it; the result is declared (``selection.declared_result``); calibration is
    by level, or at a horizon, or said to be not assessed; and the nested cross-validation interval
    runs when the split asks for it, and is offered where predictors outnumber rows.
    """
    from sklearn.base import clone

    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import FitArtifact, HoldoutDetail
    from turbotab.core.models.base import reports_coefficients
    from turbotab.core.models.artifacts import Comparison, NestedOffer
    from turbotab.core.models.baseline import no_better_concern, versus_baseline
    from turbotab.core.models.cost import duration
    from turbotab.core.models.folds import comparison_folds
    from turbotab.core.models.inference import Outcome, cluster_columns, resolve_clusters
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import (CV_DEFINITION, HEADLINE, HEADLINE_LABEL, LABELS,
                                              SE_DEFINITION, classes_of, cross_validate,
                                              higher_is_better, metric_labels, predict,
                                              primary_metric, repeated_pairs, score, tension)
    from turbotab.core.models.performance import NOT_ASSESSED, calibration, score_intervals
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.models.selection import OutOfFold, declared_result, selection_optimism
    from turbotab.core.models.survival import follow_up_columns, time_to_event_outcome
    from turbotab.core.models.validation import (COMPARISONS_NOTE, PER_ROW_LOSSES, calibration_by,
                                                 family_differences, fired_chain, headline_how,
                                                 internal_external, level_words, nested_cv_fits,
                                                 nested_cv_interval, not_applied,
                                                 optimism_bootstrap, performance_sentence,
                                                 resample_concern, score_words, wide, wide_clause)
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
    # Imputed copies the data carries (NHANES DXA, audit I18): each copy is its own analysis.
    implicate = imputed_copies_column(state)
    if implicate is not None and implicate not in columns:
        columns.append(implicate)
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
        frame = modeling_frame(store, columns, assignment.index.to_numpy(), outcome=target)
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
    if inference and implicate is not None:
        missing, copy_clusters = _copies_for_table(ctx, spec, frame.loc[table_rows], implicate,
                                                    y_tab, unit_columns)
    else:
        from turbotab.core.readings import nesting

        copy_clusters = None
        nested = nesting(state, dict((design.objects or {}).get("nested") or {}), columns=spec.inputs) \
            if inference else None
        missing = (_missing_for_table(ctx, spec, X_tab, y_tab, task, keys, survey=survey,
                                      clusters=clusters, nested=nested) if inference else None)
    # Audit IN-22: under inference the coefficient table says its dietary intakes are measured with
    # error and not corrected here (methods/dietary_caveats.py; Freedman et al. 2011).
    error_line = _measurement_error_line(ctx, spec) if inference else None

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

    # MS6 (MODELING_SEQUENCE ruling 4): the primary is a strictly proper score; AUC and C are the
    # customary headline beside it. A time to event is scored and calibrated by a horizon.
    horizon, horizon_note = (follow_up_horizon(state, y) if task == "time_to_event" else (None, None))
    primary = primary_metric(task, y if task == "time_to_event" else None)
    # MS6: under prediction the families and the baseline are compared, and the choice among them
    # corrected, on repeated k-fold (≥ 10 × K) whatever validation supplies the headline; the
    # split's own folds come first, so the headline's run is a subset of it (models/folds.py).
    seed = int(split_data.get("seed") or 0)
    k_folds = int(getattr(state.split, "folds", None) or split_data.get("folds") or 5)
    strata = (None if task == "regression" else
              np.asarray(y["event"]).astype(int) if task == "time_to_event" else y)
    if not inference:
        sub_columns, shared, sub_note = comparison_folds(
            fold_columns, validation=validation, scheme=scheme, n=len(y), strata=strata,
            groups=groups, folds=k_folds, seed=seed)
    else:  # under inference the scores describe fit: not a comparison (MODELING_SEQUENCE §1 row 11)
        sub_columns, shared, sub_note = fold_columns, len(fold_columns), None
    sub_pairs, sub_repeat_of = repeated_pairs(sub_columns, scheme)
    headline_repeats = list(range(len(fold_columns))) if shared else None
    base_sub = _baseline_cv(task, X, y, sub_pairs, sub_repeat_of, horizon)
    base = (base_sub.subset(headline_repeats) if headline_repeats is not None
            else _baseline_cv(task, X, y, pairs, repeat_of, horizon))
    base_cv = base.summary(task)
    base_sub_cv = base_sub.summary(task) if not inference else base_cv
    baseline = {"metric": primary, "value": base_cv[primary]["estimate"], "label": BASELINE_LABEL[task]}
    # The verdict against the baseline is read on R² for regression: the baseline's R² is 0 in every
    # fold, so a family's R² is its MSE gain as a share of the baseline's (the same comparison).
    versus_metric = "r2" if task == "regression" else primary
    reference = float(np.mean(y.astype(float))) if task == "regression" and len(y) else None
    n_boot = int(split_data.get("n_boot") or 0) if validation == "bootstrap" else 0
    own_pairs = 0 if headline_repeats is not None else len(pairs)
    units = max(1, len(keys) * (len(sub_pairs) + own_pairs + 2 + n_boot))
    done = 0
    results: dict[str, Any] = {}
    substrates: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    sealed_detail: dict[str, Any] = {}
    nested_runs: dict[str, Any] = {}
    n_features = len(spec.inputs)
    narrow = wide_clause(n_features, len(y)) if wide(n_features, len(y)) else None
    run_nested = bool(getattr(state.split, "nested_cv", False)) and not inference \
        and primary in PER_ROW_LOSSES
    unit_series = (pd.Series(groups, index=X.index) if groups is not None else None)

    def share(n: int) -> float:
        return 0.02 + 0.97 * n / units

    models: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    every_row: dict[str, Any] = {}  # under inference: each family refit on every analyzed row
    imputed_fits: dict[str, Any] = {}  # under multiple imputation: each family's fit on each copy
    imputed_tables: dict[str, Any] = {}  # … and each copy's coefficients and covariance
    sealed: dict[str, Any] = {}  # held-out scores: kept out of the public data (M2_CONTRACT §3)
    # Each family's out-of-fold predictions, for the selection's optimism (models/selection.py):
    # the substrate's first repeat, so each scored row is predicted once; resampled by unit.
    oof = OutOfFold(task, X, y, [p for p, r in zip(sub_pairs, sub_repeat_of) if r == 0],
                    units=groups, horizon=horizon)
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
            done += len(sub_pairs) + own_pairs + 2
            continue

        def before_fold(i: int, n: int, _label: str = family.label) -> None:
            nonlocal done
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{_label}: fold {i + 1} of {n}")
            done += 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sub_result = cross_validate(task, lambda _k=key: with_units(clone(pipelines[_k]), unit_of),
                                        X, y, sub_pairs,
                                        fit=oof.wrap(key, fit) if len(keys) > 1 else fit,
                                        before_fold=before_fold, repeat_of=sub_repeat_of,
                                        keep_predictions=True, horizon=horizon)
            result = (sub_result.subset(headline_repeats) if headline_repeats is not None else
                      cross_validate(task, lambda _k=key: with_units(clone(pipelines[_k]), unit_of),
                                     X, y, pairs, fit=fit, before_fold=before_fold,
                                     repeat_of=repeat_of, keep_predictions=True, horizon=horizon))
            if ctx.cancelled():
                raise Cancelled()
            ctx.progress(share(done), f"{family.label}: refitting on all training rows")
            final = fit(with_units(clone(pipelines[key]), unit_of), X, y)
            if levels is not None:
                final[-1].level_names_ = list(levels)
            done += 1
            # R² on the held-out rows is measured against the training rows' mean.
            holdout = (score(task, final, X_hold, y_hold, reference=reference, horizon=horizon)
                       if len(y_hold) else None)
            if len(y_hold):  # sealed with the scores: an interval on each, and calibration
                classes = classes_of(task, final)
                held_pred = predict(task, final, X_hold, horizon=horizon)
                held_levels, held_horizon, held_note = calibration_by(
                    task, y_hold, held_pred, classes, levels, hold_groups, horizon,
                    "on the held-out rows")
                sealed_detail[key] = HoldoutDetail(
                    intervals=score_intervals(task, y_hold, held_pred, classes=classes,
                                              reference=reference, groups=hold_groups,
                                              unit=grouped_by, horizon=horizon),
                    calibration=calibration(task, y_hold, held_pred, classes=classes,
                                            groups=hold_groups, where="on the held-out rows"),
                    calibration_levels=held_levels, calibration_horizon=held_horizon,
                    calibration_note=held_note,
                ).model_dump(mode="json")
            optimism = None
            if n_boot and not getattr(family, "bootstrap_optimism", True):
                # A near-interpolating family: the bootstrap would overstate it (Coley et al. 2023).
                optimism = not_applied(family.label, n_boot, grouped_by)
                done += n_boot
            elif n_boot:
                def boot_progress(b: int, total: int, _label: str = family.label) -> None:
                    nonlocal done
                    done += 1
                    ctx.progress(share(done), f"{_label}: bootstrap refit {b} of {total}")

                optimism = optimism_bootstrap(
                    task, lambda _k=key: clone(pipelines[_k]), refit_resample, X, y, n_boot=n_boot,
                    seed=seed, groups=groups, unit=grouped_by, final=final, horizon=horizon,
                    progress=boot_progress, cancelled=ctx.cancelled)
            nested = None
            if run_nested:
                ctx.progress(share(done), f"{family.label}: nested cross-validation")

                def nested_fit(model: Any, X_fit: Any, y_fit: Any) -> Any:
                    units_fit = (None if unit_series is None
                                 else unit_series.reindex(X_fit.index).to_numpy())
                    return fit_pipeline(model, X_fit, y_fit, groups=units_fit)

                nested = nested_cv_interval(
                    task, primary, lambda _k=key: with_units(clone(pipelines[_k]), unit_of),
                    nested_fit, X, y, folds=k_folds, seed=seed, groups=groups, horizon=horizon,
                    cancelled=ctx.cancelled)
                nested_runs[key] = nested
            ctx.progress(share(done), f"{family.label}: coefficients")
            concerns: list[str] = []
            interval_info = None
            n_coefficients = None
            on_all = inference and reports_coefficients(family)
            form_tests: list[dict[str, Any]] = []
            pooled = None  # the coefficient rows pooled over the imputations (multiple imputation)
            table_fit = None
            try:
                if on_all and not same_rows:
                    ctx.progress(share(done), f"{family.label}: coefficients on every analyzed row")
                # The every-row refit names an ordinal fit's cut-points by the declared levels, as
                # ``final`` does: unnamed, they would read by the internal codes (0 | 1 for 1 | 2).
                table_fit = ((final if same_rows else _with_levels(
                    fit_table(with_units(clone(pipelines[key]), unit_of)), levels))
                             if on_all else final)
                X_c, y_c = (X_tab, y_tab) if on_all else (X, y)
                if clusters is not None and hasattr(family, "inference"):
                    from turbotab.core.models.survey import (blocked, has_design_estimator,
                                                             no_design_estimator)

                    design_of = survey.design if survey is not None else None
                    if survey is not None and survey.refusal:
                        table = blocked(survey.refusal, survey.exits)
                    elif design_of is not None and not has_design_estimator(family, task):
                        # MS4: the population answer binds every family (MODELING_SEQUENCE §4).
                        table = no_design_estimator(family, task, state.models)
                    elif missing is not None and missing.refusal:
                        table = blocked(missing.refusal, missing.exits,
                                        estimator="not fitted: the missing-values answer decides it")
                    elif missing is not None and missing.imputations is not None and on_all:
                        ctx.progress(share(done), f"{family.label}: each of "
                                                  f"{missing.imputations.m} imputations")
                        def fit_copy(model: Any, X_k: Any, y_k: Any = None) -> Any:
                            if y_k is None:
                                return _with_levels(fit_table(with_units(model, unit_of), X_k),
                                                    levels)
                            return _with_levels(fit_pipeline(model, X_k, y_k), levels)

                        collected: dict[str, Any] = {}
                        table, pooled, form_tests, form_concerns = pooled_table(
                            family, pipelines[key], missing.imputations, y_c, task=task,
                            clusters=copy_clusters if copy_clusters is not None else clusters,
                            outcome=outcome, survey=design_of, design=design,
                            spec=spec, rows="all", fit=fit_copy, cancelled=ctx.cancelled,
                            collect=collected)
                        imputed_fits[key] = collected.get("fits") or []
                        imputed_tables[key] = [
                            {"names": [str(r["feature"]) for r in t.rows],
                             "estimates": [r["estimate"] for r in t.rows],
                             "cov": None if t.cov is None else np.asarray(t.cov, dtype=float),
                             "df": next((r.get("df") for r in t.rows if r.get("df") is not None), None)}
                            for t in collected.get("tables") or []]
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
                    if (survey is not None and survey.answer == "population"
                            and coefficients is not None):
                        # MS4: penalized coefficients have no design-based estimator here: blocked
                        # and recorded, as every family without one is (MODELING_SEQUENCE §4).
                        from turbotab.core.models.survey import no_design_estimator

                        table = no_design_estimator(family, task, state.models)
                        coefficients, interval_info = table.rows, table.info
                        concerns.extend(table.concerns)
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
            if coefficients and error_line:
                concerns.append(error_line)
            if coefficients and pooled is None:
                # The relative effects come from the fit the table came from: under inference
                # every analyzed row, with the table's clusters and outcome scale. (Pooled tables
                # carry each imputation's, pooled with the coefficients.)
                coefficients = _energy_rows(coefficients, design, spec, family, table_fit, X_c, y_c,
                                            task, clusters, outcome,
                                            survey.design if survey is not None else None)
            done += 1
        concerns = _concerns(caught, len(sub_pairs) + own_pairs + 1) + concerns
        if family.key == "linear":
            from turbotab.core.models.linear import collinearity_concern, model_matrix

            try:
                # Ruling 3: under inference the concern describes the matrix the table was fit on.
                use_table = on_all and table_fit is not None
                singular = collinearity_concern(model_matrix(table_fit, X_c) if use_table
                                                else model_matrix(final, X))
            except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
                singular = None
            if singular and not any("singular" in c for c in concerns):
                concerns.insert(0, singular)
        cv = result.summary(task, groups=groups, unit=grouped_by)
        sub_cv = sub_result.summary(task, groups=groups, unit=grouped_by) if not inference else cv
        # The verdict against the baseline, paired over every fold of the substrate (MS6): the
        # corrected repeated k-fold t, 1/(rK) and rK − 1 df (models/baseline.py).
        estimate = sub_cv[versus_metric]["estimate"]
        base_value = base_sub_cv[versus_metric]["estimate"]
        higher = higher_is_better(versus_metric)
        gain = ((1.0 if higher else -1.0) * (estimate - base_value)
                if estimate is not None and base_value is not None else None)
        versus = versus_baseline(versus_metric, sub_result.folds_of(versus_metric),
                                 base_sub.folds_of(versus_metric), gain=gain,
                                 test_share=sub_result.test_share)
        worse = baseline_concern(task, LABELS[versus_metric], estimate, base_value, lower=not higher)
        tie = None if worse else no_better_concern(task, LABELS[versus_metric], versus, estimate,
                                                   base_value, _two)
        if worse or tie:
            concerns.insert(0, worse or tie)
        rows_oof, y_oof, pred_oof = result.out_of_fold(0)
        oof_calibration = calibration(
            task, y_oof, pred_oof, classes=classes_of(task, final),
            groups=None if groups is None else groups[rows_oof]) if len(y_oof) else None
        if oof_calibration is not None and oof_calibration.concern:
            concerns.append(oof_calibration.concern)
        cal_levels, cal_horizon, cal_note = (calibration_by(
            task, y_oof, pred_oof, classes_of(task, final), levels,
            None if groups is None else groups[rows_oof], horizon, "out of fold")
            if len(y_oof) else (None, None, f"{NOT_ASSESSED}: no out-of-fold predictions."))
        for item in cal_levels or []:
            if item.calibration is not None and item.calibration.concern:
                said = item.calibration.concern
                concerns.append(f"{level_words(task, item.level)}: {said[0].lower()}{said[1:]}")
        if cal_horizon is not None and cal_horizon.concern:
            concerns.append(cal_horizon.concern)
        if task in ("regression", "binary") and oof_calibration is None:
            cal_note = f"{NOT_ASSESSED}: too few out-of-fold rows, or one class only."
        if optimism is not None and optimism.refused:
            concerns.append(optimism.refused)
        if n_boot and getattr(family, "bootstrap_optimism", True) and resample_concern(n_boot):
            concerns.append(resample_concern(n_boot))
        if nested is not None and nested.refused:
            concerns.append(nested.refused)
        iecv = None
        if validation == "internal_external" and split_data.get("cluster"):
            iecv = internal_external(task, result, list(split_data.get("fold_labels") or []),
                                     str(split_data["cluster"]), groups=groups, unit=grouped_by,
                                     metric=primary)
        if survey_note:
            concerns.append(survey_note)
        elif survey is not None and survey.answer == "population" and not survey.refusal:
            concerns.append(POPULATION_SCORES)
        results[key], substrates[key], summaries[key] = result, sub_result, cv
        fitted[key] = final
        if inference and (same_rows or (on_all and table_fit is not None)):
            # Under inference every estimate reads the fit on every analyzed row (BLUEPRINT §12
            # ruling 3): the substitution curve too, so it agrees with the coefficient table. A
            # family with no table is refit on them by the substitution stage, if a curve is asked.
            every_row[key] = table_fit if on_all and table_fit is not None else final
        sealed[key] = holdout
        entry = cv.get(primary) or {}
        if nested is not None and nested.estimate is not None:
            performance = performance_sentence(
                f"{LABELS[primary]}", nested.estimate, nested.ci_low, nested.ci_high,
                how=f"nested cross-validation ({nested.reps} repetitions of {nested.folds} folds; "
                    f"Bates, Hastie & Tibshirani 2023)")
        else:
            performance = performance_sentence(
                f"Cross-validated {score_words(primary)}",
                entry.get("estimate"), entry.get("ci_low"), entry.get("ci_high"),
                how=headline_how(validation, scheme, k_folds, len(fold_columns),
                                  split_data.get("cluster")),
                narrow=narrow)
        corrected = ((optimism.estimates.get(primary) if optimism is not None else None))
        if corrected is not None and corrected.corrected is not None:
            performance += " " + performance_sentence(
                f"Optimism-corrected {score_words(primary)}",
                corrected.corrected, None, None,
                how=f"Harrell's bootstrap ({optimism.n_ok:,} resamples)")
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
            "calibration_levels": ([c.model_dump(mode="json") for c in cal_levels]
                                   if cal_levels is not None else None),
            "calibration_horizon": cal_horizon.model_dump(mode="json") if cal_horizon else None,
            "calibration_note": cal_note,
            "compared_on": (sub_cv.get(primary) if not inference else None),
            "performance": performance,
            "nested_cv": nested.model_dump(mode="json") if nested is not None else None,
        })
    # Picking the best of several families by cross-validation flatters it (audit ME-13); under
    # prediction it is corrected on the substrate's out-of-fold predictions, by whole unit (MS6).
    if len(results) > 1:
        ctx.progress(0.995, "Choosing among the families: bootstrapping the out-of-fold predictions")
    labels = {m["family"]: m["label"] for m in models}
    headline = HEADLINE.get(task)
    extras = [m for m in (headline, "r2" if task == "regression" else None)
              if m is not None and m != primary]
    selection = selection_optimism(task, primary, substrates, oof, labels, score_words(primary),
                                   extras=extras)
    ctx.progress(1.0, "Done")
    n_holdout = int((~train).sum())
    comparisons = (family_differences(task, substrates, summaries, labels, metric=primary)
                   if len(results) > 1 else [])
    # The event's share beside the table: every analyzed row under inference (ruling 3), the
    # training rows the models learn from under prediction.
    imbalance = (imbalance_sentence(task, y_tab, state.event, "analyzed rows") if inference
                 else imbalance_sentence(task, y, state.event))
    predicting = [m["family"] for m in models if m.get("cv")]
    result_declared = None
    if not inference and predicting:
        only = predicting[0] if len(predicting) == 1 else None
        one = next((m for m in models if m["family"] == only), None) if only else None
        result_declared = declared_result(
            task, primary, n_holdout=n_holdout, families=predicting, labels=labels,
            summaries=summaries, selection=selection,
            optimism=(one or {}).get("optimism"),
            narrow=None if (only and only in nested_runs and nested_runs[only].estimate is not None)
            else narrow,
            nested=(nested_runs[only].model_dump() if only in nested_runs else None) if only else None,
            how_cv=headline_how(validation, scheme, k_folds, len(fold_columns),
                                 split_data.get("cluster")))
    comparison = (Comparison(folds=len(sub_pairs) // max(len(sub_columns), 1),
                             repeats=len(sub_columns), shared=shared,
                             method=("the corrected repeated k-fold t (Nadeau & Bengio 2003; "
                                     "Bouckaert & Frank 2004)"), note=sub_note)
                  if not inference else None)
    offer = None
    if narrow and not inference and not run_nested and primary in PER_ROW_LOSSES and predicting:
        fits_each = nested_cv_fits(k_folds)
        per_fit = [substrates[k].seconds / max(len(sub_pairs), 1) for k in predicting
                   if k in substrates]
        seconds = round(float(sum(per_fit)) * fits_each, 1) if per_fit else None
        offer = NestedOffer(
            label=(f"Run the nested cross-validation interval (Bates, Hastie & Tibshirani 2023): "
                   f"{fits_each:,} refits of each family"),
            fits=fits_each, seconds=seconds,
            estimate=duration(seconds) if seconds is not None else "not measured",
            decision={"kind": "set_split", **state.split.model_dump(mode="json"), "nested_cv": True})
    chain = fired_chain(task, primary, inference=inference, grouped_by=grouped_by,
                   families=predicting, n_holdout=n_holdout, comparison=comparison,
                   selection=selection, result=result_declared, n_boot=n_boot,
                   horizon_note=horizon_note, notes=[m.get("calibration_note") for m in models],
                   narrow=narrow, nested=bool(nested_runs))
    artifact = FitArtifact(task=task, primary_metric=primary, metric_labels=metric_labels(task),
                           n_train=int(train.sum()), n_holdout=n_holdout, models=models,
                           holdout_sealed=n_holdout > 0, fold_scheme=scheme,
                           cv_definition=CV_DEFINITION[task], validation=validation,
                           repeats=len(fold_columns), ranking=ranking_phrase(primary),
                           se_definition=SE_DEFINITION, comparisons=comparisons,
                           precision=precision_sentence(primary, summaries, labels, int(train.sum())),
                           imbalance=imbalance, selection=selection, levels=levels,
                           headline_metric=headline,
                           headline_label=HEADLINE_LABEL if headline else None,
                           tension=tension(task, primary), comparison=comparison,
                           comparisons_note=COMPARISONS_NOTE if comparisons else None,
                           result=result_declared, horizon=horizon, horizon_note=horizon_note,
                           wide=narrow, nested_offer=offer, chain=chain)
    frames = {SEALED_SCORES: sealed_scores_frame(models, sealed)} if n_holdout else {}
    if n_holdout and sealed_detail:
        frames[SEALED_DETAIL] = sealed_detail_frame(sealed_detail)
    return Bundle(data=artifact.model_dump(mode="json"), frames=frames,
                  objects={"fitted": fitted, "grouped_by": grouped_by,
                           "every_row": every_row if inference else None,
                           "every_row_ids": (assignment.index[table_rows].to_numpy(dtype=np.int64)
                                             if inference else None),
                           # MS4: the design the population answer binds every display to.
                           "survey_design": (survey.design if survey is not None
                                             and survey.answer == "population"
                                             and not survey.refusal else None),
                           # MS3: every estimate under multiple imputation is pooled, the
                           # substitution curve included: the copies and each family's fit on each.
                           "imputations": ({"frames": missing.imputations.frames,
                                            "outcomes": missing.imputations.outcomes,
                                            "plan": dict(missing.imputations.plan or {}),
                                            "fits": imputed_fits, "tables": imputed_tables}
                                           if inference and missing is not None
                                           and missing.imputations is not None and imputed_fits
                                           else None)})

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
    weights = None
    if survey is not None:
        # MS4: under the surveyed population the shares are the population's (survey-weighted).
        from turbotab.core.models.survey import domain_of

        domain = domain_of(matrix.index, survey)
        weights = np.zeros(len(matrix))
        weights[domain.keep] = domain.raw
    from turbotab.core.readings import per_unit_words

    units = {str(c): per_unit_words(f.get("unit")) for c, f in (spec.energy_factors or {}).items()
             if f.get("unit")}
    try:
        extra = relative_effect_rows(matrix, list(adj.nutrients), factors, table=table,
                                     coefficients=rows, weights=weights, units=units)
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

    Under a survey design (WP10) the table waits for the survey answer as the linear family's does;
    under the population answer a family with no design-based tests is blocked and recorded (MS4)."""
    from turbotab.core.models.inference import _on_rows
    from turbotab.core.models.linear import as_clusters

    concerns = [f"{family.label} tests each exposure and makes no predictions, so it has no "
                f"cross-validated or held-out score."]
    coefficients = interval_info = None
    try:
        if survey is not None and survey.refusal:
            from turbotab.core.models.survey import blocked

            table = blocked(survey.refusal, survey.exits)
        elif (survey is not None and survey.answer == "population"
              and not _accepts(family.inference, "survey")):
            # MS4: a family with no design-based estimator under the population answer is blocked
            # and recorded (MODELING_SEQUENCE §4), its exits the sample-only attestation.
            from turbotab.core.models.survey import no_design_estimator

            table = no_design_estimator(family, task, getattr(state, "models", None), what="tests")
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
            if family.key == "featurewise" and not table.info.get("refused"):
                # MS7: an exposure family's recorded multiplicity method (Benjamini–Hochberg implied).
                from turbotab.core.methods.omics import apply_multiplicity, multiplicity_policy

                table = apply_multiplicity(table, multiplicity_policy(state))
            if missing is not None:
                missing.record(table)
            if survey is not None and survey.concern():
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

    The curve averages over at most 5,000 of the rows the models were fit on: under prediction the
    training rows and their fits; under inference every analyzed row and each family refit on them
    (the fit stage's ``every_row``), as the coefficient table is (BLUEPRINT §12 ruling 3). Its
    support checks amounts and, when the total-energy column is a model input, each moved
    nutrient's share of energy. With ``n_boot > 0`` each family is also refit on that many
    bootstrap resamples of the rows it was fit on (whole units when rows repeat; resamples of
    every such row up to ``BAND_ROWS``, rescaled beyond), and the band is drawn from the refits' curves
    (:func:`~turbotab.core.methods.substitution.refit_band`). Without one, a single refit per
    family is timed, so the offer of a band can say what it costs.
    """
    from sklearn.base import clone

    from turbotab.core.methods.energy import energy_factor
    from turbotab.core.methods.substitution import (
        MIN_REFIT_SHARE,
        PERCENTILE_MIN_REFITS,
        refit_band,
        substitution_curve,
    )
    from turbotab.core.models import get_family
    from turbotab.core.models.artifacts import SubstitutionArtifact
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame

    from turbotab.core.methods.percent_energy import is_percent_of_energy

    sub = ctx.state.substitution
    n_boot = int(getattr(sub, "n_boot", 0) or 0)
    scale = getattr(sub, "scale", "kcal") or "kcal"
    percent = [c for c in (sub.donor, sub.recipient)
               if scale == "percent_energy" and is_percent_of_energy(c)]
    fit, design = ctx.inputs["fit"], ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    # The parts of totals as the ledger holds them: the design's, each column's own confirmation
    # standing over it (BLUEPRINT §14.3; the sixth gate's ``sfa_g`` confirmed as part of another
    # total moved nothing new).
    from turbotab.core.readings import nesting

    nested = nesting(ctx.state, dict(design.objects.get("nested") or {}), columns=spec.inputs)
    task = fit.data["task"]
    for column in (sub.donor, sub.recipient):
        if column not in spec.inputs:
            raise ValueError(f"{column} is not one of the model's predictors, so energy cannot be "
                             f"moved through it.")
    # BLUEPRINT §14.3 (every confirmation is honored): each kcal-per-unit factor is derived from
    # the column's recorded unit (g → its Atwater factor, kg → 1,000 × it, kcal → 1, kJ → 1/4.184)
    # or from grams the Atwater identity reads, never from its name; the substitution answer's
    # refusal asks it, and a curve recorded before the rule is not drawn on a guess.
    from turbotab.core.readings import Unsettled, factor_exits, kcal_per_unit as factor_of, listing
    from turbotab.core.readings import factor_verdicts

    moved = [c for c in (sub.donor, sub.recipient) if c not in percent]
    with open_store(ctx) as store:
        verdicts = factor_verdicts(ctx.state, store, moved)
    readings = {c: factor_of(ctx.state, c, verdict=verdicts.get(c)) for c in moved}
    waiting = [c for c, r in readings.items() if not r.settled]
    if waiting:
        raise Unsettled(f"{listing(waiting)}'s kcal per unit is not settled: "
                        + "; ".join(readings[c].why for c in waiting)
                        + ". Record the unit before the curve is drawn.",
                        exits=[e for c in waiting for e in factor_exits(c)])
    kcal_per_unit = {c: float(r.factor) for c, r in readings.items()}

    # The rows and fits the curve reads: under inference every analyzed row and the families refit
    # on them, as the coefficient table is (BLUEPRINT §12 ruling 3: a holdout is a prediction
    # concept); under prediction the training rows and the training fits.
    objects = fit.objects or {}
    every_ids = objects.get("every_row_ids")
    on_every_row = ctx.state.purpose == "inference" and every_ids is not None
    trained = objects["fitted"]
    # MS3: under multiple imputation the curve is pooled over the copies, never drawn on one fill.
    imputed = objects.get("imputations") if on_every_row else None
    pooled_entries: list[dict[str, Any]] = []
    # Under the population answer as well (MS2-MS4): each copy's design-based band, per family.
    pooled_design: list[tuple[str, dict[str, Any]]] = []
    if on_every_row:
        all_train = np.asarray(every_ids, dtype=np.int64)
        fitted = dict(objects.get("every_row") or {})  # a copy: the fit's objects stay as cached
        rows_word = "analyzed rows"
    else:
        all_train = row_ids_of(design.frames["training"])
        fitted = trained
        rows_word = "training rows"
    train_ids = all_train
    if len(train_ids) > SUBSTITUTION_ROWS:
        train_ids = np.sort(np.random.default_rng(0).choice(train_ids, SUBSTITUTION_ROWS, replace=False))
    target = ctx.state.target
    grouped_by = objects.get("grouped_by")
    ctx.progress(0.02, f"Reading the {rows_word}")
    extra = [target] + ([grouped_by] if grouped_by and grouped_by not in spec.inputs
                        and grouped_by != target else [])
    with open_store(ctx) as store:
        # Every row the models were fit on: the band's refits resample them all.
        fit_frame = modeling_frame(store, [*spec.inputs, *extra], all_train, outcome=target)
    X_fit = fit_frame[spec.inputs]
    # MS4: under the surveyed population the curve is the population's: every analyzed row in the
    # design's domain, each counted as the people its survey weight stands for, none sampled away.
    survey_design = objects.get("survey_design") if on_every_row else None
    domain = None
    if survey_design is not None:
        from turbotab.core.models.survey import domain_of

        domain = domain_of(X_fit.index, survey_design)
        curve_rows = np.flatnonzero(domain.keep)
    else:
        curve_rows = X_fit.index.get_indexer(pd.Index(train_ids))
    X = X_fit.iloc[curve_rows]
    step = float(sub.step_percent) if scale == "percent_energy" else float(sub.step_kcal)
    ks = [step * i for i in range(SUBSTITUTION_STEPS + 1)]
    total_energy = _total_energy_column(ctx.state, X, (sub.donor, sub.recipient))
    shift = shift_for(ctx.state, X, donor=sub.donor, recipient=sub.recipient,
                      kcal_per_unit=kcal_per_unit, design_nested=nested, total=total_energy,
                      scale=scale, percent=percent)
    from turbotab.core.units import outcome_unit as _unit_of_outcome
    from turbotab.core.units import recorded_unit

    # Stated only as recorded or as the name spells it out, never guessed (audit IN-05).
    outcome_unit = (_unit_of_outcome(target, recorded=recorded_unit(ctx.state, target))[0]
                    if task == "regression" else None)
    models = []
    note = None
    support = None
    skipped = []
    pipelines = design.objects["pipelines"]
    keys = [m["family"] for m in fit.data["models"] if m["family"] in trained]
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
        if imputed is not None and (imputed.get("fits") or {}).get(key):
            ctx.progress(start, f"{family.label}: each copy's curve")

            def copy_progress(done: int, total: int, _lo: float = start, _w: float = slot,
                              _label: str = family.label) -> None:
                ctx.progress(_lo + _w * done / max(total, 1), f"{_label}: copy {done} of {total}")

            started = time.perf_counter()
            entry, info = _pooled_curve(
                key, family, task, imputed, train_ids, state=ctx.state, sub=sub, ks=ks,
                kcal_per_unit=kcal_per_unit, nested=nested, total_energy=total_energy, scale=scale,
                percent=percent, outcome_unit=outcome_unit, n_boot=n_boot, pipeline=pipelines[key],
                y_fit=y_fit, group_of=group_of, interval_rows=BAND_ROWS, progress=copy_progress,
                survey_design=survey_design, models=keys)
            band_seconds += time.perf_counter() - started
            first_curve = info["curves"][0]
            note = note or first_curve["note"]
            if support is None:
                fixed = first_curve["fixed_population"]
                support = {"total": first_curve["total"], "n_rows": first_curve["n_rows"],
                           "not_recorded": first_curve["n_not_recorded"],
                           "off_amount": first_curve["n_off_amount"],
                           "off_share": first_curve["n_off_share"], "fixed_rows": fixed["n_rows"],
                           "fixed_through": fixed["through"]}
            pooled_entries.append(entry)
            models.append(entry)
            if info.get("design") is not None:
                pooled_design.append((family.label, info["design"]))
            continue
        if survey_design is not None:
            # MS4: the curve over the surveyed population, design-based or blocked and recorded.
            from turbotab.core.models.survey import population_curve

            drawn = population_curve(
                family, task, fitted.get(key), X, y_fit[curve_rows], survey_design, domain,
                models=keys, donor=sub.donor, recipient=sub.recipient,
                kcal_per_unit=kcal_per_unit, ks=ks, total_kind="variable", nested=nested,
                total=total_energy, scale=scale, shift=shift)
            curve = drawn.curve
            fixed = curve["fixed_population"]
            if support is None:
                support = {"total": curve["total"], "n_rows": curve["n_rows"],
                           "not_recorded": curve["n_not_recorded"],
                           "off_amount": curve["n_off_amount"], "off_share": curve["n_off_share"],
                           "fixed_rows": fixed["n_rows"], "fixed_through": fixed["through"]}
            note = note or curve["note"]
            if drawn.band is None:
                models.append({
                    "family": key, "label": family.label, "delta": [None] * len(ks),
                    "ci_low": None, "ci_high": None,
                    "on_support_fraction": curve["on_support_fraction"],
                    "stopped_at": curve["stopped_at"], "effect_label": None,
                    "fixed_delta": [None] * len(ks), "fixed_ci_low": None, "fixed_ci_high": None,
                    "band_ok": None, "refused": drawn.refused, "exits": drawn.exits})
                continue
            band_d = drawn.band
            models.append({
                "family": key, "label": family.label, "delta": curve["delta"],
                "ci_low": band_d["ci_low"], "ci_high": band_d["ci_high"],
                "on_support_fraction": curve["on_support_fraction"],
                "stopped_at": curve["stopped_at"],
                "effect_label": _in_outcome_unit(curve["effect_label"], outcome_unit),
                "fixed_delta": fixed["delta"], "fixed_ci_low": band_d["fixed_ci_low"],
                "fixed_ci_high": band_d["fixed_ci_high"], "band_ok": None})
            bands.append((family.label, band_d))
            continue
        if key not in fitted:
            # Under inference, a family with no coefficient table (boosted trees) was fit on the
            # training rows only; its curve reads its refit on every analyzed row.
            from turbotab.core.models.inner_cv import fit_pipeline

            ctx.progress(start, f"{family.label}: refitting on every analyzed row")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted[key] = fit_pipeline(clone(pipelines[key]), X_fit, y_fit, groups=groups)
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
    from turbotab.core.readings import confirmation as _recorded

    for c, reading in readings.items():
        if _recorded(ctx.state, "unit", c) is None:
            # Settled by the values, not recorded (BLUEPRINT §14.3: settlement is visible).
            notes.append(f"{c} is read in grams by its values (the Atwater identity holds with "
                         f"total energy), at {reading.factor:g} kcal per gram.")
        else:
            # The user's unit, and the kcal per unit it sets (the sixth gate: no record said what a
            # unit of alcohol moved).
            notes.append(f"{c} moves at {reading.factor:g} kcal per unit, {reading.why}.")
    if pooled_entries:
        notes.append(_pooled_note(len(imputed.get("frames") or []), pooled_entries, n_boot,
                                  imputed.get("outcomes") is not None, design=bool(pooled_design)))
        if any(e.get("pooled") == "per_k" for e in pooled_entries):
            notes.append("Support counts are the first copy's; the share of rows on support at each "
                         "k is the mean over the copies.")
    elif spec.multiple_imputation() and X_fit.isna().any().any():
        # WP7: the coefficient table pools the multiple imputations; the curve does not.
        which = ("the fit on every analyzed row, whose blanks are filled once without the outcome"
                 if on_every_row else "the training fit, whose blanks are filled once in each "
                                      "fold without the outcome")
        notes.append(f"The curve follows {which}; it is not pooled over the multiple imputations "
                     f"the coefficient table uses, so its band leaves out their uncertainty.")
    if skipped and task in ("multiclass", "ordinal"):
        kind = "An ordinal" if task == "ordinal" else "A multiclass"
        notes.append(f"{kind} outcome has one curve per level, which is not drawn yet.")
    elif skipped:
        notes.append("A time-to-event outcome's substitution is a hazard ratio, which is not drawn "
                     "yet; the Cox coefficients are log hazard ratios per unit of each column.")
    band = None
    if survey_design is not None:
        from turbotab.core.models.survey import curve_caption, pooled_design_caption

        for entry in models:
            if entry.get("refused"):
                notes.append(f"{entry['label']}: no curve. {entry['refused']}")
        design_bands = bands or pooled_design
        if design_bands:
            first = design_bands[0][1]
            caption = (curve_caption(survey_design, first["variance"], len(X)) if bands else
                       pooled_design_caption(survey_design, first["variance"], len(X),
                                             len(imputed.get("frames") or []),
                                             imputed.get("outcomes") is not None))
            notes.append(caption.replace("Shaded bands: ", "The shaded bands are ", 1))
            if n_boot:
                notes.append(f"The {n_boot:,} bootstrap refits asked for are not drawn: resampling "
                             f"rows ignores the strata and PSUs, so under the surveyed population "
                             f"the band is the design's.")
            band = {"n_boot": 0, "n_rows": int(len(X)), "grouped_by": None, "seconds": 0.0,
                    "failed": 0, "n_units": int(first["variance"].domain_psu),
                    "resample_units": None, "scale": None, "interval": None, "level": 0.95,
                    "min_ok_share": None, "caption": caption, "method": "design",
                    "df": int(first["df"])}
    elif n_boot and pooled_entries and not bands:
        caption = _pooled_note(len(imputed.get("frames") or []), pooled_entries, n_boot,
                               imputed.get("outcomes") is not None)
        band = {"n_boot": n_boot, "n_rows": int(len(X_fit)), "grouped_by": grouped_by,
                "seconds": round(band_seconds, 3), "failed": 0, "interval": "normal",
                "level": 0.95, "min_ok_share": MIN_REFIT_SHARE, "caption": caption}
    elif n_boot and bands:
        first = bands[0][1]
        caption = _band_caption(n_boot, interval, first, len(X_fit), grouped_by, bands,
                                rows_word=rows_word)
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
    # Under the surveyed population (MS4) the average is the population's.
    over = " over the surveyed population" if survey_design is not None else ""
    estimand = (f"The average change in {outcome}{over} when {moved} from {sub.donor} to "
                f"{sub.recipient}, with {others}.")
    if shift.carried:
        estimand = (f"The average change in {outcome}{over} when {moved} from {sub.donor} to "
                    f"{sub.recipient}, their parts or totals moving with them and {others}.")
    basis = f"Averaged over {len(X):,} {rows_word}."
    if survey_design is not None:
        weight = survey_design.weight_column
        basis = (f"Averaged over the {len(X):,} analyzed rows in the survey design, each counted as "
                 f"the people its weight{f' `{weight}`' if weight else ''} stands for.")
    artifact = SubstitutionArtifact(
        donor=sub.donor, recipient=sub.recipient, step_kcal=float(sub.step_kcal),
        ks=[float(k) for k in ks], total_kind="variable", estimand=estimand, note=" ".join(notes),
        basis=basis, models=models,
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


def contrast_per_kcal(fitted: Any, shift: Any, X: pd.DataFrame, k: float) -> pd.Series | None:
    """The change in the model matrix per kcal moved, when it is the same on every row the move is
    on support for (an all-components model: each source's kcal is its own term), else None: then
    the curve is no exact linear contrast of the coefficients."""
    from turbotab.core.models.linear import model_matrix

    rows = X.iloc[np.flatnonzero(shift.valid(X))]
    if not len(rows) or k <= 0:
        return None
    moved, mask = shift.apply(rows, float(k))
    on = np.flatnonzero(mask)
    if not len(on):
        return None
    base = model_matrix(fitted, rows.iloc[on]).astype(float)
    after = model_matrix(fitted, moved.iloc[on]).astype(float)
    diff = (after.to_numpy() - base.to_numpy()) / float(k)
    scale = max(1.0, float(np.nanmax(np.abs(diff)))) if diff.size else 1.0
    if not np.all(np.isfinite(diff)) or float(np.max(np.abs(diff - diff[0]))) > 1e-9 * scale:
        return None
    return pd.Series(diff[0], index=[str(c) for c in base.columns])


def _pooled_curve(key: str, family: Any, task: str, imputed: Mapping[str, Any], train_ids: Any,
                  *, state: Any, sub: Any, ks: Sequence[float], kcal_per_unit: Mapping[str, float],
                  nested: Mapping[str, str], total_energy: str | None, scale: str,
                  percent: Sequence[str], outcome_unit: str | None, n_boot: int,
                  pipeline: Any, y_fit: Any, group_of: Any, interval_rows: int,
                  progress: Any = None, survey_design: Any = None,
                  models: Sequence[str] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """MS3 (MODELING_SEQUENCE §2: "multiple imputation implies pooling of every estimate shown
    under inference, including substitution curves"): ``key``'s curve pooled over the copies.

    Each copy's curve averages over that copy's own rows (the curve's rows: ``train_ids``, or every
    row of a copy the data supplied) through that copy's own fit. When the move changes the model
    matrix by the same amount on every row (a linear all-components model), the pooled curve is the
    exact linear contrast of the pooled coefficients, Δ(k) = k cᵀQ̄, with Rubin's total variance
    k² cᵀTc and Barnard–Rubin degrees of freedom (``pooled = "contrast"``). Otherwise each copy's
    curve is pooled at each k: Q̄(k) the mean, and with a band (``n_boot``) each copy's bootstrap
    variance its within-copy variance (Schomaker & Heumann 2018's MI-then-bootstrap with Rubin's
    rules; the ``n_boot`` refits split over the copies), on Barnard–Rubin's ν (``pooled =
    "per_k"``). Returns (the model entry, what the band record needs).

    Under the population answer (``survey_design``; MS4 with MS2) each copy's curve is the surveyed
    population's (:func:`~turbotab.core.models.survey.population_curve`: the survey-weighted refit,
    averaged over every row of the copy in the design's domain with its weight), and each copy's
    within-copy variance is that curve's design-based (Taylor-linearized) variance, pooled by
    Rubin's rules with ν_com = the design df; a bootstrap that ignores the strata and PSUs is not
    run. The linear all-components contrast pools the copies' design-based tables the same way."""
    from statistics import NormalDist

    from sklearn.base import clone

    from turbotab.core.methods.imputation import pool_scalar
    from turbotab.core.methods.substitution import (PER_UNIT, _amount, _plain, _signed, refit_band,
                                                    substitution_curve)

    frames = list(imputed["frames"])
    fits = list((imputed.get("fits") or {}).get(key) or [])
    tables = list((imputed.get("tables") or {}).get(key) or [])
    outcomes = imputed.get("outcomes")
    m = len(fits)
    curve_ids = set(int(i) for i in np.asarray(train_ids))
    curves, shifts, rows_of = [], [], []
    design_bands: list[dict[str, Any]] = []  # under the population answer: each copy's
    for k_copy, (X_all, fitted) in enumerate(zip(frames, fits)):
        if survey_design is not None:
            from turbotab.core.models.survey import domain_of, population_curve

            # Every row of the copy in the design's domain, none sampled away (as the single fit's
            # population curve reads them).
            domain = domain_of(X_all.index, survey_design)
            rows = list(np.flatnonzero(domain.keep))
            X_k = X_all.iloc[rows]
            y_all = np.asarray(outcomes[k_copy]) if outcomes is not None else np.asarray(y_fit)
            shift = shift_for(state, X_k, donor=sub.donor, recipient=sub.recipient,
                              kcal_per_unit=kcal_per_unit, design_nested=nested,
                              total=total_energy, scale=scale, percent=percent)
            drawn = population_curve(family, task, fitted, X_k, y_all[rows], survey_design, domain,
                                     models=models, donor=sub.donor, recipient=sub.recipient,
                                     kcal_per_unit=kcal_per_unit, ks=ks, total_kind="variable",
                                     nested=nested, total=total_energy, scale=scale, shift=shift)
            if drawn.band is None:  # blocked and recorded, as the single fit's curve would be
                curve = drawn.curve
                return ({"family": key, "label": family.label, "delta": [None] * len(ks),
                         "ci_low": None, "ci_high": None,
                         "on_support_fraction": curve["on_support_fraction"],
                         "stopped_at": curve["stopped_at"], "effect_label": None,
                         "fixed_delta": [None] * len(ks), "fixed_ci_low": None,
                         "fixed_ci_high": None, "band_ok": None, "refused": drawn.refused,
                         "exits": drawn.exits},
                        {"curves": [curve], "m": m})
            curves.append(drawn.curve)
            design_bands.append(drawn.band)
            shifts.append(shift)
            rows_of.append(rows)
            continue
        rows = [i for i, rid in enumerate(X_all.index) if int(rid) in curve_ids]
        if outcomes is not None or not rows:  # a supplied copy: its own rows
            rows = list(range(min(len(X_all), SUBSTITUTION_ROWS)))
        X_k = X_all.iloc[rows]
        shift = shift_for(state, X_k, donor=sub.donor, recipient=sub.recipient,
                          kcal_per_unit=kcal_per_unit, design_nested=nested, total=total_energy,
                          scale=scale, percent=percent)
        curves.append(substitution_curve(_predictor(task, fitted), X_k, donor=sub.donor,
                                         recipient=sub.recipient, kcal_per_unit=kcal_per_unit,
                                         ks=ks, total_kind="variable", nested=nested,
                                         total=total_energy, scale=scale, shift=shift))
        shifts.append(shift)
        rows_of.append(rows)
    n_k = len(curves[0]["ks"])
    live = [all(c["delta"][i] is not None for c in curves) for i in range(n_k)]
    stops = [c["stopped_at"] for c in curves if c["stopped_at"] is not None]
    level = 0.95
    z = NormalDist().inv_cdf(0.5 + level / 2)
    delta: list[float | None] = [None] * n_k
    low: list[float | None] = [None] * n_k
    high: list[float | None] = [None] * n_k
    dfs: list[float | None] = [None] * n_k
    fixed = [float(np.mean([c["fixed_population"]["delta"][i] for c in curves]))
             if live[i] and all(c["fixed_population"]["delta"][i] is not None for c in curves) else None
             for i in range(n_k)]
    fixed_low: list[float | None] = [None] * n_k
    fixed_high: list[float | None] = [None] * n_k
    how = "per_k"
    band_info: dict[str, Any] = {}
    contrast = None
    # The prediction is the linear predictor itself only for a linear outcome model.
    if family.key == "linear" and task == "regression" and tables and len(tables) == m \
            and scale == "kcal":
        step = next((float(k) for k in curves[0]["ks"] if float(k) > 0), None)
        found = [contrast_per_kcal(f, s, frames[j].iloc[rows_of[j]], step) if step else None
                 for j, (f, s) in enumerate(zip(fits, shifts))]
        if all(c is not None for c in found) and all(c.equals(found[0]) or
                                                     np.allclose(c.to_numpy(), found[0].to_numpy(),
                                                                 rtol=1e-9, atol=1e-12)
                                                     for c in found):
            contrast = found[0]
    if contrast is not None:
        Q, U, df_rows = [], [], []
        for t in tables:
            names = list(t["names"])
            c = np.array([float(contrast.get(n, 0.0)) for n in names])
            est = np.asarray(t["estimates"], dtype=float)
            cov = np.asarray(t["cov"], dtype=float)
            Q.append(float(c @ est))
            U.append(float(c @ cov @ c))
            df_rows.append(t.get("df"))
        df_com = None if any(d is None for d in df_rows) else float(np.mean(df_rows))
        pooled = pool_scalar(Q, U, df_com, level)
        how = "contrast"
        half_per = (pooled.ci_high - pooled.estimate) if pooled.ci_high is not None else None
        for i, k in enumerate(curves[0]["ks"]):
            if not live[i]:
                continue
            delta[i] = pooled.estimate * float(k)
            if half_per is not None:
                low[i] = delta[i] - half_per * float(k)
                high[i] = delta[i] + half_per * float(k)
                dfs[i] = pooled.df
        band_info = {"contrast": {n: float(v) for n, v in contrast.items() if v != 0.0},
                     "per_kcal": pooled.estimate, "se_per_kcal": math.sqrt(pooled.total)
                     if pooled.total > 0 else None, "df": pooled.df}
    else:
        within = None
        df_within: float | None = None
        if design_bands:
            from scipy import stats

            # Each copy's design-based variance at each k (the fixed population's from its
            # interval on the design's t), on the design's df as the complete-data df.
            within = np.full((m, n_k), np.nan)
            fixed_within = np.full((m, n_k), np.nan)
            for j, b in enumerate(design_bands):
                q_j = float(stats.t.ppf(0.5 + level / 2, b["df"]))
                for i in range(n_k):
                    if b["se"][i] is not None:
                        within[j, i] = float(b["se"][i]) ** 2
                    if b["fixed_ci_low"][i] is not None:
                        fixed_within[j, i] = ((b["fixed_ci_high"][i] - b["fixed_ci_low"][i])
                                              / (2 * q_j)) ** 2
            df_within = float(min(b["df"] for b in design_bands))
            band_info = {"design": design_bands[0]}
        elif n_boot:
            per_copy = max(10, math.ceil(n_boot / max(m, 1)))
            within = np.full((m, n_k), np.nan)
            fixed_within = np.full((m, n_k), np.nan)
            n_ok = 0
            for j, (X_all, fitted) in enumerate(zip(frames, fits)):
                y_j = np.asarray(outcomes[j]) if outcomes is not None else y_fit

                def refit(Xb: pd.DataFrame, yb: Any, _full: Any = fitted) -> Any:
                    from turbotab.core.models.inner_cv import fit_pipeline

                    inner = (group_of.reindex(Xb.index).to_numpy() if group_of is not None
                             and outcomes is None else Xb.index.to_numpy())
                    pipe = pinned_to_full_fit(clone(pipeline), _full)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return _predictor(task, fit_pipeline(pipe, Xb, yb, groups=inner))

                band = refit_band(refit, X_all, y_j, shift=shifts[j], ks=ks, live=live,
                                  n_boot=per_copy, random_state=j, center=curves[j]["delta"],
                                  fixed_center=curves[j]["fixed_population"]["delta"],
                                  curve_rows=rows_of[j], max_rows=interval_rows, interval="normal",
                                  groups=(group_of.reindex(X_all.index).to_numpy()
                                          if group_of is not None and outcomes is None else None))
                n_ok += band["n_ok"]
                for i in range(n_k):
                    if band["ci_low"][i] is not None:
                        within[j, i] = ((band["ci_high"][i] - band["ci_low"][i]) / (2 * z)) ** 2
                    if band["fixed_ci_low"][i] is not None:
                        fixed_within[j, i] = ((band["fixed_ci_high"][i] - band["fixed_ci_low"][i])
                                              / (2 * z)) ** 2
                if progress is not None:
                    progress(j + 1, m)
            band_info = {"per_copy": per_copy, "n_ok": n_ok}
        for i in range(n_k):
            if not live[i]:
                continue
            q = [float(c["delta"][i]) for c in curves]
            if within is not None and np.all(np.isfinite(within[:, i])):
                pooled = pool_scalar(q, list(within[:, i]), df_within, level)
                delta[i], low[i], high[i], dfs[i] = (pooled.estimate, pooled.ci_low, pooled.ci_high,
                                                     pooled.df)
                fq = [c["fixed_population"]["delta"][i] for c in curves]
                if all(v is not None for v in fq) and np.all(np.isfinite(fixed_within[:, i])):
                    fp = pool_scalar([float(v) for v in fq], list(fixed_within[:, i]), df_within,
                                     level)
                    fixed_low[i], fixed_high[i] = fp.ci_low, fp.ci_high
            else:
                delta[i] = float(np.mean(q))
    per = PER_UNIT[scale]
    chosen = curves[0]["label_k"]
    ks_out = [float(k) for k in curves[0]["ks"]]
    label = None
    if chosen is not None and chosen in ks_out and delta[ks_out.index(chosen)] is not None:
        value = delta[ks_out.index(chosen)]
        label = f"{_signed(value * per / chosen)} per {_amount(per, scale)} at k = {_plain(chosen)}"
    banded = bool(n_boot) or how == "contrast" or bool(design_bands)
    fixed_banded = bool(n_boot) or bool(design_bands)
    if design_bands and how == "contrast":
        band_info["design"] = design_bands[0]
    entry = {
        "family": key, "label": family.label, "delta": delta, "ci_low": low if banded else None,
        "ci_high": high if banded else None,
        "on_support_fraction": [float(np.mean([c["on_support_fraction"][i] for c in curves]))
                                for i in range(n_k)],
        "stopped_at": min(stops) if stops else None,
        "effect_label": _in_outcome_unit(label, outcome_unit),
        "fixed_delta": fixed, "fixed_ci_low": fixed_low if fixed_banded else None,
        "fixed_ci_high": fixed_high if fixed_banded else None, "band_ok": band_info.get("n_ok"),
        "pooled": how, "df": dfs,
    }
    return entry, {"curves": curves, "m": m, **band_info}


def _pooled_note(m: int, entries: Sequence[Mapping[str, Any]], n_boot: int, supplied: bool,
                 design: bool = False) -> str:
    """The substitution note's sentence on how the curves were pooled over the copies (MS3); with
    ``design``, each copy's curve and variance are the surveyed population's (MS4)."""
    what = f"the data's {m} imputed copies" if supplied else f"the {m} imputations"
    parts = []
    if any(e.get("pooled") == "contrast" for e in entries):
        parts.append("for the linear all-components model the curve is the exact contrast of the "
                     "pooled coefficients, with Rubin's total variance and Barnard–Rubin degrees of "
                     "freedom")
    if any(e.get("pooled") == "per_k" for e in entries):
        how = ("each copy's survey-weighted curve over the surveyed population, pooled at each k by "
               "Rubin's rules with each copy's Taylor-linearized variance as its within-copy "
               "variance and the design's degrees of freedom as the complete-data df" if design else
               "each copy's curve, pooled at each k by Rubin's rules with each copy's bootstrap "
               f"variance as its within-copy variance ({n_boot:,} refits split over the copies; "
               "Schomaker & Heumann 2018)" if n_boot else
               "the mean at each k of each copy's curve, with no band until one is asked for")
        parts.append(f"otherwise {how}" if parts else how)
    joined = "; ".join(parts)
    return f"The curve is pooled over {what}: {joined}."


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
                  grouped_by: str | None, bands: Sequence[tuple[str, Mapping[str, Any]]],
                  rows_word: str = "training rows") -> str:
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
        drawn = (f"whole {grouped_by} units ({n_units:,} units, {n_rows:,} {rows_word})"
                 if m == n_units else f"{m:,} of the {n_units:,} {grouped_by} units "
                 f"({n_rows:,} {rows_word})")
    else:
        drawn = (f"all {n_rows:,} {rows_word}" if m == n_units
                 else f"{m:,} of the {n_rows:,} {rows_word}")
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

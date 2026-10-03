"""The ``calibration`` stage: energy-adjusted exposures corrected for day-to-day error (audit IN-22).

When each person's row is the mean of two or more 24-hour recalls, the mean still carries
day-to-day error, and a coefficient on it is biased. Freedman et al. 2011 (*J Natl Cancer Inst*
103:1086) recommend "statistical adjustment of relative risks ... using univariate (only for
energy-adjusted intakes such as densities or residuals) or multivariate regression calibration".
This stage does the univariate one (``turbotab/core/methods/calibration.py`` has the arithmetic and
its sources) for each energy-adjusted exposure of the linear model, from the recalls themselves:
the within-person variance comes from the days each person's mean was made of.

**Where it applies, and where it refuses** (the artifact's ``reason`` says which):

* inference only. Under prediction the model is used on recalls measured the same way as these,
  so its predictions need no correction (BLUEPRINT §12: the leash is right for prediction);
* the linear model only: the correction is to a coefficient, and only that family reports one
  with an interval;
* rows combined per person from repeats of one quantity (``set_grain`` repeated, ``set_repeat_kind``
  repeats, ``set_aggregation`` by the mean). Visits over time are not replicate measurements of
  one usual intake;
* energy-adjusted exposures only (residual or density outputs of the energy step). Freedman et al.:
  "Univariate adjustment for the unadjusted intakes used in the standard and partition models is
  inappropriate because the attenuation factor for the nutrient would be too small; the
  multivariate adjustment is recommended in this case." The multivariate method is not offered,
  so under the standard, partition or no-adjustment answers the stage refuses and says so.

**The recalls.** The oriented table's rows (the row-local repairs applied, as the working stage
applies them) are mapped to the analysis rows by the working table's row map. Each day goes through
the same fitted pipeline steps as the person-level rows, so a day's energy-adjusted value is
computed with the slope fitted on the analysis rows; a day missing the nutrient or energy is not a
recall of it. Each person's days are then centered on the person's own model-matrix value, so the
person-level exposure is exactly the one the coefficient table used and the days contribute only
their spread (for the linear residual transform the centering moves nothing; for a density it
replaces the mean of daily ratios by the ratio the model saw).

**Rows** are every eligible row (BLUEPRINT §12 ruling 3: inference estimates from all eligible rows,
with honest intervals), the pipeline refit on them as the sensitivity stage refits it.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext

FREEDMAN = "Freedman et al. 2011, J Natl Cancer Inst 103:1086"
CARROLL = "Carroll, Ruppert, Stefanski & Crainiceanu 2006, Measurement Error in Nonlinear Models, §4.4"
CALIBRATED_OPERATIONS = ("residual", "density")  # the energy step's energy-adjusted outputs
UNIVARIATE_METHODS = ("residual", "density", "density_multivariate")

PREDICTION = ("Under prediction the model is used on recalls measured the same way as these, so its "
              "predictions need no correction; regression calibration corrects an exposure's "
              "coefficient, which is an inference question.")
NO_LINEAR = ("Regression calibration corrects a coefficient, and only the linear model reports one "
             "with an interval; choose it among the model families.")
NOT_COMBINED = ("Each analysis row is one record, not the mean of a person's repeated recalls, so the "
                "day-to-day variance cannot be estimated. Record the rows as repeats of one person "
                "and combine them by the mean.")
TIME_POINTS = ("The rows repeat as time points, not as repeated recalls of one usual intake, so their "
               "spread is change over time rather than day-to-day error.")
NOT_ENERGY_ADJUSTED = (
    "Univariate regression calibration is for energy-adjusted intakes (residuals or densities). "
    f"{FREEDMAN}: “Univariate adjustment for the unadjusted intakes used in the standard and "
    "partition models is inappropriate because the attenuation factor for the nutrient would be "
    "too small; the multivariate adjustment is recommended in this case.” This version does not "
    "fit the multivariate one; choose the residual or density method to calibrate.")
ASSUMPTIONS = (
    "Each recall's error is independent of the person's true intake and of their other days' "
    "errors (classical error). Recalls share a person's own reporting bias, which this reads as true "
    "intake, so the corrected estimate can still be off in either direction (Freedman et al. 2011, "
    "Table 2).",
    "Each exposure is corrected on its own; total energy and the other covariates are treated as "
    "measured without error (Freedman's univariate method takes the contamination factors as zero).",
    "For a single error-prone exposure the test of no association is the uncorrected one's: "
    "“the usual statistical test of the null hypothesis (no exposure effect) remains theoretically "
    "valid even though the estimated relative risk is attenuated” (Freedman et al. 2011).",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class CalibratedExposure(_Model):
    """One energy-adjusted exposure: its uncorrected and its calibrated coefficient."""

    feature: str  # the model-matrix column (``protein_adj``)
    source: str  # the raw nutrient column
    operation: str  # "residual" or "density"
    naive: float | None
    naive_ci_low: float | None = None
    naive_ci_high: float | None = None
    p: float | None = None  # the uncorrected test of no association (Freedman 2011)
    estimate: float | None = None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    attenuation: float | None = None  # λ at the most common number of recalls
    attenuation_se: float | None = None
    within_variance: float | None = None  # σ²_u, day to day
    between_variance: float | None = None  # σ²_{x|z}, of true intake given the covariates
    n_persons: int = 0
    n_repeat: int = 0  # people with two or more recalls
    recalls: dict[str, int] = {}  # number of recalls -> people
    n_boot: int = 0
    n_boot_ok: int = 0
    refused: str | None = None


class CalibrationArtifact(_Model):
    method: Literal["none", "regression_calibration"]
    purpose: Literal["inference", "prediction"]
    applies: bool
    reason: str | None = None  # why nothing was calibrated
    family: str | None = None
    rows: Literal["all eligible rows"] = "all eligible rows"
    n_persons: int = 0
    recalls: dict[str, int] = {}  # number of recalls -> people (energy recorded on each)
    exposures: list[CalibratedExposure] = []
    assumptions: list[str] = []
    concerns: list[str] = []
    methods: str


# ── the recalls ──────────────────────────────────────────────────────────────


def day_rows(ctx: StageContext, columns: Sequence[str], row_ids: np.ndarray) -> pd.DataFrame:
    """The oriented table's rows behind ``row_ids`` (working ids), the recorded row-local repairs
    applied, with ``__unit`` naming the working row each became part of."""
    import duckdb

    from turbotab.core.models.pipeline import normalize_frame
    from turbotab.core.stages.working import (ROW_ID, _bundle_table, _ident, repair_expressions,
                                              row_map, source_sql)

    oriented = ctx.inputs["oriented"]
    source = _bundle_table(oriented)
    if source is None:
        raise RuntimeError("the oriented artifact has no table")
    names = [str(c["name"]) for c in oriented.data["columns"]]
    repairs = repair_expressions(ctx.inputs.get("findings"), ctx.state.findings)
    mapping = row_map(ctx.inputs["working"])
    mapping = mapping[mapping["row_id"].isin(row_ids)]
    con = duckdb.connect()
    try:
        con.register("unit_map", mapping)
        select = ", ".join(f"s.{_ident(c)}" for c in dict.fromkeys(columns))
        frame = con.execute(
            f"SELECT m.row_id AS __unit, {select} FROM {source_sql(source, names, repairs)} s "
            f"JOIN unit_map m ON s.{ROW_ID} = m.source_row_id ORDER BY m.row_id, s.{ROW_ID}").df()
    finally:
        con.close()
    units = frame.pop("__unit").to_numpy(dtype=np.int64)
    out = normalize_frame(frame)
    out.insert(0, "__unit", units)
    return out


def adjusted_exposures(fitted: Any, roles: Mapping[str, str]) -> list[dict[str, str]]:
    """The energy step's energy-adjusted outputs whose nutrient is an exposure."""
    step = dict(fitted.steps).get("energy")
    if step is None or not hasattr(step, "lineage"):
        return []
    out = []
    for entry in step.lineage():
        if entry["operation"] in CALIBRATED_OPERATIONS and roles.get(str(entry["inputs"][0])) == "exposure":
            out.append({"feature": str(entry["output"]), "source": str(entry["inputs"][0]),
                        "energy": str(entry["inputs"][1]), "operation": str(entry["operation"])})
    return out


def combine_rule(state: Any, working: Mapping[str, Any], column: str) -> str:
    """How the working table combined ``column`` per person: the answer's own rule for it, the
    receipt's, else the answer's method."""
    agg = getattr(state, "aggregation", None)
    chosen = dict(getattr(agg, "columns", None) or {})
    if column in chosen:
        return str(chosen[column])
    receipt = (working.get("aggregation") or {})
    listed = {c["column"]: c["rule"] for c in receipt.get("columns") or []}
    return str(listed.get(column) or receipt.get("method") or getattr(agg, "method", ""))


def replicate_values(day_matrix: pd.DataFrame, raw_days: pd.DataFrame, units: np.ndarray,
                     person_of: Mapping[int, int], item: Mapping[str, str],
                     person_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(values, person)`` of one exposure's recalls, each person's days centered on the person's
    own model-matrix value (so their mean is exactly that value)."""
    w = pd.to_numeric(day_matrix[item["feature"]], errors="coerce").to_numpy(dtype=float)
    present = (pd.to_numeric(raw_days[item["source"]], errors="coerce").notna()
               & pd.to_numeric(raw_days[item["energy"]], errors="coerce").notna()).to_numpy()
    person = np.array([person_of.get(int(u), -1) for u in units], dtype=np.int64)
    keep = present & np.isfinite(w) & (person >= 0)
    w, person = w[keep], person[keep]
    if not len(w):
        return w, person
    sums = np.bincount(person, weights=w, minlength=len(person_values))
    counts = np.bincount(person, minlength=len(person_values))
    with np.errstate(invalid="ignore", divide="ignore"):
        day_mean = sums / counts
    return w - day_mean[person] + person_values[person], person


# ── the stage ────────────────────────────────────────────────────────────────


def _recall_counts(counts: np.ndarray) -> dict[str, int]:
    values, n = np.unique(counts[counts > 0].astype(np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, n)}


def _days_phrase(recalls: Mapping[str, int]) -> str:
    if not recalls:
        return "no recall"
    ks = sorted(int(k) for k in recalls)
    if len(ks) == 1:
        return f"{ks[0]} {'recall' if ks[0] == 1 else 'recalls'} each"
    return f"{ks[0]} to {ks[-1]} recalls each"


def methods_sentence(method: str, exposures: Sequence[Mapping[str, Any]], recalls: Mapping[str, int],
                     n_persons: int) -> str:
    from turbotab.core.voice import listing

    days = _days_phrase(recalls)
    if method == "none" and not recalls:  # nothing was read: purpose or rows ruled it out first
        return "Energy-adjusted exposures were not corrected for day-to-day error in the recalls."
    if method == "none":
        return (f"Energy-adjusted exposures were the mean of each participant's recalls ({days}, "
                f"{n_persons:,} participants) and were not corrected for day-to-day error.")
    done = [e for e in exposures if e.get("estimate") is not None]
    if not done:
        return ("Regression calibration was asked for, but no energy-adjusted exposure could be "
                "calibrated; the estimates are uncorrected.")
    lam = "; ".join(f"`{e['feature']}` λ = {e['attenuation']:.2f}" for e in done)
    n_repeat = max(int(e["n_repeat"]) for e in done)
    boots = min(int(e["n_boot_ok"]) for e in done)
    noun = "exposure" if len(done) == 1 else "exposures"
    return (f"The energy-adjusted {noun} {listing([e['feature'] for e in done], limit=6)} "
            f"{'was' if len(done) == 1 else 'were'} corrected for day-to-day variation in the recalls "
            f"by univariate regression calibration ({FREEDMAN}; {CARROLL}), each on its own: the "
            f"within-person variance came from the {n_repeat:,} participants with two or more "
            f"recalls ({days}), the calibration regression adjusted for the model's other "
            f"covariates, and intervals came from {boots:,} bootstrap refits over participants "
            f"(attenuation factors: {lam}).")


def calibration_stage(ctx: StageContext) -> Bundle:
    from sklearn.base import clone

    from turbotab.core.methods.calibration import (CalibrationRefused, Replicates, logistic_fit,
                                                   ols_fit, regression_calibration)
    from turbotab.core.models.inference import INDEPENDENT, inference_table
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import _task, coded_outcome
    from turbotab.core.stages.working import effective_repeat_kind

    state = ctx.state
    spec_me = state.measurement_error
    method = spec_me.method
    inference = state.purpose == "inference"
    working = dict(getattr(ctx.inputs["working"], "data", ctx.inputs["working"]))
    structure = ctx.inputs.get("structure")
    structure = getattr(structure, "data", structure)

    def done(**fields: Any) -> Bundle:
        fields.setdefault("methods", methods_sentence(method, [], fields.get("recalls") or {},
                                                      int(fields.get("n_persons") or 0)))
        artifact = CalibrationArtifact(method=method, purpose="inference" if inference else "prediction",
                                       **fields)
        ctx.progress(1.0, "Done")
        return Bundle(data=artifact.model_dump(mode="json"))

    if not inference:
        return done(applies=False, reason=PREDICTION)
    if working.get("aggregation") is None:
        return done(applies=False, reason=NOT_COMBINED)
    if effective_repeat_kind(state, structure) == "time_points":
        return done(applies=False, reason=TIME_POINTS)

    task = _task(ctx)
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    adj = state.energy_adjustment
    energy = adj.energy_column if adj is not None and adj.method != "none" else None
    rows = ctx.inputs["cohort"].frames["rows"]["row_id"].to_numpy(dtype=np.int64)

    ctx.progress(0.05, "Reading the analysis rows and each person's recalls")
    with open_store(ctx) as store:
        frame = modeling_frame(store, [*spec.inputs, state.target], rows, outcome=state.target)
    y = coded_outcome(task, frame[state.target].to_numpy(), state.event)
    X = frame[list(spec.inputs)]
    raw_columns = list(dict.fromkeys([*spec.inputs, *([energy] if energy else [])]))
    days = day_rows(ctx, raw_columns, rows)
    units = days.pop("__unit").to_numpy(dtype=np.int64)
    person_of = {int(r): i for i, r in enumerate(frame.index.to_numpy(dtype=np.int64))}
    if energy and energy in days.columns:
        recorded = pd.to_numeric(days[energy], errors="coerce").notna().to_numpy()
        per_person = np.bincount([person_of[int(u)] for u in units[recorded]], minlength=len(frame))
    else:
        per_person = np.bincount([person_of[int(u)] for u in units], minlength=len(frame))
    recalls = _recall_counts(per_person)
    n_persons = int(len(frame))

    if method == "none":
        return done(applies=False, reason="Not corrected, as answered.", recalls=recalls,
                    n_persons=n_persons)
    if "linear" not in (state.models or []) or "linear" not in pipelines:
        return done(applies=False, reason=NO_LINEAR, recalls=recalls, n_persons=n_persons)
    if task not in ("regression", "binary"):
        return done(applies=False, reason=(
            f"Regression calibration here corrects a least-squares or logistic coefficient; a "
            f"{task.replace('_', '-')} outcome's coefficients are not corrected."),
            recalls=recalls, n_persons=n_persons)
    if adj is None or adj.method not in UNIVARIATE_METHODS:
        return done(applies=False, reason=NOT_ENERGY_ADJUSTED, recalls=recalls, n_persons=n_persons)

    ctx.progress(0.15, "Refitting the linear model on every eligible row")
    fitted = fit_pipeline(clone(pipelines["linear"]), X, y)
    matrix = model_matrix(fitted, X).astype(float)
    items = adjusted_exposures(fitted, spec.roles)
    wanted = set(spec_me.exposures)
    if wanted:
        items = [i for i in items if i["source"] in wanted or i["feature"] in wanted]
    if not items:
        reason = ("None of the model's exposures is energy-adjusted, so there is nothing univariate "
                  "regression calibration applies to." if not wanted else
                  f"{', '.join(f'`{c}`' for c in sorted(wanted))} "
                  f"{'is' if len(wanted) == 1 else 'are'} not among the model's energy-adjusted "
                  f"exposures.")
        return done(applies=False, reason=reason, family="linear", recalls=recalls,
                    n_persons=n_persons)

    day_matrix = model_matrix(fitted, days[list(spec.inputs)])
    columns = [str(c) for c in matrix.columns]
    Xm = matrix.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    fit = ols_fit if task == "regression" else logistic_fit
    classes = list(getattr(fitted[-1], "classes_", [])) or None
    seed = int(getattr(state.split, "seed", 0) or 0) if state.split is not None else 0
    out: list[dict[str, Any]] = []
    for n, item in enumerate(items):
        ctx.progress(0.2 + 0.75 * n / len(items), f"Calibrating `{item['feature']}`")
        base = {"feature": item["feature"], "source": item["source"], "operation": item["operation"],
                "naive": None}
        rule = combine_rule(state, working, item["source"])
        if rule != "mean":
            out.append({**base, "refused": (f"`{item['source']}` was combined by its {rule} record, "
                                            f"not the mean of the recalls, so its error is a single "
                                            f"day's; combine it by the mean to calibrate it.")})
            continue
        j = columns.index(item["feature"])
        values, person = replicate_values(day_matrix, days, units, person_of, item, Xm[:, j])
        rep = Replicates.of(values, person, n_persons)
        have = rep.counts() >= 1
        try:
            table = inference_table(task, matrix.loc[have], yv[have], classes, INDEPENDENT)
            row = next(r for r in table.rows if r["feature"] == item["feature"])
            result = regression_calibration(rep, Xm, j, yv, fit=fit, n_boot=spec_me.n_boot, seed=seed)
        except CalibrationRefused as refused:
            out.append({**base, "refused": str(refused)})
            continue
        out.append({**base, "naive": result.naive, "naive_ci_low": row.get("ci_low"),
                    "naive_ci_high": row.get("ci_high"), "p": row.get("p"),
                    "estimate": result.estimate, "se": result.se, "ci_low": result.ci_low,
                    "ci_high": result.ci_high, "attenuation": result.attenuation,
                    "attenuation_se": result.attenuation_se, "within_variance": result.sigma2_u,
                    "between_variance": result.sigma2_xz, "n_persons": result.n_persons,
                    "n_repeat": result.n_repeat,
                    "recalls": {str(k): v for k, v in result.recalls.items()},
                    "n_boot": result.n_boot, "n_boot_ok": result.n_boot_ok})

    concerns: list[str] = []
    good = [e for e in out if e.get("estimate") is not None]
    if len(good) > 1:
        concerns.append(f"{len(good)} error-prone exposures are in the model, each corrected on its "
                        f"own. {FREEDMAN}: with two or more mismeasured exposures “estimated relative "
                        f"risks may become attenuated, inflated, or can even change direction”; the "
                        f"univariate correction does not undo that.")
    if energy and energy in columns:
        concerns.append(f"Total energy (`{energy}`) stays in the model and comes from the same "
                        f"recalls; it is treated as measured without error.")
    survey = getattr(state, "survey", None)
    if good and survey is not None and survey.estimand == "population":
        # WP10: the fit stage's table is design-based under this answer; this correction is not.
        concerns.append("The calibration is fit without the survey weights: it corrects these "
                        "participants' coefficient, not the surveyed population's design-based "
                        "one in the fit's table.")
    if good and spec.multiple_imputation() and X.isna().any().any():
        # WP7: the fit's table pools multiple imputations; this correction refits one fill.
        concerns.append("The calibration refits the model with blanks filled once without the "
                        "outcome, not over the multiple imputations the fit's table pools; its "
                        "naive coefficient can differ from the table's.")
    short = [e for e in good if e.get("n_boot_ok", 0) < e.get("n_boot", 0)]
    for e in short:
        concerns.append(f"`{e['feature']}`: {e['n_boot_ok']:,} of {e['n_boot']:,} bootstrap refits "
                        f"could be calibrated; the interval rests on those.")
    return done(applies=bool(good), reason=None if good else "No exposure could be calibrated; each "
                "says why.", family="linear", recalls=recalls, n_persons=n_persons, exposures=out,
                assumptions=list(ASSUMPTIONS), concerns=concerns,
                methods=methods_sentence(method, out, recalls, n_persons))


CALIBRATION_READS = ("measurement_error", "purpose", "models", "task", "event", "target", "roles",
                     "survey",
                     "energy_adjustment", "aggregation", "repeat_kind", "grain", "split", "findings")

__all__ = [
    "ASSUMPTIONS", "CALIBRATION_READS", "CalibratedExposure", "CalibrationArtifact",
    "adjusted_exposures", "calibration_stage", "combine_rule", "day_rows", "methods_sentence",
    "replicate_values",
]

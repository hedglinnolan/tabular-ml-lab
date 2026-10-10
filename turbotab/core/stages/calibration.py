"""The ``calibration`` stage: regression calibration from repeated recalls, a declared secondary
analysis beside the uncorrected estimate (MS5; MODELING_SEQUENCE §0 ruling 7, §1.1, §2, §6 chain 2).

When each person's row is the mean of two or more 24-hour recalls, the mean still carries
day-to-day error, and coefficients on it are biased. This stage corrects them by regression
calibration (``turbotab/core/methods/calibration.py`` has the arithmetic and its sources), from
the recalls themselves: the within-person covariance comes from the days each person's mean was
made of.

**What is calibrated** (MODELING_SEQUENCE §2, "Regression calibration implies multivariate
calibration when several intakes are error-prone"): every column of the outcome model that the
recalls measure, jointly. The energy model says which: the energy-adjusted nutrient of the residual
or density model and total energy beside it (when the model keeps it); the nutrient and total
energy of the standard model; every energy source and "other" of the partition and all-components
models; an exposure taken as it is. One such column is the univariate method (Rosner, Willett &
Spiegelman 1989); several, the multivariate one (Rosner, Spiegelman & Willett 1990), which is what
lets the all-components model (BLUEPRINT §12 ruling 2) be calibrated at all. The exposures among
them are reported; the others are calibrated with them and named.

**The calibration's covariates** are every other column of the outcome model (Boe et al. 2023), and
never the outcome. A change to the adjustment set invalidates the declaration (§2): the decision
records the set it was declared under (``SetMeasurementError.adjustment``), and the stage refuses
to run on another one until it is declared again (:func:`current_calibration`).

**The run order** (MODELING_SEQUENCE §1.1, inference, for each imputed copy): (1) the imputation,
compatible with the analysis model, the copy the coefficient table pools; (2) the energy model
refit on the copy and applied to each recall day (energy adjusted per day first, then calibrated);
(3) the calibration; (4) the outcome model. The point estimate is the mean over the copies (Rubin's
Q̄); its interval comes from a bootstrap over the whole chain, each replicate drawn as the design
says (PSUs within strata, clusters, or people) and imputed again (Schomaker & Heumann's Boot MI).
Model-based and Rubin-only intervals are refused for a calibrated coefficient.

**A declared secondary analysis** (ruling 7): it sits beside the uncorrected estimate, whose test of
no association stays the uncorrected model's (Freedman et al. 2011), and it is labeled with what it
corrects and what it assumes (:data:`~turbotab.core.methods.calibration.LABEL`).

**The uncorrected estimate beside it is the primary's**: its estimate, interval and test of no
association are those of the outcome model on every participant in the primary analysis (the fit
stage's table). A participant with no recall day on which every calibrated intake is recorded has
no replicate and is left out of the calibration (Carroll et al. 2006, §4.4); the artifact then says
how many, and gives the uncorrected coefficient refit on the calibrated participants beside it
(``naive_refit``), so the two estimates can be read on the same people.

**Where it refuses** (the artifact's ``reason`` and ``exits`` say which): under prediction (the
model is used on recalls measured the same way); without the linear family; outside a continuous
or yes/no outcome; on rows not combined from repeated recalls by the mean; on time points; on an
error-prone column with a declared spline or quintiles (the calibration corrects a linear term);
when the survey design has no stratum with two PSUs (no bootstrap by PSU within strata); with
fewer clusters than the cluster floor (no bootstrap by cluster); and when the data cannot carry the
calibration (no one with two recalls, the true intakes' covariance not positive). Each refusal of a
declared calibration is blocked and recorded: ``blocked`` says why in the methods' words, the
record's sentence says it (``voice``, through :func:`record_facts`), and the export carries it.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.methods.calibration import (BOOT_COPIES, CARROLL, FREEDMAN, LABEL, LONELY_RULE,
                                               NO_CALIBRATION, RAO_WU, ROSNER_1989, ROSNER_1990,
                                               SCHOMAKER)

# The energy step's outputs the recalls measure (``methods.energy``'s lineage operations).
ERROR_PRONE_OPERATIONS = ("residual", "density", "kept", "partition", "partition-other")
ENERGY_ADJUSTED = ("residual", "density")
ALL_SOURCES = ("partition", "all_components")

PREDICTION = ("Under prediction the model is used on recalls measured the same way as these, so its "
              "predictions need no correction; regression calibration corrects the coefficient of "
              "what you study, which is an inference question.")
NO_LINEAR = ("Regression calibration corrects a coefficient, and only the linear model reports one "
             "with an interval; choose it among the model families.")
# MS4 → MS5: under the population answer the calibration is design-based (weighted, with a
# bootstrap by PSU within strata); it is blocked only where no such bootstrap exists.
POPULATION = ("Under the surveyed population a calibrated coefficient's interval comes from a "
              "bootstrap by PSU within strata over the whole chain (MODELING_SEQUENCE §0 ruling 7), "
              "and no stratum of this design holds two PSUs with analyzed participants, so it is "
              "blocked and recorded. To calibrate for these participants instead, answer the survey "
              "question \"these participants\" (the sample-only attestation).")
UNCORRECTED_EXIT = "Keep the population's estimates uncorrected: record no calibration"
# Why a declared calibration was not run, in the methods' words (``blocked``; MODELING_SEQUENCE §4,
# block and record): the record's sentence restates it and the export carries it.
POPULATION_BLOCKED = ("under the surveyed population no stratum held two PSUs with analyzed "
                      "participants, leaving no bootstrap by PSU within strata for its interval")
BLOCKED_NO_LINEAR = ("the linear model, the only family whose coefficient it corrects, was not "
                     "fitted")
BLOCKED_NOT_COMBINED = ("each analysis row was one record, not the mean of a person's repeated "
                        "recalls")
BLOCKED_TIME_POINTS = ("the rows repeated as time points, not as repeated recalls of one usual "
                       "intake")
BLOCKED_NONE = "no column of the outcome model was measured by the recalls"
BLOCKED_UNSETTLED = "a question it rests on was not settled"
BLOCKED_MISSING = "the handling of missing values it rests on was blocked"
BLOCKED_PREDICTION = ("under prediction the model is used on recalls measured the same way, so "
                      "there is no coefficient to correct")
REASKED = ("Regression calibration was declared under another adjustment set, so it is re-asked "
           "and was not run; the estimates are uncorrected.")


def not_run_sentence(blocked: str) -> str:
    """The methods text of a declared calibration that was not run."""
    return (f"Regression calibration was declared as a secondary analysis, but {blocked}; it was "
            f"blocked and recorded, and the estimates are uncorrected.")


def record_facts(artifact: Any) -> dict[str, Any]:
    """What the record's sentence of ``set_measurement_error`` reads from the calibration stage
    when the methods text is assembled (``voice.restate``; ``ProjectService.methods``): why a
    declared calibration was not run, by the fresh artifact as served; empty otherwise."""
    if not isinstance(artifact, Mapping) or artifact.get("applies"):
        return {}
    blocked = artifact.get("blocked")
    return {"set_measurement_error": {"blocked": str(blocked)}} if blocked else {}


def population_exits() -> list[dict[str, Any]]:
    """The population block's exits, each a decision: the sample-only attestation (MODELING_SEQUENCE
    §4's exit), or no correction, which keeps the design-based estimates as they are."""
    from turbotab.core.models.survey import SAMPLE_EXIT

    return [{"label": SAMPLE_EXIT, "decision": {"kind": "set_survey", "estimand": "sample"}},
            {"label": UNCORRECTED_EXIT,
             "decision": {"kind": "set_measurement_error", "method": "none"}}]


# At most 30 words: the consequence preview says it as its one line (the preview_note budget).
NOT_COMBINED = ("Each analysis row is one record, not the mean of a person's repeated recalls, so "
                "day-to-day variance cannot be estimated: record the rows as repeats and combine "
                "by the mean.")
TIME_POINTS = ("The rows repeat as time points, not as repeated recalls of one usual intake, so their "
               "spread is change over time rather than day-to-day error.")
NONE_ERROR_PRONE = ("No column of the outcome model is measured by the recalls (nothing you study, and no "
                    "energy model's term), so there is nothing to calibrate.")
TEST = ("The test of no association is the uncorrected model's: “the usual statistical test of the "
        f"null hypothesis (no exposure effect) remains theoretically valid” ({FREEDMAN}); the "
        "calibrated estimate gives the size, with its own interval.")
ASSUMPTIONS = (
    "Each recall's error is independent of the person's true intake and of their other days' "
    "errors (classical error). Recalls share a person's own reporting bias, which this reads as "
    "true intake, so the corrected estimate can still be off in either direction "
    f"({FREEDMAN}, Table 2; Kipnis et al. 2003).",
    "The covariates of the outcome model are taken as measured without error; each is in the "
    "calibration equation, and the outcome never is.",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class CalibratedExposure(_Model):
    """One error-prone column: its uncorrected and its calibrated coefficient."""

    feature: str  # the model-matrix column (``protein_adj``, ``kcal_from_fat``)
    source: str  # the raw nutrient column
    operation: str  # the energy step's operation ("residual", "partition", …; "as is")
    naive: float | None  # the primary's estimate: every participant it analyzes
    naive_ci_low: float | None = None
    naive_ci_high: float | None = None
    p: float | None = None  # the primary's test of no association (Freedman 2011)
    # The uncorrected coefficient refit on the calibrated participants, when they are fewer than
    # the primary's (None when they are the same people): the like-for-like comparison.
    naive_refit: float | None = None
    estimate: float | None = None  # calibrated
    se: float | None = None  # the whole-chain bootstrap's standard deviation
    ci_low: float | None = None  # its percentile interval
    ci_high: float | None = None
    delta_se: float | None = None  # Rosner's delta method: a check, where it is defined
    attenuation: float | None = None  # Γ_jj at the most common number of recalls
    within_variance: float | None = None  # Σ_uu,jj: day to day
    between_variance: float | None = None  # Σ_x|z,jj: of true intake given the covariates
    n_persons: int = 0
    n_repeat: int = 0  # people with two or more recalls
    recalls: dict[str, int] = {}  # number of recalls -> people
    n_boot: int = 0
    n_boot_ok: int = 0
    refused: str | None = None


class CalibratedContrast(_Model):
    """The substitution as the difference of calibrated coefficients (all-components)."""

    donor: str
    recipient: str
    step_kcal: float
    naive: float | None  # the uncorrected contrast (its interval: the substitution curve's)
    estimate: float | None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class CalibrationArtifact(_Model):
    method: Literal["none", "regression_calibration"]
    purpose: Literal["inference", "prediction"]
    applies: bool
    reason: str | None = None  # why nothing was calibrated
    # Why a declared calibration was not run, in the methods' words (block and record): the
    # record's sentence and the export say it. None when it ran, or when none was declared.
    blocked: str | None = None
    # The ways past a block, each a decision the client can post (BLUEPRINT §11.3).
    exits: list[dict[str, Any]] = []
    family: str | None = None
    rows: Literal["all eligible rows"] = "all eligible rows"
    role: Literal["secondary"] = "secondary"  # beside the uncorrected estimate (ruling 7)
    label: str = LABEL
    test: str = TEST
    calibration: Literal["univariate", "multivariate"] | None = None
    calibrated: list[str] = []  # every error-prone column, calibrated jointly
    covariates: list[str] = []  # the calibration's covariates: every other outcome-model column
    order: list[str] = []  # MODELING_SEQUENCE §1.1 as run
    resampling: Literal["persons", "clusters", "psu_within_strata"] | None = None
    weighted: bool = False  # survey-weighted (the surveyed population)
    imputations: int = 0  # the copies the point estimate is the mean over (0: none)
    boot_copies: int = 0  # each bootstrap replicate's own imputations (0: none)
    attenuation: list[list[float]] | None = None  # Γ at the most common k, in ``calibrated`` order
    within_covariance: list[list[float]] | None = None  # Σ_uu, in ``calibrated`` order
    n_persons: int = 0  # the participants calibrated
    n_primary: int = 0  # the participants in the primary analysis (the uncorrected estimate's)
    n_boot: int = 0
    n_boot_ok: int = 0
    recalls: dict[str, int] = {}  # number of recalls -> people (energy recorded on each)
    exposures: list[CalibratedExposure] = []
    contrasts: list[CalibratedContrast] = []
    assumptions: list[str] = []
    concerns: list[str] = []
    methods: str


# ── the declaration and its adjustment set (MODELING_SEQUENCE §2, "invalidates") ──────────────


def declared_adjustment(state: Any) -> list[str]:
    """The outcome model's covariates as the state declares them under inference: every settled
    predictor but the exposures, less the answered covariates the primary leaves out (sorted). The
    set a calibration is declared under; a change to it invalidates the declaration."""
    from turbotab.core.estimand import adjustment_left_out, predictor_roles

    if getattr(state, "purpose", None) != "inference":
        return []
    gone = set(adjustment_left_out(state))
    return sorted(c for c, r in predictor_roles(state).items() if r != "exposure" and c not in gone)


def current_calibration(state: Any) -> tuple[Any, list[str] | None]:
    """(the measurement-error answer, the adjustment set it was declared under when that set has
    changed since, else None). A calibration declared under another adjustment set is re-asked,
    never silently kept (MODELING_SEQUENCE §2)."""
    spec = getattr(state, "measurement_error", None)
    if spec is None or spec.method == "none":
        return spec, None
    recorded = getattr(spec, "adjustment", None)
    if recorded is None:
        return spec, None
    return spec, (None if sorted(recorded) == declared_adjustment(state) else list(recorded))


def reasked_clause(adjustment: Sequence[str] | None, state: Any) -> str | None:
    """The record's clause when the adjustment set a calibration was declared under
    (``adjustment``, the answer's own) is no longer the outcome model's under ``state``: it is
    re-asked and not run (MODELING_SEQUENCE §2). None while the set stands."""
    if adjustment is None or getattr(state, "purpose", None) != "inference":
        return None
    if sorted(adjustment) == declared_adjustment(state):
        return None
    from turbotab.core.voice import listing

    was = listing(list(adjustment), limit=8) if adjustment else "no covariate"
    return (f"when the model adjusted for {was}; the adjustment set has changed since, so the "
            f"declaration is re-asked and was not run, and the estimates are uncorrected")


def invalidated(spec: Any, recorded: Sequence[str], state: Any) -> tuple[str, list[dict[str, Any]]]:
    from turbotab.core.voice import listing

    now = declared_adjustment(state)
    was = listing(list(recorded), limit=8) if recorded else "no covariate"
    is_ = listing(now, limit=8) if now else "no covariate"
    again = {"kind": "set_measurement_error", "method": spec.method,
             "exposures": list(spec.exposures), "n_boot": int(spec.n_boot)}
    return ((f"Regression calibration was declared when the model adjusted for {was}; it now "
             f"adjusts for {is_}. Its calibration equation must hold every covariate of the outcome "
             f"model, so the declaration is re-asked, not kept (MODELING_SEQUENCE §2)."),
            [{"label": "Declare the calibration again under the current adjustment set",
              "decision": again},
             {"label": "Record no calibration",
              "decision": {"kind": "set_measurement_error", "method": "none"}}])


def unread_refusal(state: Any, working: Mapping[str, Any],
                   structure: Any) -> dict[str, Any] | None:
    """The stage's refusals that read no row, in its order, as its artifact's fields (``reason``,
    ``exits``, ``blocked``, ``methods``): under prediction; declared under another adjustment set;
    rows that are not the mean of repeated recalls; rows that repeat as time points. None when
    none applies. The consequence preview says the same before the answer is recorded."""
    from turbotab.core.stages.working import effective_repeat_kind

    if getattr(state, "purpose", None) != "inference":
        return {"reason": PREDICTION, "blocked": BLOCKED_PREDICTION}
    spec_now, recorded = current_calibration(state)
    if recorded is not None:
        reason, exits = invalidated(spec_now, recorded, state)
        return {"reason": reason, "exits": exits, "methods": REASKED}
    if working.get("aggregation") is None:
        return {"reason": NOT_COMBINED, "blocked": BLOCKED_NOT_COMBINED}
    if effective_repeat_kind(state, structure) == "time_points":
        return {"reason": TIME_POINTS, "blocked": BLOCKED_TIME_POINTS}
    return None


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
        if entry["operation"] in ENERGY_ADJUSTED and roles.get(str(entry["inputs"][0])) == "exposure":
            out.append({"feature": str(entry["output"]), "source": str(entry["inputs"][0]),
                        "energy": str(entry["inputs"][1]), "operation": str(entry["operation"])})
    return out


def error_prone(fitted: Any, columns: Sequence[str], roles: Mapping[str, str],
                energy: str | None) -> list[dict[str, Any]]:
    """Every model-matrix column the recalls measure, in the matrix's order: the energy step's
    outputs other than pass-through (``ERROR_PRONE_OPERATIONS``) and each exposure taken as it is.
    Each with its ``inputs`` (the raw columns read on each recall day) and ``source``."""
    have = set(columns)
    found: dict[str, dict[str, Any]] = {}
    step = dict(fitted.steps).get("energy")
    if step is not None and hasattr(step, "lineage"):
        for entry in step.lineage():
            out, op = str(entry["output"]), str(entry["operation"])
            inputs = [str(c) for c in entry["inputs"]]
            if op not in ERROR_PRONE_OPERATIONS or out not in have:
                continue
            if op == "partition-other" and energy and energy not in inputs:
                inputs = [energy, *inputs]
            found[out] = {"feature": out, "source": inputs[0], "operation": op,
                          "inputs": inputs}
    for c, r in roles.items():
        if r == "exposure" and str(c) in have and str(c) not in found:
            found[str(c)] = {"feature": str(c), "source": str(c), "operation": "as is",
                             "inputs": [str(c)]}
    return [found[c] for c in columns if c in found]


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


def recall_matrix(fitted: Any, X: pd.DataFrame, days: pd.DataFrame, person: np.ndarray,
                  raw: Sequence[str], features: Sequence[str], matrix: pd.DataFrame) -> tuple[
                      np.ndarray, np.ndarray]:
    """Each recall day of the error-prone ``features``, energy adjusted on the day itself.

    A day's row is its person's row of ``X`` with the recall-measured ``raw`` columns taken from the
    day (``days``); it goes through the fitted pipeline's own steps (the energy model the copy fit),
    so the day's energy-adjusted value uses the copy's slope (MODELING_SEQUENCE §2: "for residual or
    density models, energy is adjusted per recall day first, then calibrated"). A day missing one of
    ``raw`` is not a recall of the set. Each person's days are then centered on the person's own
    model-matrix value, so their mean is exactly the value the outcome model saw and the days add
    only their spread (for the residual the centering moves nothing; for a density it replaces the
    mean of daily ratios by the ratio of means the model saw). Returns ``(values, person)``."""
    from turbotab.core.models.linear import model_matrix

    rows = X.iloc[person].reset_index(drop=True).copy()
    present = np.ones(len(person), dtype=bool)
    for c in raw:
        if c in rows.columns and c in days.columns:
            v = pd.to_numeric(days[c], errors="coerce").to_numpy(dtype=float)
            rows[c] = v
            present &= np.isfinite(v)
    if not present.any():
        return np.empty((0, len(features))), np.empty(0, dtype=np.int64)
    rows, person = rows.loc[present].reset_index(drop=True), person[present]
    day_matrix = model_matrix(fitted, rows)
    w = day_matrix[list(features)].to_numpy(dtype=float)
    keep = np.isfinite(w).all(axis=1)
    w, person = w[keep], person[keep]
    n = len(X)
    sums = np.column_stack([np.bincount(person, weights=w[:, j], minlength=n)
                            for j in range(w.shape[1])])
    counts = np.bincount(person, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        day_mean = sums / counts[:, None]
    target = matrix[list(features)].to_numpy(dtype=float)
    return w - day_mean[person] + target[person], person


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


# ── the words ────────────────────────────────────────────────────────────────


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


RESAMPLED = {"psu_within_strata": "PSUs within strata", "clusters": "whole clusters",
             "persons": "participants"}
# The share of bootstrap resamples that must carry the calibration for an interval to be shown (a
# stated floor: past it the resamples left out would shape the interval).
MIN_BOOT_SHARE = 0.9
FAILURE_WORDS = {"CalibrationRefused": "had no positive true-intake covariance",
                 "LinAlgError": "had a singular fit", "ValueError": "could not be refit",
                 "ImputationRefused": "could not be imputed", "not finite": "gave no finite estimate"}


def main_clause(run: Mapping[str, Any]) -> str | None:
    """The calibration's clause of the methods paragraph (MODELING_SEQUENCE §6 chain 2's reviewers'
    sentence, as the run makes it): what was calibrated, the substitution when one is drawn, and
    the bootstrap. ``run``: ``calibration`` ("univariate"/"multivariate"), ``all_sources``,
    ``calibrated`` (the columns), ``contrast`` (bool), ``resampling``, ``imputed`` (bool)."""
    from turbotab.core.voice import listing

    kind = run.get("calibration")
    if kind is None:
        return None
    if kind == "multivariate":
        what = ("Usual intakes of all energy sources were calibrated jointly"
                if run.get("all_sources") else
                f"Usual intakes of {listing(list(run.get('calibrated') or []), limit=8)} were "
                f"calibrated jointly")
    else:
        (column,) = run.get("calibrated") or ["what you study"]
        what = f"The usual intake of `{column}` was calibrated"
    parts = [f"{what} with every outcome-model covariate in the calibration model"]
    if run.get("contrast"):
        parts.append("the substitution is the difference of calibrated coefficients in the "
                     "all-components model")
    steps = "calibration, imputation and the outcome model" if run.get("imputed") else \
        "calibration and the outcome model"
    if run.get("interval", True) is not False:
        parts.append(f"CIs by a bootstrap resampling {RESAMPLED[str(run.get('resampling'))]} that "
                     f"repeats {steps}")
    return "; ".join(parts)


def methods_sentence(method: str, exposures: Sequence[Mapping[str, Any]], recalls: Mapping[str, int],
                     n_persons: int, run: Mapping[str, Any] | None = None) -> str:
    """The methods text: the reviewers' clause (:func:`main_clause`) and what it rests on."""
    days = _days_phrase(recalls)
    if method == "none" and not recalls:  # nothing was read: purpose or rows ruled it out first
        return "Intakes were not corrected for day-to-day error in the recalls."
    if method == "none":
        return (f"Intakes were the mean of each participant's recalls ({days}, "
                f"{n_persons:,} participants) and were not corrected for day-to-day error.")
    done = [e for e in exposures if e.get("estimate") is not None]
    if not done or run is None:
        return ("Regression calibration was asked for, but no intake could be calibrated; the "
                "estimates are uncorrected.")
    clause = main_clause(run) or ""
    n_repeat = max(int(e["n_repeat"]) for e in done)
    source = ROSNER_1990 if run.get("calibration") == "multivariate" else ROSNER_1989
    by_copy = " by each imputed copy's own energy model" if run.get("imputed") else ""
    per_day = (f" Energy was adjusted on each recall day{by_copy} before calibration."
               if run.get("per_day") else "")
    n_primary = int(run.get("n_primary") or 0)
    left = (f" Of the {n_primary:,} participants in the primary analysis, {n_primary - n_persons:,} "
            f"had no recall day with every calibrated intake recorded and were left out of the "
            f"calibration; the uncorrected estimate beside it is the primary's, on all "
            f"{n_primary:,}." if n_primary > n_persons else "")
    spread = "covariance" if run.get("calibration") == "multivariate" else "variance"
    weighted = (" The calibration and the outcome model were survey-weighted, PSUs resampled by "
                f"Rao and Wu's bootstrap ({RAO_WU})." if run.get("weighted") else "")
    boot = int(run.get("n_boot_ok") or 0)
    if run.get("imputed"):
        interval = (f" Each of {boot:,} bootstrap resamples was imputed {BOOT_COPIES} "
                    f"times and the interval is the percentile interval of their means (Boot MI, "
                    f"{SCHOMAKER}); the estimate is the mean over the {int(run.get('m') or 0)} "
                    f"imputed copies.")
    else:
        interval = f" The interval is the percentile interval of {boot:,} bootstrap resamples."
    if run.get("interval") is False:
        interval = (f" Only {boot:,} of {int(run.get('n_boot') or 0):,} bootstrap resamples could "
                    f"be calibrated, too few for an interval, so none is reported.")
    return (f"{clause}. Calibration ({source}; {CARROLL}) used the within-person {spread} of the "
            f"recalls of the {n_repeat:,} participants with two or more ({days}).{left}{per_day}"
            f"{weighted}"
            f"{interval} It is a declared secondary analysis beside the uncorrected estimate, whose "
            f"test of no association is the primary's; it {LABEL}.")


# ── the stage ────────────────────────────────────────────────────────────────


def _nonlinear(columns: Sequence[str], feature: str) -> bool:
    return any(c != feature and (c.startswith(f"{feature}'") or c.startswith(f"{feature}_Q"))
               for c in columns)


def resampling_of(survey: Any, clusters: Any, row_ids: np.ndarray) -> Any:
    """How the whole chain is redrawn: PSUs within strata under the population answer, whole
    clusters under repeated units, participants otherwise (MODELING_SEQUENCE §0 ruling 7)."""
    from turbotab.core.methods.calibration import Resampling

    n = len(row_ids)
    design = survey.design if survey is not None and survey.answer == "population" else None
    if design is not None:
        at = pd.Series(np.arange(len(design.row_ids)), index=design.row_ids)
        pos = at.reindex(row_ids).to_numpy().astype(np.int64)
        every = {int(u): int(s) for u, s in zip(design.psu, design.stratum)}
        per = pd.Series(list(every.values())).value_counts()
        present = set(design.stratum[pos].tolist())
        lonely = sum(1 for s in present if int(per.get(s, 0)) == 1)
        return Resampling("psu_within_strata", n, stratum=design.stratum[pos], psu=design.psu[pos],
                          design_psus=every, lonely=lonely)
    if clusters is not None and getattr(clusters, "clustered", False):
        return Resampling("clusters", n, codes=pd.factorize(np.asarray(clusters.codes))[0])
    return Resampling("persons", n)


def cluster_floor(clusters: Any, persons: np.ndarray) -> tuple[str, str] | None:
    """``(reason, blocked)`` when the analyzed participants ``persons`` (positions among the
    clusters' rows) lie in fewer whole clusters than the floor (``models.inference.min_clusters``,
    the one the fit refuses cluster-robust intervals below, MODELING_SEQUENCE §2 "repeated units or
    clusters ... with refusal below a floor"); else None. A bootstrap that redraws so few clusters
    rests on a handful of units, so the calibrated coefficient has no interval to carry."""
    from turbotab.core.models.inference import min_clusters

    if clusters is None or not getattr(clusters, "clustered", False):
        return None
    G = int(len(np.unique(np.asarray(clusters.codes)[persons])))
    floor = min_clusters()
    if G >= floor:
        return None
    column = clusters.column
    return ((f"The analyzed participants lie in {G} `{column}` units, fewer than the {floor} "
             f"TurboTab requires for cluster-robust intervals (the fit reports none for the "
             f"uncorrected estimate either). A calibrated coefficient's interval would come from a "
             f"bootstrap redrawing {G} whole clusters, which rests on a handful of units, so "
             f"regression calibration is refused here."),
            (f"the participants lay in {G} `{column}` units, fewer than the {floor} that "
             f"cluster-robust intervals require, too few for a bootstrap of whole clusters to "
             f"carry its interval"))


def analysis_design(ctx: Any, state: Any, index: Any) -> tuple[Any, Any, np.ndarray,
                                                                 np.ndarray | None]:
    """``(clusters, survey, keep, weights)`` for the analysis rows ``index`` (row ids): the units
    the intervals cluster by (a confirmed grouping; settled readings only), the survey answer as
    the fit applies it (``stages.modeling._survey``), which rows the calibration may hold (the
    survey domain under the population answer, else all) and their survey weights (None unless
    the population answer). The stage and its preview (``method_previews.calibration_numbers``)
    both read the design through this, so the preview's λ is the stage's."""
    from turbotab.core.models.inference import INDEPENDENT, cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import _survey

    rows = np.asarray(index, dtype=np.int64)
    with open_store(ctx) as store:
        unit_columns = cluster_columns(state, store.columns)
        units = modeling_frame(store, unit_columns, rows) if unit_columns else None
    clusters = resolve_clusters(state, units) if units is not None else INDEPENDENT
    if clusters.refusal:
        clusters = INDEPENDENT
    survey, _ = _survey(ctx, clusters)
    keep = np.ones(len(rows), dtype=bool)
    weights: np.ndarray | None = None
    if survey is not None and survey.answer == "population" and survey.design is not None \
            and not survey.refusal:
        from turbotab.core.models.survey import domain_of

        domain = domain_of(rows, survey.design)
        keep &= np.asarray(domain.keep, dtype=bool)
        weights = np.full(len(rows), np.nan)
        weights[np.flatnonzero(domain.keep)] = domain.weight
    return clusters, survey, keep, weights


def _impute(spec: Any, X: pd.DataFrame, y: Any, task: str, m: int, seed: int, survey: Any,
            clusters: Any, nested: Any, factors: Any,
            time_invariant: Sequence[str] | None = None) -> list[pd.DataFrame]:
    """``m`` completed copies of ``X``, drawn as the coefficient table's are
    (``missing.impute_for_inference``) with ``m`` fixed: a bootstrap replicate's own imputations.
    ``time_invariant``: the columns the analysis's own copies carried and imputed once per unit
    (the confirmed ones, its plan's ``unit_level``), so a replicate imputes them as those did."""
    from turbotab.core.methods.missing import imputation_plan
    from turbotab.core.methods.smcfcs import impute

    plan = imputation_plan(spec, X, y, task, survey=survey, clusters=clusters, nested=nested,
                           factors=factors, time_invariant=time_invariant)
    out = impute(plan.data, plan.variables, mode=plan.mode, substantive=plan.substantive,
                 identity=plan.identity, units=plan.units, m=int(m), seed=seed, kinds=plan.kinds)
    frames = []
    for f in out.frames:
        done = X.copy()
        for c in X.columns:
            if c in f.columns and c not in plan.levels:
                done[c] = f[c].to_numpy() if not isinstance(f[c].dtype, pd.CategoricalDtype) else f[c]
        frames.append(done)
    return frames


def calibration_stage(ctx: StageContext) -> Bundle:
    from sklearn.base import clone

    from turbotab.core.methods.calibration import (CalibrationRefused, correct, delta_covariance,
                                                   logistic_fit, model_covariance, ols_fit,
                                                   whole_chain)
    from turbotab.core.methods.missing import copy_template
    from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import (_missing_for_table, _settled_factors, _task,
                                               coded_outcome)

    state = ctx.state
    spec_me = state.measurement_error
    method = spec_me.method
    inference = state.purpose == "inference"
    working = dict(getattr(ctx.inputs["working"], "data", ctx.inputs["working"]))
    structure = ctx.inputs.get("structure")
    structure = getattr(structure, "data", structure)

    def done(**fields: Any) -> Bundle:
        # A declared calibration that does not run is blocked and recorded (MODELING_SEQUENCE §4):
        # ``blocked`` says why in the methods' words, and the methods text is that sentence.
        if method == "none" or fields.get("applies"):
            fields.pop("blocked", None)
        elif fields.get("blocked"):
            fields.setdefault("methods", not_run_sentence(str(fields["blocked"])))
        fields.setdefault("methods", methods_sentence(method, [], fields.get("recalls") or {},
                                                      int(fields.get("n_persons") or 0)))
        artifact = CalibrationArtifact(method=method, purpose="inference" if inference else "prediction",
                                       **fields)
        ctx.progress(1.0, "Done")
        return Bundle(data=artifact.model_dump(mode="json"))

    unread = unread_refusal(state, working, structure)
    if unread is not None:
        return done(applies=False, **unread)

    task = _task(ctx)
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    adj = state.energy_adjustment
    energy = adj.energy_column if adj is not None and adj.method != "none" else None
    rows = ctx.inputs["cohort"].frames["rows"]["row_id"].to_numpy(dtype=np.int64)

    ctx.progress(0.03, "Reading the analysis rows and each person's recalls")
    with open_store(ctx) as store:
        frame = modeling_frame(store, [*spec.inputs, state.target], rows, outcome=state.target)
    clusters, survey, keep, weights_all = analysis_design(ctx, state, frame.index)
    if survey is not None and survey.refusal and method != "none":
        return done(applies=False, reason=survey.refusal, exits=list(survey.exits),
                    blocked=BLOCKED_UNSETTLED)
    y_all = coded_outcome(task, frame[state.target].to_numpy(), state.event)
    X_all = frame[list(spec.inputs)]
    raw_columns = list(dict.fromkeys([*spec.inputs, *([energy] if energy else [])]))
    days = day_rows(ctx, raw_columns, rows)
    day_unit = days.pop("__unit").to_numpy(dtype=np.int64)
    position = {int(r): i for i, r in enumerate(frame.index.to_numpy(dtype=np.int64))}
    day_person = np.array([position.get(int(u), -1) for u in day_unit], dtype=np.int64)
    if energy and energy in days.columns:
        recorded_e = pd.to_numeric(days[energy], errors="coerce").notna().to_numpy()
        per_person = np.bincount(day_person[recorded_e & (day_person >= 0)], minlength=len(frame))
    else:
        per_person = np.bincount(day_person[day_person >= 0], minlength=len(frame))
    recalls = _recall_counts(per_person)
    n_persons = int(len(frame))

    if method == "none":
        return done(applies=False, reason="Not corrected, as answered.", recalls=recalls,
                    n_persons=n_persons)
    if "linear" not in (state.models or []) or "linear" not in pipelines:
        return done(applies=False, reason=NO_LINEAR, recalls=recalls, n_persons=n_persons,
                    blocked=BLOCKED_NO_LINEAR)
    if task not in ("regression", "binary"):
        outcome = task.replace("_", "-")
        return done(applies=False, reason=(
            f"Regression calibration here corrects a least-squares or logistic coefficient; a "
            f"{outcome} outcome's coefficients are not corrected."),
            recalls=recalls, n_persons=n_persons,
            blocked=f"it corrects a least-squares or logistic coefficient, not a {outcome} one")

    # ── the rows: the domain under the population answer, the imputations under MI ──
    population = survey is not None and survey.answer == "population" and survey.design is not None
    resampling = resampling_of(survey, clusters, frame.index.to_numpy(dtype=np.int64))
    if resampling.kind == "psu_within_strata":
        per = pd.Series(list((resampling.design_psus or {}).values())).value_counts()
        present = set(resampling.stratum[keep].tolist())
        if not any(int(per.get(s, 0)) >= 2 for s in present):
            return done(applies=False, reason=POPULATION, exits=population_exits(),
                        recalls=recalls, n_persons=n_persons, blocked=POPULATION_BLOCKED)
    from turbotab.core.readings import nesting

    nested = nesting(state, dict((design.objects or {}).get("nested") or {}), columns=spec.inputs)
    clustered = clusters is not None and clusters.clustered
    missing = _missing_for_table(ctx, spec, X_all, y_all, task, ["linear"], survey=survey,
                                 clusters=clusters if clustered else None, nested=nested)
    if missing is not None and missing.refusal:
        return done(applies=False, reason=missing.refusal, exits=list(missing.exits),
                    recalls=recalls, n_persons=n_persons, blocked=BLOCKED_MISSING)
    imputations = getattr(missing, "imputations", None) if missing is not None else None
    imputed = imputations is not None and getattr(imputations, "method", "") != "supplied"
    plan = dict(getattr(imputations, "plan", None) or {}) if imputed else {}
    template = copy_template(pipelines["linear"], plan) if imputed else clone(pipelines["linear"])
    copies = list(imputations.frames) if imputed else [X_all]
    factors = _settled_factors(ctx, spec) if imputed else {}
    concerns: list[str] = []
    fit = ols_fit if task == "regression" else logistic_fit
    yv_all = np.asarray(y_all, dtype=float)

    # ── the error-prone columns, from the first copy's pipeline ──
    ctx.progress(0.08, "Refitting the linear model on every eligible row")
    first = fit_pipeline(clone(template), copies[0], y_all)
    matrix0 = model_matrix(first, copies[0])
    columns = [str(c) for c in matrix0.columns]
    items = error_prone(first, columns, spec.roles, energy)
    if not items:
        return done(applies=False, reason=NONE_ERROR_PRONE, family="linear", recalls=recalls,
                    n_persons=n_persons, blocked=BLOCKED_NONE)
    shaped = [i["feature"] for i in items if _nonlinear(columns, i["feature"])]
    if shaped:
        from turbotab.core.voice import listing

        has = "has" if len(shaped) == 1 else "have"
        return done(applies=False, family="linear", recalls=recalls, n_persons=n_persons, reason=(
            f"{listing(shaped)} {has} a "
            f"declared spline or quintiles; regression calibration here corrects a linear term, so "
            f"E[X | W̄, Z] would be put through a curve it was not fit for."),
            exits=[{"label": "Calibrate the linear form of the intake (the form question)",
                    "decision": None}, dict(NO_CALIBRATION)],
            blocked=(f"{listing(shaped)} had a declared spline or quintiles, and it corrects a "
                     f"linear term"))
    raw = list(dict.fromkeys(c for i in items for c in i["inputs"]))
    by_mean = {c: combine_rule(state, working, c) for c in raw}
    other = sorted(c for c, rule in by_mean.items() if rule != "mean")
    if other:
        from turbotab.core.voice import listing

        c = other[0]
        combined = (f"{listing(other)} {'was' if len(other) == 1 else 'were'} combined "
                    f"by {'its' if len(other) == 1 else 'their'} {by_mean[c]} record, not the mean "
                    f"of the recalls")
        return done(applies=False, family="linear", recalls=recalls, n_persons=n_persons, reason=(
            f"{combined}, so the error is a single day's; combine by the mean to calibrate."),
            exits=[{"label": "Combine the recalls by the mean", "decision": None},
                   dict(NO_CALIBRATION)], blocked=combined)
    features = [i["feature"] for i in items]
    J = [columns.index(f) for f in features]
    covariates = [c for c in columns if c not in set(features)]
    # Reported: the intakes the answer names, else the exposures among them, else all of them;
    # every error-prone column is calibrated jointly whichever are reported.
    wanted = set(spec_me.exposures)
    named = [i for i in items if i["source"] in wanted or i["feature"] in wanted]
    if wanted and not named:
        names = ", ".join(f"`{c}`" for c in sorted(wanted))
        are = "is" if len(wanted) == 1 else "are"
        return done(applies=False, family="linear", recalls=recalls, n_persons=n_persons, reason=(
            f"{names} {are} not among the model's intakes the recalls measure."),
            exits=[dict(NO_CALIBRATION)],
            blocked=f"{names} {are} not among the model's intakes the recalls measure")
    exposure_roles = {c for c, r in spec.roles.items() if r == "exposure"}
    reported = named or [i for i in items if i["source"] in exposure_roles] or list(items)

    # recall days of the analyzed persons
    on_day = day_person >= 0
    day_frame = days.loc[on_day].reset_index(drop=True)
    day_of = day_person[on_day]
    keep &= np.bincount(day_of, minlength=n_persons) >= 1
    if not imputed and X_all.isna().to_numpy().any() and (spec.missing or {}).get("strategy") == \
            "complete_case":
        keep &= ~X_all.isna().any(axis=1).to_numpy()
    persons = np.flatnonzero(keep)
    floor = cluster_floor(clusters, persons) if resampling.kind == "clusters" else None
    if floor is not None:
        return done(applies=False, family="linear", recalls=recalls, n_persons=n_persons,
                    reason=floor[0], exits=[dict(NO_CALIBRATION)], blocked=floor[1])
    remap = np.full(n_persons, -1)
    remap[persons] = np.arange(len(persons))
    day_mask = remap[day_of] >= 0
    day_frame = day_frame.loc[day_mask].reset_index(drop=True)
    day_of = remap[day_of[day_mask]]
    day_order = np.argsort(day_of, kind="stable")
    day_frame, day_of = day_frame.iloc[day_order].reset_index(drop=True), day_of[day_order]
    day_starts = np.concatenate([[0], np.cumsum(np.bincount(day_of, minlength=len(persons)))])
    yv = yv_all[persons]
    w = None if weights_all is None else weights_all[persons]
    classes = list(getattr(first[-1], "classes_", [])) or None
    seed = int(getattr(state.split, "seed", 0) or 0) if state.split is not None else 0
    contrast = _contrast(state, adj, features)

    def chain(frames: Sequence[pd.DataFrame], yy: np.ndarray, dframe: pd.DataFrame,
              dperson: np.ndarray, ww: np.ndarray | None) -> list[Any]:
        """Each copy's (fitted, matrix, Corrected): steps (2)–(4) of §1.1."""
        out = []
        for Xc in frames:
            fitted = fit_pipeline(clone(template), Xc, yy)
            matrix = model_matrix(fitted, Xc)
            cols = [str(c) for c in matrix.columns]
            if [cols.index(f) for f in features if f in cols] != J or len(cols) != len(columns):
                raise ValueError("a replicate's model matrix differs from the analysis's")
            values, who = recall_matrix(fitted, Xc, dframe, dperson, raw, features, matrix)
            from turbotab.core.methods.calibration import Recalls

            rec = Recalls.of(values, who, len(Xc))
            corrected = correct(matrix.to_numpy(dtype=float), J, rec, yy, fit=fit, weights=ww)
            out.append((fitted, matrix, corrected))
        return out

    def quantities(results: Sequence[Any]) -> np.ndarray:
        naive = np.mean([r[2].naive[J] for r in results], axis=0)
        est = np.mean([r[2].estimate[J] for r in results], axis=0)
        extra = []
        if contrast is not None:
            a, b, step = contrast
            extra = [step * (est[features.index(a)] - est[features.index(b)])]
        return np.concatenate([est, extra, naive])

    ctx.progress(0.12, "Calibrating each copy" if imputed else "Calibrating the intakes")
    X_rows = [c.iloc[persons] for c in copies]
    try:
        results = chain(X_rows, yv, day_frame, day_of, w)
    except CalibrationRefused as refused:
        said = str(refused).rstrip(".")
        return done(applies=False, family="linear", recalls=recalls, n_persons=n_persons,
                    reason=str(refused), exits=refused.exits or [dict(NO_CALIBRATION)],
                    blocked=f"{said[:1].lower()}{said[1:]}")
    point = quantities(results)
    cal0 = results[0][2].calibration
    p = len(J)
    n_ok = len(persons)  # the participants in the primary analysis
    n_cal = int(cal0.n)  # those with a recall day of every calibrated intake: the calibration's

    # ── the whole-chain bootstrap ──
    design_obj = survey.design if population else None

    def replicate(draw: Any, b: int) -> np.ndarray:
        idx = np.asarray(draw.rows, dtype=np.int64)  # positions among the analyzed persons
        Xb = [X_all.iloc[persons[idx]].reset_index(drop=True)]
        yb = yv[idx]
        wb = None if w is None else w[idx] * draw.factor
        counts = day_starts[idx + 1] - day_starts[idx]
        take = np.concatenate([np.arange(day_starts[i], day_starts[i + 1]) for i in idx]) \
            if len(idx) else np.zeros(0, dtype=np.int64)
        db = day_frame.iloc[take].reset_index(drop=True)
        pb = np.repeat(np.arange(len(idx)), counts)
        if imputed:
            from turbotab.core.models.survey import SurveyDesign

            sb = None
            if design_obj is not None:
                sb = SurveyDesign(row_ids=np.arange(len(idx)), weight=wb,
                                  stratum=np.asarray(draw.stratum, dtype=np.int64),
                                  psu=np.asarray(draw.unit, dtype=np.int64),
                                  weight_column=design_obj.weight_column,
                                  strata_column=design_obj.strata_column,
                                  psu_column=design_obj.psu_column)
            cb = None
            if clustered:
                codes = np.asarray(clusters.codes)[persons][idx]
                pair = pd.factorize(pd.MultiIndex.from_arrays([draw.unit, codes]))[0]
                cb = Clusters(column=clusters.column, codes=pair, n_clusters=int(pair.max()) + 1)
            Xb = _impute(spec, Xb[0], yb, task, BOOT_COPIES, seed + 1009 * (b + 1), sb, cb,
                         nested, factors, time_invariant=plan.get("unit_level"))
        return quantities(chain(Xb, yb, db, pb, wb))

    n_boot = int(spec_me.n_boot)
    from turbotab.core.methods.calibration import Resampling

    boot_resampling = Resampling(resampling.kind, n_ok,
                                 codes=None if resampling.codes is None else
                                 pd.factorize(resampling.codes[persons])[0],
                                 stratum=None if resampling.stratum is None else
                                 resampling.stratum[persons],
                                 psu=None if resampling.psu is None else resampling.psu[persons],
                                 design_psus=resampling.design_psus, lonely=resampling.lonely)

    def progress(i: int, total: int) -> None:
        if i == total or i % max(1, total // 40) == 0:
            ctx.progress(0.15 + 0.8 * i / max(total, 1),
                         f"Bootstrap resample {i:,} of {total:,} (the whole chain)")

    boot = whole_chain(replicate, boot_resampling, n_boot, seed, progress=progress,
                       cancelled=ctx.cancelled)
    # The resamples a calibration cannot carry are left out, and said; past a tenth of them the
    # ones kept are no longer the bootstrap distribution, so no interval is shown.
    enough = boot.n_ok >= 2 and boot.n_ok >= MIN_BOOT_SHARE * n_boot
    low, high = boot.interval()
    se = boot.se()

    # ── the uncorrected estimate beside it: the primary's, on every participant it analyzes ──
    tables = []
    for _, matrix, _ in results:
        m_rows = matrix.copy()
        m_rows.index = frame.index[persons]
        if population:
            from turbotab.core.models.survey import survey_table

            table = survey_table(task, m_rows, y_all[persons], classes, survey.design)
        else:
            cl = INDEPENDENT
            if clustered:
                codes = np.asarray(clusters.codes)[persons]
                cl = Clusters(column=clusters.column, codes=codes,
                              n_clusters=int(len(np.unique(codes))))
            table = inference_table(task, m_rows, y_all[persons], classes, cl)
        tables.append(table.rows)
    if len(tables) > 1:
        from turbotab.core.methods.imputation import pool_rows

        rows_naive = {str(r["feature"]): r for r in pool_rows(tables)}
    else:
        rows_naive = {str(r["feature"]): r for r in tables[0]}
    delta = None
    if not imputed and not population and not clustered:
        corrected = results[0][2]
        naive_cov = model_covariance(fit, corrected.X, yv[corrected.rows])[np.ix_(J, J)]
        delta = delta_covariance(corrected, naive_cov)

    modal = cal0.modal_k
    gamma = cal0.slope(modal)
    k_counts = cal0.counts[cal0.counts >= 1].astype(np.int64)
    rec_counts = {str(int(k)): int(c) for k, c in enumerate(np.bincount(k_counts)) if c and k}
    out: list[dict[str, Any]] = []
    for i in reported:
        j = features.index(i["feature"])
        row = rows_naive.get(i["feature"], {})
        out.append({
            "feature": i["feature"], "source": i["source"], "operation": i["operation"],
            "naive": _f(row["estimate"]) if row.get("estimate") is not None else None,
            "naive_ci_low": row.get("ci_low"), "naive_ci_high": row.get("ci_high"),
            "p": row.get("p"),
            "naive_refit": float(point[p + (1 if contrast else 0) + j]) if n_cal < n_ok else None,
            "estimate": float(point[j]),
            "se": _f(se[j]) if enough else None,
            "ci_low": _f(low[j]) if enough else None,
            "ci_high": _f(high[j]) if enough else None,
            "delta_se": _f(np.sqrt(delta[j, j])) if delta is not None else None,
            "attenuation": float(gamma[j, j]),
            "within_variance": float(cal0.sigma_uu[j, j]),
            "between_variance": float(cal0.conditional[j, j]),
            "n_persons": n_cal, "n_repeat": int(cal0.within.n_repeat), "recalls": rec_counts,
            "n_boot": n_boot, "n_boot_ok": boot.n_ok})
    contrasts = []
    if contrast is not None:
        a, b, step = contrast
        k = p
        primary = [rows_naive.get(f, {}).get("estimate") for f in (a, b)]
        naive_c = (step * (float(primary[0]) - float(primary[1]))
                   if None not in primary else None)
        contrasts.append({
            "donor": items[features.index(b)]["source"], "recipient": items[features.index(a)]["source"],
            "step_kcal": float(step), "naive": naive_c, "estimate": float(point[k]),
            "se": _f(se[k]) if enough else None,
            "ci_low": _f(low[k]) if enough else None,
            "ci_high": _f(high[k]) if enough else None})

    if boot.n_ok < n_boot:
        why = "; ".join(f"{n:,} {FAILURE_WORDS.get(k, k)}" for k, n in sorted(boot.failures.items()))
        rests = ("the interval rests on those" if enough else
                 f"fewer than {MIN_BOOT_SHARE:.0%} of them, so no interval is shown")
        # A resample with no positive true-intake covariance is one whose day-to-day error swamped
        # the spread between people: its attenuation was at or past zero, where the correction
        # 1/λ is largest, so leaving it out trims the tail away from zero (said, REPAIR-RC).
        tail = (" Those left out are resamples whose day-to-day error swamped the spread between "
                "people, where the correction is largest, so the interval's end away from zero "
                "may be too near." if enough and boot.failures.get("CalibrationRefused") else "")
        concerns.append(f"{boot.n_ok:,} of {n_boot:,} bootstrap resamples could be calibrated "
                        f"({why}); {rests}.{tail}")
    if resampling.kind == "psu_within_strata" and resampling.lonely:
        one = resampling.lonely == 1
        concerns.append(f"{resampling.lonely:,} {'stratum has' if one else 'strata have'} a single "
                        f"PSU, {'its PSU' if one else 'each PSU'} {LONELY_RULE} in the primary's "
                        f"table: this overstates rather than understates the variance.")
    if p > 1:
        concerns.append("With several error-prone intakes the uncorrected test of one can be off, "
                        "because the error in the others leaves some confounding uncorrected "
                        f"({ROSNER_1990}); the calibrated interval is the check.")
    if task == "binary":
        concerns.append("For a logistic outcome model, substituting E[X | W̄, Z] is an "
                        "approximation (Carroll et al. 2006, §4.2), close when the effect is "
                        "moderate.")
    if n_cal < n_ok:
        filled = " (the primary fills their intakes in each imputed copy)" if imputed else ""
        concerns.append(f"{n_ok - n_cal:,} of the {n_ok:,} participants in the primary analysis have "
                        f"no recall day with every calibrated intake recorded{filled}, so they are "
                        f"not in the calibration: the calibrated estimate describes the other "
                        f"{n_cal:,}, while the uncorrected estimate beside it, with its interval "
                        f"and test, is the primary's on all {n_ok:,}. Refit on the same "
                        f"{n_cal:,}, the uncorrected coefficient is given beside it.")
    per_day = adj is not None and adj.method in ("residual", "residual_energy_dropped", "density",
                                                 "density_multivariate")
    m = len(copies) if imputed else 0
    run = {"calibration": "multivariate" if p > 1 else "univariate",
           "all_sources": adj is not None and adj.method in ALL_SOURCES,
           "calibrated": features, "contrast": contrast is not None,
           "resampling": resampling.kind, "imputed": imputed, "per_day": per_day,
           "weighted": population, "n_boot_ok": boot.n_ok, "m": m, "interval": enough,
           "n_boot": n_boot, "n_primary": n_ok}
    copy = " on each copy" if imputed else ""
    order = [*([f"multiple imputation compatible with the analysis model (m = {m})"]
               if imputed else []),
             (f"the energy model refit{copy}, then applied to each recall day" if adj is not None
              and adj.method != "none" else f"the pipeline refit{copy}, then applied to each "
                                            f"recall day"),
             "regression calibration" + (" in each copy" if imputed else ""),
             "the outcome model" + (" in each copy" if imputed else ""),
             "the whole-chain bootstrap" + (" (Boot MI)" if imputed else "")]
    return done(
        applies=True, family="linear", recalls=rec_counts, n_persons=n_cal, n_primary=n_ok,
        exposures=out,
        contrasts=contrasts, calibration=run["calibration"], calibrated=features,
        covariates=covariates, order=order, resampling=resampling.kind, weighted=population,
        imputations=m, boot_copies=BOOT_COPIES if imputed else 0,
        attenuation=[[float(v) for v in r] for r in gamma],
        within_covariance=[[float(v) for v in r] for r in cal0.sigma_uu],
        n_boot=n_boot, n_boot_ok=boot.n_ok, assumptions=list(ASSUMPTIONS), concerns=concerns,
        methods=methods_sentence(method, out, rec_counts, n_cal, run))


def _f(value: Any) -> float | None:
    v = float(value)
    return v if np.isfinite(v) else None


def _contrast(state: Any, adj: Any, features: Sequence[str]) -> tuple[str, str, float] | None:
    """(the recipient's feature, the donor's, the step in kcal) of the declared substitution, when
    the all-components (or partition) model holds both as calibrated kcal terms; else None."""
    sub = getattr(state, "substitution", None)
    if sub is None or adj is None or adj.method not in ALL_SOURCES:
        return None
    if getattr(sub, "scale", "kcal") != "kcal":
        return None
    a, b = f"kcal_from_{sub.recipient}", f"kcal_from_{sub.donor}"
    if a not in features or b not in features:
        return None
    return a, b, float(sub.step_kcal)


CALIBRATION_READS = ("measurement_error", "purpose", "models", "task", "event", "target", "roles",
                     "roles_unconfirmed", "role_confirmations", "reading_confirmations",
                     "shape_confirmations", "survey", "missing", "substitution",
                     "energy_adjustment", "aggregation", "repeat_kind", "grain", "split", "findings")

__all__ = [
    "ASSUMPTIONS", "CALIBRATION_READS", "CalibratedContrast", "CalibratedExposure",
    "CalibrationArtifact", "POPULATION", "POPULATION_BLOCKED", "REASKED", "TEST",
    "UNCORRECTED_EXIT", "adjusted_exposures", "analysis_design", "calibration_stage",
    "cluster_floor", "combine_rule", "current_calibration", "day_rows", "declared_adjustment",
    "error_prone", "invalidated", "main_clause", "methods_sentence", "not_run_sentence",
    "population_exits", "recall_matrix", "record_facts", "reasked_clause", "replicate_values",
    "resampling_of",
]

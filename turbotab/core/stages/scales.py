"""The ``scales`` stage (MS8): each declared scale's reliability and, under inference, its
coefficient corrected for measurement error as a declared secondary analysis.

The arithmetic and its sources are in :mod:`turbotab.core.methods.scales`, the leash in
:mod:`turbotab.core.scales`. Here the rows, the copies and the model matrix:

* **Rows.** Under inference every analyzed row (BLUEPRINT §12 ruling 3), the rows the coefficient
  table is estimated from; under prediction the training rows, where the reliability is descriptive
  (it changes no modeling choice) and nothing is corrected.
* **Copies.** Under inference with multiple imputation and a blank among the model's inputs, the
  items are multiply imputed at the item level, by the same imputation the fit's table pools
  (``methods.missing.impute_for_inference`` on the same rows, inputs, outcome and seed), and in each
  completed copy the score is formed, its reliability estimated, and the correction run with its
  own bootstrap; the copies are combined by Rubin's rules, each copy's variance its bootstrap
  variance. Otherwise there is one copy, the rows as they are (complete cases: the cohort has left
  out a row with a blank predictor).
* **The matrix.** The family whose coefficient is corrected is refit on each copy: the linear
  family for a least-squares or logistic outcome, the proportional-odds family for an ordinal one.
  The score's column of its model matrix is W, every other column is a covariate of the calibration
  (Boe et al. 2023), and the uncorrected estimate beside the corrected one is that refit's, with
  the interval its own table gives on independent rows.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext

FAMILY_FOR = {"regression": "linear", "binary": "linear", "ordinal": "proportional_odds"}
COEFFICIENT_LABELS = {"omega_total": "ω-total", "omega_hierarchical": "ω-hierarchical",
                      "test_retest_icc": "test–retest ICC(3,1)",
                      "calibration_slope": "calibration slope"}
# What a reflective scale's internal consistency still reports beside a test–retest or substudy
# reliability.
CARRIED = ("omega_total", "omega_hierarchical", "omega_total_standardized",
           "omega_hierarchical_standardized", "alpha", "alpha_label")
ALPHA_LABEL = ("Cronbach's α, customary: shown because the field reports it, not as the "
               "reliability (McNeish 2018: its assumptions usually fail and it usually understates "
               "reliability).")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class ScaleReliability(_Model):
    source: Literal["internal_consistency", "test_retest", "calibration_substudy"] | None = None
    coefficient: str | None = None  # omega_total, omega_hierarchical, test_retest_icc, …
    label: str | None = None
    value: float | None = None
    # ω on the standardized items, as R's psych::omega prints it (the score's own value is ``value``)
    omega_total: float | None = None
    omega_hierarchical: float | None = None
    omega_total_standardized: float | None = None
    omega_hierarchical_standardized: float | None = None
    alpha: float | None = None
    alpha_label: str | None = None
    factors: int = 1
    n: int = 0
    across_copies: list[float] = []  # each completed copy's value, under multiple imputation
    reason: str | None = None  # why there is none


class ScaleCorrection(_Model):
    family: str
    feature: str
    naive: float
    naive_ci_low: float | None = None
    naive_ci_high: float | None = None
    p: float | None = None  # the uncorrected test of no association (Freedman et al. 2011)
    estimate: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    scale: Literal["difference", "odds_ratio"] = "difference"
    ratio: float | None = None
    ratio_low: float | None = None
    ratio_high: float | None = None
    naive_ratio: float | None = None
    attenuation: float  # λ given the covariates (mean over copies)
    error_variance: float | None = None
    covariates: list[str] = []  # every other column of the model's matrix
    n: int = 0
    n_boot: int = 0  # bootstrap replicates in each completed copy
    n_boot_ok: int = 0  # the fewest any copy could recalibrate
    copies: int = 1  # the completed copies pooled (1 without multiple imputation)
    labels: list[str] = []


class ScaleResult(_Model):
    name: str
    items: list[str]
    reverse: list[str]
    scoring: str
    kind: str
    structure: str
    role: str
    n_rows: int
    reliability: ScaleReliability
    correction: ScaleCorrection | None = None
    not_corrected: str | None = None
    # A correction blocked and recorded (the population answer with no design-based estimator,
    # MODELING_SEQUENCE §4): its ways forward, each a decision the client can post
    exits: list[dict[str, Any]] = []
    imputation: dict[str, Any] | None = None  # {m, imputed: {item: cells}}
    concerns: list[str] = []
    methods: str


class ScalesArtifact(_Model):
    purpose: Literal["inference", "prediction"]
    rows: Literal["all analyzed rows", "training rows"]
    scales: list[ScaleResult] = []
    methods: str


def _labels(task: str, source: str) -> list[str]:
    from turbotab.core.scales import APPROXIMATE, SECONDARY, TRANSIENT

    out = [SECONDARY]
    if source == "internal_consistency":
        out.append(TRANSIENT)
    if task in ("binary", "ordinal"):
        out.append(APPROXIMATE)
    return out


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _retest_score(frame: pd.DataFrame, spec: Any) -> np.ndarray | None:
    from turbotab.core.methods.scales import keyed_items, score_items

    if not spec.retest:
        return None
    if len(spec.retest) == 1:
        return pd.to_numeric(frame[spec.retest[0]], errors="coerce").to_numpy(dtype=float)
    renamed = frame[list(spec.retest)].set_axis(list(spec.items), axis=1)
    return score_items(keyed_items(renamed, spec.items, spec.reverse, spec.low, spec.high),
                       spec.scoring)


# MS4 (MODELING_SEQUENCE §4, "population estimand without a design-based estimator"): the
# calibration and the refit beside it are fit on these rows as sampled, so under the surveyed
# population the correction is blocked and recorded, with the sample-only attestation its exit.
POPULATION = ("The correction and the uncorrected coefficient beside it have no design-based "
              "estimator here: they would describe these participants, not the surveyed population "
              "the survey answer names, and their intervals would ignore the strata and PSUs.")


def population_block() -> tuple[str, list[dict[str, Any]]]:
    from turbotab.core.models.survey import SAMPLE_EXIT

    return POPULATION, [{"label": SAMPLE_EXIT,
                         "decision": {"kind": "set_survey", "estimand": "sample"}}]


def scale_result(spec: Any, copies: Sequence[pd.DataFrame], frame: pd.DataFrame, y: np.ndarray, *,
                 task: str, inference: bool, family: Any, pipeline: Any, outcome: Any,
                 seed: int, imputation: dict[str, Any] | None,
                 progress: Any = None,
                 blocked: tuple[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """One scale's result over the completed copies (one copy without multiple imputation).
    ``blocked``: the reason and exits that hold its correction (the population answer, MS4)."""
    from turbotab.core.methods import scales as S
    from turbotab.core.scales import FORMATIVE, PREDICTION

    nf = spec.group_factors()
    structure = spec.structure
    concerns: list[str] = []
    retest = _retest_score(frame, spec)
    reference = (pd.to_numeric(frame[spec.reference], errors="coerce").to_numpy(dtype=float)
                 if spec.reference else None)

    # ── reliability, copy by copy ──
    keyed_copies = [S.keyed_items(c, spec.items, spec.reverse, spec.low, spec.high) for c in copies]
    rel: dict[str, Any] = {"source": None, "factors": nf}
    omegas: list[Any] = []
    if spec.kind == "reflective":
        for K in keyed_copies:
            M = K.to_numpy(dtype=float)
            M = M[np.isfinite(M).all(axis=1)]
            try:
                omegas.append(S.omega(M, nf))
            except S.ScaleRefused as refused:
                rel["reason"] = str(refused)
                break
            rel["n"] = int(len(M))
    if omegas and len(omegas) == len(keyed_copies):
        coefficient, _ = omegas[0].coefficient(structure)
        values = [o.coefficient(structure)[1] for o in omegas]
        rel.update(source="internal_consistency", coefficient=coefficient,
                   value=_mean(values), omega_total=_mean([o.omega_total for o in omegas]),
                   omega_total_standardized=_mean([o.omega_total_standardized for o in omegas]),
                   alpha=_mean([o.alpha for o in omegas]), alpha_label=ALPHA_LABEL,
                   across_copies=values if len(values) > 1 else [])
        if nf > 1:
            rel.update(omega_hierarchical=_mean([o.omega_h for o in omegas]),
                       omega_hierarchical_standardized=_mean([o.omega_h_standardized
                                                              for o in omegas]))
        if omegas[0].caution:
            concerns.append(omegas[0].caution)
        negative = [c for c, g in zip(spec.items, omegas[0].general) if g < 0]
        if negative:
            concerns.append(
                f"{', '.join(f'`{c}`' for c in negative)} "
                f"{'loads' if len(negative) == 1 else 'load'} negatively on the scale's general "
                f"factor after the declared reverse coding. Either the instrument's key reverses "
                f"{'it' if len(negative) == 1 else 'them'} and the answer does not, "
                f"{'it was' if len(negative) == 1 else 'they were'} already reversed in the source, "
                f"or {'it does' if len(negative) == 1 else 'they do'} not belong to the scale; the "
                f"values cannot tell which, so nothing is reversed on a guess.")
    if spec.reliability == "test_retest" and retest is not None:
        iccs = []
        for K in keyed_copies:
            W1 = S.score_items(K, spec.scoring)
            try:
                found = S.retest_reliability(W1, retest)
            except S.ScaleRefused as refused:
                rel["reason"] = str(refused)
                iccs = []
                break
            iccs.append(found)
        if iccs:
            rel = {"source": "test_retest", "coefficient": "test_retest_icc", "factors": nf,
                   "value": _mean([r.icc for r in iccs]), "n": iccs[0].n,
                   "across_copies": [r.icc for r in iccs] if len(iccs) > 1 else [],
                   **{k: rel[k] for k in CARRIED if k in rel}}
    if spec.reliability == "calibration_substudy" and reference is not None:
        rel = {"source": "calibration_substudy", "coefficient": "calibration_slope", "factors": nf,
               "n": int(np.isfinite(reference).sum()),
               **{k: rel[k] for k in CARRIED if k in rel}}
    if spec.kind == "formative" and rel.get("source") is None:
        rel["reason"] = FORMATIVE
    if rel.get("coefficient"):
        rel["label"] = COEFFICIENT_LABELS[rel["coefficient"]]

    # ── the correction ──
    correction = None
    why = None
    exits: list[dict[str, Any]] = []
    if spec.correction == "none":
        why = None
    elif not inference:
        why = PREDICTION
    elif blocked is not None:
        why, exits = blocked[0], list(blocked[1])
    elif family is None or pipeline is None:
        why = (f"The correction refits a least-squares, logistic or proportional-odds model; choose "
               f"the {'proportional-odds' if task == 'ordinal' else 'linear'} family among the "
               f"models." if task in FAMILY_FOR else
               f"A {task.replace('_', '-')} outcome's coefficients are not corrected here.")
    elif spec.kind == "formative" and spec.reliability == "internal_consistency":
        why = FORMATIVE
    elif any(c.isna().any().any() for c in copies):
        why = ("Blanks among the model's inputs are filled once here, and the correction would "
               "treat each filled value as measured; answer multiple imputation to correct the "
               "score.")
    else:
        try:
            correction = _correct(spec, copies, keyed_copies, y, retest, reference, task=task,
                                  family=family, pipeline=pipeline, outcome=outcome,
                                  seed=seed, progress=progress)
        except S.ScaleRefused as refused:
            why = str(refused)
    if correction is not None:
        if spec.reliability == "calibration_substudy":
            rel["value"] = correction["attenuation"]
        if spec.reliability == "internal_consistency" and nf > 1:
            concerns.append("The correction uses ω-hierarchical, so the group factors' variance "
                            "counts as error about the general construct; if a group factor itself "
                            "relates to the outcome, the corrected estimate is off by that much.")
        short = correction["n_boot"] - correction["n_boot_ok"]
        if short:
            concerns.append(f"{correction['n_boot_ok']:,} of {correction['n_boot']:,} bootstrap "
                            f"replicates could be recalibrated; the interval rests on those.")
    return {"name": spec.name, "items": list(spec.items), "reverse": list(spec.reverse),
            "scoring": spec.scoring, "kind": spec.kind, "structure": structure, "role": spec.role,
            "n_rows": int(len(copies[0])), "reliability": rel, "correction": correction,
            "not_corrected": why, "exits": exits, "imputation": imputation, "concerns": concerns}


def _correct(spec: Any, copies: Sequence[pd.DataFrame], keyed_copies: Sequence[pd.DataFrame],
             y: np.ndarray, retest: np.ndarray | None, reference: np.ndarray | None, *, task: str,
             family: Any, pipeline: Any, outcome: Any, seed: int,
             progress: Any = None) -> dict[str, Any]:
    """The correction in each copy, combined by Rubin's rules (one copy: as it is)."""
    from sklearn.base import clone

    from turbotab.core.methods import scales as S
    from turbotab.core.methods.imputation import pool_rows, pool_scalar
    from turbotab.core.models.inference import INDEPENDENT
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.linear import model_matrix

    fit = S.FITS[task]
    corrections, tables, covariates = [], [], []
    for k, (X_k, K) in enumerate(zip(copies, keyed_copies)):
        if progress is not None:
            progress(k, len(copies))
        fitted = fit_pipeline(clone(pipeline), X_k, y)
        matrix = model_matrix(fitted, X_k)
        names = [str(c) for c in matrix.columns]
        if spec.name not in names:
            raise S.ScaleRefused(f"`{spec.name}` is not one column of the model's matrix, so its "
                                 f"coefficient has no single correction.")
        j = names.index(spec.name)
        values = matrix.to_numpy(dtype=float)
        W, Z = values[:, j], np.delete(values, j, axis=1)
        covariates = [c for c in names if c != spec.name]
        if spec.reliability == "internal_consistency":
            source = S.Source("internal_consistency", reliability=S.internal_consistency(
                K.to_numpy(dtype=float), spec.scoring, spec.structure, spec.group_factors()))
        elif spec.reliability == "test_retest":
            source = S.Source("test_retest", reliability=S.retest_source(W, retest))
        else:
            source = S.Source("calibration_substudy", reference=reference)
        corrections.append(S.correct(W, Z, y, j, source, fit=fit, n_boot=spec.n_boot,
                                     seed=seed + k))
        table = family.inference_matrix(matrix, y, task=task,
                                        classes=list(getattr(fitted[-1], "classes_", [])) or None,
                                        clusters=INDEPENDENT, outcome=outcome)
        tables.append(table.rows)
    naive_row = next(r for r in (pool_rows(tables) if len(tables) > 1 else tables[0])
                     if str(r["feature"]) == spec.name)
    if len(corrections) == 1:
        c = corrections[0]
        estimate, se, low, high = c.estimate, c.se, c.ci_low, c.ci_high
    else:
        if any(c.se is None for c in corrections):
            raise S.ScaleRefused("Too few bootstrap replicates could be recalibrated in a completed "
                                 "copy to give its variance.")
        pooled = pool_scalar([c.estimate for c in corrections], [c.se ** 2 for c in corrections])
        estimate, se, low, high = (pooled.estimate, float(np.sqrt(pooled.total)), pooled.ci_low,
                                   pooled.ci_high)
    ratio = task in ("binary", "ordinal")
    exp = (lambda v: None if v is None else float(np.exp(v)))
    return {
        "family": "proportional_odds" if task == "ordinal" else "linear", "feature": spec.name,
        "naive": float(naive_row["estimate"]), "naive_ci_low": naive_row.get("ci_low"),
        "naive_ci_high": naive_row.get("ci_high"), "p": naive_row.get("p"),
        "estimate": float(estimate), "se": se, "ci_low": low, "ci_high": high,
        "scale": "odds_ratio" if ratio else "difference",
        "ratio": exp(estimate) if ratio else None, "ratio_low": exp(low) if ratio else None,
        "ratio_high": exp(high) if ratio else None,
        "naive_ratio": exp(float(naive_row["estimate"])) if ratio else None,
        "attenuation": float(np.mean([c.attenuation for c in corrections])),
        "error_variance": _mean([c.error_variance for c in corrections]),
        "covariates": covariates, "n": int(corrections[0].n),
        "n_boot": int(spec.n_boot), "n_boot_ok": int(min(c.n_boot_ok for c in corrections)),
        "copies": len(corrections),
        "labels": _labels(task, spec.reliability)}


def scales_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.methods.imputation import ImputationRefused
    from turbotab.core.methods.missing import impute_for_inference
    from turbotab.core.models import get_family
    from turbotab.core.models.inference import Outcome
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.scales import methods_sentence
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import _task, coded_outcome, outcome_levels, read_assignment

    state = ctx.state
    specs = list(state.scales or [])
    inference = state.purpose == "inference"
    task = _task(ctx)
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    assignment = read_assignment(ctx.inputs["split"])
    ids = (assignment.index if inference else assignment.index[assignment["train"]]).to_numpy()
    target = state.target
    extra = [c for s in specs for c in [*s.retest, *([s.reference] if s.reference else [])]
             if c not in spec.inputs]
    ctx.progress(0.02, "Reading the analysis rows")
    with open_store(ctx) as store:
        frame = modeling_frame(store, list(dict.fromkeys([*spec.inputs, *extra, target])), ids,
                               outcome=target)
    raw = frame[target].to_numpy()
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        y, _ = ordinal_outcome(raw, state.outcome_order, column=target)
    else:
        y = np.asarray(coded_outcome(task, raw, state.event))
    outcome = Outcome(name=target, labels=outcome_levels(task, raw, state.event))
    X = frame[list(spec.inputs)]
    seed = int(getattr(state.split, "seed", 0) or 0) if state.split is not None else 0

    copies: list[pd.DataFrame] = [X]
    imputation = None
    if inference and spec.multiple_imputation() and X.isna().any().any():
        ctx.progress(0.05, "Imputing the items, with the outcome, before scoring")
        try:
            imputations = impute_for_inference(spec, X, y, task, seed=seed,
                                               cancelled=ctx.cancelled)
        except ImputationRefused as exc:
            imputations = None
            imputation = {"refused": str(exc)}
        if imputations is not None:
            copies = list(imputations.frames)
            items = {c for s in specs for c in s.items}
            imputation = {"m": int(imputations.m),
                          "imputed": {str(c): int(n) for c, n in imputations.imputed.items()
                                      if c in items}}
    key = FAMILY_FOR.get(task)
    family = get_family(key) if key and key in (state.models or []) and key in pipelines else None
    pipeline = pipelines.get(key) if family is not None else None
    survey = getattr(state, "survey", None)
    blocked = (population_block() if inference and survey is not None
               and survey.estimand == "population" else None)
    out = []
    for n, s in enumerate(specs):
        def progress(k: int, total: int, _n: int = n, _name: str = s.name) -> None:
            ctx.progress(0.1 + 0.85 * (_n + k / max(total, 1)) / max(len(specs), 1),
                         f"`{_name}`: completed copy {k + 1} of {total}" if total > 1
                         else f"`{_name}`: regression calibration with its bootstrap")

        result = scale_result(s, copies, frame, y, task=task, inference=inference, family=family,
                              pipeline=pipeline, outcome=outcome, seed=seed,
                              imputation=imputation if imputation and imputation.get("m") else None,
                              progress=progress, blocked=blocked)
        out.append(result)
    corrected = [r for r in out if r["correction"] is not None]
    if len(corrected) > 1:
        for r in corrected:
            r["concerns"].append(
                f"{len(corrected)} scores are corrected, each on its own. With two or more "
                f"error-prone exposures in one model, estimates “may become attenuated, "
                f"inflated, or can even change direction” (Freedman et al. 2011); a univariate "
                f"correction does not undo that.")
    for r in corrected:
        grain = getattr(state, "grain", None)
        if grain is not None and grain.grain == "repeated" and state.unit == "row":
            r["concerns"].append("Rows repeat by unit, and the bootstrap resamples rows, so its "
                                 "interval treats a unit's rows as independent.")
    for r, s in zip(out, specs):
        r["methods"] = methods_sentence(s, r)
    artifact = ScalesArtifact(
        purpose="inference" if inference else "prediction",
        rows="all analyzed rows" if inference else "training rows",
        scales=[ScaleResult(**r) for r in out],
        methods=" ".join(r["methods"] for r in out) or "No multi-item scale was scored.")
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


SCALES_READS = ("scales", "purpose", "models", "task", "event", "target", "outcome_order",
                "missing", "split", "roles", "roles_unconfirmed", "role_confirmations",
                "reading_confirmations", "shape_confirmations", "survey", "grain", "unit")

__all__ = ["POPULATION", "SCALES_READS", "ScaleCorrection", "ScaleReliability", "ScaleResult",
           "ScalesArtifact", "population_block", "scale_result", "scales_stage"]

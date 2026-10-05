"""The ``scales`` stage (MS8): each declared scale's reliability and, under inference, its
coefficient corrected for measurement error as a declared secondary analysis.

The arithmetic and its sources are in :mod:`turbotab.core.methods.scales`, the leash in
:mod:`turbotab.core.scales`. Here the rows, the copies, the clusters and the model matrix:

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
* **Clusters** (MODELING_SEQUENCE §2: repeated units or clusters imply cluster-aware intervals;
  ruling 7: the calibration's bootstrap resamples clusters). The rows are grouped exactly as the
  fit's coefficient table groups them (``models.inference.resolve_clusters`` over the same rows and
  columns): the uncorrected interval beside the correction is that table's (CR2 by the grouping),
  and the correction's bootstrap resamples whole clusters. Grouped rows the table cannot give an
  interval for (too few clusters, or a grouping nobody settled) block the correction with the
  table's own reason and exits.
* **The matrix.** The family whose coefficient is corrected is refit on each copy: the linear
  family for a least-squares or logistic outcome, the proportional-odds family for an ordinal one.
  Every corrected score's column of its model matrix is one W, every other column a covariate of
  the calibration (Boe et al. 2023); several corrected scores are calibrated jointly (Rosner,
  Spiegelman & Willett 1990). The uncorrected estimate beside each is that refit's, with the
  interval its own table gives.
* **The values read** (BLUEPRINT §14.3; MODELING_SEQUENCE §4, codes counted as answers). The answer
  was refused while any item, repeat administration or reference held a code; the values are read
  again here, and a code that reached them since (a repair undone) blocks what it would move.
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
    attenuation: float  # λ given the covariates (and the scores corrected with it; mean over copies)
    error_variance: float | None = None
    covariates: list[str] = []  # every other column of the model's matrix
    jointly: list[str] = []  # the other scores calibrated with it (multivariate calibration)
    clustered_by: str | None = None  # the grouping both intervals keep together
    n_clusters: int | None = None
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
    # MODELING_SEQUENCE §4; grouped rows with no interval; a code in the values it reads): its ways
    # forward, each a decision the client can post
    exits: list[dict[str, Any]] = []
    imputation: dict[str, Any] | None = None  # {m, imputed: {item: cells}}
    concerns: list[str] = []
    methods: str


class ScalesArtifact(_Model):
    purpose: Literal["inference", "prediction"]
    rows: Literal["all analyzed rows", "training rows"]
    scales: list[ScaleResult] = []
    methods: str


def _labels(task: str, source: str, *, jointly: Sequence[str] = (),
            clustered_by: str | None = None, n_clusters: int | None = None) -> list[str]:
    from turbotab.core.scales import APPROXIMATE, SECONDARY, TRANSIENT, clustered_label, joint_label

    out = [SECONDARY]
    if source == "internal_consistency":
        out.append(TRANSIENT)
    if task in ("binary", "ordinal"):
        out.append(APPROXIMATE)
    if jointly:
        out.append(joint_label(jointly))
    if clustered_by:
        out.append(clustered_label(clustered_by, n_clusters))
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


# ── the values read, again at the stage (a code that reached them since the answer) ──


class Codes:
    """What a code in a scale's values blocks: ``items`` (no reliability, no correction),
    ``retest`` (no test–retest ICC), ``reference`` (no substudy correction); ``reason`` and
    ``exits``."""

    def __init__(self, where: str, reason: str, exits: Sequence[dict[str, Any]]):
        self.where, self.reason, self.exits = where, reason, list(exits)


def _uncorrected_exit(spec: Any, specs: Sequence[Any]) -> dict[str, Any]:
    """The scales answer with this scale's coefficient left uncorrected (and no repeat
    administration or reference read), in its JSON form."""
    from turbotab.core.decisions import SetScales

    update = {"correction": "none", "reliability": "internal_consistency", "retest": [],
              "reference": None}
    decision = SetScales(scales=[x.model_copy(update=update) if x.name == spec.name else x
                                 for x in specs])
    return {"label": f"Report `{spec.name}`'s uncorrected estimate only",
            "decision": decision.model_dump(mode="json")}


def _outside(frame: pd.DataFrame, columns: Sequence[str], low: float, high: float) -> list[str]:
    said = []
    for c in columns:
        x = pd.to_numeric(frame[c], errors="coerce")
        if x.notna().any() and (x.min() < low or x.max() > high):
            said.append(f"`{c}` {x.min():g}–{x.max():g}")
    return said


def codes_in_values(spec: Any, frame: pd.DataFrame, state: Any, store: Any = None) -> Codes | None:
    """The scale's values as the stage reads them: the items (and a repeat administration's items)
    on the response scale, a repeat administration recorded as one score within the range its score
    can take, a reference measure with a settled amount reading and no missing-value code."""
    from turbotab.core.readings import code_or_count_reading, confirmation, whole_facts
    from turbotab.core.scales import codes_beyond, score_range

    again = {"label": f"Recode the codes to missing (the findings' code repair), then answer the "
                      f"scales question for `{spec.name}` again", "decision": None}
    for where, columns in (("items", spec.items),
                           ("retest", spec.retest if len(spec.retest) == len(spec.items) else [])):
        outside = _outside(frame, columns, spec.low, spec.high)
        if outside:
            return Codes(where, f"Values outside the {spec.low}–{spec.high} response scale reached "
                                f"the answers ({'; '.join(outside[:4])}): codes counted as answers "
                                f"would move the score, its reliability and the correction, so "
                                f"{'none of them is' if where == 'items' else 'the test–retest ICC and the correction are not'} "
                                f"computed.", [again])
    if len(spec.retest) == 1:
        x = pd.to_numeric(frame[spec.retest[0]], errors="coerce")
        lo, hi = score_range(spec)
        if x.notna().any() and (x.min() < lo or x.max() > hi):
            return Codes("retest", f"`{spec.retest[0]}`, the repeat administration, holds values "
                                   f"from {x.min():g} to {x.max():g}, outside the {lo:g}–{hi:g} "
                                   f"the score can take: codes counted as scores would move the "
                                   f"test–retest ICC and the correction, so neither is computed.",
                         [again])
    if spec.reference:
        column = spec.reference
        x = pd.to_numeric(frame[column], errors="coerce")
        settled = confirmation(state, "code_or_count", column) != "code"
        if settled and store is not None:
            facts = whole_facts([column], None, store).get(column)
            found = code_or_count_reading(state, column, facts, scope="fit") if facts else None
            settled = found is None or found.settled
        if not settled:
            return Codes("reference", f"Whether `{column}`'s values are amounts or codes is not "
                                      f"settled, and the calibration regresses them on the score "
                                      f"as amounts, so the correction is not computed.",
                         [{"label": "Answer the scales question again (it asks for the reading)",
                           "decision": None}])
        found_codes = codes_beyond(x.to_numpy(dtype=float))
        if found_codes:
            return Codes("reference", f"`{column}`, the reference measure, holds "
                                      f"{', '.join(f'{v:g}' for v in found_codes)}, beyond every "
                                      f"other value with a wide gap: a missing-value code, which "
                                      f"would move the calibration's slope, so the correction is "
                                      f"not computed.", [again])
    return None


# ── each scale's reliability ─────────────────────────────────────────────────


def reliability_of(spec: Any, keyed_copies: Sequence[pd.DataFrame],
                   retest: np.ndarray | None,
                   reference: np.ndarray | None) -> tuple[dict[str, Any], list[str]]:
    """The scale's reliability over the completed copies (one without multiple imputation), and the
    concerns it raises."""
    from turbotab.core.methods import scales as S
    from turbotab.core.scales import FORMATIVE

    nf = spec.group_factors()
    structure = spec.structure
    concerns: list[str] = []
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
    if spec.kind == "formative" and rel.get("source") is None and not rel.get("reason"):
        rel["reason"] = FORMATIVE
    if rel.get("coefficient"):
        rel["label"] = COEFFICIENT_LABELS[rel["coefficient"]]
    return rel, concerns


# ── the correction ───────────────────────────────────────────────────────────


def _why_not(spec: Any, specs: Sequence[Any], *, inference: bool, task: str,
             copies: Sequence[pd.DataFrame], family: Any, pipeline: Any,
             blocked: tuple[str, list[dict[str, Any]]] | None,
             grouped: tuple[str, Sequence[dict[str, Any]]] | None,
             codes: Codes | None) -> tuple[str | None, list[dict[str, Any]], bool]:
    """``(why not, exits, eligible)`` for one scale's correction."""
    from turbotab.core.scales import FORMATIVE, PREDICTION

    if spec.correction == "none":
        return None, [], False
    if not inference:
        return PREDICTION, [], False
    if blocked is not None:
        return blocked[0], list(blocked[1]), False
    if codes is not None:
        return codes.reason, [*codes.exits, _uncorrected_exit(spec, specs)], False
    if grouped is not None:
        return grouped[0], [dict(e) for e in grouped[1]], False
    if family is None or pipeline is None:
        return ((f"The correction refits a least-squares, logistic or proportional-odds model; "
                 f"choose the {'proportional-odds' if task == 'ordinal' else 'linear'} family "
                 f"among the models." if task in FAMILY_FOR else
                 f"A {task.replace('_', '-')} outcome's coefficients are not corrected here."),
                [], False)
    if spec.kind == "formative" and spec.reliability == "internal_consistency":
        return FORMATIVE, [], False
    if any(c.isna().any().any() for c in copies):
        return ("Blanks among the model's inputs are filled once here, and the correction would "
                "treat each filled value as measured; answer multiple imputation to correct the "
                "score.", [], False)
    return None, [], True


def _correct_jointly(specs: Sequence[Any], copies: Sequence[pd.DataFrame],
                     keyed: Mapping[str, Sequence[pd.DataFrame]], y: np.ndarray,
                     retests: Mapping[str, np.ndarray | None],
                     references: Mapping[str, np.ndarray | None], *, task: str, family: Any,
                     pipeline: Any, outcome: Any, seed: int, clusters: Any,
                     progress: Any = None) -> dict[str, dict[str, Any]]:
    """Every score in ``specs`` corrected together in each copy, each combined over the copies by
    Rubin's rules (one copy: as it is). Raises ``ScaleRefused`` naming the score whose data refuse
    (``which``, its position in ``specs``), or None when they refuse together."""
    from sklearn.base import clone

    from turbotab.core.methods import scales as S
    from turbotab.core.methods.imputation import pool_rows, pool_scalar
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.linear import model_matrix

    fit = S.FITS[task]
    groups = clusters.codes if clusters.clustered else None
    n_boot = max(int(s.n_boot) for s in specs)
    per_copy: list[list[Any]] = []
    tables, covariates = [], {}
    for k, X_k in enumerate(copies):
        if progress is not None:
            progress(k, len(copies))
        fitted = fit_pipeline(clone(pipeline), X_k, y)
        matrix = model_matrix(fitted, X_k)
        names = [str(c) for c in matrix.columns]
        for i, s in enumerate(specs):
            if s.name not in names:
                raise S.ScaleRefused(f"`{s.name}` is not one column of the model's matrix, so its "
                                     f"coefficient has no single correction.", which=i)
        js = [names.index(s.name) for s in specs]
        values = matrix.to_numpy(dtype=float)
        W, Z = values[:, js], np.delete(values, js, axis=1)
        covariates = {s.name: [c for c in names if c != s.name] for s in specs}
        sources = []
        for i, s in enumerate(specs):
            if s.reliability == "internal_consistency":
                sources.append(S.Source("internal_consistency", reliability=S.internal_consistency(
                    keyed[s.name][k].to_numpy(dtype=float), s.scoring, s.structure,
                    s.group_factors())))
            elif s.reliability == "test_retest":
                sources.append(S.Source("test_retest",
                                        reliability=S.retest_source(W[:, i], retests[s.name])))
            else:
                sources.append(S.Source("calibration_substudy", reference=references[s.name]))
        per_copy.append(S.correct_jointly(W, Z, y, js, sources, fit=fit, n_boot=n_boot,
                                          seed=seed + k, groups=groups))
        table = family.inference_matrix(matrix, y, task=task,
                                        classes=list(getattr(fitted[-1], "classes_", [])) or None,
                                        clusters=clusters, outcome=outcome)
        tables.append(table.rows)
    rows = pool_rows(tables) if len(tables) > 1 else tables[0]
    ratio = task in ("binary", "ordinal")
    exp = (lambda v: None if v is None else float(np.exp(v)))
    clustered_by = clusters.column if groups is not None else None
    out: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(specs):
        corrections = [copy[i] for copy in per_copy]
        naive_row = next(r for r in rows if str(r["feature"]) == s.name)
        if len(corrections) == 1:
            c = corrections[0]
            estimate, se, low, high = c.estimate, c.se, c.ci_low, c.ci_high
        else:
            if any(c.se is None for c in corrections):
                raise S.ScaleRefused("Too few bootstrap replicates could be recalibrated in a "
                                     "completed copy to give its variance.", which=i)
            pooled = pool_scalar([c.estimate for c in corrections],
                                 [c.se ** 2 for c in corrections])
            estimate, se, low, high = (pooled.estimate, float(np.sqrt(pooled.total)),
                                       pooled.ci_low, pooled.ci_high)
        jointly = [x.name for x in specs if x.name != s.name]
        n_clusters = corrections[0].n_clusters
        out[s.name] = {
            "family": "proportional_odds" if task == "ordinal" else "linear", "feature": s.name,
            "naive": float(naive_row["estimate"]), "naive_ci_low": naive_row.get("ci_low"),
            "naive_ci_high": naive_row.get("ci_high"), "p": naive_row.get("p"),
            "estimate": float(estimate), "se": se, "ci_low": low, "ci_high": high,
            "scale": "odds_ratio" if ratio else "difference",
            "ratio": exp(estimate) if ratio else None, "ratio_low": exp(low) if ratio else None,
            "ratio_high": exp(high) if ratio else None,
            "naive_ratio": exp(float(naive_row["estimate"])) if ratio else None,
            "attenuation": float(np.mean([c.attenuation for c in corrections])),
            "error_variance": _mean([c.error_variance for c in corrections]),
            "covariates": covariates[s.name], "jointly": jointly,
            "clustered_by": clustered_by, "n_clusters": n_clusters,
            "n": int(corrections[0].n), "n_boot": n_boot,
            "n_boot_ok": int(min(c.n_boot_ok for c in corrections)), "copies": len(corrections),
            "labels": _labels(task, s.reliability, jointly=jointly, clustered_by=clustered_by,
                              n_clusters=n_clusters)}
    return out


def correct_together(specs: Sequence[Any], **kwargs: Any) -> tuple[dict[str, dict[str, Any]],
                                                                    dict[str, str]]:
    """The scores in ``specs`` corrected jointly; a score whose own data refuse leaves the set
    with its reason (the others are corrected without it), and a refusal of the set as a whole is
    every score's. Returns ``(corrections, refusals)`` by name."""
    from turbotab.core.methods import scales as S

    remaining = list(specs)
    refused: dict[str, str] = {}
    while remaining:
        try:
            return _correct_jointly(remaining, **kwargs), refused
        except S.ScaleRefused as why:
            if why.which is None or len(remaining) == 1:
                refused.update({s.name: str(why) for s in remaining})
                return {}, refused
            refused[remaining[why.which].name] = str(why)
            remaining.pop(why.which)
    return {}, refused


def scales_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.methods import scales as S
    from turbotab.core.methods.imputation import ImputationRefused
    from turbotab.core.methods.missing import impute_for_inference
    from turbotab.core.models import get_family
    from turbotab.core.models.inference import (INDEPENDENT, Outcome, cluster_columns,
                                                floor_refusal, resolve_clusters)
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.scales import methods_sentence, uncorrected_alongside
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import _task, coded_outcome, outcome_levels, read_assignment

    state = ctx.state
    specs = list(state.scales or [])
    inference = state.purpose == "inference"
    task = _task(ctx)
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    split = ctx.inputs["split"]
    assignment = read_assignment(split)
    grouped_by = (split.data or {}).get("grouped_by") if isinstance(split, Bundle) else None
    ids = (assignment.index if inference else assignment.index[assignment["train"]]).to_numpy()
    target = state.target
    extra = [c for s in specs for c in [*s.retest, *([s.reference] if s.reference else [])]
             if c not in spec.inputs]
    ctx.progress(0.02, "Reading the analysis rows")
    with open_store(ctx) as store:
        # Under inference the rows are grouped as the coefficient table groups them (fit stage).
        unit_columns = cluster_columns(state, store.columns, [grouped_by]) if inference else []
        frame = modeling_frame(store, list(dict.fromkeys([*spec.inputs, *extra, *unit_columns,
                                                          target])), ids, outcome=target)
        codes = {s.name: codes_in_values(s, frame, state, store) for s in specs}
    clusters = (resolve_clusters(state, frame.loc[:, unit_columns], [grouped_by]) if inference
                else INDEPENDENT)
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
    grouped = floor_refusal(clusters, task) if inference else None

    keyed = {s.name: [S.keyed_items(c, s.items, s.reverse, s.low, s.high) for c in copies]
             for s in specs}
    out: list[dict[str, Any]] = []
    eligible = []
    retests: dict[str, np.ndarray | None] = {}
    references: dict[str, np.ndarray | None] = {}
    for s in specs:
        found = codes[s.name]
        retests[s.name] = None if found is not None and found.where in ("items", "retest") \
            else _retest_score(frame, s)
        references[s.name] = (pd.to_numeric(frame[s.reference], errors="coerce")
                              .to_numpy(dtype=float) if s.reference and found is None else None)
        if found is not None and found.where == "items":
            rel, concerns = {"source": None, "factors": s.group_factors(),
                             "reason": found.reason}, []
        else:
            rel, concerns = reliability_of(s, keyed[s.name], retests[s.name], references[s.name])
            if found is not None and found.where == "retest" and rel.get("source") is None:
                rel["reason"] = found.reason
        why, exits, ok = _why_not(s, specs, inference=inference, task=task, copies=copies,
                                  family=family, pipeline=pipeline, blocked=blocked,
                                  grouped=grouped, codes=found)
        if ok:
            eligible.append(s)
        out.append({"name": s.name, "items": list(s.items), "reverse": list(s.reverse),
                    "scoring": s.scoring, "kind": s.kind, "structure": s.structure,
                    "role": s.role, "n_rows": int(len(copies[0])), "reliability": rel,
                    "correction": None, "not_corrected": why, "exits": exits,
                    "imputation": imputation if imputation and imputation.get("m") else None,
                    "concerns": concerns})

    def progress(k: int, total: int) -> None:
        names = ", ".join(f"`{s.name}`" for s in eligible)
        ctx.progress(0.1 + 0.85 * k / max(total, 1),
                     f"{names}: completed copy {k + 1} of {total}" if total > 1
                     else f"{names}: regression calibration with its bootstrap")

    corrections, refused = (correct_together(
        eligible, copies=copies, keyed=keyed, y=y, retests=retests, references=references,
        task=task, family=family, pipeline=pipeline, outcome=outcome, seed=seed,
        clusters=clusters, progress=progress) if eligible else ({}, {}))
    for r, s in zip(out, specs):
        correction = corrections.get(s.name)
        if s.name in refused:
            r["not_corrected"] = refused[s.name]
        if correction is None:
            continue
        r["correction"] = correction
        if s.reliability == "calibration_substudy":
            r["reliability"]["value"] = correction["attenuation"]
        if s.reliability == "internal_consistency" and s.group_factors() > 1:
            r["concerns"].append("The correction uses ω-hierarchical, so the group factors' "
                                 "variance counts as error about the general construct; if a group "
                                 "factor itself relates to the outcome, the corrected estimate is "
                                 "off by that much.")
        short = correction["n_boot"] - correction["n_boot_ok"]
        if short:
            r["concerns"].append(f"{correction['n_boot_ok']:,} of {correction['n_boot']:,} "
                                 f"bootstrap replicates could be recalibrated; the interval rests "
                                 f"on those.")
        others = [x.name for x in specs if x.name != s.name and x.name not in corrections
                  and x.name in correction["covariates"]]
        if others:
            r["concerns"].append(uncorrected_alongside(others))
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
                "reading_confirmations", "shape_confirmations", "survey", "grain", "unit",
                "clusters", "codebooks")

__all__ = ["Codes", "POPULATION", "SCALES_READS", "ScaleCorrection", "ScaleReliability",
           "ScaleResult", "ScalesArtifact", "codes_in_values", "correct_together",
           "population_block", "reliability_of", "scales_stage"]

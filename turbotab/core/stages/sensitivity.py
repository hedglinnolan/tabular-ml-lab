"""The ``sensitivity`` stage: the primary analysis beside the same model on other row sets (ME-16).

Banna et al. 2017 (*Front Nutr* 4:45), on screens for implausible energy reports: "Regardless of
which method is used, for the time being, analyses in the total sample without exclusion of
participants should also be conducted and reported. This will allow the researcher to examine the
impact of using each method". The ``set_sensitivity`` answer names analyses beside the primary, each
with its own exclusion rules (a fixed kcal screen, the Goldberg screen, none). This stage fits each
chosen family on the rows each analysis keeps and reports its coefficients side by side, so a reader
sees how far the screen moves the answer. When the primary excludes rows and no analysis keeps
every row, the every-row analysis is added, and the artifact says it was added and why.

**Rows, by purpose** (BLUEPRINT §12 ruling 3): under inference every analysis is estimated on all the
rows its rules keep, held-out rows included, with the intervals the fit stage makes (HC3, or CR2 by
the unit when one repeats; ``models/inference.py``). Under prediction no analysis reads a sealed row
(the split's ``sealed`` frame: every row drawn to be held out, including rows the primary's rules
exclude), so the seal holds; an analysis that relaxes the primary's rules gains the excluded rows
that were never drawn. A concern says what an exclusion means for a prediction model: the people it
leaves out are still in the population the model will be used on.

**The model** is the design's own pipeline, refit per analysis (every step, energy adjustment and
imputation included, is fit on that analysis's rows), so the analyses differ only by their rows.
Families without a coefficient table are left out of this answer.

A rule that reads the outcome is refused per analysis, as the primary's would be (audit RO-01). A
Goldberg screen reads body weight; when the outcome correlates with it, a concern quotes Banna et
al.: body weight "is included in both the calculation of implausible rEI and the outcome variable …
which could artificially elevate the association".
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.models.artifacts import Coefficient, Inference

BANNA = ("Banna et al. 2017: “analyses in the total sample without exclusion of participants should "
         "also be conducted and reported”.")
EVERY_ROW = "Every row"
WEIGHT_CORRELATION = 0.5  # |r| between the outcome and a weight the screen reads: say so


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class SensitivityRow(_Model):
    """One analysis: its rules (as the participant flow labels them) and the rows they keep."""

    label: str
    primary: bool
    added: bool  # the every-row analysis the stage added (Banna 2017), not one the user named
    rules: list[str]
    n_rows: int
    refused: str | None = None  # why this analysis was not fit


class SensitivityFit(_Model):
    label: str  # the analysis
    n_rows: int
    coefficients: list[Coefficient] | None
    inference: Inference | None = None
    concerns: list[str] = []


class SensitivityFamily(_Model):
    family: str
    label: str
    fits: list[SensitivityFit]  # one per analysis, in the analyses' order


class SensitivityChange(_Model):
    """How an exposure's estimate moves across the analyses, against the primary's."""

    feature: str
    primary: float | None
    lowest: float | None
    highest: float | None
    sign_changes: bool  # some analysis's estimate has the other sign
    excludes_zero: list[bool | None]  # per analysis: the interval excludes zero (None: no interval)


class SensitivityArtifact(_Model):
    purpose: Literal["inference", "prediction"]
    rows: Literal["all eligible rows", "eligible rows outside the held-out set"]
    analyses: list[SensitivityRow]
    exposures: list[str]  # the model-matrix columns that came from exposure columns
    families: list[SensitivityFamily]
    changes: dict[str, list[SensitivityChange]]  # family -> exposures
    concerns: list[str]
    methods: str  # the methods sentence


# ── the row sets ─────────────────────────────────────────────────────────────


def analyses_of(state: Any) -> list[dict[str, Any]]:
    """The primary, the named analyses, and the every-row analysis when it is owed (Banna 2017)."""
    from turbotab.core.decisions import as_rule

    primary = [as_rule(r) for r in (state.exclusions or [])]
    out = [{"label": "Primary", "rules": primary, "primary": True, "added": False}]
    named = list(state.sensitivity or [])
    for a in named:
        out.append({"label": a.label, "rules": [as_rule(r) for r in a.rules], "primary": False,
                    "added": False})
    if primary and not any(not a.rules for a in named):
        out.append({"label": EVERY_ROW, "rules": [], "primary": False, "added": True})
    return out


def sealed_rows(split: Any) -> np.ndarray:
    """Every row drawn to be held out (the split's ``sealed`` frame), in or out of the cohort; for
    a split artifact without that frame, its held-out rows."""
    from turbotab.core.stages.modeling import read_assignment

    frames = getattr(split, "frames", None) or {}
    sealed = frames.get("sealed")
    if sealed is not None:
        return np.asarray(sealed["row_id"], dtype=np.int64)
    assignment = read_assignment(split)
    return assignment.index[~assignment["train"]].to_numpy(dtype=np.int64)


def rows_of(store: Any, state: Any, ingest: Mapping[str, Any], rules: Sequence[Any]) -> np.ndarray:
    """The row ids the cohort keeps under ``rules`` in place of the primary's (every other answer,
    missing values and repairs included, as recorded)."""
    from turbotab.core.stages.rows import compute_cohort

    _, kept, _ = compute_cohort(store, state.model_copy(update={"exclusions": list(rules)}), ingest)
    return np.asarray(kept, dtype=np.int64)


def fit_on_rows(state: Any, family: Any, pipeline: Any, frame: pd.DataFrame, inputs: Sequence[str],
                y: np.ndarray, task: str, unit_columns: Sequence[str], *, outcome: Any = None,
                survey: Any = None, levels: Sequence[str] | None = None, missing: Any = None,
                spec: Any = None) -> tuple[Any, Any, list[str]]:
    """Refit ``pipeline`` on ``frame``'s rows; its coefficient table and concerns, as fit makes them.

    As the fit stage makes them (``stages.modeling.fit_stage``): a family that models the unit is
    told each row's unit (WP12b); the table is on the outcome's scale, its odds or relative-risk
    ratios named by ``outcome`` (WP8); under the "surveyed population" answer it is design-based,
    these rows its domain (``survey``, a ``FitSurvey``; WP10), or blocked while the survey question
    is unanswered; an ordinal model's cut-points carry the declared ``levels`` (WP12a). Under
    inference ``missing`` (``stages.modeling.TableMissing``) is the missing-values answer on these
    rows: the table pooled over their own multiple imputations, held while a single fill or the
    missing-indicator method is unrecorded, or carrying complete cases' assumption (WP7)."""
    from sklearn.base import clone

    from turbotab.core.models.inference import resolve_clusters
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.stages.modeling import _inference_table, with_units

    X = frame[list(inputs)]
    clusters = resolve_clusters(state, frame[list(unit_columns)]) if unit_columns else None
    units = (pd.Series(clusters.codes, index=frame.index)
             if clusters is not None and clusters.clustered else None)
    fitted = fit_pipeline(with_units(clone(pipeline), units), X, y)
    if levels is not None:
        fitted[-1].level_names_ = list(levels)
    concerns: list[str] = []
    info = None
    if state.purpose == "inference" and hasattr(family, "inference"):
        from turbotab.core.models.survey import blocked

        design = survey.design if survey is not None else None
        if survey is not None and survey.refusal:
            table = blocked(survey.refusal, survey.exits)
        elif missing is not None and missing.refusal:
            table = blocked(missing.refusal, missing.exits,
                            estimator="not fitted: the missing-values answer decides it")
        else:
            if missing is not None and missing.imputations is not None:
                from turbotab.core.stages.modeling import _with_levels, pooled_table

                table, _, _, _ = pooled_table(
                    family, pipeline, missing.imputations, y, task=task, clusters=clusters,
                    outcome=outcome, survey=design, design=None, spec=spec, rows="all",
                    fit=lambda model, X_k: _with_levels(
                        fit_pipeline(with_units(model, units), X_k, y), levels),
                    energy_rows=False)
            else:
                table = _inference_table(family, fitted, X, y, task=task, clusters=clusters,
                                         outcome=outcome, rows="all", survey=design)
            if missing is not None:
                missing.record(table)
            if survey is not None and survey.concern():
                concerns.append(survey.concern())
        rows, info = table.rows, table.info
        concerns.extend(table.concerns)
    else:
        rows = family.coefficients(fitted, X, y, task=task, purpose=state.purpose)
        if (rows is not None and survey is not None and survey.design is not None
                and state.purpose == "inference"):
            # MS4: no design-based estimator under the population answer: blocked and recorded.
            from turbotab.core.models.survey import no_design_estimator

            table = no_design_estimator(family, task, getattr(state, "models", None))
            rows, info = table.rows, table.info
            concerns.extend(table.concerns)
    return fitted, (rows, info), concerns


def exposure_features(fitted: Any, inputs: Sequence[str], roles: Mapping[str, str]) -> list[str]:
    """The model-matrix columns whose raw sources include an exposure column (the lineage's reading)."""
    from turbotab.core.models.lineage import trace

    steps = list(fitted.steps[:-1])
    if not steps:
        return [c for c in inputs if roles.get(c) == "exposure"]
    lineage = trace(steps, list(inputs), dict(roles))
    return [n.column for n in lineage.nodes if n.lane == "matrix" and n.role == "exposure"]


def _excludes_zero(row: Mapping[str, Any]) -> bool | None:
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if lo is None or hi is None:
        return None
    return bool(lo > 0 or hi < 0)


def changes_for(fits: Sequence[Mapping[str, Any]], exposures: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for feature in exposures:
        rows = []
        for f in fits:
            found = next((r for r in (f.get("coefficients") or []) if r["feature"] == feature), None)
            rows.append(found)
        if not any(r is not None and r.get("estimate") is not None for r in rows):
            continue
        estimates = [r["estimate"] for r in rows if r is not None and r.get("estimate") is not None]
        primary = rows[0]["estimate"] if rows and rows[0] is not None else None
        signs = {math.copysign(1, e) for e in estimates if e != 0}
        out.append({"feature": feature, "primary": primary, "lowest": min(estimates),
                    "highest": max(estimates), "sign_changes": len(signs) > 1,
                    "excludes_zero": [None if r is None else _excludes_zero(r) for r in rows]})
    return out


def _weight_concern(state: Any, analyses: Sequence[Mapping[str, Any]], frame: pd.DataFrame,
                    target: str) -> str | None:
    weights = {r.weight for a in analyses for r in a["rules"] if getattr(r, "kind", "") == "goldberg"}
    for w in sorted(weights):
        if w not in frame.columns or w == target:
            continue
        r = pd.to_numeric(frame[w], errors="coerce").corr(pd.to_numeric(frame[target], errors="coerce"))
        if r is not None and np.isfinite(r) and abs(r) >= WEIGHT_CORRELATION:
            return (f"`{target}` correlates {r:.2f} with `{w}`, which the Goldberg screen reads to "
                    f"estimate BMR: excluding by it selects partly on the outcome (Banna et al. "
                    f"2017).")
    return None


def methods_sentence(analyses: Sequence[Mapping[str, Any]], rows: str) -> str:
    from turbotab.core.voice import _rule_phrase, listing

    def rules_text(rules: Sequence[Any]) -> str:
        if not rules:
            return "keeping every row"
        return "excluding rows with " + listing([_rule_phrase(r) for r in rules], limit=3, ticked=False)

    primary = analyses[0]
    others = [f"{a['label']} ({rules_text(a['rules'])})" for a in analyses[1:]]
    text = f"The primary analysis was fit {rules_text(primary['rules'])}"
    if others:
        text += (f"; the same model was refit on {rows} for each sensitivity analysis: "
                 + "; ".join(others))
    return text + "."


# ── the stage ────────────────────────────────────────────────────────────────


def sensitivity_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.decisions import OUTCOME_RULE
    from turbotab.core.models import get_family
    from turbotab.core.models.inference import cluster_columns
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.data import open_store
    from turbotab.core.models.inference import Outcome
    from turbotab.core.stages.modeling import _survey, _task, coded_outcome, outcome_levels
    from turbotab.core.stages.working import table_info

    state = ctx.state
    task = _task(ctx)
    target = state.target
    inference = state.purpose == "inference"
    design = ctx.inputs["design"]
    spec = DesignSpec.from_dict(design.objects["spec"])
    pipelines = design.objects["pipelines"]
    families = [get_family(k) for k in (state.models or []) if k in pipelines]
    analyses = analyses_of(state)
    sealed = sealed_rows(ctx.inputs["split"])
    ingest = table_info(ctx)

    ctx.progress(0.05, "Counting the rows each analysis keeps")
    rows_by: list[np.ndarray | None] = []
    # A family that models the unit (WP12b) is told it under either purpose, as the fit stage does.
    unit_models = any("units" in pipelines[f.key].steps[-1][1].get_params(deep=False)
                      for f in families)
    follow_up: list[str] = []
    if task == "time_to_event":
        from turbotab.core.models.survival import follow_up_columns

        follow_up = follow_up_columns(state)
    with open_store(ctx) as store:
        unit_columns = cluster_columns(state, store.columns) if inference or unit_models else []
        for a in analyses:
            if target in {c for r in a["rules"] for c in r.reads()}:
                a["refused"] = f"A rule reads the outcome `{target}`. {OUTCOME_RULE}"
                rows_by.append(None)
                continue
            kept = rows_of(store, state, ingest, a["rules"])
            rows_by.append(kept if inference else np.setdiff1d(kept, sealed))
        every = np.unique(np.concatenate([r for r in rows_by if r is not None] or [np.empty(0, np.int64)]))
        columns = list(dict.fromkeys([*spec.inputs, target, *unit_columns, *follow_up,
                                      *[c for a in analyses for r in a["rules"] for c in r.reads()]]))
        frame = modeling_frame(store, columns, every, outcome=target)

    # The outcome as the fit stage codes it: the event as 1, an ordinal outcome's declared order
    # (WP12a), a time-to-event outcome with its follow-up (WP12b), named for the table's scale (WP8).
    outcome = Outcome(name=target, labels=outcome_levels(task, frame[target].to_numpy(), state.event))
    levels = None
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        coded, levels = ordinal_outcome(frame[target].to_numpy(), state.outcome_order, column=target)
    else:
        coded = coded_outcome(task, frame[target].to_numpy(), state.event)
    if task == "time_to_event":
        from turbotab.core.models.survival import time_to_event_outcome

        coded = time_to_event_outcome(state, frame, coded)
    y_values = np.asarray(coded)
    position = pd.Series(np.arange(len(frame)), index=frame.index)
    # Under a survey design every analysis is a domain of the one design (WP10), as the primary is.
    if inference:
        from turbotab.core.models.inference import resolve_clusters

        every_unit = resolve_clusters(state, frame[list(unit_columns)]) if unit_columns else None
        survey, _ = _survey(ctx, every_unit)
    else:
        survey = None
    out_families: list[dict[str, Any]] = []
    exposures: list[str] = []
    concerns: list[str] = []
    missing_by: dict[str, Any] = {}  # each analysis's missing-data handling, under inference
    total = max(1, len(families) * len(analyses))
    done = 0
    for family in families:
        fits: list[dict[str, Any]] = []
        for a, rows in zip(analyses, rows_by):
            done += 1
            ctx.progress(0.1 + 0.85 * done / total, f"{family.label}: {a['label']}")
            if rows is None:
                fits.append({"label": a["label"], "n_rows": 0, "coefficients": None,
                             "concerns": [a["refused"]]})
                continue
            part = frame.loc[rows]
            y_part = y_values[position.loc[rows].to_numpy()]
            try:
                if inference and a["label"] not in missing_by:
                    # Each analysis imputes its own rows (WP7), once for every family.
                    from turbotab.core.stages.modeling import _missing_for_table

                    from turbotab.core.models.inference import resolve_clusters

                    # MS2: the design and the clustering in each analysis's imputation model.
                    missing_by[a["label"]] = _missing_for_table(
                        ctx, spec, part[list(spec.inputs)], y_part, task,
                        [f.key for f in families], loss={"n_dropped": None}, survey=survey,
                        clusters=(resolve_clusters(state, part[list(unit_columns)])
                                  if unit_columns else None))
                fitted, (coef, info), worries = fit_on_rows(
                    state, family, pipelines[family.key], part, spec.inputs,
                    y_part, task, unit_columns, outcome=outcome,
                    survey=survey, levels=levels, missing=missing_by.get(a["label"]), spec=spec)
            except Exception as exc:  # noqa: BLE001 - an analysis that cannot be fit says why
                fits.append({"label": a["label"], "n_rows": int(len(rows)), "coefficients": None,
                             "concerns": [f"This analysis could not be fit: {exc}"]})
                continue
            if coef is None:
                break  # the family has no coefficient table: it is not part of this answer
            if not exposures:
                exposures = exposure_features(fitted, spec.inputs, spec.roles)
            fits.append({"label": a["label"], "n_rows": int(len(rows)), "coefficients": coef,
                         "inference": info, "concerns": worries})
        else:
            out_families.append({"family": family.key, "label": family.label, "fits": fits})
    if not out_families:
        raise ValueError("None of the chosen model families has a coefficient table, so there is "
                         "no estimate to set beside the primary's; choose the linear model.")

    if not inference:
        concerns.append("Under prediction, people a screen leaves out are still among those the "
                        "model will be used on; these fits read no held-out row.")
    weight = _weight_concern(state, analyses, frame, target)
    if weight:
        concerns.append(weight)
    if any(a["added"] for a in analyses):
        concerns.append(f"The every-row analysis was added because the primary excludes rows. {BANNA}")
    rows_word = "all eligible rows" if inference else "eligible rows outside the held-out set"
    artifact = SensitivityArtifact(
        purpose="inference" if inference else "prediction",
        rows=rows_word,
        analyses=[SensitivityRow(label=a["label"], primary=a["primary"], added=a["added"],
                                 rules=[_label(r) for r in a["rules"]],
                                 n_rows=0 if r is None else int(len(r)), refused=a.get("refused"))
                  for a, r in zip(analyses, rows_by)],
        exposures=exposures,
        families=out_families,
        changes={f["family"]: changes_for(f["fits"], exposures) for f in out_families},
        concerns=concerns,
        methods=methods_sentence(analyses, rows_word),
    )
    ctx.progress(1.0, "Done")
    return Bundle(data=artifact.model_dump(mode="json"))


def _label(rule: Any) -> str:
    from turbotab.core.stages.rows import rule_label

    return rule_label(rule)


SENSITIVITY_READS = ("sensitivity", "exclusions", "target", "roles", "roles_unconfirmed",
                     "role_confirmations", "reading_confirmations", "shape_confirmations",
                     "missing", "findings",
                     "purpose", "models", "task", "event", "categorical", "grain", "survey",
                     "outcome_order", "follow_up")

__all__ = [
    "BANNA", "EVERY_ROW", "SENSITIVITY_READS", "SensitivityArtifact", "SensitivityChange",
    "SensitivityFamily", "SensitivityFit", "SensitivityRow", "analyses_of", "changes_for",
    "exposure_features", "fit_on_rows", "methods_sentence", "rows_of", "sealed_rows",
    "sensitivity_stage",
]

"""Fixtures for the ESTIMAND acceptance tests: a cohort whose causal structure the generator
declares, and a runner that takes it through design, fit and the ``effects`` stage the way a worker
does (``modeling_fixtures``), serving the fit as the server does (``estimand.annotate_fit``).

The generator's truth is the fixture's author's: ``age``, ``sex`` and ``smoking`` cause the exposure
and the outcome (confounders); ``activity`` causes the outcome only (precision); ``bmi`` may have
been changed by the diet (unknown timing). Every expected value in the tests is computed from these
columns by an independent path (statsmodels, NumPy written out, or R), never by the engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turbotab.core import decisions as d
from turbotab.core.decisions import SplitSpec
from turbotab.core.tests import modeling_fixtures as mf

CONFOUNDER = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
PRECISION = {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "no"}
UNKNOWN_TIMING = {"causes_exposure": "unknown", "causes_outcome": "yes", "after_exposure": "unknown"}
ANSWERS = {"age": CONFOUNDER, "sex": CONFOUNDER, "smoking": CONFOUNDER, "activity": PRECISION,
           "bmi": UNKNOWN_TIMING}
ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
         "activity": "covariate", "bmi": "covariate", "fiber": "exposure"}
# BLUEPRINT §14.3: whole numbers settle nothing by their values; the author knows these are amounts.
AMOUNTS = {"code_or_count:smoking": "amount", "code_or_count:activity": "amount"}


def cohort(n: int = 800, seed: int = 7, *, prevalence: float = 0.3) -> pd.DataFrame:
    """A cohort with a linear outcome (``glucose``), a yes/no outcome (``dm``, about
    ``prevalence`` of rows) and a two-valued exposure (``supplement``) beside the continuous one."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    sex = rng.choice(["female", "male"], n)
    smoking = rng.binomial(1, 0.3, n).astype(float)
    activity = rng.integers(0, 8, n).astype(float)
    fiber = 20 - 3 * smoking + 0.05 * (age - 50) + 1.0 * (sex == "male") + rng.normal(0, 4, n)
    bmi = 27 - 0.1 * (fiber - 20) + rng.normal(0, 3, n)
    supplement = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * smoking - 0.02 * (age - 50)))), n)
    glucose = (100 - 0.4 * fiber + 6 * smoking + 0.3 * (age - 50) + 0.5 * (bmi - 27)
               - 1.2 * activity + 2 * (sex == "male") + rng.normal(0, 8, n))
    logit = (np.log(prevalence / (1 - prevalence)) - 0.06 * (fiber - 20) + 0.8 * smoking
             + 0.03 * (age - 50) - 0.25 * (activity - 3.5) + 0.5 * supplement - 0.2)
    dm = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"pid": np.arange(n), "age": age, "sex": sex, "smoking": smoking,
                         "activity": activity, "bmi": bmi, "fiber": fiber,
                         "supplement": supplement.astype(float), "glucose": glucose,
                         "dm": np.where(dm, "yes", "no")})


def answers_for(exposure: str, which: dict[str, dict[str, str]]) -> dict[str, d.AdjustmentAnswer]:
    return {c: d.AdjustmentAnswer(exposure=exposure, **a) for c, a in which.items()}


def state(*, target: str, task: str, exposure: str = "fiber", measure: str,
          roles: dict[str, str] | None = None, answers: dict[str, dict[str, str]] | None = None,
          model_1: list[str] | None = None, event: str | None = None, models: list[str] | None = None,
          missing: Any = "complete_case", responses: dict[str, str] | None = None,
          family: bool = False, multiplicity: str | None = None, seed: int = 3,
          lens: list[str] | None = None, amounts: dict[str, str] | None = None,
          **slots: Any) -> d.ProjectState:
    roles = dict(ROLES if roles is None else roles)
    answers = ANSWERS if answers is None else answers
    key = d.EXPOSURE_FAMILY if family else exposure
    spec = d.EstimandSpec(exposure=None if family else exposure, family=family, measure=measure,
                          multiplicity=multiplicity if family else None)
    return mf.state(
        roles=roles, role_confirmations=dict(roles), target=target, task=task, event=event,
        purpose="inference", models=models or ["linear"], lens=lens or ["clinical"],
        split=SplitSpec(holdout=0.0, seed=seed, folds=5),
        shape_confirmations=dict(AMOUNTS if amounts is None else amounts),
        estimand=spec, adjustment=answers_for(key, {c: a for c, a in answers.items() if c in roles}),
        model_sequence=(d.ModelSequenceSpec(exposure=key, model_1=model_1)
                        if model_1 is not None else None),
        diagnostic_responses=({c: d.DiagnosticResponse(exposure=key, action=a)
                               for c, a in responses.items()} if responses else None),
        missing=missing,
        **{"grain": d.GrainSpec(grain="one_row_per_unit", id_column="pid"), **slots})


def run(frame: pd.DataFrame, folder: Path, st: d.ProjectState, *,
        fit: bool = True) -> dict[str, Any]:
    """design → (fit, served as the server serves it) → effects, on every row (no holdout)."""
    from turbotab.core.estimand import annotate_fit
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage, fit_stage

    paths = mf.ingest_frame(frame, folder)
    n = len(frame)
    split = mf.split_bundle(np.arange(n), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    out: dict[str, Any] = {"state": st, "design": design}
    inputs = {"design": design, "split": split, "target_info": info}
    if fit:
        fitted = fit_stage(mf.context(st, inputs, paths))
        out["fit"] = annotate_fit(fitted.data, st)
        out["fit_raw"] = fitted.data
    out["effects"] = effects_stage(mf.context(st, inputs, paths)).data
    return out


def sequence(effects: dict[str, Any], family: int = 0) -> dict[str, dict[str, Any]]:
    return {s["key"]: s for s in effects["families"][family]["sequence"]}


def design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The columns as a least-squares design written out by hand: ``sex`` as its ``male``
    indicator (female the reference, as the one-hot step drops the first level), an intercept."""
    X = pd.DataFrame({"(intercept)": np.ones(len(frame))}, index=frame.index)
    for c in columns:
        if c == "sex":
            X["sex_male"] = (frame["sex"] == "male").astype(float)
        else:
            X[c] = frame[c].astype(float)
    return X


__all__ = ["AMOUNTS", "ANSWERS", "CONFOUNDER", "PRECISION", "ROLES", "UNKNOWN_TIMING",
           "answers_for", "cohort", "design_matrix", "run", "sequence", "state"]

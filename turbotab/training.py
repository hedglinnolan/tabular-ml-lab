"""`turbotab.training` — the first number the Guided door computes.

Every number this app has shown until now was read off the table or derived
from it directly. This one is produced by fitting something, and that changes
what a mistake costs: a held-out metric is the quantity the whole lockbox
constitution exists to protect, and a leak in here is invisible in the output
by construction — a leaked score looks like a good score.

So the shape of this module is decided by one rule, `DOMAIN_SCIENCE.md` §05:

> **Every parameter estimated from data must be estimated inside the
> resampling loop.**

and its consequence, which is the part that is easy to get wrong: the family is
larger than it looks. It is not only the model's coefficients. It is the
imputer's medians, the scaler's means and standard deviations, the encoder's
level set — every one of them is a fact about the rows it saw, and a fact
learned from the held-out rows has already leaked whether or not anyone
predicts with it.

**That is why this builds a `Pipeline` and fits it once, on the training rows
only.** Not because pipelines are tidy: because the alternative — transform the
frame, then split — cannot be made correct by being careful, and this one
cannot be made incorrect by being careless. The preprocessing steps are inside
the estimator, so `fit(X_train, y_train)` is the only place any parameter can
come from, and `turbotab/test_the_seal_holds.py` probes that claim rather than
trusting this docstring.

**The shelf is never shortened.** A model the coach ranks last is still
trainable, with its concern recorded beside the score. Ranking carries the
judgment; absence never does — so this module refuses a model that is not in
the registry and accepts every model that is.

Headless: no Streamlit, no HTTP, no global state. The queue owns the thread and
hands in the generator (`turbotab/jobs.py`), so the seed that produced a number
travels with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class TrainingRefusal(Exception):
    """Raised where the app will not produce a number it cannot stand behind."""


#: The smallest held-out set worth reporting a metric from. Below this the
#: number is noise with a decimal point, and reporting it would be the app
#: asserting a precision it does not have.
MIN_TEST_ROWS = 10


@dataclass
class ModelResult:
    key: str
    name: str
    concern: str
    bucket: str
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[List[float]] = None
    probabilities: Optional[List[float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "concern": self.concern,
            "bucket": self.bucket, "metrics": self.metrics,
            "error": self.error,
            "n_predictions": 0 if self.predictions is None
            else len(self.predictions),
        }


@dataclass
class TrainingRun:
    task_type: str
    target: str
    n_train: int
    n_test: int
    seal_basis: Optional[str]
    exploratory: bool
    results: List[ModelResult] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type, "target": self.target,
            "n_train": self.n_train, "n_test": self.n_test,
            "seal_basis": self.seal_basis, "exploratory": self.exploratory,
            "features": list(self.features), "notes": list(self.notes),
            "results": [r.to_dict() for r in self.results],
        }


def _feature_frame(table: pd.DataFrame, target: str,
                   group_col: Optional[str]) -> pd.DataFrame:
    """Everything except the outcome and the grouping key.

    The grouping column is dropped rather than encoded: it identifies the
    participant, so a model given it can memorize who rather than learn what,
    and on a grouped split it is a column of unseen levels at test time.
    """
    drop = {str(target)}
    if group_col:
        drop.add(str(group_col))
    return table[[c for c in table.columns if str(c) not in drop]]


def _pipeline(estimator: Any, frame: pd.DataFrame, *,
              needs_scaling: bool) -> Any:
    """Preprocessing INSIDE the estimator, so it can only see training rows.

    The alternative — transform the frame and then split — is the leak
    `DOMAIN_SCIENCE.md` §05 names, and it cannot be made safe by being careful
    about the order. This can only be made unsafe by moving a step out.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in frame.columns
               if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]

    numeric_steps: List[Any] = [("impute", SimpleImputer(strategy="median"))]
    if needs_scaling:
        numeric_steps.append(("scale", StandardScaler()))

    blocks: List[Any] = []
    if numeric:
        blocks.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        blocks.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
            categorical))
    if not blocks:
        raise TrainingRefusal(
            "Every column except the outcome was dropped, so there is nothing "
            "to fit on.")
    return Pipeline([("prep", ColumnTransformer(blocks)),
                     ("model", estimator)])


def _metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    from ml.eval import (calculate_classification_metrics,
                         calculate_regression_metrics)

    if task_type == "classification":
        out = calculate_classification_metrics(y_true, y_pred, y_proba)
    else:
        out = calculate_regression_metrics(y_true, y_pred)
    return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                else float(v))
            for k, v in out.items() if isinstance(v, (int, float, np.floating))}


def check(project: Any, model_keys: Sequence[str]) -> None:
    """Everything `train` refuses, checked without fitting anything.

    Split out so a caller can ask *could this produce a number?* before
    submitting a job — a request that cannot must fail as a refusal the caller
    reads, not as a job that goes away and comes back empty.
    """
    from ml.model_registry import get_registry

    if not project.target:
        raise TrainingRefusal("Training needs an outcome column first.")
    if not (project.lockbox and project.lockbox.get("labels")):
        raise TrainingRefusal(
            "The held-out set is not sealed yet. A score computed before the "
            "seal is a score on rows the model may have been fitted on, and "
            "there is no way to tell afterwards which it was.")
    if not model_keys:
        raise TrainingRefusal(
            "No model is selected. The shelf orders every model this task can "
            "use and never shortens itself, so choosing is the step — nothing "
            "here is chosen for you.")
    registry = get_registry()
    unknown = [k for k in model_keys if k not in registry]
    if unknown:
        raise TrainingRefusal(f"{unknown} are not in the model registry.")

    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    has_y = table[target].notna()
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    n_test = int((has_y & is_test).sum())
    n_train = int((has_y & ~is_test).sum())
    if n_test < MIN_TEST_ROWS:
        raise TrainingRefusal(
            f"{n_test} held-out rows have a value for {target!r}, which is too "
            f"few to report a metric from — below about {MIN_TEST_ROWS} the "
            f"number moves more with which rows were drawn than with which "
            f"model was fitted.")
    if n_train < MIN_TEST_ROWS:
        raise TrainingRefusal(
            f"{n_train} training rows have a value for {target!r}, which is "
            f"too few to fit on.")


def train(project: Any, model_keys: Sequence[str], *,
          ctx: Any = None, seed: int = 42) -> TrainingRun:
    """Fit each model on the training rows and score it on the sealed ones.

    `ctx` is a :class:`turbotab.jobs.JobContext` when this runs on the queue:
    its generator supplies the seed, its cancel token is checked between models
    so stopping means stopping, and its progress channel is what makes the work
    watchable rather than a spinner (`PRODUCT_VISION.md` §04).
    """
    from ml.model_registry import get_registry

    check(project, model_keys)
    registry = get_registry()
    table = project.working_table
    target = str(project.target)
    task_type = project.task_type or "regression"
    group_col = (project.grain or {}).get("group_col")

    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    # The outcome mask is applied to BOTH halves, so a row with no outcome is
    # not counted as held out and then silently dropped from the score.
    has_y = table[target].notna()

    features = _feature_frame(table, target, group_col)
    X_train = features[has_y & ~is_test]
    X_test = features[has_y & is_test]
    y_train = table.loc[X_train.index, target]
    y_test = table.loc[X_test.index, target]

    disclosures = getattr(project, "lockbox", {}) or {}
    run = TrainingRun(
        task_type=task_type, target=target,
        n_train=int(len(X_train)), n_test=int(len(X_test)),
        seal_basis=disclosures.get("seal_basis"),
        exploratory=bool(_is_exploratory(project)),
        features=[str(c) for c in features.columns])

    # THE LIMIT OF THIS SLICE, said on the run rather than left silent.
    # `GUIDED-089`. The pipeline above is this module's own — median impute,
    # standard-scale where the model needs it — and it is NOT the plan the user
    # recorded in Preprocess, nor the variants the recipe lattice resolved. The
    # numbers are honest about the seal and they are not the numbers the
    # recorded plan would produce, and a reader has no way to know that unless
    # it is written down.
    run.notes.append(
            "These fits use a default pipeline — median imputation, and "
            "standardization where the model requires it — not the recorded "
            "preprocessing plan. The plan is recorded and is not yet what "
            "gets fitted.")

    if run.exploratory:
        # The number is honest and what it MEANS is not what a clean split
        # would mean, so the run says so and every consumer carries it.
        run.notes.append(
            "This split is not a verified clean one, so these numbers are "
            "exploratory: they may read better than the models are.")

    for i, key in enumerate(model_keys):
        if ctx is not None:
            ctx.raise_if_cancelled()
            ctx.progress(i / max(len(model_keys), 1),
                         f"Fitting {registry[key].name} on {run.n_train:,} rows")
        spec = registry[key]
        caps = spec.capabilities
        result = ModelResult(
            key=key, name=spec.name,
            concern=_concern(project, key),
            bucket=_bucket(project, key))
        try:
            estimator = spec.factory(task_type, int(seed))
            pipe = _pipeline(estimator, features,
                             needs_scaling=bool(caps.requires_scaled_numeric))
            pipe.fit(X_train, y_train)
            y_pred = np.asarray(pipe.predict(X_test))
            y_proba = None
            if task_type == "classification" and hasattr(pipe, "predict_proba"):
                proba = np.asarray(pipe.predict_proba(X_test))
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_proba = proba[:, 1]
            result.metrics = _metrics(task_type, np.asarray(y_test), y_pred,
                                      y_proba)
            result.predictions = [float(v) for v in y_pred]
            if y_proba is not None:
                result.probabilities = [float(v) for v in y_proba]
        except Exception as exc:                       # one model, not the run
            # A model that will not fit is a RESULT, not a crash: the shelf is
            # never shortened, so a model that fails says why in the place its
            # score would have been.
            result.error = f"{type(exc).__name__}: {exc}"
        run.results.append(result)

    if ctx is not None:
        ctx.progress(1.0, f"{len(run.results)} model(s) scored on "
                          f"{run.n_test:,} held-out rows")
    return run


def _is_exploratory(project: Any) -> bool:
    from turbotab import grain as _grain

    lockbox = project.lockbox or {}
    if _grain.is_exploratory_basis(lockbox.get("seal_basis")):
        return True
    grain = project.grain or {}
    return bool(grain.get("design_not_described")
                or grain.get("contradiction_acknowledged"))


def _shelf_entry(project: Any, key: str) -> Dict[str, Any]:
    """The shelf's own words for a model, so a result quotes rather than
    paraphrases. Read through the project, which is the one caller that knows
    how to build the profile the shelf ranks on."""
    try:
        entries = project.model_shelf()
    except Exception:
        return {}
    for entry in entries:
        row = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        if row.get("key") == key:
            return row
    return {}


def _concern(project: Any, key: str) -> str:
    return str(_shelf_entry(project, key).get("concern") or "")


def _bucket(project: Any, key: str) -> str:
    return str(_shelf_entry(project, key).get("bucket") or "")


def y_true_for(project: Any) -> List[Any]:
    """The held-out outcomes, in the order `train` predicted them.

    Exported so a consumer — the calibration figure — pairs predictions with
    outcomes by construction rather than by both sides re-deriving the mask.
    """
    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    keep = table[target].notna() & is_test
    return list(table.loc[keep, target])

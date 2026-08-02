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

**And the pipeline is composed from the RECORD, not from this module's own
opinion** (`GUIDED-095`). It used to build its own ColumnTransformer — median
for numeric, most-frequent plus one-hot for categorical, standard-scale where
the estimator declared it needed one — with no reference to any declaration the
user had made, so the app was *safe and unfaithful*: every number honest about
the seal, none of them about the analysis the user specified.
`turbotab/pipeline_plan.py` is the executor; this module asks it, reports what
it did, and reports per model where it could not do what was recorded.

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
    predictions: Optional[List[Any]] = None
    probabilities: Optional[List[float]] = None
    #: WHICH CLASS `probabilities` IS ABOUT. `predict_proba`'s second column is
    #: `classes_[1]`, and with a 0/1 target that is 1 and nobody had to ask.
    #: With `responder` / `non-responder` it is whichever sorts second, and a
    #: calibration curve that did not know which would be a picture of the
    #: wrong event drawn confidently (`GUIDED-093`).
    positive_label: Optional[Any] = None
    error: Optional[str] = None
    #: What this model's fold-fitted pipeline actually did, and where it could
    #: not do what was recorded. Per model, because a divergence is per model:
    #: `leave` is honored by gradient boosting and not by a linear fit, and a
    #: run-level note could not say which (`GUIDED-095`).
    plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "concern": self.concern,
            "bucket": self.bucket, "metrics": self.metrics,
            "error": self.error,
            "n_predictions": 0 if self.predictions is None
            else len(self.predictions),
            "positive_label": (None if self.positive_label is None
                               else str(self.positive_label)),
            "plan": dict(self.plan),
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


def _plan(project: Any, model_key: str, frame: pd.DataFrame, *,
          seed: int = 42) -> Any:
    """What this model's fold-fitted pipeline will do, from the RECORD.

    This function used to be `_pipeline`, and it built its own
    ColumnTransformer — median for numeric, most-frequent plus one-hot for
    categorical, standard-scale where the estimator declared it needed one —
    with no reference to any declaration the user had made. `GUIDED-095`: the
    app records 36 kinds of decision and the trainer read six project
    attributes, so every decision recorded to be *fitted inside the training
    fold* reached nothing at all. The record was written, the receipt counted
    it, the sentence was composed, and the executor did not exist.

    `turbotab.pipeline_plan` is the executor. What is preserved exactly is the
    property this module's docstring is about: the preprocessing is still
    constructed INSIDE the estimator, so `fit(X_train, y_train)` is still the
    only place any parameter can come from.
    """
    from turbotab import pipeline_plan as _plan_mod

    try:
        return _plan_mod.compose(project, model_key, frame, seed=seed)
    except _plan_mod.PlanRefusal as exc:
        raise TrainingRefusal(str(exc)) from exc


def _pipeline(estimator: Any, frame: pd.DataFrame, *,
              needs_scaling: bool, project: Any = None,
              model_key: str = "ridge", seed: int = 42) -> Any:
    """Preprocessing INSIDE the estimator, so it can only see training rows.

    The alternative — transform the frame and then split — is the leak
    `DOMAIN_SCIENCE.md` §05 names, and it cannot be made safe by being careful
    about the order. This can only be made unsafe by moving a step out.

    Kept as the one-call form for callers that hold an estimator rather than a
    plan — the seal probes are the ones that matter, because they read fitted
    parameters off the pipeline this returns. `needs_scaling` is now advisory:
    whether a model gets scaled inputs is the recipe table's answer, and the
    table seeds it from the same `requires_scaled_numeric` capability this
    argument came from.
    """
    if project is None:
        raise TrainingRefusal(
            "A pipeline is composed from a project's recorded plan. Building "
            "one without it is what `GUIDED-095` was about.")
    return _plan(project, model_key, frame, seed=seed).build(estimator)


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


def _serialize(values: np.ndarray) -> List[Any]:
    """Predictions as JSON, keeping a class label a label.

    `GUIDED-093`. This was `[float(v) for v in y_pred]`, and a classification
    target with string labels — `responder` / `non-responder`, `died` /
    `survived`, `case` / `control`, which is the ordinary case in clinical
    research — made it raise *after* the metrics had been assigned. Three false
    statements came out of one line, and the third was the one that mattered:
    the calibration figure told the researcher *"the models that were fitted do
    not produce probabilities, so there is nothing to calibrate"* about
    logistic regression, because the app's own serialization bug had emptied
    the probabilities.

    A number stays a number so a regression consumer is unchanged; anything
    else becomes its own string.
    """
    out: List[Any] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            out.append(bool(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out.append(float(value))
        else:
            out.append(str(value))
    return out


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

    # `GUIDED-089` / `GUIDED-095`. This used to be a note saying the recorded
    # preprocessing plan was NOT the one fitted — the honest form of a defect,
    # and a placeholder with a deadline. **This is the deadline.** The plan is
    # composed from the record now, so the run reports what it READ rather than
    # what it ignored, and a reader can count the decisions that reached the
    # fit against the decisions that were made.
    n_declared = len(getattr(project, "missingness", None) or [])
    n_deferred = len(getattr(project, "deferred_transforms", None) or [])
    run.notes.append(
        f"Each model's pipeline was composed from the recorded plan: "
        f"{n_declared} missingness declaration(s), {n_deferred} deferred "
        f"transform(s), and the per-model recipe the table resolved for it. "
        f"Every statistic in it is fitted inside the training folds.")

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
        result = ModelResult(
            key=key, name=spec.name,
            concern=_concern(project, key),
            bucket=_bucket(project, key))
        try:
            estimator = spec.factory(task_type, int(seed))
            plan = _plan(project, key, features, seed=int(seed))
            # THE PLAN TRAVELS WITH THE RESULT, whether or not the fit
            # succeeds. A run that says what it intended to do and then fails
            # is readable; one that says nothing is a shrug.
            result.plan = plan.to_dict()
            pipe = plan.build(estimator)
            pipe.fit(X_train, y_train)
            y_pred = np.asarray(pipe.predict(X_test))
            y_proba = None
            if task_type == "classification" and hasattr(pipe, "predict_proba"):
                proba = np.asarray(pipe.predict_proba(X_test))
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_proba = proba[:, 1]
                    # WHICH class this column is about, recorded rather than
                    # assumed. With a 0/1 target it is 1 and nobody had to
                    # ask; with `responder` / `non-responder` it is whichever
                    # sorts second, and a curve drawn against the other one
                    # would be confidently wrong (`GUIDED-093`).
                    classes = getattr(pipe, "classes_", None)
                    if classes is not None and len(classes) == 2:
                        result.positive_label = classes[1]
            result.metrics = _metrics(task_type, np.asarray(y_test), y_pred,
                                      y_proba)
            # SERIALIZED AS WHAT THEY ARE (`GUIDED-093`). This was
            # `[float(v) for v in y_pred]`, which raises on a string class
            # label — `responder`, `died`, `case` — and raises AFTER the
            # metrics were assigned, so the handler below labeled a model that
            # had fitted and scored a failure. A class label is a label; it is
            # serialized as one.
            result.predictions = _serialize(y_pred)
            if y_proba is not None:
                result.probabilities = [float(v) for v in y_proba]
        except Exception as exc:                       # one model, not the run
            # A model that will not fit is a RESULT, not a crash: the shelf is
            # never shortened, so a model that fails says why in the place its
            # score would have been.
            #
            # AND A RESULT CARRIES A SCORE OR A REASON, NEVER BOTH. If anything
            # after the fit raised, the metrics assigned before it are not
            # trustworthy either — they describe a run that did not complete.
            # `GUIDED-093`: the guard was `metrics or error`, which is
            # satisfied when both are set, and the app spent two loops
            # reporting Accuracy 0.857 beside "did not fit".
            result.error = f"{type(exc).__name__}: {exc}"
            result.metrics = {}
            result.predictions = None
            result.probabilities = None
            result.positive_label = None
        if not result.metrics and not result.error:
            # A fit that completed and produced no usable metric is neither a
            # score nor a crash, and it must not be served as an empty success.
            # Stated as the reason it is, so the row still says something.
            result.error = (
                "The fit completed and produced no finite metric — every "
                "score the evaluator computed was undefined on this split.")
        run.results.append(result)

    # WHAT WAS NOT RECORDED IS COUNTED TOO. A run that reports only what it
    # honored lets a column nobody answered read as a column somebody chose a
    # default for — which is `GUIDED-089` inverted, and just as quiet.
    undeclared = sorted({c for r in run.results
                         for c in (r.plan.get("undeclared") or [])})
    if undeclared:
        run.notes.append(
            f"{len(undeclared)} column(s) with missing values had no recorded "
            f"handling and were filled with this app's default inside each "
            f"training fold: {', '.join(undeclared[:8])}"
            + (f", and {len(undeclared) - 8} more." if len(undeclared) > 8
               else "."))
    # AND WHERE A MODEL COULD NOT DO WHAT WAS RECORDED. Per model, because a
    # divergence is per model: `leave` is honored by gradient boosting and not
    # by a linear fit, and a run-level sentence could not say which.
    for result in run.results:
        for divergence in result.plan.get("divergences") or []:
            run.notes.append(divergence["fitted_sentence"])
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
